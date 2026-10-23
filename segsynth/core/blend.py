"""Compositing a defect patch into a target image."""
from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from .patch import DefectPatch


class BlendMode(str, Enum):
    RESIDUAL = "residual"
    POISSON = "poisson"
    POISSON_MIXED = "poisson_mixed"
    ALPHA = "alpha"


def _as3(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 3 else img[:, :, None]


def blend(
    target: np.ndarray,
    patch: DefectPatch,
    pos: tuple[int, int],
    mode: BlendMode | str = BlendMode.RESIDUAL,
    max_val: float = 255.0,
) -> np.ndarray:
    """Return a new image with the patch composited at top-left `pos`."""
    mode = BlendMode(mode)
    out = _as3(target).astype(np.float32).copy()
    x, y = pos
    ph, pw = patch.mask.shape
    roi = out[y:y + ph, x:x + pw]
    a = patch.alpha[:, :, None]

    if mode == BlendMode.RESIDUAL:
        roi += patch.residual * a
    elif mode == BlendMode.ALPHA:
        roi[:] = patch.pixels * a + roi * (1.0 - a)
    else:
        th, tw = out.shape[:2]
        margin = 16
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(tw, x + pw + margin), min(th, y + ph + margin)
        local = np.ascontiguousarray(out[y0:y1, x0:x1])
        out[y0:y1, x0:x1] = _poisson(local, patch, (x - x0, y - y0), mode, max_val)

    out = np.clip(out, 0, max_val)
    dtype = target.dtype
    res = np.rint(out).astype(dtype)
    return res if target.ndim == 3 else res[:, :, 0]


def _poisson(out: np.ndarray, patch: DefectPatch, pos, mode: BlendMode, max_val: float) -> np.ndarray:
    x, y = pos
    ph, pw = patch.mask.shape
    th, tw, channels = out.shape
    pad = 4
    a = patch.alpha[:, :, None]

    padded = cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    if padded.ndim == 2:
        padded = padded[:, :, None]
    src = padded[y:y + ph + 2 * pad, x:x + pw + 2 * pad].copy()
    inner = src[pad:pad + ph, pad:pad + pw]
    inner[:] = patch.pixels * a + inner * (1.0 - a)

    m = np.zeros(src.shape[:2], np.uint8)
    m[pad:pad + ph, pad:pad + pw] = (patch.alpha > 0.05).astype(np.uint8) * 255
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return out
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = bx1 - bx0, by1 - by0
    cx = x - pad + (bx0 + bx1) // 2
    cy = y - pad + (by0 + by1) // 2
    if cx - bw // 2 < 1 or cy - bh // 2 < 1 or cx + bw - bw // 2 > tw - 1 or cy + bh - bh // 2 > th - 1:
        roi = out[y:y + ph, x:x + pw]
        roi[:] = patch.pixels * a + roi * (1.0 - a)
        return out

    scale = 255.0 / max_val

    def to8(img: np.ndarray) -> np.ndarray:
        img8 = np.clip(img * scale, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR) if channels == 1 else img8

    flag = cv2.MIXED_CLONE if mode == BlendMode.POISSON_MIXED else cv2.NORMAL_CLONE
    cloned = cv2.seamlessClone(to8(src), to8(out), m, (int(cx), int(cy)), flag).astype(np.float32) / scale
    if channels == 1:
        cloned = cloned[:, :, :1]
    changed = np.zeros((th, tw), bool)
    changed[y:y + ph, x:x + pw] = patch.alpha > 0.05
    out[changed] = cloned[changed]
    return out
