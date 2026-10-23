"""Randomized geometric and photometric variation of defect patches."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .patch import DefectPatch, mask_bbox

Range = tuple[float, float]


@dataclass
class TransformParams:
    scale: Range = (0.85, 1.2)
    elongation: Range = (0.85, 1.3)
    rotation: Range = (-8.0, 8.0)
    bend: Range = (-0.15, 0.15)
    elastic_alpha: float = 3.0
    elastic_sigma: float = 12.0
    intensity: Range = (0.8, 1.2)
    flip_h: float = 0.0
    flip_v: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TransformParams":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k, v in known.items():
            if isinstance(v, list):
                known[k] = tuple(v)
        return cls(**known)


@dataclass
class AppliedTransform:
    scale: float
    elongation: float
    rotation: float
    bend: float
    intensity: float
    flip_h: bool
    flip_v: bool
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def principal_angle(mask: np.ndarray) -> float:
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return 0.0
    return 0.5 * math.degrees(math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"]))


def _uniform(rng: np.random.Generator, r: Range) -> float:
    lo, hi = float(min(r)), float(max(r))
    return lo if hi - lo < 1e-9 else float(rng.uniform(lo, hi))


def sample(params: TransformParams, rng: np.random.Generator) -> AppliedTransform:
    return AppliedTransform(
        scale=_uniform(rng, params.scale),
        elongation=_uniform(rng, params.elongation),
        rotation=_uniform(rng, params.rotation),
        bend=_uniform(rng, params.bend),
        intensity=_uniform(rng, params.intensity),
        flip_h=bool(rng.random() < params.flip_h),
        flip_v=bool(rng.random() < params.flip_v),
    )


def _warp_all(arrs, M, size, flags):
    out = []
    for a, interp in arrs:
        out.append(cv2.warpAffine(a, M, size, flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    return out


def apply(
    patch: DefectPatch,
    t: AppliedTransform,
    rng: np.random.Generator,
    elastic_alpha: float = 0.0,
    elastic_sigma: float = 12.0,
) -> DefectPatch:
    residual, pixels, alpha, mask = patch.residual, patch.pixels, patch.alpha, patch.mask.astype(np.float32)
    if t.flip_h:
        residual, pixels, alpha, mask = (a[:, ::-1].copy() for a in (residual, pixels, alpha, mask))
    if t.flip_v:
        residual, pixels, alpha, mask = (a[::-1, :].copy() for a in (residual, pixels, alpha, mask))

    h, w = mask.shape
    cx, cy = w / 2.0, h / 2.0
    theta = principal_angle(mask > 0.5)
    rad = math.radians(theta)
    c, s = math.cos(rad), math.sin(rad)
    R = np.array([[c, -s], [s, c]])
    S = np.diag([t.scale * t.elongation, t.scale])
    rr = math.radians(t.rotation)
    Rx = np.array([[math.cos(rr), -math.sin(rr)], [math.sin(rr), math.cos(rr)]])
    A = Rx @ R @ S @ R.T

    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float64) - [cx, cy]
    tc = corners @ A.T
    length = max(w, h) * t.scale * max(t.elongation, 1.0)
    pad = int(abs(t.bend) * length * 0.5 + elastic_alpha * 3 + 4)
    nw = int(math.ceil(tc[:, 0].max() - tc[:, 0].min())) + 2 * pad
    nh = int(math.ceil(tc[:, 1].max() - tc[:, 1].min())) + 2 * pad
    nw, nh = max(nw, 3), max(nh, 3)
    M = np.zeros((2, 3))
    M[:, :2] = A
    M[:, 2] = np.array([nw / 2.0, nh / 2.0]) - A @ np.array([cx, cy])

    residual, pixels, alpha, mask = _warp_all(
        [(residual, cv2.INTER_LINEAR), (pixels, cv2.INTER_LINEAR), (alpha, cv2.INTER_LINEAR), (mask, cv2.INTER_LINEAR)],
        M, (nw, nh), None,
    )

    need_bend = abs(t.bend) > 1e-4
    need_elastic = elastic_alpha > 1e-3
    if need_bend or need_elastic:
        ys, xs = np.mgrid[0:nh, 0:nw].astype(np.float32)
        map_x, map_y = xs.copy(), ys.copy()
        if need_bend:
            phi = math.radians(theta + t.rotation)
            ux, uy = math.cos(phi), math.sin(phi)
            nx, ny = -uy, ux
            dx, dy = xs - nw / 2.0, ys - nh / 2.0
            u = dx * ux + dy * uy
            half = max(length / 2.0, 1.0)
            disp = t.bend * half * ((u / half) ** 2 - 0.5)
            map_x -= disp * nx
            map_y -= disp * ny
        if need_elastic:
            sig = max(1.0, elastic_sigma)
            fx = cv2.GaussianBlur(rng.uniform(-1, 1, (nh, nw)).astype(np.float32), (0, 0), sig)
            fy = cv2.GaussianBlur(rng.uniform(-1, 1, (nh, nw)).astype(np.float32), (0, 0), sig)
            norm = max(np.abs(fx).max(), np.abs(fy).max(), 1e-6)
            map_x += fx / norm * elastic_alpha
            map_y += fy / norm * elastic_alpha
        residual, pixels, alpha, mask = (
            cv2.remap(a, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            for a in (residual, pixels, alpha, mask)
        )

    if residual.ndim == 2:
        residual, pixels = residual[:, :, None], pixels[:, :, None]
    hard = (mask > 0.5).astype(np.uint8)
    bbox = mask_bbox(alpha > 0.01)
    if bbox is None or hard.sum() == 0:
        raise ValueError("Transformed defect vanished")
    x0, y0, x1, y1 = bbox
    return DefectPatch(
        residual=(residual[y0:y1, x0:x1] * t.intensity).astype(np.float32),
        pixels=pixels[y0:y1, x0:x1].astype(np.float32),
        alpha=np.clip(alpha[y0:y1, x0:x1], 0, 1).astype(np.float32),
        mask=hard[y0:y1, x0:x1],
        origin=patch.origin,
        source_shape=patch.source_shape,
    )
