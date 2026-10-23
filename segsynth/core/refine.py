"""Snap a rough user brush stroke to the defect pixels it contains."""
from __future__ import annotations

import cv2
import numpy as np

from .background import BackgroundMode, estimate_background


def deviation_map(image: np.ndarray, bg_mode: str = "vertical", bg_size: int = 151) -> np.ndarray:
    if BackgroundMode(bg_mode) == BackgroundMode.NONE:
        bg_mode = "isotropic"
    f = image.astype(np.float32)
    bg = estimate_background(image, bg_mode, bg_size)
    dev = np.abs(f - bg)
    dev = dev.max(axis=2) if dev.ndim == 3 else dev
    scale = 65535.0 if image.dtype == np.uint16 else 255.0
    return (dev / scale).astype(np.float32)


def refine_mask(rough: np.ndarray, deviation: np.ndarray, sensitivity: float = 0.5, close_px: int = 5) -> np.ndarray:
    """Keep pixels inside `rough` that deviate from the background.

    sensitivity in [0, 1]: higher keeps fainter pixels.
    """
    region = rough > 0
    if not region.any():
        return rough.astype(np.uint8)
    vals = deviation[region]
    lo, hi = np.percentile(vals, 5), np.percentile(vals, 99.5)
    thr = hi - (hi - lo) * float(np.clip(sensitivity, 0.02, 0.98))
    m = ((deviation >= thr) & region).astype(np.uint8)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        min_area = max(6, int(stats[1:, cv2.CC_STAT_AREA].max() * 0.02))
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        m = keep[lab].astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return (filled & region.astype(np.uint8)) | m
