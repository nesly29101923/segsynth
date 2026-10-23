"""Choosing where a transformed defect goes on a target image."""
from __future__ import annotations

from enum import Enum

import cv2
import numpy as np


class PlacementMode(str, Enum):
    RANDOM = "random"
    KEEP_COLUMN = "keep_column"
    KEEP_ROW = "keep_row"
    NEAR_SOURCE = "near_source"


def find_position(
    patch_mask: np.ndarray,
    target_shape: tuple[int, int],
    rng: np.random.Generator,
    mode: PlacementMode | str = PlacementMode.RANDOM,
    source_center: tuple[float, float] | None = None,
    jitter: int = 40,
    occupied: np.ndarray | None = None,
    allowed: np.ndarray | None = None,
    min_gap: int = 8,
    edge_margin: int = 2,
    attempts: int = 200,
) -> tuple[int, int] | None:
    """Return top-left (x, y) for the patch, or None if no valid spot was found."""
    mode = PlacementMode(mode)
    ph, pw = patch_mask.shape
    th, tw = target_shape
    x_lo, x_hi = edge_margin, tw - pw - edge_margin
    y_lo, y_hi = edge_margin, th - ph - edge_margin
    if x_hi < x_lo or y_hi < y_lo:
        return None

    mask_bool = patch_mask > 0
    blocked = None
    if occupied is not None and occupied.any():
        blocked = occupied.astype(np.uint8)
        if min_gap > 0:
            k = 2 * min_gap + 1
            blocked = cv2.dilate(blocked, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    sx = sy = None
    if source_center is not None:
        sx = source_center[0] - pw / 2.0
        sy = source_center[1] - ph / 2.0

    def draw_axis(lo: int, hi: int, src: float | None, constrain: bool) -> int:
        if constrain and src is not None:
            a, b = max(lo, int(src - jitter)), min(hi, int(src + jitter))
            if a > b:
                a = b = int(np.clip(src, lo, hi))
            return int(rng.integers(a, b + 1))
        return int(rng.integers(lo, hi + 1))

    for _ in range(attempts):
        cx_constrained = mode in (PlacementMode.KEEP_COLUMN, PlacementMode.NEAR_SOURCE)
        cy_constrained = mode in (PlacementMode.KEEP_ROW, PlacementMode.NEAR_SOURCE)
        x = draw_axis(x_lo, x_hi, sx, cx_constrained)
        y = draw_axis(y_lo, y_hi, sy, cy_constrained)
        if blocked is not None and np.any(blocked[y:y + ph, x:x + pw][mask_bool]):
            continue
        if allowed is not None and not np.all(allowed[y:y + ph, x:x + pw][mask_bool]):
            continue
        return x, y
    return None
