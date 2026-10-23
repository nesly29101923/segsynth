"""Robust local background estimation used to separate a defect from surface texture.

The background is estimated with normalized convolution: pixels known to be defects
(the user mask) and strong outliers get zero weight, so the estimate is interpolated
from clean surroundings. Directional kernels follow textures such as grooves or
machining marks; the isotropic kernel suits textures without a dominant direction.
"""
from __future__ import annotations

from enum import Enum

import cv2
import numpy as np


class BackgroundMode(str, Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    ISOTROPIC = "isotropic"
    NONE = "none"


def _kernel(mode: BackgroundMode, size: int) -> tuple[int, int]:
    size = max(3, int(size) | 1)
    if mode == BackgroundMode.VERTICAL:
        return (1, size)
    if mode == BackgroundMode.HORIZONTAL:
        return (size, 1)
    return (size, size)


def estimate_background(
    img: np.ndarray,
    mode: BackgroundMode | str = BackgroundMode.VERTICAL,
    size: int = 151,
    exclude_mask: np.ndarray | None = None,
    iterations: int = 2,
    outlier_sigma: float = 2.5,
) -> np.ndarray:
    mode = BackgroundMode(mode)
    f = img.astype(np.float32)
    if f.ndim == 2:
        f = f[:, :, None]
    if mode == BackgroundMode.NONE:
        return np.zeros_like(f) if img.ndim == 3 else np.zeros_like(f[:, :, 0])

    ksize = _kernel(mode, size)
    weight = np.ones(f.shape[:2], np.float32)
    if exclude_mask is not None:
        weight[exclude_mask > 0] = 0.0

    eps = 1e-6
    bg = f
    for i in range(max(1, iterations)):
        wsum = cv2.blur(weight, ksize, borderType=cv2.BORDER_REFLECT)
        bg = np.empty_like(f)
        for c in range(f.shape[2]):
            bg[:, :, c] = cv2.blur(f[:, :, c] * weight, ksize, borderType=cv2.BORDER_REFLECT) / (wsum + eps)
        hole = wsum < 1e-3
        if hole.any():
            bg[hole] = cv2.blur(f, (max(ksize), max(ksize)), borderType=cv2.BORDER_REFLECT)[hole]
        if i < iterations - 1:
            dev = np.abs(f - bg).max(axis=2)
            sigma = 1.4826 * np.median(dev[weight > 0]) + eps if (weight > 0).any() else eps
            outliers = dev > outlier_sigma * sigma
            weight = np.where(outliers, 0.0, weight).astype(np.float32)
            if exclude_mask is not None:
                weight[exclude_mask > 0] = 0.0

    return bg if img.ndim == 3 else bg[:, :, 0]
