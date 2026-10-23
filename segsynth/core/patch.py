"""Defect patch extraction: turns a user mask on a source image into a reusable patch."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .background import BackgroundMode, estimate_background


@dataclass
class DefectPatch:
    residual: np.ndarray      # float32 HxWxC, defect minus local background
    pixels: np.ndarray        # float32 HxWxC, raw source pixels
    alpha: np.ndarray         # float32 HxW in [0,1], feathered blend weight
    mask: np.ndarray          # uint8 HxW {0,1}, hard defect mask (label)
    origin: tuple[int, int]   # (x, y) of the patch top-left in the source image
    source_shape: tuple[int, int]

    @property
    def size(self) -> tuple[int, int]:
        return self.mask.shape[1], self.mask.shape[0]

    @property
    def center(self) -> tuple[float, float]:
        h, w = self.mask.shape
        return self.origin[0] + w / 2.0, self.origin[1] + h / 2.0


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def soft_alpha(mask: np.ndarray, feather: int) -> np.ndarray:
    m = (mask > 0).astype(np.float32)
    if feather <= 0:
        return m
    k = feather * 2 + 1
    grown = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    alpha = cv2.GaussianBlur(grown, (0, 0), sigmaX=max(0.5, feather / 2.0))
    return np.maximum(alpha, m)


def extract_patch(
    image: np.ndarray,
    mask: np.ndarray,
    bg_mode: BackgroundMode | str = BackgroundMode.VERTICAL,
    bg_size: int = 151,
    feather: int = 4,
) -> DefectPatch:
    bbox = mask_bbox(mask)
    if bbox is None:
        raise ValueError("Defect mask is empty")
    h, w = mask.shape
    margin = feather * 3 + 2
    x0, y0 = max(0, bbox[0] - margin), max(0, bbox[1] - margin)
    x1, y1 = min(w, bbox[2] + margin), min(h, bbox[3] + margin)

    ctx = int(bg_size)
    cx0, cy0 = max(0, x0 - ctx), max(0, y0 - ctx)
    cx1, cy1 = min(w, x1 + ctx), min(h, y1 + ctx)

    img = image if image.ndim == 3 else image[:, :, None]
    ctx_img = img[cy0:cy1, cx0:cx1]
    ctx_mask = mask[cy0:cy1, cx0:cx1]
    exclude = cv2.dilate((ctx_mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    bg = estimate_background(ctx_img, bg_mode, bg_size, exclude_mask=exclude)
    if bg.ndim == 2:
        bg = bg[:, :, None]

    sl = (slice(y0 - cy0, y1 - cy0), slice(x0 - cx0, x1 - cx0))
    pixels = ctx_img[sl].astype(np.float32)
    residual = pixels - bg[sl]
    pmask = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    return DefectPatch(
        residual=residual.astype(np.float32),
        pixels=pixels,
        alpha=soft_alpha(pmask, feather),
        mask=pmask,
        origin=(x0, y0),
        source_shape=(h, w),
    )
