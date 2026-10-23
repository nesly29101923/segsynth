"""Unicode-safe image I/O supporting 8/16-bit grayscale and color images."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")


def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype not in (np.uint8, np.uint16):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def read_mask(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    m = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"Cannot read mask: {path}")
    return (m > 0).astype(np.uint8)


def write_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Cannot encode image: {path}")
    buf.tofile(str(path))


def max_value(img: np.ndarray) -> float:
    return 65535.0 if img.dtype == np.uint16 else 255.0


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
