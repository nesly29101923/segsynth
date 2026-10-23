from __future__ import annotations

import cv2
import numpy as np
import pytest

from segsynth.core.imageio import write_image
from segsynth.core.project import Project


def striped_surface(h=240, w=320, period=6, seed=0, channels=3, dtype=np.uint8):
    rng = np.random.default_rng(seed)
    x = np.arange(w)
    stripes = 110 + 40 * np.sin(2 * np.pi * x / period)
    img = np.repeat(stripes[None, :], h, axis=0)
    img = img + rng.normal(0, 3, (h, w))
    img = np.clip(img, 0, 255)
    if dtype == np.uint16:
        img = img * 257
    img = img.astype(dtype)
    if channels == 3:
        img = np.dstack([img, (img * 0.9).astype(dtype), (img * 0.8).astype(dtype)])
    return img


def add_scratch(img, p0=(60, 120), p1=(160, 140), value=250, thickness=3):
    out = img.copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.line(mask, p0, p1, 1, thickness)
    v = value * (257 if img.dtype == np.uint16 else 1)
    out[mask > 0] = v
    return out, mask


@pytest.fixture
def scratch_image():
    img, mask = add_scratch(striped_surface())
    return img, mask


@pytest.fixture
def small_project(tmp_path):
    clean = striped_surface(seed=1)
    defect, mask = add_scratch(striped_surface(seed=2))
    write_image(tmp_path / "clean.png", clean)
    write_image(tmp_path / "defect.png", defect)
    proj = Project(name="test")
    proj.add_image(tmp_path / "clean.png")
    src = proj.add_image(tmp_path / "defect.png")
    rough = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    proj.add_defect(src.id, rough, count=5)
    proj.settings.seed = 7
    return proj
