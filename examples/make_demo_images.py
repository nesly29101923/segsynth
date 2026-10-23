"""Create a small synthetic demo set (grooved metal surface with scratches and dents).

    python examples/make_demo_images.py demo_images
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segsynth.core.imageio import write_image  # noqa: E402


def surface(rng: np.random.Generator, h: int = 900, w: int = 1300) -> np.ndarray:
    x = np.arange(w, dtype=np.float32)
    zones = [(0, 0.35, 7), (0.35, 0.6, 11), (0.6, 0.78, 5), (0.78, 1.0, 15)]
    prof = np.zeros(w, np.float32)
    for a, b, period in zones:
        sel = (x >= a * w) & (x < b * w)
        prof[sel] = 14 * np.sin(2 * np.pi * x[sel] / period + rng.uniform(0, 6))
    base = 120 + prof + 18 * np.sin(x / w * 3 + rng.uniform(0, 3))
    img = np.repeat(base[None], h, 0)
    img += cv2.GaussianBlur(rng.normal(0, 6, (h, w)).astype(np.float32), (0, 0), 1.2)
    shade = np.linspace(-12, 12, h, dtype=np.float32)[:, None]
    img += shade * rng.uniform(-1, 1)
    bgr = np.dstack([img * 0.78, img * 0.92, img * 1.08])
    return np.clip(bgr, 0, 255).astype(np.uint8)


def scratch(img, rng, bright=True):
    h, w = img.shape[:2]
    x0, y0 = int(rng.uniform(0.1, 0.7) * w), int(rng.uniform(0.15, 0.85) * h)
    length, ang = rng.uniform(120, 320), rng.uniform(-0.6, 0.6)
    t = np.linspace(0, 1, 60)
    pts = np.stack([x0 + t * length * np.cos(ang), y0 + t * length * np.sin(ang) + 40 * rng.uniform(-1, 1) * (t - 0.5) ** 2 * 4], 1)
    hi = np.zeros(img.shape[:2], np.float32)
    lo = np.zeros_like(hi)
    cv2.polylines(hi, [pts.astype(np.int32)], False, 1.0, 3, cv2.LINE_AA)
    cv2.polylines(lo, [(pts + [0, 3]).astype(np.int32)], False, 1.0, 3, cv2.LINE_AA)
    out = img.astype(np.float32) + (hi * 110 - lo * 70)[:, :, None] * (1 if bright else -1)
    return np.clip(out, 0, 255).astype(np.uint8)


def dent(img, rng):
    h, w = img.shape[:2]
    cx, cy, r = int(rng.uniform(0.1, 0.9) * w), int(rng.uniform(0.1, 0.9) * h), int(rng.uniform(14, 30))
    yy, xx = np.mgrid[0:h, 0:w]
    d = ((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2
    bump = np.exp(-d) * ((yy - cy) / r)
    out = img.astype(np.float32) + (bump * 90)[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo_images")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    write_image(out / "part_01_clean.png", surface(rng))
    for i in range(2, 7):
        img = surface(rng)
        img = scratch(img, rng, bright=i % 2 == 0)
        if i % 3 == 0:
            img = dent(img, rng)
        write_image(out / f"part_{i:02d}.png", img)
    print(f"Wrote 6 demo images to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
