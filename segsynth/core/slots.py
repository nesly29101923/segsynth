"""Slots: fixed positions (pockets, holes, sockets) that pieces are placed into precisely.

A user boxes one pocket; every similar pocket is found by template matching, the regular
grid they form is fitted, and grid positions without a match (for example empty pockets
that look different) are added as well. Each slot keeps its match score so empty pockets
(low score against a filled template) can be selected in one click.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .patch import DefectPatch, soft_alpha


@dataclass
class Slot:
    x: float
    y: float
    r: float
    target: bool = False
    score: float = 1.0

    def to_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Slot":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def contains(self, x: float, y: float, scale: float = 1.0) -> bool:
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= (self.r * scale) ** 2


def to_gray8(img: np.ndarray) -> np.ndarray:
    g = img
    if g.ndim == 3:
        g = cv2.cvtColor(np.ascontiguousarray(g[:, :, :3]), cv2.COLOR_BGR2GRAY) if g.shape[2] >= 3 else g[:, :, 0]
    if g.dtype != np.uint8:
        g = cv2.normalize(g.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.ascontiguousarray(g)


def _peaks(res: np.ndarray, threshold: float, dist: int) -> list[tuple[int, int, float]]:
    k = max(3, dist | 1)
    local = cv2.dilate(res, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ys, xs = np.nonzero((res >= local) & (res >= threshold))
    pts = sorted(((int(x), int(y), float(res[y, x])) for x, y in zip(xs, ys)), key=lambda p: -p[2])
    kept: list[tuple[int, int, float]] = []
    for p in pts:
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= (dist * 0.8) ** 2 for q in kept):
            kept.append(p)
    return kept


def _score_at(gray: np.ndarray, tpl: np.ndarray, cx: float, cy: float, search: int) -> tuple[float, float, float]:
    """Best match of tpl centred near (cx, cy). Returns (cx, cy, score)."""
    th, tw = tpl.shape
    h, w = gray.shape
    x0 = int(round(cx - tw / 2)) - search
    y0 = int(round(cy - th / 2)) - search
    x1, y1 = x0 + tw + 2 * search, y0 + th + 2 * search
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return cx, cy, 0.0
    res = cv2.matchTemplate(gray[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
    _, v, _, (bx, by) = cv2.minMaxLoc(res)
    return x0 + bx + tw / 2.0, y0 + by + th / 2.0, float(v)


def fit_lattice(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit p = o + i*a + j*b to the points. Returns (o, a, b) or None."""
    if len(points) < 4:
        return None
    d = points[:, None, :] - points[None, :, :]
    dist = np.hypot(d[..., 0], d[..., 1])
    np.fill_diagonal(dist, np.inf)
    nn = np.median(dist.min(axis=1))
    vecs = d[(dist < nn * 1.3)]
    if len(vecs) < 2:
        return None
    horiz = vecs[np.abs(vecs[:, 0]) >= np.abs(vecs[:, 1])]
    vert = vecs[np.abs(vecs[:, 0]) < np.abs(vecs[:, 1])]
    if len(horiz) == 0 or len(vert) == 0:
        return None
    horiz = np.where(horiz[:, :1] < 0, -horiz, horiz)
    vert = np.where(vert[:, 1:] < 0, -vert, vert)
    a, b = np.median(horiz, axis=0), np.median(vert, axis=0)
    B = np.stack([a, b], axis=1)
    if abs(np.linalg.det(B)) < 1e-6:
        return None
    o = points[0]
    for _ in range(2):
        ij = np.rint(np.linalg.solve(B, (points - o).T).T)
        A = np.column_stack([np.ones(len(ij)), ij])
        sol, *_ = np.linalg.lstsq(A, points, rcond=None)
        o, a, b = sol[0], sol[1], sol[2]
        B = np.stack([a, b], axis=1)
    return o, a, b


def detect_slots(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    threshold: float = 0.5,
    complete_grid: bool = True,
) -> list[Slot]:
    """Find every pocket similar to the one inside box (x0, y0, x1, y1)."""
    gray = to_gray8(image)
    h, w = gray.shape
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    tw, th = x1 - x0, y1 - y0
    if tw < 6 or th < 6:
        raise ValueError("Draw a box around one whole pocket first.")
    return detect_with_template(gray, gray[y0:y1, x0:x1].copy(), threshold, complete_grid)


def template_from(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    gray = to_gray8(image)
    h, w = gray.shape
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    return gray[y0:y1, x0:x1].copy()


def detect_with_template(
    image: np.ndarray,
    tpl: np.ndarray,
    threshold: float = 0.5,
    complete_grid: bool = True,
) -> list[Slot]:
    """Find every pocket that looks like the template (a grayscale 8-bit crop of one pocket)."""
    gray = to_gray8(image)
    h, w = gray.shape
    th, tw = tpl.shape
    if tw < 6 or th < 6:
        raise ValueError("Draw a box around one whole pocket first.")
    if th > h or tw > w:
        raise ValueError("The pocket box is larger than the image.")
    if float(tpl.std()) < 1.0:
        raise ValueError("The box has no contrast; draw it around a pocket.")
    radius = min(tw, th) / 2.0

    f = min(1.0, 40.0 / min(tw, th))
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray
    stpl = cv2.resize(tpl, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else tpl
    res = cv2.matchTemplate(small, stpl, cv2.TM_CCOEFF_NORMED)
    sth, stw = stpl.shape
    search = int(math.ceil(2 / f)) + 2
    found = []
    for px, py, _ in _peaks(res, threshold, int(min(stw, sth) * 0.8)):
        cx, cy = (px + stw / 2.0) / f, (py + sth / 2.0) / f
        cx, cy, s = _score_at(gray, tpl, cx, cy, search)
        if s >= threshold:
            found.append((cx, cy, s))
    found = _dedupe(found, radius)
    slots = [Slot(x, y, radius, False, s) for x, y, s in found]
    if not complete_grid or len(slots) < 4:
        return _sorted(slots)

    pts = np.array([(s.x, s.y) for s in slots])
    lat = fit_lattice(pts)
    if lat is None:
        return _sorted(slots)
    o, a, b = lat
    B = np.stack([a, b], axis=1)
    ij = np.rint(np.linalg.solve(B, (pts - o).T).T).astype(int)
    pitch = min(np.hypot(*a), np.hypot(*b))
    # Rows/columns with only stray matches (screws, holes outside the tray) are not grid.
    rows, cols = _supported(ij[:, 1]), _supported(ij[:, 0])
    keep = np.isin(ij[:, 1], rows) & np.isin(ij[:, 0], cols)
    slots = [s for s, k in zip(slots, keep) if k]
    ij = ij[keep]
    if len(ij) == 0:
        return []
    taken = {tuple(v) for v in ij}
    for i in cols:
        for j in rows:
            if (i, j) in taken:
                continue
            cx, cy = o + i * a + j * b
            if not (radius <= cx <= w - radius and radius <= cy <= h - radius):
                continue
            if any((s.x - cx) ** 2 + (s.y - cy) ** 2 < (pitch * 0.4) ** 2 for s in slots):
                continue
            _, _, score = _score_at(gray, tpl, cx, cy, 0)
            slots.append(Slot(float(cx), float(cy), radius, False, score))
    return _sorted(slots)


def _supported(idx: np.ndarray) -> list[int]:
    """Contiguous run of grid indices around the densest one, each with enough matches."""
    vals, counts = np.unique(idx, return_counts=True)
    need = max(2, 0.25 * counts.max())
    ok = {int(v) for v, c in zip(vals, counts) if c >= need}
    best = int(vals[np.argmax(counts)])
    lo = hi = best
    # grow while the next index (or the one after, to bridge one mostly-empty row) is supported
    while lo - 1 in ok or lo - 2 in ok:
        lo -= 1
    while hi + 1 in ok or hi + 2 in ok:
        hi += 1
    return list(range(lo, hi + 1))


def _dedupe(found: list[tuple[float, float, float]], radius: float) -> list[tuple[float, float, float]]:
    kept: list[tuple[float, float, float]] = []
    for p in sorted(found, key=lambda p: -p[2]):
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= radius ** 2 for q in kept):
            kept.append(p)
    return kept


def _sorted(slots: list[Slot]) -> list[Slot]:
    if not slots:
        return slots
    r = slots[0].r
    return sorted(slots, key=lambda s: (round(s.y / max(r, 1)), s.x))


def anchor_for(mask: np.ndarray, slots: list[Slot]) -> tuple[float, float]:
    """Centre a picked piece is placed by: its slot centre if it sits in one, else its centroid."""
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        raise ValueError("Empty mask")
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    inside = [s for s in slots if s.contains(cx, cy)]
    if inside:
        s = min(inside, key=lambda s: (s.x - cx) ** 2 + (s.y - cy) ** 2)
        return s.x, s.y
    return cx, cy


def slot_patch(image: np.ndarray, mask: np.ndarray, center: tuple[float, float], feather: int) -> DefectPatch:
    """Square patch whose centre pixel is exactly `center`, big enough to rotate freely."""
    cx, cy = int(round(center[0])), int(round(center[1]))
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("Empty mask")
    half = int(math.ceil(np.hypot(xs - cx, ys - cy).max())) + feather * 3 + 3
    size = 2 * half + 1
    h, w = mask.shape
    img = image if image.ndim == 3 else image[:, :, None]
    x0, y0 = cx - half, cy - half
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + size), min(h, y0 + size)
    pixels = np.zeros((size, size, img.shape[2]), np.float32)
    pmask = np.zeros((size, size), np.uint8)
    pixels[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    pmask[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = mask[sy0:sy1, sx0:sx1] > 0
    return DefectPatch(
        residual=np.zeros_like(pixels),
        pixels=pixels,
        alpha=soft_alpha(pmask, feather),
        mask=pmask,
        origin=(x0, y0),
        source_shape=(h, w),
    )


def turn_patch(p: DefectPatch, angle: float, contrast: float = 1.0) -> DefectPatch:
    """Rotate about the patch centre (keeps size, so the centre stays the anchor)."""
    size = p.mask.shape[0]
    pixels = p.pixels
    if abs(contrast - 1.0) > 1e-6:
        m = p.mask > 0
        mean = pixels[m].mean(axis=0) if m.any() else 0.0
        pixels = mean + (pixels - mean) * contrast
    if abs(angle) > 1e-6:
        c = (size - 1) / 2.0
        M = cv2.getRotationMatrix2D((c, c), angle, 1.0)

        def warp(a, interp):
            out = cv2.warpAffine(a, M, (size, size), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return out

        pix = warp(pixels, cv2.INTER_LINEAR)
        if pix.ndim == 2:
            pix = pix[:, :, None]
        pixels = pix
        alpha = warp(p.alpha, cv2.INTER_LINEAR)
        mask = (warp(p.mask.astype(np.float32), cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    else:
        alpha, mask = p.alpha, p.mask
    return DefectPatch(
        residual=np.zeros_like(pixels, dtype=np.float32),
        pixels=pixels.astype(np.float32),
        alpha=np.clip(alpha, 0, 1).astype(np.float32),
        mask=mask,
        origin=p.origin,
        source_shape=p.source_shape,
    )
