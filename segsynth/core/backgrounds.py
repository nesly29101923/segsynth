"""Background generation: cut source images into pieces, drop the pieces marked not good,
and stitch the remaining pieces from different images into new clean backgrounds.

Pieces keep their slot (a top piece is only used on top) because lighting and part
geometry depend on position. Before stitching, every source is aligned to the first one
along the seam, and neighbouring pieces are cross-faded over a band so no seam line appears.
"""
from __future__ import annotations

import itertools
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

HORIZONTAL = "horizontal"   # horizontal cut lines: pieces stacked top to bottom
VERTICAL = "vertical"       # vertical cut lines: pieces side by side
ENUMERATE_LIMIT = 200_000


@dataclass
class BackgroundSettings:
    enabled: bool = False
    direction: str = HORIZONTAL
    pieces: int = 2
    rejected: list = field(default_factory=list)   # [[image_id, slot], ...]
    seam: int = 120
    align: bool = True
    max_count: int = 200
    include_in_dataset: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BackgroundSettings":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        known["rejected"] = [list(x) for x in known.get("rejected", [])]
        return cls(**known)

    def is_rejected(self, image_id: str, slot: int) -> bool:
        return [image_id, slot] in self.rejected

    def set_rejected(self, image_id: str, slot: int, value: bool) -> None:
        key = [image_id, slot]
        if value and key not in self.rejected:
            self.rejected.append(key)
        elif not value and key in self.rejected:
            self.rejected.remove(key)


def piece_bounds(length: int, pieces: int) -> list[tuple[int, int]]:
    edges = [round(i * length / pieces) for i in range(pieces + 1)]
    return list(zip(edges[:-1], edges[1:]))


def slot_weights(length: int, pieces: int, seam: int) -> list[np.ndarray]:
    """Per-slot weights along the stacking axis; they sum to 1 everywhere."""
    bounds = piece_bounds(length, pieces)
    seam = max(0, min(int(seam), min(e - s for s, e in bounds)))
    t = np.arange(length, dtype=np.float32) + 0.5
    ramps = []
    for _, cut in bounds[:-1]:
        if seam == 0:
            ramps.append((t >= cut).astype(np.float32))
        else:
            ramps.append(np.clip((t - (cut - seam / 2.0)) / seam, 0.0, 1.0))
    weights = []
    for i in range(pieces):
        lo = ramps[i - 1] if i > 0 else np.ones(length, np.float32)
        hi = ramps[i] if i < pieces - 1 else np.zeros(length, np.float32)
        weights.append(lo - hi)
    return weights


def _profile(img: np.ndarray) -> np.ndarray:
    g = img.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    p = g.mean(axis=0)
    return p - cv2.GaussianBlur(p.reshape(1, -1), (0, 0), 15).ravel()


def estimate_shift(ref: np.ndarray, img: np.ndarray, max_shift: int) -> int:
    """Shift (along the image width) that best aligns img to ref, 0 when no reliable match."""
    a, b = _profile(ref), _profile(img)
    m = max_shift + 1
    if a.size <= 4 * m:
        return 0
    core = a[m:-m]
    if core.std() < 1e-3:
        return 0
    scores = []
    for s in range(-max_shift, max_shift + 1):
        seg = b[m - s:b.size - m - s]
        scores.append(float(np.dot(core, seg)) / (np.linalg.norm(core) * np.linalg.norm(seg) + 1e-9))
    best = int(np.argmax(scores))
    return best - max_shift if scores[best] > 0.15 else 0


class BackgroundComposer:
    """Turns a project's images and BackgroundSettings into a list of new backgrounds."""

    def __init__(self, project, seed: int | None = None):
        self.project = project
        self.s: BackgroundSettings = project.backgrounds
        self.seed = project.settings.seed if seed is None else seed
        self._shifts: dict[str, int] = {}
        self._small: dict[tuple, np.ndarray] = {}
        self._combos: list[tuple] | None = None
        self._render_cache: OrderedDict = OrderedDict()
        self.sources = self._compatible_sources()

    # sources -----------------------------------------------------------
    def _compatible_sources(self) -> list:
        imgs = self.project.images
        if not imgs:
            return []
        ref = self.project.load_pixels(imgs[0].id)
        out = []
        for e in imgs:
            px = self.project.load_pixels(e.id)
            if px.shape == ref.shape and px.dtype == ref.dtype:
                out.append(e)
        return out

    @property
    def skipped(self) -> list:
        ids = {e.id for e in self.sources}
        return [e for e in self.project.images if e.id not in ids]

    @property
    def shape(self) -> tuple[int, ...] | None:
        return self.project.load_pixels(self.sources[0].id).shape if self.sources else None

    def _oriented(self, img: np.ndarray) -> np.ndarray:
        """View the image so that pieces are always stacked along axis 0."""
        return img if self.s.direction == HORIZONTAL else np.swapaxes(img, 0, 1)

    def bounds(self) -> list[tuple[int, int]]:
        if not self.sources:
            return []
        length = self._oriented(self.project.load_pixels(self.sources[0].id)).shape[0]
        return piece_bounds(length, self.pieces)

    @property
    def pieces(self) -> int:
        return max(2, int(self.s.pieces))

    def shift(self, image_id: str) -> int:
        if not self.s.align or not self.sources or image_id == self.sources[0].id:
            return 0
        if image_id not in self._shifts:
            ref = self._oriented(self.project.load_pixels(self.sources[0].id))
            img = self._oriented(self.project.load_pixels(image_id))
            max_shift = max(8, int(0.02 * ref.shape[1]))
            self._shifts[image_id] = estimate_shift(ref, img, max_shift)
        return self._shifts[image_id]

    # selection ---------------------------------------------------------
    def good(self, slot: int) -> list[str]:
        return [e.id for e in self.sources if not self.s.is_rejected(e.id, slot)]

    def counts(self) -> tuple[int, int]:
        """(good pieces, total pieces)."""
        n = self.pieces
        total = len(self.sources) * n
        good = sum(len(self.good(i)) for i in range(n))
        return good, total

    def possible(self) -> int:
        slots = [self.good(i) for i in range(self.pieces)]
        if any(not s for s in slots):
            return 0
        total = math.prod(len(s) for s in slots)
        same = set(slots[0]).intersection(*slots[1:])
        return total - len(same)

    def combos(self) -> list[tuple]:
        if self._combos is not None:
            return self._combos
        slots = [self.good(i) for i in range(self.pieces)]
        cap = max(0, int(self.s.max_count))
        rng = np.random.default_rng(self.seed + 7919)
        out: list[tuple] = []
        if cap and all(slots) and self.possible() > 0:
            possible = self.possible()
            if math.prod(len(s) for s in slots) <= ENUMERATE_LIMIT:
                every = [c for c in itertools.product(*slots) if len(set(c)) > 1]
                if len(every) > cap:
                    keep = np.sort(rng.choice(len(every), size=cap, replace=False))
                    every = [every[i] for i in keep]
                out = every
            else:
                seen: set = set()
                target = min(cap, possible)
                while len(seen) < target:
                    c = tuple(s[int(rng.integers(len(s)))] for s in slots)
                    if len(set(c)) > 1:
                        seen.add(c)
                out = sorted(seen, key=lambda c: [self._order(i) for i in c])
        self._combos = out
        return out

    def _order(self, image_id: str) -> int:
        return next(i for i, e in enumerate(self.sources) if e.id == image_id)

    def name(self, index: int) -> str:
        return f"bg_{index + 1:05d}"

    def describe(self, combo: tuple) -> str:
        labels = ["top", "bottom"] if self.s.direction == HORIZONTAL else ["left", "right"]
        n = len(combo)
        parts = []
        for slot, image_id in enumerate(combo):
            where = labels[slot] if n == 2 else f"piece {slot + 1}"
            parts.append(f"{self.project.image(image_id).name} ({where})")
        return " + ".join(parts)

    # rendering ---------------------------------------------------------
    def _source(self, image_id: str, scale: float) -> np.ndarray:
        img = self.project.load_pixels(image_id)
        if scale >= 1.0:
            return img
        key = (image_id, round(scale, 4))
        if key not in self._small:
            h, w = img.shape[:2]
            self._small[key] = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                                          interpolation=cv2.INTER_AREA)
        return self._small[key]

    @staticmethod
    def _shift_cols(a: np.ndarray, dx: float) -> np.ndarray:
        if abs(dx) < 1e-3:
            return a
        h, w = a.shape[:2]
        m = np.float32([[1, 0, dx], [0, 1, 0]])
        interp = cv2.INTER_NEAREST if a.dtype == np.uint8 and a.ndim == 2 else cv2.INTER_LINEAR
        return cv2.warpAffine(a, m, (w, h), flags=interp, borderMode=cv2.BORDER_REFLECT)

    def render(self, combo: tuple, scale: float = 1.0, with_mask: bool = True):
        """Return (image, mask) for a combo; mask marks defects picked on the used sources."""
        key = (combo, round(scale, 4), with_mask)
        if key in self._render_cache:
            self._render_cache.move_to_end(key)
            return self._render_cache[key]
        first = self._oriented(self._source(combo[0], scale))
        length = first.shape[0]
        seam = max(0, round(self.s.seam * scale))
        weights = slot_weights(length, len(combo), seam)
        dtype = first.dtype
        acc = np.zeros(first.shape, np.float32)
        mask = np.zeros(first.shape[:2], np.uint8) if with_mask else None
        for slot, image_id in enumerate(combo):
            w = weights[slot]
            rows = np.nonzero(w > 0)[0]
            if rows.size == 0:
                continue
            r0, r1 = int(rows[0]), int(rows[-1]) + 1
            src = self._oriented(self._source(image_id, scale))[r0:r1]
            src = self._shift_cols(np.ascontiguousarray(src), self.shift(image_id) * scale)
            wv = w[r0:r1].reshape(-1, *([1] * (src.ndim - 1)))
            acc[r0:r1] += src.astype(np.float32) * wv
            if with_mask:
                full = self.project.marked_mask(image_id, self.project.load_pixels(image_id).shape[:2])
                if full.any():
                    m = full if scale >= 1.0 else cv2.resize(full, (self._source(image_id, scale).shape[1],
                                                                   self._source(image_id, scale).shape[0]),
                                                             interpolation=cv2.INTER_NEAREST)
                    m = self._shift_cols(np.ascontiguousarray(self._oriented(m)[r0:r1]), self.shift(image_id) * scale)
                    keep = (w[r0:r1] > 0.1)[:, None]
                    mask[r0:r1] |= (m > 0) & keep
        maxv = 65535.0 if dtype == np.uint16 else 255.0
        img = np.clip(np.rint(acc), 0, maxv).astype(dtype)
        if self.s.direction != HORIZONTAL:
            img = np.ascontiguousarray(np.swapaxes(img, 0, 1))
            if mask is not None:
                mask = np.ascontiguousarray(mask.T)
        result = (img, mask)
        self._render_cache[key] = result
        while len(self._render_cache) > 6:
            self._render_cache.popitem(last=False)
        return result


def split_view(img: np.ndarray, direction: str, pieces: int) -> list[np.ndarray]:
    """Cut an image (any scale) into its pieces, in slot order."""
    o = img if direction == HORIZONTAL else np.swapaxes(img, 0, 1)
    crops = [o[s:e] for s, e in piece_bounds(o.shape[0], max(2, int(pieces)))]
    return [np.ascontiguousarray(c if direction == HORIZONTAL else np.swapaxes(c, 0, 1)) for c in crops]
