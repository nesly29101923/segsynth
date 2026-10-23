"""Dataset generation: runs the recipe and writes synthetic images (and optional masks)."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from . import transform as tf
from .background import BackgroundMode
from .blend import BlendMode, blend
from .imageio import max_value, write_image
from .patch import DefectPatch, extract_patch
from .placement import find_position
from .project import DefectEntry, Project


@dataclass
class GeneratedSample:
    index: int
    image: np.ndarray
    mask: np.ndarray
    meta: dict
    files: dict = field(default_factory=dict)


class GenerationError(RuntimeError):
    pass


def validate(project: Project) -> list[str]:
    problems = []
    if not project.images:
        problems.append("Add at least one image.")
    if not [e for e in project.images if e.use_as_target]:
        problems.append("Select at least one image as a target background.")
    active = [d for d in project.defects if d.enabled and d.count > 0 and d.mask is not None and d.mask.any()]
    if not active:
        problems.append("Pick at least one defect and give it a count above zero.")
    shapes = set()
    for e in project.images:
        if e.use_as_target:
            img = project.load_pixels(e.id)
            shapes.add(img.ndim == 3 and img.shape[2])
    if len(shapes) > 1:
        problems.append("Target images mix grayscale and color; use one kind.")
    return problems


class Generator:
    def __init__(self, project: Project):
        self.project = project
        self.s = project.settings
        self._patches: dict[str, DefectPatch] = {}
        self._marked: dict[str, np.ndarray] = {}

    def patch(self, d: DefectEntry) -> DefectPatch:
        if d.id not in self._patches:
            img = self.project.load_pixels(d.image_id)
            self._patches[d.id] = extract_patch(img, d.mask, self.s.bg_mode, self.s.bg_size, self.s.feather)
        return self._patches[d.id]

    def _marked_mask(self, image_id: str, shape) -> np.ndarray:
        if image_id not in self._marked:
            m = self.project.marked_mask(image_id, shape)
            if m.any():
                m = cv2.dilate(m, np.ones((9, 9), np.uint8))
            self._marked[image_id] = m
        return self._marked[image_id]

    def plan(self, rng: np.random.Generator) -> list[str]:
        jobs = [d.id for d in self.project.defects if d.enabled and d.count > 0 for _ in range(d.count)]
        rng.shuffle(jobs)
        return jobs

    def _blend_mode(self, rng) -> BlendMode:
        modes = [BlendMode(m) for m in self.s.blend_modes] or [BlendMode.RESIDUAL]
        if BackgroundMode(self.s.bg_mode) == BackgroundMode.NONE:
            modes = [m for m in modes if m != BlendMode.RESIDUAL] or [BlendMode.ALPHA]
        return modes[int(rng.integers(len(modes)))]

    def _paste(self, canvas, label, occupied, d: DefectEntry, target_shape, rng, max_val):
        base = self.patch(d)
        mode = d.placement or self.s.placement_mode
        for _ in range(6):
            t = tf.sample(self.s.transform, rng)
            try:
                p = tf.apply(base, t, rng, self.s.transform.elastic_alpha, self.s.transform.elastic_sigma)
            except ValueError:
                continue
            ph, pw = p.mask.shape
            # keep the defect's center (not its new bbox) anchored to the source location
            src_center = (base.center[0], base.center[1])
            pos = find_position(p.mask, target_shape, rng, mode, src_center, self.s.jitter, occupied, None, self.s.min_gap)
            if pos is None:
                continue
            bm = self._blend_mode(rng)
            canvas = blend(canvas, p, pos, bm, max_val)
            x, y = pos
            label[y:y + ph, x:x + pw] |= p.mask
            occupied[y:y + ph, x:x + pw] |= p.mask
            return canvas, {
                "defect_id": d.id,
                "defect_name": d.name,
                "source_image": self.project.image(d.image_id).name,
                "position": [int(x), int(y)],
                "size": [int(pw), int(ph)],
                "blend": bm.value,
                "placement": mode,
                "transform": t.to_dict(),
            }
        return canvas, None

    def generate(
        self,
        stop: threading.Event | None = None,
    ) -> Iterator[GeneratedSample]:
        problems = validate(self.project)
        if problems:
            raise GenerationError(" ".join(problems))
        rng = np.random.default_rng(self.s.seed)
        targets = [e for e in self.project.images if e.use_as_target]
        active = [d for d in self.project.defects if d.enabled and d.mask is not None and d.mask.any()]
        lo, hi = sorted(int(v) for v in self.s.defects_per_image)
        lo = max(1, lo)
        hi = max(lo, hi)

        for index, defect_id in enumerate(self.plan(rng)):
            if stop is not None and stop.is_set():
                return
            primary = self.project.defect(defect_id)
            sample = None
            for _ in range(4):
                tgt = targets[int(rng.integers(len(targets)))]
                img = self.project.load_pixels(tgt.id)
                shape = img.shape[:2]
                occupied = self._marked_mask(tgt.id, shape).copy() if self.s.avoid_marked_defects \
                    else np.zeros(shape, np.uint8)
                label = np.zeros(shape, np.uint8)
                canvas = img
                canvas, info = self._paste(canvas, label, occupied, primary, shape, rng, max_value(img))
                if info is None:
                    continue
                placed = [info]
                for _ in range(int(rng.integers(lo, hi + 1)) - 1):
                    extra = active[int(rng.integers(len(active)))]
                    canvas, info = self._paste(canvas, label, occupied, extra, shape, rng, max_value(img))
                    if info is not None:
                        placed.append(info)
                sample = GeneratedSample(
                    index=index,
                    image=canvas,
                    mask=label,
                    meta={"target_image": tgt.name, "defects": placed},
                )
                break
            if sample is None:
                raise GenerationError(
                    f"Could not place '{primary.name}' on any target. Reduce scale, jitter or gap, "
                    f"or use larger target images."
                )
            yield sample


def run_to_folder(
    project: Project,
    out_dir: str | Path,
    progress: Callable[[int, int, GeneratedSample], None] | None = None,
    stop: threading.Event | None = None,
) -> dict:
    s = project.settings
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    if s.save_masks:
        (out / "masks").mkdir(parents=True, exist_ok=True)
    if s.save_overlays:
        (out / "overlays").mkdir(parents=True, exist_ok=True)

    total = project.total_images
    records = []
    started = time.time()
    gen = Generator(project)
    for sample in gen.generate(stop):
        stem = f"{s.name_prefix}_{sample.index + 1:05d}"
        img_path = out / "images" / f"{stem}{s.output_ext}"
        write_image(img_path, sample.image)
        sample.files["image"] = str(img_path.relative_to(out))
        if s.save_masks:
            mpath = out / "masks" / f"{stem}.png"
            write_image(mpath, sample.mask * 255)
            sample.files["mask"] = str(mpath.relative_to(out))
        if s.save_overlays:
            opath = out / "overlays" / f"{stem}.jpg"
            write_image(opath, overlay(sample.image, sample.mask))
            sample.files["overlay"] = str(opath.relative_to(out))
        records.append({"file": sample.files["image"], **sample.meta})
        if progress:
            progress(len(records), total, sample)

    manifest = {
        "generator": "segsynth",
        "project": project.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": round(time.time() - started, 2),
        "count": len(records),
        "completed": len(records) == total,
        "settings": s.to_dict(),
        "samples": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def overlay(image: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha: float = 0.45) -> np.ndarray:
    img = image
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = img.copy()
    m = mask > 0
    out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out
