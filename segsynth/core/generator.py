"""Dataset generation: runs the recipe and writes synthetic images (and optional masks)."""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from . import transform as tf
from .background import BackgroundMode
from .backgrounds import BackgroundComposer
from .blend import BlendMode, blend
from .imageio import max_value, write_image
from .export import export_dataset, label_path
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
    pasted: np.ndarray | None = None


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Target:
    key: str
    name: str
    image_id: str | None = None
    combo: tuple | None = None

    @property
    def generated(self) -> bool:
        return self.combo is not None


def make_composer(project: Project) -> BackgroundComposer | None:
    return BackgroundComposer(project) if project.backgrounds.enabled and project.images else None


def list_targets(project: Project, composer: BackgroundComposer | None = None) -> list[Target]:
    targets = [Target(e.id, e.name, image_id=e.id) for e in project.images if e.use_as_target]
    if composer is not None:
        targets += [Target(f"bg:{i}", f"{composer.name(i)}.png", combo=c) for i, c in enumerate(composer.combos())]
    return targets


def validate(project: Project) -> list[str]:
    problems = []
    if not project.images:
        problems.append("Add at least one image.")
    composer = make_composer(project)
    targets = list_targets(project, composer)
    if not targets:
        if project.backgrounds.enabled:
            problems.append("Select a background image, or keep at least one good piece in every slot "
                            "so backgrounds can be created.")
        else:
            problems.append("Select at least one image as a target background.")
    active = [d for d in project.defects if d.enabled and d.count > 0 and d.mask is not None and d.mask.any()]
    if not active:
        problems.append("Pick at least one defect and give it a count above zero.")
    shapes = set()
    for e in project.images:
        if e.use_as_target:
            img = project.load_pixels(e.id)
            shapes.add(img.ndim == 3 and img.shape[2])
    if composer is not None and composer.combos():
        shp = composer.shape
        shapes.add(len(shp) == 3 and shp[2])
    if len(shapes) > 1:
        problems.append("Target images mix grayscale and color; use one kind.")
    return problems


class Generator:
    def __init__(self, project: Project):
        self.project = project
        self.s = project.settings
        self._patches: dict[str, DefectPatch] = {}
        self._marked: dict[str, np.ndarray] = {}
        self.composer = make_composer(project)
        self.targets = list_targets(project, self.composer)

    def patch(self, d: DefectEntry) -> DefectPatch:
        if d.id not in self._patches:
            img = self.project.load_pixels(d.image_id)
            self._patches[d.id] = extract_patch(img, d.mask, self.s.bg_mode, self.s.bg_size, self.s.feather)
        return self._patches[d.id]

    def pixels(self, t: Target) -> np.ndarray:
        if t.generated:
            return self.composer.render(t.combo)[0]
        return self.project.load_pixels(t.image_id)

    def labels(self, t: Target, shape) -> np.ndarray:
        """Defects already present on the target (picked by the user)."""
        if t.generated:
            return self.composer.render(t.combo)[1].copy()
        return self.project.marked_mask(t.image_id, shape)

    def _marked_mask(self, t: Target, shape) -> np.ndarray:
        if t.key in self._marked:
            return self._marked[t.key]
        m = self.labels(t, shape)
        if m.any():
            m = cv2.dilate(m, np.ones((9, 9), np.uint8))
        if not t.generated or not m.any():
            self._marked[t.key] = m
        return m

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
        targets = self.targets
        active = [d for d in self.project.defects if d.enabled and d.mask is not None and d.mask.any()]
        lo, hi = sorted(int(v) for v in self.s.defects_per_image)
        lo = max(1, lo)
        hi = max(lo, hi)

        cycle: list = []

        def next_target():
            if not cycle:
                cycle.extend(targets[i] for i in rng.permutation(len(targets)))
            return cycle.pop()

        for index, defect_id in enumerate(self.plan(rng)):
            if stop is not None and stop.is_set():
                return
            primary = self.project.defect(defect_id)
            sample = None
            for _ in range(max(4, len(targets))):
                tgt = next_target()
                img = self.pixels(tgt)
                shape = img.shape[:2]
                occupied = self._marked_mask(tgt, shape).copy() if self.s.avoid_marked_defects \
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
                pasted = label.copy()
                label |= self.labels(tgt, shape)
                meta = {"target_image": tgt.name, "defects": placed}
                if tgt.generated:
                    meta["target_pieces"] = self.composer.describe(tgt.combo)
                sample = GeneratedSample(
                    index=index,
                    image=canvas,
                    mask=label,
                    pasted=pasted,
                    meta=meta,
                )
                break
            if sample is None:
                raise GenerationError(
                    f"Could not place '{primary.name}' on any target. Reduce scale, jitter or gap, "
                    f"or use larger target images."
                )
            yield sample


SYNTHETIC_DIR = "synthetic"
BACKGROUND_DIR = "background"


def run_to_folder(
    project: Project,
    out_dir: str | Path,
    progress: Callable[[int, int, GeneratedSample], None] | None = None,
    stop: threading.Event | None = None,
    status: Callable[[str], None] | None = None,
    labeltool: bool | None = None,
) -> dict:
    """Generate the dataset into out_dir.

    Without a database: synthetic/, background/, manifest.json and a hidden .labels/ folder
    with one mask per image (used to build a database later). With labeltool (default: the
    project setting) out_dir becomes an AI tool workspace: DATA/, DATA_THUMB/, META/<db>.sqlite.
    """
    s = project.settings
    if not (s.save_labeltool_db if labeltool is None else labeltool):
        return _generate(project, Path(out_dir), progress, stop, status)
    with tempfile.TemporaryDirectory(prefix="segsynth_") as tmp:
        manifest = _generate(project, Path(tmp), progress, stop, status)
        if status:
            status("Writing label tool database…")
        return export_dataset(tmp, out_dir, manifest, project.name, True, s.labeltool_class_name,
                              s.labeltool_class_color, s.labeltool_validate_percent, s.seed)


def _generate(
    project: Project,
    out_dir: Path,
    progress: Callable[[int, int, GeneratedSample], None] | None = None,
    stop: threading.Event | None = None,
    status: Callable[[str], None] | None = None,
) -> dict:
    s = project.settings
    out = Path(out_dir)
    (out / SYNTHETIC_DIR).mkdir(parents=True, exist_ok=True)
    if s.save_masks:
        (out / "masks").mkdir(parents=True, exist_ok=True)
    if s.save_overlays:
        (out / "overlays").mkdir(parents=True, exist_ok=True)

    total = project.total_images
    records: list[dict] = []
    files: list[dict] = []
    started = time.time()
    gen = Generator(project)
    sources: list[dict] = []
    backgrounds: list[dict] = []
    if s.labeltool_include_sources:
        if status:
            status("Copying source images…")
        sources = _write_sources(project, s, out, files)
    if gen.composer is not None and project.backgrounds.include_in_dataset:
        backgrounds = _write_backgrounds(gen, s, out, files, status, stop)
    if status:
        status("Preparing defects…")
    _write_samples(gen, s, out, files, records, total, progress, stop)

    manifest = {
        "generator": "segsynth",
        "project": project.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": round(time.time() - started, 2),
        "count": len(records),
        "completed": len(records) == total,
        "settings": s.to_dict(),
        "labeltool_db": None,
        "files": files,
        "source_images": sources,
        "generated_backgrounds": backgrounds,
        "samples": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _store(out: Path, s, rel: str, mask: np.ndarray, kind: str, files: list[dict]) -> None:
    write_image(label_path(out, rel), mask.astype(np.uint8) * 255)
    if s.save_masks:
        write_image(out / "masks" / f"{Path(rel).stem}.png", mask.astype(np.uint8) * 255)
    files.append({"file": rel, "kind": kind, "label": "bad" if mask.any() else "good"})


def _write_sources(project: Project, s, out: Path, files: list[dict]) -> list[dict]:
    """Copy every source image into background/; good unless defects were picked on it."""
    written = []
    used = set()
    (out / BACKGROUND_DIR).mkdir(parents=True, exist_ok=True)
    for entry in project.images:
        src = Path(entry.path)
        name = f"real_{src.stem}{src.suffix.lower()}"
        n = 2
        while name.lower() in used:
            name = f"real_{src.stem}_{n}{src.suffix.lower()}"
            n += 1
        used.add(name.lower())
        dest = out / BACKGROUND_DIR / name
        shutil.copyfile(src, dest)
        img = project.load_pixels(entry.id)
        mask = project.marked_mask(entry.id, img.shape[:2])
        rel = dest.relative_to(out).as_posix()
        _store(out, s, rel, mask, "source", files)
        written.append({"file": rel, "source": src.name, "label": files[-1]["label"]})
    return written


def _write_backgrounds(gen: Generator, s, out: Path, files: list[dict], status, stop) -> list[dict]:
    """Write every generated background into background/."""
    written = []
    (out / BACKGROUND_DIR).mkdir(parents=True, exist_ok=True)
    generated = [t for t in gen.targets if t.generated]
    for i, t in enumerate(generated):
        if stop is not None and stop.is_set():
            break
        if status and i % 5 == 0:
            status(f"Writing generated backgrounds… {i}/{len(generated)}")
        img, mask = gen.composer.render(t.combo)
        path = out / BACKGROUND_DIR / f"{Path(t.name).stem}{s.output_ext}"
        write_image(path, img)
        rel = path.relative_to(out).as_posix()
        _store(out, s, rel, mask, "background", files)
        written.append({"file": rel, "pieces": gen.composer.describe(t.combo), "label": files[-1]["label"]})
    return written


def _write_samples(gen, s, out: Path, files: list[dict], records: list, total: int, progress, stop) -> None:
    for sample in gen.generate(stop):
        stem = f"{s.name_prefix}_{sample.index + 1:05d}"
        img_path = out / SYNTHETIC_DIR / f"{stem}{s.output_ext}"
        write_image(img_path, sample.image)
        rel = img_path.relative_to(out).as_posix()
        sample.files["image"] = rel
        _store(out, s, rel, sample.mask, "synthetic", files)
        if s.save_masks:
            sample.files["mask"] = f"masks/{stem}.png"
        if s.save_overlays:
            opath = out / "overlays" / f"{stem}.jpg"
            write_image(opath, overlay(sample.image, sample.mask))
            sample.files["overlay"] = opath.relative_to(out).as_posix()
        records.append({"file": rel, **sample.meta})
        if progress:
            progress(len(records), total, sample)


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
