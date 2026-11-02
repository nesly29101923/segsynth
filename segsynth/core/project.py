"""Project model: source images, user-picked defects, recipe and generation settings."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .backgrounds import BackgroundSettings
from .imageio import read_image, read_mask, write_image
from .transform import TransformParams

PROJECT_FILE = "project.json"
FORMAT_VERSION = 1


def new_id() -> str:
    return uuid.uuid4().hex[:10]


@dataclass
class ImageEntry:
    id: str
    path: str
    use_as_target: bool = True

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass
class DefectEntry:
    id: str
    image_id: str
    name: str
    count: int = 10
    enabled: bool = True
    placement: str = ""          # empty = use global setting
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("mask")
        return d


@dataclass
class GenerationSettings:
    seed: int = 42
    defects_per_image: tuple[int, int] = (1, 1)
    bg_mode: str = "vertical"
    bg_size: int = 151
    feather: int = 4
    blend_modes: list[str] = field(default_factory=lambda: ["residual"])
    placement_mode: str = "random"
    jitter: int = 60
    min_gap: int = 10
    avoid_marked_defects: bool = True
    transform: TransformParams = field(default_factory=TransformParams)
    save_masks: bool = False
    save_overlays: bool = False
    save_labeltool_db: bool = False
    labeltool_class_name: str = "defect"
    labeltool_class_color: str = "#ff5c6c"
    labeltool_include_sources: bool = True
    labeltool_validate_percent: float = 15.0
    output_ext: str = ".png"
    name_prefix: str = "synth"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["transform"] = self.transform.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationSettings":
        d = dict(d)
        t = TransformParams.from_dict(d.pop("transform", {}))
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "defects_per_image" in known:
            known["defects_per_image"] = tuple(known["defects_per_image"])
        return cls(transform=t, **known)


@dataclass
class Project:
    name: str = "Untitled dataset"
    images: list[ImageEntry] = field(default_factory=list)
    defects: list[DefectEntry] = field(default_factory=list)
    settings: GenerationSettings = field(default_factory=GenerationSettings)
    backgrounds: BackgroundSettings = field(default_factory=BackgroundSettings)
    folder: Path | None = None
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # images -----------------------------------------------------------
    def add_image(self, path: str | Path, use_as_target: bool = True) -> ImageEntry:
        path = str(Path(path).resolve())
        for e in self.images:
            if e.path == path:
                return e
        entry = ImageEntry(id=new_id(), path=path, use_as_target=use_as_target)
        self.images.append(entry)
        return entry

    def remove_image(self, image_id: str) -> None:
        self.images = [e for e in self.images if e.id != image_id]
        self.defects = [d for d in self.defects if d.image_id != image_id]
        self.backgrounds.rejected = [r for r in self.backgrounds.rejected if r[0] != image_id]
        self._cache.pop(image_id, None)

    def image(self, image_id: str) -> ImageEntry:
        for e in self.images:
            if e.id == image_id:
                return e
        raise KeyError(image_id)

    def load_pixels(self, image_id: str) -> np.ndarray:
        if image_id not in self._cache:
            self._cache[image_id] = read_image(self.image(image_id).path)
        return self._cache[image_id]

    # defects ----------------------------------------------------------
    def add_defect(self, image_id: str, mask: np.ndarray, name: str | None = None, count: int = 10) -> DefectEntry:
        n = sum(1 for d in self.defects if d.image_id == image_id) + 1
        entry = DefectEntry(
            id=new_id(),
            image_id=image_id,
            name=name or f"{Path(self.image(image_id).path).stem} · defect {n}",
            count=count,
            mask=(mask > 0).astype(np.uint8),
        )
        self.defects.append(entry)
        return entry

    def remove_defect(self, defect_id: str) -> None:
        self.defects = [d for d in self.defects if d.id != defect_id]

    def defect(self, defect_id: str) -> DefectEntry:
        for d in self.defects:
            if d.id == defect_id:
                return d
        raise KeyError(defect_id)

    def defects_of(self, image_id: str) -> list[DefectEntry]:
        return [d for d in self.defects if d.image_id == image_id]

    def marked_mask(self, image_id: str, shape: tuple[int, int]) -> np.ndarray:
        m = np.zeros(shape, np.uint8)
        for d in self.defects_of(image_id):
            if d.mask is not None and d.mask.shape == shape:
                m |= d.mask
        return m

    @property
    def total_images(self) -> int:
        return sum(d.count for d in self.defects if d.enabled)

    # persistence ------------------------------------------------------
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        (folder / "masks").mkdir(parents=True, exist_ok=True)
        for d in self.defects:
            if d.mask is not None:
                write_image(folder / "masks" / f"{d.id}.png", d.mask * 255)
        live = {f"{d.id}.png" for d in self.defects}
        for f in (folder / "masks").glob("*.png"):
            if f.name not in live:
                f.unlink()
        data = {
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "images": [asdict(e) for e in self.images],
            "defects": [d.to_dict() for d in self.defects],
            "settings": self.settings.to_dict(),
            "backgrounds": self.backgrounds.to_dict(),
        }
        path = folder / PROJECT_FILE
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.folder = folder
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        if path.is_dir():
            path = path / PROJECT_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        folder = path.parent
        proj = cls(
            name=data.get("name", "Untitled dataset"),
            images=[ImageEntry(**e) for e in data.get("images", [])],
            settings=GenerationSettings.from_dict(data.get("settings", {})),
            backgrounds=BackgroundSettings.from_dict(data.get("backgrounds", {})),
            folder=folder,
        )
        for d in data.get("defects", []):
            entry = DefectEntry(**{k: v for k, v in d.items() if k in DefectEntry.__dataclass_fields__})
            mpath = folder / "masks" / f"{entry.id}.png"
            if mpath.exists():
                entry.mask = read_mask(mpath)
            proj.defects.append(entry)
        return proj
