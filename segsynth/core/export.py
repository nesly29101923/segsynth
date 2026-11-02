"""Export a generated dataset to its final folder.

Plain layout:
    <dest>/synthetic/, <dest>/background/, <dest>/manifest.json

AI tool workspace layout (with a label tool database):
    <dest>/DATA/synthetic/, <dest>/DATA/background/   images; sample paths are relative to DATA
    <dest>/DATA_THUMB/
    <dest>/META/<name>.sqlite                          database, read next to DATA
    <dest>/manifest.json
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np

from .imageio import read_mask
from .labeltool import LabelToolWriter

DATA_DIR = "DATA"
THUMB_DIR = "DATA_THUMB"
META_DIR = "META"
LABELS_DIR = ".labels"


def label_path(root: str | Path, rel: str) -> Path:
    """Where the label mask of an output image is stored inside a generation folder."""
    return Path(root) / LABELS_DIR / f"{rel}.png"


def db_file_name(project_name: str) -> str:
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_name.strip()).strip("_") or "dataset"
    return f"{stem}.sqlite"


def assign_splits(files: list[dict], validate_percent: float, seed: int) -> dict[str, str]:
    """TRAIN/VALIDATE per file, stratified by label (good / not good) and kind."""
    rng = np.random.default_rng(seed + 104729)
    share = max(0.0, min(0.9, validate_percent / 100.0))
    split = {f["file"]: "TRAIN" for f in files}
    if share <= 0:
        return split
    groups: dict[tuple, list[str]] = {}
    for f in files:
        groups.setdefault((f["label"], f["kind"]), []).append(f["file"])
    for members in groups.values():
        if len(members) < 2:
            continue
        n_val = max(1, int(round(len(members) * share)))
        n_val = min(n_val, len(members) - 1)
        for i in rng.choice(len(members), size=n_val, replace=False):
            split[members[i]] = "VALIDATE"
    return split


def write_labeltool_db(
    labels_root: str | Path,
    files: list[dict],
    db_path: str | Path,
    class_name: str = "defect",
    class_color: str = "#ff5c6c",
    splits: dict[str, str] | None = None,
) -> int:
    """Create a label tool database for the listed files using their stored masks."""
    labels_root = Path(labels_root)
    with LabelToolWriter(db_path, class_name, class_color) as db:
        for f in files:
            mask = read_mask(label_path(labels_root, f["file"]))
            db.add(f["file"], mask, (splits or {}).get(f["file"], "TRAIN"))
        return db.count


def export_dataset(
    staging: str | Path,
    dest: str | Path,
    manifest: dict,
    project_name: str,
    sqlite: bool,
    class_name: str = "defect",
    class_color: str = "#ff5c6c",
    validate_percent: float = 15.0,
    seed: int = 42,
) -> dict:
    """Copy a generation folder to dest, optionally as an AI tool workspace with a database."""
    staging, dest = Path(staging), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    image_root = dest / DATA_DIR if sqlite else dest
    image_root.mkdir(parents=True, exist_ok=True)
    for item in staging.iterdir():
        if item.name.startswith(".") or item.name == "manifest.json" or item.suffix == ".sqlite":
            continue
        target = (image_root if item.name in _image_dirs(manifest) else dest) / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    out = json.loads(json.dumps(manifest))
    out["labeltool_db"] = None
    out["layout"] = "plain"
    if sqlite:
        (dest / THUMB_DIR).mkdir(exist_ok=True)
        (dest / META_DIR).mkdir(exist_ok=True)
        files = manifest.get("files", [])
        splits = assign_splits(files, validate_percent, seed)
        name = db_file_name(project_name)
        write_labeltool_db(staging, files, dest / META_DIR / name, class_name, class_color, splits)
        out["labeltool_db"] = f"{META_DIR}/{name}"
        out["layout"] = "ai_tool_workspace"
        out["splits"] = {
            "TRAIN": sum(1 for v in splits.values() if v == "TRAIN"),
            "VALIDATE": sum(1 for v in splits.values() if v == "VALIDATE"),
        }
        for f in out.get("files", []):
            f["split"] = splits[f["file"]]
    (dest / "manifest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _image_dirs(manifest: dict) -> set[str]:
    return {f["file"].split("/")[0] for f in manifest.get("files", [])}


ANNOTATION_TABLES = ("SemanticSegmentationAnnotations", "ClassificationAnnotations", "AnomalyDetectionAnnotations",
                     "SampleProperties", "SampleToGroupConnections")


def remove_samples(db_path: str | Path, should_remove) -> list[str]:
    """Delete samples (and all rows linked to them) for which should_remove(path) is true."""
    db = sqlite3.connect(str(db_path))
    try:
        rows = db.execute("SELECT SampleId, SampleFilePath FROM Samples").fetchall()
        doomed = [(sid, path) for sid, path in rows if should_remove(path)]
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for sid, _ in doomed:
            for t in ANNOTATION_TABLES:
                if t in tables:
                    db.execute(f"DELETE FROM '{t}' WHERE LinkSampleId = ?", (sid,))
            db.execute("DELETE FROM Samples WHERE SampleId = ?", (sid,))
        db.commit()
        if doomed:
            db.execute("VACUUM")
        return [p for _, p in doomed]
    finally:
        db.close()


def prune_workspace(db_path: str | Path, remove_real: bool = True) -> list[str]:
    """Repair a saved workspace: drop samples whose image is missing from DATA, and
    (by default) real source images copied in by older versions (background\\real_*)."""
    db_path = Path(db_path)
    data_root = db_path.parent.parent / DATA_DIR

    def doomed(path: str) -> bool:
        rel = path.replace("\\", "/")
        if remove_real and Path(rel).name.startswith("real_"):
            return True
        return not (data_root / rel).is_file()

    removed = remove_samples(db_path, doomed)
    for rel in removed:
        f = data_root / rel.replace("\\", "/")
        if f.is_file():
            f.unlink()
    return removed
