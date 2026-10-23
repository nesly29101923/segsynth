from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from segsynth.core.generator import GenerationError, Generator, run_to_folder, validate
from segsynth.core.imageio import read_image
from segsynth.core.project import Project


def test_generates_requested_count(small_project, tmp_path):
    out = tmp_path / "out"
    manifest = run_to_folder(small_project, out)
    assert manifest["count"] == 5 and manifest["completed"]
    files = sorted((out / "images").glob("*.png"))
    assert len(files) == 5
    assert not (out / "masks").exists()
    data = json.loads((out / "manifest.json").read_text())
    assert len(data["samples"]) == 5
    assert all(s["defects"] for s in data["samples"])


def test_masks_and_overlays(small_project, tmp_path):
    small_project.settings.save_masks = True
    small_project.settings.save_overlays = True
    out = tmp_path / "out"
    run_to_folder(small_project, out)
    masks = sorted((out / "masks").glob("*.png"))
    assert len(masks) == 5 and len(list((out / "overlays").glob("*.jpg"))) == 5
    for m in masks:
        assert read_image(m).max() == 255


def test_deterministic_with_seed(small_project, tmp_path):
    a = run_to_folder(small_project, tmp_path / "a")
    b = run_to_folder(small_project, tmp_path / "b")
    assert [s["defects"] for s in a["samples"]] == [s["defects"] for s in b["samples"]]
    for fa in (tmp_path / "a" / "images").glob("*.png"):
        assert np.array_equal(read_image(fa), read_image(tmp_path / "b" / "images" / fa.name))


def test_different_seed_differs(small_project, tmp_path):
    a = run_to_folder(small_project, tmp_path / "a")
    small_project.settings.seed = 99
    b = run_to_folder(small_project, tmp_path / "b")
    assert [s["defects"][0]["position"] for s in a["samples"]] != [s["defects"][0]["position"] for s in b["samples"]]


def test_multiple_defects_per_image(small_project):
    small_project.settings.defects_per_image = (2, 3)
    samples = list(Generator(small_project).generate())
    assert all(2 <= len(s.meta["defects"]) <= 3 for s in samples)
    for s in samples:
        assert s.mask.sum() > 0


def test_avoids_marked_defects(small_project):
    src = small_project.images[1]
    small_project.images[0].use_as_target = False
    marked = small_project.marked_mask(src.id, small_project.load_pixels(src.id).shape[:2])
    for s in Generator(small_project).generate():
        assert not (s.pasted & marked).any()
        assert np.array_equal(s.mask, s.pasted | marked)


def test_stop_event(small_project):
    import threading
    stop = threading.Event()
    gen = Generator(small_project).generate(stop)
    next(gen)
    stop.set()
    assert list(gen) == []


def test_validate_messages(tmp_path):
    assert validate(Project())
    with pytest.raises(GenerationError):
        list(Generator(Project()).generate())


def test_unplaceable_raises(small_project):
    small_project.settings.transform.scale = (8.0, 8.0)
    with pytest.raises(GenerationError):
        list(Generator(small_project).generate())


def test_project_roundtrip(small_project, tmp_path):
    small_project.settings.blend_modes = ["poisson", "residual"]
    small_project.defects[0].placement = "keep_column"
    path = small_project.save(tmp_path / "proj")
    loaded = Project.load(path)
    assert loaded.settings == small_project.settings
    assert [i.path for i in loaded.images] == [i.path for i in small_project.images]
    d0, l0 = small_project.defects[0], loaded.defects[0]
    assert l0.placement == "keep_column" and np.array_equal(d0.mask, l0.mask)
    small_project.remove_defect(d0.id)
    small_project.save(tmp_path / "proj")
    assert not list((tmp_path / "proj" / "masks").glob("*.png"))


def test_cli(small_project, tmp_path):
    path = small_project.save(tmp_path / "proj")
    out = tmp_path / "cli_out"
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-m", "segsynth", "generate", str(path), str(out), "--masks"],
                       cwd=root, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert len(list((out / "masks").glob("*.png"))) == 5
