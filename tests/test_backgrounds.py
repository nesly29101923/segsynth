from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from segsynth.core.backgrounds import (
    HORIZONTAL, VERTICAL, BackgroundComposer, BackgroundSettings, estimate_shift, piece_bounds,
    slot_weights, split_view,
)
from segsynth.core.generator import Generator, list_targets, run_to_folder, validate
from segsynth.core.imageio import read_image, write_image
from segsynth.core.labeltool import decode_rle
from segsynth.core.project import Project

from .conftest import striped_surface


def make_project(tmp_path, n=4, shifts=None, **bg):
    proj = Project(name="bg")
    shifts = shifts or [0] * n
    for i in range(n):
        img = striped_surface(seed=i, h=200, w=240)
        img = (img.astype(np.int32) + 12 * i).clip(0, 255).astype(np.uint8)
        img = np.roll(img, shifts[i], axis=1)
        write_image(tmp_path / f"s{i}.png", img)
        proj.add_image(tmp_path / f"s{i}.png")
    proj.backgrounds = BackgroundSettings(enabled=True, **bg)
    return proj


def test_piece_bounds_cover_length():
    for n in (2, 3, 7):
        b = piece_bounds(1625, n)
        assert b[0][0] == 0 and b[-1][1] == 1625
        assert all(e == s2 for (_, e), (s2, _) in zip(b[:-1], b[1:]))


@pytest.mark.parametrize("seam", [0, 40, 1000])
def test_slot_weights_partition_of_unity(seam):
    w = slot_weights(300, 3, seam)
    assert np.allclose(np.sum(w, axis=0), 1.0)
    assert all(x.min() >= 0 for x in w)
    assert w[0][0] == 1 and w[-1][-1] == 1


def test_counts_and_combinations(tmp_path):
    proj = make_project(tmp_path, n=4)
    comp = BackgroundComposer(proj)
    assert comp.counts() == (8, 8)
    assert comp.possible() == 4 * 4 - 4
    assert len(comp.combos()) == 12
    assert all(len(set(c)) == 2 for c in comp.combos())
    proj.backgrounds.set_rejected(proj.images[0].id, 1, True)
    proj.backgrounds.set_rejected(proj.images[1].id, 0, True)
    comp = BackgroundComposer(proj)
    assert comp.counts() == (6, 8)
    assert comp.possible() == 3 * 3 - 2
    for c in comp.combos():
        assert c[1] != proj.images[0].id and c[0] != proj.images[1].id


def test_cap_and_determinism(tmp_path):
    proj = make_project(tmp_path, n=4, pieces=3, max_count=10)
    a = BackgroundComposer(proj).combos()
    b = BackgroundComposer(proj).combos()
    assert a == b and len(a) == 10 and len(set(a)) == 10
    proj.settings.seed = 999
    assert BackgroundComposer(proj).combos() != a


def test_no_good_piece_in_a_slot(tmp_path):
    proj = make_project(tmp_path, n=2)
    for e in proj.images:
        proj.backgrounds.set_rejected(e.id, 0, True)
    assert BackgroundComposer(proj).combos() == []


def test_render_uses_pieces_in_their_slots(tmp_path):
    proj = make_project(tmp_path, n=3, seam=0, align=False)
    comp = BackgroundComposer(proj)
    a, b = proj.images[0].id, proj.images[2].id
    img, mask = comp.render((a, b))
    top, bottom = proj.load_pixels(a), proj.load_pixels(b)
    assert img.shape == top.shape and img.dtype == top.dtype
    assert np.array_equal(img[:100], top[:100]) and np.array_equal(img[100:], bottom[100:])
    assert not mask.any()


def test_vertical_direction(tmp_path):
    proj = make_project(tmp_path, n=2, seam=0, align=False, direction=VERTICAL)
    comp = BackgroundComposer(proj)
    a, b = proj.images[0].id, proj.images[1].id
    img, _ = comp.render((a, b))
    assert np.array_equal(img[:, :120], proj.load_pixels(a)[:, :120])
    assert np.array_equal(img[:, 120:], proj.load_pixels(b)[:, 120:])


def test_seam_is_blended(tmp_path):
    proj = make_project(tmp_path, n=2, seam=40, align=False)
    comp = BackgroundComposer(proj)
    a, b = proj.images[0].id, proj.images[1].id
    img, _ = comp.render((a, b))
    fa, fb = proj.load_pixels(a).astype(float), proj.load_pixels(b).astype(float)
    mid = img[100].astype(float)
    assert np.abs(mid - (fa[100] + fb[100]) / 2).mean() < 2
    assert np.array_equal(img[:70], proj.load_pixels(a)[:70])


def test_alignment_recovers_shift(tmp_path):
    ref = striped_surface(seed=0, h=120, w=400, period=9)
    moved = np.roll(ref, 5, axis=1)
    assert estimate_shift(ref, moved, 12) == -5
    flat = np.full_like(ref, 100)
    assert estimate_shift(ref, flat, 12) == 0


def test_alignment_applied_in_render(tmp_path):
    proj = make_project(tmp_path, n=2, shifts=[0, 4], seam=0)
    comp = BackgroundComposer(proj)
    assert comp.shift(proj.images[1].id) == -4
    proj.backgrounds.align = False
    assert BackgroundComposer(proj).shift(proj.images[1].id) == 0


def test_picked_defects_follow_their_piece(tmp_path):
    proj = make_project(tmp_path, n=2, seam=0, align=False)
    a, b = proj.images[0].id, proj.images[1].id
    m = np.zeros((200, 240), np.uint8)
    m[150:160, 20:60] = 1
    proj.add_defect(a, m)
    comp = BackgroundComposer(proj)
    assert not comp.render((a, b))[1].any()
    _, mask = comp.render((b, a))
    assert np.array_equal(mask > 0, m > 0)


def test_incompatible_sizes_skipped(tmp_path):
    proj = make_project(tmp_path, n=2)
    write_image(tmp_path / "odd.png", striped_surface(h=50, w=60))
    proj.add_image(tmp_path / "odd.png")
    comp = BackgroundComposer(proj)
    assert len(comp.sources) == 2 and [e.name for e in comp.skipped] == ["odd.png"]


def test_gray_uint16(tmp_path):
    proj = Project(name="g")
    for i in range(2):
        write_image(tmp_path / f"g{i}.png", striped_surface(seed=i, h=80, w=90, channels=1, dtype=np.uint16))
        proj.add_image(tmp_path / f"g{i}.png")
    proj.backgrounds = BackgroundSettings(enabled=True, seam=10)
    img, _ = BackgroundComposer(proj).render(BackgroundComposer(proj).combos()[0])
    assert img.dtype == np.uint16 and img.ndim == 2


def test_split_view(tmp_path):
    img = np.arange(60).reshape(6, 10)
    top, bottom = split_view(img, HORIZONTAL, 2)
    assert top.shape == (3, 10) and np.array_equal(np.vstack([top, bottom]), img)
    parts = split_view(img, VERTICAL, 2)
    assert parts[0].shape == (6, 5) and np.array_equal(np.hstack(parts), img)


def test_settings_roundtrip(tmp_path):
    proj = make_project(tmp_path, n=2, pieces=3, seam=50, max_count=7)
    proj.backgrounds.set_rejected(proj.images[0].id, 2, True)
    proj.save(tmp_path / "p")
    loaded = Project.load(tmp_path / "p")
    assert loaded.backgrounds == proj.backgrounds
    loaded.remove_image(proj.images[0].id)
    assert loaded.backgrounds.rejected == []


def _with_defect(proj):
    img = proj.load_pixels(proj.images[0].id)
    m = np.zeros(img.shape[:2], np.uint8)
    m[40:44, 30:90] = 1
    proj.add_defect(proj.images[0].id, m, count=6)
    return proj


def test_generator_targets_include_backgrounds(tmp_path):
    proj = _with_defect(make_project(tmp_path, n=3))
    for e in proj.images:
        e.use_as_target = False
    assert validate(proj) == []
    targets = list_targets(proj, BackgroundComposer(proj))
    assert len(targets) == 6 and all(t.generated for t in targets)
    samples = list(Generator(proj).generate())
    assert len(samples) == 6
    assert len({s.meta["target_image"] for s in samples}) == 6
    assert all("target_pieces" in s.meta for s in samples)


def test_no_backgrounds_and_no_targets_is_invalid(tmp_path):
    proj = _with_defect(make_project(tmp_path, n=2))
    for e in proj.images:
        e.use_as_target = False
        proj.backgrounds.set_rejected(e.id, 0, True)
    assert any("good piece" in p for p in validate(proj))


def test_export_labels_backgrounds_good(tmp_path):
    proj = _with_defect(make_project(tmp_path, n=3))
    proj.settings.save_labeltool_db = True
    out = tmp_path / "out"
    manifest = run_to_folder(proj, out)
    bgs = manifest["generated_backgrounds"]
    assert len(bgs) == 6
    c = sqlite3.connect(out / "META" / "bg.sqlite")
    rows = dict(c.execute("""select s.SampleFilePath, d.LinkAnnotationLabelId from Samples s
                             join AnomalyDetectionAnnotations d on d.LinkSampleId = s.SampleId""").fetchall())
    assert len(rows) == 3 + 6 + 6
    for b in bgs:
        key = b["file"].replace("/", "\\")
        assert rows[key] == (-2 if b["label"] == "bad" else -1)
        assert read_image(out / "DATA" / b["file"]).shape == proj.load_pixels(proj.images[0].id).shape
    blob = c.execute("select SemanticSegmentationRunLengthCode from SemanticSegmentationAnnotations a "
                     "join Samples s on s.SampleId = a.LinkSampleId where s.SampleFilePath like '%bg_00001%'").fetchone()[0]
    assert decode_rle(blob).shape == (200, 240)


def test_export_can_skip_backgrounds(tmp_path):
    proj = _with_defect(make_project(tmp_path, n=3, include_in_dataset=False))
    manifest = run_to_folder(proj, tmp_path / "o")
    assert manifest["generated_backgrounds"] == []
    assert not list((tmp_path / "o" / "background").glob("bg_*"))
