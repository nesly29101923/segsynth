"""Slots: detection on a tray-like grid and exact placement into target slots."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from segsynth.core.generator import Generator, run_to_folder, validate
from segsynth.core.imageio import write_image
from segsynth.core.project import LABEL_GOOD, Project
from segsynth.core.slots import Slot, anchor_for, detect_slots, fit_lattice, slot_patch, turn_patch

PITCH, R = 60, 18
ROWS, COLS = 6, 8
EMPTY = {(1, 2), (4, 5)}
ODD = (3, 6)


def tray(seed: int = 0, empty=EMPTY, odd=ODD) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((ROWS * PITCH + 40, COLS * PITCH + 40), 40, np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            x, y = 20 + PITCH // 2 + c * PITCH, 20 + PITCH // 2 + r * PITCH
            if (r, c) in empty:
                cv2.circle(img, (x, y), R - 4, 25, -1)
                continue
            cv2.circle(img, (x, y), R, 210, -1)
            cv2.circle(img, (x, y), 7, 90 if (r, c) != odd else 10, -1)
            for a in (0, 120, 240):
                t = np.deg2rad(a)
                cv2.line(img, (x, y), (int(x + R * np.cos(t)), int(y + R * np.sin(t))), 60, 2)
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def center(r: int, c: int) -> tuple[int, int]:
    return 20 + PITCH // 2 + c * PITCH, 20 + PITCH // 2 + r * PITCH


def test_detect_finds_full_grid_and_empty_pockets():
    img = tray()
    x, y = center(0, 0)
    slots = detect_slots(img, (x - R - 3, y - R - 3, x + R + 4, y + R + 4), 0.5)
    assert len(slots) == ROWS * COLS
    low = {(round((s.y - 20 - PITCH // 2) / PITCH), round((s.x - 20 - PITCH // 2) / PITCH))
           for s in slots if s.score < 0.5}
    assert low == EMPTY
    for s in slots:
        r, c = round((s.y - 50) / PITCH), round((s.x - 50) / PITCH)
        cx, cy = center(r, c)
        assert abs(s.x - cx) <= 1.5 and abs(s.y - cy) <= 1.5


def test_detect_rejects_flat_box():
    img = np.full((100, 100), 50, np.uint8)
    with pytest.raises(ValueError):
        detect_slots(img, (10, 10, 40, 40))


def test_fit_lattice():
    pts = np.array([(10 + 30 * i + 2 * j, 5 + 25 * j) for i in range(4) for j in range(3)], float)
    o, a, b = fit_lattice(pts)
    assert np.allclose(a, (30, 0), atol=0.01) and np.allclose(b, (2, 25), atol=0.01)


def test_slot_patch_rotation_keeps_centre():
    img = tray()
    x, y = center(3, 6)
    m = np.zeros(img.shape, np.uint8)
    cv2.circle(m, (x + 2, y - 1), R + 3, 1, -1)
    anchor = anchor_for(m, [Slot(x, y, R)])
    assert anchor == (x, y)
    p = slot_patch(img, m, anchor, feather=2)
    n = p.mask.shape[0]
    assert n % 2 == 1 and p.mask.shape == (n, n)
    q = turn_patch(p, 90, 1.0)
    ys, xs = np.nonzero(q.mask)
    assert abs(xs.mean() - n // 2) < 2.5 and abs(ys.mean() - n // 2) < 2.5


def _project(tmp_path, good_label: bool):
    write_image(tmp_path / "a.png", tray(1))
    write_image(tmp_path / "b.png", tray(2, empty={(0, 0), (2, 3), (5, 7)}, odd=None))
    p = Project(name="tray")
    a, b = p.add_image(tmp_path / "a.png"), p.add_image(tmp_path / "b.png")
    for e in (a, b):
        x, y = center(0, 1)
        e.slots = detect_slots(p.load_pixels(e.id), (x - R - 3, y - R - 3, x + R + 4, y + R + 4), 0.5)
        for s in e.slots:
            s.target = s.score < 0.5
    x, y = center(*ODD)
    m = np.zeros(p.load_pixels(a.id).shape, np.uint8)
    cv2.circle(m, (x, y), R + 3, 1, -1)
    d = p.add_defect(a.id, m, "odd part", count=6)
    if good_label:
        d.label = LABEL_GOOD
    p.settings.placement_mode = "slots"
    p.settings.defects_per_image = (1, 2)
    p.settings.transform.rotation = (-180, 180)
    return p


def test_generation_places_into_target_slots(tmp_path):
    p = _project(tmp_path, good_label=False)
    assert validate(p) == []
    targets = {(e.name, round(s.x), round(s.y)) for e in p.images for s in e.target_slots}
    samples = list(Generator(p).generate())
    assert len(samples) == 6
    for smp in samples:
        assert 1 <= len(smp.meta["defects"]) <= 2
        for info in smp.meta["defects"]:
            sx, sy = info["slot"]
            assert any(n == smp.meta["target_image"] and abs(tx - sx) <= 1 and abs(ty - sy) <= 1
                       for n, tx, ty in targets)
            assert smp.mask[int(round(sy)), int(round(sx))] == 1
        slots_used = [tuple(i["slot"]) for i in smp.meta["defects"]]
        assert len(set(slots_used)) == len(slots_used)


def test_good_pieces_are_not_labeled(tmp_path):
    p = _project(tmp_path, good_label=True)
    for smp in Generator(p).generate():
        for info in smp.meta["defects"]:
            sx, sy = info["slot"]
            assert smp.mask[int(round(sy)), int(round(sx))] == 0
            assert info["label"] == "good"


def test_piece_covering_picked_defect_clears_its_label(tmp_path):
    p = _project(tmp_path, good_label=True)
    a = p.images[0]
    # a labeled defect picked inside a target slot of image a
    s = a.target_slots[0]
    m = np.zeros(p.load_pixels(a.id).shape, np.uint8)
    cv2.circle(m, (int(s.x), int(s.y)), 6, 1, -1)
    d = p.add_defect(a.id, m, "inside", count=0)
    assert d.label == "defect"
    p.images[1].slots = []
    for x in a.slots:
        x.target = x is s
    gen = Generator(p)
    tgt = gen.targets[0]
    img = gen.pixels(tgt)
    label = gen.labels(tgt, img.shape[:2]).copy()
    assert label[int(s.y), int(s.x)] == 1
    pasted = np.zeros_like(label)
    _, info = gen.paste(img, pasted, np.zeros_like(label), p.defects[0], tgt, np.random.default_rng(0),
                        255, set(), label)
    assert info is not None and label[int(s.y), int(s.x)] == 0


def test_validate_requires_target_slots(tmp_path):
    p = _project(tmp_path, good_label=False)
    for e in p.images:
        for s in e.slots:
            s.target = False
    assert any("target slots" in m for m in validate(p))


def test_slots_survive_save_and_load(tmp_path):
    p = _project(tmp_path, good_label=True)
    p.save(tmp_path / "proj")
    q = Project.load(tmp_path / "proj")
    assert [s.to_dict() for s in q.images[0].slots] == [s.to_dict() for s in p.images[0].slots]
    assert q.defects[0].label == LABEL_GOOD
    assert len(q.images[0].target_slots) == len(p.images[0].target_slots) > 0


def test_run_to_folder_with_database(tmp_path):
    p = _project(tmp_path, good_label=False)
    manifest = run_to_folder(p, tmp_path / "ws", labeltool=True)
    assert manifest["count"] == 6
    assert (tmp_path / "ws" / "META").is_dir()
