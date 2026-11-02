"""Offscreen smoke test of the whole UI workflow."""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from segsynth.core.imageio import write_image  # noqa: E402
from segsynth.core.project import Project  # noqa: E402
from segsynth.ui.main_window import (  # noqa: E402
    STEP_BACKGROUNDS, STEP_DEFECTS, STEP_GENERATE, STEP_IMAGES, STEP_RECIPE, STEP_SAVE,
)

from .conftest import add_scratch, striped_surface  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from segsynth.ui.theme import apply_theme
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_theme(a)
    return a


@pytest.fixture
def window(app, monkeypatch):
    from segsynth.ui.main_window import MainWindow
    MB = QtWidgets.QMessageBox
    monkeypatch.setattr(MB, "question", lambda *a, **k: MB.Yes)
    monkeypatch.setattr(MB, "warning", lambda *a, **k: None)
    monkeypatch.setattr(MB, "critical", lambda *a, **k: pytest.fail(f"critical dialog: {a[2]}"))
    monkeypatch.setattr(MB, "information", lambda *a, **k: None)
    w = MainWindow()
    w.settings.clear()
    w.show()
    yield w
    w.state.mark_dirty(False)
    w.close()


def _wait(app, cond, timeout=60):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        app.processEvents()
        time.sleep(0.02)
    assert cond()


def test_full_workflow(app, window, tmp_path):
    write_image(tmp_path / "clean.png", striped_surface(seed=1))
    img, mask = add_scratch(striped_surface(seed=2))
    write_image(tmp_path / "defect.png", img)

    window.state.replace_project(Project(name="UI test"))
    window.go(STEP_IMAGES)
    window.images_page.add_paths([tmp_path / "clean.png", tmp_path / "defect.png"])
    assert len(window.state.project.images) == 2

    window.go(STEP_DEFECTS)
    page = window.defects_page
    app.processEvents()
    page.images.setCurrentRow(1)
    app.processEvents()
    rough = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    page.canvas.set_edit_mask(rough)
    page.snap()
    assert 0 < page.canvas.edit_mask().sum() <= rough.sum()
    page.commit()
    assert len(window.state.project.defects) == 1
    assert page.canvas.edit_is_empty()

    page.defect_list.setCurrentRow(0)
    page.edit_selected()
    assert page.editing is not None
    page.cancel_edit()

    window.go(STEP_RECIPE)
    app.processEvents()
    recipe = window.recipe_page
    recipe.bulk.setValue(4)
    recipe.bulk_apply.click()
    assert window.state.project.total_images == 4
    recipe.scale.a.setValue(0.9)
    assert window.state.project.settings.transform.scale[0] == pytest.approx(0.9)
    recipe.preview()

    window.go(STEP_GENERATE)
    gen = window.generate_page
    assert gen.start.isEnabled()
    gen.run()
    _wait(app, lambda: window.state.manifest is not None and gen.worker is None)
    assert gen.gallery.count() == 4

    window.go(STEP_SAVE)
    save = window.save_page
    save.path.setText(str(tmp_path / "export"))
    save.sqlite.setChecked(True)
    save.class_name.setText("scratch")
    assert window.state.project.settings.save_labeltool_db
    assert window.state.project.settings.labeltool_class_name == "scratch"
    save.zip.setChecked(True)
    save.save()
    dest = tmp_path / "export" / "UI test"
    assert len(list((dest / "DATA" / "synthetic").glob("synth_*.png"))) == 4
    assert len(list((dest / "DATA" / "background").glob("real_*.png"))) == 2
    assert not (dest / ".labels").exists()
    import sqlite3
    db = sqlite3.connect(dest / "META" / "UI_test.sqlite")
    assert db.execute("select count(*) from Samples").fetchone() == (6,)
    assert db.execute("select AnnotationLabelName from AnnotationLabelDefinitions where AnnotationLabelId=2").fetchone() == ("scratch",)
    db.close()
    assert (dest / "manifest.json").exists()
    assert not (dest / ".preview_masks").exists()
    assert (tmp_path / "export" / "UI test.zip").exists()

    window.state.project_path = None
    proj_dir = tmp_path / "projects" / "UI test"
    saved = window.state.project.save(proj_dir)
    loaded = Project.load(saved)
    assert len(loaded.defects) == 1 and loaded.defects[0].count == 4


def test_canvas_tools(app, window):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QPainterPath
    from segsynth.ui.canvas import MaskCanvas

    c = MaskCanvas()
    c.resize(600, 400)
    c.show()
    c.set_image(striped_surface())
    path = QPainterPath()
    path.addRect(10, 10, 40, 30)
    c._fill_path(path, erase=False)
    assert c.edit_mask().sum() == pytest.approx(40 * 30, rel=0.1)
    c._fill_path(path, erase=True)
    assert c.edit_mask().sum() == 0
    c.undo()
    assert c.edit_mask().sum() > 0
    c.redo()
    assert c.edit_mask().sum() == 0
    c._erasing = False
    c.set_brush_size(10)
    c._stroke(QPointF(100, 100), QPointF(150, 100))
    assert c.edit_mask()[100, 100:150].all()
    c.zoom_to(2.0)
    assert c._zoom == pytest.approx(2.0)
    c.fit()
    assert Qt.KeepAspectRatio is not None


def test_background_generation_workflow(app, window, tmp_path):
    for i in range(3):
        write_image(tmp_path / f"s{i}.png", striped_surface(seed=i))
    img, mask = add_scratch(striped_surface(seed=9))
    write_image(tmp_path / "defect.png", img)

    window.state.replace_project(Project(name="BG test"))
    window.go(STEP_IMAGES)
    images = window.images_page
    images.add_paths([tmp_path / f"s{i}.png" for i in range(3)] + [tmp_path / "defect.png"])
    assert not window.nav_buttons[STEP_BACKGROUNDS].isEnabled()
    assert window.next_btn.text().startswith("Pick defects")

    images.create_bgs.setChecked(True)
    assert window.state.project.backgrounds.enabled
    assert window.nav_buttons[STEP_BACKGROUNDS].isEnabled()
    assert window.next_btn.text().startswith("Background generating")

    window.next_btn.click()
    app.processEvents()
    page = window.backgrounds_page
    assert window.stack.currentWidget() is page
    assert len(page._cards) == 4 and all(len(c.buttons) == 2 for c in page._cards)
    assert "12 backgrounds" in page.total.text()
    assert sum(1 for t in page.tiles if t.pixmap() is not None and not t.pixmap().isNull()) == 8

    page._cards[3].buttons[0].click()
    assert window.state.project.backgrounds.is_rejected(window.state.project.images[3].id, 0)
    assert "7 of 8 pieces good" in page.counts.text()
    assert "9 backgrounds" in page.total.text()

    page.pieces.setValue(3)
    assert window.state.project.backgrounds.rejected == []
    assert all(len(c.buttons) == 3 for c in page._cards)
    page.pieces.setValue(2)

    page.max_count.setValue(5)
    assert "5 backgrounds" in page.total.text()

    window.next_btn.click()
    assert window.stack.currentWidget() is window.defects_page
    window.back_btn.click()
    assert window.stack.currentWidget() is page

    defects = window.defects_page
    window.go(STEP_DEFECTS)
    app.processEvents()
    defects.images.setCurrentRow(3)
    app.processEvents()
    defects.canvas.set_edit_mask(cv2.dilate(mask, np.ones((7, 7), np.uint8)))
    defects.commit()
    for e in window.state.project.images:
        e.use_as_target = False
    window.go(STEP_RECIPE)
    window.recipe_page.bulk.setValue(5)
    window.recipe_page.bulk_apply.click()

    window.go(STEP_GENERATE)
    gen = window.generate_page
    assert gen.stat_bgs.text() == "5"
    gen.run()
    _wait(app, lambda: window.state.manifest is not None and gen.worker is None)
    m = window.state.manifest
    assert m["count"] == 5 and len(m["generated_backgrounds"]) == 5
    assert all(s["target_image"].startswith("bg_") for s in m["samples"])

    window.go(STEP_SAVE)
    window.save_page.path.setText(str(tmp_path / "export"))
    window.save_page.sqlite.setChecked(True)
    window.save_page.save()
    dest = tmp_path / "export" / "BG test"
    assert len(list((dest / "DATA" / "background").glob("bg_*.png"))) == 5
    import sqlite3
    db = sqlite3.connect(dest / "META" / "BG_test.sqlite")
    assert db.execute("select count(*) from Samples").fetchone() == (4 + 5 + 5,)
    db.close()

    path = window.state.project.save(tmp_path / "proj")
    assert Project.load(path).backgrounds == window.state.project.backgrounds


def test_save_without_sqlite(app, window, tmp_path):
    write_image(tmp_path / "clean.png", striped_surface(seed=1))
    img, mask = add_scratch(striped_surface(seed=2))
    write_image(tmp_path / "defect.png", img)
    proj = Project(name="No DB")
    proj.add_image(tmp_path / "clean.png")
    src = proj.add_image(tmp_path / "defect.png")
    proj.add_defect(src.id, cv2.dilate(mask, np.ones((5, 5), np.uint8)), count=3)
    window.state.replace_project(proj)
    window.go(STEP_GENERATE)
    gen = window.generate_page
    gen.run()
    _wait(app, lambda: window.state.manifest is not None and gen.worker is None)
    assert not list(window.state.staging.glob("*.sqlite"))
    window.go(STEP_SAVE)
    save = window.save_page
    assert not save.sqlite.isChecked() and not save.class_name.isEnabled()
    save.path.setText(str(tmp_path / "out"))
    save.save()
    dest = tmp_path / "out" / "No DB"
    assert sorted(p.name for p in dest.iterdir()) == ["background", "manifest.json", "synthetic"]
    assert len(list((dest / "synthetic").iterdir())) == 3
    save.sqlite.setChecked(True)
    save.save()
    assert (dest / "META" / "No_DB.sqlite").exists()
    import json
    assert json.loads((dest / "manifest.json").read_text())["labeltool_db"] == "META/No_DB.sqlite"
