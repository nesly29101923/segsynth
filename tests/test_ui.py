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
    window.go(0)
    window.pages[0].add_paths([tmp_path / "clean.png", tmp_path / "defect.png"])
    assert len(window.state.project.images) == 2

    window.go(1)
    page = window.pages[1]
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

    window.go(2)
    app.processEvents()
    recipe = window.pages[2]
    recipe.bulk.setValue(4)
    recipe.bulk_apply.click()
    assert window.state.project.total_images == 4
    recipe.scale.a.setValue(0.9)
    assert window.state.project.settings.transform.scale[0] == pytest.approx(0.9)
    recipe.preview()
    recipe.labeltool.setChecked(True)
    recipe.class_name.setText("scratch")
    assert window.state.project.settings.save_labeltool_db
    assert window.state.project.settings.labeltool_class_name == "scratch"

    window.go(3)
    gen = window.pages[3]
    assert gen.start.isEnabled()
    gen.run()
    _wait(app, lambda: window.state.manifest is not None and gen.worker is None)
    assert gen.gallery.count() == 4

    window.go(4)
    save = window.pages[4]
    save.path.setText(str(tmp_path / "export"))
    save.zip.setChecked(True)
    save.save()
    dest = tmp_path / "export" / "UI test"
    assert len(list((dest / "Images").glob("synth_*.png"))) == 4
    assert len(list((dest / "Images").glob("real_*.png"))) == 2
    import sqlite3
    db = sqlite3.connect(dest / "UI test.sqlite")
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
