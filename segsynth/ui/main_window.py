"""Main window: start screen, step navigation and project handling."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QStackedWidget,
    QVBoxLayout, QWidget,
)

from .. import __version__
from ..core.project import PROJECT_FILE, Project
from .pages.backgrounds import BackgroundsPage
from .pages.defects import DefectsPage
from .pages.generate import GeneratePage
from .pages.images import ImagesPage
from .pages.recipe import RecipePage
from .pages.save import SavePage
from .qt_utils import label
from .state import AppState

APP_NAME = "SegSynth"
TAGLINE = "Synthetic data generator for semantic segmentation"
STEPS = ["Source images", "Background generating", "Pick defects", "Recipe", "Generate", "Save dataset"]
STEP_IMAGES, STEP_BACKGROUNDS, STEP_DEFECTS, STEP_RECIPE, STEP_GENERATE, STEP_SAVE = range(len(STEPS))
MAX_RECENT = 8

SHORTCUTS_HTML = """
<table cellpadding="4">
<tr><td><b>B / E / R / L / H</b></td><td>Brush · Eraser · Box · Lasso · Pan</td></tr>
<tr><td><b>Right-drag</b></td><td>Erase with brush · subtract with box / lasso</td></tr>
<tr><td><b>[ &nbsp; ]</b> or <b>Ctrl + wheel</b></td><td>Brush size</td></tr>
<tr><td><b>Wheel</b></td><td>Zoom at cursor</td></tr>
<tr><td><b>Space + drag</b> / middle mouse</td><td>Pan</td></tr>
<tr><td><b>F</b> · <b>1</b></td><td>Fit to view · actual pixels</td></tr>
<tr><td><b>S</b></td><td>Snap selection to defect</td></tr>
<tr><td><b>Enter</b></td><td>Add selection as defect / save changes</td></tr>
<tr><td><b>Esc</b></td><td>Clear selection / cancel editing</td></tr>
<tr><td><b>Ctrl+Z / Ctrl+Y</b></td><td>Undo / redo</td></tr>
<tr><td><b>Ctrl+S / Ctrl+O / Ctrl+N</b></td><td>Save / open / new project</td></tr>
</table>
"""


class StartPage(QWidget):
    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        mid = QHBoxLayout()
        mid.addStretch(1)
        col = QVBoxLayout()
        col.setSpacing(10)
        title = QLabel(APP_NAME)
        title.setStyleSheet("font-size: 40px; font-weight: 800; background: transparent;")
        col.addWidget(title)
        col.addWidget(label(TAGLINE, "PageHint"))
        col.addSpacing(6)
        col.addWidget(label(
            "Pick real defects from a few images, vary their size, shape and position, and paste "
            "them onto clean or defective backgrounds to grow a training set.",
            "Muted", wrap=True))
        col.addSpacing(18)
        btns = QHBoxLayout()
        self.new_btn = QPushButton("Create a new dataset")
        self.new_btn.setObjectName("Primary")
        self.new_btn.setMinimumHeight(40)
        self.open_btn = QPushButton("Open project…")
        self.open_btn.setMinimumHeight(40)
        btns.addWidget(self.new_btn)
        btns.addWidget(self.open_btn)
        btns.addStretch(1)
        col.addLayout(btns)
        col.addSpacing(22)
        self.recent_title = label("RECENT PROJECTS", "SectionLabel")
        col.addWidget(self.recent_title)
        self.recent = QListWidget()
        self.recent.setMaximumHeight(230)
        col.addWidget(self.recent)
        wrap = QWidget()
        wrap.setLayout(col)
        wrap.setFixedWidth(620)
        mid.addWidget(wrap)
        mid.addStretch(1)
        outer.addLayout(mid)
        outer.addStretch(2)

        self.new_btn.clicked.connect(window.new_project)
        self.open_btn.clicked.connect(lambda: window.open_project())
        self.recent.itemActivated.connect(lambda it: window.open_project(it.data(Qt.UserRole)))

    def refresh(self, recent: list[str]) -> None:
        self.recent.clear()
        for p in recent:
            item = QListWidgetItem(f"{Path(p).parent.name}\n{p}")
            item.setData(Qt.UserRole, p)
            self.recent.addItem(item)
        self.recent_title.setVisible(bool(recent))
        self.recent.setVisible(bool(recent))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.settings = QSettings(APP_NAME, APP_NAME)
        self.setWindowTitle(APP_NAME)
        self.resize(1480, 920)
        self.setMinimumSize(1180, 720)

        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # header ------------------------------------------------------------
        self.header = QFrame()
        self.header.setObjectName("Header")
        self.header.setFixedHeight(56)
        h = QHBoxLayout(self.header)
        h.setContentsMargins(18, 0, 14, 0)
        h.setSpacing(10)
        brand = QVBoxLayout()
        brand.setSpacing(0)
        brand.addWidget(label(APP_NAME, "AppTitle"))
        brand.addWidget(label(TAGLINE, "AppSubtitle"))
        h.addLayout(brand)
        h.addSpacing(24)
        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("ProjectName")
        self.name_edit.setFixedWidth(320)
        self.name_edit.setToolTip("Dataset name — click to rename")
        h.addWidget(self.name_edit)
        self.dirty_dot = label("", "Muted")
        h.addWidget(self.dirty_dot)
        h.addStretch(1)
        self.btn_new = QPushButton("New dataset")
        self.btn_open = QPushButton("Open…")
        self.btn_save = QPushButton("Save project")
        self.btn_help = QPushButton("Shortcuts")
        self.btn_help.setObjectName("Ghost")
        for b in (self.btn_help, self.btn_new, self.btn_open, self.btn_save):
            h.addWidget(b)
        v.addWidget(self.header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        v.addLayout(body, 1)

        # sidebar -----------------------------------------------------------
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(232)
        sb = QVBoxLayout(self.sidebar)
        sb.setContentsMargins(12, 18, 12, 14)
        sb.setSpacing(4)
        sb.addWidget(label("WORKFLOW", "SectionLabel"))
        sb.addSpacing(6)
        self.nav = QButtonGroup(self)
        self.nav.setExclusive(True)
        self.nav_buttons: list[QPushButton] = []
        for i, name in enumerate(STEPS):
            b = QPushButton(f"  {i + 1}   {name}")
            b.setObjectName("NavButton")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, idx=i: self.go(idx))
            self.nav.addButton(b, i)
            self.nav_buttons.append(b)
            sb.addWidget(b)
        sb.addStretch(1)
        self.step_hint = label("", "Muted", wrap=True)
        sb.addWidget(self.step_hint)
        sb.addSpacing(8)
        sb.addWidget(label(f"v{__version__}", "Muted"))
        body.addWidget(self.sidebar)

        # pages -------------------------------------------------------------
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        self.stack = QStackedWidget()
        self.start_page = StartPage(self)
        self.images_page = ImagesPage(self.state)
        self.backgrounds_page = BackgroundsPage(self.state)
        self.defects_page = DefectsPage(self.state)
        self.recipe_page = RecipePage(self.state)
        self.generate_page = GeneratePage(self.state)
        self.save_page = SavePage(self.state)
        self.pages = [
            self.images_page, self.backgrounds_page, self.defects_page,
            self.recipe_page, self.generate_page, self.save_page,
        ]
        self.stack.addWidget(self.start_page)
        for p in self.pages:
            self.stack.addWidget(p)
        right.addWidget(self.stack, 1)

        self.footer = QFrame()
        self.footer.setObjectName("Header")
        self.footer.setFixedHeight(56)
        f = QHBoxLayout(self.footer)
        f.setContentsMargins(20, 0, 20, 0)
        self.back_btn = QPushButton("← Back")
        self.next_btn = QPushButton("Next →")
        self.next_btn.setObjectName("Primary")
        self.next_btn.setMinimumWidth(130)
        self.footer_hint = label("", "Muted")
        f.addWidget(self.back_btn)
        f.addSpacing(10)
        f.addWidget(self.footer_hint, 1)
        f.addWidget(self.next_btn)
        right.addWidget(self.footer)
        body.addLayout(right, 1)

        # wiring ------------------------------------------------------------
        self.btn_new.clicked.connect(self.new_project)
        self.btn_open.clicked.connect(lambda: self.open_project())
        self.btn_save.clicked.connect(self.save_project)
        self.btn_help.clicked.connect(self.show_shortcuts)
        self.back_btn.clicked.connect(lambda: self.go(self._neighbor(self.current_step(), -1)))
        self.next_btn.clicked.connect(lambda: self.go(self._neighbor(self.current_step(), +1)))
        self.name_edit.editingFinished.connect(self._rename)
        self.state.dirtyChanged.connect(self._update_title)
        for sig in (self.state.imagesChanged, self.state.defectsChanged, self.state.settingsChanged,
                    self.state.datasetChanged, self.state.projectReplaced, self.state.backgroundsChanged):
            sig.connect(self._update_nav)
        self.generate_page.generationFinished.connect(self._update_nav)

        for key, fn in (("Ctrl+S", self.save_project), ("Ctrl+O", self.open_project), ("Ctrl+N", self.new_project)):
            act = QAction(self)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(lambda _=False, fn=fn: fn())
            self.addAction(act)

        self.show_start()

    # navigation --------------------------------------------------------
    def show_start(self) -> None:
        self.start_page.refresh(self._recent())
        self.stack.setCurrentWidget(self.start_page)
        self.sidebar.setVisible(False)
        self.footer.setVisible(False)
        self.name_edit.setVisible(False)
        self.btn_save.setVisible(False)
        self._update_title()

    def current_step(self) -> int:
        w = self.stack.currentWidget()
        return self.pages.index(w) if w in self.pages else -1

    def _step_ready(self, step: int) -> tuple[bool, str]:
        proj = self.state.project
        if step >= STEP_BACKGROUNDS and not proj.images:
            return False, "Add at least one image first."
        if step == STEP_BACKGROUNDS and not proj.backgrounds.enabled:
            return False, "Tick “Create background images” in step 1 to use this step."
        if step >= STEP_RECIPE and not proj.defects:
            return False, "Pick at least one defect first."
        if step >= STEP_SAVE and self.state.manifest is None:
            return False, "Generate the dataset first."
        return True, ""

    def _neighbor(self, step: int, direction: int) -> int:
        nxt = step + direction
        if nxt == STEP_BACKGROUNDS and not self.state.project.backgrounds.enabled:
            nxt += direction
        return max(0, min(len(self.pages) - 1, nxt))

    def go(self, step: int) -> None:
        step = max(0, min(len(self.pages) - 1, step))
        ok, why = self._step_ready(step)
        if not ok:
            self.footer_hint.setText(why)
            return
        self.sidebar.setVisible(True)
        self.footer.setVisible(True)
        self.name_edit.setVisible(True)
        self.btn_save.setVisible(True)
        self.stack.setCurrentWidget(self.pages[step])
        self._update_nav()

    def _update_nav(self, *_) -> None:
        step = self.current_step()
        for i, b in enumerate(self.nav_buttons):
            ok, why = self._step_ready(i)
            b.setEnabled(ok)
            b.setToolTip(why)
            b.setChecked(i == step)
        if step < 0:
            return
        self.back_btn.setEnabled(step > 0)
        last = step == len(self.pages) - 1
        self.next_btn.setVisible(not last)
        if not last:
            nxt = self._neighbor(step, +1)
            ok, why = self._step_ready(nxt)
            self.next_btn.setEnabled(ok)
            self.next_btn.setText(f"{STEPS[nxt]} →")
            self.footer_hint.setText("" if ok else why)
        else:
            self.footer_hint.setText("")
        hints = [
            "Add your real images and mark which ones can receive defects.",
            "Cut images into pieces, drop the bad ones and stitch new clean backgrounds.",
            "Brush, box or lasso each defect you want to reproduce.",
            "Choose how many images per defect and how much they may vary.",
            "Run the generation and review the results.",
            "Copy the dataset to its final folder.",
        ]
        self.step_hint.setText(hints[step])

    # project handling ----------------------------------------------------
    def _confirm_discard(self) -> bool:
        if self.generate_page.is_running():
            QMessageBox.information(self, "Generation running", "Cancel the running generation first.")
            return False
        if not self.state.dirty:
            return True
        r = QMessageBox.question(
            self, "Unsaved changes", "Save the current project before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            return self.save_project()
        return True

    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        name, ok = QInputDialog.getText(self, "New dataset", "Dataset name:", text="My dataset")
        if not ok:
            return
        self.state.replace_project(Project(name=name.strip() or "My dataset"))
        self.state.mark_dirty(True)
        self.name_edit.setText(self.state.project.name)
        self.go(0)

    def open_project(self, path: str | None = None) -> None:
        if not self._confirm_discard():
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open project", "", f"SegSynth project ({PROJECT_FILE})")
            if not path:
                return
        try:
            proj = Project.load(path)
            missing = [e.name for e in proj.images if not Path(e.path).exists()]
            for e in list(proj.images):
                if not Path(e.path).exists():
                    proj.remove_image(e.id)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open project", str(exc))
            self._forget_recent(path)
            return
        self.state.replace_project(proj, Path(path))
        self.name_edit.setText(proj.name)
        self._remember(path)
        if missing:
            QMessageBox.warning(self, "Missing images",
                                "These images were not found and were removed with their defects:\n" + "\n".join(missing))
            self.state.mark_dirty(True)
        self.go(STEP_RECIPE if proj.defects else STEP_IMAGES)

    def save_project(self) -> bool:
        if self.stack.currentWidget() is self.start_page:
            return False
        path = self.state.project_path
        if path is None:
            base = QFileDialog.getExistingDirectory(self, "Choose where to store the project folder")
            if not base:
                return False
            name = "".join(c if c.isalnum() or c in " -_" else "_" for c in self.state.project.name).strip() or "project"
            folder = Path(base) / name
            if (folder / PROJECT_FILE).exists():
                r = QMessageBox.question(self, "Overwrite?", f"A project already exists in\n{folder}\nOverwrite it?")
                if r != QMessageBox.Yes:
                    return False
        else:
            folder = path.parent
        try:
            saved = self.state.project.save(folder)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        self.state.project_path = saved
        self.state.mark_dirty(False)
        self._remember(str(saved))
        self.statusBar().showMessage(f"Project saved to {saved}", 4000)
        return True

    def _rename(self) -> None:
        name = self.name_edit.text().strip()
        if name and name != self.state.project.name:
            self.state.project.name = name
            self.state.mark_dirty(True)
            self.state.datasetChanged.emit()
        elif not name:
            self.name_edit.setText(self.state.project.name)

    def _update_title(self, *_) -> None:
        on_start = self.stack.currentWidget() is self.start_page
        dirty = self.state.dirty and not on_start
        self.dirty_dot.setText("● unsaved" if dirty else "")
        title = APP_NAME if on_start else f"{self.state.project.name}{' *' if dirty else ''} — {APP_NAME}"
        self.setWindowTitle(title)

    def _recent(self) -> list[str]:
        items = self.settings.value("recent", [], type=list) or []
        return [p for p in items if Path(p).exists()][:MAX_RECENT]

    def _remember(self, path: str) -> None:
        path = str(Path(path).resolve())
        items = [p for p in self._recent() if p != path]
        self.settings.setValue("recent", [path] + items[: MAX_RECENT - 1])

    def _forget_recent(self, path: str) -> None:
        self.settings.setValue("recent", [p for p in self._recent() if p != str(Path(path).resolve())])

    def show_shortcuts(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Keyboard shortcuts")
        box.setTextFormat(Qt.RichText)
        box.setText(SHORTCUTS_HTML)
        box.exec()

    def closeEvent(self, e) -> None:
        if self.generate_page.is_running():
            r = QMessageBox.question(self, "Generation running", "Stop the generation and quit?")
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self.generate_page.stop()
            self.generate_page.worker.wait(5000)
        elif self.stack.currentWidget() is not self.start_page and not self._confirm_discard():
            e.ignore()
            return
        self.state.discard_dataset()
        e.accept()
