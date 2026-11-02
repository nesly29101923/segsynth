"""Step 4 — run the generation and review the results."""
from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from ...core.generator import GenerationError, list_targets, make_composer, run_to_folder, validate
from ...core.imageio import read_image, read_mask, write_image
from ..canvas import MaskCanvas, Tool
from ..qt_utils import card, label, page_header, thumbnail, untinted_icon
from ..state import AppState

PREVIEW_MASKS = ".preview_masks"


class GenerationWorker(QThread):
    progressed = Signal(int, int, object, str)
    status = Signal(str)
    failed = Signal(str)
    finished_ok = Signal(dict)

    def __init__(self, project, out_dir: Path):
        super().__init__()
        self.project = project
        self.out_dir = out_dir
        self.stop = threading.Event()

    def run(self) -> None:
        pm = self.out_dir / PREVIEW_MASKS

        def on_progress(done, total, sample):
            name = Path(sample.files["image"]).stem
            write_image(pm / f"{name}.png", sample.mask * 255)
            self.progressed.emit(done, total, thumbnail(sample.image, 200), sample.files["image"])

        try:
            manifest = run_to_folder(self.project, self.out_dir, on_progress, self.stop, self.status.emit)
            self.finished_ok.emit(manifest)
        except GenerationError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc(limit=4)}")


class ResultViewer(QDialog):
    def __init__(self, image, mask, title: str, parent=None, highlight_text: str = "Highlight defects"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 820)
        lay = QVBoxLayout(self)
        self.canvas = MaskCanvas()
        self.canvas.set_tool(Tool.PAN)
        self.canvas.set_image(image)
        self.mask = mask if mask is not None and mask.any() else None
        lay.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.show_area = QCheckBox(highlight_text)
        self.show_area.setEnabled(self.mask is not None)
        self.show_area.toggled.connect(self._toggle)
        row.addWidget(self.show_area)
        row.addStretch(1)
        row.addWidget(label("Wheel to zoom · drag to pan", "Muted"))
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)

    def _toggle(self, on: bool) -> None:
        self.canvas.set_committed([(self.mask, QColor("#ff4d5e"))] if on and self.mask is not None else [])


class GeneratePage(QWidget):
    generationFinished = Signal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.worker: GenerationWorker | None = None
        self._started = 0.0

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)
        root.addWidget(page_header(
            "Generate",
            "Runs the recipe and creates the synthetic images. Results appear below as they are made; "
            "double-click one to inspect it at full resolution.",
        ))

        stats = QHBoxLayout()
        stats.setSpacing(12)
        self.stat_images = self._stat(stats, "Images to generate")
        self.stat_defects = self._stat(stats, "Defects in use")
        self.stat_bgs = self._stat(stats, "Backgrounds")
        root.addLayout(stats)

        row = QHBoxLayout()
        self.start = QPushButton("Start generation")
        self.start.setObjectName("Primary")
        self.start.setMinimumWidth(180)
        self.cancel = QPushButton("Cancel")
        self.cancel.setEnabled(False)
        self.status = label("", "Muted")
        row.addWidget(self.start)
        row.addWidget(self.cancel)
        row.addSpacing(10)
        row.addWidget(self.status, 1)
        root.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        root.addWidget(self.progress)

        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setIconSize(QSize(200, 136))
        self.gallery.setGridSize(QSize(222, 176))
        self.gallery.setResizeMode(QListWidget.Adjust)
        self.gallery.setMovement(QListWidget.Static)
        self.gallery.setUniformItemSizes(True)
        self.gallery.setSpacing(4)
        root.addWidget(self.gallery, 1)

        self.start.clicked.connect(self.run)
        self.cancel.clicked.connect(self.stop)
        self.gallery.itemDoubleClicked.connect(self._open)
        for sig in (state.defectsChanged, state.imagesChanged, state.settingsChanged, state.projectReplaced,
                    state.backgroundsChanged):
            sig.connect(self.refresh_stats)
        state.datasetChanged.connect(self._on_dataset_changed)
        self.refresh_stats()

    def _stat(self, parent_layout, caption: str):
        lay = QVBoxLayout()
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(2)
        value = label("0", "Big")
        lay.addWidget(value)
        lay.addWidget(label(caption, "Muted"))
        parent_layout.addWidget(card(lay), 1)
        return value

    def refresh_stats(self, *_) -> None:
        proj = self.state.project
        self.stat_images.setText(f"{proj.total_images:,}")
        self.stat_defects.setText(str(sum(1 for d in proj.defects if d.enabled and d.count > 0)))
        try:
            n_bgs = len(list_targets(proj, make_composer(proj)))
        except Exception:
            n_bgs = sum(1 for e in proj.images if e.use_as_target)
        self.stat_bgs.setText(f"{n_bgs:,}")
        if self.worker is None:
            problems = validate(proj)
            self.start.setEnabled(not problems)
            self.start.setToolTip("\n".join(problems))
            if self.state.manifest is None:
                self.status.setText(problems[0] if problems else "Ready.")

    def _on_dataset_changed(self) -> None:
        if self.state.staging is None and self.worker is None:
            self.gallery.clear()
            self.progress.setValue(0)

    def is_running(self) -> bool:
        return self.worker is not None

    def run(self) -> None:
        if self.worker is not None:
            return
        if self.state.manifest is not None:
            r = QMessageBox.question(self, "Replace dataset?",
                                     "A generated dataset already exists and will be replaced. Continue?")
            if r != QMessageBox.Yes:
                return
        out = self.state.new_staging()
        self.gallery.clear()
        total = self.state.project.total_images
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(0)
        self.worker = GenerationWorker(self.state.project, out)
        self.worker.progressed.connect(self._on_progress)
        self.worker.status.connect(self.status.setText)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.finished.connect(self._cleanup)
        self.start.setEnabled(False)
        self.cancel.setEnabled(True)
        self.status.setText("Preparing defects…")
        self._started = time.time()
        self.worker.start()

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop.set()
            self.status.setText("Cancelling…")
            self.cancel.setEnabled(False)

    def _on_progress(self, done: int, total: int, thumb, rel: str) -> None:
        self.progress.setValue(done)
        elapsed = time.time() - self._started
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        self.status.setText(f"{done:,} of {total:,} · {rate:.1f} img/s · about {self._fmt(eta)} left")
        item = QListWidgetItem(untinted_icon(thumb), Path(rel).name)
        item.setData(Qt.UserRole, rel)
        self.gallery.addItem(item)
        if done % 25 == 0:
            self.gallery.scrollToBottom()

    @staticmethod
    def _fmt(seconds: float) -> str:
        seconds = int(seconds)
        return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{seconds} s"

    def _on_failed(self, msg: str) -> None:
        self.status.setText("Generation failed.")
        QMessageBox.critical(self, "Generation failed", msg)
        self.state.discard_dataset()

    def _on_done(self, manifest: dict) -> None:
        self.state.manifest = manifest
        n = manifest["count"]
        took = self._fmt(manifest["duration_s"])
        if manifest["completed"]:
            self.status.setText(f"Done — {n:,} images in {took}. Continue to “Save dataset”.")
        else:
            self.status.setText(f"Cancelled — {n:,} images kept. You can still save them.")
        self.state.datasetChanged.emit()
        self.generationFinished.emit()

    def _cleanup(self) -> None:
        self.worker.deleteLater()
        self.worker = None
        self.cancel.setEnabled(False)
        self.refresh_stats()

    def _open(self, item: QListWidgetItem) -> None:
        if self.state.staging is None:
            return
        rel = Path(item.data(Qt.UserRole))
        mpath = self.state.staging / PREVIEW_MASKS / f"{rel.stem}.png"
        mask = read_mask(mpath) if mpath.exists() else None
        ResultViewer(read_image(self.state.staging / rel), mask, rel.name, self).exec()
