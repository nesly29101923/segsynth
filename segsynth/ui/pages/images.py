"""Step 1 — add source images and choose which ones serve as backgrounds."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QStackedLayout, QVBoxLayout, QWidget,
)

from ...core.imageio import SUPPORTED_EXTENSIONS, list_images
from ..qt_utils import label, np_to_pixmap, page_header, thumbnail
from ..state import AppState

FILTER = "Images (" + " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS) + ")"


class ImageRow(QWidget):
    toggled = Signal(str, bool)
    removed = Signal(str)

    def __init__(self, entry, pixmap, size_text: str, n_defects: int):
        super().__init__()
        self.setObjectName("Card")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(14)
        thumb = QLabel()
        thumb.setPixmap(pixmap)
        thumb.setFixedSize(132, 90)
        thumb.setAlignment(Qt.AlignCenter)
        lay.addWidget(thumb)

        info = QVBoxLayout()
        info.setSpacing(2)
        name = label(entry.name)
        name.setStyleSheet("font-weight: 600;")
        info.addWidget(name)
        info.addWidget(label(size_text, "Muted"))
        info.addWidget(label(
            f"{n_defects} defect{'s' if n_defects != 1 else ''} picked" if n_defects else "No defects picked",
            "Muted"))
        info.addStretch(1)
        lay.addLayout(info, 1)

        self.target = QCheckBox("Use as background")
        self.target.setChecked(entry.use_as_target)
        self.target.setToolTip("Synthetic defects can be pasted onto this image.")
        self.target.toggled.connect(lambda v: self.toggled.emit(entry.id, v))
        lay.addWidget(self.target)

        rm = QPushButton("Remove")
        rm.setObjectName("Ghost")
        rm.clicked.connect(lambda: self.removed.emit(entry.id))
        lay.addWidget(rm)


class ImagesPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setAcceptDrops(True)
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)
        root.addWidget(page_header(
            "Source images",
            "Add the real images you have. Defects are picked from them, and any image marked as a "
            "background receives synthetic defects. You can also drop files or folders here.",
        ))

        bar = QHBoxLayout()
        self.add_files = QPushButton("Add images…")
        self.add_files.setObjectName("Primary")
        self.add_folder = QPushButton("Add folder…")
        self.all_targets = QPushButton("Use all as backgrounds")
        self.all_targets.setObjectName("Ghost")
        self.create_bgs = QCheckBox("Create background images")
        self.create_bgs.setToolTip(
            "Turns on the “Background generating” step: cut these images into pieces, drop the\n"
            "pieces that are not good, and stitch the rest into new clean backgrounds.")
        self.summary = label("", "Muted")
        for w in (self.add_files, self.add_folder, self.all_targets):
            bar.addWidget(w)
        bar.addSpacing(8)
        bar.addWidget(self.create_bgs)
        bar.addStretch(1)
        bar.addWidget(self.summary)
        root.addLayout(bar)

        self.stack = QStackedLayout()
        self.empty = label("Drop images here or click “Add images…”\nPNG, TIFF, BMP and JPEG · 8/16-bit · gray or color", "EmptyState")
        self.empty.setAlignment(Qt.AlignCenter)
        empty_wrap = QWidget()
        empty_wrap.setObjectName("Card")
        ew = QVBoxLayout(empty_wrap)
        ew.addWidget(self.empty)
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.NoSelection)
        self.list.setSpacing(4)
        self.list.setStyleSheet("QListWidget { background: transparent; border: none; }"
                                "QListWidget::item:hover { background: transparent; }")
        self.stack.addWidget(empty_wrap)
        self.stack.addWidget(self.list)
        root.addLayout(self.stack, 1)

        self.add_files.clicked.connect(self._pick_files)
        self.add_folder.clicked.connect(self._pick_folder)
        self.all_targets.clicked.connect(self._all_targets)
        self.create_bgs.toggled.connect(self._toggle_backgrounds)
        state.backgroundsChanged.connect(self._sync_backgrounds)
        state.imagesChanged.connect(self.refresh)
        state.defectsChanged.connect(self.refresh)
        self._thumbs: dict[str, object] = {}
        self.refresh()

    def _pick_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add images", "", FILTER)
        self.add_paths([Path(f) for f in files])

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Add all images in folder")
        if d:
            self.add_paths(list_images(d))

    def add_paths(self, paths: list[Path]) -> None:
        if not paths:
            return
        failed = []
        proj = self.state.project
        for p in paths:
            try:
                entry = proj.add_image(p)
                proj.load_pixels(entry.id)
            except Exception:
                failed.append(p.name)
                proj.images = [e for e in proj.images if Path(e.path) != Path(p).resolve()]
        self.state.imagesChanged.emit()
        if failed:
            QMessageBox.warning(self, "Some files were skipped", "Could not read:\n" + "\n".join(failed))

    def _all_targets(self) -> None:
        for e in self.state.project.images:
            e.use_as_target = True
        self.state.imagesChanged.emit()

    def _toggle_backgrounds(self, on: bool) -> None:
        bg = self.state.project.backgrounds
        if bg.enabled != on:
            bg.enabled = on
            self.state.backgroundsChanged.emit()

    def _sync_backgrounds(self) -> None:
        self.create_bgs.blockSignals(True)
        self.create_bgs.setChecked(self.state.project.backgrounds.enabled)
        self.create_bgs.blockSignals(False)
        self._update_summary()

    def _toggle(self, image_id: str, value: bool) -> None:
        self.state.project.image(image_id).use_as_target = value
        self.state.mark_dirty()
        self._update_summary()

    def _remove(self, image_id: str) -> None:
        proj = self.state.project
        n = len(proj.defects_of(image_id))
        if n:
            r = QMessageBox.question(self, "Remove image",
                                     f"This image has {n} picked defect(s). Remove the image and its defects?")
            if r != QMessageBox.Yes:
                return
        proj.remove_image(image_id)
        self._thumbs.pop(image_id, None)
        self.state.imagesChanged.emit()
        self.state.defectsChanged.emit()

    def _thumb(self, entry):
        if entry.id not in self._thumbs:
            img = self.state.project.load_pixels(entry.id)
            self._thumbs[entry.id] = (np_to_pixmap(thumbnail(img, 132)), img.shape, img.dtype)
        return self._thumbs[entry.id]

    def _update_summary(self) -> None:
        imgs = self.state.project.images
        n_t = sum(e.use_as_target for e in imgs)
        text = f"{len(imgs)} image{'s' if len(imgs) != 1 else ''} · {n_t} background{'s' if n_t != 1 else ''}"
        if self.state.project.backgrounds.enabled:
            text += " · + generated backgrounds"
        self.summary.setText(text if imgs else "")

    def refresh(self) -> None:
        self.list.clear()
        proj = self.state.project
        self.stack.setCurrentIndex(1 if proj.images else 0)
        for entry in proj.images:
            try:
                pm, shape, dtype = self._thumb(entry)
            except Exception:
                continue
            ch = "gray" if len(shape) == 2 else f"{shape[2]}-channel"
            bits = "16-bit" if str(dtype) == "uint16" else "8-bit"
            row = ImageRow(entry, pm, f"{shape[1]} × {shape[0]} px · {ch} · {bits}", len(proj.defects_of(entry.id)))
            row.toggled.connect(self._toggle)
            row.removed.connect(self._remove)
            item = QListWidgetItem(self.list)
            item.setSizeHint(QSize(0, 108))
            self.list.setItemWidget(item, row)
        self._update_summary()

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        paths = []
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir():
                paths.extend(list_images(p))
            elif p.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(p)
        self.add_paths(paths)
