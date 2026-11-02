"""Step 2 (optional) — cut images into pieces, mark the bad ones, and stitch new backgrounds."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractButton, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from ...core.backgrounds import HORIZONTAL, VERTICAL, BackgroundComposer, split_view
from ..qt_utils import label, np_to_pixmap, page_header, to_display_u8
from ..state import AppState
from ..theme import ACCENT, BORDER, DANGER

CARD_WIDTH = 250
PREVIEW_TILES = 8


class PieceButton(QAbstractButton):
    """A clickable piece of a source image: checked = good, unchecked = not good."""

    def __init__(self, pixmap, direction: str):
        super().__init__()
        self.pm = pixmap
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        w, h = pixmap.width(), pixmap.height()
        self.setFixedSize(QSize(w, h))
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._hover = False

    def enterEvent(self, e) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, e) -> None:
        self._hover = False
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()
        p.drawPixmap(r, self.pm)
        if not self.isChecked():
            p.fillRect(r, QColor(10, 12, 16, 165))
            p.setPen(QPen(QColor(DANGER), 2))
            p.drawRect(QRectF(r).adjusted(1, 1, -1, -1))
            f = QFont(self.font())
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor(DANGER))
            p.drawText(r, Qt.AlignCenter, "✕  Not good")
        else:
            color = QColor(ACCENT) if self._hover else QColor(BORDER)
            p.setPen(QPen(color, 2 if self._hover else 1))
            p.drawRect(QRectF(r).adjusted(0.5, 0.5, -0.5, -0.5))
        p.end()


class ImageCard(QFrame):
    toggled = Signal(str, int, bool)

    def __init__(self, entry, pieces: list, direction: str, good: list[bool], note: str = ""):
        super().__init__()
        self.setObjectName("Card")
        self.setFixedWidth(CARD_WIDTH + 24)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(8)
        title = label(entry.name)
        title.setStyleSheet("font-weight: 600; background: transparent;")
        title.setToolTip(entry.path)
        lay.addWidget(title)
        if note:
            lay.addWidget(label(note, "Muted", wrap=True))
        box = QVBoxLayout() if direction == HORIZONTAL else QHBoxLayout()
        box.setSpacing(4)
        box.setContentsMargins(0, 0, 0, 0)
        self.buttons: list[PieceButton] = []
        for slot, pm in enumerate(pieces):
            b = PieceButton(pm, direction)
            b.setChecked(good[slot])
            b.setEnabled(not note)
            b.setToolTip("Click to mark this piece as not good" if good[slot] else "Click to mark this piece as good")
            b.toggled.connect(lambda v, s=slot, btn=b: self._on_toggle(entry.id, s, v, btn))
            self.buttons.append(b)
            box.addWidget(b, 0, Qt.AlignLeft)
        lay.addLayout(box)
        lay.addStretch(1)

    def _on_toggle(self, image_id: str, slot: int, good: bool, btn: PieceButton) -> None:
        btn.setToolTip("Click to mark this piece as not good" if good else "Click to mark this piece as good")
        self.toggled.emit(image_id, slot, good)


class ClickableLabel(QLabel):
    clicked = Signal()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self.clicked.emit()


class BackgroundsPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.composer: BackgroundComposer | None = None
        self._display: dict[str, np.ndarray] = {}
        self._cards: list[ImageCard] = []
        self._columns = 0
        self._preview_combos: list = []
        self._loading = False

        root = QHBoxLayout(self)
        root.setContentsMargins(28, 24, 20, 20)
        root.setSpacing(20)

        left = QVBoxLayout()
        left.setSpacing(12)
        left.addWidget(page_header(
            "Background generating",
            "Each image is cut into pieces. Click every piece that is not good (scratches, dents, "
            "dirt). The good pieces of different images are aligned and stitched into new clean "
            "backgrounds. A piece always stays in its own position, so a top piece is only used on top.",
        ))

        controls = QFrame()
        controls.setObjectName("Card")
        c = QGridLayout(controls)
        c.setContentsMargins(16, 12, 16, 12)
        c.setHorizontalSpacing(10)
        c.setVerticalSpacing(10)
        self.direction = QComboBox()
        self.direction.addItem("Horizontal cuts (top / bottom)", HORIZONTAL)
        self.direction.addItem("Vertical cuts (left / right)", VERTICAL)
        self.direction.setToolTip("Horizontal cuts work best when the texture runs top to bottom,\n"
                                  "because the seam then crosses the grooves instead of following them.")
        self.pieces = QSpinBox()
        self.pieces.setRange(2, 8)
        self.pieces.setToolTip("Number of pieces each image is cut into.")
        self.seam = QSpinBox()
        self.seam.setRange(0, 1000)
        self.seam.setSingleStep(20)
        self.seam.setSuffix(" px")
        self.seam.setToolTip("Width of the soft blend across each seam. Each piece contributes up to\n"
                             "half of this width beyond its own edge. 0 = hard cut.")
        self.align = QCheckBox("Align pieces")
        self.align.setToolTip("Shift every image so its texture lines up with the first image before stitching.")
        self.max_count = QSpinBox()
        self.max_count.setRange(1, 100000)
        self.max_count.setSingleStep(50)
        self.max_count.setToolTip("Upper limit; when more combinations exist, a random selection is used.")
        self.include = QCheckBox("Add to dataset as good images")
        self.include.setToolTip("Writes every generated background into the dataset; in the label tool\n"
                                "database they are labeled good.")
        self.direction.setMinimumWidth(230)
        for w in (self.pieces, self.seam, self.max_count):
            w.setMinimumWidth(90)
        c.addWidget(label("Cut", "Muted"), 0, 0)
        c.addWidget(self.direction, 0, 1)
        c.addWidget(label("Pieces per image", "Muted"), 0, 2)
        c.addWidget(self.pieces, 0, 3)
        c.addWidget(label("Seam blend", "Muted"), 0, 4)
        c.addWidget(self.seam, 0, 5)
        c.addWidget(label("Max backgrounds", "Muted"), 1, 0)
        c.addWidget(self.max_count, 1, 1, Qt.AlignLeft)
        c.addWidget(self.align, 1, 2, 1, 2)
        c.addWidget(self.include, 1, 4, 1, 2)
        c.setColumnStretch(6, 1)
        left.addWidget(controls)

        bar = QHBoxLayout()
        self.all_good = QPushButton("Mark all pieces good")
        self.all_good.setObjectName("Ghost")
        self.counts = label("", "Muted")
        bar.addWidget(self.all_good)
        bar.addStretch(1)
        bar.addWidget(self.counts)
        left.addLayout(bar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(12)
        self.grid.setContentsMargins(0, 0, 8, 0)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.grid_host)
        left.addWidget(self.scroll, 1)
        root.addLayout(left, 1)

        right_w = QWidget()
        right_w.setFixedWidth(380)
        right = QVBoxLayout(right_w)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        self.total = label("", "StatusBadge")
        right.addWidget(self.total, 0, Qt.AlignLeft)
        self.total_note = label("", "Muted", wrap=True)
        right.addWidget(self.total_note)
        head = QHBoxLayout()
        head.addWidget(label("PREVIEW", "SectionLabel"))
        head.addStretch(1)
        self.shuffle = QPushButton("Shuffle preview")
        head.addWidget(self.shuffle)
        right.addLayout(head)
        tiles = QGridLayout()
        tiles.setSpacing(8)
        self.tiles: list[ClickableLabel] = []
        for i in range(PREVIEW_TILES):
            t = ClickableLabel()
            t.setObjectName("Card")
            t.setFixedSize(182, 124)
            t.setAlignment(Qt.AlignCenter)
            t.setCursor(Qt.PointingHandCursor)
            t.setToolTip("Click to open at full resolution")
            t.clicked.connect(lambda i=i: self._open_preview(i))
            self.tiles.append(t)
            tiles.addWidget(t, i // 2, i % 2)
        right.addLayout(tiles)
        self.preview_note = label("", "Muted", wrap=True)
        right.addWidget(self.preview_note)
        right.addStretch(1)
        root.addWidget(right_w)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(lambda: self.update_preview(reshuffle=False))

        self.direction.currentIndexChanged.connect(self._layout_changed)
        self.pieces.valueChanged.connect(self._layout_changed)
        self.seam.valueChanged.connect(self._options_changed)
        self.align.toggled.connect(self._options_changed)
        self.max_count.valueChanged.connect(self._options_changed)
        self.include.toggled.connect(self._options_changed)
        self.all_good.clicked.connect(self._mark_all_good)
        self.shuffle.clicked.connect(lambda: self.update_preview(reshuffle=True))
        state.imagesChanged.connect(self._sources_changed)
        state.defectsChanged.connect(self._sources_changed)
        state.backgroundsChanged.connect(self._load_settings)
        state.projectReplaced.connect(self._sources_changed)
        state.settingsChanged.connect(lambda: self._schedule_preview(rebuild=True))

    # settings -----------------------------------------------------------
    @property
    def bg(self):
        return self.state.project.backgrounds

    def _load_settings(self) -> None:
        self._loading = True
        self.direction.setCurrentIndex(max(0, self.direction.findData(self.bg.direction)))
        self.pieces.setValue(self.bg.pieces)
        self.seam.setValue(self.bg.seam)
        self.align.setChecked(self.bg.align)
        self.max_count.setValue(self.bg.max_count)
        self.include.setChecked(self.bg.include_in_dataset)
        self._loading = False
        if self.isVisible():
            self.rebuild()

    def _layout_changed(self, *_) -> None:
        if self._loading:
            return
        direction, pieces = self.direction.currentData(), self.pieces.value()
        if direction == self.bg.direction and pieces == self.bg.pieces:
            return
        if self.bg.rejected:
            r = QMessageBox.question(
                self, "Change the cut?",
                f"Changing the cut clears the {len(self.bg.rejected)} piece(s) marked not good. Continue?")
            if r != QMessageBox.Yes:
                self._loading = True
                self.direction.setCurrentIndex(self.direction.findData(self.bg.direction))
                self.pieces.setValue(self.bg.pieces)
                self._loading = False
                return
        self.bg.direction, self.bg.pieces = direction, pieces
        self.bg.rejected = []
        self.state.backgroundsChanged.emit()

    def _options_changed(self, *_) -> None:
        if self._loading:
            return
        self.bg.seam = self.seam.value()
        self.bg.align = self.align.isChecked()
        self.bg.max_count = self.max_count.value()
        self.bg.include_in_dataset = self.include.isChecked()
        self.state.mark_dirty()
        self.composer = None
        self._update_counts()
        self._schedule_preview()

    def _mark_all_good(self) -> None:
        if self.bg.rejected:
            self.bg.rejected = []
            self.state.backgroundsChanged.emit()

    # pieces grid ----------------------------------------------------------
    def _sources_changed(self, *_) -> None:
        self._display.clear()
        self.composer = None
        if self.isVisible():
            self.rebuild()

    def _get_composer(self) -> BackgroundComposer | None:
        if not self.state.project.images:
            return None
        if self.composer is None:
            self.composer = BackgroundComposer(self.state.project)
        return self.composer

    def _display_image(self, image_id: str) -> np.ndarray:
        if image_id not in self._display:
            proj = self.state.project
            img = proj.load_pixels(image_id)
            h, w = img.shape[:2]
            scale = CARD_WIDTH / w if self.bg.direction == HORIZONTAL else min(CARD_WIDTH / w, 1.0)
            small = to_display_u8(cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                                             interpolation=cv2.INTER_AREA))
            if small.ndim == 2:
                small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            small = small.copy()
            mask = proj.marked_mask(image_id, (h, w))
            if mask.any():
                m = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
                m = cv2.dilate(m, np.ones((3, 3), np.uint8))
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(small, contours, -1, (108, 92, 255), 1)
            self._display[image_id] = small
        return self._display[image_id]

    def _columns_for_width(self) -> int:
        avail = self.scroll.viewport().width() - 8
        return max(1, avail // (CARD_WIDTH + 24 + 12))

    def rebuild(self) -> None:
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards = []
        self.composer = None
        comp = self._get_composer()
        if comp is None:
            self._update_counts()
            self.update_preview(reshuffle=True)
            return
        compatible = {e.id for e in comp.sources}
        self._columns = self._columns_for_width()
        for i, entry in enumerate(self.state.project.images):
            disp = self._display_image(entry.id)
            pieces = [np_to_pixmap(p) for p in split_view(disp, self.bg.direction, self.bg.pieces)]
            good = [not self.bg.is_rejected(entry.id, s) for s in range(self.bg.pieces)]
            note = "" if entry.id in compatible else "Different size or type than the first image — not used."
            card = ImageCard(entry, pieces, self.bg.direction, good, note)
            card.toggled.connect(self._piece_toggled)
            self._cards.append(card)
            self.grid.addWidget(card, i // self._columns, i % self._columns)
        self._update_counts()
        self.update_preview(reshuffle=True)

    def _piece_toggled(self, image_id: str, slot: int, good: bool) -> None:
        self.bg.set_rejected(image_id, slot, not good)
        self.state.mark_dirty()
        self.composer = None
        self._update_counts()
        self._schedule_preview()

    def _update_counts(self) -> None:
        comp = self._get_composer()
        if comp is None:
            self.counts.setText("")
            self.total.setText("No backgrounds yet")
            self.total_note.setText("Add images in step 1.")
            return
        good, total = comp.counts()
        possible = comp.possible()
        created = len(comp.combos())
        self.counts.setText(f"{good} of {total} pieces good")
        self.total.setText(f"{created:,} background{'s' if created != 1 else ''} will be created")
        if possible == 0:
            self.total_note.setText("Every position needs at least one good piece, and the pieces must "
                                    "come from at least two different images.")
        elif created < possible:
            self.total_note.setText(f"{possible:,} combinations are possible; a random {created:,} are used "
                                    f"(raise “Max” for more). Combinations that only rebuild an original "
                                    f"image are skipped.")
        else:
            self.total_note.setText("Every combination of good pieces from different images. Combinations "
                                    "that only rebuild an original image are skipped.")

    # preview -------------------------------------------------------------
    def _schedule_preview(self, rebuild: bool = False) -> None:
        if rebuild:
            self.composer = None
        if self.isVisible():
            self._preview_timer.start()

    def update_preview(self, reshuffle: bool = True) -> None:
        comp = self._get_composer()
        combos = comp.combos() if comp else []
        if not combos:
            self._preview_combos = []
            for t in self.tiles:
                t.clear()
            self.preview_note.setText("")
            return
        if reshuffle or not self._preview_combos or any(c not in combos for c in self._preview_combos):
            rng = np.random.default_rng()
            k = min(PREVIEW_TILES, len(combos))
            self._preview_combos = [combos[i] for i in rng.choice(len(combos), size=k, replace=False)]
        w = comp.shape[1]
        scale = min(1.0, 182 / w, 124 / comp.shape[0])
        for i, tile in enumerate(self.tiles):
            if i < len(self._preview_combos):
                img, mask = comp.render(self._preview_combos[i], scale=scale)
                disp = to_display_u8(img)
                if disp.ndim == 2:
                    disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
                if mask is not None and mask.any():
                    disp = disp.copy()
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(disp, contours, -1, (108, 92, 255), 1)
                tile.setPixmap(np_to_pixmap(disp))
                tile.setToolTip(comp.describe(self._preview_combos[i]) + "\nClick to open at full resolution")
            else:
                tile.clear()
        self.preview_note.setText("Random examples. Red outlines are defects you picked in step 3; such "
                                  "backgrounds are labeled bad.")

    def _open_preview(self, i: int) -> None:
        if i >= len(self._preview_combos):
            return
        from .generate import ResultViewer
        comp = self._get_composer()
        combo = self._preview_combos[i]
        self.setCursor(Qt.WaitCursor)
        try:
            img, mask = comp.render(combo)
        finally:
            self.unsetCursor()
        ResultViewer(img, mask, comp.describe(combo), self).exec()

    # qt ----------------------------------------------------------------
    def showEvent(self, e) -> None:
        super().showEvent(e)
        self._load_settings()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self._cards and self._columns_for_width() != self._columns:
            self._columns = self._columns_for_width()
            for i, card in enumerate(self._cards):
                self.grid.addWidget(card, i // self._columns, i % self._columns)

