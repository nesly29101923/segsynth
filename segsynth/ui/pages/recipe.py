"""Step 3 — how many images per defect, and how much each copy may vary."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...core import transform as tf
from ...core.blend import blend
from ...core.generator import Generator
from ...core.imageio import max_value
from ...core.placement import find_position
from ..qt_utils import crop_preview, label, np_to_pixmap, page_header, untinted_icon
from ..state import AppState

PLACEMENTS = [
    ("random", "Anywhere"),
    ("keep_column", "Same column (move vertically)"),
    ("keep_row", "Same row (move horizontally)"),
    ("near_source", "Near original spot"),
]
BG_MODES = [
    ("vertical", "Vertical texture (grooves run top–bottom)"),
    ("horizontal", "Horizontal texture (grooves run left–right)"),
    ("isotropic", "No dominant direction"),
    ("none", "Off (copy raw pixels)"),
]
BLENDS = [
    ("residual", "Texture-preserving", "Adds only the defect on top of the target's own surface. Best for textured parts."),
    ("poisson", "Poisson (seamless)", "Gradient-domain blending of the raw defect pixels."),
    ("poisson_mixed", "Poisson (mixed)", "Keeps the stronger gradient of defect or target; good for thin scratches."),
    ("alpha", "Soft paste", "Feathered copy of the raw pixels."),
]


class RangeEdit(QWidget):
    def __init__(self, lo: float, hi: float, step: float, decimals: int, suffix: str = ""):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.a = QDoubleSpinBox()
        self.b = QDoubleSpinBox()
        for s in (self.a, self.b):
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setSuffix(suffix)
            s.setMinimumWidth(78)
            s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(self.a)
        lay.addWidget(label("to", "Muted"))
        lay.addWidget(self.b)

    def value(self) -> tuple[float, float]:
        a, b = self.a.value(), self.b.value()
        return (min(a, b), max(a, b))

    def set_value(self, v) -> None:
        for s in (self.a, self.b):
            s.blockSignals(True)
        self.a.setValue(float(v[0]))
        self.b.setValue(float(v[1]))
        for s in (self.a, self.b):
            s.blockSignals(False)

    def connect(self, fn) -> None:
        self.a.valueChanged.connect(fn)
        self.b.valueChanged.connect(fn)


class RecipePage(QWidget):
    COL_PREVIEW, COL_NAME, COL_SOURCE, COL_COUNT, COL_PLACE, COL_USE = range(6)

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._loading = False
        root = QHBoxLayout(self)
        root.setContentsMargins(28, 24, 20, 24)
        root.setSpacing(20)

        # left: recipe table + preview ----------------------------------
        left = QVBoxLayout()
        left.setSpacing(12)
        left.addWidget(page_header(
            "Recipe",
            "Set how many synthetic images to create for each defect. Every image gets a freshly "
            "varied copy of that defect at a new position on a random background.",
        ))
        bar = QHBoxLayout()
        bar.addWidget(label("Set all counts to", "Muted"))
        self.bulk = QSpinBox()
        self.bulk.setRange(0, 100000)
        self.bulk.setValue(20)
        self.bulk_apply = QPushButton("Apply")
        bar.addWidget(self.bulk)
        bar.addWidget(self.bulk_apply)
        bar.addStretch(1)
        self.total = label("", "StatusBadge")
        bar.addWidget(self.total)
        left.addLayout(bar)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["", "Defect", "Source image", "Images", "Placement", "Use"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setIconSize(QSize(64, 64))
        self.table.setShowGrid(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(self.COL_PREVIEW, QHeaderView.Fixed)
        hh.setSectionResizeMode(self.COL_NAME, QHeaderView.Stretch)
        hh.setSectionResizeMode(self.COL_SOURCE, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(self.COL_COUNT, QHeaderView.Fixed)
        hh.setSectionResizeMode(self.COL_PLACE, QHeaderView.Fixed)
        hh.setSectionResizeMode(self.COL_USE, QHeaderView.Fixed)
        self.table.setColumnWidth(self.COL_PREVIEW, 80)
        self.table.setColumnWidth(self.COL_COUNT, 100)
        self.table.setColumnWidth(self.COL_PLACE, 230)
        self.table.setColumnWidth(self.COL_USE, 56)
        self.table.verticalHeader().setDefaultSectionSize(76)
        left.addWidget(self.table, 3)

        prev_head = QHBoxLayout()
        prev_head.addWidget(label("PREVIEW VARIATIONS", "SectionLabel"))
        prev_head.addStretch(1)
        self.preview_btn = QPushButton("Shuffle preview")
        self.preview_btn.setToolTip("Show random variations of the selected defect with the current settings.")
        prev_head.addWidget(self.preview_btn)
        left.addLayout(prev_head)
        self.preview_grid = QGridLayout()
        self.preview_grid.setSpacing(8)
        self.preview_tiles: list[QLabel] = []
        for i in range(6):
            t = QLabel()
            t.setFixedSize(150, 150)
            t.setAlignment(Qt.AlignCenter)
            t.setObjectName("Card")
            self.preview_tiles.append(t)
            self.preview_grid.addWidget(t, 0, i)
        self.preview_grid.setColumnStretch(6, 1)
        left.addLayout(self.preview_grid)
        self.preview_note = label("Select a defect in the table to preview it.", "Muted")
        left.addWidget(self.preview_note)
        root.addLayout(left, 1)

        # right: settings ---------------------------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(430)
        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 8, 0)
        pl.setSpacing(4)
        scroll.setWidget(panel)
        root.addWidget(scroll)

        g = QGroupBox("Variation")
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignLeft)
        f.setVerticalSpacing(10)
        self.scale = RangeEdit(0.1, 5, 0.05, 2, "×")
        self.elong = RangeEdit(0.2, 5, 0.05, 2, "×")
        self.rot = RangeEdit(-180, 180, 1, 0, "°")
        self.bend = RangeEdit(-1, 1, 0.05, 2)
        self.elastic = QDoubleSpinBox()
        self.elastic.setRange(0, 40)
        self.elastic.setSingleStep(0.5)
        self.elastic.setSuffix(" px")
        self.intensity = RangeEdit(0, 3, 0.05, 2, "×")
        self.flip_h = QCheckBox("Allow horizontal flip")
        self.flip_v = QCheckBox("Allow vertical flip")
        f.addRow(self._tip("Size", "Uniform scale of the defect."), self.scale)
        f.addRow(self._tip("Elongation", "Stretch along the defect's own long axis. >1 makes scratches longer."), self.elong)
        f.addRow(self._tip("Rotation", "Extra rotation. Keep it small for shading-based images: lighting direction is baked in."), self.rot)
        f.addRow(self._tip("Bend", "Changes curvature along the long axis. Negative and positive bend opposite ways."), self.bend)
        f.addRow(self._tip("Wobble", "Small random elastic distortion of the outline."), self.elastic)
        f.addRow(self._tip("Contrast", "How strong the defect appears compared with the original."), self.intensity)
        flips = QVBoxLayout()
        flips.addWidget(self.flip_h)
        flips.addWidget(self.flip_v)
        flip_lbl = self._tip("Flips", "Flipping reverses light/shadow sides; leave off for photometric or shading images.")
        f.addRow(flip_lbl, flips)
        pl.addWidget(g)

        g = QGroupBox("Placement")
        f = QFormLayout(g)
        f.setVerticalSpacing(10)
        self.placement = QComboBox()
        for v, t in PLACEMENTS:
            self.placement.addItem(t, v)
        self.jitter = QSpinBox()
        self.jitter.setRange(0, 10000)
        self.jitter.setSuffix(" px")
        self.gap = QSpinBox()
        self.gap.setRange(0, 500)
        self.gap.setSuffix(" px")
        self.per_image = RangeEdit(1, 50, 1, 0)
        self.avoid = QCheckBox("Keep away from picked defects")
        f.addRow(self._tip("Default placement", "Per-defect placement in the table overrides this."), self.placement)
        f.addRow(self._tip("Movement range", "Maximum offset for the “same column/row” and “near original” modes."), self.jitter)
        f.addRow(self._tip("Minimum gap", "Clearance between pasted defects."), self.gap)
        f.addRow(self._tip("Defects per image", "1 = only the recipe defect. Higher adds random extra defects."), self.per_image)
        f.addRow("", self.avoid)
        pl.addWidget(g)

        g = QGroupBox("Blending")
        f = QFormLayout(g)
        f.setVerticalSpacing(10)
        self.bg_mode = QComboBox()
        for v, t in BG_MODES:
            self.bg_mode.addItem(t, v)
        self.bg_size = QSpinBox()
        self.bg_size.setRange(15, 2001)
        self.bg_size.setSingleStep(10)
        self.bg_size.setSuffix(" px")
        self.feather = QSpinBox()
        self.feather.setRange(0, 50)
        self.feather.setSuffix(" px")
        f.addRow(self._tip("Surface texture", "How the surface behind a defect is estimated and removed before pasting."), self.bg_mode)
        f.addRow(self._tip("Texture window", "Length of the background estimation window. Larger than the defect works best."), self.bg_size)
        f.addRow(self._tip("Edge softness", "Feathering around the defect outline."), self.feather)
        self.blend_checks: dict[str, QCheckBox] = {}
        blends = QVBoxLayout()
        for v, t, tip in BLENDS:
            cb = QCheckBox(t)
            cb.setToolTip(tip)
            self.blend_checks[v] = cb
            blends.addWidget(cb)
        f.addRow(self._tip("Methods", "If several are ticked, each paste picks one at random."), blends)
        pl.addWidget(g)

        g = QGroupBox("Output")
        f = QFormLayout(g)
        f.setVerticalSpacing(10)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.prefix = QComboBox()
        self.prefix.setEditable(True)
        self.prefix.addItems(["synth", "sample", "img"])
        self.ext = QComboBox()
        for e in (".png", ".tif", ".bmp"):
            self.ext.addItem(e[1:].upper(), e)
        self.masks = QCheckBox("Also save masks")
        self.masks.setToolTip("Binary masks of the pasted defects (255 = defect). Off when you label by hand.")
        self.overlays = QCheckBox("Also save review overlays")
        self.include_sources = QCheckBox("Include source images in the dataset")
        self.include_sources.setToolTip(
            "Copies every source image into the background folder of the dataset.\n"
            "Images with no picked defects are labeled good; images with picked defects are\n"
            "labeled not good with those defects as their mask.")
        f.addRow(self._tip("Random seed", "Same seed and settings give exactly the same dataset."), self.seed)
        f.addRow("File prefix", self.prefix)
        f.addRow("Format", self.ext)
        f.addRow("", self.masks)
        f.addRow("", self.overlays)
        f.addRow("", self.include_sources)
        pl.addWidget(g)
        reset = QPushButton("Reset settings to defaults")
        reset.setObjectName("Ghost")
        reset.clicked.connect(self._reset_defaults)
        pl.addWidget(reset)
        pl.addStretch(1)

        for combo in (self.placement, self.bg_mode, self.ext, self.prefix):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            combo.view().setMinimumWidth(320)

        # wiring --------------------------------------------------------
        for w in (self.scale, self.elong, self.rot, self.bend, self.intensity, self.per_image):
            w.connect(self._settings_from_ui)
        for w in (self.elastic, self.jitter, self.gap, self.bg_size, self.feather, self.seed):
            w.valueChanged.connect(self._settings_from_ui)
        for w in (self.flip_h, self.flip_v, self.avoid, self.masks, self.overlays, self.include_sources,
                  *self.blend_checks.values()):
            w.toggled.connect(self._settings_from_ui)
        for w in (self.placement, self.bg_mode, self.ext):
            w.currentIndexChanged.connect(self._settings_from_ui)
        self.prefix.currentTextChanged.connect(self._settings_from_ui)
        self.bulk_apply.clicked.connect(self._apply_bulk)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self.preview)
        self.preview_btn.clicked.connect(self.preview)

        state.defectsChanged.connect(self.refresh_table)
        state.imagesChanged.connect(self.refresh_table)
        state.projectReplaced.connect(self.load_settings)
        self.load_settings()
        self.refresh_table()

    @staticmethod
    def _tip(text: str, tip: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setToolTip(tip)
        lbl.setStyleSheet("background: transparent;")
        return lbl

    # settings <-> ui ---------------------------------------------------
    def load_settings(self) -> None:
        self._loading = True
        s = self.state.project.settings
        t = s.transform
        self.scale.set_value(t.scale)
        self.elong.set_value(t.elongation)
        self.rot.set_value(t.rotation)
        self.bend.set_value(t.bend)
        self.elastic.setValue(t.elastic_alpha)
        self.intensity.set_value(t.intensity)
        self.flip_h.setChecked(t.flip_h > 0)
        self.flip_v.setChecked(t.flip_v > 0)
        self.placement.setCurrentIndex(max(0, self.placement.findData(s.placement_mode)))
        self.jitter.setValue(s.jitter)
        self.gap.setValue(s.min_gap)
        self.per_image.set_value(s.defects_per_image)
        self.avoid.setChecked(s.avoid_marked_defects)
        self.bg_mode.setCurrentIndex(max(0, self.bg_mode.findData(s.bg_mode)))
        self.bg_size.setValue(s.bg_size)
        self.feather.setValue(s.feather)
        for v, cb in self.blend_checks.items():
            cb.setChecked(v in s.blend_modes)
        self.seed.setValue(s.seed)
        self.prefix.setCurrentText(s.name_prefix)
        self.ext.setCurrentIndex(max(0, self.ext.findData(s.output_ext)))
        self.masks.setChecked(s.save_masks)
        self.overlays.setChecked(s.save_overlays)
        self.include_sources.setChecked(s.labeltool_include_sources)
        self._loading = False

    def _settings_from_ui(self, *_) -> None:
        if self._loading:
            return
        s = self.state.project.settings
        t = s.transform
        t.scale, t.elongation, t.rotation = self.scale.value(), self.elong.value(), self.rot.value()
        t.bend, t.intensity = self.bend.value(), self.intensity.value()
        t.elastic_alpha = self.elastic.value()
        t.flip_h = 0.5 if self.flip_h.isChecked() else 0.0
        t.flip_v = 0.5 if self.flip_v.isChecked() else 0.0
        s.placement_mode = self.placement.currentData()
        s.jitter, s.min_gap = self.jitter.value(), self.gap.value()
        lo, hi = self.per_image.value()
        s.defects_per_image = (int(lo), int(hi))
        s.avoid_marked_defects = self.avoid.isChecked()
        s.bg_mode = self.bg_mode.currentData()
        s.bg_size, s.feather = self.bg_size.value(), self.feather.value()
        modes = [v for v, cb in self.blend_checks.items() if cb.isChecked()]
        if not modes:
            self._loading = True
            self.blend_checks["residual"].setChecked(True)
            self._loading = False
            modes = ["residual"]
        s.blend_modes = modes
        s.seed = self.seed.value()
        s.name_prefix = "".join(c for c in self.prefix.currentText().strip() if c.isalnum() or c in "-_") or "synth"
        s.output_ext = self.ext.currentData()
        s.save_masks, s.save_overlays = self.masks.isChecked(), self.overlays.isChecked()
        s.labeltool_include_sources = self.include_sources.isChecked()
        self.state.settingsChanged.emit()

    def _reset_defaults(self) -> None:
        from ...core.project import GenerationSettings
        self.state.project.settings = GenerationSettings()
        self.load_settings()
        self.state.settingsChanged.emit()
        self.refresh_table()

    # table -------------------------------------------------------------
    def refresh_table(self) -> None:
        proj = self.state.project
        selected = self._selected_id()
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for row, d in enumerate(proj.defects):
            self.table.insertRow(row)
            img = proj.load_pixels(d.image_id)
            pv = QTableWidgetItem()
            pv.setIcon(untinted_icon(crop_preview(img, d.mask, side=64)))
            pv.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            pv.setData(Qt.UserRole, d.id)
            self.table.setItem(row, self.COL_PREVIEW, pv)
            name = QTableWidgetItem(d.name)
            name.setToolTip("Double-click to rename")
            self.table.setItem(row, self.COL_NAME, name)
            src = QTableWidgetItem(proj.image(d.image_id).name)
            src.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, self.COL_SOURCE, src)

            spin = QSpinBox()
            spin.setRange(0, 100000)
            spin.setValue(d.count)
            spin.valueChanged.connect(lambda v, did=d.id: self._set_count(did, v))
            self.table.setCellWidget(row, self.COL_COUNT, self._centered(spin))

            combo = QComboBox()
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(14)
            combo.view().setMinimumWidth(260)
            combo.addItem("Default", "")
            for v, t in PLACEMENTS:
                combo.addItem(t, v)
            combo.setCurrentIndex(max(0, combo.findData(d.placement)))
            combo.currentIndexChanged.connect(lambda _, c=combo, did=d.id: self._set_placement(did, c.currentData()))
            self.table.setCellWidget(row, self.COL_PLACE, self._centered(combo))

            use = QCheckBox()
            use.setChecked(d.enabled)
            use.toggled.connect(lambda v, did=d.id: self._set_enabled(did, v))
            self.table.setCellWidget(row, self.COL_USE, self._centered(use, Qt.AlignCenter))
        self.table.blockSignals(False)
        self._update_total()
        if proj.defects:
            ids = [d.id for d in proj.defects]
            self.table.selectRow(ids.index(selected) if selected in ids else 0)
        else:
            self._clear_preview("Pick defects in step 2 first.")

    @staticmethod
    def _centered(w: QWidget, align=Qt.AlignVCenter | Qt.AlignLeft) -> QWidget:
        box = QWidget()
        box.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.addWidget(w)
        lay.setAlignment(align)
        return box

    def _selected_id(self) -> str | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        it = self.table.item(rows[0].row(), self.COL_PREVIEW)
        return it.data(Qt.UserRole) if it else None

    def _set_count(self, did: str, v: int) -> None:
        self.state.project.defect(did).count = v
        self.state.mark_dirty()
        self._update_total()

    def _set_enabled(self, did: str, v: bool) -> None:
        self.state.project.defect(did).enabled = v
        self.state.mark_dirty()
        self._update_total()

    def _set_placement(self, did: str, v: str) -> None:
        self.state.project.defect(did).placement = v
        self.state.mark_dirty()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != self.COL_NAME:
            return
        did = self.table.item(item.row(), self.COL_PREVIEW).data(Qt.UserRole)
        name = item.text().strip()
        if name:
            self.state.project.defect(did).name = name
            self.state.mark_dirty()

    def _apply_bulk(self) -> None:
        for d in self.state.project.defects:
            d.count = self.bulk.value()
        self.state.mark_dirty()
        self.refresh_table()

    def _update_total(self) -> None:
        n = self.state.project.total_images
        self.total.setText(f"{n:,} image{'s' if n != 1 else ''} will be generated")

    # preview -----------------------------------------------------------
    def _clear_preview(self, note: str) -> None:
        for t in self.preview_tiles:
            t.clear()
        self.preview_note.setText(note)

    def preview(self) -> None:
        did = self._selected_id()
        proj = self.state.project
        if did is None:
            self._clear_preview("Select a defect in the table to preview it.")
            return
        gen = Generator(proj)
        targets = gen.targets
        if not targets:
            self._clear_preview("Mark at least one image as a background in step 1, or create backgrounds.")
            return
        d = proj.defect(did)
        s = proj.settings
        rng = np.random.default_rng()
        try:
            base = gen.patch(d)
        except Exception as exc:
            self._clear_preview(f"Cannot preview: {exc}")
            return
        mode = d.placement or s.placement_mode
        shown = 0
        for tile in self.preview_tiles:
            tgt = targets[int(rng.integers(len(targets)))]
            img = gen.pixels(tgt)
            done = False
            for _ in range(6):
                t = tf.sample(s.transform, rng)
                try:
                    p = tf.apply(base, t, rng, s.transform.elastic_alpha, s.transform.elastic_sigma)
                except ValueError:
                    continue
                occ = gen._marked_mask(tgt, img.shape[:2]).copy() if s.avoid_marked_defects else None
                pos = find_position(p.mask, img.shape[:2], rng, mode, base.center, s.jitter, occ, None, s.min_gap)
                if pos is None:
                    continue
                out = blend(img, p, pos, gen._blend_mode(rng), max_value(img))
                m = np.zeros(img.shape[:2], np.uint8)
                x, y = pos
                m[y:y + p.mask.shape[0], x:x + p.mask.shape[1]] = p.mask
                tile.setPixmap(np_to_pixmap(crop_preview(out, m, pad=60, side=150)))
                done = True
                shown += 1
                break
            if not done:
                tile.clear()
        self.preview_note.setText(
            "Outline shows the pasted area. Click “Shuffle preview” for new random variations."
            if shown else "No valid placement found — reduce size or gap, or allow more movement.")
