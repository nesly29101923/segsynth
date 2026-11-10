"""Pick defects by brushing, rectangle cropping or lasso; mark slots (pockets) to place pieces into."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QSlider, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from ...core.project import LABEL_DEFECT, LABEL_GOOD
from ...core.refine import deviation_map, refine_mask
from ...core.slots import detect_with_template, template_from
from ..canvas import MaskCanvas, Tool
from ..qt_utils import crop_preview, label, thumbnail, untinted_icon
from ..state import AppState
from ..theme import DEFECT_COLORS

TOOLS = [
    (Tool.BRUSH, "Brush", "B", "Paint the defect. Right-drag erases. Ctrl+wheel changes size."),
    (Tool.ERASER, "Eraser", "E", "Erase parts of the current selection."),
    (Tool.RECT, "Box", "R", "Drag a box around the defect (crop). Alt or right-drag subtracts."),
    (Tool.LASSO, "Lasso", "L", "Draw a free outline around the defect. Alt or right-drag subtracts."),
    (Tool.PAN, "Pan", "H", "Drag to move the view. You can also hold Space or use the middle mouse button."),
]


LABELS = [
    (LABEL_DEFECT, "Defect (labeled)", "Pasted copies are labeled with the defect class (not good)."),
    (LABEL_GOOD, "Good part (not labeled)", "Pasted copies stay background (good), e.g. a normal part filling an empty pocket."),
]


def defect_color(index: int) -> QColor:
    return QColor(DEFECT_COLORS[index % len(DEFECT_COLORS)])


class DefectsPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.current_image: str | None = None
        self.editing: str | None = None
        self._dev_cache: dict[tuple, object] = {}
        self._template = None

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        # left: images ----------------------------------------------------
        left = QVBoxLayout()
        left.setSpacing(8)
        left.addWidget(label("IMAGES", "SectionLabel"))
        self.images = QListWidget()
        self.images.setIconSize(QSize(176, 120))
        self.images.setFixedWidth(212)
        self.images.setSpacing(2)
        left.addWidget(self.images, 1)
        root.addLayout(left)

        # center: toolbar + canvas ----------------------------------------
        center = QVBoxLayout()
        center.setSpacing(8)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_defects = QPushButton("Defects")
        self.mode_slots = QPushButton("Slots")
        self.mode_defects.setToolTip("Pick the defects or pieces to copy.")
        self.mode_slots.setToolTip("Mark slots (pockets, holes) that pieces are placed into exactly.")
        for b in (self.mode_defects, self.mode_slots):
            b.setObjectName("Tool")
            b.setCheckable(True)
            b.setMinimumWidth(b.fontMetrics().horizontalAdvance(b.text()) + 28)
            self.mode_group.addButton(b)
            bar.addWidget(b)
        self.mode_defects.setChecked(True)
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #323946;")
        bar.addSpacing(4)
        bar.addWidget(sep)
        bar.addSpacing(4)
        self.tool_widgets: list[QWidget] = []
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_buttons: dict[Tool, QPushButton] = {}
        for tool, text, key, tip in TOOLS:
            b = QPushButton(text)
            b.setObjectName("Tool")
            b.setCheckable(True)
            b.setMinimumWidth(b.fontMetrics().horizontalAdvance(text) + 28)
            b.setToolTip(f"<b>{text}</b> ({key})<br>{tip}")
            b.clicked.connect(lambda _=False, t=tool: self.set_tool(t))
            self.tool_group.addButton(b)
            self.tool_buttons[tool] = b
            bar.addWidget(b)
            self.tool_widgets.append(b)
        bar.addSpacing(12)
        size_lbl = label("Size", "Muted")
        bar.addWidget(size_lbl)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 200)
        self.size_slider.setValue(18)
        self.size_slider.setFixedWidth(100)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 200)
        self.size_spin.setValue(18)
        self.size_spin.setSuffix(" px")
        bar.addWidget(self.size_slider)
        bar.addWidget(self.size_spin)
        self.tool_widgets += [size_lbl, self.size_slider, self.size_spin]
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(55)
        self.opacity.setFixedWidth(90)
        bar.addStretch(1)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setObjectName("Tool")
        self.undo_btn.setToolTip("Undo (Ctrl+Z)")
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.setObjectName("Tool")
        self.redo_btn.setToolTip("Redo (Ctrl+Y)")
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setObjectName("Tool")
        self.fit_btn.setToolTip("Fit image to view (F)")
        self.one_btn = QPushButton("1:1")
        self.one_btn.setObjectName("Tool")
        self.one_btn.setToolTip("Actual pixels (1)")
        for b in (self.undo_btn, self.redo_btn, self.fit_btn, self.one_btn):
            b.setMinimumWidth(b.fontMetrics().horizontalAdvance(b.text()) + 24)
            bar.addWidget(b)
        center.addLayout(bar)

        self.canvas = MaskCanvas()
        center.addWidget(self.canvas, 1)

        info = QHBoxLayout()
        self.pos_label = label("", "Muted")
        self.zoom_label = label("", "Muted")
        self.hint = label("Select an image on the left to start picking defects.", "Muted")
        info.addWidget(self.hint, 1)
        info.addWidget(label("Overlay", "Muted"))
        info.addWidget(self.opacity)
        info.addSpacing(12)
        info.addWidget(self.pos_label)
        info.addSpacing(12)
        info.addWidget(self.zoom_label)
        center.addLayout(info)
        root.addLayout(center, 1)

        # right: selection + defect list -----------------------------------
        right = QVBoxLayout()
        right.setSpacing(8)
        right_w = QWidget()
        right_w.setLayout(right)
        right.setContentsMargins(0, 0, 0, 0)

        right.addWidget(label("CURRENT SELECTION", "SectionLabel"))
        self.sel_state = label("Nothing selected", "Muted")
        right.addWidget(self.sel_state)
        self.add_btn = QPushButton("Add as defect")
        self.add_btn.setObjectName("Primary")
        self.add_btn.setToolTip("Save the current selection as a defect (Enter)")
        self.cancel_edit_btn = QPushButton("Cancel editing")
        self.cancel_edit_btn.setObjectName("Ghost")
        self.cancel_edit_btn.setVisible(False)
        right.addWidget(self.add_btn)
        right.addWidget(self.cancel_edit_btn)

        snap_row = QHBoxLayout()
        self.snap_btn = QPushButton("Snap to defect")
        self.snap_btn.setToolTip(
            "Shrinks a rough selection to the pixels that differ from the surface background.\n"
            "Paint or box generously around the defect, then snap.")
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setToolTip("Clear the current selection (Esc)")
        snap_row.addWidget(self.snap_btn, 1)
        snap_row.addWidget(self.clear_btn)
        right.addLayout(snap_row)
        sens_row = QHBoxLayout()
        sens_row.addWidget(label("Snap sensitivity", "Muted"))
        self.sensitivity = QSlider(Qt.Horizontal)
        self.sensitivity.setRange(5, 95)
        self.sensitivity.setValue(55)
        self.sensitivity.setToolTip("Higher keeps fainter pixels; lower keeps only the strongest.")
        sens_row.addWidget(self.sensitivity, 1)
        right.addLayout(sens_row)

        right.addSpacing(10)
        self.list_title = label("DEFECTS ON THIS IMAGE", "SectionLabel")
        right.addWidget(self.list_title)
        self.defect_list = QListWidget()
        self.defect_list.setIconSize(QSize(56, 56))
        right.addWidget(self.defect_list, 1)
        label_row = QHBoxLayout()
        label_row.addWidget(label("Label", "Muted"))
        self.label_combo = QComboBox()
        for v, t, tip in LABELS:
            self.label_combo.addItem(t, v)
            self.label_combo.setItemData(self.label_combo.count() - 1, tip, Qt.ToolTipRole)
        self.label_combo.setToolTip("How pasted copies of the selected piece are labeled in the dataset.")
        label_row.addWidget(self.label_combo, 1)
        right.addLayout(label_row)
        row = QHBoxLayout()
        self.edit_btn = QPushButton("Edit")
        self.del_btn = QPushButton("Delete")
        self.del_btn.setObjectName("Danger")
        row.addWidget(self.edit_btn)
        row.addWidget(self.del_btn)
        right.addLayout(row)
        self.total_label = label("", "Muted")
        right.addWidget(self.total_label)

        self.right_stack = QStackedWidget()
        self.right_stack.setFixedWidth(290)
        self.right_stack.addWidget(right_w)
        self.right_stack.addWidget(self._build_slots_panel())
        root.addWidget(self.right_stack)

        # wiring ------------------------------------------------------------
        self.images.currentItemChanged.connect(self._on_image_selected)
        self.size_slider.valueChanged.connect(self.size_spin.setValue)
        self.size_spin.valueChanged.connect(self.size_slider.setValue)
        self.size_spin.valueChanged.connect(self.canvas.set_brush_size)
        self.opacity.valueChanged.connect(lambda v: self.canvas.set_layer_opacity(v / 100))
        self.undo_btn.clicked.connect(self.canvas.undo)
        self.redo_btn.clicked.connect(self.canvas.redo)
        self.fit_btn.clicked.connect(self.canvas.fit)
        self.one_btn.clicked.connect(lambda: self.canvas.zoom_to(1.0))
        self.canvas.zoomChanged.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f}%"))
        self.canvas.cursorMoved.connect(self._on_cursor)
        self.canvas.editChanged.connect(self._update_buttons)
        self.add_btn.clicked.connect(self.commit)
        self.cancel_edit_btn.clicked.connect(self.cancel_edit)
        self.snap_btn.clicked.connect(self.snap)
        self.clear_btn.clicked.connect(self.canvas.clear_edit)
        self.edit_btn.clicked.connect(self.edit_selected)
        self.del_btn.clicked.connect(self.delete_selected)
        self.defect_list.itemDoubleClicked.connect(lambda _: self.edit_selected())
        self.defect_list.currentItemChanged.connect(lambda *_: self._update_buttons())
        self.label_combo.currentIndexChanged.connect(self._set_label)
        self.mode_slots.toggled.connect(self._set_slot_mode)
        self.canvas.slotsEdited.connect(self._slots_edited)
        self.canvas.slotBoxChanged.connect(self._update_slot_panel)
        self.canvas.set_brush_size(18)
        self.canvas.set_layer_opacity(0.55)

        shortcuts = {
            "B": lambda: self.set_tool(Tool.BRUSH), "E": lambda: self.set_tool(Tool.ERASER),
            "R": lambda: self.set_tool(Tool.RECT), "L": lambda: self.set_tool(Tool.LASSO),
            "H": lambda: self.set_tool(Tool.PAN), "F": self.canvas.fit, "1": lambda: self.canvas.zoom_to(1.0),
            "Ctrl+Z": self.canvas.undo, "Ctrl+Y": self.canvas.redo, "Ctrl+Shift+Z": self.canvas.redo,
            "Return": self.commit, "Enter": self.commit, "Escape": self._escape,
            "S": self._s_key, "D": self._d_key, "[": lambda: self.size_spin.setValue(self.size_spin.value() - 2),
            "]": lambda: self.size_spin.setValue(self.size_spin.value() + 2),
        }
        self._shortcuts = []
        for key, fn in shortcuts.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)

        state.imagesChanged.connect(self.refresh_images)
        state.defectsChanged.connect(self.refresh_defects)
        state.projectReplaced.connect(self._reset)
        state.settingsChanged.connect(self._dev_cache.clear)
        self.set_tool(Tool.BRUSH)
        self.refresh_images()

    # slots -------------------------------------------------------------
    def _build_slots_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        v.addWidget(label("SLOTS", "SectionLabel"))
        v.addWidget(label(
            "Slots are the exact positions pieces are placed into, for example the pockets of a tray.\n\n"
            "1. Drag a box around one pocket (a filled one works best).\n"
            "2. Detect: every similar pocket in the grid is found.\n"
            "3. Click the pockets pieces may go into. They turn amber.", "Muted", wrap=True))
        det = QHBoxLayout()
        self.detect_btn = QPushButton("Detect")
        self.detect_btn.setObjectName("Primary")
        self.detect_btn.setToolTip("Find every pocket like the boxed one on this image (D).")
        self.detect_all_btn = QPushButton("Detect on all images")
        self.detect_all_btn.setToolTip("Use the boxed pocket to find slots on every image of the dataset.\n"
                                       "Images that already have slots are replaced.")
        det.addWidget(self.detect_btn)
        det.addWidget(self.detect_all_btn, 1)
        v.addLayout(det)
        thr = QHBoxLayout()
        thr.addWidget(label("Match", "Muted"))
        self.match_spin = QDoubleSpinBox()
        self.match_spin.setRange(0.1, 0.95)
        self.match_spin.setSingleStep(0.05)
        self.match_spin.setDecimals(2)
        self.match_spin.setToolTip("How similar a pocket must be to the boxed one to count as a match.\n"
                                   "Grid positions that match less (dashed red) look different, usually empty.")
        thr.addWidget(self.match_spin, 1)
        thr.addWidget(label("Size", "Muted"))
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(3, 2000)
        self.radius_spin.setSuffix(" px")
        self.radius_spin.setToolTip("Radius of every slot on this image.")
        thr.addWidget(self.radius_spin, 1)
        v.addLayout(thr)
        self.slot_status = label("", "Muted", wrap=True)
        v.addWidget(self.slot_status)

        v.addSpacing(6)
        v.addWidget(label("TARGETS", "SectionLabel"))
        r1 = QHBoxLayout()
        self.sel_empty_btn = QPushButton("Select empty-looking")
        self.sel_empty_btn.setToolTip("Mark every dashed red slot (low match) as a target.")
        self.sel_none_btn = QPushButton("Clear")
        self.sel_none_btn.setToolTip("Unmark all targets on this image.")
        r1.addWidget(self.sel_empty_btn, 1)
        r1.addWidget(self.sel_none_btn)
        v.addLayout(r1)
        r2 = QHBoxLayout()
        self.sel_all_btn = QPushButton("Select all")
        self.del_slots_btn = QPushButton("Remove slots")
        self.del_slots_btn.setObjectName("Danger")
        self.del_slots_btn.setToolTip("Delete every slot on this image.")
        r2.addWidget(self.sel_all_btn, 1)
        r2.addWidget(self.del_slots_btn, 1)
        v.addLayout(r2)

        v.addSpacing(6)
        v.addWidget(label("MOUSE", "SectionLabel"))
        v.addWidget(label(
            "Click: target on / off\nDrag a slot: move it\nShift+click: add a slot\nRight-click: remove a slot\n"
            "Space or middle button: pan", "Muted", wrap=True))
        legend = label(
            "<span style='color:#e3b341'>●</span> target &nbsp; "
            "<span style='color:#ff6b78'>◌</span> looks empty &nbsp; "
            "<span style='color:#c8ccd4'>○</span> slot", "Muted")
        legend.setTextFormat(Qt.RichText)
        v.addWidget(legend)
        v.addStretch(1)
        v.addWidget(label("Pieces go into target slots when their placement is “Into slots” on the "
                          "Recipe page.", "Muted", wrap=True))

        self.detect_btn.clicked.connect(lambda: self.detect_slots(False))
        self.detect_all_btn.clicked.connect(lambda: self.detect_slots(True))
        self.match_spin.valueChanged.connect(self._set_match)
        self.radius_spin.valueChanged.connect(self._set_radius)
        self.sel_empty_btn.clicked.connect(lambda: self._select_slots("empty"))
        self.sel_all_btn.clicked.connect(lambda: self._select_slots("all"))
        self.sel_none_btn.clicked.connect(lambda: self._select_slots("none"))
        self.del_slots_btn.clicked.connect(self._remove_slots)
        return w

    def _slots(self) -> list:
        return self.state.project.image(self.current_image).slots if self.current_image else []

    def _set_slot_mode(self, on: bool) -> None:
        if on and not self.canvas.edit_is_empty():
            r = QMessageBox.question(self, "Discard selection?",
                                     "The current selection has not been added as a defect. Discard it?")
            if r != QMessageBox.Yes:
                self.mode_defects.setChecked(True)
                return
            self.canvas.set_edit_mask(None, record=False)
        if on and self.editing:
            self.cancel_edit()
        for wdg in self.tool_widgets + [self.undo_btn, self.redo_btn]:
            wdg.setVisible(not on)
        self.right_stack.setCurrentIndex(1 if on else 0)
        self.canvas.set_slot_mode(on)
        self._show_slots()
        self.hint.setText(self._hint_text())

    def _hint_text(self) -> str:
        if not self.images.count():
            return "Select an image on the left to start picking defects."
        if self.canvas.slot_mode:
            return "Drag a box around one pocket, then Detect. Click pockets to make them targets."
        return "Brush, box or lasso a defect, then press Enter to add it."

    def _show_slots(self) -> None:
        self.canvas.set_slots(self._slots(), self.state.project.settings.slot_match)
        self._update_slot_panel()

    def _update_slot_panel(self) -> None:
        slots = self._slots()
        thr = self.state.project.settings.slot_match
        self.match_spin.blockSignals(True)
        self.match_spin.setValue(thr)
        self.match_spin.blockSignals(False)
        self.radius_spin.blockSignals(True)
        if slots:
            rs = sorted(s.r for s in slots)
            self.radius_spin.setValue(int(round(rs[len(rs) // 2])))
        self.radius_spin.setEnabled(bool(slots))
        self.radius_spin.blockSignals(False)
        has_box = self.canvas.slot_box is not None
        self.detect_btn.setEnabled(has_box and self.current_image is not None)
        self.detect_all_btn.setEnabled((has_box or self._template is not None) and self.images.count() > 0)
        for b in (self.sel_empty_btn, self.sel_all_btn, self.sel_none_btn, self.del_slots_btn):
            b.setEnabled(bool(slots))
        if not slots:
            self.slot_status.setText("No slots on this image yet." if not has_box
                                     else "Box drawn. Press Detect.")
        else:
            n_t = sum(s.target for s in slots)
            n_e = sum(s.score < thr for s in slots)
            self.slot_status.setText(f"{len(slots)} slots · {n_e} look empty · <b>{n_t} target{'s' if n_t != 1 else ''}</b>")
            self.slot_status.setTextFormat(Qt.RichText)
        total = sum(len(e.target_slots) for e in self.state.project.images)
        imgs = sum(1 for e in self.state.project.images if e.target_slots)
        if total:
            self.slot_status.setText(self.slot_status.text() +
                                     f"<br>Whole dataset: {total} target slot{'s' if total != 1 else ''} "
                                     f"on {imgs} image{'s' if imgs != 1 else ''}")
            self.slot_status.setTextFormat(Qt.RichText)

    def _slots_edited(self) -> None:
        self.state.mark_dirty()
        self._update_slot_panel()
        self._update_image_caption(self.current_image)

    def detect_slots(self, all_images: bool) -> None:
        proj = self.state.project
        box = self.canvas.slot_box
        if box is not None and self.current_image:
            r = box.toRect()
            self._template = template_from(proj.load_pixels(self.current_image),
                                           (r.left(), r.top(), r.right() + 1, r.bottom() + 1))
        if self._template is None:
            return
        ids = [e.id for e in proj.images] if all_images else [self.current_image]
        if all_images and any(proj.image(i).slots for i in ids):
            r = QMessageBox.question(self, "Replace slots?",
                                     "Some images already have slots. Detecting replaces them, including "
                                     "the targets you marked. Continue?")
            if r != QMessageBox.Yes:
                return
        self.setCursor(Qt.WaitCursor)
        found = {}
        try:
            for image_id in ids:
                try:
                    found[image_id] = detect_with_template(proj.load_pixels(image_id), self._template,
                                                           proj.settings.slot_match)
                except ValueError as exc:
                    QMessageBox.warning(self, "Cannot detect", str(exc))
                    return
        finally:
            self.unsetCursor()
        for image_id, slots in found.items():
            proj.image(image_id).slots = slots
        self.canvas.clear_slot_box()
        self.state.mark_dirty()
        self._show_slots()
        for image_id in found:
            self._update_image_caption(image_id)
        if not found.get(self.current_image):
            QMessageBox.information(self, "No slots found",
                                    "Nothing matched the boxed pocket. Box one whole pocket, or lower Match.")

    def _set_match(self, v: float) -> None:
        self.state.project.settings.slot_match = round(float(v), 2)
        self.state.mark_dirty()
        self._show_slots()

    def _set_radius(self, v: int) -> None:
        for s in self._slots():
            s.r = float(v)
        self.canvas.refresh_slots()
        self.state.mark_dirty()

    def _select_slots(self, which: str) -> None:
        thr = self.state.project.settings.slot_match
        for s in self._slots():
            if which == "all":
                s.target = True
            elif which == "none":
                s.target = False
            elif s.score < thr:
                s.target = True
        self.canvas.refresh_slots()
        self._slots_edited()

    def _remove_slots(self) -> None:
        if not self._slots():
            return
        r = QMessageBox.question(self, "Remove slots?", "Delete every slot on this image?")
        if r != QMessageBox.Yes:
            return
        self.state.project.image(self.current_image).slots = []
        self._show_slots()
        self._slots_edited()

    def _d_key(self) -> None:
        if self.canvas.slot_mode and self.canvas.slot_box is not None:
            self.detect_slots(False)

    def _s_key(self) -> None:
        if not self.canvas.slot_mode:
            self.snap()

    def _update_image_caption(self, image_id: str | None) -> None:
        if image_id is None:
            return
        proj = self.state.project
        for i in range(self.images.count()):
            it = self.images.item(i)
            if it.data(Qt.UserRole) == image_id:
                it.setText(self._caption(image_id))

    def _caption(self, image_id: str) -> str:
        proj = self.state.project
        n = len(proj.defects_of(image_id))
        text = f"{proj.image(image_id).name}\n{n} defect{'s' if n != 1 else ''}"
        t = len(proj.image(image_id).target_slots)
        if t:
            text += f"\n{t} target slot{'s' if t != 1 else ''}"
        return text

    def _set_label(self, _=None) -> None:
        item = self.defect_list.currentItem()
        if item is None:
            return
        d = self.state.project.defect(item.data(Qt.UserRole))
        value = self.label_combo.currentData()
        if d.label != value:
            d.label = value
            self.state.mark_dirty()
            area = int(d.mask.sum()) if d.mask is not None else 0
            item.setText(f"{d.name}\n{'good part' if value == LABEL_GOOD else f'{area:,} px'}")

    # tools -------------------------------------------------------------
    def set_tool(self, tool: Tool) -> None:
        self.tool_buttons[tool].setChecked(True)
        self.canvas.set_tool(tool)
        self.size_slider.setEnabled(tool in (Tool.BRUSH, Tool.ERASER))
        self.size_spin.setEnabled(tool in (Tool.BRUSH, Tool.ERASER))

    def _escape(self) -> None:
        if self.canvas.slot_mode:
            self.canvas.clear_slot_box()
        elif self.editing:
            self.cancel_edit()
        else:
            self.canvas.clear_edit()

    def _on_cursor(self, x: int, y: int) -> None:
        if self.canvas.has_image():
            w, h = self.canvas.image_size()
            self.pos_label.setText(f"x {x}  y {y}" if 0 <= x < w and 0 <= y < h else "")

    # images ------------------------------------------------------------
    def _reset(self) -> None:
        self.current_image = None
        self.editing = None
        self._dev_cache.clear()
        self._template = None
        self.canvas.set_image(None)
        self.canvas.set_slots([], 0.5)

    def refresh_images(self) -> None:
        keep = self.current_image
        self.images.blockSignals(True)
        self.images.clear()
        proj = self.state.project
        select_row = 0
        for i, e in enumerate(proj.images):
            try:
                icon = untinted_icon(thumbnail(proj.load_pixels(e.id), 176))
            except Exception:
                continue
            item = QListWidgetItem(icon, self._caption(e.id))
            item.setData(Qt.UserRole, e.id)
            item.setSizeHint(QSize(190, 178))
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            self.images.addItem(item)
            if e.id == keep:
                select_row = i
        self.images.setViewMode(QListWidget.IconMode)
        self.images.setFlow(QListWidget.TopToBottom)
        self.images.setWrapping(False)
        self.images.setMovement(QListWidget.Static)
        self.images.blockSignals(False)
        if self.images.count():
            ids = [self.images.item(i).data(Qt.UserRole) for i in range(self.images.count())]
            if keep not in ids:
                keep = None
            self.images.setCurrentRow(select_row)
            if keep is None or self.current_image != keep:
                self._on_image_selected(self.images.currentItem(), None)
        else:
            self._reset()
            self.refresh_defects()
        self.hint.setText(self._hint_text())

    def _on_image_selected(self, item, _prev) -> None:
        if item is None:
            return
        image_id = item.data(Qt.UserRole)
        if image_id == self.current_image and self.canvas.has_image():
            return
        if not self.canvas.edit_is_empty() and self.current_image:
            r = QMessageBox.question(self, "Discard selection?",
                                     "The current selection has not been added as a defect. Discard it?")
            if r != QMessageBox.Yes:
                self._select_row_for(self.current_image)
                return
        self.editing = None
        self.current_image = image_id
        self.canvas.set_image(self.state.project.load_pixels(image_id))
        self.refresh_defects()
        self._show_slots()

    def _select_row_for(self, image_id: str) -> None:
        self.images.blockSignals(True)
        for i in range(self.images.count()):
            if self.images.item(i).data(Qt.UserRole) == image_id:
                self.images.setCurrentRow(i)
        self.images.blockSignals(False)

    # defects -----------------------------------------------------------
    def _color_index(self, defect_id: str) -> int:
        ids = [d.id for d in self.state.project.defects]
        return ids.index(defect_id) if defect_id in ids else 0

    def refresh_defects(self) -> None:
        proj = self.state.project
        self.defect_list.clear()
        if self.current_image is None:
            self.canvas.set_committed([])
            self._update_buttons()
            self.total_label.setText("")
            return
        img = proj.load_pixels(self.current_image)
        layers = []
        for d in proj.defects_of(self.current_image):
            color = defect_color(self._color_index(d.id))
            if d.id != self.editing:
                layers.append((d.mask, color))
            area = int(d.mask.sum()) if d.mask is not None else 0
            tag = "good part" if d.label == LABEL_GOOD else f"{area:,} px"
            item = QListWidgetItem(untinted_icon(crop_preview(img, d.mask, side=56)), f"{d.name}\n{tag}")
            item.setData(Qt.UserRole, d.id)
            item.setForeground(color)
            self.defect_list.addItem(item)
        self.canvas.set_committed(layers)
        total = len(proj.defects)
        self.total_label.setText(f"{total} defect{'s' if total != 1 else ''} in the whole dataset")
        self._update_image_caption(self.current_image)
        self._update_buttons()

    def _update_buttons(self) -> None:
        has = self.canvas.has_image()
        mask = self.canvas.edit_mask() if has else None
        area = int(mask.sum()) if mask is not None else 0
        self.add_btn.setEnabled(area > 0)
        self.snap_btn.setEnabled(area > 0)
        self.clear_btn.setEnabled(area > 0)
        self.add_btn.setText("Save changes" if self.editing else "Add as defect")
        self.cancel_edit_btn.setVisible(self.editing is not None)
        if self.editing:
            self.sel_state.setText(f"Editing “{self.state.project.defect(self.editing).name}” · {area:,} px")
        else:
            self.sel_state.setText(f"{area:,} px selected" if area else "Nothing selected")
        sel = self.defect_list.currentItem() is not None
        self.edit_btn.setEnabled(sel and self.editing is None)
        self.del_btn.setEnabled(sel)
        self.label_combo.setEnabled(sel)
        if sel:
            d = self.state.project.defect(self.defect_list.currentItem().data(Qt.UserRole))
            self.label_combo.blockSignals(True)
            self.label_combo.setCurrentIndex(max(0, self.label_combo.findData(d.label)))
            self.label_combo.blockSignals(False)

    def commit(self) -> None:
        if not self.canvas.has_image() or self.current_image is None:
            return
        mask = self.canvas.edit_mask()
        if not mask.any():
            return
        proj = self.state.project
        if self.editing:
            proj.defect(self.editing).mask = mask
            self.editing = None
        else:
            proj.add_defect(self.current_image, mask)
        self.canvas.set_edit_mask(None, record=False)
        self.state.defectsChanged.emit()

    def cancel_edit(self) -> None:
        self.editing = None
        self.canvas.set_edit_mask(None, record=False)
        self.refresh_defects()

    def edit_selected(self) -> None:
        item = self.defect_list.currentItem()
        if item is None:
            return
        if not self.canvas.edit_is_empty():
            r = QMessageBox.question(self, "Replace selection?", "Discard the current selection and edit this defect?")
            if r != QMessageBox.Yes:
                return
        self.editing = item.data(Qt.UserRole)
        self.canvas.set_edit_mask(self.state.project.defect(self.editing).mask)
        self.refresh_defects()

    def delete_selected(self) -> None:
        item = self.defect_list.currentItem()
        if item is None:
            return
        did = item.data(Qt.UserRole)
        if did == self.editing:
            self.cancel_edit()
        self.state.project.remove_defect(did)
        self.state.defectsChanged.emit()

    def snap(self) -> None:
        if self.current_image is None or self.canvas.edit_is_empty():
            return
        s = self.state.project.settings
        key = (self.current_image, s.bg_mode, s.bg_size)
        if key not in self._dev_cache:
            self.setCursor(Qt.WaitCursor)
            try:
                self._dev_cache[key] = deviation_map(self.state.project.load_pixels(self.current_image), s.bg_mode, s.bg_size)
            finally:
                self.unsetCursor()
        refined = refine_mask(self.canvas.edit_mask(), self._dev_cache[key], self.sensitivity.value() / 100)
        if not refined.any():
            QMessageBox.information(self, "Nothing found", "No pixels stood out from the background. Raise the sensitivity.")
            return
        self.canvas.set_edit_mask(refined)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self.canvas.setFocus()
