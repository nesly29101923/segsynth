"""Step 2 — pick defects by brushing, rectangle cropping or lasso."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup, QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ...core.refine import deviation_map, refine_mask
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


def defect_color(index: int) -> QColor:
    return QColor(DEFECT_COLORS[index % len(DEFECT_COLORS)])


class DefectsPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.current_image: str | None = None
        self.editing: str | None = None
        self._dev_cache: dict[tuple, object] = {}

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
        bar.addSpacing(12)
        bar.addWidget(label("Size", "Muted"))
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
        right_w.setFixedWidth(290)
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
        row = QHBoxLayout()
        self.edit_btn = QPushButton("Edit")
        self.del_btn = QPushButton("Delete")
        self.del_btn.setObjectName("Danger")
        row.addWidget(self.edit_btn)
        row.addWidget(self.del_btn)
        right.addLayout(row)
        self.total_label = label("", "Muted")
        right.addWidget(self.total_label)
        root.addWidget(right_w)

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
        self.canvas.set_brush_size(18)
        self.canvas.set_layer_opacity(0.55)

        shortcuts = {
            "B": lambda: self.set_tool(Tool.BRUSH), "E": lambda: self.set_tool(Tool.ERASER),
            "R": lambda: self.set_tool(Tool.RECT), "L": lambda: self.set_tool(Tool.LASSO),
            "H": lambda: self.set_tool(Tool.PAN), "F": self.canvas.fit, "1": lambda: self.canvas.zoom_to(1.0),
            "Ctrl+Z": self.canvas.undo, "Ctrl+Y": self.canvas.redo, "Ctrl+Shift+Z": self.canvas.redo,
            "Return": self.commit, "Enter": self.commit, "Escape": self._escape,
            "S": self.snap, "[": lambda: self.size_spin.setValue(self.size_spin.value() - 2),
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

    # tools -------------------------------------------------------------
    def set_tool(self, tool: Tool) -> None:
        self.tool_buttons[tool].setChecked(True)
        self.canvas.set_tool(tool)
        self.size_slider.setEnabled(tool in (Tool.BRUSH, Tool.ERASER))
        self.size_spin.setEnabled(tool in (Tool.BRUSH, Tool.ERASER))

    def _escape(self) -> None:
        if self.editing:
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
        self.canvas.set_image(None)

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
            n = len(proj.defects_of(e.id))
            item = QListWidgetItem(icon, f"{e.name}\n{n} defect{'s' if n != 1 else ''}")
            item.setData(Qt.UserRole, e.id)
            item.setSizeHint(QSize(190, 160))
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
        self.hint.setText("Select an image on the left to start picking defects." if not self.images.count()
                          else "Brush, box or lasso a defect, then press Enter to add it.")

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
            item = QListWidgetItem(untinted_icon(crop_preview(img, d.mask, side=56)),
                                   f"{d.name}\n{area:,} px")
            item.setData(Qt.UserRole, d.id)
            item.setForeground(color)
            self.defect_list.addItem(item)
        self.canvas.set_committed(layers)
        total = len(proj.defects)
        self.total_label.setText(f"{total} defect{'s' if total != 1 else ''} in the whole dataset")
        n = len(proj.defects_of(self.current_image))
        for i in range(self.images.count()):
            it = self.images.item(i)
            if it.data(Qt.UserRole) == self.current_image:
                it.setText(f"{proj.image(self.current_image).name}\n{n} defect{'s' if n != 1 else ''}")
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
