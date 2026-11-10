"""Zoomable image canvas with brush, eraser, rectangle and lasso mask tools."""
from __future__ import annotations

from enum import Enum

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsView,
)

from ..core.slots import Slot
from .qt_utils import mask_to_qimage, np_to_pixmap, qimage_alpha_to_mask
from .theme import ACCENT, WARN

EDIT_COLOR = QColor("#3fe0cf")
SLOT_TARGET = QColor(WARN)
SLOT_EMPTY = QColor("#ff6b78")
SLOT_OTHER = QColor(255, 255, 255, 150)


class Tool(str, Enum):
    BRUSH = "brush"
    ERASER = "eraser"
    RECT = "rect"
    LASSO = "lasso"
    PAN = "pan"


class ImageLayer(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption, True)

    def set_image(self, image: QImage) -> None:
        self.prepareGeometryChange()
        self.image = image
        self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.image.width(), self.image.height())

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if not self.image.isNull():
            painter.drawImage(option.exposedRect, self.image, option.exposedRect)


class SlotLayer(QGraphicsItem):
    """Draws slots: targets solid amber, empty-looking ones dashed red, others thin white."""

    def __init__(self):
        super().__init__()
        self.slots: list[Slot] = []
        self.threshold = 0.5
        self.hover = -1
        self._rect = QRectF()

    def set_slots(self, slots: list[Slot], threshold: float) -> None:
        self.prepareGeometryChange()
        self.slots = slots
        self.threshold = threshold
        if slots:
            pad = max(s.r for s in slots) + 8
            xs = [s.x for s in slots]
            ys = [s.y for s in slots]
            self._rect = QRectF(min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad)
        else:
            self._rect = QRectF()
        self.update()

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        for i, s in enumerate(self.slots):
            if s.target:
                pen = QPen(SLOT_TARGET, 3.0)
                fill = QColor(SLOT_TARGET)
                fill.setAlpha(60)
            elif s.score < self.threshold:
                pen = QPen(SLOT_EMPTY, 2.0, Qt.DashLine)
                fill = QColor(0, 0, 0, 0)
            else:
                pen = QPen(SLOT_OTHER, 1.2)
                fill = QColor(0, 0, 0, 0)
            if i == self.hover:
                pen.setWidthF(pen.widthF() + 1.5)
                pen.setColor(pen.color().lighter(130))
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(fill)
            painter.drawEllipse(QPointF(s.x, s.y), s.r, s.r)


class MaskCanvas(QGraphicsView):
    editChanged = Signal()
    slotsEdited = Signal()
    slotBoxChanged = Signal()
    zoomChanged = Signal(float)
    cursorMoved = Signal(int, int)

    MAX_UNDO = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.base = QGraphicsPixmapItem()
        self.base.setTransformationMode(Qt.SmoothTransformation)
        self.committed = ImageLayer()
        self.committed.setOpacity(0.45)
        self.edit = ImageLayer()
        self.edit.setOpacity(0.55)
        for z, it in enumerate((self.base, self.committed, self.edit)):
            it.setZValue(z)
            self.scene().addItem(it)
        self.base.setFlag(QGraphicsItem.ItemIsSelectable, False)

        self.cursor_item = QGraphicsEllipseItem()
        pen = QPen(QColor("white"), 1.5)
        pen.setCosmetic(True)
        self.cursor_item.setPen(pen)
        self.cursor_item.setZValue(10)
        self.cursor_item.setVisible(False)
        self.scene().addItem(self.cursor_item)

        self.rubber = QGraphicsRectItem()
        self.lasso_item = QGraphicsPathItem()
        for it in (self.rubber, self.lasso_item):
            p = QPen(QColor(ACCENT), 1.5, Qt.DashLine)
            p.setCosmetic(True)
            it.setPen(p)
            it.setBrush(QBrush(QColor(63, 182, 168, 50)))
            it.setZValue(9)
            it.setVisible(False)
            self.scene().addItem(it)

        self.slot_layer = SlotLayer()
        self.slot_layer.setZValue(5)
        self.slot_layer.setVisible(False)
        self.scene().addItem(self.slot_layer)
        self.slot_mode = False
        self.slot_box: QRectF | None = None
        self._slot_press = -1
        self._slot_drag = False
        self._slot_offset = QPointF()

        self.tool = Tool.BRUSH
        self.brush_size = 18
        self._editable = False
        self._drawing = False
        self._erasing = False
        self._last = QPointF()
        self._start = QPointF()
        self._lasso = QPainterPath()
        self._panning = False
        self._space = False
        self._pan_origin = None
        self._undo: list[QImage] = []
        self._redo: list[QImage] = []
        self._zoom = 1.0
        self._auto_fit = True

    # image & layers ------------------------------------------------------
    def set_image(self, img: np.ndarray | None) -> None:
        self._undo.clear()
        self._redo.clear()
        self.clear_slot_box()
        if img is None:
            self.base.setPixmap(QPixmap())
            self.committed.set_image(QImage())
            self.edit.set_image(QImage())
            self._editable = False
            return
        pm = np_to_pixmap(img)
        self.base.setPixmap(pm)
        empty = QImage(pm.width(), pm.height(), QImage.Format_ARGB32_Premultiplied)
        empty.fill(Qt.transparent)
        self.edit.set_image(empty)
        self.committed.set_image(empty.copy())
        self.scene().setSceneRect(QRectF(0, 0, pm.width(), pm.height()).adjusted(-200, -200, 200, 200))
        self._editable = True
        self.fit()

    def has_image(self) -> bool:
        return self._editable

    def image_size(self) -> tuple[int, int]:
        return self.edit.image.width(), self.edit.image.height()

    def set_committed(self, layers: list[tuple[np.ndarray, QColor]]) -> None:
        if not self._editable:
            return
        w, h = self.image_size()
        img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)
        p = QPainter(img)
        for mask, color in layers:
            if mask is not None and mask.shape == (h, w):
                p.drawImage(0, 0, mask_to_qimage(mask, color))
        p.end()
        self.committed.set_image(img)

    def edit_mask(self) -> np.ndarray:
        return qimage_alpha_to_mask(self.edit.image)

    def set_edit_mask(self, mask: np.ndarray | None, record: bool = True) -> None:
        if record:
            self._push_undo()
        w, h = self.image_size()
        if mask is None or not mask.any():
            img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
            img.fill(Qt.transparent)
        else:
            img = mask_to_qimage(mask, EDIT_COLOR).convertToFormat(QImage.Format_ARGB32_Premultiplied)
        self.edit.set_image(img)
        self.editChanged.emit()

    def clear_edit(self) -> None:
        self.set_edit_mask(None)

    def edit_is_empty(self) -> bool:
        return not self.edit_mask().any() if self._editable else True

    def set_layer_opacity(self, value: float) -> None:
        self.committed.setOpacity(value * 0.8)
        self.edit.setOpacity(value)

    # undo ----------------------------------------------------------------
    def _push_undo(self) -> None:
        self._undo.append(self.edit.image.copy())
        if len(self._undo) > self.MAX_UNDO:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self) -> None:
        if self._undo:
            self._redo.append(self.edit.image.copy())
            self.edit.set_image(self._undo.pop())
            self.editChanged.emit()

    def redo(self) -> None:
        if self._redo:
            self._undo.append(self.edit.image.copy())
            self.edit.set_image(self._redo.pop())
            self.editChanged.emit()

    # slots ---------------------------------------------------------------
    def set_slot_mode(self, on: bool) -> None:
        self.slot_mode = on
        self.slot_layer.setVisible(on)
        self.edit.setVisible(not on)
        if not on:
            self.clear_slot_box()
        self._update_cursor()

    def set_slots(self, slots: list[Slot], threshold: float) -> None:
        self.slot_layer.set_slots(slots, threshold)

    def refresh_slots(self) -> None:
        self.slot_layer.set_slots(self.slot_layer.slots, self.slot_layer.threshold)

    def clear_slot_box(self) -> None:
        self.slot_box = None
        self.rubber.setVisible(False)
        self.slotBoxChanged.emit()

    def slot_at(self, p: QPointF) -> int:
        best, best_d = -1, None
        for i, s in enumerate(self.slot_layer.slots):
            d = (s.x - p.x()) ** 2 + (s.y - p.y()) ** 2
            if d <= s.r ** 2 and (best_d is None or d < best_d):
                best, best_d = i, d
        return best

    def _default_radius(self) -> float:
        rs = sorted(s.r for s in self.slot_layer.slots)
        return rs[len(rs) // 2] if rs else max(8.0, min(self.image_size()) / 40)

    def _slot_press_event(self, e, sp: QPointF) -> None:
        slots = self.slot_layer.slots
        hit = self.slot_at(sp)
        if e.button() == Qt.RightButton:
            if hit >= 0:
                del slots[hit]
                self.slot_layer.hover = -1
                self.refresh_slots()
                self.slotsEdited.emit()
            return
        if e.button() != Qt.LeftButton:
            return
        if e.modifiers() & Qt.ShiftModifier:
            if hit < 0:
                slots.append(Slot(sp.x(), sp.y(), self._default_radius(), True, 0.0))
                self.refresh_slots()
                self.slotsEdited.emit()
            return
        if hit >= 0:
            self._slot_press = hit
            self._slot_drag = False
            self._start = sp
            self._slot_offset = QPointF(slots[hit].x - sp.x(), slots[hit].y - sp.y())
            return
        self._drawing = True
        self._start = sp
        self.rubber.setRect(QRectF(sp, sp))
        self.rubber.setVisible(True)

    def _slot_move_event(self, sp: QPointF) -> None:
        if self._slot_press >= 0:
            d = sp - self._start
            if not self._slot_drag and (d.x() ** 2 + d.y() ** 2) * self._zoom ** 2 > 16:
                self._slot_drag = True
            if self._slot_drag:
                s = self.slot_layer.slots[self._slot_press]
                c = self._clamp(sp + self._slot_offset)
                s.x, s.y = c.x(), c.y()
                self.refresh_slots()
            return
        if self._drawing:
            self.rubber.setRect(QRectF(self._start, self._clamp(sp)).normalized())
            return
        hover = self.slot_at(sp)
        if hover != self.slot_layer.hover:
            self.slot_layer.hover = hover
            self.slot_layer.update()

    def _slot_release_event(self) -> None:
        if self._slot_press >= 0:
            s = self.slot_layer.slots[self._slot_press]
            if not self._slot_drag:
                s.target = not s.target
            self._slot_press = -1
            self.refresh_slots()
            self.slotsEdited.emit()
            return
        if self._drawing:
            self._drawing = False
            r = self.rubber.rect()
            if r.width() >= 6 and r.height() >= 6:
                self.slot_box = r
            else:
                self.slot_box = None
                self.rubber.setVisible(False)
            self.slotBoxChanged.emit()

    # tools ---------------------------------------------------------------
    def set_tool(self, tool: Tool) -> None:
        self.tool = Tool(tool)
        self._update_cursor()

    def set_brush_size(self, size: int) -> None:
        self.brush_size = max(1, int(size))
        self._update_cursor()

    def _update_cursor(self, scene_pos: QPointF | None = None) -> None:
        brushy = self.tool in (Tool.BRUSH, Tool.ERASER) and not self._space and not self.slot_mode
        if self._space or (self.tool == Tool.PAN and not self.slot_mode):
            self.viewport().setCursor(Qt.ClosedHandCursor if self._panning else Qt.OpenHandCursor)
        elif brushy:
            self.viewport().setCursor(Qt.BlankCursor if self.underMouse() else Qt.ArrowCursor)
        else:
            self.viewport().setCursor(Qt.CrossCursor)
        self.cursor_item.setVisible(brushy and self._editable and self.underMouse())
        if scene_pos is None:
            scene_pos = self.mapToScene(self.mapFromGlobal(QCursor.pos()))
        r = self.brush_size / 2.0
        self.cursor_item.setRect(scene_pos.x() - r, scene_pos.y() - r, 2 * r, 2 * r)
        color = QColor("#ff8a8a") if self.tool == Tool.ERASER else QColor("white")
        pen = self.cursor_item.pen()
        pen.setColor(color)
        self.cursor_item.setPen(pen)

    # zoom ----------------------------------------------------------------
    def fit(self) -> None:
        self._auto_fit = True
        if self.base.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.base, Qt.KeepAspectRatio)
        self._zoom = self.transform().m11()
        self.zoomChanged.emit(self._zoom)

    def zoom_to(self, factor: float) -> None:
        self._auto_fit = False
        factor = max(0.02, min(32.0, factor))
        s = factor / self._zoom
        self.scale(s, s)
        self._zoom = factor
        self.zoomChanged.emit(self._zoom)

    def wheelEvent(self, e) -> None:
        if not self._editable:
            return
        if e.modifiers() & Qt.ControlModifier and self.tool in (Tool.BRUSH, Tool.ERASER) and not self.slot_mode:
            step = 1 if e.angleDelta().y() > 0 else -1
            self.set_brush_size(self.brush_size + step * max(1, self.brush_size // 8))
            return
        f = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        self.zoom_to(self._zoom * f)

    # painting ------------------------------------------------------------
    def _clamp(self, p: QPointF) -> QPointF:
        w, h = self.image_size()
        return QPointF(min(max(p.x(), 0), w - 1), min(max(p.y(), 0), h - 1))

    def _painter(self, erase: bool) -> QPainter:
        p = QPainter(self.edit.image)
        p.setRenderHint(QPainter.Antialiasing, False)
        if erase:
            p.setCompositionMode(QPainter.CompositionMode_Clear)
        return p

    def _stroke(self, a: QPointF, b: QPointF) -> None:
        p = self._painter(self._erasing)
        pen = QPen(EDIT_COLOR, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        if a == b:
            p.setBrush(EDIT_COLOR)
            p.setPen(Qt.NoPen)
            r = self.brush_size / 2.0
            p.drawEllipse(a, r, r)
        else:
            p.drawLine(a, b)
        p.end()
        r = self.brush_size
        self.edit.update(QRectF(a, b).normalized().adjusted(-r, -r, r, r))

    def _fill_path(self, path: QPainterPath, erase: bool) -> None:
        self._push_undo()
        p = self._painter(erase)
        p.setPen(Qt.NoPen)
        p.setBrush(EDIT_COLOR)
        p.drawPath(path)
        p.end()
        self.edit.update()
        self.editChanged.emit()

    def mousePressEvent(self, e) -> None:
        if not self._editable:
            return
        pan_tool = self.tool == Tool.PAN and not self.slot_mode
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and (self._space or pan_tool)):
            self._panning = True
            self._pan_origin = e.position()
            self._update_cursor()
            return
        sp = self._clamp(self.mapToScene(e.position().toPoint()))
        if self.slot_mode:
            self._slot_press_event(e, sp)
            return
        if e.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        subtract = e.button() == Qt.RightButton or bool(e.modifiers() & Qt.AltModifier)
        if self.tool in (Tool.BRUSH, Tool.ERASER):
            self._push_undo()
            self._drawing = True
            self._erasing = self.tool == Tool.ERASER or subtract
            self._last = sp
            self._stroke(sp, sp)
        elif self.tool == Tool.RECT:
            self._drawing = True
            self._erasing = subtract
            self._start = sp
            self.rubber.setRect(QRectF(sp, sp))
            self.rubber.setVisible(True)
        elif self.tool == Tool.LASSO:
            self._drawing = True
            self._erasing = subtract
            self._lasso = QPainterPath(sp)
            self.lasso_item.setPath(self._lasso)
            self.lasso_item.setVisible(True)

    def mouseMoveEvent(self, e) -> None:
        sp = self.mapToScene(e.position().toPoint())
        if self._editable:
            self.cursorMoved.emit(int(sp.x()), int(sp.y()))
        if self._panning and self._pan_origin is not None:
            d = e.position() - self._pan_origin
            self._pan_origin = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        self._update_cursor(sp)
        if self.slot_mode:
            self._slot_move_event(sp)
            return
        if not self._drawing:
            return
        sp = self._clamp(sp)
        if self.tool in (Tool.BRUSH, Tool.ERASER):
            self._stroke(self._last, sp)
            self._last = sp
        elif self.tool == Tool.RECT:
            self.rubber.setRect(QRectF(self._start, sp).normalized())
        elif self.tool == Tool.LASSO:
            self._lasso.lineTo(sp)
            self.lasso_item.setPath(self._lasso)

    def mouseReleaseEvent(self, e) -> None:
        if self._panning:
            self._panning = False
            self._update_cursor()
            return
        if self.slot_mode:
            self._slot_release_event()
            return
        if not self._drawing:
            return
        self._drawing = False
        if self.tool in (Tool.BRUSH, Tool.ERASER):
            self.editChanged.emit()
        elif self.tool == Tool.RECT:
            r = self.rubber.rect()
            self.rubber.setVisible(False)
            if r.width() >= 2 and r.height() >= 2:
                path = QPainterPath()
                path.addRect(r)
                self._fill_path(path, self._erasing)
        elif self.tool == Tool.LASSO:
            self.lasso_item.setVisible(False)
            self._lasso.closeSubpath()
            if self._lasso.boundingRect().width() > 2 and self._lasso.boundingRect().height() > 2:
                poly = QPolygonF(self._lasso.toFillPolygon())
                path = QPainterPath()
                path.addPolygon(poly)
                self._fill_path(path, self._erasing)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self._auto_fit:
            self.fit()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if self._auto_fit:
            self.fit()

    def enterEvent(self, e) -> None:
        super().enterEvent(e)
        self._update_cursor()

    def leaveEvent(self, e) -> None:
        super().leaveEvent(e)
        self.cursor_item.setVisible(False)

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self._update_cursor()
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self._panning = False
            self._update_cursor()
            return
        super().keyReleaseEvent(e)
