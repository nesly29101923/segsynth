"""NumPy ↔ Qt conversions and small widget helpers."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


def to_display_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        return (img >> 8).astype(np.uint8)
    if img.dtype != np.uint8:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def np_to_qimage(img: np.ndarray) -> QImage:
    a = np.ascontiguousarray(to_display_u8(img))
    h, w = a.shape[:2]
    if a.ndim == 2:
        q = QImage(a.data, w, h, a.strides[0], QImage.Format_Grayscale8)
    elif a.shape[2] == 3:
        q = QImage(a.data, w, h, a.strides[0], QImage.Format_BGR888)
    else:
        q = QImage(a.data, w, h, a.strides[0], QImage.Format_ARGB32)
    return q.copy()


def np_to_pixmap(img: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(np_to_qimage(img))


def untinted_icon(img: np.ndarray) -> QIcon:
    pm = np_to_pixmap(img)
    icon = QIcon()
    for mode in (QIcon.Normal, QIcon.Selected, QIcon.Active):
        icon.addPixmap(pm, mode)
    return icon


def thumbnail(img: np.ndarray, max_side: int = 240) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def qimage_alpha_to_mask(q: QImage) -> np.ndarray:
    q = q.convertToFormat(QImage.Format_ARGB32)
    w, h = q.width(), q.height()
    ptr = q.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=q.bytesPerLine() * h).reshape(h, q.bytesPerLine())
    arr = arr[:, : w * 4].reshape(h, w, 4)
    return (arr[:, :, 3] > 0).astype(np.uint8)


def mask_to_qimage(mask: np.ndarray, color, alpha: int = 255) -> QImage:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    m = mask > 0
    rgba[m, 0] = color.blue()
    rgba[m, 1] = color.green()
    rgba[m, 2] = color.red()
    rgba[m, 3] = alpha
    return QImage(rgba.data, w, h, w * 4, QImage.Format_ARGB32).copy()


def crop_preview(img: np.ndarray, mask: np.ndarray, pad: int = 24, side: int = 96) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return thumbnail(img, side)
    h, w = mask.shape
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    crop = to_display_u8(img[y0:y1, x0:x1]).copy()
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, contours, -1, (108, 92, 255), 1)
    ch, cw = crop.shape[:2]
    s = side / max(ch, cw)
    crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((side, side, 3), 20, np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return canvas


def label(text: str, name: str | None = None, wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    if name:
        lbl.setObjectName(name)
    lbl.setWordWrap(wrap)
    return lbl


def page_header(title: str, hint: str) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 8)
    lay.setSpacing(4)
    lay.addWidget(label(title, "PageTitle"))
    lay.addWidget(label(hint, "PageHint", wrap=True))
    return w


def card(inner_layout=None) -> QFrame:
    f = QFrame()
    f.setObjectName("Card")
    if inner_layout is not None:
        f.setLayout(inner_layout)
    return f


def hbox(*widgets, spacing: int = 8, margins=(0, 0, 0, 0), stretch_at: int | None = None) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setSpacing(spacing)
    lay.setContentsMargins(*margins)
    for i, w in enumerate(widgets):
        if stretch_at is not None and i == stretch_at:
            lay.addStretch(1)
        if w is None:
            lay.addStretch(1)
        elif isinstance(w, QWidget):
            lay.addWidget(w)
        else:
            lay.addLayout(w)
    return lay


def set_alignment_center(lbl: QLabel) -> QLabel:
    lbl.setAlignment(Qt.AlignCenter)
    return lbl
