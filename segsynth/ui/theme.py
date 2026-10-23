"""Application palette and stylesheet."""
from __future__ import annotations

import os
import tempfile

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

BG = "#14171c"
SURFACE = "#1b1f26"
SURFACE_2 = "#232831"
SURFACE_3 = "#2c323d"
BORDER = "#323946"
TEXT = "#e7eaf0"
TEXT_DIM = "#9aa3b2"
TEXT_FAINT = "#6b7383"
ACCENT = "#3fb6a8"
ACCENT_HOVER = "#52c9bb"
ACCENT_DIM = "#1f4f4a"
DANGER = "#e5606b"
WARN = "#e3b341"

DEFECT_COLORS = [
    "#ff5c6c", "#ffb547", "#5ecbff", "#9b7bff", "#5fe08f",
    "#ff7bd5", "#f7e463", "#4fd1c5", "#ff8f5c", "#8fa8ff",
]

STYLESHEET = f"""
* {{ outline: none; }}
QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QToolTip {{ background: {SURFACE_3}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px 8px; border-radius: 6px; }}

#Header {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
#AppTitle {{ font-size: 15px; font-weight: 700; letter-spacing: 0.3px; background: transparent; }}
#AppSubtitle {{ color: {TEXT_FAINT}; background: transparent; font-size: 12px; }}
#Sidebar {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
#PageTitle {{ font-size: 20px; font-weight: 700; background: transparent; }}
#PageHint {{ color: {TEXT_DIM}; background: transparent; }}
#SectionLabel {{ color: {TEXT_FAINT}; font-size: 11px; font-weight: 700; letter-spacing: 1px; background: transparent; }}
#Muted {{ color: {TEXT_DIM}; background: transparent; }}
#Big {{ font-size: 28px; font-weight: 700; background: transparent; }}
#Card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; }}
#Card QLabel {{ background: transparent; }}
#Card QCheckBox {{ background: transparent; }}
#EmptyState {{ color: {TEXT_FAINT}; font-size: 14px; background: transparent; }}
#StatusBadge {{ background: {ACCENT_DIM}; color: {ACCENT_HOVER}; border-radius: 10px; padding: 2px 10px; font-weight: 600; }}

QPushButton {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 7px;
    padding: 7px 14px; color: {TEXT};
}}
QPushButton:hover {{ background: {SURFACE_3}; border-color: #3d4554; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background: {SURFACE}; border-color: {SURFACE_2}; }}
QPushButton#Primary {{ background: {ACCENT}; border-color: {ACCENT}; color: #071513; font-weight: 700; }}
QPushButton#Primary:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton#Primary:disabled {{ background: {ACCENT_DIM}; border-color: {ACCENT_DIM}; color: #5b7d79; }}
QPushButton#Danger {{ color: {DANGER}; }}
QPushButton#Danger:hover {{ background: #3a2226; border-color: #5a2d33; }}
QPushButton#Ghost {{ background: transparent; border-color: transparent; color: {TEXT_DIM}; }}
QPushButton#Ghost:hover {{ background: {SURFACE_2}; color: {TEXT}; }}

QToolButton {{
    background: transparent; border: 1px solid transparent; border-radius: 7px;
    padding: 6px 12px; min-width: 28px; color: {TEXT_DIM};
}}
QToolButton:hover {{ background: {SURFACE_2}; color: {TEXT}; }}
QToolButton:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; color: {ACCENT_HOVER}; }}

QPushButton#Tool {{ background: transparent; border: 1px solid transparent; color: {TEXT_DIM}; padding: 6px 11px; }}
QPushButton#Tool:hover {{ background: {SURFACE_2}; color: {TEXT}; }}
QPushButton#Tool:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; color: {ACCENT_HOVER}; font-weight: 600; }}
QPushButton#Tool:disabled {{ background: transparent; color: {TEXT_FAINT}; }}

#NavButton {{
    text-align: left; padding: 10px 14px; border-radius: 8px; border: 1px solid transparent;
    background: transparent; color: {TEXT_DIM}; font-weight: 600;
}}
#NavButton:hover {{ background: {SURFACE_2}; color: {TEXT}; }}
#NavButton:checked {{ background: {SURFACE_3}; color: {TEXT}; border-color: {BORDER}; }}
#NavButton:disabled {{ color: #4a5160; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 8px; min-height: 18px; selection-background-color: {ACCENT}; selection-color: #071513;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QLineEdit#ProjectName {{ background: transparent; border-color: transparent; font-weight: 600; font-size: 14px; }}
QLineEdit#ProjectName:hover {{ border-color: {BORDER}; }}
QLineEdit#ProjectName:focus {{ background: {SURFACE_2}; border-color: {ACCENT}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 18px; border: none; border-top-right-radius: 6px; background: transparent; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 18px; border: none; border-bottom-right-radius: 6px; background: transparent; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {SURFACE_3}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url(:/segsynth/up.svg); width: 8px; height: 8px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url(:/segsynth/down.svg); width: 8px; height: 8px; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url(:/segsynth/down.svg); width: 9px; height: 9px; margin-right: 8px; }}
QComboBox QAbstractItemView {{ background: {SURFACE_2}; border: 1px solid {BORDER}; selection-background-color: {SURFACE_3}; selection-color: {TEXT}; padding: 4px; }}

QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid #4a5363; background: {SURFACE_2}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url(:/segsynth/check.svg); }}

QSlider::groove:horizontal {{ height: 4px; background: {SURFACE_3}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider {{ background: transparent; }}

QProgressBar {{ background: {SURFACE_2}; border: none; border-radius: 5px; height: 10px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}

QListWidget, QTableWidget, QTreeWidget {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 4px;
    alternate-background-color: {SURFACE_2};
}}
QListWidget::item {{ border-radius: 8px; padding: 4px; color: {TEXT}; }}
QListWidget::item:hover {{ background: {SURFACE_2}; }}
QListWidget::item:selected {{ background: {SURFACE_3}; color: {TEXT}; }}
QTableWidget {{ gridline-color: transparent; }}
QTableWidget::item {{ padding: 4px; border-bottom: 1px solid {SURFACE_2}; }}
QTableWidget::item:selected {{ background: {SURFACE_3}; color: {TEXT}; }}
QHeaderView::section {{
    background: {SURFACE}; color: {TEXT_FAINT}; border: none; border-bottom: 1px solid {BORDER};
    padding: 8px 6px; font-size: 11px; font-weight: 700; text-transform: uppercase;
}}
QTableCornerButton::section {{ background: {SURFACE}; border: none; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {SURFACE_3}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #3d4554; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {SURFACE_3}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QSplitter::handle {{ background: {BORDER}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 10px; margin-top: 14px; padding: 12px 12px 10px 12px; background: {SURFACE}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {TEXT_DIM}; font-weight: 700; }}
QGroupBox QLabel, QGroupBox QCheckBox {{ background: transparent; }}
QStatusBar {{ background: {SURFACE}; border-top: 1px solid {BORDER}; color: {TEXT_DIM}; }}
QStatusBar QLabel {{ background: transparent; color: {TEXT_DIM}; }}
QMenu {{ background: {SURFACE_2}; border: 1px solid {BORDER}; padding: 4px; border-radius: 8px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 5px; }}
QMenu::item:selected {{ background: {SURFACE_3}; }}
QMessageBox, QDialog {{ background: {SURFACE}; }}
QMessageBox QLabel {{ background: transparent; }}
QGraphicsView {{ background: #0e1014; border: 1px solid {BORDER}; border-radius: 10px; }}
"""


def _register_icons() -> None:
    global _ICON_DIR
    if _ICON_DIR:
        return
    _ICON_DIR = tempfile.mkdtemp(prefix="segsynth_icons_")
    svgs = {
        "up": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M1.5 6.8 5 3.2l3.5 3.6" fill="none" stroke="%s" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>' % TEXT_DIM,
        "check": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M3.5 8.4 6.6 11.4 12.5 4.8" fill="none" stroke="#071513" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        "down": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M1.5 3.2 5 6.8l3.5-3.6" fill="none" stroke="%s" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>' % TEXT_DIM,
    }
    for name, svg in svgs.items():
        with open(os.path.join(_ICON_DIR, f"{name}.svg"), "w", encoding="utf-8") as f:
            f.write(svg)


_ICON_DIR = ""


def apply_theme(app: QApplication) -> None:
    _register_icons()
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(SURFACE_2))
    pal.setColor(QPalette.AlternateBase, QColor(SURFACE))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(SURFACE_2))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#071513"))
    pal.setColor(QPalette.ToolTipBase, QColor(SURFACE_3))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.PlaceholderText, QColor(TEXT_FAINT))
    app.setPalette(pal)
    font = QFont()
    font.setFamilies(["Segoe UI", "Inter", "SF Pro Text", "Helvetica Neue", "Arial"])
    font.setPointSizeF(10)
    app.setFont(font)
    icon_dir = _ICON_DIR.replace("\\", "/")
    app.setStyleSheet(STYLESHEET.replace(":/segsynth/", icon_dir + "/"))
