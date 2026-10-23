"""Qt application bootstrap."""
from __future__ import annotations

import sys


def run() -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from .ui.main_window import APP_NAME, MainWindow
    from .ui.theme import apply_theme

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    apply_theme(app)
    win = MainWindow()
    win.show()
    return app.exec()
