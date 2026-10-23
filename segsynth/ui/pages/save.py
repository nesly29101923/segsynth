"""Step 5 — copy the generated dataset to its final location."""
from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from ..qt_utils import card, label, page_header
from ..state import AppState
from .generate import PREVIEW_MASKS


def _safe_name(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_" else "_" for c in text).strip()
    return keep or "dataset"


class SavePage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.saved_to: Path | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)
        root.addWidget(page_header(
            "Save dataset",
            "Choose where the generated images go. The folder also gets a manifest.json describing every "
            "synthetic image: background, source defect, position and the exact variation applied. "
            "If a label tool database was created, keep it next to its Images folder.",
        ))

        info = QVBoxLayout()
        info.setContentsMargins(18, 14, 18, 14)
        self.summary = label("", wrap=True)
        info.addWidget(self.summary)
        root.addWidget(card(info))

        dest = QVBoxLayout()
        dest.setContentsMargins(18, 14, 18, 16)
        dest.setSpacing(10)
        dest.addWidget(label("DESTINATION", "SectionLabel"))
        row = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setPlaceholderText("Choose a folder…")
        browse = QPushButton("Browse…")
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        dest.addLayout(row)
        self.subfolder = QCheckBox("Create a subfolder named after the dataset")
        self.subfolder.setChecked(True)
        self.zip = QCheckBox("Also create a .zip archive")
        dest.addWidget(self.subfolder)
        dest.addWidget(self.zip)
        self.target_preview = label("", "Muted")
        dest.addWidget(self.target_preview)
        root.addWidget(card(dest))

        row = QHBoxLayout()
        self.save_btn = QPushButton("Save dataset")
        self.save_btn.setObjectName("Primary")
        self.save_btn.setMinimumWidth(160)
        self.open_btn = QPushButton("Open folder")
        self.open_btn.setEnabled(False)
        self.result = label("", "Muted")
        row.addWidget(self.save_btn)
        row.addWidget(self.open_btn)
        row.addSpacing(10)
        row.addWidget(self.result, 1)
        root.addLayout(row)
        root.addStretch(1)

        browse.clicked.connect(self._browse)
        self.save_btn.clicked.connect(self.save)
        self.open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.saved_to))))
        self.path.textChanged.connect(self._update)
        self.subfolder.toggled.connect(self._update)
        state.datasetChanged.connect(self._update)
        state.projectReplaced.connect(self._update)
        self._update()

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Save dataset to", self.path.text())
        if d:
            self.path.setText(d)

    def destination(self) -> Path | None:
        text = self.path.text().strip()
        if not text:
            return None
        base = Path(text)
        return base / _safe_name(self.state.project.name) if self.subfolder.isChecked() else base

    def _update(self, *_) -> None:
        m = self.state.manifest
        if m is None:
            self.summary.setText("No dataset yet. Generate one in step 4 first.")
        else:
            s = m["settings"]
            extras = [x for x, on in (("masks", s["save_masks"]), ("review overlays", s["save_overlays"])) if on]
            if m.get("labeltool_db"):
                src = m.get("source_images") or []
                good = sum(1 for x in src if x["label"] == "good")
                extras.append(f"label tool database {m['labeltool_db']}"
                              + (f" (with {len(src)} source images: {good} good, {len(src) - good} bad)" if src else ""))
            self.summary.setText(
                f"<b>{m['count']:,} synthetic images</b> ready"
                + (f" · plus {' and '.join(extras)}" if extras else "")
                + f" · created {m['created']}"
                + ("" if m["completed"] else " · <span style='color:#e3b341'>generation was cancelled early</span>"))
        dest = self.destination()
        self.target_preview.setText(f"Will be saved to: {dest}" if dest else "")
        self.save_btn.setEnabled(m is not None and dest is not None)

    def save(self) -> None:
        dest = self.destination()
        src = self.state.staging
        if dest is None or src is None or self.state.manifest is None:
            return
        if dest.exists() and any(dest.iterdir()):
            r = QMessageBox.question(self, "Folder not empty",
                                     f"{dest}\nalready contains files. Files with the same name will be overwritten. Continue?")
            if r != QMessageBox.Yes:
                return
        try:
            dest.mkdir(parents=True, exist_ok=True)
            for item in src.iterdir():
                if item.name == PREVIEW_MASKS:
                    continue
                target = dest / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)
            if self.zip.isChecked():
                archive = shutil.make_archive(str(dest), "zip", root_dir=dest)
                self.result.setText(f"Saved to {dest} and {Path(archive).name}")
            else:
                self.result.setText(f"Saved to {dest}")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.saved_to = dest
        self.open_btn.setEnabled(True)
