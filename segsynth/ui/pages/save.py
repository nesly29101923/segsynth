"""Last step — copy the generated dataset to its final location, optionally with a label tool database."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QFileDialog, QHBoxLayout, QLineEdit, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from ...core.generator import BACKGROUND_DIR, SYNTHETIC_DIR, labeltool_db_name, write_labeltool_db
from ..qt_utils import card, label, page_header
from ..state import AppState


def _safe_name(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_" else "_" for c in text).strip()
    return keep or "dataset"


class SavePage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.saved_to: Path | None = None
        self._loading = False
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)
        root.addWidget(page_header(
            "Save dataset",
            f"Synthetic images are saved in “{SYNTHETIC_DIR}”, source and generated backgrounds in "
            f"“{BACKGROUND_DIR}”. A manifest.json describes every image: background, source defect, "
            "position and the exact variation applied.",
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
        self.target_preview = label("", "Muted", wrap=True)
        dest.addWidget(self.target_preview)
        root.addWidget(card(dest))

        db = QVBoxLayout()
        db.setContentsMargins(18, 14, 18, 16)
        db.setSpacing(10)
        db.addWidget(label("LABEL TOOL", "SectionLabel"))
        self.sqlite = QCheckBox("Generate a .sqlite file")
        self.sqlite.setToolTip("Creates a label tool database next to the image folders.")
        db.addWidget(self.sqlite)
        db.addWidget(label(
            "Every saved image becomes a TRAIN sample. Background areas are labeled good; pasted defects "
            "and defects you picked are labeled with the defect class. Images without defects are good, "
            "images with defects are not good.", "Muted", wrap=True))
        class_row = QHBoxLayout()
        class_row.setSpacing(8)
        class_row.addWidget(label("Defect class", "Muted"))
        self.class_name = QLineEdit()
        self.class_name.setPlaceholderText("defect")
        self.class_name.setMaximumWidth(260)
        self.class_color = QPushButton()
        self.class_color.setFixedWidth(44)
        self.class_color.setToolTip("Colour of the defect class in the label tool.")
        class_row.addWidget(self.class_name)
        class_row.addWidget(self.class_color)
        class_row.addStretch(1)
        db.addLayout(class_row)
        root.addWidget(card(db))

        row = QHBoxLayout()
        self.save_btn = QPushButton("Save dataset")
        self.save_btn.setObjectName("Primary")
        self.save_btn.setMinimumWidth(160)
        self.open_btn = QPushButton("Open folder")
        self.open_btn.setEnabled(False)
        self.result = label("", "Muted", wrap=True)
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
        self.sqlite.toggled.connect(self._settings_from_ui)
        self.class_name.textChanged.connect(self._settings_from_ui)
        self.class_color.clicked.connect(self._pick_color)
        state.datasetChanged.connect(self._update)
        state.projectReplaced.connect(self._load_settings)
        self._load_settings()

    # settings -----------------------------------------------------------
    def _load_settings(self) -> None:
        s = self.state.project.settings
        self._loading = True
        self.sqlite.setChecked(s.save_labeltool_db)
        self.class_name.setText(s.labeltool_class_name)
        self._show_color(s.labeltool_class_color)
        self._loading = False
        self._update()

    def _settings_from_ui(self, *_) -> None:
        if self._loading:
            return
        s = self.state.project.settings
        s.save_labeltool_db = self.sqlite.isChecked()
        s.labeltool_class_name = self.class_name.text().strip() or "defect"
        self.state.mark_dirty()
        self._update()

    def _show_color(self, color: str) -> None:
        self.class_color.setStyleSheet(
            f"QPushButton {{ background: {color}; border: 1px solid #3d4554; border-radius: 6px; }}"
            f"QPushButton:disabled {{ background: #2c323d; }}")

    def _pick_color(self) -> None:
        s = self.state.project.settings
        c = QColorDialog.getColor(QColor(s.labeltool_class_color), self, "Defect class colour")
        if c.isValid():
            s.labeltool_class_color = c.name()
            self._show_color(s.labeltool_class_color)
            self.state.mark_dirty()

    # ui -----------------------------------------------------------------
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
        on = self.sqlite.isChecked()
        self.class_name.setEnabled(on)
        self.class_color.setEnabled(on)
        m = self.state.manifest
        if m is None:
            self.summary.setText("No dataset yet. Generate one first.")
        else:
            files = m.get("files", [])
            bg = [f for f in files if f["kind"] != "synthetic"]
            good = sum(1 for f in files if f["label"] == "good")
            parts = [f"<b>{m['count']:,} synthetic images</b>"]
            if bg:
                parts.append(f"{len(bg):,} background images")
            self.summary.setText(
                " + ".join(parts) + " ready"
                + (f" · {good:,} good, {len(files) - good:,} not good" if files else "")
                + f" · created {m['created']}"
                + ("" if m["completed"] else " · <span style='color:#e3b341'>generation was cancelled early</span>"))
        dest = self.destination()
        if dest:
            lines = [f"Will be saved to: {dest}",
                     f"    {SYNTHETIC_DIR}\\   {BACKGROUND_DIR}\\   manifest.json"]
            if on:
                lines[-1] += f"   {labeltool_db_name(self.state.project)}"
            self.target_preview.setText("\n".join(lines))
        else:
            self.target_preview.setText("")
        self.save_btn.setEnabled(m is not None and dest is not None)

    def save(self) -> None:
        dest = self.destination()
        src = self.state.staging
        manifest = self.state.manifest
        if dest is None or src is None or manifest is None:
            return
        if dest.exists() and any(dest.iterdir()):
            r = QMessageBox.question(self, "Folder not empty",
                                     f"{dest}\nalready contains files. Files with the same name will be overwritten. Continue?")
            if r != QMessageBox.Yes:
                return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            dest.mkdir(parents=True, exist_ok=True)
            for item in src.iterdir():
                if item.name.startswith(".") or item.name == "manifest.json":
                    continue
                target = dest / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)
            out_manifest = dict(manifest)
            db_name = None
            if self.sqlite.isChecked():
                s = self.state.project.settings
                db_name = labeltool_db_name(self.state.project)
                write_labeltool_db(src, manifest.get("files", []), dest / db_name,
                                   s.labeltool_class_name, s.labeltool_class_color)
            out_manifest["labeltool_db"] = db_name
            (dest / "manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
            text = f"Saved to {dest}" + (f" with {db_name}" if db_name else "")
            if self.zip.isChecked():
                archive = shutil.make_archive(str(dest), "zip", root_dir=dest)
                text += f" and {Path(archive).name}"
            self.result.setText(text)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.saved_to = dest
        self.open_btn.setEnabled(True)

