"""Shared application state passed to every page."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..core.project import Project


class AppState(QObject):
    projectReplaced = Signal()
    imagesChanged = Signal()
    defectsChanged = Signal()
    settingsChanged = Signal()
    datasetChanged = Signal()
    dirtyChanged = Signal(bool)

    def __init__(self):
        super().__init__()
        self.project = Project()
        self.project_path: Path | None = None
        self.staging: Path | None = None
        self.manifest: dict | None = None
        self._dirty = False
        for sig in (self.imagesChanged, self.defectsChanged, self.settingsChanged):
            sig.connect(lambda *_: self.mark_dirty())

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self, value: bool = True) -> None:
        if value != self._dirty:
            self._dirty = value
            self.dirtyChanged.emit(value)

    def replace_project(self, project: Project, path: Path | None = None) -> None:
        self.project = project
        self.project_path = path
        self.discard_dataset()
        self.projectReplaced.emit()
        self.imagesChanged.emit()
        self.defectsChanged.emit()
        self.settingsChanged.emit()
        self.mark_dirty(False)

    def new_staging(self) -> Path:
        self.discard_dataset()
        self.staging = Path(tempfile.mkdtemp(prefix="segsynth_"))
        return self.staging

    def discard_dataset(self) -> None:
        if self.staging and self.staging.exists():
            shutil.rmtree(self.staging, ignore_errors=True)
        self.staging = None
        self.manifest = None
        self.datasetChanged.emit()
