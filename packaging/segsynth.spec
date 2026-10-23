# PyInstaller spec: standalone build that needs no Python on the target PC.
#   pyinstaller packaging/segsynth.spec --noconfirm
# Output: dist/SegSynth/ (zip that folder and copy it to the target PC)
from pathlib import Path

ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=["segsynth.app", "segsynth.ui.main_window"],
    excludes=["tkinter", "matplotlib", "PySide6.QtWebEngineCore", "PySide6.Qt3DCore", "PySide6.QtQml", "PySide6.QtQuick"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SegSynth",
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SegSynth")
