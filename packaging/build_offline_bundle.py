"""Build an offline bundle for a PC without internet.

Run on any machine with internet access:

    python packaging/build_offline_bundle.py --platform win_amd64
    python packaging/build_offline_bundle.py --platform linux_x86_64

Windows (win_amd64): a self-contained folder with its own Python 3.12 and every package
already installed. Nothing needs to be installed on the target PC: unzip and double-click
start.bat.

Linux: the app wheel, every dependency wheel for Python 3.10-3.13, and install/run scripts
(the target PC needs Python 3.10-3.13).

The result is written to release/SegSynth-<version>-offline-<platform>.zip.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHONS = ["3.10", "3.11", "3.12", "3.13"]
PLATFORM_ALIASES = {
    "linux_x86_64": ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64"],
    "linux_aarch64": ["manylinux_2_28_aarch64", "manylinux_2_17_aarch64", "manylinux2014_aarch64"],
}

# Standalone CPython for Windows (python-build-standalone, "install_only" variant).
EMBEDDED_PYTHON_VERSION = "3.12"
EMBEDDED_PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20241016/"
    "cpython-3.12.7+20241016-x86_64-pc-windows-msvc-install_only.tar.gz"
)
# Not needed by SegSynth; removed to keep the bundle small.
PYTHON_TRIM = [
    "include", "libs", "tcl", "Lib/idlelib", "Lib/tkinter", "Lib/turtledemo", "Lib/test",
    "Lib/ensurepip", "DLLs/_tkinter.pyd", "DLLs/tcl86t.dll", "DLLs/tk86t.dll",
]

WIN_START = r"""@echo off
rem SegSynth __VERSION__ - double-click to start.
rem This folder contains its own Python with everything installed; nothing is installed
rem on this PC. "start.bat debug" keeps a console window open to show error messages.
setlocal
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONPATH="
if not exist "%~dp0python\pythonw.exe" (
  echo.
  echo  The "python" folder is missing. Unzip the whole SegSynth bundle again
  echo  and start start.bat from the unzipped folder, not from inside the zip file.
  echo.
  pause
  exit /b 1
)
if /i "%~1"=="debug" (
  "%~dp0python\python.exe" -s -m segsynth
  echo.
  echo  SegSynth exited with code %errorlevel%.
  pause
  exit /b
)
start "" "%~dp0python\pythonw.exe" -s -m segsynth
"""

SH_INSTALL = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
"$PY" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' \\
  || { echo "SegSynth needs Python 3.10-3.13"; exit 1; }
rm -rf .venv
"$PY" -m venv .venv
.venv/bin/python -m pip install --no-index --find-links wheelhouse segsynth
echo "Installed. Start SegSynth with ./run.sh"
"""

SH_RUN = """#!/usr/bin/env bash
cd "$(dirname "$0")"
exec .venv/bin/python -m segsynth "$@"
"""

WIN_README = """SegSynth {version} - offline bundle for Windows (64-bit)

1. Copy the zip to the target PC (USB stick, network share, Ethernet) and unzip it.
2. Double-click start.bat in the unzipped folder.

Nothing needs to be installed: the "python" folder holds its own Python {pyver} with all
packages. The folder can live anywhere, including paths with spaces.

Update: delete this folder and unzip the new bundle in its place. Your projects and
datasets are stored elsewhere and are not affected.
Problems: open a console here and run "start.bat debug" to see error messages.
Repair a saved workspace:
    python\\python.exe -m segsynth prune "path\\to\\META\\name.sqlite"
"""

SH_README = """SegSynth {version} - offline bundle ({platform})

1. Copy this folder to the target PC.
2. Run ./install.sh (creates .venv from the wheelhouse folder, nothing is downloaded).
3. Start the app with ./run.sh

Requirements: 64-bit Python 3.10, 3.11, 3.12 or 3.13 on the target PC.
"""


def run(cmd: list) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def requirements() -> list[str]:
    return [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines() if l.strip()]


def build_app_wheel(dest: Path) -> Path:
    print("Building SegSynth wheel")
    run([sys.executable, "-m", "pip", "wheel", ROOT, "--no-deps", "-w", dest])
    return next(dest.glob("segsynth-*.whl"))


def fetch_python(source: str, cache: Path) -> Path:
    if Path(source).is_file():
        return Path(source)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / source.rsplit("/", 1)[-1].replace("%2B", "+")
    if not target.exists():
        print(f"Downloading Python: {source}")
        tmp = target.with_name(target.name + ".part")
        with urllib.request.urlopen(source) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(target)
    return target


def build_windows(stage: Path, version: str, python_dist: str, cache: Path) -> None:
    archive = fetch_python(python_dist, cache)
    print(f"Unpacking {archive.name}")
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(stage, filter="data")
        else:
            tar.extractall(stage)
    py = stage / "python"
    if not (py / "python.exe").exists():
        raise SystemExit("The Python archive does not contain python/python.exe")
    for rel in PYTHON_TRIM:
        p = py / rel
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    for pdb in py.rglob("*.pdb"):
        pdb.unlink()

    site = py / "Lib" / "site-packages"
    with tempfile.TemporaryDirectory() as tmp:
        wheel = build_app_wheel(Path(tmp))
        print(f"Installing SegSynth and dependencies into the bundled Python {EMBEDDED_PYTHON_VERSION}")
        run([sys.executable, "-m", "pip", "install", "--target", site, "--upgrade",
             "--platform", "win_amd64", "--python-version", EMBEDDED_PYTHON_VERSION,
             "--implementation", "cp", "--only-binary=:all:", "--no-compile",
             "--disable-pip-version-check", "-c", ROOT / "packaging" / "windows-bundle-constraints.txt",
             wheel, *requirements()])
    shutil.rmtree(site / "bin", ignore_errors=True)
    for cache_dir in list(site.rglob("__pycache__")):
        shutil.rmtree(cache_dir, ignore_errors=True)

    (stage / "start.bat").write_text(WIN_START.replace("__VERSION__", version), encoding="ascii", newline="\r\n")
    (stage / "README.txt").write_text(WIN_README.format(version=version, pyver=EMBEDDED_PYTHON_VERSION),
                                      encoding="utf-8", newline="\r\n")


def build_wheelhouse(stage: Path, version: str, platform: str, pythons: list[str]) -> None:
    wheels = stage / "wheelhouse"
    wheels.mkdir(parents=True)
    build_app_wheel(wheels)
    tags = PLATFORM_ALIASES.get(platform, [platform])
    platform_args = [a for t in tags for a in ("--platform", t)]
    for py in pythons:
        print(f"Downloading dependencies for Python {py} / {platform}")
        run([sys.executable, "-m", "pip", "download", *requirements(), "-d", wheels,
             "--only-binary=:all:", *platform_args, "--python-version", py, "--implementation", "cp"])
    for fname, body in (("install.sh", SH_INSTALL), ("run.sh", SH_RUN)):
        p = stage / fname
        p.write_text(body, encoding="utf-8", newline="\n")
        p.chmod(0o755)
    (stage / "README.txt").write_text(SH_README.format(version=version, platform=platform), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="win_amd64",
                    help="win_amd64 (self-contained), linux_x86_64, linux_aarch64 or any pip platform tag")
    ap.add_argument("--python", nargs="+", default=PYTHONS, help="Linux only: Python versions to include wheels for")
    ap.add_argument("--python-dist", default=EMBEDDED_PYTHON_URL,
                    help="Windows only: URL or local path of a python-build-standalone install_only archive")
    ap.add_argument("--out", default=str(ROOT / "release"))
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from segsynth import __version__

    name = f"SegSynth-{__version__}-offline-{args.platform}"
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / name
        stage.mkdir()
        if args.platform.startswith("win"):
            build_windows(stage, __version__, args.python_dist, out_root / "cache")
        else:
            build_wheelhouse(stage, __version__, args.platform, args.python)
        shutil.copy(ROOT / "LICENSE", stage / "LICENSE.txt")
        archive = shutil.make_archive(str(out_root / name), "zip", root_dir=stage.parent, base_dir=name)
    size = Path(archive).stat().st_size / 1e6
    print(f"\nBundle ready: {archive} ({size:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
