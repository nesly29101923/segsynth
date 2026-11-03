"""Build an offline installer bundle for a PC without internet.

Run on any machine with internet access:

    python packaging/build_offline_bundle.py --platform win_amd64
    python packaging/build_offline_bundle.py --platform linux_x86_64

The result (release/SegSynth-<version>-offline-<platform>.zip) contains the app wheel,
every dependency wheel for Python 3.10–3.13, and one-click install/run scripts.
Copy it to the target PC, unzip, and double-click start.bat.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHONS = ["3.10", "3.11", "3.12", "3.13"]
PLATFORM_ALIASES = {
    "linux_x86_64": ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64"],
    "linux_aarch64": ["manylinux_2_28_aarch64", "manylinux_2_17_aarch64", "manylinux2014_aarch64"],
}

WIN_START = r"""@echo off
rem SegSynth __VERSION__ - double-click to start. The first start installs SegSynth from the
rem wheelhouse folder (no internet needed). "start.bat debug" keeps a console open for errors.
setlocal
cd /d "%~dp0"
set "APP_VERSION=__VERSION__"
set "VPY=%~dp0.venv\Scripts\python.exe"
if exist "%VPY%" (
  "%VPY%" -c "import sys, segsynth, cv2, PySide6; sys.exit(segsynth.__version__ != '%APP_VERSION%')" >nul 2>nul
  if not errorlevel 1 goto :launch
)
call :install
if errorlevel 1 goto :fail

:launch
if /i "%~1"=="debug" (
  "%VPY%" -m segsynth
  pause
  exit /b
)
start "" "%~dp0.venv\Scripts\pythonw.exe" -m segsynth
exit /b 0

:install
echo.
echo  SegSynth %APP_VERSION% - first start, installing (about one minute)...
echo.
set "PY="
where py >nul 2>nul
if not errorlevel 1 (
  for %%v in (3.12 3.13 3.11 3.10) do if not defined PY call :try_python "py -%%v"
)
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 call :try_python "python"
)
if not defined PY (
  echo  Python 3.10 - 3.13 ^(64-bit^) was not found on this PC.
  echo  Install it first, for example python-3.12.x-amd64.exe copied from another PC,
  echo  and tick "Add python.exe to PATH" in the installer. Then start this file again.
  exit /b 1
)
if exist .venv rmdir /s /q .venv
%PY% -m venv .venv
if errorlevel 1 exit /b 1
"%VPY%" -m pip install --no-index --find-links wheelhouse --disable-pip-version-check -q segsynth
if errorlevel 1 exit /b 1
echo  Installed.
exit /b 0

:fail
echo.
echo  SegSynth could not be installed. See the messages above.
pause
exit /b 1

:try_python
%~1 -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=%~1"
exit /b 0
"""

WIN_REINSTALL = r"""@echo off
rem Removes the installed environment and installs SegSynth again from the wheelhouse folder.
cd /d "%~dp0"
if exist .venv rmdir /s /q .venv
call "%~dp0start.bat"
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

README = """SegSynth {version} - offline bundle ({platform})

1. Copy this whole folder to the target PC (USB stick, network share, Ethernet).
2. Double-click {start}. The first start installs SegSynth into a private .venv
   folder next to this file (about one minute, nothing is downloaded). Every later
   start opens the app directly.

Requirements: 64-bit Python 3.10, 3.11, 3.12 or 3.13 on the target PC.
Update: unzip the new bundle over this folder and double-click {start};
        a new version is detected and installed automatically.
Problems: run "{start} debug" from a console to see error messages, or {reinstall}
          to remove the environment and install again.
Repair a saved workspace: {python} -m segsynth prune "path\\to\\META\\name.sqlite"
"""


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="win_amd64",
                    help="win_amd64, linux_x86_64, linux_aarch64 or any pip platform tag")
    ap.add_argument("--python", nargs="+", default=PYTHONS, help="Python versions to include wheels for")
    ap.add_argument("--out", default=str(ROOT / "release"))
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from segsynth import __version__

    windows = args.platform.startswith("win")
    name = f"SegSynth-{__version__}-offline-{args.platform}"
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / name
        wheels = stage / "wheelhouse"
        wheels.mkdir(parents=True)
        print("Building SegSynth wheel")
        run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "-w", str(wheels)])
        reqs = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines() if l.strip()]
        tags = PLATFORM_ALIASES.get(args.platform, [args.platform])
        platform_args = [a for t in tags for a in ("--platform", t)]
        for py in args.python:
            print(f"Downloading dependencies for Python {py} / {args.platform}")
            run([sys.executable, "-m", "pip", "download", *reqs, "-d", str(wheels),
                 "--only-binary=:all:", *platform_args, "--python-version", py,
                 "--implementation", "cp"])
        if windows:
            (stage / "start.bat").write_text(WIN_START.replace("__VERSION__", __version__),
                                             encoding="ascii", newline="\r\n")
            (stage / "reinstall.bat").write_text(WIN_REINSTALL, encoding="ascii", newline="\r\n")
        else:
            for fname, body in (("install.sh", SH_INSTALL), ("run.sh", SH_RUN)):
                p = stage / fname
                p.write_text(body, encoding="utf-8", newline="\n")
                p.chmod(0o755)
        (stage / "README.txt").write_text(README.format(
            version=__version__, platform=args.platform,
            start="start.bat" if windows else "./install.sh, then ./run.sh",
            reinstall="reinstall.bat" if windows else "./install.sh",
            python=r".venv\Scripts\python" if windows else ".venv/bin/python",
        ), encoding="utf-8", newline="\r\n" if windows else "\n")
        shutil.copy(ROOT / "LICENSE", stage / "LICENSE.txt")
        archive = shutil.make_archive(str(out_root / name), "zip", root_dir=stage.parent, base_dir=name)
    size = Path(archive).stat().st_size / 1e6
    print(f"\nBundle ready: {archive} ({size:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
