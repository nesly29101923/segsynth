"""Build an offline installer bundle for a PC without internet.

Run on any machine with internet access:

    python packaging/build_offline_bundle.py --platform win_amd64
    python packaging/build_offline_bundle.py --platform linux_x86_64

The result (release/SegSynth-<version>-offline-<platform>.zip) contains the app wheel,
every dependency wheel for Python 3.10–3.13, and one-click install/run scripts.
Copy it to the target PC by USB, unzip, run install, then run.
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

WIN_INSTALL = r"""@echo off
setlocal
cd /d "%~dp0"
echo.
echo  SegSynth offline installer
echo  --------------------------
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo  Python 3.10-3.13 was not found on this PC.
  echo  Install Python first, or use the standalone SegSynth.exe build.
  pause & exit /b 1
)
%PY% -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)"
if errorlevel 1 (
  echo  Unsupported Python version. SegSynth needs Python 3.10 to 3.13.
  %PY% --version
  pause & exit /b 1
)
if exist .venv rmdir /s /q .venv
%PY% -m venv .venv || goto :fail
.venv\Scripts\python -m pip install --no-index --find-links wheelhouse segsynth || goto :fail
echo.
echo  Installed. Start SegSynth with run.bat
pause
exit /b 0
:fail
echo.
echo  Installation failed. See the messages above.
pause
exit /b 1
"""

WIN_RUN = r"""@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
  echo Run install.bat first.
  pause
  exit /b 1
)
start "" .venv\Scripts\pythonw.exe -m segsynth
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

README = """SegSynth {version} — offline bundle ({platform})

1. Copy this folder to the target PC (USB stick, network share, remote desktop).
2. Run {install}. It creates a private environment in .venv next to this file.
   Nothing is downloaded; all packages come from the wheelhouse folder.
3. Start the app with {run}.

Requirements: 64-bit Python 3.10, 3.11, 3.12 or 3.13 on the target PC.
To update, unzip the new bundle over this folder and run {install} again.
Headless use:  {python} -m segsynth generate path\\to\\project.json output_folder
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
            (stage / "install.bat").write_text(WIN_INSTALL, encoding="utf-8", newline="\r\n")
            (stage / "run.bat").write_text(WIN_RUN, encoding="utf-8", newline="\r\n")
        else:
            for fname, body in (("install.sh", SH_INSTALL), ("run.sh", SH_RUN)):
                p = stage / fname
                p.write_text(body, encoding="utf-8", newline="\n")
                p.chmod(0o755)
        (stage / "README.txt").write_text(README.format(
            version=__version__, platform=args.platform,
            install="install.bat" if windows else "./install.sh",
            run="run.bat" if windows else "./run.sh",
            python=r".venv\Scripts\python" if windows else ".venv/bin/python",
        ), encoding="utf-8", newline="\r\n" if windows else "\n")
        shutil.copy(ROOT / "LICENSE", stage / "LICENSE.txt")
        archive = shutil.make_archive(str(out_root / name), "zip", root_dir=stage.parent, base_dir=name)
    size = Path(archive).stat().st_size / 1e6
    print(f"\nBundle ready: {archive} ({size:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
