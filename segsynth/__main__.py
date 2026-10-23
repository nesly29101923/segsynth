"""Entry point: `python -m segsynth` opens the app; `python -m segsynth generate …` runs headless."""
from __future__ import annotations

import argparse
import sys


def _cli(argv: list[str]) -> int:
    from .core.generator import GenerationError, run_to_folder
    from .core.project import Project

    parser = argparse.ArgumentParser(prog="segsynth generate", description="Generate a dataset from a saved project.")
    parser.add_argument("project", help="Path to project.json or its folder")
    parser.add_argument("output", help="Output folder")
    parser.add_argument("--seed", type=int, help="Override the random seed")
    parser.add_argument("--masks", action="store_true", help="Also write masks")
    args = parser.parse_args(argv)

    proj = Project.load(args.project)
    if args.seed is not None:
        proj.settings.seed = args.seed
    if args.masks:
        proj.settings.save_masks = True

    def progress(done, total, _sample):
        print(f"\r{done}/{total}", end="", flush=True)

    try:
        manifest = run_to_folder(proj, args.output, progress)
    except GenerationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"\nWrote {manifest['count']} images to {args.output}")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        return _cli(sys.argv[2:])
    from .app import run
    return run()


if __name__ == "__main__":
    sys.exit(main())
