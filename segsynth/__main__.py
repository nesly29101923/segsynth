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


def _prune(argv: list[str]) -> int:
    from .core.export import prune_workspace

    parser = argparse.ArgumentParser(
        prog="segsynth prune",
        description="Repair a saved workspace: remove real source images (real_*) and samples whose "
                    "image file is missing from DATA, from both the database and the DATA folder.")
    parser.add_argument("database", help="Path to the .sqlite file in the META folder")
    parser.add_argument("--keep-real", action="store_true", help="Keep real_* images")
    args = parser.parse_args(argv)
    removed = prune_workspace(args.database, remove_real=not args.keep_real)
    print(f"Removed {len(removed)} sample(s) from {args.database}")
    for p in removed[:20]:
        print("  " + p)
    if len(removed) > 20:
        print(f"  … and {len(removed) - 20} more")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        return _cli(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "prune":
        return _prune(sys.argv[2:])
    from .app import run
    return run()


if __name__ == "__main__":
    sys.exit(main())
