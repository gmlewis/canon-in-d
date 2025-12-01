#!/usr/bin/env python3
"""Orchestrate the full Canon In D build pipeline from the DAW project."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DAW_FILE = ROOT_DIR / "CanonInD.dawproject"


def run_step(description: str, command: list[str]) -> None:
    """Run a subprocess step and fail fast on error."""
    print("\n" + "=" * 66)
    print(description)
    print("=" * 66)
    result = subprocess.run(command, cwd=ROOT_DIR)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    if not DAW_FILE.exists():
        print(f"ERROR: Missing source DAW project: {DAW_FILE}", file=sys.stderr)
        return 1

    run_step("Converting DAW project to CanonInD.json",
             ["python3", "dawproject_to_json.py", "CanonInD.dawproject"])
    run_step("Renaming SVG note head paths", ["python3", "rename_svg_paths.py"])
    run_step("Building canonical notehead map", ["python3", "build-notehead-map.py"])
    run_step("Generating jumping-curve data", ["python3", "gen-note-jumping-curves-to-json.py"])
    run_step("Running regression test suite", ["./test-all.sh"])

    print("\n✅ Build pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
