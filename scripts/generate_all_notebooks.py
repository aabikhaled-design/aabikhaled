#!/usr/bin/env python3
"""Batch generate interactive Jupyter notebooks across lesson directories in phases/.

Usage:
    python scripts/generate_all_notebooks.py [--phase N] [--overwrite] [--dry-run]

Flags:
    --phase N     Restrict notebook generation to a single phase number (e.g., --phase 1)
    --overwrite   Overwrite existing experiment.ipynb files (default: skip if already present)
    --dry-run     Display which lesson notebooks would be created without writing to disk
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"

PHASE_DIR_RE = re.compile(r"^[0-9]{2}-[a-z0-9][a-z0-9-]*[a-z0-9]$")
LESSON_DIR_RE = re.compile(r"^[0-9]{2}-[a-z0-9][a-z0-9-]*[a-z0-9]$")


def iter_lesson_dirs(phase_filter: int | None) -> Iterable[Path]:
    if not PHASES_DIR.is_dir():
        return
    for phase in sorted(PHASES_DIR.iterdir()):
        if not phase.is_dir() or not PHASE_DIR_RE.match(phase.name):
            continue
        if phase_filter is not None:
            try:
                phase_num = int(phase.name.split("-", 1)[0])
            except ValueError:
                continue
            if phase_num != phase_filter:
                continue
        for lesson in sorted(phase.iterdir()):
            if lesson.is_dir() and LESSON_DIR_RE.match(lesson.name):
                yield lesson


def parse_lesson_docs(doc_path: Path) -> tuple[str, str]:
    title = doc_path.parent.parent.name
    motto = ""
    if doc_path.is_file():
        try:
            lines = doc_path.read_text(encoding="utf-8").splitlines()
            for l in lines:
                if l.startswith("# "):
                    title = l[2:].strip()
                elif l.startswith("> ") and not motto:
                    motto = l[2:].strip()
        except Exception:
            pass
    return title, motto


def generate_notebook(lesson_dir: Path, overwrite: bool, dry_run: bool) -> bool:
    notebook_dir = lesson_dir / "notebook"
    nb_path = notebook_dir / "experiment.ipynb"

    if nb_path.exists() and not overwrite:
        return False

    title, motto = parse_lesson_docs(lesson_dir / "docs" / "en.md")
    main_py = lesson_dir / "code" / "main.py"
    code_text = main_py.read_text(encoding="utf-8") if main_py.is_file() else ""

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {title}: Interactive Visual Explorer\n",
                "\n",
                f"> {motto}\n" if motto else "\n",
                "\n",
                f"Welcome to the interactive companion notebook for **{title}**.\n",
                "\n",
                "In this notebook, you can interactively execute the lesson's raw implementation, plot state transformations, and run experiment variations.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import math\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Configure plotting aesthetics\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "plt.rcParams['figure.figsize'] = (8, 5)\n",
                "plt.rcParams['font.size'] = 11\n"
            ]
        }
    ]

    if code_text.strip():
        raw_blocks = [b.strip() for b in code_text.split("\n\n") if b.strip()]
        current_chunk: list[str] = []
        for block in raw_blocks:
            current_chunk.append(block)
            if len("\n\n".join(current_chunk)) > 400:
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["\n\n".join(current_chunk) + "\n"]
                })
                current_chunk = []
        if current_chunk:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["\n\n".join(current_chunk) + "\n"]
            })

    notebook_json = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    if dry_run:
        print(f"[DRY-RUN] Would create: {nb_path.relative_to(ROOT)}")
        return True

    notebook_dir.mkdir(parents=True, exist_ok=True)
    nb_path.write_text(json.dumps(notebook_json, indent=2), encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, default=None, help="restrict generation to a single phase number")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing notebook files")
    parser.add_argument("--dry-run", action="store_true", help="display actions without writing files")
    args = parser.parse_args(argv)

    total_lessons = 0
    created_count = 0
    skipped_count = 0

    for lesson in iter_lesson_dirs(args.phase):
        total_lessons += 1
        created = generate_notebook(lesson, overwrite=args.overwrite, dry_run=args.dry_run)
        if created:
            created_count += 1
        else:
            skipped_count += 1

    action_label = "Would create" if args.dry_run else "Created"
    print(f"generate_all_notebooks.py — {total_lessons} lesson(s) checked | {action_label}: {created_count} | Skipped: {skipped_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
