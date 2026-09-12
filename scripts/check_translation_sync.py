#!/usr/bin/env python3
"""Check that hand-authored lesson translations still match their English source.

A language registered in ``languages.json`` with ``"lessons": "human"`` keeps its
lesson markdown on main next to the English file, at
``phases/<phase>/<lesson>/docs/<code>.md``. Machine translations are regenerated
whenever the English changes; a hand-authored one is not, so it silently rots the
moment someone edits ``en.md``.

This script is the tripwire for that. It compares the skeleton of each pair --
line count, fenced-code delimiters, table rows, headings, list markers, figure
blocks, and the literal fenced-block bodies -- and reports the first place a file
diverges. Prose is never compared: only the structure the translation is supposed
to mirror line for line.

Usage:
    python3 scripts/check_translation_sync.py            # all human languages
    python3 scripts/check_translation_sync.py --lang ko
    python3 scripts/check_translation_sync.py --lang ko --limit 5
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASES = ROOT / "phases"

LIST_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s")
FENCE_RE = re.compile(r"^\s*```")
HEADING_RE = re.compile(r"^#{1,6}\s")
FIGURE_RE = re.compile(r"^```figure\s*$")


def human_languages():
    registry = json.loads((ROOT / "languages.json").read_text(encoding="utf-8"))
    return [e["code"] for e in registry["languages"] if e.get("lessons") == "human"]


def classify(line):
    """Reduce a line to the structural role the translation must preserve."""
    stripped = line.strip()
    if not stripped:
        return "blank"
    if FENCE_RE.match(line):
        return "fence"
    if HEADING_RE.match(stripped):
        return "heading:" + str(len(stripped) - len(stripped.lstrip("#")))
    if stripped.startswith("|"):
        return "table-row"
    if LIST_RE.match(line):
        return "list-item"
    return "text"


def fenced_bodies(lines):
    """Every fenced block's delimiter and body, in order.

    Code, mermaid graphs, and ``figure`` markers are copied verbatim into a
    translation, so any difference here is a real drift rather than a wording
    choice.
    """
    blocks = []
    open_at = None
    for index, line in enumerate(lines):
        if not FENCE_RE.match(line):
            continue
        if open_at is None:
            open_at = index
        else:
            blocks.append((lines[open_at], tuple(lines[open_at + 1:index])))
            open_at = None
    return blocks


def compare(en_lines, tr_lines):
    """Return a human-readable reason the two files diverge, or None."""
    if len(en_lines) != len(tr_lines):
        first = next(
            (i for i in range(min(len(en_lines), len(tr_lines)))
             if classify(en_lines[i]) != classify(tr_lines[i])),
            min(len(en_lines), len(tr_lines)),
        )
        return (f"line count {len(en_lines)} != {len(tr_lines)}; "
                f"first structural difference at line {first + 1}")

    for index, (en_line, tr_line) in enumerate(zip(en_lines, tr_lines), start=1):
        en_kind, tr_kind = classify(en_line), classify(tr_line)
        if en_kind != tr_kind:
            return f"line {index}: English is {en_kind}, translation is {tr_kind}"

    en_blocks, tr_blocks = fenced_bodies(en_lines), fenced_bodies(tr_lines)
    if len(en_blocks) != len(tr_blocks):
        return f"fenced block count {len(en_blocks)} != {len(tr_blocks)}"
    for position, (en_block, tr_block) in enumerate(zip(en_blocks, tr_blocks), start=1):
        if FIGURE_RE.match(en_block[0]) and en_block != tr_block:
            return f"figure block {position} does not match the English source"
    return None


def lesson_pairs(lang):
    for en in sorted(PHASES.glob("*/*/docs/en.md")):
        yield en, en.with_name(f"{lang}.md")


def check(lang, limit):
    missing, stale, checked = [], [], 0
    for en, translated in lesson_pairs(lang):
        if not translated.is_file():
            missing.append(str(en.parent.relative_to(ROOT)))
            continue
        checked += 1
        reason = compare(
            en.read_text(encoding="utf-8").split("\n"),
            translated.read_text(encoding="utf-8").split("\n"),
        )
        if reason:
            stale.append((str(translated.relative_to(ROOT)), reason))

    print(f"{lang}: {checked} lesson(s) checked, "
          f"{len(missing)} missing, {len(stale)} out of sync")
    for path in missing[:limit]:
        print(f"  missing {lang}.md: {path}")
    if len(missing) > limit:
        print(f"  ... and {len(missing) - limit} more missing")
    for path, reason in stale[:limit]:
        print(f"  out of sync: {path}\n      {reason}")
    if len(stale) > limit:
        print(f"  ... and {len(stale) - limit} more out of sync")
    return not (missing or stale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", help="one language code; default is every human-authored language")
    parser.add_argument("--limit", type=int, default=20, help="how many problem files to print per language")
    args = parser.parse_args()

    languages = [args.lang] if args.lang else human_languages()
    if not languages:
        print("no hand-authored lesson languages registered in languages.json")
        return 0
    return 0 if all([check(lang, args.limit) for lang in languages]) else 1


if __name__ == "__main__":
    sys.exit(main())
