#!/usr/bin/env python3
"""
Build src/ for mdBook from per-stub folders + top-level docs.

mdBook requires:
- book.toml at repo root (already present)
- src/ with chapter .md files referenced by src/SUMMARY.md

This script:
1. Resets src/
2. Copies README.md -> src/index.md
3. Copies RESULTS.md -> src/results.md
4. Copies VISUAL_TOUR.md -> src/visual-tour.md
5. Copies BUILD_NOTES.md -> src/build-notes.md
6. Copies each stub folder -> src/<slug>/ (READMEs + viz/ + .gif)
7. Generates src/SUMMARY.md grouped by era

Usage:
    python3 bin/build_book.py

CI runs this before `mdbook build`. src/ is gitignored.

Mirrors hinton-problems/bin/build_book.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo",
    ".cache", "*.npz", "*.tar.gz", "*.gz",
)

# Era grouping for SUMMARY.md, mirroring the README's catalog.
# Order within each era is curated.
ERAS = [
    ("1980s — Local rules and the Neural Bucket Brigade", [
        "nbb-xor",
        "nbb-moving-light",
    ]),
    ("1990 — Controller + world-model + flip-flop", [
        "flip-flop",
        "pole-balance-non-markov",
        "pole-balance-markov-vac",
        "saccadic-target-detection",
    ]),
    ("1991 — Curiosity, subgoals, the chunker", [
        "curiosity-three-regions",
        "subgoal-obstacle-avoidance",
        "pomdp-flag-maze",
        "chunker-22-symbol",
    ]),
    ("1992 — Neural Computation triple", [
        "fast-weights-unknown-delay",
        "fast-weights-key-value",
        "predictability-min-binary-factors",
    ]),
    ("1993 — Predictable classifications, self-reference, very deep chunking", [
        "predictable-stereo",
        "self-referential-weight-matrix",
        "chunker-very-deep-1200",
    ]),
    ("1995–1997 — Levin search and the LSTM benchmark suite", [
        "levin-count-inputs",
        "levin-add-positions",
        "rs-two-sequence",
        "rs-parity",
        "rs-tomita",
        "adding-problem",
        "embedded-reber",
        "noise-free-long-lag",
        "two-sequence-noise",
        "multiplication-problem",
        "temporal-order-3bit",
        "temporal-order-4bit",
    ]),
    ("Mid-90s — Evolutionary, RL, and feature detection", [
        "pipe-symbolic-regression",
        "pipe-6-bit-parity",
        "ssa-bias-transfer-mazes",
        "hq-learning-pomdp",
        "semilinear-pm-image-patches",
        "lococode-ica",
    ]),
    ("2000–2002 — LSTM follow-ups", [
        "continual-embedded-reber",
        "anbn-anbncn",
        "timing-counting-spikes",
        "blues-improvisation",
    ]),
    ("2002–2010 — Evolutionary RL, OOPS, BLSTM+CTC", [
        "evolino-sines-mackey-glass",
        "double-pole-no-velocity",
        "timit-blstm-ctc",
        "iam-handwriting",
        "oops-towers-of-hanoi",
    ]),
    ("2010–2017 — Deep learning at scale", [
        "mnist-deep-mlp",
        "mcdnn-image-bench",
        "em-segmentation-isbi",
        "compete-to-compute",
        "highway-networks",
        "lstm-search-space-odyssey",
        "clockwork-rnn",
        "torcs-vision-evolution",
        "neural-em-shapes",
        "relational-nem-bouncing-balls",
    ]),
    ("2018–2025 — World models, fast-weight Transformers, systematic generalization", [
        "world-models-carracing",
        "world-models-vizdoom-dream",
        "upside-down-rl",
        "linear-transformers-fwp",
        "neural-data-router",
    ]),
]


def stub_title(slug: str) -> str:
    """Pretty title for nav."""
    return slug


def main() -> None:
    if SRC.exists():
        shutil.rmtree(SRC)
    SRC.mkdir()

    # Top-level pages. Source uses uppercase filenames (so links work on
    # GitHub's repo view); mdBook generates lowercase-hyphenated HTML, so
    # rewrite the inter-page references after copy.
    shutil.copy(ROOT / "README.md", SRC / "index.md")
    shutil.copy(ROOT / "RESULTS.md", SRC / "results.md")
    shutil.copy(ROOT / "VISUAL_TOUR.md", SRC / "visual-tour.md")
    shutil.copy(ROOT / "BUILD_NOTES.md", SRC / "build-notes.md")

    LINK_REWRITES = [
        ("RESULTS.md", "results.md"),
        ("VISUAL_TOUR.md", "visual-tour.md"),
        ("BUILD_NOTES.md", "build-notes.md"),
        ("README.md", "index.md"),
    ]
    for top in ("index.md", "results.md", "visual-tour.md", "build-notes.md"):
        path = SRC / top
        text = path.read_text()
        for old, new in LINK_REWRITES:
            text = text.replace(f"({old})", f"({new})")
            text = text.replace(f"][{old}]", f"][{new}]")
        path.write_text(text)

    # Per-stub folders
    all_stubs: list[str] = []
    for _, slugs in ERAS:
        all_stubs.extend(slugs)

    missing: list[str] = []
    for slug in all_stubs:
        src_dir = ROOT / slug
        if not src_dir.exists():
            missing.append(slug)
            continue
        dst_dir = SRC / slug
        shutil.copytree(src_dir, dst_dir, ignore=IGNORE)

    if missing:
        print(f"WARNING: {len(missing)} stub folders missing: {missing}")

    # Generate SUMMARY.md
    summary = ["# Summary", ""]
    summary.append("[Home](index.md)")
    summary.append("[Visual tour](visual-tour.md)")
    summary.append("[Results catalog](results.md)")
    summary.append("[Build notes](build-notes.md)")
    summary.append("")
    for era, slugs in ERAS:
        summary.append(f"# {era}")
        summary.append("")
        for slug in slugs:
            if slug in missing:
                continue
            summary.append(f"- [{stub_title(slug)}]({slug}/README.md)")
        summary.append("")

    (SRC / "SUMMARY.md").write_text("\n".join(summary) + "\n")

    n_chapters = len(all_stubs) - len(missing)
    print(f"Built {SRC} with {n_chapters} stub chapters + 4 top-level pages")


if __name__ == "__main__":
    main()
