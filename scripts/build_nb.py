"""Build the `advanced_experiments/*.ipynb` notebooks from readable cell sources.

Keeping the notebook JSON out of the source tree (and the prose in plain Python
strings) makes the notebooks reviewable in a diff and regenerable after any
change to the underlying scripts.

Usage:
    python scripts/build_nb.py <notebook_name>
    python scripts/build_nb.py --all
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Flood predictor")
NB_DIR = ROOT / "advanced_experiments"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


def build(name: str, cells: list[dict]) -> Path:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NB_DIR / f"{name}.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}  ({len(cells)} cells)")
    return path


def main(argv: list[str]) -> int:
    from nb_cells import NOTEBOOKS       # noqa: E402
    names = argv or list(NOTEBOOKS)
    for n in names:
        cells = NOTEBOOKS[n]()
        build(n, cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
