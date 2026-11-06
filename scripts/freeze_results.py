r"""Freeze the paper's evidence: hashes, headline numbers, environment.

The correction sprint changes numbers.  Before the paper is written, this script
creates one manifest that says *exactly* which files and values are the frozen
record, so a later run cannot silently change a table.

    .venv\Scripts\python.exe scripts\freeze_results.py          # write the manifest
    .venv\Scripts\python.exe scripts\freeze_results.py --check  # verify, do not write

The check fails if any hashed file changed or any headline number drifted.
It never touches the frozen test.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.getenv("AABHAS_ROOT", Path(__file__).resolve().parents[1])).resolve()
REP = ROOT / "reports"
OUT = REP / "freeze_manifest.json"
DOC = REP / "FREEZE.md"

# Files that define the paper's quantitative evidence.
FILES = [
    "data/processed/experimental_dataset.csv",
    "data/processed/mi_v2_features.csv",
    "data/processed/mi8_features.csv",
    "models/final_v2/selected_model_v2.pkl",
    "models/final_v2/final_config_v2.json",
    "reports/p2p3_rescore/summary.json",
    "reports/p2p3_rescore/probe_p2p3.csv",
    "reports/p2p3_rescore/arms_p2p3.csv",
    "reports/p2p3_rescore/d7_p2p3.csv",
    "reports/p2p3_rescore/d7_margins_p2p3.csv",
    "reports/p2p3_correction.md",
    "reports/event_level/summary.json",
    "reports/event_level/event_metrics.csv",
    "reports/event_level/summary.md",
    "reports/mi6_summary.json",
    "reports/mi9_summary.json",
    "reports/phase27_summary.json",
    "reports/phase31_summary.json",
    "reports/phase28_summary.json",
    "reports/phase34_summary.json",
    "reports/test_read_ledger.md",
    "reports/phase27_ml_vs_dl.md",
    "reports/phase31_robustness.md",
    "reports/phase28_calibration.md",
    "reports/phase34_xai.md",
    "reports/mi6_test_read.md",
]

PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "xgboost", "lightgbm",
            "torch", "matplotlib", "shap"]

# Human-readable revision history for the evidence lock. Append a new entry
# whenever a hashed artefact changes; never rewrite history in place.
REVISIONS = [
    {
        "version": 1,
        "date": "2026-10-22",
        "note": "Initial lock: D16 frozen model, MI-6 test read, correction-sprint numbers.",
    },
    {
        "version": 2,
        "date": "2026-11-06",
        "note": (
            "Documentation-only refresh. `reports/test_read_ledger.md` and "
            "`reports/mi6_test_read.md` were re-worded after the original lock; "
            "all 13 numeric/evidence checks still pass. No dataset, model, "
            "prediction, or headline number changed and the frozen test was not read."
        ),
    },
]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        p = subprocess.run(["git", "-C", str(ROOT), *args],
                           capture_output=True, text=True, timeout=30)
        return p.stdout.strip() if p.returncode == 0 else ""
    except Exception:
        return ""


def head(probe_or_arm: pd.DataFrame, model: str, cell: str, hz: str, col="pr_auc") -> float:
    r = probe_or_arm[(probe_or_arm.model == model) & (probe_or_arm.cell == cell)
                     & (probe_or_arm.horizon == hz)]
    return float(r[col].iloc[0])


def build() -> dict:
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # ---- frozen headline numbers -------------------------------------
    mi10 = json.loads((REP / "p2p3_rescore" / "summary.json").read_text(encoding="utf-8"))
    check("MI-10 summary not void", mi10.get("status") != "VOID", str(mi10.get("status")))
    check("MI-10 gates", all(g["pass"] for g in mi10["gates"]),
          f"{sum(g['pass'] for g in mi10['gates'])}/{len(mi10['gates'])} pass")

    arms = pd.read_csv(REP / "p2p3_rescore" / "arms_p2p3.csv")
    v22 = {hz: head(arms, "d16_frozen", "val_2022", hz) for hz in
           ["label_24h", "label_48h", "label_72h"]}
    exp = {"label_24h": 0.970789, "label_48h": 0.956485, "label_72h": 0.945007}
    for hz in v22:
        check(f"D16 val_2022 {hz}", abs(v22[hz] - exp[hz]) < 1e-5,
              f"{v22[hz]:.6f} vs {exp[hz]:.6f}")
    v22_mean = float(np.mean(list(v22.values())))
    check("D16 val_2022 horizon-mean", abs(v22_mean - 0.957427) < 1e-5, f"{v22_mean:.6f}")

    mi6 = json.loads((REP / "mi6_summary.json").read_text(encoding="utf-8"))
    rows = mi6["headline_all_season"]
    d16 = {h: float(np.mean([r[h]["d16"] for r in rows])) for h in ("24h", "48h", "72h")}
    d7 = {h: float(np.mean([r[h]["d7"] for r in rows])) for h in ("24h", "48h", "72h")}
    check("D16 test per-horizon", all(abs(d16[h] - x) < 1e-4 for h, x in
                                     {"24h": 0.836226, "48h": 0.768958, "72h": 0.707088}.items()),
          " / ".join(f"{d16[h]:.6f}" for h in d16))
    d16_mean = float(np.mean(list(d16.values())))
    d7_mean = float(np.mean(list(d7.values())))
    check("D16 test horizon-mean", abs(d16_mean - 0.770757) < 1e-5, f"{d16_mean:.6f}")
    check("D7 test horizon-mean", abs(d7_mean - 0.66755) < 1e-4, f"{d7_mean:.6f}")

    ev = json.loads((REP / "event_level" / "summary.json").read_text(encoding="utf-8"))
    check("event analysis is dev-only", ev.get("test_rows_read") == 0,
          str(ev.get("test_rows_read")))
    check("event analysis has all cells", len(ev.get("cells", [])) == 7,
          str(len(ev.get("cells", []))))

    dataset_rows = sum(1 for _ in (ROOT / "data/processed/experimental_dataset.csv")
                       .open(encoding="utf-8", errors="ignore")) - 1
    check("dataset row count", dataset_rows == 200665, f"{dataset_rows}")

    # ---- hashes ------------------------------------------------------
    files = []
    for rel in FILES:
        p = ROOT / rel
        if not p.exists():
            files.append({"path": rel, "exists": False, "sha256": None, "bytes": None})
            continue
        files.append({"path": rel, "exists": True, "sha256": sha256(p), "bytes": p.stat().st_size})
    check("all frozen files exist", all(f["exists"] for f in files),
          f"{sum(f['exists'] for f in files)}/{len(files)}")

    pkgs = {}
    for name in PACKAGES:
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = None

    manifest = {
        "frozen_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "seed": 20261008,
        "test_policy": "closed - no read in this freeze; D16 headline is the valid MI-6 read",
        "revisions": REVISIONS,
        "headline": {
            "val_2022_d16_per_horizon": v22,
            "val_2022_d16_horizon_mean": round(v22_mean, 6),
            "test_d16_per_horizon": {k: round(v, 6) for k, v in d16.items()},
            "test_d16_horizon_mean": round(d16_mean, 6),
            "test_d7_per_horizon": {k: round(v, 6) for k, v in d7.items()},
            "test_d7_horizon_mean": round(d7_mean, 6),
            "best_ml_val_2022": 0.954634,
            "best_dl_val_2022": 0.93557,
            "event_level": {
                "mean_event_recall": ev.get("mean_event_recall"),
                "mean_fa_per_station_year": ev.get("mean_fa_per_station_year"),
            },
        },
        "checks": checks,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": pkgs,
        },
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        },
        "files": files,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(sorted((f["path"], f["sha256"]) for f in files), sort_keys=True).encode()
    ).hexdigest()
    return manifest


def write_doc(m: dict) -> None:
    h = m["headline"]
    ok = all(c["pass"] for c in m["checks"])
    lines = [
        "# Frozen results — paper evidence lock",
        "",
        f"**Frozen at:** {m['frozen_at']} · **Seed:** `{m['seed']}` · "
        f"**Manifest SHA-256:** `{m['manifest_sha256'][:24]}…`",
        "",
        f"**Verification:** {'ALL CHECKS PASS' if ok else '**CHECK FAILURE — DO NOT WRITE**'} "
        f"({sum(c['pass'] for c in m['checks'])}/{len(m['checks'])})",
        "",
        "This file freezes the numbers the IEEE paper may quote. A later change to any",
        "hashed artefact invalidates the manifest and must be documented as a new version.",
        "",
        "## Revision history",
        "",
        "| version | date | change |",
        "| ---: | --- | --- |",
    ]
    for rev in REVISIONS:
        lines.append(f"| {rev['version']} | {rev['date']} | {rev['note']} |")
    lines += [
        "",
        "## Headline numbers",
        "",
        "| quantity | value |",
        "| --- | ---: |",
        f"| D16 `val_2022` 24 / 48 / 72 h | {h['val_2022_d16_per_horizon']['label_24h']:.6f} / "
        f"{h['val_2022_d16_per_horizon']['label_48h']:.6f} / "
        f"{h['val_2022_d16_per_horizon']['label_72h']:.6f} |",
        f"| D16 `val_2022` horizon mean | **{h['val_2022_d16_horizon_mean']:.6f}** |",
        f"| D16 frozen test 24 / 48 / 72 h | {h['test_d16_per_horizon']['24h']:.6f} / "
        f"{h['test_d16_per_horizon']['48h']:.6f} / {h['test_d16_per_horizon']['72h']:.6f} |",
        f"| D16 frozen test horizon mean | **{h['test_d16_horizon_mean']:.6f}** |",
        f"| D7 frozen test horizon mean | {h['test_d7_horizon_mean']:.6f} |",
        f"| best classical / best deep, `val_2022` | {h['best_ml_val_2022']:.4f} / "
        f"{h['best_dl_val_2022']:.4f} |",
        "",
        "## Checks",
        "",
        "| check | result | detail |",
        "| --- | --- | --- |",
    ]
    for c in m["checks"]:
        lines.append(f"| {c['check']} | {'PASS' if c['pass'] else '**FAIL**'} | {c['detail']} |")
    lines += [
        "",
        "## Artefact hashes",
        "",
        "| file | bytes | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    for f in m["files"]:
        if f["exists"]:
            lines.append(f"| `{f['path']}` | {f['bytes']:,} | `{f['sha256']}` |")
        else:
            lines.append(f"| `{f['path']}` | — | **MISSING** |")
    lines += [
        "",
        "## Environment",
        "",
        f"- Python {m['environment']['python']} on {m['environment']['platform']}",
        f"- Git: `{m['git']['commit'][:12] or 'not a repo'}` "
        f"({'dirty' if m['git']['dirty'] else 'clean'})",
    ]
    for k, v in m["environment"]["packages"].items():
        lines.append(f"- {k} {v or 'not installed'}")
    lines += [
        "",
        "## Test policy at freeze",
        "",
        m["test_policy"],
        "See `reports/test_read_ledger.md` for the full six-open history.",
        "",
        "## Verify",
        "",
        "```powershell",
        r".venv\Scripts\python.exe scripts\freeze_results.py --check",
        "```",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    m = build()
    ok = all(c["pass"] for c in m["checks"])

    if a.check:
        if not OUT.exists():
            print("FATAL: no manifest to check; run without --check first")
            return 2
        old = json.loads(OUT.read_text(encoding="utf-8"))
        old_h = {f["path"]: f["sha256"] for f in old["files"]}
        bad = [f["path"] for f in m["files"] if old_h.get(f["path"]) != f["sha256"]]
        if bad:
            print("CHANGED FILES:")
            for b in bad:
                print("  ", b)
        print(f"checks {sum(c['pass'] for c in m['checks'])}/{len(m['checks'])}; "
              f"changed files {len(bad)}")
        return 0 if (ok and not bad) else 1

    OUT.write_text(json.dumps(m, indent=2), encoding="utf-8")
    write_doc(m)
    print(f"WROTE {OUT}")
    print(f"WROTE {DOC}")
    print(f"checks {sum(c['pass'] for c in m['checks'])}/{len(m['checks'])}")
    if not ok:
        print("FATAL: a freeze check failed; the manifest is written but marked invalid.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
