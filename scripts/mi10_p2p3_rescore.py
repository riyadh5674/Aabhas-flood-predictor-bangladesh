r"""MI-10 — P2/P3 correction of the MI-sprint 7-cell claims.

Why this exists
---------------
`mi3_compare.py`, `mi3_search.py`, `mi5_selection.py`, `mi8_select.py` and
`mi8_probe.py` fit one model on the <=2021 training rows and then score all seven
"development cells".  Six of those cells (val_2020, val_2021 and the four spatial
folds) are subsets of the fit rows, so their scores are in-sample.  Only val_2022
is out-of-sample under that fit.

This script re-runs the two affected comparisons with the project's own per-cell
protocol (`baseline_model.build_tasks()`): every cell gets its own fit and its own
training-derived threshold.

Blocks
------
probe : fixed-config feature A/B -- FULL vs FULL_V2 vs FULL_V3, identical XGB spec
arms  : frozen D16 vs the MI-8 selected arms (full_v3, full_v2), per-cell refits
d7    : the D7 persistence floor, refit per cell

Test policy
-----------
Dev cells only.  The evaluation masks are asserted to contain zero `split == "test"`
rows.  `test_rows_read = 0`.

Design and reporting rules are pre-registered in
`reports/p2p3_correction_protocol.md` (written before this script was run).

Run
---
    .venv\Scripts\python.exe -u scripts\mi10_p2p3_rescore.py
    .venv\Scripts\python.exe -u scripts\mi10_p2p3_rescore.py --smoke
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B                                  # noqa: E402
import mi3_compare as C                                     # noqa: E402
import mi3_search as S                                      # noqa: E402
import mi8_search as M8                                     # noqa: E402
from mi8_select import cfg_from_row, D16, D16_EXPECTED       # noqa: E402

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
OUT = REPORTS / "p2p3_rescore"

HOR = B.HORIZONS
SEED = B.SEED
CELLS = ["val_2020", "val_2021", "val_2022",
         "spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]
PROTOCOL = {"val_2020": "P2", "val_2021": "P2", "val_2022": "P1",
            **{f"spatial_fold_{k}": "P3" for k in (1, 2, 3, 4)}}
PROBE_SETS = ["FULL", "FULL_V2", "FULL_V3"]
ARMS = ["d16_frozen", "mi8_full_v3", "mi8_full_v2"]
EPS = 1e-9


# ------------------------------------------------------------------ data + tasks
def load_all():
    """Base table + MI-2 block + MI-8 block, and the three feature lists."""
    d, fsets = B.load()
    v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
    d = d.merge(v2, on=["station_id", "date"], validate="one_to_one")
    m8 = pd.read_csv(PROC / "mi8_features.csv", parse_dates=["date"])
    d = d.merge(m8, on=["station_id", "date"], validate="one_to_one")
    assert len(d) == 200665, f"row count changed: {len(d)}"

    fs_v2 = json.loads((PROC / "feature_sets_v2.json").read_text(encoding="utf-8"))["sets"]["FULL_V2"]
    fs_v3 = json.loads((PROC / "feature_sets_mi8.json").read_text(encoding="utf-8"))["sets"]["FULL_V3"]
    feats = {"FULL": fsets["FULL"], "FULL_V2": fs_v2, "FULL_V3": fs_v3}
    assert len(feats["FULL"]) == 62 and len(feats["FULL_V2"]) == 110 and len(feats["FULL_V3"]) == 165
    return d, feats


def cell_tasks(d: pd.DataFrame) -> dict[str, dict]:
    """The project's P1/P2/P3 masks, one task per cell (baseline_model.build_tasks)."""
    holdout = d.station_holdout.to_numpy()
    year = d.date.dt.year.to_numpy()
    split = d.split.to_numpy()
    fold = d.spatial_fold.to_numpy()

    tasks = {
        "val_2020": {"protocol": "P2", "train_name": "train_<=2019",
                     "train": (year <= 2019) & ~holdout,
                     "eval": (year == 2020) & ~holdout},
        "val_2021": {"protocol": "P2", "train_name": "train_<=2020",
                     "train": (year <= 2020) & ~holdout,
                     "eval": (year == 2021) & ~holdout},
        "val_2022": {"protocol": "P1", "train_name": "train_<2022",
                     "train": (split == "train") & ~holdout,
                     "eval": (split == "val") & ~holdout},
    }
    for k in (1, 2, 3, 4):
        tasks[f"spatial_fold_{k}"] = {
            "protocol": "P3", "train_name": f"excl_fold_{k}",
            "train": (split == "train") & ~holdout & (fold != k),
            "eval": (split == "train") & (fold == k),
        }

    # the two surfaces this correction depends on must agree with the frozen split
    assert np.array_equal(tasks["val_2022"]["train"], (year <= 2021) & ~holdout), \
        "val_2022 train fold is not the <=2021 P1 fold"
    test = split == "test"
    for name, t in tasks.items():
        assert not (t["train"] & test).any(), f"{name}: test row in train mask"
        assert not (t["eval"] & test).any(), f"{name}: test row in eval mask"
    return tasks


# ------------------------------------------------------------------ model specs
def d7_pipeline(_feats):
    """Exactly the D7 reference in `baseline_model.build_models`."""
    return Pipeline([
        ("pre", B.preprocessor(["wl_pct_danger"], [], scale=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)),
    ])


def resolve_arm_specs():
    """Frozen per-horizon configs for D16 and the two MI-8 arms."""
    mi3 = pd.read_csv(REPORTS / "mi3_hpo.csv")
    d16 = {}
    for hz in HOR:
        g = mi3[(mi3.horizon == hz) & (mi3.tag == D16[hz])]
        if g.empty:
            raise SystemExit(f"FATAL: D16 config {D16[hz]}/{hz} missing from mi3_hpo.csv")
        row = g.iloc[0]
        d16[hz] = {"family": row.family, "tag": row.tag, "cfg": cfg_from_row(row)}

    def winners(path):
        hpo = M8._read(path)
        if hpo.empty:
            raise SystemExit(f"FATAL: missing or empty {path}")
        out = {}
        for hz in HOR:
            g = hpo[hpo.horizon == hz]
            if g.empty:
                raise SystemExit(f"FATAL: no {hz} rows in {path.name}")
            row = g.sort_values("val_2022", ascending=False).iloc[0]
            out[hz] = {"family": row.family, "tag": row.tag, "cfg": cfg_from_row(row)}
        return out

    v3 = winners(REPORTS / "mi8_hpo_full_v3.csv")
    v2 = winners(REPORTS / "mi8_hpo_full_v2.csv")
    return {"d16_frozen": d16, "mi8_full_v3": v3, "mi8_full_v2": v2}


def arm_roster(hz, specs, feats):
    return [
        ("d16_frozen", feats["FULL_V2"], specs["d16_frozen"][hz]),
        ("mi8_full_v3", feats["FULL_V3"], specs["mi8_full_v3"][hz]),
        ("mi8_full_v2", feats["FULL_V2"], specs["mi8_full_v2"][hz]),
    ]


# ------------------------------------------------------------------ scoring
def _ln(x):
    return math.log(x) if x > 0 else math.log(EPS)


def skill_scores(tp, fp, fn, tn):
    """SEDI/EDI/PSS from a confusion table (same definitions as sedi_edi_report.py)."""
    h = tp / (tp + fn) if (tp + fn) else float("nan")
    f = fp / (fp + tn) if (fp + tn) else float("nan")
    if not (math.isfinite(h) and math.isfinite(f)):
        return float("nan"), float("nan"), float("nan")
    lnf, lnh, ln1f, ln1h = _ln(f), _ln(h), _ln(1 - f), _ln(1 - h)
    den_e = lnf + lnh
    den_s = lnf + lnh + ln1f + ln1h
    edi = (_ln_ratio(lnf, lnh)) if abs(den_e) >= 1e-12 else float("nan")
    sedi = ((lnf - lnh - ln1f + ln1h) / den_s) if abs(den_s) >= 1e-12 else float("nan")
    return sedi, edi, h - f


def _ln_ratio(lnf, lnh):
    den = lnf + lnh
    return (lnf - lnh) / den if abs(den) >= 1e-12 else float("nan")


def add_skill_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute SEDI/EDI/PSS from confusion counts when a source table lacks them."""
    if df.empty or "sedi" in df.columns:
        return df
    df = df.copy()
    vals = [skill_scores(int(tp), int(fp), int(fn), int(tn))
            for tp, fp, fn, tn in zip(df.tp, df.fp, df.fn, df.tn)]
    df["sedi"], df["edi"], df["pss"] = zip(*vals)
    return df


def fit_score(pipe, d, feats, tr, ev, y, hz, cell, task, label, family, config):
    t0 = time.time()
    pipe.fit(d.loc[tr, feats], y[tr])
    p_tr = pipe.predict_proba(d.loc[tr, feats])[:, 1]
    thr = B.best_f1_threshold(y[tr], p_tr)
    p_ev = pipe.predict_proba(d.loc[ev, feats])[:, 1]
    met = B.metrics(y[ev], p_ev, thr)
    sedi, edi, pss = skill_scores(met["tp"], met["fp"], met["fn"], met["tn"])
    return {
        "model": label, "family": family, "config": config,
        "horizon": hz, "cell": cell, "protocol": task["protocol"],
        "train": task["train_name"], "n_train": int(tr.sum()),
        "n": int(ev.sum()), "n_pos": int(y[ev].sum()),
        "base_rate": met["base_rate"], "threshold": met["threshold"],
        "pr_auc": met["pr_auc"], "roc_auc": met["roc_auc"],
        "brier": met["brier"], "logloss": met["logloss"], "ece": met["ece"],
        "tp": met["tp"], "fp": met["fp"], "fn": met["fn"], "tn": met["tn"],
        "recall": met["recall"], "precision": met["precision"],
        "specificity": met["specificity"], "f1": met["f1"],
        "sedi": sedi, "edi": edi, "pss": pss,
        "fit_s": round(time.time() - t0, 1),
    }


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=CELLS)
    ap.add_argument("--horizons", nargs="*", default=HOR)
    ap.add_argument("--blocks", nargs="*", default=["probe", "arms", "d7"],
                    choices=["probe", "arms", "d7"])
    ap.add_argument("--smoke", action="store_true",
                    help="one cell x one horizon x all blocks; writes smoke_* files")
    a = ap.parse_args(argv)

    cells = ["val_2022"] if a.smoke else [c for c in CELLS if c in a.cells]
    horizons = ["label_24h"] if a.smoke else [h for h in HOR if h in a.horizons]
    blocks = set(a.blocks)
    full = (not a.smoke and len(cells) == len(CELLS) and len(horizons) == len(HOR)
            and blocks == {"probe", "arms", "d7"})
    OUT.mkdir(exist_ok=True)

    t0 = time.time()
    print("=== MI-10 P2/P3 correction ===")
    print(f"seed={SEED}  cells={cells}  horizons={horizons}  blocks={sorted(blocks)}")
    print("per-cell refits; val_2022 is the only cell that reproduces a published fit")
    print("test_rows_read = 0")
    print()

    d, feats = load_all()
    tasks = cell_tasks(d)
    specs = resolve_arm_specs()

    rows_probe, rows_arms, rows_d7 = [], [], []
    for hz in horizons:
        y = d[hz].to_numpy()
        for cell in cells:
            task = tasks[cell]
            tr = task["train"] & ~pd.isna(y)
            ev = task["eval"] & ~pd.isna(y)
            assert not (ev & (d.split.to_numpy() == "test")).any()

            if "probe" in blocks:
                for fs in PROBE_SETS:
                    pipe = C.make_xgb(feats[fs])
                    rows_probe.append(fit_score(
                        pipe, d, feats[fs], tr, ev, y, hz, cell, task,
                        fs, "xgb_fixed", "mi3_compare.XGB_PARAMS"))
                    print(f"  probe {hz[-3:]} {cell:15s} {fs:8s} "
                          f"n={int(ev.sum()):6d} pr_auc={rows_probe[-1]['pr_auc']:.6f} "
                          f"({rows_probe[-1]['fit_s']:.0f}s)", flush=True)

            if "arms" in blocks:
                for label, fs_feats, spec in arm_roster(hz, specs, feats):
                    pipe = S.build(spec["family"], spec["cfg"], fs_feats)
                    rows_arms.append(fit_score(
                        pipe, d, fs_feats, tr, ev, y, hz, cell, task,
                        label, spec["family"], spec["tag"]))
                    print(f"  arm   {hz[-3:]} {cell:15s} {label:12s} "
                          f"n={int(ev.sum()):6d} pr_auc={rows_arms[-1]['pr_auc']:.6f} "
                          f"({rows_arms[-1]['fit_s']:.0f}s)", flush=True)

            if "d7" in blocks:
                pipe = d7_pipeline(None)
                rows_d7.append(fit_score(
                    pipe, d, ["wl_pct_danger"], tr, ev, y, hz, cell, task,
                    "d7_persistence", "logreg", "wl_pct_danger"))
                print(f"  d7    {hz[-3:]} {cell:15s} "
                      f"pr_auc={rows_d7[-1]['pr_auc']:.6f}", flush=True)

            # checkpoint after every cell so a late failure cannot lose the run
            if full:
                pd.DataFrame(rows_probe).to_csv(OUT / "probe_p2p3.csv", index=False)
                pd.DataFrame(rows_arms).to_csv(OUT / "arms_p2p3.csv", index=False)
                pd.DataFrame(rows_d7).to_csv(OUT / "d7_p2p3.csv", index=False)

    probe = pd.DataFrame(rows_probe)
    arms = pd.DataFrame(rows_arms)
    d7 = pd.DataFrame(rows_d7)

    prefix = "smoke_" if not full else ""
    for df, name in ((probe, "probe"), (arms, "arms"), (d7, "d7")):
        if not df.empty:
            df.to_csv(OUT / f"{prefix}{name}_p2p3.csv", index=False)

    if not full:
        print("\npartial/smoke run: files written with the smoke_ prefix; summary not written")
        if not probe.empty:
            print(probe[["model", "horizon", "cell", "pr_auc", "n", "n_train"]].to_string(index=False))
        return 0

    # ------------------------------------------------------------- gates
    gates = []

    def gate(name, ok, detail):
        gates.append({"gate": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    arch_probe = pd.read_csv(REPORTS / "mi8_probe.csv")
    arch_sc = add_skill_columns(pd.read_csv(REPORTS / "mi8_scorecard.csv"))

    print("\n=== reproduction gates (val_2022) ===")
    v22p = probe[probe.cell == "val_2022"]
    for fs in PROBE_SETS:
        for hz in HOR:
            new = float(v22p[(v22p.model == fs) & (v22p.horizon == hz)].pr_auc.iloc[0])
            old = float(arch_probe[(arch_probe.featureset == fs) & (arch_probe.horizon == hz)
                                   & (arch_probe.cell == "val_2022")].pr_auc.iloc[0])
            gate(f"probe_{fs}_{hz[-3:]}", abs(new - old) < 1e-6,
                 f"new={new:.9f} archived={old:.9f} d={new - old:+.2e}")

    v22a = arms[arms.cell == "val_2022"]
    for hz in HOR:
        new = float(v22a[(v22a.model == "d16_frozen") & (v22a.horizon == hz)].pr_auc.iloc[0])
        exp = D16_EXPECTED[hz]
        gate(f"d16_{hz[-3:]}", abs(new - exp) < 1e-5,
             f"new={new:.9f} frozen={exp:.9f} d={new - exp:+.2e}")
    for label in ("mi8_full_v3", "mi8_full_v2"):
        for hz in HOR:
            new = float(v22a[(v22a.model == label) & (v22a.horizon == hz)].pr_auc.iloc[0])
            old = float(arch_sc[(arch_sc.model == label) & (arch_sc.horizon == hz)
                                & (arch_sc.cell == "val_2022")].pr_auc.iloc[0])
            gate(f"{label}_{hz[-3:]}", abs(new - old) < 1e-5,
                 f"new={new:.9f} archived={old:.9f} d={new - old:+.2e}")

    if any(not g["pass"] for g in gates):
        print("\nFATAL: a reproduction gate failed — corrected numbers are void.")
        (OUT / "summary.json").write_text(
            json.dumps({"gates": gates, "status": "VOID"}, indent=2), encoding="utf-8")
        return 1

    # ------------------------------------------------------------- summaries
    def mean_by(df, label_col, label, hz, col="pr_auc"):
        s = df[(df[label_col] == label) & (df.horizon == hz)]
        return float(s[col].mean()) if len(s) else float("nan")

    def mean7(df, label_col, label, col="pr_auc"):
        s = df[df[label_col] == label]
        return float(s[col].mean()) if len(s) else float("nan")

    probe_piv = probe.pivot_table(index=["horizon", "cell"], columns="model", values="pr_auc")
    probe_piv["d_v3_v2"] = probe_piv["FULL_V3"] - probe_piv["FULL_V2"]
    probe_piv["d_v3_full"] = probe_piv["FULL_V3"] - probe_piv["FULL"]

    arch_piv = arch_probe.pivot_table(index=["horizon", "cell"], columns="featureset",
                                      values="pr_auc")
    arch_piv["d_v3_v2"] = arch_piv["FULL_V3"] - arch_piv["FULL_V2"]

    probe_summary = {
        "mean7": {fs: mean7(probe, "model", fs) for fs in PROBE_SETS},
        "archived_mean7": {fs: mean7(arch_probe, "featureset", fs) for fs in PROBE_SETS},
        "per_horizon_mean7": {
            hz: {fs: mean_by(probe, "model", fs, hz) for fs in PROBE_SETS} for hz in HOR
        },
        "archived_per_horizon_mean7": {
            hz: {fs: mean_by(arch_probe, "featureset", fs, hz) for fs in PROBE_SETS}
            for hz in HOR
        },
        "delta_v3_vs_v2": {
            "mean7": float(probe_piv["d_v3_v2"].mean()),
            "archived_mean7": float(arch_piv["d_v3_v2"].mean()),
            "per_horizon": {hz: float(probe_piv.loc[hz, "d_v3_v2"].mean()) for hz in HOR},
            "wins": int((probe_piv["d_v3_v2"] > 0).sum()),
            "total": int(probe_piv["d_v3_v2"].notna().sum()),
            "min": float(probe_piv["d_v3_v2"].min()),
            "max": float(probe_piv["d_v3_v2"].max()),
            "archived_wins": int((arch_piv["d_v3_v2"] > 0).sum()),
            "archived_total": int(arch_piv["d_v3_v2"].notna().sum()),
            "by_cell": {c: float(probe_piv.loc[(slice(None), c), "d_v3_v2"].mean())
                        for c in CELLS},
        },
        "delta_v3_vs_full": {
            "mean7": float(probe_piv["d_v3_full"].mean()),
            "archived_mean7": float((arch_piv["FULL_V3"] - arch_piv["FULL"]).mean()),
            "wins": int((probe_piv["d_v3_full"] > 0).sum()),
            "total": int(probe_piv["d_v3_full"].notna().sum()),
        },
    }

    arm_piv = arms.pivot_table(index="horizon", columns="model", values="pr_auc")
    arm_summary = {
        "mean7": {m: mean7(arms, "model", m) for m in ARMS},
        "archived_mean7": {m: mean7(arch_sc, "model", m) for m in ARMS},
        "per_horizon_mean7": {hz: {m: mean_by(arms, "model", m, hz) for m in ARMS}
                              for hz in HOR},
        "delta_v3_vs_d16": {hz: float(arm_piv.loc[hz, "mi8_full_v3"] - arm_piv.loc[hz, "d16_frozen"])
                            for hz in HOR},
        "delta_v2_vs_d16": {hz: float(arm_piv.loc[hz, "mi8_full_v2"] - arm_piv.loc[hz, "d16_frozen"])
                            for hz in HOR},
        "sedi_mean7": {m: {hz: mean_by(arms, "model", m, hz, "sedi") for hz in HOR}
                       for m in ARMS},
        "archived_sedi_mean7": {
            m: {hz: mean_by(arch_sc, "model", m, hz, "sedi") for hz in HOR} for m in ARMS
        },
    }
    arm_summary["delta_v3_vs_d16_mean7"] = (
        arm_summary["mean7"]["mi8_full_v3"] - arm_summary["mean7"]["d16_frozen"])
    arm_summary["delta_v2_vs_d16_mean7"] = (
        arm_summary["mean7"]["mi8_full_v2"] - arm_summary["mean7"]["d16_frozen"])

    # D7 margins, cell by cell
    margin_rows = []
    for hz in HOR:
        for cell in CELLS:
            base = float(d7[(d7.horizon == hz) & (d7.cell == cell)].pr_auc.iloc[0])
            for label in ("d16_frozen", "mi8_full_v3", "mi8_full_v2"):
                val = float(arms[(arms.horizon == hz) & (arms.cell == cell)
                                 & (arms.model == label)].pr_auc.iloc[0])
                margin_rows.append({"horizon": hz, "cell": cell, "model": label,
                                    "pr_auc": val, "d7_pr_auc": base,
                                    "margin": val - base})
    margins = pd.DataFrame(margin_rows)
    margins.to_csv(OUT / "d7_margins_p2p3.csv", index=False)
    d7_summary = {
        "mean7": {hz: float(d7[d7.horizon == hz].pr_auc.mean()) for hz in HOR},
        "margins": {
            m: {"min": float(margins[margins.model == m].margin.min()),
                "above": int((margins[margins.model == m].margin > 0).sum()),
                "total": int(len(margins[margins.model == m]))}
            for m in ("d16_frozen", "mi8_full_v3", "mi8_full_v2")
        },
    }

    summary = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "test_rows_read": 0,
        "protocol": "per-cell refits; baseline_model.build_tasks masks",
        "cells": CELLS,
        "horizons": HOR,
        "gates": gates,
        "probe": probe_summary,
        "arms": arm_summary,
        "d7": d7_summary,
        "wall_s": round(time.time() - t0, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- console
    print("\n=== corrected feature A/B (mean over 7 cells, PR-AUC) ===")
    for fs in PROBE_SETS:
        print(f"  {fs:8s} corrected={probe_summary['mean7'][fs]:.6f}  "
              f"archived(in-sample)={probe_summary['archived_mean7'][fs]:.6f}")
    dv = probe_summary["delta_v3_vs_v2"]
    print(f"  FULL_V3 - FULL_V2: corrected {dv['mean7']:+.6f} "
          f"(archived {dv['archived_mean7']:+.6f}); "
          f"wins {dv['wins']}/{dv['total']} (archived {dv['archived_wins']}/{dv['archived_total']})")

    print("\n=== corrected MI-8 arms (mean over 7 cells, PR-AUC) ===")
    for m in ARMS:
        print(f"  {m:14s} corrected={arm_summary['mean7'][m]:.6f}  "
              f"archived(in-sample)={arm_summary['archived_mean7'][m]:.6f}")
    print(f"  mi8_v3 - d16: {arm_summary['delta_v3_vs_d16_mean7']:+.6f}")

    print("\n=== corrected D7 margins ===")
    for m, v in d7_summary["margins"].items():
        print(f"  {m:14s} above D7 {v['above']}/{v['total']}, min margin {v['min']:+.6f}")

    print(f"\nWROTE {OUT}  wall={summary['wall_s']:.0f}s")
    print("reports/p2p3_correction.md must be regenerated from summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
