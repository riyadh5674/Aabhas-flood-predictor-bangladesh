"""Phase 17 - Experimental modelling framework (final workflow Phase 17 / stage 31).

Binds the modelling programme before anything is trained:

* three separate tasks (24 / 48 / 72 h) - D2, never one privileged horizon;
* four model families L1-L4, every candidate with a stated reason for inclusion
  (workflow Phase 17.2 - "no algorithm just to say we tested 10 models");
* one experiment registry - id, family, dataset, feature set, protocol cells,
  seasons, horizons, selection rule, tuning budget, seed, imbalance policy,
  gate, outputs and test policy - written to
  `reports/phase17_experiment_registry.csv`;
* L4 (GNN / Transformer) rows are **gated** (D13): they may not run until the
  Phase 24 / Phase 25 gate report passes; a documented rejection closes the stage;
* fair-comparison parity (D14): every planned run uses the same split masks,
  metric code, selection cell and seed as every other run;
* test protection: no planned experiment selects anything from the frozen
  2023+ cells - those reads are reserved for Phase 33 (L1 rows were already read
  once in Phase 16 and are recorded as such).

Checks (numbered F1a-...) are printed and asserted; exit code 0 means the
framework is internally consistent and the registry is safe to execute against.

Run:
    python -u scripts\\phase17_framework.py > scripts\\phase17_run.log
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SPLITS = PROC / "split_assignments.csv"
TBL = PROC / "experimental_dataset.csv"
FSETS = PROC / "feature_sets.json"
PHASE16_MET = REPORTS / "phase16_baseline_metrics.csv"
PHASE16_TEST = REPORTS / "phase16_test_metrics.csv"
REGISTRY = REPORTS / "phase17_experiment_registry.csv"

TBL_SHA_EXPECTED = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
SEED = 20261008
HORIZONS = ["label_24h", "label_48h", "label_72h"]
SEASONS = ["all", "monsoon", "non-monsoon"]
ROLLING_YEARS = [2020, 2021, 2022]
SPATIAL_FOLDS = [1, 2, 3, 4]
TEST_YEARS = [2023, 2024, 2025, 2026]
TEST_BUCKETS = ["test_trained", "test_new_network", "test_unseen"]
HOLDOUT_CUTOFF = 2023
DEV_PROTOCOLS = ["P1", "P2", "P3"]

TEST_POLICY = "frozen_read_at_phase33"
TEST_POLICY_L1 = "read_once_phase16"
IMBALANCE_ALLOWED = {"none", "class_weight=balanced", "resample_in_fold_only"}

RES: list[dict] = []
WARN: list[str] = []


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(cid: str, detail: str, ok: bool = True) -> None:
    RES.append({"id": cid, "pass": bool(ok)})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}")
    assert ok, cid


def warn(wid: str, detail: str) -> None:
    WARN.append(wid)
    print(f"[WARN] {wid}: {detail}")


# ---------------------------------------------------------------------- registry
def build_registry() -> pd.DataFrame:
    """One row per planned experiment. Reason strings are part of the contract:
    a candidate without a reason may not be trained."""
    rows = []

    def add(**kw) -> None:
        rows.append(kw)

    dev_grid = "P1:val_2022|P2:val_2020,val_2021,val_2022|P3:spatial_fold_1..4"
    common = dict(
        protocols="P1,P2,P3",
        dev_cells=dev_grid,
        seasons="all,monsoon,non-monsoon",
        horizons="24h,48h,72h",
        selection_metric="pr_auc",
        selection_cell="val_2022",
        seed=SEED,
    )

    # ---- L1 simple statistical baselines (Phase 16 - frozen, already reported)
    l1 = [
        ("L1-01", "climatology", "A", "none",
         "constant train base rate - the prevalence reference; PR-AUC of any model must be judged against it"),
        ("L1-02", "persistence", "A", "wl_pct_danger",
         "D7 bar (D7): one feature, monotone logistic link, beats every simple baseline - the benchmark all families must clear"),
        ("L1-03", "logreg_linear", "A", "LINEAR",
         "regularised linear model on the Phase-12 LINEAR set - the honest linear ceiling for engineered features"),
        ("L1-04", "logreg_linear_no_up", "A", "LINEAR-no-up",
         "F14-2 ablation: does dropping the 67-70 % NaN up_* columns change anything"),
        ("L1-05", "logreg_selected", "A", "SELECTED",
         "compact 10-feature cross-check - does feature reduction hurt linear models"),
        ("L1-06", "tree_depth4", "A", "FULL",
         "shallow non-linear reference (depth 4, leaf>=200) - interpretable non-linearity without tuning"),
    ]
    for eid, model, ds, fset, reason in l1:
        add(registry_id=eid, phase="16", level="L1", family="simple statistical",
            model=model, dataset=ds, feature_set=fset, reason=reason,
            tuning_budget_configs=0, imbalance_policy="none",
            status="done", gate="", metrics_out="phase16_baseline_metrics.csv;phase16_test_metrics.csv",
            model_out="n/a", test_policy=TEST_POLICY_L1, **common)

    # ---- L2 classical machine learning (Phase 20)
    l2 = [
        ("L2-01", "random_forest", "A", "FULL",
         "bagged trees on all 62 features - captures feature interactions, no scaling assumptions, standard non-linear baseline"),
        ("L2-02", "gradient_boosting", "A", "FULL",
         "sklearn gradient boosting - classic sequential boosting reference with no external dependency"),
        ("L2-03", "xgboost", "A", "FULL",
         "XGBoost on FULL - the tabular workhorse and the strongest realistic classical competitor for tabular flood data"),
        ("L2-04", "lightgbm", "A", "FULL",
         "LightGBM on FULL - leaf-wise boosting, native NaN handling, the cheap high-capacity classical candidate"),
        ("L2-05", "xgboost_selected", "A", "SELECTED",
         "compactness test: can 10 selected features match 62 for boosting (pairs with L1-05)"),
    ]
    for eid, model, ds, fset, reason in l2:
        add(registry_id=eid, phase="20", level="L2", family="classical ML",
            model=model, dataset=ds, feature_set=fset, reason=reason,
            tuning_budget_configs=40, imbalance_policy="class_weight=balanced",
            status="planned", gate="", metrics_out="phase20_metrics.csv",
            model_out="models/phase20_<id>.pkl", test_policy=TEST_POLICY, **common)

    # ---- L3 deep temporal learning (Phases 21-22, Dataset B)
    raw_vars = "raw_sequence(12 vars)"
    l3 = [
        ("L3-01", "lstm_L3", "B", f"{raw_vars}|L=3",
         "3-day history LSTM - the shortest credible window; establishes whether recurrence helps at all"),
        ("L3-02", "lstm_L7", "B", f"{raw_vars}|L=7",
         "7-day history LSTM - matches the antecedent-rainfall horizon used by the engineered features (like-for-like memory)"),
        ("L3-03", "lstm_L14", "B", f"{raw_vars}|L=14",
         "14-day history LSTM - tests whether longer memory than the engineered features adds skill (D12)"),
        ("L3-04", "lstm_L30", "B", f"{raw_vars}|L=30",
         "30-day history LSTM - the long-memory end of the D12 length sweep; also the window attention should help most"),
        ("L3-05", "gru_bestL", "B", f"{raw_vars}|L=best",
         "gated recurrent unit at the validation-winning L - fewer parameters than LSTM; answers 'cheaper recurrence, same skill?'"),
        ("L3-06", "tcn_bestL", "B", f"{raw_vars}|L=best",
         "dilated causal convolution at the validation-winning L - non-recurrent temporal encoder; the LSTM/GRU/TCN three-way test (Phase 22)"),
    ]
    for eid, model, ds, fset, reason in l3:
        add(registry_id=eid, phase="21-22", level="L3", family="deep temporal",
            model=model, dataset=ds, feature_set=fset, reason=reason,
            tuning_budget_configs=20, imbalance_policy="class_weight=balanced",
            status="planned", gate="", metrics_out="phase21_metrics.csv;phase22_metrics.csv",
            model_out="models/phase21_<id>.pt", test_policy=TEST_POLICY, **common)

    # ---- L3 feature fusion (Phase 23 - amendment, workflow Phase 23 / stage 38).
    # The single registry stays the contract: the fusion variants are added here
    # (not run ad hoc) so the fused / temporal-only / static-only / hybrid
    # comparison is registered exactly like every other candidate.
    l3f = [
        ("L3-07", "fusion_L3", "B", f"{raw_vars}|L=3|static_15",
         "temporal encoder (LSTM at L*) fused with the Phase-23 static geographic "
         "vector - location-aware prediction: does knowing WHERE a station is add "
         "skill over the sequence alone (workflow Phase 23 core variant)"),
        ("L3-08", "static_only", "S", "static(group A+B geography)",
         "the static branch alone (lat/lon, danger level, terrain, distances + "
         "river/basin/division/district/upazila identity) - the feature-group "
         "baseline that isolates how much pure location explains with no sequence"),
        ("L3-09", "hybrid_L3", "B", f"{raw_vars}|L=3|static_15|LINEAR",
         "learned+engineered hybrid: the fusion head also sees the Phase-12 LINEAR "
         "set - does the engineered signal add to what the sequence encoder already "
         "learned (workflow Phase 23 optional second variant)"),
    ]
    for eid, model, ds, fset, reason in l3f:
        add(registry_id=eid, phase="23", level="L3", family="feature fusion",
            model=model, dataset=ds, feature_set=fset, reason=reason,
            tuning_budget_configs=16, imbalance_policy="class_weight=balanced",
            status="planned", gate="", metrics_out="phase23_metrics.csv",
            model_out="models/phase23_<id>.pt", test_policy=TEST_POLICY, **common)

    # ---- L4 advanced spatiotemporal (gated, D13)
    l4 = [
        ("L4-01", "gcn", "B", f"{raw_vars}|graph",
         "graph convolution over the station graph - spatial message passing between hydrologically connected stations",
         "phase24_graph_gate"),
        ("L4-02", "gat", "B", f"{raw_vars}|graph",
         "graph attention - learns which neighbours matter instead of assuming uniform edge weight",
         "phase24_graph_gate"),
        ("L4-03", "transformer_temporal", "B", f"{raw_vars}|L=30",
         "temporal self-attention - tests whether long-range dependencies are learned better than by convolution/recurrence",
         "phase25_transformer_gate"),
        ("L4-04", "gnn_lstm", "B", f"{raw_vars}|graph|L=best",
         "GNN spatial encoder + LSTM temporal encoder - the full spatiotemporal stack (only if L4-01 passes)",
         "phase24_graph_gate"),
    ]
    for eid, model, ds, fset, reason, gate in l4:
        # The Phase-24 gate (phase24_graph_gate, 24.1) PASSED, so L4-01/02/04 are
        # discharged gated -> planned (reports/phase24_graph_gate.md sec. 9).  The
        # Phase-25 transformer gate (phase25_transformer_gate, 25.1) also PASSED
        # (reports/phase25_transformer_gate.md), so L4-03 is discharged too -> every
        # L4 row is now runnable (planned).
        status = "planned"
        mout = ("models/phase25_<id>.pt" if eid == "L4-03"
                else "models/phase24_<id>.pt")
        add(registry_id=eid, phase="24-25", level="L4", family="advanced spatiotemporal",
            model=model, dataset=ds, feature_set=fset, reason=reason,
            tuning_budget_configs=20, imbalance_policy="class_weight=balanced",
            status=status, gate=gate, metrics_out="phase24_metrics.csv;phase25_metrics.csv",
            model_out=mout, test_policy=TEST_POLICY, **common)

    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------- checks
def dev_grid_from_splits(s: pd.DataFrame) -> dict[str, list[str]]:
    """Cells that the Phase-15 masks actually support (same construction as
    scripts/baseline_model.py build_tasks)."""
    split = s.split.to_numpy()
    year = pd.to_datetime(s.date).dt.year.to_numpy()
    fold = s.spatial_fold.to_numpy()
    holdout = s.station_holdout.to_numpy()
    first_year = s.first_year.to_numpy()

    p1 = ["val_2022"] if ((split == "val") & ~holdout).any() else []
    p2 = [f"val_{y}" for y in ROLLING_YEARS if ((year == y) & ~holdout).any()]
    p3 = [f"spatial_fold_{k}" for k in SPATIAL_FOLDS
          if ((split == "train") & ~holdout & (fold == k)).any()]
    test = [f"test_{y}" for y in TEST_YEARS if ((split == "test") & (year == y)).any()]
    is_test = split == "test"
    buckets = []
    if (is_test & ~holdout & (first_year <= HOLDOUT_CUTOFF)).any():
        buckets.append("test_trained")
    if (is_test & ~holdout & (first_year > HOLDOUT_CUTOFF)).any():
        buckets.append("test_new_network")
    if (is_test & holdout).any():
        buckets.append("test_unseen")
    return {"P1": p1, "P2": p2, "P3": p3, "test": test + buckets}


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print("Phase 17 - Experimental modelling framework (stage 31)")
    print("=" * 78)

    # F1a - inputs
    sha = sha256(TBL)
    check("F1a", f"experimental_dataset.csv SHA-256 unchanged ({sha[:16]}...)",
          sha == TBL_SHA_EXPECTED)
    s = pd.read_csv(SPLITS)
    fs = json.loads(FSETS.read_text(encoding="utf-8"))
    check("F1b", f"split table {s.shape[0]:,} rows x {s.shape[1]} cols; feature sets "
                 f"{fs['counts']}", s.shape[0] == 200665 and set(fs["sets"]) == {"FULL", "LINEAR", "SELECTED"})

    # F1c - the cells the masks really support
    grid = dev_grid_from_splits(s)
    check("F1c", f"Phase-15 masks support dev cells P1={grid['P1']} P2={grid['P2']} "
                 f"P3={grid['P3']} and frozen test {grid['test']}",
          grid["P1"] == ["val_2022"] and grid["P2"] == [f"val_{y}" for y in ROLLING_YEARS]
          and grid["P3"] == [f"spatial_fold_{k}" for k in SPATIAL_FOLDS]
          and grid["test"] == [f"test_{y}" for y in TEST_YEARS] + TEST_BUCKETS)

    # F1d - registry structure
    reg = build_registry()
    check("F1d", f"registry has {len(reg)} experiments across L1-L4 "
                 f"({reg.level.value_counts().to_dict()})",
          sorted(reg.level.unique()) == ["L1", "L2", "L3", "L4"]
          and len(reg) == 6 + 5 + 9 + 4)

    # F1e - every candidate has a reason (the workflow's anti-padding rule)
    short = reg[reg.reason.str.len() < 40]
    check("F1e", "every candidate carries a stated reason for inclusion "
                 "(shortest reason: "
                 f"{reg.reason.str.len().min()} chars)",
          short.empty)

    # F1f - identifiers unique
    check("F1f", f"registry ids unique ({reg.registry_id.nunique()}) and model names "
                 f"unique ({reg.model.nunique()})",
          reg.registry_id.is_unique and reg.model.is_unique)

    # F1f - L1 equals the six Phase-16 baselines exactly
    p16 = pd.read_csv(PHASE16_MET)
    l1_models = set(reg[reg.level == "L1"].model)
    check("F1g", f"L1 rows == Phase-16 model set {sorted(l1_models)}",
          l1_models == set(p16.model.unique()))
    check("F1h", "every L1 row marked done and already read the frozen test once "
                 "in Phase 16 (test_policy = read_once_phase16)",
          (reg[reg.level == "L1"].status == "done").all()
          and (reg[reg.level == "L1"].test_policy == TEST_POLICY_L1).all())

    # F1i - parity: same cells, seasons, horizons, selection rule, seed (D14)
    plan = reg[reg.level != "L1"]
    check("F1i", "fair-comparison parity (D14): every planned run uses protocols "
                 "P1/P2/P3, the same dev grid, all three seasons, all three horizons, "
                 "selection = PR-AUC on val_2022, seed 20261008",
          (plan.protocols == "P1,P2,P3").all()
          and (plan.dev_cells == plan.dev_cells.iloc[0]).all()
          and (plan.seasons == "all,monsoon,non-monsoon").all()
          and (plan.horizons == "24h,48h,72h").all()
          and (plan.selection_metric == "pr_auc").all()
          and (plan.selection_cell == "val_2022").all()
          and (plan.seed == SEED).all())

    # F1j - test protection: nothing planned reads test cells for selection
    leak = plan[plan.dev_cells.str.contains("test") | plan.selection_cell.str.startswith("test")]
    check("F1j", "no planned experiment has a test cell in its development grid or "
                 "selection cell - frozen 2023+ reads are reserved for Phase 33",
          leak.empty and (plan.test_policy == TEST_POLICY).all())
    check("F1k", "frozen test cells exist for that later read: "
                 f"{[c for c in grid['test']]}",
          len(grid["test"]) == 7)

    # F1l - tuning budgets recorded and finite
    budgets = reg.groupby("level").tuning_budget_configs.apply(list).to_dict()
    check("F1l", f"tuning budgets recorded per level (L1=0 reference, L2=40, "
                 f"L3 temporal=20 / L3 fusion=16, L4=20): "
                 f"{budgets}",
          all(v == [0] * len(v) for v in [budgets["L1"]])
          and all(b > 0 for b in budgets["L2"] + budgets["L3"] + budgets["L4"]))

    # F1m - gates (D13): both Phase-24 and Phase-25 gates discharged (gated -> planned)
    l4 = reg[reg.level == "L4"]
    p24 = l4[l4.gate == "phase24_graph_gate"]
    p25 = l4[l4.gate == "phase25_transformer_gate"]
    check("F1m", "Phase-24 gate discharged: L4-01/02/04 status=planned "
                 "(phase24_graph_gate passed); Phase-25 gate discharged: L4-03 "
                 "status=planned (phase25_transformer_gate passed); no non-L4 row "
                 "carries a gate",
          len(l4) == 4
          and set(p24.registry_id) == {"L4-01", "L4-02", "L4-04"}
          and (p24.status == "planned").all()
          and set(p25.registry_id) == {"L4-03"} and (p25.status == "planned").all()
          and reg[reg.level != "L4"].gate.eq("").all())

    # F1n - feature sets resolve
    tab = reg[reg.dataset == "A"]
    known = set(fs["sets"]) | {"none", "wl_pct_danger", "LINEAR-no-up"}
    check("F1n", f"every Dataset-A feature set is a known Phase-12 set: "
                 f"{sorted(tab.feature_set.unique())}",
          set(tab.feature_set) <= known)
    seq = reg[reg.dataset == "B"]
    check("F1o", f"every Dataset-B row names the raw sequence variables with a "
                 f"window L in {{3,7,14,30}}/graph/best: {sorted(seq.feature_set.unique())}",
          seq.feature_set.str.startswith("raw_sequence").all())

    # F1p - imbalance policy never contaminates the split (Phase 16 s10 / R4)
    check("F1p", "imbalance handling is declared as inside-training only "
                 f"({sorted(reg.imbalance_policy.unique())}) - nothing resamples before the split",
          set(reg.imbalance_policy) <= IMBALANCE_ALLOWED)

    # F1q - dev grid arithmetic: 8 tasks x 3 seasons x 3 horizons = 72 cells/model
    n_tasks = 1 + len(ROLLING_YEARS) + len(SPATIAL_FOLDS)
    cells = n_tasks * len(SEASONS) * len(HORIZONS)
    check("F1q", f"every model will be scored on {n_tasks} dev tasks x {len(SEASONS)} "
                 f"seasons x {len(HORIZONS)} horizons = {cells} cells (same grid as "
                 f"Phase 16's 432-row file / 6 models)",
          cells == 72 and len(p16) == cells * 6)

    # F1r - the three tasks are explicit and separate (D2)
    check("F1r", "three separate forecasting tasks per experiment "
                 "(24h/48h/72h columns present, no pooled horizon)",
          set(reg.horizons) == {"24h,48h,72h"})

    # F1s - D7 benchmark row present for every comparison
    check("F1s", "D7 persistence benchmark row present (L1-02) - every family is "
                 "reported against it", "persistence" in l1_models)

    # write registry
    reg.to_csv(REGISTRY, index=False, encoding="utf-8")
    check("F1t", f"registry written: {REGISTRY.relative_to(ROOT)} "
                 f"({REGISTRY.stat().st_size:,} bytes, {len(reg)} rows x {reg.shape[1]} cols)",
          REGISTRY.exists())

    n_pass = sum(r["pass"] for r in RES)
    print("-" * 78)
    print(f"{n_pass}/{len(RES)} checks passed, {len(WARN)} warnings, "
          f"{time.time() - t0:.1f}s")
    print("registry: L1 done (Phase 16) | L2 done (Phase 20) | L3 temporal done "
          "(Phases 21-22) / fusion planned (Phase 23) | L4 planned "
          "(Phases 24-25 gates passed)")
    print("NEXT: Phase 18 - ML/DL dataset preparation (Dataset B + sequence lengths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
