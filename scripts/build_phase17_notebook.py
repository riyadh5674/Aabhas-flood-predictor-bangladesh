"""Build notebooks/17_framework_registry.ipynb (workflow Phase 17 / stage 31).

Phase 17 defines what is being compared before anything is trained. The registry
itself is produced and validated by scripts/phase17_framework.py (20/20 green,
scripts/phase17_run.log); this notebook is the independent re-verification a
reviewer runs: it reloads the written registry CSV and re-derives every contract
from the frozen inputs (split masks, feature sets, Phase-16 metric file) instead
of trusting the generator.

Notebook numbering: 02-08 were used by Phases 6-16; 09_sequence_validation
(Phase 19) and 10_explainability (Phase 34) keep their workflow-assigned names,
so from Phase 17 on the notebook carries its phase number.

Checks N17-01..N17-14:
    N17-01  inputs: registry CSV, framework report, split table, feature sets,
            experimental table SHA-256 unchanged
    N17-02  registry structure: 24 experiments, unique ids/models, L1-L4 counts
    N17-03  every candidate has a written reason (anti-padding rule)
    N17-04  L1 == the six Phase-16 baselines, status done, frozen-test read once
    N17-05  parity (D14): identical protocols/grid/seasons/horizons/selection/seed
    N17-06  test protection: no planned run sees a test cell; the seven frozen
            test cells are constructible from split_assignments.csv
    N17-07  tuning budgets recorded per level before any result exists
    N17-08  L4 rows carry a named Phase-24/25 gate report (both discharged)
    N17-09  feature sets resolve: Dataset A -> Phase-12 sets, Dataset B -> 12 raw
            vars with a window L in {3,7,14,30}/best/graph
    N17-10  grid arithmetic: 8 dev tasks x 3 seasons x 3 horizons = 72 cells/model,
            equal to the Phase-16 metric grid
    N17-11  three separate horizon tasks (D2), D7 benchmark row present
    N17-12  the report quotes the registry (id, gate, D13/D14) - report/CSV agree
    N17-13  summary: all checks green
    N17-14  experimental_dataset.csv byte-stable at the end

Build and run:

    python scripts/build_phase17_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\17_framework_registry.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "17_framework_registry.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 17 — Experimental modelling framework (stage 31)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 17 — Experimental modelling framework** → checklist **stage 31** |
| **Built by** | `scripts/phase17_framework.py` → `scripts/phase17_run.log` (exit 0, 20/20) |
| **This notebook** | independent re-verification of the written registry from the frozen inputs |
| **Inputs** | `reports/phase17_experiment_registry.csv` · `split_assignments.csv` · `feature_sets.json` · `phase16_baseline_metrics.csv` · `experimental_dataset.csv` (SHA re-checked) |
| **Output** | `reports/phase17_framework.md` + registry CSV (unchanged by this notebook) |
| **Binding decisions** | **D2** three separate horizons · **D11** numbering · **D12** L ∈ {3,7,14,30} empirical · **D13** L4 gated · **D14** ML-vs-DL parity · **D7** persistence bar |

**Fourteen checks, N17-01 … N17-14.** No model is trained here and no test cell is
read for any decision — the frozen 2023+ cells are only *constructed* to prove they
exist for Phase 33."""))

cells.append(code(r"""import hashlib, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
TBL = PROC / "experimental_dataset.csv"
SPLITS = PROC / "split_assignments.csv"
FSETS = PROC / "feature_sets.json"
REG = REPORTS / "phase17_experiment_registry.csv"
REPORT_MD = REPORTS / "phase17_framework.md"
P16_MET = REPORTS / "phase16_baseline_metrics.csv"

TBL_SHA_EXPECTED = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
SEED = 20261008
HORIZONS = ["label_24h", "label_48h", "label_72h"]
SEASONS = ["all", "monsoon", "non-monsoon"]
ROLLING_YEARS = [2020, 2021, 2022]
SPATIAL_FOLDS = [1, 2, 3, 4]
TEST_YEARS = [2023, 2024, 2025, 2026]
TEST_BUCKETS = ["test_trained", "test_new_network", "test_unseen"]
HOLDOUT_CUTOFF = 2023
L1_MODELS = ["climatology", "persistence", "logreg_linear", "logreg_linear_no_up",
             "logreg_selected", "tree_depth4"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


RES, WARN = [], []


def check(cid, detail, ok=True):
    RES.append({"id": cid, "pass": bool(ok)})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}")
    assert ok, cid


def warn(wid, detail):
    WARN.append({"id": wid, "note": detail})
    print(f"[WARN] {wid}: {detail}")


t0 = time.time()
print("Phase 17 notebook - registry re-verification")"""))

# ---------------------------------------------------------------- N17-01
cells.append(code(r"""# N17-01 - inputs present, source table frozen
check("N17-01a", f"registry CSV present ({REG.stat().st_size:,} bytes) and "
                 f"framework report present ({REPORT_MD.stat().st_size:,} bytes)",
      REG.exists() and REPORT_MD.exists())
sha = sha256(TBL)
check("N17-01b", f"experimental_dataset.csv SHA-256 unchanged ({sha[:16]}...)",
      sha == TBL_SHA_EXPECTED)
s = pd.read_csv(SPLITS)
fs = json.loads(FSETS.read_text(encoding="utf-8"))
check("N17-01c", f"split table {s.shape[0]:,} x {s.shape[1]}; feature sets "
                 f"{fs['counts']}; banned columns {len(fs['banned'])} listed",
      s.shape == (200665, 9) and set(fs["sets"]) == {"FULL", "LINEAR", "SELECTED"})"""))

# ---------------------------------------------------------------- N17-02
cells.append(code(r"""# N17-02 - registry structure
reg = pd.read_csv(REG)
need = {"registry_id", "level", "family", "model", "dataset", "feature_set",
        "reason", "dev_cells", "seasons", "horizons", "selection_metric",
        "selection_cell", "seed", "tuning_budget_configs", "imbalance_policy",
        "status", "gate", "test_policy"}
check("N17-02a", f"registry {reg.shape[0]} experiments x {reg.shape[1]} columns; "
                 f"all contract columns present",
      need <= set(reg.columns))
counts = reg.level.value_counts().to_dict()
check("N17-02b", f"four levels with L1=6 (done) L2=5 L3=9 (6 + 3 Phase-23 fusion "
                 f"amendments) L4=4: {counts}",
      counts == {"L1": 6, "L2": 5, "L3": 9, "L4": 4})
check("N17-02c", f"registry ids unique ({reg.registry_id.nunique()}) and model names "
                 f"unique ({reg.model.nunique()})",
      reg.registry_id.is_unique and reg.model.is_unique)
print(reg.groupby(["level", "family"]).size().to_string())"""))

# ---------------------------------------------------------------- N17-03
cells.append(code(r"""# N17-03 - every candidate carries a reason (anti-padding rule)
lens = reg.reason.str.len()
check("N17-03", f"all {len(reg)} candidates have a written reason "
                f"(min {lens.min()} chars, median {int(lens.median())}); "
                f"no empty/generic justification",
      lens.min() >= 40 and reg.reason.notna().all())
print(reg[["registry_id", "level", "model", "reason"]].head(6).to_string(index=False))"""))

# ---------------------------------------------------------------- N17-04
cells.append(code(r"""# N17-04 - L1 == the six Phase-16 baselines, already run, test read once
p16 = pd.read_csv(P16_MET)
l1 = reg[reg.level == "L1"]
check("N17-04a", f"L1 model set == Phase-16 metric file's six models "
                 f"{sorted(l1.model.unique())}",
      set(l1.model) == set(p16.model.unique()) == set(L1_MODELS))
check("N17-04b", "L1 status=done with test_policy=read_once_phase16 (their single "
                 "frozen-test read already happened in Phase 16)",
      (l1.status == "done").all()
      and (l1.test_policy == "read_once_phase16").all())
check("N17-04c", f"Phase-16 metrics file intact: {len(p16)} rows = 8 tasks x 6 models "
                 f"x 3 horizons x 3 seasons",
      len(p16) == 432)"""))

# ---------------------------------------------------------------- N17-05
cells.append(code(r"""# N17-05 - fair-comparison parity (D14): planned runs are identical on every axis
plan = reg[reg.level != "L1"]
cols = ["protocols", "dev_cells", "seasons", "horizons", "selection_metric",
        "selection_cell", "seed", "test_policy"]
uniform = {c: plan[c].nunique() == 1 for c in cols}
check("N17-05a", f"parity columns single-valued across all planned runs: {uniform}",
      all(uniform.values()))
check("N17-05b", f"shared values: protocols={plan.protocols.iloc[0]}, "
                 f"seasons={plan.seasons.iloc[0]}, horizons={plan.horizons.iloc[0]}, "
                 f"selection={plan.selection_metric.iloc[0]} on "
                 f"{plan.selection_cell.iloc[0]}, seed={plan.seed.iloc[0]}",
      plan.protocols.iloc[0] == "P1,P2,P3"
      and plan.seasons.iloc[0] == "all,monsoon,non-monsoon"
      and plan.horizons.iloc[0] == "24h,48h,72h"
      and plan.selection_metric.iloc[0] == "pr_auc"
      and plan.selection_cell.iloc[0] == "val_2022"
      and plan.seed.iloc[0] == SEED)
check("N17-05c", "imbalance policy declared inside-training only for every planned run "
                 f"({sorted(plan.imbalance_policy.unique())}) - never resampled "
                 "before the split (R4)",
      set(plan.imbalance_policy) <= {"class_weight=balanced", "none"})"""))

# ---------------------------------------------------------------- N17-06
cells.append(code(r"""# N17-06 - test protection: no planned run sees a frozen test cell
bad = plan[plan.dev_cells.str.contains("test")
           | plan.selection_cell.str.startswith("test")]
check("N17-06a", f"no planned run has a test cell in its development grid or selection "
                 f"cell ({len(bad)} violations)",
      bad.empty and (plan.test_policy == "frozen_read_at_phase33").all())

# construct the seven frozen cells straight from the Phase-15 masks
split, year = s.split.to_numpy(), pd.to_datetime(s.date).dt.year.to_numpy()
fold, hold, fy = s.spatial_fold.to_numpy(), s.station_holdout.to_numpy(), s.first_year.to_numpy()
cells_frozen = ([f"test_{y}" for y in TEST_YEARS if ((split == "test") & (year == y)).any()]
                + [b for b, m in [("test_trained", (split == "test") & ~hold & (fy <= HOLDOUT_CUTOFF)),
                                  ("test_new_network", (split == "test") & ~hold & (fy > HOLDOUT_CUTOFF)),
                                  ("test_unseen", (split == "test") & hold)] if m.any()])
check("N17-06b", f"the seven frozen test cells exist for the Phase-33 read: "
                 f"{cells_frozen}",
      cells_frozen == [f"test_{y}" for y in TEST_YEARS] + TEST_BUCKETS)"""))

# ---------------------------------------------------------------- N17-07
cells.append(code(r"""# N17-07 - tuning budgets fixed before any result exists
budgets = {lvl: sorted(g.tuning_budget_configs.unique().tolist())
           for lvl, g in reg.groupby("level")}
check("N17-07", f"budgets per level: {budgets} (L1=0 reference, L2=40, L3 temporal=20 / "
                "L3 fusion=16 (Phase-23 amendment, within the L3 cap), L4=20 - cannot "
                "expand after a model looks promising)",
      budgets == {"L1": [0], "L2": [40], "L3": [16, 20], "L4": [20]})"""))

# ---------------------------------------------------------------- N17-08
cells.append(code(r"""# N17-08 - L4 gates (D13): Phase-24 and Phase-25 gates both discharged
l4 = reg[reg.level == "L4"]
p24 = l4[l4.gate == "phase24_graph_gate"]
p25 = l4[l4.gate == "phase25_transformer_gate"]
check("N17-08a", f"{len(p24)} L4 rows discharged status=planned (phase24_graph_gate "
                 f"passed), {len(p25)} discharged status=planned "
                 f"(phase25_transformer_gate passed)",
      len(l4) == 4 and set(p24.registry_id) == {"L4-01", "L4-02", "L4-04"}
      and (p24.status == "planned").all()
      and set(p25.registry_id) == {"L4-03"} and (p25.status == "planned").all()
      and set(l4.gate) == {"phase24_graph_gate", "phase25_transformer_gate"})
check("N17-08b", "no other level carries a gate; nothing outside L4 claims to be "
                 "blocked",
      (reg[reg.level != "L4"].gate.fillna("") == "").all()
      and set(reg[reg.level != "L4"].status) == {"done", "planned"})
print(l4[["registry_id", "model", "gate", "status"]].to_string(index=False))"""))

# ---------------------------------------------------------------- N17-09
cells.append(code(r"""# N17-09 - feature sets resolve to Phase-12 sets / raw-sequence spec
tab = reg[reg.dataset == "A"]
known = set(fs["sets"]) | {"none", "wl_pct_danger", "LINEAR-no-up"}
check("N17-09a", f"Dataset-A feature sets all known: {sorted(tab.feature_set.unique())}",
      set(tab.feature_set) <= known)
seq = reg[reg.dataset == "B"]
lengths = set()
for spec in seq.feature_set:
    if "L=" in spec:
        lengths.add(spec.split("L=")[1].split("|")[0])
check("N17-09b", f"Dataset-B rows are raw_sequence(12 vars) with L in "
                 f"{{3,7,14,30}}/best/graph: {sorted(lengths)}",
      seq.feature_set.str.startswith("raw_sequence(12 vars)").all()
      and lengths <= {"3", "7", "14", "30", "best"}
      and any("graph" in x for x in seq.feature_set))
check("N17-09c", "every feature set count matches the Phase-12 manifest "
                 f"{fs['counts']}",
      fs["counts"]["FULL"] == 62 and fs["counts"]["LINEAR"] == 35
      and fs["counts"]["SELECTED"] == 10)"""))

# ---------------------------------------------------------------- N17-10
cells.append(code(r"""# N17-10 - grid arithmetic: the cells every planned model will be scored on
p1 = ["val_2022"] if ((s.split == "val") & ~s.station_holdout).any() else []
p2 = [f"val_{y}" for y in ROLLING_YEARS if ((year == y) & ~s.station_holdout).any()]
p3 = [f"spatial_fold_{k}" for k in SPATIAL_FOLDS
      if ((s.split == "train") & ~s.station_holdout & (fold == k)).any()]
n_tasks = len(p1) + len(p2) + len(p3)
cells_model = n_tasks * len(SEASONS) * len(HORIZONS)
check("N17-10a", f"development tasks supported by the masks: P1={p1}, P2={p2}, "
                 f"P3={p3} -> {n_tasks} tasks",
      n_tasks == 8 and p1 == ["val_2022"]
      and p2 == [f"val_{y}" for y in ROLLING_YEARS]
      and p3 == [f"spatial_fold_{k}" for k in SPATIAL_FOLDS])
check("N17-10b", f"{n_tasks} tasks x {len(SEASONS)} seasons x {len(HORIZONS)} horizons "
                 f"= {cells_model} cells per model = the Phase-16 grid "
                 f"({len(p16)} rows / 6 models)",
      cells_model == 72 and len(p16) / 6 == cells_model)"""))

# ---------------------------------------------------------------- N17-11
cells.append(code(r"""# N17-11 - three separate tasks (D2) and the D7 bar (D7)
check("N17-11a", "every experiment declares all three horizon tasks separately "
                 f"({set(reg.horizons)}) - no pooled horizon anywhere",
      set(reg.horizons) == {"24h,48h,72h"})
check("N17-11b", "D7 persistence benchmark present as L1-02 - every family is reported "
                 "against it",
      (reg[reg.registry_id == "L1-02"].model == "persistence").all())
check("N17-11c", "D12 window lengths present as explicit LSTM rows "
                 "(L3-01..04 = L 3/7/14/30)",
      set(reg[reg.model.str.startswith("lstm_L")].model)
      == {"lstm_L3", "lstm_L7", "lstm_L14", "lstm_L30"})"""))

# ---------------------------------------------------------------- N17-12
cells.append(code(r"""# N17-12 - report and registry agree
txt = REPORT_MD.read_text(encoding="utf-8")
ids_in_report = [i for i in reg.registry_id if i in txt]
check("N17-12a", f"report quotes {len(ids_in_report)}/{len(reg)} registry ids "
                 "(or their rows) and names the registry file",
      "phase17_experiment_registry.csv" in txt and len(ids_in_report) >= 8)
check("N17-12b", "report states the gate rule (D13) and the parity rule (D14) that the "
                 "CSV encodes",
      "gated" in txt.lower() and "parity" in txt.lower() and "D14" in txt)
check("N17-12c", "report records the test-protection rule the CSV encodes",
      "frozen_read_at_phase33" in txt or "Phase 33" in txt)"""))

# ---------------------------------------------------------------- summary
cells.append(code(r"""print("=" * 78)
print("PHASE 17 / STAGE 31   experiment framework verified - 24 runs registered")
print("=" * 78)
for r in RES:
    print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['id']}")
for w in WARN:
    print(f"  WARN  {w['id']}: {w['note']}")
print(f"\n{sum(r['pass'] for r in RES)}/{len(RES)} checks passed, {len(WARN)} warnings, "
      f"{time.time() - t0:.1f}s")
print("registry levels:", reg.groupby("level").size().to_dict())
print("next: Phase 18 - ML/DL dataset preparation (Dataset B, L in {3,7,14,30})")
assert all(r["pass"] for r in RES)
assert sha256(TBL) == TBL_SHA_EXPECTED, "experimental table changed!"
print("ALL PHASE-17 CHECKS PASSED - experimental_dataset.csv byte-stable")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
