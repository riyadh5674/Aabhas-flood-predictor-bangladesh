"""Build notebooks/06_experimental_dataset.ipynb (workflow Phase 14).

Phase 14 has no numbered body steps - the workflow asks for the final modelling table:

    Location / Timestamp / Predictor 1..N / Target / Prediction horizon
    Keep a copy of the raw dataset separately. Never overwrite the original.

The notebook therefore *defines and proves* the table contract:

    E1  provenance: source table, Phase-13 whitelist and Phase-12 sets agree
    E2  schema contract: Location + Timestamp + 62 predictors (FULL) + 3 targets
    E3  the 6 columns that leave, each with the banned/constant reason
    E4  row + key contract: 200,665 unique station-days, 133 stations, date range
    E5  dtype contract: 5 text predictors, 57 numeric predictors, 3 float targets
    E6  missingness profile of the final table (structural vs gap-driven NaN)
    E7  target contract: counts/base rates re-checked against phase0 section 4
    E8  write + exact round-trip of data/processed/experimental_dataset.csv
    E9  immutability: the source file's SHA-256 is unchanged after the run
    E10 data dictionary (experimental_dataset_dictionary.csv) + preview

Build and run:

    python scripts/build_phase14_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\06_experimental_dataset.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "06_experimental_dataset.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 14 - Define the experimental dataset

| | |
| --- | --- |
| **Workflow** | `flood prediction workflow.md` **Phase 14 - Define the experimental dataset**: *create the final modelling table (Location / Timestamp / Predictor 1..N / Target / Prediction horizon). Keep a copy of the raw dataset separately. Never overwrite the original.* |
| **Checklist #** | 26 (experimental / model-ready table) |
| **Input** | `data/processed/flood_dataset_model_ready.csv` (200,665 x 73) + `feature_whitelist.csv` (Phase 13) + `feature_sets.json` (Phase 12) |
| **Boundary** | `reports/phase0_research_definition.md` §5 - issue instant = 24:00 UTC of day `t` |
| **Outputs** | `data/processed/experimental_dataset.csv` (200,665 x 67) · `data/processed/experimental_dataset_dictionary.csv` |
| **Source rule** | `flood_dataset_model_ready.csv` is read-only: SHA-256 before == after (E9) |

Ten checks, **E1-E10**. This table is what Phases 15-16 split and train on; it is written
by *selecting columns*, never by recomputing them, so every value is byte-identical to the
audited source."""))

cells.append(code(r"""import hashlib, json, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
SRC = PROC / "flood_dataset_model_ready.csv"
HORIZONS = ["label_24h", "label_48h", "label_72h"]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

SRC_SHA = sha256(SRC)
mr = pd.read_csv(SRC)
mr["date"] = pd.to_datetime(mr["date"])
WL = pd.read_csv(PROC / "feature_whitelist.csv")
sets = json.loads((PROC / "feature_sets.json").read_text(encoding="utf-8"))
FULL = list(sets["sets"]["FULL"])
BANNED = list(sets["banned"])

RES = []
def check(cid, detail, ok=True):
    RES.append({"id": cid, "pass": bool(ok)})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}")
    assert ok, cid

print(f"source {mr.shape} | whitelist {WL.shape} | Phase-12 FULL {len(FULL)} | "
      f"source sha256 {SRC_SHA[:16]}...")"""))

cells.append(md("""## What the workflow asks for

> *At this point create the **final modelling table**. Conceptually: Location, Timestamp,
> Predictor 1 ... Predictor N, Target, Prediction horizon. Keep a copy of the raw dataset
> separately. Never overwrite the original.* (workflow **Phase 14**)

So the table has exactly four column blocks:

| block | columns | count |
| --- | --- | --- |
| **Location** | `station_id` (join key for `lat`/`lon` and for spatial validation in Phase 15 step 26) | 1 |
| **Timestamp** | `date` (the day `t` the prediction is issued for) | 1 |
| **Predictors** | the Phase-12 `FULL` set = the Phase-13 whitelist's `ADMIT` + `ADMIT-WATCH` columns | 62 |
| **Target** | `label_24h` `label_48h` `label_72h` (the three prediction horizons, one column each) | 3 |

Everything else in the 73-column source is **deliberately excluded**: the 10 banned
columns minus the 3 targets (`name`, `source`, `obs_days_24h/48h/72h` are identity or
label-window information), plus the 1 constant that Phase 12 dropped (`day_observed`).
No value is recalculated - only columns are selected, so the audit of Phase 13 still holds
for every cell of this table."""))

cells.append(code(r"""# E1 - provenance: the three artifacts that define the table agree with each other
assert mr.shape == (200665, 73), mr.shape
assert WL.shape[0] == 73 and set(WL["column"]) == set(mr.columns)
counts = WL["ruling"].value_counts().to_dict()
assert counts == {"ADMIT": 61, "BAN": 10, "ADMIT-WATCH": 1, "ADMIT-CONSTANT": 1}, counts
admit = set(WL.loc[WL["ruling"].isin(["ADMIT", "ADMIT-WATCH"]), "column"])
assert admit == set(FULL), admit ^ set(FULL)
assert set(BANNED) == {"station_id", "name", "date", "source",
                       "label_24h", "label_48h", "label_72h",
                       "obs_days_24h", "obs_days_48h", "obs_days_72h"}
assert "phase13" in sets and "phase12" not in sets  # whitelist ruling recorded downstream
assert set(FULL) <= set(mr.columns) and set(HORIZONS) <= set(mr.columns)

check("E1 provenance",
      f"source {mr.shape} + whitelist 73 rows {counts} + Phase-12 FULL {len(FULL)} are "
      f"mutually consistent; the 10 banned columns and the 3 targets are all present, "
      f"so every exclusion below is a choice")"""))

cells.append(code(r"""# E2 - schema contract: Location | Timestamp | 62 predictors | 3 targets
KEYS = ["station_id", "date"]
PRED = list(FULL)
TARGETS = list(HORIZONS)
ORDER = KEYS + PRED + TARGETS

assert ORDER[:2] == KEYS and ORDER[-3:] == TARGETS
assert len(ORDER) == 67 and len(set(ORDER)) == 67
assert set(PRED) == admit                      # ADMIT + ADMIT-WATCH, nothing else
assert not (set(PRED) & set(BANNED))           # banned columns never re-enter as features
assert set(TARGETS) == set(BANNED) & set(HORIZONS)
assert "day_observed" not in PRED              # ADMIT-CONSTANT, dropped at Phase 12 step 18a

check("E2 schema contract",
      f"67 columns = 1 location + 1 timestamp + {len(PRED)} predictors (FULL = whitelist "
      f"ADMIT 61 + ADMIT-WATCH 1) + {len(TARGETS)} targets; 0 banned columns among the "
      f"predictors")"""))

cells.append(code(r"""# E3 - exactly 6 columns leave the source, each for a stated reason
REMOVED = {
    "name": "identity/date key (banned); redundant with station_id",
    "source": "identity/date key (banned)",
    "obs_days_24h": "label-window information (banned) - label availability is visible "
                    "directly from the target being NaN",
    "obs_days_48h": "label-window information (banned)",
    "obs_days_72h": "label-window information (banned)",
    "day_observed": "ADMIT-CONSTANT (constant in the model-ready table, dropped at "
                    "Phase 12 step 18a)",
}
assert set(mr.columns) - set(ORDER) == set(REMOVED), set(mr.columns) ^ set(ORDER)
assert len(REMOVED) == 6 and 73 - 6 == 67

# the table is a pure column selection: every retained cell is byte-identical to the source
EXPL = mr[ORDER].copy()
assert EXPL.shape == (200665, 67)
assert all(EXPL[c].equals(mr[c]) for c in ORDER)

check("E3 column selection",
      "6 columns removed (" + ", ".join(REMOVED) + "); all 67 retained columns are "
      "identical to the source cells - no recomputation, so the Phase-13 audit carries over")"""))

cells.append(code(r"""# E4 - row and key contract
assert len(EXPL) == 200665
assert EXPL.duplicated(KEYS).sum() == 0
n_st = EXPL["station_id"].nunique()
assert n_st == 133, n_st
dmin, dmax = EXPL["date"].min(), EXPL["date"].max()
assert dmin == pd.Timestamp("2017-01-06") and dmax == pd.Timestamp("2026-10-05")

# strictly increasing calendar time within every station, never a duplicate day
srt = EXPL.sort_values(["station_id", "date"])
assert int((srt.groupby("station_id")["date"].diff().dt.days.dropna() <= 0).sum()) == 0
# the 2018 archive hole never entered the model-ready table (Phase 13 L10)
assert int((EXPL["date"].dt.year == 2018).sum()) == 0

yrs = EXPL.groupby(EXPL["date"].dt.year).size()
check("E4 rows and keys",
      f"{len(EXPL):,} rows, {EXPL.duplicated(KEYS).sum()} duplicate station-days, "
      f"{n_st} stations, {dmin.date()} -> {dmax.date()}, time strictly increasing within "
      f"every station, 0 rows in 2018; years {dict(yrs)}")"""))

cells.append(code(r"""# E5 - dtype contract
num_pred = EXPL[PRED].select_dtypes(include=[np.number]).columns.tolist()
text_pred = [c for c in PRED if c not in num_pred]
assert set(text_pred) == {"river", "basin", "division", "district", "upazilla"}, text_pred
assert len(num_pred) + len(text_pred) == 62
assert pd.api.types.is_datetime64_any_dtype(EXPL["date"])
assert np.issubdtype(EXPL["station_id"].dtype, np.integer)
for h in TARGETS:
    vals = set(EXPL[h].dropna().unique())
    assert vals <= {0.0, 1.0}, (h, vals)
    assert EXPL[h].isna().any()

check("E5 dtypes",
      f"predictors: {len(num_pred)} numeric + {len(text_pred)} text "
      f"({', '.join(text_pred)} - frequency-encoded later, raw in this table); keys: int64 + "
      f"datetime64; targets: {len(TARGETS)} binary float columns with NaN for unlabelled "
      f"windows")"""))

cells.append(md("""## Missingness of the final table

Two different kinds of NaN, and Phase 15/16 must not confuse them:

* **gap-driven NaN** - a value that *would* exist but the observation record does not
  reach back far enough (the 2018 archive hole behind `wl_last_lag1..7`);
* **structural NaN** - a value that *cannot* exist for a station (no upstream gauge
  `up_*`, no rain weather station, no official danger level).

Every model here is a tree model first (Phase 16+), so NaN is kept as *unknown*, never
filled - except in the LINEAR set, where imputation is a Phase-16 decision."""))

cells.append(code(r"""# E6 - missingness profile of the 67-column table
def pct(c):
    return float(EXPL[c].isna().mean())

prof = pd.DataFrame({
    "column": ORDER,
    "block": (["location", "timestamp"] + ["predictor"] * 62 + ["target"] * 3),
    "dtype": [str(EXPL[c].dtype) for c in ORDER],
    "pct_null": [pct(c) for c in ORDER],
})
prof["pct_null"] = (prof["pct_null"] * 100).round(2)

pred_null = prof[prof["block"] == "predictor"].sort_values("pct_null", ascending=False)
zero_nan = pred_null[pred_null["pct_null"] == 0.0]["column"].tolist()
full_nan = [c for c in PRED if EXPL[c].isna().all()]
assert not full_nan, full_nan
up = [c for c in PRED if c.startswith("up_")]
assert all(0.60 <= pct(c) <= 0.80 for c in up), {c: round(pct(c), 3) for c in up}
assert pct("wl_last_lag1") > 0          # the archive hole leaves a mark

print(f"predictors with 0% NaN: {len(zero_nan)}/62 -> {zero_nan}")
print("\npredictors by missingness (top 20):")
print(pred_null.head(20).to_string(index=False))
print("\ntargets:")
print(prof[prof['block'] == 'target'].to_string(index=False))

check("E6 missingness profile",
      f"0 fully-empty predictors; {len(zero_nan)} predictors are complete; structural NaN "
      f"up_* {min(pct(c) for c in up):.0%}-{max(pct(c) for c in up):.0%}; wl_last_lag1 "
      f"{pct('wl_last_lag1'):.1%} (2018 hole); targets NaN as contracted")"""))

cells.append(code(r"""# E7 - target contract, re-checked against phase0 section 4
EXPECT = {"label_24h": (200559, 9982, 106, 4.98),
          "label_48h": (197748, 11241, 2917, 5.68),
          "label_72h": (195174, 12340, 5491, 6.32)}
rows = []
for h in TARGETS:
    n = int(EXPL[h].notna().sum())
    pos = int(EXPL[h].sum())
    nan = int(EXPL[h].isna().sum())
    rate = round(100 * pos / n, 2)
    exp_n, exp_pos, exp_nan, exp_rate = EXPECT[h]
    assert (n, pos, nan) == (exp_n, exp_pos, exp_nan), (h, n, pos, nan)
    assert abs(rate - exp_rate) < 0.01, (h, rate, exp_rate)
    rows.append({"target": h, "n": n, "positives": pos, "base_rate_%": rate, "NaN": nan,
                 "imbalance_0:1": round((n - pos) / pos, 1)})
TGT = pd.DataFrame(rows)
assert TGT["n"].is_monotonic_decreasing and TGT["positives"].is_monotonic_increasing

print(TGT.to_string(index=False))
check("E7 target contract",
      "label_24h/48h/72h = 200,559 / 197,748 / 195,174 labelled rows with 9,982 / 11,241 / "
      "12,340 positives (4.98 / 5.68 / 6.32 %) - matches phase0 §4 exactly; PR-AUC is the "
      "headline metric (rare, not degenerate)")"""))

cells.append(md("""## Writing the table

The source (`flood_dataset_model_ready.csv`) is **never** written: the notebook records its
SHA-256 first (setup cell), writes `experimental_dataset.csv`, then re-reads it and proves

1. round-trip equality - every value survives the CSV, order of columns preserved;
2. source immutability - the SHA-256 of the input is byte-identical after the run."""))

cells.append(code(r"""# E8 - write, re-read, compare
OUTCSV = PROC / "experimental_dataset.csv"
t0 = time.time()
EXPL.to_csv(OUTCSV, index=False)
back = pd.read_csv(OUTCSV)
back["date"] = pd.to_datetime(back["date"])
assert list(back.columns) == ORDER, "column order lost"
pd.testing.assert_frame_equal(EXPL.reset_index(drop=True), back, check_dtype=False)
size = OUTCSV.stat().st_size

check("E8 write + round-trip",
      f"{OUTCSV.name}: {back.shape[0]:,} x {back.shape[1]} written and re-read in "
      f"{time.time() - t0:.1f}s ({size / 1e6:.1f} MB); values, dtypes (after inference) and "
      f"column order identical - no silent coercion")"""))

cells.append(code(r"""# E9 - immutability of the source (workflow: never overwrite the original)
assert sha256(SRC) == SRC_SHA, "source file changed"
recheck = pd.read_csv(SRC, usecols=["station_id"]).shape[0]
assert recheck == 200665
check("E9 source immutability",
      f"SHA-256 of flood_dataset_model_ready.csv unchanged ({SRC_SHA[:16]}..., "
      f"{SRC.stat().st_size / 1e6:.1f} MB); the raw copy stays as it was")"""))

cells.append(code(r"""# E10 - data dictionary + preview
GRP = dict(zip(WL["column"], WL["group"]))
LATEST = dict(zip(WL["column"], WL["latest_day_used"]))
RULE = dict(zip(WL["column"], WL["ruling"]))
BLOCK = {c: "location" if c == "station_id" else "timestamp" if c == "date"
         else "target" for c in ORDER}
BLOCK.update({c: "predictor" for c in PRED})
ROLE = {c: "key - never a feature" for c in KEYS}
ROLE.update({c: "feature (Phase-12 FULL)" for c in PRED})
ROLE.update({c: "target" for c in TARGETS})

DICT = pd.DataFrame({
    "position": range(1, 68),
    "block": [BLOCK[c] for c in ORDER],
    "column": ORDER,
    "role": [ROLE[c] for c in ORDER],
    "whitelist_group": [GRP.get(c, "target" if c in TARGETS else "") for c in ORDER],
    "latest_day_used": [LATEST.get(c, "t+1 .. t+h") for c in ORDER],
    "ruling": [RULE.get(c, "TARGET") for c in ORDER],
    "dtype": [str(EXPL[c].dtype) for c in ORDER],
    "non_null": [int(EXPL[c].notna().sum()) for c in ORDER],
    "pct_null": [round(pct(c) * 100, 2) for c in ORDER],
})
DICT.to_csv(PROC / "experimental_dataset_dictionary.csv", index=False)
assert len(DICT) == 67 and (DICT["block"] == "predictor").sum() == 62

pd.set_option("display.max_rows", 70, "display.width", 200)
print(DICT.to_string(index=False))
print(f"\nexperimental_dataset.csv + experimental_dataset_dictionary.csv written "
      f"({size / 1e6:.1f} MB, {EXPL.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory)")
check("E10 data dictionary",
      "67-row dictionary written (position, block, role, whitelist group, latest day used, "
      "ruling, dtype, null profile)")"""))

cells.append(code(r"""print("=" * 78)
print("FINAL EXPERIMENTAL DATASET  data/processed/experimental_dataset.csv")
print("=" * 78)
print(f"shape            : {EXPL.shape[0]:,} rows x {EXPL.shape[1]} columns")
print(f"location keys    : 133 stations (station_id; lat/lon are predictors)")
print(f"timestamps       : {dmin.date()} .. {dmax.date()} (day t, one row per observed day)")
print(f"predictors       : 62 (Phase-12 FULL = Phase-13 ADMIT 61 + ADMIT-WATCH 1)")
print(f"targets          : label_24h / label_48h / label_72h")
print(f"dropped from src : {', '.join(sorted(REMOVED))}")
print(f"split            : NOT assigned here - Phase 15 decides train/val/test")
print("\nfirst 3 rows (transposed):")
print(EXPL.head(3).T.to_string())
print("\nall checks:")
for r in RES:
    print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['id']}")
print("ALL PHASE-14 CHECKS PASSED")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
