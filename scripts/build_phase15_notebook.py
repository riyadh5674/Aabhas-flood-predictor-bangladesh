"""Build notebooks/07_split_design.ipynb (workflow Phase 15, body steps 25-27).

Workflow Phase 15: do NOT start with a random 80/20 split.
    25 temporal split      earlier years -> train, later -> val, most recent -> test
    26 spatial validation  known locations -> train, unseen locations -> test
    27 monsoon validation  evaluate monsoon vs non-monsoon separately

Binding phase0 decisions applied here:
    D1  monsoon = JJAS                    D8  rolling-origin across years, never one pooled split
    R1  base-rate drift (per-year rates)  R6  2018 gap: split around it, never interpolate
    R7  station network 85 -> 133: unseen-location claims must not be network growth in disguise

Cell order note: the station axis (step 26 folds) is assigned BEFORE the rolling-origin
folds of step 25, because every training set must know which stations are eligible for
training (the frozen unseen-location holdout trains on nothing).

Checks:
    S1  input contract: the Phase-14 experimental table is intact
    S2  spatial folds (step 26 / R7) - 5 cohort-balanced station folds, fold 0 frozen
    S3  temporal split (step 25) - counts, ranges, per-split label rates
    S4  embargo (step 25 / R6) - natural dry-season gaps at every cut + explicit 7-day rule
    S5  rolling-origin folds (D8 / R1) - val years 2020/2021/2022, holdout excluded
    S6  spatial + seen/unseen protocols (step 26) - station-disjoint by construction;
        test period reported as three buckets (trained / new-network / holdout-unseen)
    S7  monsoon season (step 27 / D1) - JJAS share and base rates re-verified
    S8  integrity - completeness, determinism, conservation, per-protocol exclusion
    S9  write + round-trip of split_assignments.csv and splits.json
    S10 summary tables

Outputs: data/processed/split_assignments.csv, data/processed/splits.json
(never touches experimental_dataset.csv - the Phase-14 table stays byte-stable)
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "07_split_design.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 15 - Train / validation / test strategy (workflow body steps 25-27)

| | |
| --- | --- |
| **Workflow** | **Phase 15 - Train/validation/test strategy**: *do NOT start with a normal random 80/20 split* -> `25` temporal split -> `26` spatial validation (unseen locations) -> `27` monsoon validation |
| **Checklist #** | 27 (temporal train/val/test, must respect **R1** + **R6**) + 28 (spatial validation design, must respect **R7**) |
| **Input** | `data/processed/experimental_dataset.csv` (200,665 x 67, Phase 14) |
| **Binding decisions** | **D1** monsoon = JJAS · **D8** rolling-origin, per-year, never pooled · **R1** base-rate drift · **R6** 2018 gap · **R7** station network 85 -> 133 |
| **Outputs** | `data/processed/split_assignments.csv` (one row per station-day) · `data/processed/splits.json` (design + evidence manifest) |
| **Not modified** | `experimental_dataset.csv` (Phase 14) - splits live in a sidecar keyed on `(station_id, date)`; its SHA-256 is re-checked (S9) |

Ten checks, **S1-S10**. *No model is trained here* - this notebook defines and proves the
strategy; Phase 16 fits baselines. One ordering note: the **station axis** (step 26 folds)
is assigned before the rolling folds of step 25, because every training set must know which
stations are eligible for training at all."""))

cells.append(code(r"""import hashlib, json, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
TABLE = PROC / "experimental_dataset.csv"
HORIZONS = ["label_24h", "label_48h", "label_72h"]
SEED = 20261008
EMBARGO_DAYS = 7
CUT_VAL = pd.Timestamp("2022-01-01")
CUT_TEST = pd.Timestamp("2023-01-01")
MONSOON_MONTHS = (6, 7, 8, 9)                 # D1: JJAS
N_FOLDS, HOLDOUT_FOLD = 5, 0                  # step 26: fold 0 = unseen-location test

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

TBL_SHA = sha256(TABLE)
t0 = time.time()
df = pd.read_csv(TABLE)
df["date"] = pd.to_datetime(df["date"])

RES = []
def check(cid, detail, ok=True):
    RES.append({"id": cid, "pass": bool(ok)})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}")
    assert ok, cid

def base_rate(g, h):
    n = int(g[h].notna().sum()); p = int(g[h].sum())
    return 100 * p / n if n else np.nan

def rate_tbl(frame, key):
    rows = []
    for k, g in frame.groupby(key):
        r = {"part": k, "rows": len(g), "stations": g["station_id"].nunique()}
        for h in HORIZONS:
            n = int(g[h].notna().sum()); p = int(g[h].sum())
            r[h + "_labelled"] = n
            r[h + "_pos"] = p
            r[h + "_rate_%"] = round(100 * p / n, 2) if n else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("part")

print(f"experimental table {df.shape} read in {time.time() - t0:.1f}s | sha256 {TBL_SHA[:16]}...")"""))

cells.append(code(r"""# S1 - the Phase-14 table is intact before anything is assigned
assert df.shape == (200665, 67), df.shape
assert df.duplicated(["station_id", "date"]).sum() == 0
sets = json.loads((PROC / "feature_sets.json").read_text(encoding="utf-8"))
FULL = sets["sets"]["FULL"]
assert set(FULL) <= set(df.columns) and set(HORIZONS) <= set(df.columns)
assert df["station_id"].nunique() == 133
assert df["date"].min() == pd.Timestamp("2017-01-06") and df["date"].max() == pd.Timestamp("2026-10-05")
assert not {"split", "spatial_fold", "season"} & set(df.columns), "split columns already present"

check("S1 input contract",
      "experimental_dataset.csv 200,665 x 67, unique station-days, 133 stations, "
      "62 predictors + 3 targets present, no split column yet - Phase-14 table untouched")"""))

cells.append(md("""## The station axis first (feeds steps 25 and 26)

The temporal protocols need to know *which stations may be trained on* before they can
define a training set, so the station design comes first.

**R7 is the danger here**: the network grew **85 -> 133 stations**, so "unseen station" can
quietly mean "recent station" - a time effect wearing a spatial costume. The design:

* **5 station folds**, balanced on label count *within* early/late cohorts
  (first observed <= 2017 vs >= 2018) so every fold mixes both eras;
* **fold 0 is frozen**: those stations train on nothing and validate on nothing - their rows
  in the 2023+ test period are the *unseen-location* readout;
* folds 1-4 form the development-time **spatial CV** (S6).

Assignment is a pure function of `(positives, rows, station_id)` inside each cohort, seeded
and deterministic - re-running reproduces identical folds."""))

cells.append(code(r"""# S2 - spatial folds (step 26), cohort-balanced (R7)
stat = df.groupby("station_id").agg(
    rows=("date", "size"),
    pos24=("label_24h", "sum"),
    pos72=("label_72h", "sum"),
    first_date=("date", "min"),
    basin=("basin", "first"),
).reset_index()
stat["first_year"] = stat["first_date"].dt.year
stat["cohort"] = np.where(stat["first_year"] <= 2017, "early", "late")
n_early = int((stat["cohort"] == "early").sum())
n_late = int((stat["cohort"] == "late").sum())
assert 80 <= n_early <= 90 and 40 <= n_late <= 55, (n_early, n_late)
assert n_early + n_late == 133

load_c = {c: np.zeros(N_FOLDS) for c in ("early", "late")}
load_p_all = np.zeros(N_FOLDS)          # global positives load, used to break count ties
fold_of = {}
for cohort, grp in stat.groupby("cohort"):
    ordered = grp.sort_values(["pos24", "rows", "station_id"], ascending=[False, False, True])
    for _, r in ordered.iterrows():
        # primary = fewest stations of THIS cohort (keeps 17/17/17/17/17 early + 9-10 late),
        # tie-break = fewest positives on the fold overall (stops both cohorts' biggest
        # station landing on the same fold), then lowest fold index -> deterministic
        k = int(np.lexsort((np.arange(N_FOLDS), load_p_all, load_c[cohort]))[0])
        fold_of[int(r["station_id"])] = k
        load_c[cohort][k] += 1
        load_p_all[k] += r["pos24"]
stat["spatial_fold"] = stat["station_id"].map(fold_of)
assert len(fold_of) == 133 and set(fold_of.values()) == set(range(N_FOLDS))

df = df.merge(stat[["station_id", "spatial_fold", "cohort", "first_year"]],
              on="station_id", how="left", validate="many_to_one")
df["station_holdout"] = df["spatial_fold"] == HOLDOUT_FOLD
assert df["spatial_fold"].notna().all() and df["cohort"].notna().all()

fold_tbl = df.groupby("spatial_fold").agg(
    stations=("station_id", "nunique"), rows=("date", "size"),
    pos24=("label_24h", "sum"), pos72=("label_72h", "sum"))
fold_tbl["early_st"] = df[df["cohort"] == "early"].groupby("spatial_fold")["station_id"].nunique()
fold_tbl["late_st"] = df[df["cohort"] == "late"].groupby("spatial_fold")["station_id"].nunique()
fold_tbl = fold_tbl[["stations", "early_st", "late_st", "rows", "pos24", "pos72"]]
print(fold_tbl.to_string())
print()

ratio = fold_tbl["pos24"].max() / fold_tbl["pos24"].mean()
assert fold_tbl["stations"].between(25, 29).all(), fold_tbl["stations"].to_dict()
assert (fold_tbl["early_st"] >= 10).all() and (fold_tbl["late_st"] >= 5).all()
assert ratio <= 1.25, ratio
hold_st = set(stat.loc[stat["spatial_fold"] == HOLDOUT_FOLD, "station_id"])
hold_rows = df[df["station_holdout"]]
n_hold_early = df.loc[df["station_holdout"] & (df["cohort"] == "early"), "station_id"].nunique()
n_hold_late = df.loc[df["station_holdout"] & (df["cohort"] == "late"), "station_id"].nunique()
print(f"fold {HOLDOUT_FOLD} frozen: {len(hold_st)} stations "
      f"({n_hold_early} early / {n_hold_late} late), {len(hold_rows):,} rows total "
      f"(train-period rows of these stations are never used for training - asserted in S5/S6/S8)")

check("S2 spatial folds (step 26 / R7)",
      f"5 folds over 133 stations ({n_early} early <=2017 / {n_late} late >=2018), "
      f"{fold_tbl['stations'].min()}-{fold_tbl['stations'].max()} stations per fold, every "
      f"fold has both cohorts, positives max/mean {ratio:.2f}; fold {HOLDOUT_FOLD} = "
      f"{len(hold_st)} stations frozen out of every training and validation set")"""))

cells.append(md("""## Step 25 - temporal split

> *Earlier years -> training. Later period -> validation. Most recent period -> test. The
> exact years depend on your dataset.* (workflow step 25)

The years are forced by the data, not chosen for convenience:

| cut | why |
| --- | --- |
| **2022-01-01** | sits inside a **131-day** stretch containing no model-ready row at all (dry season, Phase 13 L11) |
| **2023-01-01** | sits inside an **88-day** dry-season stretch; 2023-2026 is the most recent, most complete era (23,869 -> 28,252 rows/yr) |
| never inside | the 2018 archive hole (**R6**): 0 model-ready rows exist in 2018, so no cut can land there |

So **train = 2017-2021, validation = 2022, test = 2023-2026 (frozen)** - the same dates
Phase 12 trained on and Phase 13 audited, now official.

`EMBARGO_DAYS = 7` (max label window 3 days + buffer) applies at every cut; where the
natural dry-season gap already exceeds it, the embargo flags nothing - which is why S4
*measures* the gap instead of assuming it."""))

cells.append(code(r"""# S3 - temporal split (step 25): the three periods
df["split"] = np.where(df["date"] < CUT_VAL, "train",
                       np.where(df["date"] < CUT_TEST, "val", "test"))
counts = df["split"].value_counts().to_dict()
assert counts == {"train": 77571, "val": 21920, "test": 101174}, counts
assert counts["train"] + counts["val"] + counts["test"] == len(df)

rng = df.groupby("split")["date"].agg(["min", "max", "size"])
assert rng.loc["train", "max"] < CUT_VAL <= rng.loc["val", "min"]
assert rng.loc["val", "max"] < CUT_TEST <= rng.loc["test", "min"]
assert not set(df.loc[df.split == "train", "date"]) & set(df.loc[df.split == "val", "date"])
assert not set(df.loc[df.split == "train", "date"]) & set(df.loc[df.split == "test", "date"])
assert not set(df.loc[df.split == "val", "date"]) & set(df.loc[df.split == "test", "date"])

SPLIT_TBL = rate_tbl(df, "split")
print(rng.to_string())
print()
print(SPLIT_TBL.to_string())

check("S3 temporal split (step 25)",
      f"train {counts['train']:,} (2017-01-06..2021-11-03) / val {counts['val']:,} "
      f"(2022-03-15..2022-10-30) / test {counts['test']:,} (2023-01-27..2026-10-05); "
      "0 shared dates between any two parts; label rates reported per horizon per part "
      "(R1: never pool years)")"""))

cells.append(code(r"""# S4 - embargo evidence (step 25 + R6): what the data does, then the rule we apply
CUTS = [pd.Timestamp(y) for y in ("2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01")]
cut_rows = []
for c in CUTS:
    before = df.loc[df["date"] < c, "date"]
    after = df.loc[df["date"] >= c, "date"]
    last, first = before.max(), after.min()
    free = int((first - last).days) - 1                       # model-ready-free days
    near = int(((df["date"] > c - pd.Timedelta(days=EMBARGO_DAYS)) & (df["date"] < c)).sum())
    strad = int(((df["date"] > c - pd.Timedelta(days=3)) & (df["date"] < c)).sum())
    cut_rows.append({"cut": str(c.date()), "last_before": str(last.date()),
                     "first_after": str(first.date()), "natural_free_days": free,
                     f"within_{EMBARGO_DAYS}d": near, "within_3d": strad})
CUTS_TBL = pd.DataFrame(cut_rows).set_index("cut")
print(CUTS_TBL.to_string())

assert int(CUTS_TBL.loc["2022-01-01", "natural_free_days"]) == 131
assert int(CUTS_TBL.loc["2023-01-01", "natural_free_days"]) == 88
assert int(CUTS_TBL.loc["2022-01-01", f"within_{EMBARGO_DAYS}d"]) == 0
assert int(CUTS_TBL.loc["2023-01-01", f"within_{EMBARGO_DAYS}d"]) == 0
assert int(CUTS_TBL.loc["2022-01-01", "within_3d"]) == 0
assert int(CUTS_TBL.loc["2023-01-01", "within_3d"]) == 0
assert int((df["date"].dt.year == 2018).sum()) == 0           # R6: no cut can land in 2018

in_emb = pd.Series(False, index=df.index)
for c in CUTS:
    in_emb |= (df["date"] > c - pd.Timedelta(days=EMBARGO_DAYS)) & (df["date"] < c)
df["in_embargo_window"] = in_emb

check("S4 embargo (step 25 / R6)",
      "natural model-ready-free days at the cuts: "
      + ", ".join(f"{c} = {int(r)}" for c, r in CUTS_TBL["natural_free_days"].items())
      + f"; explicit {EMBARGO_DAYS}-day rule flags {int(in_emb.sum())} rows overall, 0 of "
        "them before either primary cut (2022/2023); 2018 empty, so no cut lands in the hole")"""))

cells.append(code(r"""# S5 - rolling-origin folds (D8: per-year, never one pooled split; R1: drift)
ROLLING = {2020: "2020-01-01", 2021: "2021-01-01", 2022: "2022-01-01"}
rows = []
for y, cut in ROLLING.items():
    c = pd.Timestamp(cut)
    tr = df[(df["date"] < c - pd.Timedelta(days=EMBARGO_DAYS)) & ~df["station_holdout"]]
    va = df[(df["date"].dt.year == y) & ~df["station_holdout"]]
    assert len(tr) > 0 and len(va) > 0, y
    assert tr["date"].max() < va["date"].min()
    assert not set(tr["station_id"]) & set(df.loc[df["station_holdout"], "station_id"])
    r = {"fold": f"train <= {y - 1} -> val {y}", "train_rows": len(tr),
         "train_stations": tr["station_id"].nunique(), "val_rows": len(va),
         "val_stations": va["station_id"].nunique()}
    for h in HORIZONS:
        r[f"val_{h}_labelled"] = int(va[h].notna().sum())
        r[f"val_{h}_pos"] = int(va[h].sum())
        r[f"val_{h}_rate_%"] = round(base_rate(va, h), 2)
    rows.append(r)
ROLL_TBL = pd.DataFrame(rows).set_index("fold")
print(ROLL_TBL.to_string())
print()

# the primary split's train is exactly the 2022 fold's train (embargo is a no-op there)
prim_train = df[(df["split"] == "train") & ~df["station_holdout"]]
roll22 = df[(df["date"] < CUT_VAL - pd.Timedelta(days=EMBARGO_DAYS)) & ~df["station_holdout"]]
assert set(prim_train["station_id"].astype(str) + prim_train["date"].astype(str)) == \
       set(roll22["station_id"].astype(str) + roll22["date"].astype(str))

# R1 drift must be visible, not assumed
r17 = base_rate(df[df["date"].dt.year == 2017], "label_72h")
r23 = base_rate(df[df["date"].dt.year == 2023], "label_72h")
assert abs(r17 - 16.2) < 1.0, r17
assert abs(r23 - 1.2) < 0.6, r23
assert r17 > 5 * r23

check("S5 rolling-origin folds (D8 / R1)",
      "3 pre-test folds (val 2020 / 2021 / 2022), train strictly before the val year minus "
      f"the embargo, holdout stations excluded from both sides; 2017 label_72h {r17:.1f}% vs "
      f"2023 {r23:.1f}% ({r17 / r23:.0f}x drift) - hence per-fold, per-year reporting; "
      "fold-2022 train is byte-identical to the primary train set")"""))

cells.append(md("""## Step 26 - spatial validation

> *Test whether the model generalizes to locations it hasn't seen: known locations -> train,
> different locations -> test.* (workflow step 26)

Two protocols, kept strictly apart:

| protocol | trains on | tests on | measures |
| --- | --- | --- | --- |
| **P3 spatial CV** (development) | train years (2017-2021), folds 1-4 minus one | the left-out non-holdout fold, same years | unseen-location transfer **during development**, without touching time |
| **P4 frozen unseen test** (final) | *nothing* - fold 0 never trains or validates | fold-0 stations in 2023-2026 | the final unseen-location claim |
| *reference* | the stations a model actually trains on (folds 1-4, 2017-2021) | those same stations in 2023-2026 | pure **temporal** generalisation |

**R7 again:** the test period also contains stations that first appear *after* 2022
(network growth 85 -> 133). They are not holdout stations, but they were never trained on
either, so S6 scores them as a **third bucket** instead of quietly calling them "seen" -
network growth must not masquerade as generalisation in either direction.

The test period is never used for selection (**D8**); the temporal axis never mixes with
the station axis."""))

cells.append(code(r"""# S6 - spatial CV (P3) + the trained / new-network / unseen readout (P4, R7)
TRAIN_PERIOD = df[(df["split"] == "train") & ~df["station_holdout"]]
assert set(TRAIN_PERIOD["date"].dt.year) == {2017, 2019, 2020, 2021}
train_st = set(TRAIN_PERIOD["station_id"])          # stations a model actually trains on
assert len(train_st) == 84, len(train_st)

cv_rows = []
for k in range(N_FOLDS):
    if k == HOLDOUT_FOLD:
        continue                                    # frozen: never trained, never CV'd
    tr = TRAIN_PERIOD[TRAIN_PERIOD["spatial_fold"] != k]
    te = TRAIN_PERIOD[TRAIN_PERIOD["spatial_fold"] == k]
    assert set(tr["station_id"]).isdisjoint(set(te["station_id"]))
    assert set(tr["station_id"]).isdisjoint(hold_st)
    assert set(tr["date"]) & set(te["date"])        # same era, different places
    n, p = int(te["label_24h"].notna().sum()), int(te["label_24h"].sum())
    cv_rows.append({"spatial CV": f"train folds != {k} -> test fold {k}",
                    "train_rows": len(tr), "test_rows": len(te),
                    "test_stations": te["station_id"].nunique(),
                    "test_labelled": n, "test_pos": p, "test_rate_%": round(100 * p / n, 2)})
CV_TBL = pd.DataFrame(cv_rows).set_index("spatial CV")

testp = df[df["split"] == "test"]
seen = testp[(~testp["station_holdout"]) & testp["station_id"].isin(train_st)]
newnet = testp[(~testp["station_holdout"]) & ~testp["station_id"].isin(train_st)]
unseen = testp[testp["station_holdout"]]
assert len(seen) + len(newnet) + len(unseen) == len(testp)
assert set(unseen["station_id"]) == hold_st
assert set(seen["station_id"]) <= train_st and not (set(newnet["station_id"]) & train_st)
SEEN_TBL = rate_tbl(pd.concat([
    seen.assign(part="test / trained stations"),
    newnet.assign(part="test / new-network stations (R7)"),
    unseen.assign(part="test / UNSEEN stations (holdout)")]), "part")
nn_years = newnet.groupby("station_id")["date"].min().dt.year.value_counts().sort_index()

print(CV_TBL.to_string())
print()
print(SEEN_TBL.to_string())
print(f"\nnew-network stations first observed in the test period: "
      f"{dict((int(y), int(n)) for y, n in nn_years.items())} -> {int(nn_years.sum())} stations, "
      f"{len(newnet):,} rows - never in any training set either, so they are reported "
      f"apart from the trained bucket (R7: network growth is not spatial generalisation)")
check("S6 spatial protocols (step 26)",
      f"4-fold spatial CV on the train years - station-disjoint every time, holdout absent "
      f"from every train set ({CV_TBL['test_rows'].min()}-{CV_TBL['test_rows'].max()} test "
      f"rows per fold); frozen test period splits into {len(seen):,} trained-station rows "
      f"({seen['station_id'].nunique()} stations) / {len(newnet):,} new-network rows "
      f"({newnet['station_id'].nunique()} stations) / {len(unseen):,} unseen rows "
      f"({len(hold_st)} holdout stations never trained on)")"""))

cells.append(md("""## Step 27 - monsoon validation

> *Evaluate separately during monsoon and non-monsoon. This tells you whether performance
> is robust during the period that actually matters most.* (workflow step 27)

**D1 fixes the window at JJAS (June-September)** for every seasonal evaluation in this
project. Phase 0 already measured why: ~55 % of rows and a `label_72h` base rate of roughly
9.8 % in monsoon vs 1.8 % outside - the phenomenon is overwhelmingly a monsoon phenomenon,
so a single pooled score would hide the only period that matters."""))

cells.append(code(r"""# S7 - monsoon season (step 27 / D1), rates re-verified against phase0 section 6
df["season"] = np.where(df["date"].dt.month.isin(MONSOON_MONTHS), "monsoon", "non-monsoon")
share = float((df["season"] == "monsoon").mean())
assert abs(share - 0.555) < 0.01, share

m72 = base_rate(df[df["season"] == "monsoon"], "label_72h")
n72 = base_rate(df[df["season"] == "non-monsoon"], "label_72h")
assert abs(m72 - 9.79) < 0.4, m72
assert abs(n72 - 1.83) < 0.4, n72
assert m72 > 4 * n72

SEASON_TBL = rate_tbl(df, "season")
print(f"monsoon (JJAS) share of rows: {share:.1%}   label_72h base rate: "
      f"monsoon {m72:.2f}% vs non-monsoon {n72:.2f}%\n")
print(SEASON_TBL.to_string())
print()
print(pd.crosstab(df["split"], df["season"], margins=True).to_string())
print()
for s, g in df.groupby("split"):
    assert len(g[g["season"] == "monsoon"]) > 1000, s
    assert len(g[g["season"] == "non-monsoon"]) > 1000, s

check("S7 monsoon validation (step 27 / D1)",
      f"JJAS = {share:.1%} of rows (phase0 §6: 55.5%); label_72h base rate {m72:.2f}% in "
      f"monsoon vs {n72:.2f}% outside ({m72 / n72:.1f}x); every split holds >1000 monsoon "
      "and >1000 non-monsoon rows - all three horizons can be reported per season")"""))

cells.append(code(r"""# S8 - integrity: completeness, determinism, conservation, per-protocol exclusion
for c in ("split", "spatial_fold", "cohort", "season", "station_holdout", "in_embargo_window"):
    assert df[c].notna().all(), c
assert int(df["split"].value_counts().sum()) == len(df) == 200665

# determinism: every assignment is a pure function of date or station_id
re_split = np.where(df["date"] < CUT_VAL, "train", np.where(df["date"] < CUT_TEST, "val", "test"))
assert (re_split == df["split"]).all()
assert (np.where(df["date"].dt.month.isin(MONSOON_MONTHS), "monsoon", "non-monsoon")
        == df["season"]).all()
assert (df["station_id"].map(fold_of) == df["spatial_fold"]).all()

# per-protocol exclusion: no training set ever contains a holdout station
sets_built = {
    "primary train": df[(df["split"] == "train") & ~df["station_holdout"]],
    "rolling 2020": df[(df["date"] < pd.Timestamp("2020-01-01") - pd.Timedelta(days=EMBARGO_DAYS)) & ~df["station_holdout"]],
    "rolling 2021": df[(df["date"] < pd.Timestamp("2021-01-01") - pd.Timedelta(days=EMBARGO_DAYS)) & ~df["station_holdout"]],
    "rolling 2022": df[(df["date"] < CUT_VAL - pd.Timedelta(days=EMBARGO_DAYS)) & ~df["station_holdout"]],
    "spatial CV train": df[(df["split"] == "train") & ~df["station_holdout"] & (df["spatial_fold"] != HOLDOUT_FOLD)],
}
for nm, g in sets_built.items():
    assert not set(g["station_id"]) & hold_st, nm
    assert len(g) > 0, nm

assert set(df.loc[df["split"] == "test", "date"].dt.year) == {2023, 2024, 2025, 2026}
assert int((df["date"].dt.year == 2018).sum()) == 0                       # R6
assert int(df["in_embargo_window"].sum()) < 3000

hold = df[df["station_holdout"]]
period_counts = hold["split"].value_counts().to_dict()
print("holdout stations:", len(hold_st))
print("their rows by period (used ONLY in test):",
      {k: int(v) for k, v in period_counts.items()})
print("cohort x split rows:")
print(pd.crosstab(df["cohort"], df["split"]).to_string())
check("S8 integrity",
      f"all 6 assignment columns complete; re-derived from date/station_id and identical; "
      f"200,665 rows conserved; 5 training sets built here all exclude the "
      f"{len(hold_st)} holdout stations; 2018 absent; "
      f"{int(df['in_embargo_window'].sum())} rows flagged in embargo windows (recorded, "
      "never silently dropped)")"""))

cells.append(md("""## Outputs

`split_assignments.csv` is a **sidecar**: one row per station-day with the design columns.
`experimental_dataset.csv` is not modified (its SHA-256 is re-checked in S9), so the
Phase-14 contract stays true and Phase 16 joins the two with `validate="one_to_one"`.

`splits.json` records the whole strategy - cuts, embargo, fold recipe, protocol counts and
the evidence above - so any later phase rebuilds exactly these parts without guessing."""))

cells.append(code(r"""# S9 - write sidecar + manifest, verify round-trip, leave Phase-14 table untouched
OUTCSV = PROC / "split_assignments.csv"
ASSIGN = df[["station_id", "date", "split", "spatial_fold", "cohort", "first_year",
             "station_holdout", "season", "in_embargo_window"]].copy()
ASSIGN["date"] = ASSIGN["date"].dt.strftime("%Y-%m-%d")
assert len(ASSIGN) == 200665 and ASSIGN.duplicated(["station_id", "date"]).sum() == 0
ASSIGN.to_csv(OUTCSV, index=False)

back = pd.read_csv(OUTCSV)
back["date"] = pd.to_datetime(back["date"])
pd.testing.assert_frame_equal(
    ASSIGN.assign(date=pd.to_datetime(ASSIGN["date"])).reset_index(drop=True),
    back, check_dtype=False)
assert sha256(TABLE) == TBL_SHA, "Phase-14 table changed"

manifest = {
    "created": "2026-10-08", "phase": "Phase 15 split strategy (steps 25-27; checklist 27-28)",
    "source_table": "data/processed/experimental_dataset.csv",
    "source_sha256": TBL_SHA,
    "assignments": "data/processed/split_assignments.csv",
    "design": {
        "temporal_cuts": {"val": "2022-01-01", "test": "2023-01-01"},
        "embargo_days": EMBARGO_DAYS,
        "monsoon_months": list(MONSOON_MONTHS),
        "spatial_folds": N_FOLDS, "holdout_fold": HOLDOUT_FOLD,
        "station_cohorts": {"early": "first observed <= 2017", "late": "first observed >= 2018"},
        "seed": SEED,
        "holdout_stations": sorted(int(s) for s in hold_st),
    },
    "protocols": {
        "P1_temporal_primary": "train < 2022 (holdout excluded) / val 2022 (holdout excluded) / "
                               "test 2023+ (all stations; report seen vs unseen separately)",
        "P2_rolling_origin": ["train <= 2019 -> val 2020", "train <= 2020 -> val 2021",
                              "train <= 2021 -> val 2022"],
        "P3_spatial_cv": "4 folds over non-holdout stations, train-period rows only, "
                         "station-disjoint by construction",
        "P4_unseen_location_test": f"fold {HOLDOUT_FOLD} stations, test period only, "
                                   "never trained on or validated on",
        "P5_season": "every metric additionally reported for JJAS vs rest (D1)",
        "frozen": "test period (2023+) never used for model selection (D8)",
    },
    "counts": {
        "split_rows": {k: int(v) for k, v in df["split"].value_counts().items()},
        "split_rows_non_holdout": {k: int(v) for k, v in
                                   df[~df["station_holdout"]]["split"].value_counts().items()},
        "stations_per_fold": {int(k): int(v) for k, v in
                              df.groupby("spatial_fold")["station_id"].nunique().items()},
        "train_sets_excluding_holdout": {k: len(v) for k, v in sets_built.items()},
        "test_trained_station_rows": len(seen),
        "test_new_network_rows": len(newnet),
        "test_unseen_rows": len(unseen),
        "stations_trained_on": len(train_st),
        "stations_new_network": int(newnet["station_id"].nunique()),
        "embargo_flagged_rows": int(df["in_embargo_window"].sum()),
    },
    "evidence": {
        "natural_free_days_at_cuts": {c: int(r) for c, r in CUTS_TBL["natural_free_days"].items()},
        "base_rate_drift_label_72h": {str(y): round(base_rate(df[df.date.dt.year == y], "label_72h"), 2)
                                      for y in sorted(df["date"].dt.year.unique())},
        "monsoon": {"share_rows": round(share, 4),
                    "label_72h_rate_monsoon": round(m72, 2),
                    "label_72h_rate_non_monsoon": round(n72, 2)},
    },
    "checks": RES,
}
check("S9 write + round-trip",
      f"split_assignments.csv {len(back):,} x {back.shape[1]} re-read identical; "
      f"splits.json written; experimental_dataset.csv SHA-256 unchanged ({TBL_SHA[:16]}...)")
(PROC / "splits.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")"""))

cells.append(code(r"""print("=" * 78)
print("PHASE 15 - THE SPLIT STRATEGY   data/processed/split_assignments.csv")
print("=" * 78)
print("\n[step 25] temporal")
print(f"  train  2017-01-06 .. 2021-11-03   {counts['train']:>7,} rows   "
      f"{SPLIT_TBL.loc['train', 'label_24h_rate_%']:.2f}% 24h base rate")
print(f"  val    2022-03-15 .. 2022-10-30   {counts['val']:>7,} rows   "
      f"{SPLIT_TBL.loc['val', 'label_24h_rate_%']:.2f}% 24h")
print(f"  test   2023-01-27 .. 2026-10-05   {counts['test']:>7,} rows   FROZEN, "
      f"{SPLIT_TBL.loc['test', 'label_24h_rate_%']:.2f}% 24h")
print(f"  embargo {EMBARGO_DAYS} d at every cut; natural free days: "
      + ", ".join(f"{c}: {int(r)}" for c, r in CUTS_TBL["natural_free_days"].items()))
print("\n[step 26] spatial")
print(f"  5 station folds (early/late cohorts balanced); fold {HOLDOUT_FOLD} frozen: "
      f"{len(hold_st)} stations, 0 rows in any training set")
print(f"  test period, three buckets: {len(seen):,} trained-station rows "
      f"({seen['station_id'].nunique()} stations) | {len(newnet):,} new-network rows "
      f"({newnet['station_id'].nunique()} stations first seen after 2022, R7) | "
      f"{len(unseen):,} unseen rows ({len(hold_st)} holdout stations)")
print("\n[step 27] monsoon")
print(f"  JJAS = {share:.1%} of rows; label_72h {m72:.2f}% monsoon vs {n72:.2f}% rest")
print("\nper-year label_72h base rate (R1 - never pool):")
drift = manifest["evidence"]["base_rate_drift_label_72h"]
print("  " + "  ".join(f"{y}: {r}%" for y, r in drift.items()))
print("\nall checks:")
for r in RES:
    print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['id']}")
print("ALL PHASE-15 CHECKS PASSED")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
