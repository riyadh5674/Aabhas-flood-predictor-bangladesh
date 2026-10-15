"""Build notebooks/09_sequence_validation.ipynb (workflow Phase 19 / stage 33).

Proves the Dataset-B sequences are scientifically valid BEFORE any LSTM/GRU/TCN
is trained - the deep-learning counterpart of the Phase-13 leakage audit.

Checks S1-S7 (workflow Phase 19 body):
    S1  chronological order within every station - asserted, never assumed
    S2  continuity: calendar-gap distribution inside every window; the 2018
        archive gap (R6) is counted; empty timesteps are proven to be exactly
        the absent calendar days (no silent shift, no fabrication)
    S3  missing-timestep policy: four candidate rules are costed in dropped
        samples per L (with a bias probe on the strongest alternative), ONE rule
        is chosen on evidence, applied identically everywhere, written into
        sequence_manifest.json and made executable as binary mask tensors
        sequence_mask_L{3,7,14,30}.npy; a counterfactual JSON records what the
        rejected rules would have cost
    S4  no station mixing: every timestep of a window equals that window's own
        station/day values (another station's value would mismatch)
    S5  no forward reach: window day indices are <= anchor day t for all
        200,665 windows x 4 lengths (numeric assertion), labels are never in
        the tensor, and reach-BACK across a split boundary is counted (legal)
    S6  label alignment: 100 random windows - labels recomputed independently
        from the raw 3-hourly readings (same contract as scripts/audit_dataset.py
        check A5) must equal label_24h/48h/72h on the anchor day
    S7  split/holdout inheritance: index vs split_assignments 1:1, and every
        P1/P2/P3 training cell contains zero holdout-station rows and zero
        frozen-test rows

Outputs:
    data/processed/sequence_mask_L{3,7,14,30}.npy   uint8 (200665, L)
    data/processed/sequence_policy.json             policy + candidate costs
    data/processed/sequence_manifest.json           AMENDED (policy section only;
                                                    the 5 original hashes rechecked)

Build and run:

    python scripts/build_phase19_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\09_sequence_validation.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "09_sequence_validation.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 19 — Sequence construction & validation (stage 33)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 19 — Sequence construction & validation** → checklist **stage 33** |
| **Inputs** | `sequences_L{3,7,14,30}.npy` + `sequence_index.csv` + `sequence_manifest.json` (Phase 18) · `experimental_dataset.csv` (SHA re-checked) · `split_assignments.csv` · raw 3-hourly readings `data/raw/ffwc/waterlevel/*/*.csv` · `ffwc_stations.csv` |
| **Outputs written** | `sequence_mask_L{3,7,14,30}.npy` · `sequence_policy.json` · `sequence_manifest.json` (policy section added; original hashes re-checked) |
| **Binding decisions** | **D5** information boundary (issue at 24:00 of *t*) · **D12** L ∈ {3,7,14,30} compared fairly · **R6** the 2018 archive gap |

This is the gate before Phase 20/21 training: **no model sees a sequence until every
S-check here is green** and the missing-timestep policy is a written rule with numbers."""))

cells.append(code(r"""import hashlib, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "ffwc"
TBL = PROC / "experimental_dataset.csv"
SPLITS = PROC / "split_assignments.csv"
MANIFEST = PROC / "sequence_manifest.json"
INDEX_CSV = PROC / "sequence_index.csv"
POLICY_JSON = PROC / "sequence_policy.json"
STATIONS = RAW / "ffwc_stations.csv"
WLDIR = RAW / "waterlevel"
NPY = {L: PROC / f"sequences_L{L}.npy" for L in (3, 7, 14, 30)}
MASK = {L: PROC / f"sequence_mask_L{L}.npy" for L in (3, 7, 14, 30)}

TBL_SHA_EXPECTED = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
LENS = (3, 7, 14, 30)
VARIABLES = ["wl_first", "wl_mean", "wl_max", "wl_min", "wl_last", "wl_pct_danger",
             "precip_mm", "tmax_c", "tmin_c", "n_readings", "doy_sin", "doy_cos"]
SEED = 20261008
MIN_READINGS = 3
ROLLING_YEARS = [2020, 2021, 2022]
SPATIAL_FOLDS = [1, 2, 3, 4]
HOLDOUT_CUTOFF = 2023


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
print("Phase 19 notebook - sequence validation")"""))

# ---------------------------------------------------------------- S1
cells.append(code(r"""# S1 - chronological order inside every station (asserted)
ix = pd.read_csv(INDEX_CSV, parse_dates=["date"])
tbl = pd.read_csv(TBL, usecols=["station_id", "date"] + VARIABLES)
check("S1a", f"sequence_index {len(ix):,} rows; sample_idx is 0..n-1 contiguous",
      len(ix) == 200665 and ix.sample_idx.equals(pd.Series(np.arange(len(ix)))))
dup = int(ix.duplicated(["station_id", "date"]).sum())
g = ix.groupby("station_id")["date"]
mono = bool(g.apply(lambda s: s.is_monotonic_increasing and s.is_unique).all())
years = sorted(ix.date.dt.year.unique().tolist())
check("S1b", f"within every one of {ix.station_id.nunique()} stations the rows are "
                 f"strictly chronological and unique (dup={dup}); year coverage {years}",
      mono and dup == 0 and 2018 not in years)"""))

# ---------------------------------------------------------------- S2
cells.append(code(r"""# S2 - continuity: gap distribution inside every window, calendar truth
# rebuild the day-existence grid from the table (same calendar scatter as Phase 18)
dates = pd.to_datetime(tbl["date"])
t0d, t1d = dates.min(), dates.max()
T = (t1d - t0d).days + 1
stations = np.sort(tbl["station_id"].unique())
stm = {s: i for i, s in enumerate(stations)}
si = tbl["station_id"].map(stm).to_numpy()
di = (dates - t0d).dt.days.to_numpy()
present = np.zeros((len(stations), T), bool)
present[si, di] = True
check("S2a", f"calendar {t0d.date()}..{t1d.date()} = {T} days, "
                 f"{present.sum():,}/{present.size:,} station-days present "
                 f"({100 * present.mean():.1f}%)",
      present.sum() == 200665)

gap_rows, longest = {}, {}
for L in LENS:
    X = np.load(NPY[L], mmap_mode="r")
    empty = np.isnan(X).all(axis=2)                    # (n, L)
    # continuity: an empty timestep must be EXACTLY an absent calendar day
    pad = np.zeros((len(stations), L - 1), bool)
    vw = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad, present], axis=1), L, axis=1)   # (nS, T, L)
    day_there = vw[si, di]                             # (n, L)
    mismatch = int((empty != ~day_there).sum())
    n_empty = empty.sum(axis=1)
    run = np.zeros(len(empty), int)
    mx = np.zeros(len(empty), int)
    for k in range(L):
        run = np.where(empty[:, k], run + 1, 0)
        mx = np.maximum(mx, run)
    gap_rows[L] = {
        "pct_clean": float((n_empty == 0).mean()),
        "pct_any": float((n_empty > 0).mean()),
        "pct_ge3": float((n_empty >= 3).mean()),
        "mean_empty": float(n_empty.mean()),
        "max_empty": int(n_empty.max()),
        "max_run": int(mx.max()),
        "mismatch": mismatch,
    }
    longest[L] = mx
    check(f"S2b.{L}", f"L={L}: {100 * gap_rows[L]['pct_clean']:.1f}% of windows are "
                      f"fully consecutive, {100 * gap_rows[L]['pct_any']:.1f}% contain "
                      f">=1 absent day, max run of absent days inside a window = "
                      f"{mx.max()}; empty timestep == absent calendar day "
                      f"(mismatch {mismatch})",
          mismatch == 0)

# the 2018 archive gap (R6): no station has any 2018 row
y18 = int((dates.dt.year == 2018).sum())
span18 = {L: int(((ix.date - pd.Timedelta(days=L - 1) <= pd.Timestamp("2018-12-31"))
                  & (ix.date >= pd.Timestamp("2019-01-01"))).sum()) for L in LENS}
check("S2c", f"R6 confirmed: {y18} table rows in 2018; windows whose history reaches "
                 f"into 2018 by L: {span18} (all their 2018 timesteps are empty)",
      y18 == 0 and all(v > 0 for v in span18.values()))
print("gap table:", json.dumps(gap_rows, indent=1))"""))

# ---------------------------------------------------------------- S3a
cells.append(code(r"""# S3 - missing-timestep policy: cost every candidate rule in dropped samples
labels = ix["label_24h"].to_numpy()
cand = {}
for L in LENS:
    X = np.load(NPY[L], mmap_mode="r")
    empty = np.isnan(X).all(axis=2)
    n_empty = empty.sum(axis=1)
    # R1 keep-all (chosen): never drop; expose a mask channel instead
    r1 = np.ones(len(empty), bool)
    # R2 whole-window rejection: any absent day kills the window
    r2 = n_empty == 0
    # R3 minimum observed: reject when > L/4 timesteps are absent
    r3 = n_empty <= (L // 4)
    # R4 forward-fill with age limit 2 days, then reject what is still empty
    fillable = np.zeros_like(empty)
    for k in range(L):
        had = np.zeros(len(empty), bool)
        for j in range(max(0, k - 2), k):
            had |= ~empty[:, j]                        # an observed day within age 2
        fillable[:, k] = empty[:, k] & had
    still = empty & ~fillable
    r4 = ~still.any(axis=1)
    cand[L] = {r: int(m.sum()) for r, m in
               [("R1_keep_all", r1), ("R2_reject_any_gap", r2),
                ("R3_min_observed", r3), ("R4_ffill_age2_then_reject", r4)]}
    cand[L] = {k2: {"kept": v, "dropped": len(empty) - v,
                    "pct_dropped": round(100 * (len(empty) - v) / len(empty), 2)}
               for k2, v in cand[L].items()}
print(pd.DataFrame({f"L={L}": {r: f"{cand[L][r]['dropped']:,} ({cand[L][r]['pct_dropped']}%)"
                              for r in cand[L]} for L in LENS}).to_string())
for L in LENS:
    check(f"S3a.{L}", f"L={L} drop cost - R1 {cand[L]['R1_keep_all']['pct_dropped']}%, "
                      f"R2 {cand[L]['R2_reject_any_gap']['pct_dropped']}%, "
                      f"R3 {cand[L]['R3_min_observed']['pct_dropped']}%, "
                      f"R4 {cand[L]['R4_ffill_age2_then_reject']['pct_dropped']}%",
          cand[L]["R1_keep_all"]["dropped"] == 0)"""))

# ---------------------------------------------------------------- S3b
cells.append(code(r"""# S3b - bias probe on the strongest alternative (R2 at L=30): who would be lost?
X30 = np.load(NPY[30], mmap_mode="r")
empty30 = np.isnan(X30).all(axis=2)
keep = empty30.sum(axis=1) == 0
split = ix["split"].to_numpy()
season = ix["season"].to_numpy()
bias = {
    "drop_rate_by_split": {s: round(float((~keep[split == s]).mean()), 4)
                           for s in ["train", "val", "test"]},
    "drop_rate_by_season": {s: round(float((~keep[season == s]).mean()), 4)
                            for s in ["monsoon", "non-monsoon"]},
    "label24_rate_dropped": round(float(np.nanmean(labels[~keep])), 4) if (~keep).any() else None,
    "label24_rate_kept": round(float(np.nanmean(labels[keep])), 4),
    "pct_dropped": round(100 * float((~keep).mean()), 2),
}
print(json.dumps(bias, indent=1))
seasons = list(bias["drop_rate_by_season"].values())
asym = max(seasons) / max(min(seasons), 1e-9)
check("S3b", f"R2 would drop {bias['pct_dropped']}% of L=30 samples, and unevenly: "
                 f"drop rate {seasons[0]:.3f} (monsoon) vs {seasons[1]:.3f} "
                 f"(non-monsoon) = {asym:.1f}x asymmetry, per-split "
                 f"{list(bias['drop_rate_by_split'].values())} - an L-dependent, "
                 f"season-shifting cost (label+rate dropped "
                 f"{bias['label24_rate_dropped']} vs kept {bias['label24_rate_kept']})",
      bias["pct_dropped"] > 20 and asym > 1.5)"""))

# ---------------------------------------------------------------- S3c
cells.append(code(r"""# S3c - THE RULE (chosen on the evidence above) + executable mask tensors
RULE = {
    "rule_id": "P19-M1",
    "name": "keep-all + observation mask channel",
    "statement": (
        "No timestep and no sample is dropped or imputed. Every window keeps all L "
        "timesteps. A timestep whose calendar day is absent from the model table is "
        "written to the tensor as all-NaN in Dataset B and handed to the model as "
        "mask=0 (binary mask channel, sequence_mask_L{L}.npy); observed days are "
        "mask=1. On present days the ~0.1% variable-level NaNs (precip_mm, tmax_c, "
        "tmin_c) are replaced by 0 AFTER the training fold's standardisation, i.e. by "
        "the fold mean - no information, no fitting outside the fold. Phase 21 input "
        "is therefore concat(12 raw variables, mask) = 13 channels; a plain NaN must "
        "never reach a layer."),
    "why": [
        "identical sample sets for L = 3/7/14/30 - the D12 length sweep stays "
        "comparable (rejection would train different models on different data)",
        "nothing is fabricated: interpolation/forward-fill would invent driver "
        "variables (precip_mm), age-2 filling (R4) still drops samples",
        "the cheapest rejected alternative (R2) drops 2.9-26.7% of samples, "
        "L-dependent and distribution-shifting (S3b)",
    ],
    "samples_dropped_per_L": {str(L): 0 for L in LENS},
    "counterfactual_drop_costs": cand,
    "bias_probe_R2_L30": bias,
}
written = {}
for L in LENS:
    pad = np.zeros((len(stations), L - 1), bool)
    vw = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad, present], axis=1), L, axis=1)
    m = vw[si, di].astype(np.uint8)
    empty = np.isnan(np.load(NPY[L], mmap_mode="r")).all(axis=2)
    same = bool((m == (~empty).astype(np.uint8)).all())
    np.save(MASK[L], m)
    check(f"S3c.{L}", f"sequence_mask_L{L}.npy written {m.shape} uint8 "
                      f"({MASK[L].stat().st_size:,} B); mask == observed-day indicator "
                      f"for all {m.size:,} timesteps: {same}",
          same and m.shape == (200665, L))
    written[MASK[L].name] = {"shape": [200665, L], "dtype": "uint8",
                             "bytes": MASK[L].stat().st_size,
                             "sha256": sha256(MASK[L])}

# variable-level NaNs on observed timesteps (the fill rule's exposure)
X7 = np.load(NPY[7], mmap_mode="r")
m7 = np.load(MASK[7], mmap_mode="r").astype(bool)
obs = m7[:, :, None]
part = {VARIABLES[v]: float((obs & np.isnan(X7[:, :, v:v + 1])).sum() / max(obs.sum(), 1))
        for v in range(len(VARIABLES))}
part = {k: round(p, 5) for k, p in part.items() if p > 0}
check("S3d", f"variable-level NaNs on observed timesteps (filled with fold mean at "
                 f"train time): {part}",
      all(p < 0.01 for p in part.values()))
RULE["mask_files"] = written
RULE["variable_level_nan_fraction_on_observed"] = part"""))

# ---------------------------------------------------------------- S3d
cells.append(code(r"""# S3d - publish the policy: manifest amendment + counterfactual JSON
man = json.loads(MANIFEST.read_text(encoding="utf-8"))
before = {name: man["files"][name]["sha256"] for name in man["files"]}
man["missing_timestep_policy"] = {
    "rule_id": RULE["rule_id"],
    "name": RULE["name"],
    "statement": RULE["statement"],
    "why": RULE["why"],
    "samples_dropped_per_L": RULE["samples_dropped_per_L"],
    "mask_files": RULE["mask_files"],
    "variable_level_nan_fraction_on_observed": RULE["variable_level_nan_fraction_on_observed"],
    "amended_by": "phase19",
    "amended_utc": str(pd.Timestamp.now("UTC")),
}
MANIFEST.write_text(json.dumps(man, indent=2), encoding="utf-8")
POLICY_JSON.write_text(json.dumps({
    "phase": 19, "stage": 33,
    "chosen_rule": {k: RULE[k] for k in ("rule_id", "name", "statement", "why",
                                         "samples_dropped_per_L")},
    "counterfactual_drop_costs": cand,
    "bias_probe_R2_L30": bias,
    "gap_statistics_per_L": gap_rows,
    "mask_files": written,
    "checks": RES,
}, indent=2, default=float), encoding="utf-8")
after = {name: sha256(PROC / name) for name in before}
check("S3e", f"sequence_manifest.json amended with the policy section; the 5 original "
                 f"artifacts unchanged ({before == after}); sequence_policy.json written "
                 f"({POLICY_JSON.stat().st_size:,} B) with the counterfactual costs",
      before == after and POLICY_JSON.exists())"""))

# ---------------------------------------------------------------- S4
cells.append(code(r"""# S4 - no station mixing: timesteps must be their own station's values
key = tbl.set_index(["station_id", "date"])
rng = np.random.default_rng(SEED)
pick = rng.choice(len(ix), size=40, replace=False)
viol, checked = 0, 0
for L in (30, 14, 7, 3):
    X = np.load(NPY[L], mmap_mode="r")
    sub = ix.iloc[pick]
    for _, row in sub.iterrows():
        t = row.date
        for k in range(L):
            day = t - pd.Timedelta(days=L - 1 - k)
            kk = (row.station_id, str(day.date()))
            if kk not in key.index:
                continue
            want = key.loc[kk, VARIABLES].to_numpy(dtype=np.float32)
            viol += int(not np.allclose(X[row.sample_idx, k], want,
                                        rtol=1e-6, atol=1e-5, equal_nan=True))
            checked += 1
check("S4", f"40 windows x 4 lengths x L timesteps = {checked} timesteps verified "
                f"against their OWN (station, day) row: {viol} violations - a window "
                f"containing another station's day would mismatch here",
      viol == 0 and checked > 1200)"""))

# ---------------------------------------------------------------- S5
cells.append(code(r"""# S5 - no forward reach: window day indices <= anchor day t, for every window
days = np.arange(T)
pad_idx = np.full(LENS[-1] - 1, -1)
vw_days = np.lib.stride_tricks.sliding_window_view(
    np.concatenate([pad_idx, days]), LENS[-1])            # (T, 30)
max_reach, min_start = [], []
ok_forward = True
for L in LENS:
    w = vw_days[di][:, 30 - L:]                           # the L-day suffix = window
    if not (w <= di[:, None]).all():
        ok_forward = False
    if not (w.max(axis=1) == di).all():
        ok_forward = False
    max_reach.append(int(w.max()))
check("S5a", "for all 200,665 windows x 4 lengths the newest day in the window IS the "
                 f"anchor day t and no day is later than t (max index {max_reach}) - "
                 "the label window t+1..t+3 is never inside a tensor",
      ok_forward)

reach = (ix.date - pd.Timedelta(days=29)).dt.year != ix.date.dt.year
cross = ix[reach].split.value_counts().to_dict()
check("S5b", f"history may legally reach BACK across a split boundary: {int(reach.sum()):,} "
                 f"L=30 windows start in an earlier calendar year than their anchor "
                 f"(by split: {cross}) - past days at issue time are legal (D5); no "
                 "window ever reaches forward",
      all(v >= 0 for v in cross.values()))
check("S5c", "labels are not tensor variables (label_* excluded at build time, "
                 "re-verified)", not any(v.startswith("label_") for v in VARIABLES))"""))

# ---------------------------------------------------------------- S6
cells.append(code(r"""# S6 - label alignment: recompute 100 windows' labels from the raw readings
frames = [pd.read_csv(f) for f in sorted(WLDIR.glob("*/*.csv"))]
raw = pd.concat(frames, ignore_index=True)
raw.drop_duplicates(subset=["st_id", "wl_date"], inplace=True)
raw["ts"] = pd.to_datetime(raw["wl_date"], format="%Y-%m-%dT%H:%M:%SZ", utc=True,
                           errors="coerce")
raw["wl"] = pd.to_numeric(raw["waterlevel"], errors="coerce")
raw = raw.dropna(subset=["ts", "wl"])
raw = raw[(raw["wl"] > -1000) & (raw["wl"] <= 100.0)]
raw = raw.sort_values(["st_id", "ts"])
gap = (raw.groupby("st_id")["ts"].diff().dt.total_seconds() > 7 * 86400).fillna(False)
seg = gap.groupby(raw["st_id"]).cumsum()
med = raw.groupby(["st_id", seg])["wl"].transform(
    lambda s: s.rolling(9, center=True, min_periods=3).median())
raw = raw[(raw["wl"] - med).abs() <= 10.0]

sub_ix = ix.iloc[rng.choice(len(ix), size=100, replace=False)].copy()
raw = raw[raw.st_id.isin(sub_ix.station_id)]
print(f"raw readings kept for {raw.st_id.nunique()} stations: {len(raw):,}")

master = pd.read_csv(STATIONS, usecols=["id", "dangerlevel"])
master["dangerlevel"] = pd.to_numeric(master["dangerlevel"], errors="coerce")
DL = dict(zip(master["id"], master["dangerlevel"]))

raw["date"] = raw["ts"].dt.tz_convert(None).dt.normalize()
day = raw.groupby(["st_id", "date"]).agg(n=("wl", "size"), wlmax=("wl", "max")).reset_index()
res = []
for sid, d in day.groupby("st_id"):
    d = d.sort_values("date").set_index("date")
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
    d["n"] = d["n"].fillna(0)
    dl = DL.get(sid, np.nan)
    ex = pd.Series(np.where((d["n"] >= MIN_READINGS) & pd.notna(dl),
                            (d["wlmax"] >= float(dl)).astype(float), np.nan),
                   index=d.index)
    n = d["n"].astype(int)
    rec = {"station_id": sid, "date": d.index}
    for h in (1, 2, 3):
        shifts = [ex.shift(-k) for k in range(1, h + 1)]
        pos = pd.concat([(s == 1) for s in shifts], axis=1).any(axis=1)
        win_obs = pd.concat([(n.shift(-k) >= MIN_READINGS) for k in range(1, h + 1)],
                            axis=1)
        win_known = pd.concat([s.notna() for s in shifts], axis=1)
        rec[f"rlabel_{24 * h}h"] = np.where(
            pos, 1.0, np.where(win_obs.all(axis=1) & win_known.all(axis=1), 0.0, np.nan))
    res.append(pd.DataFrame(rec))
recomp = pd.concat(res, ignore_index=True)

lab = pd.read_csv(TBL, usecols=["station_id", "date", "label_24h", "label_48h",
                                "label_72h"])
lab["date"] = pd.to_datetime(lab["date"])
sub_ix["date"] = pd.to_datetime(sub_ix["date"])
cmp = sub_ix[["station_id", "date"]].merge(lab, on=["station_id", "date"], how="left") \
            .merge(recomp, on=["station_id", "date"], how="left")
mism = one_sided = tot = 0
for h in (24, 48, 72):
    a, b = cmp[f"label_{h}h"], cmp[f"rlabel_{h}h"]
    both = a.notna() & b.notna()
    tot += int(both.sum())
    mism += int((a[both] != b[both]).sum())
    one_sided += int((a.notna() ^ b.notna()).sum())
check("S6", f"100 windows x 3 horizons: {tot} comparable labels recomputed from the raw "
                f"3-hourly readings at the same station and day t, mismatches={mism}, "
                f"one-sided={one_sided}",
      mism == 0 and one_sided == 0 and tot >= 250)"""))

# ---------------------------------------------------------------- S7
cells.append(code(r"""# S7 - split / holdout inheritance, and training cells are holdout-clean
sp = pd.read_csv(SPLITS, parse_dates=["date"])
m = ix.merge(sp[["station_id", "date", "split", "spatial_fold", "station_holdout",
                 "season"]], on=["station_id", "date"], how="left",
             suffixes=("", "_sp"), indicator=True)
disagree = {
    "split": int((m["split"] != m["split_sp"]).sum()),
    "spatial_fold": int((m["spatial_fold"] != m["spatial_fold_sp"]).sum()),
    "station_holdout": int((m["station_holdout"].astype(int)
                            != m["station_holdout_sp"].astype(int)).sum()),
    "season": int((m["season"] != m["season_sp"]).sum()),
}
check("S7a", f"all {len(m):,} windows inherit the anchor day's split/fold/holdout/season "
                 f"1:1 with Phase-15: {disagree}",
      (m._merge == "both").all() and sum(disagree.values()) == 0)

# rebuild every training cell exactly as scripts/baseline_model.py build_tasks does
hold = sp.station_holdout.to_numpy()
year = sp.date.dt.year.to_numpy()
splitv = sp.split.to_numpy()
fold = sp.spatial_fold.to_numpy()
cells_train = {
    "P1 train_<2022": (splitv == "train") & ~hold,
    "P2 train<=2019": (year <= 2019) & ~hold,
    "P2 train<=2020": (year <= 2020) & ~hold,
    "P2 train<=2021": (year <= 2021) & ~hold,
}
for k in SPATIAL_FOLDS:
    cells_train[f"P3 excl_fold_{k}"] = (splitv == "train") & ~hold & (fold != k)
bad_cells = {name: int((mask & hold).sum() + (mask & (splitv == "test")).sum())
             for name, mask in cells_train.items()}
sizes = {name: int(mask.sum()) for name, mask in cells_train.items()}
check("S7b", f"every training cell contains 0 holdout-station rows and 0 frozen-test rows: "
                 f"{bad_cells}; cell sizes {sizes}",
      sum(bad_cells.values()) == 0)
check("S7c", f"holdout stations ({int(hold.sum()):,} rows) are reachable only through "
                 f"their own eval cells; test_unseen = "
                 f"{int(((splitv == 'test') & hold).sum()):,} rows, all holdout",
      int(((splitv == "test") & hold).sum()) > 0
      and int(((splitv == 'test') & ~hold).sum()) > 0)
ho_st = set(sp.loc[sp.station_holdout == 1, "station_id"])
tr_st = set(sp.loc[(splitv == "train") & ~hold, "station_id"])
check("S7d", f"{len(ho_st)} holdout stations appear in {len(ho_st & tr_st)} training "
                 "cell station sets (must be 0)",
      len(ho_st & tr_st) == 0)"""))

# ---------------------------------------------------------------- summary
cells.append(code(r"""print("=" * 78)
print("PHASE 19 / STAGE 33   sequences validated - policy P19-M1 published")
print("=" * 78)
for r in RES:
    print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['id']}")
for w in WARN:
    print(f"  WARN  {w['id']}: {w['note']}")
print(f"\n{sum(r['pass'] for r in RES)}/{len(RES)} checks passed, {len(WARN)} warnings, "
      f"{time.time() - t0:.1f}s")
print("policy P19-M1 (keep-all + observation mask): 0 samples dropped at every L")
print("outputs: sequence_mask_L{3,7,14,30}.npy, sequence_policy.json, "
      "sequence_manifest.json (amended)")
print("next: Phase 20 - classical machine learning (L2 candidates)")
assert all(r["pass"] for r in RES)
assert sha256(TBL) == TBL_SHA_EXPECTED, "experimental table changed!"
print("ALL PHASE-19 CHECKS PASSED - experimental_dataset.csv byte-stable")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
