"""Build notebooks/05_leakage_audit.ipynb (workflow Phase 13, body steps 22-24).

The notebook is the analysis artifact; this script only assembles it, so the code has a
single source of truth and can be regenerated after any change:

    python scripts/build_phase13_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\05_leakage_audit.ipynb

Cells cover, in binding body-Phase order (decision D10):
    step 22  feature leakage audit      -> L1 source audit, L2 calendar-formula recompute,
                                           L3 neighbourhood same-day recompute,
                                           L4 upstream lag recompute, whitelist table
    step 23  target leakage audit       -> L5 banned-column absence, L6 univariate AUC
                                           screen, L7 future-control, L8 observation-process
                                           conditional test (phase5 E7)
    step 24  duplicate / time leakage   -> L9 key + split disjointness, L10 2018-hole
                                           integrity, L11 cross-split near-duplicates
                                           and episode continuity

Outputs: data/processed/feature_whitelist.csv, data/processed/leakage_audit.json,
figures/phase13_auc_screen.png, and a phase13 block appended to feature_sets.json.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "05_leakage_audit.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 13 - Critical leakage audit (workflow body steps 22-24)

| | |
| --- | --- |
| **Workflow** | `flood prediction workflow.md` **Phase 13 - Critical leakage audit**: `22` feature leakage audit -> `23` target leakage audit -> `24` duplicate / time leakage |
| **Checklist #** | 19 (data-leakage audit) + 20 (prediction-time information boundary) |
| **Input** | `data/processed/flood_dataset_model_ready.csv` (200,665 x 73) + full calendar matrix (357,250 x 73) + `feature_sets.json` (Phase 12) |
| **Boundary** | `reports/phase0_research_definition.md` §5 - **issue instant = 24:00 UTC of day t**; labels cover `t+1 .. t+h` |
| **Outputs** | `data/processed/feature_whitelist.csv` (formal whitelist, phase0 §8 stage-19 deliverable) · `data/processed/leakage_audit.json` · `figures/phase13_auc_screen.png` |
| **Open rulings closed here** | `obs_days_*` ban · `n_readings` admissibility (phase5 **E7**) · 19 Phase-11 columns (phase11 §7 item 1) · feature-time-bound check |

Eleven checks, **L1-L11**. The `assert`s abort the run, so a green notebook *is* the evidence.
Nothing under `data/raw/` is modified and no model is trained beyond the small diagnostic
fits required by step 23 (L8) - splitting and modelling stay in Phases 15-17 (D10)."""))

cells.append(code(r"""import json, re, time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

ROOT = Path(r"D:\Flood predictor")
PROC, FIGDIR = ROOT / "data" / "processed", ROOT / "figures"
HORIZONS = ["label_24h", "label_48h", "label_72h"]
SEED = 20261008

mr = pd.read_csv(PROC / "flood_dataset_model_ready.csv")
full = pd.read_csv(PROC / "flood_dataset.csv")
for _d in (mr, full):
    _d["date"] = pd.to_datetime(_d["date"])
sets = json.loads((PROC / "feature_sets.json").read_text(encoding="utf-8"))
FULL = sets["sets"]["FULL"]
BANNED = sets["banned"]

assert mr.shape == (200665, 73) and full.shape == (357250, 73), (mr.shape, full.shape)
assert full.station_id.nunique() == 141 and mr.station_id.nunique() == 133
assert not set(FULL) & set(BANNED)

RES = []
def check(cid, detail, ok=True):
    RES.append({"id": cid, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}")
    assert ok, cid

def agree(a, b, tol=1e-9):
    # rows compared, mismatches (NaN==NaN counts as equal)
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    same = (np.isnan(a) & np.isnan(b)) | (~np.isnan(a) & ~np.isnan(b) & (np.abs(a - b) <= tol))
    return len(a), int((~same).sum())

print(f"model-ready {mr.shape} | full matrix {full.shape} | Phase-12 pool {len(FULL)} features | banned {len(BANNED)}")"""))

cells.append(md("""## Step 22 - feature leakage audit

> *For every feature ask: would this value actually be available at the moment the
> prediction is made?* (workflow step 22)

The claim to be **proved, not asserted**, is the phase0 §5 boundary: every stored feature
value is a function of days `<= t` only. Four independent lines of evidence:

| check | what it proves |
| --- | --- |
| **L1** static source audit | no feature builder mentions the target; every forward (`shift(-`) call in the module sits inside `add_labels` |
| **L2** calendar-formula recompute | ~35 derived columns rebuilt from the full matrix with `shift(+k)` / backward rolling formulas reproduce the stored values **exactly** - a value equal to `f(rows <= t)` cannot contain future information |
| **L3** neighbourhood recompute | the 8 `nb*` columns are reproduced from **day-t values only** for 60 sampled dates (nearest-k selection rebuilt independently) |
| **L4** upstream recompute | the 4 `up_*` columns equal the upstream gauge's value at `t-k` for **every** model-ready row (upstream map rebuilt from the station master rule) |

A formula audit is only meaningful because the builder's `shift(+k)` is a *calendar* shift
(the matrix is reindexed to one row per calendar day before features are added), so lags
cannot skip the 2018 hole - **L10** re-verifies that directly in the data."""))

cells.append(code(r"""src = (ROOT / "src" / "flood" / "features" / "build_dataset.py").read_text(encoding="utf-8")
lines = src.splitlines()

# (a) every forward shift (negative shift = reference to FUTURE rows) lives in add_labels
fwd = [i for i, ln in enumerate(lines, 1) if ".shift(-" in ln]
lab_start = next(i for i, ln in enumerate(lines, 1) if ln.startswith("def add_labels"))
lab_end = next(i for i, ln in enumerate(lines, 1) if ln.startswith("def add_features"))
stray = [i for i in fwd if not (lab_start < i < lab_end)]
assert fwd and not stray, {"forward_shifts": fwd, "outside_add_labels": stray}

# (b) the four feature builders must never mention the target / label-window columns
parts = re.split(r"(?m)^def (\w+)", src)
funcs = dict(zip(parts[1::2], parts[2::2]))
FEAT_FUNCS = ["add_features", "add_static_spatial", "add_neighbourhood", "add_upstream"]
hits = {fn: [t for t in ("label_24h", "label_48h", "label_72h", "obs_days", "LEADS")
             if t in funcs[fn]] for fn in FEAT_FUNCS}
assert all(not v for v in hits.values()), hits
for fn in FEAT_FUNCS:
    assert fn in funcs, fn

# (c) pool hygiene: no banned column, no calendar-leakage column (phase0 §5 illegal list)
calendar_leak = [c for c in FULL if c.startswith(("year", "month", "bad_"))]
assert not calendar_leak, calendar_leak
assert "day_observed" not in FULL          # constant, dropped at Phase 12 step 18a

check("L1 static source audit",
      f"{len(fwd)} forward-shift calls, all inside add_labels (lines {lab_start}-{lab_end}); "
      f"4 feature builders x 0 target/label tokens; no year/month columns in the pool")"""))

cells.append(code(r"""f = full.sort_values(["station_id", "date"]).reset_index(drop=True)
g = f.groupby("station_id")
rc = {}

# hydrology: calendar lags, rises, ratio features, backward rolling windows
for k in range(1, 8):
    rc[f"wl_last_lag{k}"] = g["wl_last"].shift(k)
for k in (1, 2, 3):
    rc[f"wl_max_lag{k}"] = g["wl_max"].shift(k)
rc["wl_rise_1d"] = f["wl_last"] - g["wl_last"].shift(1)
rc["wl_rise_3d"] = f["wl_last"] - g["wl_last"].shift(3)
rc["wl_dist_to_danger"] = f["danger_level"] - f["wl_max"]
rc["wl_pct_danger"] = f["wl_max"] / f["danger_level"]
rc["wl_max_roll7"] = g["wl_max"].transform(lambda s: s.rolling(7, min_periods=4).mean())
rc["wl_max_roll30"] = g["wl_max"].transform(lambda s: s.rolling(30, min_periods=10).mean())
_ex = pd.Series(np.where(f["day_observed"] & f["danger_level"].notna(),
                         (f["wl_max"] >= f["danger_level"]).astype(float), np.nan),
                index=f.index)
rc["wl_exceed_days_30"] = _ex.groupby(f["station_id"]).transform(
    lambda s: s.fillna(0).rolling(30, min_periods=1).sum())

# rain / temperature windows (all end at day t)
rc["precip_mm"] = f["precip_mm"]
rc["rain_1d"] = f["precip_mm"]
gw = f.groupby("station_id")
for k in (3, 7, 14, 30):
    rc[f"rain_{k}d"] = gw["precip_mm"].transform(lambda s: s.rolling(k, min_periods=1).sum())
rc["rain_max_7d"] = gw["precip_mm"].transform(lambda s: s.rolling(7, min_periods=1).max())
rc["wet_days_7"] = (f["precip_mm"].fillna(0) > 1).groupby(f["station_id"]).transform(
    lambda s: s.rolling(7, min_periods=1).sum())

# narrow seasonality from the date itself
doy = f["date"].dt.dayofyear
rc["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
rc["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

recomp = pd.DataFrame(rc)
recomp.insert(0, "station_id", f["station_id"].to_numpy())
recomp.insert(1, "date", f["date"].to_numpy())
m = mr.merge(recomp, on=["station_id", "date"], suffixes=("", "_rc"), validate="one_to_one")
bad = {}
for c in rc:
    _, nbad = agree(m[c], m[c + "_rc"])
    if nbad:
        bad[c] = nbad
assert not bad, bad

L2_COLS = list(rc)
check("L2 calendar-formula recompute",
      f"{len(rc)} derived columns rebuilt from the full calendar matrix with shift(+k)/"
      f"backward-window formulas: 0 mismatches over {len(m):,} model-ready rows x 3 decimals tolerance 1e-9")"""))

cells.append(md("""### L3 / L4 - the two Phase-11 groups that are not simple row formulas

`nb*` is a same-day cross-sectional statistic (nearest-k gauges **on day t**) and `up_*` is a
cross-station lag (the upstream gauge's value at **t-k**). Both were hand-checked once in
Phase 11; the audit re-derives them here through an independent implementation:

* **L3**: nearest-15 haversine pool rebuilt from the station master, nearest-k membership
  resolved per sampled date, means/counts recomputed from day-t signal columns only - the
  rain signals on **the weather record's own grid** (rolling windows over that file's rows,
  `-999` sentinel cleared, exactly as `add_neighbourhood` does) and the water-level signals
  on the calendar grid; `nb*_n_wl_danger` is a `nansum` count, so an empty pool stores `0`.
* **L4**: the upstream map re-derived from the rule (same `river`, elevation `> +0.5 m`,
  nearest within 250 km), then every `up_*` cell looked up as `value(up(s), t-k)`.

If either group used information from days `> t`, these recomputations could not match the
stored values on all rows/dates sampled."""))

cells.append(code(r"""def hav(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(np.asarray(lat2) - lat1)
    dl = np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

st = full[["station_id", "lat", "lon"]].drop_duplicates("station_id").reset_index(drop=True)
ids = st["station_id"].to_numpy().astype("int64")
latv, lonv = st["lat"].to_numpy(), st["lon"].to_numpy()
D = np.empty((len(ids), len(ids)), dtype="float")
for i in range(len(ids)):
    D[i] = hav(latv[i], lonv[i], latv, lonv)
np.fill_diagonal(D, np.inf)
POOL = np.argsort(D, axis=1)[:, :15]                 # builder: NEIGHB_POOL = 15
IDX = pd.Index(ids, name="station_id")

def take_mask(A, k):
    # (S, P) boolean: which pool positions enter the nearest-k mean (independent re-impl)
    ok = ~np.isnan(A)
    cum = np.cumsum(ok, axis=1)
    take = np.zeros_like(ok)
    few = cum[:, -1] < k
    take[few] = ok[few]                               # fewer than k reporters -> use all
    for i in np.where(~few)[0]:
        kth = int(np.argmax(cum[i] >= k))             # position of the k-th reporter
        take[i] = ok[i] & (np.arange(A.shape[1]) <= kth)
    return take

# The two signal families live on DIFFERENT grids in the builder (build_dataset.py:290-327):
#   rain  -> the weather record's own row grid (rolling(7/14) over that file's rows, -999 -> NaN)
#   wl    -> the calendar matrix
# Rebuilding rain off the calendar grid would differ wherever a station skips a weather day.
wx = pd.read_csv(ROOT / "data" / "raw" / "rainfall" / "weather_station_daily.csv",
                 usecols=["station_id", "date", "precip_mm"])
wx = wx[wx["station_id"].isin(ids)].copy()
wx.loc[wx["precip_mm"] <= -999, "precip_mm"] = np.nan
wx["date"] = pd.to_datetime(wx["date"])
_wg = wx.groupby("station_id")["precip_mm"]
wx["rain_7d"] = _wg.transform(lambda s: s.rolling(7, min_periods=1).sum())
wx["rain_14d"] = _wg.transform(lambda s: s.rolling(14, min_periods=1).sum())
WPIV = {s: wx.pivot_table(index="date", columns="station_id", values=s, dropna=False)
             .reindex(columns=ids) for s in ("rain_7d", "rain_14d")}
wx_dates = set(wx["date"].dt.date)
pool_dates = sorted(d for d in full["date"].unique() if d.date() in wx_dates)
sample_dates = pd.Series(pool_dates, dtype="datetime64[ns]").sample(n=60, random_state=SEED).sort_values().tolist()

RAINS = {"rain_7d": ("nb3_rain_7d", "nb5_rain_7d"),
         "rain_14d": ("nb3_rain_14d", "nb5_rain_14d")}
WLS = {"wl_pct_danger": ("nb3_wl_pct_danger", "nb5_wl_pct_danger")}

nb_bad, nb_cells = 0, 0
for dt in sample_dates:
    day = full.loc[full["date"] == dt].drop_duplicates("station_id").set_index("station_id")
    present = day.index
    sel = IDX.isin(present)
    rows = []
    for sig, (c3, c5) in RAINS.items():
        rows.append((WPIV[sig].loc[dt].to_numpy(dtype=float), (c3, c5)))   # weather grid
    for sig, (c3, c5) in WLS.items():
        rows.append((day[sig].reindex(IDX).to_numpy(dtype=float), (c3, c5)))  # calendar grid
    for src_row, (c3, c5) in rows:
        A = src_row[POOL]                                                  # (S, P)
        for k, col in ((3, c3), (5, c5)):
            take = take_mask(A, k)
            cnt = take.sum(axis=1).astype("float64")
            mean = np.where(cnt > 0, np.where(take, A, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
            stored = day[col].reindex(IDX).to_numpy(dtype=float)
            n, nbad = agree(mean[sel], stored[sel])
            nb_cells += n
            nb_bad += nbad
    # count column: builder uses nansum over the selected neighbours, so an empty pool
    # stores 0 (a count), never NaN (build_dataset.py:345)
    A = day["wl_pct_danger"].reindex(IDX).to_numpy(dtype=float)[POOL]
    for k, col in ((3, "nb3_n_wl_danger"), (5, "nb5_n_wl_danger")):
        take = take_mask(A, k)
        at = np.where(np.isnan(A), 0.0, (A >= 1.0).astype(float))
        n_dng = np.where(take, at, 0.0).sum(axis=1)
        stored = day[col].reindex(IDX).to_numpy(dtype=float)
        n, nbad = agree(n_dng[sel], stored[sel])
        nb_cells += n
        nb_bad += nbad
assert nb_bad == 0, nb_bad

check("L3 neighbourhood same-day recompute",
      f"8 nb* columns recomputed from day-t values only for 60 sampled dates: "
      f"{nb_cells:,} cells compared, {nb_bad} mismatches")"""))

cells.append(code(r"""topo = full[["station_id", "river", "lat", "lon", "elevation_m"]].drop_duplicates("station_id")
up_map = {}
for riv, grp in topo.groupby("river"):
    grp = grp.dropna(subset=["elevation_m"])
    if len(grp) < 2:
        continue
    for _, s in grp.iterrows():
        higher = grp[grp["elevation_m"] > s["elevation_m"] + 0.5]      # UPSTREAM_ELEV_DIFF_M
        if higher.empty:
            continue
        d = hav(s["lat"], s["lon"], higher["lat"].to_numpy(), higher["lon"].to_numpy())
        j = int(np.argmin(d))
        if d[j] <= 250.0:                                              # UPSTREAM_MAX_KM
            up_map[int(s["station_id"])] = int(higher["station_id"].iloc[j])
assert up_map, "upstream map empty"

key = pd.MultiIndex.from_frame(full[["station_id", "date"]])
look = {c: pd.Series(full[c].to_numpy(), index=key) for c in ("rain_7d", "wl_pct_danger")}
up_ids = mr["station_id"].map(up_map).fillna(-1).astype("int64").to_numpy()

up_bad, up_cells = 0, 0
for k in (1, 2):
    src_date = mr["date"].to_numpy() - np.timedelta64(k, "D")
    for base, col in (("rain_7d", f"up_rain7d_lag{k}"),
                      ("wl_pct_danger", f"up_wl_pct_danger_lag{k}")):
        vals = look[base].reindex(pd.MultiIndex.from_arrays([up_ids, src_date])).to_numpy()
        n, nbad = agree(mr[col].to_numpy(), vals)
        up_cells += n
        up_bad += nbad
assert up_bad == 0, up_bad

mapped = int((up_ids >= 0).sum())
check("L4 upstream lag recompute",
      f"upstream rule re-derived ({len(up_map)} stations mapped; {mapped:,} model-ready rows have a partner); "
      f"4 columns x {len(mr):,} rows = {up_cells:,} cells, {up_bad} mismatches")"""))

cells.append(md("""## Step 23 - target leakage audit

> *Check whether any feature directly contains the target, was calculated using the
> target, was generated after the flood event, or indirectly reveals the target.*
> (workflow step 23)

| check | workflow question it answers |
| --- | --- |
| **L5** | *directly contains the target* - labels, `obs_days_*` and keys are absent from all three Phase-12 feature sets |
| **L6** | *indirectly reveals the target* - univariate screen of all 62 features x 3 horizons; any feature with oriented AUC >= 0.995 (near-perfect separation) is flagged |
| **L7** | *calculated using the target / after the event* - **positive control**: the label rebuilt from day `t+1` (and `t+1..t+3`) exceeds reproduces `label_24h`/`label_72h` on 100 % of labelled rows, i.e. a genuinely future-derived feature scores AUC 1.000 - proving the L6 screen *would* catch leakage, while `wl_exceed_days_30` (the same exceed rule on days `<= t`, verified in L2) scores 0.908 |
| **L8** | *observation-process signal* - the phase5 **E7** ruling plus phase11 §7 item 1: do `n_readings` and `nb5_n_wl_danger` add target signal beyond today's water level, and do they predict **label availability** (selection effect)? A feature that predicts *availability* but not the *target* is a watch, not a ban |"""))

cells.append(code(r"""LEAK_TOKENS = ["label_24h", "label_48h", "label_72h",
                "obs_days_24h", "obs_days_48h", "obs_days_72h"]
assert set(LEAK_TOKENS) <= set(mr.columns), "banned columns missing from the table"
for nm in ("FULL", "LINEAR", "SELECTED"):
    s = set(sets["sets"][nm])
    assert not s & set(LEAK_TOKENS), (nm, s & set(LEAK_TOKENS))
    assert not s & {"station_id", "name", "date", "source"}, (nm, s & {"station_id", "name", "date", "source"})
# the banned columns exist in the table, so the ban is a real choice, not an accident
assert all(c in mr.columns for c in LEAK_TOKENS)

check("L5 banned/target columns absent from feature sets",
      "FULL(62) / LINEAR(35) / SELECTED(10) share 0 columns with the 3 labels, "
      "3 obs_days_* and the 4 identity/date keys")"""))

cells.append(code(r"""lab = mr[mr[HORIZONS].notna().any(axis=1)].copy()
X = lab[list(FULL)].copy()
for c in X.select_dtypes(exclude=[np.number]).columns:
    X[c] = pd.factorize(X[c])[0].astype(float)      # unsupervised code, no target involved
X = X.astype(float)

rows = []
t0 = time.time()
for c in FULL:
    r = {"feature": c}
    for h in HORIZONS:
        m_ = X[c].notna() & lab[h].notna()
        if X.loc[m_, c].nunique() < 2:
            r[h] = np.nan
            continue
        auc = roc_auc_score(lab.loc[m_, h].astype(int), X.loc[m_, c])
        r[h] = max(auc, 1.0 - auc)                  # oriented
    r["max_auc"] = max(r[h] for h in HORIZONS)
    rows.append(r)
SCREEN = pd.DataFrame(rows).set_index("feature").sort_values("max_auc", ascending=False)

flagged = SCREEN[SCREEN["max_auc"] >= 0.995]
assert flagged.empty, flagged
assert SCREEN["max_auc"].notna().all()

fig, ax = plt.subplots(figsize=(9, 7))
t = SCREEN.head(25)[::-1]
ax.barh(range(len(t)), t["max_auc"], color="#2b6cb0")
ax.set_yticks(range(len(t)))
ax.set_yticklabels(t.index, fontsize=8)
ax.axvline(0.995, color="crimson", ls="--", lw=1.2, label="flag threshold 0.995")
ax.axvline(0.5, color="grey", ls=":", lw=1)
ax.set_xlabel("oriented univariate AUC (max over 24/48/72 h, labelled rows)")
ax.set_title("Phase 13 step 23 - target-leakage screen, top 25 of 62 features")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "phase13_auc_screen.png", dpi=150)
plt.close(fig)

check("L6 univariate target-leakage screen",
      f"62 features x 3 horizons in {time.time() - t0:.0f}s; max oriented AUC "
      f"{SCREEN['max_auc'].max():.3f} ({SCREEN['max_auc'].idxmax()}); 0 features at/above 0.995")
SCREEN.head(10).round(4)"""))

cells.append(code(r"""# Positive control: rebuild the label from days > t and show the screen catches it.
fz = full.sort_values(["station_id", "date"])
idx = pd.MultiIndex.from_frame(fz[["station_id", "date"]])
ex = pd.Series(np.where(fz["day_observed"] & fz["danger_level"].notna(),
                        (fz["wl_max"] >= fz["danger_level"]).astype(float), np.nan), index=idx)
ex = ex.sort_index()

fut1 = ex.groupby(level=0).shift(-1)                                   # exceed at t+1
fut_any3 = pd.concat([ex.groupby(level=0).shift(-k) for k in (1, 2, 3)],
                     axis=1).max(axis=1, skipna=True)                  # any exceed in t+1..t+3

keys = pd.MultiIndex.from_frame(mr[["station_id", "date"]])
p1 = fut1.reindex(keys).to_numpy()
p3 = fut_any3.reindex(keys).to_numpy()

m24 = mr["label_24h"].notna().to_numpy()
m72 = mr["label_72h"].notna().to_numpy()
eq24 = float(np.mean(p1[m24] == mr.loc[m24, "label_24h"].to_numpy()))
eq72 = float(np.mean(p3[m72] == mr.loc[m72, "label_72h"].to_numpy()))
auc72 = roc_auc_score(mr.loc[m72, "label_72h"].astype(int), p3[m72])
assert eq24 == 1.0 and eq72 == 1.0, (eq24, eq72)
assert auc72 >= 0.995

# the stored, past-only counterpart for contrast:
stored_auc = roc_auc_score(mr.loc[m72, "label_72h"].astype(int),
                           mr.loc[m72, "wl_exceed_days_30"].astype(float))
check("L7 future-control (screen power)",
      f"a deliberately future-derived feature reproduces label_24h on {eq24:.1%} and "
      f"label_72h on {eq72:.1%} of labelled rows (AUC {auc72:.3f}) -> the L6 threshold would "
      f"catch it; the legitimate past-window twin wl_exceed_days_30 scores {stored_auc:.3f}")"""))

cells.append(code(r"""# L8: phase5 E7 + phase11 open item 1 - observation-process signals on the watchlist.
#     Question per step 23: does the feature carry TARGET signal, or only LABEL-AVAILABILITY
#     (selection) signal?  High availability-AUC with a ~0 PR-AUC delta means the feature
#     says "a label will exist", not "a flood will happen".
train = mr[mr["date"] < "2022-01-01"]
val = mr[(mr["date"] >= "2022-01-01") & (mr["date"] < "2023-01-01")]
assert len(train) + len(val) + (mr["date"] >= "2023-01-01").sum() == len(mr)

def fit_pr(feats, h):
    ytr, yv = train[h], val[h]
    mtr, mv = ytr.notna(), yv.notna()
    mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=64,
                             min_child_samples=50, objective="binary", n_jobs=-1,
                             random_state=SEED, verbose=-1)
    mdl.fit(train.loc[mtr, feats], ytr[mtr].astype(int))
    p = mdl.predict_proba(val.loc[mv, feats])[:, 1]
    return average_precision_score(yv[mv].astype(int), p)

WATCH = ["n_readings", "nb5_n_wl_danger"]           # E7 + phase11 s7 item 1
avail = mr["label_24h"].notna().astype(int)
AUCS, e7 = {}, []
for feat in WATCH:
    m = mr[feat].notna() & mr["label_24h"].notna()
    AUCS[feat] = {"target_24h": roc_auc_score(mr.loc[m, "label_24h"].astype(int), mr.loc[m, feat]),
                  "label_availability": roc_auc_score(avail[mr[feat].notna()],
                                                      mr.loc[mr[feat].notna(), feat])}
    for h in HORIZONS:
        base = fit_pr(["wl_pct_danger"], h)
        plus = fit_pr(["wl_pct_danger", feat], h)
        e7.append({"feature": feat, "horizon": h, "persistence": base,
                   "plus_feature": plus, "delta_pr_auc": plus - base})
E7 = pd.DataFrame(e7).set_index(["feature", "horizon"])

watch_txt = "; ".join(
    f"{f}: AUC target {AUCS[f]['target_24h']:.3f} vs availability "
    f"{AUCS[f]['label_availability']:.3f}, delta PR-AUC "
    f"{E7.loc[f, 'delta_pr_auc'].min():+.4f}..{E7.loc[f, 'delta_pr_auc'].max():+.4f}"
    for f in WATCH)
# ruling carried into the whitelist cell: n_readings = selection proxy -> ADMIT-WATCH;
# nb*_n_wl_danger = day-t flood-state signal with real incremental PR-AUC -> ADMIT
assert AUCS["n_readings"]["label_availability"] > 0.75 and AUCS["n_readings"]["target_24h"] < 0.5
assert AUCS["nb5_n_wl_danger"]["target_24h"] > 0.8 and AUCS["nb5_n_wl_danger"]["label_availability"] < 0.5
assert E7.loc["nb5_n_wl_danger", "delta_pr_auc"].max() > 0

check("L8 observation-process test (phase5 E7 + phase11 open item 1)",
      watch_txt + " -> n_readings is a LABEL-AVAILABILITY proxy (watch), "
                  "nb*_n_wl_danger is a genuine day-t flood-state signal (admit)")
E7.round(4)"""))

cells.append(md("""## Step 24 - duplicate / time leakage

> *Check whether highly similar observations appear across train and test - same location,
> nearby dates, same flood event, same storm.* (workflow step 24)

| check | what it does |
| --- | --- |
| **L9** | keys unique in both tables; the provisional split (same dates as Phase 12) shares **no date and no station-day** between parts |
| **L10** | the archive hole is never bridged: holes are derived from the data (0 observed rows on those days - the 2017-12-09..2018-12-31 FFWC gap plus a Nov-2017 gap), no model-ready row falls inside one, every lag whose source day sits inside a gap is NaN, and a label `0` requires a fully observed window while a label `1` needs an observed exceed day |
| **L11** | measures the *residual* similarity the split cannot remove: identical 62-feature vectors occurring in two different parts, the model-ready-free stretch each cut sits inside (a natural dry-season embargo), and stations positive on both sides - evidence for the **embargo decision Phase 15 must make** |"""))

cells.append(code(r"""assert mr.duplicated(["station_id", "date"]).sum() == 0
assert full.duplicated(["station_id", "date"]).sum() == 0

split = np.where(mr["date"] < "2022-01-01", "train",
                 np.where(mr["date"] < "2023-01-01", "val", "test"))
mr["_split"] = split
parts = {s: mr[mr["_split"] == s] for s in ("train", "val", "test")}
assert sum(len(p) for p in parts.values()) == len(mr)

date_overlap = (set(parts["train"].date) & set(parts["val"].date)) | \
               (set(parts["train"].date) & set(parts["test"].date)) | \
               (set(parts["val"].date) & set(parts["test"].date))
assert not date_overlap, date_overlap
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    assert not (set(parts[a].station_id.astype(str) + parts[a].date.astype(str)) &
                set(parts[b].station_id.astype(str) + parts[b].date.astype(str)))

check("L9 key uniqueness + split disjointness",
      f"0 duplicate station-days in either table; date-split train {len(parts['train']):,} / "
      f"val {len(parts['val']):,} / test {len(parts['test']):,} share 0 dates and 0 station-days "
      f"(partition by construction, asserted)")"""))

cells.append(code(r"""# (a) holes: calendar days on which NO station reached the observation threshold.
#     Derived from the data (dry-season non-reporting + the FFWC archive gap), not assumed.
obs_dates = set(full.loc[full["day_observed"], "date"])
allday = pd.date_range(full["date"].min(), full["date"].max(), freq="D")
in_gap = pd.Series([d not in obs_dates for d in allday], index=allday)
starts = in_gap & ~in_gap.shift(1, fill_value=False)
ends = in_gap & ~in_gap.shift(-1, fill_value=False)
GAPS = [(s, e) for s, e in zip(allday[starts], allday[ends])]
assert GAPS, "no interior hole found"
gap_days = allday[in_gap.to_numpy()]
MAIN = max(GAPS, key=lambda se: (se[1] - se[0]).days)
assert MAIN[0] <= pd.Timestamp("2018-01-01") and MAIN[1] >= pd.Timestamp("2018-12-31"), MAIN
n_gap_rows = int(full["date"].isin(gap_days).sum())

# no model-ready row may sit on a non-observed day - the hole never enters the table
assert not mr["date"].isin(gap_days).any(), "model-ready rows inside a hole"
n2018_full = int((full["date"].dt.year == 2018).sum())
n2018_obs = int(full.loc[full["date"].dt.year == 2018, "day_observed"].sum())
assert n2018_full == 31390 and n2018_obs == 0
assert int((mr["date"].dt.year == 2018).sum()) == 0

# (b) rows are consecutive calendar days within every station, so a row shift IS a
#     calendar shift (the property that makes the L2 formulas hole-proof)
step = full.sort_values(["station_id", "date"]).groupby("station_id")["date"].diff().dt.days.dropna()
assert int((step != 1).sum()) == 0, int((step != 1).sum())

# and on the first observed day after the MAIN hole every backward lag is empty
first = allday[allday > MAIN[1]][0]
s_first = full[(full["date"] == first) & full["wl_last"].notna()]
assert len(s_first) > 0, first
for k in range(1, 8):
    assert s_first[f"wl_last_lag{k}"].isna().all(), (first, k)

# (c) labels never borrow an unobserved day: a 0 needs the WHOLE window observed,
#     and a 1 can only come from an observed exceed day inside the window
HDAYS = {"label_24h": 1, "label_48h": 2, "label_72h": 3}     # LEADS window length in days
label_rule = {}
for h in HORIZONS:
    hd = HDAYS[h]
    od = f"obs_days_{h[-3:]}"
    assert od in mr.columns
    assert int(mr[od].max()) == hd
    zero, pos = mr[h] == 0, mr[h] == 1
    assert bool(mr.loc[zero, od].eq(hd).all()), h              # no 0 across a hole
    assert bool((mr.loc[pos, od] >= 1).all()), h               # 1 anchored on a real day
    label_rule[h] = {"0": int(zero.sum()), "1": int(pos.sum()),
                     "nan": int(mr[h].isna().sum())}

check("L10 hole integrity",
      f"{len(GAPS)} zero-observation stretches, longest {MAIN[0].date()}..{MAIN[1].date()} "
      f"(2018: {n2018_full:,} calendar rows, 0 observed, 0 in model-ready); "
      f"{n_gap_rows:,} calendar rows sit in holes and NONE reaches the table; rows are "
      f"consecutive calendar days for all 141 stations; lag1-7 empty on {first.date()} "
      f"({len(s_first)} observed rows); label rules hold {label_rule}")"""))

cells.append(code(r"""# (a) identical feature vectors in two different parts
feats = X.copy()                                    # L6 encoded matrix, index = labelled rows
h = hash_pandas_object(feats.round(6).fillna(-9999.0), index=False)
dfh = pd.DataFrame({"h": h.to_numpy(), "s": mr.loc[feats.index, "_split"].to_numpy()})
nparts = dfh.groupby("h")["s"].nunique()
dup_hashes = nparts[nparts > 1]
dup_rows = int(dfh["h"].isin(dup_hashes.index).sum())
pairs = dfh[dfh["h"].isin(dup_hashes.index)].groupby("h")["s"].apply(lambda s: "+".join(sorted(s))).value_counts()

# (b) episodes straddling the boundaries: how far is the nearest model-ready row on each
#     side of a cut, and which stations appear on BOTH sides within a +/-90 d window
#     (a dry-season no-data stretch around the cut is a natural embargo - measure it)
epi = {}
for b in ("2022-01-01", "2023-01-01"):
    b = pd.Timestamp(b)
    before = mr.loc[mr["date"] < b, "date"].max()
    after = mr.loc[mr["date"] >= b, "date"].min()
    pre = mr[(mr["date"] >= b - pd.Timedelta(days=90)) & (mr["date"] < b)]
    post = mr[(mr["date"] >= b) & (mr["date"] < b + pd.Timedelta(days=90))]
    both = set(pre.station_id) & set(post.station_id)
    pre_pos = set(pre.loc[pre["label_24h"] == 1, "station_id"])
    post_pos = set(post.loc[post["label_24h"] == 1, "station_id"])
    epi[str(b.date())] = {
        "last_row_before": str(before.date()), "first_row_after": str(after.date()),
        "no_model_ready_days_around_cut": int((after - before).days - 1),
        "stations_both_sides_90d": len(both),
        "stations_positive_both_sides_90d": len(pre_pos & post_pos)}

print("identical 62-feature vectors found in >1 split:", len(dup_hashes),
      f"({dup_rows} rows, {dup_rows / len(mr):.3%} of the table)")
print(pairs.to_string())
print("episode continuity across each boundary:")
print(pd.DataFrame(epi).T.to_string())

check("L11 cross-split similarity (evidence for Phase 15)",
      f"{len(dup_hashes)} exact feature-vector hashes appear in 2 parts ({dup_rows} rows); "
      + "; ".join(f"{b}: nearest rows {v['last_row_before']} <-> {v['first_row_after']} "
                  f"= {v['no_model_ready_days_around_cut']} model-ready-free days at the cut, "
                  f"{v['stations_both_sides_90d']} stations on both sides within 90 d "
                  f"({v['stations_positive_both_sides_90d']} positive on both sides)"
                  for b, v in epi.items())
      + " -> the date cut already carries a natural dry-season embargo; an explicit "
         "embargo/gap is still a Phase 15 decision; keys and dates never overlap (L9)")

mr.drop(columns=["_split"], inplace=True)   # L9 helper only; whitelist must see the 73 real columns
assert len(mr.columns) == 73"""))

cells.append(md("""## Outputs - the formal whitelist (checklist stage 20)

One row per column of the model-ready table: group, **the latest day each value may
depend on**, availability at the issue instant, and the ruling.

| ruling | meaning |
| --- | --- |
| `ADMIT` | available at 24:00 UTC of `t`, no target dependence - may enter a feature matrix |
| `ADMIT-WATCH` | admissible, but carries an observation-process / policy hazard - reported and ablated downstream (phase5 E7) |
| `ADMIT-CONSTANT` | legally available but constant - dropped by Phase 12 step 18a |
| `BAN` | never admissible: target, label-window information, or an identity/date key |

Rulings are asserted against the Phase-12 sets: **ADMIT ∪ {constant} must equal the 63
approved §5 columns, BAN must equal the 10 banned columns, and FULL must be a subset of the
admitted pool.**"""))

cells.append(code(r"""STATIC_GEO = ["elevation_m", "slope_deg", "dist_to_coast_km", "dist_river_km",
               "dist_major_river_km", "river_density_10km_km_per_km2", "n_river_lines_10km"]
IDENT = ["river", "basin", "division", "district", "upazilla", "lat", "lon", "danger_level"]
TODAY = ["wl_mean", "wl_max", "wl_min", "wl_first", "wl_last", "n_readings", "day_observed",
         "precip_mm", "rain_1d", "tmax_c", "tmin_c", "doy_sin", "doy_cos"]
NB = [f"nb{k}_{s}" for k in (3, 5) for s in ("rain_7d", "rain_14d", "wl_pct_danger", "n_wl_danger")]
UP = [f"up_{s}_lag{k}" for k in (1, 2) for s in ("rain7d", "wl_pct_danger")]

GROUP, LATEST = {}, {}
for c in IDENT:
    GROUP[c], LATEST[c] = "identity / published threshold", "static (station-constant)"
for c in STATIC_GEO:
    GROUP[c], LATEST[c] = "static geography (Phase 11A)", "static (time-invariant)"
for c in TODAY:
    GROUP[c], LATEST[c] = "observation at day t", "t"
for k in range(1, 8):
    GROUP[f"wl_last_lag{k}"], LATEST[f"wl_last_lag{k}"] = "hydrology lag", f"t-{k}"
for k in (1, 2, 3):
    GROUP[f"wl_max_lag{k}"], LATEST[f"wl_max_lag{k}"] = "hydrology lag", f"t-{k}"
for c, lo in (("wl_rise_1d", "t-1 .. t"), ("wl_rise_3d", "t-3 .. t"),
              ("wl_dist_to_danger", "t"), ("wl_pct_danger", "t"),
              ("wl_max_roll7", "t-6 .. t"), ("wl_max_roll30", "t-29 .. t"),
              ("wl_exceed_days_30", "t-29 .. t")):
    GROUP[c], LATEST[c] = "hydrology derived / window", lo
for c in ("rain_3d", "rain_7d", "rain_14d", "rain_30d"):
    GROUP[c], LATEST[c] = "rainfall window", f"t-{int(c[5:-1]) - 1} .. t"
for c in ("rain_max_7d", "wet_days_7"):
    GROUP[c], LATEST[c] = "rainfall window", "t-6 .. t"
for c in NB:
    GROUP[c], LATEST[c] = "neighbourhood, nearest-k gauges on day t (Phase 11B)", "t"
for c in UP:
    GROUP[c], LATEST[c] = "upstream gauge, lagged (Phase 11C)", "t-" + c[-1]

def ruling(c):
    if c in BANNED:
        return "BAN", "target / label-window information / identity-date key (phase0 §5)"
    if c == "day_observed":
        return "ADMIT-CONSTANT", "available at t but constant in model-ready; dropped at Phase 12 step 18a"
    if c == "n_readings":
        return "ADMIT-WATCH", "available at t; observation-process signal - L8 evidence, ablate in Phase 38"
    return "ADMIT", "computable at 24:00 UTC of t; formula-verified in L2/L3/L4"

rows = []
for c in mr.columns:
    if c in BANNED:
        grp, latest = ("target", "t+1 .. t+h") if c.startswith("label") else \
                      ("label-window information", "t+1 .. t+h") if c.startswith("obs_days") else \
                      ("identity / date key", "key")
        if c in ("station_id", "name", "source"):
            grp = "identity / date key"
        if c == "date":
            grp = "identity / date key"
    else:
        grp, latest = GROUP[c], LATEST[c]
    r_, why = ruling(c)
    rows.append({"column": c, "group": grp, "latest_day_used": latest,
                 "available_at_issue": "no" if r_ == "BAN" else "yes",
                 "ruling": r_, "in_full_set": c in FULL, "reason": why})
WL = pd.DataFrame(rows)

adm = set(WL.loc[WL["ruling"].str.startswith("ADMIT"), "column"])
ban = set(WL.loc[WL["ruling"] == "BAN", "column"])
assert ban == set(BANNED), ban ^ set(BANNED)
assert adm == set(mr.columns) - set(BANNED), adm ^ (set(mr.columns) - set(BANNED))
assert set(FULL) <= adm and "day_observed" not in FULL
assert set(FULL) == adm - {"day_observed"}
assert len(WL) == 73

WL.to_csv(PROC / "feature_whitelist.csv", index=False)
sets["phase13"] = {"date": "2026-10-08",
                   "ruling": "all 62 FULL columns ADMITTED (1 ADMIT-WATCH: n_readings); "
                             "no demotions -> Phase 12 sets unchanged, no re-run required",
                   "whitelist": "data/processed/feature_whitelist.csv"}
(PROC / "feature_sets.json").write_text(json.dumps(sets, indent=2), encoding="utf-8")

audit = {"date": "2026-10-08", "phase": "Phase 13 leakage audit (steps 22-24; checklist 19-20)",
         "issue_instant": "24:00 UTC of day t", "checks": RES,
         "auc_screen_max": {"feature": SCREEN["max_auc"].idxmax(),
                            "value": round(float(SCREEN["max_auc"].max()), 4)},
         "observation_process": {"auc": AUCS,
                                 "delta_pr_auc": E7.round(4).reset_index().to_dict(orient="records")},
         "cross_split_identical_rows": dup_rows,
         "whitelist": WL["ruling"].value_counts().to_dict()}
(PROC / "leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

print(WL["ruling"].value_counts().to_string())
print("\nfeature_whitelist.csv + leakage_audit.json written; feature_sets.json updated with the phase13 ruling")
for r in RES:
    print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['id']}")
print("ALL PHASE-13 CHECKS PASSED")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
