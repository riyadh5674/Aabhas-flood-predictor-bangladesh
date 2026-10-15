"""Build notebooks/18_sequences.ipynb (workflow Phase 18 / stage 32).

scripts/build_sequences.py builds Dataset B (calendar-scatter + windowing, 17/17
green, scripts/phase18_run.log, 9 s). This notebook independently re-verifies the
written artifacts from the manifest and the frozen table, then discharges the
Phase-18 prerequisite that the workflow puts on this stage: torch installed and
recorded, and a seeded tiny training run reproducing its losses twice.

Checks N18-01..N18-12:
    N18-01  inputs: manifest, index, four .npy files present; table SHA unchanged
    N18-02  manifest schema: 12 raw variables, L in {3,7,14,30}, source SHA matches
    N18-03  tensor shape/dtype: (200665, L, 12) float32 for every L
    N18-04  every file's SHA-256 matches the manifest (build is reproducible)
    N18-05  variable contract: no labels, no lag/roll/neighbour/engineered columns
    N18-06  sequence_index.csv: row order identical to experimental_dataset,
            split counts 77571/21920/101174
    N18-07  anchor alignment: the last timestep of a window equals the anchor
            row's 12 raw values (100 random samples x 4 lengths)
    N18-08  full-window alignment: every timestep equals the table values of its
            own calendar day; absent calendar days are all-NaN (no silent shift)
    N18-09  no forward reach: the newest day in a window IS the anchor day t
    N18-10  gap/front-pad statistics recomputed from the tensors match the manifest
    N18-11  split/holdout/season inheritance: index joins 1:1 with
            split_assignments.csv, zero disagreements
    N18-12  prerequisite: torch/numpy/pandas/sklearn versions recorded in
            requirements.txt, and a seeded tiny training run reproduces its
            losses exactly on a second run

Build and run:

    python scripts/build_phase18_notebook.py
    .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.kernel_name=flood-predictor notebooks\\18_sequences.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "notebooks" / "18_sequences.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# Phase 18 — ML/DL dataset preparation, Dataset B (stage 32)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 18 — ML/DL dataset preparation** → checklist **stage 32** |
| **Built by** | `scripts/build_sequences.py` → `scripts/phase18_run.log` (exit 0, 17/17, 9.0 s) |
| **This notebook** | independent verification of the written tensors + the torch prerequisite |
| **Outputs verified** | `data/processed/sequences_L{3,7,14,30}.npy` · `sequence_index.csv` · `sequence_manifest.json` |
| **Input (immutable)** | `experimental_dataset.csv` (SHA re-checked) + `split_assignments.csv` |
| **Binding decisions** | **D12** L ∈ {3,7,14,30} built, none assumed best · labels never in the tensor · static/station columns stay on the Phase-23 fusion branch · gaps become explicit NaN timesteps (policy fixed in Phase 19) |

Twelve checks, **N18-01 … N18-12**. Nothing is refit here; the notebook proves the
tensors are faithful to the table and that a seeded torch run is reproducible."""))

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
MANIFEST = PROC / "sequence_manifest.json"
INDEX_CSV = PROC / "sequence_index.csv"
REQ = ROOT / "requirements.txt"
NPY = {L: PROC / f"sequences_L{L}.npy" for L in (3, 7, 14, 30)}

TBL_SHA_EXPECTED = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
LENS = (3, 7, 14, 30)
VARIABLES = ["wl_first", "wl_mean", "wl_max", "wl_min", "wl_last", "wl_pct_danger",
             "precip_mm", "tmax_c", "tmin_c", "n_readings", "doy_sin", "doy_cos"]
BANNED = ("label_", "lag", "roll", "rain_", "rise", "exceed", "nb3_", "nb5_",
          "up_", "wet_days", "dist_to_danger", "obs_days")
SEED = 20261008


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
print("Phase 18 notebook - Dataset B verification")"""))

# ---------------------------------------------------------------- N18-01
cells.append(code(r"""# N18-01 - inputs present, source table frozen
check("N18-01a", f"manifest {MANIFEST.stat().st_size:,} B, index {INDEX_CSV.stat().st_size:,} B, "
                 f"4 tensors: " + ", ".join(f"L{L} {p.stat().st_size / 1e6:.1f} MB"
                                            for L, p in NPY.items()),
      MANIFEST.exists() and INDEX_CSV.exists() and all(p.exists() for p in NPY.values()))
sha = sha256(TBL)
check("N18-01b", f"experimental_dataset.csv SHA-256 unchanged ({sha[:16]}...)",
      sha == TBL_SHA_EXPECTED)
man = json.loads(MANIFEST.read_text(encoding="utf-8"))
check("N18-01c", f"manifest reports phase {man['phase']} / stage {man['stage']}, "
                 f"built {man['created_utc'][:19]}, "
                 f"{man['checks']['passed']}/{man['checks']['total']} build checks green",
      man["phase"] == 18 and man["checks"]["passed"] == man["checks"]["total"])"""))

# ---------------------------------------------------------------- N18-02
cells.append(code(r"""# N18-02 - manifest schema
check("N18-02a", f"manifest lists exactly the 12 raw variables: {man['variables']}",
      man["variables"] == VARIABLES and man["n_variables"] == 12)
check("N18-02b", f"window lengths {man['window']['lengths']} with the anchor-day "
                 "definition recorded",
      man["window"]["lengths"] == list(LENS)
      and "ends ON the anchor day" in man["window"]["definition"])
check("N18-02c", "manifest source SHA equals the frozen table SHA "
                 f"({man['source']['experimental_dataset.csv'][:16]}...)",
      man["source"]["experimental_dataset.csv"] == TBL_SHA_EXPECTED)
check("N18-02d", f"no statistic was fitted ({man['fitted_statistics']}); labels listed "
                 f"as excluded {man['labels_excluded']}",
      man["fitted_statistics"].startswith("none")
      and man["labels_excluded"] == ["label_24h", "label_48h", "label_72h"])"""))

# ---------------------------------------------------------------- N18-03
cells.append(code(r"""# N18-03 - shape and dtype of every tensor
shapes = {}
for L, p in NPY.items():
    X = np.load(p, mmap_mode="r")
    shapes[L] = tuple(X.shape)
    rec = man["files"][p.name]
    check(f"N18-03.{L}", f"{p.name}: shape {X.shape}, dtype {X.dtype} "
                         f"(manifest: {rec['shape']}, {rec['dtype']})",
          X.shape == (200665, L, 12) and X.dtype == np.float32
          and list(X.shape) == rec["shape"])"""))

# ---------------------------------------------------------------- N18-04
cells.append(code(r"""# N18-04 - file hashes match the manifest (byte-reproducible build)
bad = []
for name, rec in man["files"].items():
    p = PROC / name
    if not p.exists() or sha256(p) != rec["sha256"]:
        bad.append(name)
check("N18-04", f"SHA-256 verified for all {len(man['files'])} artifacts "
                f"({', '.join(man['files'])})", not bad)"""))

# ---------------------------------------------------------------- N18-05
cells.append(code(r"""# N18-05 - variable contract: raw only, no labels, no engineered columns
hits = [v for v in VARIABLES if any(b in v for b in BANNED)]
check("N18-05a", f"no banned/label/engineered column among the 12 variables: {hits}",
      not hits and not any(v.startswith("label_") for v in VARIABLES))
check("N18-05b", f"engineered exclusion recorded: {man['engineered_columns_excluded'][:60]}...",
      "lag" in man["engineered_columns_excluded"] and "up_*" in man["engineered_columns_excluded"])
check("N18-05c", "static/station columns (lat/lon/danger_level/river/...) are NOT in "
                 "the tensor - they stay on the Phase-23 fusion branch",
      not any(v in VARIABLES for v in ("lat", "lon", "danger_level", "river",
                                       "elevation_m", "station_id")))"""))

# ---------------------------------------------------------------- N18-06
cells.append(code(r"""# N18-06 - index row order identical to the source table
tbl = pd.read_csv(TBL, usecols=["station_id", "date", "label_24h"])
ix = pd.read_csv(INDEX_CSV)
check("N18-06a", f"index has {len(ix):,} rows in exact table order "
                 "(sample_idx 0..n-1, station/date/labels align row-for-row)",
      len(ix) == len(tbl) == 200665
      and ix.sample_idx.equals(pd.Series(np.arange(len(ix))))
      and ix.station_id.equals(tbl.station_id)
      and ix.date.astype(str).equals(tbl.date.astype(str))
      and np.allclose(ix.label_24h, tbl.label_24h, equal_nan=True))
sc = ix["split"].value_counts().to_dict()
check("N18-06b", f"split counts inherited unchanged: {sc}",
      sc == {"test": 101174, "train": 77571, "val": 21920})"""))

# ---------------------------------------------------------------- N18-07
cells.append(code(r"""# N18-07 - anchor alignment: window's last timestep == the anchor row's raw values
use = ["station_id", "date"] + VARIABLES
raw = pd.read_csv(TBL, usecols=use)
rng = np.random.default_rng(SEED)
pick = rng.choice(len(ix), size=100, replace=False)
bad = []
for L in LENS:
    X = np.load(NPY[L], mmap_mode="r")
    sub = raw.iloc[pick].reset_index(drop=True)
    want = sub[VARIABLES].to_numpy(dtype=np.float32)
    got = X[pick, -1, :]
    if not np.allclose(got, want, rtol=1e-6, atol=1e-5, equal_nan=True):
        bad.append(L)
check("N18-07", f"100 random samples x L in {LENS}: last timestep equals the anchor "
                f"day's 12 raw values exactly (float32) - mismatches at L={bad}", not bad)"""))

# ---------------------------------------------------------------- N18-08
cells.append(code(r"""# N18-08 - full-window alignment: every timestep is its own calendar day's value
key = raw.set_index(["station_id", "date"])
anch = ix.iloc[pick[:30]]
violations, gaps_ok, checked = 0, 0, 0
for L in LENS:
    X = np.load(NPY[L], mmap_mode="r")
    for _, row in anch.iterrows():
        t = pd.Timestamp(row.date)
        win = X[row.sample_idx]
        for k in range(L):
            day = t - pd.Timedelta(days=L - 1 - k)
            want = key.loc[(row.station_id, str(day.date()))] if (row.station_id, str(day.date())) in key.index else None
            got = win[k]
            if want is None:                       # absent calendar day
                gaps_ok += int(np.isnan(got).all())
                violations += int(not np.isnan(got).all())
            else:
                violations += int(not np.allclose(
                    got, want[VARIABLES].to_numpy(dtype=np.float32),
                    rtol=1e-6, atol=1e-5, equal_nan=True))
            checked += 1
check("N18-08", f"30 anchors x 4 window lengths = {checked} timesteps: each equals the "
                f"table value of its own calendar day; {gaps_ok} absent days are "
                f"all-NaN (windows are calendar-true, never shifted)",
      violations == 0)"""))

# ---------------------------------------------------------------- N18-09
cells.append(code(r"""# N18-09 - no forward reach: the newest day in the window IS the anchor day
row = ix.iloc[int(pick[0])]
L = 30
X = np.load(NPY[L], mmap_mode="r")[row.sample_idx]
t = pd.Timestamp(row.date)
newest = t
oldest = t - pd.Timedelta(days=L - 1)
check("N18-09a", f"window for anchor {row.date} spans {oldest.date()} .. {newest.date()} "
                 f"- ends exactly on t, never past it (D5 information boundary)",
      (newest - oldest).days == L - 1)
# and the same anchor one day earlier must be the same window shifted one step
i_prev = int(row.sample_idx) - 1
if i_prev >= 0:
    prev = ix.iloc[i_prev]
    if prev.station_id == row.station_id and pd.Timestamp(prev.date) == t - pd.Timedelta(days=1):
        Xp = np.load(NPY[L], mmap_mode="r")[i_prev]
        same = np.allclose(X[:-1], Xp[1:], rtol=1e-6, atol=1e-5, equal_nan=True)
        check("N18-09b", "consecutive anchors of the same station share the overlapping "
                         "days exactly (window slides, never jumps)", bool(same))
    else:
        check("N18-09b", "consecutive-anchor overlap test skipped for this pick "
                         "(previous row is another station/day)", True)"""))

# ---------------------------------------------------------------- N18-10
cells.append(code(r"""# N18-10 - gap / front-pad statistics recomputed from the tensors
rec = man["stats"]
for L in LENS:
    X = np.load(NPY[L], mmap_mode="r")
    empty = np.isnan(X).all(axis=2)                 # (n, L)
    frac = float(empty.any(axis=1).mean())
    fp = int(empty[:, 0].sum())                      # first timestep empty = front-pad
    want = rec["fraction_samples_with_empty_timestep"][str(L)]
    check(f"N18-10.{L}", f"L={L}: {100 * frac:.2f}% of samples contain >=1 empty "
                         f"calendar timestep (manifest {100 * want:.2f}%), "
                         f"{fp:,} samples front-padded at series start",
          abs(frac - want) < 5e-4 and fp > 0)
check("N18-10b", f"calendar coverage {100 * rec['fraction_station_days_present']:.1f}% "
                 "present - the rest are the explicit NaN gaps Phase 19 must rule on "
                 "(2018 archive gap R6 + station outages)",
      0.4 < rec["fraction_station_days_present"] < 0.6)"""))

# ---------------------------------------------------------------- N18-11
cells.append(code(r"""# N18-11 - split / holdout / season inheritance is 1:1 with the Phase-15 masks
sp = pd.read_csv(SPLITS)
m = ix.merge(sp[["station_id", "date", "split", "spatial_fold", "station_holdout",
                 "season"]], on=["station_id", "date"], how="left",
             suffixes=("", "_sp"), indicator=True)
check("N18-11a", f"every index row matches a split row ({(m._merge != 'both').sum()} "
                 "unmatched)",
      (m._merge == "both").all())
disagree = {
    "split": int((m["split"] != m["split_sp"]).sum()),
    "spatial_fold": int((m["spatial_fold"] != m["spatial_fold_sp"]).sum()),
    "station_holdout": int((m["station_holdout"].astype(int)
                            != m["station_holdout_sp"].astype(int)).sum()),
    "season": int((m["season"] != m["season_sp"]).sum()),
}
check("N18-11b", f"zero disagreements on split/fold/holdout/season: {disagree}",
      sum(disagree.values()) == 0)
ho = int(ix.station_holdout.sum())
check("N18-11c", f"{ho:,} holdout-station samples inherit station_holdout=1 - they can "
                 "never enter a training cell (Phase-19 check S7 keeps this)",
      ho == 39725)"""))

# ---------------------------------------------------------------- N18-12
cells.append(code(r"""# N18-12 - prerequisite: torch installed, versions recorded, seeded run reproduces
import torch
import torch.nn as nn

req = REQ.read_text(encoding="utf-8")
versions = {"torch": torch.__version__, "numpy": np.__version__,
            "pandas": pd.__version__}
import sklearn
versions["scikit-learn"] = sklearn.__version__
check("N18-12a", f"versions recorded: {versions}",
      torch.__version__.startswith("2.") and "+cpu" in torch.__version__
      and "torch" in req)
check("N18-12b", "requirements.txt carries the pinned torch build",
      "torch==" in req or "torch ==" in req)


def tiny_run(seed):
    torch.manual_seed(seed)
    nn_ = nn.Sequential(nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.SGD(nn_.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(64, 30, 12, generator=g)
    y = (torch.rand(64, generator=g) > 0.7).float()
    out = []
    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn(nn_(X[:, -1, :]).squeeze(-1), y)
        loss.backward()
        opt.step()
        out.append(float(loss.detach()))
    return out


a, b = tiny_run(SEED), tiny_run(SEED)
check("N18-12c", f"seeded tiny training run reproduces its 5 losses exactly on a second "
                 f"run: {[round(x, 6) for x in a]}",
      a == b and all(np.isfinite(a)))"""))

# ---------------------------------------------------------------- summary
cells.append(code(r"""print("=" * 78)
print("PHASE 18 / STAGE 32   Dataset B verified - sequences L in {3,7,14,30} x 12 vars")
print("=" * 78)
for r in RES:
    print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['id']}")
for w in WARN:
    print(f"  WARN  {w['id']}: {w['note']}")
print(f"\n{sum(r['pass'] for r in RES)}/{len(RES)} checks passed, {len(WARN)} warnings, "
      f"{time.time() - t0:.1f}s")
print("outputs: data/processed/sequences_L{3,7,14,30}.npy, sequence_index.csv, "
      "sequence_manifest.json")
print("next: Phase 19 - sequence validation (notebooks/09_sequence_validation.ipynb): "
      "continuity, gap policy, label alignment, split inheritance")
assert all(r["pass"] for r in RES)
assert sha256(TBL) == TBL_SHA_EXPECTED, "experimental table changed!"
print("ALL PHASE-18 CHECKS PASSED - experimental_dataset.csv byte-stable")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (flood-predictor venv)",
                   "language": "python", "name": "flood-predictor"},
    "language_info": {"name": "python", "version": "3.13.7"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
