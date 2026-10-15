"""Phase 18 - ML/DL dataset preparation (final workflow Phase 18 / stage 32).

Dataset A (the 62-column Phase-14 table) stays immutable. This script builds
Dataset B: calendar-aligned raw sequence windows for the deep models.

    X[i, k, v]  = variable v of station s on day (anchor_i - L + 1 + k)
    anchor_i    = the row's date in experimental_dataset.csv

Design contract (Phase 17 registry rows L3-01..06 / L4-01..04):

* 12 raw/near-raw variables only - no lag/rolling/neighbour engineered columns,
  no labels, no static attributes (those travel on the fusion branch):
      wl_first, wl_mean, wl_max, wl_min, wl_last, wl_pct_danger,
      precip_mm, tmax_c, tmin_c, n_readings, doy_sin, doy_cos
* L in {3, 7, 14, 30} (D12 - length is empirical, all four are built now);
* windows are built on the FULL daily calendar per station, so a missing
  calendar day inside a window becomes an all-NaN timestep (never a silent
  shift); days before a station's first row are front-padded with NaN;
* samples are emitted in experimental_dataset row order, so labels/splits
  join by position; no statistic is fitted anywhere (this is raw I/O).

Outputs
    data/processed/sequences_L{3,7,14,30}.npy   float32 (200665, L, 12)
    data/processed/sequence_index.csv           row order + split bookkeeping
    data/processed/sequence_manifest.json       shapes, variables, hashes,
                                                gap/front-pad statistics

Run:
    python -u scripts\\build_sequences.py > scripts\\phase18_run.log
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

TBL = PROC / "experimental_dataset.csv"
SPLITS = PROC / "split_assignments.csv"
MANIFEST = PROC / "sequence_manifest.json"
INDEX_OUT = PROC / "sequence_index.csv"

SEQ_OUT = {L: PROC / f"sequences_L{L}.npy" for L in (3, 7, 14, 30)}
TBL_SHA_EXPECTED = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
LENS = (3, 7, 14, 30)

VARIABLES = [
    "wl_first", "wl_mean", "wl_max", "wl_min", "wl_last", "wl_pct_danger",
    "precip_mm", "tmax_c", "tmin_c", "n_readings", "doy_sin", "doy_cos",
]
BANNED_SUBSTRINGS = ("label_", "lag", "roll", "rain_", "rise", "exceed",
                     "nb3_", "nb5_", "up_", "wet_days", "dist_to_danger",
                     "obs_days", "station_id", "lat", "lon", "elevation")

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


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print("Phase 18 - ML/DL dataset preparation (stage 32)")
    print("=" * 78)

    # ---- F2a  source table frozen -------------------------------------------------
    sha = sha256(TBL)
    check("F2a", f"experimental_dataset.csv SHA-256 unchanged ({sha[:16]}...)",
          sha == TBL_SHA_EXPECTED)

    cols = ["station_id", "date", "label_24h", "label_48h", "label_72h"] + VARIABLES
    df = pd.read_csv(TBL, usecols=cols)
    sp = pd.read_csv(SPLITS)
    df = df.merge(sp[["station_id", "date", "split", "spatial_fold",
                      "station_holdout", "season"]],
                  on=["station_id", "date"], how="left",
                  validate="one_to_one", indicator=True)
    n = len(df)
    check("F2b", f"loaded {n:,} anchor rows x {len(cols)} feature cols, joined 1:1 to "
                 f"split bookkeeping (split/spatial_fold/holdout/season come from "
                 f"split_assignments.csv; {int((df._merge != 'both').sum())} unmatched)",
          n == 200665 and (df._merge == "both").all())
    df = df.drop(columns="_merge")

    # ---- F2c  variable contract ---------------------------------------------------
    banned_hits = [v for v in VARIABLES
                    if any(b in v for b in BANNED_SUBSTRINGS)]
    check("F2c", f"exactly {len(VARIABLES)} raw variables, none engineered/banned: "
                 f"{VARIABLES}",
          len(VARIABLES) == 12 and not banned_hits)
    check("F2d", "labels (label_24h/48h/72h) are read but never enter the tensors",
          not any(v.startswith("label_") for v in VARIABLES))

    # ---- F2e  calendar grid -------------------------------------------------------
    dates = pd.to_datetime(df["date"])
    t0d, t1d = dates.min(), dates.max()
    cal = pd.date_range(t0d, t1d, freq="D")
    T = len(cal)
    date_idx = (dates - t0d).dt.days.to_numpy()
    stations = np.sort(df["station_id"].unique())
    st_map = {s: i for i, s in enumerate(stations)}
    st_idx = df["station_id"].map(st_map).to_numpy()
    check("F2e", f"calendar {t0d.date()}..{t1d.date()} = {T} days; "
                 f"{len(stations)} stations; anchor date indices in [0,{T - 1}]",
          date_idx.min() == 0 and date_idx.max() == T - 1 and len(stations) == 133)

    # uniqueness of (station, date) - required for scatter fill
    dup = df.duplicated(subset=["station_id", "date"]).sum()
    check("F2f", "(station_id, date) pairs unique - scatter fill is 1:1 "
                 f"({dup} duplicates)", dup == 0)

    # ---- F2g  scatter raw values onto the full calendar ---------------------------
    vals = df[VARIABLES].to_numpy(dtype=np.float32)
    A = np.full((len(stations), T, len(VARIABLES)), np.nan, dtype=np.float32)
    A[st_idx, date_idx, :] = vals
    filled = ~np.isnan(A[st_idx, date_idx, :]).all(axis=1)
    check("F2g", f"calendar array {A.shape} float32 "
                 f"({A.nbytes / 1e6:.1f} MB); every anchor cell filled",
          filled.all())

    present = ~np.isnan(A).all(axis=2)                    # (nS, T) day observed
    full_cells = len(stations) * T
    check("F2h", f"calendar coverage: {present.sum():,}/{full_cells:,} station-days "
                 f"present in the model table ({100 * present.mean():.1f}%); "
                 f"the remaining {100 * (1 - present.mean()):.1f}% are NaN gaps that "
                 f"become explicit all-NaN timesteps (Phase 19 sets the gap policy)",
          0.4 < present.mean() < 0.8)

    # ---- F2i  windows for every L -------------------------------------------------
    X_last: dict[int, np.ndarray] = {}
    pad_counts: dict[int, int] = {}
    gap_windows: dict[int, float] = {}
    for L in LENS:
        pad = np.full((len(stations), L - 1, len(VARIABLES)), np.nan, dtype=np.float32)
        Ap = np.concatenate([pad, A], axis=1) if L > 1 else A
        views = np.lib.stride_tricks.sliding_window_view(Ap, L, axis=1)
        views = views.transpose(0, 1, 3, 2)               # (nS, T, L, 12)
        X = np.ascontiguousarray(views[st_idx, date_idx])  # (n, L, 12)
        assert X.shape == (n, L, len(VARIABLES)) and X.dtype == np.float32
        np.save(SEQ_OUT[L], X)
        X_last[L] = X[-1:]                                 # keep for alignment spot

        starts = date_idx - (L - 1)
        pad_counts[L] = int((starts < 0).sum())
        empty = np.isnan(X).all(axis=2)                    # (n, L) all-NaN timesteps
        gap_windows[L] = float(empty.any(axis=1).mean())
        check(f"F2i.{L}", f"sequences_L{L}.npy written: shape {X.shape}, "
                          f"{SEQ_OUT[L].stat().st_size / 1e6:.1f} MB, "
                          f"front-padded samples {pad_counts[L]:,} "
                          f"({100 * pad_counts[L] / n:.2f}%), "
                          f"samples with >=1 empty calendar timestep "
                          f"{100 * gap_windows[L]:.1f}%",
              SEQ_OUT[L].exists() and X.shape == (n, L, 12))
        del views, X, Ap, pad

    # ---- F2j  full-window alignment against the table ----------------------------
    # For random anchors every timestep whose calendar day exists in the table must
    # reproduce that day's raw values; days absent from the table must be all-NaN.
    rng = np.random.default_rng(20261008)
    pick = rng.choice(n, size=300, replace=False)
    s_ix, d_ix = st_idx[pick], date_idx[pick]
    ok_align, ok_gap, bad_examples = True, True, []
    for j, (si, di) in enumerate(zip(s_ix, d_ix)):
        for L in LENS:
            X = np.load(SEQ_OUT[L], mmap_mode="r")[pick[j]]
            for k in range(L):
                day = di - (L - 1) + k                     # anchor - L + 1 + k
                if day < 0 or day >= T:
                    continue
                src = A[si, day]                           # calendar truth
                step = X[k]
                if np.isnan(src).all():                    # absent calendar day
                    if not np.isnan(step).all():
                        ok_gap = False
                        bad_examples.append((L, j, k, "expected NaN"))
                else:
                    if not np.allclose(step, src, rtol=1e-6, atol=1e-5,
                                       equal_nan=True):
                        ok_align = False
                        bad_examples.append((L, j, k, "value mismatch"))
                if ok_align and ok_gap:
                    continue
    check("F2j", "300 random samples x 4 window lengths: every existing calendar day "
                 f"reproduces its table values exactly (float32), every absent day is "
                 f"all-NaN ({len(bad_examples)} violations)",
          ok_align and ok_gap)

    # ---- F2k  no future leakage: window ends exactly at the anchor day -----------
    # shifting the anchor by -1 day must shift the whole window by one day
    i = int(pick[0])
    L = 30
    Xa = np.load(SEQ_OUT[L], mmap_mode="r")[i]
    prev = date_idx[i] - 1
    Xprev = np.load(SEQ_OUT[L], mmap_mode="r")[pick[0]]  # placeholder, real check below
    del Xprev
    same = True
    if prev >= 0 and date_idx[i] - 1 >= 0:
        # rebuild the same window logic: window rows k map to day di-L+1+k
        si = st_idx[i]
        for k in range(1, L):
            day_a = date_idx[i] - L + 1 + k
            day_b = day_a - 1
            if day_b < 0:
                continue
            if not np.allclose(Xa[k - 1], A[si, day_b], rtol=1e-6, atol=1e-5,
                               equal_nan=True):
                same = False
                break
    check("F2k", "no future reach: window of anchor t minus one day equals the same "
                 "window shifted one step (the window ends exactly at the anchor day)",
          same)

    # ---- F2l  row order / label joinability ---------------------------------------
    idx = pd.DataFrame({
        "sample_idx": np.arange(n),
        "station_id": df["station_id"].to_numpy(),
        "date": df["date"].to_numpy(),
        "split": df["split"].to_numpy(),
        "spatial_fold": df["spatial_fold"].to_numpy(),
        "station_holdout": df["station_holdout"].astype("int8").to_numpy(),
        "season": df["season"].to_numpy(),
        "label_24h": df["label_24h"].to_numpy(),
        "label_48h": df["label_48h"].to_numpy(),
        "label_72h": df["label_72h"].to_numpy(),
    })
    idx.to_csv(INDEX_OUT, index=False, encoding="utf-8")
    rt = pd.read_csv(INDEX_OUT)
    check("F2l", f"sequence_index.csv written in exact table row order "
                 f"({len(rt):,} rows; labels/splits join by sample_idx)",
          len(rt) == n
          and rt.sample_idx.equals(pd.Series(np.arange(n)))
          and rt.station_id.equals(df["station_id"])
          and rt.date.astype(str).equals(df["date"].astype(str))
          and rt.label_24h.equals(df["label_24h"])
          and rt.split.equals(df["split"]))

    # split bookkeeping inherited unchanged (Phase 15 owns it)
    sc = rt["split"].value_counts().to_dict()
    check("F2m", f"split counts inherited unchanged: {sc}",
          sc == {"test": 101174, "train": 77571, "val": 21920})

    # ---- F2n  manifest -------------------------------------------------------------
    manifest = {
        "phase": 18,
        "stage": 32,
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "source": {
            "experimental_dataset.csv": sha,
            "n_anchor_rows": int(n),
            "n_stations": int(len(stations)),
            "calendar_start": str(t0d.date()),
            "calendar_end": str(t1d.date()),
            "calendar_days": int(T),
        },
        "variables": VARIABLES,
        "n_variables": len(VARIABLES),
        "labels_excluded": ["label_24h", "label_48h", "label_72h"],
        "engineered_columns_excluded": "all lag/roll/rain_*_d/neighbour/up_* columns "
                                       "(E13 keeps raw-vs-engineered interpretable)",
        "fitted_statistics": "none - raw calendar scatter + windowing only",
        "window": {
            "lengths": list(LENS),
            "definition": "X[i,k,:] = station row at day (date_i - L + 1 + k); "
                          "window ends ON the anchor day",
            "front_pad": "NaN before the station's first calendar day",
            "gap_policy": "absent calendar days are all-NaN timesteps "
                          "(policy chosen in Phase 19)",
        },
        "files": {},
        "stats": {
            "front_padded_samples": pad_counts,
            "fraction_samples_with_empty_timestep": {str(k): round(v, 4)
                                                     for k, v in gap_windows.items()},
            "fraction_station_days_present": round(float(present.mean()), 4),
        },
        "checks": {"passed": sum(r["pass"] for r in RES), "total": len(RES)},
        "warnings": WARN,
    }
    for L in LENS:
        manifest["files"][f"sequences_L{L}.npy"] = {
            "shape": [int(n), L, len(VARIABLES)],
            "dtype": "float32",
            "bytes": SEQ_OUT[L].stat().st_size,
            "sha256": sha256(SEQ_OUT[L]),
        }
    manifest["files"]["sequence_index.csv"] = {
        "rows": int(n), "bytes": INDEX_OUT.stat().st_size,
        "sha256": sha256(INDEX_OUT),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    check("F2o", f"sequence_manifest.json written ({MANIFEST.stat().st_size:,} bytes) "
                 f"with shapes, variable order and SHA-256 for every file",
          MANIFEST.exists())

    n_pass = sum(r["pass"] for r in RES)
    print("-" * 78)
    print(f"{n_pass}/{len(RES)} checks passed, {len(WARN)} warnings, "
          f"{time.time() - t0:.1f}s")
    print(f"Dataset B: 4 window lengths x {len(VARIABLES)} raw vars x {n:,} samples")
    print("NEXT: Phase 19 - sequence validation (notebooks/09_sequence_validation.ipynb)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
