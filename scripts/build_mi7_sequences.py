"""MI-7 / Phase 35 — engineered-feature sequence builder (Track 3).

WHY THIS EXISTS
---------------
Phase 18 built Dataset B with **12 raw variables only** (no lag/rolling/neighbour
columns). The classical models were trained on the 62-column (v1) / 110-column
(v2) ENGINEERED tables. So the Phase 21-27 "ML beats DL" comparison gave the deep
models a strictly smaller information set than the trees. This script removes that
asymmetry: it builds the SAME causal calendar-scatter + window tensors that
Phase 18 built, but over the engineered feature sets, so a sequence model can be
given exactly the information the tabular model had.

It is strictly ADDITIVE: it writes new files only and touches nothing that
Phase 16-34 produced.

CONTRACT
--------
* Sources (read-only):
    data/processed/experimental_dataset.csv   (FULL 62 cols + keys; row order = index)
    data/processed/mi_v2_features.csv         (48 v2 cols + keys; row-aligned)
    data/processed/feature_sets.json          (FULL)
    data/processed/feature_sets_v2.json       (FULL_V2 + v2_new_columns)
    data/processed/sequence_index.csv         (row order + split/label bookkeeping)
* Channel sets (NUMERIC only; the 5 text columns river/basin/division/district/
  upazilla are constant per station and travel on the fusion/static branch, exactly
  as Phase 23 does):
      v1 : FULL      minus text  -> 57 channels
      v2 : FULL_V2   minus text  -> 105 channels
* L in {7, 14}. Same window definition as Phase 18:
      X[i, k, :] = value of the station row at day (anchor_i - L + 1 + k)
  the window ENDS exactly on the anchor day (no future reach). Missing calendar
  days are all-NaN timesteps; mask[i, k] = 1 iff timestep k is observed (any
  non-NaN channel), i.e. the Phase-19 P19-M1 keep-all policy.
* No statistic is fitted here (raw scatter + windowing only). Fold
  standardisation happens inside the trainer.

Outputs
    data/processed/sequences_mi7_L{L}_{v1|v2}.npy   float32 (200665, L, C)
    data/processed/mask_mi7_L{L}_{v1|v2}.npy        uint8   (200665, L)
    data/processed/mi7_sequence_manifest.json        shapes, order, SHA-256, stats

Run:
    python -u scripts\\build_mi7_sequences.py > scripts\\mi7_build_run.log 2>&1
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

EXP = PROC / "experimental_dataset.csv"
V2 = PROC / "mi_v2_features.csv"
FS_V1 = PROC / "feature_sets.json"
FS_V2 = PROC / "feature_sets_v2.json"
INDEX = PROC / "sequence_index.csv"
MANIFEST = PROC / "mi7_sequence_manifest.json"

TEXT_COLS = ["river", "basin", "division", "district", "upazilla"]
LENS = (7, 14)
SETS = ("v1", "v2")

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


def out_paths(tag: str, L: int) -> tuple[Path, Path]:
    return (PROC / f"sequences_mi7_L{L}_{tag}.npy",
            PROC / f"mask_mi7_L{L}_{tag}.npy")


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print("MI-7 / Phase 35 — engineered-feature sequence builder (Track 3)")
    print("=" * 78)

    fs1 = json.loads(FS_V1.read_text(encoding="utf-8"))["sets"]
    fs2 = json.loads(FS_V2.read_text(encoding="utf-8"))
    full = fs1["FULL"]                                       # 62
    full_v2 = fs2["sets"]["FULL_V2"]                         # 110
    v2_new = fs2.get("v2_new_columns")
    if v2_new is None:
        v2_new = [c for c in full_v2 if c not in set(full)]

    num_v1 = [c for c in full if c not in TEXT_COLS]
    num_v2 = [c for c in full_v2 if c not in TEXT_COLS]
    check("M1a", f"channel sets: v1=FULL-text={len(num_v1)} numeric, "
                 f"v2=FULL_V2-text={len(num_v2)} numeric "
                 f"(FULL {len(full)}, FULL_V2 {len(full_v2)}, text {len(TEXT_COLS)})",
          len(num_v1) == 57 and len(num_v2) == 105)
    banned = ("label_", "obs_days", "station_id", "name", "date")
    hits_v1 = [c for c in num_v1 if any(b in c for b in banned)]
    hits_v2 = [c for c in num_v2 if any(b in c for b in banned)]
    check("M1b", "no label/key/obs_days column in any channel set "
                 f"(v1 {hits_v1}, v2 {hits_v2})", not hits_v1 and not hits_v2)
    check("M1c", f"v2 adds exactly {len(v2_new)} new columns over v1 "
                 f"(FULL_V2 = FULL + v2_new)",
          set(num_v2) - set(num_v1) == set(v2_new) - set(TEXT_COLS))

    # ---- read sources ---------------------------------------------------------
    idx = pd.read_csv(INDEX, parse_dates=["date"])
    n = len(idx)
    check("M1d", f"sequence_index.csv loaded ({n:,} rows; "
                 f"{idx.station_id.nunique()} stations) — row order is the contract",
          n == 200665 and idx.station_id.nunique() == 133)
    check("M1e", f"split counts inherited: "
                 f"{idx.split.value_counts().to_dict()}",
          idx.split.value_counts().to_dict() ==
          {"test": 101174, "train": 77571, "val": 21920})

    exp = pd.read_csv(EXP, usecols=["station_id", "date"] + num_v1,
                      parse_dates=["date"])
    check("M1f", f"experimental_dataset.csv read ({len(exp):,} rows x "
                 f"{len(num_v1)} v1 numeric cols)", len(exp) == n)
    check("M1g", "experimental row order == sequence_index row order "
                 "(station_id and date align positionally)",
          exp.station_id.equals(idx.station_id)
          and exp.date.astype(str).equals(idx.date.astype(str)))

    v2 = pd.read_csv(V2, usecols=["station_id", "date"] + v2_new,
                     parse_dates=["date"])
    check("M1h", f"mi_v2_features.csv read ({len(v2):,} rows x "
                 f"{len(v2_new)} new cols)", len(v2) == n)
    check("M1i", "mi_v2_features row order == index row order",
          v2.station_id.equals(idx.station_id)
          and v2.date.astype(str).equals(idx.date.astype(str)))

    tbl = exp.copy()
    for c in v2_new:
        tbl[c] = v2[c].to_numpy()
    check("M1j", f"merged engineered table: {tbl.shape[1] - 2} feature cols "
                 f"({len(num_v1)} v1 + {len(v2_new)} v2)",
          tbl.shape[1] - 2 == len(num_v2))

    # ---- calendar grid --------------------------------------------------------
    dates = tbl["date"]
    t0d, t1d = dates.min(), dates.max()
    cal = pd.date_range(t0d, t1d, freq="D")
    T = len(cal)
    date_idx = (dates - t0d).dt.days.to_numpy()
    stations = np.sort(tbl["station_id"].unique())
    st_map = {s: i for i, s in enumerate(stations)}
    st_idx = tbl["station_id"].map(st_map).to_numpy()
    check("M1k", f"calendar {t0d.date()}..{t1d.date()} = {T} days; "
                 f"{len(stations)} stations; anchors in [0,{T - 1}]",
          date_idx.min() == 0 and date_idx.max() == T - 1 and len(stations) == 133)
    check("M1l", "(station_id, date) unique — scatter fill is 1:1",
          tbl.duplicated(subset=["station_id", "date"]).sum() == 0)

    manifest = {
        "sprint": "MI-7 / Phase 35 (fair DL rematch)",
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "purpose": "engineered-feature sequence tensors for the fair DL rematch",
        "source": {
            "experimental_dataset.csv": sha256(EXP),
            "mi_v2_features.csv": sha256(V2),
            "feature_sets.json": sha256(FS_V1),
            "feature_sets_v2.json": sha256(FS_V2),
            "sequence_index.csv": sha256(INDEX),
        },
        "n_anchor_rows": int(n),
        "n_stations": int(len(stations)),
        "calendar_start": str(t0d.date()),
        "calendar_end": str(t1d.date()),
        "calendar_days": int(T),
        "text_cols_excluded": TEXT_COLS,
        "channel_sets": {
            "v1": {"source": "FULL (62) minus text", "n_channels": len(num_v1),
                   "channels": num_v1},
            "v2": {"source": "FULL_V2 (110) minus text", "n_channels": len(num_v2),
                   "channels": num_v2},
        },
        "window": {
            "lengths": list(LENS),
            "definition": "X[i,k,:] = day (anchor_i - L + 1 + k); window ends ON "
                          "the anchor day (same as Phase 18)",
            "front_pad": "NaN before the station's first calendar day",
            "gap_policy": "absent calendar days are all-NaN timesteps; "
                          "mask = 1 iff timestep observed (P19-M1 keep-all)",
        },
        "fitted_statistics": "none — raw calendar scatter + windowing only",
        "files": {},
        "stats": {},
        "checks": {},
        "warnings": WARN,
    }

    # ---- per set: scatter, window, write -------------------------------------
    for tag, cols in (("v1", num_v1), ("v2", num_v2)):
        vals = tbl[cols].to_numpy(dtype=np.float32)
        A = np.full((len(stations), T, len(cols)), np.nan, dtype=np.float32)
        A[st_idx, date_idx, :] = vals
        filled = ~np.isnan(A[st_idx, date_idx, :]).all(axis=1)
        check(f"M2{tag}", f"[{tag}] calendar array {A.shape} float32 "
                         f"({A.nbytes / 1e6:.0f} MB); every anchor cell filled",
              filled.all())
        present = ~np.isnan(A).all(axis=2)
        check(f"M3{tag}", f"[{tag}] station-day coverage "
                         f"{present.sum():,}/{len(stations) * T:,} "
                         f"({100 * present.mean():.1f}%)",
              0.3 < present.mean() < 0.8)

        for L in LENS:
            sp, mp = out_paths(tag, L)
            pad = np.full((len(stations), L - 1, len(cols)), np.nan,
                          dtype=np.float32)
            Ap = np.concatenate([pad, A], axis=1)
            views = np.lib.stride_tricks.sliding_window_view(Ap, L, axis=1)
            views = views.transpose(0, 1, 3, 2)                 # (nS, T, L, C)
            X = np.ascontiguousarray(views[st_idx, date_idx])   # (n, L, C)
            assert X.shape == (n, L, len(cols)) and X.dtype == np.float32
            M = (~np.isnan(X).all(axis=2)).astype(np.uint8)     # (n, L)
            np.save(sp, X)
            np.save(mp, M)
            starts = date_idx - (L - 1)
            pad_n = int((starts < 0).sum())
            empty_frac = float((~M.astype(bool)).any(axis=1).mean())
            check(f"M4{tag}.{L}",
                  f"[{tag}] sequences_mi7_L{L}_{tag}.npy {X.shape} "
                  f"({sp.stat().st_size / 1e6:.0f} MB), mask mean "
                  f"{M.mean():.3f}, front-padded {pad_n:,}, "
                  f"windows with >=1 empty timestep {100 * empty_frac:.1f}%",
                  sp.exists() and mp.exists() and X.shape == (n, L, len(cols)))
            manifest["files"][sp.name] = {
                "shape": [int(n), L, len(cols)], "dtype": "float32",
                "bytes": sp.stat().st_size, "sha256": sha256(sp)}
            manifest["files"][mp.name] = {
                "shape": [int(n), L], "dtype": "uint8",
                "bytes": mp.stat().st_size, "sha256": sha256(mp)}
            manifest["stats"][f"{tag}_L{L}"] = {
                "front_padded_samples": pad_n,
                "fraction_windows_with_empty_timestep": round(empty_frac, 4),
                "mask_mean": round(float(M.mean()), 4)}
            del views, X, M, Ap, pad

        # alignment spot-check: an existing calendar day must reproduce its value
        rng = np.random.default_rng(20261008)
        pick = rng.choice(n, size=120, replace=False)
        ok = True
        for L in LENS:
            X = np.load(out_paths(tag, L)[0], mmap_mode="r")
            for j in pick:
                si, di = st_idx[j], date_idx[j]
                for k in range(L):
                    day = di - (L - 1) + k
                    if day < 0 or day >= T:
                        continue
                    src, step = A[si, day], X[j, k]
                    if np.isnan(src).all():
                        ok = ok and bool(np.isnan(step).all())
                    else:
                        ok = ok and bool(np.allclose(step, src, rtol=1e-6,
                                                     atol=1e-5, equal_nan=True))
            del X
        check(f"M5{tag}", f"[{tag}] 120 random anchors x 2 window lengths: every "
                         f"existing calendar day reproduces its table values; every "
                         f"absent day is all-NaN", ok)
        del A, vals

    manifest["checks"] = {"passed": sum(r["pass"] for r in RES),
                          "total": len(RES)}
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    n_pass = sum(r["pass"] for r in RES)
    print("-" * 78)
    print(f"{n_pass}/{len(RES)} checks passed, {len(WARN)} warnings, "
          f"{time.time() - t0:.1f}s")
    print(f"wrote {len([k for k in manifest['files'] if k.startswith('sequences_')])} "
          f"tensors x 2 sets x {len(LENS)} lengths: "
          f"v1={len(num_v1)}ch, v2={len(num_v2)}ch")
    print("NEXT: MI-7 trainer (scripts/mi7_dl_rematch.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
