r"""Phase 23 - static geographic vector (workflow Phase 23 / stage 38, branch 2).

Builds the per-station static vector the fusion model concatenates with the
temporal encoder's output. Nothing here is sequential: every field is constant
per station (verified below), so the whole file is one row per station.

Contract (workflow Phase 23):
  * static vector = lat/lon, river, basin, district/upazila, danger level,
    elevation, slope, distances (the Phase-11 static groups), standardised on
    train.
  * The RAW values are stored here (categorical strings + numerics). The trainer
    fits the standardiser / frequency-encoding on the training fold only, so this
    file carries no cross-fold information.
  * Feature-group boundary (Dataset B rules): static geography is NEVER repeated
    per timestep; it enters only through the fusion input. Labels are not here.

Numeric block (standardised on train by the trainer): lat, lon, danger_level,
elevation_m, slope_deg, dist_to_coast_km, dist_river_km, dist_major_river_km,
river_density_10km_km_per_km2, n_river_lines_10km           -> 10 dims
Categorical block (train-fitted frequency encoding, unseen -> 0, as Phase 14/16):
river, basin, division, district, upazilla                    -> 5 dims
                                                          static_dim = 15

Checks (S23-s...):
  S23-s1 experimental_dataset SHA-256 unchanged
  S23-s2 133 stations, one row each, no duplicate station_id
  S23-s3 every source field is constant within a station
  S23-s4 numeric block fully finite (no NaN); categorical NaN count reported
  S23-s5 all 133 sequence-index stations are covered by the vector
  S23-s6 manifest hashes match the files on disk

Run:
    python -u scripts\build_static.py > scripts\phase23_static_build.log
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
OUT_CSV = PROC / "static_vectors.csv"
OUT_MAN = PROC / "static_manifest.json"

TBL = PROC / "experimental_dataset.csv"
SEQ_IDX = PROC / "sequence_index.csv"
TBL_SHA_EXPECTED = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"

CAT = ["river", "basin", "division", "district", "upazilla"]
NUM = ["lat", "lon", "danger_level", "elevation_m", "slope_deg",
       "dist_to_coast_km", "dist_river_km", "dist_major_river_km",
       "river_density_10km_km_per_km2", "n_river_lines_10km"]

RES: list[dict] = []


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


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print("Phase 23 - static geographic vector builder (stage 38, branch 2)")
    print("=" * 78)

    sha = sha256(TBL)
    check("S23-s1", f"experimental_dataset.csv SHA-256 unchanged ({sha[:16]}...)",
          sha == TBL_SHA_EXPECTED)

    d = pd.read_csv(TBL, usecols=["station_id"] + CAT + NUM)
    n_st = d.station_id.nunique()
    check("S23-s2", f"{n_st} stations in the experimental table",
          n_st == 133)

    # every source field must be constant within a station
    g = d.groupby("station_id")
    const_ok = all(int(g[c].nunique(dropna=False).max()) == 1 for c in CAT + NUM)
    check("S23-s3", "all 15 source fields are constant within every station "
                    "(static, not time-varying)", const_ok)

    S = g.first().reset_index()                 # one row per station
    S = S[["station_id"] + NUM + CAT]
    check("S23-s2b", f"static table {S.shape[0]} rows x {S.shape[1]} cols, "
                     f"station_id unique", S.station_id.is_unique and len(S) == 133)

    nan_num = int(S[NUM].isna().sum().sum())
    nan_cat = {c: int(S[c].isna().sum()) for c in CAT if S[c].isna().any()}
    check("S23-s4", f"numeric block fully finite ({nan_num} NaN); categorical NaN "
                    f"{nan_cat or 'none'} (frequency encoding maps NaN->__NA__)",
          nan_num == 0)

    ix = pd.read_csv(SEQ_IDX, usecols=["station_id"])
    missing = set(ix.station_id.unique()) - set(S.station_id)
    check("S23-s5", f"all {ix.station_id.nunique()} sequence-index stations covered "
                    f"by the static vector", not missing)

    S.to_csv(OUT_CSV, index=False, encoding="utf-8")
    manifest = {
        "phase": "Phase 23 static vector (stage 38, branch 2)",
        "created": pd.Timestamp.now("UTC").isoformat(),
        "source": "data/processed/experimental_dataset.csv",
        "source_sha256": sha,
        "rows": int(len(S)),
        "numeric": NUM,
        "categorical": CAT,
        "static_dim": len(NUM) + len(CAT),
        "encoding": "numeric standardised on train; categorical frequency-encoded "
                    "on train (unseen/NaN -> 0, as Phase 14/16 FreqEncoder)",
        "cardinality": {c: int(S[c].nunique(dropna=True)) for c in CAT},
        "output_csv": "data/processed/static_vectors.csv",
        "output_sha256": sha256(OUT_CSV),
    }
    OUT_MAN.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    check("S23-s6", f"manifest written ({OUT_MAN.name}); static_dim="
                    f"{manifest['static_dim']} ({len(NUM)} numeric + {len(CAT)} "
                    f"categorical); cardinality {manifest['cardinality']}",
          manifest["output_sha256"] == sha256(OUT_CSV))

    n_pass = sum(r["pass"] for r in RES)
    print("-" * 78)
    print(f"{n_pass}/{len(RES)} checks passed ({time.time() - t0:.1f}s)")
    print("wrote", OUT_CSV.relative_to(ROOT), "and", OUT_MAN.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
