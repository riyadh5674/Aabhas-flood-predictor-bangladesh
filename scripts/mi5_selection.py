r"""MI-5 — model re-selection & freeze (v2 features), validation-only.

Mirrors the pre-registered Phase-33 rule (`scripts/phase33_selection.py`) but scores a
focused v2 candidate panel that Phase 33 could not see: the MI-3b HPO winners on
`FULL_V2`.  Nothing here reads the frozen test period (`>= 2023`).

Candidates (each = a spec with one fitted Pipeline per horizon):

  ============================  ==================================================
  name                          meaning
  ============================  ==================================================
  ``xgboost_FULL``              Phase-33 winner config, v1 features (incumbent reference)
  ``xgboost_FULL_V2``           same config, v2 features (isolates the feature gain)
  ``lgbm7_FULL_V2``             MI-3b 24/48-h winner (lgbm7) as a uniform spec
  ``xgb3_FULL_V2``              MI-3b 72-h winner (xgb3) as a uniform spec
  ``hpo_perhorizon_FULL_V2``    MI-3b per-horizon selection lgbm7/lgbm7/xgb3
  ``persistence``, ``climatology``  D7 bar + no-skill bar (references, never selected)
  ============================  ==================================================

Selection rule (decision **D16**, same spirit as Phase 33):

  1. **Gate 1** — beat D7 persistence on every dev cell x horizon (7 x 3 = 21 points).
  2. **Gate 2** — calibration usable: raw ``val_2022`` ECE <= 0.02.
  3. **Rank** survivors by horizon-mean P1 ``val_2022`` PR-AUC (primary), then fewest
     missed floods at the cost-optimal threshold (co-primary, FN >> FP, 10:1), then the
     robustness floor, then inference cost.  Overlapping 95 % station-bootstrap CIs on
     ``val_2022`` are reported and are *not* separable as a tie-breaker (D8).

Outputs: ``reports/mi5_scorecard.csv``, ``reports/mi5_selection.json``,
``models/final_v2/selected_model_v2.pkl`` + ``final_config_v2.json`` + ``README.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B          # noqa: E402
import mi3_ensemble as M3           # noqa: E402  (reuse the exact v2 pipe builder)

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

HOR = B.HORIZONS                       # ["label_24h", "label_48h", "label_72h"]
H2D = dict(zip(HOR, ["24h", "48h", "72h"]))
TEMPORAL = ["val_2020", "val_2021", "val_2022"]
SPATIAL = ["spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]
DEV_CELLS = TEMPORAL + SPATIAL
SEED = B.SEED
ECE_GATE = 0.02
C_FN, C_FP = 10, 1
REFERENCES = {"climatology", "persistence"}
V1_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
TAG = ""                     # set from --tag in main(); "" reproduces the pre-ERA5 paths


def sfx() -> str:
    return f"_{TAG}" if TAG else ""


def decision() -> str:
    return "D16" if not TAG else f"D16-{TAG}"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------------------- data
def load_all():
    d, fsets = B.load()
    v2 = pd.read_csv(PROC / f"mi_v2_features{sfx()}.csv", parse_dates=["date"])
    d = d.merge(v2, on=["station_id", "date"], validate="one_to_one")
    fs2 = json.loads((PROC / f"feature_sets_v2{sfx()}.json").read_text(encoding="utf-8"))["sets"]
    return d, fsets["FULL"], fs2["FULL_V2"]


def dev_masks(d: pd.DataFrame) -> dict:
    hold = d.station_holdout.to_numpy()
    year = d.date.dt.year.to_numpy()
    split = d.split.to_numpy()
    fold = d.spatial_fold.to_numpy()
    m = {"val_2020": (year == 2020) & ~hold,
         "val_2021": (year == 2021) & ~hold,
         "val_2022": (split == "val") & ~hold,
         **{f"spatial_fold_{k}": (split == "train") & (fold == k) for k in (1, 2, 3, 4)}}
    assert not any((split == "test")[v].any() for v in m.values()), "test leaked into dev"
    for c in DEV_CELLS:
        assert m[c].any(), f"empty dev cell {c}"
    return m


# ----------------------------------------------------------------- candidate specs
def build_candidates(FULL, FULL_V2) -> list[dict]:
    inc = json.loads((MODELS / "final" / "final_config.json").read_text(encoding="utf-8"))
    inc_cfg = dict(inc["params"])
    w24 = json.loads((MODELS / "mi" / f"mi_hpo{sfx()}_label_24h.json").read_text(encoding="utf-8"))
    w48 = json.loads((MODELS / "mi" / f"mi_hpo{sfx()}_label_48h.json").read_text(encoding="utf-8"))
    w72 = json.loads((MODELS / "mi" / f"mi_hpo{sfx()}_label_72h.json").read_text(encoding="utf-8"))
    per_hor = {HOR[0]: (w24["family"], w24["cfg"]),
               HOR[1]: (w48["family"], w48["cfg"]),
               HOR[2]: (w72["family"], w72["cfg"])}
    return [
        {"name": "xgboost_FULL", "family": "xgb", "cfg": inc_cfg, "feats": FULL,
         "per_hor": None, "note": "Phase-33 incumbent (v1 features)"},
        {"name": "xgboost_FULL_V2", "family": "xgb", "cfg": inc_cfg, "feats": FULL_V2,
         "per_hor": None, "note": "incumbent config on v2 features"},
        {"name": "lgbm7_FULL_V2", "family": "lgbm", "cfg": w24["cfg"], "feats": FULL_V2,
         "per_hor": None, "note": f"MI-3b 24/48-h winner ({w24['tag']}) as a uniform spec"},
        {"name": "xgb3_FULL_V2", "family": "xgb", "cfg": w72["cfg"], "feats": FULL_V2,
         "per_hor": None, "note": f"MI-3b 72-h winner ({w72['tag']}) as a uniform spec"},
        {"name": "hpo_perhorizon_FULL_V2", "family": "PERHOR", "cfg": None, "feats": FULL_V2,
         "per_hor": per_hor,
         "note": f"MI-3b per-horizon selection {w24['tag']}/{w48['tag']}/{w72['tag']}"},
    ]


def bar_specs() -> list[dict]:
    return [
        {"name": "persistence", "family": "persistence", "cfg": {}, "feats": ["wl_pct_danger"],
         "per_hor": None, "note": "D7 bar (one feature, logistic link)"},
        {"name": "climatology", "family": "climatology", "cfg": {}, "feats": [],
         "per_hor": None, "note": "no-skill bar (train base rate)"},
    ]


# ------------------------------------------------------------------------ scoring
def cost_opt_threshold(y: np.ndarray, p: np.ndarray) -> float:
    cand = np.unique(np.quantile(p, np.linspace(0.01, 0.999, 200)))
    best_c, best_t = np.inf, 0.5
    for t in cand:
        pred = p >= t
        fn = int((~pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        c = C_FN * fn + C_FP * fp
        if c < best_c:
            best_c, best_t = c, float(t)
    return best_t


def family_cfg(spec: dict, hor: str):
    if spec["per_hor"]:
        return spec["per_hor"][hor]
    return spec["family"], spec["cfg"]


def fit_spec(spec: dict, hor: str, d: pd.DataFrame, tr: np.ndarray):
    """Fit on the train fold; return (pipe_or_None, prob_full, fit_seconds)."""
    y = d[hor].to_numpy()
    tri = tr & ~pd.isna(y)
    ytr = y[tri].astype(int)
    if spec["name"] == "climatology":
        return None, np.full(len(d), float(ytr.mean())), 0.0
    if spec["name"] == "persistence":
        pipe = Pipeline([("pre", B.preprocessor(["wl_pct_danger"], [], scale=True)),
                         ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=SEED))])
        t0 = time.time()
        pipe.fit(d.loc[tri, ["wl_pct_danger"]], ytr)
        return pipe, pipe.predict_proba(d[["wl_pct_danger"]])[:, 1], time.time() - t0
    fam, cfg = family_cfg(spec, hor)
    pipe = M3.make_pipe(fam, cfg, spec["feats"])
    t0 = time.time()
    pipe.fit(d.loc[tri, spec["feats"]], ytr)
    fit_s = time.time() - t0
    return pipe, pipe.predict_proba(d[spec["feats"]])[:, 1], fit_s


def main() -> int:
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="",
                    help="optional suffix; 'era5' reads mi_v2_features_era5.csv + "
                         "feature_sets_v2_era5.json + models/mi/mi_hpo_era5_*.json and writes "
                         "mi5_scorecard_era5.csv / mi5_selection_era5.json / "
                         "models/final_v2_era5/ (the untagged D16 freeze is untouched)")
    TAG = ap.parse_args().tag
    t0 = time.time()
    print(f"MI-5 selection{sfx()} | seed={SEED} | {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    d, FULL, FULL_V2 = load_all()
    assert len(FULL_V2) > len(FULL), "v2 feature set should be larger than v1"
    masks = dev_masks(d)
    tr = (d.split.to_numpy() == "train") & ~d.station_holdout.to_numpy()
    assert not (d.split.to_numpy()[tr] == "test").any()
    station = d.station_id.to_numpy()
    y_all = {h: d[h].to_numpy() for h in HOR}

    specs = build_candidates(FULL, FULL_V2) + bar_specs()

    def cell_pr(p, y, cell):
        m = masks[cell] & ~pd.isna(y)
        return float(B.metrics(y[m], p[m], 0.5)["pr_auc"])

    rows, probs, pipes, fit_secs = [], {}, {}, {}
    for spec in specs:
        probs[spec["name"]] = {}
        pipes[spec["name"]] = {}
        fit_tot = 0.0
        for hor in HOR:
            pipe, p, fs = fit_spec(spec, hor, d, tr)
            probs[spec["name"]][hor] = p
            pipes[spec["name"]][hor] = (pipe, *family_cfg(spec, hor))
            fit_tot += fs
        fit_secs[spec["name"]] = fit_tot

    # D7 reference cell scores
    d7 = {h: {c: cell_pr(probs["persistence"][h], y_all[h], c) for c in DEV_CELLS}
          for h in HOR}

    for spec in specs:
        name = spec["name"]
        p1, margins_all, eces, missed_total, infer_tot = {}, [], [], 0, 0.0
        for hor in HOR:
            p = probs[name][hor]
            for c in DEV_CELLS:
                margins_all.append(cell_pr(p, y_all[hor], c) - d7[hor][c])
            p1[hor] = cell_pr(p, y_all[hor], "val_2022")
            m = masks["val_2022"] & ~pd.isna(y_all[hor])
            yv, pv = y_all[hor][m].astype(int), p[m]
            eces.append(B.ece(yv, pv))
            t_star = cost_opt_threshold(yv, pv)
            missed_total += int(((pv < t_star) & (yv == 1)).sum())
            pipe = pipes[name][hor][0]
            if pipe is not None:
                Xs = d.loc[m, spec["feats"]].iloc[:1000]
                ts = time.time()
                pipe.predict_proba(Xs)
                infer_tot += (time.time() - ts) * 1000.0     # ms per 1k, summed
        p1_mean = float(np.mean([p1[h] for h in HOR]))
        d7_min = float(min(margins_all))
        p2_worst = float(min(cell_pr(probs[name][h], y_all[h], c) - d7[h][c]
                             for h in HOR for c in TEMPORAL))
        p3_mean = float(np.mean([cell_pr(probs[name][h], y_all[h], c) - d7[h][c]
                                 for h in HOR for c in SPATIAL]))
        ref = name in REFERENCES
        rows.append(dict(
            model=name, notes=spec["note"], reference=ref,
            p1_pr_auc_mean=round(p1_mean, 5),
            beats_d7_everywhere=bool(d7_min > 0),
            d7_margin_min=round(d7_min, 4),
            p2_worst_d7_margin=round(p2_worst, 4),
            p3_mean_d7_margin=round(p3_mean, 4),
            calibration_ok=bool(max(eces) <= ECE_GATE),
            worst_ece=round(float(max(eces)), 4),
            missed_warn_total=int(missed_total),
            fit_s=round(fit_secs[name], 1),
            infer_ms_per_1k=round(infer_tot, 3),
            **{f"p1_pr_auc_{H2D[h]}": round(p1[h], 5) for h in HOR}))
        tag = " (reference)" if ref else ""
        print(f"  {name:24s} p1_mean={p1_mean:.5f}  d7_min={d7_min:+.4f}  "
              f"missed={missed_total:4d}  ece={max(eces):.4f}  fit={fit_secs[name]:5.1f}s{tag}")

    sc = pd.DataFrame(rows)
    sc["eligible"] = sc.apply(
        lambda r: bool((not r.reference) and r.beats_d7_everywhere and r.calibration_ok),
        axis=1)

    elig = sc[sc.eligible].sort_values(
        ["p1_pr_auc_mean", "missed_warn_total", "d7_margin_min", "infer_ms_per_1k"],
        ascending=[False, True, False, True]).reset_index(drop=True)
    elig["rank"] = np.arange(1, len(elig) + 1)
    sc = sc.merge(elig[["model", "rank"]], on="model", how="left")
    sc["rank"] = sc["rank"].fillna(0).astype(int)
    sc = sc.sort_values(["rank", "p1_pr_auc_mean"], ascending=[True, False])
    sc.to_csv(REPORTS / f"mi5_scorecard{sfx()}.csv", index=False)

    print("\n=== MI-5 v2 scorecard (dev cells only) ===")
    print(sc[["rank", "model", "p1_pr_auc_mean", "d7_margin_min", "p2_worst_d7_margin",
              "missed_warn_total", "worst_ece", "infer_ms_per_1k", "eligible"]]
          .to_string(index=False))

    if elig.empty:
        print("\nNO ELIGIBLE CANDIDATE — refusing to freeze.")
        (REPORTS / f"mi5_selection{sfx()}.json").write_text(json.dumps(
            {"decision": decision(), "selected": None, "reason": "no eligible candidate",
             "test_rows_read": 0}, indent=2), encoding="utf-8")
        return 1

    chosen = str(elig.iloc[0]["model"])
    print(f"\nSELECTED (horizon-mean P1 val_2022 PR-AUC): {chosen}")

    # ---- 95 % station-bootstrap CI on val_2022 (reuse stored probs) ----------
    ci = {}
    for name in [s["name"] for s in specs if s["name"] not in REFERENCES]:
        ci[name] = {}
        for hor in HOR:
            m = masks["val_2022"] & ~pd.isna(y_all[hor])
            lo, hi = B.bootstrap_pr_auc(station[m], y_all[hor][m].astype(int),
                                        probs[name][hor][m])
            ci[name][H2D[hor]] = [round(float(lo), 4), round(float(hi), 4)]

    # ---- freeze the winner ---------------------------------------------------
    spec = next(s for s in specs if s["name"] == chosen)
    per_hor_family = {h: pipes[chosen][h][1] for h in HOR}
    per_hor_cfg = {h: pipes[chosen][h][2] for h in HOR}
    out = MODELS / f"final_v2{sfx()}"
    out.mkdir(exist_ok=True)
    import joblib
    frozen = {
        "phase": "MI-5", "decision": decision(), "model": chosen, "seed": SEED,
        "feature_set": "FULL_V2", "features": spec["feats"],
        "per_horizon_family": per_hor_family, "per_horizon_cfg": per_hor_cfg,
        "pipelines": {h: pipes[chosen][h][0] for h in HOR},
        "train": "split == 'train' & not station_holdout (<= 2021)",
        "dataset_sha256_v1": V1_SHA,
        "dataset_sha256_v2": sha256(PROC / f"mi_v2_features{sfx()}.csv"),
        "test_rows_read": 0,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(frozen, out / "selected_model_v2.pkl")
    (out / "final_config_v2.json").write_text(
        json.dumps({k: v for k, v in frozen.items() if k != "pipelines"},
                   indent=2, default=str), encoding="utf-8")
    (out / "README.md").write_text(f"""# `models/final_v2{sfx()}/` — MI-5 selected model (decision {decision()})

| | |
| --- | --- |
| **Model** | `{chosen}` |
| **Features** | FULL_V2 ({len(spec['feats'])} cols = v1 {len(FULL)} + MI-v2) |
| **Per-horizon family** | {json.dumps({H2D[h]: per_hor_family[h] for h in HOR})} |
| **Seed** | {SEED} |
| **Selection** | validation + spatial-CV only (7 dev cells), rule D16 |
| **Frozen** | {datetime.now(timezone.utc).date().isoformat()} (MI-5) |

**Test period never read** (`test_rows_read = 0`).  This freeze is provisional: the
frozen test is opened exactly once, in MI-6.

```python
import sys, joblib
sys.path.insert(0, r"D:\\Flood predictor\\scripts")
s = joblib.load(r"D:\\Flood predictor\\models\\final_v2\\selected_model_v2.pkl")
pipe24 = s["pipelines"]["label_24h"]
p24 = pipe24.predict_proba(d[s["features"]])[:, 1]
```
""", encoding="utf-8")

    (REPORTS / f"mi5_selection{sfx()}.json").write_text(json.dumps({
        "decision": decision(), "seed": SEED, "selected": chosen,
        "rule": ("gate1 beat-D7 on all 7x3 dev points; gate2 ECE<=0.02; rank by "
                 "horizon-mean P1 val_2022 PR-AUC, then fewest missed floods at the "
                 "cost-optimal 10:1 threshold, then robustness floor, then inference"),
        "eligible": elig.model.tolist(),
        "ranked": elig[["rank", "model", "p1_pr_auc_mean", "p1_pr_auc_24h",
                        "p1_pr_auc_48h", "p1_pr_auc_72h", "missed_warn_total",
                        "d7_margin_min", "p2_worst_d7_margin", "p3_mean_d7_margin",
                        "worst_ece", "infer_ms_per_1k"]].to_dict("records"),
        "val_2022_bootstrap_ci": ci,
        "test_rows_read": 0,
    }, indent=2), encoding="utf-8")

    print(f"\nwrote reports/mi5_scorecard{sfx()}.csv, reports/mi5_selection{sfx()}.json, "
          f"models/final_v2{sfx()}/  [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
