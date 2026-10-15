r"""Phase 20 — classical ML candidates + validation-only tuning
(workflow body "Phase 20 — Classical ML candidates" / checklist stages 34, 35).

Contract
--------
* Registry: reports/phase17_experiment_registry.csv rows L2-01..L2-05
  (random_forest, gradient_boosting, xgboost, lightgbm, xgboost_selected),
  each with tuning_budget_configs = 40 (maximum), test_policy =
  frozen_read_at_phase33, selection_metric = pr_auc, selection_cell = val_2022,
  seed = 20261008.
* Preprocessing: EXACTLY the Phase-16 contract, imported from
  scripts/baseline_model.py — median imputation + train-fitted frequency
  encoding (no target encoding), scale=False for tree models, banned columns
  asserted absent by load().
* Cells: the Phase-16 grid restricted to DEV cells —
  P1 val_2022, P2 val_2020/2021/2022, P3 spatial_fold_1..4, each x
  {all, monsoon, non-monsoon} x {24h,48h,72h}. The frozen test period is NOT
  read (test buckets belong to Phase 33's single read; enforced here by
  stripping every eval cell that starts with "test" and asserting the result).
* Tuning (stage 35): random search over a fixed, seeded space per family,
  scored ONLY on the selection cell val_2022 (season=all), score = mean PR-AUC
  over the three horizons (D2: horizons reported equally). Budgets were set
  from a measured fit-time probe on this machine (RF400 = 36 s,
  GBM200 = 138 s, XGB600 = 15 s, LGBM600 = 12 s, 8 logical cores) and are
  recorded below — all strictly <= the registry maximum of 40:
      random_forest        12 configs   (72-cell space)
      gradient_boosting     4 configs   (48-cell space; slowest family)
      xgboost              16 configs   (648-cell space)
      lightgbm             16 configs   (216-cell space)
      xgboost_selected     12 configs   (648-cell space, SELECTED features)
* Imbalance (Phase 16 S10): weighting ON/OFF is part of every family's search
  space; which one wins is a validation result, applied inside the training
  fold only (RF class_weight, GBM sample_weight, XGB/LGBM scale_pos_weight).
* Early stopping: not used anywhere (fixed n_estimators) — no code path can
  touch a test fold during fitting (the Phase-15 defect is structurally gone).
* Outputs:
    reports/phase20_search.csv     one row per evaluated config
    reports/phase20_metrics.csv    dev-cell grid (same columns as Phase 16)
    models/phase20_L2-0X.pkl       P1 primary pipelines, {horizon: Pipeline}
    scripts/phase20_run.log        stdout of this run
* Resume: finished families (pkl + both CSVs present) are skipped; partial
  rows of an unfinished family are dropped and redone. Pass family names as
  argv to run a subset.

Run (two calls keeps each under the shell timeout):
    python -u scripts/phase20_classical_ml.py random_forest gradient_boosting
    python -u scripts/phase20_classical_ml.py xgboost lightgbm xgboost_selected
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"D:\Flood predictor")
sys.path.insert(0, str(ROOT / "scripts"))

from baseline_model import (HORIZONS, REPORTS, SEED, TEXT_COLS, best_f1_threshold,
                            bootstrap_pr_auc, build_tasks, load, macro_pr_auc,
                            metrics, preprocessor, score_cell)

from lightgbm import LGBMClassifier
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
from xgboost import XGBClassifier

MODELS = ROOT / "models"
SEARCH_CSV = REPORTS / "phase20_search.csv"
MET_CSV = REPORTS / "phase20_metrics.csv"
REGISTRY = REPORTS / "phase17_experiment_registry.csv"

BUDGETS = {"random_forest": 12, "gradient_boosting": 4, "xgboost": 16,
           "lightgbm": 16, "xgboost_selected": 12}
REG_IDS = {"random_forest": "L2-01", "gradient_boosting": "L2-02",
           "xgboost": "L2-03", "lightgbm": "L2-04",
           "xgboost_selected": "L2-05"}
FEATURE_SET = {"random_forest": "FULL", "gradient_boosting": "FULL",
               "xgboost": "FULL", "lightgbm": "FULL",
               "xgboost_selected": "SELECTED"}
DEV_CELLS = {"val_2022", "val_2020", "val_2021", "spatial_fold_1",
             "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"}


# ------------------------------------------------------------------ search space
def space(family: str) -> dict[str, list]:
    if family == "random_forest":
        return {"n_estimators": [200, 400], "max_depth": [None, 12, 24],
                "min_samples_leaf": [1, 5, 20], "max_features": ["sqrt", 0.5],
                "class_weight": [None, "balanced"]}
    if family == "gradient_boosting":
        return {"n_estimators": [100, 150], "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3], "subsample": [0.8, 1.0],
                "_weight": ["off", "on"]}
    if family in ("xgboost", "xgboost_selected"):
        return {"n_estimators": [300, 600, 1000], "learning_rate": [0.03, 0.06, 0.1],
                "max_depth": [4, 6, 8], "min_child_weight": [1, 10],
                "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0],
                "_weight": ["off", "on"]}
    if family == "lightgbm":
        return {"n_estimators": [300, 600, 1000], "learning_rate": [0.03, 0.06, 0.1],
                "num_leaves": [31, 63, 127], "min_child_samples": [20, 50],
                "feature_fraction": [0.8, 1.0], "_weight": ["off", "on"]}
    raise KeyError(family)


def sample_configs(family: str, n: int) -> list[dict]:
    sp = space(family)
    cells = [dict(zip(sp, vals)) for vals in product(*sp.values())]
    rng = np.random.default_rng(SEED + sum(map(ord, family)))
    pick = np.sort(rng.choice(len(cells), size=min(n, len(cells)), replace=False))
    return [{**cells[i], "_cell": int(i)} for i in pick]


def make_pipe(family: str, cfg: dict, features: list[str], ratio: float):
    """One sklearn Pipeline: Phase-16 preprocessing + the family's classifier."""
    c = {k: v for k, v in cfg.items() if not k.startswith("_")}
    weighting = "none"
    if family == "random_forest":
        weighting = f"class_weight={c.get('class_weight')}"
        clf = RandomForestClassifier(**{k: v for k, v in c.items()
                                        if k != "class_weight"},
                                     class_weight=c.get("class_weight"),
                                     n_jobs=-1, random_state=SEED)
        sample_weight = None
    elif family == "gradient_boosting":
        weighting = "balanced_sample_weight" if c.pop("_weight", "off") == "on" else "none"
        clf = GradientBoostingClassifier(**c, random_state=SEED)
        sample_weight = "balanced" if weighting != "none" else None
    elif family in ("xgboost", "xgboost_selected"):
        on = c.pop("_weight", "off") == "on"
        weighting = f"scale_pos_weight={ratio:.1f}" if on else "none"
        clf = XGBClassifier(**c, scale_pos_weight=(ratio if on else 1.0),
                            n_jobs=8, tree_method="hist", eval_metric="logloss",
                            random_state=SEED, verbosity=0)
        sample_weight = None
    elif family == "lightgbm":
        on = c.pop("_weight", "off") == "on"
        weighting = f"scale_pos_weight={ratio:.1f}" if on else "none"
        clf = LGBMClassifier(**c, scale_pos_weight=(ratio if on else 1.0),
                             n_jobs=8, verbose=-1, random_state=SEED)
        sample_weight = None
    else:
        raise KeyError(family)

    num = [col for col in features if col not in TEXT_COLS]
    txt = [col for col in features if col in TEXT_COLS]
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("pre", preprocessor(num, txt, scale=False)),
                     ("clf", clf)])
    return pipe, weighting, sample_weight


def fit_pipe(pipe, X, y, sample_weight):
    if sample_weight == "balanced":
        ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
        sw = np.where(y == 1, ratio, 1.0)
        return pipe.fit(X, y, clf__sample_weight=sw)
    if sample_weight is not None and not isinstance(sample_weight, str):
        return pipe.fit(X, y, clf__sample_weight=sample_weight)
    return pipe.fit(X, y)


# ------------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    families = sys.argv[1:] or list(BUDGETS)
    bad = [f for f in families if f not in BUDGETS]
    assert not bad, f"unknown families {bad}"
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    print(f"Phase 20 — classical ML | families={families} | seed={SEED} | "
          f"{pd.Timestamp.now('UTC')}")

    # registry contract
    reg = pd.read_csv(REGISTRY)
    l2 = reg[reg.level == "L2"].set_index("model")
    assert set(l2.index) == set(BUDGETS), "registry L2 rows != families"
    for f in families:
        row = l2.loc[f]
        assert row.test_policy == "frozen_read_at_phase33", f
        assert row.selection_cell == "val_2022" and row.selection_metric == "pr_auc"
        assert int(row.tuning_budget_configs) >= BUDGETS[f], "budget exceeds registry"
    print("registry OK: test_policy=frozen_read_at_phase33, budgets within <=40")

    d, fsets = load()
    tasks = build_tasks(d)
    for t in tasks:
        t["evals"] = [(p, c, m) for p, c, m in t["evals"] if not c.startswith("test")]
        cells = {c for _, c, _ in t["evals"]}
        assert cells <= DEV_CELLS, f"non-dev cell leaked: {cells}"
    assert {c for t in tasks for _, c, _ in t["evals"]} == DEV_CELLS, "dev grid != contract"
    print(f"dev grid OK: {sorted({c for t in tasks for _, c, _ in t['evals']})} "
          f"— frozen test never scored")

    holdout = d.station_holdout.to_numpy()
    split = d.split.to_numpy()
    for t in tasks:
        assert not (t["train"] & holdout).any(), "holdout in a training mask"
        assert not (t["train"] & (split == "test")).any(), "test row in a training mask"

    stations = d.station_id.to_numpy()
    season = d.season.to_numpy()

    # resume: drop partial rows of families we are about to (re)run
    search = pd.read_csv(SEARCH_CSV) if SEARCH_CSV.exists() else pd.DataFrame()
    met = pd.read_csv(MET_CSV) if MET_CSV.exists() else pd.DataFrame()
    todo = []
    for f in families:
        pkl = MODELS / f"phase20_{REG_IDS[f]}.pkl"
        done = (not met.empty and f in set(met.get("model", []))
                and pkl.exists()
                and not search.empty and f in set(search.get("family", [])))
        if done:
            print(f"  skip {f}: already complete ({pkl.name})")
            continue
        if not search.empty:
            search = search[search.family != f]
        if not met.empty:
            met = met[met.model != f]
        todo.append(f)
    if not todo:
        print("nothing to do — writing summary only")
        summarize(search, met)
        return

    rows_search: list[dict] = []
    rows_met: list[dict] = []

    for fam_i, family in enumerate(todo):
        fam_t0 = time.time()
        feats = fsets[FEATURE_SET[family]]
        ratio = None
        best_cfg, best_score, best_cfg_id = None, -1.0, None

        # ---------------- stage 35: validation-only random search
        cfgs = sample_configs(family, BUDGETS[family])
        sp_size = len([1 for _ in product(*space(family).values())])
        print(f"\n=== {family} ({REG_IDS[family]}) — tuning "
              f"{len(cfgs)}/{sp_size} configs, cell=val_2022, score=mean PR-AUC "
              f"over 3 horizons ===")
        p1 = next(t for t in tasks if t["protocol"] == "P1_temporal_primary")
        tri = p1["train"]
        ev_mask = next(m for _, c, m in p1["evals"] if c == "val_2022")

        for ci, cfg in enumerate(cfgs):
            t1 = time.time()
            aucs = {}
            pipes = {}
            for hor in HORIZONS:
                y = d[hor].to_numpy()
                tr_idx = np.flatnonzero(tri & ~np.isnan(y))
                ytr = y[tr_idx].astype(int)
                ev_idx = np.flatnonzero(ev_mask & ~np.isnan(y))
                ratio = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
                pipe, weighting, sw = make_pipe(family, cfg, feats, ratio)
                fit_pipe(pipe, d.iloc[tr_idx][feats], ytr, sw)
                p = pipe.predict_proba(d.iloc[ev_idx][feats])[:, 1]
                aucs[hor] = float(metrics(y[ev_idx].astype(int), p, 0.5)["pr_auc"])
                pipes[hor] = pipe
            mean_auc = float(np.mean(list(aucs.values())))
            cfg_id = f"{family}_c{ci:02d}"
            rows_search.append({
                "family": family, "registry_id": REG_IDS[family],
                "config_id": cfg_id, "space_cell": cfg["_cell"],
                "params": json.dumps({k: v for k, v in cfg.items()
                                      if not k.startswith("_")}),
                "weighting": weighting,
                "pr_auc_24h": aucs[HORIZONS[0]], "pr_auc_48h": aucs[HORIZONS[1]],
                "pr_auc_72h": aucs[HORIZONS[2]], "mean_pr_auc": mean_auc,
                "fit_s": round(time.time() - t1, 1),
                "n_configs_budget": BUDGETS[family], "space_size": sp_size,
                "selection_cell": "val_2022", "feature_set": FEATURE_SET[family],
            })
            if mean_auc > best_score:
                best_score, best_cfg, best_cfg_id = mean_auc, cfg, cfg_id
            print(f"  [{ci + 1}/{len(cfgs)}] {cfg_id} mean={mean_auc:.5f} "
                  f"(24h={aucs[HORIZONS[0]]:.5f} 48h={aucs[HORIZONS[1]]:.5f} "
                  f"72h={aucs[HORIZONS[2]]:.5f}) {weighting} "
                  f"({time.time() - t1:.0f}s)")

        winner = next(r for r in rows_search if r["config_id"] == best_cfg_id)
        print(f"  -> winner {best_cfg_id} mean_pr_auc={best_score:.5f} "
              f"weighting={winner['weighting']}")

        # ---------------- stage 34: dev-cell grid with the selected config
        print(f"=== {family} — grid with {best_cfg_id} "
              f"(8 dev cells x 3 seasons x 3 horizons) ===")
        saved: dict = {"family": family, "registry_id": REG_IDS[family],
                       "config_id": best_cfg_id,
                       "params": {k: v for k, v in best_cfg.items()
                                  if not k.startswith("_")},
                       "weighting": winner["weighting"],
                       "feature_set": FEATURE_SET[family],
                       "features": feats, "seed": SEED,
                       "train": "train_<2022", "horizons": {}}
        for task in tasks:
            for hor in HORIZONS:
                t1 = time.time()
                y_all = d[hor].to_numpy()
                tr_idx = np.flatnonzero(task["train"] & ~np.isnan(y_all))
                ytr = y_all[tr_idx].astype(int)
                if len(np.unique(ytr)) < 2:
                    continue
                ratio = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
                pipe, weighting, sw = make_pipe(family, best_cfg, feats, ratio)
                fit_pipe(pipe, d.iloc[tr_idx][feats], ytr, sw)
                ptrain = pipe.predict_proba(d.iloc[tr_idx][feats])[:, 1]
                thr = best_f1_threshold(ytr, ptrain)
                fit_s = round(time.time() - t1, 1)
                if task["protocol"] == "P1_temporal_primary":
                    saved["horizons"][hor] = pipe
                    saved[f"fit_rows_{hor}"] = int(tr_idx.size)

                for protocol, cell, ev_mask in task["evals"]:
                    for seas in (None, "monsoon", "non-monsoon"):
                        m = score_cell(d, ev_mask, seas)
                        idx = np.flatnonzero(m & ~np.isnan(y_all))
                        if len(idx) == 0:
                            continue
                        y = y_all[idx].astype(int)
                        p = pipe.predict_proba(d.iloc[idx][feats])[:, 1]
                        met_row = metrics(y, p, thr)
                        if seas is None and len(idx) >= 1000 and (
                                len(np.unique(stations[idx])) >= 10):
                            lo, hi = bootstrap_pr_auc(stations[idx], y, p)
                        else:
                            lo, hi = np.nan, np.nan
                        met_row.update(
                            protocol=protocol, train=task["name"], cell=cell,
                            season=seas or "all", model=family, horizon=hor,
                            n=len(idx), train_rows=int(tr_idx.size),
                            train_base_rate=float(ytr.mean()),
                            pr_auc_ci_lo=lo, pr_auc_ci_hi=hi,
                            macro_pr_auc=macro_pr_auc(stations[idx], y, p),
                            fit_s=fit_s, registry_id=REG_IDS[family],
                            feature_set=FEATURE_SET[family],
                            weighting=weighting, config_id=best_cfg_id,
                            search_score=round(best_score, 5))
                        rows_met.append(met_row)
                print(f"  {task['protocol']:18s} {task['name']:13s} {hor} "
                      f"pr_auc(val) fit {fit_s}s")

        joblib.dump(saved, MODELS / f"phase20_{REG_IDS[family]}.pkl")
        # checkpoint after every family
        search = pd.concat([search, pd.DataFrame(rows_search)], ignore_index=True)
        met = pd.concat([met, pd.DataFrame(rows_met)], ignore_index=True)
        rows_search, rows_met = [], []
        search.to_csv(SEARCH_CSV, index=False)
        met.to_csv(MET_CSV, index=False)
        print(f"  checkpoint: {SEARCH_CSV.name} ({len(search)} rows), "
              f"{MET_CSV.name} ({len(met)} rows), "
              f"{(MODELS / f'phase20_{REG_IDS[family]}.pkl').name}, "
              f"family time {time.time() - fam_t0:.0f}s")

    summarize(search, met)
    print(f"\nTOTAL {time.time() - t0:.0f}s — exit=0")


def summarize(search: pd.DataFrame, met: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("PHASE 20 summary")
    print("=" * 78)
    if not search.empty:
        print("\n-- tuning budgets actually used (registry max = 40) --")
        print(search.groupby("family").agg(
            configs=("config_id", "nunique"), space=("space_size", "max"),
            budget=("n_configs_budget", "max"),
            best=("mean_pr_auc", "max"), fit_s=("fit_s", "sum")).round(5).to_string())
        best = search.loc[search.groupby("family")["mean_pr_auc"].idxmax()]
        print("\n-- selected config per family --")
        print(best[["family", "config_id", "params", "weighting",
                    "mean_pr_auc"]].to_string(index=False))
    if met.empty:
        return
    print("\n=== selection cell val_2022 (P1, season=all) — Phase 20 vs Phase 16 ===")
    p16 = pd.read_csv(REPORTS / "phase16_baseline_metrics.csv")
    cmp = pd.concat([met, p16[p16.model.isin(["persistence", "logreg_linear"])]])
    sub = cmp[(cmp.protocol == "P1") & (cmp.cell == "val_2022")
              & (cmp.season == "all")]
    piv = sub.pivot_table(index="model", columns="horizon",
                          values=["pr_auc", "recall", "brier"])
    print(piv.round(4).to_string())
    print("\n=== spatial CV (P3, mean over folds, season=all) — PR-AUC ===")
    sub = pd.concat([met, p16[p16.model.isin(["persistence", "logreg_linear"])]])
    sub = sub[(sub.protocol == "P3") & (sub.season == "all")]
    print(sub.groupby(["model", "horizon"])["pr_auc"].mean().round(4)
          .unstack().to_string())
    print("\n=== rolling origin (P2, per year, season=all) — PR-AUC ===")
    sub = met[(met.protocol == "P2") & (met.season == "all")]
    print(sub.groupby(["model", "cell", "horizon"])["pr_auc"].mean().round(4)
          .unstack().to_string())
    print(f"\nwrote {SEARCH_CSV.name} ({len(search)} rows), {MET_CSV.name} "
          f"({len(met)} rows), models/phase20_L2-0*.pkl")


if __name__ == "__main__":
    main()
