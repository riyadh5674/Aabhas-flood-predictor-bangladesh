"""MI-3c: probability ensemble over the best HPO configs + diverse members.

Reads reports/mi3_hpo.csv, retrains the top-k configs per horizon plus two diverse
sklearn families, and compares the ensemble against the best single model on the
seven development cells (validation-only).

MI-3b companion: the HPO pool now also contains the sklearn HistGradientBoosting and
ExtraTrees families, so the family string is carried through verbatim (never folded
into "lgbm") and make_pipe knows how to rebuild each family.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B          # noqa: E402

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
HOR = B.HORIZONS
DEV = ["val_2020", "val_2021", "val_2022",
       "spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]
CFG_KEYS = {"n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample",
            "colsample_bytree", "reg_lambda", "scale_pos_weight", "num_leaves",
            "min_child_samples", "max_iter", "max_leaf_nodes", "min_samples_leaf",
            "l2_regularization", "min_samples_split", "max_features"}
# Hyperparameters that a learner insists on as int. The HPO CSV round-trips every
# numeric column as float (each family leaves the others' cells NaN), so these must be
# coerced back before a learner is rebuilt -- e.g. LightGBM rejects n_estimators="300.0".
INT_KEYS = {"n_estimators", "max_depth", "num_leaves", "min_child_samples",
            "max_iter", "max_leaf_nodes", "min_samples_leaf", "min_samples_split"}


def _coerce(cfg: dict) -> dict:
    """Restore the int/float/str type of a config read back from mi3_hpo.csv."""
    out = {}
    for k, v in cfg.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, str):
            out[k] = v
        elif k in INT_KEYS:
            out[k] = int(round(float(v)))
        else:
            out[k] = float(v)
    return out


def load_v2():
    d, _ = B.load()
    v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
    d = d.merge(v2, on=["station_id", "date"], validate="one_to_one")
    fs2 = json.loads((PROC / "feature_sets_v2.json").read_text(encoding="utf-8"))["sets"]
    return d, fs2["FULL_V2"]


def dev_masks(d):
    holdout = d.station_holdout.to_numpy()
    year = d.date.dt.year.to_numpy()
    split = d.split.to_numpy()
    fold = d.spatial_fold.to_numpy()
    return {"val_2020": (year == 2020) & ~holdout,
            "val_2021": (year == 2021) & ~holdout,
            "val_2022": (split == "val") & ~holdout,
            **{f"spatial_fold_{k}": (split == "train") & (fold == k) for k in (1, 2, 3, 4)}}


def make_pipe(family, cfg, feats):
    from sklearn.pipeline import Pipeline
    cfg = _coerce(cfg or {})
    num = [c for c in feats if c not in B.TEXT_COLS]
    txt = [c for c in feats if c in B.TEXT_COLS]
    if family == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(**cfg, n_jobs=8, tree_method="hist", eval_metric="logloss",
                            random_state=B.SEED, verbosity=0)
    elif family == "lgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(**cfg, n_jobs=8, random_state=B.SEED, verbose=-1)
    elif family == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(**cfg, random_state=B.SEED)
    elif family == "rf":
        from sklearn.ensemble import RandomForestClassifier
        base = dict(n_estimators=500, min_samples_leaf=3, n_jobs=8,
                    class_weight="balanced_subsample")
        base.update(cfg or {})
        clf = RandomForestClassifier(**base, random_state=B.SEED)
    else:  # et
        from sklearn.ensemble import ExtraTreesClassifier
        base = dict(n_estimators=500, min_samples_leaf=3, n_jobs=8,
                    class_weight="balanced_subsample")
        base.update(cfg or {})
        clf = ExtraTreesClassifier(**base, random_state=B.SEED)
    return Pipeline([("pre", B.preprocessor(num, txt, scale=False)), ("clf", clf)])


def cells_pr(prob, y_all, masks):
    out = {}
    for cell in DEV:
        m = masks[cell] & ~pd.isna(y_all)
        out[cell] = B.metrics(y_all[m], prob[m],
                              B.best_f1_threshold(y_all[m], prob[m]))["pr_auc"]
    return out


def main() -> int:
    t0 = time.time()
    d, feats = load_v2()
    masks = dev_masks(d)
    train = (d.split.to_numpy() == "train") & ~d.station_holdout.to_numpy()
    hpo = pd.read_csv(REPORTS / "mi3_hpo.csv")

    rows = []
    for h in HOR:
        y_all = d[h].to_numpy()
        tr = train & ~pd.isna(y_all)
        Xtr, ytr = d.loc[tr, feats], y_all[tr]

        top = hpo[hpo.horizon == h].sort_values("val_2022", ascending=False).head(3)
        members = [(str(r.family),
                    {k: r[k] for k in CFG_KEYS if k in r and pd.notna(r[k])})
                   for _, r in top.iterrows()]
        members += [("rf", {}), ("et", {})]

        probs = {}
        for i, (fam, cfg) in enumerate(members):
            pipe = make_pipe(fam, cfg, feats)
            pipe.fit(Xtr, ytr)
            probs[f"{fam}{i}"] = pipe.predict_proba(d[feats])[:, 1]
        mat = np.vstack(list(probs.values()))
        ens_mean = mat.mean(axis=0)
        ens_rank = np.vstack([pd.Series(p).rank(pct=True).to_numpy() for p in probs.values()]).mean(axis=0)

        for name, prob in list(probs.items()) + [("ENS_mean", ens_mean), ("ENS_rank", ens_rank)]:
            cpr = cells_pr(np.asarray(prob), y_all, masks)
            rows.append({"horizon": h, "model": name,
                         "val_2022": cpr["val_2022"],
                         "dev_mean": float(np.mean(list(cpr.values()))), **cpr})
        print(f"  {h}: best single val2022={max(r['val_2022'] for r in rows if r['horizon']==h and not r['model'].startswith('ENS')):.4f} "
              f"| ENS_mean={cells_pr(ens_mean, y_all, masks)['val_2022']:.4f} "
              f"| ENS_rank={cells_pr(ens_rank, y_all, masks)['val_2022']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "mi3_ensemble.csv", index=False)
    print("\n=== ensemble vs best single (val_2022) ===")
    for h in HOR:
        sub = out[out.horizon == h]
        print(f"  {h}: best_single={sub[~sub.model.str.startswith('ENS')].val_2022.max():.4f}  "
              f"ENS_mean={sub[sub.model=='ENS_mean'].val_2022.iloc[0]:.4f}  "
              f"ENS_rank={sub[sub.model=='ENS_rank'].val_2022.iloc[0]:.4f}")
    print(f"\nWROTE {REPORTS/'mi3_ensemble.csv'}  wall={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
