r"""Experiment 1 — Domain adaptation for distribution shift (GroupDRO / IRM / DANN).

Motivation
----------
The champion (decision D17, `models/final_v3/`) loses PR-AUC on the two most recent
test years (2025, 2026) relative to the D16 champion, while gaining on unseen
stations.  The training period (2017-2021) is a wet regime (24 h base rate 7-14 %)
and the evaluation period is progressively drier (2022: 5.1 %, 2025-26: ~1-2 %).
That is a *covariate + concept* shift: ``P_train(y|x) != P_test(y|x)``.

Domain-adaptation methods attack exactly that: they optimise a *worst-group*
objective over a partition of the training data into domains, instead of the
population-average empirical risk.

What this script does
---------------------
1. Builds station-level **domain partitions** from metadata that is available at
   prediction time (basin, elevation quartile, coastline distance, network cohort,
   and a basin x coastal cross).  No label information is used -> no leakage.
2. Trains, under a **strictly out-of-sample** protocol:
     * ``t2020``  train on years <= 2019            -> score 2020
     * ``t2021``  train on years <= 2020            -> score 2021
     * ``t2022``  train on years <= 2021 (all train)-> score 2022
     * ``sf1..sf4``  train on train-split rows with spatial_fold != k, score fold k
   Every scored row is therefore *never seen in training*.  (The MI-8/MI-4
   "7 dev cells" scorecards scored val_2020, val_2021 and spatial_fold_1..4 with a
   model trained on those very rows; only val_2022 was genuinely held out.  This
   script fixes that so the numbers are honest and the comparison meaningful.)
3. Compares, under an identical feature set (FULL_V4, 187 features) and identical
   training rows:
     * ``erm_lgbm``       LightGBM, empirical-risk minimisation (control)
     * ``groupdro_lgbm``  LightGBM, group weights re-estimated by exponentiated
                          gradient (Sagawa et al. 2020, "Online GroupDRO")
     * ``erm_mlp``        torch MLP, empirical-risk minimisation (control for the
                          neural arm)
     * ``groupdro_mlp``   torch MLP, online GroupDRO
     * ``irm_mlp``        torch MLP, IRM v1 (Arjovsky et al. 2019)
     * ``dann_mlp``       torch MLP, domain-adversarial (Ganin et al. 2016) with a
                          gradient-reversal layer and a domain classifier
4. Scores every method on every dev cell, runs a **paired station-level
   bootstrap** against the control, and writes the ledger.

Discipline
----------
* Seed ``20261008`` (``baseline_model.B.SEED``) everywhere.
* PR-AUC is the primary metric; per-cell only, never pooled (D8).
* **No test cell is touched in this script.**  A test read needs a pre-registered
  decision (see ``reports/advanced_experiments.md``).
* Resumable: every (task, horizon, method, scheme) fit is appended to a ledger
  CSV and skipped on restart.

Usage
-----
    python -u scripts/domain_adaptation.py domains
    python -u scripts/domain_adaptation.py probe --horizons label_24h
    python -u scripts/domain_adaptation.py run --horizons label_24h label_48h label_72h
    python -u scripts/domain_adaptation.py report
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
OUT = REPORTS / "domain_adaptation"
OUT.mkdir(parents=True, exist_ok=True)
PRED_DIR = OUT / "preds"
PRED_DIR.mkdir(exist_ok=True)

SEED = B.SEED
HORIZONS = B.HORIZONS
TASKS = ["t2020", "t2021", "t2022", "sf1", "sf2", "sf3", "sf4"]
CELL_OF = {"t2020": "val_2020", "t2021": "val_2021", "t2022": "val_2022",
           "sf1": "spatial_fold_1", "sf2": "spatial_fold_2",
           "sf3": "spatial_fold_3", "sf4": "spatial_fold_4"}
SCHEMES = ["basin", "elev_q", "coastal", "cohort", "hydro"]
N_BOOT = 400

LEDGER = OUT / "da_ledger.csv"
PRED_LEDGER = OUT / "da_predictions.parquet"
SUMMARY = OUT / "da_summary.json"
DOMAIN_CSV = OUT / "da_domains.csv"

# Champion LightGBM configuration for FULL_V4 (MI-4 HPO winner, 24 h: lgbm13).
# The same configuration is used for every arm so that the only thing that
# changes between arms is the *objective*, never the learner or its capacity.
LGBM_CFG = dict(n_estimators=300, learning_rate=0.02, num_leaves=63,
                min_child_samples=40, subsample=0.7, colsample_bytree=0.6,
                reg_lambda=1.0, scale_pos_weight=1.0)

# The frozen D16 champion (models/final_v2/selected_model_v2.pkl) as a reference
# arm under the SAME out-of-sample protocol as the DA arms.  Without this the DA
# gain could be confounded with the FULL_V4 vs FULL_V2 feature difference.
D16_FAMILY = {"label_24h": "lgbm", "label_48h": "lgbm", "label_72h": "xgb"}
D16_CFG = {
    "label_24h": dict(n_estimators=300, learning_rate=0.02, subsample=0.85,
                      colsample_bytree=0.6, reg_lambda=3.0, num_leaves=63,
                      min_child_samples=40),
    "label_48h": dict(n_estimators=300, learning_rate=0.02, subsample=0.85,
                      colsample_bytree=0.6, reg_lambda=3.0, num_leaves=63,
                      min_child_samples=40),
    "label_72h": dict(n_estimators=800, learning_rate=0.02, max_depth=4,
                      min_child_weight=1.0, subsample=1.0, colsample_bytree=0.8,
                      reg_lambda=3.0, scale_pos_weight=1.0),
}

# torch hyper-parameters (identical for every neural arm)
MLP_CFG = dict(hidden=(256, 64), dropout=0.1, lr=1e-3, weight_decay=1e-4,
               epochs=25, batch=4096, pos_weight="sqrt")
GROUP_DRO_ETA = 0.1          # exponentiated-gradient step
GROUP_DRO_CLIP = 5.0         # max relative group weight
IRM_LAMBDA = 1.0             # IRM v1 penalty weight (BCE-logit scale; see fit_mlp)
IRM_ANNEAL = 5               # epochs before the penalty is applied
DANN_LAMBDA = 0.1            # domain-classifier weight
DANN_ANNEAL = 5              # epochs before the adversary is switched on


# --------------------------------------------------------------------------- data
def load_data():
    """FULL_V4 (187 features) + split metadata, one row per (station, date)."""
    exp = pd.read_csv(PROC / "experimental_dataset.csv", parse_dates=["date"])
    v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
    wc = pd.read_csv(PROC / "esa_worldcover_features.csv")
    mi8 = pd.read_csv(PROC / "mi8_features.csv", parse_dates=["date"])
    spl = pd.read_csv(PROC / "split_assignments.csv", parse_dates=["date"])

    d = exp.merge(v2, on=["station_id", "date"], validate="one_to_one")
    d = d.merge(wc, on="station_id", how="left", validate="many_to_one")
    d = d.merge(mi8, on=["station_id", "date"], validate="one_to_one")
    d = d.merge(spl, on=["station_id", "date"], validate="one_to_one")
    d = d.sort_values(["station_id", "date"]).reset_index(drop=True)

    fsets = json.loads((PROC / "feature_sets_v4.json").read_text(encoding="utf-8"))["sets"]
    feats = fsets["FULL_V4"]
    missing = [c for c in feats if c not in d.columns]
    if missing:
        raise RuntimeError(f"missing FULL_V4 columns: {missing[:10]}")
    print(f"loaded {len(d):,} rows x {d.shape[1]} cols; FULL_V4 = {len(feats)} features")
    return d, feats


def build_domains(d: pd.DataFrame) -> pd.DataFrame:
    """Station-level domain partitions.

    Every variable used here is a *station attribute* known at prediction time and
    derived from FFWC metadata / the DEM / the coastline-distance rasters -- never
    from a label.  Phase-13 leakage rule is satisfied by construction.
    """
    st = pd.read_csv(PROC / "station_spatial_features.csv")
    sp = pd.read_csv(PROC / "split_assignments.csv", parse_dates=["date"])

    # network cohort: the cohort label the Phase-15 split assigns to each station
    coh = (sp.sort_values("date").groupby("station_id")
           .agg(cohort=("cohort", "first"), first_year=("first_year", "first")))

    dom = st[["station_id", "name", "river", "basin", "division",
              "elevation_m", "dist_to_coast_km", "dist_river_km"]].copy()
    dom = dom.merge(coh, on="station_id", how="left")

    dom["basin"] = dom["basin"].replace("-", "Unknown")
    dom["elev_q"] = pd.qcut(dom["elevation_m"].rank(method="first"), 4,
                            labels=["q1_low", "q2", "q3", "q4_high"])
    # coastal / inland cut at the 40th percentile of the coastline-distance raster
    # (the absolute distance distribution is extremely skewed: median 2.3 km,
    # 75th pct 6.4 km, max 101 km, so a fixed km cut is not meaningful).
    cut = float(dom["dist_to_coast_km"].quantile(0.40))
    dom["coastal"] = np.where(dom["dist_to_coast_km"] <= cut, "coastal", "inland")
    dom["coast_cut_km"] = cut
    dom["cohort"] = dom["cohort"].astype(str)
    dom["hydro"] = dom["basin"] + "|" + dom["coastal"]

    dom.to_csv(DOMAIN_CSV, index=False)
    print(f"domains written to {DOMAIN_CSV}  ({len(dom)} stations)")
    for s in SCHEMES:
        vc = dom[s].value_counts()
        print(f"  {s:8s} {len(vc):2d} groups  sizes={vc.to_dict()}")
    return dom


def attach_domains(d: pd.DataFrame, dom: pd.DataFrame) -> pd.DataFrame:
    """Attach domain partitions under a ``dom_`` prefix.

    The prefix avoids a collision with the ``basin`` feature column that already
    exists in the experimental table.
    """
    cols = ["station_id"] + [f"dom_{s}" for s in SCHEMES]
    keep = dom[["station_id"] + SCHEMES].copy()
    keep.columns = cols
    d = d.merge(keep, on="station_id", how="left", validate="many_to_one")
    for s in SCHEMES:
        d[f"dom_{s}"] = d[f"dom_{s}"].fillna("Unknown").astype(str)
    return d


def make_tasks(d: pd.DataFrame) -> dict:
    """Strictly out-of-sample train/eval masks (see module docstring)."""
    split = d.split.to_numpy()
    holdout = d.station_holdout.to_numpy()
    year = d.date.dt.year.to_numpy()
    fold = d.spatial_fold.to_numpy()
    train_split = (split == "train") & ~holdout

    tasks = {
        "t2020": (train_split & (year <= 2019), (year == 2020) & ~holdout),
        "t2021": (train_split & (year <= 2020), (year == 2021) & ~holdout),
        "t2022": (train_split, (split == "val") & ~holdout),
    }
    for k in (1, 2, 3, 4):
        tasks[f"sf{k}"] = (train_split & (fold != k),
                           (split == "train") & (fold == k) & ~holdout)
    return tasks


# ------------------------------------------------------------------- torch helpers
def _torch():
    import torch
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(False)
    return torch


class MLP(torch.nn.Module):
    """Small MLP used as the shared representation for ERM / GroupDRO / IRM / DANN."""

    def __init__(self, d_in: int, hidden, dropout: float, n_domains: int = 0):
        super().__init__()
        import torch.nn as nn
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
        self.dom_head = nn.Linear(prev, n_domains) if n_domains > 0 else None

    def forward(self, x, grl=None):
        z = self.body(x)
        out = self.head(z).squeeze(-1)
        dom = None
        if self.dom_head is not None:
            # The reversal must sit on the DOMAIN branch only: applying it to the
            # shared representation would also reverse the label gradient and
            # destroy the classifier.
            dom = self.dom_head(grl(z) if grl is not None else z)
        return out, dom


class _GRLFn(torch.autograd.Function):
    """Gradient reversal layer (Ganin et al. 2016): forward is identity, backward
    multiplies the gradient by ``-lambda``."""

    @staticmethod
    def forward(ctx, inp, lambd):
        ctx.lambd = lambd
        return inp.view_as(inp)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def _grl(lambd):
    def fn(x):
        return _GRLFn.apply(x, lambd)
    return fn


# --------------------------------------------------------------------- model arms
def _lgbm(seed_offset: int = 0):
    cfg = dict(LGBM_CFG)
    cfg["n_estimators"] = int(cfg["n_estimators"])
    return LGBMClassifier(**cfg, n_jobs=-1, random_state=SEED + seed_offset, verbose=-1)


def fit_lgbm_erm(Xtr, ytr, w=None):
    m = _lgbm()
    m.fit(Xtr, ytr, sample_weight=w)
    return m


def fit_lgbm_groupdro(Xtr, ytr, gidx, n_groups, rounds=8, eta=GROUP_DRO_ETA):
    """Online GroupDRO (exponentiated gradient) for a fixed tree learner.

    Round 0 is plain ERM.  After each round the per-group mean log-loss is
    computed on the training rows and the group weights are updated as
    ``q_g <- q_g * exp(eta * (L_g - mean(L)))`` (shift-invariant, so it is immune to
    the overall loss scale, which matters here because the base rate is ~5-14 %).
    Weights are clipped to ``[1/G, CLIP/G]`` so no group can be dropped.
    """
    import torch  # noqa: F401
    q = np.full(n_groups, 1.0 / n_groups)
    model = None
    for t in range(rounds):
        counts = np.bincount(gidx, minlength=n_groups).astype(float)
        keep = counts > 0
        if not keep.any():
            break
        w = q[gidx] / np.maximum(counts[gidx], 1.0)
        w = w * (len(gidx) / w.sum())
        model = _lgbm(seed_offset=t)
        model.fit(Xtr, ytr, sample_weight=w)

        p = np.clip(model.predict_proba(Xtr)[:, 1], 1e-6, 1 - 1e-6)
        loss = -(ytr * np.log(p) + (1 - ytr) * np.log(1 - p))
        gl = np.zeros(n_groups)
        for g in range(n_groups):
            m = gidx == g
            if m.sum():
                gl[g] = loss[m].mean()
        valid = keep & np.isfinite(gl)
        if valid.sum() == 0:
            break
        gl_c = gl[valid] - gl[valid].mean()
        upd = np.zeros(n_groups)
        upd[valid] = np.exp(eta * gl_c)
        q = q * upd
        q = np.where(keep, q, 1.0 / n_groups)
        q = np.clip(q, 1.0 / (n_groups * GROUP_DRO_CLIP), GROUP_DRO_CLIP / n_groups)
        q = q / q.sum()
    return model


def _prep_xy(Xtr, ytr, Xev, scale=True):
    """Median-impute + standardise numeric, frequency-encode text (train-fitted)."""
    num = [c for c in Xtr.columns if c not in B.TEXT_COLS]
    txt = [c for c in Xtr.columns if c in B.TEXT_COLS]
    pre = B.preprocessor(num, txt, scale=scale)
    A = pre.fit_transform(Xtr)
    E = pre.transform(Xev)
    return (np.asarray(A, dtype=np.float32), np.asarray(E, dtype=np.float32),
            pre, (num, txt))


def fit_mlp(mode: str, A, ytr, gidx, n_groups, Xev=None, seed_offset: int = 0):
    """Train the torch MLP under the requested objective.  Returns (model, pre, meta)."""
    mode = mode.replace("_mlp", "")
    torch = _torch()
    import torch.nn as nn

    torch.manual_seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)

    A_t = torch.from_numpy(A)
    y_t = torch.from_numpy(ytr.astype(np.float32))
    g_t = torch.from_numpy(gidx.astype(np.int64))

    n_in = A.shape[1]
    model = MLP(n_in, MLP_CFG["hidden"], MLP_CFG["dropout"],
                n_domains=n_groups if mode == "dann" else 0)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_CFG["lr"],
                           weight_decay=MLP_CFG["weight_decay"])

    pos = ytr.sum()
    neg = len(ytr) - pos
    pw = torch.tensor(float(np.sqrt(neg / max(pos, 1))))
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)

    n = len(ytr)
    bs = MLP_CFG["batch"]
    steps = max(1, n // bs)
    q = np.full(n_groups, 1.0 / n_groups) if mode == "groupdro" else None

    dom_lookup = None
    if mode == "dann":
        dom_lookup = torch.arange(n_groups)

    for ep in range(MLP_CFG["epochs"]):
        perm = torch.randperm(n)
        model.train()
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            if len(idx) < 32:
                continue
            xb, yb, gb = A_t[idx], y_t[idx], g_t[idx]
            opt.zero_grad()

            if mode == "dann":
                lam = DANN_LAMBDA if ep >= DANN_ANNEAL else 0.0
                lbl, dom_out = model(xb, grl=_grl(lam) if lam > 0 else None)
                loss = bce(lbl, yb).mean()
                if lam > 0:
                    dloss = nn.functional.cross_entropy(dom_out, gb)
                    loss = loss + lam * dloss
            elif mode == "irm":
                logits, _ = model(xb)
                per = bce(logits, yb)
                loss = per.mean()
                if ep >= IRM_ANNEAL:
                    # IRMv1: dummy scalar w=1 on the logits; penalty = ||d L_g / d w||^2,
                    # averaged over the environments present in the batch.
                    penalty = torch.zeros(())
                    n_present = 0
                    for g in torch.unique(gb).tolist():
                        m = gb == g
                        if int(m.sum()) < 2:
                            continue
                        scale = torch.ones(1, requires_grad=True)
                        gl = bce(logits[m] * scale, yb[m]).mean()
                        g1 = torch.autograd.grad(gl, [scale], create_graph=True)[0]
                        penalty = penalty + (g1 ** 2).sum()
                        n_present += 1
                    if n_present > 0:
                        loss = loss + (IRM_LAMBDA / n_present) * penalty
            elif mode == "groupdro":
                logits, _ = model(xb)
                per = bce(logits, yb)
                # online exponentiated gradient on the groups present in the batch
                present = torch.unique(gb)
                gl = torch.zeros(n_groups)
                cnt = torch.zeros(n_groups)
                for g in present.tolist():
                    m = gb == g
                    gl[g] = per[m].mean()
                    cnt[g] = float(m.sum())
                wq = torch.from_numpy(q[gb.numpy()].astype(np.float32))
                loss = (per * wq).mean() / max(float(wq.mean()), 1e-8)
            else:                                     # erm
                logits, _ = model(xb)
                loss = bce(logits, yb).mean()

            loss.backward()
            opt.step()

            if mode == "groupdro":
                with torch.no_grad():
                    vals = torch.tensor([gl[g].item() if cnt[g] > 0 else np.nan
                                         for g in range(n_groups)])
                    good = ~torch.isnan(vals)
                    if good.sum() > 0:
                        v = vals[good]
                        upd = torch.ones(n_groups)
                        upd[good] = torch.exp(GROUP_DRO_ETA * (v - v.mean()))
                        q = q * upd.numpy()
                        q = np.clip(q, 1.0 / (n_groups * GROUP_DRO_CLIP),
                                    GROUP_DRO_CLIP / n_groups)
                        q = q / q.sum()

    return model, pre if False else None, None


# ------------------------------------------------------------------------ helpers
def predict_mlp(model, E):
    torch = _torch()
    with torch.no_grad():
        out, _ = model(torch.from_numpy(E))
        return torch.sigmoid(out).numpy()


def pr_auc(y, p):
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


def paired_bootstrap(y, p1, p2, stations, n_boot=N_BOOT, seed=SEED):
    """Paired station-level bootstrap of PR-AUC(p1) - PR-AUC(p2)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(stations)
    idx_by = {s: np.flatnonzero(stations == s) for s in uniq}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(pr_auc(y[idx], p1[idx]) - pr_auc(y[idx], p2[idx]))
    if len(diffs) < 20:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, p_gt_0=np.nan, n=len(diffs))
    diffs = np.asarray(diffs)
    return dict(mean=float(diffs.mean()), lo=float(np.percentile(diffs, 2.5)),
                hi=float(np.percentile(diffs, 97.5)),
                p_gt_0=float((diffs > 0).mean()), n=len(diffs))


def encode_groups(d: pd.DataFrame, scheme: str, idx: np.ndarray) -> tuple[np.ndarray, int]:
    col = f"dom_{scheme}"
    all_vals = d[col].to_numpy()
    vals = all_vals[idx]
    cats = sorted(pd.unique(all_vals))
    mapping = {c: i for i, c in enumerate(cats)}
    g = np.array([mapping[v] for v in vals], dtype=np.int64)
    return g, len(cats)


# --------------------------------------------------------------------------- main
def cmd_domains(args) -> int:
    d, _ = load_data()
    dom = build_domains(d)
    d = attach_domains(d, dom)
    print("\n=== domain sizes within the 2017-2021 training rows ===")
    tr = d[(d.split == "train") & (~d.station_holdout)]
    for s in SCHEMES:
        col = f"dom_{s}"
        vc = tr[col].value_counts()
        pos = tr.groupby(col)["label_24h"].mean().round(4).to_dict()
        print(f"\n{s}:")
        for g in vc.index:
            print(f"   {str(g):28s} rows={vc[g]:7,}  base_rate_24h={pos.get(g, np.nan)}")
    return 0


LEDGER_COLS = ["horizon", "method", "scheme", "task", "cell", "n_train", "n_eval",
               "base_rate", "pr_auc", "base_rate_train", "secs"]


def _load_ledger() -> pd.DataFrame:
    if LEDGER.exists():
        df = pd.read_csv(LEDGER)
        if len(df):
            return df
    return pd.DataFrame(columns=LEDGER_COLS)


def _save_ledger(df: pd.DataFrame) -> None:
    df.to_csv(LEDGER, index=False)


def run_arm(d, feats, tasks, horizon, method, scheme, save_preds=True):
    """Train one (method, scheme) arm on every task for one horizon."""
    y_all = d[horizon].to_numpy()
    num = [c for c in feats if c not in B.TEXT_COLS]
    txt = [c for c in feats if c in B.TEXT_COLS]
    rows, preds = [], []

    for task in TASKS:
        tr_mask, ev_mask = tasks[task]
        tr_idx = np.flatnonzero(tr_mask & ~np.isnan(y_all))
        ev_idx = np.flatnonzero(ev_mask & ~np.isnan(y_all))
        if len(tr_idx) == 0 or len(ev_idx) == 0:
            continue
        ytr = y_all[tr_idx].astype(int)
        yev = y_all[ev_idx].astype(int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yev)) < 2:
            continue

        t0 = time.time()
        if method == "erm_d16":
            # Champion reference: FULL_V2 with the frozen D16 configurations.
            # Scheme-independent by construction -- it exists so that a DA gain
            # cannot be confused with the FULL_V4 vs FULL_V2 feature difference.
            v2 = json.loads((PROC / "feature_sets_v4.json").read_text(
                encoding="utf-8"))["sets"]["FULL_V2"]
            num2 = [c for c in v2 if c not in B.TEXT_COLS]
            txt2 = [c for c in v2 if c in B.TEXT_COLS]
            pre = B.preprocessor(num2, txt2, scale=False)
            A = pre.fit_transform(d.loc[tr_idx, v2])
            E = pre.transform(d.loc[ev_idx, v2])
            fam = D16_FAMILY[horizon]
            cfg = dict(D16_CFG[horizon])
            if fam == "lgbm":
                m = LGBMClassifier(**cfg, random_state=SEED, n_jobs=-1, verbose=-1)
            else:
                from xgboost import XGBClassifier
                m = XGBClassifier(**cfg, random_state=SEED, n_jobs=-1,
                                  eval_metric="logloss", tree_method="hist")
            m.fit(A, ytr)
            p_ev = m.predict_proba(E)[:, 1]
            p_tr = m.predict_proba(A)[:, 1]
        elif method in ("erm_lgbm", "groupdro_lgbm"):
            Xtr = d.loc[tr_idx, feats]
            Xev = d.loc[ev_idx, feats]
            pre = B.preprocessor(num, txt, scale=False)
            A = pre.fit_transform(Xtr)
            E = pre.transform(Xev)
            gidx, n_groups = encode_groups(d, scheme, tr_idx)
            if method == "erm_lgbm":
                m = fit_lgbm_erm(A, ytr)
            else:
                m = fit_lgbm_groupdro(A, ytr, gidx, n_groups)
            p_ev = m.predict_proba(E)[:, 1]
            p_tr = m.predict_proba(A)[:, 1]
        else:
            Xtr = d.loc[tr_idx, feats]
            Xev = d.loc[ev_idx, feats]
            A, E, pre, _ = _prep_xy(Xtr, ytr, Xev, scale=True)
            gidx, n_groups = encode_groups(d, scheme, tr_idx)
            m, _, _ = fit_mlp(method, A, ytr, gidx, n_groups)
            p_ev = predict_mlp(m, E)
            p_tr = predict_mlp(m, A)

        auc = pr_auc(yev, p_ev)
        rows.append(dict(horizon=horizon, method=method, scheme=scheme, task=task,
                         cell=CELL_OF[task], n_train=len(tr_idx), n_eval=len(ev_idx),
                         base_rate=float(yev.mean()), pr_auc=auc,
                         base_rate_train=float(ytr.mean()), secs=round(time.time() - t0, 1)))
        if save_preds:
            preds.append(pd.DataFrame(dict(
                horizon=horizon, method=method, scheme=scheme, task=task,
                row=ev_idx, y=yev, p=p_ev,
                station_id=d.station_id.to_numpy()[ev_idx])))
        print(f"    {method:14s} {scheme:8s} {task:6s} n={len(ev_idx):6,} "
              f"br={yev.mean():.4f} PR-AUC={auc:.6f}  ({time.time()-t0:.0f}s)", flush=True)

    return rows, preds


def cmd_probe(args) -> int:
    """24 h-only probe: pick the domain scheme (and check the methods work)."""
    d, feats = load_data()
    dom = build_domains(d)
    d = attach_domains(d, dom)
    tasks = make_tasks(d)
    ledger = _load_ledger()

    for scheme in SCHEMES:
        for method in args.methods:
            key = (method, scheme, "label_24h")
            done = ledger[(ledger.method == method) & (ledger.scheme == scheme) &
                          (ledger.horizon == "label_24h")]
            if len(done):
                print(f"skip {method}/{scheme}/label_24h (already in ledger)")
                continue
            print(f"\n=== {method} / {scheme} / label_24h ===", flush=True)
            rows, preds = run_arm(d, feats, tasks, "label_24h", method, scheme)
            if rows:
                ledger = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
                _save_ledger(ledger)
                _append_preds(preds)
    print("\nprobe done")
    return 0


def cmd_run(args) -> int:
    d, feats = load_data()
    dom = build_domains(d)
    d = attach_domains(d, dom)
    tasks = make_tasks(d)
    ledger = _load_ledger()

    for horizon in args.horizons:
        for scheme in args.schemes:
            for method in args.methods:
                done = ledger[(ledger.method == method) & (ledger.scheme == scheme) &
                              (ledger.horizon == horizon)]
                if len(done):
                    print(f"skip {method}/{scheme}/{horizon} (in ledger)")
                    continue
                print(f"\n=== {method} / {scheme} / {horizon} ===", flush=True)
                rows, preds = run_arm(d, feats, tasks, horizon, method, scheme)
                if rows:
                    ledger = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
                    _save_ledger(ledger)
                    _append_preds(preds)
    print("\nrun done")
    return 0


def _pred_path(method: str, scheme: str, horizon: str) -> Path:
    return PRED_DIR / f"pred_{method}__{scheme}__{horizon}.csv.gz"


def _append_preds(preds):
    """Persist one gzip CSV per (method, scheme, horizon)."""
    if not preds:
        return
    new = pd.concat(preds, ignore_index=True)
    key = (new.method.iloc[0], new.scheme.iloc[0], new.horizon.iloc[0])
    path = _pred_path(*key)
    if path.exists():
        try:
            old = pd.read_csv(path)
            new = pd.concat([old, new], ignore_index=True)
        except Exception:
            pass
    new.drop_duplicates(subset=["row"], keep="last").to_csv(path, index=False, compression="gzip")


def _load_preds() -> pd.DataFrame:
    frames = []
    for p in sorted(PRED_DIR.glob("pred_*.csv.gz")):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def cmd_report(args) -> int:
    """Assemble the scorecard + paired bootstrap vs the ERM control."""
    if not LEDGER.exists():
        print("no ledger")
        return 1
    led = pd.read_csv(LEDGER)
    preds = _load_preds()

    rows = []
    for (method, scheme, horizon), g in led.groupby(["method", "scheme", "horizon"]):
        g = g.set_index("task")
        rec = dict(method=method, scheme=scheme, horizon=horizon)
        for t in TASKS:
            if t in g.index:
                rec[CELL_OF[t]] = float(g.loc[t, "pr_auc"])
        cells = [CELL_OF[t] for t in TASKS if t in g.index]
        rec["dev_mean"] = float(np.mean([rec[c] for c in cells])) if cells else np.nan
        rec["dev_worst"] = float(np.min([rec[c] for c in cells])) if cells else np.nan
        rec["val_2022"] = rec.get("val_2022", np.nan)
        rows.append(rec)
    score = pd.DataFrame(rows)
    score.to_csv(OUT / "da_scorecard.csv", index=False)

    boots = []
    if len(preds):
        for (method, scheme, horizon), g in preds.groupby(["method", "scheme", "horizon"]):
            # Two controls: the FULL_V4 ERM arm (isolates the objective) and the
            # frozen-D16-config arm on FULL_V2 (isolates the real champion).
            for ctrl_method in ("erm_d16", "erm_lgbm"):
                if method == ctrl_method:
                    continue
                if ctrl_method == "erm_d16":
                    ctrl = preds[(preds.method == ctrl_method) & (preds.horizon == horizon)]
                    if "scheme" in ctrl.columns and len(ctrl.scheme.unique()) > 1:
                        ctrl = ctrl[ctrl.scheme == ctrl.scheme.iloc[0]]
                else:
                    ctrl = preds[(preds.method == ctrl_method) & (preds.scheme == scheme) &
                                 (preds.horizon == horizon)]
                for task in TASKS:
                    a = g[g.task == task]
                    b = ctrl[ctrl.task == task]
                    if len(a) == 0 or len(b) == 0:
                        continue
                    m = a.merge(b, on="row", suffixes=("", "_c"))
                    if len(m) == 0:
                        continue
                    bs = paired_bootstrap(m.y.to_numpy(), m.p.to_numpy(),
                                          m.p_c.to_numpy(), m.station_id.to_numpy())
                    boots.append(dict(method=method, scheme=scheme, horizon=horizon,
                                      task=task, cell=CELL_OF[task],
                                      control=ctrl_method, **bs))
    bdf = pd.DataFrame(boots)
    if len(bdf):
        bdf.to_csv(OUT / "da_bootstrap.csv", index=False)

    summary = dict(
        seed=SEED,
        protocol=dict(
            temporal="rolling origin: t2020 train<=2019, t2021 train<=2020, t2022 train<=2021",
            spatial="train-split rows with spatial_fold != k scored on fold k",
            note="every scored row is out-of-sample; no test cell is read"),
        lgbm_cfg=LGBM_CFG, mlp_cfg={k: (list(v) if isinstance(v, tuple) else v)
                                    for k, v in MLP_CFG.items()},
        group_dro_eta=GROUP_DRO_ETA, irm_lambda=IRM_LAMBDA, dann_lambda=DANN_LAMBDA,
        n_scorecards=int(len(score)), n_boot_rows=int(len(bdf)),
    )
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(score.to_string(index=False))
    print(f"\nwrote {OUT/'da_scorecard.csv'} and {OUT/'da_bootstrap.csv'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("domains").set_defaults(func=cmd_domains)

    p = sub.add_parser("probe")
    p.add_argument("--methods", nargs="+",
                   default=["erm_d16", "erm_lgbm", "groupdro_lgbm", "erm_mlp",
                            "groupdro_mlp", "irm_mlp", "dann_mlp"])
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("run")
    p.add_argument("--horizons", nargs="+", default=B.HORIZONS)
    p.add_argument("--schemes", nargs="+", default=SCHEMES)
    p.add_argument("--methods", nargs="+",
                   default=["erm_d16", "erm_lgbm", "groupdro_lgbm", "erm_mlp",
                            "groupdro_mlp", "irm_mlp", "dann_mlp"])
    p.set_defaults(func=cmd_run)

    sub.add_parser("report").set_defaults(func=cmd_report)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
