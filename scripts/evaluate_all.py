"""Phase 27 — fair ML vs DL comparison (one table, one code path, one set of cells).

Deliverables
------------
* reports/phase27_metrics.csv        master comparison table (L1-L4, all dev cells)
* reports/phase27_p1_rescore.csv     fresh P1 val_2022 re-score (all 24 models, 1 path)
* reports/phase27_operational.csv    operational block (missed floods, FAR, lead time)
* reports/phase27_cost.csv           cost block (params, train/inference time, hardware)
* reports/phase27_generalisation.csv generalisation blocks (temporal / spatial / season)
* reports/phase27_summary.json       parity + headline numbers
* figures/phase27_ml_vs_dl.png       headline PR-AUC with station-bootstrap 95 % CI

Design (D14 fairness)
---------------------
The per-phase grids (phases 16, 20-25) already scored **every** registered model on the
same 8 dev cells x 3 seasons x 3 horizons with the *same* metric function
(`baseline_model.metrics`).  Phase 27 (a) re-scores the **P1 primary cell fresh** for all
24 models through ONE load+predict+score path — this is the independent proof that the
grids are comparable — and (b) consolidates the grids into the single master table.
The parity column `delta_pr_auc` (fresh re-score minus grid) is reported for every cell.

The frozen test period (>= 2023) is never read: every mask is built from `split != test`
columns only.  The `test_unseen` / `test_new_network` buckets are deliberately deferred to
Phase 33; `P3` is the dev-time unseen-location protocol (phase15: "unseen-location
transfer during development").
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_model as B          # noqa: E402
import train_sequence as ts         # noqa: E402
import train_transformer as T       # noqa: E402
import train_fusion as tf           # noqa: E402
import train_gnn as g               # noqa: E402

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
REGISTRY = REPORTS / "phase17_experiment_registry.csv"

SEED = B.SEED
HOR = B.HORIZONS
SEASONS = ("all", "monsoon", "non-monsoon")
HOR_DAYS = {"label_24h": 1, "label_48h": 2, "label_72h": 3}
HARDWARE = (f"CPU · {platform.machine()} · {platform.system()} · "
            f"torch {torch.__version__}")

GRID_FILES = [
    "phase16_baseline_metrics.csv",
    "phase20_metrics.csv", "phase21_metrics.csv", "phase22_metrics.csv",
    "phase23_metrics.csv", "phase24_metrics.csv", "phase25_metrics.csv",
]

# model name (canonical, = grid name) -> catalogue entry
CATALOG = [
    dict(model="climatology", level="L1", rid="L1-01", kind="l1"),
    dict(model="persistence", level="L1", rid="L1-02", kind="l1"),
    dict(model="logreg_linear", level="L1", rid="L1-03", kind="l1"),
    dict(model="logreg_linear_no_up", level="L1", rid="L1-04", kind="l1"),
    dict(model="logreg_selected", level="L1", rid="L1-05", kind="l1"),
    dict(model="tree_depth4", level="L1", rid="L1-06", kind="l1"),
    dict(model="random_forest", level="L2", rid="L2-01", kind="l2",
         path="phase20_L2-01.pkl"),
    dict(model="gradient_boosting", level="L2", rid="L2-02", kind="l2",
         path="phase20_L2-02.pkl"),
    dict(model="xgboost", level="L2", rid="L2-03", kind="l2",
         path="phase20_L2-03.pkl"),
    dict(model="lightgbm", level="L2", rid="L2-04", kind="l2",
         path="phase20_L2-04.pkl"),
    dict(model="xgboost_selected", level="L2", rid="L2-05", kind="l2",
         path="phase20_L2-05.pkl"),
    dict(model="lstm_L3", level="L3", rid="L3-01", kind="seq",
         path="phase21_L3-01.pt"),
    dict(model="lstm_L7", level="L3", rid="L3-02", kind="seq",
         path="phase21_L3-02.pt"),
    dict(model="lstm_L14", level="L3", rid="L3-03", kind="seq",
         path="phase21_L3-03.pt"),
    dict(model="lstm_L30", level="L3", rid="L3-04", kind="seq",
         path="phase21_L3-04.pt"),
    dict(model="gru_bestL", level="L3", rid="L3-05", kind="seq",
         path="phase21_L3-05.pt"),
    dict(model="tcn_bestL", level="L3", rid="L3-06", kind="seq",
         path="phase22_L3-06.pt"),
    dict(model="fusion_L3", level="L3", rid="L3-07", kind="fusion",
         path="phase23_L3-07.pt"),
    dict(model="static_only", level="L3", rid="L3-08", kind="fusion",
         path="phase23_L3-08.pt"),
    dict(model="hybrid_L3", level="L3", rid="L3-09", kind="fusion",
         path="phase23_L3-09.pt"),
    dict(model="gcn", level="L4", rid="L4-01", kind="gnn",
         path="phase24_L4-01.pt"),
    dict(model="gat", level="L4", rid="L4-02", kind="gnn",
         path="phase24_L4-02.pt"),
    dict(model="transformer_bestL", level="L4", rid="L4-03", kind="transformer",
         path="phase25_L4-03.pt"),
    dict(model="gnn_lstm", level="L4", rid="L4-04", kind="gnn",
         path="phase24_L4-04.pt"),
]
CAT_BY_NAME = {c["model"]: c for c in CATALOG}


# --------------------------------------------------------------------------- utils
def _remap_legacy(sd: dict) -> dict:
    """Phase-21 LSTM/GRU checkpoints predate the `rnn` -> `enc` attribute rename."""
    return {("enc." + k[4:] if k.startswith("rnn.") else k): v for k, v in sd.items()}


def _ci(y, p, stations) -> tuple[float, float]:
    if len(np.unique(stations)) < 10:
        return np.nan, np.nan
    return B.bootstrap_pr_auc(stations, y, p)


def operational(dates: np.ndarray, stations: np.ndarray, y: np.ndarray,
                p: np.ndarray, thr: float, h_days: int) -> dict:
    """Row- and event-level operational block (all definitions stated in the report).

    * missed_flood_rows = FN (label 1, not warned)
    * false-alarm rate (FAR) = FP / (FP + TP)   (fraction of alarms that are wrong)
    * false-positive rate (FPR) = FP / (FP + TN) (probability of a false alarm)
    * event = maximal run of consecutive calendar days (gap <= 1 d) with label == 1,
      per station.  An event is *detected* if any warning day lies in
      [start - h_days, end].  Event recall = detected / total; event misses = rest.
    * warning lead time = start - first_warning_day in that window (>= 0 days)
    """
    warn = p >= thr
    tp = int((warn & (y == 1)).sum()); fp = int((warn & (y == 0)).sum())
    fn = int((~warn & (y == 1)).sum()); tn = int((~warn & (y == 0)).sum())
    out = {
        "missed_flood_rows": fn,
        "false_alarm_rate": fp / (fp + tp) if (fp + tp) else np.nan,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else np.nan,
        "alarm_count": tp + fp,
    }
    df = pd.DataFrame({"st": stations, "dt": dates, "y": y, "w": warn}) \
        .sort_values(["st", "dt"], kind="stable")
    total = det = 0
    leads: list[int] = []
    one = np.timedelta64(1, "D")
    for _, gs in df.groupby("st", sort=True):
        yy = gs.y.to_numpy(); ww = gs.w.to_numpy(); dd = gs.dt.to_numpy()
        n = len(gs); i = 0
        while i < n:
            if yy[i] != 1:
                i += 1
                continue
            j = i
            while (j + 1 < n and yy[j + 1] == 1
                   and (dd[j + 1] - dd[j]) <= one):
                j += 1
            s_e, e_e = dd[i], dd[j]
            win = (dd >= s_e - np.timedelta64(h_days, "D")) & (dd <= e_e)
            total += 1
            if ww[win].any():
                det += 1
                t_first = dd[win][ww[win]].min()
                leads.append(int(max(0, (s_e - t_first) / one)))
            i = j + 1
    out.update({
        "event_count": total,
        "event_detected": det,
        "event_recall": det / total if total else np.nan,
        "event_missed": total - det,
        "lead_mean_days": float(np.mean(leads)) if leads else np.nan,
        "lead_median_days": float(np.median(leads)) if leads else np.nan,
    })
    return out


def score(y, p, thr, stations, dates, hor, do_ci: bool) -> tuple[dict, dict]:
    """THE one scoring path: B.metrics + derived ratios + CI + macro + operational."""
    met = B.metrics(y, p, thr)
    tp, fp, fn, tn = met["tp"], met["fp"], met["fn"], met["tn"]
    met["fnr"] = fn / (fn + tp) if (fn + tp) else np.nan
    met["far"] = fp / (fp + tp) if (fp + tp) else np.nan
    met["fpr"] = fp / (fp + tn) if (fp + tn) else np.nan
    met["macro_pr_auc"] = B.macro_pr_auc(stations, y, p)
    lo, hi = _ci(y, p, stations) if do_ci else (np.nan, np.nan)
    met["pr_auc_ci_lo"], met["pr_auc_ci_hi"] = lo, hi
    op = operational(dates, stations, y, p, thr, HOR_DAYS[hor])
    return met, op


# ----------------------------------------------------------------- predictors
class Data:
    def __init__(self):
        self.d, self.fsets = B.load()
        self.ix = pd.read_csv(PROC / "sequence_index.csv", parse_dates=["date"])
        assert len(self.d) == len(self.ix) == 200665
        assert (self.d.station_id.to_numpy() == self.ix.station_id.to_numpy()).all()
        self.station = self.ix.station_id.to_numpy()
        self.season = self.d.season.to_numpy()
        self.date = self.d.date.to_numpy()
        self.models_l1 = B.build_models(self.fsets)
        # gnn index maps (built once)
        self.dp, self.sp, self.row_of, self.sp_index = g.build_index_maps(self.ix)
        tf.STATION_OF = self.station
        self.static = pd.read_csv(PROC / "static_vectors.csv").set_index("station_id")
        self._cache: dict[int, tuple] = {}

    def seq(self, L: int):
        if L not in self._cache:
            self._cache[L] = ts.prepare(L)
        return self._cache[L]


def predict_l1(data: Data, name: str, tr_idx, ev_idx, hor):
    spec = data.models_l1[name]
    y = data.d[hor].to_numpy()
    ytr = y[tr_idx].astype(int)
    if spec["kind"] == "const":
        base = float(ytr.mean())
        return np.full(len(tr_idx), base), np.full(len(ev_idx), base), 0.5
    pipe = spec["pipe"]
    pipe.fit(data.d.iloc[tr_idx][spec["features"]], ytr)
    p_tr = pipe.predict_proba(data.d.iloc[tr_idx][spec["features"]])[:, 1]
    p_ev = pipe.predict_proba(data.d.iloc[ev_idx][spec["features"]])[:, 1]
    return p_tr, p_ev, B.best_f1_threshold(ytr, p_tr)


def predict_l2(data: Data, path: str, tr_idx, ev_idx, hor):
    s = joblib.load(MODELS / path)
    pipe = s["horizons"][hor]
    feats = s["features"]
    p_tr = pipe.predict_proba(data.d.iloc[tr_idx][feats])[:, 1]
    p_ev = pipe.predict_proba(data.d.iloc[ev_idx][feats])[:, 1]
    return p_tr, p_ev, B.best_f1_threshold(data.ix[hor].to_numpy()[tr_idx].astype(int), p_tr)


def predict_seq(data: Data, path: str, tr_idx, ev_idx, hor):
    ck = torch.load(MODELS / path, weights_only=False)
    L, pr = ck["L"], ck["params"]
    X, M = data.seq(L)
    mu, sd = ts.fold_stats(X, M, tr_idx)
    net = ts.SeqNet(ck["kind"], pr["hidden"], pr["layers"], pr["dropout"])
    net.load_state_dict(_remap_legacy(ck["state"][hor])); net.eval()
    with torch.no_grad():
        p_tr = torch.sigmoid(net(torch.tensor(ts.std_rows(X, M, tr_idx, mu, sd)))).numpy()
        p_ev = torch.sigmoid(net(torch.tensor(ts.std_rows(X, M, ev_idx, mu, sd)))).numpy()
    ytr = data.ix[hor].to_numpy()[tr_idx].astype(int)
    return p_tr, p_ev, B.best_f1_threshold(ytr, p_tr)


def predict_transformer(data: Data, path: str, tr_idx, ev_idx, hor):
    ck = torch.load(MODELS / path, weights_only=False)
    L, pr = ck["L"], ck["params"]
    X, M = data.seq(L)
    mu, sd = ts.fold_stats(X, M, tr_idx)
    net = T.TransformerNet(pr["hidden"], pr["heads"], pr["layers"], pr["dropout"])
    net.load_state_dict(ck["state"][hor]); net.eval()
    with torch.no_grad():
        p_tr = torch.sigmoid(net(torch.tensor(ts.std_rows(X, M, tr_idx, mu, sd)))).numpy()
        p_ev = torch.sigmoid(net(torch.tensor(ts.std_rows(X, M, ev_idx, mu, sd)))).numpy()
    ytr = data.ix[hor].to_numpy()[tr_idx].astype(int)
    return p_tr, p_ev, B.best_f1_threshold(ytr, p_tr)


def predict_fusion(data: Data, name: str, path: str, tr_idx, ev_idx, hor):
    ck = torch.load(MODELS / path, weights_only=False)
    L, pr = ck["L"], ck["params"]
    spec = tf.FAMS[name]
    X, M = data.seq(L) if spec["use_seq"] else (None, None)
    enc = tf.fit_encoders(spec, X, M, data.static, data.d, tr_idx)
    tr_t = tf.encode(enc, spec, X, M, data.static, data.d, tr_idx)
    ev_t = tf.encode(enc, spec, X, M, data.static, data.d, ev_idx)
    net = tf.FusionNet(spec["kind"], pr["hidden"], pr["layers"], pr["dropout"],
                       spec["use_seq"], spec["use_static"],
                       tf.LIN_DIM if spec["lin"] else 0)
    net.load_state_dict(ck["state"][hor]); net.eval()

    def fwd(t):
        with torch.no_grad():
            return torch.sigmoid(net(t.get("seq"), t.get("stat"), t.get("lin"))).numpy()

    ytr = data.ix[hor].to_numpy()[tr_idx].astype(int)
    p_tr = fwd(tr_t)
    return p_tr, fwd(ev_t), B.best_f1_threshold(ytr, p_tr)


def predict_gnn(data: Data, path: str, tr_idx, ev_idx, hor):
    ck = torch.load(MODELS / path, weights_only=False)
    L, pr, agg = ck["L"], ck["params"], ck["agg"]
    y = data.ix[hor].to_numpy().astype(float)
    X, M = data.seq(L)
    state = ck["state"][hor]
    a_hat, a_mask = state["a_hat"], state["a_mask"].bool()
    if agg == "gnn_lstm":
        net = g.GNNLSTMNet(pr["hidden"], pr["enc_layers"], pr["g_layers"],
                           pr["dropout"], a_hat, a_mask)
    else:
        net = g.GNNNet(agg, pr["hidden"], pr["enc_layers"], pr["g_layers"],
                       pr["dropout"], a_hat, a_mask)
    net.load_state_dict(state); net.eval()
    bs = int(pr["batch_dates"])
    vtr = g.make_view(np.isin(np.arange(len(data.ix)), tr_idx), y, data.sp, data.dp, X, M)
    vev = g.make_view(np.isin(np.arange(len(data.ix)), ev_idx), y, data.sp, data.dp, X, M)
    p_tr = 1.0 / (1.0 + np.exp(-g.predict_all(net, X, M, data.row_of,
                                              vtr["mu"], vtr["sd"], vtr, bs)))
    p_ev = 1.0 / (1.0 + np.exp(-g.predict_all(net, X, M, data.row_of,
                                              vtr["mu"], vtr["sd"], vev, bs)))
    ytr = data.ix[hor].to_numpy()[tr_idx].astype(int)
    return p_tr, p_ev, B.best_f1_threshold(ytr, p_tr)


def predict_one(data: Data, cat: dict, tr_idx, ev_idx, hor):
    k = cat["kind"]
    if k == "l1":
        return predict_l1(data, cat["model"], tr_idx, ev_idx, hor)
    if k == "l2":
        return predict_l2(data, cat["path"], tr_idx, ev_idx, hor)
    if k == "seq":
        return predict_seq(data, cat["path"], tr_idx, ev_idx, hor)
    if k == "transformer":
        return predict_transformer(data, cat["path"], tr_idx, ev_idx, hor)
    if k == "fusion":
        return predict_fusion(data, cat["model"], cat["path"], tr_idx, ev_idx, hor)
    if k == "gnn":
        return predict_gnn(data, cat["path"], tr_idx, ev_idx, hor)
    raise ValueError(k)


# ------------------------------------------------------------------- cost block
def _n_tree_nodes(clf) -> int:
    if hasattr(clf, "tree_"):
        return int(clf.tree_.node_count)
    if hasattr(clf, "estimators_"):
        return int(sum(getattr(e, "tree_").node_count for e in np.ravel(clf.estimators_)
                       if hasattr(e, "tree_")))
    if hasattr(clf, "get_booster"):                     # xgboost sklearn API
        try:
            return int(clf.get_booster().trees_to_dataframe().shape[0])
        except Exception:
            return 0
    if hasattr(clf, "booster_"):                        # lightgbm sklearn API
        try:
            return int(clf.booster_.trees_to_dataframe().shape[0])
        except Exception:
            return 0
    return 0


def params_kind_and_count(data: Data, cat: dict) -> tuple[int, str]:
    k = cat["kind"]
    if k in ("gnn", "seq", "transformer", "fusion"):
        ck = torch.load(MODELS / cat["path"], weights_only=False)
        return int(ck.get("n_params", 0)), "weights"
    if k == "l2":
        s = joblib.load(MODELS / cat["path"])
        clf = s["horizons"][HOR[0]].named_steps["clf"]
        if hasattr(clf, "coef_"):
            return int(np.size(clf.coef_) + np.size(clf.intercept_)), "coefficients"
        return _n_tree_nodes(clf), "tree_nodes"
    if k == "l1":
        spec = data.models_l1[cat["model"]]
        if spec["kind"] == "const":
            return 1, "coefficients"
        # fit the pipeline on the P1 train to count parameters consistently
        y = data.d[HOR[0]].to_numpy()
        tri = np.flatnonzero(data.p1_train & ~np.isnan(y))
        pipe = spec["pipe"]
        pipe.fit(data.d.iloc[tri][spec["features"]], y[tri].astype(int))
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "coef_"):
            return int(np.size(clf.coef_) + np.size(clf.intercept_)), "coefficients"
        return _n_tree_nodes(clf), "tree_nodes"
    return 0, "unknown"


# ------------------------------------------------------------------------ main
def main() -> None:
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    reg = pd.read_csv(REGISTRY).set_index("registry_id")
    data = Data()

    split = data.d.split.to_numpy()
    hold = data.d.station_holdout.to_numpy()
    year = data.d.date.dt.year.to_numpy()

    # P1 primary cell masks (dev only; the frozen test split is never masked in)
    data.p1_train = (split == "train") & ~hold
    p1_eval = (split == "val") & ~hold & (year == 2022)
    print(f"Phase 27 — fair ML vs DL | {len(CATALOG)} models | seed={SEED} | "
          f"{pd.Timestamp.now('UTC')}")
    print(f"P1 train rows={int(data.p1_train.sum())}  val_2022 rows={int(p1_eval.sum())}  "
          f"hardware={HARDWARE}")

    # ---------------------------------------------------- fresh P1 re-score
    # Cached to disk: the (expensive) fresh prediction pass is reused on re-runs so
    # that downstream fixes do not force a second multi-minute model reload.
    cache_resc = REPORTS / "phase27_p1_rescore.csv"
    cache_cost = REPORTS / "phase27_cost.csv"
    resc = cost_df = None
    if cache_resc.exists() and cache_cost.exists():
        _r = pd.read_csv(cache_resc)
        _c = pd.read_csv(cache_cost)
        if len(_r) == 216 and len(_c) == len(CATALOG):
            resc, cost_df = _r, _c
            print(f"[cache] reusing fresh P1 re-score ({len(_r)} rows) and cost table "
                  f"({len(_c)} rows) — delete reports/phase27_p1_rescore.csv to recompute")
    if resc is None:
        rescored: list[dict] = []
        cost: list[dict] = []
        for cat in CATALOG:
            name = cat["model"]
            nprm, pkind = params_kind_and_count(data, cat)
            for hor in HOR:
                y = data.d[hor].to_numpy()
                tri = np.flatnonzero(data.p1_train & ~np.isnan(y))
                evi = np.flatnonzero(p1_eval & ~np.isnan(y))
                t1 = time.time()
                p_tr, p_ev, thr = predict_one(data, cat, tri, evi, hor)
                fit_s = time.time() - t1
                ytr = y[tri].astype(int); yev = y[evi].astype(int)
                st = data.station[evi]; dt = data.date[evi]; sev = data.season[evi]
                infer_ms = 1000.0 * fit_s / max(len(evi), 1)
                cost.append(dict(model=name, level=cat["level"], registry_id=cat["rid"],
                                 n_params=nprm, params_kind=pkind,
                                 p1_train_s=round(fit_s, 2),
                                 infer_ms_per_1k=round(infer_ms, 3),
                                 hardware=HARDWARE, family=reg.loc[cat["rid"], "family"]))
                for seas in SEASONS:
                    m = np.ones(len(evi), bool) if seas == "all" else (sev == seas)
                    yy, pp = yev[m], p_ev[m]
                    if len(yy) == 0 or len(np.unique(yy)) < 2:
                        continue
                    met, op = score(yy, pp, thr, st[m], dt[m], hor,
                                    do_ci=(seas == "all" and len(yy) >= 1000))
                    row = {**met, **op,
                           "model": name, "level": cat["level"], "registry_id": cat["rid"],
                           "horizon": hor, "season": seas, "cell": "val_2022",
                           "protocol": "P1", "n": int(len(yy)),
                           "n_pos": int(yy.sum()), "base_rate": float(yy.mean()),
                           "train_rows": int(len(tri)), "threshold": float(thr),
                           "source": "evaluate_all_rescore"}
                    rescored.append(row)
                print(f"  rescore {name:22s} {hor}  ({fit_s:5.1f}s, "
                      f"pr_auc={B.metrics(yev, p_ev, thr)['pr_auc']:.5f})")
        resc = pd.DataFrame(rescored)
        cost_df = pd.DataFrame(cost).drop_duplicates("model")
        resc.to_csv(cache_resc, index=False)
        cost_df.to_csv(cache_cost, index=False)

    # ---------------------------------------------------- master table (grids)
    grids = []
    for f in GRID_FILES:
        t = pd.read_csv(REPORTS / f)
        t["source_file"] = f
        grids.append(t)
    master = pd.concat(grids, ignore_index=True, sort=False)
    master["registry_id"] = master["model"].map(
        {c["model"]: c["rid"] for c in CATALOG})
    master["level"] = master["model"].map(
        {c["model"]: c["level"] for c in CATALOG})
    master["source"] = "phase_grid"
    # derived ratio columns on the grid table
    for a, b, col in (("fn", "tp", "fnr"), ("fp", "tp", "far"), ("fp", "tn", "fpr")):
        master[col] = master[a] / (master[a] + master[b]).replace(0, np.nan)
    # parity: join the fresh re-score on (model, horizon, season)
    par = resc[["model", "horizon", "season", "pr_auc", "pr_auc_ci_lo", "pr_auc_ci_hi",
                "roc_auc", "brier", "ece", "precision", "recall", "f1",
                "specificity", "fnr", "far", "fpr", "macro_pr_auc",
                "missed_flood_rows", "event_recall", "lead_mean_days"]].rename(
        columns={"pr_auc": "pr_auc_rescore", "roc_auc": "roc_auc_rescore",
                 "brier": "brier_rescore", "ece": "ece_rescore",
                 "precision": "precision_rescore", "recall": "recall_rescore",
                 "f1": "f1_rescore", "specificity": "specificity_rescore",
                 "fnr": "fnr_rescore", "far": "far_rescore", "fpr": "fpr_rescore",
                 "macro_pr_auc": "macro_rescore",
                 "pr_auc_ci_lo": "ci_lo_rescore", "pr_auc_ci_hi": "ci_hi_rescore",
                 "missed_flood_rows": "missed_rows_rescore",
                 "event_recall": "event_recall_rescore",
                 "lead_mean_days": "lead_mean_rescore"})
    master = master.merge(par, on=["model", "horizon", "season"], how="left")
    master["delta_pr_auc"] = master["pr_auc_rescore"] - master["pr_auc"]
    p1 = master[(master.cell == "val_2022") & (master.season == "all")]
    max_delta = float(np.nanmax(np.abs(p1["delta_pr_auc"])))
    print(f"\nparity: max |fresh P1 PR-AUC - grid P1 PR-AUC| = {max_delta:.2e} "
          f"over {len(p1)} model/horizon cells")

    # cost merge (drop the redundant grid n_params first so the cost table wins)
    fit_map = master[(master.cell == "val_2022") & (master.season == "all")] \
        .groupby(["model", "horizon"])["fit_s"].first().groupby("model").sum()
    cost_df["grid_fit_s_3h"] = cost_df["model"].map(fit_map)
    if "n_params" in master.columns:
        master = master.drop(columns=["n_params"])
    master = master.merge(cost_df[["model", "n_params", "params_kind", "p1_train_s",
                                   "infer_ms_per_1k"]], on="model", how="left")

    master.to_csv(REPORTS / "phase27_metrics.csv", index=False)
    cost_df.to_csv(REPORTS / "phase27_cost.csv", index=False)

    op = resc[["model", "level", "horizon", "season", "n", "n_pos", "threshold",
               "tp", "fp", "fn", "tn", "missed_flood_rows", "false_alarm_rate",
               "false_positive_rate", "alarm_count", "event_count", "event_detected",
               "event_missed", "event_recall", "lead_mean_days", "lead_median_days"]]
    op.to_csv(REPORTS / "phase27_operational.csv", index=False)

    # ------------------------------------------------- generalisation blocks
    gen_rows = []
    # dedupe the P1/P2 val_2022 duplicate so each distinct cell appears once
    m_all = master[master.season == "all"].drop_duplicates(
        ["model", "cell", "season", "horizon"])
    # temporal = P1/P2 val cells; spatial = P3 folds
    for _, r in m_all.iterrows():
        cell = r["cell"]
        if cell.startswith("val_"):
            gen_rows.append(dict(model=r["model"], level=r["level"],
                                 registry_id=r["registry_id"], horizon=r["horizon"],
                                 block="temporal_year", bucket=cell,
                                 pr_auc=r["pr_auc"], roc_auc=r["roc_auc"],
                                 n=r["n"], n_pos=int(r["tp"] + r["fn"])))
        elif cell.startswith("spatial_fold_"):
            gen_rows.append(dict(model=r["model"], level=r["level"],
                                 registry_id=r["registry_id"], horizon=r["horizon"],
                                 block="spatial_fold", bucket=cell,
                                 pr_auc=r["pr_auc"], roc_auc=r["roc_auc"],
                                 n=r["n"], n_pos=int(r["tp"] + r["fn"])))
    # monsoon block on the P1 cell
    for _, r in master[(master.cell == "val_2022") &
                       (master.season != "all")].drop_duplicates(
                           ["model", "cell", "season", "horizon"]).iterrows():
        if r["model"] not in CAT_BY_NAME:
            continue
        gen_rows.append(dict(model=r["model"], level=r["level"],
                             registry_id=r["registry_id"], horizon=r["horizon"],
                             block="season", bucket=r["season"],
                             pr_auc=r["pr_auc"], roc_auc=r["roc_auc"],
                             n=r["n"], n_pos=int(r["tp"] + r["fn"])))
    gen = pd.DataFrame(gen_rows)
    gen.to_csv(REPORTS / "phase27_generalisation.csv", index=False)

    # ---------------------------------------------------------- headline summary
    head = resc[(resc.season == "all") & resc.pr_auc.notna()]
    summary = {
        "phase": 27, "seed": SEED, "hardware": HARDWARE,
        "n_models": len(CATALOG), "n_master_rows": int(len(master)),
        "parity_max_abs_delta_pr_auc_P1": max_delta,
        "test_rows_read": 0,
        "headline_P1_val2022_all": {
            h: head[head.horizon == h].sort_values("pr_auc", ascending=False)
            .set_index("model")["pr_auc"].round(5).to_dict()
            for h in HOR},
        "ci_P1_val2022_all": {
            h: {r["model"]: [round(r["pr_auc_ci_lo"], 4), round(r["pr_auc_ci_hi"], 4)]
                for _, r in head[head.horizon == h].iterrows()}
            for h in HOR},
    }
    (REPORTS / "phase27_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharex=False)
        colors = {"L1": "#888888", "L2": "#1f77b4", "L3": "#2ca02c", "L4": "#d62728"}
        for ax, h in zip(axes, HOR):
            sub = head[head.horizon == h].sort_values("pr_auc")
            ypos = np.arange(len(sub))
            ax.barh(ypos, sub.pr_auc, color=[colors[l] for l in sub.level])
            ax.errorbar(sub.pr_auc, ypos,
                        xerr=[sub.pr_auc - sub.pr_auc_ci_lo,
                              sub.pr_auc_ci_hi - sub.pr_auc],
                        fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
            ax.set_yticks(ypos); ax.set_yticklabels(sub.model, fontsize=8)
            ax.set_xlim(0.5, 1.0); ax.set_title(f"{h[6:]} (P1 val_2022)")
            ax.set_xlabel("PR-AUC (95% station bootstrap CI)"); ax.grid(axis="x", alpha=.3)
        fig.suptitle("Phase 27 — fair ML vs DL comparison (one code path)")
        fig.tight_layout()
        fig.savefig(FIGURES / "phase27_ml_vs_dl.png", dpi=130)
        print("wrote figures/phase27_ml_vs_dl.png")
    except Exception as e:  # pragma: no cover
        print("figure skipped:", e)

    print(f"\nwrote reports/phase27_metrics.csv ({len(master)} rows), "
          f"phase27_p1_rescore.csv ({len(resc)}), phase27_operational.csv ({len(op)}), "
          f"phase27_cost.csv ({len(cost_df)}), "
          f"phase27_generalisation.csv ({len(gen)})  [{time.time() - t0:.0f}s]")
    print("\n=== P1 val_2022 headline (fresh re-score, season=all) ===")
    for h in HOR:
        top = head[head.horizon == h].sort_values("pr_auc", ascending=False).head(6)
        print(f"  {h}: " + " · ".join(
            f"{r['model']} {r['pr_auc']:.4f}" for _, r in top.iterrows()))


if __name__ == "__main__":
    main()
