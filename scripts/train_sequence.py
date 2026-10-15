r"""Phase 21/22 — Deep temporal learning: LSTM & GRU (stage 36) then TCN (stage 37).

NOTE: the deliverable file is named train_sequence.py (as the workflow body
mandates, with a lstm|gru|tcn model flag). Phase 21 runs the recurrent families
(lstm_L3/L7/L14/L30, gru_bestL); Phase 22 runs tcn_bestL at L* = 3 (registry
L3-06). Both share this driver, the same data/splits/metrics/budget class (D14);
only the temporal encoder differs (recurrence vs dilated causal convolution).

Contract
--------
* Registry: reports/phase17_experiment_registry.csv rows L3-01..L3-06
  (lstm_L3/L7/L14/L30, gru_bestL, tcn_bestL), tuning_budget_configs = 20 maximum,
  test_policy = frozen_read_at_phase33, selection_metric = pr_auc,
  selection_cell = val_2022, seed = 20261008.
* Output routing: recurrent families -> reports/phase21_*.csv +
  models/phase21_L3-0X.pt + scripts/phase21_run.log; tcn_bestL ->
  reports/phase22_*.csv + models/phase22_L3-06.pt + scripts/phase22_run.log.
* Input: Dataset B — data/processed/sequences_L{L}.npy (200665, L, 12) raw
  float32 + sequence_mask_L{L}.npy (200665, L) uint8 (policy P19-M1 keep-all).
  Phase 21 input = concat(12 raw vars, mask) = 13 channels; PER-TASK train-fold
  standardisation (mean/std over the fold's OBSERVED timesteps only, computed
  inside the training fold — no info from outside), every residual NaN -> 0
  (the P19-M1 fill rule). Train/val/cell masks are rebuilt on the
  sequence_index rows from its own split/spatial_fold/station_holdout/season
  columns, mirroring scripts/baseline_model.build_tasks (P1/P2/P3 D14 parity).
* Cells: DEV only — P1 val_2022, P2 val_2020/2021/2022, P3 spatial_fold_1..4,
  each x {all, monsoon, non-monsoon} x {24h,48h,72h}. Test/frozen period is
  never read (asserted).
* Tuning: seeded random search per family (space 144 cells), scored ONLY on
  selection cell val_2022 (season=all), score = mean PR-AUC over 3 horizons.
  Budgets fixed from the Phase-21 fit-time probe on this machine (8 threads;
  cost grows ~linearly with L):
      lstm_L3 12 configs   lstm_L7 10   lstm_L14 8   lstm_L30 6   gru_bestL 10
      all <= registry max 20.
* Training contract: BCEWithLogitsLoss, pos_weight = train ratio when the
  config draws weighting=on (unweighted otherwise); early stopping on
  validation PR-AUC (patience 4, epoch cap 14); batch size recorded; every fit
  seeded identically (python/numpy/torch) -> determinism check in the notebook.
  Capacity: one recurrent layer first, second layer only as a tuning option.
* Grid: the winner config is re-trained per task x horizon with the winner's
  BEST-EPOCH count (fixed, no val during P2/P3 fits) and scored on the task's
  cells x 3 seasons. Output rows mirror phase20_metrics.csv exactly
  (same protocol/cell/season/model/horizon + metrics() keys) so the deep
  models join the Phase-20 tables for the head-to-head (Phase 27).
* Model selection context: L* = the L whose winning LSTM has the highest
  val_2022 mean PR-AUC; gru_bestL is tuned at that L (registry L3-05).
* Outputs:
    reports/phase21_search.csv        one row per evaluated config
    reports/phase21_metrics.csv       dev-cell grid (Phase-20 schema + n_params)
    models/phase21_L3-0X.pt           P1 primary models, {horizon: state_dict} + meta
    scripts/phase21_run.log           stdout (python-side UTF-8 tee)
* Resume: finished families (pt + both CSVs) are skipped; partial rows of an
  unfinished family are dropped. Pass families as argv to run a subset.

Run (split so each call stays under the shell timeout):
    python -u scripts/train_sequence.py lstm_L3 lstm_L7 lstm_L14 lstm_L30
    python -u scripts/train_sequence.py gru_bestL
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):        # scripts have it; Jupyter's OutStream does not
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"D:\Flood predictor")
sys.path.insert(0, str(ROOT / "scripts"))

from baseline_model import HORIZONS, SEED, best_f1_threshold, bootstrap_pr_auc, \
    macro_pr_auc, metrics

ROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
SEARCH_CSV = REPORTS / "phase21_search.csv"
MET_CSV = REPORTS / "phase21_metrics.csv"
P22_SEARCH_CSV = REPORTS / "phase22_search.csv"
P22_MET_CSV = REPORTS / "phase22_metrics.csv"
REGISTRY = REPORTS / "phase17_experiment_registry.csv"
RUN_LOG = ROOT / "scripts" / "phase21_run.log"
RUN_LOG_22 = ROOT / "scripts" / "phase22_run.log"

LENS = (3, 7, 14, 30)
N_VARS = 12
CHANNELS = N_VARS + 1
SPACE = {"lr": [3e-4, 1e-3, 3e-3], "hidden": [32, 64], "layers": [1, 2],
         "dropout": [0.0, 0.3], "batch_size": [256, 512], "_weight": ["off", "on"]}
SPACE_SIZE = len([1 for _ in product(*SPACE.values())])          # 96
BUDGETS = {"lstm_L3": 12, "lstm_L7": 10, "lstm_L14": 8, "lstm_L30": 6,
           "gru_bestL": 10, "tcn_bestL": 12}
REG_IDS = {"lstm_L3": "L3-01", "lstm_L7": "L3-02", "lstm_L14": "L3-03",
           "lstm_L30": "L3-04", "gru_bestL": "L3-05", "tcn_bestL": "L3-06"}
FAM_L = {"lstm_L3": 3, "lstm_L7": 7, "lstm_L14": 14, "lstm_L30": 30,
         "gru_bestL": None, "tcn_bestL": None}
FAM_KIND = {"lstm_L3": "lstm", "lstm_L7": "lstm", "lstm_L14": "lstm",
            "lstm_L30": "lstm", "gru_bestL": "gru", "tcn_bestL": "tcn"}
TN_KERNEL = 3                                                 # TCN conv width
MAX_EPOCHS = 14
PATIENCE = 4
SEASONS = ["monsoon", "non-monsoon"]


def out_paths(family: str) -> tuple[Path, Path, Path]:
    """(search_csv, metrics_csv, model_pt) for a family (Phase 21 vs Phase 22)."""
    if family.startswith("tcn"):
        return (P22_SEARCH_CSV, P22_MET_CSV,
                MODELS / f"phase22_{REG_IDS[family]}.pt")
    return (SEARCH_CSV, MET_CSV, MODELS / f"phase21_{REG_IDS[family]}.pt")


class _Tee:
    """Duplicate print() output to scripts/phase21_run.log (UTF-8)."""

    def __init__(self, stream: io.TextIOWrapper, log: io.TextIOWrapper):
        self.stream, self.log = stream, log

    def write(self, s: str) -> int:
        self.stream.write(s)
        self.log.write(s)
        return len(s)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()


# --------------------------------------------------------------- data prep
def prepare(L: int):
    """Load + standardise cache enablers. Returns callables over the index."""

    X = np.load(ROC / f"sequences_L{L}.npy", mmap_mode="r")
    M = np.load(ROC / f"sequence_mask_L{L}.npy", mmap_mode="r").astype(np.float32)
    return X, M


def fold_stats(X: np.ndarray, M: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """"Fold-standardise: mean/std over the training fold's OBSERVED timesteps
    only (mask=1), per variable. No info from outside the fold."""
    sub = X[idx]
    m = M[idx] == 1                                    # (n, L)
    vals = np.where(m[:, :, None], sub, np.nan)
    mu = np.nanmean(vals, axis=(0, 1))
    sd = np.sqrt(np.nanmean((vals - mu) ** 2, axis=(0, 1)))
    sd = np.where(sd < 1e-9, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def std_rows(X: np.ndarray, M: np.ndarray, idx: np.ndarray, mu: np.ndarray,
             sd: np.ndarray) -> np.ndarray:
    """(n, L, 13) float32: z-scores then residual NaN -> 0 (P19-M1 fill), +mask."""
    z = (X[idx] - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return np.concatenate([z.astype(np.float32), M[idx][:, :, None]], axis=-1)


# ---------------------------------------------------------------- model
class _CausalConv1d(nn.Module):
    """Conv1d with left-only padding so step t never sees t+1..t+k."""

    def __init__(self, c_in, c_out, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class _TCNBlock(nn.Module):
    """Dilated causal residual block: conv->BN->ReLU->Drop twice + skip."""

    def __init__(self, c_in, c_out, k, dilation, dropout):
        super().__init__()
        self.conv1 = _CausalConv1d(c_in, c_out, k, dilation)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = _CausalConv1d(c_out, c_out, k, dilation)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.drop(out)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class _TCNEncoder(nn.Module):
    """Stack of dilated causal blocks -> last-timestep representation."""

    def __init__(self, hidden, levels, kernel, dropout):
        super().__init__()
        blocks = []
        for i in range(levels):
            blocks.append(_TCNBlock(CHANNELS if i == 0 else hidden, hidden,
                                    kernel, 2 ** i, dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):                       # x: (N, L, C)
        y = self.net(x.transpose(1, 2))         # (N, hidden, L)
        return y[:, :, -1]                       # last (anchor) timestep


class SeqNet(nn.Module):
    """Temporal encoder (lstm|gru|tcn) -> dense head -> sigmoid logit."""

    def __init__(self, kind: str, hidden: int, layers: int, dropout: float,
                 kernel: int = TN_KERNEL):
        super().__init__()
        assert kind in ("lstm", "gru", "tcn")
        self.kind = kind
        if kind in ("lstm", "gru"):
            rnn = nn.LSTM if kind == "lstm" else nn.GRU
            drop = dropout if layers > 1 else 0.0
            self.enc = rnn(CHANNELS, hidden, num_layers=layers, dropout=drop,
                           batch_first=True)
        else:
            self.enc = _TCNEncoder(hidden, layers, kernel, dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        if self.kind in ("lstm", "gru"):
            out, _ = self.enc(x)
            h = out[:, -1, :]
        else:
            h = self.enc(x)
        return self.head(h).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------------------------------------------------------------- fit helpers
def fit_model(kind: str, L: int, cfg: dict, X: np.ndarray, M: np.ndarray,
              mu: np.ndarray, sd: np.ndarray, tr_idx: np.ndarray, ytr: np.ndarray,
              ev_idx: np.ndarray, yev: np.ndarray,
              max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
              fixed_epochs: int | None = None) -> dict:
    """Train with early stopping on validation PR-AUC (or fixed epoch count).
    Returns model, best val pr_auc, best epoch, epochs, fit time."""
    seed_all(SEED)                                        # determinism contract
    t0 = time.time()
    Xtr = torch.tensor(std_rows(X, M, tr_idx, mu, sd))
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xev = torch.tensor(std_rows(X, M, ev_idx, mu, sd))
    yev_t = torch.tensor(yev, dtype=torch.float32)

    ratio = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    pos_w = ratio if cfg["_weight"] == "on" else 1.0
    net = SeqNet(kind, cfg["hidden"], cfg["layers"], cfg["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
    bs = int(cfg["batch_size"])
    n = Xtr.shape[0]

    best_ap, best_ep, best_state = -1.0, 0, None
    epochs = 0
    gen = torch.Generator().manual_seed(SEED)          # deterministic shuffle
    cap = max_epochs if fixed_epochs is None else fixed_epochs
    for ep in range(1, cap + 1):
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xtr[b]), ytr_t[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(Xev))
        from sklearn.metrics import average_precision_score
        ap = float(average_precision_score(yev, p.numpy()))
        epochs = ep
        if fixed_epochs is not None:
            continue
        if ap > best_ap + 1e-9:
            best_ap, best_ep = ap, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if ep - best_ep >= patience:
            break
    if fixed_epochs is None:
        net.load_state_dict(best_state)
    with torch.no_grad():
        p = torch.sigmoid(net(Xev))
    fit_s = time.time() - t0
    return {"net": net, "pred": p.numpy(), "best_ap": best_ap, "best_ep": best_ep,
            "epochs": epochs, "fit_s": fit_s, "pos_weight_used": pos_w}


# ------------------------------------------------------------------ tasks
def build_tasks_index(ix: pd.DataFrame) -> list[dict]:
    """"Mirror of scripts/baseline_model.build_tasks built on sequence_index
    rows (its own columns). Train masks are holdout- and test-clean."""
    hold = ix.station_holdout.astype(bool).to_numpy()
    year = ix.date.dt.year.to_numpy()
    split = ix.split.to_numpy()
    fold = ix.spatial_fold.to_numpy()

    tasks = []
    tr = (split == "train") & ~hold
    tasks.append({"protocol": "P1_temporal_primary", "name": "train_<2022",
                  "train": tr, "evals": [("P1", "val_2022",
                                          (split == "val") & ~hold)]})
    for Y in (2020, 2021, 2022):
        tasks.append({"protocol": "P2_rolling_origin", "name": f"train_<={Y - 1}",
                      "train": (year <= Y - 1) & ~hold,
                      "evals": [("P2", f"val_{Y}", (year == Y) & ~hold)]})
    for k in (1, 2, 3, 4):
        tasks.append({"protocol": "P3_spatial_cv", "name": f"excl_fold_{k}",
                      "train": (split == "train") & ~hold & (fold != k),
                      "evals": [("P3", f"spatial_fold_{k}",
                                 (split == "train") & (fold == k))]})
    return tasks


# -------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    fams = sys.argv[1:] or list(BUDGETS)
    bad = [f for f in fams if f not in BUDGETS]
    assert not bad, f"unknown families {bad}"
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    phase = 22 if all(f.startswith("tcn") for f in fams) else 21
    log = open(RUN_LOG_22 if phase == 22 else RUN_LOG, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log)
    _print = print
    _print(f"Phase {phase} — families={fams} | seed={SEED} | "
           f"{pd.Timestamp.now('UTC')}")

    reg = pd.read_csv(REGISTRY)
    l3 = reg[reg.level == "L3"].set_index("model")
    assert set(FAM_KIND) <= set(l3.index), "registry L3 rows != phase-21 families"
    for f in fams:
        row = l3.loc[f]
        assert row.test_policy == "frozen_read_at_phase33", f
        assert row.selection_cell == "val_2022" and row.selection_metric == "pr_auc"
        assert int(row.tuning_budget_configs) >= BUDGETS[f], "budget exceeds registry"
    _print("registry OK: test_policy=frozen_read_at_phase33, budgets <= 20")

    ix = pd.read_csv(ROC / "sequence_index.csv", parse_dates=["date"])
    assert len(ix) == 200665
    lbl = pd.read_csv(ROC / "experimental_dataset.csv",
                      usecols=["station_id", "date", "label_24h", "label_48h",
                               "label_72h"], parse_dates=["date"])
    m = ix[["station_id", "date"]].merge(lbl, on=["station_id", "date"],
                                         how="left")

    for h in HORIZONS:
        a = ix[h].to_numpy()
        b = m[h].to_numpy()
        assert np.nansum(np.abs(a - b)) == 0, f"index labels != experimental table {h}"
    _print(f"index labels re-checked 1:1 vs experimental_dataset.csv "
           f"({len(ix):,} rows, {ix.station_id.nunique()} stations)")
    assert set(ix.season.dropna().unique()) == set(SEASONS), "season values"
    tasks = build_tasks_index(ix)
    dev = {c for t in tasks for _, c, _ in t["evals"]}
    assert dev == {"val_2022", "val_2020", "val_2021",
                   "spatial_fold_1", "spatial_fold_2", "spatial_fold_3",
                   "spatial_fold_4"}, "dev grid != contract"
    for t in tasks:
        assert not (t["train"] & ix.station_holdout.astype(bool).to_numpy()).any(), \
            "holdout in a training mask"
        assert not (t["train"] & (ix.split.to_numpy() == "test")).any(), \
            "test row in a training mask"
    _print(f"dev grid OK: {sorted(dev)} — frozen test never scored")

    labels = {h: ix[h].to_numpy() for h in HORIZONS}
    stations = ix.station_id.to_numpy()
    season_v = ix.season.to_numpy()

    search = pd.read_csv(SEARCH_CSV) if SEARCH_CSV.exists() else pd.DataFrame()
    met = pd.read_csv(MET_CSV) if MET_CSV.exists() else pd.DataFrame()

    # L* resolution: highest val_2022 mean PR-AUC among the 4 LSTM families.
    LSTMS = ["lstm_L3", "lstm_L7", "lstm_L14", "lstm_L30"]
    lstm_results: dict[int, dict] = {}
    if not search.empty:
        for f in LSTMS:
            sub = search[search.family == f]
            if not sub.empty:
                lstm_results[FAM_L[f]] = {"best_mean": float(sub["mean_pr_auc"].max())}
    for f in LSTMS:
        if f in fams:
            fd = _run_family(f, FAM_L[f], "lstm", ix, tasks, labels, stations,
                             season_v, _print)
            if fd:
                lstm_results[FAM_L[f]] = fd
    lstar = None
    if lstm_results:
        lstar = max(lstm_results, key=lambda L: lstm_results[L]["best_mean"])
        _print(f"\nL* = {lstar} ({lstm_results[lstar]['best_mean']:.5f} mean "
               f"PR-AUC on val_2022 over the 4 LSTMs)")
        txt = ROOT / "data" / "processed" / "phase21_lstar.json"
        txt.write_text(json.dumps({"L_star": lstar}, indent=2), encoding="utf-8")

    def _resolve_lstar() -> int:
        if lstar is not None:
            return int(lstar)
        try:
            return int(json.loads((ROC / "phase21_lstar.json").read_text())["L_star"])
        except Exception:
            _print("ERROR: bestL family needs Phase-21 lstm results first")
            sys.exit(2)

    if "gru_bestL" in fams:                       # Phase 21 (stage 36)
        _run_family("gru_bestL", _resolve_lstar(), "gru", ix, tasks, labels,
                    stations, season_v, _print)
    if "tcn_bestL" in fams:                       # Phase 22 (stage 37)
        _run_family("tcn_bestL", _resolve_lstar(), "tcn", ix, tasks, labels,
                    stations, season_v, _print)

    _print(f"\nTOTAL {time.time() - t0:.0f}s — exit=0")
    sys.stdout.flush()


def _sample_configs(n: int) -> list[dict]:
    cells = [dict(zip(SPACE, vals)) for vals in product(*SPACE.values())]
    rng = np.random.default_rng(SEED + 7)
    pick = np.sort(rng.choice(len(cells), size=n, replace=False))
    return [{**cells[i], "_cell": int(i)} for i in pick]


def _run_family(family: str, L: int, kind: str, ix, tasks, labels, stations,
                season_v, _print) -> dict:
    fam_t0 = time.time()
    reg_id = REG_IDS[family]
    search_csv, met_csv, pkl = out_paths(family)
    search = pd.read_csv(search_csv) if search_csv.exists() else pd.DataFrame()
    met = pd.read_csv(met_csv) if met_csv.exists() else pd.DataFrame()
    done = (not met.empty and family in set(met["model"]) and pkl.exists()
            and not search.empty and family in set(search["family"]))
    if done:
        _print(f"  skip {family}: already complete ({pkl.name})")
        return {}
    if not search.empty:
        search = search[search.family != family]
    if not met.empty:
        met = met[met.model != family]

    X, M = prepare(L)
    p1 = next(t for t in tasks if t["protocol"] == "P1_temporal_primary")
    tri = p1["train"]
    ev_mask = next(m2 for _, c, m2 in p1["evals"] if c == "val_2022")

    cfgs = _sample_configs(BUDGETS[family])
    _print(f"\n=== {family} ({reg_id}) L={L} {kind} — tuning "
           f"{len(cfgs)}/{SPACE_SIZE} configs, cell=val_2022, score=mean PR-AUC "
           f"over 3 horizons ===")
    rows_search: list[dict] = []
    best_cfg, best_score, best_cfg_id = None, -1.0, None
    best_epochs = None
    n_param = None

    for ci, cfg in enumerate(cfgs):
        aucs, aps, fit_s_list = {}, {}, 0.0
        for hor in HORIZONS:
            yv = labels[hor]
            tr_idx = np.flatnonzero(tri & ~np.isnan(yv))
            ev_idx = np.flatnonzero(ev_mask & ~np.isnan(yv))
            mu, sd = fold_stats(X, M, tr_idx)
            r = fit_model(kind, L, cfg, X, M, mu, sd, tr_idx, yv[tr_idx],
                          ev_idx, yv[ev_idx])
            aucs[hor] = r["best_ap"] if r["best_ap"] > 0 else \
                float(metrics(yv[ev_idx].astype(int), r["pred"], 0.5)["pr_auc"])
            aps[f"best_ep_{hor}"] = r["best_ep"]
            fit_s_list += r["fit_s"]
            n_param = r["net"].n_params()
        mean_auc = float(np.mean(list(aucs.values())))
        cfg_id = f"{family}_c{ci:02d}"
        rows_search.append({
            "family": family, "registry_id": reg_id, "config_id": cfg_id,
            "space_cell": cfg["_cell"],
            "params": json.dumps({k: v for k, v in cfg.items()
                                  if not k.startswith("_")}),
            "weighting": "pos_weight=ratio" if cfg["_weight"] == "on"
                         else "none",
            "pr_auc_24h": aucs[HORIZONS[0]], "pr_auc_48h": aucs[HORIZONS[1]],
            "pr_auc_72h": aucs[HORIZONS[2]], "mean_pr_auc": mean_auc,
            "best_epochs": json.dumps({h.split("_")[-1]: v
                                       for h, v in aps.items()}),
            "fit_s": round(fit_s_list, 1), "n_configs_budget": BUDGETS[family],
            "space_size": SPACE_SIZE, "selection_cell": "val_2022",
            "feature_set": "raw_sequence(12 vars)|L=" + str(L),
            "n_params": n_param})
        if mean_auc > best_score:
            best_score, best_cfg, best_cfg_id = mean_auc, cfg, cfg_id
            best_epochs = {h.split("_")[-1]: v for h, v in aps.items()}
        _print(f"  [{ci + 1}/{len(cfgs)}] {cfg_id} mean={mean_auc:.5f} "
               f"(24h={aucs[HORIZONS[0]]:.5f} 48h={aucs[HORIZONS[1]]:.5f} "
               f"72h={aucs[HORIZONS[2]]:.5f}) {cfg['_weight']} "
               f"({fit_s_list:.0f}s)")

    winner = next(r for r in rows_search if r["config_id"] == best_cfg_id)
    _print(f"  -> winner {best_cfg_id} mean_pr_auc={best_score:.5f} "
           f"epochs={best_epochs} n_params={winner['n_params']:,}")

    # -------- grid: winner config x 8 tasks x 3 horizons, fixed epochs
    _print(f"=== {family} — grid with {best_cfg_id} "
           f"(8 dev tasks x 3 seasons x 3 horizons) ===")
    rows_met: list[dict] = []
    saved: dict = {"family": family, "registry_id": reg_id, "feature_set":
                   "raw_sequence(12 vars)|L=" + str(L), "kind": kind, "L": L,
                   "config_id": best_cfg_id,
                   "params": {k: v for k, v in best_cfg.items()
                              if not k.startswith("_")},
                   "weighting": winner["weighting"],
                   "seed": SEED, "train": "train_<2022", "n_params": n_param,
                   "epochs": best_epochs, "headline_metrics": {}, "state": {}}
    for task in tasks:
        for hor in HORIZONS:
            yv = labels[hor]
            tr_idx = np.flatnonzero(task["train"] & ~np.isnan(yv))
            if len(np.unique(yv[tr_idx])) < 2:
                continue
            mu, sd = fold_stats(X, M, tr_idx)
            ev0, cell0, ev_mask0 = task["evals"][0]
            ev_idx = np.flatnonzero(ev_mask0 & ~np.isnan(yv))
            fixed = int(best_epochs[hor.split("_")[-1]])
            r = fit_model(kind, L, best_cfg, X, M, mu, sd, tr_idx, yv[tr_idx],
                          ev_idx, yv[ev_idx], fixed_epochs=fixed)
            fit_s = r["fit_s"]
            with torch.no_grad():
                net = r["net"]
                Xtr_t = torch.tensor(std_rows(X, M, tr_idx, mu, sd))
                p_tr = torch.sigmoid(net(Xtr_t)).numpy()
            thr = best_f1_threshold(yv[tr_idx].astype(int), p_tr)
            if task["protocol"] == "P1_temporal_primary":
                saved["state"][hor] = {k: v.cpu() for k, v in net.state_dict().items()}
                saved["fit_rows_" + hor] = int(tr_idx.size)

            for protocol, cell, ev_mask2 in task["evals"]:
                for seas in (None, "monsoon", "non-monsoon"):
                    msk = ev_mask2.copy()
                    if seas is not None:
                        msk = ev_mask2 & (season_v == seas)
                    idx = np.flatnonzero(msk & ~np.isnan(yv))
                    if len(idx) == 0:
                        continue
                    Xev = torch.tensor(std_rows(X, M, idx, mu, sd))
                    with torch.no_grad():
                        p = torch.sigmoid(net(Xev)).numpy()
                    y = yv[idx].astype(int)
                    met_row = metrics(y, p, thr)
                    lo, hi = np.nan, np.nan
                    if seas is None and len(idx) >= 1000 and \
                            len(np.unique(stations[idx])) >= 10:
                        lo, hi = bootstrap_pr_auc(stations[idx], y, p)
                    if task["protocol"] == "P1_temporal_primary" and seas is None:
                        from sklearn.metrics import average_precision_score
                        saved["headline_metrics"][hor] = \
                            float(average_precision_score(y, p))
                    met_row.update(
                        protocol=protocol, train=task["name"], cell=cell,
                        season=seas or "all", model=family, horizon=hor,
                        n=len(idx), train_rows=int(tr_idx.size),
                        train_base_rate=float(yv[tr_idx].mean()),
                        pr_auc_ci_lo=lo, pr_auc_ci_hi=hi,
                        macro_pr_auc=macro_pr_auc(stations[idx], y, p),
                        fit_s=round(fit_s, 1), registry_id=reg_id,
                        feature_set=saved["feature_set"],
                        weighting=winner["weighting"], config_id=best_cfg_id,
                        search_score=round(best_score, 5), n_params=n_param)
                    rows_met.append(met_row)
            _print(f"  {task['protocol']:18s} {task['name']:13s} {hor} "
                   f"epochs {r['epochs']} fit {fit_s:.0f}s")

    torch.save(saved, pkl)
    # merge-on-write: never clobber other families' already-persisted rows
    disk_s = pd.read_csv(search_csv) if search_csv.exists() else pd.DataFrame()
    disk_m = pd.read_csv(met_csv) if met_csv.exists() else pd.DataFrame()
    if not disk_s.empty:
        disk_s = disk_s[disk_s.family != family]
    if not disk_m.empty:
        disk_m = disk_m[disk_m.model != family]
    disk_s = pd.concat([disk_s, pd.DataFrame(rows_search)], ignore_index=True)
    disk_m = pd.concat([disk_m, pd.DataFrame(rows_met)], ignore_index=True)
    disk_s.to_csv(search_csv, index=False)
    disk_m.to_csv(met_csv, index=False)
    _print(f"  checkpoint: {search_csv.name} ({len(disk_s)} rows), "
           f"{met_csv.name} ({len(disk_m)} rows), {pkl.name}, "
           f"family time {time.time() - fam_t0:.0f}s")
    return {"best_mean": best_score, "best_cfg": best_cfg,
            "best_cfg_id": best_cfg_id, "n_params": n_param,
            "best_epochs": best_epochs}


if __name__ == "__main__":
    main()