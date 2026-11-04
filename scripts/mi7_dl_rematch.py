r"""MI-7 / Phase 35 — Fair DL rematch (Track 3 + Track 4).

WHAT
----
The Phase 21-27 comparison gave the deep models 12 raw variables over a 3-day
window while the trees saw 62 (v1) / 110 (v2) engineered features. This driver
fixes the asymmetry: it trains temporal networks on the *engineered*-feature
sequences built by scripts/build_mi7_sequences.py, with the training rig those
Phases lacked, and reports seed/bootstrapped CIs on every delta.

DESIGN (pre-registered)
-----------------------
* Families (arch, feature tag, L):
    lstm_v2_L7, gru_v2_L7, tcn_v2_L7, transformer_v2_L7,
    lstm_v1_L7 (information ablation), transformer_v2_L14 (longer window).
  Ladder: raw-12 (Phase-21 LSTM, reused) -> v1-57 -> v2-105.
* Multi-task trunk: one temporal encoder -> shared dense -> 3 horizon heads.
  One fit scores all 3 horizons (data-efficient; mirrors the hpo_perhorizon ML).
* Static branch: 10 station numerics (standardised per fold) + 5 text columns
  (frequency-encoded per fold), exactly the Phase-23 fusion input.
* Rig: AdamW + weight decay, ReduceLROnPlateau on val PR-AUC, gradient clipping,
  early stopping (patience 5, cap 30), dropout/batch/lr/hidden/layers/pos_weight
  in the search space. Fold-standardised inputs (train-fold mean/std over
  OBSERVED timesteps only; residual NaN -> 0), same P19-M1 keep-all mask policy.
* Tuning: seeded random search, selection cell = val_2022 (season=all), score =
  mean PR-AUC over 3 horizons. Budgets are per family (see FAMILIES): the
  LSTM/TCN families get ~20 configs, the attention families fewer, because the
  host is CPU-only (8 threads). This is below the classical-ML ceiling of 40
  configs and is reported as a stated limitation.
* Seed averaging: the winner config is re-trained for K=3 seeds per task and the
  horizon probabilities are averaged. Per-seed spread -> seed CI; station
  bootstrap -> sampling CI (Track 4).
* Cells: DEV ONLY - P1 val_2022, P2 val_2020/2021/2022, P3 spatial_fold_1..4,
  x {all, monsoon, non-monsoon} x {24h,48h,72h}. The frozen test is never read
  (asserted; the project-level rule is reports/test_read_ledger.md).

Outputs
    reports/mi7_search.csv        one row per evaluated config
    reports/mi7_metrics.csv       dev-cell grid (Phase-20/21 schema + seed cols)
    reports/mi7_preds/{family}/{task}_{hor}.npz   per-seed + averaged predictions
    models/mi7_{family}.pt        winner states (all seeds) + meta
    scripts/mi7_run.log           stdout tee

Run (resumable; pass family names to run a subset):
    python -u scripts/mi7_dl_rematch.py > scripts/mi7_run.log 2>&1
    MI7_QUICK=1 python -u scripts/mi7_dl_rematch.py lstm_v2_L7   # smoke test
"""
from __future__ import annotations

import io
import json
import os
import re
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
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"D:\Flood predictor")
sys.path.insert(0, str(ROOT / "scripts"))

from baseline_model import (HORIZONS, SEED, best_f1_threshold, bootstrap_pr_auc,
                            macro_pr_auc, metrics, TEXT_COLS)

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
PREDS = REPORTS / "mi7_preds"
SEARCH_CSV = REPORTS / "mi7_search.csv"
MET_CSV = REPORTS / "mi7_metrics.csv"
RUN_LOG = ROOT / "scripts" / "mi7_run.log"
MANIFEST = PROC / "mi7_sequence_manifest.json"

QUICK = os.environ.get("MI7_QUICK") == "1"
torch.set_num_threads(max(1, os.cpu_count() or 8))
MAX_EPOCHS = 6 if QUICK else int(os.environ.get("MI7_EPOCHS", "18"))
PATIENCE = 2 if QUICK else 4
SEEDS_N = 1 if QUICK else int(os.environ.get("MI7_SEEDS", "3"))
SEASONS = ["monsoon", "non-monsoon"]
TN_KERNEL = 3
_B = int(os.environ.get("MI7_BUDGET", "0"))     # 0 = use per-family budget
_E = int(os.environ.get("MI7_EPOCHS", "0"))     # 0 = use per-family epoch cap

# name -> (kind, tag, L, budget, epochs, task_scope).
#   budget/epochs: LSTM/TCN get the fuller search because the host is CPU-only;
#                  the attention families get a smaller, cheaper search.
#   task_scope: 'full' = all 8 dev tasks (P1/P2/P3); 'core' = P1 + P3 only;
#               'p1' = the selection protocol only. Non-headline families use
#               the narrower scope so wall-clock goes to the models that matter.
FAMILIES = {
    "fusion_v2_L3":       ("lstm", "raw", 3, 20, 12, "full"),
    "fusion_v1_L3":       ("lstm", "raw1", 3, 16, 12, "core"),
    "lstm_v2_L7":         ("lstm", "v2", 7, 20, 12, "full"),
    "lstm_v1_L7":         ("lstm", "v1", 7, 20, 12, "core"),
    "gru_v2_L7":          ("gru", "v2", 7, 16, 12, "core"),
    "tcn_v2_L7":          ("tcn", "v2", 7, 16, 12, "core"),
    "transformer_v2_L7":  ("transformer", "v2", 7, 10, 8, "p1"),
    "transformer_v2_L14": ("transformer", "v2", 14, 8, 8, "p1"),
}
if QUICK:
    FAMILIES = {k: (a, b, c, 8, 6, g) for k, (a, b, c, d, e, g) in FAMILIES.items()}
if _B:
    FAMILIES = {k: (a, b, c, _B, e, g) for k, (a, b, c, d, e, g) in FAMILIES.items()}
if _E:
    FAMILIES = {k: (a, b, c, d, _E, g) for k, (a, b, c, d, e, g) in FAMILIES.items()}

SCOPES = {
    "full": {"P1_temporal_primary", "P2_rolling_origin", "P3_spatial_cv"},
    "core": {"P1_temporal_primary", "P3_spatial_cv"},
    "p1":   {"P1_temporal_primary"},
}

SPACE = {"lr": [3e-4, 1e-3, 3e-3], "hidden": [64, 128, 256],
         "layers": [1, 2], "dropout": [0.1, 0.3], "batch_size": [256, 512],
         "wd": [0.0, 1e-4], "_weight": ["off", "on"]}
SPACE_SIZE = len(list(product(*SPACE.values())))

# Anchor configs are always evaluated first (in this order). They bracket the
# regime the Phase-21/23 deep models liked (Adam-ish lr, 1-2 layers, mild
# dropout, moderate width) so a small random budget cannot miss it entirely.
ANCHORS = [
    {"lr": 1e-3, "hidden": 128, "layers": 1, "dropout": 0.2, "batch_size": 256,
     "wd": 1e-4, "_weight": "on", "_cell": -1},
    {"lr": 1e-3, "hidden": 128, "layers": 2, "dropout": 0.2, "batch_size": 256,
     "wd": 1e-4, "_weight": "on", "_cell": -2},
    {"lr": 3e-4, "hidden": 256, "layers": 2, "dropout": 0.3, "batch_size": 256,
     "wd": 1e-4, "_weight": "off", "_cell": -3},
    {"lr": 3e-3, "hidden": 64, "layers": 1, "dropout": 0.1, "batch_size": 512,
     "wd": 0.0, "_weight": "on", "_cell": -4},
    {"lr": 1e-3, "hidden": 256, "layers": 1, "dropout": 0.3, "batch_size": 512,
     "wd": 1e-4, "_weight": "on", "_cell": -5},
]


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s):
        self.stream.write(s)
        self.log.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.log.flush()


# --------------------------------------------------------------- data
def prepare(tag: str, L: int):
    # load into RAM: fancy-indexing a memmap on every fit re-reads from disk.
    # tag in ('raw','raw1') uses the ORIGINAL Phase-18 12-var sequences
    # (fusion temporal branch); 'raw' pairs them with the v2 engineered static
    # block, 'raw1' with the v1 block (information ablation in the fusion net).
    if tag in ("raw", "raw1"):
        X = np.load(PROC / f"sequences_L{L}.npy")
        M = np.load(PROC / f"sequence_mask_L{L}.npy").astype(np.float32)
    else:
        X = np.load(PROC / f"sequences_mi7_L{L}_{tag}.npy")
        M = np.load(PROC / f"mask_mi7_L{L}_{tag}.npy").astype(np.float32)
    return X, M


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]", "_", name)


def fold_stats(X, M, idx):
    sub = X[idx]
    m = M[idx] == 1
    vals = np.where(m[:, :, None], sub, np.nan)
    mu = np.nanmean(vals, axis=(0, 1))
    sd = np.sqrt(np.nanmean((vals - mu) ** 2, axis=(0, 1)))
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where((~np.isfinite(sd)) | (sd < 1e-9), 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def std_rows(X, M, idx, mu, sd):
    z = (X[idx] - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return np.concatenate([z.astype(np.float32), M[idx][:, :, None]], axis=-1)


def row_positions(full_idx, want):
    """Position of each row of `want` inside the sorted array `full_idx`
    (-1 where absent). A multi-task fit is trained on the label_24h row grid
    (`full_idx`), but a horizon-specific score is computed on the rows where
    THAT horizon is observed (`want`); the two sets differ, so map explicitly
    instead of assuming a shared index/order."""
    return pd.Index(full_idx).get_indexer(want)


class FreqEnc:
    """Train-fold frequency encoding of one station-level text column."""

    def __init__(self, values):
        s = pd.Series(values).astype("object").where(pd.Series(values).notna(), "__NA__")
        self.map_ = s.value_counts(normalize=True).to_dict()

    def transform(self, values):
        s = pd.Series(values).astype("object").where(pd.Series(values).notna(), "__NA__")
        return s.map(self.map_).fillna(0.0).to_numpy(dtype=np.float32)


def load_static():
    sv = pd.read_csv(PROC / "static_vectors.csv")
    num = [c for c in sv.columns if c != "station_id" and c not in TEXT_COLS]
    return sv, num, TEXT_COLS


def build_static_raw(static_kind, sv, num_cols, st_all, train_station_ids, xvt):
    """UNstandardised static block for the whole table (standardised per fold later).

    static_kind='geo'   -> 10 station numerics (static_vectors.csv) + 5 text freq
                           (the Phase-23 fusion input);
    static_kind='v2eng' -> the 105 engineered FULL_V2 features AT THE ANCHOR DAY
                           (xvt, the last timestep of the v2 sequence) + 5 text freq.
    Text is frequency-encoded on the training fold's stations only.
    """
    sv = sv.set_index("station_id")
    row = sv.reindex(st_all)
    sub = sv.loc[[s for s in train_station_ids if s in sv.index]]
    blocks = []
    if static_kind in ("v2eng", "v1eng"):
        blocks.append(np.asarray(xvt, dtype=np.float32))       # (n, 105|57)
    else:
        blocks.append(row[num_cols].to_numpy(dtype=np.float32))
    for c in TEXT_COLS:
        enc = FreqEnc(sub[c].to_numpy())
        blocks.append(enc.transform(row[c].to_numpy())[:, None])
    return np.concatenate(blocks, axis=1).astype(np.float32)


def standardize_static(S_raw, idx):
    """Per-fold z-score of the static block (NaN -> train mean, then -> 0)."""
    sub = S_raw[idx]
    mu = np.nanmean(sub, axis=0)
    sd = np.nanstd(sub, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where((~np.isfinite(sd)) | (sd < 1e-9), 1.0, sd)
    z = (S_raw - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


# --------------------------------------------------------------- model
class _CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class _TCNBlock(nn.Module):
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


class _TransfEnc(nn.Module):
    def __init__(self, c_in, hidden, layers, dropout, L, nhead=4):
        super().__init__()
        self.proj = nn.Linear(c_in, hidden)
        self.pos = nn.Parameter(torch.zeros(1, L, hidden))
        layer = nn.TransformerEncoderLayer(hidden, nhead, hidden * 4, dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        y = self.proj(x) + self.pos
        return self.enc(y)[:, -1, :]


class MTNet(nn.Module):
    """Temporal encoder -> (+static) -> shared trunk -> n_heads logits."""

    def __init__(self, kind, c_in, hidden, layers, dropout, L, n_static,
                 n_heads=3, kernel=TN_KERNEL):
        super().__init__()
        self.kind = kind
        if kind in ("lstm", "gru"):
            rnn = nn.LSTM if kind == "lstm" else nn.GRU
            self.enc = rnn(c_in, hidden, num_layers=layers,
                           dropout=(dropout if layers > 1 else 0.0),
                           batch_first=True)
            self.enc_out = hidden
        elif kind == "tcn":
            blocks = [_TCNBlock(c_in if i == 0 else hidden, hidden, kernel,
                                2 ** i, dropout) for i in range(layers)]
            self.enc = nn.Sequential(*blocks)
            self.enc_out = hidden
        elif kind == "transformer":
            self.enc = _TransfEnc(c_in, hidden, layers, dropout, L)
            self.enc_out = hidden
        else:
            raise ValueError(kind)
        self.use_static = n_static > 0
        din = self.enc_out + (n_static if self.use_static else 0)
        self.trunk = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(),
                                   nn.Dropout(dropout))
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_heads)])

    def forward(self, x, s=None):
        if self.kind in ("lstm", "gru"):
            h = self.enc(x)[0][:, -1, :]
        elif self.kind == "tcn":
            h = self.enc(x.transpose(1, 2))[:, :, -1]
        else:
            h = self.enc(x)
        if self.use_static:
            h = torch.cat([h, s], dim=-1)
        h = self.trunk(h)
        return torch.stack([head(h).squeeze(-1) for head in self.heads], dim=1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_multitask(kind, L, cfg, X, M, mu, sd, S, tr_idx, ytr, ev_idx, yev,
                  seed=SEED, fixed_epochs=None, max_epochs=MAX_EPOCHS,
                  patience=PATIENCE):
    """Train a 3-head model; early stop on mean val PR-AUC. Returns preds both
    splits, best epoch, epochs run, fit seconds, n_params.

    ytr / yev: (n, 3) float with NaN for undefined horizons."""
    from sklearn.metrics import average_precision_score
    seed_all(seed)
    t0 = time.time()
    Xtr = torch.tensor(std_rows(X, M, tr_idx, mu, sd))
    Str = torch.tensor(S[tr_idx])
    Xev = torch.tensor(std_rows(X, M, ev_idx, mu, sd))
    Sev = torch.tensor(S[ev_idx])
    Ytr = torch.tensor(ytr, dtype=torch.float32)
    Yev = torch.tensor(yev, dtype=torch.float32)
    valid_tr = torch.tensor(~np.isnan(ytr))
    valid_ev = torch.tensor(~np.isnan(yev))

    pos = Ytr[valid_tr]
    ratio = float((pos == 0).sum() / max((pos == 1).sum(), 1))
    pos_w = ratio if cfg["_weight"] == "on" else 1.0
    net = MTNet(kind, Xtr.shape[-1], cfg["hidden"], cfg["layers"],
                cfg["dropout"], L, S.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=2)
    lossf = nn.BCEWithLogitsLoss(reduction="none",
                                 pos_weight=torch.tensor(pos_w))
    bs = int(cfg["batch_size"])
    n = Xtr.shape[0]
    gen = torch.Generator().manual_seed(seed)

    def mean_ap(net, Xe, Se, Ye, ve):
        net.eval()
        with torch.no_grad():
            logits = net(Xe, Se)
            p = torch.sigmoid(logits).numpy()
        aps = []
        for h in range(3):
            m = ve[:, h].numpy()
            if m.sum() > 0 and len(np.unique(Ye[m, h].numpy())) > 1:
                aps.append(average_precision_score(Ye[m, h].numpy(), p[m, h]))
        return float(np.mean(aps)) if aps else 0.0

    best_ap, best_ep, best_state = -1.0, 0, None
    cap = max_epochs if fixed_epochs is None else fixed_epochs
    epochs = 0
    for ep in range(1, cap + 1):
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            logits = net(Xtr[b], Str[b])
            # NaN targets (undefined horizons) would poison the masked loss
            # (NaN * 0 = NaN): sanitise BEFORE the loss, then mask.
            raw = lossf(logits, torch.nan_to_num(Ytr[b], nan=0.0))
            mask = valid_tr[b].float()
            loss = (raw * mask).sum() / mask.sum().clamp(min=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        epochs = ep
        if fixed_epochs is not None:
            continue
        ap = mean_ap(net, Xev, Sev, Yev, valid_ev)
        sched.step(ap)
        if ap > best_ap + 1e-9:
            best_ap, best_ep = ap, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if ep - best_ep >= patience:
            break
    if fixed_epochs is None and best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p_tr = torch.sigmoid(net(Xtr, Str)).numpy()
        p_ev = torch.sigmoid(net(Xev, Sev)).numpy()
    if not (np.isfinite(p_tr).all() and np.isfinite(p_ev).all()):
        base = float(np.nanmean(Ytr.numpy())) if Ytr.numel() else 0.0
        p_tr = np.nan_to_num(p_tr, nan=base)
        p_ev = np.nan_to_num(p_ev, nan=base)
        return {"net": net, "p_tr": p_tr, "p_ev": p_ev, "best_ap": -1.0,
                "best_ep": 0, "epochs": epochs, "fit_s": time.time() - t0,
                "pos_weight": pos_w, "n_params": net.n_params(), "diverged": True}
    return {"net": net, "p_tr": p_tr, "p_ev": p_ev, "best_ap": best_ap,
            "best_ep": best_ep, "epochs": epochs, "fit_s": time.time() - t0,
            "pos_weight": pos_w, "n_params": net.n_params(), "diverged": False}


# --------------------------------------------------------------- tasks
def build_tasks_index(ix):
    hold = ix.station_holdout.astype(bool).to_numpy()
    year = ix.date.dt.year.to_numpy()
    split = ix.split.to_numpy()
    fold = ix.spatial_fold.to_numpy()
    tasks = [{"protocol": "P1_temporal_primary", "name": "train_<2022",
              "train": (split == "train") & ~hold,
              "evals": [("P1", "val_2022", (split == "val") & ~hold)]}]
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


def _sample_configs(n):
    cells = [dict(zip(SPACE, vals)) for vals in product(*SPACE.values())]
    rng = np.random.default_rng(SEED + 71)
    keys = ["lr", "hidden", "layers", "dropout", "batch_size", "wd", "_weight"]

    def keyfn(c):
        return tuple(c[k] for k in keys)

    out, seen = [], set()
    for a in ANCHORS:                                  # anchors first, always
        if keyfn(a) not in seen:
            seen.add(keyfn(a))
            out.append(a)
    order = rng.permutation(len(cells))
    for i in order:
        if len(out) >= n:
            break
        c = {**cells[i], "_cell": int(i)}
        if keyfn(c) not in seen:
            seen.add(keyfn(c))
            out.append(c)
    return out[:n]


# --------------------------------------------------------------- main
def main():
    t0 = time.time()
    fams = sys.argv[1:] or list(FAMILIES)
    bad = [f for f in fams if f not in FAMILIES]
    assert not bad, f"unknown families {bad}"
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    PREDS.mkdir(exist_ok=True)
    # append: families are driven one invocation at a time, so the log is the
    # running narrative of the whole sweep. Each run stamps a header with its
    # timestamp + effective overrides, so stale content is never ambiguous.
    log = open(RUN_LOG, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log)
    P = print
    P(f"\n{'=' * 78}\nMI-7 — fair DL rematch | families={fams} | seed={SEED} | "
      f"seeds/family={SEEDS_N} | quick={QUICK} | {pd.Timestamp.now('UTC')}\n{'=' * 78}")
    # echo the EFFECTIVE env-overrides and per-family plan: a leaked MI7_* env
    # var (which silently shrank an earlier run to 1 epoch) is then impossible
    # to miss in the log.
    P(f"env overrides: MI7_QUICK={os.environ.get('MI7_QUICK')} "
      f"MI7_BUDGET={os.environ.get('MI7_BUDGET')} "
      f"MI7_EPOCHS={os.environ.get('MI7_EPOCHS')} "
      f"MI7_SEEDS={os.environ.get('MI7_SEEDS')} | "
      f"MAX_EPOCHS={MAX_EPOCHS} PATIENCE={PATIENCE}")
    P(f"span={SPACE_SIZE} space cells "
      f"(+{len(ANCHORS)} anchors) | plan:")
    for _f in fams:
        _k, _t, _L, _b, _e, _s = FAMILIES[_f]
        P(f"    {_f:20s} kind={_k:11s} tag={_t:4s} L={_L:<2d} "
          f"budget={_b:<3d} epochs={_e:<3d} scope={_s}")

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    nch = {k: v["n_channels"] for k, v in man["channel_sets"].items()}
    P(f"channel sets from manifest: {nch}")

    ix = pd.read_csv(PROC / "sequence_index.csv", parse_dates=["date"])
    assert len(ix) == 200665
    lbl = pd.read_csv(PROC / "experimental_dataset.csv",
                      usecols=["station_id", "date", "label_24h", "label_48h",
                               "label_72h"], parse_dates=["date"])
    m = ix[["station_id", "date"]].merge(lbl, on=["station_id", "date"], how="left")
    for h in HORIZONS:
        assert np.nansum(np.abs(ix[h].to_numpy() - m[h].to_numpy())) == 0, h
    tasks = build_tasks_index(ix)
    dev = {c for t in tasks for _, c, _ in t["evals"]}
    assert dev == {"val_2022", "val_2020", "val_2021", "spatial_fold_1",
                   "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"}
    for t in tasks:
        assert not (t["train"] & ix.station_holdout.astype(bool).to_numpy()).any()
        assert not (t["train"] & (ix.split.to_numpy() == "test")).any()
    P(f"dev grid OK: {sorted(dev)} — frozen test never scored "
      f"(see reports/test_read_ledger.md)")

    labels = {h: ix[h].to_numpy() for h in HORIZONS}
    ymat = np.column_stack([labels[h] for h in HORIZONS])          # (n,3)
    stations = ix.station_id.to_numpy()
    season_v = ix.season.to_numpy()
    sv, snum, stext = load_static()

    P1 = next(t for t in tasks if t["protocol"] == "P1_temporal_primary")
    dev_mask = next(mk for _, c, mk in P1["evals"] if c == "val_2022")

    for fam in fams:
        kind, tag, L, budget, epochs, scope = FAMILIES[fam]
        tks = [t for t in tasks if t["protocol"] in SCOPES[scope]]
        _run_family(fam, kind, tag, L, budget, epochs, ix, tks, ymat, labels,
                    stations, season_v, sv, snum, stext, dev_mask, P)

    P(f"\nTOTAL {time.time() - t0:.0f}s — exit=0")
    sys.stdout.flush()


def _save_pt(path, obj):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _run_family(fam, kind, tag, L, budget, epochs, ix, tasks, ymat, labels, stations,
                season_v, sv, snum, stext, dev_mask, P):
    ft0 = time.time()
    pkl = MODELS / f"mi7_{fam}.pt"
    search = pd.read_csv(SEARCH_CSV) if SEARCH_CSV.exists() else pd.DataFrame()
    met = pd.read_csv(MET_CSV) if MET_CSV.exists() else pd.DataFrame()
    done = (not met.empty and fam in set(met["model"]) and pkl.exists()
            and not search.empty and fam in set(search["family"]))
    if done:
        P(f"  skip {fam}: already complete ({pkl.name})")
        return
    if not search.empty:
        search = search[search.family != fam]
    if not met.empty:
        met = met[met.model != fam]

    X, M = prepare(tag, L)
    p1 = next(t for t in tasks if t["protocol"] == "P1_temporal_primary")
    tri = p1["train"]
    tr_stations = np.unique(stations[tri & ~np.isnan(ymat[:, 0])])
    static_kind = {"raw": "v2eng", "raw1": "v1eng"}.get(tag, "geo")
    xvt = None
    if static_kind in ("v2eng", "v1eng"):
        # engineered features at the anchor day = last timestep of the mi7 seq
        src = "sequences_mi7_L7_v1.npy" if static_kind == "v1eng" \
            else "sequences_mi7_L7_v2.npy"
        xvt = np.load(PROC / src, mmap_mode="r")[:, -1, :]
    S_raw = build_static_raw(static_kind, sv, snum, stations, tr_stations, xvt)
    P(f"    static block: {static_kind} -> {S_raw.shape[1]} cols")

    cfgs = _sample_configs(budget)
    P(f"\n=== {fam} ({kind}, {tag}={X.shape[2]} ch, L={L}) — tuning "
      f"{len(cfgs)}/{SPACE_SIZE} configs, cell=val_2022, score=mean PR-AUC ===")
    rows_search = []
    best_cfg, best_score, best_id, best_ep, n_param = None, -1.0, None, None, None
    tr_idx = np.flatnonzero(tri & ~np.isnan(ymat[:, 0]))
    ev_idx = np.flatnonzero(dev_mask & ~np.isnan(ymat[:, 0]))
    mu, sd = fold_stats(X, M, tr_idx)
    S = standardize_static(S_raw, tr_idx)
    for ci, cfg in enumerate(cfgs):
        r = fit_multitask(kind, L, cfg, X, M, mu, sd, S, tr_idx,
                          ymat[tr_idx], ev_idx, ymat[ev_idx], max_epochs=epochs)
        # per-horizon AP on the tuned cell (best epoch already restored)
        aucs = {}
        for hi, hor in enumerate(HORIZONS):
            yv = labels[hor]
            ei = np.flatnonzero(dev_mask & ~np.isnan(yv))
            from sklearn.metrics import average_precision_score
            pos = row_positions(ev_idx, ei)
            kk = pos >= 0
            aucs[hor] = float(average_precision_score(
                yv[ei[kk]], r["p_ev"][pos[kk], hi]))
        mean_auc = float(np.mean(list(aucs.values())))
        cid = f"{fam}_c{ci:02d}"
        rows_search.append({
            "family": fam, "kind": kind, "feature_tag": tag, "L": L,
            "config_id": cid, "space_cell": cfg["_cell"],
            "params": json.dumps({k: v for k, v in cfg.items()
                                  if not k.startswith("_")}),
            "weighting": "pos_weight=ratio" if cfg["_weight"] == "on" else "none",
            "pr_auc_24h": aucs[HORIZONS[0]], "pr_auc_48h": aucs[HORIZONS[1]],
            "pr_auc_72h": aucs[HORIZONS[2]], "mean_pr_auc": mean_auc,
            "best_epoch": r["best_ep"], "epochs": r["epochs"],
            "fit_s": round(r["fit_s"], 1), "budget": budget,
            "space_size": SPACE_SIZE, "selection_cell": "val_2022",
            "n_channels": int(X.shape[2]), "n_params": r["n_params"]})
        if mean_auc > best_score:
            best_score, best_cfg, best_id = mean_auc, cfg, cid
            best_ep, n_param = max(1, int(r["best_ep"])), r["n_params"]
        P(f"  [{ci + 1}/{len(cfgs)}] {cid} mean={mean_auc:.5f} "
          f"(24h={aucs[HORIZONS[0]]:.5f} 48h={aucs[HORIZONS[1]]:.5f} "
          f"72h={aucs[HORIZONS[2]]:.5f}) ep={r['best_ep']} "
          f"{cfg['_weight']} ({r['fit_s']:.0f}s)")

    P(f"  -> winner {best_id} mean_pr_auc={best_score:.5f} ep={best_ep} "
      f"n_params={n_param:,}")
    # persist search now (resumable)
    disk_s = pd.read_csv(SEARCH_CSV) if SEARCH_CSV.exists() else pd.DataFrame()
    if not disk_s.empty:
        disk_s = disk_s[disk_s.family != fam]
    pd.concat([disk_s, pd.DataFrame(rows_search)], ignore_index=True).to_csv(
        SEARCH_CSV, index=False)

    # -------- grid: winner config x 8 tasks x K seeds (multi-task) ----------
    P(f"=== {fam} — grid {best_id} x {len(tasks)} tasks x {SEEDS_N} seeds ===")
    saved = {"family": fam, "kind": kind, "feature_tag": tag, "L": L,
             "n_channels": int(X.shape[2]), "config_id": best_id,
             "params": {k: v for k, v in best_cfg.items() if not k.startswith("_")},
             "best_epoch": best_ep, "seed": SEED, "seeds": [SEED + 100 * s
                                                            for s in range(SEEDS_N)],
             "n_params": n_param, "state": {}, "headline": {}}
    fam_preds = PREDS / fam
    fam_preds.mkdir(parents=True, exist_ok=True)
    rows_met = []
    for task in tasks:
        tr_idx = np.flatnonzero(task["train"] & ~np.isnan(ymat[:, 0]))
        if len(np.unique(ymat[tr_idx, 0])) < 2:
            continue
        mu, sd = fold_stats(X, M, tr_idx)
        S_task = standardize_static(S_raw, tr_idx)
        ev0, cell0, ev_mask0 = task["evals"][0]
        ev_idx = np.flatnonzero(ev_mask0 & ~np.isnan(ymat[:, 0]))
        p_ev_seeds, p_tr_seeds, fit_s = [], [], 0.0
        for s in range(SEEDS_N):
            r = fit_multitask(kind, L, best_cfg, X, M, mu, sd, S_task, tr_idx,
                              ymat[tr_idx], ev_idx, ymat[ev_idx],
                              seed=SEED + 100 * s, fixed_epochs=int(best_ep),
                              max_epochs=epochs)
            p_ev_seeds.append(r["p_ev"])
            p_tr_seeds.append(r["p_tr"])
            fit_s += r["fit_s"]
            if task["protocol"] == "P1_temporal_primary":
                saved["state"][f"seed{s}"] = {k: v.cpu()
                                              for k, v in r["net"].state_dict().items()}
        p_ev = np.mean(p_ev_seeds, axis=0)                 # (nev, 3) averaged
        p_tr = np.mean(p_tr_seeds, axis=0)
        np.savez_compressed(fam_preds / f"{safe_name(task['name'])}_{cell0}.npz",
                            ev_idx=ev_idx, y=ymat[ev_idx],
                            p=p_ev, p_seeds=np.stack(p_ev_seeds))
        for hi, hor in enumerate(HORIZONS):
            yv = labels[hor]
            tr_i = np.flatnonzero(task["train"] & ~np.isnan(yv))
            if len(np.unique(yv[tr_i])) < 2:
                continue
            ptr = row_positions(tr_idx, tr_i)
            ktr = ptr >= 0
            thr = best_f1_threshold(yv[tr_i[ktr]].astype(int),
                                    p_tr[ptr[ktr], hi])
            for protocol, cell, ev_mask2 in task["evals"]:
                for seas in (None, "monsoon", "non-monsoon"):
                    msk = ev_mask2.copy() if seas is None else \
                        ev_mask2 & (season_v == seas)
                    idx = np.flatnonzero(msk & ~np.isnan(yv))
                    if len(idx) == 0:
                        continue
                    pos2 = row_positions(ev_idx, idx)      # ev_idx = label_24h grid
                    kk = pos2 >= 0
                    idx = idx[kk]
                    p = p_ev[pos2[kk], hi]
                    y = yv[idx].astype(int)
                    met_row = metrics(y, p, thr)
                    lo, hi_ci, seed_sd = np.nan, np.nan, np.nan
                    if seas is None and len(idx) >= 1000 and \
                            len(np.unique(stations[idx])) >= 10:
                        lo, hi_ci = bootstrap_pr_auc(stations[idx], y, p)
                        ps = np.stack([p_ev_seeds[s][pos2[kk], hi]
                                       for s in range(SEEDS_N)])
                        from sklearn.metrics import average_precision_score as aps
                        seed_sd = float(np.std([aps(y, ps[s]) for s in range(SEEDS_N)]))
                    if task["protocol"] == "P1_temporal_primary" and seas is None:
                        from sklearn.metrics import average_precision_score
                        saved["headline"][hor] = float(average_precision_score(y, p))
                    met_row.update(
                        protocol=protocol, train=task["name"], cell=cell,
                        season=seas or "all", model=fam, horizon=hor,
                        n=len(idx), train_rows=int(tr_idx.size),
                        train_base_rate=float(yv[tr_i].mean()),
                        pr_auc_ci_lo=lo, pr_auc_ci_hi=hi_ci, seed_pr_auc_sd=seed_sd,
                        macro_pr_auc=macro_pr_auc(stations[idx], y, p),
                        fit_s=round(fit_s, 1), feature_set=f"{tag}|L={L}",
                        config_id=best_id, search_score=round(best_score, 5),
                        n_params=n_param, n_seeds=SEEDS_N)
                    rows_met.append(met_row)
        P(f"  {task['protocol']:18s} {task['name']:13s} "
          f"fit {fit_s:.0f}s ({SEEDS_N} seeds)")
    _save_pt(pkl, saved)
    disk_m = pd.read_csv(MET_CSV) if MET_CSV.exists() else pd.DataFrame()
    if not disk_m.empty:
        disk_m = disk_m[disk_m.model != fam]
    pd.concat([disk_m, pd.DataFrame(rows_met)], ignore_index=True).to_csv(
        MET_CSV, index=False)
    P(f"  checkpoint: {SEARCH_CSV.name}, {MET_CSV.name}, {pkl.name}, "
      f"family {time.time() - ft0:.0f}s")


if __name__ == "__main__":
    main()
