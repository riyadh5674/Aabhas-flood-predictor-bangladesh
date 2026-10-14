r"""Phase 25 (stage 40, D13) — temporal Transformer (registry L4-03).

Gate 25.1 PASSED (reports/phase25_transformer_gate.md): the Transformer earns one
registered attempt under pre-registered acceptance/kill criteria.  This driver fits
the temporal Transformer — per-day token -> linear embedding + positional embedding
-> TransformerEncoder (self-attention) -> masked-mean pool -> sigmoid head — on the
SAME data/splits/metrics/budget class as Phase 21/22 (D14), with the temporal window
L as a tuning dimension ("L = best").

Contract (parity with train_sequence.py)
----------------------------------------
* Registry: reports/phase17_experiment_registry.csv row L4-03 transformer_temporal,
  gate=phase25_transformer_gate, tuning_budget_configs = 20 (max), test_policy =
  frozen_read_at_phase33, selection_metric = pr_auc, selection_cell = val_2022,
  seed = 20261008.
* Input: Dataset B sequences_L{L}.npy (200665, L, 12) + mask, standardised per task
  fold on OBSERVED timesteps only (fill rule P19-M1).  Channels = 12 raw + mask = 13.
* Cells: DEV only — P1 val_2022, P2 val_2020/2021/2022, P3 spatial_fold_1..4,
  each x {all, monsoon, non-monsoon} x {24h,48h,72h}.  Frozen test never read.
* Tuning: seeded random search (budget 16 <= registry 20), window L in {3,7,14,30}
  (the workflow's {7,14,30} + the L* = 3 parity point), scored ONLY on val_2022
  (season=all), score = mean PR-AUC over 3 horizons.  Early stopping on val PR-AUC
  (patience 4, cap 14); deterministic seed; pos_weight only when weighting=on.
* Outputs:
    reports/phase25_search.csv    one row per evaluated config
    reports/phase25_metrics.csv   dev-cell grid (Phase-20 schema + n_params)
    models/phase25_L4-03.pt       P1 primary model {horizon: state_dict} + meta
    scripts/phase25_run.log       stdout tee
* Resume: a finished family (pt + both CSVs) is skipped.

Run:
    python -u scripts/train_transformer.py
"""
from __future__ import annotations

import io
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"D:\Flood predictor")
sys.path.insert(0, str(ROOT / "scripts"))

import train_sequence as ts                                    # noqa: E402
from baseline_model import HORIZONS, SEED, best_f1_threshold, bootstrap_pr_auc, \
    macro_pr_auc, metrics                                      # noqa: E402

ROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REGISTRY = REPORTS / "phase17_experiment_registry.csv"
RUN_LOG = ROOT / "scripts" / "phase25_run.log"

FAMILY = "transformer_bestL"
REG_ID = "L4-03"
MODEL_NAME = "transformer_temporal"
SEARCH_CSV = REPORTS / "phase25_search.csv"
MET_CSV = REPORTS / "phase25_metrics.csv"
PT = MODELS / "phase25_L4-03.pt"

CHANNELS = 13
MAX_EPOCHS = 14
PATIENCE = 4
MAX_LEN = 30
BUDGET = 16                                                    # <= registry 20
SPACE = {"L": [3, 7, 14, 30], "lr": [3e-4, 1e-3, 3e-3], "hidden": [32, 64],
         "heads": [2, 4], "layers": [1, 2], "dropout": [0.0, 0.3],
         "batch_size": [256, 512], "_weight": ["off", "on"]}
SPACE_SIZE = len([1 for _ in product(*SPACE.values())])


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


class TransformerNet(nn.Module):
    """Per-day token -> embed + position -> TransformerEncoder -> masked pool."""

    def __init__(self, hidden: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.embed = nn.Linear(CHANNELS, hidden)
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
            dropout=dropout, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                    # x: (N, L, 13), last channel = mask
        mask = x[:, :, -1].clamp(0.0, 1.0)   # (N, L) observed = 1
        pad = mask < 0.5                     # True = ignore (key padding)
        h = self.embed(x) + self.pos[:, :x.shape[1]]
        h = self.enc(h, src_key_padding_mask=pad)
        m = mask.unsqueeze(-1)
        h = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(h).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def fit_transformer(cfg: dict, X, M, mu, sd, tr_idx, ytr, ev_idx, yev,
                    fixed_epochs: int | None = None) -> dict:
    ts.seed_all(SEED)
    t0 = time.time()
    Xtr = torch.tensor(ts.std_rows(X, M, tr_idx, mu, sd))
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xev = torch.tensor(ts.std_rows(X, M, ev_idx, mu, sd))

    ratio = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    pos_w = ratio if cfg["_weight"] == "on" else 1.0
    net = TransformerNet(cfg["hidden"], cfg["heads"], cfg["layers"], cfg["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
    bs = int(cfg["batch_size"])
    n = Xtr.shape[0]

    best_ap, best_ep, best_state = -1.0, 0, None
    epochs = 0
    gen = torch.Generator().manual_seed(SEED)
    cap = MAX_EPOCHS if fixed_epochs is None else fixed_epochs
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
        if ep - best_ep >= PATIENCE:
            break
    if fixed_epochs is None:
        net.load_state_dict(best_state)
    with torch.no_grad():
        p = torch.sigmoid(net(Xev))
    return {"net": net, "pred": p.numpy(), "best_ap": best_ap, "best_ep": best_ep,
            "epochs": epochs, "fit_s": time.time() - t0, "pos_weight_used": pos_w}


def sample_configs(n: int) -> list[dict]:
    cells = [dict(zip(SPACE, vals)) for vals in product(*SPACE.values())]
    rng = np.random.default_rng(SEED + 7)
    pick = np.sort(rng.choice(len(cells), size=n, replace=False))
    return [{**cells[i], "_cell": int(i)} for i in pick]


def main() -> None:
    t0 = time.time()
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    log = open(RUN_LOG, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log)
    _print = print
    _print(f"Phase 25 — temporal Transformer ({REG_ID}) | seed={SEED} | "
           f"{pd.Timestamp.now('UTC')}")

    reg = pd.read_csv(REGISTRY)
    row = reg[(reg.level == "L4") & (reg.registry_id == REG_ID)]
    assert len(row) == 1, "registry L4-03 missing"
    r = row.iloc[0]
    assert r.model == MODEL_NAME and r.gate == "phase25_transformer_gate", (r.model, r.gate)
    assert int(r.tuning_budget_configs) >= BUDGET, "budget exceeds registry"
    assert r.test_policy == "frozen_read_at_phase33", r.test_policy
    assert r.selection_cell == "val_2022" and r.selection_metric == "pr_auc"
    _print(f"registry OK: {REG_ID} {r.model}, gate={r.gate}, budget {BUDGET} <= "
           f"{int(r.tuning_budget_configs)}, test={r.test_policy}")

    if PT.exists() and MET_CSV.exists() and SEARCH_CSV.exists():
        s = pd.read_csv(SEARCH_CSV)
        m = pd.read_csv(MET_CSV)
        if FAMILY in set(s.family) and FAMILY in set(m.model):
            _print(f"skip {FAMILY}: already complete ({PT.name})")
            sys.stdout.flush()
            return

    ix = pd.read_csv(ROC / "sequence_index.csv", parse_dates=["date"])
    assert len(ix) == 200665
    lbl = pd.read_csv(ROC / "experimental_dataset.csv",
                      usecols=["station_id", "date", "label_24h", "label_48h",
                               "label_72h"], parse_dates=["date"])
    mm = ix[["station_id", "date"]].merge(lbl, on=["station_id", "date"], how="left")
    for h in HORIZONS:
        assert np.nansum(np.abs(ix[h].to_numpy() - mm[h].to_numpy())) == 0, \
            f"index labels != experimental table {h}"
    _print(f"index labels re-checked 1:1 vs experimental_dataset.csv "
           f"({len(ix):,} rows, {ix.station_id.nunique()} stations)")

    tasks = ts.build_tasks_index(ix)
    dev = {c for t in tasks for _, c, _ in t["evals"]}
    assert dev == {"val_2022", "val_2020", "val_2021", "spatial_fold_1",
                   "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"}
    for t in tasks:
        assert not (t["train"] & ix.station_holdout.astype(bool).to_numpy()).any()
        assert not (t["train"] & (ix.split.to_numpy() == "test")).any()
    _print(f"dev grid OK: {sorted(dev)} — frozen test never scored")

    labels = {h: ix[h].to_numpy() for h in HORIZONS}
    stations = ix.station_id.to_numpy()
    season_v = ix.season.to_numpy()

    p1 = next(t for t in tasks if t["protocol"] == "P1_temporal_primary")
    tri = p1["train"]
    ev_mask = next(m2 for _, c, m2 in p1["evals"] if c == "val_2022")

    Xc: dict[int, tuple] = {}

    def data(L):
        if L not in Xc:
            Xc[L] = ts.prepare(L)
        return Xc[L]

    cfgs = sample_configs(BUDGET)
    _print(f"\n=== {FAMILY} ({REG_ID}) — tuning {len(cfgs)}/{SPACE_SIZE} configs, "
           f"cell=val_2022, score=mean PR-AUC over 3 horizons, windows "
           f"{SPACE['L']} ===")
    rows_search: list[dict] = []
    best_cfg, best_score, best_cfg_id, best_epochs, n_param = None, -1.0, None, None, None
    for ci, cfg in enumerate(cfgs):
        X, M = data(cfg["L"])
        aucs, epochs_h, fit_s_list = {}, {}, 0.0
        for hor in HORIZONS:
            yv = labels[hor]
            tr_idx = np.flatnonzero(tri & ~np.isnan(yv))
            ev_idx = np.flatnonzero(ev_mask & ~np.isnan(yv))
            mu, sd = ts.fold_stats(X, M, tr_idx)
            res = fit_transformer(cfg, X, M, mu, sd, tr_idx, yv[tr_idx],
                                  ev_idx, yv[ev_idx])
            aucs[hor] = res["best_ap"] if res["best_ap"] > 0 else \
                float(metrics(yv[ev_idx].astype(int), res["pred"], 0.5)["pr_auc"])
            epochs_h[hor] = res["best_ep"]
            fit_s_list += res["fit_s"]
            n_param = res["net"].n_params()
        mean_auc = float(np.mean(list(aucs.values())))
        cfg_id = f"{FAMILY}_c{ci:02d}"
        rows_search.append({
            "family": FAMILY, "registry_id": REG_ID, "config_id": cfg_id,
            "space_cell": cfg["_cell"],
            "params": json.dumps({k: v for k, v in cfg.items()
                                  if not k.startswith("_")}),
            "weighting": "pos_weight=ratio" if cfg["_weight"] == "on" else "none",
            "pr_auc_24h": aucs[HORIZONS[0]], "pr_auc_48h": aucs[HORIZONS[1]],
            "pr_auc_72h": aucs[HORIZONS[2]], "mean_pr_auc": mean_auc,
            "best_epochs": json.dumps({h.split("_")[-1]: v for h, v in epochs_h.items()}),
            "fit_s": round(fit_s_list, 1), "n_configs_budget": BUDGET,
            "space_size": SPACE_SIZE, "selection_cell": "val_2022",
            "feature_set": "raw_sequence(12 vars)|L=" + str(cfg["L"]),
            "n_params": n_param})
        if mean_auc > best_score:
            best_score, best_cfg, best_cfg_id = mean_auc, cfg, cfg_id
            best_epochs = {h.split("_")[-1]: v for h, v in epochs_h.items()}
        _print(f"  [{ci + 1}/{len(cfgs)}] {cfg_id} L={cfg['L']} mean={mean_auc:.5f} "
               f"(24h={aucs[HORIZONS[0]]:.5f} 48h={aucs[HORIZONS[1]]:.5f} "
               f"72h={aucs[HORIZONS[2]]:.5f}) {cfg['_weight']} ({fit_s_list:.0f}s)")

    winner = next(x for x in rows_search if x["config_id"] == best_cfg_id)
    winL = int(best_cfg["L"])
    _print(f"  -> winner {best_cfg_id} L={winL} mean_pr_auc={best_score:.5f} "
           f"epochs={best_epochs} n_params={winner['n_params']:,}")

    X, M = data(winL)
    _print(f"=== {FAMILY} — grid with {best_cfg_id} (L={winL}) "
           f"(8 dev tasks x 3 seasons x 3 horizons) ===")
    rows_met: list[dict] = []
    saved = {"family": FAMILY, "registry_id": REG_ID,
             "feature_set": "raw_sequence(12 vars)|L=" + str(winL),
             "kind": "transformer", "L": winL, "config_id": best_cfg_id,
             "params": {k: v for k, v in best_cfg.items() if not k.startswith("_")},
             "weighting": winner["weighting"], "seed": SEED, "train": "train_<2022",
             "n_params": n_param, "epochs": best_epochs, "headline_metrics": {},
             "state": {}}
    for task in tasks:
        for hor in HORIZONS:
            yv = labels[hor]
            tr_idx = np.flatnonzero(task["train"] & ~np.isnan(yv))
            if len(np.unique(yv[tr_idx])) < 2:
                continue
            mu, sd = ts.fold_stats(X, M, tr_idx)
            ev0, cell0, ev_mask0 = task["evals"][0]
            ev_idx = np.flatnonzero(ev_mask0 & ~np.isnan(yv))
            fixed = int(best_epochs[hor.split("_")[-1]])
            res = fit_transformer(best_cfg, X, M, mu, sd, tr_idx, yv[tr_idx],
                                  ev_idx, yv[ev_idx], fixed_epochs=fixed)
            fit_s = res["fit_s"]
            net = res["net"]
            with torch.no_grad():
                Xtr_t = torch.tensor(ts.std_rows(X, M, tr_idx, mu, sd))
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
                    Xev = torch.tensor(ts.std_rows(X, M, idx, mu, sd))
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
                        season=seas or "all", model=FAMILY, horizon=hor,
                        n=len(idx), train_rows=int(tr_idx.size),
                        train_base_rate=float(yv[tr_idx].mean()),
                        pr_auc_ci_lo=lo, pr_auc_ci_hi=hi,
                        macro_pr_auc=macro_pr_auc(stations[idx], y, p),
                        fit_s=round(fit_s, 1), registry_id=REG_ID,
                        feature_set=saved["feature_set"],
                        weighting=winner["weighting"], config_id=best_cfg_id,
                        search_score=round(best_score, 5), n_params=n_param)
                    rows_met.append(met_row)
            _print(f"  {task['protocol']:18s} {task['name']:13s} {hor} "
                   f"epochs {res['epochs']} fit {fit_s:.0f}s")

    torch.save(saved, PT)
    disk_s = pd.read_csv(SEARCH_CSV) if SEARCH_CSV.exists() else pd.DataFrame()
    disk_m = pd.read_csv(MET_CSV) if MET_CSV.exists() else pd.DataFrame()
    if not disk_s.empty:
        disk_s = disk_s[disk_s.family != FAMILY]
    if not disk_m.empty:
        disk_m = disk_m[disk_m.model != FAMILY]
    disk_s = pd.concat([disk_s, pd.DataFrame(rows_search)], ignore_index=True)
    disk_m = pd.concat([disk_m, pd.DataFrame(rows_met)], ignore_index=True)
    disk_s.to_csv(SEARCH_CSV, index=False)
    disk_m.to_csv(MET_CSV, index=False)
    _print(f"  checkpoint: {SEARCH_CSV.name} ({len(disk_s)} rows), "
           f"{MET_CSV.name} ({len(disk_m)} rows), {PT.name}")
    _print(f"\nTOTAL {time.time() - t0:.0f}s — exit=0")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
