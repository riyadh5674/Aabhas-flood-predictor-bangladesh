r"""Phase 24.2 - Station-graph spatiotemporal GNNs (workflow stage 39, D13 pass).

Graph = the frozen hydrology-first artifact data/processed/station_graph.csv
(build_graph.py): Phase-11C river-upstream edges UNION the minimal nearest-k
haversine fallback that connects all 133 stations.  Message passing runs over
this static graph; distance is a connectivity fallback only.

Models (registry is the contract - L4-01 / L4-02 / L4-04, gate phase24_graph_gate)
-----------------------------------------------------------------------------
* L4-01 gcn      : per-node temporal encoder -> GCN layer(s) over the station graph
                   -> per-node head.  Uniform normalised neighbour weights.
* L4-02 gat      : per-node temporal encoder -> graph-attention layer(s) (learned
                   neighbour weights) -> per-node head.
* L4-04 gnn_lstm : spatial-first full stack - GCN message passing per timestep,
                   then an LSTM over the per-node spatial states -> per-node head.
                   The same engine as L4-01 but the spatial mechanism runs first.

A per-date forward encodes every station present on that date (a node set),
runs the graph layers, then reads the target nodes.  The temporal encoder is an
LSTM over the 13-channel window (12 raw vars + mask, L=L*=3), the exact Phase-21
input contract; only the spatial mechanism differs from the L3 family.  Graph
layers carry a residual connection (out + h) so a node keeps its own signal when
neighbour message passing is uninformative.

Parity / honesty (D14)
----------------------
* Same Dataset B, same Phase-15 split masks (build_tasks_index), same dev cells,
  same metric code, same selection cell (val_2022, season=all), same seed.
* Fold standardisation fit on the training fold's observed timesteps only.
* Only non-test rows are ever gathered as graph nodes; test labels are never
  read (asserted).  Frozen 2023+ period untouched.

Outputs
    reports/phase24_search.csv     one row per evaluated config
    reports/phase24_metrics.csv    dev-cell grid (Phase-21 schema + n_params)
    models/phase24_L4-0X.pt        P1 primary models, {horizon: state_dict} + meta
    scripts/phase24_run.log        stdout (UTF-8 tee)

Run:
    python -u scripts\train_gnn.py gcn gat
    python -u scripts\train_gnn.py gcn gat gnn_lstm --smoke
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)   # cap threads: this host is memory-tight under load

ROOT = Path(r"D:\Flood predictor")
sys.path.insert(0, str(ROOT / "scripts"))

import train_sequence as ts                       # noqa: E402
from baseline_model import (HORIZONS, SEED, best_f1_threshold,  # noqa: E402
                            bootstrap_pr_auc, macro_pr_auc, metrics)

ROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REGISTRY = REPORTS / "phase17_experiment_registry.csv"
SEARCH_CSV = REPORTS / "phase24_search.csv"
MET_CSV = REPORTS / "phase24_metrics.csv"
RUN_LOG = ROOT / "scripts" / "phase24_run.log"
GRAPH_CSV = ROC / "station_graph.csv"

FAMS = {
    "gcn": dict(reg="L4-01", agg="gcn", budget=8),
    "gat": dict(reg="L4-02", agg="gat", budget=8),
    "gnn_lstm": dict(reg="L4-04", agg="gnn_lstm", budget=8),
}
GSPACE = {"lr": [1e-3, 3e-3], "hidden": [32, 64], "enc_layers": [1, 2],
          "g_layers": [1, 2], "dropout": [0.0, 0.3], "batch_dates": [8, 16],
          "_weight": ["off", "on"]}
GSPACE_SIZE = len([1 for _ in product(*GSPACE.values())])
MAX_EPOCHS = 14
PATIENCE = 4
FSET = "raw_sequence(12 vars)|graph"


# ----------------------------------------------------------------- model
def make_encoder(hidden, layers, dropout):
    drop = dropout if layers > 1 else 0.0
    return nn.LSTM(ts.CHANNELS, hidden, num_layers=layers, dropout=drop,
                   batch_first=True)


class GCNLayer(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, a_hat, node_mask):
        wh = self.lin(h)                                   # (D,N,H)
        out = torch.einsum("ij,bjh->bih", a_hat, wh)
        out = F.relu(out)
        out = self.drop(out)
        return (out + h) * node_mask.unsqueeze(-1)         # residual: keep node identity


class GATLayer(nn.Module):
    """Graph attention over an explicit edge list - mathematically identical to
    the dense (D,N,N) formulation (softmax over the same neighbours incl. the
    self-loop), but O(E) instead of O(N^2).  The edge list is the nonzero
    pattern of a_mask, so per-edge weight = softmax over a node's in-edges."""

    def __init__(self, hidden, dropout):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.a1 = nn.Parameter(torch.zeros(hidden))
        self.a2 = nn.Parameter(torch.zeros(hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, node_mask):
        tgt, src = edge_index[0], edge_index[1]            # (E,), (E,)
        D, N, H = h.shape
        E = tgt.numel()
        wh = self.lin(h)                                   # (D,N,H)
        w_src = wh.index_select(1, src)                    # (D,E,H)
        w_tgt = wh.index_select(1, tgt)
        e = (w_tgt * self.a1).sum(-1) + (w_src * self.a2).sum(-1)   # (D,E)
        e = F.leaky_relu(e, 0.2)
        valid = node_mask.index_select(1, src)             # (D,E): source present
        e = e.masked_fill(~valid, float("-inf"))
        idx = tgt.unsqueeze(0).expand(D, E)                # scatter index (D,E)
        m = torch.full((D, N), float("-inf"), dtype=e.dtype, device=e.device)
        m = m.scatter_reduce(1, idx, e, reduce="amax", include_self=True)
        e = torch.exp(e - m.gather(1, idx))
        denom = torch.zeros(D, N, dtype=e.dtype, device=e.device).scatter_add(
            1, idx, e)
        attn = e / denom.gather(1, idx).clamp_min(1e-12)
        attn = torch.nan_to_num(attn, 0.0)
        attn = self.drop(attn)
        msg = attn.unsqueeze(-1) * w_src                   # (D,E,H)
        out = torch.zeros(D, N, H, dtype=h.dtype, device=h.device).scatter_add(
            1, tgt.view(1, E, 1).expand(D, E, H), msg)
        out = F.elu(out)
        return (out + h) * node_mask.unsqueeze(-1)         # residual: keep node identity


class GNNNet(nn.Module):
    def __init__(self, agg, hidden, enc_layers, g_layers, dropout, a_hat, a_mask):
        super().__init__()
        self.agg = agg
        self.enc = make_encoder(hidden, enc_layers, dropout)
        self.register_buffer("a_hat", a_hat)
        self.register_buffer("a_mask", a_mask)
        rows, cols = torch.nonzero(a_mask, as_tuple=True)  # (tgt, src)
        self.register_buffer("edge_index",
                             torch.stack([rows, cols]).to(torch.long))
        layer = GCNLayer if agg == "gcn" else GATLayer
        self.graph = nn.ModuleList([layer(hidden, dropout) for _ in range(g_layers)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, nodes, node_mask):
        d, n, L, c = nodes.shape
        out, _ = self.enc(nodes.reshape(d * n, L, c))
        h = out[:, -1, :].reshape(d, n, -1) * node_mask.unsqueeze(-1)
        for gl in self.graph:
            if self.agg == "gcn":
                h = gl(h, self.a_hat, node_mask)
            else:
                h = gl(h, self.edge_index, node_mask)
        return self.head(h).squeeze(-1)                    # (D,N)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GNNLSTMNet(nn.Module):
    """Spatial-first spatiotemporal stack (L4-04, "full" GNN -> LSTM).

    Temporal encoder first (LSTM, as L4-01/02) runs before the graph layers in
    GNNNet; this variant reverses the composition so the *spatial* mechanism is
    the first signal a node sees: per-timestep linear projection, GCN message
    passing per timestep, then an LSTM over the per-node spatial states, then a
    per-node head.  Uniform neighbour weights (GCNLayer) - attention is already
    covered by the separate L4-02 family."""

    def __init__(self, hidden, enc_layers, g_layers, dropout, a_hat, a_mask):
        super().__init__()
        self.lin = nn.Linear(ts.CHANNELS, hidden)
        self.register_buffer("a_hat", a_hat)
        self.register_buffer("a_mask", a_mask)
        self.graph = nn.ModuleList(
            [GCNLayer(hidden, dropout) for _ in range(g_layers)])
        rnn_drop = dropout if enc_layers > 1 else 0.0
        self.rnn = nn.LSTM(hidden, hidden, num_layers=enc_layers,
                           dropout=rnn_drop, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, nodes, node_mask):
        d, n, L, c = nodes.shape
        h = self.lin(nodes) * node_mask.unsqueeze(-1).unsqueeze(-1)  # (D,N,L,H)
        hs = []
        for t in range(L):
            ht = h[:, :, t, :]
            for gl in self.graph:
                ht = gl(ht, self.a_hat, node_mask)
            hs.append(ht)
        seq = torch.stack(hs, dim=2)                                 # (D,N,L,H)
        out, _ = self.rnn(seq.reshape(d * n, L, -1))
        hh = out[:, -1, :].reshape(d, n, -1) * node_mask.unsqueeze(-1)
        return self.head(hh).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------- graph + index
def load_graph(sp_index: dict) -> tuple[torch.Tensor, torch.Tensor]:
    g = pd.read_csv(GRAPH_CSV)
    n = len(sp_index)
    adj = np.zeros((n, n), np.float32)
    for a, b in zip(g.station_id_a, g.station_id_b):
        i, j = sp_index[a], sp_index[b]
        adj[i, j] = adj[j, i] = 1.0
    tilde = adj + np.eye(n, dtype=np.float32)
    deg = tilde.sum(1)
    d_inv = np.diag(1.0 / np.sqrt(deg))
    a_hat = torch.tensor(d_inv @ tilde @ d_inv, dtype=torch.float32)
    a_mask = torch.tensor((adj + np.eye(n)) > 0)
    return a_hat, a_mask


def build_index_maps(ix):
    dp, sp = pd.factorize(ix.date)
    sid, st = pd.factorize(ix.station_id)
    n_dates, n_st = len(dp), len(st)
    row_of = np.full((n_dates, n_st), -1, np.int32)
    present = (ix.split.to_numpy() != "test")
    rows = np.arange(len(ix))
    row_of[dp[present], sid[present]] = rows[present]
    sp_index = {int(s): i for i, s in enumerate(st)}
    return dp, sid, row_of, sp_index


def build_view(idx, y, sp, tp):
    groups = {}
    for local, d in enumerate(tp[idx]):
        groups.setdefault(int(d), []).append(local)
    udates = np.array(sorted(groups), dtype=np.int64)
    return {"idx": idx, "y": y[idx], "sp": sp[idx], "groups": groups,
            "udates": udates, "n": len(idx)}


def gather(X, M, rows_mat, mu, sd):
    present = rows_mat >= 0
    safe = np.where(present, rows_mat, 0)
    sub = X[safe]                                          # (D,N,L,12)
    msub = M[safe].astype(np.float32)
    z = (sub - mu) / sd
    z[~np.isfinite(z)] = 0.0
    ch = np.concatenate([z.astype(np.float32), msub[:, :, :, None]], axis=-1)
    return (torch.from_numpy(np.ascontiguousarray(ch)),
            torch.from_numpy(present))


def forward_dates(net, X, M, row_of, mu, sd, dpos):
    nodes, nmask = gather(X, M, row_of[dpos], mu, sd)
    return net(nodes, nmask)                               # (D,N)


def predict_all(net, X, M, row_of, mu, sd, view, bs):
    net.eval()
    out = np.zeros(view["n"], np.float32)
    with torch.no_grad():
        for i in range(0, len(view["udates"]), bs):
            dpos = view["udates"][i:i + bs]
            logits = forward_dates(net, X, M, row_of, mu, sd, dpos)
            for bi, d in enumerate(dpos):
                locs = view["groups"][int(d)]
                sps = view["sp"][locs]
                out[locs] = logits[bi, sps].numpy()
    return out


# -------------------------------------------------------------------- fit
def fit_gnn(spec, cfg, X, M, row_of, a_hat, a_mask, vtr, vev, fixed_epochs=None):
    ts.seed_all(SEED)
    t0 = time.time()
    ratio = float((vtr["y"] == 0).sum() / max((vtr["y"] == 1).sum(), 1))
    pos_w = ratio if cfg["_weight"] == "on" else 1.0
    if spec["agg"] == "gnn_lstm":
        net = GNNLSTMNet(cfg["hidden"], cfg["enc_layers"], cfg["g_layers"],
                         cfg["dropout"], a_hat, a_mask)
    else:
        net = GNNNet(spec["agg"], cfg["hidden"], cfg["enc_layers"],
                     cfg["g_layers"], cfg["dropout"], a_hat, a_mask)
    # Station-disjoint training view of the index (D13/D14).  Only stations that
    # have rows in THIS training view may appear as graph nodes while gradients
    # are computed, so excluded-fold / held-out station features can never enter
    # the message passing during training.  Inference (vev / grid) still uses the
    # full observed graph - that matches deployment-time input availability.
    rows_tr = row_of.copy()
    keep = np.zeros(row_of.shape[1], dtype=bool)
    keep[np.unique(vtr["sp"])] = True
    rows_tr[:, ~keep] = -1
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
    bs = int(cfg["batch_dates"])
    ytr = torch.tensor(vtr["y"], dtype=torch.float32)
    gen = torch.Generator().manual_seed(SEED)
    cap = MAX_EPOCHS if fixed_epochs is None else fixed_epochs
    best_ap, best_ep, best_state, epochs = -1.0, 0, None, 0
    for ep in range(1, cap + 1):
        net.train()
        order = torch.randperm(len(vtr["udates"]), generator=gen)
        for i in range(0, len(order), bs):
            dpos = vtr["udates"][order[i:i + bs].numpy()]
            logits = forward_dates(net, X, M, rows_tr, vtr["mu"], vtr["sd"], dpos)
            preds, ys = [], []
            for bi, d in enumerate(dpos):
                locs = vtr["groups"][int(d)]
                preds.append(logits[bi, vtr["sp"][locs]])
                ys.append(ytr[locs])
            opt.zero_grad()
            loss = lossf(torch.cat(preds), torch.cat(ys))
            loss.backward()
            opt.step()
        epochs = ep
        if fixed_epochs is not None:
            continue
        p = predict_all(net, X, M, row_of, vtr["mu"], vtr["sd"], vev, bs)
        from sklearn.metrics import average_precision_score
        ap = float(average_precision_score(vev["y"], 1.0 / (1.0 + np.exp(-p))))
        if ap > best_ap + 1e-9:
            best_ap, best_ep = ap, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if ep - best_ep >= PATIENCE:
            break
    if fixed_epochs is None:
        net.load_state_dict(best_state)
    logits = predict_all(net, X, M, row_of, vtr["mu"], vtr["sd"], vev, bs)
    return {"net": net, "pred": 1.0 / (1.0 + np.exp(-logits)), "best_ap": best_ap,
            "best_ep": best_ep, "epochs": epochs, "fit_s": time.time() - t0,
            "pos_weight_used": pos_w}


def make_view(mask, y, sp, tp, X, M):
    idx = np.flatnonzero(mask & ~np.isnan(y))
    v = build_view(idx, y, sp, tp)
    v["mu"], v["sd"] = ts.fold_stats(X, M, idx)
    return v


def sample_cfgs(n):
    cells = [dict(zip(GSPACE, v)) for v in product(*GSPACE.values())]
    rng = np.random.default_rng(SEED + 7)
    pick = np.sort(rng.choice(len(cells), size=n, replace=False))
    return [{**cells[i], "_cell": int(i)} for i in pick]


# -------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    fams = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FAMS)
    bad = [f for f in fams if f not in FAMS]
    assert not bad, f"unknown families {bad}"
    global MAX_EPOCHS
    if smoke:
        MAX_EPOCHS = 2
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    log = open(RUN_LOG, "a", encoding="utf-8", buffering=1)
    sys.stdout = ts._Tee(sys.stdout, log)
    _print = print
    _print(f"Phase 24.2 — families={fams} | seed={SEED} | "
           f"{pd.Timestamp.now('UTC')}{' | SMOKE' if smoke else ''}")

    reg = pd.read_csv(REGISTRY)
    l4 = reg[reg.level == "L4"].set_index("model")
    for f in fams:
        row = l4.loc[f]
        assert row.test_policy == "frozen_read_at_phase33", f
        assert row.selection_cell == "val_2022" and row.selection_metric == "pr_auc"
        assert int(row.tuning_budget_configs) >= FAMS[f]["budget"], "budget > registry"
        assert row.gate == "phase24_graph_gate", f
    _print("registry OK: L4-01/L4-02/L4-04 present, gate=phase24_graph_gate, "
           "budget<=20")

    gm = json.loads((ROC / "station_graph_manifest.json").read_text())
    _print(f"graph: {gm['n_edges']} edges ({gm['n_river_edges']} river + "
           f"{gm['n_fallback_edges']} fallback k={gm['fallback_k']}), "
           f"mean deg {gm['mean_degree']}, connected={gm['connected']}, "
           f"sha={gm['sha256'][:12]}")

    ix = pd.read_csv(ROC / "sequence_index.csv", parse_dates=["date"])
    assert len(ix) == 200665
    lbl = pd.read_csv(ROC / "experimental_dataset.csv",
                      usecols=["station_id", "date"] + HORIZONS, parse_dates=["date"])
    m = ix[["station_id", "date"]].merge(lbl, on=["station_id", "date"], how="left")
    for h in HORIZONS:
        assert np.nansum(np.abs(ix[h].to_numpy() - m[h].to_numpy())) == 0, h
    _print(f"index labels re-checked 1:1 vs experimental_dataset.csv ({len(ix):,} rows)")

    dp, sp, row_of, sp_index = build_index_maps(ix)
    a_hat, a_mask = load_graph(sp_index)
    assert a_hat.shape[0] == len(sp_index) == 133
    _print(f"graph tensors: {a_hat.shape}, edge-mask nnz {int(a_mask.sum())}")

    tasks = ts.build_tasks_index(ix)
    dev = {c for t in tasks for _, c, _ in t["evals"]}
    assert dev == {"val_2022", "val_2020", "val_2021", "spatial_fold_1",
                   "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"}
    for t in tasks:
        assert not (t["train"] & ix.station_holdout.astype(bool).to_numpy()).any()
        assert not (t["train"] & (ix.split.to_numpy() == "test")).any()
    _print(f"dev grid OK: {sorted(dev)} — frozen test never scored")

    labels = {h: ix[h].to_numpy() for h in HORIZONS}
    L = int(json.loads((ROC / "phase21_lstar.json").read_text())["L_star"])
    X, M = ts.prepare(L)
    _print(f"temporal encoder: LSTM(hidden), windows L*=L={L}, {ts.CHANNELS} channels")

    for f in fams:
        _run_family(f, L, X, M, row_of, dp, sp, labels, tasks, a_hat, a_mask,
                    ix, smoke, _print)

    _print(f"\nTOTAL {time.time() - t0:.0f}s — exit=0")
    sys.stdout.flush()


def _run_family(family, L, X, M, row_of, dp, sp, labels, tasks, a_hat, a_mask,
                ix, smoke, _print):
    spec = FAMS[family]
    fam_t0 = time.time()
    reg_id = spec["reg"]
    pkl = MODELS / f"phase24_{reg_id}.pt"
    search = pd.read_csv(SEARCH_CSV) if SEARCH_CSV.exists() else pd.DataFrame()
    met = pd.read_csv(MET_CSV) if MET_CSV.exists() else pd.DataFrame()
    if (not met.empty and family in set(met["model"]) and pkl.exists()
            and not search.empty and family in set(search["family"])):
        _print(f"  skip {family}: already complete ({pkl.name})")
        return
    if not search.empty:
        search = search[search.family != family]
    if not met.empty:
        met = met[met.model != family]

    p1 = next(t for t in tasks if t["protocol"] == "P1_temporal_primary")
    season_v = ix.season.to_numpy()
    budget = 1 if smoke else spec["budget"]
    cfgs = sample_cfgs(budget)
    _print(f"\n=== {family} ({reg_id}) — tuning {len(cfgs)}/{GSPACE_SIZE} configs, "
           f"cell=val_2022, score=mean PR-AUC over 3 horizons ===")

    rows_search, best_cfg, best_score, best_cfg_id, best_epochs, n_param = \
        [], None, -1.0, None, None, None
    hz = HORIZONS[:1] if smoke else HORIZONS
    for ci, cfg in enumerate(cfgs):
        aucs, aps, fit_s = {}, {}, 0.0
        for hor in hz:
            yv = labels[hor]
            vtr = make_view(p1["train"], yv, sp, dp, X, M)
            ev_mask = next(m2 for _, c, m2 in p1["evals"] if c == "val_2022")
            vev = make_view(ev_mask, yv, sp, dp, X, M)
            r = fit_gnn(spec, cfg, X, M, row_of, a_hat, a_mask, vtr, vev)
            aucs[hor] = r["best_ap"] if r["best_ap"] > 0 else \
                float(metrics(vev["y"].astype(int), r["pred"], 0.5)["pr_auc"])
            aps[hor] = r["best_ep"]
            fit_s += r["fit_s"]
            n_param = r["net"].n_params()
        mean_auc = float(np.mean(list(aucs.values())))
        cfg_id = f"{family}_c{ci:02d}"
        rows_search.append({
            "family": family, "registry_id": reg_id, "config_id": cfg_id,
            "space_cell": cfg["_cell"],
            "params": json.dumps({k: v for k, v in cfg.items()
                                  if not k.startswith("_")}),
            "weighting": "pos_weight=ratio" if cfg["_weight"] == "on" else "none",
            **{f"pr_auc_{h[6:]}": aucs[h] for h in hz},
            "mean_pr_auc": mean_auc,
            "best_epochs": json.dumps({h[6:]: aps[h] for h in hz}),
            "fit_s": round(fit_s, 1), "n_configs_budget": budget,
            "space_size": GSPACE_SIZE, "selection_cell": "val_2022",
            "feature_set": FSET, "n_params": n_param})
        if mean_auc > best_score:
            best_score, best_cfg, best_cfg_id = mean_auc, cfg, cfg_id
            best_epochs = {h[6:]: aps[h] for h in hz}
        _print(f"  [{ci + 1}/{len(cfgs)}] {cfg_id} mean={mean_auc:.5f} "
               f"({', '.join(f'{h[6:]}={aucs[h]:.5f}' for h in hz)}) "
               f"{cfg['_weight']} ({fit_s:.0f}s)")

    winner = next(r for r in rows_search if r["config_id"] == best_cfg_id)
    n_param = int(winner["n_params"])
    _print(f"  -> winner {best_cfg_id} mean_pr_auc={best_score:.5f} "
           f"epochs={best_epochs} n_params={n_param:,}")

    _print(f"=== {family} — grid with {best_cfg_id} "
           f"(8 dev tasks x 3 seasons x 3 horizons) ===")
    rows_met = []
    saved = {"family": family, "registry_id": reg_id, "feature_set": FSET,
             "agg": spec["agg"], "L": L, "config_id": best_cfg_id,
             "params": {k: v for k, v in best_cfg.items() if not k.startswith("_")},
             "weighting": winner["weighting"], "seed": SEED, "train": "train_<2022",
             "n_params": n_param, "graph_sha256": json.loads(
                 (ROC / "station_graph_manifest.json").read_text())["sha256"],
             "epochs": best_epochs, "headline_metrics": {}, "state": {}}
    grid_tasks = [t for t in tasks if t["protocol"] == "P1_temporal_primary"] if smoke \
        else tasks
    grid_hz = HORIZONS[:1] if smoke else HORIZONS
    for task in grid_tasks:
        for hor in grid_hz:
            yv = labels[hor]
            vtr = make_view(task["train"], yv, sp, dp, X, M)
            if len(np.unique(vtr["y"])) < 2:
                continue
            ev0, cell0, ev_mask0 = task["evals"][0]
            vev = make_view(ev_mask0, yv, sp, dp, X, M)
            fixed = int(best_epochs[hor[6:]])
            r = fit_gnn(spec, best_cfg, X, M, row_of, a_hat, a_mask, vtr, vev,
                        fixed_epochs=fixed)
            net = r["net"]
            p_tr = 1.0 / (1.0 + np.exp(-predict_all(
                net, X, M, row_of, vtr["mu"], vtr["sd"], vtr,
                int(best_cfg["batch_dates"]))))
            thr = best_f1_threshold(vtr["y"].astype(int), p_tr)
            if task["protocol"] == "P1_temporal_primary":
                saved["state"][hor] = {k: v.cpu() for k, v in net.state_dict().items()}
                saved["fit_rows_" + hor] = int(vtr["n"])

            for protocol, cell, ev_mask2 in task["evals"]:
                for seas in (None, "monsoon", "non-monsoon"):
                    if smoke and seas is not None:
                        continue
                    msk = ev_mask2.copy()
                    if seas is not None:
                        msk = ev_mask2 & (season_v == seas)
                    idx = np.flatnonzero(msk & ~np.isnan(yv))
                    if len(idx) == 0:
                        continue
                    vsub = build_view(idx, yv, sp, dp)
                    vsub["mu"], vsub["sd"] = vtr["mu"], vtr["sd"]
                    p = 1.0 / (1.0 + np.exp(-predict_all(
                        net, X, M, row_of, vtr["mu"], vtr["sd"], vsub,
                        int(best_cfg["batch_dates"]))))
                    y = yv[idx].astype(int)
                    met_row = metrics(y, p, thr)
                    lo, hi = np.nan, np.nan
                    if seas is None and len(idx) >= 1000 and \
                            len(np.unique(sp[idx])) >= 10:
                        lo, hi = bootstrap_pr_auc(sp[idx], y, p)
                    if task["protocol"] == "P1_temporal_primary" and seas is None:
                        from sklearn.metrics import average_precision_score
                        saved["headline_metrics"][hor] = \
                            float(average_precision_score(y, p))
                    met_row.update(
                        protocol=protocol, train=task["name"], cell=cell,
                        season=seas or "all", model=family, horizon=hor,
                        n=len(idx), train_rows=int(vtr["n"]),
                        train_base_rate=float(vtr["y"].mean()),
                        pr_auc_ci_lo=lo, pr_auc_ci_hi=hi,
                        macro_pr_auc=macro_pr_auc(sp[idx], y, p),
                        fit_s=round(r["fit_s"], 1), registry_id=reg_id,
                        feature_set=FSET, weighting=winner["weighting"],
                        config_id=best_cfg_id, search_score=round(best_score, 5),
                        n_params=n_param)
                    rows_met.append(met_row)
            _print(f"  {task['protocol']:18s} {task['name']:13s} {hor} "
                   f"epochs {r['epochs']} fit {r['fit_s']:.0f}s")

    torch.save(saved, pkl)
    disk_s = pd.read_csv(SEARCH_CSV) if SEARCH_CSV.exists() else pd.DataFrame()
    disk_m = pd.read_csv(MET_CSV) if MET_CSV.exists() else pd.DataFrame()
    if not disk_s.empty:
        disk_s = disk_s[disk_s.family != family]
    if not disk_m.empty:
        disk_m = disk_m[disk_m.model != family]
    disk_s = pd.concat([disk_s, pd.DataFrame(rows_search)], ignore_index=True)
    disk_m = pd.concat([disk_m, pd.DataFrame(rows_met)], ignore_index=True)
    disk_s.to_csv(SEARCH_CSV, index=False)
    disk_m.to_csv(MET_CSV, index=False)
    _print(f"  checkpoint: {SEARCH_CSV.name} ({len(disk_s)} rows), "
           f"{MET_CSV.name} ({len(disk_m)} rows), {pkl.name}, "
           f"family time {time.time() - fam_t0:.0f}s")


if __name__ == "__main__":
    main()
