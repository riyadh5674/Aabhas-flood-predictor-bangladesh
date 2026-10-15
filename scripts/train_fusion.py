r"""Phase 23 - Multi-input / feature-fusion deep learning (workflow stage 38).

Location-aware prediction: the temporal encoder is allowed to see *where* it is.

Branches
--------
* Branch 1 - sequence encoder: the Phase-21/22 winner is the LSTM at L* = 3
  (registry L3-01). Fusion therefore reuses that encoder (kind=lstm, L=L*) and
  only adds the static branch, so any gain is attributable to the fusion input,
  not to a different temporal model.
* Branch 2 - static vector: data/processed/static_vectors.csv (script
  build_static.py) - lat/lon, danger level, terrain + distances (10 numerics,
  standardised on the training fold) and river/basin/division/district/upazilla
  (5 categoricals, frequency-encoded on the training fold; unseen -> 0, the same
  FreqEncoder rule as Phase 14/16).

Variants (all registered; the registry is the contract)
-------------------------------------------------------
* L3-07 fusion_L3  - sequence encoder + static MLP -> concat -> dense -> sigmoid.
* L3-08 static_only- static MLP alone (no sequence) - the feature-group baseline.
* L3-09 hybrid_L3  - fusion + the Phase-12 LINEAR engineered set appended to the
                     head (the "learned + engineered" hybrid; optional variant).
temporal-only is registry L3-01 (epoch for epoch, from Phase 21) - reported, not
re-trained here.

Parity / honesty
----------------
* Same data (Dataset B), same Phase-15 split masks, same dev cells, same metric
  code, same selection cell (val_2022) and seed as every other family (D14).
* Standardisation is fit on the training fold only - sequence stats from the
  observed timesteps of the fold (train_sequence.fold_stats), static stats and
  frequency maps from the fold's training stations. Nothing is fit on val dates
  or on stations that only appear in an evaluation cell.
* The frozen 2023+ test period is never read (asserted).

Outputs
-------
    reports/phase23_search.csv    one row per evaluated config
    reports/phase23_metrics.csv   dev-cell grid (Phase-21 schema + n_params)
    models/phase23_L3-0X.pt       P1 primary models, {horizon: state_dict} + meta
    scripts/phase23_run.log       stdout (UTF-8 tee)

Run:
    python -u scripts\train_fusion.py static_only fusion_L3 hybrid_L3
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(r"D:\Flood predictor")
sys.path.insert(0, str(ROOT / "scripts"))

import train_sequence as ts                       # noqa: E402  (reuse data/split helpers)
from baseline_model import (HORIZONS, SEED, best_f1_threshold,  # noqa: E402
                            bootstrap_pr_auc, macro_pr_auc, metrics,
                            TEXT_COLS, preprocessor)

ROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REGISTRY = REPORTS / "phase17_experiment_registry.csv"
SEARCH_CSV = REPORTS / "phase23_search.csv"
MET_CSV = REPORTS / "phase23_metrics.csv"
RUN_LOG = ROOT / "scripts" / "phase23_run.log"
STATIC_CSV = ROC / "static_vectors.csv"
FSETS = json.loads((ROC / "feature_sets.json").read_text(encoding="utf-8"))["sets"]

NUM = ["lat", "lon", "danger_level", "elevation_m", "slope_deg",
       "dist_to_coast_km", "dist_river_km", "dist_major_river_km",
       "river_density_10km_km_per_km2", "n_river_lines_10km"]
CAT = ["river", "basin", "division", "district", "upazilla"]
STATIC_DIM = len(NUM) + len(CAT)                 # 15
LINEAR = FSETS["LINEAR"]
LIN_DIM = len(LINEAR)                            # 35

# family -> spec.  temporal encoder is fixed to the Phase-21 winner (lstm, L*).
FAMS = {
    "fusion_L3": dict(reg="L3-07", use_seq=True, use_static=True, lin=False,
                      kind="lstm", budget=16),
    "static_only": dict(reg="L3-08", use_seq=False, use_static=True, lin=False,
                        kind=None, budget=16),
    "hybrid_L3": dict(reg="L3-09", use_seq=True, use_static=True, lin=True,
                      kind="lstm", budget=12),
}
SPACE = ts.SPACE                                 # {lr,hidden,layers,dropout,batch,_weight}
SPACE_SIZE = ts.SPACE_SIZE
MAX_EPOCHS = ts.MAX_EPOCHS
PATIENCE = ts.PATIENCE


def resolve_lstar() -> int:
    return int(json.loads((ROC / "phase21_lstar.json").read_text())["L_star"])


# --------------------------------------------------------------- static branch
class _StaticMLP(nn.Module):
    def __init__(self, in_dim, hidden, depth, dropout):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_encoder(kind, hidden, layers, dropout):
    if kind in ("lstm", "gru"):
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        drop = dropout if layers > 1 else 0.0
        return rnn(ts.CHANNELS, hidden, num_layers=layers, dropout=drop,
                   batch_first=True)
    return ts._TCNEncoder(hidden, layers, ts.TN_KERNEL, dropout)


class FusionNet(nn.Module):
    """Static MLP and/or temporal encoder -> concat -> dense -> sigmoid logit."""

    def __init__(self, kind, hidden, layers, dropout, use_seq, use_static,
                 lin_dim, static_dim=STATIC_DIM):
        super().__init__()
        self.kind, self.use_seq, self.use_static, self.lin_dim = \
            kind, use_seq, use_static, lin_dim
        if use_seq:
            self.enc = make_encoder(kind, hidden, layers, dropout)
        if use_static:
            self.stat = _StaticMLP(static_dim, hidden, layers, dropout)
        head_in = ((hidden if use_seq else 0)
                   + (hidden if use_static else 0) + lin_dim)
        self.head = nn.Linear(head_in, 1)

    def forward(self, xs, xst, xl):
        parts = []
        if self.use_seq:
            if self.kind in ("lstm", "gru"):
                out, _ = self.enc(xs)
                parts.append(out[:, -1, :])
            else:
                parts.append(self.enc(xs))
        if self.use_static:
            parts.append(self.stat(xst))
        if self.lin_dim:
            parts.append(xl)
        h = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return self.head(h).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------- static encoding (train)
def fit_static(S: pd.DataFrame, st_train: np.ndarray):
    sub = S.loc[st_train]
    mu = sub[NUM].mean().to_numpy(np.float32)
    sd = sub[NUM].std(ddof=0).replace(0.0, 1.0).fillna(1.0).to_numpy(np.float32)
    freq = {}
    for c in CAT:
        s = sub[c].astype("object").where(sub[c].notna(), "__NA__")
        freq[c] = s.value_counts(normalize=True).to_dict()
    return mu, sd, freq


def encode_static(S: pd.DataFrame, st_idx: np.ndarray, mu, sd, freq) -> np.ndarray:
    sub = S.loc[st_idx]
    z = (sub[NUM].to_numpy(np.float64) - mu) / sd
    z[~np.isfinite(z)] = 0.0
    cats = [sub[c].astype("object").where(sub[c].notna(), "__NA__")
            .map(freq[c]).fillna(0.0).to_numpy(np.float64) for c in CAT]
    return np.concatenate([z, np.column_stack(cats)], axis=1).astype(np.float32)


def fit_encoders(spec, X, M, S, d_lin, tr_idx) -> dict:
    """Fit every fold encoder ONCE on the training rows (sequence stats / static
    standardiser+freq map / LINEAR preprocessor). Nothing here sees eval rows."""
    enc = {}
    if spec["use_seq"]:
        enc["mu"], enc["sd"] = ts.fold_stats(X, M, tr_idx)
    if spec["use_static"]:
        enc["smu"], enc["ssd"], enc["freq"] = fit_static(S, STATION_OF[tr_idx])
    if spec["lin"]:
        pre = preprocessor([c for c in LINEAR if c not in TEXT_COLS],
                           [c for c in LINEAR if c in TEXT_COLS], scale=True)
        pre.fit(d_lin.iloc[tr_idx][LINEAR], None)
        enc["pre"] = pre
    return enc


def encode(enc, spec, X, M, S, d_lin, idx) -> dict:
    """Apply the training-fit encoders to any row subset -> dict-of-tensors."""
    d = {}
    if spec["use_seq"]:
        d["seq"] = torch.tensor(ts.std_rows(X, M, idx, enc["mu"], enc["sd"]))
    if spec["use_static"]:
        d["stat"] = torch.tensor(
            encode_static(S, STATION_OF[idx], enc["smu"], enc["ssd"], enc["freq"]))
    if spec["lin"]:
        d["lin"] = torch.tensor(
            np.asarray(enc["pre"].transform(d_lin.iloc[idx][LINEAR]), dtype=np.float32))
    return d


STATION_OF: np.ndarray = None                  # station_id per ix row (set in main)


def fit_net(spec, cfg, tr_t, ytr, ev_t, yev, fixed_epochs=None):
    ts.seed_all(SEED)
    t0 = time.time()
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    yev_t = torch.tensor(yev, dtype=torch.float32)
    ratio = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    pos_w = ratio if cfg["_weight"] == "on" else 1.0
    net = FusionNet(spec["kind"] or "lstm", cfg["hidden"], cfg["layers"],
                    cfg["dropout"], spec["use_seq"], spec["use_static"],
                    LIN_DIM if spec["lin"] else 0)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
    bs = int(cfg["batch_size"])
    n = ytr.shape[0]
    best_ap, best_ep, best_state, epochs = -1.0, 0, None, 0
    gen = torch.Generator().manual_seed(SEED)
    cap = MAX_EPOCHS if fixed_epochs is None else fixed_epochs
    for ep in range(1, cap + 1):
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            xs = tr_t["seq"][b] if spec["use_seq"] else None
            xst = tr_t["stat"][b] if spec["use_static"] else None
            xl = tr_t["lin"][b] if spec["lin"] else None
            opt.zero_grad()
            loss = lossf(net(xs, xst, xl), ytr_t[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(ev_t.get("seq"), ev_t.get("stat"), ev_t.get("lin")))
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
        p = torch.sigmoid(net(ev_t.get("seq"), ev_t.get("stat"), ev_t.get("lin")))
    return {"net": net, "pred": p.numpy(), "best_ap": best_ap, "best_ep": best_ep,
            "epochs": epochs, "fit_s": time.time() - t0, "pos_weight_used": pos_w}


# ----------------------------------------------------------------------- main
def out_paths(family):
    return (SEARCH_CSV, MET_CSV, MODELS / f"phase23_{FAMS[family]['reg']}.pt")


def main() -> None:
    global STATION_OF
    t0 = time.time()
    fams = sys.argv[1:] or list(FAMS)
    bad = [f for f in fams if f not in FAMS]
    assert not bad, f"unknown families {bad}"
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    log = open(RUN_LOG, "a", encoding="utf-8")
    sys.stdout = ts._Tee(sys.stdout, log)
    _print = print
    _print(f"Phase 23 — families={fams} | seed={SEED} | {pd.Timestamp.now('UTC')}")

    reg = pd.read_csv(REGISTRY)
    l3 = reg[reg.level == "L3"].set_index("model")
    for f in fams:
        row = l3.loc[f]
        assert row.test_policy == "frozen_read_at_phase33", f
        assert row.selection_cell == "val_2022" and row.selection_metric == "pr_auc"
        assert int(row.tuning_budget_configs) >= FAMS[f]["budget"], "budget > registry"
    _print("registry OK: L3-07/08/09 present, test_policy=frozen_read_at_phase33, "
           "budgets <= 16")

    ix = pd.read_csv(ROC / "sequence_index.csv", parse_dates=["date"])
    assert len(ix) == 200665
    lbl = pd.read_csv(ROC / "experimental_dataset.csv",
                      usecols=["station_id", "date"] + HORIZONS,
                      parse_dates=["date"])
    m = ix[["station_id", "date"]].merge(lbl, on=["station_id", "date"], how="left")
    for h in HORIZONS:
        assert np.nansum(np.abs(ix[h].to_numpy() - m[h].to_numpy())) == 0, h
    _print(f"index labels re-checked 1:1 vs experimental_dataset.csv ({len(ix):,} rows)")

    STATION_OF = ix.station_id.to_numpy()
    S = pd.read_csv(STATIC_CSV).set_index("station_id")
    assert set(np.unique(STATION_OF)) <= set(S.index), "static vector missing stations"
    _print(f"static vector: {S.shape[0]} stations x {STATIC_DIM} dims "
           f"({len(NUM)} numeric + {len(CAT)} categorical)")

    d_lin = None
    if any(FAMS[f]["lin"] for f in fams):
        d_lin = pd.read_csv(ROC / "experimental_dataset.csv",
                            usecols=["station_id", "date"] + LINEAR,
                            parse_dates=["date"])
        d_lin = ix[["station_id", "date"]].merge(d_lin, on=["station_id", "date"],
                                                 how="left")
        _print(f"engineered LINEAR block: {LIN_DIM} features for hybrid_L3")

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
    L = resolve_lstar()
    _print(f"temporal encoder fixed to Phase-21 winner: kind=lstm, L*=L={L}")

    for f in fams:
        _run_family(f, L, ix, tasks, labels, stations, season_v, S, d_lin, _print)

    _print(f"\nTOTAL {time.time() - t0:.0f}s — exit=0")
    sys.stdout.flush()


def _run_family(family, L, ix, tasks, labels, stations, season_v, S, d_lin, _print):
    spec = FAMS[family]
    fam_t0 = time.time()
    reg_id = spec["reg"]
    search_csv, met_csv, pkl = out_paths(family)
    search = pd.read_csv(search_csv) if search_csv.exists() else pd.DataFrame()
    met = pd.read_csv(met_csv) if met_csv.exists() else pd.DataFrame()
    if (not met.empty and family in set(met["model"]) and pkl.exists()
            and not search.empty and family in set(search["family"])):
        _print(f"  skip {family}: already complete ({pkl.name})")
        return
    if not search.empty:
        search = search[search.family != family]
    if not met.empty:
        met = met[met.model != family]

    X = M = None
    if spec["use_seq"]:
        X, M = ts.prepare(L)
    p1 = next(t for t in tasks if t["protocol"] == "P1_temporal_primary")
    tri = p1["train"]
    ev_mask = next(m2 for _, c, m2 in p1["evals"] if c == "val_2022")

    cfgs = ts._sample_configs(spec["budget"])
    _print(f"\n=== {family} ({reg_id}) — tuning {len(cfgs)}/{SPACE_SIZE} configs, "
           f"cell=val_2022, score=mean PR-AUC over 3 horizons ===")
    # encoders depend only on the horizon's train/val masks (identical for every
    # config), so fit once per horizon and reuse -> far fewer passes over X.
    TUNE = {}
    for hor in HORIZONS:
        yv = labels[hor]
        tr_idx = np.flatnonzero(tri & ~np.isnan(yv))
        ev_idx = np.flatnonzero(ev_mask & ~np.isnan(yv))
        enc = fit_encoders(spec, X, M, S, d_lin, tr_idx)
        TUNE[hor] = (tr_idx, ev_idx,
                     encode(enc, spec, X, M, S, d_lin, tr_idx),
                     encode(enc, spec, X, M, S, d_lin, ev_idx))
    rows_search, best_cfg, best_score, best_cfg_id, best_epochs, n_param = \
        [], None, -1.0, None, None, None

    for ci, cfg in enumerate(cfgs):
        aucs, aps, fit_s = {}, {}, 0.0
        for hor in HORIZONS:
            yv = labels[hor]
            tr_idx, ev_idx, tr_t, ev_t = TUNE[hor]
            r = fit_net(spec, cfg, tr_t, yv[tr_idx], ev_t, yv[ev_idx])
            aucs[hor] = r["best_ap"] if r["best_ap"] > 0 else \
                float(metrics(yv[ev_idx].astype(int), r["pred"], 0.5)["pr_auc"])
            aps[f"best_ep_{hor}"] = r["best_ep"]
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
            "pr_auc_24h": aucs[HORIZONS[0]], "pr_auc_48h": aucs[HORIZONS[1]],
            "pr_auc_72h": aucs[HORIZONS[2]], "mean_pr_auc": mean_auc,
            "best_epochs": json.dumps({h.split("_")[-1]: v for h, v in aps.items()}),
            "fit_s": round(fit_s, 1), "n_configs_budget": spec["budget"],
            "space_size": SPACE_SIZE, "selection_cell": "val_2022",
            "feature_set": _fset(family, L), "n_params": n_param})
        if mean_auc > best_score:
            best_score, best_cfg, best_cfg_id = mean_auc, cfg, cfg_id
            best_epochs = {h.split("_")[-1]: v for h, v in aps.items()}
        _print(f"  [{ci + 1}/{len(cfgs)}] {cfg_id} mean={mean_auc:.5f} "
               f"(24h={aucs[HORIZONS[0]]:.5f} 48h={aucs[HORIZONS[1]]:.5f} "
               f"72h={aucs[HORIZONS[2]]:.5f}) {cfg['_weight']} ({fit_s:.0f}s)")

    winner = next(r for r in rows_search if r["config_id"] == best_cfg_id)
    n_param = int(winner["n_params"])            # the WINNER's params, not the last cfg
    _print(f"  -> winner {best_cfg_id} mean_pr_auc={best_score:.5f} "
           f"epochs={best_epochs} n_params={n_param:,}")

    _print(f"=== {family} — grid with {best_cfg_id} "
           f"(8 dev tasks x 3 seasons x 3 horizons) ===")
    rows_met = []
    saved = {"family": family, "registry_id": reg_id, "feature_set": _fset(family, L),
             "kind": spec["kind"], "L": L, "config_id": best_cfg_id,
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
            ev0, cell0, ev_mask0 = task["evals"][0]
            ev_idx = np.flatnonzero(ev_mask0 & ~np.isnan(yv))
            enc = fit_encoders(spec, X, M, S, d_lin, tr_idx)
            tr_t = encode(enc, spec, X, M, S, d_lin, tr_idx)
            ev_t = encode(enc, spec, X, M, S, d_lin, ev_idx)
            fixed = int(best_epochs[hor.split("_")[-1]])
            r = fit_net(spec, cfg=best_cfg, tr_t=tr_t, ytr=yv[tr_idx], ev_t=ev_t,
                        yev=yv[ev_idx], fixed_epochs=fixed)
            net = r["net"]
            with torch.no_grad():
                p_tr = torch.sigmoid(net(tr_t.get("seq"), tr_t.get("stat"),
                                         tr_t.get("lin"))).numpy()
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
                    # encode the (possibly season-subset) eval rows with the
                    # task's training-fit encoders
                    ev_s = encode(enc, spec, X, M, S, d_lin, idx)
                    with torch.no_grad():
                        p = torch.sigmoid(net(ev_s.get("seq"), ev_s.get("stat"),
                                              ev_s.get("lin"))).numpy()
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
                        fit_s=round(r["fit_s"], 1), registry_id=reg_id,
                        feature_set=saved["feature_set"], weighting=winner["weighting"],
                        config_id=best_cfg_id, search_score=round(best_score, 5),
                        n_params=n_param)
                    rows_met.append(met_row)
            _print(f"  {task['protocol']:18s} {task['name']:13s} {hor} "
                   f"epochs {r['epochs']} fit {r['fit_s']:.0f}s")

    torch.save(saved, pkl)
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


def _fset(family, L):
    if family == "static_only":
        return "static(group A+B geography)"
    if family == "hybrid_L3":
        return f"raw_sequence(12 vars)|L={L} + static + engineered LINEAR"
    return f"raw_sequence(12 vars)|L={L} + static"


if __name__ == "__main__":
    main()
