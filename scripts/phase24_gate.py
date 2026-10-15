"""Phase 24 gate (workflow stage 39, decision D13) - is a station-graph GNN justified?

Builds and audits candidate edge sets for the 133-station network and writes
reports/phase24_graph_gate.md's evidence (JSON + CSV + log). It does NOT train a GNN:
the gate only decides whether 24.2 may run.

Evidence produced, per the workflow Phase 24 body (24.1):
  * defensible edge set - river-network / basin connectivity first, distance fallback
  * coverage: stations with edges, edges, degree, components, connected?
  * edge temporal overlap and per-edge missingness
  * cross-station dependence: upstream/downstream lagged WL correlation vs a null
  * incremental value: does a neighbour column add over the station's own state
    (re-tests the Phase-16 F14-2 result on the graph edge subset)

ASCII-only console output (cp1252), seed fixed at 20261008.
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261008
rng = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REP = ROOT / "reports"

LOG = REP / "phase24_graph_gate.log"
OUT_JSON = REP / "phase24_graph_gate.json"
OUT_EDGES = REP / "phase24_graph_edges.csv"
FIG = ROOT / "figures" / "phase24_gate.png"

_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _lines.append(msg)


def check(cid: str, detail: str, ok: bool) -> bool:
    log(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}")
    return bool(ok)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


# --------------------------------------------------------------------------- #
# 1. station master (the 133 model-ready stations)
# --------------------------------------------------------------------------- #
log("=" * 78)
log("PHASE 24 GATE (stage 39, D13) - station graph & GNN justification")
log("=" * 78)

sv = pd.read_csv(PROC / "static_vectors.csv")
sv = sv.sort_values("station_id").reset_index(drop=True)
sid = sv["station_id"].to_numpy()
lat = sv["lat"].to_numpy(float)
lon = sv["lon"].to_numpy(float)
elev = sv["elevation_m"].to_numpy(float)
river = sv["river"].astype(str).to_numpy()
basin = sv["basin"].astype(str).to_numpy()
n = len(sv)
idx = {s: i for i, s in enumerate(sid)}

check("G1", f"station master: {n} stations, {sv.river.nunique()} rivers, "
            f"{sv.basin.nunique()} basins", n == 133)

# pairwise haversine (133 x 133, cheap)
D = np.full((n, n), np.inf)
for i in range(n):
    for j in range(i + 1, n):
        d = haversine_km(lat[i], lon[i], lat[j], lon[j])
        D[i, j] = D[j, i] = d


# --------------------------------------------------------------------------- #
# 2. candidate edge sets - hydrology first (Phase-11C rule), distance fallback
# --------------------------------------------------------------------------- #
def river_upstream_edges(same_river_dist_km: float = 250.0, rise_m: float = 0.5):
    """Phase-11C rule: same river, elevation higher by >= rise_m, nearest, <= dist.

    Returns directed edges (downstream -> upstream) as (d, u) index pairs.
    """
    edges = []
    for i in range(n):
        cand = [j for j in range(n)
                if j != i and river[j] == river[i]
                and elev[j] >= elev[i] + rise_m and D[i, j] <= same_river_dist_km]
        if cand:
            j = min(cand, key=lambda k: D[i, k])
            edges.append((i, j))
    return edges


river_up = river_upstream_edges()
river_up_set = set(river_up)

# all same-river pairs within 250 km (undirected)
same_river = [(i, j) for i, j in combinations(range(n), 2)
              if river[i] == river[j] and D[i, j] <= 250.0]
# basin co-membership (undirected)
basin_edges = [(i, j) for i, j in combinations(range(n), 2) if basin[i] == basin[j]]


def knn_edges(k: int):
    e = set()
    for i in range(n):
        order = np.argsort(D[i])
        c = 0
        for j in order:
            if j == i:
                continue
            e.add((min(i, int(j)), max(i, int(j))))
            c += 1
            if c == k:
                break
    return sorted(e)


knn3 = knn_edges(3)
knn5 = knn_edges(5)

EDGE_SETS = {
    "river_upstream": [tuple(sorted(e)) for e in river_up],
    "river_undirected": [tuple(sorted(e)) for e in river_up],
    "same_river_250km": same_river,
    "basin_clique": basin_edges,
    "knn3": knn3,
    "knn5": knn5,
}


# --------------------------------------------------------------------------- #
# 3. graph coverage / connectivity
# --------------------------------------------------------------------------- #
def graph_stats(edges):
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    deg = np.array([len(adj[i]) for i in range(n)])
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return {
        "nodes_covered": int((deg > 0).sum()),
        "n_edges": len(edges),
        "mean_degree": round(float(deg.mean()), 3),
        "max_degree": int(deg.max()),
        "isolated": int((deg == 0).sum()),
        "n_components": len(comps),
        "largest_component": len(comps[0]) if comps else 0,
        "is_connected": len(comps) == 1,
        "component_sizes": [len(c) for c in comps[:12]],
    }


stats = {name: graph_stats(e) for name, e in EDGE_SETS.items()}
# directed river detail
down = {a for a, b in river_up}
up = {b for a, b in river_up}

log("")
log("-- edge sets / coverage ------------------------------------------------")
log(f"{'graph':<18}{'edges':>7}{'cover':>7}{'iso':>5}{'comp':>6}{'largest':>9}{'conn':>7}")
for name, s in stats.items():
    log(f"{name:<18}{s['n_edges']:>7}{s['nodes_covered']:>7}{s['isolated']:>5}"
        f"{s['n_components']:>6}{s['largest_component']:>9}{str(s['is_connected']):>7}")
log(f"river_upstream directed: {len(down)} downstream stations, {len(up)} upstream "
    f"roles, {len(river_up)} directed edges")

# hydrology coverage of the *distance fallback*: how many river-uncovered stations
river_cov = {i for e in river_up for i in e}
log(f"stations reached by the river-upstream graph: {len(river_cov)}/{n}; "
    f"uncovered (need fallback): {n - len(river_cov)}")


# --------------------------------------------------------------------------- #
# 4. temporal coverage + WL series
# --------------------------------------------------------------------------- #
si = pd.read_csv(PROC / "sequence_index.csv", usecols=["station_id", "date", "split"])
si["date"] = pd.to_datetime(si["date"])
active = {s: set(g["date"]) for s, g in si.groupby("station_id")}

cols = ["station_id", "date", "wl_mean", "wl_last", "wl_pct_danger",
        "wl_dist_to_danger", "up_wl_pct_danger_lag1", "up_wl_pct_danger_lag2",
        "label_24h", "label_48h", "label_72h"]
exp = pd.read_csv(PROC / "experimental_dataset.csv", usecols=cols)
exp["date"] = pd.to_datetime(exp["date"])
wl = exp.pivot_table(index="date", columns="station_id", values="wl_pct_danger")
wl = wl.reindex(columns=sid)
wl_finite = {s: set(wl.index[np.isfinite(wl[s].to_numpy())]) for s in sid}
upwl = exp.pivot_table(index="date", columns="station_id",
                       values="up_wl_pct_danger_lag1").reindex(columns=sid)
upwl = upwl.reindex(wl.index)

check("G2", f"sequence_index {si.station_id.nunique()} stations, "
            f"{si.date.min().date()}..{si.date.max().date()}; "
            f"WL matrix {wl.shape[0]} dates x {wl.shape[1]} stations",
      si.station_id.nunique() == n and wl.shape[1] == n)

# edge temporal overlap + missingness
def edge_overlap(edges):
    ov_days, jac = [], []
    for a, b in edges:
        A, B = wl_finite[sid[a]], wl_finite[sid[b]]
        inter, union = len(A & B), len(A | B)
        ov_days.append(inter)
        jac.append(inter / union if union else 0.0)
    return np.array(ov_days), np.array(jac)


ov, jc = edge_overlap([tuple(sorted(e)) for e in river_up])
# edge-identity validation: reproduce the stored Phase-11C up_* feature exactly
val_rows = []
for d_i, u_i in river_up:
    a = upwl[sid[d_i]].to_numpy(float)
    b = wl[sid[u_i]].shift(1).to_numpy(float)
    m = np.isfinite(a) & np.isfinite(b)
    rate = float((np.abs(a[m] - b[m]) < 1e-6).mean()) if m.sum() else float("nan")
    val_rows.append((sid[d_i], sid[u_i], int(m.sum()), rate))
val = pd.DataFrame(val_rows, columns=["downstream", "upstream", "overlap", "match_rate"])
valid = val.dropna(subset=["match_rate"])
n_exact = int((valid["match_rate"] >= 0.99).sum())
check("G3", f"edge identity vs the stored Phase-11C up_* column: {n_exact}/{len(valid)} "
            f"edges reproduce upstream WL(t-1) exactly (match>=0.99), "
            f"{int(val.overlap.gt(0).sum())} edges have data; "
            f"{exp.up_wl_pct_danger_lag1.notna().groupby(exp.station_id).any().sum()} "
            f"stations carry up_* (Phase 11C reported 50 in model-ready)",
      n_exact >= 48 and int(val.overlap.gt(0).sum()) == 50)

log("")
log("-- edge temporal overlap & missingness (river-upstream edges) ----------")
log(f"overlap days (WL finite at both ends): min {int(ov.min())}, "
    f"median {int(np.median(ov))}, max {int(ov.max())}")
log(f"Jaccard overlap: median {np.median(jc):.3f}; "
    f"edges with < 1 year (365 d) overlap: {int((ov < 365).sum())}/{len(ov)}")
log(f"per-edge missingness (1 - Jaccard): median {1 - np.median(jc):.3f}")


# --------------------------------------------------------------------------- #
# 5. cross-station dependence: upstream(t-lag) vs downstream(t)
# --------------------------------------------------------------------------- #
def lag_corr(a_idx, b_idx, lag):
    u = wl[sid[a_idx]].to_numpy(float)          # upstream
    d = wl[sid[b_idx]].to_numpy(float)          # downstream
    u = np.roll(u, lag)
    u[:lag] = np.nan
    m = np.isfinite(u) & np.isfinite(d)
    if m.sum() < 200:
        return np.nan, int(m.sum())
    return float(np.corrcoef(u[m], d[m])[0, 1]), int(m.sum())


LAGS = [0, 1, 2, 3]
edge_corr = {L: [] for L in LAGS}
edge_n = {L: [] for L in LAGS}
for d_i, u_i in river_up:
    for L in LAGS:
        r, m = lag_corr(u_i, d_i, L)
        if np.isfinite(r):
            edge_corr[L].append(r)
            edge_n[L].append(m)

# null: random ordered pairs (excluding true edges and self)
all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
true_pairs = set(river_up)
non_edge = [p for p in all_pairs if p not in true_pairs]
null_draw = rng.choice(len(non_edge), size=min(4000, len(non_edge)), replace=False)
null_pairs = [non_edge[k] for k in null_draw]
null_corr = {L: [] for L in LAGS}
for u_i, d_i in null_pairs:
    for L in LAGS:
        r, _ = lag_corr(u_i, d_i, L)
        if np.isfinite(r):
            null_corr[L].append(r)

log("")
log("-- cross-station dependence: upstream WL(t-lag) vs downstream WL(t) -----")
log(f"{'lag':>4}{'edge_r_mean':>13}{'edge_r_med':>12}{'null_r_mean':>13}"
    f"{'null_r_med':>12}{'delta':>9}")
dep = {}
for L in LAGS:
    e = np.array(edge_corr[L])
    nn = np.array(null_corr[L])
    dep[L] = {"edge_mean": float(e.mean()), "edge_median": float(np.median(e)),
              "edge_p10": float(np.percentile(e, 10)), "edge_p90": float(np.percentile(e, 90)),
              "null_mean": float(nn.mean()), "null_median": float(np.median(nn)),
              "delta": float(e.mean() - nn.mean()), "n_edges": int(len(e))}
    log(f"{L:>4}{e.mean():>13.3f}{np.median(e):>12.3f}{nn.mean():>13.3f}"
        f"{np.median(nn):>12.3f}{e.mean() - nn.mean():>9.3f}")

# fraction of edges with strong positive lag-1 correlation
e1 = np.array(edge_corr[1])
frac_strong = float((e1 >= 0.5).mean())
log(f"fraction of river edges with lag-1 r >= 0.5: {frac_strong:.3f}")


# --------------------------------------------------------------------------- #
# 6. incremental value: does a neighbour column add over the station's own state?
# --------------------------------------------------------------------------- #
sub = exp.dropna(subset=["up_wl_pct_danger_lag1"]).copy()
res = {"n_rows_with_upstream": int(len(sub)),
       "n_stations_with_upstream": int(sub.station_id.nunique())}
if len(sub):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score, roc_auc_score

    sub["year"] = sub["date"].dt.year
    sub["split"] = np.where(sub["year"] < 2022, "train",
                            np.where(sub["year"] == 2022, "val", "test"))
    for split, tr in [("P1 val_2022 (label_24h)", "label_24h"),
                      ("P1 val_2022 (label_48h)", "label_48h"),
                      ("P1 val_2022 (label_72h)", "label_72h")]:
        d = sub.dropna(subset=[tr]).copy()
        tr_m = d["split"] == "train"
        va_m = d["split"] == "val"
        own = ["wl_pct_danger", "wl_dist_to_danger"]
        add = ["up_wl_pct_danger_lag1", "up_wl_pct_danger_lag2"]
        out = {}
        for tag, feats in [("own", own), ("own+upstream", own + add)]:
            X = d[feats].to_numpy(float)
            imp = SimpleImputer(strategy="median").fit(X[tr_m.to_numpy()])
            sc = StandardScaler().fit(imp.transform(X[tr_m.to_numpy()]))
            m = LogisticRegression(max_iter=2000, class_weight="balanced")
            m.fit(sc.transform(imp.transform(X[tr_m.to_numpy()])),
                  d.loc[tr_m, tr].to_numpy())
            p = m.predict_proba(sc.transform(imp.transform(X[va_m.to_numpy()])))[:, 1]
            y = d.loc[va_m, tr].to_numpy()
            out[tag] = {"pr_auc": round(float(average_precision_score(y, p)), 4),
                        "roc_auc": round(float(roc_auc_score(y, p)), 4)}
        res[tr] = {"own": out["own"], "own+upstream": out["own+upstream"],
                   "delta_pr_auc": round(out["own+upstream"]["pr_auc"] - out["own"]["pr_auc"], 4)}
        log(f"incremental {tr}: own pr_auc {out['own']['pr_auc']} -> "
            f"own+upstream {out['own+upstream']['pr_auc']} "
            f"(delta {res[tr]['delta_pr_auc']:+.4f})")

log("")
log(f"edge-subset rows with an upstream: {res['n_rows_with_upstream']} "
    f"({res['n_stations_with_upstream']} stations)")


# --------------------------------------------------------------------------- #
# 7. write evidence
# --------------------------------------------------------------------------- #
edges_rows = []
for d_i, u_i in river_up:
    edges_rows.append({
        "downstream": sid[d_i], "upstream": sid[u_i],
        "downstream_river": river[d_i], "upstream_river": river[u_i],
        "elev_down": float(elev[d_i]), "elev_up": float(elev[u_i]),
        "dist_km": round(float(D[d_i, u_i]), 2),
        "overlap_days": int(len(wl_finite[sid[d_i]] & wl_finite[sid[u_i]])),
        "lag1_r": round(float(lag_corr(u_i, d_i, 1)[0]), 4),
    })
pd.DataFrame(edges_rows).to_csv(OUT_EDGES, index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(LAGS, [dep[L]["edge_mean"] for L in LAGS], "o-", color="#c0392b",
           label="river-upstream edges")
ax[0].plot(LAGS, [dep[L]["null_mean"] for L in LAGS], "s--", color="#7f8c8d",
           label="null (random station pairs)")
ax[0].set_xlabel("propagation lag (days)")
ax[0].set_ylabel("mean Pearson r (wl_pct_danger)")
ax[0].set_title("Cross-station dependence: upstream(t-lag) vs downstream(t)")
ax[0].legend()
ax[0].grid(alpha=0.3)
names = list(stats)
cov = [stats[nm]["nodes_covered"] for nm in names]
ax[1].bar(range(len(names)), cov, color="#2980b9")
ax[1].axhline(n, color="k", ls=":", lw=1)
ax[1].text(len(names) - 1, n - 10, f"all {n}", ha="right", fontsize=8)
ax[1].set_xticks(range(len(names)))
ax[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax[1].set_ylabel("stations covered")
ax[1].set_title("Graph coverage")
ax[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIG, dpi=120)
log(f"figure: {FIG.relative_to(ROOT)}")

summary = {
    "seed": SEED, "n_stations": n,
    "rivers": int(sv.river.nunique()), "basins": int(sv.basin.nunique()),
    "river_upstream": {"directed_edges": len(river_up),
                       "downstream_stations": len(down), "upstream_roles": len(up)},
    "coverage": stats,
    "dep": dep, "frac_lag1_r_ge_0.5": frac_strong,
    "incremental_value": res,
    "edge_validation": {
        "n_edges": int(len(val)),
        "n_with_data": int(val.overlap.gt(0).sum()),
        "n_exact_match_ge_099": n_exact,
        "mean_match_rate": round(float(valid.match_rate.mean()), 4),
    },
}
OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
LOG.write_text("\n".join(_lines) + "\n", encoding="utf-8")
log("")
log(f"wrote {OUT_JSON.relative_to(ROOT)}, {OUT_EDGES.relative_to(ROOT)}, "
    f"{LOG.relative_to(ROOT)}")
