r"""Phase 24.2 - freeze the station graph artifact (workflow stage 39, D13 pass).

Graph = hydrology-first message-passing edges (Phase-11C rule: same river, higher
elevation, <= 250 km) UNION the nearest-k haversine fallback that connects all 133
stations.  Distance is a *fallback only* (workflow 24.1).

k is fixed at 5 - the value approved by the pre-registered gate
(reports/phase24_graph_gate.md sec. 8: G = river_upstream edges UNION knn5 to
connect the 51 isolated nodes).  The earlier k=2 freeze was an implementation
shortcut (minimal k that happened to connect the union); that artifact was
invalidated and rebuilt with the approved knn5.

Outputs (freeze before any model sees them):
    data/processed/station_graph.csv      station_id_a, station_id_b, kind
    data/processed/station_graph_manifest.json   counts + SHA-256 of the CSV

Run:
    python scripts\build_graph.py
"""
from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

FALLBACK_K = 5          # gate-approved: river_upstream UNION knn5


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def connected(adj, n):
    seen, start = {0}, 0
    stack = [0]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == n


def main() -> None:
    sv = pd.read_csv(PROC / "static_vectors.csv").sort_values("station_id").reset_index(drop=True)
    sid = sv.station_id.to_numpy()
    la, lo = sv.lat.to_numpy(float), sv.lon.to_numpy(float)
    el = sv.elevation_m.to_numpy(float)
    riv = sv.river.astype(str).to_numpy()
    n = len(sv)

    D = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = haversine_km(la[i], lo[i], la[j], lo[j])

    # river-upstream rule (Phase-11C): same river, >= 0.5 m higher, nearest, <= 250 km
    river = set()
    for i in range(n):
        cand = [j for j in range(n) if j != i and riv[j] == riv[i]
                and el[j] >= el[i] + 0.5 and D[i, j] <= 250.0]
        if cand:
            j = min(cand, key=lambda k: D[i, k])
            river.add((min(i, j), max(i, j)))

    def knn(k):
        e = set()
        for i in range(n):
            order = [int(j) for j in np.argsort(D[i]) if int(j) != i][:k]
            for j in order:
                e.add((min(i, j), max(i, j)))
        return e

    chosen_k = FALLBACK_K
    edges = river | knn(chosen_k)
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    assert connected(adj, n), f"river + knn{FALLBACK_K} is not connected"

    rows = []
    for a, b in sorted(edges):
        kind = "river" if (a, b) in river else "knn_fallback"
        rows.append({"station_id_a": int(sid[a]), "station_id_b": int(sid[b]),
                     "kind": kind, "dist_km": round(float(D[a, b]), 3)})
    g = pd.DataFrame(rows)
    out = PROC / "station_graph.csv"
    g.to_csv(out, index=False)
    h = hashlib.sha256(out.read_bytes()).hexdigest()

    deg = {}
    for a, b in edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    manifest = {
        "created": pd.Timestamp.now("UTC").isoformat(),
        "phase": "24.2 (station graph & GNN, stage 39)",
        "rule": "Phase-11C river-upstream (same river, +0.5m, <=250km) UNION "
                f"nearest-k={chosen_k} haversine fallback (gate-approved knn5, "
                "reports/phase24_graph_gate.md sec. 8)",
        "n_stations": n,
        "n_edges": len(edges),
        "n_river_edges": int((g.kind == "river").sum()),
        "n_fallback_edges": int((g.kind == "knn_fallback").sum()),
        "fallback_k": chosen_k,
        "mean_degree": round(len(edges) * 2 / n, 3),
        "min_degree": int(min(deg.values())),
        "max_degree": int(max(deg.values())),
        "connected": True,
        "file": "data/processed/station_graph.csv",
        "sha256": h,
    }
    (PROC / "station_graph_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                      encoding="utf-8")
    print("station graph frozen:")
    for k, v in manifest.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
