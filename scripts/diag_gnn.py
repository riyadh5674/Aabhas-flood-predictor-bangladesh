import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(r"D:\Flood predictor")
sys.path.insert(0, str(ROOT / "scripts"))
import train_sequence as ts
import train_gnn as g

g.MAX_EPOCHS = 14
ROC = ROOT / "data" / "processed"
ix = pd.read_csv(ROC / "sequence_index.csv", parse_dates=["date"])
dp, sp, row_of, sp_index = g.build_index_maps(ix)
L = int(json.loads((ROC / "phase21_lstar.json").read_text())["L_star"])
X, M = ts.prepare(L)
labels = {h: ix[h].to_numpy() for h in g.HORIZONS}
tasks = ts.build_tasks_index(ix)
p1 = next(t for t in tasks if t["protocol"] == "P1_temporal_primary")
ev_mask = next(m2 for _, c, m2 in p1["evals"] if c == "val_2022")
hor = g.HORIZONS[0]; yv = labels[hor]
vtr = g.make_view(p1["train"], yv, sp, dp, X, M)
vev = g.make_view(ev_mask, yv, sp, dp, X, M)
from sklearn.metrics import average_precision_score

graph = pd.read_csv(ROC / "station_graph.csv")
n = len(sp_index)

def build(sub):
    adj = np.zeros((n, n), np.float32)
    for a, b in zip(sub.station_id_a, sub.station_id_b):
        i, j = sp_index[int(a)], sp_index[int(b)]
        adj[i, j] = adj[j, i] = 1.0
    tilde = adj + np.eye(n, dtype=np.float32)
    deg = tilde.sum(1)
    d = np.diag(1.0 / np.sqrt(deg))
    return (torch.tensor(d @ tilde @ d, dtype=torch.float32),
            torch.tensor((adj + np.eye(n)) > 0))

G = {"full": build(graph), "river": build(graph[graph.kind == "river"]),
     "id": build(graph.iloc[0:0])}

base = {"lr": 3e-3, "hidden": 32, "dropout": 0.0, "batch_dates": 16,
        "_weight": "on", "enc_layers": 2}
print("REF temporal-only(no graph):")
r = g.fit_gnn({"agg": "gcn"}, {**base, "g_layers": 0}, X, M, row_of,
              *G["full"], vtr, vev)
print(f"  ap={average_precision_score(vev['y'], r['pred']):.5f}")
for agg in ("gcn", "gat"):
    for gname in ("full", "river", "id"):
        cfg = {**base, "g_layers": 1}
        r = g.fit_gnn({"agg": agg}, cfg, X, M, row_of, *G[gname], vtr, vev)
        ap = average_precision_score(vev["y"], r["pred"])
        print(f"agg={agg} graph={gname} g_layers=1 ap={ap:.5f} "
              f"ep={r['epochs']} fit={r['fit_s']:.0f}s npar={r['net'].n_params()}")
