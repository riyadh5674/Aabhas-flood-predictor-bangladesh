r"""Phase 26 (stage 41 + 59/60/61) — multi-horizon strategy.

Every surviving L1-L4 model is assembled into one model x horizon matrix and the
lead-time question is answered: how does difficulty grow with lead time, and does
it grow *differently* for engineered-feature models (L1/L2) than for sequence/DL
models (L3/L4)?  This phase integrates existing, frozen dev-cell results only — it
trains nothing and never reads the frozen test (phase 27 owns the full CIs/cost
comparison; this phase owns the horizon axis).

Outputs:
    reports/phase26_matrix.csv    consolidated dev grid, all 24 models (1728 rows)
    reports/phase26_summary.csv   per model x protocol: 24/48/72h PR-AUC + mean + drop
    reports/phase26_multihorizon.md (written separately, tables from here)

Run:
    python -u scripts/phase26_matrix.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Flood predictor")
REPORTS = ROOT / "reports"
PROC = ROOT / "data" / "processed"
TBL = PROC / "experimental_dataset.csv"
TBL_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"
SEED = 20261008
HOR = ["label_24h", "label_48h", "label_72h"]

# source metrics files: (file, registry level)
SOURCES = [
    ("phase16_baseline_metrics.csv", "L1"),
    ("phase20_metrics.csv", "L2"),
    ("phase21_metrics.csv", "L3"),
    ("phase22_metrics.csv", "L3"),
    ("phase23_metrics.csv", "L3"),
    ("phase24_metrics.csv", "L4"),
    ("phase25_metrics.csv", "L4"),
]

# family label per level (the D14 taxonomy)
LEVEL_FAMILY = {
    "L1": "baseline/statistical (engineered)",
    "L2": "classical ML (engineered)",
    "L3": "deep temporal (sequence)",
    "L4": "spatiotemporal/attention (sequence)",
}
# levels that count as "engineered-feature" vs "sequence" for the interaction
ENGINEERED = {"L1", "L2"}
SEQUENCE = {"L3", "L4"}
# trivial reference baselines (not trained on the feature matrix) - excluded from group means
TRIVIAL = {"climatology", "persistence"}

# model -> registry id
REG_ID = {
    "climatology": "L1-01", "persistence": "L1-02", "logreg_linear": "L1-03",
    "logreg_linear_no_up": "L1-04", "logreg_selected": "L1-05", "tree_depth4": "L1-06",
    "random_forest": "L2-01", "gradient_boosting": "L2-02", "xgboost": "L2-03",
    "lightgbm": "L2-04", "xgboost_selected": "L2-05",
    "lstm_L3": "L3-01", "lstm_L7": "L3-02", "lstm_L14": "L3-03", "lstm_L30": "L3-04",
    "gru_bestL": "L3-05", "tcn_bestL": "L3-06", "fusion_L3": "L3-07",
    "static_only": "L3-08", "hybrid_L3": "L3-09",
    "gcn": "L4-01", "gat": "L4-02", "transformer_bestL": "L4-03", "gnn_lstm": "L4-04",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_grid() -> pd.DataFrame:
    frames = []
    for fn, lvl in SOURCES:
        d = pd.read_csv(REPORTS / fn)
        d["level"] = lvl
        d["source"] = fn
        frames.append(d)
    g = pd.concat(frames, ignore_index=True)
    g["family"] = g["level"].map(LEVEL_FAMILY)
    g["registry_id"] = g["model"].map(REG_ID)
    assert g["registry_id"].notna().all(), \
        sorted(g.loc[g.registry_id.isna(), "model"].unique())
    g["horizon_h"] = g["horizon"].str.extract(r"(\d+)").astype(int)
    return g


def cell_pivot(g: pd.DataFrame, protocol: str, cell: str | None, value="pr_auc"):
    d = g[(g.protocol == protocol) & (g.season == "all")]
    if cell is not None:
        d = d[d.cell == cell]
    else:  # mean over the protocol's cells
        d = (d.groupby(["model", "horizon_h"])[value].mean().reset_index())
    m = d.pivot_table(index="model", columns="horizon_h", values=value)
    return m.reindex(columns=[24, 48, 72])


def main() -> None:
    print("=" * 78)
    print("Phase 26 - multi-horizon matrix (stage 41 + 59/60/61)")
    print("=" * 78)
    sha = sha256(TBL)
    print(f"experimental_dataset.csv SHA-256 {sha[:12]}... "
          f"(unchanged: {sha == TBL_SHA})")
    assert sha == TBL_SHA, "experimental table changed"

    g = load_grid()
    assert not g.cell.astype(str).str.startswith("test").any(), "a test cell leaked"
    print(f"consolidated dev grid: {len(g)} rows, {g.model.nunique()} models, "
          f"levels {sorted(g.level.unique())}")
    g.to_csv(REPORTS / "phase26_matrix.csv", index=False)

    models = list(REG_ID.keys())
    lvl = {m: g.loc[g.model == m, "level"].iloc[0] for m in models}

    # ---- summary: per model x protocol, 24/48/72 + mean + drop ----
    protos = [("P1", "val_2022"), ("P2", None), ("P3", None)]
    rows = []
    for m in models:
        for proto, cell in protos:
            pv = cell_pivot(g, proto, cell)
            if m not in pv.index:
                continue
            r = pv.loc[m]
            rows.append({
                "level": lvl[m], "family": LEVEL_FAMILY[lvl[m]],
                "registry_id": REG_ID[m], "model": m, "protocol": proto,
                "cell": cell or ("mean_years" if proto == "P2" else "mean_folds"),
                "pr_auc_24h": round(float(r[24]), 5),
                "pr_auc_48h": round(float(r[48]), 5),
                "pr_auc_72h": round(float(r[72]), 5),
                "mean": round(float(r.mean()), 5),
                "drop_72_24": round(float(r[72] - r[24]), 5),
            })
    summ = pd.DataFrame(rows)
    summ.to_csv(REPORTS / "phase26_summary.csv", index=False)

    # ---- P1 val_2022 headline matrix (season=all) ----
    p1 = cell_pivot(g, "P1", "val_2022")
    p1 = p1.reindex(models)
    print("\n--- P1 val_2022 (season=all) PR-AUC ---")
    print(p1.round(4).to_string())

    # ---- lead-time difficulty: per model drop on P1 ----
    print("\n--- lead-time degradation on P1 (24h -> 72h) ---")
    drops = (p1[72] - p1[24]).sort_values()
    for m, v in drops.items():
        print(f"  {m:22s} {lvl[m]}  {p1.loc[m,24]:.4f} -> {p1.loc[m,72]:.4f}  "
              f"drop {v:+.4f}")

    # ---- interaction: engineered vs sequence (P1, P2, P3) ----
    print("\n--- interaction: engineered (L1/L2) vs sequence (L3/L4) ---")
    analysis = {}
    for proto, cell in protos:
        pv = cell_pivot(g, proto, cell)
        non_trivial = [m for m in models if m not in TRIVIAL]
        eng = [m for m in non_trivial if lvl[m] in ENGINEERED]
        seq = [m for m in non_trivial if lvl[m] in SEQUENCE]
        e = pv.loc[eng]
        s = pv.loc[seq]
        ed = float((e[72] - e[24]).mean())
        sd = float((s[72] - s[24]).mean())
        em = float(np.median(e[72] - e[24]))
        sm = float(np.median(s[72] - s[24]))
        # best model in each group by P1 mean over horizons
        best_e = e.mean(axis=1).idxmax()
        best_s = s.mean(axis=1).idxmax()
        analysis[proto] = {
            "engineered_n": len(eng), "sequence_n": len(seq),
            "engineered_24_48_72": [round(float(e[h].mean()), 5) for h in (24, 48, 72)],
            "sequence_24_48_72": [round(float(s[h].mean()), 5) for h in (24, 48, 72)],
            "engineered_drop_mean": round(ed, 5), "sequence_drop_mean": round(sd, 5),
            "interaction_drop_diff_mean": round(sd - ed, 5),
            "engineered_drop_median": round(em, 5), "sequence_drop_median": round(sm, 5),
            "interaction_drop_diff_median": round(sm - em, 5),
            "best_engineered": best_e, "best_engineered_curve":
                [round(float(e.loc[best_e, h]), 5) for h in (24, 48, 72)],
            "best_engineered_drop": round(float(e.loc[best_e, 72] - e.loc[best_e, 24]), 5),
            "best_sequence": best_s, "best_sequence_curve":
                [round(float(s.loc[best_s, h]), 5) for h in (24, 48, 72)],
            "best_sequence_drop": round(float(s.loc[best_s, 72] - s.loc[best_s, 24]), 5),
        }
        print(f"  {proto}: engineered (n={len(eng)}) drop mean {ed:+.4f} / median {em:+.4f}"
              f" | sequence (n={len(seq)}) drop mean {sd:+.4f} / median {sm:+.4f}")
        print(f"        best engineered {best_e} {analysis[proto]['best_engineered_curve']} "
              f"(drop {analysis[proto]['best_engineered_drop']:+.4f})")
        print(f"        best sequence   {best_s} {analysis[proto]['best_sequence_curve']} "
              f"(drop {analysis[proto]['best_sequence_drop']:+.4f})")

    # ---- best per horizon (all models, P1) ----
    print("\n--- best per horizon (P1 val_2022) ---")
    for h in (24, 48, 72):
        b = p1[h].sort_values(ascending=False)
        print(f"  {h}h: " + ", ".join(f"{m}={b[m]:.4f}" for m in b.index[:4]))

    (REPORTS / "phase26_interaction.json").write_text(
        json.dumps({"seed": SEED, "sha256": TBL_SHA, "models": len(models),
                    "interaction": analysis}, indent=2), encoding="utf-8")
    print(f"\nwrote phase26_matrix.csv ({len(g)} rows), phase26_summary.csv "
          f"({len(summ)} rows), phase26_interaction.json")


if __name__ == "__main__":
    main()
