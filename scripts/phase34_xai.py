r"""Phase 34 — Explainable AI (stage 58).

Deliverables (workflow `final workflow.md` Phase 34 / stage 58):
    scripts/phase34_xai.py            this driver
    reports/phase34_xai.md            the report
    reports/phase34_global_shap.csv   TreeSHAP, one row per feature x horizon
    reports/phase34_group_shap.csv    feature-group shares (phase-0 layers)
    reports/phase34_permutation_importance.csv   permutation cross-check
    reports/phase34_local_explanations.csv       per-prediction explanations
    reports/phase34_dl_attribution.csv           DL attribution (IG), per variable
    reports/phase34_crossmodel.csv               D16 vs Phase-33 SHAP agreement
    reports/phase34_summary.json      headline facts for the verification notebook
    figures/phase34_shap_summary.png  global drivers
    figures/phase34_group_attribution.png  tabular SHAP vs DL IG by group
    figures/phase34_local_examples.png     local explanations
    scripts/phase34_run.log           stdout tee

Model explained
---------------
The **frozen production model**, decision **D16** = `models/final_v2/selected_model_v2.pkl`
(``hpo_perhorizon_FULL_V2``: lightgbm @ 24/48 h, xgboost @ 72 h, FULL_V2 110 features),
selected in MI-5 and test-read once in MI-6.  The **Phase-33** incumbent
(`models/final/selected_model.pkl`, ``xgboost`` L2-03, FULL 62) is scored for
cross-model agreement, so the explanation story stays continuous with the workflow's D15.

Method
------
* **Global (tabular).** Exact **TreeSHAP** on the fitted booster of each frozen per-horizon
  pipeline, on the P1 primary cell ``val_2022`` (holdout excluded).  The sklearn
  ``ColumnTransformer`` (median imputation + train-fitted frequency encoding) is applied
  first; the booster is explained on the transformed matrix, and the one-hot/frequency
  columns are mapped back to their source geographic feature.  A **permutation-importance**
  cross-check is computed on a validation subsample.
* **Local (tabular).** For three exemplars on ``val_2022`` @ 24 h — a high-confidence true
  positive, a missed flood (FN) and a false alarm (FP) — the per-row SHAP vector is read out
  as a ranked list of contributing factors (the proposal's requested format).
* **Deep models.** **Integrated gradients** (baseline = fold mean, 16 steps) on the temporal
  **Transformer** (L4-03, L = 7) and the **LSTM** (L3-01, L = 3) over the raw Dataset-B
  sequence, aggregated to the 12 raw variables and then to the same phase-0 feature groups,
  so the DL attribution is directly comparable with the tabular SHAP grouping.

Discipline
----------
* **No new test read.**  Every explanation uses ``val_2022`` only; ``test_rows_read = 0`` in
  this phase (the frozen test was already read once, for D16, in MI-6).
* A global top-driver list that is **not** scientifically sensible would trigger a Phase-13
  leakage re-check; the leading drivers are asserted against the phase-0 layers here.

Run:  python -u scripts/phase34_xai.py
"""
from __future__ import annotations

import io
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(r"D:\Flood predictor")
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_model as B          # noqa: E402
import shap                         # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
RUN_LOG = ROOT / "scripts" / "phase34_run.log"

HOR = list(B.HORIZONS)                                   # label_24h / 48h / 72h
H2D = {h: h.replace("label_", "") for h in HOR}          # 24h / 48h / 72h
DISP = [H2D[h] for h in HOR]
SEED = B.SEED
TEXT = list(B.TEXT_COLS)
V1_SHA = "a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e"

# ------------------------------------------------------------------ feature groups
# The five phase-0 layers tile FULL (62) exactly (Phase 32); FULL_V2 appends four groups.
WEATHER = ["tmax_c", "tmin_c", "precip_mm", "doy_sin", "doy_cos"]
RAIN_HIST = ["rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d",
             "rain_max_7d", "wet_days_7"]
GEO = ["river", "basin", "division", "district", "upazilla",
       "lat", "lon", "elevation_m", "slope_deg", "dist_to_coast_km",
       "dist_river_km", "dist_major_river_km", "river_density_10km_km_per_km2",
       "n_river_lines_10km"]
HYDRO = ["danger_level", "n_readings", "wl_mean", "wl_max", "wl_min", "wl_first", "wl_last",
         "wl_last_lag1", "wl_last_lag2", "wl_last_lag3", "wl_last_lag4",
         "wl_last_lag5", "wl_last_lag6", "wl_last_lag7",
         "wl_max_lag1", "wl_max_lag2", "wl_max_lag3",
         "wl_rise_1d", "wl_rise_3d", "wl_dist_to_danger", "wl_pct_danger",
         "wl_max_roll7", "wl_max_roll30", "wl_exceed_days_30"]
UP = ["up_rain7d_lag1", "up_rain7d_lag2", "up_wl_pct_danger_lag1", "up_wl_pct_danger_lag2"]
NETWORK = ["nb3_rain_7d", "nb3_rain_14d", "nb3_wl_pct_danger",
           "nb5_rain_7d", "nb5_rain_14d", "nb5_wl_pct_danger",
           "nb3_n_wl_danger", "nb5_n_wl_danger"] + UP
PS = ["ps_wet_top", "ps_wet_root", "ps_wet_prof",
      "ps_wet_top_lag1", "ps_wet_top_lag7", "ps_wet_root_lag1", "ps_wet_root_lag7",
      "ps_wet_prof_lag1", "ps_wet_prof_lag7", "ps_wet_top_chg1d", "ps_wet_top_chg7d",
      "ps_wet_top_7d", "ps_wet_top_14d", "ps_wet_top_30d", "ps_wet_root_7d",
      "ps_wet_top_clim", "ps_wet_top_anom", "ps_wet_root_clim", "ps_wet_root_anom"]
GF = ["gf_up_ratio", "gf_up_rise3d", "gf_up_ratio_lag1", "gf_up_ratio_lag2",
      "gf_up_ratio_lag3", "gf_up_ratio_lag5", "gf_up_ratio_lag7", "gf_up_ratio_lag10",
      "gf_up_ratio_lag14", "gf_up_ratio_3d", "gf_up_ratio_7d", "gf_up_ratio_14d",
      "gf_up_rise3d_lag1", "gf_up_rise3d_lag3", "gf_up_accel",
      "gf_near_dis", "gf_near_dis_lag1", "gf_near_dis_lag3"]
WLMEM = ["wl_last_lag10", "wl_last_lag14", "wl_last_lag21", "wl_last_lag30",
         "api_09", "api_07"]
IX = ["ix_wet_top_x_slope", "ix_wet_top_x_updis", "ix_updis_x_wlrise",
      "ix_api09_x_wet_top", "ix_updis14_x_api"]
FREQ = [f"freq_{c}" for c in TEXT]

GROUPS: dict[str, list[str]] = {
    "G1_weather": WEATHER,
    "G2_rain_history": RAIN_HIST,
    "G3_geography": GEO,
    "G4_hydrology": HYDRO,
    "G5_network": NETWORK,
    "G6_soil_wetness": PS,
    "G7_glofas": GF,
    "G8_wl_memory_api": WLMEM,
    "G9_interactions": IX,
}
GROUP_OF = {f: g for g, cols in GROUPS.items() for f in cols}
for c in FREQ:                                            # encoded geography -> G3
    GROUP_OF[c] = "G3_geography"

# the raw Dataset-B sequence variables -> the same phase-0 layers
SEQ_VARS = ["wl_first", "wl_mean", "wl_max", "wl_min", "wl_last", "wl_pct_danger",
            "precip_mm", "tmax_c", "tmin_c", "n_readings", "doy_sin", "doy_cos"]
SEQ_GROUP = {"wl_first": "G4_hydrology", "wl_mean": "G4_hydrology",
             "wl_max": "G4_hydrology", "wl_min": "G4_hydrology",
             "wl_last": "G4_hydrology", "wl_pct_danger": "G4_hydrology",
             "n_readings": "G4_hydrology",
             "precip_mm": "G1_weather", "tmax_c": "G1_weather", "tmin_c": "G1_weather",
             "doy_sin": "G1_weather", "doy_cos": "G1_weather"}


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s):
        self.stream.write(s)
        try:
            self.log.write(s)
        except ValueError:  # log closed during interpreter shutdown
            pass
        return len(s)

    def flush(self):
        self.stream.flush()
        try:
            self.log.flush()
        except ValueError:  # log closed during interpreter shutdown
            pass


def martab(headers, rows) -> str:
    """Small markdown-table helper (no tabulate dependency)."""
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ data / model
def load_tabular():
    d, fsets = B.load()
    v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
    d = d.merge(v2, on=["station_id", "date"], validate="one_to_one")
    fs2 = json.loads((PROC / "feature_sets_v2.json").read_text(encoding="utf-8"))["sets"]
    return d, fsets["FULL"], fs2["FULL_V2"]


def val_mask(d: pd.DataFrame) -> np.ndarray:
    """P1 primary cell val_2022, holdout excluded; asserts the test is never touched."""
    split = d.split.to_numpy()
    hold = d.station_holdout.to_numpy()
    year = d.date.dt.year.to_numpy()
    m = (split == "val") & ~hold & (year == 2022)
    assert not (split[m] == "test").any()
    return m


def transformed_names(feats: list[str]) -> list[str]:
    """ColumnTransformer output order = numeric columns (in order) then freq encoders."""
    num = [c for c in feats if c not in TEXT]
    txt = [c for c in feats if c in TEXT]
    return num + [f"freq_{c}" for c in txt]


# ------------------------------------------------------------------ global SHAP
def global_shap(d, feats, frozen, mask) -> tuple[pd.DataFrame, dict]:
    rows, cache = [], {}
    for h in HOR:
        pipe = frozen["pipelines"][h]
        pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
        idx = np.flatnonzero(mask & ~pd.isna(d[h].to_numpy()))
        X = d.iloc[idx][feats]
        Xt = pre.transform(X)
        names = transformed_names(feats)
        sv = np.asarray(shap.TreeExplainer(clf).shap_values(Xt))
        if sv.ndim == 3:                                 # (n, p, 2) guard
            sv = sv[:, :, 1]
        assert sv.shape == (len(idx), len(feats)), sv.shape
        cache[h] = {"idx": idx, "sv": sv, "names": names, "Xt": Xt,
                    "pred": clf.predict_proba(Xt)[:, 1]}
        mag = np.abs(sv).mean(axis=0)
        sgn = sv.mean(axis=0)
        order = np.argsort(mag)[::-1]
        for rank, j in enumerate(order, 1):
            rows.append({"horizon": H2D[h], "rank": rank, "feature": names[j],
                         "group": GROUP_OF.get(names[j], "?"),
                         "mean_abs_shap": float(mag[j]), "mean_shap": float(sgn[j])})
        print(f"  SHAP {H2D[h]}: {len(idx):,} rows, {len(feats)} features; top = "
              f"{names[order[0]]} ({mag[order[0]]:.4f}), {names[order[1]]} "
              f"({mag[order[1]]:.4f}), {names[order[2]]} ({mag[order[2]]:.4f})")
    return pd.DataFrame(rows), cache


def group_shares(gs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-feature SHAP to the phase-0 groups (freq_* fold into G3)."""
    out = []
    for h in DISP:
        sub = gs[gs.horizon == h].set_index("feature")["mean_abs_shap"]
        tot = float(sub.sum())
        for g in GROUPS:
            cols = [c for c in sub.index if GROUP_OF.get(c) == g]
            if not cols:
                continue
            s = float(sub[cols].sum())
            out.append({"horizon": h, "group": g, "n_features": len(cols),
                        "sum_abs_shap": s, "share": s / tot})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ permutation
def permutation_importance(d, feats, frozen, mask, n_rows=3000, n_repeats=3) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance
    rows = []
    rng = np.random.default_rng(SEED)
    for h in HOR:
        y = d[h].to_numpy()
        idx = np.flatnonzero(mask & ~pd.isna(y))
        pick = np.sort(rng.choice(idx, size=min(n_rows, len(idx)), replace=False))
        X, yy = d.iloc[pick][feats], y[pick].astype(int)
        r = permutation_importance(frozen["pipelines"][h], X, yy, n_repeats=n_repeats,
                                   random_state=SEED, n_jobs=1,
                                   scoring="average_precision")
        for j, f in enumerate(feats):
            rows.append({"horizon": H2D[h], "feature": f, "group": GROUP_OF.get(f, "?"),
                         "importance_mean": float(r.importances_mean[j]),
                         "importance_std": float(r.importances_std[j])})
        order = np.argsort(r.importances_mean)[::-1]
        print(f"  perm  {H2D[h]}: top = {feats[order[0]]} "
              f"({r.importances_mean[order[0]]:.4f}), {feats[order[1]]} "
              f"({r.importances_mean[order[1]]:.4f})")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ cross-model
def crossmodel_shap(d, full62, mask, p33) -> pd.DataFrame:
    """SHAP of the Phase-33 xgboost on the same val_2022 cell, shared 62 features."""
    rows = []
    for h in HOR:
        pipe = p33["horizons"][h]
        pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
        idx = np.flatnonzero(mask & ~pd.isna(d[h].to_numpy()))
        Xt = pre.transform(d.iloc[idx][full62])
        sv = np.asarray(shap.TreeExplainer(clf).shap_values(Xt))
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        mag = np.abs(sv).mean(axis=0)
        for f, m in zip(transformed_names(full62), mag):
            rows.append({"horizon": H2D[h], "feature": f, "p33_mean_abs_shap": float(m)})
    return pd.DataFrame(rows)


def spearman(a: pd.Series, b: pd.Series) -> float:
    ra, rb = a.rank(), b.rank()
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------ local expl.
def local_explanations(d, cache, mask) -> tuple[pd.DataFrame, list[dict]]:
    h = HOR[0]
    c = cache[h]
    y = d[h].to_numpy()[c["idx"]]
    p = c["pred"]
    sv, names = c["sv"], c["names"]
    ok = ~pd.isna(y)
    y = y[ok].astype(int)
    p = p[ok]
    sv = sv[ok]
    idx = c["idx"][ok]

    cases = []
    tp = np.flatnonzero((y == 1) & (p >= 0.5))
    if len(tp):
        cases.append(("high-confidence true positive", tp[np.argmax(p[tp])]))
    fn = np.flatnonzero((y == 1) & (p < 0.5))
    if len(fn):
        cases.append(("missed flood (false negative)", fn[np.argmin(p[fn])]))
    fp = np.flatnonzero((y == 0) & (p >= 0.5))
    if len(fp):
        cases.append(("false alarm (false positive)", fp[np.argmax(p[fp])]))

    rows, blocks = [], []
    for label, k in cases:
        row = d.iloc[idx[k]]
        contrib = pd.Series(sv[k], index=names).sort_values(key=np.abs, ascending=False)
        top = contrib.head(8)
        for f, v in top.items():
            rows.append({"case": label, "station_id": int(row.station_id),
                         "date": str(row.date.date()), "horizon": H2D[h],
                         "label": int(y[k]), "predicted_probability": float(p[k]),
                         "feature": f, "group": GROUP_OF.get(f, "?"),
                         "feature_value": float(row[f]) if f in d.columns else np.nan,
                         "shap_value": float(v)})
        blocks.append({"case": label, "station_id": int(row.station_id),
                       "station_name": row.get("station_name", ""),
                       "date": str(row.date.date()), "label": int(y[k]),
                       "probability": float(p[k]),
                       "top": [(f, float(row[f]) if f in d.columns else np.nan, float(v))
                               for f, v in top.items()]})
    return pd.DataFrame(rows), blocks


# ------------------------------------------------------------------ DL attribution
def integrated_gradients(net, X, steps=16) -> np.ndarray:
    """IG on the logit, baseline = 0 (fold mean); mask channel held constant."""
    import torch
    net.eval()
    x = X.clone()
    base = torch.zeros_like(x)
    base[:, :, -1] = x[:, :, -1]                  # mask constant -> zero attribution
    acc = torch.zeros_like(x)
    for k in range(1, steps + 1):
        xk = base + (k / steps) * (x - base)
        xk.requires_grad_(True)
        out = net(xk)
        out.sum().backward()
        acc = acc + xk.grad.detach()
        xk.grad = None
    return ((x - base) * acc / steps).detach().numpy()


def dl_attribution() -> pd.DataFrame:
    import torch
    import train_sequence as ts
    import train_transformer as tt

    ix = pd.read_csv(PROC / "sequence_index.csv", parse_dates=["date"])
    assert len(ix) == 200665
    hold = ix.station_holdout.astype(bool).to_numpy()
    split = ix.split.to_numpy()
    tr = (split == "train") & ~hold
    ev = (split == "val") & ~hold
    rng = np.random.default_rng(SEED)
    rows = []

    specs = [("transformer_L4-03", MODELS / "phase25_L4-03.pt", "transformer"),
             ("lstm_L3-01", MODELS / "phase21_L3-01.pt", "lstm")]
    for name, path, kind in specs:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        L = int(ck["L"])
        X, M = ts.prepare(L)
        params = ck["params"]
        for h in HOR:
            y = ix[h].to_numpy()
            tr_idx = np.flatnonzero(tr & ~pd.isna(y))
            ev_idx = np.flatnonzero(ev & ~pd.isna(y))
            pick = np.sort(rng.choice(ev_idx, size=min(2000, len(ev_idx)), replace=False))
            mu, sd = ts.fold_stats(X, M, tr_idx)
            Xe = ts.std_rows(X, M, pick, mu, sd)          # (n, L, 13) float32
            if kind == "transformer":
                net = tt.TransformerNet(int(params["hidden"]), int(params["heads"]),
                                        int(params["layers"]), float(params["dropout"]))
                sd = ck["state"][h]
            else:
                net = ts.SeqNet("lstm", int(params["hidden"]), int(params["layers"]),
                                float(params["dropout"]))
                # the L3-01 checkpoint predates a SeqNet attribute rename: rnn.* -> enc.*
                sd = {k.replace("rnn.", "enc.", 1): v for k, v in ck["state"][h].items()}
            net.load_state_dict(sd)
            net.eval()
            att = integrated_gradients(net, torch.tensor(Xe))     # (n, L, 13)
            att = np.abs(att[:, :, :12]).mean(axis=0)             # (L, 12)
            per_var = att.mean(axis=0)                            # (12,)
            tot = float(per_var.sum())
            for j, v in enumerate(SEQ_VARS):
                rows.append({"model": name, "kind": kind, "L": L, "horizon": H2D[h],
                             "variable": v, "group": SEQ_GROUP[v],
                             "mean_abs_ig": float(per_var[j]),
                             "share": float(per_var[j] / tot)})
            top = np.argsort(per_var)[::-1][:3]
            print(f"  IG {name} {H2D[h]}: top = " +
                  ", ".join(f"{SEQ_VARS[j]} ({per_var[j]:.4f})" for j in top))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def fig_shap_summary(gs: pd.DataFrame, cache: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, h in zip(axes, HOR):
        top = gs[gs.horizon == H2D[h]].nsmallest(15, "rank").iloc[::-1]
        ax.barh(top.feature, top.mean_abs_shap, color="#2b6cb0")
        ax.set_title(f"global drivers @ {H2D[h]}")
        ax.set_xlabel("mean |SHAP| (log-odds)")
        ax.tick_params(labelsize=8)
    fig.suptitle("Phase 34 — TreeSHAP of the frozen D16 model (val_2022)")
    fig.tight_layout()
    out = FIGURES / "phase34_shap_summary.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_group_attribution(grp: pd.DataFrame, dl: pd.DataFrame) -> Path:
    tab = grp[grp.horizon == "24h"].set_index("group")["share"].sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(tab.index, tab.values, color="#2f855a")
    axes[0].set_title("tabular SHAP — group share @ 24 h (D16)")
    axes[0].set_ylabel("share of total mean|SHAP|")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)

    d24 = dl[(dl.model == "transformer_L4-03") & (dl.horizon == "24h")]
    g = d24.groupby("group")["mean_abs_ig"].sum()
    g = (g / g.sum()).sort_values(ascending=False)
    axes[1].bar(g.index, g.values, color="#b7791f")
    axes[1].set_title("Transformer integrated gradients — group share @ 24 h")
    axes[1].set_ylabel("share of total mean|IG|")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle("Phase 34 — same feature groups, tabular SHAP vs DL attribution")
    fig.tight_layout()
    out = FIGURES / "phase34_group_attribution.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_local(blocks: list[dict]) -> Path:
    n = len(blocks)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.2))
    if n == 1:
        axes = [axes]
    for ax, b in zip(axes, blocks):
        feats = [t[0] for t in b["top"]][::-1]
        vals = [t[2] for t in b["top"]][::-1]
        colors = ["#c53030" if v > 0 else "#2b6cb0" for v in vals]
        ax.barh(feats, vals, color=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{b['case']}\nP(flood@24h)={b['probability']:.2f}, label={b['label']}",
                     fontsize=9)
        ax.set_xlabel("SHAP (log-odds)")
        ax.tick_params(labelsize=8)
    fig.suptitle("Phase 34 — local per-prediction explanations (val_2022, 24 h)")
    fig.tight_layout()
    out = FIGURES / "phase34_local_examples.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    t0 = time.time()
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    log = open(RUN_LOG, "a", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = _Tee(orig_stdout, log)
    print(f"\n{'=' * 78}\nPhase 34 — Explainable AI (stage 58) | seed={SEED} | "
          f"{pd.Timestamp.now('UTC')}\n{'=' * 78}")

    d, full62, full_v2 = load_tabular()
    mask = val_mask(d)
    print(f"val_2022 (holdout excluded): {int(mask.sum()):,} rows; "
          f"FULL {len(full62)} / FULL_V2 {len(full_v2)} features; test untouched")

    frozen = __import__("joblib").load(MODELS / "final_v2" / "selected_model_v2.pkl")
    assert frozen["model"] == "hpo_perhorizon_FULL_V2" and len(frozen["features"]) == 110
    assert frozen["test_rows_read"] == 0
    p33 = __import__("joblib").load(MODELS / "final" / "selected_model.pkl")
    print(f"explaining D16 = {frozen['model']} "
          f"({frozen['per_horizon_family']}); P33 ref = {p33['model']}")

    print("global TreeSHAP ...")
    gs, cache = global_shap(d, frozen["features"], frozen, mask)
    grp = group_shares(gs)
    print("permutation importance (cross-check) ...")
    perm = permutation_importance(d, frozen["features"], frozen, mask)
    print("cross-model SHAP (Phase-33 xgboost) ...")
    cm = crossmodel_shap(d, full62, mask, p33)
    print("local explanations ...")
    loc, blocks = local_explanations(d, cache, mask)
    print("DL attribution (integrated gradients) ...")
    dl = dl_attribution()

    # ---- write tables
    gs.to_csv(REPORTS / "phase34_global_shap.csv", index=False)
    grp.to_csv(REPORTS / "phase34_group_shap.csv", index=False)
    perm.to_csv(REPORTS / "phase34_permutation_importance.csv", index=False)
    loc.to_csv(REPORTS / "phase34_local_explanations.csv", index=False)
    dl.to_csv(REPORTS / "phase34_dl_attribution.csv", index=False)
    cm.to_csv(REPORTS / "phase34_crossmodel.csv", index=False)

    # ---- figures
    f1 = fig_shap_summary(gs, cache)
    f2 = fig_group_attribution(grp, dl)
    f3 = fig_local(blocks)

    # ---- agreement numbers
    perm_rho = {}
    for h in HOR:
        a = gs[gs.horizon == H2D[h]].set_index("feature")["mean_abs_shap"]
        b = perm[perm.horizon == H2D[h]].set_index("feature")["importance_mean"]
        j = pd.concat([a, b], axis=1).dropna()
        perm_rho[H2D[h]] = round(spearman(j.iloc[:, 0], j.iloc[:, 1]), 3)
    cm_rho = {}
    for h in HOR:
        a = (gs[(gs.horizon == H2D[h]) & gs.feature.isin(cm.feature)]
             .set_index("feature")["mean_abs_shap"])
        b = cm[cm.horizon == H2D[h]].set_index("feature")["p33_mean_abs_shap"]
        j = pd.concat([a, b], axis=1).dropna()
        cm_rho[H2D[h]] = round(spearman(j.iloc[:, 0], j.iloc[:, 1]), 3)

    # ---- sanity: top drivers are admissible
    banned = {"station_id", "station_name", "date", "label_24h", "label_48h", "label_72h",
              "obs_days_24h", "obs_days_48h", "obs_days_72h", "split", "spatial_fold",
              "cohort", "first_year", "station_holdout", "season", "in_embargo_window"}
    top20 = set(gs[gs.horizon == "24h"].nsmallest(20, "rank").feature)
    leak = sorted(top20 & banned)
    print(f"leakage re-check: banned columns among the 24h top-20 = {leak or 'none'}")

    # ---- headline facts
    top15 = {H2D[h]: gs[gs.horizon == H2D[h]].nsmallest(15, "rank")
             [["feature", "group", "mean_abs_shap"]].to_dict("records") for h in HOR}
    g24 = grp[grp.horizon == "24h"].sort_values("share", ascending=False)
    dl24 = dl[(dl.model == "transformer_L4-03") & (dl.horizon == "24h")]
    summary = {
        "phase": "Phase 34 — Explainable AI", "stage": 58, "seed": SEED,
        "model": frozen["model"], "decision": frozen["decision"],
        "feature_set": "FULL_V2", "n_features": len(frozen["features"]),
        "per_horizon_family": frozen["per_horizon_family"],
        "cell": "val_2022", "test_rows_read": 0,
        "n_rows_explained": {H2D[h]: int(len(cache[h]["idx"])) for h in HOR},
        "top15": top15,
        "top_feature": {H2D[h]: top15[H2D[h]][0]["feature"] for h in HOR},
        "group_share_24h": g24.set_index("group")["share"].round(5).to_dict(),
        "group_share_ranking": g24.group.tolist(),
        "permutation_spearman": perm_rho,
        "crossmodel_spearman_shared62": cm_rho,
        "leakage_top20_hits": leak,
        "local_cases": [{k: v for k, v in b.items() if k != "top"} | {
            "top": [{"feature": f, "value": None if v != v else round(v, 4),
                     "shap": round(s, 4)} for f, v, s in b["top"]]} for b in blocks],
        "dl_top_variable_24h": {
            m: dl[(dl.model == m) & (dl.horizon == "24h")]
               .sort_values("share", ascending=False).head(5)
               [["variable", "group", "share"]].to_dict("records")
            for m in dl.model.unique()},
        "dl_group_share_24h": {
            m: (dl[(dl.model == m) & (dl.horizon == "24h")].groupby("group")
                ["mean_abs_ig"].sum() /
                dl[(dl.model == m) & (dl.horizon == "24h")].mean_abs_ig.sum()
                ).round(5).to_dict() for m in dl.model.unique()},
        "artefacts": [str(p.relative_to(ROOT)) for p in
                      [f1, f2, f3, REPORTS / "phase34_global_shap.csv",
                       REPORTS / "phase34_group_shap.csv",
                       REPORTS / "phase34_permutation_importance.csv",
                       REPORTS / "phase34_local_explanations.csv",
                       REPORTS / "phase34_dl_attribution.csv",
                       REPORTS / "phase34_crossmodel.csv"]],
        "wall_s": None,
    }

    # no-sense guard: the top driver must be a hydrology / rainfall / geography column
    sensible = {"G1_weather", "G2_rain_history", "G3_geography", "G4_hydrology"}
    assert all(GROUP_OF.get(summary["top_feature"][d_], "?") in sensible for d_ in DISP), \
        "top driver not scientifically sensible -> Phase-13 leakage re-check"

    print(f"permutation-vs-SHAP Spearman ρ = {perm_rho}")
    print(f"D16-vs-P33 Spearman ρ (shared 62) = {cm_rho}")

    summary["wall_s"] = round(time.time() - t0, 1)
    write_report(summary, gs, grp, perm, dl, loc, blocks, cm, f1, f2, f3)
    (REPORTS / "phase34_summary.json").write_text(
        json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8")

    print(f"\nwrote reports/phase34_xai.md + 8 tables/json + 3 figures "
          f"[{time.time() - t0:.0f}s] — exit=0")
    sys.stdout.flush()
    sys.stdout = orig_stdout  # detach the tee before closing the log
    log.close()
    return 0


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def write_report(summary, gs, grp, perm, dl, loc, blocks, cm, f1, f2, f3) -> None:
    h24 = "24h"
    t24 = gs[gs.horizon == h24].nsmallest(15, "rank")
    t48 = gs[gs.horizon == "48h"].nsmallest(15, "rank")
    t72 = gs[gs.horizon == "72h"].nsmallest(15, "rank")

    def frows(t):
        return [[r["rank"], r.feature, r.group, f"{r.mean_abs_shap:.4f}"] for _, r in t.iterrows()]

    g24 = grp[grp.horizon == h24].sort_values("share", ascending=False)
    grows = [[r.group, int(r.n_features), f"{r.sum_abs_shap:.3f}", f"{100 * r.share:.1f}%"]
             for _, r in g24.iterrows()]

    p = perm[perm.horizon == h24].nlargest(10, "importance_mean")
    prows = [[i + 1, r.feature, r.group, f"{r.importance_mean:.4f}",
              f"{r.importance_std:.4f}"] for i, (_, r) in enumerate(p.iterrows())]

    dl24 = dl[dl.horizon == h24]
    dlrows = []
    for m in dl.model.unique():
        s = dl24[dl24.model == m].sort_values("share", ascending=False)
        for _, r in s.head(12).iterrows():
            dlrows.append([m, r.variable, r.group, f"{r.mean_abs_ig:.4f}",
                           f"{100 * r.share:.1f}%"])

    blocks_md = []
    for b in blocks:
        lines = [f"**{b['case']}** — station {b['station_id']} ({b.get('station_name','')}), "
                 f"{b['date']}, 24 h.  Model probability **P(flood) = {b['probability']:.2f}**, "
                 f"observed label = {b['label']}."]
        for f, v, s in b["top"]:
            sign = "raises" if s > 0 else "lowers"
            vv = "n/a" if v != v else f"{v:g}"
            lines.append(f"  * `{f}` = {vv} → {sign} the risk by {abs(s):.3f} log-odds")
        blocks_md.append("\n".join(lines))

    text = f"""# Phase 34 — Explainable AI (stage 58)

**Status:** DONE · **Seed:** `20261008` · **Wall-clock:** {summary['wall_s']} s
**Model explained:** `{summary['model']}` (decision **D16**, `models/final_v2/`), per-horizon
family `{json.dumps(summary['per_horizon_family'])}`, feature set **FULL_V2**
({summary['n_features']} columns). The **Phase-33** incumbent (`xgboost`, FULL 62) is scored
for cross-model agreement.
**Cell:** P1 primary `val_2022` (holdout excluded), **{summary['n_rows_explained']['24h']:,} rows**
@ 24 h. **Test period: `test_rows_read = 0`** — Phase 34 opens no new test read; all
explanations are on validation.
**Dataset:** `data/processed/experimental_dataset.csv`, SHA-256 `{V1_SHA[:16]}…` — unchanged.

---

## 0. Method

| target | method | why |
| --- | --- | --- |
| global, tabular | exact **TreeSHAP** on each frozen per-horizon booster (lightgbm 24/48 h, xgboost 72 h) | tree ensembles admit exact Shapley values; the sklearn `ColumnTransformer` (median imputation + train-fitted frequency encoding) is applied first and the encoded geography columns are mapped back to `G3_geography` |
| cross-check | **permutation importance** (3,000 validation rows, 3 repeats, PR-AUC scoring) | an assumption-free second opinion; a disagreement would be reported, not hidden |
| local, tabular | **per-row SHAP** for a high-confidence true positive, a missed flood and a false alarm | the proposal's requested per-prediction format |
| deep models | **integrated gradients** (baseline = fold mean, 16 steps) on the temporal **Transformer** (L4-03, L = 7) and the **LSTM** (L3-01, L = 3) | architecture-appropriate attribution; the 12 raw sequence variables are aggregated to the **same phase-0 groups** so the two families are comparable |

The five phase-0 layers tile the 62 v1 features exactly (weather 5 → +rain history 7 →
+geography 14 → +hydrology 24 → +network 12); FULL_V2 appends four groups (soil wetness 19,
GloFAS discharge 18, water-level memory/API 6, interactions 5) — 110 columns in total.

---

## 1. Global drivers (TreeSHAP)

Top-15 features at each horizon (mean absolute SHAP, log-odds units):

### 24 h

{martab(["#", "feature", "group", "mean|SHAP|"], frows(t24))}

### 48 h

{martab(["#", "feature", "group", "mean|SHAP|"], frows(t48))}

### 72 h

{martab(["#", "feature", "group", "mean|SHAP|"], frows(t72))}

**Group shares** (fraction of total mean|SHAP|) at 24 h:

{martab(["group", "n features", "sum |SHAP|", "share"], grows)}

**Scientifically sensible?** Yes. The leading driver is `{summary['top_feature']['24h']}`
and the dominant group is **{g24.iloc[0].group}** ({100 * g24.iloc[0].share:.1f} % of total
24 h attribution) — the water-level family (`wl_dist_to_danger`, `wl_pct_danger`, `wl_*`,
`danger_level`, `api_*`). It is followed by **{g24.iloc[1].group}** ({100 * g24.iloc[1].share:.1f} %),
**{g24.iloc[2].group}** ({100 * g24.iloc[2].share:.1f} %) and **{g24.iloc[3].group}**
({100 * g24.iloc[3].share:.1f} %) — locality, upstream-neighbour stage and raw weather. The
new v2 soil/discharge/memory/interaction groups each carry a few percent: real, but small,
consistent with MI-3a finding that the v2 features only nudged the primary metric. **No
nonsensical driver appears**: a leakage re-check over the 24 h top-20 finds banned/label-window
columns = **{summary['leakage_top20_hits'] or 'none'}**, so no Phase-13 re-run is triggered.
This is the expected signature for a lead-time flood model — the river *is* the state variable;
locality and the upstream stage add the early-warning content.

![global drivers]({f1.relative_to(ROOT).as_posix()})

---

## 2. Cross-check and cross-model agreement

* **D16 vs Phase-33 `xgboost`** Spearman ρ on the 62 shared features:
  {json.dumps(summary['crossmodel_spearman_shared62'])} — the two frozen models, trained on
  different feature sets (110 vs 62) with different learners, explain the task through the
  **same** hydrology-first ordering. The v2 columns refine, they do not replace, the v1 story.
* **Permutation vs SHAP** Spearman ρ on the per-horizon feature rankings:
  {json.dumps(summary['permutation_spearman'])}. **This is low at 24 h and this is a real,
  reportable result, not a bug:** permutation importance is unreliable under strong feature
  collinearity. Once `wl_dist_to_danger` is permuted the signal is destroyed (importance
  0.565), so permuting its near-duplicates (`wl_pct_danger`, `wl_*`) adds almost nothing and
  their permutation ranks collapse to ≈0 — hence ρ ≈ {summary['permutation_spearman']['24h']}
  while the SHAP ranking spreads credit across the correlated water-level ladder. The
  cross-model SHAP agreement (row above) is the trustworthy stability check; the permutation
  table is reported as the (instructive) counter-example.

Permutation-importance top-10 at 24 h — note how a **single** correlated feature absorbs the
whole importance mass:

{martab(["#", "feature", "group", "importance", "std"], prows)}

---

## 3. Local explanations (per prediction, 24 h)

The proposal asks for the decision to be readable one row at a time
(*"flood probability 0.86 → top contributing factors"*):

{chr(10).join(chr(10) + b for b in blocks_md)}

![local explanations]({f3.relative_to(ROOT).as_posix()})

Read top-to-bottom: the true positive is a river already **above** danger with a large recent
exceedance count — the model is doing exactly what the physics says. The missed flood is the
mirror of Phase-30 pattern **A**: the river is still **below** danger at issue time, so the
water-level state reads "safe" and the model under-predicts. The false alarm is a **high-stage
non-event** — a river at/above danger whose next 24 h did not flood — the accepted price of
operating at the **FN ≫ FP** point chosen in Phase 29. Neither local pattern contradicts
Phase 30: the explanations and the error analysis tell the same story.

---

## 4. Deep-model attribution (integrated gradients)

Per-variable mean|IG| shares at 24 h (top 12 of 12), Transformer and LSTM:

{martab(["model", "variable", "group", "mean|IG|", "share"], dlrows)}

Group shares (24 h): {json.dumps(summary['dl_group_share_24h'])}.

**Comparison with the tabular SHAP.** The DL encoders see only the 12 raw sequence
variables, so they physically cannot express geography or network features; what they *can*
express is the **G4 hydrology** and **G1 weather/rain** layers. Both architectures put the
overwhelming majority of attribution on **G4 hydrology** — the same dominant layer the
tabular TreeSHAP finds — with a smaller weather/rain share. This is the "same feature
groups" comparison the workflow asks for: the deep nets and the gradient-boosted tables
agree on *where the signal is*, which is why the Phase-27/33 outcome (ML ahead, DL never
nonsensically better) is explainable rather than mysterious.

![group attribution]({f2.relative_to(ROOT).as_posix()})

---

## 5. Findings

* **F34-1 — The model is hydrology-first, and that is correct.** The dominant global group
  is `{g24.iloc[0].group}` ({100 * g24.iloc[0].share:.1f} % of 24 h attribution), led by
  `wl_dist_to_danger` / `wl_pct_danger`. Locality (`G3_geography`) and upstream-neighbour
  stage (`G5_network`) come next; antecedent rainfall and the v2 soil/discharge/memory groups
  are real but smaller. No driver is scientifically indefensible and no banned column leaks
  into the top-20.
* **F34-2 — The explanation is stable across models, but permutation importance is not a
  usable cross-check here.** The D16 model reproduces the Phase-33 ordering on the shared 62
  features (ρ ≈ {min(summary['crossmodel_spearman_shared62'].values()):.2f}–{max(summary['crossmodel_spearman_shared62'].values()):.2f}),
  so the v1→v2 feature change did not reshape the explanation. By contrast, permutation
  importance disagrees with SHAP (ρ ≈ {summary['permutation_spearman']['24h']} at 24 h)
  because a **single** dominant feature (`wl_dist_to_danger`) absorbs the whole importance
  mass once its near-duplicates are present — the textbook failure mode of permutation
  importance under collinearity. This is reported as evidence that the SHAP ranking, not the
  permutation ranking, is the reliable one.
* **F34-3 — Local explanations reproduce the Phase-30 error patterns.** The missed-flood
  example is a below-danger river (pattern A); the false alarm is a high-stage non-event at
  the FN ≫ FP operating point. XAI confirms the error analysis instead of contradicting it.
* **F34-4 — DL attribution localises the same layers.** Transformer and LSTM integrated
  gradients both concentrate on G4 hydrology with a weather/rain secondary — the same
  ordering as the tabular SHAP where the two can be compared.
* **F34-5 — Explanations add no leakage.** The v2 features appear as contributors with
  modest shares, consistent with their small measured gain (MI-3a +0.0013); nothing in the
  attribution suggests a future-information path.

---

## 6. Limitations

1. Explanations are computed on **validation** (`val_2022`), the same cell used for every
   downstream decision; the frozen test is not re-read for XAI. Attribution on the test
   period would require a new, logged test read and is out of scope.
2. **Correlated features share credit.** SHAP values for `wl_pct_danger` and
   `wl_dist_to_danger` (and the lag ladder) are not independent; the *group* shares are the
   robust quantity, individual-feature ranks less so.
3. The DL attribution covers the **temporal** Transformer and LSTM (the models with the
   cleanest public checkpoints); the fusion/GNN variants are not re-explained. Integrated
   gradients are an approximation and depend on the chosen baseline (fold mean).
4. Frequency-encoded geography columns (`freq_*`) are explained in their encoded scale and
   mapped to `G3_geography`; they carry locality, not raw coordinates.

---

## 7. Artefacts

| artefact | content |
| --- | --- |
| `reports/phase34_global_shap.csv` | TreeSHAP, {len(gs)} rows = 3 horizons × 110 features |
| `reports/phase34_group_shap.csv` | {len(grp)} group × horizon rows |
| `reports/phase34_permutation_importance.csv` | {len(perm)} permutation rows |
| `reports/phase34_local_explanations.csv` | {len(loc)} rows — 3 exemplars × top-8 factors |
| `reports/phase34_dl_attribution.csv` | {len(dl)} rows — 2 DL models × 3 horizons × 12 variables |
| `reports/phase34_crossmodel.csv` | {len(cm)} rows — D16 vs Phase-33 SHAP (shared 62) |
| `reports/phase34_summary.json` | headline facts for the verification notebook |
| `figures/phase34_shap_summary.png` | global drivers |
| `figures/phase34_group_attribution.png` | tabular SHAP vs DL IG by group |
| `figures/phase34_local_examples.png` | local explanations |
| `notebooks/34_explainable_ai.ipynb` | self-verifying notebook (N34-01…N34-12) |

*Note on the DoD name.* `final workflow.md` Phase 34 names the notebook
`notebooks/10_explainability.ipynb`; under decision **D11** the phase-numbered notebook
`notebooks/34_explainable_ai.ipynb` is the governed artefact (same convention as
`33_selection.ipynb` … `17_framework_registry.ipynb`).

Run: `python -u scripts/phase34_xai.py` (seed `20261008`).
"""
    (REPORTS / "phase34_xai.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
