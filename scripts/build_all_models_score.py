"""
build_all_models_score.py — one ranked reference of every model in the project.

Emit  project study/all models score.md

Scope (owner decision, 2026-10-17):
  * EVERY model ever trained is in the file.
  * Ranked table of the PRE-improvement models comes FIRST (Phases 0-34, i.e. everything
    that existed before the MI campaign started on 2026-10-15).
  * The MI campaign (MI-1..MI-8) follows in its own section.
  * Rank key = PR-AUC horizon-mean on the project's decision cell (val_2022, season=all),
    per the D8/D16 rule.  Cells are never pooled.

Sources are read-only; this script only writes the one markdown file.
Test stays SEALED: this document reads no test data.  Test numbers quoted in section 3 come
from the single logged MI-6 read (reports/mi6_test_metrics.csv, n_reads=1) and the Phase-33
read (reports/phase33_test_metrics.csv, n_reads=1); they are reproduced, not re-derived.

Run:  .venv\\Scripts\\python.exe scripts\\build_all_models_score.py
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
OUT = ROOT / "project study" / "all models score.md"

SEED = 20261008
DECISION_CELL = "val_2022"
HORIZONS = ["label_24h", "label_48h", "label_72h"]
HZ = {h: f"{h.split('_')[1][:-1]}h" for h in HORIZONS}

FULL_METRICS = [
    "pr_auc", "roc_auc", "brier", "logloss", "ece", "base_rate", "threshold",
    "recall", "precision", "specificity", "f1", "tp", "fp", "fn", "tn",
]


# ----------------------------------------------------------------------------- helpers
def r(x, n=5):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{round(float(x), n) + 0.0:.{n}f}"


def ri(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{int(round(x))}"


def acc(row):
    n = row.get("tp", 0) + row.get("fp", 0) + row.get("fn", 0) + row.get("tn", 0)
    if not n or pd.isna(n):
        return float("nan")
    return (row.get("tp", 0) + row.get("tn", 0)) / n


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------- 2. pre-MI era
def load_pre_mi():
    """Phase 0-34 roster (24 models) on the decision cell, full metrics."""
    d = pd.read_csv(REP / "phase27_metrics.csv")
    v = d[(d.cell == DECISION_CELL) & (d.season == "all")].copy()
    # P1 and P2 are bit-identical on val_2022 (train_<2022 == train_<=2021); keep P1.
    v = v[v.protocol == "P1"].drop_duplicates(subset=["model", "horizon"])
    v["f1"] = v.apply(
        lambda x: (2 * x.precision * x.recall / (x.precision + x.recall))
        if pd.notna(x.precision) and (x.precision + x.recall) > 0 else float("nan"),
        axis=1,
    )
    v["accuracy"] = v.apply(acc, axis=1)

    # dev-cell mean (secondary, non-selecting column)
    dev = d[d.season == "all"].groupby(["model", "horizon"]).pr_auc.mean().rename("dev_mean")

    reg = pd.read_csv(REP / "phase17_experiment_registry.csv")[
        ["registry_id", "level", "family", "model", "feature_set"]
    ].rename(columns={"model": "reg_model"})

    wide = v.pivot(index="model", columns="horizon", values="pr_auc")[HORIZONS]
    wide["mean"] = wide.mean(axis=1)
    wide = wide.sort_values("mean", ascending=False)
    wide["rank"] = range(1, len(wide) + 1)
    wide = wide.join(dev.groupby("model").mean().rename("dev7_mean"), on="model")
    wide = wide.join(v.groupby("model")[["registry_id"]].first(), on="model")
    wide = wide.join(reg.set_index("registry_id"), on="registry_id")
    return v, wide, d


# ------------------------------------------------------------------- 3. MI campaign
def load_mi():
    sc = pd.read_csv(REP / "mi5_scorecard.csv")            # D16 candidate set
    probe = pd.read_csv(REP / "mi8_probe.csv")              # feature A/B, full metrics
    ladder = pd.read_csv(REP / "mi7_ladder.csv")            # MI-7 DL ladder
    deltas = pd.read_csv(REP / "mi7_paired_deltas.csv")     # paired station bootstrap
    ens = pd.read_csv(REP / "mi3_ensemble.csv")             # rejected ensembles
    era5 = pd.read_csv(REP / "mi3_era5_probe.csv")          # rejected ERA5
    return sc, probe, ladder, deltas, ens, era5


def test_block():
    a = pd.read_csv(REP / "mi6_test_metrics.csv")
    b = pd.read_csv(REP / "phase33_test_metrics.csv")
    d = pd.concat([a, b], ignore_index=True)
    d = d[(d.season == "all") & (d.horizon.isin(HORIZONS))]
    return d


# ------------------------------------------------------------------------- narrative
def main() -> None:
    v, wide, dal = load_pre_mi()
    sc, probe, ladder, deltas, ens, era5 = load_mi()
    td = test_block()

    L = []
    A = L.append

    A("# All Models — Score & Ranking")
    A("")
    A(f"**Created:** {_dt.date.today().isoformat()} · **seed** `{SEED}`  ")
    A("**Built by:** `scripts/build_all_models_score.py` (read-only over `reports/`)  ")
    A("**One-line purpose:** a single ranked reference of *every* model this project has "
      "trained, with every metric that was recorded for it.")
    A("")
    A("| governing fact | value |")
    A("|---|---|")
    A(f"| **Rank key** | PR-AUC **horizon-mean** (24/48/72 h) on the decision cell `{DECISION_CELL}`, season `all` |")
    A("| **Why that cell** | D8 (cells never pooled) + D16 (selection is decided on `val_2022`). Spatial folds sit at 0.98–1.00 and `dev_mean` is a poor selector (MI-3c lesson), so the 7-dev mean is shown but never ranks. |")
    A("| **Frozen test** | **SEALED** for this document — 0 rows read. Section 3 *reproduces* the two already-logged single reads (Phase-33 and MI-6). |")
    A("| **Headline metric** | PR-AUC. Accuracy is reported in the tables but is **banned as a headline** (Phase-16 §10). |")
    A("| **Ceiling** | MI-4 put the empirical PR-AUC ceiling at **0.96–0.97** on high-base-rate cells; 0.99 across 0.7 %-base-rate test years is not reachable. |")
    A("")

    # ---------------------------------------------------------------- how to read
    A("## 0. How to read this file")
    A("")
    A("Three eras, in the order the project actually ran them:")
    A("")
    A("1. **§1 — Before any model-improvement work** (Phases 0–34). The full 24-model roster "
      "that existed when the MI campaign began on 2026-10-15. This is the baseline snapshot.")
    A("2. **§2 — The improvement campaign** (MI-1…MI-8). What was added, what won, what was "
      "rejected and why.")
    A("3. **§3 — The frozen test.** The one logged read each for Phase-33 and MI-6/D16.")
    A("")
    A("**Metric glossary.** `PR-AUC` average precision (headline) · `ROC-AUC` area under ROC · "
      "`Brier` mean squared probability error · `LogLoss` · `ECE` expected calibration error · "
      "`Thr` operating threshold chosen by `best_f1_threshold` · `Rec`/`Prec`/`Spec` at that "
      "threshold · `F1` · `TP/FP/FN/TN` confusion counts · `Base` base rate of the cell · "
      "`Macro` macro (per-station) PR-AUC · `CI` 95 % bootstrap CI on PR-AUC · `Acc` accuracy "
      "(reported, never headline).")
    A("")
    A("Cells are **never averaged across** for any decision; the 7-dev column is descriptive only.")
    A("")

    # =============================================================== SECTION 1
    A("---")
    A("")
    A("## 1. Before any model-improvement work (Phases 0–34)")
    A("")
    A(f"Source: `reports/phase27_metrics.csv` · cell `{DECISION_CELL}` · season `all` · protocol P1 "
      "(P2 is bit-identical on this cell: `max|Δ| = 0.0`). 24 models × 3 horizons.")
    A("")
    A("### 1.1 Ranking — PR-AUC on the decision cell")
    A("")
    rows = []
    best = wide["mean"].iloc[0]
    for idx, row in wide.iterrows():
        rows.append([
            int(row["rank"]), idx, row.get("registry_id", "—"), row.get("level", "—"),
            row.get("family", "—"), row.get("feature_set", "—"),
            r(row["label_24h"]), r(row["label_48h"]), r(row["label_72h"]),
            f"**{r(row['mean'])}**", r(row["mean"] - best), r(row.get("dev7_mean")),
        ])
    A(md_table(
        ["#", "model", "reg", "level", "family", "feature set",
         "24 h", "48 h", "72 h", "**mean**", "Δ vs #1", "7-dev mean (not a selector)"],
        rows))
    A("")
    A("### 1.2 Full metric detail — discrimination")
    A("")
    rows = []
    for idx in wide.index:
        for h in HORIZONS:
            x = v[(v.model == idx) & (v.horizon == h)]
            if x.empty:
                continue
            x = x.iloc[0]
            rows.append([
                idx, HZ[h], ri(x["n"]), r(x.base_rate, 4),
                f"**{r(x.pr_auc)}**", f"[{r(x.pr_auc_ci_lo, 4)}, {r(x.pr_auc_ci_hi, 4)}]",
                r(x.roc_auc), r(x.macro_pr_auc),
            ])
    A(md_table(["model", "hor", "n", "base", "PR-AUC", "95 % CI", "ROC-AUC", "Macro PR-AUC"], rows))
    A("")
    A("### 1.3 Full metric detail — calibration & operating point")
    A("")
    A("*Accuracy (`Acc`) is shown for completeness only; per Phase-16 §10 it is banned as a "
      "headline because it is inflated by the ~90 % negative class.*")
    A("")
    rows = []
    for idx in wide.index:
        for h in HORIZONS:
            x = v[(v.model == idx) & (v.horizon == h)]
            if x.empty:
                continue
            x = x.iloc[0]
            rows.append([
                idx, HZ[h], r(x.brier, 4), r(x.logloss, 4), r(x.ece, 4), r(x.threshold, 3),
                r(x.recall, 3), r(x.precision, 3), r(x.specificity, 4), r(x.f1, 3),
                r(x.accuracy, 3), ri(x.tp), ri(x.fp), ri(x.fn), ri(x.tn),
            ])
    A(md_table(
        ["model", "hor", "Brier", "LogLoss", "ECE", "Thr", "Rec", "Prec", "Spec", "F1",
         "Acc", "TP", "FP", "FN", "TN"], rows))
    A("")
    A("### 1.4 What §1 shows")
    A("")
    top = wide.iloc[0]
    A(f"* **Best pre-improvement model: `{wide.index[0]}`** at **{r(top['mean'])}** horizon-mean "
      f"PR-AUC on the decision cell ({r(top['label_24h'])} / {r(top['label_48h'])} / {r(top['label_72h'])}).")
    A(f"* The two no-skill bars behave as they must: `climatology` scores the base rate itself "
      f"({r(wide.loc['climatology','mean'], 4)}), and `persistence` "
      f"({r(wide.loc['persistence','mean'], 4)}) is strong at 24 h and decays with lead time — "
      "the signature of a genuinely temporal problem.")
    A("* Gradient boosting (`xgboost`, `lightgbm`, `xgboost_selected`) owns the top of the "
      "table. Deep learning (LSTM/GRU/TCN/Transformer/GNN, levels L3–L4) sits **below** it "
      "across every horizon — this is the finding that later motivated the MI-7 fair rematch.")
    A("* Spatial folds are near-saturated for every model; the discrimination that separates "
      "them is almost entirely in the temporal cells (`val_2020/2021/2022`).")
    A("")

    # =============================================================== SECTION 2
    A("---")
    A("")
    A("## 2. The improvement campaign (MI-1 … MI-8)")
    A("")
    A("Everything below is **after** the baseline in §1. It is ranked separately because the "
      "candidates used different feature sets (`FULL_V2`, `FULL_V3`) and a different selection "
      "protocol (D16), so mixing them into one table with §1 would misrepresent the experiment.")
    A("")

    A("### 2.1 MI-5 selection → decision D16 (the frozen champion)")
    A("")
    A("Source: `reports/mi5_scorecard.csv`. Candidates re-scored on frozen validation evidence "
      "only; gates G1/G2 applied; test never read at selection time.")
    A("")
    rows = []
    for _, x in sc.iterrows():
        rows.append([
            x["rank"] if pd.notna(x["rank"]) else "—", x.model,
            "yes" if x.eligible else "no",
            r(x.p1_pr_auc_24h), r(x.p1_pr_auc_48h), r(x.p1_pr_auc_72h),
            f"**{r(x.p1_pr_auc_mean)}**", r(x.worst_ece, 4), ri(x.missed_warn_total),
            r(x.fit_s, 1), r(x.infer_ms_per_1k, 1),
        ])
    A(md_table(
        ["rank", "model", "eligible", "24 h", "48 h", "72 h", "**mean**",
         "worst ECE", "missed warns", "fit s", "infer ms/1k"], rows))
    A("")
    A("**Result (D16):** `hpo_perhorizon_FULL_V2` — LightGBM `lgbm7` @ 24/48 h + XGBoost `xgb3` "
      "@ 72 h, on `FULL_V2` — at **0.95743** horizon-mean, **+0.00280** over the Phase-33 "
      "incumbent (`xgboost` / `FULL`, §1 row #1). That +0.0028 decomposes as **+0.00126 from "
      "the data / `FULL_V2` features** and **+0.00154 from search on top of them** (§2.6) — and "
      "it is the ceiling this project has moved its own frozen champion by.")
    A("")

    A("### 2.2 MI-8 feature engineering — fixed-model A/B")
    A("")
    A("Source: `reports/mi8_probe.csv`. **Same model config, only the feature block changes** — "
      "this isolates the features from the search. `FULL` = 62 (v1), `FULL_V2` = 110 "
      "(v1 + 48 MI-v2 columns; ERA5 gated, not adopted), `FULL_V3` = 165 (`FULL_V2` + the "
      "55-column MI-8 block).")
    A("")
    rows = []
    for fs in ["FULL", "FULL_V2", "FULL_V3"]:
        for h in HORIZONS:
            x = probe[(probe.featureset == fs) & (probe.cell == DECISION_CELL) & (probe.horizon == h)]
            if x.empty:
                continue
            x = x.iloc[0]
            rows.append([fs, HZ[h], ri(x.n), r(x.n_pos / x.n, 4), r(x.pr_auc), r(x.roc_auc),
                         r(x.brier, 4), r(x.ece, 4), r(x.recall, 3), ri(x.fn)])
    A(md_table(["feature set", "hor", "n", "base", "PR-AUC", "ROC-AUC", "Brier", "ECE",
                "Rec", "FN"], rows))
    A("")
    piv = (probe[probe.cell == DECISION_CELL]
           .pivot(index="featureset", columns="horizon", values="pr_auc")[HORIZONS])
    piv["mean"] = piv.mean(axis=1)
    rows = []
    for fs in ["FULL", "FULL_V2", "FULL_V3"]:
        rows.append(
            [fs] + [r(piv.loc[fs, h]) for h in HORIZONS]
            + [f"**{r(piv.loc[fs, 'mean'])}**",
               r(piv.loc[fs, "mean"] - piv.loc["FULL", "mean"]),
               r(piv.loc[fs, "mean"] - piv.loc["FULL_V2", "mean"])]
        )
    A(md_table(["feature set", "24 h", "48 h", "72 h", "**mean**",
                "Δ vs FULL (v1)", "Δ vs FULL_V2"], rows))
    A("")
    A("**Result:** the MI-8 block is a **clean, reproducible win at fixed budget** — "
      "`FULL_V3` − `FULL_V2` (its direct predecessor) = **+0.00058** on the decision cell "
      "(0.95589 → 0.95647) and **+0.00135** across the 7 dev cells, with **21/21 cells "
      "improved**; the full v2+v3 arc vs v1 is `FULL_V3` − `FULL` = **+0.00184** on the "
      "decision cell (+0.00479 across dev). Both reproduction anchors were rebuilt at "
      "`diff = 0.0`; the Phase-30 pattern-A (response-lag) diagnosis was correct.")
    A("")
    A("Honest caveat that must travel with that number: the probe holds the model config fixed, "
      "so it measures *features*, not *the champion*. Whether the new features move the frozen "
      "champion is decided by the extended-budget search (§2.4), not by this table.")
    A("")

    A("### 2.3 MI-7 fair deep-learning rematch")
    A("")
    A("Source: `reports/mi7_ladder.csv` + `reports/mi7_paired_deltas.csv`. Equal features, equal "
      "fusion family, equal `SPACE_SIZE`, per-family budget — the rig the earlier DL phases "
      "never had. Verdict was **negative** — the DL loses at every horizon, and 8 of 9 paired "
      "CIs exclude 0 in ML's favour.")
    A("")
    A(md_table(["model", "24 h", "48 h", "72 h", "**mean**", "Δ vs D16"],
               [[x.model, r(x.h24), r(x.h48), r(x.h72), f"**{r(x['mean'])}**",
                 r(x["mean"] - 0.957427)] for _, x in ladder.iterrows()]))
    A("")
    rows = []
    for _, x in deltas.iterrows():
        for m in ["lightgbm", "xgboost_selected", "xgboost"]:
            rows.append([HZ[x.horizon], f"DL − {m}", r(x[f"d_vs_{m}"]),
                         r(x[f"lo_vs_{m}"]), r(x[f"hi_vs_{m}"]),
                         r(x[f"pgt_vs_{m}"], 3)])
    A(md_table(["hor", "contrast", "Δ PR-AUC", "CI lo", "CI hi", "P(Δ>0)"], rows))
    A("")
    A("**Result:** the DL loses to classical ML at every horizon — `fusion_v2_L3` **0.93399** "
      "vs `xgboost` **0.95463** (−0.02064) and vs D16 **0.95743** (−0.02344). **8 of 9** paired "
      "station-bootstrap CIs exclude 0 in ML's favour (24 h −0.02284, 48 h −0.01767, "
      "72 h −0.01397 vs LightGBM); the lone exception is the 72 h vs `lightgbm` contrast, whose "
      "CI just straddles 0 (`[−0.03993, +0.00069]`, P(Δ>0) = 0.034) — a tie, not a win. The DL "
      "queue was therefore **abandoned** and its compute redirected to score improvement. The "
      "stacking attempt (MI-7 Track 5) was a **wash** — every CI straddled 0.")
    A("")

    A("### 2.4 MI-8 extended-budget search — 71 configs/horizon, 3 families")
    A("")
    A("71 configs × 3 horizons = 213 fits per arm (XGBoost 30 / LightGBM 30 / HistGB 10), "
      "selected on `val_2022` per D16. The `full_v3` arm carries the MI-8 feature block; the "
      "`full_v2` arm is the **equal-budget control** on the old feature set, and is what "
      "separates “the features helped” from “more search helped”.")
    A("")
    arm_means = {}
    for arm in ("full_v3", "full_v2"):
        final = REP / f"mi8_hpo_{arm}.csv"
        live = REP / f"mi8_hpo_partial_{arm}.csv"
        src = final if final.exists() else live
        if not src.exists():
            continue
        h = pd.read_csv(src)
        rows, means = [], []
        for hz_ in HORIZONS:
            g = h[h.horizon == hz_]
            if g.empty:
                continue
            b = g.sort_values("val_2022", ascending=False).iloc[0]
            means.append(b.val_2022)
            rows.append([HZ[hz_], r(b.val_2022), f"{b.family}/{b.tag}", len(g)])
        status = "FINAL" if final.exists() else "in flight"
        A(f"**arm `{arm}` — {status}** ({len(h)} configs)")
        A("")
        if len(means) == 3:
            arm_means[arm] = sum(means) / 3
            rows.append(["**mean**", f"**{r(arm_means[arm])}**",
                         f"vs D16 0.95743 → **{r(arm_means[arm] - 0.957427)}**", len(h)])
        A(md_table(["hor", "best PR-AUC", "config", "configs"], rows))
        A("")
    if len(arm_means) == 2:
        d = arm_means["full_v3"] - arm_means["full_v2"]
        A(f"**Reading.** `full_v3` − `full_v2` = **{r(d)}** at equal budget. "
          + ("The feature block *does* carry through search — the residual gap to D16 is then "
             "explained by D16's own per-horizon choice, not by the features."
             if d > 0.0003 else
             "The equal-budget control is **level with — in fact marginally ahead of — the v3 "
             "arm**, so the extra 55 features buy **nothing under extended search**: the "
             f"probe's fixed-config gain (+0.00058) did not transfer under a full search. The "
             f"control's own nominal {r(arm_means['full_v2'] - 0.957427)} over D16 lands inside "
             "the bootstrap CI (every delta straddles 0), so there is **no real gain from either "
             "the features or the extra search** — the champion D16 stands.")
          )
        A("")
    elif arm_means.get("full_v3") is not None:
        A(f"`full_v3` lands at **{r(arm_means['full_v3'])}** horizon-mean vs D16 "
          f"**0.95743** → **{r(arm_means['full_v3'] - 0.957427)}**: "
          "parity, not a gain. The equal-budget `full_v2` control decides whether that is "
          "“features helped but D16 was already optimal” or “the lever is dead”.")
        A("")


    A("### 2.5 Levers tested and rejected")
    A("")
    A("Recorded so the project does not re-run dead ends.")
    A("")
    rows = [[x.model, r(x.val_2022), r(x.dev_mean)] for _, x in ens.iterrows()
            if x.horizon == "label_24h"]
    A(md_table(["MI-3c ensemble member / blend", "val_2022 (24 h)", "dev mean"], rows[:8]))
    A("")
    A("* **MI-3c ensembling — REJECTED.** Every blend loses to the best single config on the "
      "decision cell at every horizon.")
    e22 = era5[era5.cell == DECISION_CELL].pivot(index="featureset", columns="horizon",
                                                 values="pr_auc")[HORIZONS]
    e22["mean"] = e22.mean(axis=1)
    e_delta = e22.loc["FULL_V2_era5", "mean"] - e22.loc["FULL_V2_pre", "mean"]
    A("")
    A("MI-1's ERA5-Land block, same fixed config, three feature sets:")
    A("")
    A(md_table(["feature set", "24 h", "48 h", "72 h", "**mean**"],
               [[fs] + [r(e22.loc[fs, h]) for h in HORIZONS]
                + [f"**{r(e22.loc[fs, 'mean'])}**"] for fs in e22.index]))
    A("")
    A(f"* **ERA5-Land — NOT ADOPTED.** Wash on the decision cell "
      f"(`FULL_V2_era5` − `FULL_V2_pre` = **{r(e_delta)}** horizon-mean, i.e. negative) and only "
      f"+0.00075 across dev cells; D16 stood.")
    A("* **Top-2 lgbm/xgb blend — REJECTED (free test).** Member correlation 0.996–0.997, best "
      "gain ≤ +0.00067 — confirms MI-3c on a second variant.")
    A("* **Label ceiling (MI-4).** PR-AUC 0.99 is label-limited: base rate plus 4–10 % "
      "near-threshold noise caps the reachable value at 0.96–0.97.")
    A("")

    if arm_means.get("full_v3") is not None and arm_means.get("full_v2") is None:
        hpo_row = ("MI-8 extended HPO (`full_v3`)", 
                   f"{r(arm_means['full_v3'])} vs D16 0.95743 = {r(arm_means['full_v3'] - 0.957427)}",
                   "parity — no new champion")
    elif arm_means.get("full_v2") is not None and arm_means.get("full_v3") is not None:
        d8 = arm_means["full_v3"] - arm_means["full_v2"]
        hpo_row = ("MI-8 extended HPO (v3 vs v2 control)",
                   f"{r(d8)} at equal budget; {r(arm_means['full_v3'] - 0.957427)} vs D16",
                   "gain kept" if d8 > 0.0003 and arm_means["full_v3"] > 0.957427 else
                   ("features carry, D16 still ahead" if d8 > 0.0003 else "no gain — D16 stands"))
    else:
        hpo_row = ("MI-8 extended HPO", "in flight", "pending")

    A("### 2.6 Improvement campaign — bottom line")
    A("")
    A(md_table(["stage", "what it bought", "verdict"],
               [["MI-1/2 data + `FULL_V2` features", "+0.00126 (0.95463 → 0.95589)", "adopted (D16)"],
                ["MI-3b HPO on top of that", "+0.00154 (0.95589 → 0.95743)", "adopted (D16)"],
                ["**total vs Phase-33 incumbent**", "**+0.00280 (0.95463 → 0.95743)**", "adopted (D16)"],
                ["MI-3c ensemble", "≤ 0 / negative", "rejected"],
                ["MI-1 ERA5 block", "−0.00025 on decision cell", "not adopted"],
                ["MI-4 label audit", "defines ceiling 0.96–0.97", "informational"],
                ["MI-7 DL rematch", "−0.0206 vs ML, 8/9 CIs exclude 0", "queue abandoned"],
                ["MI-7 stacking", "wash, CI straddles 0", "rejected"],
                ["MI-8 `FULL_V3` features (fixed config)", "+0.00058 / +0.00135 dev", "win, but not in champion"],
                [hpo_row[0], hpo_row[1], hpo_row[2]]]))
    A("")

    # =============================================================== SECTION 3
    A("---")
    A("")
    A("## 3. The frozen test — reproduced from the two logged single reads")
    A("")
    A("**No test data was read to build this file.** These rows are copied from the guarded "
      "artefacts `reports/phase33_test_metrics.csv` (`n_reads = 1`) and "
      "`reports/mi6_test_metrics.csv` (`n_reads = 1`). Cells are shown separately (D8); "
      "the mean row is a convenience summary of the same three horizons, not a pooled cell.")
    A("")
    means = {}
    for m in td.model.unique():
        g = td[td.model == m]
        per = {h: g[g.horizon == h].pr_auc.mean() for h in HORIZONS}
        means[m] = sum(per.values()) / 3
    order = sorted(means, key=means.get, reverse=True)
    A(md_table(["model", "read", "24 h", "48 h", "72 h", "**mean**"],
               [[m, "MI-6 (D16)" if m == "hpo_perhorizon_FULL_V2" else
                 "Phase-33" if m == "xgboost" else "both",
                 *[r(td[(td.model == m) & (td.horizon == h)].pr_auc.mean()) for h in HORIZONS],
                 f"**{r(means[m])}**"] for m in order]))
    A("")
    rows = []
    for c in sorted(td.cell.unique()):
        row = [c]
        for m in order:
            g = td[(td.model == m) & (td.cell == c)]
            row.append(r(g.pr_auc.mean()) if len(g) else "—")
        rows.append(row)
    A(md_table(["test cell"] + order, rows))
    A("")
    A("**Reading.** The same model that scores 0.957 on `val_2022` scores 0.771 on the frozen "
      "test. That is not a bug — PR-AUC is base-rate-scaled, and the base rate falls from "
      "9.6–11.5 % in training to 5.1–6.6 % on `val_2022` to **0.7–5.9 %** (mean ≈ 2.2 %) across "
      "the test cells; the page-33 study puts PR-AUC vs base rate at **r = +0.87** on the dev "
      "cells. ROC-AUC stays high (0.933–0.997 across the test cells). The unchanged `persistence` "
      "bar collapses the same way (0.920 → 0.628 at 24 h), while D16's **margin over persistence "
      "actually grows** on the test set. Any claim of “0.99 PR-AUC everywhere” would be false; "
      "this file does not make it.")
    A("")

    # ------------------------------------------------------------------ provenance
    A("---")
    A("")
    A("## 4. Provenance")
    A("")
    A(md_table(["section", "source file(s)"],
               [["§1 pre-improvement roster & metrics", "`reports/phase27_metrics.csv`, `reports/phase17_experiment_registry.csv`"],
                ["§2.1 MI-5 / D16 selection", "`reports/mi5_scorecard.csv`"],
                ["§2.2 MI-8 feature A/B", "`reports/mi8_probe.csv`"],
                ["§2.3 MI-7 DL rematch", "`reports/mi7_ladder.csv`, `reports/mi7_paired_deltas.csv`"],
                ["§2.4 MI-8 extended search", "`reports/mi8_hpo_full_v3.csv`, `reports/mi8_hpo_full_v2.csv`"],
                ["§2.5 rejected levers", "`reports/mi3_ensemble.csv`, `reports/mi3_era5_probe.csv`"],
                ["§3 frozen test", "`reports/mi6_test_metrics.csv`, `reports/phase33_test_metrics.csv`"]]))
    A("")
    A(f"Regenerate with `.venv\\Scripts\\python.exe scripts\\build_all_models_score.py` "
      f"(seed `{SEED}`). The script reads `reports/` only and writes this one file; it touches "
      "no frozen artefact, no notebook and neither governing document.")
    A("")
    A("*Not in this file:* MI-7 per-architecture DL configs that were killed mid-run "
      "(`lstm_v2_L7` and the unrun `fusion_v1_L3` / `gru` / `tcn` / `transformer_v2` families) "
      "have no completed scorecard and are therefore absent rather than estimated. The MI-8 "
      "`full_v2` equal-budget control has now run (213/213) and appears in §2.4.")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    print(f"pre-MI models ranked: {len(wide)} | MI scorecard rows: {len(sc)} | test rows: {len(td)}")


if __name__ == "__main__":
    main()
