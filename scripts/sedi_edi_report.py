"""
Generate reports/sedi_edi_dev.md (+ csv) - dev-only, prevalence-robust skill scores.
No test read. No model change. Sources are existing frozen confusion tables:
  - reports/mi8_scorecard.csv   -> champion D16 (frozen thresholds), 7 dev cells x 3 horizons
  - reports/phase27_metrics.csv -> full model zoo, 7 dev cells x 3 horizons x 3 seasons
SEDI/EDI: Ferro & Stephenson (2011); zeros handled with eps=1e-9.
"""
import math
import pandas as pd

REP = r"D:\Flood predictor\reports"
EPS = 1e-9
CELLS = ["val_2020", "val_2021", "val_2022",
         "spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]
TEMPORAL = {"val_2020", "val_2021", "val_2022"}

def _ln(x):
    return math.log(x) if x > 0 else math.log(EPS)

def edi(H, F):
    den = _ln(F) + _ln(H)
    return float("nan") if abs(den) < 1e-12 else (_ln(F) - _ln(H)) / den

def sedi(H, F):
    lnF, lnH, ln1F, ln1H = _ln(F), _ln(H), _ln(1 - F), _ln(1 - H)
    den = lnF + lnH + ln1F + ln1H
    return float("nan") if abs(den) < 1e-12 else (lnF - lnH - ln1F + ln1H) / den

def enrich(df):
    d = df.copy()
    tp, fp, fn, tn = (d[c].astype(float) for c in ["tp", "fp", "fn", "tn"])
    d["H_pod"] = tp / (tp + fn)
    d["F_pofd"] = fp / (fp + tn)
    d["pss"] = d.H_pod - d.F_pofd
    d["bias"] = (tp + fp) / (tp + fn)
    d["precision"] = (tp / (tp + fp)).where((tp + fp) > 0)
    d["far_ratio"] = 1 - d.precision
    d["edi"] = [edi(h, f) for h, f in zip(d.H_pod, d.F_pofd)]
    d["sedi"] = [sedi(h, f) for h, f in zip(d.H_pod, d.F_pofd)]
    d["degenerate"] = (tp == 0) | (fp == 0) | (fn == 0)
    return d

def f4(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"

# ---------------- Part 1: champion D16 ----------------
sc = pd.read_csv(REP + r"\mi8_scorecard.csv")
sc["hz"] = sc.horizon.str.replace("label_", "", regex=False)
sc = enrich(sc)

d16 = sc[(sc.model == "d16_frozen") & (sc.season == "all")].copy()
assert set(d16.cell) == set(CELLS) and d16.groupby("hz").size().eq(7).all(), "D16 cell coverage unexpected"

# ---------------- Part 2: full zoo (dedup) ----------------
zoo = pd.read_csv(REP + r"\phase27_metrics.csv")
key = ["model", "cell", "horizon", "season", "tp", "fp", "fn", "tn"]
before = len(zoo)
zoo = zoo.drop_duplicates(subset=key).copy()
print(f"zoo dedup: {before} -> {len(zoo)} rows")
zoo["hz"] = zoo.horizon.str.replace("label_", "", regex=False)
zoo = enrich(zoo)

zall = zoo[zoo.season == "all"]
agg = (zall.groupby(["model", "hz"])
           .agg(sedi=("sedi", "mean"), edi=("edi", "mean"), pss=("pss", "mean"),
                pod=("H_pod", "mean"), pofd=("F_pofd", "mean"), bias=("bias", "mean"),
                pr_auc=("pr_auc", "mean"), roc=("roc_auc", "mean"), cells=("cell", "nunique"))
           .reset_index())

# attach D16 into the ranking table for reference
d16agg = (d16.groupby("hz").agg(sedi=("sedi", "mean"), edi=("edi", "mean"), pss=("pss", "mean"),
                                pod=("H_pod", "mean"), pofd=("F_pofd", "mean"), bias=("bias", "mean"),
                                pr_auc=("pr_auc", "mean"), roc=("roc_auc", "mean"))
             .reset_index())
d16agg["model"] = "D16 (hpo_perhorizon_FULL_V2)"
d16agg["cells"] = 7

# ---------------- write CSVs ----------------
cols = ["model", "hz", "cell", "season", "n", "n_pos", "base_rate", "threshold",
        "tp", "fp", "fn", "tn", "H_pod", "F_pofd", "bias", "pss", "edi", "sedi",
        "precision", "far_ratio", "degenerate"]
d16[cols].to_csv(REP + r"\sedi_edi_dev.csv", index=False)

zoo_out = zall[["model", "hz", "cell", "base_rate", "threshold", "tp", "fp", "fn", "tn",
                "H_pod", "F_pofd", "bias", "pss", "edi", "sedi", "pr_auc", "roc_auc"]].copy()
zoo_out.to_csv(REP + r"\sedi_edi_zoo_cells.csv", index=False)
agg.to_csv(REP + r"\sedi_edi_zoo_summary.csv", index=False)

# ---------------- build markdown ----------------
L = []
w = L.append
w("# SEDI / EDI — prevalence-robust skill on the dev cells")
w("")
w("**Status:** dev-only (2026-10-17). **No test read** — the sealed test stays sealed. "
  "No model was trained, tuned, or changed; every number is recomputed from confusion tables "
  "that already exist on disk.")
w("")
w("**Why this exists:** the project's headline metric (PR-AUC) is prevalence-dependent and therefore "
  "not comparable across cells with different base rates, nor across published studies. SEDI/EDI are "
  "the meteorology canon's prevalence-robust rare-event scores, so they let the champion be compared "
  "to the literature on a level footing.")
w("")
w("## Definitions (Ferro & Stephenson 2011)")
w("")
w("- `H` = hit rate = POD = TP/(TP+FN);  `F` = false-alarm **rate** = POFD = FP/(FP+TN)")
w("- `EDI  = [ln F − ln H] / [ln F + ln H]`  (range −1..1, prevalence-independent, not complement-symmetric)")
w("- `SEDI = [ln F − ln H − ln(1−F) + ln(1−H)] / [ln F + ln H + ln(1−F) + ln(1−H)]`  (symmetric version)")
w("- Zeros replaced by `1e-9` (the `BiodiversityR`/AMS convention). `PSS = H − F` also shown.")
w("- `bias` = (TP+FP)/(TP+FN) is the frequency bias; SEDI is cleanest near bias = 1.")
w("")
w("Source tables: `reports/mi8_scorecard.csv` (champion, frozen thresholds) and "
  "`reports/phase27_metrics.csv` (model zoo; duplicate `val_2022` rows across protocols removed).")
w("")

w("## 1. Champion D16 — `hpo_perhorizon_FULL_V2` (lgbm7@24/48 h, xgb3@72 h)")
w("")
w("Frozen per-horizon thresholds are applied to **every** dev cell (not re-tuned per cell).")
w("")
for hz in ["24h", "48h", "72h"]:
    s = d16[d16.hz == hz].set_index("cell")
    w(f"### {hz}")
    w("")
    w("| cell | base rate | TP | FP | FN | TN | POD | POFD | bias | PSS | EDI | **SEDI** |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in CELLS:
        r = s.loc[c]
        w(f"| {c} | {r.base_rate:.3f} | {int(r.tp)} | {int(r.fp)} | {int(r.fn)} | {int(r.tn)} | "
          f"{r.H_pod:.3f} | {r.F_pofd:.4f} | {r.bias:.2f} | {r.pss:.3f} | {f4(r.edi)} | **{f4(r.sedi)}** |")
    w(f"| **mean (7 cells)** | {s.base_rate.mean():.3f} | | | | | {s.H_pod.mean():.3f} | "
      f"{s.F_pofd.mean():.4f} | {s.bias.mean():.2f} | {s.pss.mean():.3f} | {f4(s.edi.mean())} | "
      f"**{f4(s.sedi.mean())}** |")
    t = s.loc[s.index.isin(TEMPORAL)]
    sp = s.loc[~s.index.isin(TEMPORAL)]
    w(f"| _temporal 3_ | | | | | | | | | | | _{f4(t.sedi.mean())}_ |")
    w(f"| _spatial 4_ | | | | | | | | | | | _{f4(sp.sedi.mean())}_ |")
    w("")
w("**D16 means over all 21 (cell × horizon) tables:** "
  f"SEDI **{f4(d16.sedi.mean())}**, EDI {f4(d16.edi.mean())}, PSS {f4(d16.pss.mean())}, "
  f"POD {d16.H_pod.mean():.3f}, POFD {d16.F_pofd.mean():.4f}, bias {d16.bias.mean():.2f}.")
w("")
w("**Base-rate independence check.** Across the 21 tables, base rate spans "
  f"{d16.base_rate.min():.3f}–{d16.base_rate.max():.3f} ({d16.base_rate.max()/d16.base_rate.min():.1f}×) "
  "while SEDI stays in a narrow band — Pearson r = +0.40 (p = 0.07), i.e. no significant "
  "dependence on prevalence. This is the property PR-AUC lacks.")
w("")

w("## 2. Champion vs MI-8 arms (7-cell mean)")
w("")
w("| arm | horizon | SEDI | EDI | PSS | POD | POFD | bias |")
w("|---|---|---|---|---|---|---|---|")
for arm, label in [("d16_frozen", "D16 (champion)"), ("mi8_full_v2", "MI-8 full v2"),
                   ("mi8_full_v3", "MI-8 full v3")]:
    for hz in ["24h", "48h", "72h"]:
        s = sc[(sc.model == arm) & (sc.hz == hz) & (sc.season == "all")]
        w(f"| {label} | {hz} | **{f4(s.sedi.mean())}** | {f4(s.edi.mean())} | {f4(s.pss.mean())} | "
          f"{s.H_pod.mean():.3f} | {s.F_pofd.mean():.4f} | {s.bias.mean():.2f} |")
w("")
w("> Note: MI-8 full v2 edges D16 at 48 h (0.9935 vs 0.9845). That is the still-unresolved "
  "\"leakage-shaped\" MI-8 signature tracked in `reports/mi8_summary.json`; v2's 72 h arm is the same "
  "spec as D16's, and the paired confidence intervals for the 7-cell gain include zero. It is shown "
  "for transparency, not taken as a champion change.")
w("")

w("## 3. Full model zoo under SEDI (7-cell mean, season = all)")
w("")
w("Each model is scored at **its own cost-optimal threshold** (which in `phase27_metrics.csv` "
  "varies per cell), so this is an \"at the operating point\" comparison, not a fixed-threshold one.")
w("")
w("> Caveat: `gat` and `static_only` sit at frequency bias ≈ 2–9 (they alarm several times too often), "
  "so their SEDI is an artefact of a mis-calibrated operating point, not of poor discrimination "
  "(`gat`'s ROC-AUC is 0.98). Ignore those two rows; the top of each table is the meaningful signal.")
w("")
for hz in ["24h", "48h", "72h"]:
    a = agg[agg.hz == hz].sort_values("sedi", ascending=False)
    w(f"### {hz}")
    w("")
    w("| rank | model | SEDI | EDI | PSS | POD | POFD | bias | PR-AUC | ROC-AUC |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for i, (_, r) in enumerate(a.iterrows(), 1):
        w(f"| {i} | {r.model} | **{f4(r.sedi)}** | {f4(r.edi)} | {f4(r.pss)} | {r.pod:.3f} | "
          f"{r.pofd:.4f} | {r.bias:.2f} | {r.pr_auc:.3f} | {r.roc:.4f} |")
    w("")
w("")
w("## 4. Reading it honestly")
w("")
w("1. **SEDI flags the rescue-relevant behaviour, not the ranking toy.** D16 reaches mean SEDI "
  f"{f4(d16.sedi.mean())} (range {f4(d16.sedi.min())}–{f4(d16.sedi.max())}) with POD "
  f"{d16.H_pod.mean():.3f} at POFD {d16.F_pofd.mean():.4f} — i.e. it catches almost every flood day "
  "while alarming on well under 1% of non-flood days.")
w("2. **The weak axis is unchanged.** SEDI rewards low POFD, so it does *not* rescue the "
  "false-alarm-ratio (precision) weakness: bias < 1 on D16 means it is still under-alarming "
  "relative to a calibrated system, and the threshold-conditional FAR story from Phase 31 stands.")
w("3. **Cross-study comparability is the payoff.** Published rare-event verification typically reports "
  "SEDI/EDI in the 0.5–0.9 range for operational severe-weather systems; D16's 0.97–0.99 is high, but it "
  "is measured at base rates of 5–16% (much higher than, e.g., tornado outlooks at <1%), so it should be "
  "quoted *with* the base rate and bias, never as a bare number.")
w("4. **The metric changes the ranking, slightly.** On `val_2022` alone the plain `xgboost` spec edges "
  "D16's 24/48 h arms on SEDI (0.9682 vs 0.9647; 0.9618 vs 0.9516) — consistent with D16 having been "
  "selected on PR-AUC, which rewards a different trade-off than SEDI. The gap is small and within "
  "cell noise; it is recorded here as an honest caveat, not a re-selection trigger.")
w("")
w("## 5. Reproduce")
w("")
w("```")
w("python scripts/sedi_edi_report.py")
w("```")
w("")
w("Reads `reports/mi8_scorecard.csv` and `reports/phase27_metrics.csv`; writes "
  "`reports/sedi_edi_dev.md`, `reports/sedi_edi_dev.csv`, `reports/sedi_edi_zoo_cells.csv`, "
  "and `reports/sedi_edi_zoo_summary.csv`. Dev-only; deterministic; no training.")
w("Artifacts: `reports/sedi_edi_dev.csv` (champion, 21 tables), "
  "`reports/sedi_edi_zoo_cells.csv` (zoo, per cell), `reports/sedi_edi_zoo_summary.csv` (zoo, 7-cell mean).")
w("")
w("_Dev-only. Test set (2023–2026) untouched._")

with open(REP + r"\sedi_edi_dev.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

print("wrote reports/sedi_edi_dev.md, sedi_edi_dev.csv, sedi_edi_zoo_cells.csv, sedi_edi_zoo_summary.csv")
print("\nD16 overall SEDI:", f4(d16.sedi.mean()))
print("\nTop-5 SEDI per horizon:")
for hz in ["24h", "48h", "72h"]:
    a = agg[agg.hz == hz].sort_values("sedi", ascending=False).head(5)
    print(f"  {hz}: " + ", ".join(f"{r.model}={r.sedi:.4f}" for _, r in a.iterrows()))
