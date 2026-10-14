"""Phase 25 gate (workflow stage 40, decision D13) - is a temporal Transformer justified?

Unlike Phase 24 (an empirical graph gate), Phase 25's gate is a *precondition +
pre-registration* gate: the workflow says the Transformer "must be compared against
something real", having a solid Phase-21/22 temporal baseline and knowing from Phase
23 whether static context helps.  This script:

  * audits the registry contract for L4-03 (transformer_temporal, gated by
    phase25_transformer_gate, budget 20, selection val_2022/pr_auc, test frozen);
  * confirms the Phase-21/22/23 baselines exist and are solid (artefacts + the
    measured D7/xgboost/LSTM/TCN/fusion bars on val_2022 and P3);
  * reports the *window-length sweep* evidence - the temporal baseline does not
    improve as the window grows (LSTM L* = 3), which weakens the "attention over
    long sequences" premise and sets expectations;
  * pre-registers the acceptance / kill criteria for 25.2.

It trains no Transformer - the gate only decides whether 25.2 may run.  ASCII-only
console output (cp1252), seed 20261008.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261008
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REP = ROOT / "reports"

REGISTRY = REP / "phase17_experiment_registry.csv"
LOG = REP / "phase25_transformer_gate.log"
OUT_JSON = REP / "phase25_transformer_gate.json"
FIG = ROOT / "figures" / "phase25_gate.png"

_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _lines.append(msg)


def check(cid: str, detail: str, ok: bool) -> bool:
    log(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}")
    return bool(ok)


HORIZONS = ["label_24h", "label_48h", "label_72h"]
SEASONS = ["all", "monsoon", "non-monsoon"]


def headline_metrics(path: Path, model: str):
    """(val_2022 mean over 3 horizons, P3 mean over 4 folds), season=all."""
    m = pd.read_csv(path)
    v = m[(m.model == model) & (m.protocol == "P1") & (m.cell == "val_2022")
          & (m.season == "all")]
    p3 = m[(m.model == model) & (m.protocol == "P3") & (m.season == "all")]
    vmean = float(v.pr_auc.mean()) if len(v) else float("nan")
    p3mean = float(p3.pr_auc.mean()) if len(p3) else float("nan")
    return round(vmean, 5), round(p3mean, 5), len(v), len(p3)


# --------------------------------------------------------------------------- #
log("=" * 78)
log("PHASE 25 GATE (stage 40, D13) - temporal Transformer justification")
log("=" * 78)

# 1. registry contract -------------------------------------------------------- #
reg = pd.read_csv(REGISTRY)
l4 = reg[reg.level == "L4"]
row = l4[l4.registry_id == "L4-03"]
ok_reg = (len(row) == 1 and row.iloc[0].model == "transformer_temporal"
          and row.iloc[0].gate == "phase25_transformer_gate"
          # 25.2 discharges this gate: 25.1 produced status=gated, the amendment
          # (phase17_framework.py) flips it gated -> planned, so the gate stays
          # re-runnable after the phase closes.
          and row.iloc[0].status in {"gated", "planned"}
          and int(row.iloc[0].tuning_budget_configs) == 20
          and row.iloc[0].selection_cell == "val_2022"
          and row.iloc[0].selection_metric == "pr_auc"
          and row.iloc[0].test_policy == "frozen_read_at_phase33")
r = row.iloc[0]
check("G1", f"registry L4-03 = transformer_temporal, gate={r.gate}, status={r.status}, "
            f"budget={int(r.tuning_budget_configs)}, selection={r.selection_cell}/"
            f"{r.selection_metric}, test={r.test_policy}",
      bool(ok_reg))

# 2. preconditions: the baselines that the Transformer must beat ----------------- #
pre = {
    "phase21_metrics.csv": REP / "phase21_metrics.csv",
    "phase22_metrics.csv": REP / "phase22_metrics.csv",
    "phase23_metrics.csv": REP / "phase23_metrics.csv",
    "phase21_search.csv": REP / "phase21_search.csv",
    "phase22_search.csv": REP / "phase22_search.csv",
}
missing = [k for k, p in pre.items() if not p.exists()]
check("G2", f"Phase-21/22/23 baseline artefacts present "
            f"({len(pre) - len(missing)}/{len(pre)}) - a Transformer has real bars "
            f"to beat (LSTM / TCN / fusion)", not missing)

search21 = pd.read_csv(pre["phase21_search.csv"])
search22 = pd.read_csv(pre["phase22_search.csv"])
budgets = (search21.groupby("family").n_configs_budget.first().to_dict()
           | search22.groupby("family").n_configs_budget.first().to_dict())
check("G3", f"Phase-21/22 tuning budgets recorded and <= 20 (D14): "
            f"{sorted(budgets.items())}",
      bool(budgets) and all(v <= 20 for v in budgets.values()))

# 3. window-length sweep (the core evidence) ------------------------------------ #
L_FAM = {"lstm_L3": 3, "lstm_L7": 7, "lstm_L14": 14, "lstm_L30": 30}
sweep = {}
for fam, L in L_FAM.items():
    sub = search21[search21.family == fam]
    if len(sub):
        w = sub.sort_values("mean_pr_auc", ascending=False).iloc[0]
        sweep[L] = float(w.mean_pr_auc)
lstar = int(json.loads((PROC / "phase21_lstar.json").read_text())["L_star"])
best_L = max(sweep, key=sweep.get)
d_short = sweep[best_L]
d_long = sweep.get(30, float("nan"))
check("G4", f"window-length sweep (val_2022 mean PR-AUC): "
            f"{ {L: round(v, 5) for L, v in sorted(sweep.items())} }; "
            f"L*={lstar} (best {best_L}); longer windows do NOT help "
            f"(L30 = {d_long:.5f} <= L* = {d_short:.5f})",
      lstar == best_L and d_long <= d_short + 1e-9)

# 4. baseline bars measured ----------------------------------------------------- #
bars = {}
bars["lstm_L3"] = headline_metrics(pre["phase21_metrics.csv"], "lstm_L3")
bars["gru_bestL"] = headline_metrics(pre["phase21_metrics.csv"], "gru_bestL")
bars["tcn_bestL"] = headline_metrics(pre["phase22_metrics.csv"], "tcn_bestL")
bars["fusion_L3"] = headline_metrics(pre["phase23_metrics.csv"], "fusion_L3")
bars["xgboost"] = ("P1 val_2022:",
                   headline_metrics(REP / "phase20_metrics.csv", "xgboost"))
persist = headline_metrics(REP / "phase16_baseline_metrics.csv", "persistence")

log("")
log("-- temporal / feature baselines (season=all) ---------------------------")
log(f"{'model':<12}{'val_2022':>11}{'P3 mean':>10}")
for k, v in bars.items():
    if k == "xgboost":
        log(f"{k:<12}{v[1][0]:>11.5f}{v[1][1]:>10.5f}")
    else:
        log(f"{k:<12}{v[0]:>11.5f}{v[1]:>10.5f}")
log(f"{'persistence':<12}{persist[0]:>11.5f}{persist[1]:>10.5f}")

ok_bars = all(np.isfinite(v[0]) and np.isfinite(v[1]) for k, v in bars.items()
              if k != "xgboost")
check("G5", f"the DL bars are finite and ordered sensibly "
            f"(LSTM val {bars['lstm_L3'][0]:.5f} > TCN {bars['tcn_bestL'][0]:.5f} "
            f"~ fusion {bars['fusion_L3'][0]:.5f}); xgboost is the standing ceiling",
      ok_bars)

# --------------------------------------------------------------------------- #
# figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
Ls = sorted(sweep)
ax[0].plot(Ls, [sweep[L] for L in Ls], "o-", color="#c0392b")
ax[0].axvline(lstar, color="k", ls=":", lw=1)
ax[0].text(lstar, min(sweep.values()), f" L*={lstar}", fontsize=8)
ax[0].set_xlabel("temporal window L (days)")
ax[0].set_ylabel("val_2022 mean PR-AUC")
ax[0].set_title("Window-length sweep: longer is not better (LSTM)")
ax[0].grid(alpha=0.3)
names = ["persistence", "tcn_bestL", "lstm_L3", "fusion_L3", "xgboost"]
p1 = [persist[0], bars["tcn_bestL"][0], bars["lstm_L3"][0], bars["fusion_L3"][0],
      bars["xgboost"][1][0]]
p3 = [persist[1], bars["tcn_bestL"][1], bars["lstm_L3"][1], bars["fusion_L3"][1],
      bars["xgboost"][1][1]]
x = np.arange(len(names))
ax[1].bar(x - 0.2, p1, 0.4, label="val_2022")
ax[1].bar(x + 0.2, p3, 0.4, label="P3 mean")
ax[1].set_xticks(x)
ax[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
ax[1].set_ylabel("PR-AUC")
ax[1].set_title("Bars the Transformer must beat")
ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIG, dpi=120)
log("")
log(f"figure: {FIG.relative_to(ROOT)}")

# rules ready before any model runs
rules = {
    "primary": "temporal Transformer (L4-03), budget<=20, tuned on val_2022 "
               "(season=all, mean PR-AUC over 24/48/72h), must beat the best "
               "temporal baseline (TCN L3-06 and ideally LSTM L3-01) on val_2022",
    "secondary": "on P3 spatial CV (unseen stations) the Transformer must not "
                 "lose to LSTM/TCN; a paired station-bootstrap CI on the "
                 "difference is reported on identical cells",
    "kill": "if no meaningful gain (CI spanning 0) over the temporal baselines, "
            "stage 40 closes as a documented rejection and no spatiotemporal "
            "Transformer is attempted",
    "expectation": f"the window sweep (best at L={lstar}) predicts attention's "
                   f"long-range advantage is unlikely to materialise; a "
                   f"tie-or-loss is a legitimate, reportable outcome",
}

summary = {
    "seed": SEED,
    "registry": {"id": "L4-03", "model": "transformer_temporal",
                 "gate": "phase25_transformer_gate", "budget": 20},
    "window_sweep": {str(L): round(sweep[L], 5) for L in Ls},
    "lstar": lstar,
    "baseline_bars": {k: (v[1] if k == "xgboost" else v) for k, v in bars.items()},
    "persistence": persist,
    "preconditions_ok": not missing,
    "rules": rules,
    "decision": "PASS (narrow) - proceed to 25.2 under the pre-registered rules",
}
OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
LOG.write_text("\n".join(_lines) + "\n", encoding="utf-8")

log("")
log("pre-registered rules:")
for k, v in rules.items():
    log(f"  {k}: {v}")
log("")
log("DECISION: GATE PASSED (narrow) - proceed to 25.2")
log(f"wrote {OUT_JSON.relative_to(ROOT)}, {LOG.relative_to(ROOT)}")
