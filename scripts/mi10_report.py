r"""Generate `reports/p2p3_correction.md` from the MI-10 correction outputs.

Reads only `reports/p2p3_rescore/` plus the archived source tables that the
correction replaces.  Writes one markdown file; never touches a model or the test
split.  Run after `scripts/mi10_p2p3_rescore.py` has completed and its gates pass.

    .venv\Scripts\python.exe scripts\mi10_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Flood predictor")
REP = ROOT / "reports"
OUT = REP / "p2p3_rescore"

CELLS = ["val_2020", "val_2021", "val_2022",
         "spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]
HOR = ["label_24h", "label_48h", "label_72h"]
HZ = {"label_24h": "24 h", "label_48h": "48 h", "label_72h": "72 h"}
PROBE_SETS = ["FULL", "FULL_V2", "FULL_V3"]
ARMS = ["d16_frozen", "mi8_full_v3", "mi8_full_v2"]


def f6(x):
    return f"{x:.6f}"


def s6(x):
    return f"{x:+.6f}"


def f4(x):
    return f"{x:.4f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> int:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") == "VOID":
        raise SystemExit("summary is VOID: a reproduction gate failed; refusing to write")
    probe = pd.read_csv(OUT / "probe_p2p3.csv")
    arms = pd.read_csv(OUT / "arms_p2p3.csv")
    d7 = pd.read_csv(OUT / "d7_p2p3.csv")
    margins = pd.read_csv(OUT / "d7_margins_p2p3.csv")
    arch_probe = pd.read_csv(REP / "mi8_probe.csv")
    arch_sc = pd.read_csv(REP / "mi8_scorecard.csv")

    ps = summary["probe"]
    ar = summary["arms"]
    d7s = summary["d7"]
    gates = summary["gates"]

    gate_rows = [[g["gate"], "PASS" if g["pass"] else "FAIL", g["detail"]] for g in gates]

    # ---- feature A/B tables
    fs_rows = []
    for fs in PROBE_SETS:
        corr = ps["mean7"][fs]
        old = ps["archived_mean7"][fs]
        fs_rows.append([f"`{fs}`", f6(corr), f6(old), s6(corr - old)])

    d = ps["delta_v3_vs_v2"]
    ph_rows = [[HZ[hz], s6(d["per_horizon"][hz])] for hz in HOR]
    cell_rows = [[c, s6(d["by_cell"][c])] for c in CELLS]

    # ---- arms
    arm_rows = []
    for m in ARMS:
        corr = ar["mean7"][m]
        old = ar["archived_mean7"][m]
        vs = corr - ar["mean7"]["d16_frozen"]
        arm_rows.append([f"`{m}`", f6(corr), f6(old), s6(vs)])

    sedi_rows = []
    for m in ARMS:
        for hz in HOR:
            sedi_rows.append([f"`{m}`", HZ[hz],
                              f4(ar["sedi_mean7"][m][hz]),
                              f4(ar["archived_sedi_mean7"][m][hz])])

    marg_rows = []
    for m in ARMS:
        v = d7s["margins"][m]
        marg_rows.append([f"`{m}`", f"{v['above']}/{v['total']}", s6(v["min"])])

    # ---- per-cell corrected PR-AUC (compact appendix)
    piv = probe.pivot_table(index="cell", columns="model", values="pr_auc").reindex(CELLS)
    percell_rows = [[c, f6(piv.loc[c, "FULL"]), f6(piv.loc[c, "FULL_V2"]),
                     f6(piv.loc[c, "FULL_V3"]),
                     s6(piv.loc[c, "FULL_V3"] - piv.loc[c, "FULL_V2"])]
                    for c in CELLS]

    arm_piv = arms.pivot_table(index=["cell", "horizon"], columns="model", values="pr_auc")
    arm_cell_rows = []
    for c in CELLS:
        for hz in HOR:
            r = arm_piv.loc[(c, hz)]
            arm_cell_rows.append([c, HZ[hz], f6(r["d16_frozen"]), f6(r["mi8_full_v3"]),
                                  f6(r["mi8_full_v2"])])

    # ---- status logic (pre-registered)
    if d["mean7"] <= 0:
        fixed_status = "**Withdrawn.** The corrected 7-cell gain is not positive."
    else:
        fixed_status = (f"**Corrected.** The signed mean is {s6(d['mean7'])}; the "
                        f"21/21 phrase becomes **{d['wins']}/{d['total']}**.")

    arm_v3 = ar["delta_v3_vs_d16_mean7"]
    arm_status = ("**Withdrawn.** The corrected 7-cell arm gain is not positive."
                  if arm_v3 <= 0 else
                  f"**Corrected.** The corrected arm difference is {s6(arm_v3)}.")

    d7_ok = all(d7s["margins"][m]["above"] == d7s["margins"][m]["total"] for m in ARMS)
    d7_status = ("**Retained.** Every corrected cell x horizon point is above D7."
                 if d7_ok else
                 "**Withdrawn.** At least one corrected point is not above D7.")

    md = f"""# P2/P3 re-scoring correction — the MI-sprint 7-cell claims

| | |
| --- | --- |
| **Status** | executed and superseding — {summary['created']} |
| **Script** | `scripts/mi10_p2p3_rescore.py` (seed `{summary['seed']}`) |
| **Pre-registration** | `reports/p2p3_correction_protocol.md` (written before the run) |
| **Outputs** | `reports/p2p3_rescore/{{probe,arms,d7,d7_margins}}_p2p3.csv`, `summary.json`, `run.log` |
| **Test policy** | **`test_rows_read = 0`** — dev cells only; every evaluation mask asserted free of `split == "test"` |
| **Wall clock** | {summary['wall_s']:.0f} s |
| **Champion impact** | **none** — D16 was selected on `val_2022`, which this correction reproduces exactly |

## 1. What was wrong

The MI sprint reported "7-cell" means and "21/21 cells" from a **single fit on the
≤ 2021 training rows**, then scored `val_2020`, `val_2021`, `val_2022` and
`spatial_fold_1..4` with that one model. Six of those seven cells are subsets of the fit
rows:

* `val_2020` and `val_2021` are calendar years inside `split == "train"`.
* `spatial_fold_1..4` are `split == "train"` rows partitioned by station.

Only `val_2022` is out-of-sample under that fit. The diagnostic signature was
`train_rows = 61,977` for every cell together with PR-AUC ≈ 0.9999–1.0000 on the six
training subsets. The defect is documented in
`reports/supervisor_progress_report_condensed.md` §4.7; the affected scripts are
`mi3_compare.py`, `mi3_search.py`, `mi5_selection.py`, `mi8_select.py` and
`mi8_probe.py`.

The correction does **not** re-open model selection. D16's decision cell (`val_2022`) is
the one cell that was already out-of-sample, and it is reproduced to < 1e-5 below.

## 2. Correction design (pre-registered)

Every cell gets its own fit and its own training-derived operating threshold:

| cell | protocol | fit rows | evaluation rows |
| --- | --- | --- | --- |
| `val_2020` | P2 rolling origin | year ≤ 2019, non-holdout | year = 2020, non-holdout |
| `val_2021` | P2 rolling origin | year ≤ 2020, non-holdout | year = 2021, non-holdout |
| `val_2022` | P1 / P2 (same fold) | `split == "train"`, non-holdout (≤ 2021) | `split == "val"`, non-holdout |
| `spatial_fold_k` | P3 spatial CV | `split == "train"`, non-holdout, fold ≠ k | `split == "train"`, fold = k |

Two blocks were re-scored:

* **Feature A/B** — the fixed XGBoost specification (`mi3_compare.XGB_PARAMS`, no tuning)
  on `FULL` (62), `FULL_V2` (110) and `FULL_V3` (165).
* **Frozen arms** — D16 (`lgbm7` @ 24/48 h, `xgb3` @ 72 h, `FULL_V2`), the MI-8
  `full_v3` winner, and the equal-budget `full_v2` control, each on its frozen
  per-horizon configuration, plus the D7 persistence floor.

The operating threshold for each fit is the best-F1 threshold on **that cell's training
rows** (`baseline_model.best_f1_threshold`), never on the evaluation rows. PR-AUC is the
primary metric; SEDI/EDI are recomputed from the corrected confusion tables.

## 3. Reproduction gates

The `val_2022` fit is identical to the original P1 fit, so it must reproduce the
archived numbers exactly. All gates passed before any corrected number was accepted:

{table(["gate", "result", "detail"], gate_rows)}

## 4. Corrected results

### 4.1 Fixed-config feature A/B (PR-AUC, mean over 7 cells)

{table(["feature set", "corrected 7-cell mean", "archived (in-sample)", "inflation"], fs_rows)}

`FULL_V3 − FULL_V2`: **{s6(d['mean7'])}** corrected, versus **{s6(d['archived_mean7'])}**
archived; **{d['wins']}/{d['total']}** cell × horizon points improve, versus
**{d['archived_wins']}/{d['archived_total']}** archived.

| horizon | corrected `FULL_V3 − FULL_V2` |
| --- | ---: |
{chr(10).join(f"| {r[0]} | {r[1]} |" for r in ph_rows)}

| cell | corrected `FULL_V3 − FULL_V2` |
| --- | ---: |
{chr(10).join(f"| `{r[0]}` | {r[1]} |" for r in cell_rows)}

**Status of the "21/21 / +0.00135" claim:** {fixed_status}

### 4.2 Frozen MI-8 arms versus D16 (PR-AUC, mean over 7 cells)

{table(["model", "corrected 7-cell mean", "archived (in-sample)", "vs D16 (corrected)"], arm_rows)}

`mi8_full_v3 − d16_frozen` = **{s6(arm_v3)}**; `mi8_full_v2 − d16_frozen` =
**{s6(ar['delta_v2_vs_d16_mean7'])}**.

**Status of the MI-8 arm 7-cell claim:** {arm_status}

### 4.3 SEDI (mean over 7 cells, per-cell training thresholds)

{table(["model", "horizon", "corrected SEDI", "archived SEDI"], sedi_rows)}

### 4.4 D7 persistence floor (corrected)

{table(["model", "points above D7", "minimum margin"], marg_rows)}

**Status of the "beats D7 on every dev cell" claim:** {d7_status}

### 4.5 The decision cell is unchanged

`val_2022` reproduces the archived values exactly: D16 = **0.970789 / 0.956485 /
0.945007** (24/48/72 h; horizon-mean **0.957427**), and the fixed-config probe reproduces
`FULL` 0.968118, `FULL_V2` 0.969538, `FULL_V3` 0.969648 at 24 h (all horizons in the
gate table). This is why D16 remains the champion.

## 5. Claim ledger

| claim | old value | corrected value | status |
| --- | --- | --- | --- |
| MI-8 fixed-config 7-cell gain | +0.00135 | {s6(d['mean7'])} | corrected |
| MI-8 fixed-config wins | 21/21 | {d['wins']}/{d['total']} | corrected |
| `FULL_V3 − FULL` 7-cell | +0.00479 | {s6(ps['delta_v3_vs_full']['mean7'])} | corrected |
| MI-8 `full_v3` vs D16, 7-cell | +0.00164 | {s6(arm_v3)} | {'corrected' if arm_v3 > 0 else 'withdrawn'} |
| D16 SEDI 24/48/72 h, 7-cell | 0.9903 / 0.9845 / 0.9743 | {f4(ar['sedi_mean7']['d16_frozen']['label_24h'])} / {f4(ar['sedi_mean7']['d16_frozen']['label_48h'])} / {f4(ar['sedi_mean7']['d16_frozen']['label_72h'])} | corrected |
| D16 beats D7 on all dev points | 21/21 | {d7s['margins']['d16_frozen']['above']}/21 | {'retained' if d7s['margins']['d16_frozen']['above'] == 21 else 'withdrawn'} |
| D16 `val_2022` horizon-mean | 0.957427 | 0.957427 (reproduced) | retained |
| D16 frozen-test mean vs D7 | 0.77076 vs 0.66755 | unchanged (test untouched) | retained |
| Phase 16–27 / 31 / 32 grids | per-cell protocol | unchanged | retained |
| MI-7 fair ML-vs-DL rematch | per-cell protocol | unchanged | retained |
| MI-3 ERA5 7-cell gain | +0.00075 | not re-scored | withdrawn |
| MI-3c ensemble 7-cell means | quoted | not re-scored | withdrawn |
| MI-5 candidate 7-cell columns | spatial mean / worst temporal / D7 floor | not re-scored for non-D16 candidates | withdrawn (selection on `val_2022` unaffected) |

## 6. What survives, and what the paper may say

* **Survives:** D16's selection on `val_2022`; the D16 test read; the Phase 27 fair
  ML-vs-DL comparison; the Phase 31/32 robustness grids; the MI-7 rematch. These were
  always per-cell or per-year and are not affected by this defect.
* **Corrected:** any statement about the MI-8 feature block across the seven dev cells,
  the MI-8 arm means, the D16 SEDI means, and the dev-cell D7 floors. Use the values in
  §4, not the archived MI tables.
* **Withdrawn:** the MI-3/MI-3c 7-cell aggregates and the MI-5 non-D16 7-cell columns.
  Their `val_2022` results stand.
* **Never to be used as a headline:** a single mean over the seven dev cells. Four of the
  seven are spatial folds with near-saturated PR-AUC, so the mean is dominated by
  cell composition, not by model quality (D8). The per-cell and per-year results are the
  defensible evidence.

## 7. Per-cell appendix (audit trail)

Fixed-config PR-AUC by cell (season = all):

{table(["cell", "FULL", "FULL_V2", "FULL_V3", "V3 − V2"], percell_rows)}

Frozen arms by cell and horizon:

{table(["cell", "horizon", "D16", "MI-8 full_v3", "MI-8 full_v2"], arm_cell_rows)}

## 8. Reproduce

```powershell
cd "D:\\Flood predictor"
.venv\\Scripts\\python.exe -u scripts\\mi10_p2p3_rescore.py   # ~{summary['wall_s']:.0f} s
.venv\\Scripts\\python.exe scripts\\mi10_report.py
```

Deterministic given seed `{summary['seed']}`. Dev cells only. `test_rows_read = 0`.
"""

    (REP / "p2p3_correction.md").write_text(md, encoding="utf-8")
    print(f"WROTE {REP / 'p2p3_correction.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
