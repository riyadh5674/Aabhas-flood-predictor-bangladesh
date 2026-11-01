# MI-6 — the single logged frozen-test read (decision D16)

| | |
| --- | --- |
| **Stage** | MI-6 (improvement plan §MI-6) — one frozen-test read |
| **Model read** | `hpo_perhorizon_FULL_V2` (MI-5 selected, decision **D16**) + the **D7 persistence** bar |
| **Command** | `.venv\Scripts\python.exe -u scripts\mi6_test_read.py` |
| **Run log** | `scripts/mi6_run.log` |
| **Exit / wall** | `0` / **119 s** |
| **Seed** | `20261008` |
| **Reads** | **1** (`n_reads = 1`) — guarded; `--reread` required to repeat |
| **Rows scored** | 1,198,056 (= 2 models × 3 horizons × 7 cells); **101,174 distinct test rows** |
| **Artefacts** | `reports/mi6_test_metrics.csv` · `mi6_test_read_log.json` · `mi6_thresholds.csv` · `mi6_summary.json` |

> **Programme-level disclosure (2026-10-22).** `n_reads = 1` is local to this script. The
> frozen window had already been opened by Phase 16 and Phase 33, and was later opened by
> MI-4 (invalid, retracted) and MI-9 (pre-registered correction, which reproduced this read
> **bit-for-bit**). See `reports/test_read_ledger.md`: six opens, five predictive reads.
> The numbers in this file are the valid D16 test result and are the ones the paper quotes.
> Do **not** describe the test as "read exactly once" at programme level.

## 1. What this stage does

Mi-6 opens the frozen test period (`split == "test"`, `>= 2023`) **exactly once**, for the
MI-5/D16 model and the D7 bar only, scoring with the project's one scoring path
(`evaluate_all.score`). It is guarded: if the metrics + log already exist the read is
reused, and the log records `n_reads`.

- **Thresholds.** D7 carries the Phase-29 period-specific cost-optimal thresholds verbatim;
  the D16 model has no Phase-29 entry, so its period thresholds are derived on **`val_2022`
  only** with the identical method (`phase29_thresholds.best_exact`, `C_FN = 10`, `C_FP = 1`)
  and frozen to `reports/mi6_thresholds.csv`. No test label picks a threshold.
  All-season: D16 `t = 0.103 / 0.053 / 0.082` (24/48/72 h); D7 `t = 0.259 / 0.170 / 0.225`.
- **Reproducibility cross-check.** The D7 test read reproduces the historical Phase-16 read
  **exactly** — max |Δ PR-AUC| = **5.6e-17**, max |Δ Brier| = 9.9e-17 over 63 shared rows —
  proving this read is comparable to the earlier one.

## 2. Headline — all-season PR-AUC on the frozen test

| cell | **D16** 24 / 48 / 72 h | Phase-33 `xgboost` (horizon-mean) | D7 (horizon-mean) | D16 − D7 | D16 − P33 |
| --- | --- | ---: | ---: | ---: | ---: |
| `test_2023` | 0.7956 / 0.6814 / 0.6257 | 0.6767 | 0.5636 | **+0.1373** | **+0.0242** |
| `test_2024` | 0.9177 / 0.8573 / 0.8043 | 0.8465 | 0.7350 | +0.1247 | +0.0132 |
| `test_2025` | 0.8048 / 0.7598 / 0.7019 | 0.7592 | 0.6680 | +0.0875 | −0.0037 |
| `test_2026` | 0.7935 / 0.7467 / 0.7023 | 0.7351 | 0.6552 | +0.0923 | +0.0124 |
| `test_trained` | 0.8686 / 0.8091 / 0.7614 | 0.8012 | 0.6805 | +0.1326 | +0.0118 |
| `test_new_network` | 0.8434 / 0.7868 / 0.7129 | 0.7721 | 0.6914 | +0.0896 | +0.0090 |
| `test_unseen` | 0.8299 / 0.7416 / 0.6410 | 0.7012 | 0.6792 | +0.0583 | +0.0363 |
| **mean of 7 cells** | — | **0.77076** | 0.75601 | 0.66755 | **+0.1032** | **+0.01475** |

Per-horizon, averaged over the 7 cells:

| horizon | D16 | Phase-33 `xgboost` | D7 |
| --- | ---: | ---: | ---: |
| 24 h | **0.83623** | 0.80624 | 0.73396 |
| 48 h | **0.76896** | 0.75255 | 0.66252 |
| 72 h | 0.70709 | **0.70923** | 0.60617 |

- **D16 beats D7 on all 21 test points** (7 cells × 3 horizons, all-season).
- **D16 beats the Phase-33 `xgboost` on 6 of 7 cells.** The one loss is `test_2025`
  (−0.0037), and at **72 h** the two are a statistical wash (D16 −0.0021 overall).

## 3. Operational behaviour (all-season, chosen thresholds)

Pooled over the 7 test cells at the per-horizon cost-optimal threshold:

| horizon | recall | precision | FAR | missed flood rows | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | --- |
| 24 h | 0.918 | 0.549 | 0.451 | 234 | 2616 / 2152 / 234 |
| 48 h | 0.903 | 0.388 | 0.612 | 342 | 3194 / 5030 / 342 |
| 72 h | 0.856 | 0.351 | 0.649 | 602 | 3586 / 6624 / 602 |

The 10:1 cost stance (missing a flood ≫ false alarm) buys high recall at the price of many
false alarms — the expected operating point, unchanged in spirit from Phase 29/33. Full
per-cell × season rows (incl. monsoon / non-monsoon) are in `mi6_test_metrics.csv`.

## 4. What the MI-v2 programme actually bought on the frozen test

| | val_2022 horizon-mean (selection) | test, mean of 7 cells (horizon-mean) |
| --- | ---: | ---: |
| Phase-33 `xgboost` (v1) | 0.95463 | 0.75601 |
| **MI-5/D16 model (v2)** | **0.95743** | **0.77076** |
| Δ | **+0.00280** | **+0.01475** |

The validation-side gain **carries to the test**, and is in fact larger there (+0.0148) — but
it is **concentrated at 24–48 h**; at 72 h the v2 model is on par with (marginally behind)
the Phase-33 model.

## 5. Honest read (programme close-out)

- The MI-v2 model is a genuine, consistent improvement — it clears the D7 bar on every test
  point and edges the Phase-33 winner on 6/7 test cells — but the **absolute test numbers
  (~0.62–0.92) remain far below the ~0.96 validation and the owner's 0.99 aspiration**.
- This is exactly what **MI-4** predicted: the test years have much lower base rates
  (`test_2023` 24 h ≈ 0.7 %) and 4–10 % near-threshold label noise, so a uniform 0.99 is
  **not reachable**. The honest ceiling is model-and-year dependent, not a single number.
- The largest **unused** lever is still the **ERA5-Land block** (`e5_*`), now 123/133 stations
  (block activates at ≥130). Landing the last stations and re-running MI-3a/b/MI-5 would
  price it in.
- The frozen test has now been read **once** for D16 (this stage). No further test reads
  should occur without a new, pre-registered decision.

## 6. Reproduce

```
cd "D:\Flood predictor"
.venv\Scripts\python.exe -u scripts\mi6_test_read.py            # the one read (guarded)
.venv\Scripts\python.exe -u scripts\mi6_test_read.py --dry-run  # thresholds only, no test
```

Exit `0`; deterministic given seed `20261008`; `n_reads = 1`.
