# Audit — is the D17 test comparison valid?

**Date:** 2026-10-21
**Verdict:** **NO. The D17 test read is invalid and the decision is suspended.**
**Severity:** high — it affected a frozen-model decision (`models/final_v3/`).

---

## 1. What was claimed

`reports/mi4_test_read_decision.md` claimed that the hybrid V4/D16 model
(decision **D17**) beat the D16 champion by a wide margin on the frozen test
period:

| Horizon | Claimed MI-4 hybrid | Claimed MI-6 D16 | Claimed Δ |
| --- | --- | --- | --- |
| 24 h | 0.8572 | 0.8061 | **+0.0511** |
| 48 h | 0.7932 | 0.7289 | **+0.0643** |
| 72 h | 0.7444 | 0.6662 | **+0.0782** |

and that the gains were largest on unseen stations (+0.13 … +0.20).

## 2. What is actually true

The two numbers were computed on **different rows, different cell definitions,
and different base rates**. They are not comparable.

### 2.1 Different row sets

`scripts/mi4_test_read.py` scored only rows with **no missing feature values**:

```python
valid = ~d[feats].isna().any(axis=1)
p_all[valid] = pipe.predict_proba(d.loc[valid, feats])[:, 1]
```

`FULL_V4` has 187 features, and the `m8_*` / `up_*` blocks are NaN for most
rows (they are only defined where an upstream neighbour and a lag exist). The
result:

| | rows |
| --- | --- |
| all test rows | 101,174 |
| test rows with a complete `FULL_V4` vector | **≈ 30 %** |

So MI-4 scored roughly **one third** of the test set, while MI-6 scored **all**
rows. The MI-4 HPO (`scripts/mi4_hpo_v4.py`) and the MI-6 read both use the
pipeline's median imputer and score every row — only the MI-4 *test read* (and
the MI-4 *selection scorecard*) applied the complete-case filter.

### 2.2 The retained subset is easier

Dropping rows is not neutral: it correlates with the label. The retained rows
have systematically higher base rates, and **PR-AUC rises mechanically with the
base rate** in this range.

| Cell (24 h) | MI-4 rows | MI-4 base rate | MI-6 rows | MI-6 base rate |
| --- | --- | --- | --- | --- |
| `test_2023` | 6,949 | 0.0076 | 23,858 | 0.0068 |
| `test_2024` | 7,468 | **0.0455** | 24,339 | **0.0254** |
| `test_2025` | 9,199 | 0.0095 | 28,250 | 0.0117 |
| `test_2026` | 8,243 | 0.0082 | 24,689 | 0.0127 |
| `test_unseen` | 5,567 | **0.0180** | 19,761 | **0.0103** |

`test_2024` was scored on a subset with a **1.8× higher** positive rate than the
full cell. That alone is worth a large fraction of the claimed "+0.05".

### 2.3 `test_new_network` was a different cell entirely

| | definition | rows |
| --- | --- | --- |
| MI-6 (Phase-15) | `first_year > 2023` (stations that did not exist at model freeze) | 4,805 |
| MI-4 | `year >= 2023 & ~station_holdout` — i.e. **almost the entire test set** | 26,292 |

The claimed "+0.0434 on new network" compares a 26 k-row cell with a 4.8 k-row
cell. It carries no information.

### 2.4 Test-read guard counter

The guard in `mi4_test_read.py` incremented `n_reads` on its own file and
recorded "2 reads" project-wide, but the invalid read still **consumed a test
read**. That cannot be undone — the correct response is to log the correction
explicitly (read #3) rather than pretend it did not happen.

## 3. What about the MI-4 *selection*?

Mixed, but salvageable:

| Artefact | Filter | Status |
| --- | --- | --- |
| `mi4_hpo_v4.py` (HPO, val_2022 objective) | **no filter** — imputes and scores all rows | **valid** |
| `mi4_select_v4.py` (scorecard + bootstrap) | complete-case | **invalid** |
| `mi4_test_read.py` (test read) | complete-case | **invalid** |

The HPO objective is therefore trustworthy: `lgbm13` really did reach
`val_2022 = 0.9713` on all val rows. But the *bootstrap CIs* and *per-cell
scorecard* that justified "adopt V4 at 24 h only" were computed on the
complete-case subset, so the "clear win with 94 % bootstrap support" is unproven.

Independently, the HPO showed:

* 24 h: V4 winner 0.9713 vs D16 0.9708 → **+0.0005** (not +0.005)
* 48 h: V4 winner 0.9573 vs D16 0.9565 → **+0.0008**
* 72 h: V4 winner 0.9457 vs D16 0.9450 → **+0.0007**

These are **marginal** differences (≤ 0.001), i.e. exactly the regime where a
leakage / protocol bug can flip the sign.

## 4. Actions taken

1. **D17 suspended.** `models/final_v3/` is retained on disk but is no longer
   the champion. The champion reverts to **D16** (`models/final_v2/`) until a
   corrected read says otherwise.
2. **Corrected read (test read #3)** specified in
   `scripts/mi9_test_read_corrected.py`, mirroring the MI-6 protocol exactly:
   * all rows with a non-NaN label, pipeline imputation (no complete-case filter);
   * the Phase-15 test-cell definitions (`test_2023 … test_2026`,
     `test_trained`, `test_new_network`, `test_unseen`);
   * `evaluate_all.score` (the single scoring path);
   * thresholds from `phase29_thresholds.best_exact` on val_2022, C_FN = 10;
   * seasons `all` / `monsoon` / `non-monsoon`;
   * a **sanity check**: re-scoring the frozen D16 model must reproduce
     `reports/mi6_test_metrics.csv` to < 1e-9, otherwise the read is aborted.
3. **Pre-registered decision rule** for read #3 (written before running):

   | # | Rule |
   | --- | --- |
   | 1 | Sanity check on D16 must pass (< 1e-9 vs MI-6) |
   | 2 | Hybrid must beat D16 on the 7-cell horizon-mean PR-AUC at every horizon |
   | 3 | Hybrid must beat D7 on ≥ 20 of 21 cells |
   | 4 | Paired station bootstrap on `test_2024` (the largest-base-rate year) must not exclude 0 against the hybrid |

   All four → D17 reinstated. Any failure → **D16 stays champion** and the V4
   contribution is recorded as inconclusive.

## 5. Lesson recorded for the programme

**Complete-case filtering is not a neutral preprocessing choice.** Any
evaluation that drops rows must report (a) the fraction of rows retained and
(b) the base rate of the retained subset, and must never be compared against a
result computed on all rows. This is added to the checklist in
`reports/advanced_experiments.md` and to the governing rules in
`reports/mi_status.md`.
