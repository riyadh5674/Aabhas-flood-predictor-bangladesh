# MI-9 — corrected frozen-test read: **decision D18**

**Date:** 2026-10-21
**Read:** test read **#3** (project-wide), `scripts/mi9_test_read_corrected.py`
**Supersedes:** `reports/mi4_test_read_decision.md` (decision D17)
**Verdict:** **D17 is RETRACTED. D16 stands as champion.**

> **Ledger note (2026-10-22).** "Test read #3" here counts only the D16/hybrid reads. In the
> programme ledger (`reports/test_read_ledger.md`) this is **open #6 / predictive read #5**;
> the ledger supersedes this internal numbering. The sanity gate below is the authoritative
> proof that MI-6 and MI-9 are the same fit.

---

## 1. Why a third read was necessary

`reports/d17_comparability_audit.md` documented that the MI-4 read applied a
complete-case filter (31.5 % of test rows retained, with a 1.5–1.8× higher base
rate) and used a non-Phase-15 `test_new_network` mask. The D17 gains were
therefore not measurable.

The correction was **pre-registered before running** (audit §4, four rules:
sanity gate, horizon-mean at every horizon, ≥ 20/21 vs D7, bootstrap on
`test_2024`).

## 2. Sanity gate — PASSED exactly

Re-scoring the frozen D16 model under the MI-9 code reproduced the MI-6 metrics
bit-for-bit:

| Check | Result |
| --- | --- |
| shared (cell × horizon × season) rows | 63 |
| max \|Δ PR-AUC\| | **0.000e+00** |
| max \|Δ Brier\| | 9.9e-17 |
| max \|Δ n\| | 0 |
| tolerance | 1e-9 → **PASS** |

This is the proof that the corrected read is on the *same protocol* as MI-6, and
therefore that the D16-vs-hybrid comparison below is valid.

## 3. Result — the D17 "gain" does not exist

Horizon-mean PR-AUC over the 7 test cells (`season=all`):

| Horizon | D16 | D17 hybrid | Δ |
| --- | --- | --- | ---: |
| 24 h | **0.836226** | 0.835923 | **−0.00030** |
| 48 h | 0.768958 | 0.768958 | 0.00000 (identical model) |
| 72 h | 0.707088 | 0.707088 | 0.00000 (identical model) |
| **Horizon-mean** | **0.770757** | 0.770657 | **−0.00010** |

The 48 h and 72 h arms are *the same frozen D16 pipelines* in both models, so
only the 24 h arm can differ — and it is a wash. The claimed **+0.065 / +0.064 /
+0.078** was entirely an artefact of scoring a favourable 31.5 % subset.

### 24 h per-cell detail

| Cell | D16 | D17 hybrid | Δ |
| --- | ---: | ---: | ---: |
| `test_2023` | 0.7956 | 0.7830 | **−0.0126** |
| `test_2024` | 0.9177 | 0.9166 | −0.0011 |
| `test_2025` | 0.8048 | 0.7944 | **−0.0104** |
| `test_2026` | 0.7935 | 0.8110 | **+0.0175** |
| `test_trained` | 0.8686 | 0.8644 | −0.0042 |
| `test_new_network` (4,805 rows) | 0.8434 | 0.8369 | −0.0065 |
| `test_unseen` | 0.8299 | 0.8452 | **+0.0153** |

So the only genuine V4 gains are `test_2026` (+0.0175) and `test_unseen`
(+0.0153) — not the "+0.13 to +0.20" that was claimed. The bootstrap on
`test_2024` (24 h) gives **−0.0013 [−0.0061, +0.0031], P(Δ>0) = 0.283**, i.e.
no evidence for the hybrid.

## 4. Decision

> ### D18 — the V4 feature block is **not adopted**.
> `models/final_v2/` (D16, `hpo_perhorizon_FULL_V2`) is the champion.
> `models/final_v3/` remains on disk, marked **RETRACTED**, and must not be
> deployed.

| Rule | Result |
| --- | --- |
| R1 sanity gate | ✅ PASS (exact) |
| R2 hybrid ≥ D16 horizon-mean at every horizon | ❌ **FAIL** (24 h −0.0003) |
| R3 hybrid beats D7 on ≥ 20/21 | ✅ 21/21 |
| R4 bootstrap on `test_2024` favours hybrid | ❌ FAIL (P = 0.283) |

R4 is partly degenerate by construction (identical pipelines give a zero
bootstrap difference at 48/72 h), but R2 already fails on the only arm that can
differ. The decision is unambiguous.

## 5. What this means for the programme

1. **The V4 feature work is closed as a null result.** MI-8 (+0.00058) and MI-4
   (≤ 0.001 under a fair HPO) both agree: adding WorldCover + `m8_*` columns to
   `FULL_V2` does not move the champion.
2. **The 2017–21 / 2025–26 gap is not a feature problem.** D16 loses nothing to
   a 187-feature model on the same protocol. This strengthens the case for the
   *label / data-source* experiments (Experiment 1 domain adaptation,
   Experiment 2 SAR labels, Experiment 3 NWP ensembles) over more feature
   engineering.
3. **Test-read accounting:** three reads have now been consumed (MI-6 for D16,
   MI-4 invalid, MI-9 correction). The next read requires a new pre-registered
   decision.
4. **Process fix:** `mi4_test_read.py` is quarantined. It must not be re-run.
   `scripts/mi9_test_read_corrected.py` is the reference implementation for any
   future frozen read.

## 6. Artefacts

| File | Contents |
| --- | --- |
| `reports/mi9_test_metrics.csv` | 3 models × 3 horizons × 7 cells × 3 seasons |
| `reports/mi9_headline.csv` | season=all pivot with Δ vs D16 and vs D7 |
| `reports/mi9_test_thresholds.csv` | cost-optimal thresholds (val_2022 only) |
| `reports/mi9_sanity.json` | the D16 == MI-6 proof |
| `reports/mi9_summary.json` | the four pre-registered rules + bootstrap |
| `reports/mi9_test_read_log.json` | read #3 log, `project_test_reads_total = 3` |
