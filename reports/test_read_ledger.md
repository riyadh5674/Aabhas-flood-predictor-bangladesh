# Frozen-test read ledger (project-level)

**Created:** 2026-10-16 · **Updated:** 2026-10-22 · **Source of truth:** `reports/test_read_ledger.json` (machine copy) + this file.
**MI-7, Track 1; reconciled after MI-4/MI-9 and the P2/P3 correction.**

**Frozen test** = `split == "test"`, dates `>= 2023-01-01` (`TEST_CUTOFF`), years 2023–2026.
**Distinct test rows:** 101,174. **Scored rows per full read:** 1,198,056 (model × horizon × cell — *not* rows).

---

## Why this file exists

The project repeatedly states that the frozen test is **"read exactly once"**:

- `scripts/phase33_selection.py:8-10` — *"The frozen test period (`>= 2023`) is read exactly once"*
- `scripts/mi6_test_read.py:1-16` — the same claim
- `reports/mi_status.md` — *"the test is read exactly once, in MI-6"*

Each statement is true **of that one script's own guard file** (`..._test_read_log.json`, `n_reads = 1`).
**No guard can see a different script's read.** At the programme level the frozen window has been
opened **six** times: **five predictive reads** (one of them invalid and retracted) plus one
label-statistics read. This ledger is the authoritative count.

---

## Chronology

| # | When (UTC) | Local (UTC+6) | Phase | Kind | Models | Status |
|---|---|---|---|---|---|---|
| 1 | 2026-10-09 (mtime) | 2026-10-09 01:18 | **Phase 16** | baseline scoring | 6 L1 baselines | valid, pre-freeze |
| 2 | 2026-10-15T05:31:10Z | 2026-10-15 11:31 | **Phase 33** | selection read | `xgboost` + `persistence` | valid; is the Phase-33 freeze |
| 3 | 2026-10-15T23:34:38Z | 2026-10-16 05:34 | **MI-6** | champion read | `hpo_perhorizon_FULL_V2` (D16) + `persistence` | **valid; reported headline** |
| 4 | 2026-10-16 (log) | 2026-10-16 04:54 | **MI-4** | label statistics only | *(none — no predictions)* | valid, no model score |
| 5 | 2026-10-21T16:05:57Z | 2026-10-21 22:05 | **MI-4 / D17** | comparison read | V4 hybrid + D16 + D7 | **INVALID — retracted by D18** |
| 6 | 2026-10-21T17:26:28Z | 2026-10-21 23:26 | **MI-9** | correction read | D16 + V4 hybrid + D7 | **valid; D16 reproduced bit-for-bit** |

Freeze points: Phase-33 model frozen `2026-10-15T05:31:10Z`; D16 frozen `2026-10-15T23:05:47Z`
(27 min before the MI-6 read).

**Numbering note.** `reports/mi9_test_read_decision.md` calls itself "read #3" because it counts
only the D16/hybrid reads. In the programme ledger it is **open #6 / predictive read #5**.
This file supersedes the internal numbering.

---

## What each read may be used for

1. **Phase 16 (open #1)** is a pre-freeze baseline comparison with different models. It is
   context, never a comparison for D16.
2. **Phase 33 (open #2)** selected the v1 incumbent. It is the reason the programme-level
   "single read" claim is already false before MI-6.
3. **MI-6 (open #3)** is the reported D16 test result: horizon-mean **0.77076**, per-horizon
   **0.83623 / 0.76896 / 0.70709**, D7 **0.66755**. It is fully logged and is the number the
   paper quotes.
4. **MI-4 / D17 (open #5)** is **not usable**. `reports/d17_comparability_audit.md` shows it
   applied a complete-case filter (31.5 % of test rows retained, 1.5–1.8× higher base rate) and
   a non-Phase-15 `test_new_network` mask. Its 0.7983 headline is **retracted**. It must not be
   quoted, plotted, or compared with D16.
5. **MI-9 (open #6)** was pre-registered in the D17 audit. Its sanity gate re-scored D16 and
   reproduced MI-6 **bit-for-bit** (max |Δ PR-AUC| = 0, 63 shared rows). It also re-scored the
   V4 hybrid under the MI-6 protocol: **0.770657 vs D16 0.770757 (Δ = −0.00010)**. **D18:
   D16 remains champion; D17 is retracted.** MI-9 is a reproducibility read, not a new
   selection opportunity.

---

## The uncomfortable part (stated plainly)

1. **The test was not pristine at the Phase-33 freeze.** Phase 16 already scored it on
   2026-10-09 — documented and intended (`phase17_experiment_registry.csv`:
   `test_policy = read_once_phase16`), but real.
2. **A second predictive read happened (MI-6) 18 h after Phase 33.** It is fully logged, but it
   is **test-informed**: the MI-v2 programme was launched *because* the Phase-33 test number
   dropped to 0.71–0.80 (`reports/improvement_plan.md:40`).
3. **A third predictive read (MI-4) was invalid**, and a fourth (MI-9) was needed to retract it.
   The V4 hybrid's apparent +0.065 test gain was an artefact of row filtering. D16 is the
   champion by a pre-registered rule, not because its test number was the largest observed.
4. **The D16 headline is the MI-6 read, not the maximum of several reads.** The paper must say
   this explicitly. "D16 0.77076" is valid; "the best test result we observed" is not.
5. **`test_rows_read` is a misleading name** — it counts scored rows (1,198,056), not window
   opens, and is per-script.

---

## Operating rule going forward

- **The frozen test is now CLOSED for this paper.** No further read, re-score, event analysis,
  calibration fit, or figure may touch `split == "test"`. The event-level evaluation is
  development-only for exactly this reason.
- Any future test read must be **pre-registered**, **added to this ledger**, and reported as
  *an additional read* — the "exactly once" wording must not be reused.
- New scripts must consult this ledger rather than assert a local `n_reads = 1` as if it were
  global.
- The paper's Limitations section carries the disclosure: the frozen window was opened before
  the MI-v2 model existed, so the test is frozen for D16 but not pristine for the programme.

This correction makes the project's single-read claim weaker in the honest direction. It is
recorded because a warning system people can trust must be able to show its reasoning —
including the reasoning about its own evidence.
