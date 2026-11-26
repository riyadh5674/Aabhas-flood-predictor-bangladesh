# Aabhas Flood Predictor — Threshold Sign-Off

**Version:** 2.0  
**Date:** 2026-11-10  
**Status:** **SIGNED — internal research prototype only**  
**Supersedes:** v1.0 draft (2026-11-04), whose threshold table and metrics were stale.  
**Scope:** D16 frozen model, 24/48/72-hour station-level probabilistic flood outlook.

---

## 1. Purpose and scope

This document records the review and adoption of the probability thresholds and risk bands used
by the Aabhas Flood Predictor. The thresholds convert the frozen D16 model's probability into
four decision bands (low, moderate, high, very high) and drive the prototype alert workflow.

**This is a research-prototype sign-off, not an operational warning-service approval.**

- Aabhas is **not** an official flood-warning service.
- FFWC advisories and government instructions always take precedence.
- The signed decision below authorises the thresholds for **research, demonstration and
  decision-support** use only.
- External delivery remains disabled (`notification.enabled = false`). No public alert is sent
  by the system.
- An agency / named duty-officer sign-off, delivery credentials and a legal review are still
  required before any operational use.

---

## 2. Adopted thresholds

Source of truth: `system/config/model_config.json` (`thresholds.values`), derived from
`reports/phase29_chosen_thresholds.csv` and locked in the frozen evidence manifest.

| Horizon | Moderate / watch (P ≥) | High / warning (P ≥) | Very high / high-risk (P ≥) | Low |
| --- | ---: | ---: | ---: | --- |
| **24 h** | 0.0303 | 0.0605 | 0.4340 | P < 0.0303 |
| **48 h** | 0.0434 | 0.0868 | 0.5630 | P < 0.0434 |
| **72 h** | 0.0482 | 0.0963 | 0.5277 | P < 0.0482 |

Per-horizon model family (D16, `hpo_perhorizon_FULL_V2`, FULL_V2 110 features):

| Horizon | Family |
| --- | --- |
| 24 h | LightGBM |
| 48 h | LightGBM |
| 72 h | XGBoost |

The cuts are probability values from the frozen D16 model, not water levels. The frontend,
API, alert engine and static snapshot all read the same configuration.

### 2.1 Band-to-action mapping

| Band | Alert level | Prototype action |
| --- | --- | --- |
| **Low** | none | Routine monitoring |
| **Moderate** | watch | Heightened monitoring; check freshness and upstream sources |
| **High** | warning | Prepare response; draft advisory for human review |
| **Very high** | high | Activate review; draft high-risk advisory for human approval |

Alerts are **draft → human approval → dispatch → hash-chained audit**. Cooldown is 24 hours,
with escalation bypass. Nothing is dispatched automatically to the public.

---

## 3. Derivation

- **Decision method:** row-level cost-loss analysis with asymmetric costs,
  **C_FN : C_FP = 10 : 1** (a missed flood is treated as ten times more costly than a false
  alarm). Sensitivity at 5:1 and 20:1 was computed in Phase 29.
- **Selection period:** the pre-registered development cell `val_2022`, all seasons, using the
  frozen D16 out-of-fold probabilities. The test split was **not** used to choose thresholds.
- **Sweep:** probability grid 0.01–0.60; exact cost argmin per horizon.
- **Band derivation:** watch = ½ · `t_warn`; warning = `t_warn`; very high = the cut where
  precision reaches ≈ 0.90. The 48 h and 72 h bands use the documented four-level scheme so
  that `very_high` is represented consistently in the config, API, UI and snapshot.
- **Freeze:** thresholds are part of the frozen D16 artefact and the evidence manifest
  (`reports/FREEZE.md`). Changing a cut is a new version, not an edit.

### 3.1 Validation performance at the adopted cuts (D16, `val_2022`)

| Horizon | `t_warn` | Recall | Precision | F1 | FAR | Alarm rate | Missed rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 h | 0.0605 | 0.980 | 0.746 | 0.847 | 25.4 % | 6.7 % | 18 |
| 48 h | 0.0868 | 0.967 | 0.727 | 0.830 | 27.3 % | 7.9 % | 34 |
| 72 h | 0.0963 | 0.954 | 0.684 | 0.796 | 31.6 % | 9.2 % | 52 |

These are development-cell numbers used for threshold selection, not a test claim.

### 3.2 Frozen-test behaviour (the valid MI-6 read, for context only)

The test split is **closed**. The programme-level ledger (`reports/test_read_ledger.md`)
records **six opens — five predictive reads (one invalid and retracted) plus one
label-statistics read**. The reported D16 headline is the valid MI-6 read; it was not used to
re-tune these thresholds.

| Horizon | Recall | Precision | FAR | Missed flood rows (D16 / D7) |
| --- | ---: | ---: | ---: | ---: |
| 24 h | 0.918 | 0.549 | 0.451 | 234 / 474 |
| 48 h | 0.903 | 0.388 | 0.612 | 342 / 762 |
| 72 h | 0.856 | 0.351 | 0.649 | 602 / 1,242 |

The high false-alarm rate is the deliberate consequence of the 10:1 cost stance. It is
acceptable for a human-reviewed prototype; it would need explicit agency acceptance before
operational use.

---

## 4. Evidence reviewed

| Evidence | Role |
| --- | --- |
| `reports/phase29_thresholds.md`, `reports/phase29_chosen_thresholds.csv` | Cost-loss derivation and sensitivity |
| `reports/phase29_risk_bands.csv` | Band lift and observed flood rate |
| `reports/FREEZE.md` | Frozen artefact hashes and headline numbers |
| `reports/mi6_test_read.md`, `reports/mi6_test_metrics.csv` | Valid D16 test read and operational behaviour |
| `reports/test_read_ledger.md` | Programme-level test-read disclosure |
| `reports/phase43_final_evaluation.md` | Validity assessment |
| `system/config/model_config.json` | Deployed cuts and band definitions |
| `system/config/alert_config.json` | Alert levels, cooldown, notification state |
| `tests/run_tests.py` | 89 Python contract/regression tests |
| `scripts/freeze_results.py --check` | 13/13 frozen-evidence checks |

---

## 5. Known limitations and conditions

1. **Research prototype.** Not validated for official warning use. No legal or agency mandate.
2. **Training period ends 2021.** Validation is 2022; the frozen test is 2023–2026. The model
   is not retrained on test data, so climate and land-use non-stationarity are not represented
   in the fitted parameters.
3. **High false-alarm rate.** 45–65 % on the frozen test at the adopted cuts, by design of the
   10:1 cost ratio. Human review is mandatory.
4. **Station coverage.** The modelling network has 133 FFWC stations; the latest public
   snapshot has 81 stations with predictions on the latest feature date and 127 in the 90-day
   trend window. Coverage is not the full national network.
5. **Lead time.** 24/48/72 h only; no sub-daily update. Freshness depends on upstream sources
   and is surfaced in `/health` and the UI banner.
6. **Data incident.** The October–November 2026 silent-null incident is fixed and guarded by
   the quality gate (`reports/phase45_thesis.md` §7). It did not change the frozen model or
   these thresholds.
7. **Delivery disabled.** No SMTP/webhook recipient is configured. The alert lifecycle is
   implemented and tested, but no public message is sent.
8. **External sign-off absent.** The signatures in §6 are internal research sign-offs. Agency,
   hydrology and legal sign-off remain blank and are required before operational use.

---

## 6. Sign-off

### 6.1 Internal research-prototype sign-off (effective 2026-11-10)

| Role | Name / entity | Decision | Signature | Date |
| --- | --- | --- | --- | --- |
| Research programme owner | Aabhas project (internal) | **APPROVED** for research-prototype decision support only | Aabhas research programme | 2026-11-10 |
| Model owner | D16 frozen artefact (`models/final_v2/selected_model_v2.pkl`) | **APPROVED** as the threshold-bearing model | D16 freeze record | 2026-11-10 |
| Threshold reviewer | Phase-29 cost-loss analysis + Phase-43 validity review | **APPROVED** with limitations in §5 | Phase-29 / Phase-43 evidence | 2026-11-10 |

**Decision:** The thresholds in §2 are adopted for the Aabhas research prototype. They may be
used in demonstrations, research, and human-reviewed decision support. They may **not** be
presented as official warnings or used for automatic public alerting.

### 6.2 External / operational sign-off (PENDING — required before operational use)

| Role | Name | Signature | Date | Conditions |
| --- | --- | --- | --- | --- |
| Technical lead (external) | | | | |
| Hydrology expert | | | | |
| FFWC representative | | | | |
| Disaster-management authority | | | | |
| Legal / compliance | | | | |

No external signature is claimed. The system must remain labelled a research prototype until
these are completed.

---

## 7. Review schedule

- **Current review:** 2026-11-10 (internal research prototype).
- **Before any operational pilot:** full external review, delivery credentials, and a named
  duty officer.
- **Quarterly:** review source freshness, false-alarm burden, and incident log during any pilot.
- **After any model change or retraining:** new freeze, new threshold derivation, new version.
- **After a major flood event:** review missed events and false alarms; record findings without
  re-opening the frozen test.

---

## 8. Amendment procedure

1. Open a new version of this document; never edit a signed version in place.
2. State the technical justification, the affected horizon(s), and the new validation evidence.
3. Re-run `python scripts/freeze_results.py --check`; update the freeze manifest if the model
   or thresholds change.
4. Obtain internal research sign-off and, for operational use, all external signatures.
5. Update `system/config/model_config.json`, the API, the frontend fallback and the static
   snapshot together; verify that all four agree.
6. Record the change in `CHANGELOG.md` and communicate it to every user of the thresholds.

---

*Part of the Aabhas governance framework. See also `docs/DUTY_OFFICER_PROCESS.md`,
`docs/FREE_TIER_OPERATIONS.md`, `DEPLOYMENT.md`, and `reports/test_read_ledger.md`.*
