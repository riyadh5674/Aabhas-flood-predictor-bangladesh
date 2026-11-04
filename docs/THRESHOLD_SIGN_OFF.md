# Aabhas Flood Predictor — Threshold Sign-Off Document

**Version:** 1.0  
**Date:** 2026-11-04  
**Status:** DRAFT — For Review

---

## 1. Purpose

This document records the formal review and adoption of flood risk thresholds used by the Aabhas Flood Predictor for operational alerting in Bangladesh. These thresholds were originally derived in Phase 29 of the research workflow using cost-loss analysis on historical data (2017–2025).

**Important:** The Aabhas system is a **research prototype** and **not an official warning service**. FFWC advisories always take precedence. This sign-off is for adopting thresholds as **decision-support** inputs, not as official warning criteria.

---

## 2. Current Thresholds (Phase 29, Frozen D16 Model)

| Horizon | Watch (moderate) | Warning (high) | High-Risk (very_high) |
|---------|------------------|----------------|----------------------|
| 24h     | P ≥ 0.18         | P ≥ 0.35       | P ≥ 0.62             |
| 48h     | P ≥ 0.15         | P ≥ 0.30       | P ≥ 0.55             |
| 72h     | P ≥ 0.12         | P ≥ 0.25       | P ≥ 0.48             |

*Thresholds are probability values from the D16 model (LightGBM for 24h/48h, XGBoost for 72h).*

---

## 3. Derivation Summary

- **Method:** Cost-loss analysis with asymmetric costs (missed event vs. false alarm)
- **Data:** Independent test set (2022–2025), 127 stations, 2,615 feature dates
- **Validation:** Phase 43 evaluation (reliability diagrams, Brier scores, ROC-AUC)
- **Calibration:** Isotonic regression per horizon

| Horizon | Brier Score | ROC-AUC | Reliability (ECE) |
|---------|-------------|---------|-------------------|
| 24h     | 0.087       | 0.89    | 0.023             |
| 48h     | 0.092       | 0.86    | 0.031             |
| 72h     | 0.098       | 0.82    | 0.041             |

---

## 4. Operational Implications

| Alert Level | Probability Range | Recommended Action |
|-------------|-------------------|-------------------|
| **Watch**   | Moderate risk     | Heightened monitoring, pre-position resources |
| **Warning** | High risk         | Prepare response, issue advisories to communities |
| **High**    | Very high risk    | Activate emergency protocols, consider evacuations |

**Cooldown:** 24 hours (bypassed on escalation)

---

## 5. Known Limitations

1. **Research prototype** — not validated for official warning use
2. **Training data** ends 2025; climate non-stationarity not accounted for
3. **Null feature incident** (Oct–Nov 2026): 37/110 features were null for ~3 weeks due to source fetchers stopping at 2026-10-12. Fixed but evaluation delta not yet quantified.
4. **Station coverage:** 127/153 FFWC stations have sufficient history
5. **Lead time:** 24h/48h/72h horizons; no sub-daily updates

---

## 6. Sign-Off Required

| Role | Name | Signature | Date | Comments |
|------|------|-----------|------|----------|
| **Technical Lead** | | | | |
| **Hydrology Expert** | | | | |
| **FFWC Representative** | | | | |
| **Disaster Management Authority** | | | | |
| **Legal/Compliance** | | | | |

---

## 7. Review Schedule

- **Initial review:** Upon deployment
- **Quarterly review:** First year of operation
- **Annual review:** Thereafter
- **Trigger review:** After any major flood event or model retraining

---

## 8. Amendment Procedure

Changes to thresholds require:
1. Technical justification with updated evaluation metrics
2. Review by all signatories
3. Version increment and documentation update
4. Communication to all operational users

---

*This document is part of the Aabhas Flood Predictor governance framework. See also: Duty Officer Process, Legal Disclaimer Review.*