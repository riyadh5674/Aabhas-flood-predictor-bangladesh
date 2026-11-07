# Aabhas — Investor & Partner Brief

**Station-level probabilistic flood forecasting for Bangladesh.**
*Know your river. Plan ahead.*

**Live demo:** https://riyadh5674.github.io/Aabhas-flood-predictor-bangladesh/
**Repository:** https://github.com/riyadh5674/Aabhas-flood-predictor-bangladesh
**Stage:** research-complete, working prototype, pre-revenue
**Contact:** open an issue on the repository or reach the maintainer via GitHub

> This is an honest brief. Aabhas has a validated model and a working product, but no
> paying customers yet. The next milestone is an institutional pilot, not a growth
> campaign. Everything below is traceable to the repository.

---

## 1. One-page summary

Bangladesh is one of the most flood-exposed countries on earth. Around **173 million
people** live in a delta where the Ganges, Brahmaputra and Meghna converge, and severe
monsoon flooding routinely displaces millions. Existing public warnings are largely
district-level and deterministic: they tell people *that* a district may flood, not
*which station*, *when*, or *how likely*.

Aabhas turns open hydrological data into a **station-level probability** that a river
crosses its danger level in the next **24, 48 or 72 hours**, with:

- a calibrated probability and an action-oriented risk band,
- a map and station pages designed for non-specialists,
- an approval-gated alert workflow with a hash-chained audit trail,
- a fully reproducible, leakage-audited evaluation.

The model is trained on **2,617 days of data across 153 FFWC stations** and validated
on a **frozen 2023–2026 test period** that was opened exactly once. It beats the
operational persistence baseline on **21/21** test points and reduces missed floods by
**58 %**.

**The ask:** an initial **US$150k–350k** grant/seed to convert the validated prototype
into a paid institutional pilot with one agency or NGO, plus the operational
infrastructure to run it reliably. Detailed use of funds in §7.

---

## 2. The problem

| Today | Consequence |
|---|---|
| Warnings are district-level | A household cannot tell whether *its* river gauge is at risk |
| Forecasts are binary or deterministic | No way to express uncertainty or trade off false alarms vs. missed events |
| Lead time is short | Evacuation, shelter opening and asset pre-positioning need 48–72 h |
| Data is fragmented | Rainfall, upstream discharge, soil moisture and gauge levels live in different systems |
| Trust is brittle | One silent data outage can degrade a forecast without anyone noticing |

The people who need this most — union councils, schools, clinics, NGOs, agricultural
cooperatives, logistics operators, insurers — are not served by a research paper or a
single national bulletin. They need a **local, probabilistic, explainable outlook**.

## 3. The product

**Aabhas** is a web application and API that answers one question at one place:

> *P(flood at station X within the next 24 / 48 / 72 h), with a risk band and the
> drivers behind it.*

### Product surfaces

1. **National overview** — current situation, horizon switch, freshness banner, top-risk stations.
2. **Risk map** — administrative choropleth plus station markers, filterable by band.
3. **Station detail** — probability trend, band history, model drivers, plain-language guidance.
4. **Alerts** — rule-based escalation with cooldown, human approval, and an immutable audit trail.
5. **Model page** — thresholds, calibration and evaluation evidence in plain sight.
6. **API** — documented FastAPI endpoints for integration with partner systems.
7. **Offline PWA** — bilingual (English / Bengali), installable, with an offline shell.

### What the user gets that they do not get today

| Capability | Why it matters |
|---|---|
| Station-level probability | Actionable at the level where decisions are made |
| 24 / 48 / 72 h horizons | Enough lead time to move people and assets |
| Explicit 10:1 miss-to-false-alarm cost | Thresholds reflect the real asymmetry of flood decisions |
| Explainability | Officers can sanity-check a warning before it goes out |
| Freshness and quality gates | Decision-makers can see when data is stale or degraded |
| Human-in-the-loop alerts | Institutional trust and accountability |

## 4. Evidence

The strongest asset is not a slide — it is a reproducible evidence base.

| Claim | Number | Where to verify |
|---|---|---|
| Validation PR-AUC (horizon mean) | **0.9574** | `reports/FREEZE.md`, `reports/mi6_summary.json` |
| Frozen test PR-AUC | **0.7708** | `reports/mi6_summary.json` |
| Persistence baseline (test) | 0.6675 | `reports/mi6_summary.json` |
| Test points beating persistence | **21 / 21** | `reports/mi6_summary.json` |
| Missed floods vs. persistence | **−58 %** | `reports/phase43_final_evaluation.md` |
| Out-of-fold calibration error | **0.0031** | `reports/phase33_selection.md` |
| Spatial robustness points vs. persistence | **63 / 63** | `reports/phase31_robustness.md` |
| Automated tests | **84 Python + 21 frontend** | `tests/`, `system/web` |
| Frozen evidence checks | **13 / 13** | `scripts/freeze_results.py --check` |
| Data coverage | **2,617 days**, **153 stations** | `system/data/processed/model_ready/` |

### Why the evaluation is trustworthy

- **Leakage-controlled splits:** 7-day embargo, station-disjoint spatial folds, banned
  label-window features asserted absent.
- **Frozen test set:** opened once in MI-6; every later analysis records
  `test_rows_read = 0`.
- **Fair ML vs. DL comparison:** identical features, budgets and seeds; deep learning
  did not beat the classical model, so the DL queue was closed.
- **Operational realism:** cost-derived thresholds, calibration, missingness and
  monsoon/non-monsoon robustness were tested explicitly.
- **Documented failure:** a hard-coded date in three fetchers silently degraded 42 of
  110 features for three weeks. It is documented, fixed, and now guarded by a mandatory
  quality gate. That story is evidence of engineering maturity, not a hidden flaw.

## 5. Market and go-to-market

### Who pays

| Segment | Pain | Product |
|---|---|---|
| **Government agencies** (FFWC, disaster management) | Need station-level decision support without replacing their mandate | API + dashboard licence, co-branded pilot |
| **NGOs / humanitarian agencies** (BRAC, IFRC, Oxfam, WFP) | Pre-positioning and anticipatory action require 48–72 h triggers | Alert feed + API, per-programme subscription |
| **Climate / development finance** | Adaptation programmes need measurable early-warning impact | Grant-funded pilots, impact reporting |
| **Agriculture & logistics** | Harvest, storage and route decisions depend on flood windows | Station-level API, seasonal subscription |
| **Insurance / index products** | Parametric triggers need transparent station data | Data licensing and audit trail |

### Go-to-market sequence

1. **Anchor pilot** — one agency or NGO, one basin, 6 months, defined success metrics.
2. **Evidence upgrade** — prospective evaluation: lead time, action rate, false-alarm cost.
3. **Integration** — API/SMS/webhook delivery into the partner's existing workflow.
4. **Expansion** — additional basins, then additional delta countries.
5. **Commercial layer** — subscriptions and data licensing once the pilot proves use.

Aabhas does not need to replace national warning services. It is a **decision-support
layer** that sits beside them, and it says so everywhere in the product.

## 6. Differentiation

| Alternative | Gap Aabhas fills |
|---|---|
| National bulletins | District-level, deterministic, not station-specific |
| Global flood models (GloFAS, Google Flood Hub) | Useful, but not tuned to FFWC danger levels or local station decisions |
| Research papers | Not deployable, not maintained, rarely audited for leakage |
| Generic ML flood demos | Often leak label-window features; no calibration, no ops, no alerts |
| Commercial risk platforms | Expensive, opaque, not built around a public duty officer's workflow |

**Moat:** local station knowledge + a leakage-audited evaluation + an operational
workflow (quality gate, threshold governance, approval and audit). The model is
reproducible; the trust layer is the product.

## 7. Use of funds (indicative, 12 months)

| Workstream | Share | What it buys |
|---|---:|---|
| Institutional pilot (1 agency/NGO, 1 basin) | 30 % | MOU, duty-officer training, threshold sign-off, evaluation |
| Always-on infrastructure | 15 % | Managed API, scheduler, monitoring, backups, modest data egress |
| Product & integrations | 25 % | SMS/webhook delivery, partner dashboard, accessibility, Bengali polish |
| Impact evaluation | 15 % | Prospective study, baseline, independent review |
| Data & model operations | 10 % | Source monitoring, retraining policy, drift and incident response |
| Legal, governance, contingency | 5 % | Data agreements, disclaimers, audit, contingency |

**Milestones the funding buys**

- M1 (month 1–2): managed deployment + uptime and freshness SLAs.
- M2 (month 3–4): signed pilot, named duty officer, formal threshold sign-off.
- M3 (month 5–9): live alert delivery and a prospective evaluation dataset.
- M4 (month 10–12): published impact report and a renewal/scale decision.

## 8. Team

Aabhas is currently a solo-founder research and engineering project. The repository
demonstrates the full stack: data ingestion, feature engineering, model selection,
leakage auditing, calibration, API, frontend, alert governance, tests and deployment.

What the project needs next is **not more code** — it is an institutional partner,
a modest operating budget, and domain mentorship from a flood or disaster-management
agency. Advisory roles are open.

## 9. Risks and how they are handled

| Risk | Mitigation |
|---|---|
| **False alarms erode trust** | Explicit 10:1 cost function, calibration, human approval before public alerts |
| **Missed events** | Recall-first thresholds, event analysis, published error taxonomy |
| **Stale or broken data** | Per-source freshness SLAs, degraded flag in `/health`, hard quality gate |
| **Model drift** | Frozen model + documented retraining and evaluation policy |
| **Overclaiming** | The product and README label it a research prototype, not an official warning service |
| **Single maintainer** | Documentation, reproducibility and tests are treated as first-class; advisory support is part of the ask |
| **Regulatory / liability** | No public auto-alerts; delivery disabled by config until a partner signs off |

## 10. What we are asking for

1. **A pilot partner** — an agency, NGO or research group willing to run one basin for
   one season with a named duty officer.
2. **US$150k–350k** in grant or pre-seed funding for the 12-month plan in §7.
3. **Advisors** — flood operations, disaster governance, and climate finance.

If you want to evaluate the work before a conversation, the fastest path is:

1. Open the [live demo](https://riyadh5674.github.io/Aabhas-flood-predictor-bangladesh/).
2. Read [the 5-minute demo script](DEMO.md).
3. Run `tests\run_tests.py` and `scripts\freeze_results.py --check` yourself.
4. Read [the final evaluation](../reports/phase43_final_evaluation.md).
5. The original project proposal is in [Proposal.docx](../Proposal.docx).

---

*Aabhas is a research prototype and decision-support tool. It is not an official flood
warning service and must never replace FFWC advisories or government instructions.*
