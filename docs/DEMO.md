# 5-Minute Demo Script

For investors, agencies, NGOs and reviewers. The goal is to show **one real decision**
being made with Aabhas — not to tour every screen.

**Live demo:** https://riyadh5674.github.io/Aabhas-flood-predictor-bangladesh/
**Local fallback:** see §6.

---

## Before you start

- Open the live demo in a desktop browser at 1440 px width or wider.
- Have the repository open in a second tab.
- Optional: open `reports/phase43_final_evaluation.md` for the evidence Q&A.
- Say this once, early: *"This is a research prototype and decision-support tool, not an
  official warning service. FFWC advisories take precedence."*

---

## Minute 1 — The question

> "In Bangladesh, a warning usually tells a district that it might flood. A family,
> a union council or an NGO needs a different answer: *what is the chance that **our
> station** crosses its danger level in the next 48 hours?* Aabhas answers that at
> station level, with a probability and an explicit uncertainty trade-off."

Show the **Overview** page.

Point out:
- the horizon switch **24 / 48 / 72 h**,
- the **data freshness banner**,
- the national risk summary and the list of highest-risk stations.

Talking point: *"The first thing on screen is not a prediction — it is how fresh the
data is. A stale forecast is never presented as a live one."*

## Minute 2 — From country to station

Click **Map**. Then click a high-risk station (or open **Stations** and pick one).

On the station page, show:
- the probability trend,
- the current band (Low / Moderate / High / Very high),
- the model drivers,
- the plain-language guidance.

Talking point: *"The band is not arbitrary. The threshold comes from an explicit
10-to-1 cost: a missed flood is treated as ten times worse than a false alarm. That is
a policy choice, and it is visible in the product and the config."*

## Minute 3 — Why the number is credible

Open the **Model** page.

Show:
- horizon-specific thresholds and expected recall / precision,
- calibration evidence,
- the frozen model identity.

Then switch to the repository and show:

```powershell
.\.venv\Scripts\python.exe tests\run_tests.py          # 84/84
.\.venv\Scripts\python.exe scripts\freeze_results.py --check   # 13/13
```

Talking point: *"The test period was opened once. Every later analysis records zero
test reads. That is why the number is believable — and you can run the checks yourself."*

## Minute 4 — The operational workflow

Open **Alerts**.

Show:
- how a band change becomes a **draft** alert,
- the **human approval** step,
- the **hash-chained audit log**.

Talking point: *"We deliberately do not auto-publish public warnings. A duty officer
reviews the draft, approves it, and the system records who approved what and when.
External delivery is disabled by configuration until a partner signs off on thresholds."*

Mention the failure story briefly:

> "In October a hard-coded date in three data fetchers silently degraded 42 of 110
> features for three weeks. We found it, documented it, and rebuilt the pipeline around
> a mandatory quality gate and per-source freshness telemetry. That incident is why the
> product shows freshness so prominently."

## Minute 5 — The ask

> "The model and system are complete and reproducible. What is missing is operational:
> one institutional pilot, a managed deployment, and a named duty officer to approve
> alerts. We are looking for a pilot partner and US$150k–350k for a 12-month plan.
> The details are in `docs/PITCH.md`."

Close on the live link and the repository.

---

## 6. Local fallback

If the public demo is unavailable:

```powershell
# Full stack (API + web)
cd system\web
npm run dev
# open http://localhost:5173

# Static snapshot only (no API)
cd system\web
$env:VITE_AABHAS_STATIC_SNAPSHOT="true"
npm run dev
```

Docker:

```bash
docker compose up --build
# web: http://localhost:5173
# api: http://localhost:8000/health
```

---

## 7. Likely questions

**"Is this an official warning service?"**
No. It is a decision-support prototype. The UI, README and API all state this. FFWC
remains the authority.

**"How do you know the model is not leaking?"**
The splits use a 7-day embargo and station-disjoint folds. Label-window columns are
banned as features and asserted absent in tests. The test set was opened once, and a
read ledger records zero reads after that.

**"Why not deep learning?"**
We ran a fair DL rematch on identical features, budgets and seeds. It did not beat the
gradient-boosting model, so the DL queue was closed. The comparison is committed in
`reports/mi7_report.md`.

**"What happens when data is stale?"**
The quality gate fails the date, `/health` marks the source degraded, and the UI shows
an aging/stale banner. The pipeline does not silently serve an old forecast as current.

**"How does this become a business?"**
Pilot-funded deployments with agencies/NGOs, then API subscriptions and data licensing.
The honest answer today is that the model is validated and the revenue model is not yet.

**"What is the biggest risk?"**
Institutional trust, not model accuracy. That is why the product is built around human
approval, auditability and freshness rather than automatic public alerts.

---

## 8. Reset / troubleshooting

- **Page shows an old date:** the static demo ships a snapshot; check the freshness
  banner. Rebuild with `python system/deploy/build_static_snapshot.py` and redeploy.
- **Map does not load:** reload once; the service worker caches the boundary files.
  Hard-refresh (`Ctrl+Shift+R`) clears a stale shell.
- **Local API port in use:** set `AABHAS_API=http://127.0.0.1:<port>` before
  `npm run dev`, or start `python -m system.api.run_api --port <port>` separately.
- **No `.venv`:** create it with `py -3.13 -m venv .venv` and install
  `requirements.txt`.
