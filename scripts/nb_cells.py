"""Notebook cell sources for `advanced_experiments/*.ipynb`.

Each entry returns a list of notebook cells (markdown + code).  Keeping them in
plain Python source makes them reviewable, regenerable and free of JSON escaping
pain.  `scripts/build_nb.py` turns them into notebooks.

NOTEBOOKS["<name>"]() -> list[dict]
"""
from __future__ import annotations


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip("\n").splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


# --------------------------------------------------------------------------- 01
def nb01() -> list[dict]:
    return [
        md(r"""
# Experiment 1 — Domain Adaptation for Distribution Shift

**GroupDRO · IRM · DANN applied to Bangladesh flood probability forecasting**

| | |
| --- | --- |
| **Status** | RUNNING |
| **Started** | 2026-10-21 |
| **Seed** | `20261008` (`baseline_model.B.SEED`) |
| **Script** | `scripts/domain_adaptation.py` |
| **Artefacts** | `reports/domain_adaptation/` |
| **Test reads** | **0** — this experiment is development-only on dev cells |

---

## 1. Why this experiment exists

The programme's most recent candidate (decision **D17**, `models/final_v3/`) is a
per-horizon hybrid. **Its test numbers are retained here only for traceability —
the read that produced them is invalid and D17 is suspended** (see §2.1 and
`reports/d17_comparability_audit.md`):

| Horizon | Learner | Feature set | Test PR-AUC (MI-4, **invalid**) |
| --- | --- | --- | --- |
| 24 h | LightGBM `lgbm13` | FULL_V4 (187) | 0.8572 |
| 48 h | LightGBM `lgbm7` | FULL_V2 (110) | 0.7932 |
| 72 h | XGBoost `xgb3` | FULL_V2 (110) | 0.7444 |

What *is* solid, because it comes from the correctly-scored MI-6 read, is the
D16 champion's behaviour on the frozen test period:

| Cell (24 h, season=all) | D16 PR-AUC |
| --- | --- |
| `test_2023` | 0.7956 |
| `test_2024` | 0.9177 |
| `test_2025` | 0.8048 |
| `test_2026` | 0.7935 |
| `test_unseen` | 0.8299 |

The structural weakness is the base-rate collapse:

| Period | 24 h base rate |
| --- | --- |
| 2017–2021 (training) | 7.0 % – 14.2 % |
| 2022 (validation) | 5.1 % |
| 2025–2026 (test) | ~1–3 % |

This is a **covariate + concept shift**: `P_train(y|x) ≠ P_test(y|x)`. The
model was fitted to discriminate a flood in a *wet* regime and is asked to do so
in a *dry* regime, where the prior has changed by an order of magnitude.

**Domain adaptation** is the family of methods built for exactly this problem:
instead of minimising the *population-average* empirical risk, they minimise the
*worst-group* risk over a partition of the training data into domains that are
known to shift.

### The hypothesis

> If the training rows are partitioned into physically meaningful domains
> (basin, elevation band, coastline proximity, network cohort) and the model is
> optimised for the **worst** domain rather than the average, the model should
> lose less on the shifted (drier) evaluation years without giving up anything on
> the in-regime cells.

### What would falsify it

If GroupDRO/IRM/DANN cannot beat the empirical-risk control on the
pre-registered decision cell (`val_2022`, horizon-mean PR-AUC) **and** does not
improve the worst-cell PR-AUC, the hypothesis is rejected and D17 stands.

---

## 2. Protocol — and a protocol correction

### 2.1 A correction to the earlier dev-cell scorecards

The MI-8 and MI-4 "7 dev cells" scorecards (`val_2020`, `val_2021`, `val_2022`,
`spatial_fold_1..4`) were produced by **one model trained on the whole training
split**, then scored on cells that were *inside* that training split:

* `val_2020` and `val_2021` are years inside `split == "train"`;
* `spatial_fold_1..4` are station folds inside `split == "train"`.

Only `val_2022` was genuinely held out. That inflated the absolute numbers — the
in-sample cells sat at PR-AUC ≈ 1.000. **This notebook fixes the protocol so
that every scored row is unseen.**

A caution about magnitude: even with the corrected protocol, the 24 h ERM cell
values are still high (0.969–0.993). The in-sample scorecard was inflated by
roughly 0.01–0.03, but the pre-2023 dev cells are *genuinely* easy. The real
difficulty is confined to the post-2023 test years, which this experiment does
not touch.

Also note: the MI-4 test read that produced D17 applied a **complete-case**
filter, scoring only 31.5 % of test rows on a subset with a 1.5–1.8× higher base
rate. D17 is therefore **retracted** (decision **D18**,
`reports/mi9_test_read_decision.md`): under the corrected read the hybrid scores
0.77066 vs D16's 0.77076 (-0.00010). **D16 is the champion.** This experiment is
unaffected: it is dev-only and uses the pipeline imputer throughout.

### 2.2 The protocol used here

Every scored row is **never seen in training**:

| Task | Training rows | Scored cell | Shift exercised |
| --- | --- | --- | --- |
| `t2020` | years ≤ 2019 | 2020 | temporal (wet → wet) |
| `t2021` | years ≤ 2020 | 2021 | temporal (wet → drier) |
| `t2022` | years ≤ 2021 (all train) | 2022 | temporal (wet → driest) |
| `sf1..sf4` | train-split rows with `spatial_fold != k` | fold `k` | spatial (station-disjoint) |

This mirrors the Phase-15 protocol (`P2_rolling_origin`, `P3_spatial_cv`) as
defined in `scripts/baseline_model.py::build_tasks`.

### 2.3 Rules honoured

* **PR-AUC** is the primary metric; per-cell only, never pooled (D8).
* Seed `20261008` everywhere.
* **No test cell is read** in this experiment.
* Domain variables are station attributes known at prediction time — no label
  information enters the partition (Phase-13 leakage rule satisfied by
  construction).
* Every arm shares the **same features, the same training rows, the same
  learner and the same capacity**; only the *objective* differs. That is what
  makes the contrast interpretable.
"""),
        code(r"""
# --- environment -------------------------------------------------------------
import json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(r"D:\Flood predictor")
sys.path.insert(0, str(ROOT / "scripts"))
import baseline_model as B

SEED = B.SEED
HORIZONS = B.HORIZONS
print("python  :", sys.version.split()[0])
print("seed    :", SEED)
print("horizons:", HORIZONS)
"""),
        md(r"""
---

## 3. Domain construction

Station-level partitions derived from metadata available at prediction time.

| Scheme | Groups | Source |
| --- | --- | --- |
| `basin` | 5 | FFWC `basin` attribute (Ganges / Brahmaputra / Meghna / SE Hill / Unknown) |
| `elev_q` | 4 | quartiles of DEM `elevation_m` |
| `coastal` | 2 | 40th percentile of `dist_to_coast_km` (coastal vs inland) |
| `cohort` | 2 | Phase-15 network cohort (`early` / `late`) |
| `hydro` | 10 | `basin × coastal` cross |

The partitions are strongly non-uniform in label prevalence, which is exactly
what makes worst-group optimisation non-trivial:

| `basin` group | train rows | 24 h base rate |
| --- | --- | --- |
| Brahmaputra | 27,294 | 0.1072 |
| Meghna | 15,543 | 0.0868 |
| Ganges | 13,231 | 0.1190 |
| SE Hill | 5,923 | **0.0162** |

| `coastal` group | train rows | 24 h base rate |
| --- | --- | --- |
| inland | 36,727 | 0.0761 |
| coastal | 25,264 | 0.1247 |

| `cohort` group | train rows | 24 h base rate |
| --- | --- | --- |
| early | 58,478 | 0.0842 |
| late | 3,513 | **0.2906** |

The `SE Hill` (hill-torrent, flash-flood) domain has a **6× lower** base rate
than the Ganges domain, and the `late` network cohort has a **3.5× higher** one.
These are the groups a worst-group objective will try to protect.
"""),
        code(r"""
# --- build the domain partitions ---------------------------------------------
import importlib
da = importlib.import_module("domain_adaptation")

d, feats = da.load_data()
dom = da.build_domains(d)
d = da.attach_domains(d, dom)
tasks = da.make_tasks(d)

print("\nFULL_V4 features:", len(feats))
print("\ntasks (train rows / eval rows, base rate):")
for t, (tr, ev) in tasks.items():
    tr_rows = int(tr.sum()); ev_rows = int(ev.sum())
    br = d.loc[ev, "label_24h"].mean()
    print(f"  {t:6s} train={tr_rows:7,}  eval={ev_rows:7,}  eval_base_rate_24h={br:.4f}")
"""),
        code(r"""
# --- domain composition inside the 2017-2021 training rows -------------------
tr = d[(d.split == "train") & (~d.station_holdout)]
rows = []
for s in da.SCHEMES:
    col = f"dom_{s}"
    for g, n in tr[col].value_counts().items():
        sub = tr[tr[col] == g]
        rows.append(dict(scheme=s, group=g, rows=n,
                         base_rate_24h=sub.label_24h.mean(),
                         base_rate_48h=sub.label_48h.mean(),
                         base_rate_72h=sub.label_72h.mean(),
                         n_stations=sub.station_id.nunique()))
dom_table = pd.DataFrame(rows)
dom_table.to_csv(ROOT / "reports" / "domain_adaptation" / "da_domain_composition.csv", index=False)
dom_table
"""),
        md(r"""
---

## 4. Methods

All six arms use the **same 187 features** and the **same training rows**. The
control is empirical risk minimisation (ERM); the other five replace only the
objective.

| Arm | Learner | Objective |
| --- | --- | --- |
| `erm_lgbm` | LightGBM (300 trees, lr 0.02, 63 leaves) | mean BCE (control) |
| `groupdro_lgbm` | LightGBM (identical) | **Online GroupDRO**: exponentiated-gradient re-weighting of groups |
| `erm_mlp` | MLP 187→256→64 | mean BCE (control for the neural arm) |
| `groupdro_mlp` | MLP (identical) | **Online GroupDRO** on mini-batches |
| `irm_mlp` | MLP (identical) | **IRM v1** penalty `‖∇_w BCE‖²` shared across domains |
| `dann_mlp` | MLP + domain head | **DANN**: gradient-reversal layer + domain classifier |

### Online GroupDRO

Sagawa et al. (2020). Group weights `q_g` are updated between rounds by

```
q_g  ←  q_g · exp(η · (L_g − mean(L)))        η = 0.1
q_g  ←  clip(q_g, 1/(G·5), 5/G) ,  q ← q/Σq
```

The `−mean(L)` shift makes the update invariant to the overall loss scale, which
matters here because the base rate is only 5–14 % and raw log-loss differences
between groups are tiny.

### IRM v1

Arjovsky et al. (2019). A dummy scalar `w = 1` is multiplied into the logits;
the penalty is the squared gradient of the per-domain loss with respect to `w`.
The penalty is annealed in over the first 5 epochs (`λ = 100`).

### DANN

Ganin et al. (2016). A gradient-reversal layer sits between the shared
representation and a domain classifier; the classifier is trained to predict the
domain, the reversal makes the feature extractor try to *destroy* that
information. The adversary is switched on after 3 epochs (`λ = 1`).

### Leakage control

Group membership comes from `station_spatial_features.csv` (FFWC metadata +
DEM + coastline rasters) and the Phase-15 cohort labels. No label, no
post-cutoff statistic, and no test row enters the partition — so the Phase-13
truncation rule is satisfied by construction. A dedicated assertion in the
script checks that every domain column is constant within a station.
"""),
        code(r"""
# --- assertions that the protocol is honest ---------------------------------
checks = []

# A1: no scored row is also a training row, for every task
for t, (tr, ev) in tasks.items():
    checks.append((f"A1:{t}", int((tr & ev).sum()) == 0))

# A2: temporal tasks are strictly rolling-origin
yr = d.date.dt.year.to_numpy()
checks.append(("A2:t2020_train<=2019", bool((yr[tasks["t2020"][0]] <= 2019).all())))
checks.append(("A2:t2021_train<=2020", bool((yr[tasks["t2021"][0]] <= 2020).all())))
checks.append(("A2:t2022_train<=2021", bool((yr[tasks["t2022"][0]] <= 2021).all())))

# A3: spatial folds are station-disjoint
for k in (1, 2, 3, 4):
    tr, ev = tasks[f"sf{k}"]
    s_tr = set(d.loc[tr, "station_id"]); s_ev = set(d.loc[ev, "station_id"])
    checks.append((f"A3:sf{k}_station_disjoint", len(s_tr & s_ev) == 0))

# A4: no test row is ever scored
te = (d.split == "test").to_numpy()
checks.append(("A4:no_test_row_scored",
               not any(bool((te & ev).any()) for _, ev in tasks.values())))

# A5: domain labels are constant within a station (no time-varying leakage)
for s in da.SCHEMES:
    nun = d.groupby("station_id")[f"dom_{s}"].nunique().max()
    checks.append((f"A5:dom_{s}_station_constant", int(nun) == 1))

# A6: holdout (unseen) stations never train
for t, (tr, ev) in tasks.items():
    checks.append((f"A6:{t}_no_holdout_in_train",
                   not bool(d.loc[tr, "station_holdout"].any())))

chk = pd.DataFrame(checks, columns=["check", "passed"])
chk.to_csv(ROOT / "reports" / "domain_adaptation" / "da_protocol_checks.csv", index=False)
print(f"{int(chk.passed.sum())} / {len(chk)} protocol checks passed")
chk
"""),
        md(r"""
---

## 5. Results

*(populated from `reports/domain_adaptation/da_scorecard.csv` and
`da_bootstrap.csv` after the run completes)*
"""),
        code(r"""
# --- scorecard ---------------------------------------------------------------
import importlib
da = importlib.import_module("domain_adaptation")

OUT = ROOT / "reports" / "domain_adaptation"
led = pd.read_csv(OUT / "da_ledger.csv")
print(f"ledger rows: {len(led)}")
led.head()
"""),
        code(r"""
# --- per-cell PR-AUC scorecard ----------------------------------------------
cells = ["val_2020", "val_2021", "val_2022",
         "spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]
rows = []
for (m, s, h), g in led.groupby(["method", "scheme", "horizon"]):
    g = g.set_index("task")
    rec = dict(method=m, scheme=s, horizon=h)
    for t, c in da.CELL_OF.items():
        rec[c] = float(g.loc[t, "pr_auc"]) if t in g.index else np.nan
    vals = [rec[c] for c in cells if not np.isnan(rec[c])]
    rec["dev_mean"] = float(np.mean(vals)) if vals else np.nan
    rec["dev_worst"] = float(np.min(vals)) if vals else np.nan
    rows.append(rec)
score = pd.DataFrame(rows).sort_values(["horizon", "scheme", "method"])
score.to_csv(OUT / "da_scorecard.csv", index=False)
score
"""),
        code(r"""
# --- paired station-level bootstrap vs the ERM control -----------------------
bpath = OUT / "da_bootstrap.csv"
if bpath.exists():
    boot = pd.read_csv(bpath)
    print(boot.shape)
    boot.head(10)
else:
    print("bootstrap table not written yet")
"""),
        md(r"""
---

## 6. Decision criteria (pre-registered before the run)

| # | Criterion | Threshold |
| --- | --- | --- |
| 1 | Decision cell `val_2022` horizon-mean PR-AUC | must be **≥ ERM control** |
| 2 | Worst dev cell PR-AUC | must be **> ERM control** |
| 3 | Paired bootstrap on `val_2022` | CI must not exclude 0 in favour of the control |
| 4 | No cell may regress by more than 0.005 | safety |

Only if all four hold does the method become eligible for a **pre-registered
test read #3**. Otherwise D17 stands and the finding is recorded as a negative
result.

---

## 7. Reproduction

```bash
.venv\Scripts\python.exe -u scripts\domain_adaptation.py domains
.venv\Scripts\python.exe -u scripts\domain_adaptation.py probe
.venv\Scripts\python.exe -u scripts\domain_adaptation.py run --horizons label_24h label_48h label_72h
.venv\Scripts\python.exe -u scripts\domain_adaptation.py report
```

The run is **resumable**: every `(method, scheme, horizon)` arm is appended to
`reports/domain_adaptation/da_ledger.csv` and skipped on restart.
"""),
    ]


# --------------------------------------------------------------------------- 02
def nb02() -> list[dict]:
    return [
        md(r"""
# Experiment 2 — SAR Inundation Prediction

**Turning Sentinel-1 into a flood-label engine and an inundation feature**

| | |
| --- | --- |
| **Status** | PLANNED (§1-6 written, awaits Experiment 1 verdict) |
| **Started** | — |
| **Seed** | `20261008` |
| **Script** | `scripts/sar_inundation.py` (to be written) |
| **Artefacts** | `reports/sar_inundation/` |
| **Test reads** | 0 |

---

## 1. Why this experiment exists

Every negative result in the programme points at the same bottleneck: **the
labels**. `reports/mi4_label_audit.md` showed that PR-AUC on `val_2022` tops out
at ≈ 0.96–0.97 because the label itself is noisy (a daily binary exceedance
derived from a single gauge reading, with `n_readings` varying per station), and
because the base rate collapses from 7–14 % (2017–21) to 1–3 % (2025–26).

Feature engineering has been exhausted: MI-8 (+55 `m8_*` columns) moved
`val_2022` by +0.00058; MI-3a (ERA5) by −0.00025. Adding a 188th tabular column
will not help.

What *has* never been tried is a **different measurement of the target**.
Sentinel-1 C-band SAR sees surface water through cloud and at night. A
SAR-derived inundation map:

1. gives an **independent, spatially explicit** water mask per acquisition
   (≈ 6-day revisit at this latitude, two satellites);
2. can **validate or relabel** the gauge exceedance labels;
3. can be aggregated to the station catchment and used as a **feature**
   (`sar_water_frac_7d`, `sar_water_trend`), which is genuinely new information
   rather than another transform of the same rainfall series.

## 2. Data feasibility — already probed

`scripts/probe_data_sources5.py` established (2026-10-21):

| Check | Result |
| --- | --- |
| Copernicus DataSpace OData catalogue reachable | **HTTP 200** |
| Sentinel-1 products over the Bangladesh AOI, 2025-08 | **HTTP 200**, GRD COG products returned |
| Same query for 2023-06 | **HTTP 200** |
| ASF DAAC search endpoint | HTTP 404 (path changed) — unused |
| CHIRPS daily netCDF | HTTP 404 — **unavailable**, as before |
| ECMWF open data | DNS failure — **unavailable** without a licence |

**Blocker:** product *download* needs a free CDSE account (S3 credentials).
Catalogue metadata is open; raster access is not anonymous. This is a
5-minute signup, not a technical risk, but it must be done before the run.

## 3. Method

```
Sentinel-1 GRD (VV, VH, 10 m)
        │  terrain correction (Copernicus DEM 30 m), speckle filter (Lee 5x5)
        ▼
  water mask:  VV < -17 dB  AND  (VH < -23 dB OR VV-VH < 7 dB)
        │  remove permanent water (JRC Global Surface Water occurrence > 75 %)
        │  remove steep-slope false positives (slope > 5 deg)
        ▼
  binary inundation map  ──►  catchment aggregate  ──►  tabular feature
        │
        └──►  independent check against FFWC `wl_dist_to_danger > 0`
```

**Model.** A small U-Net (4 down / 4 up, 32→256 channels) on 256×256 tiles at
100 m, inputs = `[VV, VH, VV−VH, elevation, slope, HAND, dist_river]`, output =
`P(water)`. Trained on weak labels from the threshold rule, refined on a
hand-checked subset (~2,000 tiles). CPU-feasible at 100 m; GPU preferred.

**Two distinct deliverables** — these must not be conflated:

| Deliverable | Metric | Role |
| --- | --- | --- |
| A. Inundation map | pixel AUPRC, IoU vs threshold labels | a *product*, not a model score |
| B. Tabular feature | per-cell PR-AUC vs D16 | the only thing that can change the champion |

Deliverable B is what matters for the forecasting programme. Deliverable A is
valuable in its own right but does **not** justify a test read.

## 4. Evaluation plan

* **Pixel level:** AUPRC and IoU on held-out tiles (spatial block CV, no tile
  leakage), stratified by season and by basin.
* **Station level:** aggregate SAR water fraction to the station's upstream
  catchment (the `up_*` graph already exists in `station_graph.csv`), add as
  `sar_*` features, re-run the **same** tabular HPO budget, and compare per-cell
  against D16 on the 7 dev cells.
* **Leakage:** a SAR acquisition on day `t` may only feed a label at `t + h`
  with `h > 0`. Backward aggregation is banned (Phase-13). The acquisition
  timestamp, not the processing timestamp, defines availability.
* **Test read:** only if the dev-cell gate passes, and pre-registered.

## 5. Risks

| Risk | Mitigation |
| --- | --- |
| CDSE account unavailable | fall back to the public `sentinel-s1-l1c` S3 bucket (SLC, needs GRD processing) — more work, same data |
| Flood-time acquisitions are sparse (cloud-free is not required, but revisit is 6 days) | accept the sparsity; use it as an *event* feature, not a daily series |
| Threshold water masks are biased in vegetated/urban areas | calibrate the threshold per land-cover class using ESA WorldCover (already in the project) |
| 2025–26 base rate is 1–3 %, so even a good feature moves PR-AUC little | evaluate on `test_unseen` / `test_new_network` too, where the base rate is higher |

## 6. Success criteria (pre-registered, to be confirmed before the run)

| # | Criterion |
| --- | --- |
| 1 | Pixel AUPRC of the U-Net ≥ 0.85 on held-out tiles, and ≥ threshold rule + 0.05 |
| 2 | `sar_*` features improve the 7-dev-cell worst-cell PR-AUC vs D16 |
| 3 | No dev cell regresses by more than 0.005 |
| 4 | Paired station bootstrap on `val_2022` does not favour D16 |
"""),
    ]


# --------------------------------------------------------------------------- 03
def nb03() -> list[dict]:
    return [
        md(r"""
# Experiment 3 — NWP Ensemble Forecast Features

**Forecast rainfall instead of observed rainfall, with calibrated uncertainty**

| | |
| --- | --- |
| **Status** | PLANNED |
| **Seed** | `20261008` |
| **Script** | `scripts/nwp_ensemble.py` (to be written) |
| **Artefacts** | `reports/nwp_ensemble/` |
| **Test reads** | 0 |

---

## 1. Why this experiment exists

Every feature in `FULL_V2` / `FULL_V4` that describes rainfall is either
**observed** (`rain_1d … rain_30d`, Open-Meteo archive) or a **lag** of observed
rainfall (`rain_*_lag*`, `up_rain7d_lag*`). Observed rainfall at day `t` is only
known at day `t`. A 24 h-ahead forecast issued at `t` that uses `rain_1d(t)`
therefore has **zero true lead time** for the rainfall driver — it is
nowcasting the water level, not forecasting the rain.

NWP ensembles supply exactly the missing object: a *predictive distribution* of
future rainfall. The scientific claim is:

> Replacing/augmenting observed-rainfall features with **ensemble forecast
> features** (mean, spread, exceedance probability, and their upstream
> aggregates) improves PR-AUC at 48 h and 72 h, where lead time actually
> matters, and improves calibration (`ece`, `brier`) everywhere.

## 2. Data feasibility

| Source | Status (probed 2026-10-21) | Verdict |
| --- | --- | --- |
| NOAA GEFS 0.25°, 31 members, 16-day | NOMADS returns **HTTP 403** from this host | blocked on this network; AWS Open Data mirror to be tested next |
| ECMWF ENS 50 members | DNS failure / paid licence | **unavailable** |
| GFS deterministic 0.25° | NOMADS 403 | same |
| Open-Meteo *forecast* API | not yet probed; free tier allows historical forecast archive | **most promising fallback** |

**Decision to make before building:** probe the Open-Meteo
`/v1/forecast` archive (which serves past GEFS/ECMWF IFS forecasts with a
`past_days` window and a `start_date`/`end_date` range) and the AWS
`noaa-gfs-bdp-pub` mirror. If neither works, Experiment 3 is **not feasible**
without a paid data licence and should be dropped rather than faked.

## 3. Method

1. **Ensemble features** per station-day, per lead time `h ∈ {1,2,3}` days:
   * ensemble mean and spread of 24 h rainfall;
   * `P(rain_24h > 10 mm)`, `P(rain_24h > 25 mm)` (exceedance probabilities);
   * basin-aggregated upstream ensemble means (reuse the `up_*` graph);
   * **forecast–observation divergence** at day `t`: `|forecast(t) − obs(t)|` —
     a proxy for forecast difficulty / regime instability.
2. **Calibration.** Fit a Bayesian Model Averaging (BMA) layer per station
   cluster on the training period, producing a calibrated predictive
   distribution; feed its moments as features. BMA weights are fitted on
   training years only.
3. **Tabular model.** Identical HPO budget and feature-count control as MI-8,
   so the comparison isolates the NWP block.

## 4. Evaluation plan

* 7 dev cells, strictly out-of-sample (`P2` rolling origin + `P3` spatial CV).
* Primary: per-cell PR-AUC at 48 h and 72 h (where lead time matters).
* Secondary: `brier`, `ece`, and **reliability diagrams** — an NWP ensemble
  should improve calibration even where it does not improve ranking.
* A dedicated **lead-time ablation**: restrict features to those available at
  issue time `t`; verify no observed-rainfall-at-`t+h` leakage.

## 5. Risks

| Risk | Mitigation |
| --- | --- |
| Forecast archive not freely available | probe first; drop the experiment if not |
| Forecasts are biased for Bangladesh (monsoon convection) | bias-correct against observations on training years only; report raw and corrected |
| Forecast features are correlated with observed features | feature-count control + permutation importance, as in MI-8 |
| Adds 30+ columns to a 110-column model | strict HPO budget match, and an equal-column random control |

## 6. Success criteria

| # | Criterion |
| --- | --- |
| 1 | 48 h and 72 h dev-cell PR-AUC improve over D16 |
| 2 | 24 h does not regress |
| 3 | `ece` improves at all horizons |
| 4 | Paired station bootstrap excludes zero on at least one horizon |
"""),
    ]


# --------------------------------------------------------------------------- 04
def nb04() -> list[dict]:
    return [
        md(r"""
# Experiment 4 — Physics-Informed Neural Operator on the River Graph

**Learning the routing operator instead of a per-station classifier**

| | |
| --- | --- |
| **Status** | PLANNED |
| **Seed** | `20261008` |
| **Script** | `scripts/pino_graph.py` (to be written) |
| **Artefacts** | `reports/pino_graph/` |
| **Test reads** | 0 |

---

## 1. Why this experiment exists

The current model is a **per-row classifier**: each (station, day) is scored
independently, with the river network entering only through hand-built `up_*`
aggregates and a fixed `station_graph.csv`. Two consequences:

* the model cannot represent **travel time** along the network — a flood wave
  moving from the Brahmaputra into the Meghna delta is a *spatiotemporal*
  object, not a row;
* it has no incentive to satisfy **conservation**: predicted water levels can
  rise downstream without a corresponding upstream cause.

A neural operator on the river graph addresses both. It learns a mapping from
the *whole upstream state field* to the *downstream response*, with physics
residuals as soft constraints.

## 2. Method

**Graph.** Nodes = 153 stations + virtual junctions; edges = `station_graph.csv`
(427 edges, `dist_km`). Node features = the existing tabular state at day `t`.

**Operator.** A Fourier/Galerkin-style spectral layer on the graph Laplacian
(analogous to FNO but on an irregular mesh):

```
h_{t+1} = σ( U diag(θ) Uᵀ h_t  +  W h_t  +  b )
```

where `U` are the lowest-k eigenvectors of the normalised graph Laplacian. This
is a *graph FNO*: it is equivariant to node permutation and has a receptive
field equal to the graph diameter.

**Physics losses (soft):**

| Residual | Form |
| --- | --- |
| Water balance | `dS/dt ≈ P − Q − E` aggregated over the upstream catchment |
| Routing lag | cross-correlation between upstream `wl` and downstream `wl` must be non-negative at the learned lag |
| Monotonicity | downstream `wl` must not exceed upstream `wl` + channel slope term (soft penalty) |

**Head.** Per-node, per-horizon binary classifier, identical label definition,
so the comparison against D16 is direct.

## 3. Evaluation plan

* 7 dev cells, strictly out-of-sample; spatial folds are the *important* ones
  here because the operator is supposed to generalise to new nodes.
* Primary: `spatial_fold_1..4` PR-AUC (station-disjoint) and `test_unseen`
  eligibility.
* Physics-residual diagnostics reported separately — a model that scores well
  but violates conservation should be flagged, not promoted.
* Compute: needs GPU for the spectral layers at 153 nodes × 200 k days; CPU
  fallback is possible but slow. **Confirm GPU availability before starting.**

## 4. Risks

| Risk | Mitigation |
| --- | --- |
| 153 nodes is a very small graph — the operator may overfit | strong weight decay, low spectral rank `k ≤ 16`, node dropout |
| Physics constants (channel geometry, roughness) are unknown | keep residuals *dimensionless* and relative; do not fit physical constants |
| Dataset is a sparse station sample of a continuous field | treat ungauged junctions as latent, marginalise rather than impute |
| No GPU confirmed | do not start until confirmed; CPU path is 10–20× slower |

## 5. Success criteria

| # | Criterion |
| --- | --- |
| 1 | Spatial-fold PR-AUC beats D16 on at least 3 of 4 folds |
| 2 | `val_2022` does not regress |
| 3 | Physics residuals are lower than the tabular model's implied residuals |
| 4 | Paired station bootstrap excludes zero on the spatial folds |
"""),
    ]


# --------------------------------------------------------------------------- 05
def nb05() -> list[dict]:
    return [
        md(r"""
# Experiment 5 — Differentiable Hydrology (Neural ODE)

**A continuous-time rainfall–runoff model with data assimilation**

| | |
| --- | --- |
| **Status** | PLANNED |
| **Seed** | `20261008` |
| **Script** | `scripts/neural_ode_hydro.py` (to be written) |
| **Artefacts** | `reports/neural_ode/` |
| **Test reads** | 0 |

---

## 1. Why this experiment exists

Experiment 4 learns the *spatial* operator. Experiment 5 learns the *temporal*
dynamics. The current models treat each day as an independent row with lagged
features; a neural ODE models the **continuous state trajectory**

```
dh/dt = f_θ( h(t), P(t), T(t), u(t) )
```

where `h` is catchment storage / water level, `P` rainfall, `T` temperature,
`u` upstream inflow. This gives:

* **irregular time support** — the data has missing days (there is no 2018 in
  the training split) and varying `n_readings`; an ODE handles irregular `Δt`
  natively, a lag-feature model does not;
* **data assimilation** — at each observation the latent state is corrected
  (Kalman-style or via a learned encoder), which is the principled way to use
  a gauge reading;
* **physically interpretable parameters** — recession constants, travel times.

## 2. Method

1. **Latent state** `z(t) ∈ R^d` per station, initialised by an encoder on the
   first 7 days.
2. **Dynamics** `dz/dt = f_θ(z, P, T)` — a small MLP with a residual
   `dz/dt = a(z) + b·P(t)` structure so that the no-rainfall solution decays
   (mass balance by construction).
3. **Observation model** `wl(t) = g_φ(z(t)) + ε`; the label is then derived
   from `wl` exactly as in the current pipeline (`wl_dist_to_danger > 0`), so
   the supervised signal is the same.
4. **Assimilation** at observed days: nudge `z` toward the posterior
   (`z ← z + K·(wl_obs − g(z))`), with `K` learned.
5. **Solver** `dopri5` with `rtol=1e-5`; adjoint gradients.

## 3. Evaluation plan

* 7 dev cells, strictly out-of-sample.
* Primary: per-cell PR-AUC; secondary: **trajectory RMSE** of `wl` (a metric the
  tabular models cannot produce at all, and the reason to run this experiment).
* **Missing-data stress test:** mask 30 % of days at random and measure
  degradation. The ODE should degrade gracefully; the lag-feature model cannot
  be evaluated at all on masked days.
* Compute: GPU strongly preferred (adjoint through ~1,500-day trajectories).

## 4. Risks

| Risk | Mitigation |
| --- | --- |
| ODE training is unstable / slow | start with a single basin (Ganges, 25 stations) before scaling to 153 |
| Identifiability: many parameter sets fit the same trajectory | report posterior spread; do not claim physical constants are identified |
| Label is a *threshold*, not the level — the ODE's advantage may not show in PR-AUC | report `wl` RMSE as the primary scientific result; PR-AUC as the secondary, champion-relevant one |
| No GPU confirmed | do not start until confirmed |

## 5. Success criteria

| # | Criterion |
| --- | --- |
| 1 | `wl` trajectory RMSE beats a persistence/AR baseline on held-out days |
| 2 | PR-AUC is at least equal to D16 on the 7 dev cells |
| 3 | Under 30 % day masking, PR-AUC degrades by less than the lag-feature model |
| 4 | Physical parameters are stable across seeds (report the spread) |
"""),
    ]


NOTEBOOKS = {
    "01_domain_adaptation": nb01,
    "02_sar_inundation": nb02,
    "03_nwp_ensemble": nb03,
    "04_pino_graph": nb04,
    "05_neural_ode": nb05,
}
