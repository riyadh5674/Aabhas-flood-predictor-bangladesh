# MI-8 — score push: feature block + extended-budget HPO (dev cells only)

| | |
| --- | --- |
| **Stage** | MI-8 — the score push: a 55-column feature block, an extended-budget HPO on it, and an equal-budget control |
| **Decision** | **D16 stands.** No candidate beats `hpo_perhorizon_FULL_V2` on the pre-registered decision cell (`val_2022`, season=all, horizon-mean PR-AUC). The champion is unchanged at **0.957427**. |
| **Commands** | `scripts/mi8_features.py` · `scripts/mi8_probe.py` · `scripts/mi8_search.py --arm full_v3` · `scripts/mi8_search.py --arm full_v2` · `scripts/mi8_select.py --arm full_v3` |
| **Run logs** | `scripts/mi8_probe.log` · `scripts/mi8_search_v3.log` · `scripts/mi8_search_v2.log` · `scripts/mi8_search_v2_resume.log` · `scripts/mi8_select_run.log` |
| **Exit / wall** | probe `0` / 169 s · search `full_v3` `0` / **11 685 s** · search `full_v2` `0` / first leg + **3 587 s** resume · select `0` / **298 s** |
| **Seed** | `20261008` (`baseline_model.B.SEED`) — every stage |
| **Evidence** | **7 dev cells only** — `test_rows_read = 0`; no `>= 2023` row is scored by any script in this stage |
| **Artefacts** | `reports/mi8_probe.{csv,json}` · `reports/mi8_hpo_full_{v2,v3}.csv` + `partial_*` · `models/mi/mi8_hpo_*_<horizon>.json` · `reports/mi8_scorecard.csv` · `reports/mi8_paired_deltas.csv` · `reports/mi8_summary.json` |

> **Correction (2026-10-22).** All **seven-cell** results in this report were produced by a
> single model fitted on `split == "train"` (<=2021) and then scored on `val_2020`,
> `val_2021`, and `spatial_fold_1..4`. Those six cells are subsets of the fit rows, so their
> absolute scores are **in-sample** and are **withdrawn** (the 0.9999 saturation is gone).
> The per-cell re-scoring in `reports/p2p3_correction.md` gives: FULL 0.9641, FULL_V2
> 0.9659, FULL_V3 0.9678; V3-V2 **+0.00187** on **20/21** points (not +0.00135/21-21);
> the MI-8 arm `full_v3` is **+0.00203** above D16 on the corrected seven-cell mean, but
> **-0.00004** below it on `val_2022`. **D16 remains the champion** under the pre-registered
> `val_2022` rule. Do not quote the archived seven-cell magnitudes; use the corrected values.

## 1. What this stage did

MI-5 froze **D16** at `val_2022` horizon-mean **0.957427** (`lgbm7` @ 24 h/48 h, `xgb3` @ 72 h;
`FULL_V2` = 110 features). MI-8 attacked that number with the two levers still available:

1. **Features** — 55 new columns (`m8_*`) derived from the response history, rain, upstream and
   neighbour gauges, GloFAS, plus a data-availability flag (`FULL_V3` = 165 features).
2. **Budget** — a per-horizon HPO at 71 configs/horizon against MI-3b's 41, run on **both**
   feature sets so the two effects are separable.

The design point is that lever 2 is run **twice**: `full_v3` (new features) and `full_v2`
(same budget, *old* features). Without the control, any `full_v3` gain could be attributed to
either lever.

## 2. The feature block is a real fixed-config win

`mi8_probe.py` holds the estimator **fixed** (the Phase-33 `xgboost` config) and swaps feature
sets, so it measures the features alone. It reproduces MI-3a exactly (`diff = 0.0` for both
`FULL` and `FULL_V2`), which validates the probe before it reports anything new.

| feature set | cols | `val_2022` h-mean | all-dev horizon-mean |
| --- | ---: | ---: | ---: |
| `FULL` (v1) | 62 | 0.95463 | 0.98341 |
| `FULL_V2` | 110 | 0.95589 | 0.98685 |
| **`FULL_V3`** | **165** | **0.95647** | **0.98820** |
| **Δ (`v3 − v2`)** | +55 | **+0.00058** | **+0.00135** |

- Improvement is **uniform**: **21/21** (cell × horizon) points improve.
- It is monotone in horizon: `v3 − v2` = +0.00091 / +0.00154 / **+0.00161** (24 h/48 h/72 h) on
  the all-dev mean — the long horizons gain most, which is the physically sensible direction.
- `test_rows_read = 0`.

> **Correction.** Only the `val_2022` column survives unchanged. The all-dev column and the
> 21/21 uniformity above are in-sample and are superseded. The per-cell re-scoring gives a
> corrected `FULL_V3 − FULL_V2` mean of **+0.00187** (vs +0.00135 archived), positive in
> **20/21** points; the single loss is `val_2021` 24 h (−0.0007). See
> `reports/p2p3_correction.md`.

So the feature block is **proven** at fixed config. It is *not* yet a champion win, and neither
is the ceiling it implies: the fixed-config `FULL_V3` number is **0.95647**, still below D16's
**0.95743** — i.e. the features alone do not close the gap that MI-3b's search already closed.

## 3. Extended HPO — the budget helps, and more so on the *old* features

71 trials/horizon = 1 `frozen` + 30 `xgb` + 30 `lgbm` + 10 `hgb`; 213 fits total. Selection is
per-horizon `argmax` of `val_2022` PR-AUC — the pre-registered D16 rule.

| arm | 24 h | 48 h | 72 h | horizon-mean | vs D16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D16 frozen (`FULL_V2`, MI-3b, 41 cfg) | 0.970789 | 0.956485 | 0.945007 | 0.957427 | — |
| **`full_v3`** (new feats, 71 cfg) | 0.970662 `xgb23` | 0.956765 `xgb6` | 0.944737 `xgb17` | **0.957388** | **−0.00004** |
| **`full_v2`** (old feats, 71 cfg) | **0.971983** `lgbm6` | 0.956365 `xgb26` | 0.945007 `xgb3` | **0.957785** | **+0.00036** |

Read against the fixed-config probe:

| feature set | fixed config | after 71-cfg HPO | search gain |
| --- | ---: | ---: | ---: |
| `FULL_V2` | 0.95589 | 0.957785 | **+0.00190** |
| `FULL_V3` | 0.95647 | 0.957388 | **+0.00092** |

So on the decision cell the **search** was worth roughly 2–3× the **feature block**, and the
new feature set actually extracted *less* from the same budget. That reverses the fixed-config
ordering and is the first sign that the two arms are inside the noise band of one cell.

### Why more budget cannot fix it

The search is saturated, and D16 sits on the plateau:

- Top-5 configs per horizon are within **0.00031 / 0.00044 / 0.00112** of each other (24/48/72 h);
  the full 71-config range is only 0.005–0.008.
- Of 71 configs, **0 beat D16 at 24 h, 0 at 72 h, and 4 at 48 h — by at most +0.00028**.
- Selector disagreement: Spearman(`val_2022`, 7-cell mean) = 0.69 / 0.52 / **0.31**. At 72 h the
  two cells barely agree, so the ranking at that horizon is largely noise.

## 4. Verification — refit, 7-cell scorecard, paired bootstrap

`mi8_select.py` refits each arm's per-horizon winner plus a reconstructed D16, scores all seven
dev cells, and runs a **paired station-level bootstrap** (500 resamples, stations resampled not
rows, identical resample index for both models; `D8`).

**Honesty gate — D16 must reproduce from `mi3_hpo.csv` before any comparison is reported:**

| horizon | refit | archived | Δ |
| --- | ---: | ---: | ---: |
| 24 h | 0.970789 | 0.970789 | **−1.8e-07** |
| 48 h | 0.956485 | 0.956485 | **−4.7e-07** |
| 72 h | 0.945007 | 0.945007 | **−2.8e-07** |

A free end-to-end check confirms the rig: **`full_v2` @ 72 h selects `xgb3` on `FULL_V2` —
the identical spec to D16 — and is bit-identical to it** (paired delta `0.0`, sd `0.0`). The
same features, same seed and same config give the same model.

**Decision cell (`val_2022`, season=all, horizon-mean PR-AUC):**

| arm | 24 h | 48 h | 72 h | **mean** | vs D16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D16 frozen | 0.970789 | 0.956485 | 0.945007 | **0.957427** | — |
| `full_v3` | 0.970662 | 0.956765 | 0.944737 | 0.957388 | −0.00004 |
| `full_v2` | 0.971983 | 0.956365 | 0.945007 | 0.957785 | +0.00036 |

**Paired 95 % CI, Δ = arm − D16 (500 station resamples):**

| contrast | 24 h | 48 h | 72 h |
| --- | --- | --- | --- |
| `full_v3` − D16 | −0.00024 [−0.00340, +0.00199] | +0.00018 [−0.00220, +0.00219] | −0.00024 [−0.00462, +0.00425] |
| `full_v2` − D16 | +0.00122 [−0.00072, +0.00406] | −0.00013 [−0.00197, +0.00146] | 0.00000 [0, 0] |

**Every contrast straddles zero at every horizon.** The control's best-looking number
(+0.00122 at 24 h, 89.2 % of resamples positive) still has a lower bound of −0.00072.

Note also that these deltas are **upward-biased as estimates of procedure quality**: the winner
in each arm is chosen by `argmax` on the *same* `val_2022` cell the bootstrap then resamples,
so the winner's curse is *inside* the point estimate and *not* inside the CI.

## 5. The one disagreement worth recording: the 7-cell mean

> **Corrected (2026-10-22).** The seven-cell means in this section are **in-sample for six
> cells**: one model was fitted on <=2021 and then scored on `val_2020`, `val_2021`, and
> folds 1-4, which are subsets of that fit. The per-cell refits are in
> `reports/p2p3_correction.md`. The disagreement **survives** after correction: on the
> corrected seven-cell mean, `full_v3` is **+0.00203** above D16, while on `val_2022` it is
> **-0.00004** below. The disagreement is therefore real, not an artefact of the in-sample
> cells. The pre-registered rule selects on `val_2022`, so **D16 stands**; both numbers are
> reported rather than re-selecting.

On the mean over all seven dev cells, the new features are **unambiguously better**:

| arm | dev7 horizon-mean | vs D16 |
| --- | ---: | ---: |
| D16 frozen | 0.988520 | — |
| `full_v2` | 0.988922 | +0.00040 |
| **`full_v3`** | **0.990160** | **+0.00164** |

Per-horizon, `full_v3` − D16 on the dev7 mean: **24 h +0.00153 · 48 h +0.00263 · 72 h +0.00076**
— positive at all three, consistent with the fixed-config probe.

**So the `val_2022` decision cell and the 7-cell mean disagree about which arm is better.** Both
selectors are known to be imperfect (`dev_mean` is a poor selector — the MI-3c lesson), and here
the disagreement is structural: `val_2022` is a single held-out year in a different regime, and
at 24 h it is the *only* dev cell with headroom (0.9708 vs 0.9999–1.0000 on the other six).

This is recorded, not resolved. No post-hoc rule change is made: the pre-registered decision
cell governs, and it says no.

## 6. Resolved - the 48 h saturation pattern (audited 2026-10-22)

> **Resolved.** This section correctly flagged the saturation signature. The audit showed it
> was **in-sample evaluation**, not a feature leak or a base-rate ceiling: six of the seven
> cells were subsets of the single ≤2021 fit. The per-cell correction is in
> `reports/p2p3_correction.md`. The recommendation below ("saturation is a feature") is
> **retracted**; the corrected cells do not saturate. The `val_2022` cell was always
> out-of-sample and is unchanged.

At 48 h, `full_v3` drives **six** dev cells to 0.99997–1.00000 while `val_2022` moves only
0.956485 → 0.956765. D16's same six cells span 0.9943–0.9985. Independently, at 24 h `full_v2`
reaches **exactly 1.00000** on six cells while `val_2022` sits at 0.97198.

A block of cells saturating to perfect separation while the held-out year does not is the
pattern a leakage audit exists to rule out. It is *not* evidence of leakage — the block is
`val_2020`, `val_2021` and the four spatial folds, which are all drawn from the `train` split
and are therefore plausibly just easier — but the `m8_*` block contains upstream (`m8_up_*`),
neighbour (`m8_nb5_*`, `m8_nb3_*`) and long-window (`m8_pct_z180`, 180-day z-score) features,
which is exactly where a non-causal window would hide.

**Recommended check before any of the +0.00164 is trusted:** audit the 55 `m8_*` columns against
the Phase-13 leakage rule — specifically that every window is strictly causal (no negative
`shift`), that no feature is computed over rows including the dev cells, and that the 48 h gain
is not concentrated in the already-saturated cells.

## 7. Decision

> **D16 stands.** The MI-8 feature block and the extended-budget HPO were both run to completion
> on the seven dev cells with the frozen test unread (`test_rows_read = 0`). No candidate beats
> `hpo_perhorizon_FULL_V2` on the pre-registered decision cell (`val_2022`, season=all,
> horizon-mean PR-AUC = 0.957427): `full_v3` is −0.00004 and the equal-budget `full_v2` control
> is +0.00036, and **no paired 95 % station-bootstrap CI excludes zero at any horizon**. The
> champion is unchanged; the MI-8 feature block is recorded as a real cross-cell improvement
> (+0.00164 on the 7-cell horizon-mean, +0.00058 at fixed config on `val_2022`) that the decision
> cell cannot resolve, pending the §6 audit.

## 8. Honest read

- **MI-8 did not move the score.** The decision cell is unchanged to four decimals and the
  paired CIs are wide enough to contain both the proposed gain and a comparable loss.
- **The feature block is not useless — it is unconfirmed.** It wins at fixed config (+0.00058),
  across cells (+0.00164), and at every horizon; the single pre-registered year cannot see it.
- **The budget lever is exhausted.** D16 sits on the top of the 71-config plateau: beating it
  requires fewer than 1 config in 71 to be better, at every horizon.
- **The remaining headroom is not in this cell.** `val_2022` at 24 h is the only dev cell with
  slack, and MI-4 already bounded the label at ≈0.96–0.97. Every lever inside this cell is now
  capped by label noise, not by model capacity.
- **The honest next questions are elsewhere:** the §6 leakage audit (cheap, and it either
  strengthens or voids the +0.00164), and — if a real score move is still the goal — the years
  the programme has never modelled, which means a pre-registered additional test read. That is
  an owner decision, not a scripted one.

## 9. Reproduce

```
cd "D:\Flood predictor"
.venv\Scripts\python.exe -u scripts\mi8_features.py                    # -> data/processed/mi8_features.csv (FULL_V3 = 165)
.venv\Scripts\python.exe -u scripts\mi8_probe.py                       # -> reports/mi8_probe.{csv,json}      (~169 s)
.venv\Scripts\python.exe -u scripts\mi8_search.py --arm full_v3        # -> reports/mi8_hpo_full_v3.csv       (~11 685 s)
.venv\Scripts\python.exe -u scripts\mi8_search.py --arm full_v2        # -> reports/mi8_hpo_full_v2.csv       (resumable)
.venv\Scripts\python.exe -u scripts\mi8_select.py   --arm full_v3      # -> reports/mi8_scorecard.csv, mi8_paired_deltas.csv, mi8_summary.json (~298 s)
```

Exit `0`; deterministic given seed `20261008`; `test_rows_read = 0`.

`mi8_search.py` is **resumable**: it skips any `(horizon, tag)` pair already present in
`reports/mi8_hpo_partial_<arm>.csv`, so an interrupted run is continued by re-issuing the same
command. `mi8_select.py` accepts `--horizons` for a cheap single-horizon check and refuses to
write its artefacts unless all three horizons are present.

## 10. Files

| file | role |
| --- | --- |
| `scripts/mi8_features.py` | builds the 55 `m8_*` columns → `FULL_V3` |
| `scripts/mi8_probe.py` | fixed-config A/B of `FULL` / `FULL_V2` / `FULL_V3` |
| `scripts/mi8_search.py` | per-horizon HPO, arm `full_v3` or `full_v2`, resumable |
| `scripts/mi8_select.py` | winner refit + 7-cell scorecard + paired station bootstrap vs D16 |
| `reports/mi8_probe.json` | feature A/B, reproduction `diff = 0.0`, 21/21 cells |
| `reports/mi8_hpo_full_v3.csv` | 213 configs on `FULL_V3` |
| `reports/mi8_hpo_full_v2.csv` | 213 configs on `FULL_V2` (equal-budget control) |
| `models/mi/mi8_hpo_*_<horizon>.json` | per-arm per-horizon winners + configs |
| `reports/mi8_scorecard.csv` | 63 rows = 3 arms × 3 horizons × 7 cells |
| `reports/mi8_paired_deltas.csv` | 6 paired station-bootstrap contrasts |
| `reports/mi8_summary.json` | machine-readable summary of the stage |
