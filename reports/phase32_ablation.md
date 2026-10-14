# Phase 32 — Ablation & comparative experiment matrix (stages 55, 56)

**Status:** DONE · **Seed:** `20261008` · **Wall-clock:** 1,658 s
**Dataset:** `data/processed/experimental_dataset.csv` (200,665 rows), SHA-256
`a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e` — unchanged.
**Test period (`>= 2023`) never read:** `test_rows_read = 0`.

**Question (workflow Phase 32).** *Which information actually contributes to flood
prediction, and how do the model families compare once the input is the only thing that
changes?* One planned matrix (E0–E16) is run once and reported as a research story — every
result **per cell, never pooled** (D8).

---

## 0. Method

### 0.1 Nested feature-group ablation (E1–E5, E11) — the new run

The FULL 62-feature set is partitioned into five **disjoint layers** whose prefix sums give
five cumulative nested groups; `G5_full == FULL` is asserted at runtime:

| Group | Layers | n |
| --- | --- | --- |
| `G1_weather` | weather | 5 |
| `G2_weather_rainhist` | + rainfall history | 12 |
| `G3_plus_geography` | + geography (admin text + terrain/distances) | 26 |
| `G4_plus_hydrology` | + local hydrology (danger level + water-level family) | 50 |
| `G5_full` | + station network (`nb3_*`/`nb5_*` neighbourhood + `up_*` upstream) | **62** |
| `G5_no_up` | FULL minus the four `up_*` columns (E11 / F14-2) | 58 |

Layers are `weather (5) + rain_hist (7) + geography (14) + hydrology (24) + network (12) = 62`
and are verified to be disjoint and to tile FULL exactly.

**The learner is held fixed and only the input changes.** Two fixed learners are used — the
Phase-20 family winners `xgboost_c01` and `lightgbm_c00` — each with its **complete**
FULL-tuned hyper-parameters (not re-tuned per subset), so the feature contribution is
isolated from the tuning. Search space: 6 groups × 2 learners × 7 dev cells × 3 seasons ×
3 horizons = **756 scored rows** (252 fits).

Every group is scored on all seven Phase-15 dev cells (`val_2020/2021/2022`, `spatial_fold_1..4`),
each with micro (pooled) and macro (mean per-station) PR-AUC, ROC-AUC, Brier, ECE, plus a
**station-bootstrap 95 % CI** (100 reps, season `all`). The scientific heart is the **paired
station-bootstrap CI on the increments** between consecutive groups (G2−G1, G3−G2, G4−G3,
G5−G4, G5_no_up−G5) — e.g. "what does adding hydrology actually buy?".

### 0.2 Bar and reuse

* **E0** adds fresh D7 bars: `persistence` (rank by `wl_pct_danger`) and the no-skill
  `climatology` constant, re-fit on every cell × horizon × season (126 reference rows).
* **E6–E10** fall out of the same run (full model on the spatial folds, on the monsoon
  slice, and per horizon).
* **E12–E16** integrate already-frozen evidence (Phase-16/21/23/24/25/26/27) — no retraining —
  and are given explicit status (reused / blocked), never silently dropped.
* **Test (`>= 2023`) is never read:** training and eval masks are asserted test-free inside
  the loop; `test_rows_read = 0`.

---

## 1. Nested feature-group ablation (E1–E5)

**Temporal macro PR-AUC** (mean over `val_2020/21/22`, micro = pooled), xgboost; lightgbm is
within ±0.001 everywhere (full table in `phase32_groups.csv`):

| Group | n | 24 h | 48 h | 72 h | macro/station 24 h |
| --- | ---: | ---: | ---: | ---: | ---: |
| `G1_weather` | 5 | 0.1585 | 0.1717 | 0.1852 | 0.2901 |
| `G2_weather_rainhist` | 12 | 0.1673 | 0.1797 | 0.1986 | 0.3567 |
| `G3_plus_geography` | 26 | 0.4759 | 0.4995 | 0.5095 | 0.4281 |
| `G4_plus_hydrology` | 50 | **0.9775** | **0.9660** | **0.9520** | 0.8941 |
| `G5_full` (E5) | 62 | 0.9782 | 0.9664 | 0.9534 | 0.8938 |
| `G5_no_up` (E11) | 58 | 0.9783 | 0.9665 | 0.9531 | 0.8934 |
| **D7 `persistence`** (E0) | — | 0.9429 | 0.9250 | 0.9048 | — |
| `climatology` (E0) | — | 0.0877 | 0.0967 | 0.1043 | — |

**Increment table** (xgboost, temporal cells; paired station-bootstrap mean, over 9
cell × horizon points):

| Increment | mean ΔPR-AUC | range | significant |
| --- | ---: | ---: | ---: |
| `G2 − G1` (+ rainfall history) | **+0.0101** | −0.0121 … +0.0500 | 3/9 |
| `G3 − G2` (+ geography) | **+0.3131** | +0.1852 … +0.4244 | 9/9 |
| `G4 − G3` (+ hydrology) | **+0.4702** | +0.3892 … +0.5385 | 9/9 |
| `G5 − G4` (+ station network) | +0.0009 | −0.0002 … +0.0037 | 2/9 |
| `G5_no_up − G5` (drop `up_*`) | −0.0001 | −0.0011 … +0.0004 | 2/9 |

The full per-cell increment matrix (all 7 cells × 3 horizons, both learners, with CIs) is in
`phase32_increments.csv`.

---

## 2. Full model across protocols (E6–E10) and the D7 bar

**`G5_full` (xgboost) per cell**, season `all`, 24 h micro / macro / CI:

| Cell | micro PR-AUC | macro/station | 95 % CI | n | stations | base rate |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| val_2020 | 0.9859 | 0.9150 | [0.9758, 0.9916] | 15,901 | 74 | 0.1418 |
| val_2021 | 0.9815 | 0.8685 | [0.9627, 0.9905] | 17,369 | 84 | 0.0703 |
| val_2022 | 0.9674 | 0.8979 | [0.9323, 0.9865] | 17502 | 84 | 0.0509 |
| spatial_fold_1 | 0.9653 | 0.9146 | [0.9472, 0.9865] | 15196 | 21 | 0.0881 |
| spatial_fold_2 | 0.9910 | 0.9284 | [0.9791, 0.9964] | 16260 | 22 | 0.0982 |
| spatial_fold_3 | 0.9866 | 0.9049 | [0.9586, 0.9965] | 14778 | 19 | 0.1078 |
| spatial_fold_4 | 0.9588 | 0.8817 | [0.9074, 0.9808] | 15743 | 22 | 0.0898 |

* **E6** spatial (mean 4 folds): 0.9754 / 0.9617 / 0.9490.
* **E7** monsoon vs non-monsoon (temporal macro): 0.9801 / 0.9684 / 0.9561 vs
  0.9695 / 0.9577 / 0.9406 — a gap of only 0.0106 / 0.0107 / 0.0155.
* **E8–E10** temporal full: **0.9782** [0.9570, 0.9895] · **0.9664** [0.9374, 0.9826] ·
  **0.9534** [0.9186, 0.9737].
* **The full model beats the D7 bar in 21/21 (cell × horizon) points**, min margin **+0.0144**
  (`spatial_fold_2`, 24 h: 0.9910 vs 0.9766). The D7 bar beats climatology in 63/63 reference
  points (7 cells × 3 horizons × 3 seasons) — a real bar, not a dummy.

`G5_full` remains essentially flat across the season slices (monsoon 0.9801 vs non-monsoon
0.9695 at 24 h) and degrades monotonically with lead time on every protocol.

---

## 3. The `up_*` contrast (E11 / F14-2)

Dropping the four upstream (`up_*`) columns changes the temporal macro PR-AUC by
**−0.0001 on average**, paired 95 % CI [−0.0023, +0.0016]; only 1 of 9 temporal points has a
CI excluding zero (val_2021/72 h, −0.0011). The Phase-16 finding **F14-2 is confirmed at full
capacity**: `up_*` is redundant once the local hydrology + neighbourhood layers are present.

---

## 4. Reused comparisons (E13–E16) and the blocked row (E12)

All numbers below come from the frozen artefacts (`phase26_interaction.json`,
`phase27_metrics.csv`, `phase23_ablation.csv`, `phase24_ablation.csv`, `phase25_ablation.csv`);
they were re-read and tabulated in `phase32_reuse_detail.csv`.

* **E13 — engineered (XGBoost) vs raw sequence (DL).** Temporal macro gaps of
  **+0.0204 / +0.0180 / +0.0209** (24/48/72 h); on the unseen-station folds the gap widens to
  **+0.0603 / +0.0416 / +0.0322**. The Phase-26 interaction ("ML beats DL on P1, DL decays
  more slowly on P3") is reproduced and made explicit.
* **E14 — sequence length L = 3/7/14/30.** L=3 is best (temporal macro 24 h **0.9519** vs
  0.9460 / 0.9474 / 0.9485); longer windows do not help.
* **E15 — spatial ablation.** temporal-only `lstm_L3` 0.9519 → +location `fusion_L3` 0.9556
  (matched-capacity P1 Δ24 h **+0.0024**); +graph edges on the unseen-station folds: GNN
  0.9088 vs LSTM 0.9011 (Δ **+0.0077**). The "+river-basin" variant *is* the approved
  river-first graph (52 river-adjacency edges ∪ kNN fallback), so it is covered by the graph
  row rather than a separate run.
* **E16 — architecture ablation.** LSTM 0.9519 · +static 0.9556 · +attention 0.9521 · TCN
  0.9460 · GRU 0.9470 · GNN-temporal 0.9529. **TCN+static** was not run (no gated budget after
  the static branch proved marginal); recorded as *not-run*, not as a null.
* **E12 — soil/landcover groups: BLOCKED by R8.** The processed table carries no
  soil/landcover groups (Phase-13 dependency register), so there is nothing to ablate. The
  blocker is recorded in the matrix, not hidden.

The optional **multi-task head deferred from Phase 26 §9** is outside the E0–E16 scope; it
would require a new architecture + gate (D13) and is left as future work.

---

## 5. Named findings and limitations

**F32-1 — the ablation is a hydrology story.** Weather alone (G1, 0.1585–0.1852) and
weather+rain (G2, 0.1673–0.1986) sit only marginally above the no-skill climatology bar
(0.0877–0.1043) and are **far below the D7 persistence bar** (0.9048–0.9429) — i.e. no usable
skill. Adding geography (G3) lifts skill to 0.4759–0.5095, and
adding hydrology (G4) is the single largest layer (**+0.4702 mean, significant in 9/9**) —
it is hydrology that clears the D7 bar and reaches 0.95–0.98.

**F32-2 — the station-network layer is marginal.** +network (G5−G4) buys **+0.0009** mean
(significant at only 2/9 points, max +0.0037). The neighbourhood/upstream information adds
essentially nothing on top of local hydrology.

**F32-3 — `up_*` is redundant (F14-2 confirmed).** E11 mean −0.0001, CI [−0.0023, +0.0016] —
indistinguishable from zero. The 58-feature model is equivalent to the 62-feature model.

**F32-4 — geography is a high-value, cheap layer.** G3 (26 features) already recovers ~half
the achievable skill over the weather baseline, and it is only the hydrology layer that
carries the rest.

**F32-5 — macro/micro gap is a model-strength artefact.** For the weak groups macro exceeds
micro (G1 0.29 vs 0.16; G2 0.36 vs 0.17) because a weak but *station-consistent* ranking can
average above the pooled score; for the strong model macro (0.89) < micro (0.98). Both are
reported per cell, never pooled (D8).

**F32-6 — the leader ordering is robust to the fixed-learner choice.** xgboost and lightgbm
agree to ±0.001 at every group × horizon, so the ordering is a property of the features, not
of one learner's idiosyncrasies.

**F32-7 — full beats D7 everywhere, monotone in lead time.** 21/21 cell × horizon points beat
the D7 bar (min margin +0.0144), and PR-AUC decays monotonically 24 h > 48 h > 72 h on every
protocol.

**F32-8 — season slices are close** (monsoon vs non-monsoon gap ≤ 0.0155 on the temporal
cells), unlike the unseen-station folds in Phase 31.

**Limitations.** (i) Hyper-parameters are held fixed at the FULL-tuned winners across all
groups — a deliberate design choice to isolate the feature contribution, but a subset could in
principle be tuned better; the two-learner agreement mitigates this. (ii) `G3` geography
includes station-identifying text (river/basin/district/upazila) whose value partly reflects
station-level propensity rather than transferable terrain signal — a known consequence of the
Phase-12 design. (iii) E13–E16 are **reused** evidence from earlier phases, not re-run here.
**No pooled-only headline:** macro is always reported beside micro, and every result is per
cell.

---

## 6. Provenance

| Artefact | Role |
| --- | --- |
| `scripts/phase32_ablation.py` | driver (exit 0, 1658 s, seed `20261008`) |
| `scripts/phase32_run.log` | run record (`… TOTAL 1658s - exit=0`) |
| `reports/phase32_groups.csv` (756) | nested groups × learner × cell × season × horizon |
| `reports/phase32_reference.csv` (126) | E0 persistence / climatology bars |
| `reports/phase32_increments.csv` (210) | paired station-bootstrap increments |
| `reports/phase32_matrix.csv` (22) | E0–E16 matrix (status / value / CI / source) |
| `reports/phase32_reuse_detail.csv` (39) | reused E13–E16 per-model detail |
| `reports/phase32_summary.json` | headline aggregates + seed/SHA/`test_rows_read` |
| `figures/phase32_ablation.png` | PR-AUC vs feature group, micro + macro |
| `scripts/build_phase32_notebook.py`, `notebooks/32_ablation.ipynb` | executed verification |

**Reproduce:** `.venv\Scripts\python.exe -u scripts/phase32_ablation.py`
