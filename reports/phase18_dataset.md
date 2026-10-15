# Phase 18 — ML/DL dataset preparation (Dataset B)

| | |
| --- | --- |
| **Workflow** | `final workflow.md` **Phase 18 — ML / DL dataset preparation** → checklist **stage 32** |
| **Build** | `scripts/build_sequences.py` → `scripts/phase18_run.log`, **exit 0, 17/17 green, 9.0 s** |
| **Notebook** | `notebooks/18_sequences.ipynb` — **32/32 green, 16.7 s** (built by `scripts/build_phase18_notebook.py`) |
| **Outputs** | `data/processed/sequences_L{3,7,14,30}.npy` · `sequence_index.csv` (200,665 rows) · `sequence_manifest.json` (2,779 B) |
| **Input (immutable)** | `experimental_dataset.csv` SHA-256 `a67485…` **re-verified twice** (script F2a + notebook N18-01b/N18-final) |
| **Prerequisite (workflow)** | `torch==2.14.1+cpu` added to `requirements.txt`; versions recorded below; seeded tiny run reproduces losses exactly twice |
| **Binding decisions** | **D12** four window lengths, none assumed best · labels never in the tensor · static/station columns stay on the Phase-23 fusion branch · gaps = explicit NaN timesteps (policy decided in Phase 19) |

## 1. Dataset A vs Dataset B

| | Dataset A (Phase 14) | Dataset B (this phase) |
| --- | --- | --- |
| form | 200,665 × 67 engineered columns, one row = station-day | `(200665, L, 12)` float32 windows + index |
| consumers | L1/L2 models, fusion branch | L3/L4 models (LSTM/GRU/TCN/…) |
| contents | lags, rolls, neighbour, up_*, labels | 12 raw/near-raw variables only |
| status | **immutable** (SHA re-checked) | reproducible from A/raw in one command |

## 2. The 12 sequence variables

| variable | why it is in the sequence |
| --- | --- |
| `wl_first wl_mean wl_max wl_min wl_last` | the day's observed hydrograph (all readings) |
| `wl_pct_danger` | station's danger utilisation — per-row normalisation, not a rolling aggregate |
| `precip_mm` | the driver of the whole system |
| `tmax_c tmin_c` | thermal season / snowmelt-proxy context |
| `n_readings` | observation-process metadata — a real missingness signal |
| `doy_sin doy_cos` | narrow season encoding (D1), legal at *t* |

**Excluded by contract:** all lag/roll/`rain_*_d`/`rise`/`exceed`/`nb3_`/`nb5_`/`up_*`
columns (keeps **E13** — raw sequence → DL vs engineered → XGBoost — interpretable), all
labels, and every static/geographic column (`lat lon danger_level river elevation_m …`)
which travels on the Phase-23 fusion branch.

## 3. Window semantics

```
X[i, k, :]  = variable values of station s on day (date_i − L + 1 + k)
window      = L calendar days ending ON the anchor day t   (never reaching past t)
```

* built by **calendar scatter**: each station is laid onto its full daily grid
  (2017-01-06 … 2026-10-05 = 3,560 days), then windowed — so an absent calendar day
  becomes an **all-NaN timestep**, never a silent shift (notebook N18-08 proves this
  on 30 anchors × 4 lengths = up to 1,200 timesteps, 0 violations);
* days before a station's first row are **front-padded** with NaN;
* no statistic is fitted anywhere — raw I/O only (manifest records `fitted_statistics: none`);
* row order = `experimental_dataset.csv` row order, so labels/splits join by
  `sample_idx` (`sequence_index.csv`).

## 4. What the tensors look like

| L | file | shape | size | samples with ≥ 1 empty calendar timestep | front-padded |
| ---: | --- | --- | ---: | ---: | ---: |
| 3 | `sequences_L3.npy` | (200665, 3, 12) | 28.9 MB | **2.9 %** | 19 |
| 7 | `sequences_L7.npy` | (200665, 7, 12) | 67.4 MB | **7.5 %** | 19 |
| 14 | `sequences_L14.npy` | (200665, 14, 12) | 134.8 MB | **14.1 %** | 35 |
| 30 | `sequences_L30.npy` | (200665, 30, 12) | 289.0 MB | **26.7 %** | 62 |

Calendar coverage: **200,665 / 473,480 = 42.4 %** of station-days exist in the model
table — the other 57.6 % are gaps (the 2018 archive hole R6 plus station outages). They
are *not* imputed here: **Phase 19 picks the missing-timestep rule on evidence.**
The rising gap fraction with L is the central number of that decision: at L = 30 more
than a quarter of windows contain at least one empty day.

## 5. Reproducibility

```powershell
python -u scripts\build_sequences.py > scripts\phase18_run.log     # 17/17, 9.0 s
```

`sequence_manifest.json` records for every artifact: shape, dtype, byte size and
**SHA-256**, plus the source table hash, variable order, calendar range, gap statistics
and build timestamp. The notebook recomputes all four tensor hashes and the index hash
and compares them to the manifest (N18-04) — a rebuild must be byte-identical.

## 6. Torch prerequisite (workflow Phase 18 bullet)

| package | version |
| --- | --- |
| torch | **2.14.1+cpu** (CPU build from `download.pytorch.org/whl/cpu`, pinned in `requirements.txt`) |
| numpy | 2.3.2 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| CUDA | not available (CPU-only machine) |

Determinism: a seeded (20261008) 5-step tiny training run reproduces its five losses
**exactly** on a second run (notebook N18-12c).

## 7. Checks

**Script — 17/17** (`scripts/phase18_run.log`): F2a source SHA · F2b 1:1 join to split
bookkeeping · F2c/D 12 raw variables, no labels/engineered · F2e/f calendar grid and
unique (station, date) · F2g/h scatter fill + 42.4 % coverage · F2i.{3,7,14,30} shapes,
sizes, gap and front-pad counts · F2j 300-sample full-window value equality · F2k no
future reach (window shifts, never jumps) · F2l/m index order and split counts
77,571 / 21,920 / 101,174 · F2o manifest written.

**Notebook — 32/32** (`notebooks/18_sequences.ipynb`): N18-01 inputs + SHA · N18-02
manifest schema · N18-03 shapes/dtypes vs manifest · N18-04 file hashes vs manifest ·
N18-05 variable contract · N18-06 index order + split counts · N18-07 100-anchor anchor-
day equality · N18-08 30-anchor full-window equality incl. NaN gaps · N18-09 window ends
on *t*, consecutive anchors overlap exactly · N18-10 gap stats recomputed from the
tensors · N18-11 split/fold/holdout/season 1:1 with Phase-15 masks (39,725 holdout
samples flagged) · N18-12 torch versions + seeded-run reproducibility.

---

**Next: Phase 19 — Sequence construction & validation (stage 33)** —
`notebooks/09_sequence_validation.ipynb` with S-ids: chronological order, gap
distribution, the written missing-timestep rule (kept/dropped per L), no station
mixing, no forward reach, 100-window label recomputation, split/holdout inheritance —
then `reports/phase19_sequences.md`.
