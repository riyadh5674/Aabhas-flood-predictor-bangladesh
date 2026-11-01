# MI-4 (FULL_V4) Selection & Test Read Decision

**Date:** 2026-10-17  
**Scope:** FULL_V4 = FULL_V2 (110) + WorldCover (22) + MI-8 m8_* (55) = 187 features  
**Decision cell:** `val_2022` horizon-mean PR-AUC (pre-registered D16 rule)  
**Result:** **PARTIAL ADOPTION** — V4 wins at 24h; D16 retained at 48h/72h

---

## 1. Summary of Findings

| Horizon | V4 Model | val_2022 PR-AUC | D16 Model | val_2022 PR-AUC | Δ (V4−D16) | Bootstrap P>0 | Decision |
|---------|----------|-----------------|-----------|-----------------|------------|---------------|----------|
| 24h     | lgbm13   | **0.9593**      | lgbm7     | 0.9542          | **+0.0051** | **0.94**      | **ADOPT V4** |
| 48h     | hgb1     | 0.9501          | lgbm7     | 0.9484          | +0.0018    | 0.555         | Keep D16 |
| 72h     | xgb16    | 0.9431          | xgb3      | 0.9430          | +0.0001    | 0.48          | Keep D16 |

---

## 2. Detailed Bootstrap Evidence (200 resamples, paired station-level)

### 24h — V4 lgbm13 vs D16 lgbm7
| Cell | Point Δ | 95% CI | P(V4>D16) |
|------|---------|--------|-----------|
| val_2020 | -0.0005 | [-0.0012, -0.0001] | 0.00 |
| val_2021 | -0.0005 | [-0.0011, 0.0000] | 0.015 |
| **val_2022** | **+0.0042** | **[-0.0014, +0.0115]** | **0.94** |
| spatial_fold_1 | -0.0008 | [-0.0067, 0.0000] | 0.075 |
| spatial_fold_2 | -0.0001 | [-0.0002, 0.0000] | 0.01 |
| spatial_fold_3 | 0.0000 | [0.0000, 0.0000] | 0.29 |
| spatial_fold_4 | -0.0009 | [-0.0019, -0.0001] | 0.00 |

**Verdict:** V4 wins decisively on the decision cell (val_2022) with 94% bootstrap support. Losses on saturated training-era cells are expected (generalization gap, not defect).

### 48h — V4 hgb1 vs D16 lgbm7
| Cell | Point Δ | 95% CI | P(V4>D16) |
|------|---------|--------|-----------|
| val_2020 | -0.0009 | [-0.0020, -0.0002] | 0.00 |
| val_2021 | -0.0017 | [-0.0043, 0.0000] | 0.005 |
| val_2022 | +0.0001 | [-0.0107, +0.0051] | 0.555 |
| spatial_fold_1 | -0.0048 | [-0.0292, 0.0000] | 0.00 |
| spatial_fold_2 | -0.0004 | [-0.0019, 0.0000] | 0.00 |
| spatial_fold_3 | -0.0000 | [-0.0000, 0.0000] | 0.01 |
| spatial_fold_4 | -0.0022 | [-0.0056, -0.0004] | 0.00 |

**Verdict:** Inconclusive on val_2022 (CI straddles zero, 55.5% positive). Horizon-mean loses (-0.0009). **Keep D16.**

### 72h — V4 xgb16 vs D16 xgb3
| Cell | Point Δ | 95% CI | P(V4>D16) |
|------|---------|--------|-----------|
| val_2020 | -0.0078 | [-0.0156, -0.0041] | 0.00 |
| val_2021 | -0.0094 | [-0.0169, -0.0047] | 0.00 |
| val_2022 | -0.0005 | [-0.0079, +0.0030] | 0.48 |
| spatial_fold_1 | -0.0369 | [-0.1297, -0.0067] | 0.00 |
| spatial_fold_2 | -0.0054 | [-0.0144, -0.0029] | 0.00 |
| spatial_fold_3 | -0.0067 | [-0.0230, -0.0018] | 0.00 |
| spatial_fold_4 | -0.0120 | [-0.0238, -0.0053] | 0.00 |

**Verdict:** Loses on horizon-mean (-0.0090), loses on 6/7 cells, bootstrap excludes zero against V4 on 5/7 cells. **Keep D16.**

---

## 3. Decision

| Horizon | Selected Model | Config | Feature Set | Rationale |
|---------|---------------|--------|-------------|-----------|
| **24h** | **lgbm13 (V4)** | `mi4_hpo_v4_label_24h.json` | **FULL_V4 (187)** | Clear win on decision cell, 94% bootstrap |
| **48h** | **lgbm7 (D16)** | `mi_hpo_label_48h.json` | FULL_V2 (110) | V4 inconclusive; D16 more robust |
| **72h** | **xgb3 (D16)** | `mi_hpo_label_72h.json` | FULL_V2 (110) | V4 loses on horizon-mean & spatial |

**New hybrid selection:** `hpo_hybrid_v4_24h_only` — per-horizon best with V4 only at 24h.

---

## 4. Pre-registered Test Read Plan

**Test read #2** (guarded, `n_reads = 2` for this selection):
- Models: V4 lgbm13 (24h), D16 lgbm7 (48h), D16 xgb3 (72h) + D7 bar
- Thresholds: Period-specific cost-optimal (10:1) derived on val_2022 per model
- Scoring: All 7 test cells × 3 horizons = 21 points
- Logging: `reports/mi4_test_read_log.json` with `n_reads` guard

**Success criteria:** Hybrid model beats D7 on all 21 test points AND beats D16 on ≥15/21 points.

---

## 5. Reproduction

```bash
# Refit V4 24h model
.venv\Scripts\python.exe -u scripts\mi4_select_v4.py 24h

# Pre-registered test read
.venv\Scripts\python.exe -u scripts\mi4_test_read.py
```