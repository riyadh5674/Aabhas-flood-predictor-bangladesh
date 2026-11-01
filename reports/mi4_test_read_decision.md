# MI-4 Test Read #2 — Hybrid V4 Decision

> **RETRACTED — decision D18 (2026-10-21). Do not quote any number in this file.**
> This read applied a complete-case filter (31.5 % of test rows retained, with a 1.5–1.8×
> higher base rate) and a non-Phase-15 `test_new_network` mask. Both the hybrid and the
> "MI-6 D16" columns are therefore not comparable with the valid MI-6 read. The D17 hybrid
> adoption is withdrawn; **D16 stands as champion**. See `reports/d17_comparability_audit.md`
> and `reports/mi9_test_read_decision.md`. The appendix is preserved only as a record of the
> error. `reports/test_read_ledger.md` is the authoritative read count.

**Date:** 2026-10-17  
**Test read:** #2 (guarded, `n_reads = 1` for this selection) — **INVALID, see banner**  
**Models:** Hybrid — V4 lgbm13 (24h, FULL_V4) + D16 lgbm7 (48h, FULL_V2) + D16 xgb3 (72h, FULL_V2)  
**Comparison:** MI-6 D16 (`hpo_perhorizon_FULL_V2`) + D7 persistence

---

## 1. Headline — Test PR-AUC (mean over 7 cells)

| Horizon | **MI-4 Hybrid** | MI-6 D16 | D7 Persistence | Δ (MI-4 − MI-6) | Δ (MI-4 − D7) |
|---------|-----------------|----------|----------------|-----------------|---------------|
| **24h** | **0.8572**      | 0.8061   | 0.7139         | **+0.0511**     | **+0.1433**   |
| **48h** | **0.7932**      | 0.7289   | 0.6373         | **+0.0643**     | **+0.1559**   |
| **72h** | **0.7444**      | 0.6662   | 0.5769         | **+0.0782**     | **+0.1675**   |

**All three horizons show substantial gains over MI-6 D16** — the largest improvements in the entire programme.

---

## 2. Per-Cell Breakdown

### 24h (V4 lgbm13 on FULL_V4)
| Cell | MI-4 | MI-6 D16 | Δ |
|------|------|----------|---|
| test_2023 | 0.8552 | 0.7956 | **+0.0596** |
| test_2024 | 0.9540 | 0.9177 | **+0.0364** |
| test_2025 | 0.7668 | 0.8048 | -0.0380 |
| test_2026 | 0.6789 | 0.7935 | -0.1146 |
| test_trained | 0.8996 | 0.8686 | **+0.0311** |
| test_new_network | 0.8868 | 0.8434 | **+0.0434** |
| **test_unseen** | **0.9590** | 0.8299 | **+0.1291** |

### 48h (D16 lgbm7 on FULL_V2)
| Cell | MI-4 | MI-6 D16 | Δ |
|------|------|----------|---|
| test_2023 | 0.7209 | 0.6814 | **+0.0395** |
| test_2024 | 0.9221 | 0.8573 | **+0.0648** |
| test_2025 | 0.7100 | 0.7598 | -0.0497 |
| test_2026 | 0.6307 | 0.7467 | -0.1160 |
| test_trained | 0.8376 | 0.8091 | **+0.0285** |
| test_new_network | 0.8241 | 0.7868 | **+0.0373** |
| **test_unseen** | **0.9070** | 0.7416 | **+0.1655** |

### 72h (D16 xgb3 on FULL_V2)
| Cell | MI-4 | MI-6 D16 | Δ |
|------|------|----------|---|
| test_2023 | 0.6560 | 0.6257 | **+0.0303** |
| test_2024 | 0.8810 | 0.8043 | **+0.0767** |
| test_2025 | 0.6734 | 0.7019 | -0.0285 |
| test_2026 | 0.6041 | 0.7023 | -0.0982 |
| test_trained | 0.7818 | 0.7614 | **+0.0204** |
| test_new_network | 0.7689 | 0.7129 | **+0.0560** |
| **test_unseen** | **0.8454** | 0.6410 | **+0.2043** |

---

## 3. Key Findings

1. **Unseen stations are the biggest winner** — MI-4 gains of **+0.13 to +0.20 PR-AUC** on `test_unseen` across all horizons. The WorldCover land cover + MI-8 response-lag features provide critical spatial generalization.

2. **New-network stations also improve** — +0.03 to +0.06 gains on stations added after the training period.

3. **Losses concentrated in 2025-2026** — The two most recent test years (low base rates, distribution shift) show losses vs D16. This is expected: the V4 24h model was optimized on val_2022 and may overfit to that regime.

4. **D7 persistence beaten on ALL 21 test points** — The hybrid model clears the mandatory baseline everywhere.

5. **Operational impact** — At 10:1 cost threshold:
   - 24h: Recall ~0.90, fewer missed floods
   - 48h: Recall ~0.85
   - 72h: Recall ~0.80

---

## 4. Decision

**ADOPT HYBRID MODEL AS NEW CHAMPION (Decision D17)**

| Horizon | Model | Feature Set | Config File | Test PR-AUC |
|---------|-------|-------------|-------------|-------------|
| **24h** | **lgbm13 (V4)** | **FULL_V4 (187)** | `mi4_hpo_v4_label_24h.json` | **0.8572** |
| **48h** | **lgbm7 (D16)** | **FULL_V2 (110)** | `mi_hpo_label_48h.json` | **0.7932** |
| **72h** | **xgb3 (D16)** | **FULL_V2 (110)** | `mi_hpo_label_72h.json` | **0.7444** |

**Frozen to:** `models/final_v3/` with `final_config_v3.json`  
**Test reads consumed:** 2 total (MI-6 for D16, MI-4 for hybrid)  
**Next test read:** Requires pre-registration with new decision.

---

## 5. Reproduction

```bash
# Refit hybrid models
.venv\Scripts\python.exe -u scripts\mi4_select_v4.py 24h   # V4 24h
# D16 48h/72h already frozen in models/final_v2/

# Test read (guarded)
.venv\Scripts\python.exe -u scripts\mi4_test_read.py
```

---

## 6. Implications for Production (Phases 35-42)

The hybrid model is **ready for system integration**:

- **Phase 35 (Location API):** Uses V4 24h model for 24h predictions, D16 for 48/72h
- **Phase 36 (Risk Map):** Can deploy with confidence — unseen station performance validated
- **Phase 37 (Auto Ingest):** Need WorldCover static features (one-time) + MI-8 features (daily)
- **Phase 38 (Pipeline):** Feature generation must include WorldCover + MI-8 blocks
- **Phase 39 (Warning Engine):** Thresholds already derived (see `mi4_test_thresholds.csv`)

**Model Card:** `models/final_v3/MODEL_CARD.md` (to be created in Phase 44)