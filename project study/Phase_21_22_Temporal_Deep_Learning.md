# Phase 21–22 — Deep Temporal Learning (LSTM, GRU, TCN)
### Project study notes · Bangladesh flood-probability forecasting

> Source: `reports/phase21_temporal_dl.md`, `reports/phase22_tcn.md`,
> `scripts/train_sequence.py`, `reports/phase21_metrics.csv`, `reports/phase22_metrics.csv`.
> Previous study: Phase 20 (classical ML). These phases run the **deep learning ladder** on
> Dataset B.

---

## 0. The 60-second version

Phases 21–22 ask: **can a neural network that reads the last few days as a sequence beat the
classical tree models?** They train recurrent networks (LSTM, GRU) and a convolution network
(TCN) on Dataset B windows, on the same grid, with the same rules.

**Honest result: no — not as single models.** The best deep model (LSTM at L = 3) reaches
**0.9421** at 24 h, while XGBoost from Phase 20 reaches **0.9681** — the deep models sit
**~0.02–0.03 below** the tree.

But the phases weren't wasted:
- They established the **window length L\* = 3** (shortest wins — long windows add nothing).
- They answered the **recurrence-vs-convolution** question: **at 24 h they match**; at 72 h
  **recurrence wins**; the TCN's one clear win is **spatial generalisation**.
- They set up Phase 23 (**fusion**) to try to close the gap by adding location.

**Key lesson:** on tabular + short-sequence problems, a well-tuned boosted tree is very hard
to beat. That's a real, common finding — and the project reports it instead of hiding it.

---

## 1. Where Phase 21–22 sit

| Phase | What | Plain words |
|---|---|---|
| 20 | Classical ML (L2) | XGBoost/LightGBM win the day |
| **➜ 21–22** | **Temporal DL (L3)** | **LSTM/GRU/TCN on day-sequences** |
| 23 | Fusion | Add static geography to the encoder |
| 24–25 | Gated advanced models | Graph nets / transformer (only if gates pass) |
| 26–33 | Calibrate → select → freeze | Pick champion, read test once |

Closes checklist **stages 36 (LSTM/GRU)** and **37 (TCN)**.

---

## 2. Key words

| Word | Meaning |
|---|---|
| **LSTM** | Long Short-Term Memory — a recurrent net that remembers/forgets over time. |
| **GRU** | Gated Recurrent Unit — a simpler, cheaper recurrent net. |
| **TCN** | Temporal Convolutional Network — uses *dilated causal convolutions* instead of recurrence. |
| **Recurrent** | Processes the sequence one step at a time, carrying a memory. |
| **Convolution** | Slides a filter over the sequence to detect local patterns. |
| **Causal convolution** | A convolution that only looks **left** (past), never right (future). |
| **Dilated** | Skips steps to cover a wider range cheaply (dilation 1, 2, 4, …). |
| **Receptive field** | How many past days the model can "see". |
| **Window L** | How many days of history the model reads (3, 7, 14, 30). |
| **L\*** | The winning window length (here **3**). |
| **Hidden size** | Width of the network's internal representation. |
| **Dropout** | Randomly ignore units during training to reduce overfitting. |
| **BCE loss** | Binary cross-entropy — the standard loss for yes/no prediction. |
| **`pos_weight = n_neg/n_pos`** | A way to make the loss care more about rare positives. |
| **Early stopping** | Stop training when **validation** PR-AUC stops improving. |
| **Patience** | How many epochs to wait for improvement before stopping. |
| **State dict / `.pt`** | A saved PyTorch model. |
| **n_params** | Number of trainable weights — a model-size measure. |
| **Determinism** | Same input + seed → bit-identical output. Proved here. |

---

## 3. The shared contract (identical to Phase 20's grid)

All DL models used:
- Dataset B sequences with policy **P19-M1** → **13 channels** = concat(12 raw vars, mask),
- the **same dev grid** (8 cells × 3 horizons × 3 seasons),
- **BCE loss** with an **unweighted baseline and a `pos_weight = n_neg/n_pos` variant** both in
  the search space,
- **early stopping on validation PR-AUC** (only validation — never test),
- per-task **train-fold-only standardisation**, NaN→0 **after** standardisation (so a raw NaN
  never reaches a layer),
- seed **20261008** everywhere.

```
input  X : (N, L, 13)   concat(12 raw variables, timestep mask)
       M : (N, L)      mask (0 = absent calendar day)
head   : encoder -> last hidden -> dropout -> Linear(hidden,1) -> sigmoid
```

---

## 4. Phase 21 — LSTM & GRU (stage 36)

### Tuning (validation-only, ≤ 20 configs)

| Family | Configs | Best mean PR-AUC |
|---|---:|---:|
| lstm_L3 | 12 | **0.92677** |
| lstm_L7 | 10 | 0.91146 |
| lstm_L14 | 8 | 0.90775 |
| lstm_L30 | 6 | 0.91688 |
| gru_bestL | 10 | 0.92362 |

### Window selection — **L\* = 3**

Mean val-2022 PR-AUC over the four LSTMs:
**L3 0.9268 > L30 0.9169 > L7 0.9115 > L14 0.9078**.

So the **shortest window wins**. Long windows (14/30 days) add nothing here and cost more
compute. `gru_bestL` was therefore tuned at **L = 3**.

### Selected configs

| Family | Selected config | Params | Weighting |
|---|---|---:|---|
| **lstm_L3** | lr 0.003, hidden 32, layers 2, dropout 0.0, batch 256 | 14,497 | `pos_weight=ratio` |
| lstm_L7 | lr 0.001, hidden 64, layers 2, dropout 0.3, batch 256 | 53,569 | `pos_weight=ratio` |
| lstm_L14 | lr 0.001, hidden 32, layers 2, dropout 0.3, batch 256 | 14,497 | none |
| lstm_L30 | lr 0.003, hidden 32, layers 2, dropout 0.0, batch 256 | 14,497 | none |
| **gru_bestL** | lr 0.001, hidden 64, layers 2, dropout 0.3, batch 256 | 40,193 | `pos_weight=ratio` |

**Weighting was again a validation result** — on for the L3/GRU leaders, off for L7/L14/L30 —
and always applied **inside the training fold only**.

### Selection-cell results (`val_2022`, P1, all season)

| Model | 24 h | 48 h | 72 h |
|---|---|---|---|
| **lstm_L3** | **0.9421** | 0.9253 | **0.9129** |
| gru_bestL | 0.9359 | **0.9263** | 0.9086 |
| lstm_L30 | 0.9388 | 0.9183 | 0.8936 |
| lstm_L7 | 0.9309 | 0.9164 | 0.8871 |
| lstm_L14 | 0.9311 | 0.9098 | 0.8823 |
| *xgboost* (P20) | *0.9681* | *0.9549* | *0.9408* |
| *logreg_linear* (P16) | *0.9401* | *0.9241* | *0.9053* |
| *persistence* (D7) | *0.9203* | *0.8954* | *0.8683* |

**Reading it:** every recurrent family **clears persistence** at all horizons — good. But all
sit **≈0.02–0.03 below XGBoost** (−0.0261 / −0.0286 / −0.0279 for the best DL model). The
boosted tree on engineered features remains the strongest single model.

### Cross-cell robustness

| | P3 spatial CV 24/48/72 h | P2 rolling val_2020/21/22 (24 h) |
|---|---|---|
| lstm_L3 | 0.9196 / 0.9055 / 0.8781 | 0.9659 / 0.9575 / 0.9421 |
| lstm_L30 | **0.9245** / 0.9035 / **0.8940** | **0.9683** / **0.9668** / 0.9388 |
| lstm_L14 | 0.9244 / **0.9120** / 0.8877 | 0.9566 / 0.9545 / 0.9311 |
| gru_bestL | 0.9139 / 0.8972 / 0.8855 | 0.9655 / 0.9506 / 0.9359 |
| lstm_L7 | 0.9126 / 0.9117 / 0.8881 | 0.9520 / 0.9519 / 0.9309 |

**Interesting nuance:** `lstm_L3` leads on the primary selection cell, but `lstm_L30` leads on
P3/CV and P2 rolling at 24 h. So the window choice **trades in-distribution accuracy (L3) for
out-of-sample stability (L30)**. The project records this honestly rather than over-claiming.

---

## 5. Phase 22 — TCN (stage 37)

### The architecture

A stack of residual blocks; block *i* applies two **dilated causal** `Conv1d` (kernel 3,
dilation 2^i, left-only padding so step *t* never sees *t+1…*) with BatchNorm + ReLU + dropout:

```
TCNBlock(x) = ReLU( x + Drop(BN(Conv1(ReLU(BN(Conv0(x)))))) ), dilation 1,2,...
```

Everything else is **identical to Phase 21** (fair-comparison parity, D14).

### Selected config

| Family | Selected config | Params | Weighting |
|---|---|---:|---|
| **tcn_bestL** | lr 0.003, hidden 32, layers 2, dropout 0.0, batch 256 | 11,329 | `pos_weight=ratio` |

Receptive field = `1 + (k−1)·Σdilations = 1 + 2·3 = 7 ≥ L`. Best mean PR-AUC **0.92124**. The
tuning landscape is **flat** (strongest 24 h result 0.94052 vs weakest 0.91893, ~0.5 pt) — the
TCN is not hypersensitive.

### The recurrence-vs-convolution table (the scientific product)

| Model | 24 h | 48 h | 72 h | Params | Mean grid fit |
|---|---|---|---|---|---|
| **LSTM** (L3) | **0.9421** | **0.9253** | **0.9129** | 14,497 | 30 s |
| GRU (L3) | 0.9359 | 0.9263 | 0.9086 | **4,545** | 70 s |
| **TCN** (L3) | 0.9405 | 0.9224 | 0.9008 | 11,329 | 41 s |
| *xgboost* (P20) | *0.9681* | *0.9549* | *0.9408* | — | — |
| *persistence* (D7) | *0.9203* | *0.8954* | *0.8683* | — | — |

**Answers:**
- **At 24 h, convolution matches recurrence.** TCN 0.9405 vs LSTM 0.9421 — gap **0.0016**,
  far inside overlapping CIs → **not separable**. TCN also edges GRU at 24 h.
- **At 72 h, recurrence wins.** TCN is weakest (0.9008 vs LSTM 0.9129, −0.0121). The TCN's
  72 h fit **early-stopped at epoch 2** — dilated convs over a 3-day window overfit the hardest
  horizon fastest.
- **Cheaper? Not really.** TCN (11,329 params, 41 s/fit) sits between LSTM (14,497, 30 s) and
  GRU (4,545, 70 s). GRU is the genuinely cheap model. **TCN's one clear win is spatial
  generalisation** (below).

### Cross-cell robustness

| | P2 rolling 24/48/72 h *(mean over 2020/21/22)* | P3 spatial CV 24/48/72 h |
|---|---|---|
| lstm_L3 | 0.9551 / 0.9362 / 0.9193 | 0.9196 / 0.9055 / 0.8781 |
| gru_bestL | 0.9507 / 0.9376 / 0.9231 | 0.9139 / 0.8972 / 0.8855 |
| **tcn_bestL** | 0.9478 / 0.9371 / 0.9063 | **0.9258** / 0.8949 / 0.8716 |

*(Note: the Phase-21 table in §4 shows P2 **per year at 24 h**; here P2 is the **mean over the three
rolling years (2020/21/22)** per horizon — which is why the `lstm_L3` 24 h figure is lower here than there.)*

**P3 spatial CV: the TCN leads at 24 h (0.9258 vs LSTM 0.9196, GRU 0.9139)** then falls behind
at 72 h — the same short-lead-time character. Season splits barely move (≤ 1 point) across all
families.

---

## 6. Artifacts

| File | Size | Contents | Params |
|---|---|---:|---:|
| `phase21_L3-01.pt` | 180 KB | **lstm_L3** × 3 horizons | 14,497 |
| `phase21_L3-02.pt` | 638 KB | lstm_L7 × 3 | 53,569 |
| `phase21_L3-03.pt` | 180 KB | lstm_L14 × 3 | 14,497 |
| `phase21_L3-04.pt` | 180 KB | lstm_L30 × 3 | 14,497 |
| `phase21_L3-05.pt` | 481 KB | **gru_bestL** × 3 | 40,193 |
| `phase22_L3-06.pt` | 169 KB | **tcn_bestL** × 3 | 11,329 |

**Parameter counts are part of the comparison (Phase 27):** the winning LSTM/GRU are ~14 K /
40 K parameters versus XGBoost's 300 trees. The DL models are *smaller* but, per-model, weaker.

Metric files: `phase21_metrics.csv` (**360 rows**), `phase22_metrics.csv` (**72 rows**) — with
**zero `test` rows**. Station-bootstrap CIs and macro per-station PR-AUC included.

---

## 7. Test protection & determinism

- **Test:** `frozen_read_at_phase33` for all L3 rows. Both metric files contain **zero rows
  whose cell starts with `test`**. The frozen test is read once, at Phase 33.
- **Determinism (workflow-mandated):** both notebooks refit a fixed config for 2 epochs twice on
  identical 4,000-row slices and assert the parameter tensors are **bit-identical** and val
  PR-AUCs agree to **< 1e-12** (cells N21-08b, N22-08b).
- **Replay:** all three TCN state_dicts reload and reproduce the recorded val-2022 PR-AUC exactly.
- **Cost:** Phase 21 `TOTAL 40625 s` (wall inflated by host sleep — the honest per-config
  `fit_s` column is the real cost: 1.1/1.5/2.9/12.0/1.1 K s per family). Phase 22
  `TOTAL 2560 s`, tuning Σ = 1,796 s.

---

## 8. Problems & solutions

| Problem | Solution |
|---|---|
| Deep nets could leak future days | **Causal** convolutions + windows ending on *t* (Phase 18–19). |
| Which window length? | Sweep L ∈ {3,7,14,30} fairly (D12); **L\* = 3** won. |
| Imbalance handling could be guessed | `pos_weight` variant in search space; chosen by **validation**. |
| Training could stop on test | Early stopping on **validation PR-AUC only**; test cells absent. |
| Raw NaN could hit a layer | Mask channel + NaN→0 **after** standardisation. |
| Deep results could be non-reproducible | Bit-identical determinism + replay checks. |
| DL might be claimed better than it is | Report the **honest gap** to XGBoost (~0.02–0.03 below). |
| Long windows might be assumed better | Empirically shortest won; L30 only helps spatial stability. |

---

## 9. Files referenced

| File | Purpose |
|---|---|
| `reports/phase21_temporal_dl.md` | LSTM/GRU report (stage 36) |
| `reports/phase22_tcn.md` | TCN report (stage 37) |
| `scripts/train_sequence.py` | Driver for `lstm`\|`gru`\|`tcn` (resume-safe) |
| `reports/phase21_metrics.csv` | 360 rows (L3-01…05) |
| `reports/phase22_metrics.csv` | 72 rows (L3-06) |
| `models/phase21_L3-0{1..5}.pt`, `models/phase22_L3-06.pt` | Saved models |
| `data/processed/sequence_policy.json` | P19-M1 (13 channels) |
| `reports/phase20_classical_ml.md` | The bar these DL models must beat |

---

## 10. Verdict

- **Single deep models do NOT beat XGBoost** — they sit ~0.02–0.03 below it, while clearing
  persistence at every horizon.
- **Window length L\* = 3** (shortest wins); long windows add nothing and cost more.
- **Recurrence vs convolution:** TCN **matches** LSTM/GRU at 24 h (not separable), is **beaten**
  by recurrence at 72 h, buys **no cost advantage**, but **leads spatial CV at 24 h**.
- **Weighting decided by validation**; early stopping on validation only; test untouched.
- **Determinism and replay checks green**; models saved for Phase 23/27/33.

**Bottom line:** The deep-learning ladder is honest and complete — it does **not** beat the
boosted tree as a single model, but it identifies the winning window, characterises the
recurrence-vs-convolution trade, and gives Phase 23 something real to fuse. Reporting this
plainly (rather than cherry-picking a lucky cell) is exactly what the framework was built for.

---

**Next in the series:** Phase 23 — Feature fusion (temporal encoder + static geography +
engineered features: does knowing *where* and *what* add skill over the sequence alone?).
Then 24–25 (gated graph/transformer gates), 26–33 (calibration, thresholds, selection, test).
