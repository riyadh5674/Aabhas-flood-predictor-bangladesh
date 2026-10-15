# `models/final/` — Phase 33 selected model

| | |
| --- | --- |
| **Model** | `xgboost` (L2-03, config `xgboost_c01`) |
| **Family** | xgboost |
| **Feature set** | FULL (62 features) |
| **Training rows** | P1 `train_<2022` (split train & not holdout, <= 2021); 61,977 rows @ 24 h |
| **Horizons** | 24 h / 48 h / 72 h (one fitted pipeline per horizon) |
| **Seed** | 20261008 |
| **Dataset SHA-256** | `a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e` |
| **Frozen** | 2026-10-15 (Phase 33 / stage 57) |

## Load

```python
import sys, joblib
sys.path.insert(0, r"D:\Flood predictor\scripts")   # FreqEncoder lives in baseline_model
s = joblib.load(r"D:\Flood predictor\models\final\selected_model.pkl")
pipe = s["horizons"]["label_24h"]        # -> sklearn Pipeline (.predict_proba)
X = d[s["features"]]
p24 = pipe.predict_proba(X)[:, 1]
```

The selected model is the Phase-20 P1-trained artefact (reproducible from
`reports/phase20_search.csv` config `xgboost_c01`); nothing was re-tuned in
Phase 33. The frozen test period was read once for this model **and** the D7 persistence
bar — see `reports/phase33_selection.md` and `reports/phase33_test_read_log.json`.
