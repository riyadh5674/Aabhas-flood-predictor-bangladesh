# `models/final_v2/` — MI-5 selected model (decision D16)

| | |
| --- | --- |
| **Model** | `hpo_perhorizon_FULL_V2` |
| **Features** | FULL_V2 (110 cols = v1 62 + MI-v2) |
| **Per-horizon family** | {"24h": "lgbm", "48h": "lgbm", "72h": "xgb"} |
| **Seed** | 20261008 |
| **Selection** | validation + spatial-CV only (7 dev cells), rule D16 |
| **Frozen** | 2026-10-15 (MI-5) |

**Test governance note (updated).** This README's original freeze text is historical. The
programme-level frozen-test history is recorded in `reports/test_read_ledger.md`: six
exposures/two full model reads; MI-6 is the valid reported headline. `test_rows_read = 0` in
the config reflects this file's local/script state and must not be read as a programme-level
claim. D16 remains the governed model.

```python
import sys, joblib
sys.path.insert(0, r"D:\Flood predictor\scripts")
s = joblib.load(r"D:\Flood predictor\models\final_v2\selected_model_v2.pkl")
pipe24 = s["pipelines"]["label_24h"]
p24 = pipe24.predict_proba(d[s["features"]])[:, 1]
```
