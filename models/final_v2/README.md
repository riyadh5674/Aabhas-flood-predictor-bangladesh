# `models/final_v2/` — MI-5 selected model (decision D16)

| | |
| --- | --- |
| **Model** | `hpo_perhorizon_FULL_V2` |
| **Features** | FULL_V2 (110 cols = v1 62 + MI-v2) |
| **Per-horizon family** | {"24h": "lgbm", "48h": "lgbm", "72h": "xgb"} |
| **Seed** | 20261008 |
| **Selection** | validation + spatial-CV only (7 dev cells), rule D16 |
| **Frozen** | 2026-10-15 (MI-5) |

**Test period never read** (`test_rows_read = 0`).  This freeze is provisional: the
frozen test is opened exactly once, in MI-6.

```python
import sys, joblib
sys.path.insert(0, r"D:\Flood predictor\scripts")
s = joblib.load(r"D:\Flood predictor\models\final_v2\selected_model_v2.pkl")
pipe24 = s["pipelines"]["label_24h"]
p24 = pipe24.predict_proba(d[s["features"]])[:, 1]
```
