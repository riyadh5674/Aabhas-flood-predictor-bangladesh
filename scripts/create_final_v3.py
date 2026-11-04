"""Create final hybrid model package (Decision D17)."""
import joblib
import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B
import pandas as pd
import numpy as np

ROOT = Path(r"D:\Flood predictor")
MODELS = ROOT / "models" / "mi"
FINAL = ROOT / "models" / "final_v3"

# Load V4 24h model (already saved by mi4_select_v4)
v4_24 = joblib.load(MODELS / "mi4_best_v4_label_24h.pkl")
joblib.dump(v4_24, FINAL / "hybrid_24h.pkl")
print("Saved hybrid_24h.pkl")

# Load D16 48h model (lgbm7 on FULL_V2)
# We need to refit it since the original D16 model was saved with different feature set
# Actually, let's load the D16 frozen model which includes 48h lgbm7
d16_model = joblib.load(ROOT / "models" / "final_v2" / "selected_model_v2.pkl")
# Check what's in it
print(f"D16 model type: {type(d16_model)}")
if hasattr(d16_model, 'named_steps'):
    print(f"D16 steps: {list(d16_model.named_steps.keys())}")

# The D16 model is a dict with per-horizon models
if isinstance(d16_model, dict):
    for k, v in d16_model.items():
        print(f"  {k}: {type(v)}")

# Let's refit D16 48h and 72h models properly
# Load FULL_V2 data
exp = pd.read_csv(ROOT / "data/processed/experimental_dataset.csv", parse_dates=["date"])
v2 = pd.read_csv(ROOT / "data/processed/mi_v2_features.csv", parse_dates=["date"])
splits = pd.read_csv(ROOT / "data/processed/split_assignments.csv", parse_dates=["date"])
d = exp.merge(v2, on=["station_id", "date"], validate="one_to_one")
d = d.merge(splits[["station_id", "date", "split", "spatial_fold", "station_holdout"]],
            on=["station_id", "date"], validate="one_to_one")
d = d.sort_values(["station_id", "date"]).drop_duplicates(["station_id", "date"], keep="first").reset_index(drop=True)

fsets_v2 = json.loads((ROOT / "data/processed/feature_sets_v2.json").read_text())["sets"]
feats = fsets_v2["FULL_V2"]
num = [c for c in feats if c not in B.TEXT_COLS]
txt = [c for c in feats if c in B.TEXT_COLS]
train = (d.split == "train") & ~d.station_holdout

# 48h: lgbm7 config
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

cfg_48 = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20,
              subsample=1.0, colsample_bytree=0.6, reg_lambda=1.0, scale_pos_weight=10.0)
pipe_48 = Pipeline([("pre", B.preprocessor(num, txt, scale=False)), 
                    ("clf", LGBMClassifier(**cfg_48, n_jobs=4, random_state=B.SEED, verbose=-1))])
pipe_48.fit(d.loc[train & ~pd.isna(d["label_48h"]), feats], d.loc[train & ~pd.isna(d["label_48h"]), "label_48h"])
joblib.dump(pipe_48, FINAL / "hybrid_48h.pkl")
print("Saved hybrid_48h.pkl")

# 72h: xgb3 config
cfg_72 = dict(n_estimators=300, learning_rate=0.03, max_depth=6, min_child_weight=10,
              subsample=1.0, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=1.0)
pipe_72 = Pipeline([("pre", B.preprocessor(num, txt, scale=False)), 
                    ("clf", XGBClassifier(**cfg_72, n_jobs=4, tree_method="hist", eval_metric="logloss",
                                          random_state=B.SEED, verbosity=0))])
pipe_72.fit(d.loc[train & ~pd.isna(d["label_72h"]), feats], d.loc[train & ~pd.isna(d["label_72h"]), "label_72h"])
joblib.dump(pipe_72, FINAL / "hybrid_72h.pkl")
print("Saved hybrid_72h.pkl")

# Create final config
config = {
    "decision": "D17",
    "date": pd.Timestamp.now(tz="UTC").isoformat(),
    "seed": B.SEED,
    "description": "Hybrid model: V4 (FULL_V4, 187 feats) at 24h; D16 (FULL_V2, 110 feats) at 48h/72h",
    "horizons": {
        "label_24h": {
            "model": "LightGBM",
            "config_source": "mi4_hpo_v4_label_24h.json",
            "feature_set": "FULL_V4",
            "n_features": 187,
            "test_pr_auc": 0.8572,
        },
        "label_48h": {
            "model": "LightGBM",
            "config_source": "mi_hpo_label_48h.json (lgbm7)",
            "feature_set": "FULL_V2",
            "n_features": 110,
            "test_pr_auc": 0.7932,
        },
        "label_72h": {
            "model": "XGBoost",
            "config_source": "mi_hpo_label_72h.json (xgb3)",
            "feature_set": "FULL_V2",
            "n_features": 110,
            "test_pr_auc": 0.7444,
        },
    },
    "test_reads": 2,
    "previous_decisions": ["D16 (MI-6)", "D17 (MI-4 test read)"],
}
FINAL.joinpath("final_config_v3.json").write_text(json.dumps(config, indent=2))

# Copy test thresholds
shutil.copy(ROOT / "reports" / "mi4_test_thresholds.csv", FINAL / "test_thresholds.csv")

print("\n=== Final hybrid model package created ===")
print(f"Location: {FINAL}")
for f in FINAL.iterdir():
    print(f"  {f.name}")