import pandas as pd
import numpy as np

score = pd.read_csv("reports/domain_adaptation/da_scorecard.csv")
boot = pd.read_csv("reports/domain_adaptation/da_bootstrap.csv")

print("=========================================================================")
print("                   EXPERIMENT 1: DOMAIN ADAPTATION SCORECARD             ")
print("=========================================================================")

methods = ["erm_d16", "erm_lgbm", "groupdro_lgbm", "erm_mlp", "groupdro_mlp", "irm_mlp", "dann_mlp"]
horizons = ["label_24h", "label_48h", "label_72h"]

rows = []
for m in methods:
    gm = score[(score.method == m) & (score.scheme == "basin")]
    if len(gm) < 3:
        continue
    h_v22 = gm.set_index("horizon")["val_2022"]
    h_devm = gm.set_index("horizon")["dev_mean"]
    h_devw = gm.set_index("horizon")["dev_worst"]
    rows.append({
        "method": m,
        "val_2022_24h": h_v22.get("label_24h", np.nan),
        "val_2022_48h": h_v22.get("label_48h", np.nan),
        "val_2022_72h": h_v22.get("label_72h", np.nan),
        "val_2022_mean": h_v22.mean(),
        "dev_mean": h_devm.mean(),
        "dev_worst_mean": h_devw.mean(),
    })

res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\n=========================================================================")
print("                 PAIRED BOOTSTRAP VS CONTROLS (val_2022)                ")
print("=========================================================================")
val_boot = boot[boot.cell == "val_2022"].copy()
for horizon in horizons:
    print(f"\n--- {horizon} ---")
    hb = val_boot[val_boot.horizon == horizon][["method", "control", "mean", "lo", "hi", "p_gt_0"]]
    print(hb.to_string(index=False, float_format=lambda x: f"{x:+.6f}"))
