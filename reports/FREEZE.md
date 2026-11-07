# Frozen results — paper evidence lock

**Frozen at:** 2026-11-07T14:06:19 · **Seed:** `20261008` · **Manifest SHA-256:** `67e88d1be7eeaae7afd0816c…`

**Verification:** ALL CHECKS PASS (13/13)

This file freezes the numbers the IEEE paper may quote. A later change to any
hashed artefact invalidates the manifest and must be documented as a new version.

## Revision history

| version | date | change |
| ---: | --- | --- |
| 1 | 2026-10-22 | Initial lock: D16 frozen model, MI-6 test read, correction-sprint numbers. |
| 2 | 2026-11-06 | Documentation-only refresh. `reports/test_read_ledger.md` and `reports/mi6_test_read.md` were re-worded after the original lock; all 13 numeric/evidence checks still pass. No dataset, model, prediction, or headline number changed and the frozen test was not read. |
| 3 | 2026-11-07 | Mark the two multi-hundred-MB MI feature caches as optional, regenerable artefacts. They were never shipped in git and are not inputs to the frozen D16 result. All core D16 evidence files remain strictly hash-checked; no number, model, or dataset changed. |

## Headline numbers

| quantity | value |
| --- | ---: |
| D16 `val_2022` 24 / 48 / 72 h | 0.970789 / 0.956485 / 0.945007 |
| D16 `val_2022` horizon mean | **0.957427** |
| D16 frozen test 24 / 48 / 72 h | 0.836227 / 0.768959 / 0.707087 |
| D16 frozen test horizon mean | **0.770758** |
| D7 frozen test horizon mean | 0.667549 |
| best classical / best deep, `val_2022` | 0.9546 / 0.9356 |

## Checks

| check | result | detail |
| --- | --- | --- |
| MI-10 summary not void | PASS | None |
| MI-10 gates | PASS | 18/18 pass |
| D16 val_2022 label_24h | PASS | 0.970789 vs 0.970789 |
| D16 val_2022 label_48h | PASS | 0.956485 vs 0.956485 |
| D16 val_2022 label_72h | PASS | 0.945007 vs 0.945007 |
| D16 val_2022 horizon-mean | PASS | 0.957427 |
| D16 test per-horizon | PASS | 0.836227 / 0.768959 / 0.707087 |
| D16 test horizon-mean | PASS | 0.770758 |
| D7 test horizon-mean | PASS | 0.667549 |
| event analysis is dev-only | PASS | 0 |
| event analysis has all cells | PASS | 7 |
| dataset row count | PASS | 200665 |
| all frozen files exist | PASS | 24/24 |

## Artefact hashes

| file | bytes | SHA-256 |
| --- | ---: | --- |
| `data/processed/experimental_dataset.csv` | 125,682,549 | `a6748529083fcdceb8bb8a657b067bf12b5c861f3a7a6b09c8977ea63be73c8e` |
| `models/final_v2/selected_model_v2.pkl` | 5,496,212 | `0b1cd34074d860b6797889410728905003909902bc1613a7da512a77291a0d79` |
| `models/final_v2/final_config_v2.json` | 3,721 | `091f08bbad0c3bb4d2a160eb1618fb6c4a97a58933afb854108a3b64215f1d37` |
| `reports/p2p3_rescore/summary.json` | 7,921 | `b0ab1f97723da72d795054f2de271dd79a5b1b70f70c754fbe705ba97e710300` |
| `reports/p2p3_rescore/probe_p2p3.csv` | 24,727 | `3326ccceb63b364fd523a4a99a6fa762bea4cd38771534e14060dc0486c9e53e` |
| `reports/p2p3_rescore/arms_p2p3.csv` | 23,575 | `1e1c8c8cd5ec891d4f933f2f8bcab3cbe7e3ce7b0dbb8f910bc1548e09bda5bb` |
| `reports/p2p3_rescore/d7_p2p3.csv` | 8,246 | `518215b84168d5bcbc80bbd8d5a70a20ad684df5db96e0f324f9cdedc2d69493` |
| `reports/p2p3_rescore/d7_margins_p2p3.csv` | 5,911 | `d3f1b77e5a06f59cb3249787217a5c3fddd15e95bf0e208207c66fdbf021b901` |
| `reports/p2p3_correction.md` | 11,654 | `94f71f43e52babcd84541e09a1dc0166756e0256e695e2a910cfee5fbe86dff3` |
| `reports/event_level/summary.json` | 1,816 | `1db43aaf293ab3582650dc1efbbe873235bd20c068a34c4018712f17181081b3` |
| `reports/event_level/event_metrics.csv` | 6,060 | `77d89891a7b0630489f335f1f45ac99c20f82712a24e922a2113f9b1dd8f3eb9` |
| `reports/event_level/summary.md` | 3,506 | `3379cf5c932846a23652a6bbaf1e328ac41646d4541741b430a27c9708342bdd` |
| `reports/mi6_summary.json` | 2,361 | `ece2cfe9803e2a3f14247324141a443dff78e8d67b8b96b32726e87f6fd3e158` |
| `reports/mi9_summary.json` | 1,116 | `60203f2f9c90302c0e74c5eca66e709dc49e4e40499aa50d8248124ef21a6d52` |
| `reports/phase27_summary.json` | 7,391 | `a9555d7d28803bb452346a4271bca1b524ca4227827b729f5e864704e5ed252d` |
| `reports/phase31_summary.json` | 4,374 | `2122d3f9ceb99c83f7a25d20c8f0538051031612851e9c334f9e9226f14fc05e` |
| `reports/phase28_summary.json` | 1,782 | `f86d8379915c233062405f76c21e7ddd136f1686fec028a472b7b0a1659008d0` |
| `reports/phase34_summary.json` | 12,971 | `c0b2a4635433185b9ee2ee0cea049fd82051a6cd568fc81b67205a79b2d02c86` |
| `reports/test_read_ledger.md` | 5,865 | `84cdbcd28b3ade24a857911f931d467e06c8f492d8c48c36df9089991d264109` |
| `reports/phase27_ml_vs_dl.md` | 20,394 | `4a86be65cebe95222c43d54dc80ed5822c91496ecc170d5d873a5fb5316d700a` |
| `reports/phase31_robustness.md` | 14,117 | `693430167cbfad749c0591470e3214ed25d02e4aceeab5b3df72a3070af3a092` |
| `reports/phase28_calibration.md` | 12,641 | `99f8b1b0573221b6e24ad08d3aa0648aa65100bac4cc8daeb9a69a2b6339b442` |
| `reports/phase34_xai.md` | 17,511 | `296215e0afc8e385d930d380aca13178d863cb762ae861022d7f506d1706a8f4` |
| `reports/mi6_test_read.md` | 6,240 | `e6e4e5b2b5ba20ac0837eb21b86f028aa558c71a0ab87941ca53bb885ef799d5` |
| `data/processed/mi_v2_features.csv` *(optional)* | — | not shipped (regenerable) |
| `data/processed/mi8_features.csv` *(optional)* | — | not shipped (regenerable) |

## Environment

- Python 3.13.7 on Windows-11-10.0.26300-SP0
- Git: `312fb3818b3f` (dirty)
- numpy 2.3.2
- pandas 3.0.6
- scikit-learn 1.9.1
- scipy 1.18.1
- xgboost 3.4.1
- lightgbm 4.7.0
- torch 2.14.0
- matplotlib 3.10.6
- shap not installed

## Test policy at freeze

closed - no read in this freeze; D16 headline is the valid MI-6 read
See `reports/test_read_ledger.md` for the full six-open history.

## Verify

```powershell
.venv\Scripts\python.exe scripts\freeze_results.py --check
```
