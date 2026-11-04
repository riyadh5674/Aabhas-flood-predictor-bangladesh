r"""MI-8 select — pick the per-horizon winners, verify them against D16, and say plainly
whether any gain is CI-separable.

What this does
--------------
1. Reads the finished HPO ledgers (`mi8_hpo_<arm>.csv`) and picks the per-horizon winner on
   `val_2022` (season=all) — the pre-registered D16 selection rule.
2. **Refits** the winners and scores them on all seven dev cells (full metric set), alongside:
   * the **D16 reference** — `lgbm7` @24/48h + `xgb3` @72h on `FULL_V2`, reconstructed from
     `reports/mi3_hpo.csv`, and
   * the **`full_v2` equal-budget control** winner, when that arm has finished.
3. Runs a **paired station-cluster bootstrap** on `val_2022`: delta = PR-AUC(winner) −
   PR-AUC(D16), 500 resamples, stations resampled not rows (the project's D8 rule, same
   resample index set for both models so the pairing is exact).

Why the refit is not redundant
-----------------------------
The HPO ledger already contains a `val_2022` number per config, but a *paired* bootstrap needs
row-level probabilities from both models, and the ledger stores none. The refit is also the
**honesty check**: a D16 refit on `FULL_V2` must reproduce the archived MI-3b number
(lgbm7 0.970789 / 0.956485, xgb3 0.945007) to 1e-6. If it does not, the whole comparison is
void and this script fails rather than reporting a number.

Test policy
-----------
Development cells only. `S.dev_masks()` contains no test cell and there is no code path here
that scores `>= 2023`. `test_rows_read` is asserted to be 0 and written into the summary.

Run
---
    .venv\Scripts\python.exe -u scripts\mi8_select.py --arm full_v3
    .venv\Scripts\python.exe -u scripts\mi8_select.py --arm full_v3 --reps 1000
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B                                    # noqa: E402
import mi3_search as S                                         # noqa: E402
import mi8_search as M8                                        # noqa: E402

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models" / "mi"
HOR = B.HORIZONS
SEED = B.SEED
DEV = S.DEV
CFG_COLS = S.CFG_COLS
TEST_ROWS_READ = 0

# Hyper-parameters that must reach the estimator as a Python `int`.  The HPO/ledger CSV
# widens every config column to float64 (the family union is NaN-padded), so a config read
# back from disk carries `n_estimators=800.0` and XGBoost aborts inside `train()` with
# "'float' object cannot be interpreted as an integer".  The nine names below are sampled
# from integer lists in XGB_SPACE / LGBM_SPACE / HGB_SPACE / ET_SPACE (verified in
# mi3_search.py).  `max_features` is deliberately excluded: ET_SPACE samples the *string*
# "sqrt" as well as the float 0.5, and `max_depth` may legitimately be `None` (ET), which
# arrives as NaN and is dropped rather than coerced.
INT_PARAMS = {"n_estimators", "max_depth", "min_child_weight", "num_leaves",
              "min_child_samples", "max_iter", "max_leaf_nodes",
              "min_samples_leaf", "min_samples_split"}

# The frozen D16 champion, per horizon (tag in reports/mi3_hpo.csv, arm = FULL_V2).
D16 = {"label_24h": "lgbm7", "label_48h": "lgbm7", "label_72h": "xgb3"}
D16_EXPECTED = {"label_24h": 0.970789, "label_48h": 0.956485, "label_72h": 0.945007}
D16_MEAN_EXPECTED = 0.957427

RESULTS: list[tuple[str, bool, str]] = []


def check(cid: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((cid, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {detail}")
    return bool(ok)


def _native(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ------------------------------------------------------------------ bootstrap
def paired_station_bootstrap(stations, y, p_new, p_ref, reps=500):
    """delta = PR-AUC(new) - PR-AUC(ref), resampling STATIONS with replacement.

    Both models are scored on the identical resampled row index, so the delta is a paired
    contrast rather than two independent CIs subtracted after the fact.
    """
    uniq = np.unique(stations)
    if len(uniq) < 10:
        return None
    idx_by_st = {s: np.flatnonzero(stations == s) for s in uniq}
    rng = np.random.default_rng(SEED)
    d_new, d_ref, deltas = [], [], []
    for _ in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_st[s] for s in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        a = average_precision_score(y[idx], p_new[idx])
        b = average_precision_score(y[idx], p_ref[idx])
        d_new.append(a)
        d_ref.append(b)
        deltas.append(a - b)
    if len(deltas) < reps // 2:
        return None
    d = np.asarray(deltas)
    return {"d_new": float(np.mean(d_new)), "d_ref": float(np.mean(d_ref)),
            "delta": float(np.mean(d)), "sd": float(np.std(d, ddof=1)),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "pgt": float((d > 0).mean()), "n_reps": int(len(d))}


# ---------------------------------------------------------------------- data
def load_featherless(arm: str):
    """(frame, feats) for an arm, plus the FULL_V2 frame shared with D16."""
    d3, fsets3 = B.load()
    v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
    d3 = d3.merge(v2, on=["station_id", "date"], validate="one_to_one")
    feats_v2 = json.loads((PROC / "feature_sets_v2.json").read_text(encoding="utf-8"))["sets"]["FULL_V2"]
    if arm == "full_v3":
        m8 = pd.read_csv(PROC / "mi8_features.csv", parse_dates=["date"])
        d3 = d3.merge(m8, on=["station_id", "date"], validate="one_to_one")
        feats = json.loads((PROC / "feature_sets_mi8.json").read_text(encoding="utf-8"))["sets"]["FULL_V3"]
    else:
        feats = feats_v2
    check(f"L_{arm}", len(d3) == 200665, f"rows={len(d3):,} feats={len(feats)}")
    return d3, feats, feats_v2


def cfg_from_row(row) -> dict:
    """Drop every column the family does not use (NaN), keep the rest as plain Python.

    Integer-typed parameters (see INT_PARAMS) are cast back to `int`; a non-integral value
    there is a corruption, not something to round silently, so it raises instead.
    """
    out = {}
    for k in CFG_COLS:
        if k not in row.index:
            continue
        v = row[k]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if k in INT_PARAMS:
            fv = float(v)
            if not fv.is_integer():
                raise ValueError(f"non-integral value for integer param {k!r}: {v!r}")
            out[k] = int(fv)
        else:
            out[k] = _native(v)
    return out


# ---------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["full_v3", "full_v2"], default="full_v3")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--skip-v2-control", action="store_true")
    ap.add_argument("--horizons", nargs="*", default=None,
                    help="restrict to a subset of label_24h / label_48h / label_72h")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve winners + D16 configs and stop before any fit")
    a = ap.parse_args(argv)
    arm = a.arm
    run_hors = [h for h in HOR if not a.horizons or h in a.horizons]
    if not run_hors:
        print(f"FATAL: --horizons {a.horizons} matched none of {HOR}")
        return 2

    t0 = time.time()
    print(f"=== MI-8 select  arm={arm}  bootstrap reps={a.reps}  seed={SEED} ===")
    print(f"horizons: {run_hors}")
    print(f"selection cell: val_2022 (season=all), pre-registered D16 rule")
    print(f"test_rows_read = {TEST_ROWS_READ} (dev cells only; no >=2023 row is scored)")
    print()

    # ---- HPO ledgers -------------------------------------------------------
    hpo_v3 = M8._read(REPORTS / "mi8_hpo_full_v3.csv")
    hpo_v2 = M8._read(REPORTS / "mi8_hpo_full_v2.csv")
    if hpo_v3.empty:
        print("FATAL: mi8_hpo_full_v3.csv missing — run the search first")
        return 2

    winners = {}
    for hz in run_hors:
        g = hpo_v3[hpo_v3.horizon == hz]
        if g.empty:
            print(f"FATAL: no {hz} rows in the full_v3 ledger")
            return 2
        b = g.sort_values("val_2022", ascending=False).iloc[0]
        winners[hz] = {"family": b.family, "tag": b.tag, "cfg": cfg_from_row(b),
                       "val_2022_ledger": float(b.val_2022)}

    ctrl = {}
    if not hpo_v2.empty and not a.skip_v2_control:
        for hz in run_hors:
            g = hpo_v2[hpo_v2.horizon == hz]
            if g.empty:
                continue
            b = g.sort_values("val_2022", ascending=False).iloc[0]
            ctrl[hz] = {"family": b.family, "tag": b.tag, "cfg": cfg_from_row(b),
                        "val_2022_ledger": float(b.val_2022)}

    # ---- data --------------------------------------------------------------
    d, feats, feats_v2 = load_featherless(arm)
    masks = S.dev_masks(d)
    check("M1", set(masks) == {"val_2020", "val_2021", "val_2022",
                               "spatial_fold_1", "spatial_fold_2",
                               "spatial_fold_3", "spatial_fold_4"},
          f"masks={sorted(masks)}")
    check("M2", not any("test" in k for k in masks), "no test cell in the scoring grid")
    dev_union = np.zeros(len(d), dtype=bool)
    for m in masks.values():
        dev_union |= m
    check("M3", not (dev_union & (d.split.to_numpy() == "test")).any(),
          f"dev union {int(dev_union.sum()):,} rows excludes split=='test'")

    split = d.split.to_numpy()
    holdout = d.station_holdout.to_numpy()
    train = (split == "train") & ~holdout
    keep_pos = np.where(train | dev_union)[0]
    v22 = masks["val_2022"]
    stations = d.station_id.to_numpy()[v22]
    print(f"train={int(train.sum()):,}  dev_union={int(dev_union.sum()):,}  "
          f"val_2022={int(v22.sum()):,}")

    # ---- D16 config lookup -------------------------------------------------
    mi3 = pd.read_csv(REPORTS / "mi3_hpo.csv")
    d16_cfg = {}
    for hz in run_hors:
        tag = D16[hz]
        g = mi3[(mi3.horizon == hz) & (mi3.tag == tag)]
        if g.empty:
            print(f"FATAL: D16 config {tag}/{hz} not found in mi3_hpo.csv")
            return 2
        row = g.iloc[0]
        d16_cfg[hz] = {"family": row.family, "tag": tag, "cfg": cfg_from_row(row),
                       "val_2022_ledger": float(row.val_2022)}

    # ---- fit + score -------------------------------------------------------
    if a.dry_run:
        print("\n--dry-run: resolved roster, no fits executed --")
        for hz in run_hors:
            w = winners[hz]
            print(f"  {hz}: mi8_{arm} {w['family']}/{w['tag']} "
                  f"(ledger val_2022={w['val_2022_ledger']:.6f})  cfg={w['cfg']}")
            if hz in ctrl:
                print(f"  {hz}: mi8_full_v2 {ctrl[hz]['family']}/{ctrl[hz]['tag']} "
                      f"(ledger val_2022={ctrl[hz]['val_2022_ledger']:.6f})")
            print(f"  {hz}: d16_frozen {d16_cfg[hz]['family']}/{d16_cfg[hz]['tag']} "
                  f"(archived val_2022={d16_cfg[hz]['val_2022_ledger']:.6f})")
        n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
        print(f"\n{len(RESULTS)} checks: {len(RESULTS) - n_fail} PASS / {n_fail} FAIL")
        return 0 if n_fail == 0 else 1

    scorecard_rows, preds = [], {}
    for hz in run_hors:
        y_all = d[hz].to_numpy()
        tr = train & ~pd.isna(y_all)
        Xtr, ytr = d.loc[tr, (feats if arm == "full_v3" else feats)], y_all[tr]

        roster = [(f"mi8_{arm}", winners[hz], feats)]
        if hz in ctrl:
            roster.append(("mi8_full_v2", ctrl[hz], feats_v2))
        roster.append(("d16_frozen", d16_cfg[hz], feats_v2))

        for label, spec, fs in roster:
            t1 = time.time()
            pipe = S.build(spec["family"], spec["cfg"], fs)
            Xtr_, ytr_ = (d.loc[tr, fs], y_all[tr])
            pipe.fit(Xtr_, ytr_)
            p_sub = pipe.predict_proba(d.iloc[keep_pos][fs])[:, 1]
            p_all = np.full(len(d), np.nan)
            p_all[keep_pos] = p_sub
            preds[(hz, label)] = p_all

            thr = B.best_f1_threshold(y_all[tr], p_all[tr])
            v22_pr = float("nan")
            for cell in DEV:
                m = masks[cell] & ~pd.isna(y_all)
                if m.sum() == 0 or len(np.unique(y_all[m])) < 2:
                    continue
                met = B.metrics(y_all[m], p_all[m], thr)
                if cell == "val_2022":
                    v22_pr = met["pr_auc"]
                scorecard_rows.append({
                    "model": label, "family": spec["family"], "config": spec["tag"],
                    "horizon": hz, "cell": cell, "n": int(m.sum()),
                    "n_pos": int(y_all[m].sum()), "season": "all",
                    "train_rows": int(tr.sum()), "threshold": thr, **met,
                    "fit_s": round(time.time() - t1, 1)})
            print(f"  {hz} {label:12s} {spec['family']:5s}/{spec['tag']:8s} "
                  f"val_2022={v22_pr:.5f}  {time.time() - t1:.0f}s", flush=True)
            del pipe

    scorecard = pd.DataFrame(scorecard_rows)

    # ---- D16 reproduction gate --------------------------------------------
    d16_chk = scorecard[(scorecard.model == "d16_frozen") & (scorecard.cell == "val_2022") &
                        (scorecard.season == "all")] if "season" in scorecard.columns else \
              scorecard[(scorecard.model == "d16_frozen") & (scorecard.cell == "val_2022")]
    print()
    for hz in run_hors:
        row = d16_chk[d16_chk.horizon == hz]
        if row.empty:
            check(f"R_{hz}", False, "D16 refit produced no val_2022 score")
            continue
        got = float(row.pr_auc.iloc[0])
        exp = D16_EXPECTED[hz]
        check(f"R_{hz}", abs(got - exp) < 1e-5,
              f"D16 refit {got:.6f} vs archived {exp:.6f} (Δ={got - exp:+.2e})")

    # ---- paired bootstrap --------------------------------------------------
    delta_rows = []
    for hz in run_hors:
        y = d[hz].to_numpy()[v22]
        ok = ~pd.isna(y)
        p_new = preds[(hz, f"mi8_{arm}")][v22][ok]
        p_ref = preds[(hz, "d16_frozen")][v22][ok]
        res = paired_station_bootstrap(stations[ok], y[ok], p_new, p_ref, reps=a.reps)
        if res is None:
            continue
        res.update({"horizon": hz, "contrast": f"mi8_{arm} - d16_frozen"})
        delta_rows.append(res)
        ci_excl = "excludes 0" if (res["lo"] > 0 or res["hi"] < 0) else "STRADDLES 0"
        print(f"  {hz}: Δ={res['delta']:+.5f} CI[{res['lo']:+.5f},{res['hi']:+.5f}] "
              f"P(Δ>0)={res['pgt']:.3f}  -> {ci_excl}")

        if hz in ctrl:
            r2 = paired_station_bootstrap(stations[ok], y[ok],
                                          preds[(hz, "mi8_full_v2")][v22][ok], p_ref, reps=a.reps)
            if r2 is not None:
                r2.update({"horizon": hz, "contrast": "mi8_full_v2 - d16_frozen"})
                delta_rows.append(r2)

    deltas = pd.DataFrame(delta_rows)

    # ---- headline ----------------------------------------------------------
    def horizon_mean(label):
        sel = scorecard[(scorecard.model == label) & (scorecard.cell == "val_2022")]
        per = {r.horizon: r.pr_auc for r in sel.itertuples() if r.horizon in run_hors}
        return per, (float(np.mean(list(per.values()))) if len(per) == len(run_hors)
                     else float("nan"))

    per_new, hm_new = horizon_mean(f"mi8_{arm}")
    per_ref, hm_ref = horizon_mean("d16_frozen")
    per_ctl, hm_ctl = horizon_mean("mi8_full_v2") if "mi8_full_v2" in set(scorecard.model) else ({}, float("nan"))

    print()
    print("=== horizon-mean on the decision cell (val_2022, season=all) ===")
    print(f"  mi8_{arm} : {hm_new:.5f}   {[f'{k[-3:]}={v:.5f}' for k, v in per_new.items()]}")
    print(f"  d16_frozen: {hm_ref:.5f}   {[f'{k[-3:]}={v:.5f}' for k, v in per_ref.items()]}")
    if per_ctl:
        print(f"  full_v2   : {hm_ctl:.5f}   {[f'{k[-3:]}={v:.5f}' for k, v in per_ctl.items()]}")
    print(f"  mi8 - d16 : {hm_new - hm_ref:+.5f}")

    # ---- write -------------------------------------------------------------
    if len(run_hors) != len(HOR):
        print(f"\nNOTE: partial run ({run_hors}) — artefacts NOT written; "
              f"re-run with all horizons to publish.")
        n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
        print(f"\n{len(RESULTS)} checks: {len(RESULTS) - n_fail} PASS / {n_fail} FAIL  "
              f"wall={time.time() - t0:.0f}s")
        return 0 if n_fail == 0 else 1

    scorecard.to_csv(REPORTS / "mi8_scorecard.csv", index=False)
    deltas.to_csv(REPORTS / "mi8_paired_deltas.csv", index=False)

    summary = {
        "seed": SEED, "arm": arm, "reps": a.reps, "test_rows_read": TEST_ROWS_READ,
        "selection_cell": "val_2022", "selection_season": "all",
        "rule": "per-horizon argmax of val_2022 PR-AUC (pre-registered D16 rule)",
        "winners": {h: {"family": v["family"], "tag": v["tag"], "cfg": v["cfg"]}
                    for h, v in winners.items()},
        "control_full_v2": ({h: {"family": v["family"], "tag": v["tag"]} for h, v in ctrl.items()}
                            if ctrl else None),
        "per_horizon": {"mi8": per_new, "d16": per_ref, "full_v2": per_ctl},
        "horizon_mean": {"mi8": hm_new, "d16": hm_ref, "full_v2": hm_ctl,
                         "mi8_minus_d16": hm_new - hm_ref,
                         "full_v2_minus_d16": (hm_ctl - hm_ref) if per_ctl else None,
                         "mi8_minus_full_v2": (hm_new - hm_ctl) if per_ctl else None},
        "paired_bootstrap": delta_rows,
        "d16_reproduction": {h: D16_EXPECTED[h] for h in run_hors},
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    (REPORTS / "mi8_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS)} checks: {len(RESULTS) - n_fail} PASS / {n_fail} FAIL")
    print(f"WROTE reports/mi8_scorecard.csv ({len(scorecard)} rows), "
          f"mi8_paired_deltas.csv ({len(deltas)} rows), mi8_summary.json  "
          f"wall={time.time() - t0:.0f}s")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
