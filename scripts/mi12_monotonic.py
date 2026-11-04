r"""Experiment 6 — monotonic physical priors for flood-probability forecasting.

Idea
----
Every hydrological feature has a known *direction*: a higher stage or more rain can
only raise the probability of exceeding the danger level; a larger margin below the
danger level can only lower it.  Gradient boosting does not know this and will happily
learn an inverted response from a wet training regime (2017-21, base rate 7-14 %) that
does not transfer to the dry evaluation regime (2025-26, ~1-2 %).  Monotone constraints
(LightGBM ``monotone_constraints`` / XGBoost ``monotone_constraints``) impose the sign
as a hard shape prior on every tree split.

This is a *knowledge-informed inductive bias*, not a PDE solver: it constrains the
sign of the response, not its magnitude or functional form.

Arms (identical features FULL_V2, identical rows, identical D16 hyper-parameters)
--------------------------------------------------------------------------------
  a0_unconstrained  D16 exactly, no constraints                  (control)
  a1_strict         42 high-confidence hydrological signs
  a2_broad          the strict 42 + 48 second-order proxies (90 total)
  a3_placebo        a1 signs multiplied by -1                    (anti-physics control)
  a4_random         random +/-1 on the same 42 features           (regularisation control)
  b1_targeted       per-horizon signs on features whose unconstrained PDP response
                    is materially wrong-signed in BOTH 2020 and 2021 audits (Part B)
  b2_targeted_placebo  b1 signs multiplied by -1                 (Part B anti-physics)

Part B (targeted priors)
------------------------
Selection uses only the 2020 and 2021 out-of-sample audits: a feature is constrained at
horizon h iff its PDP response is wrong-signed with |delta P| >= 0.001 in both years.
val_2022 is never consulted for selection.  The 24 h target set is empty, so b1/b2 are
exact no-ops there; Part B is therefore decided per horizon with the same four criteria.

Protocol (strictly out-of-sample; copied from Experiment 1)
-----------------------------------------------------------
  t2020    train years <= 2019            -> score 2020
  t2021    train years <= 2020            -> score 2021
  t2022    train split (<= 2021)          -> score val_2022
  sf1..4   train split with fold != k     -> score fold k
Every scored row is unseen by the model that scores it.  This is the corrected
protocol; the MI sprint's 7-cell means were in-sample for six of seven cells.

Test policy
-----------
NO test row is read.  ``make_tasks`` asserts every evaluation mask is disjoint from
``split == "test"``; the summary records ``test_rows_read = 0``.  A test read needs a
separate pre-registered decision.

Pre-registered decision rule (written before the run)
-----------------------------------------------------
A treatment arm is SUPPORTED only if all four hold:
  1. horizon-mean val_2022 PR-AUC        >= a0_unconstrained
  2. worst (cell, horizon) PR-AUC        >  a0_unconstrained
  3. no (cell, horizon) regresses        >  0.005 below a0
  4. horizon-mean val_2022               >  both placebo arms
Otherwise the result is recorded as not supported and D16 stands.

Usage
-----
    python -u scripts/mi12_monotonic.py signs
    python -u scripts/mi12_monotonic.py audit [--task t2022] [--suffix 2020]
    python -u scripts/mi12_monotonic.py targeted
    python -u scripts/mi12_monotonic.py run [--horizons ...] [--tasks ...] [--arms ...]
    python -u scripts/mi12_monotonic.py report
    python -u scripts/mi12_monotonic.py all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B                                    # noqa: E402

from lightgbm import LGBMClassifier                            # noqa: E402
from sklearn.metrics import average_precision_score            # noqa: E402
from xgboost import XGBClassifier                              # noqa: E402

ROOT = Path(r"D:\Flood predictor")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
OUT = REPORTS / "mi12_monotonic"
OUT.mkdir(parents=True, exist_ok=True)

SEED = B.SEED
HORIZONS = B.HORIZONS
N_JOBS = 4

SIGNS_JSON = PROC / "monotonic_signs_v1.json"
LEDGER = OUT / "ledger.csv"
PRED_DIR = OUT / "preds"
PRED_DIR.mkdir(exist_ok=True)
SCORECARD = OUT / "scorecard.csv"
BOOTSTRAP = OUT / "bootstrap.csv"
DELTAS = OUT / "deltas_vs_a0.csv"
AUDIT_CSV = OUT / "pdp_audit.csv"
SUMMARY = OUT / "summary.json"

ARMS = ["a0_unconstrained", "a1_strict", "a2_broad", "a3_placebo", "a4_random",
        "b1_targeted", "b2_targeted_placebo"]
TREATMENTS = ["a1_strict", "a2_broad", "b1_targeted"]
PLACEBOS = ["a3_placebo", "a4_random", "b2_targeted_placebo"]
TARGETED_JSON = PROC / "monotonic_signs_v1b_targeted.json"

TASKS = ["t2020", "t2021", "t2022", "sf1", "sf2", "sf3", "sf4"]
CELL_OF = {"t2020": "val_2020", "t2021": "val_2021", "t2022": "val_2022",
           "sf1": "spatial_fold_1", "sf2": "spatial_fold_2",
           "sf3": "spatial_fold_3", "sf4": "spatial_fold_4"}
DEV_CELLS = [CELL_OF[t] for t in TASKS]

# D16 = the frozen champion's per-horizon family and configuration
# (models/final_v2/final_config_v2.json).  Nothing is re-tuned.
D16_FAMILY = {"label_24h": "lgbm", "label_48h": "lgbm", "label_72h": "xgb"}
D16_CFG = {
    "label_24h": dict(n_estimators=300, learning_rate=0.02, subsample=0.85,
                      colsample_bytree=0.6, reg_lambda=3.0, num_leaves=63,
                      min_child_samples=40),
    "label_48h": dict(n_estimators=300, learning_rate=0.02, subsample=0.85,
                      colsample_bytree=0.6, reg_lambda=3.0, num_leaves=63,
                      min_child_samples=40),
    "label_72h": dict(n_estimators=800, learning_rate=0.02, max_depth=4,
                      min_child_weight=1.0, subsample=1.0, colsample_bytree=0.8,
                      reg_lambda=3.0, scale_pos_weight=1.0),
}


# --------------------------------------------------------------------------- data
def load_data() -> tuple[pd.DataFrame, list[str]]:
    """experimental_dataset + split assignments + the frozen MI-v2 feature block."""
    d, _ = B.load()
    v2 = pd.read_csv(PROC / "mi_v2_features.csv", parse_dates=["date"])
    d = d.merge(v2, on=["station_id", "date"], validate="one_to_one")
    feats = json.loads((PROC / "feature_sets_v2.json").read_text(
        encoding="utf-8"))["sets"]["FULL_V2"]
    d = d.sort_values(["station_id", "date"]).reset_index(drop=True)
    missing = [c for c in feats if c not in d.columns]
    if missing:
        raise RuntimeError(f"missing FULL_V2 columns: {missing[:10]}")
    print(f"loaded {len(d):,} rows x {d.shape[1]} cols; FULL_V2 = {len(feats)} features")
    return d, feats


def make_tasks(d: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Strictly out-of-sample train/eval masks, identical to Experiment 1."""
    split = d.split.to_numpy()
    holdout = d.station_holdout.to_numpy()
    year = d.date.dt.year.to_numpy()
    fold = d.spatial_fold.to_numpy()
    train_split = (split == "train") & ~holdout

    tasks = {
        "t2020": (train_split & (year <= 2019), (year == 2020) & ~holdout),
        "t2021": (train_split & (year <= 2020), (year == 2021) & ~holdout),
        "t2022": (train_split, (split == "val") & ~holdout),
    }
    for k in (1, 2, 3, 4):
        tasks[f"sf{k}"] = (train_split & (fold != k),
                           (split == "train") & (fold == k) & ~holdout)

    # hard guard: the frozen test period is never an evaluation cell
    for name, (_, ev) in tasks.items():
        if (ev & (split == "test")).any():
            raise AssertionError(f"task {name} touches the frozen test period")
    return tasks


# ------------------------------------------------------------------------- signs
def load_signs(feats: list[str]) -> dict:
    s = json.loads(SIGNS_JSON.read_text(encoding="utf-8"))
    feats_in = s["features"]
    if set(feats_in) != set(feats):
        missing = sorted(set(feats) - set(feats_in))
        extra = sorted(set(feats_in) - set(feats))
        raise RuntimeError(f"sign table mismatch: missing={missing} extra={extra}")
    for name, rec in feats_in.items():
        if rec["sign"] not in (-1, 0, 1):
            raise RuntimeError(f"{name}: sign must be -1/0/1")
        if rec["tier"] not in ("strict", "broad", "none"):
            raise RuntimeError(f"{name}: bad tier {rec['tier']}")
        if rec["tier"] == "none" and rec["sign"] != 0:
            raise RuntimeError(f"{name}: tier none must have sign 0")
        if rec["tier"] != "none" and rec["sign"] == 0:
            raise RuntimeError(f"{name}: tier {rec['tier']} must have a sign")
    strict = [f for f in feats if feats_in[f]["tier"] == "strict"]
    broad = [f for f in feats if feats_in[f]["tier"] in ("strict", "broad")]
    print(f"sign table OK: {len(strict)} strict / {len(broad)} broad / "
          f"{len(feats) - len(broad)} unconstrained")
    # Part B: targeted sets, mechanically derived from the pre-2022 audits
    s["targeted"] = {}
    if TARGETED_JSON.exists():
        t = json.loads(TARGETED_JSON.read_text(encoding="utf-8"))
        for h in HORIZONS:
            feats_h = t["per_horizon"].get(h, [])
            bad = [f for f in feats_h if feats_in[f]["sign"] == 0]
            if bad:
                raise RuntimeError(f"targeted feature without a physical sign: {bad}")
            s["targeted"][h] = feats_h
        print("targeted sets: " + ", ".join(
            f"{h}={len(s['targeted'][h])}" for h in HORIZONS))
    return s


def _order(feats: list[str]) -> tuple[list[str], list[str], list[str]]:
    """ColumnTransformer output order: numeric first, then the 5 text columns."""
    num = [c for c in feats if c not in B.TEXT_COLS]
    txt = [c for c in feats if c in B.TEXT_COLS]
    return num, txt, num + txt


def constraint_vector(feats: list[str], signs: dict, arm: str,
                      horizon: str = "label_72h") -> list[int]:
    """Monotone vector in ColumnTransformer output order.

    Part B arms depend on the horizon: the targeted set is derived separately for
    24/48/72 h from the pre-2022 audits.  Part A arms are horizon-invariant.
    """
    num, txt, order = _order(feats)
    table = signs["features"]
    vec = [0] * len(order)
    if arm == "a0_unconstrained":
        return vec
    if arm == "a1_strict":
        chosen = {f for f in feats if table[f]["tier"] == "strict"}
    elif arm == "a2_broad":
        chosen = {f for f in feats if table[f]["tier"] in ("strict", "broad")}
    elif arm == "a3_placebo":
        chosen = {f for f in feats if table[f]["tier"] == "strict"}
    elif arm == "a4_random":
        chosen = {f for f in feats if table[f]["tier"] == "strict"}
    elif arm in ("b1_targeted", "b2_targeted_placebo"):
        if horizon not in signs.get("targeted", {}):
            raise RuntimeError(f"no targeted sign set for {horizon}; run 'targeted'")
        chosen = set(signs["targeted"][horizon])
    else:
        raise ValueError(arm)

    rng = np.random.default_rng(SEED)
    for f in order:
        if f not in chosen:
            continue
        s = table[f]["sign"]
        if arm in ("a3_placebo", "b2_targeted_placebo"):
            s = -s
        elif arm == "a4_random":
            s = int(rng.choice([-1, 1]))
        vec[order.index(f)] = s

    if len(txt) != 5:
        raise AssertionError(f"expected 5 text columns, got {len(txt)}")
    return vec


def make_clf(arm: str, horizon: str, cons: list[int]):
    fam = D16_FAMILY[horizon]
    cfg = dict(D16_CFG[horizon])
    if fam == "lgbm":
        return LGBMClassifier(**cfg, monotone_constraints=list(cons),
                              random_state=SEED, n_jobs=N_JOBS, verbose=-1)
    return XGBClassifier(**cfg, monotone_constraints=tuple(cons),
                         random_state=SEED, n_jobs=N_JOBS, eval_metric="logloss",
                         tree_method="hist")


# ----------------------------------------------------------------------- running
LEDGER_COLS = ["task", "horizon", "arm", "n_train", "n_eval", "base_rate",
               "base_rate_train", "pr_auc", "secs"]


def _load_ledger() -> pd.DataFrame:
    if LEDGER.exists() and LEDGER.stat().st_size > 0:
        return pd.read_csv(LEDGER, on_bad_lines="skip")
    return pd.DataFrame(columns=LEDGER_COLS)


def _append(row: dict) -> None:
    df = pd.DataFrame([row]).reindex(columns=LEDGER_COLS)
    header = (not LEDGER.exists()) or LEDGER.stat().st_size == 0
    df.to_csv(LEDGER, mode="a", header=header, index=False)


def _pred_path(arm: str, horizon: str) -> Path:
    return PRED_DIR / f"pred_{arm}__{horizon}.csv.gz"


def cmd_run(args) -> int:
    d, feats = load_data()
    signs = load_signs(feats)
    tasks = make_tasks(d)
    ledger = _load_ledger()
    done = set(zip(ledger.task, ledger.horizon, ledger.arm)) if len(ledger) else set()

    num, txt, order = _order(feats)
    want_h = [h for h in args.horizons if h in HORIZONS] or list(HORIZONS)
    want_t = [t for t in args.tasks if t in TASKS] or list(TASKS)
    want_a = [a for a in (getattr(args, "arms", None) or ARMS) if a in ARMS]
    if not TARGETED_JSON.exists():
        skipped = [a for a in want_a if a.startswith("b")]
        want_a = [a for a in want_a if not a.startswith("b")]
        if skipped:
            print(f"  skipping {skipped}: no targeted sign table yet")
    # Part B vectors are horizon-specific, so cache on (arm, horizon)
    vecs = {(a, h): constraint_vector(feats, signs, a, h)
            for a in want_a for h in want_h}
    for a in want_a:
        counts = [sum(1 for v in vecs[(a, h)] if v != 0) for h in want_h]
        print(f"  {a:20s} constrained features = "
              + "/".join(f"{c:3d}" for c in counts)
              + "  (" + "/".join(h.replace("label_", "") for h in want_h) + ")")

    y_all = {h: d[h].to_numpy() for h in HORIZONS}
    t0 = time.time()

    for task in want_t:
        tr_mask, ev_mask = tasks[task]

        for horizon in want_h:
            y = y_all[horizon]
            # Fit the imputer / frequency encoder on the SAME rows the frozen D16
            # pipeline used: training rows with a non-missing label for this horizon.
            tr_idx = np.flatnonzero(tr_mask & ~np.isnan(y))
            ev_idx = np.flatnonzero(ev_mask & ~np.isnan(y))
            ytr = y[tr_idx].astype(int)
            yev = y[ev_idx].astype(int)
            st_ev = d.station_id.to_numpy()[ev_idx]
            rows_ev = ev_idx
            if len(np.unique(ytr)) < 2 or len(np.unique(yev)) < 2:
                print(f"  {task} {horizon}: skipped (single class)")
                continue
            pre = B.preprocessor(num, txt, scale=False)
            Atr = pre.fit_transform(d.iloc[tr_idx][feats])
            Eev = pre.transform(d.iloc[ev_idx][feats])
            print(f"\n=== task {task} / {horizon}  train={len(tr_idx):,}  "
                  f"eval={len(ev_idx):,} ===", flush=True)

            for arm in want_a:
                key = (task, horizon, arm)
                if key in done:
                    print(f"  {horizon} {arm:20s} skip (ledger)")
                    continue
                t1 = time.time()
                clf = make_clf(arm, horizon, vecs[(arm, horizon)])
                clf.fit(Atr, ytr)
                p = clf.predict_proba(Eev)[:, 1]
                auc = float(average_precision_score(yev, p))
                _append(dict(task=task, horizon=horizon, arm=arm,
                             n_train=int(len(ytr)), n_eval=int(len(yev)),
                             base_rate=float(yev.mean()),
                             base_rate_train=float(ytr.mean()),
                             pr_auc=auc, secs=round(time.time() - t1, 1)))
                done.add(key)

                path = _pred_path(arm, horizon)
                new = pd.DataFrame(dict(task=task, horizon=horizon, arm=arm,
                                        row=rows_ev, y=yev, p=p,
                                        station_id=st_ev))
                header = not path.exists()
                new.to_csv(path, mode="a", header=header, index=False,
                           compression="gzip")
                print(f"  {horizon} {arm:20s} n={len(yev):6,} "
                      f"br={yev.mean():.4f} PR-AUC={auc:.6f} "
                      f"({time.time() - t1:.0f}s)", flush=True)

    print(f"\nrun done in {time.time() - t0:.0f}s  ledger={LEDGER}")
    n_test = int((d.split == "test").sum())
    print(f"test_rows_read = 0  (frozen test rows in table: {n_test:,}, none scored)")
    return 0


# ------------------------------------------------------------------------- audit
def _pdp(clf, E: np.ndarray, j: int, sign: int, n_rows: int, rng) -> dict:
    """Partial-dependence direction for column j, on a row sample of E."""
    n = min(n_rows, len(E))
    pick = rng.choice(len(E), size=n, replace=False)
    sub = np.array(E[pick], dtype=np.float64)
    qs = [0.01, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 0.99]
    vals = np.unique(np.quantile(sub[:, j], qs))
    if len(vals) < 3:
        return dict(n_grid=int(len(vals)), pdp_start=np.nan, pdp_end=np.nan,
                    pdp_delta=np.nan, frac_correct=np.nan, violated=False)
    tile = np.tile(sub, (len(vals), 1))
    for k, v in enumerate(vals):
        tile[k * n:(k + 1) * n, j] = v
    p = clf.predict_proba(tile)[:, 1].reshape(len(vals), n).mean(axis=1)
    dp = np.diff(p)
    correct = (dp >= -1e-12) if sign > 0 else (dp <= 1e-12)
    delta = float(p[-1] - p[0])
    return dict(n_grid=int(len(vals)), pdp_start=float(p[0]), pdp_end=float(p[-1]),
                pdp_delta=delta, frac_correct=float(correct.mean()),
                violated=bool(delta * sign < -1e-9))


def cmd_audit(args) -> int:
    """How physically consistent is the UNCONSTRAINED champion already?

    ``--task`` selects the rolling-origin task; the model is always fitted on that
    task's training rows only, so the audited cell is out-of-sample.  ``--suffix``
    keeps the pre-2022 audits separate from the val_2022 one.
    """
    d, feats = load_data()
    signs = load_signs(feats)
    tasks = make_tasks(d)
    num, txt, order = _order(feats)
    idx_of = {f: i for i, f in enumerate(order)}
    task = args.task
    if task not in tasks:
        raise SystemExit(f"unknown task {task}; choose from {TASKS}")
    tr_mask, ev_mask = tasks[task]
    rng = np.random.default_rng(SEED)

    rows = []
    for horizon in (args.horizons or HORIZONS):
        y = d[horizon].to_numpy()
        tr_idx = np.flatnonzero(tr_mask & ~np.isnan(y))
        ev_idx = np.flatnonzero(ev_mask & ~np.isnan(y))
        ytr = y[tr_idx].astype(int)
        pre = B.preprocessor(num, txt, scale=False)
        Atr = pre.fit_transform(d.iloc[tr_idx][feats])
        Eev = pre.transform(d.iloc[ev_idx][feats])
        clf = make_clf("a0_unconstrained", horizon, [0] * len(order))
        t0 = time.time()
        clf.fit(Atr, ytr)
        print(f"\n=== audit {task} / {horizon}: unconstrained fit "
              f"({time.time() - t0:.0f}s), {len(Eev):,} eval rows ===", flush=True)
        for f in order:
            rec = signs["features"][f]
            if rec["sign"] == 0:
                continue
            t1 = time.time()
            pdp = _pdp(clf, Eev, idx_of[f], rec["sign"],
                       n_rows=args.audit_rows, rng=rng)
            rows.append(dict(horizon=horizon, feature=f, sign=rec["sign"],
                             tier=rec["tier"], n_eval=int(len(Eev)), **pdp,
                             secs=round(time.time() - t1, 2)))
            flag = "VIOLATION" if pdp["violated"] else "ok"
            print(f"  {f:32s} sign={rec['sign']:+d} delta={pdp['pdp_delta']:+.4f} "
                  f"frac={pdp['frac_correct']:.2f} {flag}", flush=True)
    df = pd.DataFrame(rows)
    out_csv = OUT / (f"pdp_audit_{args.suffix}.csv" if args.suffix else "pdp_audit.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(df)} feature x horizon rows)")
    return 0


# ---------------------------------------------------------------------- targeted
def _material_violations(path: Path, horizon: str) -> dict[str, float]:
    """Wrong-sign PDP response with |ΔP| >= 0.001, from one audit CSV."""
    if not path.exists():
        raise SystemExit(f"missing audit: {path}; run 'audit --task ... --suffix ...'")
    a = pd.read_csv(path)
    a = a[a.horizon == horizon]
    wrong = (a.pdp_delta * a.sign < -1e-9) & (a.pdp_delta.abs() >= 0.001)
    return {str(r.feature): float(r.pdp_delta) for r in a[wrong].itertuples()}


def cmd_targeted(args) -> int:
    """Build the Part B target sets from the 2020 and 2021 audits only.

    Selection rule, fixed before any Part B fit: constrain a feature at horizon h
    iff its unconstrained PDP response has the wrong sign with |ΔP| >= 0.001 in
    BOTH the out-of-sample 2020 audit (t2020) and the out-of-sample 2021 audit
    (t2021).  The 2022 cell is never consulted, so it stays a clean decision cell.
    """
    d, feats = load_data()
    signs = load_signs(feats)
    table = signs["features"]
    a20 = OUT / "pdp_audit_2020.csv"
    a21 = OUT / "pdp_audit_2021.csv"

    per_h, prov, counts = {}, {}, {}
    for horizon in HORIZONS:
        v20 = _material_violations(a20, horizon)
        v21 = _material_violations(a21, horizon)
        both = sorted(set(v20) & set(v21))
        bad = [f for f in both if table[f]["sign"] == 0]
        if bad:
            raise RuntimeError(f"targeted feature without a physical sign: {bad}")
        per_h[horizon] = both
        prov[horizon] = {f: dict(sign=int(table[f]["sign"]),
                                 tier=table[f]["tier"],
                                 delta_2020=round(v20[f], 6),
                                 delta_2021=round(v21[f], 6)) for f in both}
        counts[horizon] = len(both)
        print(f"\n=== targeted set {horizon} ===")
        print(f"  material violations 2020: {len(v20):2d}  2021: {len(v21):2d}  "
              f"in both: {len(both):2d}")
        for f in both:
            print(f"  {f:32s} sign={table[f]['sign']:+d} "
                  f"2020={v20[f]:+.4f} 2021={v21[f]:+.4f}")

    payload = dict(
        experiment="Experiment 6b — targeted monotonic priors",
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed=SEED,
        rule=("constrain feature at horizon h iff its unconstrained PDP response has "
              "the wrong sign with |delta P| >= 0.001 in BOTH the 2020 and 2021 "
              "out-of-sample audits (t2020, t2021); val_2022 is never consulted"),
        material_threshold=0.001,
        audit_rows=int(getattr(args, "audit_rows", 2000)),
        source_audits={"val_2020": str(a20), "val_2021": str(a21)},
        per_horizon=per_h,
        n_per_horizon=counts,
        empty_horizons=[h for h in HORIZONS if counts[h] == 0],
        provenance=prov,
        primary_analysis=(
            "per-horizon, because the target set is horizon-specific and may be empty. "
            "Each horizon is tested separately against a0 and the placebo arms with the "
            "same four criteria. A horizon with an empty target set is a no-op by "
            "construction and cannot test the hypothesis; it is reported but not counted "
            "as evidence for or against."),
        secondary_analysis=(
            "pooled horizon mean and worst cell, reported descriptively only. The pooled "
            "test is not primary because an empty-horizon no-op makes criterion 4 "
            "mathematically impossible to pass."),
        pre_registered_criteria={
            "c1_val2022_ge_a0": "horizon val_2022 PR-AUC >= a0_unconstrained at the same horizon",
            "c2_worst_gt_a0": "worst (cell) PR-AUC > a0_unconstrained at the same horizon",
            "c3_no_cell_regression_gt_0.005": "no (cell) more than 0.005 below a0 at the same horizon",
            "c4_beats_all_placebo_arms": "horizon val_2022 > a3_placebo, a4_random and b2_targeted_placebo at the same horizon",
            "decision": "all four must hold; otherwise not supported and D16 stands",
        },
        test_policy="no test row is read; no test read is authorised by this experiment",
    )
    TARGETED_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {TARGETED_JSON}")
    return 0


# ------------------------------------------------------------------------ report
def _read_preds() -> pd.DataFrame:
    frames = []
    for p in sorted(PRED_DIR.glob("pred_*__*.csv.gz")):
        try:
            # path encodes arm and horizon: pred_<arm>__<horizon>.csv.gz
            body = p.name[len("pred_"):-len(".csv.gz")]
            arm, horizon = body.split("__", 1)
            df = pd.read_csv(p)
            if "arm" not in df.columns:
                df["arm"] = arm
            if "horizon" not in df.columns:
                df["horizon"] = horizon
            frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired_bootstrap(y: np.ndarray, p: np.ndarray, q: np.ndarray,
                     stations: np.ndarray, reps: int = 400) -> dict:
    rng = np.random.default_rng(SEED)
    uniq = np.unique(stations)
    idx_by = {s: np.flatnonzero(stations == s) for s in uniq}
    deltas = []
    for _ in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(float(average_precision_score(y[idx], p[idx])
                            - average_precision_score(y[idx], q[idx])))
    if not deltas:
        return dict(delta_mean=np.nan, lo=np.nan, hi=np.nan, p_gt0=np.nan,
                    n_boot=0)
    d = np.asarray(deltas)
    return dict(delta_mean=float(d.mean()), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), p_gt0=float((d > 0).mean()),
                n_boot=int(len(d)))


def cmd_report(args) -> int:
    if not LEDGER.exists():
        print("no ledger — run first")
        return 1
    led = pd.read_csv(LEDGER)

    # ---- scorecard -------------------------------------------------------
    rows = []
    for (arm, horizon), g in led.groupby(["arm", "horizon"]):
        g = g.set_index("task")
        rec = dict(arm=arm, horizon=horizon)
        for t in TASKS:
            if t in g.index:
                rec[CELL_OF[t]] = float(g.loc[t, "pr_auc"])
        cells = [CELL_OF[t] for t in TASKS if t in g.index]
        rec["dev_mean"] = float(np.mean([rec[c] for c in cells])) if cells else np.nan
        rec["dev_worst"] = float(np.min([rec[c] for c in cells])) if cells else np.nan
        rec["val_2022"] = rec.get("val_2022", np.nan)
        rows.append(rec)
    score = pd.DataFrame(rows)
    score.to_csv(SCORECARD, index=False)

    # ---- paired bootstrap vs a0, val_2022 only ---------------------------
    preds = _read_preds()
    boots = []
    if len(preds):
        base = preds[(preds.task == "t2022") & (preds.arm == "a0_unconstrained")]
        for arm in ARMS:
            if arm == "a0_unconstrained":
                continue
            for horizon in HORIZONS:
                a = preds[(preds.task == "t2022") & (preds.arm == arm) &
                          (preds.horizon == horizon)]
                b = base[base.horizon == horizon]
                if len(a) == 0 or len(b) == 0:
                    continue
                m = a.merge(b, on=["task", "row"], suffixes=("", "_c"))
                if len(m) == 0:
                    continue
                bs = paired_bootstrap(m.y.to_numpy(), m.p.to_numpy(),
                                      m.p_c.to_numpy(),
                                      m.station_id.to_numpy())
                boots.append(dict(arm=arm, horizon=horizon,
                                  control="a0_unconstrained", **bs))
        # Part B: is the targeted arm distinguishable from its own flipped control?
        if ({"b1_targeted", "b2_targeted_placebo"} <= set(preds.arm)):
            for horizon in HORIZONS:
                a = preds[(preds.task == "t2022") & (preds.arm == "b1_targeted") &
                          (preds.horizon == horizon)]
                b = preds[(preds.task == "t2022") &
                          (preds.arm == "b2_targeted_placebo") &
                          (preds.horizon == horizon)]
                if len(a) == 0 or len(b) == 0:
                    continue
                m = a.merge(b, on=["task", "row"], suffixes=("", "_c"))
                if len(m) == 0:
                    continue
                bs = paired_bootstrap(m.y.to_numpy(), m.p.to_numpy(),
                                      m.p_c.to_numpy(),
                                      m.station_id.to_numpy())
                boots.append(dict(arm="b1_targeted", horizon=horizon,
                                  control="b2_targeted_placebo", **bs))
    bdf = pd.DataFrame(boots)
    if len(bdf):
        bdf.to_csv(BOOTSTRAP, index=False)

    # ---- deltas vs a0 ----------------------------------------------------
    drow = []
    for horizon in HORIZONS:
        a0 = score[(score.arm == "a0_unconstrained") & (score.horizon == horizon)]
        if a0.empty:
            continue
        a0 = a0.iloc[0]
        for arm in ARMS:
            s = score[(score.arm == arm) & (score.horizon == horizon)]
            if s.empty:
                continue
            s = s.iloc[0]
            rec = dict(arm=arm, horizon=horizon)
            for c in DEV_CELLS:
                rec[c] = float(s[c] - a0[c])
            rec["dev_mean"] = float(s.dev_mean - a0.dev_mean)
            rec["dev_worst"] = float(s.dev_worst - a0.dev_worst)
            drow.append(rec)
    delta = pd.DataFrame(drow)
    if len(delta):
        delta.to_csv(DELTAS, index=False)

    # ---- audit summary (material vs negligible sign flips) ---------------
    audit = {}
    if AUDIT_CSV.exists():
        a = pd.read_csv(AUDIT_CSV)
        a["wrong"] = a.pdp_delta * a.sign < -1e-9
        a["material"] = a.wrong & (a.pdp_delta.abs() >= 0.001)
        a["negligible"] = a.wrong & (a.pdp_delta.abs() < 0.001)
        a["strong"] = a.pdp_delta.abs() >= 0.005
        for h, g in a.groupby("horizon"):
            strong = g[g.strong]
            audit[h] = dict(
                n_signed=int(len(g)),
                n_material_violations=int(g.material.sum()),
                n_negligible_flips=int(g.negligible.sum()),
                n_strong=int(len(strong)),
                strong_correct=int((strong.pdp_delta * strong.sign > 0).sum()),
                strongest=str(g.loc[g.pdp_delta.abs().idxmax(), "feature"]),
                strongest_delta=round(float(g.pdp_delta.abs().max()), 4),
                material_features=sorted(g.loc[g.material, "feature"].tolist()),
            )
        print("\n=== monotonicity audit of the unconstrained champion (val_2022 PDP) ===")
        for h in sorted(audit):
            r = audit[h]
            print(f"  {h}: {r['n_strong']} strong features "
                  f"({r['strong_correct']}/{r['n_strong']} correct), "
                  f"{r['n_material_violations']} material violations, "
                  f"{r['n_negligible_flips']} negligible sign flips")

    # ---- pre-registered decision ----------------------------------------
    def summarise(arm: str):
        s = score[score.arm == arm]
        return dict(
            val_2022_mean=float(s.val_2022.mean()),
            dev_mean=float(s.dev_mean.mean()),
            dev_worst=float(s.dev_worst.min()))

    verdicts = {}
    verdicts_h = {}
    if len(score):
        a0 = summarise("a0_unconstrained")
        placebo_of = {
            "a1_strict": ["a3_placebo", "a4_random"],
            "a2_broad": ["a3_placebo", "a4_random"],
            "b1_targeted": ["a3_placebo", "a4_random", "b2_targeted_placebo"],
        }
        for arm in TREATMENTS:
            if score[score.arm == arm].empty:
                continue
            ps = [p for p in placebo_of[arm] if not score[score.arm == p].empty]
            if not ps:
                continue
            placebo_best = max(summarise(p)["val_2022_mean"] for p in ps)
            s = summarise(arm)
            regress = delta[delta.arm == arm]
            min_cell_delta = float(min(regress[c].min() for c in DEV_CELLS))
            crit = {
                "c1_val2022_ge_a0": s["val_2022_mean"] >= a0["val_2022_mean"],
                "c2_worst_gt_a0": s["dev_worst"] > a0["dev_worst"],
                "c3_no_cell_regression_gt_0.005": min_cell_delta >= -0.005,
                "c4_beats_declared_placebos": s["val_2022_mean"] > placebo_best,
            }
            verdicts[arm] = dict(
                metrics=s, a0=a0, placebo_best=placebo_best,
                placebos=ps, min_cell_delta=min_cell_delta, criteria=crit,
                supported=bool(all(crit.values())))

            # Part B is tested per horizon (target sets are horizon-specific and may
            # be empty).  The same four criteria are applied inside each horizon.
            for h in HORIZONS:
                sh = score[(score.arm == arm) & (score.horizon == h)]
                a0h = score[(score.arm == "a0_unconstrained") &
                            (score.horizon == h)]
                if sh.empty or a0h.empty:
                    continue
                sh, a0h = sh.iloc[0], a0h.iloc[0]
                ps_h = [p for p in placebo_of[arm]
                        if not score[(score.arm == p) &
                                     (score.horizon == h)].empty]
                pb_h = max(float(score[(score.arm == p) &
                                       (score.horizon == h)].val_2022.iloc[0])
                           for p in ps_h) if ps_h else np.nan
                dh = delta[(delta.arm == arm) & (delta.horizon == h)]
                min_d = (float(min(dh[c].iloc[0] for c in DEV_CELLS))
                         if len(dh) else np.nan)
                crit_h = {
                    "c1_val2022_ge_a0": float(sh.val_2022) >= float(a0h.val_2022),
                    "c2_worst_gt_a0": float(sh.dev_worst) > float(a0h.dev_worst),
                    "c3_no_cell_regression_gt_0.005": min_d >= -0.005,
                    "c4_beats_declared_placebos": float(sh.val_2022) > pb_h,
                }
                verdicts_h[f"{arm}|{h}"] = dict(
                    arm=arm, horizon=h, val_2022=float(sh.val_2022),
                    a0_val_2022=float(a0h.val_2022),
                    dev_worst=float(sh.dev_worst),
                    a0_dev_worst=float(a0h.dev_worst),
                    placebo_best=pb_h, min_cell_delta=min_d,
                    criteria=crit_h, supported=bool(all(crit_h.values())))

    summary = dict(
        experiment="Experiment 6 — monotonic physical priors",
        seed=SEED,
        protocol=dict(
            tasks="t2020/t2021/t2022 rolling origin; sf1..sf4 station-disjoint",
            note="every scored row is out-of-sample; no test row is read",
            test_rows_read=0,
        ),
        arms=ARMS,
        n_strict=int(sum(1 for f in json.loads(SIGNS_JSON.read_text(
            encoding="utf-8"))["features"].values() if f["tier"] == "strict")),
        n_broad=int(sum(1 for f in json.loads(SIGNS_JSON.read_text(
            encoding="utf-8"))["features"].values()
            if f["tier"] in ("strict", "broad"))),
        n_targeted=(json.loads(TARGETED_JSON.read_text(encoding="utf-8"))["n_per_horizon"]
                    if TARGETED_JSON.exists() else {}),
        targeted=(json.loads(TARGETED_JSON.read_text(encoding="utf-8"))["per_horizon"]
                  if TARGETED_JSON.exists() else {}),
        audit=audit,
        scorecard=score.round(6).to_dict(orient="records") if len(score) else [],
        verdicts=verdicts,
        verdicts_by_horizon=verdicts_h,
    )
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # ---- figures ---------------------------------------------------------
    _figures(score, delta, bdf)

    # ---- console ---------------------------------------------------------
    print("\n=== scorecard (PR-AUC) ===")
    print(score.round(6).to_string(index=False))
    if len(bdf):
        print("\n=== paired station bootstrap vs a0 (val_2022) ===")
        print(bdf.round(6).to_string(index=False))
    print("\n=== pre-registered decision ===")
    for arm, v in verdicts.items():
        print(f"{arm}: supported={v['supported']}  "
              f"val2022={v['metrics']['val_2022_mean']:.6f} "
              f"(a0={v['a0']['val_2022_mean']:.6f})  "
              f"worst={v['metrics']['dev_worst']:.6f} "
              f"(a0={v['a0']['dev_worst']:.6f})  "
              f"min_cell_delta={v['min_cell_delta']:+.6f}")
        for k, ok in v["criteria"].items():
            print(f"   [{'PASS' if ok else 'FAIL'}] {k}")
    if verdicts_h:
        print("\n=== per-horizon decision (Part B primary) ===")
        for key, v in verdicts_h.items():
            if not v["arm"].startswith("b"):
                continue
            print(f"{v['arm']} {v['horizon']}: supported={v['supported']}  "
                  f"val2022={v['val_2022']:.6f} "
                  f"(a0={v['a0_val_2022']:.6f})  "
                  f"worst={v['dev_worst']:.6f} "
                  f"(a0={v['a0_dev_worst']:.6f})  "
                  f"min_cell_delta={v['min_cell_delta']:+.6f}")
            for k, ok in v["criteria"].items():
                print(f"   [{'PASS' if ok else 'FAIL'}] {k}")
    print(f"\nwrote {SCORECARD}, {BOOTSTRAP}, {DELTAS}, {SUMMARY}")
    return 0


def _figures(score: pd.DataFrame, delta: pd.DataFrame, boot: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if score.empty:
        return
    arms = [a for a in ARMS if not score[score.arm == a].empty]
    colors = {"a0_unconstrained": "#444444", "a1_strict": "#1f77b4",
              "a2_broad": "#2ca02c", "a3_placebo": "#d62728",
              "a4_random": "#ff9f1c", "b1_targeted": "#9467bd",
              "b2_targeted_placebo": "#8c564b"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=False)
    x = np.arange(len(DEV_CELLS))
    w = 0.8 / max(len(arms), 1)
    for ax, horizon in zip(axes, HORIZONS):
        s = score[score.horizon == horizon]
        for i, arm in enumerate(arms):
            g = s[s.arm == arm]
            if g.empty:
                continue
            g = g.iloc[0]
            ax.bar(x + (i - (len(arms) - 1) / 2) * w,
                   [g[c] for c in DEV_CELLS], width=w, label=arm.replace("_", " "),
                   color=colors[arm])
        ax.set_title(horizon.replace("label_", "").replace("h", " h"))
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("spatial_fold_", "sf").replace("val_", "v")
                            for c in DEV_CELLS], rotation=30, fontsize=8)
        ax.set_ylabel("PR-AUC")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0.5, 1.0)
    axes[0].legend(fontsize=6, ncol=2, loc="lower right")
    fig.suptitle("Experiment 6 — monotonic priors: PR-AUC per out-of-sample dev cell "
                 "(no test rows read)")
    fig.tight_layout()
    fig.savefig(FIGURES / "mi12_pr_auc_by_cell.png", dpi=140)
    plt.close(fig)

    if not delta.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.3))
        hs = HORIZONS
        panels = [
            ("all arms", [a for a in ARMS if a != "a0_unconstrained"]),
            ("excluding the flipped-sign control",
             [a for a in ARMS if a not in ("a0_unconstrained", "a3_placebo")]),
        ]
        for ax, (title, arms_plot) in zip(axes, panels):
            for arm in arms_plot:
                g = delta[delta.arm == arm].set_index("horizon").reindex(hs)
                if g.dev_mean.isna().all():
                    continue
                ax.plot(range(len(hs)), g.dev_mean, marker="o",
                        label=arm.replace("_", " "))
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xticks(range(len(hs)))
            ax.set_xticklabels([h.replace("label_", "") for h in hs])
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        axes[0].set_ylabel("Δ mean dev-cell PR-AUC vs D16")
        fig.suptitle("Experiment 6 — dose-response: constrained minus unconstrained")
        fig.tight_layout()
        fig.savefig(FIGURES / "mi12_delta_vs_d16.png", dpi=140)
        plt.close(fig)

    if not boot.empty:
        fig, ax = plt.subplots(figsize=(7.6, 4.0))
        label, lo, hi, mean = [], [], [], []
        for arm in [a for a in ARMS if a != "a0_unconstrained"]:
            g = boot[(boot.arm == arm) &
                     (boot.control == "a0_unconstrained")]
            if g.empty:
                continue
            label.append(arm.replace("_", " "))
            lo.append(g.lo.mean())
            hi.append(g.hi.mean())
            mean.append(g.delta_mean.mean())
        y = np.arange(len(label))
        ax.errorbar(mean, y, xerr=[np.array(mean) - np.array(lo),
                                   np.array(hi) - np.array(mean)],
                    fmt="o", capsize=4)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(label)
        ax.set_xlabel("Δ PR-AUC vs D16 (val_2022, 400 station bootstraps, mean of horizons)")
        ax.set_title("Experiment 6 — paired bootstrap vs the unconstrained champion")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "mi12_bootstrap.png", dpi=140)
        plt.close(fig)


def cmd_signs(args) -> int:
    d, feats = load_data()
    signs = load_signs(feats)
    n_num, n_txt, order = _order(feats)
    print(f"transformed width = {len(order)} (numeric {len(n_num)} + text {len(n_txt)})")
    for a in ARMS:
        if a.startswith("b"):
            if not signs.get("targeted"):
                print(f"  {a:20s} skipped (no targeted sign table yet)")
                continue
            for h in HORIZONS:
                v = constraint_vector(feats, signs, a, h)
                print(f"  {a:20s} {h.replace('label_', ''):4s} "
                      f"+1={sum(x == 1 for x in v):3d}  -1={sum(x == -1 for x in v):3d}"
                      f"  0={sum(x == 0 for x in v):3d}")
        else:
            v = constraint_vector(feats, signs, a)
            print(f"  {a:20s}       +1={sum(x == 1 for x in v):3d}  "
                  f"-1={sum(x == -1 for x in v):3d}  0={sum(x == 0 for x in v):3d}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("signs").set_defaults(func=cmd_signs)

    p = sub.add_parser("audit")
    p.add_argument("--horizons", nargs="*", default=HORIZONS)
    p.add_argument("--task", default="t2022", choices=TASKS)
    p.add_argument("--suffix", default="",
                   help="output suffix: pdp_audit_<suffix>.csv")
    p.add_argument("--audit-rows", type=int, default=2000)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("run")
    p.add_argument("--horizons", nargs="*", default=HORIZONS)
    p.add_argument("--tasks", nargs="*", default=TASKS)
    p.add_argument("--arms", nargs="*", default=ARMS)
    p.set_defaults(func=cmd_run)

    sub.add_parser("targeted").set_defaults(func=cmd_targeted)

    sub.add_parser("report").set_defaults(func=cmd_report)

    p = sub.add_parser("all")
    p.add_argument("--horizons", nargs="*", default=HORIZONS)
    p.add_argument("--tasks", nargs="*", default=TASKS)
    p.add_argument("--audit-rows", type=int, default=2000)
    p.set_defaults(func=None)

    a = ap.parse_args(argv)
    if a.cmd == "all":
        rc = cmd_audit(argparse.Namespace(horizons=a.horizons,
                                          audit_rows=a.audit_rows,
                                          task="t2022", suffix=""))
        if rc:
            return rc
        rc = cmd_run(argparse.Namespace(horizons=a.horizons, tasks=a.tasks,
                                        arms=ARMS))
        if rc:
            return rc
        return cmd_report(argparse.Namespace())
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
