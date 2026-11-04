r"""MI-11 — event-level evaluation of the governed model against D7 persistence.

The station-day metrics in the rest of the project answer "is this row ranked above the
threshold?".  An early-warning system also has to answer "did we warn before the flood
event, how early, and how often did we cry wolf?".  This script measures that.

Definitions (dev cells only)
----------------------------
* **Event**: a maximal run of consecutive days on which the observed daily maximum water
  level is at or above the station's danger level (`wl_pct_danger >= 1`).  A single
  one-day dip is bridged, because a flood wave commonly oscillates around the threshold.
  Events are built on the non-test timeline only; the frozen test is never read.
* **Detected**: for horizon h (1/2/3 days), at least one forecast issue day in
  `{onset-1, ..., onset-h}` has `p >= threshold`.  The threshold is the cost-optimal threshold (C_FN=10, C_FP=1) on that
  cell's *training* rows, the same operating point as MI-6 and Phase 29.
* **Lead time**: `onset - first warning day`, in days (1..h).
* **False-alarm episode**: a maximal run of warning days outside every event span and
  every event's lead window.  Warnings in the last h days of a cell are excluded because
  their target window is not observable inside the cell.

Cells: the seven development cells (P2 `val_2020/2021`, P1 `val_2022`, P3 folds 1-4).
Models: D16 (frozen per-horizon config) and the D7 persistence reference, each refit
per cell.  No test row is fitted, predicted or scored.

Run
---
    .venv\Scripts\python.exe -u scripts\mi11_event_eval.py
    .venv\Scripts\python.exe -u scripts\mi11_event_eval.py --smoke
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_model as B                       # noqa: E402
import mi3_search as S                           # noqa: E402
import mi10_p2p3_rescore as M10                  # noqa: E402
import phase29_thresholds as P29                 # noqa: E402

ROOT = Path(r"D:\Flood predictor")
OUT = ROOT / "reports" / "event_level"
FIG = ROOT / "figures"
HOR = B.HORIZONS
HDAYS = {"label_24h": 1, "label_48h": 2, "label_72h": 3}
HZ = {"label_24h": "24 h", "label_48h": "48 h", "label_72h": "72 h"}
CELLS = ["val_2020", "val_2021", "val_2022",
         "spatial_fold_1", "spatial_fold_2", "spatial_fold_3", "spatial_fold_4"]
MODELS = ["d16_frozen", "d7_persistence"]


# ------------------------------------------------------------------ events
def build_event_catalog(d: pd.DataFrame) -> pd.DataFrame:
    """One row per observed danger-level exceedance event, non-test rows only."""
    use = d[d.split != "test"][["station_id", "date"]].copy()
    use["above"] = (pd.to_numeric(d.loc[use.index, "wl_pct_danger"], errors="coerce") >= 1.0).to_numpy()
    use = use.sort_values(["station_id", "date"])

    rows = []
    for sid, g in use.groupby("station_id", sort=False):
        dates = g.date.to_numpy()
        a = g.above.to_numpy().astype(bool)
        # bridge an isolated one-day dip (observed below or missing), but only
        # when the neighbouring rows are exactly one calendar day apart - the
        # record has a missing year and station gaps, and those must not merge
        for k in range(1, len(a) - 1):
            if (not a[k] and a[k - 1] and a[k + 1]
                    and (dates[k] - dates[k - 1]) / np.timedelta64(1, "D") == 1
                    and (dates[k + 1] - dates[k]) / np.timedelta64(1, "D") == 1):
                a[k] = True
        i = 0
        while i < len(a):
            if not a[i]:
                i += 1
                continue
            j = i
            while (j + 1 < len(a) and a[j + 1]
                   and (dates[j + 1] - dates[j]) / np.timedelta64(1, "D") == 1):
                j += 1
            rows.append({
                "station_id": sid,
                "onset": pd.Timestamp(dates[i]),
                "end": pd.Timestamp(dates[j]),
                "duration_days": (pd.Timestamp(dates[j]) - pd.Timestamp(dates[i])).days + 1,
            })
            i = j + 1
    ev = pd.DataFrame(rows)
    print(f"event catalog: {len(ev)} events at {ev.station_id.nunique()} stations "
          f"(non-test timeline)")
    return ev


def episodes(days: list[pd.Timestamp]) -> int:
    """Count maximal runs, treating a one-day gap as the same episode."""
    n, prev = 0, None
    for d in sorted(days):
        if prev is None or (d - prev).days > 1:
            n += 1
        prev = d
    return n


def evaluate_cell(cell, events, pred, thr, h, stations, start, end):
    """Return event-level metrics for one cell x horizon x model."""
    pmap = dict(zip(zip(pred.station_id, pred.date), pred.p.astype(float)))

    # events whose *onset* falls inside the cell and whose full lead window is observed
    ev = events[(events.station_id.isin(stations))
                & (events.onset >= start) & (events.onset <= end)
                & (events.onset - pd.Timedelta(days=h) >= start)]

    hits, leads = 0, []
    covered: set[tuple] = set()

    # coverage includes events already running at the cell boundary, so their
    # continuation warnings are not charged as false alarms
    ev_cov = events[(events.station_id.isin(stations))
                    & (events.onset <= end) & (events.end >= start)]
    for _, e in ev_cov.iterrows():
        sid = e.station_id
        for d in pd.date_range(max(e.onset, start), min(e.end, end)):
            covered.add((sid, d))
        for k in range(1, h + 1):
            d = e.onset - pd.Timedelta(days=k)
            if start <= d <= end:
                covered.add((sid, d))

    eligible = 0
    for _, e in ev.iterrows():
        lead_keys = [(e.station_id, e.onset - pd.Timedelta(days=k))
                     for k in range(1, h + 1)
                     if (e.station_id, e.onset - pd.Timedelta(days=k)) in pmap]
        if not lead_keys:
            continue        # no observable issue day: unverifiable, not a miss
        eligible += 1
        hit_days = [key[1] for key in lead_keys if pmap[key] >= thr]
        if hit_days:
            hits += 1
            leads.append((e.onset - min(hit_days)).days)

    # false-alarm episodes: warning days outside every event/lead window.
    # Warnings in the last h days are unverifiable inside this cell.
    fa_by_station: dict[str, list[pd.Timestamp]] = {}
    for (sid, d), p in pmap.items():
        if p < thr or d > end - pd.Timedelta(days=h) or (sid, d) in covered:
            continue
        fa_by_station.setdefault(sid, []).append(d)
    fa = sum(episodes(v) for v in fa_by_station.values())

    n_ev = int(len(ev))
    years = max((end - start).days / 365.25, 1e-9)
    n_st = int(len(stations))
    return {
        "n_events": n_ev, "n_eligible": eligible, "n_missing_lead": n_ev - eligible,
        "detected": int(hits),
        "event_recall": (hits / eligible) if eligible else float("nan"),
        "median_lead_days": float(np.median(leads)) if leads else float("nan"),
        "fa_episodes": int(fa),
        "fa_per_station_year": float(fa / (n_st * years)),
        "event_precision": (hits / (hits + fa)) if (hits + fa) else float("nan"),
        "n_stations": n_st, "years": round(years, 2),
    }


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=CELLS)
    ap.add_argument("--horizons", nargs="*", default=HOR)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    cells = ["val_2022"] if a.smoke else [c for c in CELLS if c in a.cells]
    horizons = ["label_24h"] if a.smoke else [h for h in HOR if h in a.horizons]
    full = (not a.smoke and len(cells) == len(CELLS) and len(horizons) == len(HOR))
    OUT.mkdir(exist_ok=True)

    t0 = time.time()
    print("=== MI-11 event-level evaluation ===")
    print(f"cells={cells}  horizons={horizons}  test_rows_read = 0")
    d, feats = M10.load_all()
    tasks = M10.cell_tasks(d)
    specs = M10.resolve_arm_specs()
    events = build_event_catalog(d)

    rows = []
    for hz in horizons:
        h = HDAYS[hz]
        y = d[hz].to_numpy()
        for cell in cells:
            task = tasks[cell]
            tr = task["train"] & ~pd.isna(y)
            ev = task["eval"] & ~pd.isna(y)
            assert not (ev & (d.split.to_numpy() == "test")).any()
            start = d.loc[ev, "date"].min()
            end = d.loc[ev, "date"].max()
            stations = set(d.loc[ev, "station_id"].unique())

            # --- D16
            spec = specs["d16_frozen"][hz]
            pipe = S.build(spec["family"], spec["cfg"], feats["FULL_V2"])
            pipe.fit(d.loc[tr, feats["FULL_V2"]], y[tr])
            p_tr = pipe.predict_proba(d.loc[tr, feats["FULL_V2"]])[:, 1]
            thr, _ = P29.best_exact(y[tr], p_tr, 10.0)
            p = pipe.predict_proba(d.loc[ev, feats["FULL_V2"]])[:, 1]
            pred = pd.DataFrame({"station_id": d.loc[ev, "station_id"].to_numpy(),
                                 "date": d.loc[ev, "date"].to_numpy(), "p": p})
            r = evaluate_cell(cell, events, pred, thr, h, stations, start, end)
            r.update({"cell": cell, "horizon": hz, "model": "d16_frozen", "threshold": thr,
                      "fit_s": None})
            rows.append(r)
            print(f"  {hz[-3:]} {cell:15s} d16  events={r['n_events']:3d} "
                  f"det={r['detected']:3d} rec={r['event_recall']:.3f} "
                  f"lead={r['median_lead_days']} FA={r['fa_episodes']:3d}", flush=True)

            # --- D7
            pipe = M10.d7_pipeline(None)
            pipe.fit(d.loc[tr, ["wl_pct_danger"]], y[tr])
            p_tr = pipe.predict_proba(d.loc[tr, ["wl_pct_danger"]])[:, 1]
            thr7, _ = P29.best_exact(y[tr], p_tr, 10.0)
            p7 = pipe.predict_proba(d.loc[ev, ["wl_pct_danger"]])[:, 1]
            pred7 = pd.DataFrame({"station_id": d.loc[ev, "station_id"].to_numpy(),
                                  "date": d.loc[ev, "date"].to_numpy(), "p": p7})
            r = evaluate_cell(cell, events, pred7, thr7, h, stations, start, end)
            r.update({"cell": cell, "horizon": hz, "model": "d7_persistence", "threshold": thr7,
                      "fit_s": None})
            rows.append(r)
            print(f"  {hz[-3:]} {cell:15s} d7   events={r['n_events']:3d} "
                  f"det={r['detected']:3d} rec={r['event_recall']:.3f} "
                  f"lead={r['median_lead_days']} FA={r['fa_episodes']:3d}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / ("smoke_event_metrics.csv" if not full else "event_metrics.csv"), index=False)
    if not full:
        print(df.to_string(index=False))
        return 0

    # summary + markdown
    def m(mean_df, model, hz, col):
        s = df[(df.model == model) & (df.horizon == hz)]
        return float(s[col].mean()) if len(s) else float("nan")

    summary = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "seed": B.SEED,
        "test_rows_read": 0,
        "cells": cells,
        "horizons": horizons,
        "n_events": {hz: int(df[(df.horizon == hz)].n_events.max()) for hz in horizons},
        "cell_event_pairs_eligible": {hz: int(df[(df.horizon == hz) & (df.model == "d16_frozen")].n_eligible.sum())
                                      for hz in horizons},
        "mean_event_recall": {m_: {hz: m(df, m_, hz, "event_recall") for hz in horizons}
                              for m_ in MODELS},
        "mean_fa_per_station_year": {m_: {hz: m(df, m_, hz, "fa_per_station_year") for hz in horizons}
                                     for m_ in MODELS},
        "mean_event_precision": {m_: {hz: m(df, m_, hz, "event_precision") for hz in horizons}
                                 for m_ in MODELS},
        "median_lead": {m_: {hz: m(df, m_, hz, "median_lead_days") for hz in horizons}
                        for m_ in MODELS},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Event-level evaluation — D16 vs D7 persistence", "",
             f"**Created:** {summary['created']} · **Seed:** `{B.SEED}` · "
             "**Test rows read:** 0", "",
             "Definitions are in `scripts/mi11_event_eval.py`; per-cell rows are in "
             "`reports/event_level/event_metrics.csv`.", ""]
    for hz in horizons:
        lines += [f"## {HZ[hz]}", "",
                  "| cell | events (eligible/total) | D16 detected | D16 recall | D16 median lead (d) | "
                  "D7 detected | D7 recall | D7 median lead (d) | D16 FA episodes | D7 FA episodes |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for cell in cells:
            a = df[(df.cell == cell) & (df.horizon == hz) & (df.model == "d16_frozen")].iloc[0]
            b = df[(df.cell == cell) & (df.horizon == hz) & (df.model == "d7_persistence")].iloc[0]
            lines.append(
                f"| `{cell}` | {a.n_eligible}/{a.n_events} | {a.detected} | {a.event_recall:.3f} | "
                f"{a.median_lead_days:.1f} | {b.detected} | {b.event_recall:.3f} | "
                f"{b.median_lead_days:.1f} | {a.fa_episodes} | {b.fa_episodes} |")
        lines += ["",
                  "Recall is over events with at least one observable issue day in the lead "
                  "window; events with no prior row are excluded (they are unverifiable, not "
                  "misses) and are shown in the events column.", "",
                  f"Mean event recall: D16 **{m(df,'d16_frozen',hz,'event_recall'):.3f}**, "
                  f"D7 {m(df,'d7_persistence',hz,'event_recall'):.3f}; "
                  f"mean false-alarm episodes per station-year: D16 "
                  f"{m(df,'d16_frozen',hz,'fa_per_station_year'):.2f}, D7 "
                  f"{m(df,'d7_persistence',hz,'fa_per_station_year'):.2f}.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWROTE {OUT}  wall={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
