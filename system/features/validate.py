"""Feature quality gate.

The frozen D16 model uses GloFAS discharge and NASA POWER soil wetness directly.
If those sources are stale, the feature row still builds, but the predictions are
degraded. This module makes that failure explicit: it measures per-group null
shares for every output date, writes a report, and blocks the build unless the
caller explicitly accepts a degraded run.

Usage (called from ``system.features.build``):
    validate_features(df, allow_degraded=False)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(os.getenv("AABHAS_ROOT", Path(__file__).resolve().parents[2])).resolve()
QUALITY_DIR = ROOT / "system" / "data" / "processed" / "quality"

# Groups that feed the frozen model and must not be mostly null.
CRITICAL_GROUPS = {
    "glofas": ("gf_",),
    "soil": ("ps_",),
    "interactions": ("ix_",),
    "temperature": ("tmax_c", "tmin_c"),
}
# Informational groups: reported, but allowed to be sparse for structural reasons.
WATCH_GROUPS = {
    "hydrology": ("wl_",),
    "rainfall": ("rain_",),
    "upstream": ("up_",),
}
MAX_NULL_SHARE = 0.05


def _group_share(group: pd.DataFrame, selectors: tuple[str, ...]) -> float | None:
    cols = [c for c in group.columns if any(c.startswith(s) for s in selectors)]
    if not cols:
        return None
    return float(group[cols].isna().mean().mean())


def validate_features(df: pd.DataFrame, allow_degraded: bool = False) -> dict:
    if df.empty:
        raise SystemExit("Feature quality gate: empty feature frame")

    date_rows = []
    for day, group in df.groupby(df["date"].dt.date):
        critical = {
            name: _group_share(group, selectors)
            for name, selectors in CRITICAL_GROUPS.items()
        }
        watch = {
            name: _group_share(group, selectors)
            for name, selectors in WATCH_GROUPS.items()
        }
        critical = {k: v for k, v in critical.items() if v is not None}
        watch = {k: v for k, v in watch.items() if v is not None}
        all_null = [c for c in group.columns if group[c].isna().all()]

        failures = [
            f"{name}={share:.1%}"
            for name, share in critical.items()
            if share > MAX_NULL_SHARE
        ]
        passed = not failures and not all_null

        date_rows.append(
            {
                "date": day.isoformat(),
                "rows": int(len(group)),
                "stations": int(group["station_id"].nunique()),
                "critical_null_share": critical,
                "watch_null_share": watch,
                "all_null_columns": all_null,
                "passed": passed,
                "failures": failures,
            }
        )

    payload = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "max_null_share": MAX_NULL_SHARE,
        "ok": all(row["passed"] for row in date_rows),
        "dates": date_rows,
    }

    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    (QUALITY_DIR / "latest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    for row in date_rows:
        (QUALITY_DIR / f"{row['date']}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )

    if not payload["ok"]:
        bad = [row for row in date_rows if not row["passed"]]
        detail = "; ".join(
            f"{row['date']}: {', '.join(row['failures']) or 'all-null column'}"
            for row in bad[:5]
        )
        message = (
            f"Feature quality gate failed for {len(bad)} date(s) "
            f"(threshold {MAX_NULL_SHARE:.0%}): {detail}"
        )
        if allow_degraded:
            print(f"WARNING: {message}")
        else:
            raise SystemExit(
                message
                + ". Refresh the sources or re-run with --allow-degraded to publish anyway."
            )
    else:
        print(
            f"Feature quality gate passed for {len(date_rows)} date(s) "
            f"(critical groups <= {MAX_NULL_SHARE:.0%} null)"
        )

    return payload
