"""
System Ingestion — Provenance manifest management.

Records every (station_id, query_date, source) API call with status and row count,
making ingestion resumable and auditable. Mirrors the parent repo's
data/raw/ffwc/harvest_manifest.csv pattern but extended for all sources.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_COLUMNS = [
    "station_id",
    "query_date",
    "source",
    "rows_fetched",
    "fetched_at",
    "status",
]


class ProvenanceManifest:
    """Thread-unsafe manifest for single-process ingestion runs."""

    def __init__(self, path: str | Path, columns: Optional[list[str]] = None):
        self.path = Path(path)
        self.columns = columns or DEFAULT_COLUMNS
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[tuple] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            df = pd.read_csv(self.path, dtype={"query_date": str})
            for _, row in df.iterrows():
                key = (
                    int(row["station_id"]),
                    str(row["query_date"]),
                    str(row["source"]),
                )
                self._done.add(key)
        except (pd.errors.EmptyDataError, FileNotFoundError, KeyError):
            self._done = set()

    def is_done(self, station_id: int, query_date: str, source: str) -> bool:
        return (int(station_id), str(query_date), str(source)) in self._done

    def record(
        self,
        station_id: int,
        query_date: str,
        source: str,
        rows_fetched: int,
        status: str = "ok",
    ) -> None:
        key = (int(station_id), str(query_date), str(source))
        if key in self._done:
            return
        self._done.add(key)
        self._append_row(
            station_id=int(station_id),
            query_date=str(query_date),
            source=str(source),
            rows_fetched=int(rows_fetched),
            fetched_at=datetime.now(timezone.utc).isoformat(),
            status=str(status),
        )

    def _append_row(self, **kwargs) -> None:
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.columns)
            if write_header:
                writer.writeheader()
            row = {c: kwargs.get(c, "") for c in self.columns}
            writer.writerow(row)

    def flush(self) -> None:
        """No-op for append-mode manifest; kept for interface symmetry."""
        pass

    def done_count(self) -> int:
        return len(self._done)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=self.columns)
        return pd.read_csv(self.path, dtype={"query_date": str})

    def summary(self) -> dict:
        df = self.to_dataframe()
        if df.empty:
            return {"total": 0, "by_source": {}, "by_status": {}}
        return {
            "total": len(df),
            "by_source": df["source"].value_counts().to_dict() if "source" in df.columns else {},
            "by_status": df["status"].value_counts().to_dict() if "status" in df.columns else {},
        }

__all__ = ["ProvenanceManifest", "DEFAULT_COLUMNS"]
