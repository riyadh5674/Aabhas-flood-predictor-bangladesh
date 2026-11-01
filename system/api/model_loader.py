"""
System API — Model loader for frozen D16 artefact.

Loads `models/final_v2/selected_model_v2.pkl` with joblib. The pickle references
`baseline_model` (FreqEncoder) and a synthetic `decision` module (numpy.dtype),
both of which are shimmed here so the artefact loads outside the parent repo's
`scripts/` directory.

The model is frozen at D16 (MI-5, 2026-10-15). SHA-256 verification is performed
at load time if a hash is configured.
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from system.config.model_config import (
    MODEL_PATH,
    MODEL_SHA256,
    FEATURE_SET,
    N_FEATURES,
    SEED,
    HORIZON_TO_LABEL,
    THRESHOLDS,
    RISK_BANDS,
)

# ---------------------------------------------------------------------------
# Shim modules required by the pickled artefact
# ---------------------------------------------------------------------------

_decision = types.ModuleType("decision")
_decision.dtype = np.dtype
sys.modules.setdefault("decision", _decision)

# baseline_model must be importable; import it if available
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import baseline_model  # noqa: F401
except Exception:
    # baseline_model provides FreqEncoder used inside the pipelines.
    # If it cannot be imported, the pipelines will fail at predict time —
    # that's a hard error we surface clearly.
    pass


class ModelNotLoadedError(RuntimeError):
    """Raised when the model artefact cannot be loaded."""


class D16Model:
    """Wrapper around the frozen D16 model artefact.

    Provides:
      - predict_proba(station_features_df, horizon) -> np.ndarray of P(flood=1)
      - apply_threshold(probability, horizon) -> (risk_band, threshold_used)
      - model metadata (version, features, seed, etc.)
    """

    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self._artefact: dict | None = None
        self._pipelines: dict = {}
        self._features: list[str] = []
        self._loaded_at: datetime | None = None
        self._sha256: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._artefact is not None

    @property
    def version(self) -> str:
        return self._artefact.get("decision", "D16") if self._artefact else "D16"

    @property
    def seed(self) -> int:
        return self._artefact.get("seed", SEED) if self._artefact else SEED

    @property
    def features(self) -> list[str]:
        return list(self._features)

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def frozen_at_utc(self) -> str | None:
        return self._artefact.get("frozen_at_utc") if self._artefact else None

    @property
    def dataset_sha256_v2(self) -> str | None:
        return self._artefact.get("dataset_sha256_v2") if self._artefact else None

    def load(self, verify_sha256: bool = True) -> "D16Model":
        """Load the model artefact from disk.

        If verify_sha256 and MODEL_SHA256 is set, the file hash is checked
        before unpickling.
        """
        if not self.model_path.exists():
            raise ModelNotLoadedError(f"Model artefact not found: {self.model_path}")

        if verify_sha256 and MODEL_SHA256:
            self._sha256 = self._compute_sha256()
            if self._sha256 != MODEL_SHA256:
                raise ModelNotLoadedError(
                    f"SHA-256 mismatch for {self.model_path}: "
                    f"expected {MODEL_SHA256}, got {self._sha256}"
                )

        try:
            artefact = joblib.load(self.model_path)
        except Exception as exc:
            raise ModelNotLoadedError(
                f"Failed to load model artefact {self.model_path}: {exc}"
            ) from exc

        if not isinstance(artefact, dict):
            raise ModelNotLoadedError(
                f"Model artefact is not a dict (got {type(artefact).__name__})"
            )

        required = {"pipelines", "features"}
        missing = required - set(artefact.keys())
        if missing:
            raise ModelNotLoadedError(
                f"Model artefact missing required keys: {missing}"
            )

        self._artefact = artefact
        self._pipelines = artefact["pipelines"]
        self._features = list(artefact["features"])
        self._loaded_at = datetime.now(timezone.utc)

        # Validate feature count
        if len(self._features) != N_FEATURES:
            raise ModelNotLoadedError(
                f"Model expects {N_FEATURES} features, artefact has {len(self._features)}"
            )

        # Validate all pipelines exist for all horizons
        for horizon in HORIZON_TO_LABEL:
            label = HORIZON_TO_LABEL[horizon]
            if label not in self._pipelines:
                raise ModelNotLoadedError(
                    f"Model artefact missing pipeline for {label} ({horizon})"
                )

        return self

    def _compute_sha256(self) -> str:
        h = hashlib.sha256()
        with open(self.model_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def predict_proba(
        self,
        features_df,
        horizon: str,
    ):
        """Return P(flood=1) for each row in features_df for the given horizon.

        features_df must contain all FULL_V2 feature columns (in any order).
        Returns a 1-D numpy array of probabilities.
        """
        if not self.is_loaded:
            raise ModelNotLoadedError("Model not loaded. Call .load() first.")

        label = HORIZON_TO_LABEL[horizon]
        pipeline = self._pipelines[label]

        # Select and order features exactly as the model expects
        missing = [c for c in self._features if c not in features_df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = features_df[self._features]

        # predict_proba returns (n_rows, 2); column 1 is P(class=1)
        probs = pipeline.predict_proba(X)[:, 1]
        return probs

    def risk_band(
        self,
        probability: float,
        horizon: str,
    ) -> dict:
        """Map a probability to a risk band using Phase-29 cost-derived bands.

        Returns dict with band name, probability, threshold, and action.
        """
        bands = RISK_BANDS[horizon]["bands"]
        p = float(probability)

        for band in bands:
            p_lo = band["p_lo"]
            p_hi = band["p_hi"]
            lo_ok = p_lo is None or p >= p_lo
            hi_ok = p_hi is None or p < p_hi
            if lo_ok and hi_ok:
                return {
                    "band": band["name"],
                    "probability": p,
                    "threshold": THRESHOLDS[horizon]["t_warn"],
                    "action": band["action"],
                }

        # Fallback (should not happen with complete band coverage)
        return {
            "band": "low",
            "probability": p,
            "threshold": THRESHOLDS[horizon]["t_warn"],
            "action": "routine monitoring",
        }

    def predict(
        self,
        features_df,
        horizon: str,
    ):
        """Run inference: probabilities + risk bands.

        Returns a DataFrame with columns:
          station_id, date, horizon, probability, band, action, threshold,
          model_version, computed_at
        """
        import pandas as pd

        probs = self.predict_proba(features_df, horizon)
        threshold = float(THRESHOLDS[horizon]["t_warn"])
        computed_at = datetime.now(timezone.utc).isoformat()
        model_version = str(self.version)
        n_rows = len(features_df)

        if n_rows == 0:
            return pd.DataFrame(
                columns=[
                    "station_id",
                    "date",
                    "horizon",
                    "probability",
                    "band",
                    "action",
                    "threshold",
                    "model_version",
                    "computed_at",
                ]
            )

        # Extract columns once instead of building a Series per row with iloc.
        if "station_id" in features_df.columns:
            station_values = (
                pd.to_numeric(features_df["station_id"], errors="coerce")
                .fillna(0)
                .astype(int)
                .tolist()
            )
        else:
            station_values = [0] * n_rows

        if "date" in features_df.columns:
            date_values = (
                pd.to_datetime(features_df["date"], errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .fillna("")
                .tolist()
            )
        else:
            date_values = [""] * n_rows

        band_names: list[str] = []
        actions: list[str] = []
        for p in probs:
            band_info = self.risk_band(float(p), horizon)
            band_names.append(band_info["band"])
            actions.append(band_info["action"])

        return pd.DataFrame(
            {
                "station_id": station_values,
                "date": date_values,
                "horizon": horizon,
                "probability": [float(p) for p in probs],
                "band": band_names,
                "action": actions,
                "threshold": threshold,
                "model_version": model_version,
                "computed_at": computed_at,
            }
        )


__all__ = ["D16Model", "ModelNotLoadedError"]
