import type { Band, BandDef, Horizon, RiskConfig } from "./types";

/**
 * Fallback risk configuration.
 *
 * Mirrors system/config/model_config.json exactly. The app prefers the live
 * `/thresholds` endpoint and only falls back to this copy when the API is
 * unreachable, so the numbers stay in sync with the frozen artefact.
 */
export const FALLBACK_RISK: RiskConfig = {
  cost_ratio: "C_FN=10, C_FP=1",
  period: "val_2022 all-season",
  thresholds: {
    "24h": {
      model: "lightgbm",
      t_warn: 0.0605,
      recall: 0.98,
      precision: 0.746,
      f1: 0.847,
      far: 0.254,
      alarm_rate: 0.067,
      missed_rows: 18,
      cost_per_1k: 27.25,
    },
    "48h": {
      model: "lightgbm",
      t_warn: 0.0868,
      recall: 0.967,
      precision: 0.727,
      f1: 0.83,
      far: 0.273,
      alarm_rate: 0.079,
      missed_rows: 34,
      cost_per_1k: 41.16,
    },
    "72h": {
      model: "xgboost",
      t_warn: 0.0963,
      recall: 0.954,
      precision: 0.684,
      f1: 0.796,
      far: 0.316,
      alarm_rate: 0.092,
      missed_rows: 52,
      cost_per_1k: 59.45,
    },
  },
  risk_bands: {
    "24h": {
      model: "lightgbm",
      bands: [
        { name: "low", p_lo: null, p_hi: 0.0303, action: "routine monitoring" },
        { name: "moderate", p_lo: 0.0303, p_hi: 0.0605, action: "watch" },
        { name: "high", p_lo: 0.0605, p_hi: 0.434, action: "warning" },
        { name: "very_high", p_lo: 0.434, p_hi: null, action: "high-risk warning / activate" },
      ],
    },
    "48h": {
      model: "lightgbm",
      bands: [
        { name: "low", p_lo: null, p_hi: 0.0868, action: "routine monitoring" },
        { name: "moderate", p_lo: 0.0868, p_hi: 0.1736, action: "watch" },
        { name: "high", p_lo: 0.1736, p_hi: null, action: "warning" },
      ],
    },
    "72h": {
      model: "xgboost",
      bands: [
        { name: "low", p_lo: null, p_hi: 0.0963, action: "routine monitoring" },
        { name: "moderate", p_lo: 0.0963, p_hi: 0.1926, action: "watch" },
        { name: "high", p_lo: 0.1926, p_hi: null, action: "warning" },
      ],
    },
  },
};

export const BAND_ORDER: Band[] = ["very_high", "high", "moderate", "low", "unknown"];

export const SEVERITY: Record<Band, number> = {
  very_high: 3,
  high: 2,
  moderate: 1,
  low: 0,
  unknown: -1,
};

interface BandMeta {
  label: string;
  short: string;
  color: string;
  /** Plain-language guidance for the public. */
  guidance: string;
  /** What an operator should do. */
  operational: string;
  alertLevel: "none" | "watch" | "warning" | "high";
}

export const BAND_META: Record<Band, BandMeta> = {
  very_high: {
    label: "Very high",
    short: "VH",
    color: "var(--band-very-high)",
    guidance:
      "Flooding is likely within the forecast window. Move valuables and livestock to higher ground, keep documents and medicines dry, and follow local evacuation instructions.",
    operational:
      "Activate the high-risk protocol, notify the duty officer, and issue a public advisory for the affected unions.",
    alertLevel: "high",
  },
  high: {
    label: "High",
    short: "H",
    color: "var(--band-high)",
    guidance:
      "Conditions are favourable for flooding. Prepare now: check your escape route, charge phones, and store drinking water and dry food.",
    operational:
      "Issue a warning, verify the upstream gauge trend, and pre-position response capacity in the affected districts.",
    alertLevel: "warning",
  },
  moderate: {
    label: "Moderate",
    short: "M",
    color: "var(--band-moderate)",
    guidance:
      "Some risk is present. Stay informed and re-check the forecast before travelling near rivers or low-lying roads.",
    operational: "Place the station on watch and re-evaluate at the next model cycle.",
    alertLevel: "watch",
  },
  low: {
    label: "Low",
    short: "L",
    color: "var(--band-low)",
    guidance:
      "No significant flood signal in this window. Normal seasonal precautions still apply.",
    operational: "Routine monitoring.",
    alertLevel: "none",
  },
  unknown: {
    label: "No signal",
    short: "—",
    color: "var(--band-none)",
    guidance:
      "There is not enough recent sensor data to produce a forecast for this station. Do not read this as 'safe'.",
    operational: "Check ingestion health for this station and the upstream gauges.",
    alertLevel: "none",
  },
};

export function bandFor(probability: number | null, horizon: Horizon, config: RiskConfig): Band {
  if (probability == null || Number.isNaN(probability)) return "unknown";
  const bands = config.risk_bands[horizon]?.bands ?? FALLBACK_RISK.risk_bands[horizon].bands;
  for (const band of bands) {
    const loOk = band.p_lo == null || probability >= band.p_lo;
    const hiOk = band.p_hi == null || probability < band.p_hi;
    if (loOk && hiOk) return band.name;
  }
  return "low";
}

export function bandDefFor(band: Band, horizon: Horizon, config: RiskConfig): BandDef | undefined {
  return config.risk_bands[horizon]?.bands.find((b) => b.name === band);
}

export function isElevated(band: Band): boolean {
  return band === "high" || band === "very_high";
}

export function worstBand(bands: Band[]): Band {
  if (!bands.length) return "unknown";
  return bands.reduce<Band>((worst, b) => (SEVERITY[b] > SEVERITY[worst] ? b : worst), "unknown");
}

/** Position of a probability on the 0..1 gauge, clamped. */
export function gaugeFraction(probability: number | null): number {
  if (probability == null) return 0;
  return Math.max(0, Math.min(1, probability));
}
