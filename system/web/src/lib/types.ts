export type Horizon = "24h" | "48h" | "72h";
export type Band = "low" | "moderate" | "high" | "very_high" | "unknown";
export type MapLevel = "adm1" | "adm2";
export type DataMode = "auto" | "demo";
export type Connection = "checking" | "live" | "offline";

export const HORIZONS: Horizon[] = ["24h", "48h", "72h"];

export interface Station {
  id: number;
  name: string;
  river: string;
  basin: string;
  division: string;
  divisionKey: string;
  district: string;
  districtKey: string;
  upazilla: string;
  union: string;
  lat: number;
  lon: number;
  dangerLevel: number | null;
  recordHigh: number | null;
}

/** Raw prediction record as returned by the FastAPI service. */
export interface RawPrediction {
  station_id: number;
  date: string;
  horizon: Horizon;
  probability: number | null;
  band: Band;
  action: string;
  threshold: number | null;
  model_version: string;
  computed_at: string;
  error?: string | null;
}

export interface Prediction {
  stationId: number;
  date: string;
  horizon: Horizon;
  probability: number | null;
  band: Band;
  action: string;
  threshold: number | null;
  modelVersion: string;
  computedAt: string;
}

/** A prediction joined to its station master record. */
export interface StationRisk extends Prediction {
  station: Station;
}

export interface Health {
  status: string;
  model_loaded: boolean;
  model_version: string | null;
  feature_store_dates: number;
  latest_feature_date: string | null;
  timestamp: string;
  /** Round-trip time measured by the client, if it performed the request. */
  latencyMs?: number;
}

export interface ThresholdRow {
  model: string;
  t_warn: number;
  recall: number;
  precision: number;
  f1: number;
  far: number;
  alarm_rate: number;
  missed_rows: number;
  cost_per_1k: number;
}

export interface BandDef {
  name: Band;
  p_lo: number | null;
  p_hi: number | null;
  action: string;
}

export interface RiskConfig {
  thresholds: Record<Horizon, ThresholdRow>;
  risk_bands: Record<Horizon, { model: string; bands: BandDef[] }>;
  cost_ratio: string;
  period: string;
}

export interface ModelInfo {
  version: string;
  decision: string;
  seed: number;
  feature_set: string | null;
  n_features: number;
  frozen_at_utc: string | null;
  dataset_sha256_v2: string | null;
  per_horizon_family: Record<string, string> | null;
  horizons: string[];
  loaded_at: string | null;
}

export interface PredictionResult {
  rows: StationRisk[];
  source: "live" | "demo" | "snapshot";
  fetchedAt: number;
}
