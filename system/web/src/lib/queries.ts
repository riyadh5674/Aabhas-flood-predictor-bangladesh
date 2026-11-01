import { useQueries, useQuery } from "@tanstack/react-query";
import {
  fetchDates,
  fetchHealth,
  fetchModelInfo,
  fetchPredictions,
  fetchStationTrend,
  STATIC_SNAPSHOT,
  fetchThresholds,
} from "./api";
import { demoPredictions, demoSeries } from "./demo";
import { bandFor, FALLBACK_RISK } from "./risk";
import { loadStations } from "./stations";
import type {
  DataMode,
  Health,
  Horizon,
  ModelInfo,
  Prediction,
  PredictionResult,
  RiskConfig,
  Station,
  StationRisk,
} from "./types";

export const queryKeys = {
  stations: ["stations"] as const,
  dates: ["dates"] as const,
  health: ["health"] as const,
  risk: ["risk-config"] as const,
  model: ["model-info"] as const,
  predictions: (date: string, horizon: Horizon) => ["predictions", date, horizon] as const,
  trend: (id: number, horizon: Horizon) => ["trend", id, horizon] as const,
};

export function useStations() {
  return useQuery<Station[]>({
    queryKey: queryKeys.stations,
    queryFn: loadStations,
    staleTime: Infinity,
  });
}

export function useHealth() {
  return useQuery<Health>({
    queryKey: queryKeys.health,
    queryFn: () => fetchHealth(),
    refetchInterval: 45_000,
    retry: false,
    staleTime: 20_000,
  });
}

export function useDates() {
  return useQuery<{ dates: string[] }>({
    queryKey: queryKeys.dates,
    queryFn: () => fetchDates(),
    retry: 1,
    refetchInterval: 60_000,
    staleTime: 5 * 60_000,
  });
}

export function useRiskConfig() {
  return useQuery<RiskConfig>({
    queryKey: queryKeys.risk,
    queryFn: () => fetchThresholds(),
    retry: false,
    staleTime: 30 * 60_000,
    placeholderData: FALLBACK_RISK,
  });
}

export function useModelInfo() {
  return useQuery<ModelInfo>({
    queryKey: queryKeys.model,
    queryFn: () => fetchModelInfo(),
    retry: false,
    staleTime: 30 * 60_000,
  });
}

function toPrediction(raw: {
  station_id: number;
  date: string;
  horizon: Horizon;
  probability: number | null;
  band?: StationRisk["band"];
  action?: string;
  threshold?: number | null;
  model_version?: string;
  computed_at?: string;
}): Prediction {
  const horizon = raw.horizon;
  const probability = raw.probability;
  return {
    stationId: raw.station_id,
    date: raw.date,
    horizon,
    probability,
    band: raw.band ?? bandFor(probability, horizon, FALLBACK_RISK),
    action: raw.action ?? "",
    threshold: raw.threshold ?? FALLBACK_RISK.thresholds[horizon]?.t_warn ?? null,
    modelVersion: raw.model_version ?? "D16",
    computedAt: raw.computed_at ?? new Date().toISOString(),
  };
}

function join(rows: Prediction[], stations: Station[]): StationRisk[] {
  const byId = new Map(stations.map((s) => [s.id, s]));
  const out: StationRisk[] = [];
  for (const row of rows) {
    const station = byId.get(row.stationId);
    if (station) out.push({ ...row, station });
  }
  return out;
}

function demoResult(stations: Station[], date: string, horizon: Horizon): PredictionResult {
  const raw = demoPredictions(stations, date, horizon);
  const rows = join(raw.map(toPrediction), stations);
  return { rows, source: "demo", fetchedAt: Date.now() };
}

export function usePredictions(date: string, horizon: Horizon, mode: DataMode = "auto") {
  return useQuery<PredictionResult>({
    queryKey: [...queryKeys.predictions(date, horizon), mode],
    enabled: Boolean(date),
    staleTime: 60_000,
    retry: false,
    queryFn: async () => {
      const stations = await loadStations();
      if (mode === "demo") return demoResult(stations, date, horizon);

      const raw = await fetchPredictions(date, horizon);
      const rows = join(raw.map(toPrediction), stations);
      if (!rows.length) throw new Error("The API returned no predictions for this date and horizon.");
      return { rows, source: STATIC_SNAPSHOT ? "snapshot" as const : "live" as const, fetchedAt: Date.now() };
    },
  });
}

/** Runs the same date across all three horizons in parallel. */
export function useHorizonSet(date: string, mode: DataMode = "auto") {
  const horizons: Horizon[] = ["24h", "48h", "72h"];
  const results = useQueries({
    queries: horizons.map((horizon) => ({
      queryKey: [...queryKeys.predictions(date, horizon), mode],
      enabled: Boolean(date),
      staleTime: 60_000,
      retry: false,
      queryFn: async (): Promise<PredictionResult> => {
        const stations = await loadStations();
        if (mode === "demo") return demoResult(stations, date, horizon);
        const raw = await fetchPredictions(date, horizon);
        const rows = join(raw.map(toPrediction), stations);
        if (!rows.length) throw new Error("The API returned no predictions for this date and horizon.");
        return { rows, source: STATIC_SNAPSHOT ? "snapshot" as const : "live" as const, fetchedAt: Date.now() };
      },
    })),
  });

  return {
    "24h": results[0],
    "48h": results[1],
    "72h": results[2],
  } as Record<Horizon, (typeof results)[number]>;
}

export interface TrendPoint {
  date: string;
  probability: number | null;
  band: StationRisk["band"];
  action?: string;
  threshold?: number | null;
}

export function useStationTrend(
  station: Station | undefined,
  dates: string[],
  horizon: Horizon,
  mode: DataMode = "auto",
) {
  return useQuery<{ points: TrendPoint[]; source: "live" | "demo" | "snapshot" }>({
    queryKey: [...queryKeys.trend(station?.id ?? -1, horizon), dates[0], dates.at(-1), mode],
    enabled: Boolean(station && dates.length),
    staleTime: 5 * 60_000,
    retry: false,
    queryFn: async () => {
      if (!station) return { points: [], source: "demo" as const };
      if (mode === "demo") {
        return { points: demoSeries(station, dates, horizon), source: "demo" as const };
      }
      if (STATIC_SNAPSHOT) {
        const trend = await fetchStationTrend(station.id, horizon);
        const wanted = new Set(dates);
        return { points: trend.points.filter((point) => wanted.has(point.date)), source: "snapshot" as const };
      }
      const settled = await Promise.allSettled(dates.map((d) => fetchPredictions(d, horizon)));
      const points: TrendPoint[] = [];
      let failures = 0;
      settled.forEach((res, i) => {
        const date = dates[i];
        if (res.status !== "fulfilled") {
          failures += 1;
          points.push({ date, probability: null, band: "unknown" });
          return;
        }
        const row = res.value.find((r) => r.station_id === station.id);
        points.push({
          date,
          probability: row?.probability ?? null,
          band: row?.band ?? "unknown",
          action: row?.action ?? "",
          threshold: row?.threshold ?? null,
        });
      });
      if (points.every((p) => p.probability == null)) throw new Error("The API returned no live station history.");
      if (failures === dates.length) throw new Error("The API could not load station history.");
      return { points, source: "live" as const };
    },
  });
}
