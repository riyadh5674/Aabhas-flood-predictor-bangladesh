/**
 * Tiny typed fetch layer for the Aabhas inference API.
 *
 * The browser never talks to :8000 directly — requests go to `/api`, which the
 * Vite dev server (or a production reverse proxy) forwards to FastAPI. That
 * keeps the API free of CORS configuration.
 */
import type { Health, Horizon, ModelInfo, RawPrediction, RiskConfig } from "./types";

export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";
export const STATIC_SNAPSHOT = import.meta.env.VITE_AABHAS_STATIC_SNAPSHOT === "true";
const SNAPSHOT_BASE = `${import.meta.env.BASE_URL}data/snapshot`;

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

interface RequestOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
}

async function getJson<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const { timeoutMs = 8000, signal } = opts;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const onAbort = () => controller.abort();
  signal?.addEventListener("abort", onAbort, { once: true });

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new ApiError(`API ${path} responded ${res.status}`, res.status);
    }
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    const message = err instanceof Error ? err.message : String(err);
    throw new ApiError(message);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}

export function fetchHealth(opts?: RequestOptions) {
  if (STATIC_SNAPSHOT) return getStatic<Health>("health.json", opts);
  return getJson<Health>("/health", { timeoutMs: 4000, ...opts });
}

export function fetchDates(opts?: RequestOptions) {
  if (STATIC_SNAPSHOT) return getStatic<{ dates: string[] }>("dates.json", opts);
  return getJson<{ dates: string[] }>("/dates", { timeoutMs: 8000, ...opts });
}

export function fetchPredictions(date: string, horizon: Horizon, opts?: RequestOptions) {
  if (STATIC_SNAPSHOT) return getStatic<Record<Horizon, RawPrediction[]>>(`predictions/${date}.json`, opts).then((all) => all[horizon]);
  return getJson<RawPrediction[]>(
    `/predict/all?date=${encodeURIComponent(date)}&horizon=${encodeURIComponent(horizon)}`,
    { timeoutMs: 12000, ...opts },
  );
}

export function fetchThresholds(opts?: RequestOptions) {
  if (STATIC_SNAPSHOT) return getStatic<RiskConfig>("thresholds.json", opts);
  return getJson<RiskConfig>("/thresholds", { timeoutMs: 5000, ...opts });
}

export function fetchModelInfo(opts?: RequestOptions) {
  if (STATIC_SNAPSHOT) return getStatic<ModelInfo>("model.json", opts);
  return getJson<ModelInfo>("/model/info", { timeoutMs: 5000, ...opts });
}

export function fetchStationTrend(stationId: number, horizon: Horizon, opts?: RequestOptions) {
  return getStatic<{ points: Array<{ date: string; probability: number | null; band: "low" | "moderate" | "high" | "very_high" | "unknown"; action?: string; threshold?: number | null }> }>(
    `trends/${horizon}/${stationId}.json`, opts,
  );
}

async function getStatic<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const res = await fetch(`${SNAPSHOT_BASE}/${path}`, { headers: { Accept: "application/json" }, signal: opts.signal });
  if (!res.ok) throw new ApiError(`Snapshot ${path} responded ${res.status}`, res.status);
  return (await res.json()) as T;
}

