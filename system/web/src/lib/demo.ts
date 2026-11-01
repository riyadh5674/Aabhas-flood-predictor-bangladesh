/**
 * Deterministic demo dataset.
 *
 * Used only when the inference API is unreachable. It is built from the real
 * FFWC station master (real names, rivers, districts, coordinates) and a
 * seeded generator, so every reload shows the same forecast for the same date.
 * A monsoon seasonal curve and four spatial "surge" kernels give the national
 * picture a believable shape without ever being mistaken for model output —
 * the UI marks this data clearly as demo.
 */
import { bandFor, FALLBACK_RISK } from "./risk";
import type { Band, Horizon, Prediction, RawPrediction, Station } from "./types";

const DAY_MS = 86_400_000;

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function random() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(value: string): number {
  let h = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function isoDay(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
    d.getDate(),
  ).padStart(2, "0")}`;
}

export function demoToday(): string {
  return isoDay(new Date());
}

export function demoDates(count = 180): string[] {
  const out: string[] = [];
  const today = new Date();
  today.setHours(12, 0, 0, 0);
  for (let i = count - 1; i >= 0; i -= 1) {
    out.push(isoDay(new Date(today.getTime() - i * DAY_MS)));
  }
  return out;
}

/** Monsoon weight: high June–October, low in the dry season. */
function monsoon(dateIso: string): number {
  const d = new Date(`${dateIso}T12:00:00`);
  const doy = Math.floor((d.getTime() - new Date(d.getFullYear(), 0, 0).getTime()) / DAY_MS);
  const phase = ((doy - 227) / 365) * Math.PI * 2;
  return Math.max(0.14, 0.55 + 0.45 * Math.cos(phase));
}

interface Kernel {
  lat: number;
  lon: number;
  radius: number;
  weight: number;
}

// Surge kernels over the four basins the model cares about most.
const KERNELS: Kernel[] = [
  { lat: 24.95, lon: 91.35, radius: 0.95, weight: 0.62 }, // Surma–Kushiyara
  { lat: 22.95, lon: 91.55, radius: 0.8, weight: 0.4 }, // Muhuri–Feni coast
  { lat: 23.78, lon: 89.65, radius: 0.85, weight: 0.34 }, // Padma–Jamuna
  { lat: 25.85, lon: 89.7, radius: 0.75, weight: 0.26 }, // Brahmaputra
];

function surge(station: Station, dateIso: string): number {
  let sum = 0;
  for (const k of KERNELS) {
    const dy = station.lat - k.lat;
    const dx = (station.lon - k.lon) * Math.cos((station.lat * Math.PI) / 180);
    const dist = Math.sqrt(dx * dx + dy * dy);
    sum += k.weight * Math.exp(-((dist / k.radius) ** 2));
  }
  const d = new Date(`${dateIso}T12:00:00`);
  const doy = Math.floor((d.getTime() - new Date(d.getFullYear(), 0, 0).getTime()) / DAY_MS);
  const pulse = 0.82 + 0.18 * Math.sin(doy * 0.42) + 0.08 * Math.sin(doy * 1.7);
  return Math.min(1.15, sum * monsoon(dateIso) * pulse);
}

function probabilityFor(station: Station, dateIso: string, horizon: Horizon): number {
  const seed = hashString(`${station.id}|${dateIso}|${horizon}`);
  const rnd = mulberry32(seed);
  const u = rnd();
  const v = rnd();
  const s = surge(station, dateIso);

  // Horizon widens the distribution: longer lead time, more uncertainty.
  const horizonScale = horizon === "24h" ? 1 : horizon === "48h" ? 1.08 : 1.16;

  const baseline = 0.008 + 0.055 * u ** 3;
  const surgeTerm = s * 0.1;
  const spike = s > 0.5 ? ((s - 0.5) / 0.65) * 0.62 * v ** 0.75 : 0;
  const raw = (baseline + surgeTerm + spike) * horizonScale;
  return Math.max(0.0004, Math.min(0.97, raw));
}

export function demoPredictions(
  stations: Station[],
  dateIso: string,
  horizon: Horizon,
): RawPrediction[] {
  return stations.map((station) => {
    const p = probabilityFor(station, dateIso, horizon);
    const band: Band = bandFor(p, horizon, FALLBACK_RISK);
    const def = FALLBACK_RISK.risk_bands[horizon].bands.find((b) => b.name === band);
    return {
      station_id: station.id,
      date: dateIso,
      horizon,
      probability: Number(p.toFixed(6)),
      band,
      action: def?.action ?? "routine monitoring",
      threshold: FALLBACK_RISK.thresholds[horizon].t_warn,
      model_version: "D16 (demo)",
      computed_at: new Date().toISOString(),
    };
  });
}

/** Probability series for a single station, newest last. */
export function demoSeries(
  station: Station,
  dates: string[],
  horizon: Horizon,
): Prediction[] {
  return dates.map((dateIso) => {
    const p = probabilityFor(station, dateIso, horizon);
    const band = bandFor(p, horizon, FALLBACK_RISK);
    const def = FALLBACK_RISK.risk_bands[horizon].bands.find((b) => b.name === band);
    return {
      stationId: station.id,
      date: dateIso,
      horizon,
      probability: Number(p.toFixed(6)),
      band,
      action: def?.action ?? "routine monitoring",
      threshold: FALLBACK_RISK.thresholds[horizon].t_warn,
      modelVersion: "D16 (demo)",
      computedAt: new Date().toISOString(),
    };
  });
}
