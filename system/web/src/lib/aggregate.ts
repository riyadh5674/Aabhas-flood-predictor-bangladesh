import { SEVERITY, worstBand } from "./risk";
import type { Band, MapLevel, StationRisk } from "./types";

export interface RegionAgg {
  name: string;
  worst: Band;
  maxProbability: number | null;
  meanProbability: number | null;
  stations: StationRisk[];
  elevated: number;
  reporting: number;
}

export function regionKey(row: StationRisk, level: MapLevel): string {
  return level === "adm1" ? row.station.divisionKey : row.station.districtKey;
}

export function aggregateRegions(rows: StationRisk[], level: MapLevel): Map<string, RegionAgg> {
  const map = new Map<string, RegionAgg>();
  for (const row of rows) {
    const key = regionKey(row, level);
    if (!key) continue;
    let agg = map.get(key);
    if (!agg) {
      agg = {
        name: key,
        worst: "unknown",
        maxProbability: null,
        meanProbability: null,
        stations: [],
        elevated: 0,
        reporting: 0,
      };
      map.set(key, agg);
    }
    agg.stations.push(row);
    if (row.probability != null) {
      agg.reporting += 1;
      agg.maxProbability =
        agg.maxProbability == null ? row.probability : Math.max(agg.maxProbability, row.probability);
    }
    if (row.band === "high" || row.band === "very_high") agg.elevated += 1;
    if (SEVERITY[row.band] > SEVERITY[agg.worst]) agg.worst = row.band;
  }
  for (const agg of map.values()) {
    const probs = agg.stations.map((s) => s.probability).filter((p): p is number => p != null);
    agg.meanProbability = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : null;
  }
  return map;
}

export interface DivisionAgg {
  name: string;
  worst: Band;
  maxProbability: number | null;
  meanProbability: number | null;
  stations: number;
  elevated: number;
  reporting: number;
}

export function aggregateDivisions(rows: StationRisk[]): DivisionAgg[] {
  const map = new Map<string, DivisionAgg>();
  for (const row of rows) {
    const key = row.station.divisionKey || "Unassigned";
    let agg = map.get(key);
    if (!agg) {
      agg = {
        name: key,
        worst: "unknown",
        maxProbability: null,
        meanProbability: null,
        stations: 0,
        elevated: 0,
        reporting: 0,
      };
      map.set(key, agg);
    }
    agg.stations += 1;
    if (row.probability != null) {
      agg.reporting += 1;
      agg.maxProbability =
        agg.maxProbability == null ? row.probability : Math.max(agg.maxProbability, row.probability);
    }
    if (row.band === "high" || row.band === "very_high") agg.elevated += 1;
    if (SEVERITY[row.band] > SEVERITY[agg.worst]) agg.worst = row.band;
  }
  const out = [...map.values()];
  for (const agg of out) {
    const probs = rows
      .filter((r) => (r.station.divisionKey || "Unassigned") === agg.name)
      .map((r) => r.probability)
      .filter((p): p is number => p != null);
    agg.meanProbability = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : null;
  }
  return out.sort((a, b) => {
    if (SEVERITY[b.worst] !== SEVERITY[a.worst]) return SEVERITY[b.worst] - SEVERITY[a.worst];
    return (b.maxProbability ?? 0) - (a.maxProbability ?? 0);
  });
}

export function bandCounts(rows: StationRisk[]): Record<Band, number> {
  const counts: Record<Band, number> = {
    very_high: 0,
    high: 0,
    moderate: 0,
    low: 0,
    unknown: 0,
  };
  for (const row of rows) counts[row.band] += 1;
  return counts;
}

export function nationalSummary(rows: StationRisk[]) {
  const reporting = rows.filter((r) => r.probability != null);
  const probs = reporting.map((r) => r.probability as number);
  const elevated = rows.filter((r) => r.band === "high" || r.band === "very_high");
  const worst = worstBand(rows.map((r) => r.band));
  return {
    stations: rows.length,
    reporting: reporting.length,
    maxProbability: probs.length ? Math.max(...probs) : null,
    meanProbability: probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : null,
    elevated: elevated.length,
    worst,
  };
}
