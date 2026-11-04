import { describe, expect, it } from "vitest";
import {
  aggregateDivisions,
  aggregateRegions,
  bandCounts,
  nationalSummary,
  regionKey,
} from "./aggregate";
import type { Band, Station, StationRisk } from "./types";

function station(id: number, overrides: Partial<Station> = {}): Station {
  return {
    id,
    name: `Station ${id}`,
    river: "River",
    basin: "Basin",
    division: "Dhaka",
    divisionKey: "Dhaka",
    district: "Dhaka",
    districtKey: "Dhaka",
    upazilla: "Sadar",
    union: "Union",
    lat: 23.8,
    lon: 90.4,
    dangerLevel: 5,
    recordHigh: 7,
    ...overrides,
  };
}

function row(id: number, probability: number | null, band: Band, overrides: Partial<Station> = {}): StationRisk {
  return {
    stationId: id,
    date: "2026-11-02",
    horizon: "24h",
    probability,
    band,
    action: "",
    threshold: null,
    modelVersion: "D16",
    computedAt: "2026-11-02T00:00:00Z",
    station: station(id, overrides),
  };
}

const rows: StationRisk[] = [
  row(1, 0.2, "high", { divisionKey: "Dhaka", districtKey: "Dhaka" }),
  row(2, 0.1, "moderate", { divisionKey: "Dhaka", districtKey: "Gazipur", district: "Gazipur" }),
  row(3, null, "unknown", { divisionKey: "Khulna", districtKey: "Khulna", division: "Khulna", district: "Khulna" }),
  row(4, 0.6, "very_high", { divisionKey: "Khulna", districtKey: "Khulna", division: "Khulna", district: "Khulna" }),
];

describe("aggregate", () => {
  it("chooses region keys by map level", () => {
    expect(regionKey(rows[0], "adm1")).toBe("Dhaka");
    expect(regionKey(rows[0], "adm2")).toBe("Dhaka");
  });

  it("aggregates districts, reporting counts and the worst band", () => {
    const regions = aggregateRegions(rows, "adm2");
    expect(regions.get("Dhaka")).toMatchObject({
      worst: "high",
      maxProbability: 0.2,
      reporting: 1,
      elevated: 1,
    });
    expect(regions.get("Khulna")).toMatchObject({
      worst: "very_high",
      maxProbability: 0.6,
      reporting: 1,
      elevated: 1,
    });
    expect(regions.get("Khulna")?.meanProbability).toBeCloseTo(0.6);
  });

  it("ranks divisions by severity and probability", () => {
    const divisions = aggregateDivisions(rows);
    expect(divisions.map((division) => division.name)).toEqual(["Khulna", "Dhaka"]);
    expect(divisions[0]).toMatchObject({
      worst: "very_high",
      stations: 2,
      reporting: 1,
      elevated: 1,
    });
    expect(divisions[1]?.meanProbability).toBeCloseTo(0.15);
  });

  it("counts every band even when absent", () => {
    expect(bandCounts(rows)).toEqual({
      very_high: 1,
      high: 1,
      moderate: 1,
      low: 0,
      unknown: 1,
    });
  });

  it("summarises only stations with a probability", () => {
    expect(nationalSummary(rows)).toMatchObject({
      stations: 4,
      reporting: 3,
      maxProbability: 0.6,
      elevated: 2,
      worst: "very_high",
    });
    expect(nationalSummary(rows).meanProbability).toBeCloseTo(0.3);
    expect(nationalSummary([]).maxProbability).toBeNull();
  });
});
