import { describe, expect, it } from "vitest";
import {
  calendarDaysBetween,
  evaluateFreshness,
  parseDateOnly,
} from "./freshness";

describe("freshness", () => {
  it("parses date-only strings as UTC calendar days", () => {
    expect(parseDateOnly("2026-11-02")).toBe(Date.UTC(2026, 10, 2));
    expect(parseDateOnly("2026-02-30")).toBe(Date.UTC(2026, 2, 2));
    expect(parseDateOnly("")).toBeNull();
    expect(parseDateOnly("2026/11/02")).toBeNull();
    expect(parseDateOnly("not-a-date")).toBeNull();
  });

  it("computes whole calendar-day differences without timezone drift", () => {
    expect(calendarDaysBetween("2026-10-30", "2026-11-02")).toBe(3);
    expect(calendarDaysBetween("2026-11-02", "2026-10-30")).toBe(-3);
    expect(calendarDaysBetween("2026-10-30", null)).toBeNull();
  });

  it("treats yesterday and today as current", () => {
    const now = new Date("2026-11-02T23:59:00Z");
    expect(evaluateFreshness("2026-11-02", now)).toMatchObject({
      level: "current",
      daysBehind: 0,
      warn: false,
    });
    expect(evaluateFreshness("2026-11-01", now)).toMatchObject({
      level: "current",
      daysBehind: 1,
      warn: false,
    });
  });

  it("warns during the aging window", () => {
    const now = new Date("2026-11-02T12:00:00Z");
    for (const days of [2, 3]) {
      const latest = new Date(Date.UTC(2026, 10, 2 - days)).toISOString().slice(0, 10);
      expect(evaluateFreshness(latest, now)).toMatchObject({
        level: "aging",
        daysBehind: days,
        warn: true,
      });
    }
  });

  it("escalates to stale and expired", () => {
    const now = new Date("2026-11-02T12:00:00Z");
    expect(evaluateFreshness("2026-10-29", now)).toMatchObject({
      level: "stale",
      daysBehind: 4,
      warn: true,
    });
    expect(evaluateFreshness("2026-10-20", now)).toMatchObject({
      level: "expired",
      daysBehind: 13,
      warn: true,
    });
  });

  it("fails closed when the API does not report a date", () => {
    expect(evaluateFreshness(null)).toMatchObject({
      level: "unknown",
      daysBehind: null,
      warn: true,
    });
    expect(evaluateFreshness("garbage")).toMatchObject({
      level: "unknown",
      warn: true,
    });
  });
});
