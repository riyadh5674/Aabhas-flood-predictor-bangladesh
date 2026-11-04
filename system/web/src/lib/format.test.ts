import { describe, expect, it } from "vitest";
import {
  clockTime,
  dec,
  dayMonth,
  int,
  longDate,
  meters,
  pct,
  pctPoints,
  relativeTime,
  shortDate,
  slugify,
  titleCase,
  truncate,
  weekday,
} from "./format";

describe("format", () => {
  it("formats numeric values and missing values consistently", () => {
    expect(pct(0.1234, 1)).toBe("12.3%");
    expect(pct(null)).toBe("—");
    expect(pctPoints(0.0605)).toBe("6.05");
    expect(pctPoints(Number.NaN)).toBe("—");
    expect(int(1234.6)).toBe("1,235");
    expect(int(null)).toBe("—");
    expect(dec(1.2)).toBe("1.20");
    expect(dec(undefined)).toBe("—");
    expect(meters(3.456)).toBe("3.46 m");
    expect(meters(null)).toBe("—");
  });

  it("survives invalid dates instead of throwing", () => {
    expect(shortDate("")).toBe("");
    expect(shortDate("not-a-date")).toBe("not-a-date");
    expect(longDate("")).toBe("");
    expect(weekday("")).toBe("");
    expect(dayMonth("")).toBe("");
    expect(clockTime("not-a-date")).toBe("—");
  });

  it("formats valid dates", () => {
    expect(shortDate("2026-11-02")).toBe("2 Nov 2026");
    expect(longDate("2026-11-02")).toBe("2 November 2026");
    expect(weekday("2026-11-02")).toBe("Mon");
    expect(dayMonth("2026-11-02")).toBe("2 Nov");
  });

  it("formats relative time in useful units", () => {
    const now = Date.UTC(2026, 10, 2, 12, 0, 0);
    expect(relativeTime(now - 10_000, now)).toBe("just now");
    expect(relativeTime(now - 5 * 60_000, now)).toBe("5 min ago");
    expect(relativeTime(now - 3 * 60 * 60_000, now)).toBe("3 h ago");
    expect(relativeTime(now - 2 * 24 * 60 * 60_000, now)).toBe("2 d ago");
  });

  it("normalises labels and slugs", () => {
    expect(titleCase("very_high river-level")).toBe("Very High River Level");
    expect(slugify("Kushiyara River (Fulbari)")).toBe("kushiyara-river-fulbari");
    expect(truncate("abcdef", 5)).toBe("abcd…");
    expect(truncate("abc", 5)).toBe("abc");
  });
});
