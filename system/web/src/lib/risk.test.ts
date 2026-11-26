import { describe, expect, it } from "vitest";
import {
  bandFor,
  BAND_ORDER,
  FALLBACK_RISK,
  gaugeFraction,
  isElevated,
  SEVERITY,
  worstBand,
} from "./risk";

describe("risk bands", () => {
  it("uses inclusive lower and exclusive upper bounds", () => {
    expect(bandFor(0.0302, "24h", FALLBACK_RISK)).toBe("low");
    expect(bandFor(0.0303, "24h", FALLBACK_RISK)).toBe("moderate");
    expect(bandFor(0.0604, "24h", FALLBACK_RISK)).toBe("moderate");
    expect(bandFor(0.0605, "24h", FALLBACK_RISK)).toBe("high");
    expect(bandFor(0.4339, "24h", FALLBACK_RISK)).toBe("high");
    expect(bandFor(0.434, "24h", FALLBACK_RISK)).toBe("very_high");
  });

  it("respects horizon-specific schemes", () => {
    // 48h: watch cut = t_warn/2, warning cut = t_warn=0.0868,
    // very-high cut = first validation score with precision >= 0.90.
    expect(bandFor(0.0433, "48h", FALLBACK_RISK)).toBe("low");
    expect(bandFor(0.0434, "48h", FALLBACK_RISK)).toBe("moderate");
    expect(bandFor(0.0867, "48h", FALLBACK_RISK)).toBe("moderate");
    expect(bandFor(0.0868, "48h", FALLBACK_RISK)).toBe("high");
    expect(bandFor(0.5629, "48h", FALLBACK_RISK)).toBe("high");
    expect(bandFor(0.563, "48h", FALLBACK_RISK)).toBe("very_high");

    // 72h: t_warn=0.0963, very-high cut=0.5277.
    expect(bandFor(0.0481, "72h", FALLBACK_RISK)).toBe("low");
    expect(bandFor(0.0482, "72h", FALLBACK_RISK)).toBe("moderate");
    expect(bandFor(0.0962, "72h", FALLBACK_RISK)).toBe("moderate");
    expect(bandFor(0.0963, "72h", FALLBACK_RISK)).toBe("high");
    expect(bandFor(0.5276, "72h", FALLBACK_RISK)).toBe("high");
    expect(bandFor(0.5277, "72h", FALLBACK_RISK)).toBe("very_high");
    expect(bandFor(0.999, "48h", FALLBACK_RISK)).toBe("very_high");
  });

  it("never turns missing values into low risk", () => {
    expect(bandFor(null, "24h", FALLBACK_RISK)).toBe("unknown");
    expect(bandFor(Number.NaN, "24h", FALLBACK_RISK)).toBe("unknown");
  });

  it("orders and compares bands without treating unknown as safe", () => {
    expect(worstBand([])).toBe("unknown");
    expect(worstBand(["low", "moderate", "high"])).toBe("high");
    expect(worstBand(["high", "very_high", "moderate"])).toBe("very_high");
    expect(worstBand(["low", "unknown"])).toBe("low");
    expect(SEVERITY.unknown).toBeLessThan(SEVERITY.low);
    expect(BAND_ORDER.at(-1)).toBe("unknown");
    expect(isElevated("high")).toBe(true);
    expect(isElevated("very_high")).toBe(true);
    expect(isElevated("moderate")).toBe(false);
    expect(isElevated("unknown")).toBe(false);
  });

  it("clamps gauge positions", () => {
    expect(gaugeFraction(null)).toBe(0);
    expect(gaugeFraction(-0.4)).toBe(0);
    expect(gaugeFraction(0.42)).toBe(0.42);
    expect(gaugeFraction(1.7)).toBe(1);
  });
});
