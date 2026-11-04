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
    expect(bandFor(0.0868, "48h", FALLBACK_RISK)).toBe("moderate");
    expect(bandFor(0.1736, "48h", FALLBACK_RISK)).toBe("high");
    expect(bandFor(0.0963, "72h", FALLBACK_RISK)).toBe("moderate");
    expect(bandFor(0.1926, "72h", FALLBACK_RISK)).toBe("high");
    // 48h/72h have no very_high band in the frozen configuration.
    expect(bandFor(0.999, "48h", FALLBACK_RISK)).toBe("high");
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
