/**
 * Data-freshness policy for the flood outlook.
 *
 * `/health` reports the newest date in the feature store. A forecast is only
 * meaningful while that date is close to today, so the UI uses one shared
 * policy to decide when to warn the user instead of quietly showing a stale
 * map. Keep this pure and testable: no React, no clock reads by default.
 */

export type FreshnessLevel = "current" | "aging" | "stale" | "expired" | "unknown";

export interface Freshness {
  level: FreshnessLevel;
  /** Whole calendar days between the latest feature date and today. */
  daysBehind: number | null;
  /** Short label for a chip, e.g. "3 days behind". */
  label: string;
  /** One-sentence explanation suitable for a banner. */
  message: string;
  /** True when the UI should show a warning the user cannot miss. */
  warn: boolean;
}

const DAY_MS = 86_400_000;

/** Parse a YYYY-MM-DD string as a UTC calendar day without timezone drift. */
export function parseDateOnly(value: string | null | undefined): number | null {
  if (!value || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const [year, month, day] = value.split("-").map(Number);
  const utc = Date.UTC(year, month - 1, day);
  return Number.isNaN(utc) ? null : utc;
}

/** Calendar-day difference between two YYYY-MM-DD dates (b - a). */
export function calendarDaysBetween(
  earlier: string | null | undefined,
  later: string | null | undefined,
): number | null {
  const a = parseDateOnly(earlier);
  const b = parseDateOnly(later);
  if (a == null || b == null) return null;
  return Math.round((b - a) / DAY_MS);
}

function todayIso(now: Date): string {
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()))
    .toISOString()
    .slice(0, 10);
}

export function evaluateFreshness(
  latestFeatureDate: string | null | undefined,
  now: Date = new Date(),
): Freshness {
  const daysBehind = calendarDaysBetween(latestFeatureDate, todayIso(now));
  if (daysBehind == null) {
    return {
      level: "unknown",
      daysBehind: null,
      label: "Freshness unknown",
      message: "The API did not report a latest feature date. Treat these forecasts with caution.",
      warn: true,
    };
  }

  // A feature store normally ends yesterday (FFWC posts with a short lag), so
  // 0–1 days behind is healthy, 2–3 is worth flagging, 4–7 is stale.
  if (daysBehind <= 1) {
    return {
      level: "current",
      daysBehind,
      label: daysBehind <= 0 ? "Up to date" : "1 day behind",
      message: "Forecasts use the latest available feature date.",
      warn: false,
    };
  }
  if (daysBehind <= 3) {
    return {
      level: "aging",
      daysBehind,
      label: `${daysBehind} days behind`,
      message: `The newest model features are ${daysBehind} days old. Re-check before making decisions.`,
      warn: true,
    };
  }
  if (daysBehind <= 7) {
    return {
      level: "stale",
      daysBehind,
      label: `${daysBehind} days behind`,
      message: `The newest model features are ${daysBehind} days old. This is a stale outlook, not a current warning.`,
      warn: true,
    };
  }
  return {
    level: "expired",
    daysBehind,
    label: `${daysBehind} days behind`,
    message: `The newest model features are ${daysBehind} days old. Forecasts may no longer reflect river conditions.`,
    warn: true,
  };
}
