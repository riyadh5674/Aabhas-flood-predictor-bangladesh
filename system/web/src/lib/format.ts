import { format, parseISO } from "date-fns";

export function pct(value: number | null | undefined, digits = 1): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

/** Probability rendered in percent-points, e.g. 0.0605 -> "6.05". */
export function pctPoints(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) return "—";
  return (value * 100).toFixed(digits);
}

export function int(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en-US").format(Math.round(value));
}

export function dec(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function meters(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${value.toFixed(2)} m`;
}

export function longDate(iso: string): string {
  try {
    return format(parseISO(iso), "d MMMM yyyy");
  } catch {
    return iso;
  }
}

export function shortDate(iso: string): string {
  try {
    return format(parseISO(iso), "d MMM yyyy");
  } catch {
    return iso;
  }
}

export function weekday(iso: string): string {
  try {
    return format(parseISO(iso), "EEE");
  } catch {
    return "";
  }
}

export function dayMonth(iso: string): string {
  try {
    return format(parseISO(iso), "d MMM");
  } catch {
    return iso;
  }
}

export function clockTime(value: string | number | Date): string {
  const d = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(d.getTime())) return "—";
  return format(d, "HH:mm");
}

export function relativeTime(from: number, now = Date.now()): string {
  const seconds = Math.round((now - from) / 1000);
  if (seconds < 45) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return `${days} d ago`;
}

export function horizonLabel(h: string): string {
  return h.replace("h", " h");
}

export function titleCase(value: string): string {
  return value
    .split(/[\s_-]+/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

export function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

export function truncate(value: string, max = 40): string {
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}
