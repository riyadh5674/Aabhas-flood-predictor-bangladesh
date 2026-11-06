import { AlertTriangle, FlaskConical, WifiOff } from "lucide-react";
import { Link } from "react-router-dom";
import { shortDate } from "@/lib/format";
import { evaluateFreshness, type FreshnessLevel } from "@/lib/freshness";
import { useHealth } from "@/lib/queries";
import { useApp } from "@/state/AppProvider";

const TONE: Record<Exclude<FreshnessLevel, "current" | "unknown">, string> = {
  aging: "var(--band-moderate)",
  stale: "var(--band-high)",
  expired: "var(--band-very-high)",
};

function Banner({
  tone,
  icon,
  title,
  children,
}: {
  tone: string;
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="no-print border-b px-4 py-2 sm:px-6"
      style={{
        borderColor: `color-mix(in srgb, ${tone} 35%, transparent)`,
        background: `color-mix(in srgb, ${tone} 12%, var(--bg))`,
      }}
      role="status"
      aria-live="polite"
    >
      <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-2 gap-y-1 text-[12px]">
        <span className="flex shrink-0 items-center gap-1.5 font-semibold" style={{ color: tone }}>
          {icon}
          {title}
        </span>
        <span className="text-2">{children}</span>
      </div>
    </div>
  );
}

/**
 * Two safety-critical notices that must never be hidden behind a tooltip:
 *  1. synthetic demo data is active, and
 *  2. the feature store is behind today's date.
 */
export function DataFreshnessBanner() {
  const health = useHealth();
  const { dataMode, date, latestDate } = useApp();

  if (dataMode === "demo") {
    return (
      <Banner
        tone="var(--band-moderate)"
        icon={<FlaskConical size={14} />}
        title="Demo data — not a forecast"
      >
        Values on this screen are synthetic samples for interface development. Do not use them for any decision.
      </Banner>
    );
  }

  if (health.isError) {
    return (
      <Banner tone="var(--band-high)" icon={<WifiOff size={14} />} title="Live API unavailable">
        Forecasts cannot be loaded right now. The app will not substitute invented values.{" "}
        <button type="button" className="underline" onClick={() => void health.refetch()}>
          Retry
        </button>
      </Banner>
    );
  }

  if (health.isLoading || !health.data) return null;

  const latestFeatureDate = health.data.latest_feature_date;
  const staleCritical = Object.entries(health.data.sources ?? {}).filter(
    ([, source]) => source.critical && source.status !== "ok",
  );
  const degraded = Boolean(health.data.degraded) || staleCritical.length > 0;

  if (degraded) {
    const names = staleCritical
      .map(([name, source]) => (source.last_date ? `${name} (${shortDate(source.last_date)})` : name))
      .join(", ");
    return (
      <Banner tone="var(--band-high)" icon={<AlertTriangle size={14} />} title="Data source degraded">
        {names ? (
          <>
            Critical inputs are behind their freshness SLA: <strong>{names}</strong>. Forecasts may
            be less reliable.
          </>
        ) : (
          <>A critical data source is behind its freshness SLA. Forecasts may be less reliable.</>
        )}
        {latestFeatureDate ? (
          <>
            {" "}
            Latest model features: <strong>{shortDate(latestFeatureDate)}</strong>.
          </>
        ) : null}{" "}
        <Link to="/model" className="underline">
          Model details
        </Link>
      </Banner>
    );
  }

  const freshness = evaluateFreshness(latestFeatureDate);

  if (!freshness.warn) return null;

  const level = freshness.level === "unknown" ? "stale" : freshness.level;
  const tone = level === "expired" ? TONE.expired : level === "stale" ? TONE.stale : TONE.aging;

  return (
    <Banner tone={tone} icon={<AlertTriangle size={14} />} title={freshness.label}>
      {freshness.message}
      {latestFeatureDate ? (
        <>
          {" "}
          Latest model features: <strong>{shortDate(latestFeatureDate)}</strong>.
        </>
      ) : null}{" "}
      <Link to="/model" className="underline">
        Model details
      </Link>
      {date && latestDate && date !== latestDate ? (
        <>
          {" "}
          You are viewing <strong>{shortDate(date)}</strong>.
        </>
      ) : null}
    </Banner>
  );
}
