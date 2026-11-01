import { useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { fetchHealth } from "@/lib/api";
import { STATIC_SNAPSHOT } from "@/lib/api";
import { cn } from "@/lib/cn";
import { useApp } from "@/state/AppProvider";
import { useI18n } from "@/state/i18n";

/**
 * Live status pill. Measures real round-trip latency against /health and
 * distinguishes API/model readiness, explicit demo mode, and checking.
 */
export function StatusPill({ compact = false }: { compact?: boolean }) {
  const { t } = useI18n();
  const { dataMode } = useApp();
  const [latency, setLatency] = useState<number | null>(null);

  const health = useQuery({
    queryKey: ["health", "pill"],
    queryFn: async () => {
      const started = performance.now();
      const data = await fetchHealth();
      setLatency(Math.round(performance.now() - started));
      return data;
    },
    refetchInterval: 45_000,
    retry: false,
    staleTime: 20_000,
  });

  useEffect(() => {
    if (health.isError) setLatency(null);
  }, [health.isError]);

  const live = health.data?.status === "ok" && health.data.model_loaded && health.data.feature_store_dates > 0;
  const snapshot = STATIC_SNAPSHOT && health.isSuccess && health.data?.status === "snapshot";
  const demo = dataMode === "demo";
  const offline = !snapshot && (health.isError || (health.isSuccess && !live));
  const checking = health.isLoading && !health.isError;

  const color = demo || offline ? "var(--band-moderate)" : live ? "var(--band-low)" : "var(--muted)";
  const latestFeatureDate = health.data?.latest_feature_date;
  const label = demo ? "Demo" : snapshot ? "Snapshot" : offline ? "Unavailable" : live ? "API ready" : t("common.loading");
  const formattedFeatureDate = latestFeatureDate
    ? new Date(`${latestFeatureDate}T00:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })
    : null;

  return (
    <div
      className="flex items-center gap-2 rounded-full border px-2.5 py-1"
      style={{
        borderColor: `color-mix(in srgb, ${color} 32%, transparent)`,
        background: `color-mix(in srgb, ${color} 10%, transparent)`,
      }}
      title={
        demo
          ? "Demo mode is on. Values are synthetic and are not forecasts."
          : snapshot
            ? `Precomputed D16 forecast snapshot through ${latestFeatureDate ?? "an unknown date"}. It is not real-time; forecasts refresh when the snapshot is manually rebuilt and redeployed.`
          : offline
            ? "The inference API or model is unavailable. No sample forecast is being substituted."
            : live
              ? `Inference API and model are ready. Latest available model feature date: ${latestFeatureDate ?? "unknown"}. This is feature-store coverage, not a real-time observation timestamp.${latency != null ? ` API response: ${latency} ms` : ""}`
              : "Checking API and model availability"
      }
    >
      <motion.span
        className="relative flex h-1.5 w-1.5 rounded-full"
        style={{ background: color }}
        animate={checking ? { opacity: [1, 0.35, 1] } : { opacity: 1 }}
        transition={checking ? { duration: 1.4, repeat: Infinity } : { duration: 0.2 }}
      />
      <span className="num text-[10px] font-medium uppercase tracking-[0.12em]" style={{ color }}>
        {label}
      </span>
      {!compact && latency != null && live ? (
        <span className={cn("num text-[10px] faint")}>{latency}ms</span>
      ) : null}
      {!compact && formattedFeatureDate && (live || snapshot) ? (
        <span className={cn("num text-[10px] faint")}>data through {formattedFeatureDate}</span>
      ) : null}
    </div>
  );
}
