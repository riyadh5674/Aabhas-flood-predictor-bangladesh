import { motion } from "motion/react";
import { STATIC_SNAPSHOT } from "@/lib/api";
import { cn } from "@/lib/cn";
import { useHealth } from "@/lib/queries";
import { useApp } from "@/state/AppProvider";
import { useI18n } from "@/state/i18n";

/**
 * Live status pill. Shares the app-wide health query so one open tab makes a
 * single health request every five minutes instead of one per component.
 */
export function StatusPill({ compact = false }: { compact?: boolean }) {
  const { t } = useI18n();
  const { dataMode } = useApp();
  const health = useHealth();
  const latency = health.data?.latencyMs ?? null;

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
