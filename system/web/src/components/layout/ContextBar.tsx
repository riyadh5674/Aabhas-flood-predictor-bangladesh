import { useQueryClient } from "@tanstack/react-query";
import { FlaskConical, RotateCw, Share2 } from "lucide-react";
import { Segmented } from "@/components/ui/Segmented";
import { DatePicker } from "@/components/ui/DatePicker";
import { DEMO_ENABLED } from "@/lib/config";
import { HORIZONS, type Horizon } from "@/lib/types";
import { useApp } from "@/state/AppProvider";
import { useI18n } from "@/state/i18n";
import { useToast } from "@/state/ToastProvider";

const HORIZON_OPTIONS = HORIZONS.map((h) => ({ value: h, label: h.replace("h", " h") }));

export function ContextBar() {
  const { t } = useI18n();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const { date, setDate, dates, latestDate, horizon, setHorizon, dataMode, setDataMode } = useApp();

  const share = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      toast({ title: "Link copied", description: "This view is shareable — date and horizon included.", tone: "success" });
    } catch {
      toast({ title: "Could not copy link", description: "Your browser blocked clipboard access.", tone: "danger" });
    }
  };

  const refresh = () => {
    void queryClient.invalidateQueries();
    toast({ title: "Refreshing forecasts", tone: "info" });
  };

  const toggleMode = () => {
    const next = dataMode === "auto" ? "demo" : "auto";
    setDataMode(next);
    toast({
      title: next === "demo" ? "Demo mode on" : "Auto mode on",
      description:
        next === "demo"
          ? "Showing a deterministic synthetic dataset. Live API calls are paused."
          : "Using the inference API. If it fails, the app will show an error instead of invented forecast values.",
      tone: "info",
    });
  };

  return (
    <div
      className="sticky top-16 z-20 border-b px-4 no-print sm:px-6"
      style={{
        background: "color-mix(in srgb, var(--bg) 86%, transparent)",
        backdropFilter: "blur(16px)",
        borderColor: "var(--border)",
      }}
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 py-2.5">
        {/* Date */}
        <div className="flex items-center gap-1.5">
          <span className="micro hidden sm:inline" style={{ fontSize: 9 }}>
            {t("common.date")}
          </span>
          <DatePicker
            value={date}
            availableDates={dates}
            onChange={setDate}
            minDate={dates[0]}
            maxDate={dates.at(-1)}
          />
          {date !== latestDate ? (
            <button type="button" className="btn btn--sm btn--ghost" onClick={() => setDate(latestDate)}>
              Latest
            </button>
          ) : (
            <span className="chip chip--accent hidden md:inline-flex" style={{ height: 22, fontSize: 10.5 }}>
              Latest
            </span>
          )}
        </div>

        <div className="hidden h-5 w-px sm:block" style={{ background: "var(--border)" }} />

        {/* Horizon */}
        <div className="flex items-center gap-2">
          <span className="micro hidden sm:inline" style={{ fontSize: 9 }}>
            {t("common.horizon")}
          </span>
          <Segmented<Horizon>
            value={horizon}
            options={HORIZON_OPTIONS}
            onChange={setHorizon}
            ariaLabel="Forecast horizon"
          />
        </div>

        <div className="ml-auto flex items-center gap-1.5">
          {DEMO_ENABLED ? (
            <button
              type="button"
              onClick={toggleMode}
              className="chip hidden lg:inline-flex"
              style={{
                borderColor: dataMode === "demo" ? "var(--band-moderate)" : "var(--border)",
                color: dataMode === "demo" ? "var(--band-moderate)" : "var(--muted)",
                background: dataMode === "demo" ? "color-mix(in srgb, var(--band-moderate) 10%, transparent)" : "var(--surface-2)",
              }}
              title={dataMode === "demo" ? "Demo mode is on — click for API mode" : "API mode — forecast values come from the configured model"}
            >
              <FlaskConical size={12} />
              {dataMode === "demo" ? "Demo" : "API"}
            </button>
          ) : null}
          <button type="button" className="icon-btn h-8 w-8" onClick={refresh} aria-label="Refresh data" title="Refresh">
            <RotateCw size={14} />
          </button>
          <button type="button" className="icon-btn h-8 w-8" onClick={share} aria-label="Copy shareable link" title="Share view">
            <Share2 size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
