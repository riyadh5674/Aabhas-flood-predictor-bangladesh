import { useQueryClient } from "@tanstack/react-query";
import {
  CalendarDays,
  ChevronLeft,
  ChevronRight,
  FlaskConical,
  RotateCw,
  Share2,
} from "lucide-react";
import { useCallback } from "react";
import { Segmented } from "@/components/ui/Segmented";
import { longDate } from "@/lib/format";
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

  const index = dates.indexOf(date);
  const atStart = index <= 0;
  const atEnd = index === -1 ? true : index >= dates.length - 1;

  const step = (delta: number) => {
    if (index === -1) return;
    const next = dates[index + delta];
    if (next) setDate(next);
  };

  const onPickDate = useCallback(
    (value: string) => {
      if (!value) return;
      if (dates.includes(value)) {
        setDate(value);
        return;
      }
      const target = Date.parse(value);
      let nearest = dates[0];
      let best = Infinity;
      for (const candidate of dates) {
        const delta = Math.abs(Date.parse(candidate) - target);
        if (delta < best) {
          best = delta;
          nearest = candidate;
        }
      }
      if (nearest) {
        setDate(nearest);
        toast({
          title: "Nearest available date",
          description: `${longDate(nearest)} is the closest date with model features.`,
          tone: "info",
        });
      }
    },
    [dates, setDate, toast],
  );

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
          <div
            className="flex items-center gap-0.5 rounded-xl border p-0.5"
            style={{ borderColor: "var(--border)", background: "var(--surface-2)" }}
          >
            <button
              type="button"
              className="icon-btn h-7 w-7"
              onClick={() => step(-1)}
              disabled={atStart}
              aria-label="Previous day"
              style={{ opacity: atStart ? 0.35 : 1 }}
            >
              <ChevronLeft size={14} />
            </button>
            <label className="relative flex cursor-pointer items-center gap-1.5 px-1.5">
              <CalendarDays size={13} className="accent" />
              <span className="num whitespace-nowrap text-[12px] font-medium">{longDate(date)}</span>
              <input
                type="date"
                value={date}
                min={dates[0]}
                max={dates.at(-1)}
                onChange={(event) => onPickDate(event.target.value)}
                className="absolute inset-0 cursor-pointer opacity-0"
                aria-label="Forecast date"
              />
            </label>
            <button
              type="button"
              className="icon-btn h-7 w-7"
              onClick={() => step(1)}
              disabled={atEnd}
              aria-label="Next day"
              style={{ opacity: atEnd ? 0.35 : 1 }}
            >
              <ChevronRight size={14} />
            </button>
          </div>
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
