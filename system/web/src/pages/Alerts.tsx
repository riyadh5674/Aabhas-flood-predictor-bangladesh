import { useMemo } from "react";
import { BellRing, Clock, ShieldAlert } from "lucide-react";
import { useHorizonSet } from "@/lib/queries";
import { SEVERITY, BAND_META } from "@/lib/risk";
import { pctPoints } from "@/lib/format";
import type { Horizon, StationRisk } from "@/lib/types";
import { useApp } from "@/state/AppProvider";
import { useI18n } from "@/state/i18n";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card } from "@/components/ui/Card";
import { Reveal } from "@/components/ui/Reveal";
import { Segmented } from "@/components/ui/Segmented";
import { BandChip } from "@/components/ui/BandChip";
import { StatusPill } from "@/components/StatusPill";
import { EmptyState, ErrorState } from "@/components/ui/States";
import { Link } from "react-router-dom";

export function AlertsPage() {
  const { t } = useI18n();
  const { date, horizon, setHorizon, dataMode } = useApp();
  const horizons = useHorizonSet(date, dataMode);

  const forecast = horizons[horizon];
  const rows = forecast.data?.rows ?? [];
  const elevated = useMemo(
    () =>
      [...rows]
        .filter((r) => r.band === "high" || r.band === "very_high")
        .sort((a, b) => SEVERITY[b.band] - SEVERITY[a.band] || (b.probability ?? 0) - (a.probability ?? 0)),
    [rows],
  );

  const counts = useMemo(
    () => ({
      watch: rows.filter((r) => r.band === "moderate").length,
      warning: rows.filter((r) => r.band === "high").length,
      high: rows.filter((r) => r.band === "very_high").length,
    }),
    [rows],
  );

  return (
    <div className="space-y-6">
      <Reveal>
        <PageHeader
          eyebrow={t("alerts.title")}
          title="Signals worth checking"
          description={
            <>
              Model signals for the selected {horizon.toUpperCase()} outlook. Open a station to review its river context and trend; follow FFWC for official warnings.
            </>
          }
          actions={
            <>
              <StatusPill compact />
              <Segmented<Horizon>
                value={horizon}
                options={[
                  { value: "24h", label: "24 h" },
                  { value: "48h", label: "48 h" },
                  { value: "72h", label: "72 h" },
                ] as { value: Horizon; label: string }[]}
                onChange={setHorizon}
                ariaLabel="Horizon"
              />
            </>
          }
        />
      </Reveal>

      <Reveal delay={0.06}>
        <div className="grid gap-3 md:grid-cols-3">
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="micro mb-0.5">{t("alerts.watch")}</div>
                <div className="text-[28px] font-semibold leading-none" style={{ color: BAND_META.moderate.color }}>
                  {forecast.isError || forecast.isLoading || !date ? "—" : counts.watch}
                </div>
              </div>
              <ShieldAlert size={18} style={{ color: BAND_META.moderate.color }} />
            </div>
          </Card>
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="micro mb-0.5">{t("alerts.warning")}</div>
                <div className="text-[28px] font-semibold leading-none" style={{ color: BAND_META.high.color }}>
                  {forecast.isError || forecast.isLoading || !date ? "—" : counts.warning}
                </div>
              </div>
              <BellRing size={18} style={{ color: BAND_META.high.color }} />
            </div>
          </Card>
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="micro mb-0.5">{t("alerts.high")}</div>
                <div className="text-[28px] font-semibold leading-none" style={{ color: BAND_META.very_high.color }}>
                  {forecast.isError || forecast.isLoading || !date ? "—" : counts.high}
                </div>
              </div>
              <Clock size={18} style={{ color: BAND_META.very_high.color }} />
            </div>
          </Card>
        </div>
      </Reveal>

      <Reveal delay={0.1}>
        <div className="space-y-2">
          {!date ? (
            <EmptyState title="No forecast date is available" description="Connect the Aabhas API and load its available feature dates to see risk signals." />
          ) : forecast.isError ? (
            <ErrorState title="Forecasts are unavailable" description={forecast.error instanceof Error ? forecast.error.message : "Check the API connection and try again."} onRetry={() => void forecast.refetch()} />
          ) : forecast.isLoading ? (
            <EmptyState title="Loading forecast signals…" description="We’re retrieving model output for the selected date and horizon." />
          ) : elevated.length ? (
            elevated.map((row, i) => (
              <Card key={row.stationId} className="p-4 card--hover" style={{ borderColor: `color-mix(in srgb, ${BAND_META[row.band].color} 20%, var(--border))` }}>
                <div className="flex items-center justify-between gap-3 mb-2">
                  <div className="flex items-center gap-2">
                    <span className="num text-[11px] faint">{String(i + 1).padStart(2, "0")}</span>
                    <span className="text-[13px] font-medium">{row.station.name}</span>
                  </div>
                  <BandChip band={row.band} />
                </div>
                <div className="flex items-center gap-1 text-[11px]">
                  <span className="muted">{row.station.district}</span>
                  <span className="faint">·</span>
                  <span className="muted">{row.station.division}</span>
                </div>
                <div className="mt-1.5 flex items-end gap-2">
                  <span className="num text-[18px] font-semibold">{pctPoints(row.probability, 1)}</span>
                  <span className="text-[11px] muted">%</span>
                  <span className="num text-[11px] faint ml-auto">{row.horizon}</span>
                </div>
                <p className="mt-2.5 text-[11.5px] leading-relaxed text-2">{BAND_META[row.band].operational}</p>
                <Link className="alert-open-link" to={`/stations/${row.stationId}`}>Review station outlook <span aria-hidden="true">→</span></Link>
              </Card>
            ))
          ) : (
            <div className="p-10 text-center">
              <EmptyState
                title={t("alerts.none")}
                description="No station signals are in a high or very high band for this date and outlook. Always keep an eye on official local advisories."
              />
            </div>
          )}
        </div>
      </Reveal>
      <div className="alert-trust-note"><ShieldAlert size={16} /><span><strong>These are model signals, not official alerts.</strong> Confirm conditions and any action with FFWC and your local authorities.</span><Link to="/map">Open map</Link></div>
    </div>
  );
}

export default AlertsPage;
