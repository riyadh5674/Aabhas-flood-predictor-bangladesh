import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { Compass, Clock, MapPin, AlertTriangle } from "lucide-react";
import { useStationTrend } from "@/lib/queries";
import { BAND_META } from "@/lib/risk";
import { pctPoints, dayMonth, dec, meters, longDate } from "@/lib/format";
import type { Horizon, Station, StationRisk } from "@/lib/types";
import { loadStations } from "@/lib/stations";
import { useApp } from "@/state/AppProvider";
import { useI18n } from "@/state/i18n";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card } from "@/components/ui/Card";
import { Reveal } from "@/components/ui/Reveal";
import { Segmented } from "@/components/ui/Segmented";
import { ProbabilityGauge } from "@/components/viz/ProbabilityGauge";
import { Sparkline } from "@/components/viz/Sparkline";
import { TrendChart } from "@/components/viz/TrendChart";
import { BandChip } from "@/components/ui/BandChip";
import { StatusPill } from "@/components/StatusPill";
import { EmptyState, ErrorState } from "@/components/ui/States";
import { downloadCsv } from "@/lib/csv";

const HORIZON_OPTIONS = ["24h", "48h", "72h"].map((h) => ({ value: h, label: h.replace("h", " h") }));

export function StationDetailPage() {
  const { stationId: rawId } = useParams<{ stationId: string }>();
  const stationId = Number(rawId);
  const { t } = useI18n();
  const { date, dates, horizon, setHorizon, dataMode } = useApp();

  return <StationDetailView stationId={stationId} date={date} dates={dates} horizon={horizon} setHorizon={setHorizon} dataMode={dataMode} />;
}

function StationDetailView({
  stationId,
  date,
  dates,
  horizon,
  setHorizon,
  dataMode,
}: {
  stationId: number;
  date: string;
  dates: string[];
  horizon: Horizon;
  setHorizon: (h: Horizon) => void;
  dataMode: "auto" | "demo";
}) {
  const { t } = useI18n();
  const [station, setStation] = useState<Station | null>(null);
  const [stationLoading, setStationLoading] = useState(true);
  const [stationError, setStationError] = useState<Error | null>(null);
  const [stationRetry, setStationRetry] = useState(0);

  const trendDates = useMemo(() => {
    const selected = /^\d{4}-\d{2}-\d{2}$/.test(date) ? date : "";
    const available = dates
      .filter((day) => /^\d{4}-\d{2}-\d{2}$/.test(day) && (!selected || day <= selected))
      .slice(-30);
    if (selected && !available.includes(selected)) return [...available, selected].slice(-30);
    return available;
  }, [dates, date]);
  const trend = useStationTrend(station ?? undefined, trendDates, horizon, dataMode);

  useEffect(() => {
    let cancelled = false;
    setStation(null);
    setStationLoading(true);
    setStationError(null);
    loadStations().then((list) => {
      if (cancelled) return;
      setStation(list.find((x) => x.id === stationId) ?? null);
      setStationLoading(false);
    }).catch((error: unknown) => {
      if (cancelled) return;
      setStationError(error instanceof Error ? error : new Error("Could not load the station directory."));
      setStationLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [stationId, stationRetry]);

  const trendPoints = useMemo(() => trend.data?.points ?? [], [trend.data]);
  // Missing days are explicit `unknown` points. The headline must show the most
  // recent real forecast, never turn "no data" into a reassuring 0%.
  const latest = useMemo(
    () => trendPoints.filter((point) => point.probability != null).at(-1) ?? null,
    [trendPoints],
  );
  const meta = BAND_META[latest?.band ?? "unknown"];

  if (!station) {
    if (stationError) {
      return <ErrorState title="Couldn’t load station details" description={stationError.message} onRetry={() => setStationRetry((value) => value + 1)} />;
    }
    return (
      <div className="flex h-[560px] flex-col items-center justify-center text-center">
        <EmptyState title={stationLoading ? "Loading station…" : "Station not found"} description={stationLoading ? "Please wait while we load the station master." : `No station with ID ${stationId} exists in the station directory.`} />
      </div>
    );
  }

  if (!latest) {
    if (trend.isError) {
      return <ErrorState title="Couldn’t load this station forecast" description={trend.error instanceof Error ? trend.error.message : "Check the API connection and try again."} onRetry={() => void trend.refetch()} />;
    }
    if (!date) {
      return <EmptyState title="No forecast date is available" description="Connect the Aabhas API and load available feature dates to see this station forecast." />;
    }
    if (trend.isLoading) {
      return <EmptyState title="Loading station forecast…" description={`Retrieving the ${horizon.toUpperCase()} outlook for ${station.name}.`} />;
    }
    return (
      <div className="flex h-[560px] flex-col items-center justify-center text-center">
        <EmptyState
          title="No forecast for this station"
          description={`There is no ${horizon.toUpperCase()} forecast for ${station.name} on ${date}.`}
        />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Reveal>
        <PageHeader
          eyebrow={station.name}
          title={`${station.river ? station.river + " · " : ""}${station.district}, ${station.division}`}
          description={
            <>
              <span className="flex items-center gap-1.5">
                <MapPin size={13} className="faint" />
                {station.upazilla}, {station.district}, {station.division}
              </span>
            </>
          }
          actions={
            <>
              <StatusPill compact />
              <button
                type="button"
                className="icon-btn"
                onClick={() => {
                  downloadCsv(
                    `aabhas-station-${stationId}-${date}.csv`,
                    trendPoints.map((p) => ({ date: p.date, probability: p.probability, band: p.band })),
                  );
                }}
                title="Export station trend CSV"
              >
                <span className="flex items-center gap-1">Export</span>
              </button>
            </>
          }
        />
      </Reveal>

      <Reveal delay={0.06}>
        <div className="grid gap-5 lg:grid-cols-[1fr_260px]">
          <div className="flex flex-col gap-5">
            <Card className="p-5 flex flex-col gap-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="micro mb-0.5">Latest probability · {horizon.toUpperCase()}</div>
                  <div className="text-[34px] font-semibold leading-none tracking-[-0.02em]" style={{ color: meta.color }}>
                    {latest.probability == null ? "—" : (latest.probability * 100).toFixed(1)}
                    <span className="text-[14px] font-normal muted">%</span>
                  </div>
                </div>
                <BandChip band={latest.band} />
              </div>
              {date && latest.date !== date ? (
                <p className="text-[11.5px] muted" role="status">
                  No {horizon.toUpperCase()} forecast for {longDate(date)}. Showing the most recent available forecast from{" "}
                  {longDate(latest.date)}.
                </p>
              ) : null}
              <p className="text-[13px] leading-relaxed text-2" style={{ borderLeft: `3px solid ${meta.color}`, paddingLeft: "12px", margin: 0 }}>
                {meta.guidance}
              </p>
              <div className="mt-2 flex items-center gap-4 text-[11.5px]">
                <span className="flex items-center gap-1">
                  <Clock size={12} className="faint" />
                  <span>{longDate(latest.date)}</span>
                </span>
                <span className="flex items-center gap-1">
                  <Compass size={12} className="faint" />
                  <span>{dec(station.dangerLevel)} m danger level</span>
                </span>
                <span className="flex items-center gap-1">
                  <AlertTriangle size={12} className="faint" />
                  <span>{dec(station.recordHigh)} m record high</span>
                </span>
              </div>
              <div className="mt-3 h-1 w-full overflow-hidden rounded-full" style={{ background: "var(--surface-3)" }}>
                <div
                  className="h-full rounded-full viz-bar"
                  style={{
                    width: `${Math.min(100, (latest.probability ?? 0) * 100 * 1.6)}%`,
                    background: meta.color,
                  }}
                />
              </div>
            </Card>

            <Card className="p-5">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="display text-[16px]">Trend over time</h3>
                <Segmented<Horizon>
                  value={horizon}
                  options={[
                    { value: "24h", label: "24 h" },
                    { value: "48h", label: "48 h" },
                    { value: "72h", label: "72 h" },
                  ] as { value: Horizon; label: string }[]}
                  onChange={setHorizon}
                  ariaLabel="Trend horizon"
                />
              </div>
              <TrendChart points={trendPoints} threshold={latest.threshold ?? null} height={230} />
            </Card>

            <Card className="p-5">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="display text-[16px]">Probability spark</h3>
              </div>
              <div className="flex items-end gap-2">
                <Sparkline
                  values={trendPoints.map((p) => p.probability)}
                  width={320}
                  height={70}
                  color={meta.color}
                  showArea={true}
                />
                <div className="text-[12px] text-2">
                  {pctPoints(trendPoints.at(-1)?.probability ?? null, 1)}%
                  {" "}
                  {trendPoints.at(-1) ? dayMonth(trendPoints.at(-1)!.date) : ""}
                </div>
              </div>
            </Card>
          </div>

          <aside className="space-y-4">
            <Card className="p-4">
              <div className="mb-3 flex items-center gap-2">
                <Compass size={15} style={{ color: "var(--accent)" }} />
                <h3 className="display text-[15px]">Station profile</h3>
              </div>
              <table className="tbl text-[12.5px]">
                <tbody>
                  <tr>
                    <td className="muted w-1/3">Station ID</td>
                    <td className="num">{station.id}</td>
                  </tr>
                  <tr>
                    <td className="muted">River</td>
                    <td>{station.river || "—"}</td>
                  </tr>
                  <tr>
                    <td className="muted">Basin</td>
                    <td>{station.basin || "—"}</td>
                  </tr>
                  <tr>
                    <td className="muted">Division</td>
                    <td>{station.division}</td>
                  </tr>
                  <tr>
                    <td className="muted">District</td>
                    <td>{station.district}</td>
                  </tr>
                  <tr>
                    <td className="muted">Upazilla</td>
                    <td>{station.upazilla || "—"}</td>
                  </tr>
                  <tr>
                    <td className="muted">Coordinates</td>
                    <td className="num">{station.lat.toFixed(4)}, {station.lon.toFixed(4)}</td>
                  </tr>
                </tbody>
              </table>
            </Card>

            <Card className="p-4">
              <div className="mb-3 micro" style={{ fontSize: 9 }}>
                Recommended action
              </div>
              <div className="p-3 rounded-xl border" style={{ background: "var(--surface-2)", borderColor: "var(--border)" }}>
                <p className="text-[12.5px] leading-relaxed text-2">{latest.action}</p>
              </div>
            </Card>
          </aside>
        </div>
      </Reveal>
    </div>
  );
}

export default StationDetailPage;
