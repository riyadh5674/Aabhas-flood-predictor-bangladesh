import { useNavigate, useSearchParams } from "react-router-dom";
import { Search, Filter, X } from "lucide-react";
import { useState } from "react";
import { useMemo } from "react";
import { useHorizonSet } from "@/lib/queries";
import { SEVERITY, BAND_META } from "@/lib/risk";
import { pctPoints, dec } from "@/lib/format";
import type { Band, Horizon, StationRisk } from "@/lib/types";
import { useApp } from "@/state/AppProvider";
import { useI18n } from "@/state/i18n";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Segmented } from "@/components/ui/Segmented";
import { Reveal } from "@/components/ui/Reveal";
import { BandChip } from "@/components/ui/BandChip";
import { StatusPill } from "@/components/StatusPill";
import { downloadCsv } from "@/lib/csv";
import { ErrorState } from "@/components/ui/States";

export function StationsPage() {
  const { t } = useI18n();
  const { date, horizon, setHorizon, dataMode } = useApp();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [search, setSearch] = useState("");
  const [division, setDivision] = useState("all");
  const [band, setBand] = useState("all");
  const [district] = useState(() => searchParams.get("district") ?? "");
  const horizons = useHorizonSet(date, dataMode);

  const forecast = horizons[horizon];
  const rows = forecast.data?.rows ?? [];
  const stations: StationRisk[] = useMemo(
    () => [...rows].sort((a, b) => SEVERITY[b.band] - SEVERITY[a.band] || (b.probability ?? 0) - (a.probability ?? 0)),
    [rows],
  );
  const divisions = useMemo(() => [...new Set(stations.map((row) => row.station.division))].sort(), [stations]);
  const filteredStations = useMemo(() => {
    const q = search.trim().toLowerCase();
    return stations.filter((row) => {
      const matchesText = !q || [row.station.name, row.station.river, row.station.district, row.station.division, row.station.upazilla].join(" ").toLowerCase().includes(q);
      const matchesDivision = division === "all" || row.station.division === division;
      const matchesBand = band === "all" || row.band === band;
      const matchesDistrict = !district || row.station.districtKey.toLowerCase() === district.toLowerCase() || row.station.district.toLowerCase() === district.toLowerCase();
      return matchesText && matchesDivision && matchesBand && matchesDistrict;
    });
  }, [stations, search, division, band, district]);

  return (
    <div className="space-y-6">
      <Reveal>
        <PageHeader
          eyebrow={t("stations.title")}
          title="Station directory"
          description={
            <>
              {forecast.isError ? "Forecast data is unavailable." : forecast.isLoading ? "Loading station forecasts…" : `All ${rows.length} FFWC stations for the ${horizon.toUpperCase()} horizon, ranked by risk band then probability.`}
              {!forecast.isError && !forecast.isLoading ? " Click a station for its full detail page." : null}
            </>
          }
          actions={
            <>
              <StatusPill compact />
              <button type="button" className="btn btn--sm" title="Export filtered stations" disabled={forecast.isLoading || forecast.isError || filteredStations.length === 0} onClick={() => exportCsv(filteredStations)}>
                <DownloadIcon size={14} />
                <span className="hidden sm:inline">Export</span>
              </button>
            </>
          }
        />
      </Reveal>

      <Reveal delay={0.06}>
        <div className="station-filters mb-3">
          <div className="relative flex-1 min-w-0">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 muted" />
            <input
              type="text"
              placeholder={t("stations.search")}
              className="input pl-9 pr-3"
              aria-label="Search stations"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>
          <label className="station-filter-select"><Filter size={13} /><select className="input" aria-label="Filter by division" value={division} onChange={(event) => setDivision(event.target.value)}><option value="all">All divisions</option>{divisions.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
          <select className="input station-filter-select__band" aria-label="Filter by risk level" value={band} onChange={(event) => setBand(event.target.value)}><option value="all">All risk levels</option><option value="very_high">Very high</option><option value="high">High</option><option value="moderate">Watch</option><option value="low">Low</option><option value="unknown">No signal</option></select>
          {district && <button type="button" className="chip chip--accent" onClick={() => navigate("/stations")}>{district}<X size={12} /></button>}
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
        </div>

        <div className="overflow-x-auto border border-b-0 rounded-xl">
          <table className="tbl text-[12.5px]">
            <thead>
              <tr>
                <th style={{ width: "22%" }}>Station</th>
                <th style={{ width: "14%" }}>River</th>
                <th style={{ width: "14%" }}>District</th>
                <th style={{ width: "14%" }}>Division</th>
                <th className="text-right" style={{ width: "12%" }}>Probability</th>
                <th className="text-right" style={{ width: "12%" }}>Band</th>
                <th style={{ width: "12%" }}>Danger level</th>
              </tr>
            </thead>
            <tbody>
              {filteredStations.length ? (
                filteredStations.map((row) => (
                  <tr key={row.stationId} className="group hover:text-[var(--text)] cursor-pointer" tabIndex={0} role="link" aria-label={`Open ${row.station.name} station details`} onClick={() => navigate(`/stations/${row.stationId}`)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); navigate(`/stations/${row.stationId}`); } }}>
                    <td>
                      <div className="flex items-center gap-2">
                        <BandChip band={row.band} className="shrink-0" />
                        <span className="truncate text-[12px] font-medium">{row.station.name}</span>
                      </div>
                      <div className="truncate text-[11px] muted">{row.station.upazilla || row.station.district}</div>
                    </td>
                    <td className="text-[11px] muted truncate">{row.station.river || "—"}</td>
                    <td className="text-[11px]">{row.station.district}</td>
                    <td className="text-[11px]">{row.station.division}</td>
                    <td className="text-right">
                      <span className="num">{row.probability == null ? "—" : pctPoints(row.probability, 1)}%</span>
                    </td>
                    <td className="text-right">
                      <BandChip band={row.band} className="shrink-0" />
                    </td>
                    <td className="text-right">
                      <span className="num">{dec(row.station.dangerLevel)}</span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="h-24 text-center text-[12px] muted">
                    {forecast.isError ? "Forecasts could not be loaded. Use the retry button above." : forecast.isLoading ? "Loading station forecasts…" : !date ? "No forecast date is available from the API." : "No stations match these filters. Try a different search or clear a filter."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="mt-2 text-right text-[10px] muted">Showing {forecast.isError || !date ? "—" : filteredStations.length} of {forecast.isError || !date ? "—" : stations.length} stations</div>
        {forecast.isError && <ErrorState title="Couldn’t load station forecasts" description={forecast.error instanceof Error ? forecast.error.message : "Check your API connection and try again."} onRetry={() => void forecast.refetch()} />}
      </Reveal>
    </div>
  );
}

function DownloadIcon({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

function exportCsv(rows: StationRisk[]) {
  downloadCsv(
    `aabhas-stations-${new Date().toISOString().slice(0, 10)}.csv`,
    rows.map((r) => ({
      station_id: r.stationId,
      station: r.station.name,
      river: r.station.river,
      division: r.station.division,
      district: r.station.district,
      probability: r.probability ?? null,
      band: r.band,
      action: r.action,
      threshold: r.threshold ?? null,
    })),
  );
}

export default StationsPage;
