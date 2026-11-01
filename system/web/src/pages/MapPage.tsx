import { useEffect, useMemo, useState } from "react";
import { ArrowUpRight, LocateFixed, Map as MapIcon, MapPinned, SlidersHorizontal } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { useSearchParams } from "react-router-dom";
import { usePredictions } from "@/lib/queries";
import { aggregateRegions } from "@/lib/aggregate";
import { BAND_META, SEVERITY } from "@/lib/risk";
import { pctPoints } from "@/lib/format";
import type { Band, MapLevel } from "@/lib/types";
import { useApp } from "@/state/AppProvider";
import { useI18n } from "@/state/i18n";
import { BangladeshMap } from "@/components/map/BangladeshMap";
import { BandChip } from "@/components/ui/BandChip";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";

const BANDS: Band[] = ["very_high", "high", "moderate", "low", "unknown"];
const bandLabel: Record<Band, string> = { very_high: "Very high", high: "High", moderate: "Watch", low: "Low", unknown: "No signal" };

export function MapPage() {
  const { t } = useI18n();
  const { date, horizon, dataMode } = useApp();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [level, setLevel] = useState<MapLevel>("adm1");
  const [selectedRegion, setSelectedRegion] = useState<string | null>(() => searchParams.get("region"));
  const [activeBands, setActiveBands] = useState<Band[]>([]);
  const result = usePredictions(date, horizon, dataMode);
  useEffect(() => setSelectedRegion(searchParams.get("region")), [searchParams]);
  const rows = result.data?.rows ?? [];
  const regions = useMemo(() => aggregateRegions(rows, level), [rows, level]);
  const rankedRegions = useMemo(() => [...regions.values()].sort((a, b) => SEVERITY[b.worst] - SEVERITY[a.worst] || (b.maxProbability ?? 0) - (a.maxProbability ?? 0)), [regions]);
  const selected = selectedRegion ? regions.get(selectedRegion) : null;
  const focusRows = selected?.stations ?? [...rows].sort((a, b) => SEVERITY[b.band] - SEVERITY[a.band] || (b.probability ?? 0) - (a.probability ?? 0));
  const visibleRows = focusRows.filter((row) => activeBands.length === 0 || activeBands.includes(row.band)).slice(0, 6);
  const reporting = rows.filter((row) => row.probability != null).length;
  const elevated = rows.filter((row) => row.band === "high" || row.band === "very_high").length;

  const toggleBand = (band: Band) => setActiveBands((current) => current.includes(band) ? current.filter((item) => item !== band) : [...current, band]);
  const chooseRegion = (region: string | null) => {
    setSelectedRegion(region);
    const params = new URLSearchParams(searchParams);
    if (region) params.set("region", region); else params.delete("region");
    setSearchParams(params, { replace: true });
  };
  const focusBangladesh = () => { chooseRegion(null); setActiveBands([]); };

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Explore Bangladesh"
        title="Flood risk, on the map."
        description={<>See which river stations are reporting elevated risk. Select a division or district to explore nearby stations.</>}
        actions={<span className={`source-badge source-badge--${result.data?.source ?? (result.isError ? "error" : "loading")}`}><span />{result.data?.source === "demo" ? "Sample outlook" : result.data?.source === "live" ? "Model output" : result.isError ? "Forecast unavailable" : "Loading outlook"}</span>}
      />

      {result.isError && <div className="map-request-error" role="alert">
        <strong>We couldn’t load this forecast.</strong>
        <span>{result.error instanceof Error ? result.error.message : "Check the API connection or choose another available date."}</span>
        <button type="button" className="btn btn--sm" onClick={() => void result.refetch()}>Try again</button>
      </div>}
      {!date && <div className="map-request-error" role="status">
        <strong>No forecast dates are available.</strong>
        <span>Connect the Aabhas API and confirm that the model feature store is available.</span>
      </div>}

      <section className="map-workspace" aria-label="Interactive Bangladesh flood risk map">
        <div className="map-workspace__main">
          <div className="map-toolbar">
            <div className="map-toolbar__location"><span className="map-toolbar__icon"><MapPinned size={17} /></span><span><strong>{selected?.name ?? "Bangladesh"}</strong><small>{result.isError ? "Forecast unavailable" : result.isLoading || !date ? "Loading forecast data…" : selected ? `${selected.reporting} stations reporting · ${selected.elevated} elevated` : `${reporting} of ${rows.length} stations reporting · ${elevated} elevated`}</small></span></div>
            <div className="map-toolbar__controls" role="group" aria-label="Map detail level">
              <button type="button" className={level === "adm1" ? "map-level is-active" : "map-level"} onClick={() => { setLevel("adm1"); chooseRegion(null); }}>Divisions</button>
              <button type="button" className={level === "adm2" ? "map-level is-active" : "map-level"} onClick={() => { setLevel("adm2"); chooseRegion(null); }}>Districts</button>
              <button type="button" className="map-icon-button" onClick={focusBangladesh} aria-label="Reset map and filters" title="Reset map"><LocateFixed size={16} /></button>
            </div>
          </div>

          <BangladeshMap rows={rows} level={level} height={540} showStations activeBands={activeBands} selectedRegion={selectedRegion} onSelectRegion={chooseRegion} onSelectStation={(id) => navigate(`/stations/${id}`)} />

          <div className="map-band-filter" aria-label="Filter by risk level">
            <div className="map-band-filter__label"><SlidersHorizontal size={14} /> Show</div>
            {BANDS.filter((band) => band !== "unknown").map((band) => (
              <button key={band} type="button" onClick={() => toggleBand(band)} aria-pressed={activeBands.includes(band)} className={`map-filter-chip ${activeBands.includes(band) ? "is-active" : ""}`} style={{ "--chip-color": BAND_META[band].color } as React.CSSProperties}>
                <span />{bandLabel[band]}
              </button>
            ))}
            {activeBands.length > 0 && <button type="button" className="map-clear" onClick={() => setActiveBands([])}>Clear</button>}
            <span className="map-gesture">Drag to move · scroll to zoom</span>
          </div>
          <div className="map-data-note">{result.data?.source === "demo" ? "Illustrative sample values · not a live forecast" : result.data?.source === "live" ? `Forecast horizon: ${horizon} · ${date}` : "No forecast values are shown until the API responds."} <span>·</span> Area shading summarizes the highest station risk; it is not a flood extent map.</div>
        </div>

        <aside className="map-workspace__side">
          <Card className="map-side-card">
            <div className="map-side-card__heading"><div><div className="micro">Risk outlook</div><h2>{selected ? selected.name : "Areas to watch"}</h2></div><MapIcon size={18} className="accent" /></div>
            <p className="map-side-intro">{selected ? `Station outlook for ${selected.name}. Choose a station to see its forecast and river details.` : "Areas are ranked by the highest station risk in each region."}</p>
            <div className="region-list">
              {(selected ? [selected] : rankedRegions).slice(0, selected ? 1 : 6).map((region) => (
                <button type="button" key={region.name} className={`region-row ${selectedRegion === region.name ? "is-selected" : ""}`} onClick={() => chooseRegion(region.name)}>
                  <span className="region-row__dot" style={{ background: BAND_META[region.worst].color }} />
                  <span className="region-row__name">{region.name}</span>
                  <span className="region-row__meta">{region.reporting} stn</span>
                  <span className="num region-row__value" style={{ color: BAND_META[region.worst].color }}>{region.maxProbability == null ? "—" : `${pctPoints(region.maxProbability, 1)}%`}</span>
                </button>
              ))}
              {!selected && rankedRegions.length === 0 && <p className="map-empty">No regional data is available for this date.</p>}
            </div>
            {selected && <button type="button" className="map-back-link" onClick={() => chooseRegion(null)}>← All areas</button>}
          </Card>

          <Card className="map-side-card map-side-card--stations">
            <div className="map-side-card__heading"><div><div className="micro">{selected ? "Local stations" : "Station watch"}</div><h2>{activeBands.length ? "Filtered stations" : "Highest signals"}</h2></div><span className="station-count">{focusRows.filter((row) => activeBands.length === 0 || activeBands.includes(row.band)).length}</span></div>
            <div className="station-watch-list">
              {visibleRows.map((row) => <button key={row.stationId} type="button" className="station-watch-row" onClick={() => navigate(`/stations/${row.stationId}`)}>
                <span className="station-watch-row__band" style={{ background: BAND_META[row.band].color }} />
                <span className="station-watch-row__copy"><strong>{row.station.name}</strong><small>{row.station.river || row.station.district} · {row.station.district}</small></span>
                <span className="station-watch-row__risk"><strong style={{ color: BAND_META[row.band].color }}>{row.probability == null ? "—" : `${pctPoints(row.probability, 1)}%`}</strong><small>{t(`band.${row.band}`)}</small></span>
                <ArrowUpRight size={14} className="station-watch-row__arrow" />
              </button>)}
              {visibleRows.length === 0 && <p className="map-empty">No stations match these filters here.</p>}
            </div>
            <Link className="map-all-stations" to={selected ? `/stations?district=${encodeURIComponent(selected.name)}` : "/stations"}>Browse all stations <ArrowUpRight size={14} /></Link>
          </Card>
          <div className="map-trust"><strong>About this outlook</strong><p>Station forecasts are model estimates. They don’t replace FFWC alerts or local emergency advice.</p></div>
        </aside>
      </section>
    </div>
  );
}
