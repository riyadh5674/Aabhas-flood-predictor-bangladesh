import { useMemo } from "react";
import { ArrowRight, ArrowUpRight, Compass, MapPinned, ShieldCheck, Waves } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { useHorizonSet, useStations } from "@/lib/queries";
import { aggregateDivisions } from "@/lib/aggregate";
import { BAND_META, SEVERITY } from "@/lib/risk";
import { pctPoints } from "@/lib/format";
import type { Horizon, StationRisk } from "@/lib/types";
import { useApp } from "@/state/AppProvider";
import { useI18n } from "@/state/i18n";
import { Card } from "@/components/ui/Card";
import { BandChip } from "@/components/ui/BandChip";
import { BangladeshMap } from "@/components/map/BangladeshMap";
import { StatusPill } from "@/components/StatusPill";
import { BrandMark } from "@/components/layout/BrandMark";

const HORIZONS: Horizon[] = ["24h", "48h", "72h"];

export function OverviewPage() {
  const { t } = useI18n();
  const { date, dataMode, setHorizon } = useApp();
  const navigate = useNavigate();
  const forecast = useHorizonSet(date, dataMode);
  const stations = useStations();
  const rows = forecast["24h"].data?.rows ?? [];
  const divisions = useMemo(() => aggregateDivisions(rows), [rows]);
  const topStations = useMemo(() => [...rows].sort((a, b) => SEVERITY[b.band] - SEVERITY[a.band] || (b.probability ?? 0) - (a.probability ?? 0)).slice(0, 4), [rows]);
  const reporting = rows.filter((r) => r.probability != null).length;
  const elevated = rows.filter((r) => r.band === "high" || r.band === "very_high").length;
  const leadRow = topStations[0];
  const dateLabel = date ? new Date(`${date}T12:00:00`).toLocaleDateString("en-BD", { weekday: "long", day: "numeric", month: "long", timeZone: "Asia/Dhaka" }) : "Today";
  const source = forecast["24h"].data?.source;

  return (
    <div className="nodi-home space-y-6">
      <section className="home-hero">
        <div className="home-hero__copy">
          <div className="home-brand-line"><BrandMark compact size={19} /><span>AABHAS <i>·</i> BANGLADESH FLOOD OUTLOOK</span><StatusPill compact /></div>
          <p className="micro home-greeting">{dateLabel} <span>·</span> {t("hero.eyebrow")}</p>
          <h1 className="display">{t("home.titleA")}<br /><em>{t("home.titleB")}</em></h1>
          <p className="home-hero__description">{t("home.description")}</p>
          <div className="home-hero__actions"><Link to="/map" className="btn btn--primary">{t("home.openMap")} <ArrowRight size={16} /></Link><Link to="/stations" className="btn btn--soft"><MapPinned size={15} /> {t("home.findStation")}</Link></div>
          {source === "demo" && <div className="home-demo-note"><span /> Showing illustrative sample values. These are not live forecasts.</div>}
          {forecast["24h"].isError && <div className="map-request-error" role="alert"><strong>Forecast unavailable</strong><span>{forecast["24h"].error instanceof Error ? forecast["24h"].error.message : "Check the API connection or choose another date."}</span><button type="button" className="btn btn--sm" onClick={() => void forecast["24h"].refetch()}>Try again</button></div>}
        </div>
        <div className="home-hero__visual" aria-label="Bangladesh flood outlook map preview">
          <div className="home-hero__visual-label"><span className="live-dot" /> BANGLADESH AT A GLANCE <span className="num">{date}</span></div>
          <div className="home-map-preview"><BangladeshMap rows={rows} level="adm1" height={340} showStations={false} /></div>
          <div className="home-map-caption"><span><span className="home-map-caption__dot" /> River station coverage</span><Link to="/map">Open interactive map <ArrowUpRight size={13} /></Link></div>
        </div>
        <div className="home-hero__wave" aria-hidden="true"><Waves size={280} strokeWidth={0.6} /></div>
      </section>

      <section className="home-metrics" aria-label="Forecast summary">
        <div className="home-metric"><div className="micro">Stations reporting</div><div className="home-metric__number">{forecast["24h"].isError || forecast["24h"].isLoading || !date ? "—" : reporting}<small> / {stations.data?.length ?? "—"}</small></div><p>Gauge outlooks available</p></div>
        <div className="home-metric home-metric--risk"><div className="micro">Elevated signals</div><div className="home-metric__number" style={{ color: elevated ? "var(--band-high)" : "var(--band-low)" }}>{forecast["24h"].isError || forecast["24h"].isLoading || !date ? "—" : elevated}<small> stations</small></div><p>High or very high model risk</p></div>
        <div className="home-metric"><div className="micro">Forecast window</div><div className="home-metric__horizons">{HORIZONS.map((h) => <button key={h} type="button" onClick={() => { setHorizon(h); navigate(`/map?h=${h}`); }} aria-label={`View ${h} outlook`}>{h}</button>)}</div><p>Choose a lead time to explore</p></div>
        <div className="home-metric home-metric--source"><div className="micro">Data status</div><div className="home-metric__status"><StatusPill /></div><p>{source === "demo" ? "Sample mode · clearly labeled" : "Connection status updates automatically"}</p></div>
      </section>

      <section className="home-section-head"><div><div className="micro">{t("home.startRegion")}</div><h2>{t("home.regionTitle")}</h2><p>{t("home.regionHelp")}</p></div><Link to="/map" className="text-link">{t("home.allRegions")} <ArrowRight size={14} /></Link></section>
      <section className="division-grid" aria-label="Flood outlook by division">
        {divisions.slice(0, 8).map((division, index) => <button type="button" key={division.name} className="division-card" onClick={() => navigate(`/map?region=${encodeURIComponent(division.name)}`)} style={{ "--division-color": BAND_META[division.worst].color } as React.CSSProperties}>
          <div className="division-card__top"><span className="division-card__index">{String(index + 1).padStart(2, "0")}</span><span className="division-card__risk"><span />{t(`band.${division.worst}`)}</span></div>
          <strong>{division.name}</strong><div className="division-card__bottom"><span>{division.reporting} stations reporting</span><span className="num">{division.maxProbability == null ? "—" : `${pctPoints(division.maxProbability, 1)}%`}<ArrowUpRight size={13} /></span></div>
          <span className="division-card__meter"><i style={{ width: `${Math.max(4, Math.min(100, (division.maxProbability ?? 0) * 100))}%` }} /></span>
        </button>)}
        {divisions.length === 0 && <Card className="home-loading-card">{forecast["24h"].isError ? "Division outlooks are unavailable until the API returns predictions." : forecast["24h"].isLoading ? <><span className="home-loading-pulse" />Loading division outlooks…</> : "No division outlooks are available for this date."}</Card>}
      </section>

      <section className="home-lower-grid">
        <Card className="home-watch-card">
          <div className="home-card-heading"><div><div className="micro">Station watch · 24 hours</div><h2>{elevated ? "Signals to keep an eye on" : "Your station outlook"}</h2></div><Link to="/stations" aria-label="See all stations"><ArrowUpRight size={17} /></Link></div>
          <p className="home-card-subtitle">The strongest station-level signals in the selected outlook.</p>
          <div className="home-station-list">{topStations.map((row, index) => <StationWatchItem key={row.stationId} row={row} rank={index + 1} onOpen={() => navigate(`/stations/${row.stationId}`)} />)}{!topStations.length && <div className="home-loading">{forecast["24h"].isError ? "Station forecasts are unavailable." : forecast["24h"].isLoading ? "Preparing your station outlook…" : "No station forecasts are available for this date."}</div>}</div>
          <Link to="/stations" className="home-card-footer">Browse all {stations.data?.length ?? "stations"} <ArrowRight size={14} /></Link>
        </Card>
        <Card className="home-ready-card">
          <div className="home-ready-card__icon"><Compass size={19} /></div><div className="micro">A useful next step</div><h2>Make a plan before you need one.</h2>
          <p>Know your nearest safe, higher ground. Keep medicines and documents together, and agree how your household will stay in touch if water rises.</p>
          <div className="home-ready-card__trust"><ShieldCheck size={16} /><span>Use forecasts as context. Follow FFWC and local authority instructions for official warnings.</span></div>
          <Link to="/about#prepare" className="home-ready-card__link">Preparedness guidance <ArrowRight size={14} /></Link>
        </Card>
      </section>
      {leadRow && <div className="sr-only">Highest current station signal: {leadRow.station.name}, {pctPoints(leadRow.probability, 1)} percent at 24 hours.</div>}
    </div>
  );
}

function StationWatchItem({ row, rank, onOpen }: { row: StationRisk; rank: number; onOpen: () => void }) {
  return <button className="home-station-row" type="button" onClick={onOpen}>
    <span className="home-station-row__rank">{String(rank).padStart(2, "0")}</span><span className="home-station-row__signal" style={{ background: BAND_META[row.band].color }} />
    <span className="home-station-row__identity"><strong>{row.station.name}</strong><small>{row.station.river || row.station.district} · {row.station.district}</small></span>
    <span className="home-station-row__readout"><strong style={{ color: BAND_META[row.band].color }}>{pctPoints(row.probability, 1)}%</strong><BandChip band={row.band} /></span><ArrowUpRight size={15} className="home-station-row__arrow" />
  </button>;
}
