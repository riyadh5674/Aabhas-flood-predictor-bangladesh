import { useQuery } from "@tanstack/react-query";
import { Minus, Plus, RotateCcw } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { loadGeo, projectGeo } from "@/lib/geo";
import { useMeasure } from "@/lib/hooks";
import { aggregateRegions } from "@/lib/aggregate";
import { BAND_META, SEVERITY } from "@/lib/risk";
import { hasPlausibleCoords } from "@/lib/stations";
import type { Band, MapLevel, StationRisk } from "@/lib/types";
import { useI18n } from "@/state/i18n";

interface MapProps {
  rows: StationRisk[];
  level: MapLevel;
  height?: number;
  showStations?: boolean;
  activeBands?: Band[];
  selectedRegion?: string | null;
  onSelectRegion?: (region: string | null) => void;
  onSelectStation?: (stationId: number) => void;
}

export function BangladeshMap({
  rows,
  level,
  height = 520,
  showStations = true,
  activeBands,
  selectedRegion,
  onSelectRegion,
  onSelectStation,
}: MapProps) {
  const { ref, width } = useMeasure<HTMLDivElement>();
  const { t } = useI18n();
  const svgRef = useRef<SVGSVGElement | null>(null);

  const [view, setView] = useState({ k: 1, x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);
  const moved = useRef(false);
  const [hover, setHover] = useState<{
    region: string;
    x: number;
    y: number;
  } | null>(null);

  const geo = useQuery({
    queryKey: ["geo", level],
    queryFn: () => loadGeo(level),
    staleTime: Infinity,
  });

  const projected = useMemo(() => {
    if (!geo.data || width < 10) return null;
    return projectGeo(geo.data, width, height, 8);
  }, [geo.data, width, height]);

  const aggregates = useMemo(() => aggregateRegions(rows, level), [rows, level]);

  const regionState = (name: string) => {
    const agg = aggregates.get(name);
    if (!agg || agg.reporting === 0) return { band: "unknown" as Band, agg: null };
    return { band: agg.worst, agg };
  };

  const bandVisible = (band: Band) => !activeBands || activeBands.length === 0 || activeBands.includes(band);

  const stations = useMemo(() => {
    if (!projected || !showStations) return [];
    return rows
      .filter((r) => hasPlausibleCoords(r.station))
      .map((r) => {
        const p = projected.project(r.station.lon, r.station.lat);
        return p ? { row: r, x: p[0], y: p[1] } : null;
      })
      .filter((v): v is { row: StationRisk; x: number; y: number } => v !== null);
  }, [projected, rows, showStations]);

  const pulses = useMemo(() => {
    if (!projected) return [];
    return projected.regions
      .map((region) => {
        const { band } = regionState(region.name);
        if (band !== "very_high" && band !== "high") return null;
        return { name: region.name, x: region.centroid[0], y: region.centroid[1], band };
      })
      .filter(
        (v): v is { name: string; x: number; y: number; band: "high" | "very_high" } => v !== null,
      );
  }, [projected, aggregates]);

  const zoomBy = (factor: number) => {
    setView((v) => {
      const k = Math.min(4.5, Math.max(1, v.k * factor));
      return { k, x: v.x * (k / v.k), y: v.y * (k / v.k) };
    });
  };

  const onPointerDown = (event: React.PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return;
    (event.currentTarget as SVGSVGElement).setPointerCapture(event.pointerId);
    drag.current = { x: event.clientX, y: event.clientY, vx: view.x, vy: view.y };
    moved.current = false;
  };
  const onPointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    if (drag.current) {
      const dx = event.clientX - drag.current.x;
      const dy = event.clientY - drag.current.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) moved.current = true;
      setView((v) => ({ ...v, x: drag.current!.vx + dx, y: drag.current!.vy + dy }));
    }
  };
  const onPointerUp = () => {
    drag.current = null;
  };

  const hovered = hover ? regionState(hover.region) : null;

  return (
    <div
      ref={ref}
      className="map-shell"
      style={{ height, cursor: drag.current ? "grabbing" : "default" }}
      onWheel={(event) => {
        if (!onSelectRegion) return;
        event.preventDefault();
        zoomBy(event.deltaY < 0 ? 1.12 : 1 / 1.12);
      }}
    >
      {geo.isError ? (
        <div className="map-load-error"><div className="micro">Map unavailable</div><p>We couldn’t load the Bangladesh boundary data.</p><button type="button" className="btn btn--sm" onClick={() => void geo.refetch()}>Try again</button></div>
      ) : projected ? (
        <svg
          ref={svgRef}
          className="map-svg"
          width={width}
          height={height}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={() => {
            onPointerUp();
            setHover(null);
          }}
          role="group"
          aria-label={`Bangladesh flood risk map by ${level === "adm1" ? "division" : "district"}`}
        >
          <defs>
            <filter id="map-glow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <g
            transform={`translate(${view.x} ${view.y}) scale(${view.k})`}
            style={{ transition: drag.current ? "none" : "transform 0.35s cubic-bezier(0.16,1,0.3,1)" }}
          >
            <path
              d={projected.graticule}
              fill="none"
              stroke="var(--border)"
              strokeWidth={0.5}
              opacity={0.5}
              pointerEvents="none"
            />

            {projected.regions.map((region) => {
              const { band, agg } = regionState(region.name);
              const visible = bandVisible(band);
              const isSelected = selectedRegion === region.name;
              const isHovered = hover?.region === region.name;
              const fill = band === "unknown" ? "var(--surface-3)" : BAND_META[band].color;
              const opacity = !visible ? 0.06 : band === "unknown" ? 0.32 : isHovered || isSelected ? 1 : 0.78;
              return (
                <path
                  key={region.name}
                  d={region.path}
                  className={`map-region ${!visible ? "map-region--dim" : ""} ${
                    isSelected ? "map-region--active" : ""
                  }`}
                  fill={fill}
                  fillOpacity={opacity}
                  stroke={isHovered ? "var(--text)" : undefined}
                  role="button"
                  tabIndex={0}
                  aria-label={`Show ${region.name} outlook`}
                  onPointerEnter={(event) => {
                    if (drag.current) return;
                    const rect = event.currentTarget.ownerSVGElement?.getBoundingClientRect();
                    setHover({
                      region: region.name,
                      x: rect ? event.clientX - rect.left : 0,
                      y: rect ? event.clientY - rect.top : 0,
                    });
                  }}
                  onPointerMove={(event) => {
                    if (!hover || hover.region !== region.name) return;
                    const rect = event.currentTarget.ownerSVGElement?.getBoundingClientRect();
                    setHover({
                      region: region.name,
                      x: rect ? event.clientX - rect.left : 0,
                      y: rect ? event.clientY - rect.top : 0,
                    });
                  }}
                  onClick={() => {
                    if (moved.current) return;
                    onSelectRegion?.(isSelected ? null : region.name);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onSelectRegion?.(isSelected ? null : region.name);
                    }
                  }}
                >
                  <title>
                    {`${region.name} — ${agg ? t(`band.${agg.worst}`) : t("map.noStation")}`}
                  </title>
                </path>
              );
            })}

            {pulses.map((pulse) => (
              <g key={`pulse-${pulse.name}`} pointerEvents="none">
                <circle
                  className="pulse-ring"
                  cx={pulse.x}
                  cy={pulse.y}
                  r={3}
                  stroke={BAND_META[pulse.band].color}
                  opacity={0.5}
                />
                <circle
                  className="pulse-ring"
                  cx={pulse.x}
                  cy={pulse.y}
                  r={3}
                  stroke={BAND_META[pulse.band].color}
                  opacity={0.35}
                  style={{ animationDelay: "1.2s" }}
                />
              </g>
            ))}

            {level === "adm1"
              ? projected.regions.map((region) => (
                  <text
                    key={`label-${region.name}`}
                    x={region.centroid[0]}
                    y={region.centroid[1]}
                    textAnchor="middle"
                    fontSize={9.5}
                    fontFamily="var(--font-mono)"
                    letterSpacing="0.08em"
                    fill="var(--text)"
                    opacity={0.72}
                    pointerEvents="none"
                    style={{ textTransform: "uppercase" }}
                  >
                    {region.name}
                  </text>
                ))
              : null}

            {stations.map(({ row, x, y }) => (
              <circle
                key={row.stationId}
                className="map-station"
                cx={x}
                cy={y}
                r={2.4}
                fill={row.band === "unknown" ? "var(--faint)" : BAND_META[row.band].color}
                opacity={bandVisible(row.band) ? 1 : 0.12}
                stroke="var(--map-water)"
                role={onSelectStation ? "button" : undefined}
                tabIndex={onSelectStation ? 0 : undefined}
                aria-label={onSelectStation ? `Open ${row.station.name} station forecast` : undefined}
                onClick={(event) => {
                  event.stopPropagation();
                  if (onSelectStation) onSelectStation(row.stationId);
                  else onSelectRegion?.(regionOf(row, level));
                }}
                onKeyDown={(event) => {
                  if (onSelectStation && (event.key === "Enter" || event.key === " ")) {
                    event.preventDefault();
                    event.stopPropagation();
                    onSelectStation(row.stationId);
                  }
                }}
              >
                <title>{`${row.station.name} · ${row.probability == null ? "—" : (row.probability * 100).toFixed(1) + "%"}`}</title>
              </circle>
            ))}
          </g>
        </svg>
      ) : (
        <div className="grid h-full place-items-center">
          <div className="skel h-3 w-32" />
        </div>
      )}

      {/* zoom controls */}
      <div className="absolute bottom-4 left-4 flex flex-col gap-1 no-print">
        <button type="button" className="icon-btn" style={controlStyle} onClick={() => zoomBy(1.35)} aria-label="Zoom in">
          <Plus size={15} />
        </button>
        <button type="button" className="icon-btn" style={controlStyle} onClick={() => zoomBy(1 / 1.35)} aria-label="Zoom out">
          <Minus size={15} />
        </button>
        <button
          type="button"
          className="icon-btn"
          style={controlStyle}
          onClick={() => setView({ k: 1, x: 0, y: 0 })}
          aria-label="Reset view"
        >
          <RotateCcw size={14} />
        </button>
      </div>

      {hover && hovered ? (
        <div
          className="tip no-print"
          style={{
            position: "absolute",
            left: Math.min(hover.x + 14, width - 190),
            top: Math.max(hover.y - 10, 8),
            width: 176,
          }}
        >
          <div className="flex items-center justify-between gap-2">
            <span className="text-[12.5px] font-medium">{hover.region}</span>
            {hovered.agg ? (
              <span className="num text-[11px]" style={{ color: BAND_META[hovered.band].color }}>
                {(hovered.agg.maxProbability! * 100).toFixed(1)}%
              </span>
            ) : null}
          </div>
          {hovered.agg ? (
            <div className="mt-1.5 flex items-center justify-between">
              <span className="text-[11px]" style={{ color: BAND_META[hovered.band].color }}>
                {t(`band.${hovered.band}`)}
              </span>
              <span className="num text-[10.5px] faint">{hovered.agg.reporting} stn</span>
            </div>
          ) : (
            <div className="mt-1 text-[11px] muted">{t("map.noStation")}</div>
          )}
        </div>
      ) : null}
    </div>
  );
}

const controlStyle: React.CSSProperties = {
  background: "color-mix(in srgb, var(--bg-deep) 78%, transparent)",
  border: "1px solid var(--border)",
  backdropFilter: "blur(10px)",
};

function regionOf(row: StationRisk, level: MapLevel): string {
  return level === "adm1" ? row.station.divisionKey : row.station.districtKey;
}
