import { area, curveMonotoneX, line } from "d3-shape";
import { motion, useReducedMotion } from "motion/react";
import { useMemo, useState } from "react";
import { useMeasure } from "@/lib/hooks";
import { dayMonth } from "@/lib/format";
import { BAND_META } from "@/lib/risk";
import type { Band } from "@/lib/types";
import type { TrendPoint } from "@/lib/queries";

export function TrendChart({
  points,
  threshold,
  height = 210,
  color = "var(--accent)",
}: {
  points: TrendPoint[];
  threshold?: number | null;
  height?: number;
  color?: string;
}) {
  const { ref, width } = useMeasure<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const reduce = useReducedMotion();

  const margin = { top: 16, right: 14, bottom: 26, left: 42 };
  const w = Math.max(width, 260);
  const plotW = w - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const { maxY, xs, ys, linePath, areaPath, ticks } = useMemo(() => {
    const probs = points.map((p) => p.probability).filter((p): p is number => p != null);
    const peak = probs.length ? Math.max(...probs) : 0;
    const rawMax = Math.max(peak * 1.35, (threshold ?? 0) * 2.2, 0.05);
    const step = rawMax > 0.4 ? 0.2 : rawMax > 0.2 ? 0.1 : rawMax > 0.1 ? 0.05 : 0.02;
    const yMax = Math.min(1, Math.ceil(rawMax / step) * step);
    const xScale = (i: number) => margin.left + (i / Math.max(1, points.length - 1)) * plotW;
    const yScale = (v: number) => margin.top + (1 - v / yMax) * plotH;

    const coords: [number, number][] = points.map((p, i) => [xScale(i), yScale(p.probability ?? 0)]);
    const lineGen = line<[number, number]>()
      .x((d) => d[0])
      .y((d) => d[1])
      .curve(curveMonotoneX);
    const areaGen = area<[number, number]>()
      .x((d) => d[0])
      .y0(margin.top + plotH)
      .y1((d) => d[1])
      .curve(curveMonotoneX);

    const tickValues: number[] = [];
    for (let v = 0; v <= yMax + 1e-9; v += step) tickValues.push(Number(v.toFixed(4)));

    return {
      maxY: yMax,
      xs: coords.map((c) => c[0]),
      ys: coords.map((c) => c[1]),
      linePath: lineGen(coords) ?? "",
      areaPath: areaGen(coords) ?? "",
      ticks: tickValues,
    };
  }, [points, plotW, plotH, margin.left, margin.top, threshold]);

  const yOf = (v: number) => margin.top + (1 - v / maxY) * plotH;
  const xOf = (i: number) => margin.left + (i / Math.max(1, points.length - 1)) * plotW;

  const onMove = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!points.length) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const ratio = (x - margin.left) / Math.max(1, plotW);
    const index = Math.round(ratio * (points.length - 1));
    setHover(Math.max(0, Math.min(points.length - 1, index)));
  };

  const hovered = hover != null ? points[hover] : null;
  const hoveredBand = (hovered?.band ?? "unknown") as Band;

  return (
    <div ref={ref} className="relative w-full select-none" style={{ height }}>
      {width > 0 ? (
        <svg
          width={w}
          height={height}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
          style={{ touchAction: "pan-y" }}
          role="img"
          aria-label="Flood probability trend"
        >
          <defs>
            <linearGradient id="trend-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.26" />
              <stop offset="100%" stopColor={color} stopOpacity="0" />
            </linearGradient>
          </defs>

          {ticks.map((tick) => (
            <g key={tick}>
              <line
                x1={margin.left}
                x2={w - margin.right}
                y1={yOf(tick)}
                y2={yOf(tick)}
                stroke="var(--hairline)"
                strokeDasharray={tick === 0 ? undefined : "3 5"}
              />
              <text
                x={margin.left - 8}
                y={yOf(tick) + 3.5}
                textAnchor="end"
                fontSize={10}
                fill="var(--faint)"
                fontFamily="var(--font-mono)"
              >
                {(tick * 100).toFixed(0)}%
              </text>
            </g>
          ))}

          {threshold != null && threshold > 0 && threshold <= maxY ? (
            <g>
              <line
                x1={margin.left}
                x2={w - margin.right}
                y1={yOf(threshold)}
                y2={yOf(threshold)}
                stroke="var(--band-high)"
                strokeWidth={1}
                strokeDasharray="5 4"
                opacity={0.7}
              />
              <text
                x={w - margin.right}
                y={yOf(threshold) - 5}
                textAnchor="end"
                fontSize={9.5}
                fill="var(--band-high)"
                fontFamily="var(--font-mono)"
              >
                warn {(threshold * 100).toFixed(2)}%
              </text>
            </g>
          ) : null}

          <path d={areaPath} fill="url(#trend-fill)" />
          <motion.path
            d={linePath}
            fill="none"
            stroke={color}
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            pathLength={1}
            initial={reduce ? undefined : { pathLength: 0 }}
            animate={reduce ? undefined : { pathLength: 1 }}
            transition={{ duration: 1.3, ease: [0.16, 1, 0.3, 1] }}
          />

          {hover != null && hovered ? (
            <g>
              <line
                x1={xs[hover]}
                x2={xs[hover]}
                y1={margin.top}
                y2={margin.top + plotH}
                stroke="var(--border-strong)"
                strokeDasharray="3 3"
              />
              <circle
                cx={xs[hover]}
                cy={ys[hover]}
                r={4.5}
                fill={BAND_META[hoveredBand].color}
                stroke="var(--surface)"
                strokeWidth={2}
              />
            </g>
          ) : null}

          {points.map((p, i) => {
            if (i !== points.length - 1) return null;
            return (
              <circle
                key={p.date}
                cx={xs[i]}
                cy={ys[i]}
                r={4}
                fill={BAND_META[p.band].color}
                stroke="var(--surface)"
                strokeWidth={2}
              />
            );
          })}

          {[0, Math.floor((points.length - 1) / 2), points.length - 1]
            .filter((v, i, arr) => arr.indexOf(v) === i && v >= 0)
            .map((i) => (
              <text
                key={i}
                x={xOf(i)}
                y={height - 7}
                textAnchor={i === 0 ? "start" : i === points.length - 1 ? "end" : "middle"}
                fontSize={10}
                fill="var(--faint)"
                fontFamily="var(--font-mono)"
              >
                {dayMonth(points[i].date)}
              </text>
            ))}
        </svg>
      ) : null}

      {hovered ? (
        <div
          className="tip"
          style={{
            position: "absolute",
            left: Math.min(Math.max(xs[hover!] - 60, 0), w - 150),
            top: Math.max(ys[hover!] - 58, 0),
            width: 140,
          }}
        >
          <div className="micro mb-1" style={{ fontSize: 8.5 }}>
            {dayMonth(hovered.date)}
          </div>
          <div className="num text-[15px] font-semibold" style={{ color: BAND_META[hoveredBand].color }}>
            {hovered.probability == null ? "—" : `${(hovered.probability * 100).toFixed(2)}%`}
          </div>
          <div className="mt-0.5 text-[10.5px] muted">{BAND_META[hoveredBand].label}</div>
        </div>
      ) : null}
    </div>
  );
}
