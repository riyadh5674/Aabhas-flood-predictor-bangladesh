import { line, area, curveMonotoneX } from "d3-shape";
import { motion, useReducedMotion } from "motion/react";
import { useId } from "react";

export function Sparkline({
  values,
  width = 120,
  height = 34,
  color = "var(--accent)",
  showArea = true,
  strokeWidth = 1.6,
}: {
  values: (number | null)[];
  width?: number;
  height?: number;
  color?: string;
  showArea?: boolean;
  strokeWidth?: number;
}) {
  const reduce = useReducedMotion();
  const gid = useId().replace(/[:]/g, "");
  const clean = values.filter((v): v is number => v != null);
  if (clean.length < 2) {
    return (
      <svg width={width} height={height} aria-hidden="true">
        <line
          x1={0}
          y1={height / 2}
          x2={width}
          y2={height / 2}
          stroke="var(--border)"
          strokeDasharray="3 4"
        />
      </svg>
    );
  }

  const pad = strokeWidth + 1;
  const max = Math.max(...clean, 0.0001);
  const points: [number, number][] = values.map((v, i) => [
    (i / Math.max(1, values.length - 1)) * (width - pad * 2) + pad,
    height - pad - ((v ?? 0) / max) * (height - pad * 2),
  ]);

  const lineGen = line<[number, number]>()
    .x((d) => d[0])
    .y((d) => d[1])
    .curve(curveMonotoneX);
  const areaGen = area<[number, number]>()
    .x((d) => d[0])
    .y0(height)
    .y1((d) => d[1])
    .curve(curveMonotoneX);

  const d = lineGen(points) ?? "";
  const a = areaGen(points) ?? "";
  const last = points[points.length - 1];

  return (
    <svg width={width} height={height} aria-hidden="true" className="overflow-visible">
      <defs>
        <linearGradient id={`spark-${gid}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.28" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {showArea ? <path d={a} fill={`url(#spark-${gid})`} /> : null}
      <motion.path
        d={d}
        className="spark-line"
        stroke={color}
        strokeWidth={strokeWidth}
        initial={reduce ? undefined : { pathLength: 0, opacity: 0 }}
        animate={reduce ? undefined : { pathLength: 1, opacity: 1 }}
        transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1] }}
      />
      <circle cx={last[0]} cy={last[1]} r={2.2} fill={color} />
    </svg>
  );
}
