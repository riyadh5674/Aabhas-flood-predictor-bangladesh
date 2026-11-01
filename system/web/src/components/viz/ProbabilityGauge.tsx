import { motion, useReducedMotion } from "motion/react";
import { BAND_META } from "@/lib/risk";
import type { Band, Horizon } from "@/lib/types";
import { CountUp } from "@/components/ui/CountUp";

function polar(cx: number, cy: number, r: number, angleDeg: number): [number, number] {
  const rad = (angleDeg * Math.PI) / 180;
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
}

const START = 135;
const SWEEP = 270;

function arcPath(cx: number, cy: number, r: number, from: number, to: number): string {
  const [x1, y1] = polar(cx, cy, r, from);
  const [x2, y2] = polar(cx, cy, r, to);
  const large = Math.abs(to - from) > 180 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
}

export function ProbabilityGauge({
  probability,
  band,
  horizon,
  threshold,
  size = 232,
  compact = false,
}: {
  probability: number | null;
  band: Band;
  horizon: Horizon;
  threshold?: number | null;
  size?: number;
  compact?: boolean;
}) {
  const reduce = useReducedMotion();
  const meta = BAND_META[band];
  const frac = probability == null ? 0 : Math.max(0.001, Math.min(1, probability));
  const cx = 100;
  const cy = 100;
  const r = 80;
  const track = arcPath(cx, cy, r, START, START + SWEEP);

  const ticks = Array.from({ length: 37 }, (_, i) => {
    const t = i / 36;
    const angle = START + SWEEP * t;
    const major = i % 6 === 0;
    const [x1, y1] = polar(cx, cy, r + (major ? 9 : 5.5), angle);
    const [x2, y2] = polar(cx, cy, r + (major ? 13.5 : 8.5), angle);
    return { x1, y1, x2, y2, major };
  });

  const thresholdAngle =
    threshold != null && threshold > 0 && threshold < 1 ? START + SWEEP * threshold : null;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg viewBox="0 0 200 200" width={size} height={size} role="img" aria-label={`${meta.label} risk`}>
        <defs>
          <linearGradient id="gauge-grad" x1="0" y1="1" x2="1" y2="0">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.55" />
            <stop offset="55%" stopColor="var(--accent)" />
            <stop offset="100%" stopColor={meta.color} />
          </linearGradient>
        </defs>

        {ticks.map((tick, i) => (
          <line
            key={i}
            x1={tick.x1}
            y1={tick.y1}
            x2={tick.x2}
            y2={tick.y2}
            stroke="var(--border-strong)"
            strokeWidth={tick.major ? 1.1 : 0.6}
            strokeLinecap="round"
            opacity={tick.major ? 0.75 : 0.4}
          />
        ))}

        <path d={track} fill="none" stroke="var(--surface-3)" strokeWidth={9} strokeLinecap="round" />

        <motion.path
          d={track}
          fill="none"
          stroke="url(#gauge-grad)"
          strokeWidth={9}
          strokeLinecap="round"
          pathLength={1}
          initial={{ pathLength: 0 }}
          animate={{ pathLength: frac }}
          transition={reduce ? { duration: 0 } : { duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
          style={{ filter: `drop-shadow(0 0 6px ${meta.color}55)` }}
        />

        {thresholdAngle != null ? (
          <line
            x1={polar(cx, cy, r - 8, thresholdAngle)[0]}
            y1={polar(cx, cy, r - 8, thresholdAngle)[1]}
            x2={polar(cx, cy, r + 8, thresholdAngle)[0]}
            y2={polar(cx, cy, r + 8, thresholdAngle)[1]}
            stroke="var(--text)"
            strokeWidth={1.2}
            strokeDasharray="2 2"
            opacity={0.55}
          />
        ) : null}
      </svg>

      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center pt-1">
        <div className="micro mb-1" style={{ fontSize: 9 }}>
          {horizon}
        </div>
        <div
          className="num font-semibold leading-none"
          style={{ fontSize: compact ? 34 : 44, color: probability == null ? "var(--muted)" : "var(--text)" }}
        >
          {probability == null ? (
            "—"
          ) : (
            <>
              <CountUp value={probability * 100} decimals={1} duration={1.2} />
              <span className="text-[0.55em] font-normal muted">%</span>
            </>
          )}
        </div>
        <div className="mt-2 text-[11.5px] font-medium" style={{ color: meta.color }}>
          {meta.label}
        </div>
        {threshold != null && !compact ? (
          <div className="mt-1 num text-[9px] faint">warn ≥ {(threshold * 100).toFixed(2)}%</div>
        ) : null}
      </div>
    </div>
  );
}
