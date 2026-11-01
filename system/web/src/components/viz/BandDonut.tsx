import { arc as d3Arc } from "d3-shape";
import { motion, useReducedMotion } from "motion/react";
import { BAND_META, BAND_ORDER } from "@/lib/risk";
import type { Band } from "@/lib/types";
import { useI18n } from "@/state/i18n";

const SIZE = 168;
const OUTER = 76;
const INNER = 54;

export function BandDonut({
  counts,
  total,
  onSelect,
  active,
}: {
  counts: Record<Band, number>;
  total: number;
  onSelect?: (band: Band | null) => void;
  active?: Band | null;
}) {
  const reduce = useReducedMotion();
  const { t } = useI18n();

  const arcGen = d3Arc<{ start: number; end: number }>()
    .startAngle((d) => d.start)
    .endAngle((d) => d.end)
    .innerRadius(INNER)
    .outerRadius(OUTER)
    .padAngle(0.022)
    .cornerRadius(3);

  let cursor = 0;
  const segments = BAND_ORDER.filter((b) => counts[b] > 0).map((band) => {
    const value = counts[band];
    const angle = (value / Math.max(1, total)) * Math.PI * 2;
    const seg = { band, start: cursor, end: cursor + angle, value };
    cursor += angle;
    return seg;
  });

  return (
    <div className="relative" style={{ width: SIZE, height: SIZE }}>
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} width={SIZE} height={SIZE} role="img" aria-label="Risk band distribution">
        <g transform={`translate(${SIZE / 2} ${SIZE / 2})`}>
          {segments.length === 0 ? (
            <circle r={(OUTER + INNER) / 2} fill="none" stroke="var(--surface-3)" strokeWidth={OUTER - INNER} />
          ) : null}
          {segments.map((seg) => {
            const d = arcGen(seg) ?? "";
            const color = BAND_META[seg.band].color;
            const dimmed = active != null && active !== seg.band;
            return (
              <motion.path
                key={seg.band}
                d={d}
                fill={color}
                initial={reduce ? undefined : { opacity: 0, scale: 0.9 }}
                animate={{ opacity: dimmed ? 0.25 : 1, scale: 1 }}
                transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                style={{ transformOrigin: "0 0", cursor: onSelect ? "pointer" : "default" }}
                onClick={() => onSelect?.(active === seg.band ? null : seg.band)}
              >
                <title>{`${t(`band.${seg.band}`)}: ${seg.value}`}</title>
              </motion.path>
            );
          })}
        </g>
      </svg>
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
        <div className="num text-[26px] font-semibold leading-none">{total}</div>
        <div className="micro mt-1" style={{ fontSize: 8.5 }}>
          stations
        </div>
      </div>
    </div>
  );
}

export function BandLegend({
  counts,
  total,
  active,
  onSelect,
}: {
  counts: Record<Band, number>;
  total: number;
  active?: Band | null;
  onSelect?: (band: Band | null) => void;
}) {
  const { t } = useI18n();
  return (
    <ul className="flex w-full flex-col gap-0.5">
      {BAND_ORDER.map((band) => {
        const value = counts[band] ?? 0;
        const share = total ? Math.round((value / total) * 100) : 0;
        const isActive = active === band;
        return (
          <li key={band}>
            <button
              type="button"
              onClick={() => onSelect?.(isActive ? null : band)}
              className="flex w-full items-center gap-2.5 rounded-lg px-2 py-[7px] text-left transition-colors"
              style={{ background: isActive ? "var(--surface-2)" : "transparent" }}
            >
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ background: BAND_META[band].color }}
              />
              <span className="flex-1 truncate text-[12.5px]" style={{ color: "var(--text-2)" }}>
                {t(`band.${band}`)}
              </span>
              <span className="num text-[11.5px] tabular-nums" style={{ color: "var(--text)" }}>
                {value}
              </span>
              <span className="num w-8 text-right text-[10.5px] faint">{share}%</span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
