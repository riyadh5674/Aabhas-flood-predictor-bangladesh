import { useMemo } from "react";

function blob(cx: number, cy: number, base: number, seed: number): string {
  const points: string[] = [];
  const N = 88;
  for (let i = 0; i < N; i += 1) {
    const a = (i / N) * Math.PI * 2;
    const r =
      base *
      (1 +
        0.16 * Math.sin(a * 3 + seed) +
        0.08 * Math.sin(a * 7 + seed * 2.1) +
        0.045 * Math.sin(a * 11 + seed * 3.7));
    const x = cx + Math.cos(a) * r * 1.45;
    const y = cy + Math.sin(a) * r * 0.82;
    points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }
  return `M${points.join("L")}Z`;
}

function TopoContours() {
  const paths = useMemo(() => {
    const out: string[] = [];
    for (let ring = 0; ring < 15; ring += 1) {
      const base = 48 + ring * 36;
      out.push(blob(880, 150, base, ring * 1.31));
      out.push(blob(-60, 640, base * 0.92, ring * 2.07 + 1.1));
      out.push(blob(520, 980, base * 1.15, ring * 0.87 + 2.4));
    }
    return out;
  }, []);

  return (
    <svg
      className="fx__topo"
      viewBox="0 0 1200 900"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
      focusable="false"
    >
      <g fill="none" stroke="var(--accent)" strokeWidth={0.7} opacity={0.14}>
        {paths.map((d, i) => (
          <path key={i} d={d} />
        ))}
      </g>
    </svg>
  );
}

export function BackgroundFX() {
  return (
    <div className="fx no-print" aria-hidden="true">
      <div className="fx__grid" />
      <div className="fx__glow fx__glow--a" />
      <div className="fx__glow fx__glow--b" />
      <TopoContours />
      <div className="fx__grain" />
    </div>
  );
}
