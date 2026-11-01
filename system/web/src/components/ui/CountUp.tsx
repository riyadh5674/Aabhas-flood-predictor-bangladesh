import { useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";

const easeOutExpo = (t: number) => (t === 1 ? 1 : 1 - Math.pow(2, -10 * t));

/**
 * Animated numeric readout. Starts from zero, eases to the target, and
 * re-animates from the current value whenever the target changes.
 */
export function CountUp({
  value,
  decimals = 0,
  duration = 1.1,
  prefix = "",
  suffix = "",
  className,
}: {
  value: number | null | undefined;
  decimals?: number;
  duration?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
}) {
  const reduce = useReducedMotion();
  const [display, setDisplay] = useState(0);
  const fromRef = useRef(0);
  const frameRef = useRef<number>(0);

  useEffect(() => {
    const target = value == null || Number.isNaN(value) ? 0 : value;
    const from = fromRef.current;
    const start = performance.now();
    const total = reduce ? 0 : duration * 1000;

    cancelAnimationFrame(frameRef.current);

    const tick = (now: number) => {
      const elapsed = now - start;
      const t = total === 0 ? 1 : Math.min(1, elapsed / total);
      const next = from + (target - from) * easeOutExpo(t);
      setDisplay(next);
      if (t < 1) frameRef.current = requestAnimationFrame(tick);
      else fromRef.current = target;
    };

    frameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameRef.current);
  }, [value, duration, reduce]);

  if (value == null) return <span className={className}>—</span>;

  return (
    <span className={className}>
      {prefix}
      {display.toFixed(decimals)}
      {suffix}
    </span>
  );
}
