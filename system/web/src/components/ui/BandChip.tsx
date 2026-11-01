import { cn } from "@/lib/cn";
import { BAND_META } from "@/lib/risk";
import type { Band } from "@/lib/types";
import { useI18n } from "@/state/i18n";

export function BandChip({
  band,
  solid = false,
  showDot = true,
  className,
  label,
}: {
  band: Band;
  solid?: boolean;
  showDot?: boolean;
  className?: string;
  label?: string;
}) {
  const { t } = useI18n();
  const text = label ?? t(`band.${band}`);
  return (
    <span
      className={cn("band", `band--${band}`, solid && "band--solid", className)}
      title={BAND_META[band].guidance}
    >
      {showDot && !solid ? <span className="band__dot" /> : null}
      {text}
    </span>
  );
}

export function Chip({
  children,
  accent = false,
  dot = false,
  className,
  title,
}: {
  children: React.ReactNode;
  accent?: boolean;
  dot?: boolean;
  className?: string;
  title?: string;
}) {
  return (
    <span className={cn("chip", accent && "chip--accent", dot && "chip--dot", className)} title={title}>
      {children}
    </span>
  );
}

export function MicroLabel({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("micro", className)}>{children}</div>;
}
