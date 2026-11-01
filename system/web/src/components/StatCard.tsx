import { ArrowDownRight, ArrowUpRight, Minus, type LucideIcon } from "lucide-react";
import { Sparkline } from "@/components/viz/Sparkline";
import { cn } from "@/lib/cn";

export function StatCard({
  label,
  value,
  unit,
  hint,
  icon: Icon,
  color = "var(--accent)",
  spark,
  trend,
  className,
}: {
  label: string;
  value: React.ReactNode;
  unit?: string;
  hint?: React.ReactNode;
  icon?: LucideIcon;
  color?: string;
  spark?: (number | null)[];
  trend?: "up" | "down" | "flat";
  className?: string;
}) {
  const TrendIcon = trend === "up" ? ArrowUpRight : trend === "down" ? ArrowDownRight : Minus;
  return (
    <div className={cn("card card--hover group overflow-hidden p-4", className)}>
      <div className="flex items-start justify-between gap-3">
        <div className="micro" style={{ fontSize: 9.5 }}>
          {label}
        </div>
        {Icon ? (
          <span
            className="grid h-7 w-7 shrink-0 place-items-center rounded-lg transition-transform duration-300 group-hover:scale-105"
            style={{ background: `color-mix(in srgb, ${color} 12%, transparent)`, color }}
          >
            <Icon size={14} />
          </span>
        ) : null}
      </div>

      <div className="mt-3 flex items-end gap-1.5">
        <div className="num text-[28px] font-semibold leading-none tracking-[-0.03em]">{value}</div>
        {unit ? <div className="mb-0.5 text-[12px] muted">{unit}</div> : null}
        {trend ? (
          <span
            className="mb-1 ml-auto flex items-center gap-0.5 text-[10.5px]"
            style={{ color: trend === "up" ? "var(--band-high)" : trend === "down" ? "var(--band-low)" : "var(--faint)" }}
          >
            <TrendIcon size={11} />
          </span>
        ) : null}
      </div>

      <div className="mt-3 flex items-end justify-between gap-3">
        {hint ? <div className="min-w-0 text-[11.5px] leading-snug muted">{hint}</div> : <div />}
        {spark && spark.length > 1 ? (
          <div className="shrink-0 opacity-90">
            <Sparkline values={spark} width={84} height={26} color={color} strokeWidth={1.4} />
          </div>
        ) : null}
      </div>
    </div>
  );
}
