import { motion, useReducedMotion } from "motion/react";
import { useId } from "react";
import { cn } from "@/lib/cn";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  hint?: string;
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  className,
  ariaLabel,
}: {
  value: T;
  options: SegmentedOption<T>[];
  onChange: (value: T) => void;
  className?: string;
  ariaLabel?: string;
}) {
  const id = useId();
  const reduce = useReducedMotion();

  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={cn(
        "inline-flex items-center gap-0.5 rounded-xl border p-0.5",
        className,
      )}
      style={{ background: "var(--surface-2)", borderColor: "var(--border)" }}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            role="tab"
            aria-selected={active}
            type="button"
            onClick={() => onChange(option.value)}
            className="relative rounded-[9px] px-3 py-1.5 text-[12.5px] font-medium transition-colors"
            style={{ color: active ? "var(--accent-ink)" : "var(--text-2)" }}
          >
            {active ? (
              <motion.span
                layoutId={`seg-${id}`}
                className="absolute inset-0 rounded-[9px]"
                style={{ background: "var(--accent)" }}
                transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 520, damping: 38 }}
              />
            ) : null}
            <span className="relative z-10 whitespace-nowrap">
              {option.label}
              {option.hint ? (
                <span className="ml-1.5 opacity-60">{option.hint}</span>
              ) : null}
            </span>
          </button>
        );
      })}
    </div>
  );
}
