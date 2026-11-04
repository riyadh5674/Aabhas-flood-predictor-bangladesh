import { ChevronLeft, ChevronRight } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/cn";
import { format, parseISO, startOfMonth, endOfMonth, startOfWeek, endOfWeek, addMonths, isSameDay, isSameMonth } from "date-fns";

interface DatePickerProps {
  value: string;
  availableDates: string[];
  onChange: (date: string) => void;
  disabled?: boolean;
  minDate?: string;
  maxDate?: string;
  className?: string;
}

function parseDateOrNull(d: string): Date | null {
  if (!d) return null;
  try {
    const parsed = parseISO(d);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  } catch {
    return null;
  }
}

function safeParseISO(d: string): Date {
  return parseDateOrNull(d) ?? new Date();
}

function safeFormatISO(d: string, fmt: string): string {
  const parsed = parseDateOrNull(d);
  if (!parsed) return "";
  try {
    return format(parsed, fmt);
  } catch {
    return "";
  }
}

export function DatePicker({
  value,
  availableDates,
  onChange,
  disabled = false,
  minDate,
  maxDate,
  className,
}: DatePickerProps) {
  const [open, setOpen] = useState(false);
  const [viewDate, setViewDate] = useState(() => safeParseISO(value));
  const popoverRef = useRef<HTMLDivElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const availableSet = useMemo(() => new Set(availableDates), [availableDates]);

  // Keep the visible month in sync when the selected date changes (including
  // when AppProvider swaps in the latest feature date after /dates loads).
  useEffect(() => {
    const parsed = parseDateOrNull(value);
    if (parsed) setViewDate(parsed);
  }, [value]);

  const isAvailable = useCallback((dateStr: string) => {
    if (availableSet.size === 0) return true;
    return availableSet.has(dateStr);
  }, [availableSet]);

  const isDisabled = useCallback((dateStr: string) => {
    if (disabled) return true;
    if (!isAvailable(dateStr)) return true;
    if (minDate && dateStr < minDate) return true;
    if (maxDate && dateStr > maxDate) return true;
    return false;
  }, [disabled, isAvailable, minDate, maxDate]);

  const handleClickOutside = useCallback((event: MouseEvent) => {
    if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
  }, []);

  useEffect(() => {
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [handleClickOutside]);

  const isSelected = (dateStr: string) => value === dateStr;

  const goToMonth = (delta: number) => {
    setViewDate(addMonths(viewDate, delta));
  };

  const selectDate = (dateStr: string) => {
    onChange(dateStr);
    setOpen(false);
  };

  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const monthStart = startOfMonth(viewDate);
  const monthEnd = endOfMonth(viewDate);
  const calendarStart = startOfWeek(monthStart, { weekStartsOn: 1 });
  const calendarEnd = endOfWeek(monthEnd, { weekStartsOn: 1 });

  const weeks: Date[][] = [];
  let week: Date[] = [];
  let current = calendarStart;

  while (current <= calendarEnd) {
    week.push(current);
    if (week.length === 7) {
      weeks.push(week);
      week = [];
    }
    current = new Date(current.getTime() + 86400000);
  }
  if (week.length > 0) weeks.push(week);

  const monthLabel = format(viewDate, "MMMM yyyy");

  const formatDateKey = (date: Date) => format(date, "yyyy-MM-dd");

  return (
    <div className="relative" ref={wrapperRef}>
      <button
        type="button"
        className={cn(
          "flex items-center gap-1.5 rounded-xl border px-2 py-1.5 transition-colors",
          "hover:border-[var(--border-strong)]",
          disabled && "opacity-50 cursor-not-allowed",
          className
        )}
        style={{
          borderColor: "var(--border)",
          background: "var(--surface-2)",
        }}
        onClick={() => !disabled && setOpen((v) => !v)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label="Select forecast date"
        disabled={disabled}
      >
        <span className="num whitespace-nowrap text-[12px] font-medium tabular-nums">
          {safeFormatISO(value, "d MMM yyyy") || "Select date"}
        </span>
      </button>

      {open && (
        <div
          ref={popoverRef}
          className="fixed z-50 mt-2 min-w-[280px] rounded-xl border bg-[var(--bg)] shadow-xl overflow-hidden"
          style={{ borderColor: "var(--border)", boxShadow: "0 20px 40px -12px var(--shadow)" }}
          role="dialog"
          aria-label="Select forecast date"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-3 border-b" style={{ borderColor: "var(--border)" }}>
            <button
              type="button"
              className="icon-btn h-8 w-8"
              onClick={() => goToMonth(-1)}
              aria-label="Previous month"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="flex-1 text-center text-[13px] font-medium">{monthLabel}</span>
            <button
              type="button"
              className="icon-btn h-8 w-8"
              onClick={() => goToMonth(1)}
              aria-label="Next month"
            >
              <ChevronRight size={16} />
            </button>
          </div>

          {/* Weekday headers */}
          <div className="grid grid-cols-7 px-3 pb-1 text-center text-[10px] font-medium text-[var(--muted)]" role="row">
            {["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((day) => (
              <div key={day} className="h-7" style={{ fontSize: 10 }}>{day}</div>
            ))}
          </div>

          {/* Calendar grid */}
          <div className="grid grid-cols-7 gap-0.5 px-2 pb-3" role="grid" aria-label="Calendar">
            {weeks.map((week, wIndex) => (
              <div key={wIndex} className="contents">
                {week.map((day, dIndex) => {
                  const dateStr = formatDateKey(day);
                  const isCurrentMonth = isSameMonth(day, viewDate);
                  const disabled = isDisabled(dateStr);
                  const selected = isSelected(dateStr);
                  const isToday = isSameDay(day, today);
                  const available = isAvailable(dateStr);

                  return (
                    <button
                      key={formatDateKey(day)}
                      type="button"
                      role="gridcell"
                      onClick={() => !disabled && selectDate(dateStr)}
                      disabled={disabled}
                      aria-selected={selected}
                      aria-disabled={disabled}
                      aria-label={format(day, "EEEE, d MMMM yyyy")}
                      className={cn(
                        "relative h-9 w-full rounded-xl text-[12px] font-medium transition-all",
                        !isCurrentMonth && "text-[var(--muted)] opacity-40",
                        disabled && "opacity-30 cursor-not-allowed",
                        selected && "bg-[var(--accent)] text-[var(--accent-fg)] font-semibold",
                        isToday && !selected && "ring-2 ring-[var(--accent)]",
                        available && !selected && !isToday && "hover:bg-[var(--surface-3)]",
                      )}
                      style={{
                        color: disabled ? "var(--muted)" : selected ? "var(--accent-fg)" : isToday ? "var(--accent)" : "var(--text)",
                      }}
                      tabIndex={disabled ? -1 : 0}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          if (!disabled) selectDate(dateStr);
                        }
                      }}
                    >
                      {format(day, "d")}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>

          {/* Quick actions */}
          <div className="flex items-center justify-between p-3 border-t" style={{ borderColor: "var(--border)" }}>
            <button
              type="button"
              className="text-[12px] text-[var(--accent)] hover:underline"
              onClick={() => selectDate(format(today, "yyyy-MM-dd"))}
              disabled={isDisabled(format(today, "yyyy-MM-dd"))}
            >
              Today
            </button>
            <button
              type="button"
              className="text-[12px] text-[var(--muted)] hover:text-[var(--text)]"
              onClick={() => setOpen(false)}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export function DatePickerTrigger({
  value,
  availableDates,
  onChange,
  disabled = false,
  minDate,
  maxDate,
  className,
}: DatePickerProps) {
  return <DatePicker value={value} availableDates={availableDates} onChange={onChange} disabled={disabled} minDate={minDate} maxDate={maxDate} className={className} />;
}