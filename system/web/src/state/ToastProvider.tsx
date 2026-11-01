import { AnimatePresence, motion } from "motion/react";
import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";
import { CheckCircle2, Info, TriangleAlert, X } from "lucide-react";

type Tone = "default" | "success" | "danger" | "info";

export interface Toast {
  id: number;
  title: string;
  description?: string;
  tone: Tone;
}

interface ToastContextValue {
  toast: (input: { title: string; description?: string; tone?: Tone; duration?: number }) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

const TONE_ICON = {
  default: Info,
  info: Info,
  success: CheckCircle2,
  danger: TriangleAlert,
} as const;

const TONE_COLOR = {
  default: "var(--accent)",
  info: "var(--accent)",
  success: "var(--band-low)",
  danger: "var(--band-very-high)",
} as const;

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const counter = useRef(0);

  const dismiss = useCallback((id: number) => {
    setItems((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const toast = useCallback<ToastContextValue["toast"]>(
    ({ title, description, tone = "default", duration = 4200 }) => {
      counter.current += 1;
      const id = counter.current;
      setItems((prev) => [...prev.slice(-3), { id, title, description, tone }]);
      window.setTimeout(() => dismiss(id), duration);
    },
    [dismiss],
  );

  const value = useMemo(() => ({ toast }), [toast]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed bottom-5 right-5 z-[120] flex w-[min(360px,calc(100vw-2.5rem))] flex-col gap-2 no-print">
        <AnimatePresence initial={false}>
          {items.map((item) => {
            const Icon = TONE_ICON[item.tone];
            return (
              <motion.div
                key={item.id}
                layout
                initial={{ opacity: 0, y: 14, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 8, scale: 0.98, transition: { duration: 0.16 } }}
                transition={{ type: "spring", stiffness: 420, damping: 32 }}
                className="pointer-events-auto flex items-start gap-3 rounded-xl border p-3.5"
                style={{
                  background: "color-mix(in srgb, var(--bg-deep) 92%, transparent)",
                  borderColor: "var(--border-strong)",
                  boxShadow: "var(--shadow-2)",
                  backdropFilter: "blur(14px)",
                }}
                role="status"
              >
                <Icon size={17} style={{ color: TONE_COLOR[item.tone], marginTop: 1 }} />
                <div className="min-w-0 flex-1">
                  <p className="text-[13px] font-medium leading-snug">{item.title}</p>
                  {item.description ? (
                    <p className="mt-0.5 text-[12px] leading-snug muted">{item.description}</p>
                  ) : null}
                </div>
                <button
                  type="button"
                  onClick={() => dismiss(item.id)}
                  className="icon-btn -mr-1 -mt-1 h-6 w-6"
                  aria-label="Dismiss notification"
                >
                  <X size={13} />
                </button>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside ToastProvider");
  return ctx;
}
