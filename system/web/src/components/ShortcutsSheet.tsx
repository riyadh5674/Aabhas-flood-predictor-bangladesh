import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { X } from "lucide-react";
import { useEffect } from "react";
import { useApp } from "@/state/AppProvider";

const GROUPS = [
  {
    title: "Global",
    rows: [
      { keys: ["Ctrl", "K"], label: "Open command palette" },
      { keys: ["/"], label: "Focus search" },
      { keys: ["?"], label: "Keyboard shortcuts" },
      { keys: ["Esc"], label: "Close dialog" },
    ],
  },
  {
    title: "Forecast",
    rows: [
      { keys: ["Alt", "←"], label: "Previous day" },
      { keys: ["Alt", "→"], label: "Next day" },
      { keys: ["1"], label: "24-hour horizon" },
      { keys: ["2"], label: "48-hour horizon" },
      { keys: ["3"], label: "72-hour horizon" },
    ],
  },
  {
    title: "Map",
    rows: [
      { keys: ["+ / −"], label: "Zoom in / out" },
      { keys: ["Drag"], label: "Pan the map" },
      { keys: ["Click"], label: "Select a region" },
    ],
  },
];

export function ShortcutsSheet() {
  const { shortcutsOpen, setShortcutsOpen } = useApp();
  const reduce = useReducedMotion();

  useEffect(() => {
    if (!shortcutsOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setShortcutsOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [shortcutsOpen, setShortcutsOpen]);

  return (
    <AnimatePresence>
      {shortcutsOpen ? (
        <motion.div
          className="fixed inset-0 z-[105] flex items-center justify-center px-4 no-print"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.16 }}
        >
          <div
            className="absolute inset-0"
            style={{ background: "color-mix(in srgb, var(--bg-deep) 66%, transparent)", backdropFilter: "blur(6px)" }}
            onClick={() => setShortcutsOpen(false)}
            aria-hidden="true"
          />
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-label="Keyboard shortcuts"
            className="relative w-full max-w-[520px] rounded-2xl border p-5"
            style={{ background: "var(--surface)", borderColor: "var(--border-strong)", boxShadow: "var(--shadow-2)" }}
            initial={reduce ? { opacity: 0 } : { opacity: 0, y: 14, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={reduce ? { opacity: 0 } : { opacity: 0, y: 8, scale: 0.99 }}
            transition={{ duration: 0.24, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="mb-4 flex items-start justify-between">
              <div>
                <h2 className="display text-[18px]">Keyboard shortcuts</h2>
                <p className="mt-1 text-[12.5px] muted">Move through Aabhas without leaving the keyboard.</p>
              </div>
              <button type="button" className="icon-btn -mr-1 -mt-1" onClick={() => setShortcutsOpen(false)} aria-label="Close">
                <X size={15} />
              </button>
            </div>

            <div className="space-y-5">
              {GROUPS.map((group) => (
                <div key={group.title}>
                  <div className="micro mb-2" style={{ fontSize: 9 }}>
                    {group.title}
                  </div>
                  <ul className="space-y-1">
                    {group.rows.map((row) => (
                      <li key={row.label} className="flex items-center justify-between rounded-lg px-2 py-1.5">
                        <span className="text-[13px] text-2">{row.label}</span>
                        <span className="flex items-center gap-1">
                          {row.keys.map((key) => (
                            <kbd key={key}>{key}</kbd>
                          ))}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
