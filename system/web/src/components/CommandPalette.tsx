import {
  ArrowRight,
  CornerDownLeft,
  FlaskConical,
  Keyboard,
  Languages,
  Link2,
  Map as MapIcon,
  Moon,
  Search,
  Sun,
  type LucideIcon,
} from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useStations } from "@/lib/queries";
import { cn } from "@/lib/cn";
import { useApp } from "@/state/AppProvider";
import { useI18n } from "@/state/i18n";
import { useToast } from "@/state/ToastProvider";
import { NAV_ITEMS } from "./layout/nav";

interface PaletteItem {
  id: string;
  group: string;
  label: string;
  hint?: string;
  icon: LucideIcon;
  run: () => void;
}

export function CommandPalette() {
  const { paletteOpen, setPaletteOpen, toggleTheme, theme, setLang, lang, dataMode, setDataMode, setShortcutsOpen } =
    useApp();
  const { t } = useI18n();
  const { toast } = useToast();
  const navigate = useNavigate();
  const reduce = useReducedMotion();
  const { data: stations = [] } = useStations();

  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (paletteOpen) {
      setQuery("");
      setActive(0);
      window.setTimeout(() => inputRef.current?.focus(), 30);
    }
  }, [paletteOpen]);

  const items = useMemo<PaletteItem[]>(() => {
    const pages: PaletteItem[] = NAV_ITEMS.map((item) => ({
      id: `page-${item.to}`,
      group: "Navigate",
      label: t(item.key),
      hint: item.to,
      icon: item.icon,
      run: () => navigate(item.to),
    }));

    const actions: PaletteItem[] = [
      {
        id: "action-theme",
        group: "Actions",
        label: theme === "dark" ? "Switch to light theme" : "Switch to dark theme",
        icon: theme === "dark" ? Sun : Moon,
        run: toggleTheme,
      },
      {
        id: "action-lang",
        group: "Actions",
        label: lang === "en" ? "বাংলায় দেখুন" : "View in English",
        icon: Languages,
        run: () => setLang(lang === "en" ? "bn" : "en"),
      },
      {
        id: "action-demo",
        group: "Actions",
        label: dataMode === "demo" ? "Turn off demo mode" : "Turn on demo mode",
        hint: dataMode === "demo" ? "currently demo" : "currently auto",
        icon: FlaskConical,
        run: () => setDataMode(dataMode === "demo" ? "auto" : "demo"),
      },
      {
        id: "action-link",
        group: "Actions",
        label: "Copy shareable link",
        icon: Link2,
        run: () => {
          void navigator.clipboard
            .writeText(window.location.href)
            .then(() => toast({ title: "Link copied", tone: "success" }))
            .catch(() => toast({ title: "Clipboard unavailable", tone: "danger" }));
        },
      },
      {
        id: "action-keys",
        group: "Actions",
        label: "Keyboard shortcuts",
        icon: Keyboard,
        run: () => setShortcutsOpen(true),
      },
      {
        id: "action-map",
        group: "Actions",
        label: "Open national risk map",
        icon: MapIcon,
        run: () => navigate("/map"),
      },
    ];

    const stationItems: PaletteItem[] = stations.map((station) => ({
      id: `station-${station.id}`,
      group: "Stations",
      label: station.name,
      hint: [station.river, station.district, station.division].filter(Boolean).join(" · "),
      icon: ArrowRight,
      run: () => navigate(`/stations/${station.id}`),
    }));

    return [...pages, ...actions, ...stationItems];
  }, [dataMode, lang, navigate, setDataMode, setLang, setShortcutsOpen, stations, t, theme, toast, toggleTheme]);

  const q = query.trim().toLowerCase();
  const filtered = useMemo(() => {
    if (!q) {
      return items.filter((item) => item.group !== "Stations").concat(items.filter((i) => i.group === "Stations").slice(0, 6));
    }
    return items
      .filter((item) => `${item.label} ${item.hint ?? ""} ${item.group}`.toLowerCase().includes(q))
      .slice(0, 18);
  }, [items, q]);

  const groups = useMemo(() => {
    const map = new Map<string, PaletteItem[]>();
    for (const item of filtered) {
      const list = map.get(item.group) ?? [];
      list.push(item);
      map.set(item.group, list);
    }
    return [...map.entries()];
  }, [filtered]);

  const flat = useMemo(() => groups.flatMap(([, list]) => list), [groups]);

  useEffect(() => {
    setActive(0);
  }, [q]);

  useEffect(() => {
    if (!paletteOpen) return;
    const node = listRef.current?.querySelector<HTMLElement>(`[data-index="${active}"]`);
    node?.scrollIntoView({ block: "nearest" });
  }, [active, paletteOpen]);

  const close = () => setPaletteOpen(false);

  const run = (item: PaletteItem | undefined) => {
    if (!item) return;
    close();
    window.setTimeout(() => item.run(), 40);
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Escape") {
      event.preventDefault();
      close();
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive((i) => (flat.length ? (i + 1) % flat.length : 0));
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((i) => (flat.length ? (i - 1 + flat.length) % flat.length : 0));
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      run(flat[active]);
    }
  };

  let cursor = -1;

  return (
    <AnimatePresence>
      {paletteOpen ? (
        <motion.div
          className="fixed inset-0 z-[110] flex items-start justify-center px-4 pt-[12vh] no-print"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.16 }}
        >
          <div
            className="absolute inset-0"
            style={{ background: "color-mix(in srgb, var(--bg-deep) 68%, transparent)", backdropFilter: "blur(6px)" }}
            onClick={close}
            aria-hidden="true"
          />
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-label="Command palette"
            className="relative w-full max-w-[620px] overflow-hidden rounded-2xl border"
            style={{
              background: "var(--surface)",
              borderColor: "var(--border-strong)",
              boxShadow: "var(--shadow-2)",
            }}
            initial={reduce ? { opacity: 0 } : { opacity: 0, y: -12, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={reduce ? { opacity: 0 } : { opacity: 0, y: -8, scale: 0.99 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
            onKeyDown={onKeyDown}
          >
            <div className="flex items-center gap-3 border-b px-4" style={{ borderColor: "var(--border)" }}>
              <Search size={17} className="muted" />
              <input
                ref={inputRef}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search stations, pages, actions…"
                className="h-14 flex-1 bg-transparent text-[14.5px] outline-none placeholder:text-[var(--faint)]"
                aria-label="Search"
                autoComplete="off"
                spellCheck={false}
              />
              <kbd>Esc</kbd>
            </div>

            <div ref={listRef} className="max-h-[52vh] overflow-y-auto p-2">
              {flat.length === 0 ? (
                <div className="px-4 py-10 text-center">
                  <p className="text-[13.5px] font-medium">No matches</p>
                  <p className="mt-1 text-[12px] muted">Try a station name, river, district or page.</p>
                </div>
              ) : (
                groups.map(([group, list]) => (
                  <div key={group} className="mb-1">
                    <div className="micro px-3 pb-1 pt-2.5" style={{ fontSize: 9 }}>
                      {group}
                    </div>
                    <ul>
                      {list.map((item) => {
                        cursor += 1;
                        const index = cursor;
                        const Icon = item.icon;
                        const isActive = index === active;
                        return (
                          <li key={item.id}>
                            <button
                              type="button"
                              data-index={index}
                              onMouseEnter={() => setActive(index)}
                              onClick={() => run(item)}
                              className={cn(
                                "flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors",
                              )}
                              style={{ background: isActive ? "var(--accent-soft)" : "transparent" }}
                            >
                              <Icon
                                size={15}
                                style={{ color: isActive ? "var(--accent)" : "var(--muted)" }}
                                className="shrink-0"
                              />
                              <span className="min-w-0 flex-1">
                                <span
                                  className="block truncate text-[13.5px]"
                                  style={{ color: isActive ? "var(--text)" : "var(--text-2)" }}
                                >
                                  {item.label}
                                </span>
                                {item.hint ? (
                                  <span className="mt-0.5 block truncate text-[11px] faint">{item.hint}</span>
                                ) : null}
                              </span>
                              {isActive ? <CornerDownLeft size={13} className="shrink-0 faint" /> : null}
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                ))
              )}
            </div>

            <div
              className="flex items-center justify-between gap-3 border-t px-4 py-2.5 text-[10.5px] faint"
              style={{ borderColor: "var(--border)" }}
            >
              <div className="flex items-center gap-3">
                <span className="flex items-center gap-1">
                  <kbd>↑</kbd>
                  <kbd>↓</kbd> navigate
                </span>
                <span className="flex items-center gap-1">
                  <kbd>↵</kbd> open
                </span>
              </div>
              <span className="num">{flat.length} results</span>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
