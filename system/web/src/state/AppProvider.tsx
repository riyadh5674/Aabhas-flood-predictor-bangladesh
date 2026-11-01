import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { demoDates } from "@/lib/demo";
import { useDates } from "@/lib/queries";
import { HORIZONS, type DataMode, type Horizon } from "@/lib/types";
import { I18nProvider, type Lang } from "./i18n";

export type Theme = "dark" | "light";

const STORAGE = {
  theme: "aabhas.theme",
  lang: "aabhas.lang",
  mode: "aabhas.mode",
};

function readStored<T extends string>(key: string, fallback: T): T {
  try {
    const value = localStorage.getItem(key);
    return (value as T) ?? fallback;
  } catch {
    return fallback;
  }
}

function isHorizon(value: string | null): value is Horizon {
  return value === "24h" || value === "48h" || value === "72h";
}

interface AppContextValue {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  lang: Lang;
  setLang: (lang: Lang) => void;
  date: string;
  setDate: (date: string) => void;
  horizon: Horizon;
  setHorizon: (horizon: Horizon) => void;
  dataMode: DataMode;
  setDataMode: (mode: DataMode) => void;
  latestDate: string;
  dates: string[];
  paletteOpen: boolean;
  setPaletteOpen: (open: boolean) => void;
  shortcutsOpen: boolean;
  setShortcutsOpen: (open: boolean) => void;
}

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [searchParams, setSearchParams] = useSearchParams();

  const [theme, setThemeState] = useState<Theme>(() => {
    try {
      const stored = localStorage.getItem(STORAGE.theme);
      if (stored === "dark" || stored === "light") return stored;
      return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    } catch {
      return "dark";
    }
  });

  const [lang, setLangState] = useState<Lang>(() => readStored<Lang>(STORAGE.lang, "en"));
  const [dataMode, setDataModeState] = useState<DataMode>(() =>
    readStored<DataMode>(STORAGE.mode, "auto"),
  );

  const [date, setDateState] = useState<string>(() => searchParams.get("d") ?? "");
  const [horizon, setHorizonState] = useState<Horizon>(() => {
    const h = searchParams.get("h");
    return isHorizon(h) ? h : "24h";
  });

  const [paletteOpen, setPaletteOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

  const { data: datesPayload, isSuccess: datesReady } = useDates();
  const dates = dataMode === "demo" ? demoDates(180) : (datesPayload?.dates ?? []);
  const latestDate = dates.at(-1) ?? "";
  const effectiveDate = date || latestDate;

  // Keep shared or remembered dates inside the real feature store. A stale
  // browser date must never silently produce a sample forecast.
  useEffect(() => {
    if (dataMode === "demo" || !datesReady || !latestDate) return;
    if (!date || !dates.includes(date)) setDateState(latestDate);
  }, [dataMode, datesReady, latestDate, date, dates]);

  /* Theme + language side effects */
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(STORAGE.theme, theme);
    } catch {
      /* storage may be unavailable */
    }
  }, [theme]);

  useEffect(() => {
    document.documentElement.setAttribute("lang", lang === "bn" ? "bn" : "en");
    try {
      localStorage.setItem(STORAGE.lang, lang);
    } catch {
      /* ignore */
    }
  }, [lang]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE.mode, dataMode);
    } catch {
      /* ignore */
    }
  }, [dataMode]);

  /* Keep date + horizon shareable through the URL */
  useEffect(() => {
    const next = new URLSearchParams(searchParams);
    if (effectiveDate) next.set("d", effectiveDate);
    next.set("h", horizon);
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveDate, horizon]);

  const setTheme = useCallback((next: Theme) => setThemeState(next), []);
  const toggleTheme = useCallback(
    () => setThemeState((prev) => (prev === "dark" ? "light" : "dark")),
    [],
  );
  const setLang = useCallback((next: Lang) => setLangState(next), []);
  const setDate = useCallback((next: string) => setDateState(next), []);
  const setHorizon = useCallback((next: Horizon) => {
    if (HORIZONS.includes(next)) setHorizonState(next);
  }, []);
  const setDataMode = useCallback((next: DataMode) => setDataModeState(next), []);

  const value = useMemo<AppContextValue>(
    () => ({
      theme,
      setTheme,
      toggleTheme,
      lang,
      setLang,
      date: effectiveDate,
      setDate,
      horizon,
      setHorizon,
      dataMode,
      setDataMode,
      latestDate,
      dates,
      paletteOpen,
      setPaletteOpen,
      shortcutsOpen,
      setShortcutsOpen,
    }),
    [
      theme,
      setTheme,
      toggleTheme,
      lang,
      setLang,
      effectiveDate,
      setDate,
      horizon,
      setHorizon,
      dataMode,
      setDataMode,
      latestDate,
      dates,
      paletteOpen,
      setPaletteOpen,
      shortcutsOpen,
      setShortcutsOpen,
    ],
  );

  return (
    <AppContext.Provider value={value}>
      <I18nProvider lang={lang} setLang={setLang}>
        {children}
      </I18nProvider>
    </AppContext.Provider>
  );
}

export function useApp(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used inside AppProvider");
  return ctx;
}
