import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { CommandPalette } from "@/components/CommandPalette";
import { ShortcutsSheet } from "@/components/ShortcutsSheet";
import { cn } from "@/lib/cn";
import { useApp } from "@/state/AppProvider";
import { BackgroundFX } from "./BackgroundFX";
import { ContextBar } from "./ContextBar";
import { Footer } from "./Footer";
import { MobileNav } from "./MobileNav";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { WelcomeIntro } from "./WelcomeIntro";

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

export function AppShell() {
  const isPreview = import.meta.env.VITE_AABHAS_PREVIEW === "true";
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem("aabhas.sidebar") === "collapsed";
    } catch {
      return false;
    }
  });
  const { setPaletteOpen, paletteOpen, shortcutsOpen, setShortcutsOpen, setHorizon, dates, date, setDate } = useApp();
  const location = useLocation();
  const reduce = useReducedMotion();

  useEffect(() => {
    try {
      localStorage.setItem("aabhas.sidebar", collapsed ? "collapsed" : "expanded");
    } catch {
      /* ignore */
    }
  }, [collapsed]);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location.pathname]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const mod = event.metaKey || event.ctrlKey;
      if (mod && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen(!paletteOpen);
        return;
      }
      if (isTypingTarget(event.target)) return;

      if (event.key === "/") {
        event.preventDefault();
        setPaletteOpen(true);
        return;
      }
      if (event.key === "?") {
        event.preventDefault();
        setShortcutsOpen(!shortcutsOpen);
        return;
      }
      if (event.key === "Escape") {
        setPaletteOpen(false);
        setShortcutsOpen(false);
        return;
      }
      if (event.key === "1") setHorizon("24h");
      if (event.key === "2") setHorizon("48h");
      if (event.key === "3") setHorizon("72h");

      if (event.altKey && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
        event.preventDefault();
        const index = dates.indexOf(date);
        if (index === -1) return;
        const next = dates[index + (event.key === "ArrowRight" ? 1 : -1)];
        if (next) setDate(next);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    date,
    dates,
    paletteOpen,
    setDate,
    setHorizon,
    setPaletteOpen,
    setShortcutsOpen,
    shortcutsOpen,
  ]);

  return (
    <div className="relative min-h-dvh">
      <BackgroundFX />
      <WelcomeIntro />
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((v) => !v)} />

      <div
        className={cn(
          "relative z-10 min-h-dvh transition-[padding] duration-300",
          collapsed ? "lg:pl-[76px]" : "lg:pl-[248px]",
        )}
      >
        {isPreview ? (
          <div className="flex min-h-9 items-center justify-center gap-2 border-b border-amber-400/25 bg-amber-400/10 px-3 py-2 text-center text-[10px] font-semibold uppercase tracking-[0.12em] text-amber-200 sm:text-xs">
            <span>Preview environment</span>
            <span className="hidden font-normal normal-case tracking-normal text-amber-100/80 sm:inline">
              Model output is experimental and is not an official flood warning
            </span>
          </div>
        ) : null}
        <TopBar />
        <ContextBar />
        <main className="mx-auto w-full max-w-[1500px] px-4 pb-8 pt-6 sm:px-6">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={location.pathname}
              initial={reduce ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reduce ? undefined : { opacity: 0, y: -6 }}
              transition={{ duration: 0.26, ease: [0.16, 1, 0.3, 1] }}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
        <Footer />
      </div>

      <MobileNav />
      <CommandPalette />
      <ShortcutsSheet />
    </div>
  );
}
