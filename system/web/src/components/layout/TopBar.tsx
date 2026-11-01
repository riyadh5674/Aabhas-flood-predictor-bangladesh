import { ChevronRight, Command, Search } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { StatusPill } from "@/components/StatusPill";
import { LangToggle, ThemeToggle } from "@/components/PreferenceToggles";
import { useApp } from "@/state/AppProvider";
import { useI18n } from "@/state/i18n";
import { BrandMark } from "./BrandMark";
import { titleForPath } from "./nav";

function DhakaClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const time = new Intl.DateTimeFormat("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZone: "Asia/Dhaka",
    hour12: false,
  }).format(now);
  return (
    <div className="hidden items-center gap-2 xl:flex" title="Bangladesh Standard Time (UTC+6)">
      <span className="micro" style={{ fontSize: 9 }}>
        Dhaka
      </span>
      <span className="num text-[11.5px] tabular-nums text-2">{time}</span>
    </div>
  );
}

export function TopBar() {
  const { t } = useI18n();
  const { setPaletteOpen } = useApp();
  const location = useLocation();
  const title = titleForPath(location.pathname);

  return (
    <header
      className="sticky top-0 z-30 flex h-16 items-center justify-between gap-3 border-b px-4 no-print sm:px-6"
      style={{
        background: "color-mix(in srgb, var(--bg) 82%, transparent)",
        backdropFilter: "blur(18px) saturate(1.2)",
        borderColor: "var(--border)",
      }}
    >
      <div className="flex min-w-0 items-center gap-3">
        <div className="lg:hidden">
          <Link to="/" aria-label="Aabhas home">
            <BrandMark />
          </Link>
        </div>
        <nav className="hidden min-w-0 items-center gap-1.5 lg:flex" aria-label="Breadcrumb">
          <Link to="/" className="text-[12.5px] muted transition-colors hover:text-[var(--text)]">
            Aabhas
          </Link>
          <ChevronRight size={13} className="faint" />
          <span className="truncate text-[12.5px] font-medium">{title}</span>
        </nav>
      </div>

      <div className="flex items-center gap-1.5 sm:gap-2">
        <DhakaClock />
        <div className="mx-1 hidden h-5 w-px sm:block" style={{ background: "var(--border)" }} />
        <button
          type="button"
          onClick={() => setPaletteOpen(true)}
          className="hidden items-center gap-2 rounded-xl border px-2.5 py-1.5 text-[12px] transition-colors hover:border-[var(--border-strong)] md:flex"
          style={{ borderColor: "var(--border)", background: "var(--surface-2)", color: "var(--muted)" }}
          aria-label={t("common.search")}
        >
          <Search size={13} />
          <span className="hidden xl:inline">{t("common.search")} stations, pages…</span>
          <span className="ml-1 flex items-center gap-0.5">
            <kbd className="hidden xl:inline-flex">
              <Command size={9} />
            </kbd>
            <kbd>K</kbd>
          </span>
        </button>
        <button
          type="button"
          onClick={() => setPaletteOpen(true)}
          className="icon-btn md:hidden"
          aria-label={t("common.search")}
        >
          <Search size={16} />
        </button>
        <div className="hidden sm:block">
          <StatusPill />
        </div>
        <div className="mx-0.5 hidden h-5 w-px sm:block" style={{ background: "var(--border)" }} />
        <LangToggle />
        <ThemeToggle />
      </div>
    </header>
  );
}
