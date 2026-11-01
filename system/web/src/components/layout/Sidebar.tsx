import { motion } from "motion/react";
import { PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { NavLink } from "react-router-dom";
import { cn } from "@/lib/cn";
import { useApp } from "@/state/AppProvider";
import { useI18n } from "@/state/i18n";
import { BrandMark } from "./BrandMark";
import { NAV_ITEMS } from "./nav";
import { StatusPill } from "@/components/StatusPill";

export function Sidebar({
  collapsed,
  onToggle,
}: {
  collapsed: boolean;
  onToggle: () => void;
}) {
  const { t } = useI18n();
  const { latestDate } = useApp();

  return (
    <aside
      className={cn(
        "fixed inset-y-0 left-0 z-40 hidden flex-col border-r transition-[width] duration-300 lg:flex no-print",
        collapsed ? "w-[76px]" : "w-[248px]",
      )}
      style={{
        background: "color-mix(in srgb, var(--bg) 88%, transparent)",
        backdropFilter: "blur(18px)",
        borderColor: "var(--border)",
      }}
    >
      <div
        className={cn("flex h-16 items-center border-b", collapsed ? "justify-center px-2" : "justify-between px-5")}
        style={{ borderColor: "var(--border)" }}
      >
        <NavLink to="/" aria-label="Aabhas home">
          <BrandMark compact={collapsed} />
        </NavLink>
        {!collapsed ? (
          <button type="button" className="icon-btn h-7 w-7" onClick={onToggle} aria-label="Collapse sidebar">
            <PanelLeftClose size={15} />
          </button>
        ) : null}
      </div>

      <nav className={cn("flex-1 overflow-y-auto overflow-x-hidden py-3", collapsed ? "px-2" : "px-3")}>
        <div className={cn("micro mb-2 px-2", collapsed && "sr-only")}>{t("nav.overview") ? "Navigate" : ""}</div>
        <ul className="space-y-0.5">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  title={collapsed ? t(item.key) : undefined}
                  className={({ isActive }) =>
                    cn(
                      "group relative flex items-center gap-3 rounded-xl px-2.5 py-2 text-[13.5px] font-medium transition-colors",
                      collapsed && "justify-center px-0",
                      isActive ? "text-[var(--text)]" : "text-[var(--text-2)] hover:text-[var(--text)]",
                    )
                  }
                  style={({ isActive }) => ({
                    background: isActive ? "var(--accent-soft)" : undefined,
                  })}
                >
                  {({ isActive }) => (
                    <>
                      {isActive ? (
                        <motion.span
                          layoutId="nav-active-bar"
                          className="absolute left-0 top-1/2 h-5 w-[2.5px] -translate-y-1/2 rounded-full"
                          style={{ background: "var(--accent)" }}
                          transition={{ type: "spring", stiffness: 480, damping: 38 }}
                        />
                      ) : null}
                      <Icon
                        size={17}
                        strokeWidth={isActive ? 2.1 : 1.8}
                        style={{ color: isActive ? "var(--accent)" : undefined }}
                        className="shrink-0"
                      />
                      {!collapsed ? <span className="truncate">{t(item.key)}</span> : null}
                    </>
                  )}
                </NavLink>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className={cn("border-t p-3", collapsed && "px-2")} style={{ borderColor: "var(--border)" }}>
        {collapsed ? (
          <div className="mb-2 flex justify-center">
            <button type="button" className="icon-btn h-7 w-7" onClick={onToggle} aria-label="Expand sidebar">
              <PanelLeftOpen size={15} />
            </button>
          </div>
        ) : (
          <div className="mb-3 rounded-xl border p-3" style={{ borderColor: "var(--border)", background: "var(--surface-2)" }}>
            <div className="mb-2 flex items-center justify-between">
              <span className="micro" style={{ fontSize: 9 }}>
                System
              </span>
              <StatusPill compact />
            </div>
            <div className="num text-[11px] text-2">
              D16 · 110 features
            </div>
            <div className="num mt-0.5 text-[10.5px] faint">data to {latestDate || "—"}</div>
          </div>
        )}

        {!collapsed ? (
          <p className="px-1 text-[10.5px] leading-relaxed faint">
            Forecast guidance to help you prepare. Always follow official FFWC advisories.
          </p>
        ) : null}
      </div>
    </aside>
  );
}
