import { motion } from "motion/react";
import { NavLink } from "react-router-dom";
import { useI18n } from "@/state/i18n";
import { NAV_ITEMS } from "./nav";

const MOBILE_KEYS = ["nav.overview", "nav.map", "nav.stations", "nav.alerts", "nav.model"];

export function MobileNav() {
  const { t } = useI18n();
  const items = NAV_ITEMS.filter((item) => MOBILE_KEYS.includes(item.key));

  return (
    <nav
      className="fixed inset-x-0 bottom-0 z-40 border-t px-2 pb-[max(env(safe-area-inset-bottom),8px)] pt-1.5 lg:hidden no-print"
      style={{
        background: "color-mix(in srgb, var(--bg) 92%, transparent)",
        backdropFilter: "blur(20px) saturate(1.2)",
        borderColor: "var(--border)",
      }}
      aria-label="Primary"
    >
      <ul className="mx-auto flex max-w-md items-stretch justify-between">
        {items.map((item) => {
          const Icon = item.icon;
          return (
            <li key={item.to} className="flex-1">
              <NavLink
                to={item.to}
                end={item.end}
                className="relative flex flex-col items-center gap-1 rounded-xl px-1 py-1.5"
              >
                {({ isActive }) => (
                  <>
                    {isActive ? (
                      <motion.span
                        layoutId="mobile-nav-active"
                        className="absolute inset-x-2 inset-y-0 rounded-xl"
                        style={{ background: "var(--accent-soft)" }}
                        transition={{ type: "spring", stiffness: 480, damping: 36 }}
                      />
                    ) : null}
                    <Icon
                      size={18}
                      strokeWidth={isActive ? 2.2 : 1.8}
                      className="relative z-10"
                      style={{ color: isActive ? "var(--accent)" : "var(--muted)" }}
                    />
                    <span
                      className="relative z-10 text-[10px] font-medium"
                      style={{ color: isActive ? "var(--accent)" : "var(--muted)" }}
                    >
                      {t(item.key)}
                    </span>
                  </>
                )}
              </NavLink>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
