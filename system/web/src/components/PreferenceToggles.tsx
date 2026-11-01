import { Moon, Sun } from "lucide-react";
import { flushSync } from "react-dom";
import { motion } from "motion/react";
import { useApp } from "@/state/AppProvider";
import { useI18n } from "@/state/i18n";

type ViewTransitionDocument = Document & {
  startViewTransition?: (cb: () => void) => { finished: Promise<void> };
};

export function ThemeToggle() {
  const { theme, setTheme } = useApp();

  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    const doc = document as ViewTransitionDocument;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!doc.startViewTransition || reduce) {
      setTheme(next);
      return;
    }
    doc.startViewTransition(() => {
      flushSync(() => setTheme(next));
    });
  };

  return (
    <button
      type="button"
      onClick={toggle}
      className="icon-btn"
      aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      title={theme === "dark" ? "Light theme" : "Dark theme"}
    >
      <motion.span
        key={theme}
        initial={{ rotate: -35, opacity: 0, scale: 0.7 }}
        animate={{ rotate: 0, opacity: 1, scale: 1 }}
        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
        className="grid place-items-center"
      >
        {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
      </motion.span>
    </button>
  );
}

export function LangToggle() {
  const { lang, setLang } = useApp();
  const { t } = useI18n();

  return (
    <button
      type="button"
      onClick={() => setLang(lang === "en" ? "bn" : "en")}
      className="icon-btn w-auto px-2.5"
      aria-label={lang === "en" ? "বাংলায় দেখুন" : "View in English"}
      title={lang === "en" ? "বাংলা" : "English"}
    >
      <span
        className="num text-[11px] font-semibold tracking-wide"
        style={{ fontFamily: lang === "en" ? "var(--font-bangla)" : undefined }}
      >
        {lang === "en" ? "বাং" : "EN"}
      </span>
    </button>
  );
}

export function ShortcutHint() {
  const { setPaletteOpen } = useApp();
  const { t } = useI18n();
  return (
    <button
      type="button"
      onClick={() => setPaletteOpen(true)}
      className="hidden items-center gap-2 rounded-xl border px-3 py-2 text-[12.5px] transition-colors md:flex"
      style={{
        borderColor: "var(--border)",
        background: "var(--surface-2)",
        color: "var(--muted)",
      }}
      aria-label={t("common.search")}
    >
      <span className="truncate">{t("common.search")}</span>
      <kbd>Ctrl</kbd>
      <kbd>K</kbd>
    </button>
  );
}
