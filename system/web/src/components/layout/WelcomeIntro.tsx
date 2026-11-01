import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { ArrowRight, MapPinned, Waves } from "lucide-react";
import { useEffect, useState } from "react";
import { BrandMark } from "./BrandMark";

const SEEN_KEY = "aabhas.welcome-seen";

export function WelcomeIntro() {
  const reduce = useReducedMotion();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    try {
      setOpen(localStorage.getItem(SEEN_KEY) !== "yes");
    } catch {
      setOpen(false);
    }
  }, []);

  const dismiss = () => {
    setOpen(false);
    try {
      localStorage.setItem(SEEN_KEY, "yes");
    } catch {
      /* The welcome experience still works when storage is unavailable. */
    }
  };

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          className="nodi-welcome no-print"
          initial={reduce ? { opacity: 0 } : { opacity: 0, backdropFilter: "blur(0px)" }}
          animate={{ opacity: 1, backdropFilter: "blur(10px)" }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
          role="dialog"
          aria-modal="true"
          aria-labelledby="welcome-title"
        >
          <motion.div
            className="nodi-welcome__panel"
            initial={reduce ? { opacity: 0 } : { opacity: 0, y: 18, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.99 }}
            transition={{ duration: 0.55, delay: 0.08, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="nodi-welcome__art" aria-hidden="true">
              <div className="nodi-welcome__sun" />
              <div className="nodi-welcome__mapline nodi-welcome__mapline--one" />
              <div className="nodi-welcome__mapline nodi-welcome__mapline--two" />
              <Waves className="nodi-welcome__waves" size={240} strokeWidth={0.8} />
              <span className="nodi-welcome__pin nodi-welcome__pin--one"><MapPinned size={18} /></span>
              <span className="nodi-welcome__pin nodi-welcome__pin--two"><MapPinned size={16} /></span>
              <div className="nodi-welcome__caption">BANGLADESH · FLOOD OUTLOOK</div>
            </div>
            <div className="nodi-welcome__copy">
              <BrandMark size={42} />
              <div className="micro mt-8">Know your river. Plan ahead.</div>
              <h1 id="welcome-title" className="display mt-3 text-[clamp(2.2rem,6vw,4rem)]">A clearer view of what’s ahead.</h1>
              <p className="mt-4 max-w-[48ch] text-[15px] leading-relaxed text-2">
                Explore river-station forecasts, understand local flood risk, and find practical ways to prepare.
              </p>
              <div className="mt-7 flex flex-wrap items-center gap-3">
                <button type="button" className="btn btn--primary nodi-welcome__cta" onClick={dismiss} autoFocus>
                  Explore Aabhas <ArrowRight size={16} />
                </button>
                <button type="button" className="btn btn--ghost" onClick={dismiss}>Skip introduction</button>
              </div>
              <p className="mt-6 text-[11.5px] leading-relaxed muted">
                Forecast guidance supports preparedness. For official warnings, follow the Flood Forecasting and Warning Centre (FFWC).
              </p>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
