/**
 * Build-time feature flags.
 *
 * Demo data exists so the UI can be developed without the API. It must never
 * be reachable in a normal production build: a synthetic flood forecast shown
 * without an unmistakable label is a safety problem, not a feature. Enable it
 * explicitly with VITE_AABHAS_ALLOW_DEMO=true (staging only) or via the Vite
 * dev server.
 */
export const DEMO_ENABLED =
  import.meta.env.DEV || import.meta.env.VITE_AABHAS_ALLOW_DEMO === "true";

export const IS_PRODUCTION = import.meta.env.PROD;

export const APP_VERSION = (import.meta.env.VITE_AABHAS_VERSION as string | undefined) ?? "1.0.0";
