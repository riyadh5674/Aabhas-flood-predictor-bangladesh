/**
 * Register the offline shell in production builds only. Development keeps
 * Vite's HMR path untouched, and the service worker is intentionally skipped
 * in static-snapshot mode's preview builds where a stale shell would confuse
 * deploy verification.
 */
export function registerServiceWorker(): void {
  if (!import.meta.env.PROD || !("serviceWorker" in navigator)) return;
  window.addEventListener("load", () => {
    navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`).catch((error) => {
      console.warn("[Aabhas] Service worker registration failed", error);
    });
  });
}
