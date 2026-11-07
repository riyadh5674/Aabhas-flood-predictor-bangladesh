/*
 * Aabhas offline shell.
 *
 * Flood information is most useful when the network is worst. This worker
 * keeps the app shell, station directory and static snapshot files available
 * offline. It deliberately does NOT cache /api responses: a stale probability
 * served without the normal freshness checks could be mistaken for a current
 * forecast. The UI's own freshness banner handles that distinction.
 *
 * All paths are derived from the registration scope so the same build works at
 * a root domain, a GitHub Pages project path, or a Hugging Face Space.
 */
const VERSION = "aabhas-v2";
const SHELL_CACHE = `${VERSION}-shell`;
const DATA_CACHE = `${VERSION}-data`;
const ASSET_CACHE = `${VERSION}-assets`;
const BASE = new URL(self.registration.scope).pathname;
const SHELL = [BASE, `${BASE}index.html`, `${BASE}manifest.webmanifest`, `${BASE}favicon.svg`];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      .then((cache) => cache.addAll(SHELL))
      .catch(() => undefined)
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => !key.startsWith(VERSION))
            .map((key) => caches.delete(key)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("message", (event) => {
  if (event.data === "SKIP_WAITING") self.skipWaiting();
});

function isAsset(url) {
  return (
    url.pathname.startsWith(`${BASE}assets/`) ||
    /\.(?:css|js|mjs|woff2?|ttf|png|jpg|jpeg|svg|webp|ico)$/i.test(url.pathname)
  );
}

async function networkFirst(request, cacheName, fallback) {
  const cache = await caches.open(cacheName);
  try {
    const response = await fetch(request);
    if (response && response.ok) cache.put(request, response.clone());
    return response;
  } catch (error) {
    const cached = await cache.match(request);
    if (cached) return cached;
    if (fallback) {
      const shell = await caches.open(SHELL_CACHE).then((c) => c.match(fallback));
      if (shell) return shell;
    }
    throw error;
  }
}

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response && response.ok) cache.put(request, response.clone());
  return response;
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/")) return;

  if (request.mode === "navigate") {
    event.respondWith(networkFirst(request, SHELL_CACHE, `${BASE}index.html`));
    return;
  }

  if (url.pathname.startsWith(`${BASE}data/`)) {
    event.respondWith(networkFirst(request, DATA_CACHE));
    return;
  }

  if (isAsset(url)) {
    event.respondWith(cacheFirst(request, ASSET_CACHE).catch(() => fetch(request)));
  }
});
