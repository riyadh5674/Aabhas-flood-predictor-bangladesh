import type { Station } from "./types";

let cache: Station[] | null = null;
let inflight: Promise<Station[]> | null = null;

/** Loads the real FFWC station master (153 stations) bundled with the app. */
export async function loadStations(): Promise<Station[]> {
  if (cache) return cache;
  if (inflight) return inflight;
  inflight = fetch(`${import.meta.env.BASE_URL}data/stations.json`)
    .then((r) => {
      if (!r.ok) throw new Error(`stations.json ${r.status}`);
      return r.json() as Promise<Station[]>;
    })
    .then((rows) => {
      cache = rows;
      return rows;
    })
    .finally(() => {
      inflight = null;
    });
  return inflight;
}

export function stationsSync(): Station[] {
  return cache ?? [];
}

export function stationById(id: number): Station | undefined {
  return cache?.find((s) => s.id === id);
}

/** Bangladesh bounding box — used to sanity-check station coordinates. */
export const BD_BOUNDS = { minLat: 20.4, maxLat: 26.7, minLon: 87.9, maxLon: 92.8 };

export function hasPlausibleCoords(s: Station): boolean {
  return (
    s.lat >= BD_BOUNDS.minLat &&
    s.lat <= BD_BOUNDS.maxLat &&
    s.lon >= BD_BOUNDS.minLon &&
    s.lon <= BD_BOUNDS.maxLon
  );
}
