import { geoGraticule10, geoMercator, geoPath } from "d3-geo";
import type { MapLevel } from "./types";

export interface GeoFeature {
  type: "Feature";
  properties: { shapeName: string; [key: string]: unknown };
  geometry: { type: string; coordinates: unknown };
}

export interface GeoCollection {
  type: "FeatureCollection";
  features: GeoFeature[];
}

export interface RegionShape {
  name: string;
  path: string;
  centroid: [number, number];
}

export interface ProjectedGeo {
  regions: RegionShape[];
  graticule: string;
  width: number;
  height: number;
  /** Project a lon/lat pair into the same pixel space as the region paths. */
  project: (lon: number, lat: number) => [number, number] | null;
}

const cache = new Map<MapLevel, Promise<GeoCollection>>();

export function loadGeo(level: MapLevel): Promise<GeoCollection> {
  const existing = cache.get(level);
  if (existing) return existing;
  const promise = fetch(`/data/geo/${level}.geojson`)
    .then((r) => {
      if (!r.ok) throw new Error(`${level}.geojson ${r.status}`);
      return r.json() as Promise<GeoCollection>;
    })
    .catch((err) => {
      cache.delete(level);
      throw err;
    });
  cache.set(level, promise);
  return promise;
}

/**
 * Projects the boundary file into the given pixel box and pre-computes SVG
 * path strings + centroids. Called once per level/size, not per render.
 */
export function projectGeo(geo: GeoCollection, width: number, height: number, pad = 10): ProjectedGeo {
  // geoBoundaries exports GeoJSON rings in RFC 7946 winding order. d3-geo's
  // spherical polygon renderer expects the opposite winding, otherwise it
  // fills the polygon's complement (the whole viewport) instead of Bangladesh.
  // Reverse a render-only copy so the source data remains untouched.
  const renderGeo: GeoCollection = {
    ...geo,
    features: geo.features.map((feature) => ({
      ...feature,
      geometry: {
        ...feature.geometry,
        coordinates: reversePolygonRings(feature.geometry),
      },
    })),
  };
  const projection = geoMercator().fitExtent(
    [
      [pad, pad],
      [width - pad, height - pad],
    ],
    renderGeo as never,
  );
  const path = geoPath(projection);

  const regions = renderGeo.features
    .map((feature) => {
      const d = path(feature as never);
      const centroid = path.centroid(feature as never);
      if (!d) return null;
      return {
        name: feature.properties.shapeName,
        path: d,
        centroid: [centroid[0], centroid[1]] as [number, number],
      };
    })
    .filter((r): r is RegionShape => r !== null);

  return {
    regions,
    graticule: path(geoGraticule10()) ?? "",
    width,
    height,
    project: (lon: number, lat: number) => {
      const p = projection([lon, lat]);
      return p ? [p[0], p[1]] : null;
    },
  };
}

function reversePolygonRings(geometry: GeoFeature["geometry"]): unknown {
  const reverseRings = (polygon: unknown) =>
    Array.isArray(polygon) ? polygon.map((ring) => (Array.isArray(ring) ? [...ring].reverse() : ring)) : polygon;
  if (geometry.type === "Polygon") return reverseRings(geometry.coordinates);
  if (geometry.type === "MultiPolygon" && Array.isArray(geometry.coordinates)) {
    return geometry.coordinates.map(reverseRings);
  }
  return geometry.coordinates;
}

/** Rough visual centre of Bangladesh, used to anchor pulse rings. */
export const BD_CENTER: [number, number] = [23.8, 90.3];
