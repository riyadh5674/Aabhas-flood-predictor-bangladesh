import { Navigate, Route, Routes } from "react-router-dom";
import { Suspense, lazy } from "react";
import { AppShell } from "@/components/layout/AppShell";

const Overview = lazy(() => import("@/pages/Overview").then((m) => ({ default: m.OverviewPage })));
const MapPage = lazy(() => import("@/pages/MapPage").then((m) => ({ default: m.MapPage })));
const Stations = lazy(() => import("@/pages/Stations").then((m) => ({ default: m.StationsPage })));
const StationDetail = lazy(() => import("@/pages/StationDetail").then((m) => ({ default: m.StationDetailPage })));
const Alerts = lazy(() => import("@/pages/Alerts").then((m) => ({ default: m.AlertsPage })));
const Model = lazy(() => import("@/pages/Model").then((m) => ({ default: m.ModelPage })));
const About = lazy(() => import("@/pages/About").then((m) => ({ default: m.AboutPage })));
const NotFound = lazy(() => import("@/pages/NotFound").then((m) => ({ default: m.NotFoundPage })));

function LoadingFallback() {
  return (
    <div className="mx-auto max-w-[1500px] px-4 py-10 sm:px-6" role="status" aria-live="polite" aria-label="Loading page">
      <div className="mb-6 space-y-3">
        <div className="skel h-3 w-28" />
        <div className="skel h-10 w-full max-w-md" />
        <div className="skel h-4 w-full max-w-2xl" />
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <div className="skel h-36 rounded-[18px]" />
        <div className="skel h-36 rounded-[18px]" />
        <div className="skel h-36 rounded-[18px]" />
      </div>
    </div>
  );
}

export function App() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<Overview />} />
          <Route path="/map" element={<MapPage />} />
          <Route path="/stations" element={<Stations />} />
          <Route path="/stations/:stationId" element={<StationDetail />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/model" element={<Model />} />
          <Route path="/about" element={<About />} />
        </Route>
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Suspense>
  );
}
