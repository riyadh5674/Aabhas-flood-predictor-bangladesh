import { Navigate, Route, Routes } from "react-router-dom";
import { OverviewPage as Overview } from "@/pages/Overview";
import { MapPage } from "@/pages/MapPage";
import { StationsPage as Stations } from "@/pages/Stations";
import { StationDetailPage as StationDetail } from "@/pages/StationDetail";
import { AlertsPage as Alerts } from "@/pages/Alerts";
import { ModelPage as Model } from "@/pages/Model";
import { AboutPage as About } from "@/pages/About";
import { NotFoundPage as NotFound } from "@/pages/NotFound";
import { AppShell } from "@/components/layout/AppShell";

export function App() {
  return (
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
  );
}
