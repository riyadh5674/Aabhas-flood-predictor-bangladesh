import {
  BellRing,
  Compass,
  Cpu,
  Gauge,
  LayoutDashboard,
  Map as MapIcon,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  to: string;
  key: string;
  label: string;
  icon: LucideIcon;
  end?: boolean;
}

export const NAV_ITEMS: NavItem[] = [
  { to: "/", key: "nav.overview", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/map", key: "nav.map", label: "Risk map", icon: MapIcon },
  { to: "/stations", key: "nav.stations", label: "Stations", icon: Gauge },
  { to: "/alerts", key: "nav.alerts", label: "Alerts", icon: BellRing },
  { to: "/model", key: "nav.model", label: "Model", icon: Cpu },
  { to: "/about", key: "nav.about", label: "About", icon: Compass },
];

export const ROUTE_TITLES: Record<string, string> = {
  "/": "Overview",
  "/map": "Risk map",
  "/stations": "Stations",
  "/alerts": "Alerts",
  "/model": "Model",
  "/about": "About",
};

export function titleForPath(pathname: string): string {
  if (ROUTE_TITLES[pathname]) return ROUTE_TITLES[pathname];
  if (pathname.startsWith("/stations/")) return "Station detail";
  return "Aabhas";
}
