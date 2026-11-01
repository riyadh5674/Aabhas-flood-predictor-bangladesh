import { ArrowUpRight, ClipboardCopy } from "lucide-react";
import { Link } from "react-router-dom";
import { BandChip } from "@/components/ui/BandChip";
import { BAND_META, SEVERITY } from "@/lib/risk";
import { longDate } from "@/lib/format";
import type { StationRisk } from "@/lib/types";
import { useToast } from "@/state/ToastProvider";

export function AlertCard({ row, rank }: { row: StationRisk; rank?: number }) {
  const { toast } = useToast();
  const meta = BAND_META[row.band];
  const level = meta.alertLevel;

  const bulletin = [
    `AABHAS FLOOD ADVISORY — ${meta.label.toUpperCase()}`,
    `Station: ${row.station.name}${row.station.river ? ` (${row.station.river} river)` : ""}`,
    `Location: ${[row.station.upazilla, row.station.district, row.station.division].filter(Boolean).join(", ")}`,
    `Forecast: ${longDate(row.date)} · ${row.horizon}`,
    `Probability: ${row.probability == null ? "—" : (row.probability * 100).toFixed(2) + "%"}`,
    `Warning threshold: ${row.threshold == null ? "—" : (row.threshold * 100).toFixed(2) + "%"}`,
    `Recommended action: ${row.action || meta.operational}`,
    "",
    meta.guidance,
    "",
    "Research prototype — not an official warning. Follow FFWC and local authority advisories.",
  ].join("\n");

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(bulletin);
      toast({ title: "Advisory copied", description: "Plain-text bulletin ready to paste.", tone: "success" });
    } catch {
      toast({ title: "Clipboard unavailable", tone: "danger" });
    }
  };

  return (
    <article
      className="card card--hover relative overflow-hidden p-4"
      style={{ borderColor: `color-mix(in srgb, ${meta.color} 26%, var(--border))` }}
    >
      <span
        className="absolute inset-y-0 left-0 w-[3px]"
        style={{ background: meta.color, opacity: level === "high" ? 1 : 0.7 }}
        aria-hidden="true"
      />

      <div className="flex items-start justify-between gap-3 pl-1.5">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {rank != null ? <span className="num text-[10.5px] faint">#{String(rank).padStart(2, "0")}</span> : null}
            <Link
              to={`/stations/${row.stationId}`}
              className="truncate text-[14px] font-medium transition-colors hover:text-[var(--accent)]"
            >
              {row.station.name}
            </Link>
          </div>
          <div className="mt-0.5 truncate text-[11.5px] muted">
            {[row.station.river, row.station.upazilla, row.station.district].filter(Boolean).join(" · ")}
          </div>
        </div>
        <BandChip band={row.band} />
      </div>

      <div className="mt-3 flex items-end gap-4 pl-1.5">
        <div>
          <div className="micro" style={{ fontSize: 8.5 }}>
            Probability
          </div>
          <div className="num mt-0.5 text-[24px] font-semibold leading-none" style={{ color: meta.color }}>
            {row.probability == null ? "—" : (row.probability * 100).toFixed(1)}
            <span className="text-[12px] font-normal muted">%</span>
          </div>
        </div>
        <div className="mb-0.5">
          <div className="micro" style={{ fontSize: 8.5 }}>
            Horizon
          </div>
          <div className="num mt-0.5 text-[12.5px]">{row.horizon}</div>
        </div>
        <div className="mb-0.5">
          <div className="micro" style={{ fontSize: 8.5 }}>
            Threshold
          </div>
          <div className="num mt-0.5 text-[12.5px]">
            {row.threshold == null ? "—" : `${(row.threshold * 100).toFixed(2)}%`}
          </div>
        </div>
      </div>

      <p className="mt-3 border-t pt-2.5 pl-1.5 text-[12px] leading-relaxed text-2" style={{ borderColor: "var(--hairline)" }}>
        {meta.operational}
      </p>

      <div className="mt-3 flex items-center justify-between gap-2 pl-1.5">
        <span className="num text-[10px] uppercase tracking-[0.1em]" style={{ color: meta.color }}>
          {level === "high" ? "High-risk alert" : level === "warning" ? "Warning" : "Watch"}
        </span>
        <div className="flex items-center gap-1">
          <button type="button" className="btn btn--sm btn--ghost" onClick={copy} title="Copy advisory">
            <ClipboardCopy size={12} />
            Copy
          </button>
          <Link to={`/stations/${row.stationId}`} className="btn btn--sm">
            Station
            <ArrowUpRight size={12} />
          </Link>
        </div>
      </div>
    </article>
  );
}

export function sortBySeverity(rows: StationRisk[]): StationRisk[] {
  return [...rows].sort((a, b) => {
    if (SEVERITY[b.band] !== SEVERITY[a.band]) return SEVERITY[b.band] - SEVERITY[a.band];
    return (b.probability ?? 0) - (a.probability ?? 0);
  });
}
