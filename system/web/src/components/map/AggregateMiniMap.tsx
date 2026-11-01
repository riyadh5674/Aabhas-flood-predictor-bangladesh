import { useMemo } from "react";
import { aggregateDivisions } from "@/lib/aggregate";
import { BAND_META, SEVERITY } from "@/lib/risk";
import { BandChip } from "@/components/ui/BandChip";
import type { StationRisk } from "@/lib/types";

interface AggregateMiniMapProps {
  rows: StationRisk[];
  height?: number;
}

export function AggregateMiniMap({ rows, height = 320 }: AggregateMiniMapProps) {
  const divisions = useMemo(() => aggregateDivisions(rows).slice(0, 8), [rows]);

  return (
    <div style={{ height }} className="flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <span className="micro">Top divisions by worst band</span>
        <span className="num text-[11px] faint">{divisions.length} shown</span>
      </div>
      <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
        {divisions.length ? (
          divisions.map((d) => {
            const meta = BAND_META[d.worst];
            return (
              <div key={d.name} className="flex items-center gap-2 rounded-lg p-1.5" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
                <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: meta.color }} />
                <span className="truncate text-[11.5px]">{d.name}</span>
                <BandChip band={d.worst} className="text-[10px] shrink-0" />
                <span className="num text-[10px] faint ml-auto">{d.stations} stn</span>
              </div>
            );
          })
        ) : (
          <div className="text-[11px] italic muted">No divisions reporting.</div>
        )}
      </div>
    </div>
  );
}
