import { Cpu, BarChart3, Layers, FileText, ShieldCheck, Activity } from "lucide-react";
import { useApp } from "@/state/AppProvider";
import { useI18n } from "@/state/i18n";
import { useModelInfo, useRiskConfig } from "@/lib/queries";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card } from "@/components/ui/Card";
import { Reveal } from "@/components/ui/Reveal";
import { StatCard } from "@/components/StatCard";
import { BandChip } from "@/components/ui/BandChip";
import { StatusPill } from "@/components/StatusPill";
import { BAND_META, FALLBACK_RISK } from "@/lib/risk";
import { pctPoints } from "@/lib/format";

export function ModelPage() {
  const { t } = useI18n();
  const { dataMode } = useApp();

  const info = useModelInfo();
  const risk = useRiskConfig();

  return (
    <div className="space-y-6">
      <Reveal>
        <PageHeader
          eyebrow={t("nav.model")}
          title="Model card"
          description={
            <>
              Frozen D16 inference artefact. Thresholds and risk bands are fetched from <span className="font-medium">/thresholds</span>;
              the published model card identifies the bundled threshold copy when the service is unavailable.
            </>
          }
          actions={<StatusPill compact />}
        />
      </Reveal>

      <Reveal delay={0.06}>
        <div className="grid gap-4 md:grid-cols-4">
          <StatCard label="Model version" value={info.data?.version ?? (info.isError ? "Unavailable" : "Loading")} unit="frozen artefact" icon={Cpu} />
          <StatCard label="Feature set" value="FULL_V2" unit={info.data?.n_features ? String(info.data.n_features) + " features" : "110 features"} icon={BarChart3} />
          <StatCard label="Seed" value={String(info.data?.seed ?? 20261008)} unit="deterministic" icon={Layers} />
          <StatCard label="Status" value={info.data?.loaded_at != null ? "Loaded" : info.isError ? "Unavailable" : "Checking"} unit={dataMode === "demo" ? "Demo forecasts · model not used" : info.isError ? "API request failed" : "API model metadata"} icon={ShieldCheck} />
        </div>
      </Reveal>

      <Reveal delay={0.1}>
        <div className="grid gap-6 md:grid-cols-3">
          <Card className="p-5 flex-1 min-h-[320px]">
            <div className="flex items-center gap-2 mb-3">
              <FileText size={16} style={{ color: "var(--accent)" }} />
              <h3 className="display text-[15px]">Thresholds by horizon</h3>
            </div>
            <div className="space-y-3">
              {(["24h", "48h", "72h"] as const).map((h) => {
                const row = risk.data?.thresholds?.[h] ?? FALLBACK_RISK.thresholds[h];
                return (
                  <div key={h} className="rounded-lg border p-3" style={{ borderColor: "var(--border)", background: "var(--surface-2)" }}>
                    <div className="flex items-center justify-between text-[12px]">
                      <span className="font-medium">{h}</span>
                      <span className="num">{row.model}</span>
                    </div>
                    <div className="mt-1 grid grid-cols-2 gap-2 text-[11px]">
                      <div className="flex flex-col">
                        <span className="muted">Warn threshold</span>
                        <span className="num" style={{ color: BAND_META.high.color }}>
                          {(row.t_warn * 100).toFixed(3)}%
                        </span>
                      </div>
                      <div className="flex flex-col">
                        <span className="muted">Recall</span>
                        <span className="num">{pctPoints(row.recall)}%</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="muted">Precision</span>
                        <span className="num">{pctPoints(row.precision)}%</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="muted">F1</span>
                        <span className="num">{pctPoints(row.f1)}%</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>

          <Card className="p-5 flex-1 min-h-[320px]">
            <div className="flex items-center gap-2 mb-3">
              <Activity size={16} style={{ color: "var(--accent)" }} />
              <h3 className="display text-[15px]">Risk bands</h3>
            </div>
            <div className="space-y-2">
              {(["24h", "48h", "72h"] as const).map((h) => {
                const bands = risk.data?.risk_bands?.[h]?.bands ?? FALLBACK_RISK.risk_bands[h].bands;
                return (
                  <div key={h}>
                    <div className="flex items-center gap-1 mb-1.5">
                      <span className="micro" style={{ fontSize: 8.5 }}>{h}</span>
                      <span className="num text-[11px] faint">{bands.length} band{bands.length !== 1 ? "s" : ""}</span>
                    </div>
                    <div className="grid grid-cols-4 gap-1">
                      {bands.map((b) => (
                        <BandChip key={b.name} band={b.name} solid className="text-[11px] h-8 py-0" />
                      ))}
                    </div>
                  </div>
                );
              })}
              <p className="mt-3 text-[11.5px] leading-relaxed muted">
                The 48 h and 72 h schemes have no <code className="text-[11px]">very_high</code> band per the frozen config —
                their top band is <code className="text-[11px]">high</code>.
              </p>
            </div>
          </Card>

          <Card className="p-5 flex-1 min-h-[320px]">
            <div className="flex items-center gap-2 mb-3">
              <ShieldCheck size={16} style={{ color: "var(--accent)" }} />
              <h3 className="display text-[15px]">Model facts</h3>
            </div>
            <table className="tbl text-[12px]">
              <tbody>
                <tr>
                  <td className="muted w-2/5">Architecture</td>
                  <td>lightgbm (24 h / 48 h) · xgboost (72 h)</td>
                </tr>
                <tr>
                  <td className="muted">Frozen at</td>
                  <td className="text-2">{info.data?.frozen_at_utc ?? "2026-10-15"}</td>
                </tr>
                <tr>
                  <td className="muted">Cost ratio</td>
                  <td className="text-2">C_FN=10, C_FP=1</td>
                </tr>
                <tr>
                  <td className="muted">Evaluation</td>
                  <td className="text-2">val_2022 all-season</td>
                </tr>
                <tr>
                  <td className="muted">Batch size</td>
                  <td className="text-2">133 stations max</td>
                </tr>
              </tbody>
            </table>
            <div className="mt-4 p-3 rounded-xl border" style={{ background: "var(--surface-2)", borderColor: "var(--hairline)" }}>
              <p className="text-[11px] leading-relaxed muted">
                This is a research prototype trained on FFWC gauge data. It is not an official warning service — always
                follow FFWC and local authority advisories.
              </p>
            </div>
          </Card>
        </div>
      </Reveal>
    </div>
  );
}

export default ModelPage;
