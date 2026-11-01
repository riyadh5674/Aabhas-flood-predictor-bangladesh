import { CloudOff, RefreshCw, SearchX } from "lucide-react";

export function EmptyState({
  title,
  description,
  action,
  icon: Icon = SearchX,
}: {
  title: string;
  description?: string;
  action?: React.ReactNode;
  icon?: React.ComponentType<{ size?: number; className?: string; style?: React.CSSProperties }>;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
      <div
        className="mb-4 grid h-12 w-12 place-items-center rounded-2xl"
        style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
      >
        <Icon size={20} className="muted" />
      </div>
      <h3 className="text-[15px] font-medium">{title}</h3>
      {description ? <p className="mt-1.5 max-w-[42ch] text-[13px] leading-relaxed muted">{description}</p> : null}
      {action ? <div className="mt-5">{action}</div> : null}
    </div>
  );
}

export function ErrorState({
  title = "Something went wrong",
  description,
  onRetry,
  retryLabel = "Try again",
}: {
  title?: string;
  description?: string;
  onRetry?: () => void;
  retryLabel?: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
      <div
        className="mb-4 grid h-12 w-12 place-items-center rounded-2xl"
        style={{ background: "color-mix(in srgb, var(--band-very-high) 12%, transparent)" }}
      >
        <CloudOff size={20} style={{ color: "var(--band-very-high)" }} />
      </div>
      <h3 className="text-[15px] font-medium">{title}</h3>
      {description ? <p className="mt-1.5 max-w-[46ch] text-[13px] leading-relaxed muted">{description}</p> : null}
      {onRetry ? (
        <button type="button" className="btn btn--sm mt-5" onClick={onRetry}>
          <RefreshCw size={13} />
          {retryLabel}
        </button>
      ) : null}
    </div>
  );
}
