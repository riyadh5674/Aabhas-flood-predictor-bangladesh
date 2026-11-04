import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RotateCw } from "lucide-react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

function isChunkError(error: Error): boolean {
  return /dynamically imported module|Loading chunk|Importing a module script failed/i.test(
    error.message,
  );
}

/**
 * Last-resort boundary. Without this, any render-time exception produces a
 * blank page — the worst possible failure mode for an emergency-information
 * app. The fallback stays usable without JavaScript state recovery by
 * offering a full reload.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Keep a breadcrumb for debugging without shipping a vendor SDK.
    console.error("[Aabhas] Unhandled UI error", error, info.componentStack);
    if (isChunkError(error)) {
      // A deploy replaced the hashed asset this tab was using. Reload once so
      // the user lands on the new build automatically.
      const key = "aabhas.chunk-reload";
      try {
        if (!sessionStorage.getItem(key)) {
          sessionStorage.setItem(key, "1");
          window.location.reload();
          return;
        }
      } catch {
        /* storage unavailable */
      }
    }
  }

  render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div
        className="flex min-h-dvh items-center justify-center p-6"
        role="alert"
        aria-live="assertive"
      >
        <div
          className="w-full max-w-lg rounded-2xl border p-6"
          style={{ background: "var(--surface)", borderColor: "var(--border)" }}
        >
          <div className="mb-3 flex items-center gap-2">
            <AlertTriangle size={18} style={{ color: "var(--band-moderate)" }} />
            <h1 className="display text-[18px]">The outlook could not be displayed</h1>
          </div>
          <p className="text-[13px] leading-relaxed text-2">
            {isChunkError(error)
              ? "A new version of the app was deployed while this tab was open. Reload to continue."
              : "An unexpected error interrupted this page. Your data has not been changed."}
          </p>
          <pre className="mt-3 overflow-auto rounded-lg p-3 text-[11px] faint" style={{ background: "var(--surface-2)" }}>
            {error.message}
          </pre>
          <div className="mt-4 flex flex-wrap gap-2">
            <button type="button" className="btn btn--primary btn--sm" onClick={() => window.location.reload()}>
              <RotateCw size={14} /> Reload app
            </button>
            <button type="button" className="btn btn--soft btn--sm" onClick={() => this.setState({ error: null })}>
              Try again
            </button>
            <a className="btn btn--ghost btn--sm" href="/">
              Go to overview
            </a>
          </div>
          <p className="mt-4 text-[11px] faint">
            If this keeps happening, this is not an official warning service — follow FFWC and local authority instructions.
          </p>
        </div>
      </div>
    );
  }
}
