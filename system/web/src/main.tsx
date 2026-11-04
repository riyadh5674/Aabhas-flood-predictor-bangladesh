import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, HashRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { ToastProvider } from "@/state/ToastProvider";
import { AppProvider } from "@/state/AppProvider";
import { registerServiceWorker } from "@/lib/pwa";
import { App } from "./App";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      retry: 2,
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 10_000),
      refetchOnWindowFocus: false,
    },
  },
});

// Static hosts (Hugging Face static Spaces) do not rewrite unknown paths to
// index.html, so deep links must use hash routing. The API-served build keeps
// clean browser URLs.
const Router =
  import.meta.env.VITE_AABHAS_STATIC_SNAPSHOT === "true" ? HashRouter : BrowserRouter;

registerServiceWorker();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <Router>
        <QueryClientProvider client={queryClient}>
          <ToastProvider>
            <AppProvider>
              <App />
            </AppProvider>
          </ToastProvider>
        </QueryClientProvider>
      </Router>
    </ErrorBoundary>
  </React.StrictMode>,
);
