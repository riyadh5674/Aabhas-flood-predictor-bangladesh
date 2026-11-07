import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";

const API_TARGET = process.env.AABHAS_API ?? process.env.NODI_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  // AABHAS_BASE lets the same build be served from a sub-path (e.g. a GitHub
  // Pages project site at /<repo>/) without patching asset URLs.
  base: process.env.AABHAS_BASE ?? "/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
  build: {
    target: "es2022",
    sourcemap: process.env.AABHAS_SOURCEMAP !== "false",
    rollupOptions: {
      output: {
        manualChunks: {
          motion: ["motion", "motion/react"],
          d3: ["d3-geo", "d3-shape"],
          query: ["@tanstack/react-query"],
        },
      },
    },
  },
});
