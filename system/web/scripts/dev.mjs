import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const webDir = resolve(fileURLToPath(new URL("..", import.meta.url)));
const rootDir = resolve(webDir, "../..");
const windows = process.platform === "win32";
const python = process.env.AABHAS_PYTHON ?? resolve(rootDir, windows ? ".venv/Scripts/python.exe" : ".venv/bin/python");
const apiUrl = process.env.AABHAS_API ?? "http://127.0.0.1:8000";
const apiEndpoint = new URL(apiUrl);
const localApi = ["127.0.0.1", "localhost", "[::1]"].includes(apiEndpoint.hostname);
const children = [];

async function apiAlreadyResponds() {
  try {
    const response = await fetch(`${apiUrl}/health`, { signal: AbortSignal.timeout(1200) });
    return response.ok;
  } catch {
    return false;
  }
}

function stopChildren() {
  for (const child of children) {
    if (child && child.exitCode === null) child.kill("SIGTERM");
  }
}

process.on("SIGINT", () => { stopChildren(); process.exit(130); });
process.on("SIGTERM", () => { stopChildren(); process.exit(143); });

if (localApi && !(await apiAlreadyResponds())) {
  if (!existsSync(python)) {
    console.error(`Aabhas API was not found at ${python}. Set AABHAS_PYTHON or start the API separately.`);
    process.exitCode = 1;
    process.exit();
  }
  console.log(`Starting Aabhas API at ${apiUrl}`);
  const api = spawn(python, ["-m", "uvicorn", "system.api.app:app", "--host", "127.0.0.1", "--port", new URL(apiUrl).port || "8000"], {
    cwd: rootDir,
    stdio: "inherit",
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
  children.push(api);
  api.on("error", (error) => console.error(`Could not start the Aabhas API: ${error.message}`));
} else if (!localApi) {
  console.log(`Using configured API at ${apiUrl}`);
}

console.log("Starting Aabhas web app at http://localhost:5173");
const vite = spawn(process.execPath, [resolve(webDir, "node_modules/vite/bin/vite.js")], {
  cwd: webDir,
  stdio: "inherit",
  env: process.env,
});
children.push(vite);
vite.on("error", (error) => {
  console.error(`Could not start the web app: ${error.message}`);
  stopChildren();
  process.exitCode = 1;
});
vite.on("exit", (code) => {
  stopChildren();
  process.exitCode = code ?? 1;
});
