import { existsSync } from "node:fs";

if (!existsSync("dist/deployment-hardening.json")) {
  throw new Error("dist is missing. Run `npm run build:worker` before the minified viewer check.");
}

process.env.DEEPBOM_VIEWER_ROOT = "dist";
await import("./check-explorer-redesign-viewer.mjs");
