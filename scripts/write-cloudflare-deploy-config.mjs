import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

export function createRoutePreservingDeployConfig(config) {
  if (!config || typeof config !== "object" || Array.isArray(config)) {
    throw new Error("Wrangler source config must be a JSON object.");
  }
  if (!Array.isArray(config.routes) || config.routes.length === 0) {
    throw new Error("Wrangler source config must declare the existing production routes.");
  }
  if (config.workers_dev !== false) {
    throw new Error("Wrangler source config must explicitly disable workers_dev.");
  }
  const { routes: _existingRoutes, ...deployConfig } = config;
  return deployConfig;
}

export function writeRoutePreservingDeployConfig(sourcePath, outputPath) {
  const source = JSON.parse(readFileSync(sourcePath, "utf8"));
  const deployConfig = createRoutePreservingDeployConfig(source);
  writeFileSync(outputPath, `${JSON.stringify(deployConfig, null, 2)}\n`, "utf8");
  return deployConfig;
}

function main() {
  const sourcePath = process.argv[2] || "wrangler.jsonc";
  const outputPath = process.argv[3] || ".wrangler-deploy.json";
  writeRoutePreservingDeployConfig(sourcePath, outputPath);
  console.log(`Wrote route-preserving Wrangler deploy config: ${outputPath}`);
}

if (process.argv[1] && resolve(fileURLToPath(import.meta.url)) === resolve(process.argv[1])) {
  main();
}
