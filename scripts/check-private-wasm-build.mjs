import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { PRIVATE_WASM_MODULES } from "./private-wasm-modules.mjs";
import { runNpm } from "./run-utils.mjs";

const moduleResults = [];

for (const moduleSpec of PRIVATE_WASM_MODULES) {
  moduleResults.push(await checkPrivateWasmModule(moduleSpec));
}

console.log(
  `Private WASM build checks passed (${moduleResults.length} module(s): ${moduleResults.map((result) => `${result.id} ${formatBytes(result.totalBytes)}`).join(", ")}; manifest and runtime-call smoke=ok; public dist leak=0).`,
);

async function checkPrivateWasmModule(moduleSpec) {
  await runNpm(["run", moduleSpec.buildScript]);

  const totalBytes = checkArtifacts(moduleSpec);
  checkGlue(moduleSpec);
  await smokePrivateWasmRuntime(moduleSpec);
  checkNoTrackedBuildArtifacts(moduleSpec);
  checkNoPublicDistLeak(moduleSpec);

  return { id: moduleSpec.id, totalBytes };
}

function checkArtifacts(moduleSpec) {
  let totalBytes = 0;
  for (const [label, fileName, budgetBytes] of moduleSpec.expectedArtifacts) {
    const filePath = path.join(moduleSpec.pkgRoot, fileName);
    if (!existsSync(filePath)) {
      throw new Error(`${moduleSpec.label} WASM build is missing ${label} artifact: ${filePath}`);
    }
    const bytes = statSync(filePath).size;
    totalBytes += bytes;
    if (bytes > budgetBytes) {
      throw new Error(
        `${moduleSpec.label} ${label} artifact budget exceeded: ${formatBytes(bytes)} > ${formatBytes(budgetBytes)} (${filePath})`,
      );
    }
  }

  if (totalBytes > moduleSpec.totalBudgetBytes) {
    throw new Error(
      `${moduleSpec.label} pkg budget exceeded: ${formatBytes(totalBytes)} > ${formatBytes(moduleSpec.totalBudgetBytes)}`,
    );
  }
  return totalBytes;
}

function checkGlue(moduleSpec) {
  const jsGlue = readFileSync(path.join(moduleSpec.pkgRoot, moduleSpec.glueFile), "utf8");
  const cargoManifest = readFileSync(moduleSpec.cargoManifest, "utf8");
  const omitsDefaultModulePath = /omit-default-module-path\s*=\s*true/.test(cargoManifest);
  if (omitsDefaultModulePath && jsGlue.includes(moduleSpec.wasmFile)) {
    throw new Error(`${moduleSpec.label} JS glue exposes a default WASM path despite the hardened build contract.`);
  }
  if (!omitsDefaultModulePath && !jsGlue.includes(moduleSpec.wasmFile)) {
    throw new Error(`${moduleSpec.label} JS glue has neither an explicit WASM payload nor the hardened path-omission contract.`);
  }
  const typeDefs = readFileSync(path.join(moduleSpec.pkgRoot, moduleSpec.typeFile), "utf8");
  const missingExports = moduleSpec.requiredTypeExports.filter((exportName) => !typeDefs.includes(exportName));
  if (missingExports.length) {
    throw new Error(`${moduleSpec.label} TypeScript definitions are missing expected exports: ${missingExports.join(", ")}`);
  }
}

function checkNoTrackedBuildArtifacts(moduleSpec) {
  const trackedPrivateBuildArtifacts = execFileSync(
    "git",
    ["ls-files", ...moduleSpec.ignoredRoots.map(toGitPath)],
    { encoding: "utf8" },
  )
    .trim()
    .split(/\r?\n/)
    .filter(Boolean);
  if (trackedPrivateBuildArtifacts.length) {
    throw new Error(
      `${moduleSpec.label} build artifacts are tracked:\n${trackedPrivateBuildArtifacts.join("\n")}`,
    );
  }
}

function checkNoPublicDistLeak(moduleSpec) {
  const leakedDistPaths = existsSync("dist")
    ? collectFiles("dist").filter((filePath) => filePath.replaceAll(path.sep, "/").includes(moduleSpec.distLeakNeedle))
    : [];
  if (leakedDistPaths.length) {
    throw new Error(
      `${moduleSpec.label} artifacts leaked into public dist:\n${leakedDistPaths.join("\n")}`,
    );
  }
}

async function smokePrivateWasmRuntime(moduleSpec) {
  const jsPath = path.resolve(moduleSpec.pkgRoot, moduleSpec.glueFile);
  const wasmPath = path.resolve(moduleSpec.pkgRoot, moduleSpec.wasmFile);
  const privateModule = await import(pathToFileURL(jsPath).href);
  privateModule.initSync({ module: readFileSync(wasmPath) });

  const version = privateModule[moduleSpec.smoke.versionExport]();
  const patterns = privateModule[moduleSpec.smoke.patternsExport]();
  const manifest = privateModule[moduleSpec.smoke.manifestExport]();
  const requiredCapabilities = new Set(moduleSpec.smoke.requiredCapabilities);

  if (manifest.schema !== moduleSpec.smoke.manifestSchema) {
    throw new Error(`${moduleSpec.label} manifest schema mismatch: ${manifest.schema}`);
  }
  if (manifest.version !== version) {
    throw new Error(`${moduleSpec.label} manifest version mismatch: ${manifest.version} !== ${version}`);
  }
  if (manifest.module_id !== moduleSpec.smoke.moduleId || manifest.access_scope !== moduleSpec.smoke.accessScope) {
    throw new Error(`${moduleSpec.label} manifest identity/access scope mismatch.`);
  }
  if (manifest.frontend_exposure !== moduleSpec.smoke.frontendExposure) {
    throw new Error(`${moduleSpec.label} manifest must remain marked ${moduleSpec.smoke.frontendExposure}.`);
  }
  if (!Array.isArray(patterns) || patterns.length !== moduleSpec.smoke.expectedPatternCount || !patterns.includes(moduleSpec.smoke.requiredPattern)) {
    throw new Error(`${moduleSpec.label} target pattern runtime export is not stable.`);
  }
  if (!Array.isArray(manifest.target_patterns) || manifest.target_patterns.join("|") !== patterns.join("|")) {
    throw new Error(`${moduleSpec.label} manifest target_patterns must match ${moduleSpec.smoke.patternsExport}().`);
  }
  if (!Array.isArray(manifest.capabilities) || manifest.capabilities.length < requiredCapabilities.size) {
    throw new Error(`${moduleSpec.label} manifest capabilities are incomplete.`);
  }
  const capabilityIds = new Set(manifest.capabilities.map((capability) => capability.id));
  for (const capabilityId of requiredCapabilities) {
    if (!capabilityIds.has(capabilityId)) {
      throw new Error(`${moduleSpec.label} manifest is missing capability ${capabilityId}.`);
    }
  }
  if (!manifest.output_schemas?.includes(moduleSpec.smoke.requiredOutputSchema)) {
    throw new Error(`${moduleSpec.label} manifest is missing required output schema ${moduleSpec.smoke.requiredOutputSchema}.`);
  }
  for (const runtimeCall of moduleSpec.smoke.runtimeCalls || []) {
    smokePrivateWasmRuntimeCall(privateModule, moduleSpec, runtimeCall);
  }
}

function smokePrivateWasmRuntimeCall(privateModule, moduleSpec, runtimeCall) {
  const callable = privateModule[runtimeCall.exportName];
  if (typeof callable !== "function") {
    throw new Error(`${moduleSpec.label} runtime smoke export is missing: ${runtimeCall.exportName}`);
  }
  const result = callable(...(runtimeCall.args || []).map(materializeSmokeArg));
  for (const fieldPath of runtimeCall.requiredFields || []) {
    if (valueAtPath(result, fieldPath) === undefined) {
      throw new Error(`${moduleSpec.label} ${runtimeCall.exportName} missing runtime smoke field: ${fieldPath}`);
    }
  }
  for (const [fieldPath, expectedValue] of Object.entries(runtimeCall.expectedValues || {})) {
    const actualValue = valueAtPath(result, fieldPath);
    if (actualValue !== expectedValue) {
      throw new Error(
        `${moduleSpec.label} ${runtimeCall.exportName} expected ${fieldPath}=${JSON.stringify(expectedValue)}, got ${JSON.stringify(actualValue)}.`,
      );
    }
  }
  for (const [fieldPath, minValue, maxValue] of runtimeCall.numericRanges || []) {
    const actualValue = valueAtPath(result, fieldPath);
    if (typeof actualValue !== "number" || actualValue < minValue || actualValue > maxValue) {
      throw new Error(
        `${moduleSpec.label} ${runtimeCall.exportName} expected ${fieldPath} in [${minValue}, ${maxValue}], got ${JSON.stringify(actualValue)}.`,
      );
    }
  }
}

function materializeSmokeArg(arg) {
  if (!arg || typeof arg !== "object" || !arg.typedArray) return arg;
  if (arg.typedArray === "Float32Array") return new Float32Array(arg.values || []);
  if (arg.typedArray === "Uint32Array") return new Uint32Array(arg.values || []);
  throw new Error(`Unsupported private WASM smoke typed array: ${arg.typedArray}`);
}

function valueAtPath(value, pathSpec) {
  return String(pathSpec)
    .split(".")
    .reduce((current, part) => {
      if (current === undefined || current === null) return undefined;
      const key = Array.isArray(current) && /^\d+$/.test(part) ? Number(part) : part;
      return current[key];
    }, value);
}

function collectFiles(root) {
  const files = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const filePath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectFiles(filePath));
    } else if (entry.isFile()) {
      files.push(filePath);
    }
  }
  return files;
}

function formatBytes(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
}

function toGitPath(filePath) {
  return filePath.replaceAll(path.sep, "/");
}
