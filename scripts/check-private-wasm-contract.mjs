import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";

import { PRIVATE_WASM_MODULES } from "./private-wasm-modules.mjs";

const gitignore = readFileSync(".gitignore", "utf8");

function fail(message) {
  console.error(message);
  process.exit(1);
}

function toGitPath(filePath) {
  return filePath.replaceAll("\\", "/");
}

function assertList(name, values, moduleSpec) {
  if (!Array.isArray(values) || values.length === 0) {
    fail(`${moduleSpec.label} contract is missing ${name}.`);
  }
}

function wasmExportNames(libSource) {
  return Array.from(
    libSource.matchAll(/#\[wasm_bindgen\]\s*(?:#\[[^\]]+\]\s*)*pub fn\s+([A-Za-z0-9_]+)/g),
    (match) => match[1],
  );
}

function checkExports(moduleSpec, libSource) {
  const expected = moduleSpec.contract.expectedExports;
  const actual = wasmExportNames(libSource);
  const missing = expected.filter((name) => !actual.includes(name));
  const unexpected = actual.filter((name) => !expected.includes(name));
  if (missing.length || unexpected.length) {
    fail(
      [
        `${moduleSpec.label} WASM export contract failed.`,
        `missing=${missing.join(",") || "-"}`,
        `unexpected=${unexpected.join(",") || "-"}`,
      ].join("\n"),
    );
  }
  return actual.length;
}

function checkModules(moduleSpec, libSource) {
  for (const moduleName of moduleSpec.contract.expectedModules) {
    if (!libSource.includes(`mod ${moduleName};`)) {
      fail(`${moduleSpec.label} module contract failed: missing mod ${moduleName};`);
    }
  }
  return moduleSpec.contract.expectedModules.length;
}

function checkSchemas(moduleSpec) {
  const combinedSource = moduleSpec.contract.sourceFiles
    .map((file) => readFileSync(file, "utf8"))
    .join("\n");
  for (const schema of moduleSpec.contract.expectedSchemas) {
    if (!combinedSource.includes(schema)) {
      fail(`${moduleSpec.label} schema contract failed: missing ${schema}`);
    }
  }
  return moduleSpec.contract.expectedSchemas.length;
}

function checkIgnoredRoots(moduleSpec) {
  for (const ignoredRoot of moduleSpec.ignoredRoots) {
    const gitPath = toGitPath(ignoredRoot);
    if (!gitignore.includes(gitPath)) {
      fail(`${moduleSpec.label} ignored root is missing from .gitignore: ${gitPath}`);
    }
  }

  const tracked = execFileSync(
    "git",
    ["ls-files", ...moduleSpec.ignoredRoots],
    { encoding: "utf8" },
  )
    .trim()
    .split(/\r?\n/)
    .filter(Boolean);
  if (tracked.length) {
    fail(
      `${moduleSpec.label} build artifacts are tracked:\n${tracked.join("\n")}`,
    );
  }
}

const summaries = [];
for (const moduleSpec of PRIVATE_WASM_MODULES) {
  const contract = moduleSpec.contract;
  if (!contract) {
    fail(`${moduleSpec.label} is missing a private WASM contract spec.`);
  }
  assertList("expectedExports", contract.expectedExports, moduleSpec);
  assertList("expectedModules", contract.expectedModules, moduleSpec);
  assertList("expectedSchemas", contract.expectedSchemas, moduleSpec);
  assertList("sourceFiles", contract.sourceFiles, moduleSpec);

  const libSource = readFileSync(contract.libFile, "utf8");
  const exportsCount = checkExports(moduleSpec, libSource);
  const modulesCount = checkModules(moduleSpec, libSource);
  const schemasCount = checkSchemas(moduleSpec);
  checkIgnoredRoots(moduleSpec);
  summaries.push(`${moduleSpec.id}: ${exportsCount} exports, ${modulesCount} modules, ${schemasCount} schemas`);
}

console.log(`Private WASM contract passed (${summaries.join("; ")}).`);
