import { existsSync, readFileSync } from "node:fs";
import {
  PRIVATE_WASM_MODULES,
  normalizeSpecPath,
  privateModuleCargoManifests,
  privateModuleDistLeakNeedles,
  privateModuleIgnoredPrefixes,
  privateModuleIgnoredRoots,
  privateModuleSourcePrefixes,
  privateModuleSourceRoots,
  privateModuleValidationCases,
} from "./private-wasm-modules.mjs";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("Private WASM registry contract check");
const packageJson = JSON.parse(readFileSync("package.json", "utf8"));
const packageScripts = packageJson.scripts || {};

expect(Array.isArray(PRIVATE_WASM_MODULES), "PRIVATE_WASM_MODULES should be an array.");

const ids = new Set();
const buildScripts = new Set();
const sourceRoots = new Set();
const cargoManifests = new Set();
const leakNeedles = new Set();

for (const moduleSpec of PRIVATE_WASM_MODULES) {
  expect(typeof moduleSpec.id === "string" && /^[a-z][a-z0-9_]*$/.test(moduleSpec.id), `${moduleSpec.label || "module"} id should be stable snake_case.`);
  expect(!ids.has(moduleSpec.id), `${moduleSpec.id} should be unique.`);
  ids.add(moduleSpec.id);

  expect(typeof moduleSpec.label === "string" && moduleSpec.label.length > 0, `${moduleSpec.id} should have a label.`);
  expect(typeof moduleSpec.buildScript === "string" && moduleSpec.buildScript.length > 0, `${moduleSpec.id} should have a buildScript.`);
  expect(!buildScripts.has(moduleSpec.buildScript), `${moduleSpec.id} buildScript should be unique.`);
  buildScripts.add(moduleSpec.buildScript);

  const sourceRoot = normalizeSpecPath(moduleSpec.sourceRoot || "");
  const cargoManifest = normalizeSpecPath(moduleSpec.cargoManifest || "");
  const pkgRoot = normalizeSpecPath(moduleSpec.pkgRoot || "");
  expect(sourceRoot.length > 0 && existsSync(sourceRoot), `${moduleSpec.id} sourceRoot should exist.`);
  expect(!sourceRoots.has(sourceRoot), `${moduleSpec.id} sourceRoot should be unique.`);
  sourceRoots.add(sourceRoot);
  expect(cargoManifest.length > 0 && existsSync(cargoManifest), `${moduleSpec.id} cargoManifest should exist.`);
  expect(!cargoManifests.has(cargoManifest), `${moduleSpec.id} cargoManifest should be unique.`);
  cargoManifests.add(cargoManifest);
  expect(pkgRoot.startsWith(`${sourceRoot}/`), `${moduleSpec.id} pkgRoot should stay under sourceRoot.`);

  const buildCommand = packageScripts[moduleSpec.buildScript] || "";
  expect(buildCommand.includes(sourceRoot), `${moduleSpec.id} package script should build from sourceRoot.`);
  expect(buildCommand.includes("--out-dir pkg"), `${moduleSpec.id} package script should write to local pkg output.`);

  const ignoredRoots = (moduleSpec.ignoredRoots || []).map(normalizeSpecPath);
  expect(ignoredRoots.includes(pkgRoot), `${moduleSpec.id} ignoredRoots should include pkgRoot.`);
  expect(ignoredRoots.some((root) => root === `${sourceRoot}/target` || root.startsWith(`${sourceRoot}/target`)), `${moduleSpec.id} ignoredRoots should include target output.`);

  expect(typeof moduleSpec.distLeakNeedle === "string" && moduleSpec.distLeakNeedle.length > 0, `${moduleSpec.id} should have a dist leak needle.`);
  expect(!leakNeedles.has(moduleSpec.distLeakNeedle), `${moduleSpec.id} dist leak needle should be unique.`);
  leakNeedles.add(moduleSpec.distLeakNeedle);

  expectArtifactContract(moduleSpec, "wasm", moduleSpec.wasmFile);
  expectArtifactContract(moduleSpec, "js", moduleSpec.glueFile);
  expectArtifactContract(moduleSpec, "types", moduleSpec.typeFile);
  expect(Array.isArray(moduleSpec.requiredTypeExports) && moduleSpec.requiredTypeExports.length > 0, `${moduleSpec.id} should declare required type exports.`);

  const contract = moduleSpec.contract || {};
  expect(existsSync(contract.libFile || ""), `${moduleSpec.id} contract.libFile should exist.`);
  expect(Array.isArray(contract.sourceFiles) && contract.sourceFiles.length > 0, `${moduleSpec.id} contract.sourceFiles should be non-empty.`);
  for (const sourceFile of contract.sourceFiles || []) {
    expect(existsSync(sourceFile), `${moduleSpec.id} contract source file should exist: ${normalizeSpecPath(sourceFile)}`);
  }
  expect(Array.isArray(contract.expectedExports) && contract.expectedExports.length > 0, `${moduleSpec.id} should declare expected exports.`);
  expect(Array.isArray(contract.expectedModules) && contract.expectedModules.length > 0, `${moduleSpec.id} should declare expected Rust modules.`);
  expect(Array.isArray(contract.expectedSchemas) && contract.expectedSchemas.length > 0, `${moduleSpec.id} should declare expected schemas.`);

  const smoke = moduleSpec.smoke || {};
  for (const field of ["versionExport", "patternsExport", "manifestExport", "manifestSchema", "moduleId", "accessScope", "frontendExposure", "requiredPattern", "requiredOutputSchema"]) {
    expect(typeof smoke[field] === "string" && smoke[field].length > 0, `${moduleSpec.id} smoke.${field} should be set.`);
  }
  expect(smoke.moduleId === moduleSpec.id, `${moduleSpec.id} smoke.moduleId should match module id.`);
  expect(Number.isInteger(smoke.expectedPatternCount) && smoke.expectedPatternCount > 0, `${moduleSpec.id} smoke.expectedPatternCount should be positive.`);
  expect(Array.isArray(smoke.requiredCapabilities) && smoke.requiredCapabilities.length > 0, `${moduleSpec.id} smoke.requiredCapabilities should be non-empty.`);
  for (const runtimeCall of smoke.runtimeCalls || []) {
    expect(typeof runtimeCall.exportName === "string" && runtimeCall.exportName.length > 0, `${moduleSpec.id} runtime smoke call should name an export.`);
    expect((contract.expectedExports || []).includes(runtimeCall.exportName), `${moduleSpec.id} runtime smoke export should be in contract.expectedExports: ${runtimeCall.exportName}`);
    expect(Array.isArray(runtimeCall.args), `${moduleSpec.id} runtime smoke call should declare args.`);
    for (const arg of runtimeCall.args) {
      if (!arg || typeof arg !== "object" || !arg.typedArray) continue;
      expect(["Float32Array", "Uint32Array"].includes(arg.typedArray), `${moduleSpec.id} runtime smoke typed array is supported: ${arg.typedArray}`);
      expect(Array.isArray(arg.values), `${moduleSpec.id} runtime smoke typed array should declare values.`);
    }
    expect(Array.isArray(runtimeCall.requiredFields) && runtimeCall.requiredFields.length > 0, `${moduleSpec.id} runtime smoke call should assert required fields.`);
    expect(Array.isArray(runtimeCall.numericRanges || []), `${moduleSpec.id} runtime smoke numericRanges should be an array when set.`);
    for (const range of runtimeCall.numericRanges || []) {
      expect(Array.isArray(range) && range.length === 3, `${moduleSpec.id} runtime smoke numeric range should be [path, min, max].`);
      if (Array.isArray(range) && range.length === 3) {
        expect(typeof range[0] === "string" && range[0].length > 0, `${moduleSpec.id} runtime smoke numeric range path should be set.`);
        expect(Number.isFinite(range[1]) && Number.isFinite(range[2]) && range[1] <= range[2], `${moduleSpec.id} runtime smoke numeric range bounds should be finite and ordered.`);
      }
    }
  }
}

expectEqual(privateModuleSourceRoots().length, PRIVATE_WASM_MODULES.length, "source root helper should cover every private module.");
expectEqual(privateModuleSourcePrefixes().length, PRIVATE_WASM_MODULES.length, "source prefix helper should cover every private module.");
expect(privateModuleSourcePrefixes().every((prefix) => prefix.endsWith("/")), "source prefixes should be slash-terminated.");
expectEqual(privateModuleCargoManifests().length, PRIVATE_WASM_MODULES.length, "cargo manifest helper should cover every private module.");
expect(privateModuleIgnoredRoots().length >= PRIVATE_WASM_MODULES.length * 2, "ignored roots helper should include pkg and target roots.");
expect(privateModuleIgnoredPrefixes().every((prefix) => prefix.endsWith("/")), "ignored prefixes should be slash-terminated.");
expectEqual(privateModuleDistLeakNeedles().length, PRIVATE_WASM_MODULES.length, "dist leak helper should cover every private module.");
const validationCases = privateModuleValidationCases();
expectEqual(validationCases.length, PRIVATE_WASM_MODULES.length, "validation case helper should cover every private module.");
for (const validationCase of validationCases) {
  expect(ids.has(validationCase.id), `${validationCase.id} validation case should refer to a registered module.`);
  expect(sourceRoots.has(validationCase.sourceRoot), `${validationCase.id} validation sourceRoot should match registry.`);
  expect(cargoManifests.has(validationCase.cargoManifest), `${validationCase.id} validation cargoManifest should match registry.`);
  expect(validationCase.primarySource.startsWith(`${validationCase.sourceRoot}/`), `${validationCase.id} validation primarySource should stay under sourceRoot.`);
  expect(validationCase.testSource.startsWith(`${validationCase.sourceRoot}/`), `${validationCase.id} validation testSource should stay under sourceRoot.`);
}

done(`Private WASM registry contract passed (${PRIVATE_WASM_MODULES.length} module spec(s)).`);

function expectArtifactContract(moduleSpec, artifactKind, fileName) {
  const artifact = (moduleSpec.expectedArtifacts || []).find(([kind]) => kind === artifactKind);
  expect(Boolean(artifact), `${moduleSpec.id} expectedArtifacts should include ${artifactKind}.`);
  if (!artifact) return;
  expectEqual(artifact[1], fileName, `${moduleSpec.id} ${artifactKind} artifact should match top-level field.`);
  expect(Number.isFinite(artifact[2]) && artifact[2] > 0, `${moduleSpec.id} ${artifactKind} artifact budget should be positive.`);
}
