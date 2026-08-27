// Optional private analyzers are intentionally not part of the DEEPBOM repository.
// The registry stays explicit so downstream deployments can validate a local
// extension without changing the public build or source boundary.
export const PRIVATE_WASM_MODULES = [];

export function privateModuleSourceRoots() {
  return PRIVATE_WASM_MODULES.map((moduleSpec) => normalizeSpecPath(moduleSpec.sourceRoot));
}

export function privateModuleSourcePrefixes() {
  return privateModuleSourceRoots().map((root) => root.endsWith("/") ? root : `${root}/`);
}

export function privateModuleCargoManifests() {
  return PRIVATE_WASM_MODULES.map((moduleSpec) => normalizeSpecPath(moduleSpec.cargoManifest));
}

export function privateModuleIgnoredRoots() {
  return PRIVATE_WASM_MODULES.flatMap((moduleSpec) => moduleSpec.ignoredRoots || []).map(normalizeSpecPath);
}

export function privateModuleIgnoredPrefixes() {
  return privateModuleIgnoredRoots().map((root) => root.endsWith("/") ? root : `${root}/`);
}

export function privateModuleDistLeakNeedles() {
  return PRIVATE_WASM_MODULES
    .map((moduleSpec) => moduleSpec.distLeakNeedle)
    .filter(Boolean);
}

export function privateModuleValidationCases() {
  return PRIVATE_WASM_MODULES.map((moduleSpec) => ({
    id: moduleSpec.id,
    label: moduleSpec.label,
    sourceRoot: normalizeSpecPath(moduleSpec.sourceRoot),
    cargoManifest: normalizeSpecPath(moduleSpec.cargoManifest),
    primarySource: normalizeSpecPath(moduleSpec.contract?.libFile || `${moduleSpec.sourceRoot}/src/lib.rs`),
    testSource: normalizeSpecPath(`${moduleSpec.sourceRoot}/src/tests.rs`),
  }));
}

export function normalizeSpecPath(value) {
  return String(value).replace(/\\/g, "/");
}
