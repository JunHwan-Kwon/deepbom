import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { RELEASE_GENERATED_TRACKED_ARTIFACTS } from "./release-generated-artifacts.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outputPath = path.join(root, "web", "lib", "build-metadata.js");

const ONNX_RULEPACK_SOURCES = [
  "web/onnx.js",
  "web/lib/onnx-container-inference.js",
  "web/lib/onnx-domain-analysis.js",
  "web/lib/onnx-external-data.js",
  "web/lib/onnx-extended-shape-inference.js",
  "web/lib/onnx-ml-imputer-conformance.js",
  "web/lib/onnx-ml-imputer.js",
  "web/lib/onnx-ml-label-encoder-conformance.js",
  "web/lib/onnx-ml-label-encoder.js",
  "web/lib/onnx-ml-linear-model-conformance.js",
  "web/lib/onnx-ml-linear-model.js",
  "web/lib/onnx-ml-normalizer-conformance.js",
  "web/lib/onnx-ml-normalizer.js",
  "web/lib/onnx-ml-one-hot-encoder-conformance.js",
  "web/lib/onnx-ml-one-hot-encoder.js",
  "web/lib/onnx-ml-scaler-conformance.js",
  "web/lib/onnx-ml-scaler.js",
  "web/lib/onnx-ml-svm-conformance.js",
  "web/lib/onnx-ml-svm.js",
  "web/lib/onnx-ml-tree-ensemble-conformance.js",
  "web/lib/onnx-ml-tree-ensemble.js",
  "web/lib/onnx-ml-value-inference.js",
  "web/lib/onnx-ort-extension-shape-inference.js",
  "web/lib/onnx-ort-matmul-shape-inference.js",
  "web/lib/onnx-ort-recurrent-shape-inference.js",
  "web/lib/onnx-ort-transformer-shape-inference.js",
  "web/lib/onnx-opset-imports.js",
  "web/lib/onnx-schema-legality.js",
  "web/lib/onnx-shape-inference.js",
  "web/lib/onnx-runtime-shape-binding.js",
  "web/lib/onnx-shape-ops.js",
  "web/lib/onnx-shape-schema-generated.js",
  "web/lib/onnx-sparse-tensor.js",
  "web/lib/onnx-static-value-evidence.js",
  "web/lib/onnx-tfidf-vectorizer-conformance.js",
  "web/lib/onnx-tfidf-vectorizer.js",
  "web/lib/onnx-type-proto.js",
];

const RULEPACK_SOURCE_SLICES = [
  {
    label: "report metadata/provenance",
    path: "web/lib/report-metadata.js",
    start: "export const ANALYZER_METADATA",
    end: "export const RUNTIME_COMPATIBILITY_EVIDENCE_LABEL",
  },
  {
    label: "reason-code labels",
    path: "web/lib/reason-codes.js",
  },
  {
    label: "generated XNNPACK delegate provenance",
    path: "web/lib/xnnpack-rulepack-metadata.js",
  },
  {
    label: "source-mapped XNNPACK delegate evaluator",
    path: "src/xnnpack_delegate.rs",
  },
  {
    label: "generated XNNPACK delegate documentary rules",
    path: "src/xnnpack_rulepack_generated.rs",
  },
  {
    label: "pinned XNNPACK delegate README snapshot",
    path: "reference/xnnpack-readme/README-v2.21.0.md",
  },
  {
    label: "XNNPACK delegate semantic rule manifest",
    path: "reference/xnnpack-readme/rule-manifest-v2.21.0.json",
  },
  {
    label: "XNNPACK delegate rule generator and source digest contract",
    path: "scripts/generate-xnnpack-delegate-rulepack.mjs",
  },
  {
    label: "target profiles and target-specific planning contracts",
    path: "src/target_profiles.rs",
  },
  {
    label: "quantization classification and XNNPACK planning rules",
    path: "src/lib.rs",
    start: "fn fused_activation_for_op(",
    end: "fn annotate_fusion(",
  },
  {
    label: "TFLite runtime-version floor derivation",
    path: "src/lib.rs",
    start: "fn compute_runtime_compat(",
    end: "fn compute_metadata_presence(",
  },
  {
    label: "pinned TFLite runtime-version floor map",
    path: "src/runtime_version.rs",
  },
  {
    label: "TFLite M001 metadata contract parser",
    path: "src/tflite_metadata.rs",
  },
  {
    label: "exact stored-weight accumulator envelope",
    path: "src/accumulator_atlas.rs",
  },
  {
    label: "independent browser accumulator validator",
    path: "web/lib/accumulator-atlas.js",
  },
  {
    label: "pinned TFLite requantization fidelity solver",
    path: "src/requantization_fidelity.rs",
  },
  {
    label: "independent browser requantization validator",
    path: "web/lib/requantization-fidelity.js",
  },
  {
    label: "exact quantized kernel extremum witness solver",
    path: "src/kernel_witness.rs",
  },
  {
    label: "independent browser kernel witness validator and viewer",
    path: "web/lib/kernel-witness.js",
  },
  {
    label: "independent browser kernel witness verification worker",
    path: "web/lib/kernel-witness-worker.js",
  },
  {
    label: "exact monotone quantized channel vitality solver",
    path: "src/channel_vitality.rs",
  },
  {
    label: "independent browser channel vitality validator and viewer",
    path: "web/lib/channel-vitality.js",
  },
  {
    label: "independent browser channel vitality verification worker",
    path: "web/lib/channel-vitality-worker.js",
  },
  {
    label: "exact fixed-point rounding equivalence solver",
    path: "src/rounding_equivalence.rs",
  },
  {
    label: "independent browser rounding equivalence validator and viewer",
    path: "web/lib/rounding-equivalence.js",
  },
  {
    label: "independent browser rounding equivalence verification worker",
    path: "web/lib/rounding-equivalence-worker.js",
  },
  {
    label: "exact numerical ABI graph propagation solver",
    path: "src/numerical_abi_propagation.rs",
  },
  {
    label: "independent browser numerical ABI propagation validator",
    path: "web/lib/numerical-abi-propagation.js",
  },
  {
    label: "numerical ABI propagation atlas viewer and export renderer",
    path: "web/lib/numerical-abi-propagation-viewer.js",
  },
  {
    label: "exhaustive residual output-contract design solver",
    path: "src/quantization_lattice.rs",
  },
  {
    label: "independent browser residual contract validator",
    path: "web/lib/quantization-lattice.js",
  },
  {
    label: "shared pinned quantization arithmetic",
    path: "src/quantization_math.rs",
  },
  {
    label: "independent browser quantization arithmetic",
    path: "web/lib/quantization-math.js",
  },
  {
    label: "residual contract migration impact solver",
    path: "src/contract_migration.rs",
  },
  {
    label: "independent browser migration validator",
    path: "web/lib/contract-migration.js",
  },
  {
    label: "exhaustive residual adjacent-code step-response solver",
    path: "src/residual_step_response.rs",
  },
  {
    label: "independent browser residual step-response validator",
    path: "web/lib/residual-step-response.js",
  },
  {
    label: "independent browser residual step-response worker",
    path: "web/lib/residual-step-response-worker.js",
  },
  {
    label: "exhaustive residual contract-distortion solver",
    path: "src/residual_contract_distortion.rs",
  },
  {
    label: "independent browser residual contract-distortion validator",
    path: "web/lib/residual-contract-distortion.js",
  },
  {
    label: "independent browser residual contract-distortion worker",
    path: "web/lib/residual-contract-distortion-worker.js",
  },
  {
    label: "protected XNNPACK source selector",
    path: "protected/deepbom_wasm/src/xnnpack_selector.rs",
  },
  {
    label: "protected ORT compatibility derivation",
    path: "protected/deepbom_wasm/src/ort_rulepack.rs",
  },
  {
    label: "generated pinned ORT EP and release rules",
    path: "protected/deepbom_wasm/src/ort_rulepack_generated.rs",
  },
  {
    label: "pinned ORT rule generator and source digest contract",
    path: "scripts/generate-ort-rulepack.mjs",
  },
  {
    label: "public XNNPACK evidence identity and arithmetic adapter",
    path: "web/lib/xnnpack-selector-evidence.js",
  },
  {
    label: "public ORT evidence identity and conservation adapter",
    path: "web/lib/ort-compatibility-evidence.js",
  },
  ...ONNX_RULEPACK_SOURCES.map((sourcePath) => ({
    label: "ONNX rule/reconstruction",
    path: sourcePath,
  })),
  {
    label: "authoritative finding rules",
    path: "web/lib/report-findings.js",
  },
  {
    label: "finding evidence-pointer contract",
    path: "web/lib/finding-contract.js",
  },
];

const BUNDLE_CONTENT_ROOTS = [
  "web",
  path.join("pkg", "tflite_wasm_audit.js"),
  path.join("pkg", "tflite_wasm_audit_bg.wasm"),
  "package.json",
  "package-lock.json",
  "Cargo.toml",
  "Cargo.lock",
  "build.rs",
  "src",
  "wrangler.jsonc",
  "worker",
];

const BUNDLE_CONTENT_FILE_EXCLUDES = new Set([
  "web/lib/build-metadata.js",
  "web/samples/mobilenet_v1_025_224_float.tflite",
  "web/samples/sample_cnn_float.onnx",
]);

const BUNDLE_CONTENT_DIR_EXCLUDES = new Set([
  ".git",
  ".wrangler",
  "dist",
  "node_modules",
  "target",
]);

const BUILD_FILE_DIGEST_SET_SCHEMA = "deepbom.analyzer_build_file_digest_set.v1";
const BUILD_CONTENT_MANIFEST_SCHEMA = "deepbom.analyzer_build_content_manifest.v1";
const BUILD_CONTENT_HASH_METHOD = `SHA-256 over RFC8785-JCS canonical ${BUILD_FILE_DIGEST_SET_SCHEMA}`;
export function writeBuildMetadata({ log = true, publicDistribution = false } = {}) {
  const commit = git(["rev-parse", "HEAD"]).trim();
  const releaseBuild = releaseCleanTreeRequired();
  const dirty = isWorkingTreeDirty({ ignoreReleaseGenerated: releaseBuild });
  if (dirty && releaseBuild) {
    throw new Error("Formal release builds require a clean git tree. Dirty development builds may proceed only without the release-clean-tree flag and will embed a bundle content SHA-256.");
  }
  const rulepack = computeRulepackDigest({ publicDistribution });
  const bundle = computeBundleContentDigest({ publicDistribution });
  const source = generatedModuleSource({
    commit,
    dirty,
    rulepackSha256: rulepack.sha256,
    rulepackHashBasis: publicRulepackHashBasis(rulepack.basis),
    bundleContentSha256: bundle.sha256,
    bundleHashBasis: bundle.basis,
    bundleContentManifest: bundle.manifest,
    bundleContentManifestSha256: bundle.manifestSha256,
    sourceDistribution: publicDistribution ? "public_channel" : "private_monorepo",
  });
  writeFileSync(outputPath, source);
  if (log) {
    console.log(`Build metadata written (${commit.slice(0, 12)}${dirty ? "+dirty" : ""}, rulepack ${rulepack.sha256.slice(0, 12)}..., bundle ${bundle.sha256.slice(0, 12)}...)`);
  }
  return {
    commit,
    sourceState: dirty ? "working-tree-dirty" : "clean",
    rulepackSha256: rulepack.sha256,
    bundleContentSha256: bundle.sha256,
    bundleContentManifestSha256: bundle.manifestSha256,
  };
}

export function computeRulepackDigest({ publicDistribution = false } = {}) {
  const hash = createHash("sha256");
  const basis = [];
  const optionalPrivateSources = new Set([
    "scripts/generate-ort-rulepack.mjs",
    "scripts/generate-xnnpack-delegate-rulepack.mjs",
  ]);
  for (const item of RULEPACK_SOURCE_SLICES) {
    const absolute = path.join(root, item.path);
    const optionalPrivate = item.path.startsWith("protected/deepbom_wasm/src/") || optionalPrivateSources.has(item.path);
    if (publicDistribution && optionalPrivate) continue;
    if (!existsSync(absolute)) {
      if (optionalPrivate) continue;
      throw new Error(`Required public rulepack source is missing: ${item.path}`);
    }
    const fullSource = readFileSync(absolute, "utf8");
    const source = sliceSource(fullSource, item);
    hash.update(`\n--- ${item.path} :: ${item.label} ---\n`);
    hash.update(source.replace(/\r\n/g, "\n"));
    basis.push(`${item.path} (${item.label})`);
  }
  if (!basis.length) throw new Error("Rulepack digest has no public source basis.");
  return { sha256: hash.digest("hex"), basis };
}

export function publicRulepackHashBasis(basis) {
  const protectedPrefix = "protected/deepbom_wasm/src/";
  const protectedCount = basis.filter((item) => item.startsWith(protectedPrefix)).length;
  return [
    ...basis.filter((item) => !item.startsWith(protectedPrefix)),
    ...(protectedCount
      ? [`protected authenticated rule source set (${protectedCount} owners; implementation paths withheld; included in RULEPACK_SHA256)`]
      : []),
  ];
}

function computeBundleContentDigest({ publicDistribution = false } = {}) {
  const files = collectBundleContentFiles({ publicDistribution });
  const selectedRoots = BUNDLE_CONTENT_ROOTS
    .map(normalizePath)
    .filter((entry) => existsSync(path.join(root, entry)));
  const fileDigests = files.map((relativePath) => {
    const bytes = readFileSync(path.join(root, relativePath));
    return {
      path: relativePath,
      size: bytes.byteLength,
      sha256: sha256Hex(bytes),
    };
  });
  const digestSet = {
    schema: BUILD_FILE_DIGEST_SET_SCHEMA,
    files: fileDigests,
  };
  const sha256 = sha256Hex(canonicalJson(digestSet));
  const generatorPath = normalizePath(path.relative(root, fileURLToPath(import.meta.url)));
  const manifest = {
    schema: BUILD_CONTENT_MANIFEST_SCHEMA,
    content_digest: {
      sha256,
      method: BUILD_CONTENT_HASH_METHOD,
      canonicalization: "RFC8785-JCS",
      digest_set_schema: BUILD_FILE_DIGEST_SET_SCHEMA,
    },
    selection: {
      roots: selectedRoots,
      file_excludes: [...BUNDLE_CONTENT_FILE_EXCLUDES, "**/.gitignore"].sort(),
      directory_excludes: [
        ...BUNDLE_CONTENT_DIR_EXCLUDES,
        ...(publicDistribution ? ["web/protected"] : []),
      ].sort(),
      ordering: "UTF-16 code-unit ascending normalized relative paths",
      path_separator: "/",
    },
    generator: {
      path: generatorPath,
      sha256: sha256Hex(readFileSync(fileURLToPath(import.meta.url))),
      toolchain: {
        node: process.version,
        npm: toolVersion(process.platform === "win32" ? "npm.cmd" : "npm", ["--version"]),
        rustc: toolVersion("rustc", ["--version"]),
        wasm_pack: toolVersion("wasm-pack", ["--version"]),
      },
    },
    release_inputs: {
      app_expires_at_epoch_ms: process.env.APP_EXPIRES_AT_EPOCH_MS || "not_set",
      app_not_before_epoch_ms: process.env.APP_NOT_BEFORE_EPOCH_MS || "not_set",
      app_expiry_days: process.env.APP_EXPIRY_DAYS || "not_set",
    },
    files: fileDigests,
  };
  const manifestSha256 = sha256Hex(canonicalJson(manifest));
  return {
    sha256,
    manifest,
    manifestSha256,
    basis: [
      `${files.length} files`,
      BUILD_CONTENT_HASH_METHOD,
      `manifest ${BUILD_CONTENT_MANIFEST_SCHEMA} SHA-256 ${manifestSha256}`,
      `roots: ${selectedRoots.join(", ")}`,
      `file excludes: ${manifest.selection.file_excludes.join(", ")}`,
      `directory excludes: ${manifest.selection.directory_excludes.join(", ")}`,
    ],
  };
}

function collectBundleContentFiles({ publicDistribution = false } = {}) {
  const files = [];
  for (const entry of BUNDLE_CONTENT_ROOTS) {
    const normalized = normalizePath(entry);
    const absolute = path.join(root, normalized);
    if (!existsSync(absolute)) continue;
    const entryStat = statSync(absolute);
    if (entryStat.isDirectory()) {
      walkBundleContentDir(normalized, files, { publicDistribution });
    } else if (entryStat.isFile() && !shouldExcludeBundleContentFile(normalized)) {
      files.push(normalized);
    }
  }
  return [...new Set(files)].sort();
}

function walkBundleContentDir(relativeDir, files, { publicDistribution = false } = {}) {
  const entries = readdirSync(path.join(root, relativeDir), { withFileTypes: true });
  for (const entry of entries) {
    if (BUNDLE_CONTENT_DIR_EXCLUDES.has(entry.name)) continue;
    const relativePath = normalizePath(path.join(relativeDir, entry.name));
    if (publicDistribution && (relativePath === "web/protected" || relativePath.startsWith("web/protected/"))) continue;
    if (entry.isDirectory()) {
      walkBundleContentDir(relativePath, files, { publicDistribution });
    } else if (entry.isFile() && !shouldExcludeBundleContentFile(relativePath)) {
      files.push(relativePath);
    }
  }
}

function shouldExcludeBundleContentFile(relativePath) {
  return BUNDLE_CONTENT_FILE_EXCLUDES.has(relativePath) || path.basename(relativePath) === ".gitignore";
}

function sliceSource(source, item) {
  if (!item.start) {
    return source;
  }
  const start = source.indexOf(item.start);
  const end = source.indexOf(item.end, start + item.start.length);
  if (start < 0 || end < 0 || end <= start) {
    throw new Error(`Could not find rulepack hash slice ${item.path}: ${item.start}..${item.end}`);
  }
  return source.slice(start, end);
}

function isWorkingTreeDirty({ ignoreReleaseGenerated = false } = {}) {
  const args = [
    "status",
    "--porcelain=v1",
    "--untracked-files=all",
    "--",
    ".",
    ":!web/lib/build-metadata.js",
    ":!dist",
  ];
  if (ignoreReleaseGenerated) args.push(...RELEASE_GENERATED_TRACKED_ARTIFACTS.map((entry) => `:!${entry}`));
  const status = git(args);
  return status.trim().length > 0;
}

function git(args) {
  return execFileSync("git", args, { cwd: root, encoding: "utf8" });
}

function releaseCleanTreeRequired() {
  return ["DEEPBOM_RELEASE_BUILD", "RELEASE_BUILD", "CI_RELEASE"].some((name) => {
    const value = String(process.env[name] || "").trim().toLowerCase();
    return value === "1" || value === "true" || value === "yes";
  });
}

function normalizePath(value) {
  return value.replaceAll(path.sep, "/");
}

function sha256Hex(value) {
  return createHash("sha256").update(value).digest("hex");
}

function toolVersion(command, args) {
  try {
    return execFileSync(command, args, { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim() || "not_available";
  } catch {
    return "not_available";
  }
}

function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
  const keys = Object.keys(value).filter((key) => value[key] !== undefined).sort();
  return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
}

function generatedModuleSource({
  commit,
  dirty,
  rulepackSha256,
  rulepackHashBasis,
  bundleContentSha256,
  bundleHashBasis,
  bundleContentManifest,
  bundleContentManifestSha256,
  sourceDistribution,
}) {
  return [
    "// Generated by scripts/write-build-metadata.mjs. Do not edit by hand.",
    `export const ANALYZER_BUILD_COMMIT = ${JSON.stringify(commit)};`,
    `export const ANALYZER_BUILD_SOURCE_STATE = ${JSON.stringify(dirty ? "working-tree-dirty" : "clean")};`,
    `export const ANALYZER_SOURCE_DISTRIBUTION = ${JSON.stringify(sourceDistribution)};`,
    `export const ANALYZER_BUNDLE_CONTENT_SHA256 = ${JSON.stringify(bundleContentSha256)};`,
    `export const ANALYZER_BUNDLE_CONTENT_HASH_METHOD = ${JSON.stringify(BUILD_CONTENT_HASH_METHOD)};`,
    `export const ANALYZER_BUNDLE_HASH_BASIS = Object.freeze(${JSON.stringify(bundleHashBasis, null, 2)});`,
    `export const ANALYZER_BUNDLE_CONTENT_MANIFEST_SHA256 = ${JSON.stringify(bundleContentManifestSha256)};`,
    `export const ANALYZER_BUNDLE_CONTENT_MANIFEST = Object.freeze(${JSON.stringify(bundleContentManifest, null, 2)});`,
    `export const RULEPACK_SHA256 = ${JSON.stringify(rulepackSha256)};`,
    `export const RULEPACK_HASH_BASIS = Object.freeze(${JSON.stringify(rulepackHashBasis, null, 2)});`,
    "",
  ].join("\n");
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const autoDistribution = process.argv.includes("--auto-distribution");
  const publicDistribution = process.argv.includes("--public-distribution")
    || (autoDistribution && !existsSync(path.join(root, "protected", "deepbom_wasm", "src", "lib.rs")));
  writeBuildMetadata({ publicDistribution });
}
