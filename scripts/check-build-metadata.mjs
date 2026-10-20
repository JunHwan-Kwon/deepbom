import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import {
  ANALYZER_BUNDLE_CONTENT_MANIFEST,
  ANALYZER_BUNDLE_CONTENT_MANIFEST_SHA256,
  ANALYZER_BUNDLE_CONTENT_SHA256,
  ANALYZER_BUILD_COMMIT,
  ANALYZER_BUILD_SOURCE_STATE,
  ANALYZER_SOURCE_DISTRIBUTION,
  RULEPACK_HASH_BASIS,
  RULEPACK_SHA256,
} from "../web/lib/build-metadata.js";
import {
  ANALYZER_SEMANTIC_VERSION,
  ANALYZER_VERSION,
  RULEPACK_VERSION,
} from "../web/lib/app-config.js";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import {
  computeRulepackDigest,
  publicRulepackHashBasis,
} from "./write-build-metadata.mjs";

const manifest = ANALYZER_BUNDLE_CONTENT_MANIFEST;
if (manifest?.schema !== "deepbom.analyzer_build_content_manifest.v1") throw new Error("Unexpected analyzer build-content manifest schema.");
if (!/^[0-9a-f]{40}$/.test(ANALYZER_BUILD_COMMIT)) throw new Error("Analyzer build commit is missing or malformed.");
if (!manifest.files?.length) throw new Error("Analyzer build-content manifest has no files.");
const observedRulepack = computeRulepackDigest();
for (const owner of ["config/scalar-types.v1.json", "src/tensor_math.rs", "src/tflite_subgraphs.rs", "src/portable_numerics.rs", "src/portable_spectrum.rs", "src/portable_weight.rs", "src/portable_distribution.rs", "src/portable_affine.rs", "src/portable_pruning.rs", "src/portable_metrics.rs", "src/portable_contracts.rs", "src/portable_graph.rs", "src/portable_cost.rs", "src/portable_integer.rs", "src/portable_einsum.rs", "src/portable_protobuf.rs", "src/portable_shape.rs", "src/portable_redesign.rs", "src/portable_evidence.rs", "src/portable_identity.rs", "src/portable_native.rs", "src/portable_onnx_type.rs", "src/portable_ir.rs", "src/portable_coreml.rs", "src/portable_workflow.rs", "src/portable_decimal.rs", "src/portable_activation_validation.rs", "src/portable_artifact_validation.rs", "src/portable_conversion.rs", "src/portable_model_validation.rs", "src/portable_numerical_validation.rs", "src/portable_onnx_conditional.rs", "src/portable_onnx_container.rs", "src/portable_onnx_dynamic.rs", "src/portable_onnx_elementwise.rs", "src/portable_onnx_merge.rs", "src/portable_onnx_ml.rs", "src/portable_onnx_ml_label.rs", "src/portable_onnx_ml_linear.rs", "src/portable_onnx_ml_mapping.rs", "src/portable_onnx_ml_schema.rs", "src/portable_onnx_ml_svm.rs", "src/portable_onnx_ml_svm_eval.rs", "src/portable_onnx_ml_tree.rs", "src/portable_onnx_ml_tree_data.rs", "src/portable_onnx_ml_tree_eval.rs", "src/portable_onnx_ml_tree_graph.rs", "src/portable_onnx_node.rs", "src/portable_onnx_ort.rs", "src/portable_onnx_regions.rs", "src/portable_onnx_semantics.rs", "src/portable_onnx_slice.rs", "src/portable_onnx_special.rs", "src/portable_onnx_tensor.rs",
    "src/portable_onnx_tfidf.rs", "src/portable_provenance.rs", "src/portable_runtime_overlay.rs", "src/portable_validation.rs", "src/onnx_ml_row_templates.json", "src/onnx_ml_schemas.json", "src/onnx_ort_specs.json", "src/onnx_shape_groups.json",  "web/lib/onnx-shape-inference.js", "web/lib/onnx-dimension-expression.js", "web/lib/redesign-codegen.js", "web/lib/evidence-identity.js", "web/lib/numerical-wasm.js", "web/lib/numerical-ir/exact-moments.js", "web/lib/numerical-ir/statistics.js", "web/lib/numerical-ir/weight-math.js", "web/lib/numerical-ir/distribution.js", "web/lib/numerical-ir/weight-contracts.js", "web/lib/weight-pruning.js", "web/lib/builtin-metrics.js", "web/lib/tensor-size.js", "web/lib/scalar-numeric.js", "web/lib/onnx-tensor-types.js", "web/lib/gguf-storage-types.js"]) {
  if (!observedRulepack.basis.some(item => item.startsWith(owner))) {
    throw new Error(`Common calculation owner is missing from rulepack identity: ${owner}`);
  }
}
if (observedRulepack.sha256 !== RULEPACK_SHA256
  || canonicalJson(publicRulepackHashBasis(observedRulepack.basis)) !== canonicalJson(RULEPACK_HASH_BASIS)) {
  throw new Error("Rulepack SHA-256 or source basis does not match the current pinned rule sources.");
}
const hasProtectedPath = RULEPACK_HASH_BASIS.some((item) => item.includes("protected/deepbom_wasm/src/"));
const hasWithheldPrivateBasis = RULEPACK_HASH_BASIS.some((item) => item.includes("implementation paths withheld"));
if (hasProtectedPath
  || (ANALYZER_SOURCE_DISTRIBUTION === "private_monorepo" && !hasWithheldPrivateBasis)
  || (ANALYZER_SOURCE_DISTRIBUTION === "public_channel" && hasWithheldPrivateBasis)) {
  throw new Error("Rulepack provenance does not match its declared source distribution.");
}
if (!["private_monorepo", "public_channel"].includes(ANALYZER_SOURCE_DISTRIBUTION)) {
  throw new Error(`Unknown analyzer source distribution ${ANALYZER_SOURCE_DISTRIBUTION}.`);
}

const observedFiles = [];
for (const item of manifest.files) {
  const bytes = await readFile(item.path);
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (bytes.byteLength !== item.size) throw new Error(`Build-content size mismatch for ${item.path}.`);
  if (sha256 !== item.sha256) throw new Error(`Build-content SHA-256 mismatch for ${item.path}.`);
  observedFiles.push({ path: item.path, size: item.size, sha256 });
}
const sortedPaths = observedFiles.map((item) => item.path).sort();
if (canonicalJson(sortedPaths) !== canonicalJson(observedFiles.map((item) => item.path))) throw new Error("Build-content manifest files are not sorted by normalized path.");
const digestSet = { schema: manifest.content_digest.digest_set_schema, files: observedFiles };
const contentSha256 = createHash("sha256").update(canonicalJson(digestSet)).digest("hex");
if (contentSha256 !== ANALYZER_BUNDLE_CONTENT_SHA256 || contentSha256 !== manifest.content_digest.sha256) throw new Error("Analyzer aggregate build-content SHA-256 mismatch.");
const manifestSha256 = createHash("sha256").update(canonicalJson(manifest)).digest("hex");
if (manifestSha256 !== ANALYZER_BUNDLE_CONTENT_MANIFEST_SHA256) throw new Error("Analyzer build-content manifest SHA-256 mismatch.");
for (const tool of ["node", "npm", "rustc", "wasm_pack"]) {
  if (!manifest.generator?.toolchain?.[tool]) throw new Error(`Build-content manifest is missing ${tool} toolchain identity.`);
}
const generatorBytes = await readFile(manifest.generator.path);
if (createHash("sha256").update(generatorBytes).digest("hex") !== manifest.generator.sha256) throw new Error("Build-content generator SHA-256 mismatch.");
if (!manifest.selection?.file_excludes?.includes("web/lib/build-metadata.js")) throw new Error("Build-content manifest must disclose its self-reference exclusion.");
if (!manifest.selection?.file_excludes?.includes("web/samples/mobilenet_v1_025_224_float.tflite")) throw new Error("Build-content manifest must disclose the deployment-excluded FLOAT32 fixture.");
if (!manifest.selection?.file_excludes?.includes("web/samples/sample_cnn_float.onnx")) throw new Error("Build-content manifest must disclose the deployment-excluded synthetic ONNX fixture.");
const excludedStandardsPathPatterns = [
  /^web\/lib\/cyclonedx-(?:20|draft|perspective)(?:-|\.|$)/i,
  /^web\/vendor\/jsonpath-rfc95\d+(?:\.|$)/i,
];
for (const excludedPattern of excludedStandardsPathPatterns) {
  if (!manifest.selection?.file_exclude_patterns?.includes(excludedPattern.source)) {
    throw new Error(`Build-content manifest must disclose deployment exclusion pattern ${excludedPattern.source}.`);
  }
  if (manifest.files.some((item) => excludedPattern.test(item.path))) {
    throw new Error(`Build-content manifest contains a deployment-excluded standards file matched by ${excludedPattern.source}.`);
  }
}
if ((process.env.CI || process.env.DEEPBOM_RELEASE_BUILD) && ANALYZER_BUILD_SOURCE_STATE !== "clean") throw new Error("CI/release metadata must identify a clean source tree.");
if (process.env.DEEPBOM_RELEASE_BUILD && [manifest.release_inputs?.app_expires_at_epoch_ms, manifest.release_inputs?.app_not_before_epoch_ms].some((value) => !/^\d{13}$/.test(String(value || "")))) {
  throw new Error("Release build metadata must bind explicit runtime-guard epoch inputs.");
}

const packageMetadata = JSON.parse(await readFile("package.json", "utf8"));
const wasmPackageMetadata = JSON.parse(await readFile("pkg/package.json", "utf8"));
const cargoManifest = await readFile("Cargo.toml", "utf8");
const cargoVersion = cargoManifest.match(/^version\s*=\s*"([^"]+)"/m)?.[1] || "";
const html = await readFile("web/index.html", "utf8");
const htmlRelease = html.match(/<meta\s+name="deepbom-release"\s+content="([^"]+)"\s*\/>/)?.[1] || "";
if ([packageMetadata.version, wasmPackageMetadata.version, cargoVersion, htmlRelease]
  .some((version) => version !== ANALYZER_SEMANTIC_VERSION)) {
  throw new Error(`Release identity drift: package=${packageMetadata.version}, wasm=${wasmPackageMetadata.version}, cargo=${cargoVersion}, html=${htmlRelease}, app=${ANALYZER_SEMANTIC_VERSION}.`);
}
if (ANALYZER_METADATA.version !== ANALYZER_VERSION
  || ANALYZER_METADATA.semanticVersion !== ANALYZER_SEMANTIC_VERSION
  || ANALYZER_METADATA.rulepackVersion !== RULEPACK_VERSION) {
  throw new Error("Report metadata identity drifted from the canonical app configuration.");
}
if (ANALYZER_METADATA.rulepackProvenance.xnnpackReadmeSha256 !== "85524b3e6acfe5429b9d8b7c1ef47dde79a4543ccbe2f73a0e276f8e7e0eb93c"
  || ANALYZER_METADATA.rulepackProvenance.xnnpackTagCommit !== "a481b10260dfdf833a1b16007eead49c1d7febf3"
  || ANALYZER_METADATA.rulepackProvenance.xnnpackDelegateImplementedConstraintCount !== 133
  || ANALYZER_METADATA.rulepackProvenance.xnnpackDelegateUnmappedConstraintCount !== 0) {
  throw new Error("XNNPACK delegate README identity or semantic-constraint coverage drifted.");
}

console.log(`Build metadata check passed (${observedFiles.length} files, ${contentSha256.slice(0, 12)}..., source ${ANALYZER_BUILD_SOURCE_STATE}, release ${ANALYZER_SEMANTIC_VERSION}, rulepack ${RULEPACK_VERSION}).`);
