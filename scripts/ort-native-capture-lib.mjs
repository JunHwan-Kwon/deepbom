import { createHash } from "node:crypto";
import { access, mkdir, readdir, readFile, realpath, rm, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { validateOrtBuildAttestation } from "../web/lib/ort-build-attestation.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { parseOrtReducedOperatorConfig } from "../web/lib/ort-reduced-operator-config.js";

export const ORT_NATIVE_CAPTURE_SCHEMA = "deepbom.ort_native_capture.v1.5";
export const ORT_NATIVE_PROFILE_SCHEMA = "deepbom.ort_native_profile.v1.4";
const ORT_EXTERNAL_ARTIFACT_SET_SCHEMA = "deepbom.onnx_external_artifact_set.v1.2";
export const ORT_RUNTIME_VERSION = "1.26.0";
export const ORT_SOURCE_COMMIT = "8c546c37b43caaca1fa25db430dab94b901cf277";
export const ORT_PACKAGE_NAME = "onnxruntime-node";
export const ORT_PACKAGE_INTEGRITY = "sha512-OHl6PiOEOqxaLHL0N9eFrbzS7IGmu3BtJNH3RTEnRAheCIkfc3gjcjl4sGcjp9C22ZC9YTquDOxSdT/stBQ6BQ==";

const SHA256 = /^[a-f0-9]{64}$/;
const MAX_EXTERNAL_FILE_COUNT = 1_024;
const MAX_EXTERNAL_FILE_BYTES = 536_870_912;
const MAX_EXTERNAL_AGGREGATE_BYTES = 1_073_741_824;
const PROFILE_ROLES = Object.freeze([
  Object.freeze({ role: "identity", graphOptimizationLevel: "disabled" }),
  Object.freeze({ role: "production", graphOptimizationLevel: "all" }),
]);
const require = createRequire(import.meta.url);
const projectRoot = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));

export async function capturePinnedOrtProfiles({
  artifactPath,
  outputDir,
  providers = ["cpu"],
  runs = 3,
  warmupRuns = 1,
  inputShapes = new Map(),
  intraOpNumThreads = 1,
  interOpNumThreads = 1,
  reducedOperatorConfigPath = null,
  runtimeModulePath = null,
  buildAttestationPath = null,
} = {}) {
  const artifact = path.resolve(requiredText(artifactPath, "artifact path"));
  const output = path.resolve(requiredText(outputDir, "output directory"));
  if (path.extname(artifact).toLowerCase() !== ".onnx") throw new Error("Pinned ORT capture requires an .onnx artifact.");
  await access(artifact);
  await requireEmptyOrMissingDirectory(output);
  validatePositiveInteger(runs, "runs", 1000);
  validateNonNegativeInteger(warmupRuns, "warmup runs", 1000);
  validatePositiveInteger(intraOpNumThreads, "intra-op thread count", 256);
  validatePositiveInteger(interOpNumThreads, "inter-op thread count", 256);

  if (Boolean(runtimeModulePath) !== Boolean(buildAttestationPath)) throw new Error("Source-built ORT capture requires both runtimeModulePath and buildAttestationPath.");
  const buildAttestation = buildAttestationPath
    ? validateOrtBuildAttestation(JSON.parse(await readFile(path.resolve(buildAttestationPath), "utf8"))) : null;
  const selectedModulePath = runtimeModulePath ? path.resolve(runtimeModulePath) : null;
  const ort = selectedModulePath ? await import(pathToFileURL(selectedModulePath).href) : await import("onnxruntime-node");
  const versions = ort.env?.versions || {};
  if (versions.node !== ORT_RUNTIME_VERSION || versions.common !== ORT_RUNTIME_VERSION) {
    throw new Error(`onnxruntime-node version must be exactly ${ORT_RUNTIME_VERSION}.`);
  }
  const supportedBackends = (await ort.listSupportedBackends())
    .map((item) => ({ name: String(item.name), bundled: item.bundled === true }))
    .sort((left, right) => left.name.localeCompare(right.name));
  const requestedProviders = validateProviders(providers, supportedBackends);
  const reducedOperatorConfig = await selectedReducedOperatorConfig(reducedOperatorConfigPath, buildAttestation);
  const runtime = await runtimeIdentity(supportedBackends, requestedProviders, reducedOperatorConfig, {
    selectedModulePath,
    buildAttestation,
  });
  const artifactBytes = await readFile(artifact);
  const artifactSet = await collectOnnxArtifactSet(artifact, artifactBytes);
  const artifactIdentity = artifactSet.identity;
  const captureStartedAt = new Date().toISOString();
  const captureId = sha256Text(canonicalJson({
    artifact_content_set_sha256: artifactIdentity.content_set_sha256,
    runtime_binary_inventory_sha256: runtime.binary_inventory_sha256,
    reduced_operator_config_sha256: runtime.reduced_operator_config?.source_sha256 || null,
    requested_execution_providers: requestedProviders,
    runs,
    warmup_runs: warmupRuns,
    input_shapes: [...inputShapes].sort(([left], [right]) => left.localeCompare(right)),
    intra_op_num_threads: intraOpNumThreads,
    inter_op_num_threads: interOpNumThreads,
    capture_started_at: captureStartedAt,
  })).slice(0, 24);

  await mkdir(output, { recursive: true });
  const snapshotRoot = path.join(output, ".artifact-snapshot");
  const profiles = [];
  try {
    const stagedArtifact = await stageOnnxArtifactSet(snapshotRoot, artifactSet, artifactBytes);
    for (const spec of PROFILE_ROLES) {
      const result = await captureProfile({
        ort,
        artifact: stagedArtifact,
        artifactIdentity,
        output,
        runtime,
        captureId,
        spec,
        providers: requestedProviders,
        runs,
        warmupRuns,
        inputShapes,
        intraOpNumThreads,
        interOpNumThreads,
      });
      profiles.push(result);
    }
    await assertOnnxArtifactIdentity(stagedArtifact, artifactIdentity, "staged ONNX content set changed during ORT execution");
  } finally {
    await rm(snapshotRoot, { recursive: true, force: true });
  }
  const pairedOutputComparison = compareProfileOutputs(profiles[0].comparisonSnapshot, profiles[1].comparisonSnapshot);
  const pairedRuntimeGraph = buildPairedRuntimeGraph(profiles);
  for (const item of profiles) {
    item.envelope.paired_profile_output_comparison = pairedOutputComparison;
    item.envelope.paired_profile_runtime_graph = pairedRuntimeGraph;
    item.envelope.capture_content_sha256 = contentSha256(item.envelope);
    const bytes = Buffer.from(stableJson(item.envelope));
    await writeFile(path.join(output, item.filename), bytes);
    item.fileSha256 = sha256Bytes(bytes);
  }

  const index = {
    schema: ORT_NATIVE_CAPTURE_SCHEMA,
    capture_id: captureId,
    artifact: artifactIdentity,
    runtime,
    collection: { started_at: captureStartedAt },
    invocation: {
      requested_execution_providers: requestedProviders,
      runs,
      warmup_runs: warmupRuns,
      intra_op_num_threads: intraOpNumThreads,
      inter_op_num_threads: interOpNumThreads,
    },
    profiles: profiles.map((item) => ({
      role: item.envelope.profile_role,
      filename: item.filename,
      file_sha256: item.fileSha256,
      profile_sha256: item.envelope.profile.sha256,
      source_event_count: item.envelope.profile.source_event_count,
      kernel_event_count: item.envelope.profile.kernel_event_count,
      observed_providers: item.envelope.profile.observed_providers,
      mapping_intent: item.envelope.mapping_intent,
    })),
    paired_profile_output_comparison: pairedOutputComparison,
    paired_profile_runtime_graph: pairedRuntimeGraph,
    evidence_boundary: "Provider, node, and duration fields are observed in the pinned native ORT profile. The collector verifies and hashes the ONNX protobuf plus every referenced external-data file/range before execution, and captures package and host-binary identities. The browser can verify the envelope, active artifact content set, and embedded profile, but does not independently re-read native runtime binaries.",
  };
  index.capture_content_sha256 = contentSha256(index);
  await writeFile(path.join(output, "capture-index.json"), stableJson(index));
  await verifyOrtNativeCapturePackage(output, { artifactPath: artifact });
  return { outputDir: output, index };
}

export async function verifyOrtNativeCapturePackage(captureDir, { artifactPath = null } = {}) {
  const root = path.resolve(requiredText(captureDir, "capture directory"));
  const index = JSON.parse(await readFile(path.join(root, "capture-index.json"), "utf8"));
  if (index.schema !== ORT_NATIVE_CAPTURE_SCHEMA || index.capture_content_sha256 !== contentSha256(index)) {
    throw new Error("ORT native capture index schema or content digest is invalid.");
  }
  validateRuntimeIdentity(index.runtime);
  validateArtifactIdentity(index.artifact);
  if (!Array.isArray(index.profiles) || index.profiles.length !== PROFILE_ROLES.length) throw new Error("ORT native capture must contain identity and production profiles.");
  if (artifactPath) {
    const resolvedArtifact = path.resolve(artifactPath);
    const bytes = await readFile(resolvedArtifact);
    const observedIdentity = await collectOnnxArtifactIdentity(resolvedArtifact, bytes);
    if (canonicalJson(observedIdentity) !== canonicalJson(index.artifact)) throw new Error("ORT native capture artifact content-set identity does not match the supplied model and external data.");
  }
  const roles = new Set();
  for (const row of index.profiles) {
    if (!PROFILE_ROLES.some((item) => item.role === row.role) || roles.has(row.role)) throw new Error("ORT native capture profile roles are invalid or duplicated.");
    roles.add(row.role);
    const filename = safeBasename(row.filename);
    const bytes = await readFile(path.join(root, filename));
    if (sha256Bytes(bytes) !== row.file_sha256) throw new Error(`ORT native capture hash mismatch for ${filename}.`);
    const envelope = JSON.parse(bytes.toString("utf8"));
    validateOrtNativeProfileEnvelope(envelope);
    if (envelope.capture_id !== index.capture_id
      || canonicalJson(envelope.artifact) !== canonicalJson(index.artifact)
      || envelope.profile_role !== row.role
      || envelope.profile.sha256 !== row.profile_sha256
      || envelope.profile.source_event_count !== row.source_event_count
      || envelope.profile.kernel_event_count !== row.kernel_event_count
      || canonicalJson(envelope.profile.observed_providers) !== canonicalJson(row.observed_providers)
      || envelope.mapping_intent !== row.mapping_intent
      || canonicalJson(envelope.paired_profile_output_comparison) !== canonicalJson(index.paired_profile_output_comparison)
      || canonicalJson(envelope.paired_profile_runtime_graph) !== canonicalJson(index.paired_profile_runtime_graph)) {
      throw new Error(`ORT native capture index identity mismatch for ${filename}.`);
    }
  }
  return { root, index };
}

export function validateOrtNativeProfileEnvelope(envelope) {
  if (!envelope || envelope.schema !== ORT_NATIVE_PROFILE_SCHEMA || envelope.capture_content_sha256 !== contentSha256(envelope)) {
    throw new Error("ORT native profile schema or content digest is invalid.");
  }
  validateArtifactIdentity(envelope.artifact);
  validateRuntimeIdentity(envelope.runtime);
  const spec = PROFILE_ROLES.find((item) => item.role === envelope.profile_role);
  if (!spec || envelope.invocation?.graph_optimization_level !== spec.graphOptimizationLevel
    || envelope.invocation?.execution_mode !== "sequential") {
    throw new Error("ORT native profile role and session options are inconsistent.");
  }
  validatePositiveInteger(envelope.invocation?.runs, "profile runs", 1000);
  validateNonNegativeInteger(envelope.invocation?.warmup_runs, "profile warmup runs", 1000);
  if (!Array.isArray(envelope.invocation?.inputs) || !envelope.invocation.inputs.length) throw new Error("ORT native profile input inventory is missing.");
  for (const input of envelope.invocation.inputs) {
    if (!input?.name || !input.type || !Array.isArray(input.shape) || input.shape.some((dim) => !Number.isSafeInteger(dim) || dim < 1) || !SHA256.test(input.data_sha256)) {
      throw new Error("ORT native profile input identity is invalid.");
    }
  }
  const profile = envelope.profile;
  if (!profile || typeof profile.json !== "string" || !SHA256.test(profile.sha256) || sha256Text(profile.json) !== profile.sha256) {
    throw new Error("ORT native profile JSON digest is invalid.");
  }
  let events;
  try { events = JSON.parse(profile.json); } catch { throw new Error("ORT native embedded profile is not valid JSON."); }
  if (!Array.isArray(events) || events.length !== profile.source_event_count) throw new Error("ORT native profile event count is invalid.");
  const kernelEvents = events.filter(isOrtKernelEvent);
  const providers = [...new Set(kernelEvents.map((event) => String(event.args.provider)))].sort();
  if (!kernelEvents.length || kernelEvents.length !== profile.kernel_event_count || canonicalJson(providers) !== canonicalJson(profile.observed_providers)) {
    throw new Error("ORT native profile kernel-event inventory is invalid.");
  }
  if (!Array.isArray(envelope.output_observations) || !envelope.output_observations.length) throw new Error("ORT native profile output observations are missing.");
  for (const output of envelope.output_observations) {
    if (!output?.name || !output.type || !Array.isArray(output.dims) || !SHA256.test(output.first_run_sha256) || !SHA256.test(output.last_run_sha256)) {
      throw new Error("ORT native profile output observation is invalid.");
    }
  }
  validatePairedOutputComparison(envelope.paired_profile_output_comparison);
  validatePairedRuntimeGraph(envelope.paired_profile_runtime_graph);
  return envelope;
}

async function captureProfile({
  ort, artifact, artifactIdentity, output, runtime, captureId, spec, providers,
  runs, warmupRuns, inputShapes, intraOpNumThreads, interOpNumThreads,
}) {
  const baseOptions = {
    executionProviders: providers,
    graphOptimizationLevel: spec.graphOptimizationLevel,
    executionMode: "sequential",
    intraOpNumThreads,
    interOpNumThreads,
  };
  let inputInventory;
  if (warmupRuns > 0) {
    const warmup = await ort.InferenceSession.create(artifact, baseOptions);
    const prepared = prepareInputs(ort, warmup, inputShapes);
    inputInventory = prepared.inventory;
    try {
      for (let index = 0; index < warmupRuns; index += 1) await warmup.run(prepared.feeds);
    } finally {
      await warmup.release();
    }
  }

  const prefix = path.join(output, `.ort-${spec.role}`);
  const collectedAt = new Date().toISOString();
  const session = await ort.InferenceSession.create(artifact, {
    ...baseOptions,
    enableProfiling: true,
    profileFilePrefix: prefix,
  });
  const prepared = prepareInputs(ort, session, inputShapes);
  if (inputInventory && canonicalJson(inputInventory) !== canonicalJson(prepared.inventory)) throw new Error("ORT warmup/profile input metadata changed between sessions.");
  inputInventory = prepared.inventory;
  const outputDigests = [];
  let comparisonSnapshot = null;
  try {
    for (let index = 0; index < runs; index += 1) {
      const observation = await observeOutputs(await session.run(prepared.feeds));
      outputDigests.push(observation.rows);
      if (!comparisonSnapshot) comparisonSnapshot = observation.comparisonSnapshot;
    }
    session.endProfiling();
  } finally {
    await session.release();
  }
  const profilePath = await locateProfile(output, path.basename(prefix));
  const profileJson = await readFile(profilePath, "utf8");
  const events = JSON.parse(profileJson);
  if (!Array.isArray(events)) throw new Error("ORT profiler did not emit a Chrome-trace event array.");
  const kernelEvents = events.filter(isOrtKernelEvent);
  if (!kernelEvents.length) throw new Error(`ORT ${spec.role} profile emitted no provider-bound node events.`);
  const outputObservations = summarizeOutputs(outputDigests);
  const envelope = {
    schema: ORT_NATIVE_PROFILE_SCHEMA,
    capture_id: captureId,
    profile_role: spec.role,
    mapping_intent: spec.role === "identity"
      ? "original_node_identity_with_graph_optimization_disabled"
      : "production_provider_assignment_with_graph_optimization_all",
    artifact: artifactIdentity,
    runtime,
    invocation: {
      requested_execution_providers: providers,
      graph_optimization_level: spec.graphOptimizationLevel,
      execution_mode: "sequential",
      runs,
      warmup_runs: warmupRuns,
      intra_op_num_threads: intraOpNumThreads,
      inter_op_num_threads: interOpNumThreads,
      inputs: inputInventory,
    },
    collection: {
      collected_at: collectedAt,
      collector: "scripts/capture-pinned-ort-profile.mjs",
      collector_schema: ORT_NATIVE_CAPTURE_SCHEMA,
    },
    profile: {
      sha256: sha256Text(profileJson),
      source_event_count: events.length,
      kernel_event_count: kernelEvents.length,
      observed_providers: [...new Set(kernelEvents.map((event) => String(event.args.provider)))].sort(),
      json: profileJson,
    },
    output_observations: outputObservations,
    evidence_boundary: "Node identity, provider, and duration are ORT profiler observations. The ONNX protobuf and referenced external-data content set are collector-verified and browser-matchable; runtime package and host binary hashes remain collector observations. Input values are deterministic zeros; this is placement/timing evidence for the declared invocation, not representative-workload validation.",
  };
  envelope.capture_content_sha256 = contentSha256(envelope);
  const filename = `${spec.role}.deepbom-ort-profile.json`;
  const bytes = Buffer.from(stableJson(envelope));
  await writeFile(path.join(output, filename), bytes);
  await rm(profilePath, { force: true });
  return { filename, fileSha256: sha256Bytes(bytes), envelope, comparisonSnapshot };
}

function prepareInputs(ort, session, inputShapes) {
  const feeds = {};
  const inventory = [];
  for (const metadata of session.inputMetadata || []) {
    if (!metadata.isTensor) throw new Error(`ORT input ${metadata.name} is not a tensor and requires an explicit capture adapter.`);
    const shape = resolveShape(metadata, inputShapes.get(metadata.name));
    const elementCount = shape.reduce((product, dim) => product * dim, 1);
    if (!Number.isSafeInteger(elementCount) || elementCount < 1 || elementCount > 500_000_000) throw new Error(`ORT input ${metadata.name} element count is unsafe.`);
    const data = zeroData(metadata.type, elementCount);
    feeds[metadata.name] = new ort.Tensor(metadata.type, data, shape);
    inventory.push({
      name: metadata.name,
      type: metadata.type,
      shape,
      generator: "deterministic_zero",
      data_sha256: tensorDataSha(metadata.type, shape, data),
    });
  }
  if (!inventory.length) throw new Error("ORT model exposes no tensor inputs.");
  return { feeds, inventory };
}

function resolveShape(metadata, override) {
  const declared = metadata.shape || [];
  const selected = override || declared;
  if (!Array.isArray(selected) || selected.length !== declared.length || selected.some((dim) => !Number.isSafeInteger(Number(dim)) || Number(dim) < 1)) {
    throw new Error(`ORT input ${metadata.name} has dynamic dimensions; provide --shape=${metadata.name}=d0,d1,... with positive integers.`);
  }
  const shape = selected.map(Number);
  for (let index = 0; index < declared.length; index += 1) {
    if (typeof declared[index] === "number" && declared[index] > 0 && declared[index] !== shape[index]) {
      throw new Error(`ORT input ${metadata.name} shape override changes static dimension ${index}.`);
    }
  }
  return shape;
}

function zeroData(type, count) {
  switch (type) {
    case "float32": return new Float32Array(count);
    case "float64": return new Float64Array(count);
    case "int8": return new Int8Array(count);
    case "uint8":
    case "bool": return new Uint8Array(count);
    case "int16": return new Int16Array(count);
    case "uint16":
    case "float16": return new Uint16Array(count);
    case "int32": return new Int32Array(count);
    case "uint32": return new Uint32Array(count);
    case "int64": return new BigInt64Array(count);
    case "uint64": return new BigUint64Array(count);
    case "string": return Array(count).fill("");
    default: throw new Error(`ORT deterministic input generator does not support ${type}.`);
  }
}

async function observeOutputs(outputs) {
  const rows = [];
  const comparisonSnapshot = [];
  for (const name of Object.keys(outputs).sort()) {
    const tensor = outputs[name];
    rows.push({ name, type: tensor.type, dims: [...tensor.dims], sha256: tensorDataSha(tensor.type, tensor.dims, tensor.data) });
    comparisonSnapshot.push({
      name,
      type: tensor.type,
      dims: [...tensor.dims],
      values: ArrayBuffer.isView(tensor.data) || Array.isArray(tensor.data) ? Array.from(tensor.data) : null,
    });
  }
  return { rows, comparisonSnapshot };
}

function summarizeOutputs(runs) {
  if (!runs.length) throw new Error("ORT capture emitted no output observations.");
  const first = runs[0];
  const last = runs.at(-1);
  if (canonicalJson(first.map(({ sha256: _sha, ...row }) => row)) !== canonicalJson(last.map(({ sha256: _sha, ...row }) => row))) {
    throw new Error("ORT output identity changed across profiled runs.");
  }
  return first.map((row, index) => ({
    name: row.name,
    type: row.type,
    dims: row.dims,
    first_run_sha256: row.sha256,
    last_run_sha256: last[index].sha256,
    stable_across_profiled_runs: runs.every((run) => run[index]?.sha256 === row.sha256),
  }));
}

function compareProfileOutputs(identity, production) {
  if (!Array.isArray(identity) || !Array.isArray(production) || identity.length !== production.length) {
    throw new Error("ORT identity/production output inventories do not match.");
  }
  const outputs = identity.map((reference, index) => {
    const candidate = production[index];
    if (reference.name !== candidate?.name || reference.type !== candidate.type || canonicalJson(reference.dims) !== canonicalJson(candidate.dims)) {
      throw new Error("ORT identity/production output identities do not match.");
    }
    const referenceHash = tensorDataSha(reference.type, reference.dims, outputValuesForHash(reference));
    const candidateHash = tensorDataSha(candidate.type, candidate.dims, outputValuesForHash(candidate));
    const bitwiseEqual = referenceHash === candidateHash;
    const numeric = Array.isArray(reference.values) && Array.isArray(candidate.values)
      && reference.values.length === candidate.values.length
      && reference.values.every((value) => typeof value === "number" && Number.isFinite(value))
      && candidate.values.every((value) => typeof value === "number" && Number.isFinite(value));
    if (!numeric) return {
      name: reference.name,
      type: reference.type,
      dims: reference.dims,
      element_count: reference.values?.length ?? null,
      identity_sha256: referenceHash,
      production_sha256: candidateHash,
      bitwise_equal: bitwiseEqual,
      numeric_comparison_status: "not_assessed_non_numeric_or_nonfinite",
    };
    let sumAbs = 0;
    let sumSquared = 0;
    let referenceSquared = 0;
    let candidateSquared = 0;
    let dot = 0;
    let maxAbs = 0;
    for (let valueIndex = 0; valueIndex < reference.values.length; valueIndex += 1) {
      const left = reference.values[valueIndex];
      const right = candidate.values[valueIndex];
      const difference = right - left;
      const absolute = Math.abs(difference);
      sumAbs += absolute;
      sumSquared += difference * difference;
      referenceSquared += left * left;
      candidateSquared += right * right;
      dot += left * right;
      maxAbs = Math.max(maxAbs, absolute);
    }
    const count = reference.values.length;
    const denominator = Math.sqrt(referenceSquared) * Math.sqrt(candidateSquared);
    return {
      name: reference.name,
      type: reference.type,
      dims: reference.dims,
      element_count: count,
      identity_sha256: referenceHash,
      production_sha256: candidateHash,
      bitwise_equal: bitwiseEqual,
      numeric_comparison_status: "assessed",
      max_abs_error: maxAbs,
      mean_abs_error: sumAbs / Math.max(1, count),
      rms_error: Math.sqrt(sumSquared / Math.max(1, count)),
      relative_l2_error: Math.sqrt(sumSquared) / Math.max(Number.MIN_VALUE, Math.sqrt(referenceSquared)),
      cosine_distance: denominator === 0 ? (referenceSquared === 0 && candidateSquared === 0 ? 0 : null) : 1 - dot / denominator,
    };
  });
  return {
    schema: "deepbom.ort_profile_output_comparison.v1",
    reference_profile_role: "identity",
    candidate_profile_role: "production",
    status: outputs.every((item) => item.numeric_comparison_status === "assessed") ? "assessed" : "partially_assessed",
    all_outputs_bitwise_equal: outputs.every((item) => item.bitwise_equal),
    outputs,
    interpretation_boundary: "Both profiles used the same deterministic input bytes and pinned runtime binaries. Numeric deltas isolate graph-optimization/provider-path effects for this synthetic input only; they are not representative-accuracy evidence.",
  };
}

function outputValuesForHash(snapshot) {
  if (snapshot.type === "string") return snapshot.values;
  const constructors = {
    float32: Float32Array, float64: Float64Array, int8: Int8Array, uint8: Uint8Array, bool: Uint8Array,
    int16: Int16Array, uint16: Uint16Array, float16: Uint16Array, int32: Int32Array, uint32: Uint32Array,
    int64: BigInt64Array, uint64: BigUint64Array,
  };
  const Constructor = constructors[snapshot.type];
  if (!Constructor || !Array.isArray(snapshot.values)) return snapshot.values;
  return new Constructor(snapshot.values);
}

function validatePairedOutputComparison(comparison) {
  if (!comparison || comparison.schema !== "deepbom.ort_profile_output_comparison.v1"
    || comparison.reference_profile_role !== "identity" || comparison.candidate_profile_role !== "production"
    || !["assessed", "partially_assessed"].includes(comparison.status)
    || typeof comparison.all_outputs_bitwise_equal !== "boolean" || !Array.isArray(comparison.outputs) || !comparison.outputs.length) {
    throw new Error("ORT paired-profile output comparison is invalid.");
  }
  for (const output of comparison.outputs) {
    if (!output?.name || !output.type || !Array.isArray(output.dims) || typeof output.bitwise_equal !== "boolean"
      || !SHA256.test(output.identity_sha256) || !SHA256.test(output.production_sha256)
      || !["assessed", "not_assessed_non_numeric_or_nonfinite"].includes(output.numeric_comparison_status)) {
      throw new Error("ORT paired-profile output row is invalid.");
    }
    if (output.numeric_comparison_status === "assessed") {
      for (const field of ["max_abs_error", "mean_abs_error", "rms_error", "relative_l2_error"]) {
        if (!Number.isFinite(output[field]) || output[field] < 0) throw new Error(`ORT paired-profile ${field} is invalid.`);
      }
      if (output.cosine_distance != null && !Number.isFinite(output.cosine_distance)) throw new Error("ORT paired-profile cosine distance is invalid.");
    }
  }
}

function buildPairedRuntimeGraph(profiles) {
  return {
    schema: "deepbom.ort_paired_runtime_graph.v1.1",
    profiles: profiles.map((item) => {
      const events = JSON.parse(item.envelope.profile.json);
      const groups = new Map();
      events.forEach((event, sourceEventIndex) => {
        if (!isOrtKernelEvent(event)) return;
        const row = {
          runtime_node_name: String(event.name).slice(0, -"_kernel_time".length),
          runtime_node_index: Number(event.args.node_index),
          op_name: String(event.args.op_name),
          provider: String(event.args.provider),
          output_size_bytes_decimal: /^\d+$/.test(String(event.args.output_size ?? "")) ? String(event.args.output_size) : null,
        };
        const key = `${row.runtime_node_name}\0${row.runtime_node_index}\0${row.op_name}\0${row.provider}`;
        const current = groups.get(key) || { ...row, first_source_event_index: sourceEventIndex, sample_count: 0, duration_sum_us: 0 };
        if (current.output_size_bytes_decimal !== row.output_size_bytes_decimal) {
          throw new Error(`ORT runtime node ${row.runtime_node_name} emitted inconsistent output_size values across repeated events.`);
        }
        current.sample_count += 1;
        current.duration_sum_us += Number(event.dur);
        groups.set(key, current);
      });
      const nodes = [...groups.values()]
        .sort((left, right) => left.first_source_event_index - right.first_source_event_index)
        .map((row) => ({ ...row, duration_mean_us: row.duration_sum_us / row.sample_count }));
      return {
        role: item.envelope.profile_role,
        profile_sha256: item.envelope.profile.sha256,
        graph_optimization_level: item.envelope.invocation.graph_optimization_level,
        execution_mode: item.envelope.invocation.execution_mode,
        invocation_run_count: item.envelope.invocation.runs,
        kernel_event_count: item.envelope.profile.kernel_event_count,
        runtime_node_count: nodes.length,
        observed_providers: item.envelope.profile.observed_providers,
        nodes,
      };
    }),
    production_original_graph_mapping_status: "NOT_INFERRED_TRANSFORMED_RUNTIME_NODE_IDENTITY",
    interpretation_boundary: "The identity profile is intended for fail-closed original-node mapping. The production profile ledger preserves optimized/fused runtime-node and provider observations without assigning them back to original graph ops unless an independent exact mapping exists.",
  };
}

function validatePairedRuntimeGraph(graph) {
  if (!graph || !["deepbom.ort_paired_runtime_graph.v1", "deepbom.ort_paired_runtime_graph.v1.1"].includes(graph.schema)
    || graph.production_original_graph_mapping_status !== "NOT_INFERRED_TRANSFORMED_RUNTIME_NODE_IDENTITY"
    || !Array.isArray(graph.profiles) || graph.profiles.length !== 2) {
    throw new Error("ORT paired runtime graph is invalid.");
  }
  const roles = new Set();
  for (const profile of graph.profiles) {
    if (!PROFILE_ROLES.some((item) => item.role === profile.role) || roles.has(profile.role) || !SHA256.test(profile.profile_sha256)
      || !Number.isSafeInteger(profile.kernel_event_count) || profile.kernel_event_count < 1
      || !Number.isSafeInteger(profile.runtime_node_count) || profile.runtime_node_count < 1
      || !Array.isArray(profile.nodes) || profile.nodes.length !== profile.runtime_node_count
      || profile.nodes.reduce((sum, node) => sum + Number(node.sample_count || 0), 0) !== profile.kernel_event_count) {
      throw new Error("ORT paired runtime graph profile is invalid.");
    }
    roles.add(profile.role);
    if (graph.schema === "deepbom.ort_paired_runtime_graph.v1.1"
      && (!Number.isSafeInteger(profile.invocation_run_count) || profile.invocation_run_count < 1)) {
      throw new Error("ORT paired runtime graph invocation count is invalid.");
    }
    for (const node of profile.nodes) {
      if (!node?.runtime_node_name || !Number.isSafeInteger(node.runtime_node_index) || node.runtime_node_index < 0 || !node.op_name || !node.provider
        || !Number.isSafeInteger(node.first_source_event_index) || node.first_source_event_index < 0
        || !Number.isSafeInteger(node.sample_count) || node.sample_count < 1
        || !Number.isFinite(node.duration_sum_us) || node.duration_sum_us < 0
        || !Number.isFinite(node.duration_mean_us) || node.duration_mean_us < 0
        || (graph.schema === "deepbom.ort_paired_runtime_graph.v1.1"
          && node.output_size_bytes_decimal != null && !/^\d+$/.test(node.output_size_bytes_decimal))
        || Math.abs(node.duration_mean_us - node.duration_sum_us / node.sample_count) > 1e-9 * Math.max(1, node.duration_mean_us)) {
        throw new Error("ORT paired runtime graph node is invalid.");
      }
    }
  }
}

async function runtimeIdentity(supportedBackends, requestedProviders, reducedOperatorConfig, { selectedModulePath, buildAttestation }) {
  const packageManifestPath = selectedModulePath
    ? await findParentFile(path.dirname(selectedModulePath), "package.json", 4)
    : require.resolve("onnxruntime-node/package.json");
  const packageRoot = path.dirname(packageManifestPath);
  const manifest = JSON.parse(await readFile(packageManifestPath, "utf8"));
  if (manifest.name !== ORT_PACKAGE_NAME || manifest.version !== ORT_RUNTIME_VERSION) throw new Error("Installed ORT package manifest does not match the pin.");
  if (!buildAttestation) {
    const lock = JSON.parse(await readFile(path.join(projectRoot, "package-lock.json"), "utf8"));
    const locked = lock.packages?.[`node_modules/${ORT_PACKAGE_NAME}`];
    if (locked?.version !== ORT_RUNTIME_VERSION || locked?.integrity !== ORT_PACKAGE_INTEGRITY) throw new Error("package-lock ORT identity does not match the pin.");
  }
  const hostRoot = path.join(packageRoot, "bin", "napi-v6", process.platform, process.arch);
  const files = await recursiveFiles(hostRoot);
  if (!files.length) throw new Error(`No pinned ORT host binaries for ${process.platform}/${process.arch}.`);
  const binaryInventory = [];
  for (const file of files) {
    const bytes = await readFile(file);
    binaryInventory.push({ path: path.relative(packageRoot, file).replaceAll("\\", "/"), byte_length: bytes.byteLength, sha256: sha256Bytes(bytes) });
  }
  binaryInventory.sort((left, right) => left.path.localeCompare(right.path));
  const primary = binaryInventory.find((item) => /(?:^|\/)(?:onnxruntime\.dll|libonnxruntime\.so(?:\.\d+)*|libonnxruntime\.dylib)$/.test(item.path))
    || binaryInventory.find((item) => item.path.endsWith("onnxruntime_binding.node"));
  if (!primary) throw new Error("Pinned ORT primary runtime binary was not found.");
  const packageManifestSha256 = sha256Bytes(await readFile(packageManifestPath));
  const binaryInventorySha256 = sha256Text(canonicalJson(binaryInventory));
  if (buildAttestation && (buildAttestation.runtime_package.package_manifest_sha256 !== packageManifestSha256
    || buildAttestation.runtime_package.binary_inventory_sha256 !== binaryInventorySha256
    || buildAttestation.runtime_package.primary_binary_sha256 !== primary.sha256
    || canonicalJson(buildAttestation.runtime_package.binary_inventory) !== canonicalJson(binaryInventory)
    || buildAttestation.runtime_package.platform !== process.platform
    || buildAttestation.runtime_package.arch !== process.arch)) {
    throw new Error("Selected ORT module/package binaries do not reproduce the source-build attestation.");
  }
  return {
    name: "ONNX Runtime Node.js",
    version: ORT_RUNTIME_VERSION,
    package_name: ORT_PACKAGE_NAME,
    distribution_identity: buildAttestation ? "SOURCE_BUILD_ATTESTED" : "NPM_PACKAGE_LOCK_ATTESTED",
    package_integrity: buildAttestation ? null : ORT_PACKAGE_INTEGRITY,
    package_manifest_sha256: packageManifestSha256,
    build_attestation: buildAttestation,
    source_commit: ORT_SOURCE_COMMIT,
    node_napi: "napi-v6",
    platform: process.platform,
    arch: process.arch,
    node_version: process.version,
    requested_execution_providers: requestedProviders,
    supported_backends: supportedBackends,
    supported_backends_sha256: sha256Text(canonicalJson(supportedBackends)),
    provider_inventory_status: "OBSERVED_FROM_ORT_LIST_SUPPORTED_BACKENDS",
    reduced_operator_inventory_status: reducedOperatorConfig
      ? buildAttestation ? "BUILD_INPUT_BINARY_ATTESTED" : "IMPORTED_CONFIG_NOT_BINARY_ATTESTED"
      : "NOT_EXPOSED_BY_ONNXRUNTIME_NODE_API_NOT_INFERRED",
    reduced_operator_config: reducedOperatorConfig,
    primary_binary_path: primary.path,
    primary_binary_sha256: primary.sha256,
    binary_inventory_sha256: binaryInventorySha256,
    binary_inventory: binaryInventory,
    host: {
      os_type: os.type(),
      os_release: os.release(),
      cpu_model: os.cpus()[0]?.model || "unknown",
      logical_cpu_count: os.cpus().length,
      endianness: os.endianness(),
    },
  };
}

function validateRuntimeIdentity(runtime) {
  if (!runtime || runtime.name !== "ONNX Runtime Node.js" || runtime.version !== ORT_RUNTIME_VERSION
    || runtime.package_name !== ORT_PACKAGE_NAME
    || !["NPM_PACKAGE_LOCK_ATTESTED", "SOURCE_BUILD_ATTESTED"].includes(runtime.distribution_identity)
    || (runtime.distribution_identity === "NPM_PACKAGE_LOCK_ATTESTED" ? runtime.package_integrity !== ORT_PACKAGE_INTEGRITY || runtime.build_attestation != null
      : runtime.package_integrity != null || validateOrtBuildAttestation(runtime.build_attestation).attestation_sha256 !== runtime.build_attestation.attestation_sha256)
    || runtime.source_commit !== ORT_SOURCE_COMMIT || runtime.node_napi !== "napi-v6"
    || !SHA256.test(runtime.package_manifest_sha256) || !SHA256.test(runtime.primary_binary_sha256)
    || !SHA256.test(runtime.binary_inventory_sha256) || !Array.isArray(runtime.binary_inventory) || !runtime.binary_inventory.length
    || runtime.binary_inventory_sha256 !== sha256Text(canonicalJson(runtime.binary_inventory))) {
    throw new Error("ORT native runtime identity is invalid.");
  }
  for (const file of runtime.binary_inventory) {
    if (safeRelativePath(file.path) !== file.path || !Number.isSafeInteger(file.byte_length) || file.byte_length < 1 || !SHA256.test(file.sha256)) throw new Error("ORT native runtime binary inventory is invalid.");
  }
  if (!runtime.binary_inventory.some((file) => file.path === runtime.primary_binary_path && file.sha256 === runtime.primary_binary_sha256)) {
    throw new Error("ORT native primary binary is not bound to the binary inventory.");
  }
  const backends = runtime.supported_backends;
  const requested = runtime.requested_execution_providers;
  if (!Array.isArray(backends) || !backends.length
    || backends.some((backend) => !backend?.name || typeof backend.bundled !== "boolean")
    || new Set(backends.map((backend) => backend.name)).size !== backends.length
    || canonicalJson([...backends].sort((left, right) => left.name.localeCompare(right.name))) !== canonicalJson(backends)
    || runtime.supported_backends_sha256 !== sha256Text(canonicalJson(backends))
    || runtime.provider_inventory_status !== "OBSERVED_FROM_ORT_LIST_SUPPORTED_BACKENDS"
    || !["NOT_EXPOSED_BY_ONNXRUNTIME_NODE_API_NOT_INFERRED", "IMPORTED_CONFIG_NOT_BINARY_ATTESTED", "BUILD_INPUT_BINARY_ATTESTED"].includes(runtime.reduced_operator_inventory_status)
    || !Array.isArray(requested) || !requested.length || new Set(requested).size !== requested.length
    || requested.some((name) => !backends.some((backend) => backend.name === name))) {
    throw new Error("ORT native provider/build inventory is invalid.");
  }
  validateReducedOperatorConfigIdentity(runtime.reduced_operator_config, runtime.reduced_operator_inventory_status);
}

async function collectReducedOperatorConfig(configPath) {
  const resolved = path.resolve(requiredText(configPath, "reduced-operator config path"));
  const bytes = await readFile(resolved);
  if (bytes.byteLength > 4 * 1024 * 1024) throw new Error("ORT reduced-operator config exceeds 4 MiB.");
  if (bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) throw new Error("ORT reduced-operator config must be UTF-8 without a byte-order mark.");
  let sourceText;
  try { sourceText = new TextDecoder("utf-8", { fatal: true }).decode(bytes); } catch { throw new Error("ORT reduced-operator config must be valid UTF-8."); }
  const normalizedConfig = parseOrtReducedOperatorConfig(sourceText);
  return {
    schema: "deepbom.ort_reduced_operator_config_identity.v1",
    source_name: path.basename(resolved),
    source_sha256: sha256Bytes(bytes),
    normalized_sha256: sha256Text(canonicalJson(normalizedConfig)),
    source_text: sourceText,
    normalized_config: normalizedConfig,
    binary_binding_status: "NOT_ATTESTED_CONFIG_INPUT_NOT_OBSERVED_FROM_SELECTED_BINARY",
  };
}

async function selectedReducedOperatorConfig(configPath, buildAttestation) {
  if (!buildAttestation) return configPath ? collectReducedOperatorConfig(configPath) : null;
  const attested = buildAttestation.reduced_operator_config;
  if (!attested) {
    if (configPath) throw new Error("A reduced-operator config was supplied, but the selected ORT build attestation has no reduced build input.");
    return null;
  }
  if (configPath) {
    const supplied = await collectReducedOperatorConfig(configPath);
    if (supplied.source_sha256 !== attested.source_sha256 || supplied.normalized_sha256 !== attested.normalized_sha256) {
      throw new Error("Supplied reduced-operator config differs from the selected ORT build input.");
    }
  }
  return {
    schema: "deepbom.ort_reduced_operator_config_identity.v1",
    source_name: attested.source_name,
    source_sha256: attested.source_sha256,
    normalized_sha256: attested.normalized_sha256,
    source_text: attested.source_text,
    normalized_config: attested.normalized_config,
    binary_binding_status: "ATTESTED_OBSERVED_BUILD_INPUT_BOUND_TO_SELECTED_BINARY_INVENTORY",
  };
}

function validateReducedOperatorConfigIdentity(identity, status) {
  if (status === "NOT_EXPOSED_BY_ONNXRUNTIME_NODE_API_NOT_INFERRED") {
    if (identity != null) throw new Error("ORT runtime exposes a reduced-operator config without a matching status.");
    return;
  }
  if (!identity || identity.schema !== "deepbom.ort_reduced_operator_config_identity.v1"
    || path.basename(identity.source_name) !== identity.source_name
    || !SHA256.test(identity.source_sha256) || !SHA256.test(identity.normalized_sha256)
    || sha256Text(identity.source_text) !== identity.source_sha256
    || (status === "BUILD_INPUT_BINARY_ATTESTED"
      ? identity.binary_binding_status !== "ATTESTED_OBSERVED_BUILD_INPUT_BOUND_TO_SELECTED_BINARY_INVENTORY"
      : identity.binary_binding_status !== "NOT_ATTESTED_CONFIG_INPUT_NOT_OBSERVED_FROM_SELECTED_BINARY")) {
    throw new Error("ORT reduced-operator config identity is invalid.");
  }
  const reparsed = parseOrtReducedOperatorConfig(identity.source_text);
  if (canonicalJson(reparsed) !== canonicalJson(identity.normalized_config)
    || sha256Text(canonicalJson(reparsed)) !== identity.normalized_sha256) {
    throw new Error("ORT reduced-operator normalized config is inconsistent.");
  }
}

async function findParentFile(start, name, maxParents) {
  let current = start;
  for (let depth = 0; depth <= maxParents; depth += 1) {
    const candidate = path.join(current, name);
    try { if ((await stat(candidate)).isFile()) return candidate; } catch {}
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  throw new Error(`Cannot locate ${name} above the selected ORT runtime module.`);
}

async function collectOnnxArtifactIdentity(artifactPath, artifactBytes) {
  return (await collectOnnxArtifactSet(artifactPath, artifactBytes)).identity;
}

async function collectOnnxArtifactSet(artifactPath, artifactBytes) {
  const modelSha256 = sha256Bytes(artifactBytes);
  const structural = analyzeOnnxModel(new Uint8Array(artifactBytes.buffer, artifactBytes.byteOffset, artifactBytes.byteLength), path.basename(artifactPath));
  const references = structural.onnx_external_data || {};
  if (Number(references.malformed_reference_count || 0) > 0) throw new Error("ORT native capture refuses malformed ONNX external_data references.");
  const root = await realpath(path.dirname(artifactPath));
  const records = [];
  const locations = new Set();
  for (const row of references.tensors || []) {
    const declared = String(row.location || "");
    const location = safeRelativePath(row.normalized_location || declared);
    if (!location) throw new Error(`ORT native capture refuses unsafe external_data location ${declared || "(missing)"}.`);
    locations.add(location);
  }
  if (locations.size > MAX_EXTERNAL_FILE_COUNT) throw new Error(`ORT native capture external_data file count exceeds ${MAX_EXTERNAL_FILE_COUNT}.`);
  let aggregateBytes = 0;
  for (const location of [...locations].sort(compareText)) {
    if (location === path.basename(artifactPath)) throw new Error(`ORT native capture external_data location conflicts with the ONNX model path: ${location}.`);
    let candidate;
    try {
      candidate = await realpath(path.resolve(root, ...location.split("/")));
    } catch (error) {
      throw new Error(`ORT native capture cannot resolve external_data file ${location}: ${error?.code || error?.message || "unknown error"}.`);
    }
    const relative = path.relative(root, candidate);
    if (!relative || relative.startsWith(`..${path.sep}`) || relative === ".." || path.isAbsolute(relative)) {
      throw new Error(`ORT native capture external_data location escapes the model directory: ${location}.`);
    }
    const fileInfo = await stat(candidate);
    if (!fileInfo.isFile()) throw new Error(`ORT native capture external_data location is not a regular file: ${location}.`);
    if (!Number.isSafeInteger(fileInfo.size) || fileInfo.size < 0 || fileInfo.size > MAX_EXTERNAL_FILE_BYTES) {
      throw new Error(`ORT native capture external_data file ${location} exceeds ${MAX_EXTERNAL_FILE_BYTES} bytes.`);
    }
    aggregateBytes += fileInfo.size;
    if (!Number.isSafeInteger(aggregateBytes) || aggregateBytes > MAX_EXTERNAL_AGGREGATE_BYTES) {
      throw new Error(`ORT native capture external_data aggregate exceeds ${MAX_EXTERNAL_AGGREGATE_BYTES} bytes.`);
    }
    const bytes = await readFile(candidate);
    if (bytes.byteLength !== fileInfo.size) throw new Error(`ORT native capture external_data file ${location} changed while it was being read.`);
    records.push({
      path: location,
      bytes,
      sha256: sha256Bytes(bytes),
      sha1: createHash("sha1").update(bytes).digest("hex"),
    });
  }
  const verified = analyzeOnnxModel(
    new Uint8Array(artifactBytes.buffer, artifactBytes.byteOffset, artifactBytes.byteLength),
    path.basename(artifactPath),
    null,
    { externalDataFiles: records },
  );
  const evidence = verified.onnx_external_data || {};
  if (Number(evidence.tensor_count || 0) > 0 && evidence.status !== "verified_payloads") {
    throw new Error(`ORT native capture external_data verification failed: ${evidence.detail || evidence.status}.`);
  }
  const files = (evidence.supplied_files || []).filter((file) => file.used).map((file) => ({
    path: file.path,
    byte_length: file.byte_length,
    sha256: file.sha256,
    sha1: file.sha1,
  })).sort((left, right) => compareText(left.path, right.path));
  const tensorRanges = (evidence.tensors || []).map((row) => ({
    scope: row.scope,
    tensor_role: row.tensor_role,
    tensor_name: row.tensor_name,
    location: row.normalized_location,
    offset: row.offset,
    length: row.length,
    payload_bytes: row.payload_bytes,
    checksum: String(row.checksum || "").toLowerCase(),
    sidecar_sha256: row.sidecar_sha256,
  })).sort(compareTensorRange);
  const externalData = {
    schema: ORT_EXTERNAL_ARTIFACT_SET_SCHEMA,
    status: tensorRanges.length ? "verified_payloads" : "assessed_absent",
    tensor_count: tensorRanges.length,
    verified_payload_bytes: Number(evidence.verified_payload_bytes || 0),
    tensor_ranges: tensorRanges,
    files,
  };
  externalData.ledger_sha256 = sha256Text(canonicalJson(externalArtifactLedgerPayload(externalData)));
  const artifact = {
    name: path.basename(artifactPath),
    byte_length: artifactBytes.byteLength,
    sha256: modelSha256,
    external_data: externalData,
  };
  artifact.content_set_sha256 = sha256Text(canonicalJson(artifactContentSetPayload(artifact)));
  return { identity: artifact, externalFiles: records };
}

async function stageOnnxArtifactSet(snapshotRoot, artifactSet, artifactBytes) {
  await mkdir(snapshotRoot, { recursive: false });
  const modelPath = path.join(snapshotRoot, artifactSet.identity.name);
  await writeFile(modelPath, artifactBytes);
  for (const file of artifactSet.externalFiles) {
    const target = path.join(snapshotRoot, ...file.path.split("/"));
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, file.bytes);
  }
  await assertOnnxArtifactIdentity(modelPath, artifactSet.identity, "staged ONNX content set does not match the verified source bytes");
  return modelPath;
}

async function assertOnnxArtifactIdentity(artifactPath, expected, reason) {
  const bytes = await readFile(artifactPath);
  const observed = await collectOnnxArtifactIdentity(artifactPath, bytes);
  if (canonicalJson(observed) !== canonicalJson(expected)) throw new Error(`ORT native capture ${reason}.`);
}

function externalArtifactLedgerPayload(externalData) {
  return {
    schema: externalData.schema,
    status: externalData.status,
    tensor_count: externalData.tensor_count,
    verified_payload_bytes: externalData.verified_payload_bytes,
    tensor_ranges: externalData.tensor_ranges,
    files: externalData.files,
  };
}

function artifactContentSetPayload(artifact) {
  return {
    byte_length: artifact.byte_length,
    sha256: artifact.sha256,
    external_data: artifact.external_data,
  };
}

function validateArtifactIdentity(artifact) {
  if (!artifact?.name || safeBasename(artifact.name) !== artifact.name || !Number.isSafeInteger(artifact.byte_length) || artifact.byte_length < 1
    || !SHA256.test(artifact.sha256) || !SHA256.test(artifact.content_set_sha256)) throw new Error("ORT native artifact identity is invalid.");
  const external = artifact.external_data;
  if (!external || external.schema !== ORT_EXTERNAL_ARTIFACT_SET_SCHEMA
    || !["assessed_absent", "verified_payloads"].includes(external.status)
    || !Number.isSafeInteger(external.tensor_count) || external.tensor_count < 0
    || !Number.isSafeInteger(external.verified_payload_bytes) || external.verified_payload_bytes < 0
    || !Array.isArray(external.tensor_ranges) || !Array.isArray(external.files)
    || !SHA256.test(external.ledger_sha256)) throw new Error("ORT native external-data artifact identity is invalid.");
  if (external.files.length > MAX_EXTERNAL_FILE_COUNT
    || canonicalJson(external.files) !== canonicalJson([...external.files].sort((left, right) => compareText(left.path, right.path)))
    || canonicalJson(external.tensor_ranges) !== canonicalJson([...external.tensor_ranges].sort(compareTensorRange))) {
    throw new Error("ORT native external-data ledger ordering is not canonical.");
  }
  const files = new Map();
  let aggregateFileBytes = 0;
  for (const file of external.files) {
    if (safeRelativePath(file.path) !== file.path || files.has(file.path)
      || !Number.isSafeInteger(file.byte_length) || file.byte_length < 0 || file.byte_length > MAX_EXTERNAL_FILE_BYTES
      || !SHA256.test(file.sha256) || !/^[a-f0-9]{40}$/.test(file.sha1 || "")) throw new Error("ORT native external-data file identity is invalid.");
    aggregateFileBytes += file.byte_length;
    if (!Number.isSafeInteger(aggregateFileBytes) || aggregateFileBytes > MAX_EXTERNAL_AGGREGATE_BYTES) throw new Error("ORT native external-data aggregate identity is invalid.");
    files.set(file.path, file);
  }
  let payloadBytes = 0;
  const usedFiles = new Set();
  for (const row of external.tensor_ranges) {
    const file = files.get(row.location);
    if (!row.scope || ![
      "graph_initializer", "node_attribute_tensor", "function_default_attribute_tensor",
      "graph_sparse_initializer_values", "graph_sparse_initializer_indices",
      "node_attribute_sparse_tensor_values", "node_attribute_sparse_tensor_indices",
      "function_default_attribute_sparse_tensor_values", "function_default_attribute_sparse_tensor_indices",
    ].includes(row.tensor_role)
      || !file || !Number.isSafeInteger(row.offset) || row.offset < 0
      || (row.length != null && (!Number.isSafeInteger(row.length) || row.length < 0))
      || !Number.isSafeInteger(row.payload_bytes) || row.payload_bytes < 0
      || !Number.isSafeInteger(row.offset + row.payload_bytes) || row.offset + row.payload_bytes > file.byte_length
      || (row.length == null ? row.payload_bytes !== file.byte_length - row.offset : row.length !== row.payload_bytes)
      || row.sidecar_sha256 !== file.sha256
      || (row.checksum && (!/^[a-f0-9]{40}$/.test(row.checksum) || row.checksum !== file.sha1))) throw new Error("ORT native external-data tensor-range identity is invalid.");
    payloadBytes += row.payload_bytes;
    if (!Number.isSafeInteger(payloadBytes)) throw new Error("ORT native external-data payload total is unsafe.");
    usedFiles.add(row.location);
  }
  if (external.tensor_count !== external.tensor_ranges.length || external.verified_payload_bytes !== payloadBytes
    || (external.status === "assessed_absent" ? external.tensor_count !== 0 || external.files.length !== 0 : external.tensor_count === 0)
    || usedFiles.size !== external.files.length
    || external.ledger_sha256 !== sha256Text(canonicalJson(externalArtifactLedgerPayload(external)))
    || artifact.content_set_sha256 !== sha256Text(canonicalJson(artifactContentSetPayload(artifact)))) {
    throw new Error("ORT native artifact content-set ledger is inconsistent.");
  }
}

function validateProviders(providers, supportedBackends) {
  const values = [...new Set((providers || []).map((item) => String(item).trim().toLowerCase()).filter(Boolean))];
  if (!values.length || values.length > 8) throw new Error("ORT capture requires one to eight execution providers.");
  const supported = new Set(supportedBackends.map((item) => item.name.toLowerCase()));
  for (const provider of values) if (!supported.has(provider)) throw new Error(`ORT backend ${provider} is not supported by this pinned host package.`);
  return values;
}

function tensorDataSha(type, dims, data) {
  const hash = createHash("sha256").update(`${type}\0${dims.join(",")}\0`);
  if (ArrayBuffer.isView(data)) hash.update(Buffer.from(data.buffer, data.byteOffset, data.byteLength));
  else hash.update(canonicalJson(data));
  return hash.digest("hex");
}

function isOrtKernelEvent(event) {
  const args = event?.args;
  return String(event?.cat || "").toLowerCase() === "node"
    && String(event?.name || "").endsWith("_kernel_time")
    && typeof args?.provider === "string" && args.provider
    && typeof args?.op_name === "string" && args.op_name
    && /^\d+$/.test(String(args?.node_index ?? ""))
    && Number.isFinite(Number(event?.dur)) && Number(event.dur) >= 0;
}

async function locateProfile(output, prefixName) {
  const candidates = (await readdir(output))
    .filter((name) => name.startsWith(`${prefixName}_`) && name.endsWith(".json"))
    .sort();
  if (candidates.length !== 1) throw new Error(`ORT profiler emitted ${candidates.length} files for ${prefixName}; expected exactly one.`);
  return path.join(output, candidates[0]);
}

async function recursiveFiles(root) {
  const output = [];
  const visit = async (directory) => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const target = path.join(directory, entry.name);
      if (entry.isDirectory()) await visit(target);
      else if (entry.isFile()) output.push(target);
    }
  };
  await visit(root);
  return output;
}

async function requireEmptyOrMissingDirectory(directory) {
  try {
    const info = await stat(directory);
    if (!info.isDirectory()) throw new Error(`ORT capture output exists and is not a directory: ${directory}`);
    if ((await readdir(directory)).length) throw new Error(`ORT capture output directory must be empty: ${directory}`);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
}

function contentSha256(value) {
  const { capture_content_sha256: _digest, ...content } = value || {};
  return sha256Text(canonicalJson(content));
}

function safeBasename(value) {
  const text = String(value || "");
  if (!text || path.basename(text) !== text || text.includes("..")) throw new Error("ORT capture filename is unsafe.");
  return text;
}

function safeRelativePath(value) {
  const text = String(value || "").replaceAll("\\", "/").replace(/^(?:\.\/)+/, "");
  if (!text || text.includes("\0") || /^[a-z][a-z0-9+.-]*:/i.test(text) || text.startsWith("/")
    || text.split("/").some((part) => !part || part === "." || part === "..")) return null;
  return text;
}

function compareTensorRange(left, right) {
  return compareText(
    [left.scope, left.tensor_role, left.tensor_name, left.location, left.offset, left.length, left.payload_bytes, left.checksum, left.sidecar_sha256].join("\0"),
    [right.scope, right.tensor_role, right.tensor_name, right.location, right.offset, right.length, right.payload_bytes, right.checksum, right.sidecar_sha256].join("\0"),
  );
}

function compareText(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function requiredText(value, label) {
  const text = String(value || "").trim();
  if (!text) throw new Error(`ORT native capture ${label} is required.`);
  return text;
}

function validatePositiveInteger(value, label, maximum) {
  if (!Number.isSafeInteger(Number(value)) || Number(value) < 1 || Number(value) > maximum) throw new Error(`ORT native capture ${label} must be 1..${maximum}.`);
}

function validateNonNegativeInteger(value, label, maximum) {
  if (!Number.isSafeInteger(Number(value)) || Number(value) < 0 || Number(value) > maximum) throw new Error(`ORT native capture ${label} must be 0..${maximum}.`);
}

function sha256Bytes(value) { return createHash("sha256").update(value).digest("hex"); }
function sha256Text(value) { return sha256Bytes(Buffer.from(String(value), "utf8")); }
function stableJson(value) { return `${JSON.stringify(value, null, 2)}\n`; }
