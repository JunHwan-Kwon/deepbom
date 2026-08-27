import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { access, mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

export const CAPTURE_RUN_SCHEMA = "deepbom.native_capture_run.v1.2";
export const CAPTURE_MANIFEST_SCHEMA = "deepbom.native_capture_manifest.v4";
export const RUNTIME_ASSIGNMENT_SCHEMA = "deepbom.runtime_assignment.v1.9";
export const CAPTURE_PACKAGE_SCHEMA = "deepbom.native_capture_package.v1.1";
export const TENSORFLOW_SOURCE_COMMIT = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
export const XNNPACK_SOURCE_COMMIT = "23a67314f7afdbb76191589ae090d82bf55afbfa";

const CAPTURE_MODES = new Set(["runtime_capture", "synthetic_contract_probe"]);

export async function sha256File(filePath) {
  return createHash("sha256").update(await readFile(filePath)).digest("hex");
}

export function stableJson(value) {
  return `${JSON.stringify(sortJson(value), null, 2)}\n`;
}

export async function runNativeCapture(configPath, options = {}) {
  const absoluteConfigPath = path.resolve(configPath);
  const configDir = path.dirname(absoluteConfigPath);
  const config = JSON.parse(await readFile(absoluteConfigPath, "utf8"));
  validateRunConfig(config);

  const outputDir = path.resolve(configDir, options.outputDir || config.output_dir);
  const artifactPath = resolveFrom(configDir, config.artifact_path);
  const runtimeBinaryPath = resolveFrom(configDir, config.runtime.binary_path);
  if (config.capture_mode === "runtime_capture" && path.basename(runtimeBinaryPath).toLowerCase().includes("contract-probe")) {
    throw new Error("the synthetic contract probe cannot produce runtime_capture evidence");
  }
  await ensureEmptyCaptureDirectory(outputDir);
  const buildIdPath = resolveFrom(configDir, config.build.microkernel_build_identifier_path);
  const debugSymbolsPath = config.build.debug_symbols_path
    ? resolveFrom(configDir, config.build.debug_symbols_path)
    : null;
  for (const [label, filePath] of [
    ["artifact", artifactPath],
    ["microkernel build identifier", buildIdPath],
    ...(debugSymbolsPath ? [["runtime debug symbols", debugSymbolsPath]] : []),
  ]) await requireFile(filePath, label);

  const replacements = {
    artifact_path: artifactPath,
    capture_dir: outputDir,
    config_dir: configDir,
    events_path: path.join(outputDir, "runtime-events.ndjson"),
  };
  if (config.runtime.build_command) {
    await runConfiguredCommand(config.runtime.build_command, configDir, replacements, "runtime build");
  }
  await requireFile(runtimeBinaryPath, "runtime binary");

  const collectorPath = path.resolve(options.collectorPath || defaultCollectorPath());
  if (!options.skipCollectorBuild) {
    await runProcess("cargo", ["build", "--release", "--manifest-path", "native/runtime_collector/Cargo.toml"], process.cwd(), {}, "collector build");
  }
  await requireFile(collectorPath, "collector binary");

  const artifactSha256 = await sha256File(artifactPath);
  const runtimeBinarySha256 = await sha256File(runtimeBinaryPath);
  const buildIdSha256 = await sha256File(buildIdPath);
  const debugSymbolsSha256 = debugSymbolsPath ? await sha256File(debugSymbolsPath) : null;
  if (debugSymbolsPath) {
    const buildIdentity = JSON.parse(await readFile(buildIdPath, "utf8"));
    if (buildIdentity.runtime_debug_symbols?.format !== "pdb"
      || buildIdentity.runtime_debug_symbols?.sha256 !== debugSymbolsSha256) {
      throw new Error("runtime PDB does not match the instrumented build identity");
    }
  }
  const buildManifestPath = path.join(outputDir, "runtime-build-manifest.json");
  const runtimeOptionsPath = path.join(outputDir, "runtime-options.json");
  const eventPath = replacements.events_path;
  const manifestPath = path.join(outputDir, "capture-manifest.json");
  const assignmentPath = path.join(outputDir, "runtime-assignment.json");
  const resourcePartitionPath = config.invocation.resource_partition
    ? path.join(outputDir, "resource-partition-observation.json")
    : null;

  await writeFile(buildManifestPath, stableJson({
    schema: "deepbom.native_runtime_build_manifest.v1",
    runtime: config.runtime.build,
    source: config.build.source,
    xnnpack_source_commit: config.build.xnnpack_source_commit,
    cmake_system_name: config.build.cmake_system_name || null,
    compile_definitions: config.build.compile_definitions,
    toolchain: config.build.toolchain,
    debug_symbols: debugSymbolsPath ? {
      format: "pdb",
      filename: path.basename(debugSymbolsPath),
      sha256: debugSymbolsSha256,
    } : null,
  }));
  await writeFile(runtimeOptionsPath, stableJson(config.invocation.runtime_options));

  const captureId = config.source?.capture_id || createHash("sha256")
    .update(stableJson({ artifactSha256, runtimeBinarySha256, buildIdSha256, invocation: config.invocation }))
    .digest("hex");
  const collectedAt = config.source?.collected_at || new Date().toISOString();
  const collectorSourceCommit = config.source?.collector_source_commit || await gitSourceIdentity();
  const environment = Object.fromEntries(Object.entries(config.runtime.environment || {})
    .map(([name, value]) => [name, expand(String(value), replacements)]));
  Object.assign(environment, {
    DEEPBOM_ARTIFACT_SHA256: artifactSha256,
    DEEPBOM_CAPTURE_ID: captureId,
    DEEPBOM_CAPTURE_MODE: config.capture_mode,
    DEEPBOM_MICROKERNEL_BUILD_ID_SHA256: buildIdSha256,
    DEEPBOM_RUNTIME_EVENTS_PATH: eventPath,
    DEEPBOM_XNNPACK_SOURCE_COMMIT: config.build.xnnpack_source_commit,
  });
  const runtimeArguments = config.runtime.arguments.map((value) => expand(value, replacements));
  if (config.invocation.resource_partition) {
    const observation = await runProcessWithResourceObservation(
      runtimeBinaryPath,
      runtimeArguments,
      configDir,
      environment,
      "instrumented runtime",
      config.invocation.resource_partition,
    );
    await writeFile(resourcePartitionPath, stableJson(observation));
    enforceResourcePartitionObservation(observation, config.invocation.resource_partition, config.invocation.thread_count);
  } else {
    await runProcess(runtimeBinaryPath, runtimeArguments, configDir, environment, "instrumented runtime");
  }
  await requireNonemptyFile(eventPath, "runtime event stream");

  const manifest = {
    schema: CAPTURE_MANIFEST_SCHEMA,
    capture_mode: config.capture_mode,
    artifact_path: artifactPath,
    artifact_sha256: artifactSha256,
    target_profile_id: config.target_profile.id,
    target_profile_sha256: config.target_profile.sha256,
    runtime: {
      name: config.runtime.name,
      version: config.runtime.version,
      backend: config.runtime.backend,
      build: config.runtime.build,
      binary_path: runtimeBinaryPath,
    },
    source: { collected_at: collectedAt, capture_id: captureId, collector_source_commit: collectorSourceCommit },
    device: {
      identity: config.device.identity,
      nnapi_runtime_feature_level: config.device.nnapi_runtime_feature_level ?? null,
      nnapi_accelerator_identity: config.device.nnapi_accelerator_identity ?? null,
      nnapi_capability_source: config.device.nnapi_capability_source ?? null,
    },
    build: {
      tensorflow_source_commit: config.build.source.tensorflow_commit,
      xnnpack_source_commit: config.build.xnnpack_source_commit,
      cmake_system_name: config.build.cmake_system_name || null,
      microkernel_build_identifier_path: buildIdPath,
      build_manifest_path: buildManifestPath,
      compile_definitions: config.build.compile_definitions,
    },
    invocation: {
      inputs: config.invocation.inputs,
      thread_count: config.invocation.thread_count,
      runtime_options_path: runtimeOptionsPath,
      resource_partition_request: config.invocation.resource_partition || null,
      resource_partition_observation_path: resourcePartitionPath,
      tflite_gpu_experimental_flags: config.invocation.runtime_options.TFLITE_GPU_EXPERIMENTAL_FLAGS ?? null,
      tflite_gpu_max_delegated_partitions: config.invocation.runtime_options.TFLITE_GPU_MAX_DELEGATED_PARTITIONS ?? null,
    },
    instrumentation: config.instrumentation,
  };
  await writeFile(manifestPath, stableJson(manifest));
  await runProcess(collectorPath, [manifestPath, eventPath, assignmentPath], configDir, {}, "runtime evidence collector");

  const assignment = JSON.parse(await readFile(assignmentPath, "utf8"));
  validateCollectedAssignment(assignment, {
    captureMode: config.capture_mode,
    artifactSha256,
    targetProfileSha256: config.target_profile.sha256,
    runtimeBinarySha256,
    instrumentation: config.instrumentation,
  });
  const files = {};
  for (const name of [
    "capture-manifest.json",
    "runtime-events.ndjson",
    "runtime-build-manifest.json",
    "runtime-options.json",
    ...(resourcePartitionPath ? ["resource-partition-observation.json"] : []),
    "runtime-assignment.json",
  ]) {
    files[name] = await sha256File(path.join(outputDir, name));
  }
  const index = {
    schema: CAPTURE_PACKAGE_SCHEMA,
    capture_mode: config.capture_mode,
    importable_runtime_evidence: config.capture_mode === "runtime_capture",
    artifact_sha256: artifactSha256,
    runtime_binary_sha256: runtimeBinarySha256,
    microkernel_build_identifier_sha256: buildIdSha256,
    files,
  };
  await writeFile(path.join(outputDir, "capture-index.json"), stableJson(index));
  return { assignment, index, outputDir };
}

export async function verifyNativeCapturePackage(captureDir) {
  const root = path.resolve(captureDir);
  const index = JSON.parse(await readFile(path.join(root, "capture-index.json"), "utf8"));
  if (index?.schema !== CAPTURE_PACKAGE_SCHEMA || !CAPTURE_MODES.has(index.capture_mode)) {
    throw new Error("native capture package schema or mode is invalid");
  }
  const baseFiles = [
    "capture-manifest.json",
    "runtime-events.ndjson",
    "runtime-build-manifest.json",
    "runtime-options.json",
    "runtime-assignment.json",
  ];
  const hasResourcePartition = Object.prototype.hasOwnProperty.call(index.files || {}, "resource-partition-observation.json");
  const expectedFiles = [...baseFiles, ...(hasResourcePartition ? ["resource-partition-observation.json"] : [])];
  if (JSON.stringify(Object.keys(index.files || {}).sort()) !== JSON.stringify([...expectedFiles].sort())) {
    throw new Error("native capture package file inventory is incomplete or contains unknown entries");
  }
  for (const name of expectedFiles) {
    requireSha(index.files[name], `capture-index files.${name}`, 64);
    const actual = await sha256File(path.join(root, name));
    if (actual !== index.files[name]) throw new Error(`native capture package hash mismatch for ${name}`);
  }
  const manifest = JSON.parse(await readFile(path.join(root, "capture-manifest.json"), "utf8"));
  const assignment = JSON.parse(await readFile(path.join(root, "runtime-assignment.json"), "utf8"));
  if (manifest.schema !== CAPTURE_MANIFEST_SCHEMA || manifest.capture_mode !== index.capture_mode
    || assignment.schema !== RUNTIME_ASSIGNMENT_SCHEMA) {
    throw new Error("capture manifest schema or mode differs from package index");
  }
  if (manifest.artifact_sha256 !== index.artifact_sha256
    || assignment.artifact_sha256 !== index.artifact_sha256
    || assignment.runtime?.binary_sha256 !== index.runtime_binary_sha256
    || assignment.selector_context?.build?.runtime_binary_sha256 !== index.runtime_binary_sha256
    || assignment.selector_context?.build?.microkernel_build_identifier_sha256 !== index.microkernel_build_identifier_sha256) {
    throw new Error("native capture package identity chain is inconsistent");
  }
  if (manifest.target_profile_id !== assignment.target_profile_id
    || manifest.target_profile_sha256 !== assignment.target_profile_sha256
    || manifest.source?.capture_id !== assignment.source?.capture_id
    || manifest.build?.xnnpack_source_commit !== assignment.selector_context?.build?.xnnpack_source_commit
    || stableJson(manifest.build?.compile_definitions) !== stableJson(assignment.selector_context?.build?.compile_definitions)
    || ["name", "version", "backend", "build"].some((field) => manifest.runtime?.[field] !== assignment.runtime?.[field])) {
    throw new Error("native capture package manifest and assignment declarations differ");
  }
  if (manifest.instrumentation?.arena_allocations === true
    ? manifest.build?.tensorflow_source_commit !== assignment.runtime_memory?.tensorflow_source_commit
    : assignment.runtime_memory != null) {
    throw new Error("native capture package arena instrumentation and TensorFlow source binding differ");
  }
  const expectedKind = index.capture_mode === "runtime_capture"
    ? "deepbom_native_runtime_capture"
    : "deepbom_native_runtime_contract_probe";
  if (assignment.source?.kind !== expectedKind
    || index.importable_runtime_evidence !== (index.capture_mode === "runtime_capture")) {
    throw new Error("native capture package evidence class is inconsistent");
  }
  if (assignment.source?.dispatch_sample_semantics !== "unique_context_function_selection_per_process") {
    throw new Error("native capture package dispatch sample semantics are missing or unsupported");
  }
  if (assignment.source?.profile_sha256 !== index.files["runtime-events.ndjson"]
    || assignment.selector_context?.build?.build_manifest_sha256 !== index.files["runtime-build-manifest.json"]
    || assignment.selector_context?.invocation?.runtime_options_sha256 !== index.files["runtime-options.json"]) {
    throw new Error("native capture package evidence file bindings are inconsistent");
  }
  const partition = assignment.selector_context?.invocation?.resource_partition;
  if (hasResourcePartition) {
    if (!partition || partition.observation_sha256 !== index.files["resource-partition-observation.json"]) {
      throw new Error("native capture package resource-partition observation binding is inconsistent");
    }
    const observationBytes = await readFile(path.join(root, "resource-partition-observation.json"));
    const observation = JSON.parse(observationBytes.toString("utf8"));
    validateResourcePartitionObservation(
      observation,
      manifest.invocation?.resource_partition_request,
      manifest.invocation?.thread_count,
    );
    const expectedPartition = resourcePartitionAssignmentSummary(
      observation,
      createHash("sha256").update(observationBytes).digest("hex"),
    );
    if (stableJson(partition) !== stableJson(expectedPartition)) {
      throw new Error("native capture package assignment summary does not reproduce the raw affinity sample ledger");
    }
  } else if (partition != null) {
    throw new Error("runtime assignment emitted resource-partition evidence without a packaged observation");
  } else if (manifest.invocation?.resource_partition_request != null
    || manifest.invocation?.resource_partition_observation_path != null) {
    throw new Error("native capture manifest declares a resource partition without packaged observation evidence");
  }
  return { assignment, index, manifest, root };
}

export function validateRunConfig(config) {
  if (config?.schema !== CAPTURE_RUN_SCHEMA) throw new Error(`capture run schema must be ${CAPTURE_RUN_SCHEMA}`);
  if (!CAPTURE_MODES.has(config.capture_mode)) throw new Error("capture_mode must be runtime_capture or synthetic_contract_probe");
  for (const [field, value] of [
    ["artifact_path", config.artifact_path],
    ["output_dir", config.output_dir],
    ["runtime.binary_path", config.runtime?.binary_path],
    ["runtime.name", config.runtime?.name],
    ["runtime.version", config.runtime?.version],
    ["runtime.backend", config.runtime?.backend],
    ["target_profile.id", config.target_profile?.id],
    ["device.identity", config.device?.identity],
    ["build.microkernel_build_identifier_path", config.build?.microkernel_build_identifier_path],
  ]) if (typeof value !== "string" || !value.trim()) throw new Error(`${field} is required`);
  if (config.build.debug_symbols_path != null
    && (typeof config.build.debug_symbols_path !== "string" || !config.build.debug_symbols_path.trim())) {
    throw new Error("build.debug_symbols_path must be a non-empty string when present");
  }
  if (config.build.cmake_system_name != null
    && (typeof config.build.cmake_system_name !== "string" || !config.build.cmake_system_name.trim())) {
    throw new Error("build.cmake_system_name must be a non-empty string when present");
  }
  requireSha(config.target_profile?.sha256, "target_profile.sha256", 64);
  requireSha(config.build?.source?.tensorflow_commit, "build.source.tensorflow_commit", 40);
  requireSha(config.build?.xnnpack_source_commit, "build.xnnpack_source_commit", 40);
  if (config.build.source.tensorflow_commit !== TENSORFLOW_SOURCE_COMMIT) throw new Error(`build.source.tensorflow_commit must match the pinned ${TENSORFLOW_SOURCE_COMMIT}`);
  if (config.build.xnnpack_source_commit !== XNNPACK_SOURCE_COMMIT) throw new Error(`build.xnnpack_source_commit must match the pinned ${XNNPACK_SOURCE_COMMIT}`);
  if (!Array.isArray(config.runtime.arguments)) throw new Error("runtime.arguments must be an array");
  if (config.runtime.environment != null && (typeof config.runtime.environment !== "object" || Array.isArray(config.runtime.environment))) throw new Error("runtime.environment must be an object");
  if (!Array.isArray(config.build.compile_definitions) || !config.build.compile_definitions.length) throw new Error("build.compile_definitions are required");
  const names = config.build.compile_definitions.map((item) => item?.name);
  if (JSON.stringify(names) !== JSON.stringify([...names].sort()) || new Set(names).size !== names.length) throw new Error("compile definitions must be unique and sorted");
  for (const name of ["XNN_BUILD_ALL_MICROKERNELS", "XNN_ENABLE_ASSEMBLY"]) {
    if (!names.includes(name)) throw new Error(`build.compile_definitions must include ${name}`);
  }
  if (config.build.compile_definitions.some((item) => typeof item?.name !== "string" || typeof item?.value !== "string" || !item.name || !item.value)) throw new Error("compile definitions require non-empty string names and values");
  for (const name of ["TFLITE_ENABLE_GPU", "TFLITE_ENABLE_NNAPI"]) {
    const value = config.build.compile_definitions.find((item) => item.name === name)?.value;
    if (value != null && !["0", "1", "OFF", "ON", "FALSE", "TRUE"].includes(value.toUpperCase())) {
      throw new Error(`${name} must carry an explicit boolean value`);
    }
  }
  const gpuFlags = config.invocation?.runtime_options?.TFLITE_GPU_EXPERIMENTAL_FLAGS;
  if (gpuFlags != null && (!Number.isSafeInteger(gpuFlags) || gpuFlags < 0)) throw new Error("TFLITE_GPU_EXPERIMENTAL_FLAGS must be a non-negative safe integer");
  const gpuPartitions = config.invocation?.runtime_options?.TFLITE_GPU_MAX_DELEGATED_PARTITIONS;
  if (gpuPartitions != null && (!Number.isSafeInteger(gpuPartitions) || gpuPartitions < 1)) throw new Error("TFLITE_GPU_MAX_DELEGATED_PARTITIONS must be a positive safe integer");
  if ((config.device.nnapi_runtime_feature_level != null || config.device.nnapi_accelerator_identity != null)
    && !config.device.nnapi_capability_source) throw new Error("NNAPI feature or accelerator declarations require device.nnapi_capability_source");
  if (config.device.nnapi_runtime_feature_level != null
    && (!Number.isSafeInteger(config.device.nnapi_runtime_feature_level) || config.device.nnapi_runtime_feature_level < 27)) throw new Error("device.nnapi_runtime_feature_level must be an integer at least 27");
  if (typeof config.build.toolchain !== "object" || config.build.toolchain == null || Array.isArray(config.build.toolchain) || !Object.keys(config.build.toolchain).length) throw new Error("build.toolchain requires at least one identity field");
  if (!Array.isArray(config.invocation?.inputs) || !config.invocation.inputs.length || !Number.isSafeInteger(config.invocation.thread_count) || config.invocation.thread_count < 1) {
    throw new Error("invocation requires inputs and a positive integer thread_count");
  }
  if (config.invocation.resource_partition != null) {
    const partition = config.invocation.resource_partition;
    if (partition.affinity_mode !== "taskset_process_and_descendants") {
      throw new Error("resource_partition.affinity_mode must be taskset_process_and_descendants");
    }
    if (!["affinity_only", "exclusive_cpuset"].includes(partition.isolation_expectation)) {
      throw new Error("resource_partition.isolation_expectation must be affinity_only or exclusive_cpuset");
    }
    const cpuIds = partition.requested_cpu_ids;
    if (!Array.isArray(cpuIds) || !cpuIds.length
      || cpuIds.some((cpu) => !Number.isSafeInteger(cpu) || cpu < 0)
      || new Set(cpuIds).size !== cpuIds.length
      || JSON.stringify(cpuIds) !== JSON.stringify([...cpuIds].sort((a, b) => a - b))) {
      throw new Error("resource_partition.requested_cpu_ids must be a non-empty, unique, ascending array of non-negative CPU IDs");
    }
    if (config.invocation.thread_count > cpuIds.length) {
      throw new Error("invocation.thread_count must not exceed the requested isolated CPU count");
    }
  }
  const inputIndices = config.invocation.inputs.map((input) => input?.tensor_index);
  if (new Set(inputIndices).size !== inputIndices.length || JSON.stringify(inputIndices) !== JSON.stringify([...inputIndices].sort((a, b) => a - b))
    || config.invocation.inputs.some((input) => !Number.isSafeInteger(input?.tensor_index) || input.tensor_index < 0 || typeof input.name !== "string" || !input.name || !Array.isArray(input.shape) || !input.shape.length || input.shape.some((dim) => !Number.isSafeInteger(dim) || dim < 1))) {
    throw new Error("invocation inputs must be unique, sorted, named, and carry concrete positive shapes");
  }
  if (typeof config.invocation.runtime_options !== "object" || config.invocation.runtime_options == null) throw new Error("invocation.runtime_options is required");
  if (typeof config.instrumentation?.lowering_ids !== "boolean" || typeof config.instrumentation?.microkernel_ids !== "boolean" || typeof config.instrumentation?.arena_allocations !== "boolean") throw new Error("instrumentation capabilities must be explicit booleans");
  if (config.instrumentation.microkernel_ids && !config.instrumentation.lowering_ids) throw new Error("microkernel instrumentation requires lowering instrumentation");
  if (config.capture_mode === "runtime_capture" && config.instrumentation.microkernel_ids) {
    const noFusionEnvironment = String(config.runtime.environment?.DEEPBOM_XNN_NO_OPERATOR_FUSION || "");
    const noFusionOption = String(config.invocation.runtime_options?.DEEPBOM_XNN_NO_OPERATOR_FUSION || "");
    if (noFusionEnvironment !== "1" || noFusionOption !== "1") {
      throw new Error("microkernel attribution capture requires DEEPBOM_XNN_NO_OPERATOR_FUSION=1 in both runtime.environment and invocation.runtime_options");
    }
  }
  if (config.source?.collected_at && !/^\d{4}-\d{2}-\d{2}T.+(?:Z|[+-]\d{2}:\d{2})$/.test(config.source.collected_at)) throw new Error("source.collected_at must be timezone-qualified ISO-8601");
}

function validateCollectedAssignment(value, expected) {
  if (value?.schema !== RUNTIME_ASSIGNMENT_SCHEMA) throw new Error(`collector output schema must be ${RUNTIME_ASSIGNMENT_SCHEMA}`);
  const expectedKind = expected.captureMode === "runtime_capture"
    ? "deepbom_native_runtime_capture"
    : "deepbom_native_runtime_contract_probe";
  if (value.source?.kind !== expectedKind) throw new Error(`collector source kind must be ${expectedKind}`);
  if (value.artifact_sha256 !== expected.artifactSha256
    || value.target_profile_sha256 !== expected.targetProfileSha256
    || value.runtime?.binary_sha256 !== expected.runtimeBinarySha256
    || value.selector_context?.build?.runtime_binary_sha256 !== expected.runtimeBinarySha256) {
    throw new Error("collector output identity binding failed");
  }
  if (value.selector_context?.build?.tensorflow_source_commit !== TENSORFLOW_SOURCE_COMMIT
    || value.tflite_delegate_build_inventory?.schema !== "deepbom.tflite_delegate_build_inventory.v1"
    || value.tflite_delegate_build_inventory?.artifact_sha256 !== expected.artifactSha256
    || value.tflite_delegate_build_inventory?.runtime_binary_sha256 !== expected.runtimeBinarySha256
    || value.tflite_delegate_build_inventory?.tensorflow_source_commit !== TENSORFLOW_SOURCE_COMMIT) {
    throw new Error("collector TFLite delegate selected-build inventory binding failed");
  }
  if (!Array.isArray(value.assignments) || !value.assignments.length) throw new Error("collector emitted no assignments");
  const indices = value.assignments.map((item) => item.op_index);
  if (JSON.stringify(indices) !== JSON.stringify([...indices].sort((a, b) => a - b)) || new Set(indices).size !== indices.length) {
    throw new Error("collector assignments must be unique and sorted by original op index");
  }
  if (expected.captureMode !== "runtime_capture" && value.source.validation_scope !== "collector_contract_only_not_runtime_evidence") {
    throw new Error("synthetic output must carry the non-runtime validation scope");
  }
  const loweringCount = value.assignments.reduce((sum, item) => sum + Number(item.lowerings?.length || 0), 0);
  const dispatchCount = value.assignments.reduce((sum, item) => sum + Number(item.dispatches?.length || 0), 0);
  if (expected.captureMode === "runtime_capture" && expected.instrumentation?.microkernel_ids
    && loweringCount > 0 && dispatchCount === 0) {
    throw new Error("microkernel instrumentation observed lowering nodes but resolved no dispatch function pointers");
  }
  if (expected.instrumentation?.arena_allocations) {
    if (value.runtime_memory?.schema !== "deepbom.runtime_memory.v1"
      || value.runtime_memory?.status !== "assessed"
      || value.runtime_memory?.evidence_class !== "OBSERVED_RUNTIME"
      || !Number.isSafeInteger(value.runtime_memory?.snapshot_count)
      || value.runtime_memory.snapshot_count < 1
      || !Array.isArray(value.runtime_memory?.snapshots)
      || value.runtime_memory.snapshots.length !== value.runtime_memory.snapshot_count
      || !/^[a-f0-9]{64}$/.test(value.runtime_memory?.allocation_ledger_sha256 || "")) {
      throw new Error("collector runtime memory evidence is missing or malformed");
    }
  } else if (value.runtime_memory != null) {
    throw new Error("collector emitted runtime memory evidence beyond declared instrumentation");
  }
}

async function ensureEmptyCaptureDirectory(outputDir) {
  try {
    await access(outputDir);
    throw new Error(`capture output already exists: ${outputDir}`);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  await mkdir(outputDir, { recursive: true });
}

async function requireFile(filePath, label) {
  const info = await stat(filePath).catch(() => null);
  if (!info?.isFile()) throw new Error(`${label} is missing: ${filePath}`);
}

async function requireNonemptyFile(filePath, label) {
  await requireFile(filePath, label);
  if ((await stat(filePath)).size === 0) throw new Error(`${label} is empty: ${filePath}`);
}

async function runConfiguredCommand(spec, configDir, replacements, label) {
  if (typeof spec.command !== "string" || !Array.isArray(spec.arguments)) throw new Error(`${label} requires command and arguments`);
  await runProcess(
    expand(spec.command, replacements),
    spec.arguments.map((value) => expand(value, replacements)),
    resolveFrom(configDir, spec.cwd || "."),
    Object.fromEntries(Object.entries(spec.environment || {}).map(([key, value]) => [key, expand(String(value), replacements)])),
    label,
  );
}

async function runProcessWithResourceObservation(command, args, cwd, extraEnvironment, label, request) {
  if (process.platform !== "linux") throw new Error("resource-partition observation requires Linux taskset, procfs, and cgroupfs");
  const requested = request.requested_cpu_ids;
  const cpuList = formatCpuList(requested);
  console.log(`\n> taskset --cpu-list ${cpuList} ${command} ${args.join(" ")}`);
  const samples = [];
  let cgroupObservation = null;
  const child = spawn("taskset", ["--cpu-list", cpuList, command, ...args], {
    cwd,
    env: { ...process.env, ...extraEnvironment },
    stdio: "inherit",
    shell: false,
  });
  let finished = false;
  let exitCode = null;
  const exit = new Promise((resolve, reject) => {
    child.on("error", (error) => {
      finished = true;
      reject(error);
    });
    child.on("exit", (code) => {
      finished = true;
      exitCode = code;
      resolve(code);
    });
  });
  while (!finished) {
    const sample = await sampleLinuxProcessAffinity(child.pid).catch(() => null);
    if (sample?.threads?.length) samples.push(sample);
    if (!cgroupObservation) cgroupObservation = await observeLinuxCgroupPartition(child.pid).catch(() => null);
    await delay(20);
  }
  await exit;
  if (exitCode !== 0) throw new Error(`${label} failed with exit code ${exitCode}`);

  const system = await observeLinuxSystemPartition(requested);
  const observedThreads = new Map();
  const observedProcessors = new Set();
  let everySampledThreadWithinRequest = samples.length > 0;
  for (const sample of samples) {
    for (const thread of sample.threads) {
      observedThreads.set(thread.tid, thread.allowed_cpu_ids);
      if (!isSubset(thread.allowed_cpu_ids, requested)) everySampledThreadWithinRequest = false;
      if (Number.isSafeInteger(thread.processor)) observedProcessors.add(thread.processor);
    }
  }
  const observedAllowedCpuIds = [...new Set(
    [...observedThreads.values()].flat(),
  )].sort((left, right) => left - right);
  everySampledThreadWithinRequest = everySampledThreadWithinRequest
    && sameNumbers(observedAllowedCpuIds, requested);
  const effective = cgroupObservation?.effective_cpu_ids || [];
  const isolatedPartition = cgroupObservation?.partition_state === "isolated"
    && sameNumbers(effective, requested);
  return {
    schema: "deepbom.resource_partition_observation.v1",
    evidence_class: "OBSERVED_OS_RESOURCE_PARTITION",
    requested_cpu_ids: requested,
    affinity_mode: request.affinity_mode,
    isolation_expectation: request.isolation_expectation,
    affinity_status: everySampledThreadWithinRequest
      ? "observed_all_sampled_threads_within_requested_set"
      : "not_observed_or_outside_requested_set",
    exclusive_isolation_status: isolatedPartition
      ? "observed_cgroup_v2_isolated_partition"
      : "not_observed_affinity_only",
    sample_count: samples.length,
    maximum_observed_thread_count: samples.reduce((maximum, sample) => Math.max(maximum, sample.threads.length), 0),
    sampled_threads: [...observedThreads.entries()]
      .sort((left, right) => left[0] - right[0])
      .map(([tid, allowed_cpu_ids]) => ({ tid, allowed_cpu_ids })),
    observed_allowed_cpu_ids_union: observedAllowedCpuIds,
    observed_processor_ids: [...observedProcessors].sort((a, b) => a - b),
    observed_effective_cpu_ids: effective,
    cgroup_v2_path: cgroupObservation?.path || null,
    cgroup_v2_partition_state: cgroupObservation?.partition_state || null,
    online_cpu_ids: system.online_cpu_ids,
    kernel_command_line: system.kernel_command_line,
    kernel_isolation_parameters: system.kernel_isolation_parameters,
    cpu_frequency_policy: system.cpu_frequency_policy,
    cache_shared_cpu_lists: system.cache_shared_cpu_lists,
    affinity_samples: samples,
    interpretation_boundary: "Affinity is established only for sampled threads in this process and descendants inheriting the taskset mask. Exclusive isolation is established only when the observed cgroup v2 cpuset partition is isolated and its effective CPU set exactly matches the request. IRQ routing, scheduler ticks, frequency stability, thermal state, and memory contention remain separately reported or unbound.",
  };
}

async function sampleLinuxProcessAffinity(pid) {
  if (!Number.isSafeInteger(pid) || pid < 1) return null;
  const taskRoot = `/proc/${pid}/task`;
  const tids = (await readdir(taskRoot))
    .map(Number)
    .filter((tid) => Number.isSafeInteger(tid) && tid > 0)
    .sort((a, b) => a - b);
  const threads = [];
  for (const tid of tids) {
    const status = await readFile(`${taskRoot}/${tid}/status`, "utf8").catch(() => null);
    if (!status) continue;
    const allowedText = status.match(/^Cpus_allowed_list:\s*(.+)$/m)?.[1]?.trim();
    if (!allowedText) continue;
    const statText = await readFile(`${taskRoot}/${tid}/stat`, "utf8").catch(() => "");
    const tail = statText.slice(statText.lastIndexOf(")") + 2).trim().split(/\s+/);
    const processor = Number(tail[36]);
    threads.push({
      tid,
      allowed_cpu_ids: parseCpuList(allowedText),
      processor: Number.isSafeInteger(processor) ? processor : null,
    });
  }
  return { observed_at_unix_ms: Date.now(), threads };
}

async function observeLinuxCgroupPartition(pid) {
  const cgroup = await readFile(`/proc/${pid}/cgroup`, "utf8");
  const unified = cgroup.split(/\r?\n/).find((line) => line.startsWith("0::"));
  if (!unified) return null;
  const relative = unified.slice(3).replace(/^\/+/, "");
  const root = path.join("/sys/fs/cgroup", relative);
  const effectiveText = await readOptionalText(path.join(root, "cpuset.cpus.effective"));
  const partitionState = await readOptionalText(path.join(root, "cpuset.cpus.partition"));
  return {
    path: `/${relative}`,
    effective_cpu_ids: effectiveText ? parseCpuList(effectiveText) : [],
    partition_state: partitionState || null,
  };
}

async function observeLinuxSystemPartition(requested) {
  const onlineText = await readOptionalText("/sys/devices/system/cpu/online");
  const commandLine = await readOptionalText("/proc/cmdline");
  const parameters = {};
  for (const name of ["isolcpus", "nohz_full", "irqaffinity", "rcu_nocbs"]) {
    const value = commandLine?.split(/\s+/).find((item) => item.startsWith(`${name}=`));
    parameters[name] = value ? value.slice(name.length + 1) : null;
  }
  const cpuFrequencyPolicy = [];
  const cacheSharedCpuLists = [];
  for (const cpu of requested) {
    cpuFrequencyPolicy.push({
      cpu_id: cpu,
      governor: await readOptionalText(`/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_governor`),
      current_khz: integerOrNull(await readOptionalText(`/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_cur_freq`)),
      maximum_khz: integerOrNull(await readOptionalText(`/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_max_freq`)),
    });
    const indexRoot = `/sys/devices/system/cpu/cpu${cpu}/cache`;
    const indices = await readdir(indexRoot).catch(() => []);
    for (const index of indices.filter((entry) => /^index\d+$/.test(entry)).sort()) {
      const shared = await readOptionalText(path.join(indexRoot, index, "shared_cpu_list"));
      const level = integerOrNull(await readOptionalText(path.join(indexRoot, index, "level")));
      const type = await readOptionalText(path.join(indexRoot, index, "type"));
      if (shared) cacheSharedCpuLists.push({ cpu_id: cpu, cache_index: index, level, type, shared_cpu_ids: parseCpuList(shared) });
    }
  }
  return {
    online_cpu_ids: onlineText ? parseCpuList(onlineText) : [],
    kernel_command_line: commandLine || null,
    kernel_isolation_parameters: parameters,
    cpu_frequency_policy: cpuFrequencyPolicy,
    cache_shared_cpu_lists: cacheSharedCpuLists,
  };
}

export function validateResourcePartitionObservation(observation, request, threadCount) {
  if (observation?.schema !== "deepbom.resource_partition_observation.v1"
    || observation.evidence_class !== "OBSERVED_OS_RESOURCE_PARTITION") {
    throw new Error("resource partition observation schema or evidence class is invalid");
  }
  if (!request || observation.affinity_mode !== request.affinity_mode
    || observation.isolation_expectation !== request.isolation_expectation
    || !sameNumbers(observation.requested_cpu_ids || [], request.requested_cpu_ids || [])) {
    throw new Error("resource partition observation does not match the manifest request");
  }
  const requested = requireCpuIds(request.requested_cpu_ids, "requested_cpu_ids");
  if (!Number.isSafeInteger(threadCount) || threadCount < 1 || threadCount > requested.length) {
    throw new Error("resource partition thread count exceeds requested CPU-set cardinality");
  }
  const samples = observation.affinity_samples;
  if (!Array.isArray(samples) || !samples.length || observation.sample_count !== samples.length) {
    throw new Error("resource partition sample_count must exactly match a non-empty affinity sample ledger");
  }
  let previousTime = 0;
  let maximumThreadCount = 0;
  const finalThreadMasks = new Map();
  const allowedUnion = new Set();
  const processorIds = new Set();
  for (const [sampleIndex, sample] of samples.entries()) {
    const observedAt = sample?.observed_at_unix_ms;
    if (!Number.isSafeInteger(observedAt) || observedAt < 1 || (sampleIndex > 0 && observedAt < previousTime)) {
      throw new Error("resource partition affinity sample timestamps must be positive and non-decreasing");
    }
    previousTime = observedAt;
    if (!Array.isArray(sample.threads) || !sample.threads.length) {
      throw new Error("resource partition affinity samples require observed threads");
    }
    maximumThreadCount = Math.max(maximumThreadCount, sample.threads.length);
    let previousTid = 0;
    for (const [threadIndex, thread] of sample.threads.entries()) {
      if (!Number.isSafeInteger(thread?.tid) || thread.tid < 1 || (threadIndex > 0 && thread.tid <= previousTid)) {
        throw new Error("resource partition sampled thread IDs must be positive, unique, and ascending per sample");
      }
      previousTid = thread.tid;
      const allowed = requireCpuIds(thread.allowed_cpu_ids, "affinity_samples[].threads[].allowed_cpu_ids");
      if (!isSubset(allowed, requested)) {
        throw new Error("resource partition sampled thread mask exceeds the requested CPU set");
      }
      finalThreadMasks.set(thread.tid, allowed);
      for (const cpu of allowed) allowedUnion.add(cpu);
      if (thread.processor != null) {
        if (!Number.isSafeInteger(thread.processor) || !requested.includes(thread.processor)) {
          throw new Error("resource partition sampled processor exceeds the requested CPU set");
        }
        processorIds.add(thread.processor);
      }
    }
  }
  if (maximumThreadCount !== observation.maximum_observed_thread_count || maximumThreadCount < threadCount) {
    throw new Error("resource partition maximum thread count does not reproduce the affinity sample ledger");
  }
  const expectedThreads = [...finalThreadMasks.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([tid, allowed_cpu_ids]) => ({ tid, allowed_cpu_ids }));
  if (stableJson(observation.sampled_threads) !== stableJson(expectedThreads)) {
    throw new Error("resource partition sampled-thread summary does not reproduce the affinity sample ledger");
  }
  const expectedUnion = [...allowedUnion].sort((left, right) => left - right);
  const expectedProcessors = [...processorIds].sort((left, right) => left - right);
  if (!sameNumbers(expectedUnion, requested)
    || !sameNumbers(observation.observed_allowed_cpu_ids_union || [], expectedUnion)
    || !sameNumbers(observation.observed_processor_ids || [], expectedProcessors)) {
    throw new Error("resource partition CPU union or processor summary does not reproduce the affinity sample ledger");
  }
  if (observation.affinity_status !== "observed_all_sampled_threads_within_requested_set") {
    throw new Error("resource partition capture failed closed: sampled-thread affinity was not established");
  }
  if (request.isolation_expectation === "exclusive_cpuset"
    && observation.exclusive_isolation_status !== "observed_cgroup_v2_isolated_partition") {
    throw new Error("resource partition capture failed closed: an exclusive cgroup v2 isolated cpuset was requested but not observed");
  }
  const online = requireCpuIds(observation.online_cpu_ids, "online_cpu_ids");
  if (!isSubset(requested, online)) {
    throw new Error("resource partition requested CPU is absent from the observed online CPU set");
  }
  const effective = Array.isArray(observation.observed_effective_cpu_ids) && observation.observed_effective_cpu_ids.length
    ? requireCpuIds(observation.observed_effective_cpu_ids, "observed_effective_cpu_ids") : [];
  if (observation.exclusive_isolation_status === "observed_cgroup_v2_isolated_partition"
    && (observation.cgroup_v2_partition_state !== "isolated" || !sameNumbers(effective, requested))) {
    throw new Error("resource partition isolated status requires an exact isolated cgroup v2 effective CPU set");
  }
  return true;
}

function enforceResourcePartitionObservation(observation, request, threadCount) {
  validateResourcePartitionObservation(observation, request, threadCount);
}

function resourcePartitionAssignmentSummary(observation, observationSha256) {
  const summary = structuredClone(observation);
  delete summary.affinity_samples;
  summary.observation_sha256 = observationSha256;
  return summary;
}

function requireCpuIds(value, field) {
  if (!Array.isArray(value) || !value.length || value.length > 4096
    || value.some((cpu) => !Number.isSafeInteger(cpu) || cpu < 0)
    || value.some((cpu, index) => index > 0 && cpu <= value[index - 1])) {
    throw new Error(`resource partition ${field} must be a non-empty, unique, ascending CPU-ID array`);
  }
  return value;
}

function parseCpuList(value) {
  const result = new Set();
  for (const token of String(value || "").trim().split(",").filter(Boolean)) {
    const match = token.match(/^(\d+)(?:-(\d+))?$/);
    if (!match) throw new Error(`invalid Linux CPU list '${value}'`);
    const first = Number(match[1]);
    const last = match[2] == null ? first : Number(match[2]);
    if (!Number.isSafeInteger(first) || !Number.isSafeInteger(last) || last < first || last - first > 65536) {
      throw new Error(`invalid Linux CPU range '${token}'`);
    }
    for (let cpu = first; cpu <= last; cpu += 1) result.add(cpu);
  }
  return [...result].sort((a, b) => a - b);
}

function formatCpuList(values) {
  return [...values].sort((a, b) => a - b).join(",");
}

function isSubset(left, right) {
  const allowed = new Set(right);
  return left.length > 0 && left.every((value) => allowed.has(value));
}

function sameNumbers(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function integerOrNull(value) {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) ? parsed : null;
}

async function readOptionalText(filePath) {
  return readFile(filePath, "utf8").then((value) => value.trim()).catch(() => null);
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function runProcess(command, args, cwd, extraEnvironment, label) {
  console.log(`\n> ${command} ${args.join(" ")}`);
  const code = await new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env: { ...process.env, ...extraEnvironment },
      stdio: "inherit",
      shell: false,
    });
    child.on("error", reject);
    child.on("exit", resolve);
  });
  if (code !== 0) throw new Error(`${label} failed with exit code ${code}`);
}

async function gitSourceIdentity() {
  const commit = await captureProcess("git", ["rev-parse", "HEAD"]).catch(() => "not-a-git-checkout");
  if (commit === "not-a-git-checkout") return commit;
  const dirty = await captureProcess("git", ["status", "--porcelain"]);
  return dirty ? `${commit}+working-tree-dirty` : commit;
}

async function captureProcess(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: process.cwd(), shell: false });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("exit", (code) => code === 0 ? resolve(stdout.trim()) : reject(new Error(stderr.trim())));
  });
}

function expand(value, replacements) {
  return String(value).replace(/\$\{([a-z_]+)\}/g, (_, key) => {
    if (!(key in replacements)) throw new Error(`unknown capture placeholder \${${key}}`);
    return replacements[key];
  });
}

function resolveFrom(base, value) {
  return path.resolve(base, value);
}

function requireSha(value, field, length) {
  if (typeof value !== "string" || value.length !== length || !/^[a-f0-9]+$/.test(value)) throw new Error(`${field} must be a lowercase ${length}-character hexadecimal identity`);
}

function sortJson(value) {
  if (Array.isArray(value)) return value.map(sortJson);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map((key) => [key, sortJson(value[key])]));
  return value;
}

function defaultCollectorPath() {
  const executable = process.platform === "win32" ? "deepbom-runtime-collector.exe" : "deepbom-runtime-collector";
  return path.join("native", "runtime_collector", "target", "release", executable);
}
