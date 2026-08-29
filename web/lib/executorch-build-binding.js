import { EXECUTORCH_PORTABLE_OPERATOR_SIGNATURES } from "./executorch-operator-signatures.generated.js";
import { BoundedFlatBufferReader } from "./flatbuffer-reader.js";
import { canonicalJson } from "./report-utils.js";
import { sha256BytesHex, sha256TextHex } from "./sha256-sync.js";
import { parseStrictJson } from "./strict-json.js";

export const EXECUTORCH_SELECTED_BUILD_ATTESTATION_SCHEMA = "deepbom.executorch_selected_build_attestation.v1";
export const EXECUTORCH_BUILD_BINDING_SCHEMA = "deepbom.executorch_selected_build_binding.v1";
export const EXECUTORCH_SELECTED_BUILD_INPUT_SCHEMA = "deepbom.executorch_selected_build_input.v1";

const SELECTED_BUILD_FILENAME = "deepbom.executorch-build.json";
const MAX_SELECTED_BUILD_BYTES = 16 * 1024 * 1024;

const SOURCE = Object.freeze({
  repository: "pytorch/executorch",
  release: "v1.4.1",
  commit: "e4d02f41f7909e8ed5bf4a14ffc520d733453d9f",
});
const FILES = Object.freeze({
  cmake_options: sourceFile("tools/cmake/preset/default.cmake", "d718e91fea803271f3febeb000bbbc1bba6c0305f4f6852db9bedf93a74c1c9b"),
  schema_version: sourceFile("exir/version.py", "d1853272c0ed0cf026ecec49f2ad6932d924cbca7b03a46d2ed16e73227a2047"),
  runtime_loader: sourceFile("runtime/executor/program.cpp", "d38be8eeec0fac0cea8f25d61820bc6f6d2bac4f07a89f3cb9ce175649260ca9"),
});

export const EXECUTORCH_BACKEND_SOURCE_REGISTRY = Object.freeze({
  XnnpackBackend: backend({
    buildOption: "EXECUTORCH_BUILD_XNNPACK",
    registration: sourceFile("backends/xnnpack/runtime/XNNPACKBackend.h", "f7455dd1d41cce5c9522e38ec90474230e1b14bb969a1604256f6bb73c42210b"),
    blobKind: "PUBLIC_FLATBUFFER_SCHEMA",
    blobSources: [
      sourceFile("backends/xnnpack/serialization/schema.fbs", "a4dca505a91b7c9ff690d0f58d5f2a49dc044a44d9261743fc32331eb644102d"),
      sourceFile("backends/xnnpack/serialization/runtime_schema.fbs", "8e2d90ec60a8e785befe762b48cee8e06276791c0c499e891ba946e01d0beeac"),
    ],
    rootType: "XNNGraph",
    boundary: "The public FlatBuffer schema supports structural decoding. Runtime XNNPACK compilation, microkernel selection, initialization, and execution remain runtime evidence.",
  }),
  VulkanBackend: backend({
    buildOption: "EXECUTORCH_BUILD_VULKAN",
    registration: sourceFile("backends/vulkan/runtime/VulkanBackend.cpp", "75a656caf4408e216d527b81f539632b65205d0ffc081e0d4f11526f0569f0f3"),
    blobKind: "PUBLIC_FLATBUFFER_SCHEMA",
    blobSources: [sourceFile("backends/vulkan/serialization/schema.fbs", "3c563312bd299e5fcc6e7930763d6d8d363e012a519856ee7dad023d6ca9ee84")],
    rootType: "VkGraph",
    boundary: "The serialized VkGraph is source-described. Shader registration, device capability, pipeline creation, and dispatch remain runtime evidence.",
  }),
  MPSBackend: backend({
    buildOption: "EXECUTORCH_BUILD_MPS",
    registration: sourceFile("backends/apple/mps/runtime/MPSBackend.mm", "8f3091007b6bff9dd57454c8277b7a0b685e1dc225dac13862a79b976db5f2e7"),
    blobKind: "PUBLIC_FLATBUFFER_SCHEMA",
    blobSources: [sourceFile("backends/apple/mps/serialization/schema.fbs", "0503ab159bf6eb1b9cef40acadb42e6528064fdcb01a8100e1f4c56f09eb6f58")],
    rootType: "MPSGraph",
    lifecycle: "DEPRECATED_AT_PINNED_SOURCE",
    boundary: "The MPSGraph payload is source-described, but the pinned backend is deprecated. MPSGraph compilation and execution remain runtime evidence.",
  }),
  CoreMLBackend: backend({
    buildOption: "EXECUTORCH_BUILD_COREML",
    registration: sourceFile("backends/apple/coreml/runtime/delegate/ETCoreMLStrings.mm", "b777560b6a03ed1c37795edb3157de81120084247cedf954d193cea58c06c552"),
    blobKind: "SOURCE_DEFINED_PACKAGE_ARCHIVE",
    blobSources: [sourceFile("backends/apple/coreml/compiler/coreml_preprocess.py", "37250b7b99527954144dfc7998c0917118d5a4bf38a70f768abfae0b5324fc8f")],
    boundary: "The compiler source defines the packaged Core ML payload. This ET12 adapter does not reinterpret the nested Core ML package as an ExecuTorch FlatBuffer.",
  }),
  QnnBackend: backend({
    buildOption: "EXECUTORCH_BUILD_QNN",
    registration: sourceFile("backends/qualcomm/runtime/QnnExecuTorch.h", "6cea1a7bcc4c76bdfa76a5041accb415d0fd9a8b4c6627841c3035586985c262"),
    blobKind: "VENDOR_CONTEXT_OR_DLC_OPAQUE",
    blobSources: [sourceFile("backends/qualcomm/serialization/qc_compiler_spec.fbs", "7b8722caea5b5d4ea13ce4f784c376eb01a229b9e93aa45bc2bb5fa9d94105ea")],
    boundary: "The public FlatBuffer covers QNN compiler options, not the vendor context/DLC payload. Delegate internals remain opaque without matching QNN tooling.",
  }),
  CudaBackend: backend({
    buildOption: "EXECUTORCH_BUILD_CUDA",
    registration: sourceFile("backends/cuda/runtime/cuda_backend.cpp", "3a43d926613164a7bc8ac3d5bbd137b9b530d50363e2cedcf9aff559fb79a544"),
    blobKind: "AOTI_NAMED_DATA_SHARED_LIBRARY",
    blobSources: [sourceFile("backends/cuda/runtime/cuda_backend.cpp", "3a43d926613164a7bc8ac3d5bbd137b9b530d50363e2cedcf9aff559fb79a544")],
    lifecycle: "EXPERIMENTAL_AT_PINNED_SOURCE",
    boundary: "The processed payload resolves AOTI named-data keys; executable code and weights are separate named data. Static presence does not establish loadability or CUDA execution.",
  }),
  MetalBackend: backend({
    buildOption: "EXECUTORCH_BUILD_METAL",
    registration: sourceFile("backends/apple/metal/runtime/metal_backend.cpp", "69da65119daf5000b51cec961224ea048b968e42584cf4d89555f4d76f0f5c33"),
    blobKind: "AOTI_NAMED_DATA_SHARED_LIBRARY",
    blobSources: [sourceFile("backends/apple/metal/runtime/metal_backend.cpp", "69da65119daf5000b51cec961224ea048b968e42584cf4d89555f4d76f0f5c33")],
    lifecycle: "EXPERIMENTAL_AT_PINNED_SOURCE",
    boundary: "The processed payload resolves AOTI named-data keys; compiled Metal code and weights are separate named data. Static presence does not establish loadability or Metal execution.",
  }),
});

const REQUIRED_BUILD_OPTIONS = Object.freeze([
  "EXECUTORCH_BUILD_COREML",
  "EXECUTORCH_BUILD_CUDA",
  "EXECUTORCH_BUILD_METAL",
  "EXECUTORCH_BUILD_MPS",
  "EXECUTORCH_BUILD_PORTABLE_OPS",
  "EXECUTORCH_BUILD_QNN",
  "EXECUTORCH_BUILD_VULKAN",
  "EXECUTORCH_BUILD_XNNPACK",
]);
const SHA256 = /^[a-f0-9]{64}$/;
const SAFE_PATH = /^(?![A-Za-z]:)(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$))[A-Za-z0-9._+@()\/-]+$/;

export const EXECUTORCH_BACKEND_REGISTRY_SOURCE = Object.freeze({
  schema: "deepbom.executorch_backend_source_registry.v1",
  ...SOURCE,
  files: FILES,
  backend_count: Object.keys(EXECUTORCH_BACKEND_SOURCE_REGISTRY).length,
  registry_sha256: sha256TextHex(canonicalJson(EXECUTORCH_BACKEND_SOURCE_REGISTRY)),
  interpretation_boundary: "A registry match establishes that the backend ID and processed-data form are present in the pinned source. It does not establish that a selected binary links the backend or that initialization and execution succeed.",
});

export function validateExecuTorchSelectedBuildAttestation(value) {
  if (!value || value.schema !== EXECUTORCH_SELECTED_BUILD_ATTESTATION_SCHEMA
    || value.evidence_class !== "REPRODUCIBLE_SELECTED_BUILD_ATTESTATION") {
    throw new Error(`ExecuTorch selected-build attestation must use ${EXECUTORCH_SELECTED_BUILD_ATTESTATION_SCHEMA}.`);
  }
  const source = {
    repository: requiredText(value.source?.repository, "ExecuTorch repository"),
    release: requiredText(value.source?.release, "ExecuTorch release"),
    commit: requiredCommit(value.source?.commit, "ExecuTorch commit"),
    pristine_before_build: value.source?.pristine_before_build === true,
    submodule_status_sha256: requiredSha(value.source?.submodule_status_sha256, "ExecuTorch submodule status SHA-256"),
    post_build_diff_sha256: requiredSha(value.source?.post_build_diff_sha256, "ExecuTorch post-build diff SHA-256"),
    backend_registry_sha256: requiredSha(value.source?.backend_registry_sha256, "ExecuTorch backend registry SHA-256"),
  };
  if (source.repository !== SOURCE.repository || source.release !== SOURCE.release || source.commit !== SOURCE.commit
    || !source.pristine_before_build || source.backend_registry_sha256 !== EXECUTORCH_BACKEND_REGISTRY_SOURCE.registry_sha256) {
    throw new Error("ExecuTorch attestation does not bind a pristine checkout and the pinned backend source registry.");
  }
  const cmakeOptions = validateBuildOptions(value.build?.cmake_options);
  const linkedBackendIds = sortedUniqueStrings(value.build?.linked_backend_ids, "linked backend IDs", 256);
  const customBackendSources = validateCustomBackendSources(value.build?.custom_backend_sources || []);
  const customBackendIds = new Set(customBackendSources.map((row) => row.backend_id));
  for (const backendId of linkedBackendIds) {
    const sourceEntry = EXECUTORCH_BACKEND_SOURCE_REGISTRY[backendId];
    if (sourceEntry && cmakeOptions[sourceEntry.build_option] !== true) {
      throw new Error(`ExecuTorch linked backend ${backendId} contradicts ${sourceEntry.build_option}=false.`);
    }
    if (!sourceEntry && !customBackendIds.has(backendId)) {
      throw new Error(`ExecuTorch linked backend ${backendId} has no pinned or custom source identity.`);
    }
  }
  if (customBackendSources.some((row) => !linkedBackendIds.includes(row.backend_id))) {
    throw new Error("ExecuTorch custom backend source does not correspond to a linked backend ID.");
  }
  const portableOperatorNames = sortedUniqueStrings(value.build?.portable_operator_names, "portable operator names", 100_000);
  if (portableOperatorNames.length && !cmakeOptions.EXECUTORCH_BUILD_PORTABLE_OPS) {
    throw new Error("ExecuTorch portable operators contradict EXECUTORCH_BUILD_PORTABLE_OPS=false.");
  }
  for (const name of portableOperatorNames) {
    if (!Object.hasOwn(EXECUTORCH_PORTABLE_OPERATOR_SIGNATURES, name)) {
      throw new Error(`ExecuTorch portable operator ${name} is outside the pinned operator signature registry.`);
    }
  }
  const customOperatorNames = sortedUniqueStrings(value.build?.custom_operator_names || [], "custom operator names", 100_000);
  if (customOperatorNames.some((name) => portableOperatorNames.includes(name))) {
    throw new Error("ExecuTorch operator cannot be listed as both portable and custom.");
  }
  const binaryInventory = validateBinaryInventory(value.runtime?.binary_inventory);
  const binaryInventorySha256 = sha256TextHex(canonicalJson(binaryInventory));
  if (requiredSha(value.runtime?.binary_inventory_sha256, "ExecuTorch binary inventory SHA-256") !== binaryInventorySha256) {
    throw new Error("ExecuTorch binary inventory SHA-256 does not reconstruct.");
  }
  const primaryBinaryPath = safePath(value.runtime?.primary_binary_path, "ExecuTorch primary binary path");
  const primaryBinarySha256 = requiredSha(value.runtime?.primary_binary_sha256, "ExecuTorch primary binary SHA-256");
  if (!binaryInventory.some((row) => row.path === primaryBinaryPath && row.sha256 === primaryBinarySha256)) {
    throw new Error("ExecuTorch primary binary does not bind to the binary inventory.");
  }
  const normalized = {
    schema: EXECUTORCH_SELECTED_BUILD_ATTESTATION_SCHEMA,
    evidence_class: "REPRODUCIBLE_SELECTED_BUILD_ATTESTATION",
    source,
    build: {
      configuration: requiredText(value.build?.configuration, "ExecuTorch build configuration"),
      cmake_options: cmakeOptions,
      linked_backend_ids: linkedBackendIds,
      custom_backend_sources: customBackendSources,
      portable_operator_names: portableOperatorNames,
      custom_operator_names: customOperatorNames,
      cmake_cache_sha256: requiredSha(value.build?.cmake_cache_sha256, "ExecuTorch CMake cache SHA-256"),
      build_stdout_sha256: requiredSha(value.build?.build_stdout_sha256, "ExecuTorch build stdout SHA-256"),
      build_stderr_sha256: requiredSha(value.build?.build_stderr_sha256, "ExecuTorch build stderr SHA-256"),
    },
    runtime: {
      platform: requiredText(value.runtime?.platform, "ExecuTorch runtime platform"),
      arch: requiredText(value.runtime?.arch, "ExecuTorch runtime architecture"),
      binary_inventory: binaryInventory,
      binary_inventory_sha256: binaryInventorySha256,
      primary_binary_path: primaryBinaryPath,
      primary_binary_sha256: primaryBinarySha256,
    },
    boundary: optionalText(value.boundary, 8192),
  };
  const attestationSha256 = sha256TextHex(canonicalJson(normalized));
  if (requiredSha(value.attestation_sha256, "ExecuTorch attestation SHA-256") !== attestationSha256) {
    throw new Error("ExecuTorch selected-build attestation SHA-256 does not reconstruct.");
  }
  return { ...normalized, attestation_sha256: attestationSha256 };
}

export function resolveExecuTorchSelectedBuildAttestation(files) {
  const candidates = (files || []).filter((file) => String(file?.path || file?.name || "").replaceAll("\\", "/").split("/").at(-1)?.toLowerCase() === SELECTED_BUILD_FILENAME);
  if (candidates.length > 1) throw new Error(`ExecuTorch sidecars contain more than one ${SELECTED_BUILD_FILENAME}.`);
  if (!candidates.length) return null;
  const file = candidates[0];
  const bytes = file?.bytes instanceof Uint8Array ? file.bytes : null;
  if (!bytes || bytes.byteLength < 1 || bytes.byteLength > MAX_SELECTED_BUILD_BYTES) {
    throw new Error(`ExecuTorch ${SELECTED_BUILD_FILENAME} is empty or exceeds ${MAX_SELECTED_BUILD_BYTES} bytes.`);
  }
  const path = safePath(file.path || file.name, "ExecuTorch selected-build sidecar path");
  const fileSha256 = sha256BytesHex(bytes);
  if (file.sha256 != null && String(file.sha256).toLowerCase() !== fileSha256) {
    throw new Error(`ExecuTorch ${SELECTED_BUILD_FILENAME} SHA-256 does not match the selected bytes.`);
  }
  let text;
  try { text = new TextDecoder("utf-8", { fatal: true }).decode(bytes); }
  catch { throw new Error(`ExecuTorch ${SELECTED_BUILD_FILENAME} is not valid UTF-8.`); }
  const attestation = validateExecuTorchSelectedBuildAttestation(parseStrictJson(text, `ExecuTorch ${SELECTED_BUILD_FILENAME}`));
  return {
    attestation,
    input: {
      schema: EXECUTORCH_SELECTED_BUILD_INPUT_SCHEMA,
      path,
      byte_length: bytes.byteLength,
      file_sha256: fileSha256,
      attestation_sha256: attestation.attestation_sha256,
      duplicate_key_validation: "complete",
    },
  };
}

export function buildExecuTorchSelectedBuildBinding(delegates, ops, selectedBuildValue = null, selectedBuildInputValue = null) {
  const attestation = selectedBuildValue == null ? null : validateExecuTorchSelectedBuildAttestation(selectedBuildValue);
  const selectedBuildInput = attestation == null ? null : validateSelectedBuildInput(selectedBuildInputValue, attestation);
  const linked = new Set(attestation?.build.linked_backend_ids || []);
  const portable = new Set(attestation?.build.portable_operator_names || []);
  const custom = new Set(attestation?.build.custom_operator_names || []);
  const delegateBindings = (delegates || []).map((row) => {
    const source = EXECUTORCH_BACKEND_SOURCE_REGISTRY[row.backend_id] || null;
    return {
      plan_index: row.plan_index,
      delegate_index: row.index,
      backend_id: row.backend_id,
      serialized_assignment_status: "OBSERVED_SERIALIZED_DELEGATE_CALL_TARGET",
      source_status: source ? "SOURCE_REGISTERED_AT_PINNED_COMMIT" : "SOURCE_NOT_IN_PINNED_STANDARD_REGISTRY",
      source_profile: source,
      selected_build_status: !attestation ? "SELECTED_BUILD_NOT_BOUND"
        : linked.has(row.backend_id) ? "ATTESTED_LINKED_IN_SELECTED_BUILD" : "ATTESTED_NOT_LINKED_IN_SELECTED_BUILD",
    };
  });
  const kernelBindings = (ops || []).filter((row) => row.instruction_kind === "KernelCall").map((row) => ({
    op_index: row.index,
    operator_name: row.name,
    serialized_call_status: "OBSERVED_SERIALIZED_KERNEL_CALL",
    source_signature_status: row.signature_status || "not_bound_operator_outside_pinned_portable_registry",
    selected_build_status: !attestation ? "SELECTED_BUILD_NOT_BOUND"
      : portable.has(row.name) ? "ATTESTED_PORTABLE_OPERATOR_INCLUDED"
        : custom.has(row.name) ? "ATTESTED_CUSTOM_OPERATOR_INCLUDED" : "ATTESTED_OPERATOR_NOT_LISTED",
  }));
  const delegateContradictions = delegateBindings.filter((row) => row.selected_build_status === "ATTESTED_NOT_LINKED_IN_SELECTED_BUILD").length;
  const kernelContradictions = kernelBindings.filter((row) => row.selected_build_status === "ATTESTED_OPERATOR_NOT_LISTED").length;
  return {
    schema: EXECUTORCH_BUILD_BINDING_SCHEMA,
    evidence_class: attestation ? "OBSERVED_ARTIFACT_PLUS_ATTESTED_SELECTED_BUILD" : "OBSERVED_ARTIFACT_PLUS_PINNED_SOURCE",
    source_registry: EXECUTORCH_BACKEND_REGISTRY_SOURCE,
    selected_build: attestation,
    selected_build_input: selectedBuildInput,
    delegate_bindings: delegateBindings,
    kernel_bindings: kernelBindings,
    delegate_contradiction_count: delegateContradictions,
    kernel_contradiction_count: kernelContradictions,
    status: !attestation ? "SOURCE_ONLY_SELECTED_BUILD_UNBOUND"
      : delegateContradictions || kernelContradictions ? "CONTRADICTION_SELECTED_BUILD_CANNOT_SATISFY_SERIALIZED_PROGRAM"
        : "SELECTED_BUILD_INVENTORY_SATISFIES_SERIALIZED_IDENTITIES",
    interpretation_boundary: attestation
      ? "The attestation binds source, declared build inputs, linked backend IDs, selected operator names, and runtime binary digests. It does not prove dead-strip behavior, backend initialization, delegate-internal validity, executed assignment, correctness, or latency."
      : "Pinned source establishes known backend and operator identities only. No selected runtime binary or build inventory is bound.",
  };
}

export function execuTorchBackendSource(backendId) {
  return EXECUTORCH_BACKEND_SOURCE_REGISTRY[String(backendId || "")] || null;
}

export function assessExecuTorchProcessedPayload(backendId, payload) {
  if (!(payload instanceof Uint8Array)) throw new TypeError("ExecuTorch processed backend payload must be a Uint8Array.");
  const source = execuTorchBackendSource(backendId);
  const base = {
    schema: "deepbom.executorch_processed_backend_payload.v1",
    evidence_class: "OBSERVED_SERIALIZED_PAYLOAD_PLUS_PINNED_SOURCE",
    backend_id: String(backendId || ""),
    byte_length: payload.byteLength,
    sha256: sha256BytesHex(payload),
    source_status: source ? "SOURCE_REGISTERED_AT_PINNED_COMMIT" : "SOURCE_NOT_IN_PINNED_STANDARD_REGISTRY",
    processed_blob_kind: source?.processed_blob?.kind || "UNKNOWN",
    root_type: source?.processed_blob?.root_type || null,
    source_files: source?.processed_blob?.source_files || [],
    interpretation_boundary: source?.interpretation_boundary
      || "The backend ID is not present in the pinned standard registry. The payload is preserved by exact bytes and SHA-256 only.",
  };
  if (!source) return { ...base, structural_status: "NOT_ASSESSED_BACKEND_SOURCE_UNBOUND", structural_error: null };
  if (source.processed_blob.kind !== "PUBLIC_FLATBUFFER_SCHEMA") {
    return { ...base, structural_status: "NOT_ASSESSED_SOURCE_BOUND_NON_FLATBUFFER_PAYLOAD", structural_error: null };
  }
  try {
    validateBareFlatBufferRoot(payload);
    return { ...base, structural_status: "OBSERVED_BOUNDED_FLATBUFFER_ROOT_ENVELOPE", structural_error: null };
  } catch (error) {
    return {
      ...base,
      structural_status: "CONTRADICTION_SOURCE_DECLARED_FLATBUFFER_ENVELOPE_INVALID",
      structural_error: String(error?.message || error),
    };
  }
}

function backend({ buildOption, registration, blobKind, blobSources, rootType = null, lifecycle = "SUPPORTED_AT_PINNED_SOURCE", boundary }) {
  return Object.freeze({
    build_option: buildOption,
    registration_source: registration,
    processed_blob: Object.freeze({ kind: blobKind, source_files: Object.freeze(blobSources), root_type: rootType }),
    lifecycle,
    interpretation_boundary: boundary,
  });
}
function sourceFile(path, sha256) { return Object.freeze({ path, sha256 }); }

function validateBuildOptions(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("ExecuTorch CMake option inventory is required.");
  const keys = Object.keys(value).sort();
  if (canonicalJson(keys) !== canonicalJson(REQUIRED_BUILD_OPTIONS)) {
    throw new Error("ExecuTorch CMake option inventory must contain the exact source-bound backend and portable-op options.");
  }
  return Object.fromEntries(keys.map((key) => {
    if (typeof value[key] !== "boolean") throw new Error(`ExecuTorch CMake option ${key} must be boolean.`);
    return [key, value[key]];
  }));
}
function validateCustomBackendSources(value) {
  if (!Array.isArray(value) || value.length > 256) throw new Error("ExecuTorch custom backend source inventory is invalid.");
  const rows = value.map((row) => ({
    backend_id: requiredText(row?.backend_id, "custom backend ID"),
    repository: requiredText(row?.repository, "custom backend repository"),
    commit: requiredCommit(row?.commit, "custom backend commit"),
    path: safePath(row?.path, "custom backend source path"),
    sha256: requiredSha(row?.sha256, "custom backend source SHA-256"),
  })).sort((left, right) => left.backend_id.localeCompare(right.backend_id));
  if (new Set(rows.map((row) => row.backend_id)).size !== rows.length
    || rows.some((row) => Object.hasOwn(EXECUTORCH_BACKEND_SOURCE_REGISTRY, row.backend_id))) {
    throw new Error("ExecuTorch custom backend sources must be unique and must not shadow the pinned standard registry.");
  }
  return rows;
}
function validateBinaryInventory(value) {
  if (!Array.isArray(value) || !value.length || value.length > 1024) throw new Error("ExecuTorch binary inventory is empty or oversized.");
  const rows = value.map((row) => ({
    path: safePath(row?.path, "ExecuTorch binary path"),
    byte_length: positiveInteger(row?.byte_length, "ExecuTorch binary byte length"),
    sha256: requiredSha(row?.sha256, "ExecuTorch binary SHA-256"),
  })).sort((left, right) => left.path.localeCompare(right.path));
  if (new Set(rows.map((row) => row.path)).size !== rows.length) throw new Error("ExecuTorch binary inventory contains duplicate paths.");
  return rows;
}
function validateSelectedBuildInput(value, attestation) {
  if (value == null) return null;
  const normalized = {
    schema: String(value.schema || ""),
    path: safePath(value.path, "ExecuTorch selected-build sidecar path"),
    byte_length: positiveInteger(value.byte_length, "ExecuTorch selected-build sidecar byte length"),
    file_sha256: requiredSha(value.file_sha256, "ExecuTorch selected-build sidecar SHA-256"),
    attestation_sha256: requiredSha(value.attestation_sha256, "ExecuTorch selected-build semantic SHA-256"),
    duplicate_key_validation: String(value.duplicate_key_validation || ""),
  };
  if (normalized.schema !== EXECUTORCH_SELECTED_BUILD_INPUT_SCHEMA
    || normalized.duplicate_key_validation !== "complete"
    || normalized.attestation_sha256 !== attestation.attestation_sha256) {
    throw new Error("ExecuTorch selected-build sidecar identity does not bind the validated attestation.");
  }
  return normalized;
}
function sortedUniqueStrings(value, label, limit) {
  if (!Array.isArray(value) || value.length > limit) throw new Error(`ExecuTorch ${label} inventory is missing or oversized.`);
  const rows = value.map((item) => requiredText(item, `ExecuTorch ${label}`)).sort((left, right) => left.localeCompare(right));
  if (new Set(rows).size !== rows.length) throw new Error(`ExecuTorch ${label} must be unique.`);
  return rows;
}
function requiredSha(value, label) { const text = String(value || "").toLowerCase(); if (!SHA256.test(text)) throw new Error(`${label} is required.`); return text; }
function requiredCommit(value, label) { const text = String(value || "").toLowerCase(); if (!/^[a-f0-9]{40,64}$/.test(text)) throw new Error(`${label} is required.`); return text; }
function requiredText(value, label) { const text = String(value || "").trim(); if (!text || text.length > 8192) throw new Error(`${label} is required or oversized.`); return text; }
function optionalText(value, limit) { if (value == null || value === "") return null; const text = String(value); if (text.length > limit) throw new Error("ExecuTorch attestation boundary is oversized."); return text; }
function safePath(value, label) { const text = String(value || "").replaceAll("\\", "/"); if (!SAFE_PATH.test(text)) throw new Error(`${label} is unsafe.`); return text; }
function positiveInteger(value, label) { const number = Number(value); if (!Number.isSafeInteger(number) || number < 1) throw new Error(`${label} must be a positive integer.`); return number; }

function validateBareFlatBufferRoot(payload) {
  if (payload.byteLength < 8) throw new Error("processed payload is shorter than a bounded FlatBuffer root envelope");
  const rootOffset = new DataView(payload.buffer, payload.byteOffset, payload.byteLength).getUint32(0, true);
  if (rootOffset < 4 || rootOffset > payload.byteLength - 4) throw new Error(`processed payload root offset ${rootOffset} is outside the payload`);
  const reader = new BoundedFlatBufferReader(payload, { maxVectorElements: 2_000_000, maxStringBytes: 16 * 1024 * 1024 });
  reader.table(rootOffset, "processed backend root table");
}
