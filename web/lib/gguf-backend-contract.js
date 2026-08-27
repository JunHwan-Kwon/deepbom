import { GGUF_BACKEND_PROFILES, GGUF_BACKEND_SOURCE, GGUF_LLAMA_ARCHITECTURES } from "./gguf-backend-contract.generated.js";

function sourceEvidence() {
  return {
    ...GGUF_BACKEND_SOURCE,
    files: Object.fromEntries(Object.entries(GGUF_BACKEND_SOURCE.files).map(([key, row]) => [key, { ...row }])),
  };
}

export function buildGgufBackendCompatibility(gguf, tensors) {
  const architecture = typeof gguf?.architecture === "string" && gguf.architecture !== "not_declared" ? gguf.architecture : null;
  const architectureRegistered = architecture ? GGUF_LLAMA_ARCHITECTURES.includes(architecture) : false;
  const tensorRows = Array.isArray(tensors) ? tensors : [];
  const invalidStorageRows = tensorRows.filter((tensor) => tensor.storage_status !== "assessed");
  const encodings = [...new Set(tensorRows.map((tensor) => tensor.dtype).filter(Boolean))].sort();
  const commonStatus = !architecture
    ? "not_assessed_architecture_not_declared"
    : !architectureRegistered
      ? "definite_exclusion_architecture_not_registered"
      : invalidStorageRows.length
        ? "definite_exclusion_invalid_or_unknown_tensor_storage"
        : "source_candidate_build_and_graph_unbound";
  const profiles = GGUF_BACKEND_PROFILES.map((profile) => ({
    ...profile,
    assessment_status: commonStatus,
    architecture_registry_match: architectureRegistered,
    serialized_storage_precheck: invalidStorageRows.length ? "definite_fail" : "passed_known_source_pinned_layouts",
    build_option_binding: "not_bound_to_selected_runtime_binary",
    backend_registration_evidence: "source_registration_candidate_only",
    operator_support_evidence: "not_assessable_without_runtime_constructed_graph_and_backend_device",
    execution_evidence_class: "NOT_OBSERVED",
  }));
  return {
    schema: "deepbom.gguf_backend_compatibility.v1",
    status: commonStatus.startsWith("definite_exclusion") ? "invalid"
      : architectureRegistered && !invalidStorageRows.length ? "source_candidate" : "not_assessed",
    evidence_class: "OBSERVED/SOURCE_PINNED",
    architecture,
    architecture_registry_match: architectureRegistered,
    pinned_architecture_count: GGUF_LLAMA_ARCHITECTURES.length,
    tensor_count: tensorRows.length,
    encoding_count: encodings.length,
    encodings,
    invalid_or_unknown_storage_tensor_count: invalidStorageRows.length,
    invalid_or_unknown_storage_tensors: invalidStorageRows.slice(0, 128).map((tensor) => ({
      tensor_index: tensor.index,
      tensor_name: tensor.name,
      dtype: tensor.dtype,
      storage_status: tensor.storage_status,
    })),
    selected_backend_profile_id: null,
    selected_runtime_build_manifest_status: "not_bound",
    execution_graph_status: "not_serialized_by_gguf",
    compatibility_conclusion: "not_concluded",
    profiles,
    source: sourceEvidence(),
    method: "Match general.architecture against the pinned llama.cpp registry, validate every serialized tensor layout against pinned GGML traits, and enumerate backend CMake option plus compiled registration gates. Do not infer runtime graph operator support, device availability, offload, fallback, or execution.",
    boundary: "A source candidate is not an executable compatibility result. Bind the exact llama.cpp binary/build manifest and an artifact-identified runtime capture to conclude backend availability or execution.",
  };
}

export function ggufBackendSourceForVerification() {
  return GGUF_BACKEND_SOURCE;
}
