import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const RUNTIME_BACKEND_EVIDENCE_LEDGER_SCHEMA = "deepbom.runtime_backend_evidence_ledger.v1";

const PROVIDERS = Object.freeze([
  Object.freeze({ id: "qnn", label: "QNN", source_profile: "qnn" }),
  Object.freeze({ id: "nnapi", label: "NNAPI", source_profile: "nnapi" }),
  Object.freeze({ id: "coreml", label: "Core ML", source_profile: "coreml" }),
  Object.freeze({ id: "webgpu", label: "WebGPU", source_profile: "webgpu" }),
  Object.freeze({ id: "webnn", label: "WebNN", source_profile: "webnn" }),
]);

export function buildRuntimeBackendEvidenceLedger(runtimeEvidence) {
  const adapter = runtimeEvidence?.source?.adapter || null;
  const isOrt = ["deepbom.ort_profile_adapter.v2.1", "deepbom.ort_profile_adapter.v2.2"].includes(adapter?.schema)
    || runtimeEvidence?.source?.kind === "onnxruntime_profile_json_adapter";
  if (!isOrt) return null;
  const capture = adapter?.native_capture || null;
  const selected = capture?.selected_build_provider_binding || null;
  const inventoryComplete = selected?.provider_inventory_status === "OBSERVED_FROM_ORT_LIST_SUPPORTED_BACKENDS";
  const assignments = Array.isArray(runtimeEvidence.assignments) ? runtimeEvidence.assignments : [];
  const rows = PROVIDERS.map((provider) => buildProviderRow(provider, selected, inventoryComplete, assignments));
  const body = {
    schema: RUNTIME_BACKEND_EVIDENCE_LEDGER_SCHEMA,
    evidence_class: "NORMALIZED_FROM_SEPARATE_SOURCE_BUILD_CAPABILITY_ASSIGNMENT_EXECUTION_LEDGERS",
    artifact_sha256: validSha(runtimeEvidence.artifact_sha256) ? runtimeEvidence.artifact_sha256 : null,
    runtime_binary_sha256: validSha(runtimeEvidence.runtime?.binary_sha256) ? runtimeEvidence.runtime.binary_sha256 : null,
    selected_build_inventory_sha256: validSha(selected?.supported_backends_sha256) ? selected.supported_backends_sha256 : null,
    selected_build_inventory_status: selected?.provider_inventory_status || "not_observed",
    provider_count: rows.length,
    configured_inclusion_assessed_count: rows.filter((row) => row.configured_inclusion.status !== "not_assessed").length,
    capability_acceptance_observed_count: rows.filter((row) => row.capability_acceptance.status === "accepted_by_observed_assignment").length,
    assigned_provider_count: rows.filter((row) => row.assignment.assigned_original_op_count > 0).length,
    executed_provider_count: rows.filter((row) => row.execution.executed_original_op_count > 0).length,
    providers: rows,
    interpretation_boundary: "Configured inclusion is read only from the selected runtime's complete backend inventory. Source eligibility, GetCapability acceptance, original-op assignment, and execution are separate layers. A missing assignment is never converted into rejection, and an observed assignment is not a microkernel or correctness claim.",
  };
  return Object.freeze({ ...body, ledger_sha256: sha256TextHex(canonicalJson(body)) });
}

export function validateRuntimeBackendEvidenceLedger(value, runtimeEvidence) {
  const expected = buildRuntimeBackendEvidenceLedger(runtimeEvidence);
  if (canonicalJson(value) !== canonicalJson(expected)) throw new Error("Runtime backend evidence ledger does not reconstruct from the imported ORT evidence.");
  return true;
}

function buildProviderRow(provider, selected, inventoryComplete, assignments) {
  const binding = (selected?.bindings || []).find((row) => row.source_profile === provider.source_profile || normalizeProvider(row.backend_name) === provider.id) || null;
  const sourceListed = Boolean(binding?.source_profile)
    || (selected?.source_profiles_not_listed_by_selected_build || []).includes(provider.source_profile);
  const providerAssignments = assignments.filter((row) => normalizeProvider(row.provider) === provider.id);
  const executed = providerAssignments.filter((row) => Number.isSafeInteger(Number(row.sample_count)) && Number(row.sample_count) > 0
    && Number.isFinite(Number(row.duration_sum_us ?? row.duration_us)));
  const sampleCount = executed.reduce((sum, row) => sum + Number(row.sample_count || 0), 0);
  const duration = executed.reduce((sum, row) => sum + Number(row.duration_sum_us ?? row.duration_us ?? 0), 0);
  return Object.freeze({
    provider_id: provider.id,
    label: provider.label,
    source_eligibility: {
      status: sourceListed ? "source_profile_present" : selected ? "source_profile_not_loaded" : "not_assessed",
      evidence_class: sourceListed ? "SOURCE_PINNED" : "NOT_ASSESSED",
      source_profile: sourceListed ? provider.source_profile : null,
    },
    configured_inclusion: {
      status: !selected || !inventoryComplete ? "not_assessed" : binding
        ? binding.bundled ? "observed_bundled" : "observed_available_not_bundled"
        : "observed_not_listed",
      evidence_class: !selected || !inventoryComplete ? "NOT_ASSESSED" : "OBSERVED_SELECTED_BUILD_INVENTORY",
      backend_name: binding?.backend_name || null,
      bundled: binding ? Boolean(binding.bundled) : null,
      binding_status: binding?.binding_status || null,
    },
    capability_acceptance: {
      status: providerAssignments.length ? "accepted_by_observed_assignment" : "not_assessed",
      evidence_class: providerAssignments.length ? "DERIVED_FROM_OBSERVED_RUNTIME_ASSIGNMENT" : "NOT_ASSESSED",
      accepted_original_op_count: providerAssignments.length,
      rejection_not_inferred: true,
    },
    assignment: {
      status: providerAssignments.length ? "observed" : "not_observed",
      evidence_class: providerAssignments.length ? "OBSERVED_RUNTIME" : "NOT_OBSERVED",
      assigned_original_op_count: providerAssignments.length,
      provider_names: [...new Set(providerAssignments.map((row) => String(row.provider)))].sort(),
    },
    execution: {
      status: executed.length ? "observed_profile_events" : "not_observed",
      evidence_class: executed.length ? "OBSERVED_RUNTIME" : "NOT_OBSERVED",
      executed_original_op_count: executed.length,
      event_sample_count: sampleCount,
      duration_sum_us: executed.length ? duration : null,
      correctness_status: "not_assessed",
    },
  });
}

function normalizeProvider(value) {
  const text = String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
  if (text.includes("qnn")) return "qnn";
  if (text.includes("nnapi")) return "nnapi";
  if (text.includes("coreml")) return "coreml";
  if (text.includes("webgpu")) return "webgpu";
  if (text.includes("webnn")) return "webnn";
  return "";
}

function validSha(value) { return /^[a-f0-9]{64}$/.test(String(value || "")); }
