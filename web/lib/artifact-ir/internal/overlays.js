import { normalizeArtifactIrRuntimeOverlay } from "../../artifact-ir-runtime.js";
import { clone, compactStrings, integerArray, list, normalizeSha256 } from "./shared.js";

export function buildOverlays(analysis, format, graph, architecture, runtimeEvidence, artifactSha256) {
  const staticRows = [];
  let staticSummary = null;
  const primaryOperators = graph.operators.filter((row) => row.scope_ref === graph.primary_scope_ref);
  if (graph.status === "serialized" && format === "tflite") {
    for (const operator of primaryOperators) {
      const op = list(analysis.ops).find((row) => Number(row?.index) === operator.native_index);
      const eligible = op?.xnnpack_supported === true;
      staticRows.push({
        subject_ref: operator.id,
        state: eligible ? "CONDITIONALLY_DELEGATABLE" : op?.xnnpack_supported === false ? "PREDICTED_FALLBACK_OR_BREAK" : "NOT_ASSESSABLE",
        backend: eligible ? "xnnpack" : "cpu",
        evidence_state: eligible ? "ARTIFACT_ELIGIBLE" : op?.xnnpack_supported === false ? "SOURCE_REGISTERED" : "NOT_ASSESSABLE",
        evidence_class: eligible ? "PREDICTED_SOURCE_AND_ARTIFACT_ELIGIBILITY" : op?.xnnpack_supported === false ? "PREDICTED" : "NOT_ASSESSABLE",
        reason_codes: compactStrings([op?.xnnpack_reason || op?.xnnpack_break_class]),
        unresolved_predicates: compactStrings([op?.xnnpack_build_requirement]),
      });
    }
    staticSummary = { profile_id: "xnnpack", backend: "XNNPACK", original_op_engine_selection_claim: false };
  } else if (graph.status === "serialized" && format === "onnx") {
    const projection = analysis?.tensorrt_static_preflight?.projection;
    if (projection && list(projection.rows).length === primaryOperators.length) {
      const byIndex = new Map(list(projection.rows).map((row) => [Number(row.op_index), row]));
      for (const operator of primaryOperators) {
        const row = byIndex.get(operator.native_index);
        if (!row) throw new Error(`Artifact IR TensorRT overlay is missing operator ${operator.native_index}.`);
        staticRows.push({
          subject_ref: operator.id,
          state: String(row.state || "UNRESOLVED"),
          backend: String(projection.profile_id || "tensorrt"),
          evidence_state: row.state === "CONDITIONALLY_ELIGIBLE" ? "ARTIFACT_ELIGIBLE" : row.state === "DEFINITE_EXCLUSION" ? "BUILD_INCLUDED" : "NOT_ASSESSABLE",
          evidence_class: String(projection.evidence_class || analysis?.tensorrt_static_preflight?.evidence_class || "NOT_ASSESSABLE"),
          reason_codes: compactStrings(row.reason_codes),
          unresolved_predicates: compactStrings(row.unresolved_predicates),
        });
      }
      const engine = analysis?.tensorrt_static_preflight?.engine_inspector_evidence;
      staticSummary = {
        profile_id: String(projection.profile_id || "tensorrt"),
        backend: String(projection.label || projection.profile_id || "TensorRT"),
        projection_schema: String(projection.schema || ""),
        evidence_class: String(projection.evidence_class || "NOT_ASSESSABLE"),
        state_counts: clone(projection.state_counts || {}),
        parser_observation_status: analysis?.tensorrt_static_preflight?.parser_observation?.coverage_status || "not_observed",
        engine_inspector_status: engine?.status || "not_observed",
        engine_sha256: normalizeSha256(engine?.engine?.sha256),
        engine_source_mapping_status: engine?.source_mapping_status || "not_exposed",
        original_op_engine_selection_claim: false,
      };
    }
  }
  const llmScenario = selectLlmPlacementScenario(analysis?.accelerator_profile_binding);
  if (llmScenario) {
    const selected = new Set(integerArray(llmScenario.serialized_layer_offload?.selected_layer_indices));
    for (const node of architecture.nodes) staticRows.push({
      subject_ref: node.id,
      state: selected.has(node.native_index) ? "CONDITIONAL_SERIALIZED_LAYER_RESIDENCY_CANDIDATE" : "CONDITIONAL_OTHER_POOL_RESIDENCY_CANDIDATE",
      backend: selected.has(node.native_index) ? "nvidia_accelerator" : "cpu_or_other_pool",
      evidence_state: "CONFIGURATION_BOUND_STATIC_LOWER_BOUND",
      evidence_class: "DERIVED_CONDITIONAL_STATIC_LOWER_BOUND",
      reason_codes: [], unresolved_predicates: [],
    });
    staticSummary = {
      profile_id: "llm_static_memory_placement",
      backend: "declared accelerator residency scenario",
      source: llmScenario.scenario_source,
      context_length: llmScenario.context_length,
      batch_size: llmScenario.batch_size,
      storage_bits: llmScenario.storage_bits,
      fit_claim: llmScenario.fit_claim,
      original_op_engine_selection_claim: false,
    };
  }
  const staticOverlay = staticRows.length ? [{
    id: "overlay:static-placement:0",
    kind: "static_placement",
    evidence_class: staticSummary?.evidence_class || staticRows[0]?.evidence_class || "DERIVED",
    summary: staticSummary,
    rows: staticRows,
    interpretation_boundary: "Static eligibility or residency candidates do not establish selected-build acceptance, executed assignment, physical transfer, kernel choice, or latency.",
  }] : [];
  const runtime = normalizeArtifactIrRuntimeOverlay(runtimeEvidence, graph, architecture, artifactSha256);
  return { static: staticOverlay, runtime };
}

function selectLlmPlacementScenario(binding) {
  const rows = list(binding?.llm_accelerator_residency?.scenarios);
  const selected = rows.filter((row) => row.scenario_source === "cli_declared");
  return selected.length === 1 && selected[0]?.serialized_layer_offload ? selected[0] : null;
}
