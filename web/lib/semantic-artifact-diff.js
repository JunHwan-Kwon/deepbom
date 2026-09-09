import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const SEMANTIC_ARTIFACT_DIFF_SCHEMA = "deepbom.semantic_artifact_diff.v1";

export function buildSemanticArtifactDiff(baselineIr, candidateIr, { tfliteDeploymentDelta = null } = {}) {
  assertArtifactIr(baselineIr, "baseline");
  assertArtifactIr(candidateIr, "candidate");
  if (baselineIr.artifact.format !== candidateIr.artifact.format) {
    throw new Error(`Semantic diff requires matching artifact formats, received ${baselineIr.artifact.format} and ${candidateIr.artifact.format}.`);
  }
  const graph = graphDelta(baselineIr.graph, candidateIr.graph);
  const storage = storageDelta(baselineIr.storage_topology, candidateIr.storage_topology);
  const quantization = quantizationDelta(baselineIr.quantization_contracts, candidateIr.quantization_contracts);
  const artifactBytesChanged = baselineIr.artifact.sha256 !== candidateIr.artifact.sha256;
  const changeImpact = classifyImpact({ graph, storage, quantization, artifactBytesChanged });
  const body = {
    schema: SEMANTIC_ARTIFACT_DIFF_SCHEMA,
    format: baselineIr.artifact.format,
    baseline: artifactIdentity(baselineIr),
    candidate: artifactIdentity(candidateIr),
    graph_delta: graph,
    storage_delta: storage,
    quantization_delta: quantization,
    change_impact: changeImpact,
    ...(tfliteDeploymentDelta ? { tflite_deployment_delta: tfliteDeploymentDelta } : {}),
    interpretation_boundary: "This document compares canonical serialized-artifact evidence. Matching coordinates do not prove training lineage or semantic layer identity. Static MAC, storage, and target-cost changes are not runtime measurements; task and regulatory significance require separately bound evidence.",
  };
  return { ...body, semantic_diff_sha256: sha256TextHex(canonicalJson(body)) };
}

export function validateSemanticArtifactDiff(document) {
  if (document?.schema !== SEMANTIC_ARTIFACT_DIFF_SCHEMA) throw new Error("Semantic diff schema is invalid.");
  if (!/^[a-f0-9]{64}$/.test(String(document?.baseline?.sha256 || ""))
      || !/^[a-f0-9]{64}$/.test(String(document?.candidate?.sha256 || ""))) {
    throw new Error("Semantic diff artifact identity is invalid.");
  }
  if (document.baseline.format !== document.format || document.candidate.format !== document.format) {
    throw new Error("Semantic diff format binding is inconsistent.");
  }
  assertAlignmentCounts(document.graph_delta?.operator_alignment, document.graph_delta, "operator");
  assertAlignmentCounts(document.storage_delta?.object_alignment, document.storage_delta, "object");
  assertAlignmentCounts(document.quantization_delta?.record_alignment, document.quantization_delta, "record");
  const { semantic_diff_sha256: observed, ...body } = document;
  const expected = sha256TextHex(canonicalJson(body));
  if (observed !== expected) throw new Error("Semantic diff SHA-256 is invalid.");
  return document;
}

function graphDelta(baseline, candidate) {
  const leftOperators = keyed(baseline?.operators, (row) => row.id);
  const rightOperators = keyed(candidate?.operators, (row) => row.id);
  const operatorRows = compareRows(leftOperators, rightOperators, operatorContract);
  const leftInputs = (baseline?.inputs || []).map((id) => valueContract((baseline?.values || []).find((row) => row.id === id)));
  const rightInputs = (candidate?.inputs || []).map((id) => valueContract((candidate?.values || []).find((row) => row.id === id)));
  const leftOutputs = (baseline?.outputs || []).map((id) => valueContract((baseline?.values || []).find((row) => row.id === id)));
  const rightOutputs = (candidate?.outputs || []).map((id) => valueContract((candidate?.values || []).find((row) => row.id === id)));
  const inputChanged = canonicalJson(leftInputs) !== canonicalJson(rightInputs);
  const outputChanged = canonicalJson(leftOutputs) !== canonicalJson(rightOutputs);
  const leftMacs = baseline?.totals?.macs?.decimal ?? null;
  const rightMacs = candidate?.totals?.macs?.decimal ?? null;
  return {
    executable_graph_status: transition(baseline?.executable_graph_status, candidate?.executable_graph_status),
    input_contract_changed: inputChanged,
    output_contract_changed: outputChanged,
    baseline_inputs: leftInputs,
    candidate_inputs: rightInputs,
    baseline_outputs: leftOutputs,
    candidate_outputs: rightOutputs,
    operator_alignment: operatorRows,
    matched_operator_count: operatorRows.filter((row) => row.status === "unchanged" || row.status === "changed").length,
    changed_operator_count: operatorRows.filter((row) => row.status === "changed").length,
    added_operator_count: operatorRows.filter((row) => row.status === "added").length,
    removed_operator_count: operatorRows.filter((row) => row.status === "removed").length,
    total_macs: { baseline_decimal: leftMacs, candidate_decimal: rightMacs, changed: leftMacs !== rightMacs },
    topology_changed: inputChanged || outputChanged || operatorRows.some((row) => row.status !== "unchanged"),
  };
}

function storageDelta(baseline, candidate) {
  const rows = compareRows(keyed(baseline?.objects, (row) => row.id), keyed(candidate?.objects, (row) => row.id), storageContract);
  const payloadChanges = rows.filter((row) => row.status === "changed" && row.changed_fields.includes("payload_sha256"));
  const layoutChanges = rows.filter((row) => row.status !== "unchanged"
    && (row.status !== "changed" || row.changed_fields.some((field) => field !== "payload_sha256")));
  return {
    object_alignment: rows,
    matched_object_count: rows.filter((row) => row.status === "unchanged" || row.status === "changed").length,
    changed_object_count: rows.filter((row) => row.status === "changed").length,
    added_object_count: rows.filter((row) => row.status === "added").length,
    removed_object_count: rows.filter((row) => row.status === "removed").length,
    payload_digest_change_count: payloadChanges.length,
    layout_change_count: layoutChanges.length,
    serialized_bytes: {
      baseline_decimal: baseline?.totals?.serialized_object_bytes_sum?.decimal ?? null,
      candidate_decimal: candidate?.totals?.serialized_object_bytes_sum?.decimal ?? null,
      changed: baseline?.totals?.serialized_object_bytes_sum?.decimal !== candidate?.totals?.serialized_object_bytes_sum?.decimal,
    },
  };
}

function quantizationDelta(baseline, candidate) {
  const rows = compareRows(keyed(baseline?.records, quantizationKey), keyed(candidate?.records, quantizationKey), quantizationContract)
    .map((row) => row.status === "changed" ? { ...row, scale_ratio: scaleRatio(row.baseline?.parameters?.scale, row.candidate?.parameters?.scale) } : row);
  return {
    status: transition(baseline?.status, candidate?.status),
    record_alignment: rows,
    matched_record_count: rows.filter((row) => row.status === "unchanged" || row.status === "changed").length,
    changed_record_count: rows.filter((row) => row.status === "changed").length,
    added_record_count: rows.filter((row) => row.status === "added").length,
    removed_record_count: rows.filter((row) => row.status === "removed").length,
    contract_changed: rows.some((row) => row.status !== "unchanged"),
  };
}

function compareRows(left, right, project) {
  return [...new Set([...left.keys(), ...right.keys()])].sort().map((key) => {
    const baseline = left.get(key);
    const candidate = right.get(key);
    if (!baseline) return { subject_key: key, status: "added", baseline: null, candidate: project(candidate), changed_fields: [] };
    if (!candidate) return { subject_key: key, status: "removed", baseline: project(baseline), candidate: null, changed_fields: [] };
    const baselineContract = project(baseline);
    const candidateContract = project(candidate);
    const changedFields = Object.keys(baselineContract).filter((field) => canonicalJson(baselineContract[field]) !== canonicalJson(candidateContract[field]));
    return {
      subject_key: key,
      status: changedFields.length ? "changed" : "unchanged",
      baseline: baselineContract,
      candidate: candidateContract,
      changed_fields: changedFields,
    };
  });
}

function operatorContract(row) {
  return {
    op_type: row?.op_type || null,
    domain: row?.domain || null,
    version: row?.version ?? null,
    inputs: (row?.inputs || []).map((port) => port.value_ref),
    outputs: (row?.outputs || []).map((port) => port.value_ref),
    macs_decimal: row?.metrics?.macs?.decimal ?? null,
    mac_assessment_status: row?.metrics?.mac_assessment_status || null,
  };
}

function valueContract(row) {
  return row ? { dtype: row.dtype, shape: row.shape, shape_signature: row.shape_signature, shape_contract_status: row.shape_contract_status } : null;
}

function storageContract(row) {
  return {
    name: row?.name || null,
    dtype: row?.dtype || null,
    shape: row?.shape || [],
    serialized_byte_length_decimal: row?.serialized_byte_length?.decimal ?? null,
    encoding: row?.encoding || null,
    payload_sha256: row?.payload_sha256 || null,
  };
}

function quantizationContract(row) {
  return {
    subject_ref: row?.subject_ref || null,
    related_storage_refs: row?.related_storage_refs || [],
    mapping: row?.mapping || null,
    parameterization: row?.parameterization || null,
    storage: row?.storage || null,
    parameters: row?.parameters || null,
    completeness: row?.completeness || null,
  };
}

function classifyImpact({ graph, storage, quantization, artifactBytesChanged }) {
  const integration = [];
  const deployment = [];
  const task = [];
  if (graph.input_contract_changed) integration.push("external_input_contract_changed");
  if (graph.output_contract_changed) integration.push("external_output_contract_changed");
  if (graph.topology_changed) integration.push("serialized_operator_topology_changed");
  if (quantization.contract_changed) integration.push("serialized_quantization_contract_changed");
  if (graph.total_macs.changed) deployment.push("nominal_compute_changed");
  if (storage.layout_change_count || storage.serialized_bytes.changed) deployment.push("serialized_storage_layout_changed");
  if (storage.payload_digest_change_count) task.push("serialized_parameter_payload_changed");
  if (artifactBytesChanged && !integration.length && !deployment.length && !task.length) task.push("artifact_bytes_changed_semantic_cause_unclassified");
  const categories = [
    impactCategory("integration_revalidation", integration),
    impactCategory("deployment_reassessment", deployment),
    impactCategory("task_performance_reassessment", task),
  ];
  return {
    schema: "deepbom.change_impact.v1",
    categories,
    highest_action: integration.length ? "integration_revalidation"
      : deployment.length ? "deployment_reassessment"
        : task.length ? "task_performance_reassessment" : "no_change_observed",
    interpretation_boundary: "These categories identify evidence that should be revisited. They do not decide release authorization, clinical significance, regulatory submission scope, or runtime impact.",
  };
}

function impactCategory(id, reasons) {
  return { id, required: reasons.length > 0, reasons: [...new Set(reasons)].sort() };
}

function keyed(rows, keyOf) {
  const result = new Map();
  for (const row of rows || []) {
    const key = String(keyOf(row) || "");
    if (!key || result.has(key)) throw new Error(`Semantic diff cannot establish a unique subject key: ${key || "empty"}.`);
    result.set(key, row);
  }
  return result;
}

function quantizationKey(row) {
  const subject = String(row?.subject_ref || "");
  const id = String(row?.id || "");
  return subject ? `subject:${subject}|record:${id}` : `record:${id}`;
}

function scaleRatio(baseline, candidate) {
  const left = baseline?.inline_values;
  const right = candidate?.inline_values;
  if (!Array.isArray(left) || !Array.isArray(right) || !left.length || left.length !== right.length) return null;
  const candidateOverBaseline = [];
  for (let index = 0; index < left.length; index += 1) {
    const a = Math.abs(Number(left[index]));
    const b = Math.abs(Number(right[index]));
    if (!Number.isFinite(a) || !Number.isFinite(b) || a === 0 || b === 0) return null;
    candidateOverBaseline.push(b / a);
  }
  const minimum = Math.min(...candidateOverBaseline);
  const maximum = Math.max(...candidateOverBaseline);
  const uniform = candidateOverBaseline.every((value) => Math.abs(value - candidateOverBaseline[0]) <= Math.max(1, Math.abs(value)) * 1e-12)
    ? candidateOverBaseline[0] : null;
  return {
    compared_value_count: candidateOverBaseline.length,
    candidate_over_baseline: { minimum, maximum, uniform_factor: uniform },
    baseline_over_candidate: { minimum: 1 / maximum, maximum: 1 / minimum, uniform_factor: uniform == null ? null : 1 / uniform },
  };
}

function assertAlignmentCounts(rows, container, noun) {
  if (!Array.isArray(rows)) throw new Error(`Semantic diff ${noun} alignment is missing.`);
  const count = (status) => rows.filter((row) => row.status === status).length;
  if (container[`matched_${noun}_count`] !== count("unchanged") + count("changed")
      || container[`changed_${noun}_count`] !== count("changed")
      || container[`added_${noun}_count`] !== count("added")
      || container[`removed_${noun}_count`] !== count("removed")) {
    throw new Error(`Semantic diff ${noun} alignment counts are inconsistent.`);
  }
}

function artifactIdentity(ir) {
  return {
    filename: ir.artifact.filename,
    format: ir.artifact.format,
    sha256: ir.artifact.sha256,
    byte_length: ir.artifact.byte_length,
    artifact_ir_sha256: ir.artifact_ir_sha256,
  };
}

function transition(baseline, candidate) {
  return { baseline: baseline ?? null, candidate: candidate ?? null, changed: baseline !== candidate };
}

function assertArtifactIr(value, label) {
  if (value?.schema !== "deepbom.artifact_ir.v2" || !value?.artifact_ir_sha256 || !value?.artifact?.sha256) {
    throw new Error(`Semantic diff requires a materialized deepbom.artifact_ir.v2 ${label}.`);
  }
}
