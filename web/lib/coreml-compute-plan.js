import { COREML_DEPLOYMENT_SOURCE } from "./coreml-deployment-contract.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const COREML_COMPUTE_PLAN_SCHEMA = "deepbom.coreml_compute_plan.v1";

const COMPUTE_UNITS = new Set(["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"]);
const STRUCTURES = new Set(["neuralnetwork", "program"]);
const COMPUTE_DEVICES = new Set(["CPU", "GPU", "NEURAL_ENGINE"]);

export function buildCoreMlComputePlanTemplate(analysis) {
  requireCoreMlAnalysis(analysis);
  return {
    schema: COREML_COMPUTE_PLAN_SCHEMA,
    evidence_class: "COREML_COMPUTE_PLAN_ESTIMATE",
    artifact: { format: "coreml", filename: analysis.filename || null, sha256: requireSha(analysis.model_sha256, "artifact SHA-256") },
    runtime: {
      coremltools_version: null,
      coremltools_compute_plan_source_sha256: null,
      compiled_model_content_sha256: null,
      platform: null,
      architecture: null,
      platform_system: null,
      macos_version: null,
      os_build: null,
      hardware_model: null,
      python_version: null,
      available_compute_devices: [],
    },
    configuration: { compute_units: "ALL", function_name: analysis.coreml?.ml_program ? analysis.coreml.ml_program.functions ? analysis.coreml.description?.default_function_name || "main" : "main" : null },
    capture: { capture_id: null, collected_at: null, collector: { name: "deepbom-coreml-compute-plan-collector", version: "2", source_sha256: null } },
    structure: {
      kind: analysis.coreml?.model_type === "mlProgram" ? "program" : "neuralnetwork",
      operation_count: Array.isArray(analysis.ops) ? analysis.ops.length : 0,
      rows: [],
    },
    boundary: "MLComputePlan reports anticipated device usage and estimated relative cost for a compiled model. It is not an execution trace, measured latency, allocation trace, fusion record, lowering identity, or proof that a preferred device executed an operation.",
  };
}

export function parseCoreMlComputePlanDocument(source, analysis, { fileSha256 = null } = {}) {
  requireCoreMlAnalysis(analysis);
  const document = typeof source === "string" ? parseJsonObject(source) : source;
  if (!document || document.schema !== COREML_COMPUTE_PLAN_SCHEMA) throw new Error(`Core ML compute-plan evidence must use ${COREML_COMPUTE_PLAN_SCHEMA}.`);
  const artifactSha256 = requireSha(document.artifact?.sha256, "compute-plan artifact SHA-256");
  if (document.artifact?.format !== "coreml" || artifactSha256 !== requireSha(analysis.model_sha256, "active artifact SHA-256")) throw new Error("Core ML compute-plan evidence is not bound to the active artifact SHA-256.");
  const sourceSha256 = requireSha(document.runtime?.coremltools_compute_plan_source_sha256, "coremltools compute-plan source SHA-256");
  const compiledSha256 = requireSha(document.runtime?.compiled_model_content_sha256, "compiled model content SHA-256");
  const computeUnits = requireText(document.configuration?.compute_units, "Core ML compute units");
  if (!COMPUTE_UNITS.has(computeUnits)) throw new Error(`Unsupported Core ML compute units: ${computeUnits}.`);
  const expectedKind = analysis.coreml?.model_type === "mlProgram" ? "program"
    : analysis.coreml?.neural_network ? "neuralnetwork" : null;
  const kind = requireText(document.structure?.kind, "Core ML compute-plan structure kind");
  if (!STRUCTURES.has(kind) || expectedKind !== kind) throw new Error(`Core ML compute-plan structure ${kind} does not match the decoded artifact representation ${expectedKind || "unavailable"}.`);
  const functionName = validateFunctionName(document.configuration?.function_name, analysis, kind);
  const availableComputeDevices = validateAvailableComputeDevices(document.runtime?.available_compute_devices);
  const graphOps = Array.isArray(analysis.ops) ? analysis.ops : [];
  const rows = validateRows(document.structure?.rows, graphOps, kind, computeUnits, new Set(availableComputeDevices.map((device) => device.type)));
  if (Number(document.structure?.operation_count) !== rows.length || rows.length !== graphOps.length) throw new Error("Core ML compute-plan operation count does not match the decoded static graph.");
  const collectedAt = requireIsoTimestamp(document.capture?.collected_at, "compute-plan capture timestamp");
  const captureId = requireText(document.capture?.capture_id, "compute-plan capture ID");
  const collectorName = requireText(document.capture?.collector?.name, "compute-plan collector name");
  const collectorVersion = requireText(document.capture?.collector?.version, "compute-plan collector version");
  const collectorSourceSha256 = requireSha(document.capture?.collector?.source_sha256, "compute-plan collector source SHA-256");
  const estimatedRows = rows.filter((row) => row.estimated_cost_weight != null);
  const preferredCounts = count(rows.map((row) => row.preferred_compute_device || "not_determined"));
  const normalized = {
    schema: COREML_COMPUTE_PLAN_SCHEMA,
    evidence_class: "COREML_COMPUTE_PLAN_ESTIMATE",
    artifact: { format: "coreml", filename: document.artifact?.filename || analysis.filename || null, sha256: artifactSha256 },
    runtime: {
      coremltools_version: requireText(document.runtime?.coremltools_version, "coremltools version"),
      coremltools_compute_plan_source_sha256: sourceSha256,
      pinned_source_alignment: sourceSha256 === COREML_DEPLOYMENT_SOURCE.compute_plan_sha256 ? "exact_pinned_compute_plan_source" : "different_coremltools_compute_plan_source",
      compiled_model_content_sha256: compiledSha256,
      platform: requireText(document.runtime?.platform, "Core ML runtime platform"),
      architecture: requireText(document.runtime?.architecture, "Core ML runtime architecture"),
      platform_system: requireExactText(document.runtime?.platform_system, "Darwin", "Core ML capture platform system"),
      macos_version: requireText(document.runtime?.macos_version, "macOS version"),
      os_build: requireText(document.runtime?.os_build, "macOS build"),
      hardware_model: requireText(document.runtime?.hardware_model, "Apple hardware model"),
      python_version: requireText(document.runtime?.python_version, "Python version"),
      available_compute_devices: availableComputeDevices,
    },
    configuration: {
      compute_units: computeUnits,
      function_name: functionName,
    },
    capture: {
      capture_id: captureId,
      collected_at: collectedAt,
      collector: { name: collectorName, version: collectorVersion, source_sha256: collectorSourceSha256 },
      import_file_sha256: fileSha256 ? requireSha(fileSha256, "compute-plan import file SHA-256") : null,
    },
    structure: { kind, operation_count: rows.length, rows },
    summary: {
      mapped_operation_count: rows.length,
      preferred_compute_device_counts: preferredCounts,
      estimated_cost_operation_count: estimatedRows.length,
      estimated_cost_weight_sum: estimatedRows.reduce((sum, row) => sum + row.estimated_cost_weight, 0),
      unresolved_device_usage_count: rows.filter((row) => !row.preferred_compute_device).length,
    },
    runtime_identity_status: "bound",
    execution_status: "not_observed_compute_plan_only",
    compatibility_conclusion: "compiled_model_compute_plan_imported",
    source: { ...COREML_DEPLOYMENT_SOURCE },
    boundary: "The imported MLComputePlan is a runtime-produced estimate for one compiled model and compute-unit configuration. Preferred/supported devices and cost weights are not observed execution placement or measured timing. The browser binds identities and static op order but does not re-run Core ML.",
  };
  return { ...normalized, normalized_manifest_sha256: sha256TextHex(canonicalJson(normalized)) };
}

export function isCoreMlComputePlanDocument(value) { return value?.schema === COREML_COMPUTE_PLAN_SCHEMA; }

function validateRows(value, graphOps, kind, computeUnits, availableDevices) {
  if (!Array.isArray(value)) throw new Error("Core ML compute-plan rows must be an array.");
  const seen = new Set();
  return value.map((row, index) => {
    const opIndex = Number(row?.op_index);
    if (!Number.isSafeInteger(opIndex) || opIndex !== index || seen.has(opIndex)) throw new Error("Core ML compute-plan op indices must be unique, contiguous, and ordered.");
    seen.add(opIndex);
    const op = graphOps[index];
    if (!op) throw new Error(`Core ML compute-plan row ${index} has no decoded static op.`);
    const operatorType = requireText(row.operator_type, `compute-plan operator type ${index}`);
    const expectedType = kind === "program" ? String(op.mil_operation_type || "") : String(op.name || "");
    if (normalizeType(operatorType) !== normalizeType(expectedType)) throw new Error(`Core ML compute-plan operator type mismatch at #${index}.`);
    const identity = requireText(row.identity, `compute-plan operation identity ${index}`);
    const expectedIdentity = String(op.coreml_layer_name || "");
    if (identity !== expectedIdentity) throw new Error(`Core ML compute-plan operation identity mismatch at #${index}.`);
    const preferred = row.preferred_compute_device == null ? null : requireText(row.preferred_compute_device, `preferred compute device ${index}`);
    const supported = uniqueStrings(row.supported_compute_devices || [], `supported compute devices ${index}`);
    if ((preferred && !COMPUTE_DEVICES.has(preferred)) || supported.some((device) => !COMPUTE_DEVICES.has(device))) throw new Error(`Unknown Core ML compute device at #${index}.`);
    if (preferred && !supported.includes(preferred)) throw new Error(`Preferred Core ML compute device is not in the supported-device set at #${index}.`);
    if ((preferred && !availableDevices.has(preferred)) || supported.some((device) => !availableDevices.has(device))) throw new Error(`Core ML compute-plan row #${index} references a device absent from the captured host inventory.`);
    const permitted = computeUnits === "CPU_ONLY" ? new Set(["CPU"])
      : computeUnits === "CPU_AND_GPU" ? new Set(["CPU", "GPU"])
        : computeUnits === "CPU_AND_NE" ? new Set(["CPU", "NEURAL_ENGINE"]) : COMPUTE_DEVICES;
    if (preferred && !permitted.has(preferred)) throw new Error(`Preferred Core ML compute device at #${index} violates compute-units ${computeUnits}.`);
    const cost = row.estimated_cost_weight == null ? null : Number(row.estimated_cost_weight);
    if (cost != null && (!Number.isFinite(cost) || cost < 0 || cost > 1)) throw new Error(`Core ML estimated cost weight at #${index} is outside [0,1].`);
    return { op_index: opIndex, operator_type: operatorType, identity, preferred_compute_device: preferred, supported_compute_devices: supported, estimated_cost_weight: cost };
  });
}

function validateFunctionName(value, analysis, kind) {
  if (kind === "neuralnetwork") {
    if (value != null) throw new Error("NeuralNetwork compute-plan evidence must not declare an ML Program function name.");
    return null;
  }
  const functions = Object.keys(analysis.coreml?.ml_program?.functions || {});
  const expected = analysis.coreml?.description?.default_function_name
    || (functions.includes("main") ? "main" : functions.length === 1 ? functions[0] : null);
  if (!expected) throw new Error("The decoded ML Program function identity is ambiguous.");
  const observed = requireText(value, "Core ML function name");
  if (observed !== expected) throw new Error(`Core ML compute-plan function ${observed} does not match decoded function ${expected}.`);
  return observed;
}

function validateAvailableComputeDevices(value) {
  if (!Array.isArray(value) || !value.length) throw new Error("Core ML available compute-device inventory is required.");
  const seen = new Set();
  return value.map((device, index) => {
    const type = requireText(device?.type, `Core ML compute-device type ${index}`);
    if (!COMPUTE_DEVICES.has(type) || seen.has(type)) throw new Error("Core ML compute-device inventory contains an unknown or duplicate type.");
    seen.add(type);
    const sourceClass = requireText(device?.source_class, `Core ML compute-device source class ${index}`);
    const instanceCount = Number(device?.instance_count);
    if (!Number.isSafeInteger(instanceCount) || instanceCount <= 0) throw new Error("Core ML compute-device inventory requires a positive instance count.");
    const totalCoreCount = device?.total_core_count == null ? null : Number(device.total_core_count);
    if (type === "NEURAL_ENGINE" && (!Number.isSafeInteger(totalCoreCount) || totalCoreCount <= 0)) throw new Error("Neural Engine compute-device inventory requires a positive total core count.");
    if (type !== "NEURAL_ENGINE" && totalCoreCount != null) throw new Error("Only Neural Engine inventory may declare total core count.");
    return { type, source_class: sourceClass, instance_count: instanceCount, total_core_count: totalCoreCount };
  });
}

function requireCoreMlAnalysis(analysis) {
  if (!analysis || analysis.format !== "coreml") throw new Error("Core ML compute-plan evidence requires an active Core ML audit.");
  if (!Array.isArray(analysis.ops) || !analysis.ops.length
    || !(analysis.coreml?.neural_network || analysis.coreml?.model_type === "mlProgram")) {
    throw new Error("Core ML compute-plan evidence requires a decoded NeuralNetwork or ML Program graph; classical and pipeline structures are not accepted.");
  }
}
function parseJsonObject(text) { let value; try { value = JSON.parse(text); } catch { throw new Error("Core ML compute-plan JSON is invalid."); } if (!value || Array.isArray(value) || typeof value !== "object") throw new Error("Core ML compute-plan evidence must be a JSON object."); return value; }
function requireSha(value, label) { const text = String(value || "").toLowerCase(); if (!/^[a-f0-9]{64}$/.test(text)) throw new Error(`${label} is required.`); return text; }
function requireText(value, label) { const text = String(value || "").trim(); if (!text || text.length > 8192) throw new Error(`${label} is required and bounded.`); return text; }
function requireExactText(value, expected, label) { const text = requireText(value, label); if (text !== expected) throw new Error(`${label} must be ${expected}.`); return text; }
function requireIsoTimestamp(value, label) { const text = requireText(value, label); const date = new Date(text); if (!Number.isFinite(date.getTime()) || date.toISOString() !== text) throw new Error(`${label} must be an ISO-8601 UTC timestamp.`); return text; }
function normalizeType(value) { return String(value || "").replaceAll(/[^a-z0-9]/gi, "").toLowerCase(); }
function uniqueStrings(value, label) { if (!Array.isArray(value)) throw new Error(`${label} must be an array.`); const rows = value.map((item) => requireText(item, label)); if (new Set(rows).size !== rows.length) throw new Error(`${label} contains duplicates.`); return rows; }
function count(values) { const result = {}; for (const value of values) result[value] = (result[value] || 0) + 1; return result; }
