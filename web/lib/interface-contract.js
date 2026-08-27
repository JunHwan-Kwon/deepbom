import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const INTERFACE_CONTRACT_SCHEMA = "deepbom.interface_quantization_contracts.v1.3";
export const INTERFACE_HASH_METHOD = "SHA-256 over UTF-8 RFC8785-JCS canonical JSON";
export const INTERFACE_CANONICALIZATION = "RFC8785-JCS";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const QUANT_PREFIX = "cdx:ai-ml:model:parameter:quantization:";

export function canonicalContractSha256(value) {
  return sha256TextHex(canonicalJson(value));
}

export function legacyContractSha256(value) {
  return sha256TextHex(JSON.stringify(value));
}

export function quantizationContractPayload({
  status,
  scheme,
  granularity,
  scales,
  zeroPoints,
  axis,
  blockSize = null,
  parameterization = null,
}) {
  const normalizedAxis = optionalInteger(axis);
  const normalizedBlockSize = optionalInteger(blockSize);
  return {
    status: String(status || "invalid_or_incomplete"),
    scheme: String(scheme || "none"),
    granularity: String(granularity || "invalid_or_incomplete"),
    parameterization: String(parameterization || granularity || "invalid_or_incomplete"),
    scales: numericArray(scales),
    zero_points: integerArray(zeroPoints),
    axis: normalizedAxis,
    block_size: normalizedBlockSize != null && normalizedBlockSize > 0 ? normalizedBlockSize : null,
  };
}

function optionalInteger(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isSafeInteger(number) ? number : null;
}

export function interfaceContractPayload(parameter) {
  const quantization = parameter?.quantization || {};
  return {
    direction: normalizeDirection(parameter?.direction),
    ordinal: nonNegativeInteger(parameter?.ordinal),
    tensor_name: String(parameter?.tensor_name ?? parameter?.name ?? ""),
    dtype: String(parameter?.dtype || "UNKNOWN").toUpperCase(),
    shape: dimensionArray(parameter?.shape),
    shape_signature: Array.isArray(parameter?.shape_signature)
      ? dimensionArray(parameter.shape_signature) : null,
    quantization: quantizationContractPayload({
      status: quantization.status,
      scheme: quantization.scheme,
      granularity: quantization.granularity,
      parameterization: quantization.parameterization,
      scales: quantization.scales,
      zeroPoints: quantization.zero_points,
      axis: quantization.axis,
      blockSize: quantization.block_size,
    }),
  };
}

export function contractHashDescriptor(canonicalSha256, legacySha256 = null, payloadContract = {}) {
  return {
    algorithm: "SHA-256",
    canonicalization: INTERFACE_CANONICALIZATION,
    method: INTERFACE_HASH_METHOD,
    sha256: canonicalSha256,
    payload_contract: {
      selection: payloadContract.selection || "included_json_pointers_only",
      included_json_pointers: [...(payloadContract.includedJsonPointers || [])],
      excluded_json_pointers: [...(payloadContract.excludedJsonPointers || [])],
      interpretation: payloadContract.interpretation
        || "Resolve the listed JSON Pointers relative to the containing contract object, construct the declared payload, canonicalize it with RFC8785-JCS, encode as UTF-8, and hash with SHA-256.",
    },
    ...(legacySha256 ? {
      legacy_v1_1: {
        method: "SHA-256 over UTF-8 JSON.stringify payload bytes",
        sha256: legacySha256,
        ordered_payload_fields: [...(payloadContract.legacyOrderedPayloadFields || [])],
      },
    } : {}),
  };
}

export function compareInterfaceContracts(ledger, declarationSource, expectedArtifactSha256 = "") {
  const parsed = parseProductionInterfaceContract(declarationSource);
  const expected = Array.isArray(ledger?.parameters) ? ledger.parameters : [];
  const expectedArtifact = normalizeSha256(expectedArtifactSha256 || ledger?.subject?.sha256);
  if (!parsed.provided) return comparisonResult("unbound", expected, parsed, [], expectedArtifact);
  if (!parsed.valid) return comparisonResult("invalid_declaration", expected, parsed, parsed.errors.map((message) => ({
    parameter_id: null,
    field: "declaration",
    expected: "valid production interface contract",
    declared: message,
    evidence_pointer: "/production_interface_contract",
  })), expectedArtifact);

  const declaration = parsed.contract;
  const diffs = [];
  const declaredArtifact = normalizeSha256(declaration.artifact_sha256);
  if (expectedArtifact && declaredArtifact && expectedArtifact !== declaredArtifact) {
    diffs.push(diff(null, "artifact_sha256", expectedArtifact, declaredArtifact, "/artifact_sha256"));
  }
  const expectedByKey = new Map(expected.map((row) => [parameterKey(row.direction, row.ordinal), row]));
  const declaredByKey = new Map(declaration.parameters.map((row) => [parameterKey(row.direction, row.ordinal), row]));
  for (const [key, expectedRow] of expectedByKey) {
    const declaredRow = declaredByKey.get(key);
    if (!declaredRow) {
      diffs.push(diff(expectedRow.parameter_id, "parameter", "present", "missing", `/parameters/${escapePointer(key)}`));
      continue;
    }
    compareParameter(expectedRow, declaredRow, diffs, key);
  }
  for (const [key, declaredRow] of declaredByKey) {
    if (!expectedByKey.has(key)) {
      diffs.push(diff(declaredRow.parameter_id || key, "parameter", "absent", "extra", `/parameters/${escapePointer(key)}`));
    }
  }

  let status = "bound_exact_contract";
  const artifactHashMissing = Boolean(expectedArtifact && !declaredArtifact);
  const implementationHashMissing = !normalizeSha256(declaration.implementation_sha256);
  if (artifactHashMissing && implementationHashMissing) status = "partial_artifact_and_implementation_hash_missing";
  else if (artifactHashMissing) status = "partial_artifact_hash_missing";
  else if (implementationHashMissing) status = "partial_implementation_hash_missing";
  if (diffs.some((row) => row.field === "artifact_sha256")) status = "contradiction_artifact_hash_mismatch";
  else if (diffs.length) status = "contradiction_interface_contract_mismatch";
  return comparisonResult(status, expected, parsed, diffs, expectedArtifact);
}

export function parseProductionInterfaceContract(source) {
  if (source == null || source === "") return { provided: false, valid: true, contract: null, errors: [], source_kind: "none" };
  let value = source;
  if (typeof value === "string") {
    try { value = JSON.parse(value); }
    catch { return { provided: true, valid: false, contract: null, errors: ["Input is not valid JSON."], source_kind: "json" }; }
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { provided: true, valid: false, contract: null, errors: ["Contract root must be a JSON object."], source_kind: "unknown" };
  }
  const fromCycloneDx = (value.bomFormat === "CycloneDX" || value.specFormat === "CycloneDX") ? contractFromCycloneDx(value) : null;
  const candidate = fromCycloneDx || value.production_interface_contract || value.interface_contract_ledger || value;
  const parameters = Array.isArray(candidate?.parameters) ? candidate.parameters
    : Array.isArray(candidate?.inputs) || Array.isArray(candidate?.outputs)
      ? [
          ...(candidate.inputs || []).map((row, ordinal) => ({ ...row, direction: "input", ordinal })),
          ...(candidate.outputs || []).map((row, ordinal) => ({ ...row, direction: "output", ordinal })),
        ]
      : null;
  const errors = [...(fromCycloneDx?.extraction_errors || [])];
  if (!parameters) errors.push("A parameters array, or inputs/outputs arrays, is required.");
  const normalized = [];
  const seen = new Set();
  for (const [index, row] of (parameters || []).entries()) {
    const parameter = normalizeDeclaredParameter(row, index);
    if (!parameter.valid) {
      errors.push(...parameter.errors.map((message) => `parameters[${index}]: ${message}`));
      continue;
    }
    const key = parameterKey(parameter.value.direction, parameter.value.ordinal);
    if (seen.has(key)) errors.push(`Duplicate parameter identity ${key}.`);
    seen.add(key);
    normalized.push(parameter.value);
  }
  const artifactSource = candidate?.artifact_sha256 || candidate?.subject?.sha256 || fromCycloneDx?.artifact_sha256;
  const implementationSource = candidate?.implementation_sha256 || candidate?.encoder_decoder_sha256;
  const contract = {
    schema: String(candidate?.schema || "deepbom.production_interface_contract.v1"),
    artifact_sha256: validatedOptionalSha256(artifactSource, "artifact_sha256", errors),
    implementation_sha256: validatedOptionalSha256(implementationSource, "implementation_sha256", errors),
    parameters: normalized,
  };
  return {
    provided: true,
    valid: errors.length === 0,
    contract,
    errors,
    source_kind: fromCycloneDx ? "cyclonedx_2_draft" : "deepbom_json",
  };
}

function comparisonResult(status, expected, parsed, diffs, expectedArtifactSha256) {
  const mismatch = status.startsWith("contradiction") || status === "invalid_declaration";
  return {
    schema: "deepbom.interface_contract_comparison.v1",
    status,
    evidence_class: mismatch ? "DERIVED_FROM_ARTIFACT_AND_DECLARATION" : status === "bound_exact_contract" ? "DECLARED_RELEASE_BINDING" : "NOT_ASSESSED",
    expected_artifact_sha256: expectedArtifactSha256 || null,
    declared_artifact_sha256: parsed.contract?.artifact_sha256 || null,
    implementation_sha256: parsed.contract?.implementation_sha256 || null,
    expected_parameter_count: expected.length,
    declared_parameter_count: parsed.contract?.parameters?.length || 0,
    mismatch_count: diffs.length,
    mismatches: diffs,
    declaration_validation: {
      source_kind: parsed.source_kind,
      valid: parsed.valid,
      errors: parsed.errors,
    },
    gate_result: mismatch ? "block" : status === "bound_exact_contract" ? "pass" : "pending",
    interpretation_boundary: "A mismatch is an observed deployment-configuration integrity contradiction. An absent declaration is unbound evidence, not a vulnerability or proof of incorrect inference.",
  };
}

function compareParameter(expectedRow, declaredRow, diffs, key) {
  const expectedPayload = interfaceContractPayload(expectedRow);
  const base = `/parameters/${escapePointer(key)}`;
  const declaredHash = normalizeSha256(declaredRow.interface_contract_sha256);
  if (!declaredRow.has_structured_contract) {
    if (declaredHash !== expectedRow.interface_contract_sha256) {
      diffs.push(diff(expectedRow.parameter_id, "interface_contract_sha256", expectedRow.interface_contract_sha256, declaredHash || null, `${base}/interface_contract_sha256`));
    }
    return;
  }
  const declaredPayload = interfaceContractPayload(declaredRow);
  const fields = [
    ["direction", expectedPayload.direction, declaredPayload.direction],
    ["ordinal", expectedPayload.ordinal, declaredPayload.ordinal],
    ["tensor_name", expectedPayload.tensor_name, declaredPayload.tensor_name],
    ["dtype", expectedPayload.dtype, declaredPayload.dtype],
    ["shape", expectedPayload.shape, declaredPayload.shape],
    ["shape_signature", expectedPayload.shape_signature, declaredPayload.shape_signature],
    ["quantization.status", expectedPayload.quantization.status, declaredPayload.quantization.status],
    ["quantization.scheme", expectedPayload.quantization.scheme, declaredPayload.quantization.scheme],
    ["quantization.granularity", expectedPayload.quantization.granularity, declaredPayload.quantization.granularity],
    ["quantization.parameterization", expectedPayload.quantization.parameterization, declaredPayload.quantization.parameterization],
    ["quantization.scales", expectedPayload.quantization.scales, declaredPayload.quantization.scales],
    ["quantization.zero_points", expectedPayload.quantization.zero_points, declaredPayload.quantization.zero_points],
    ["quantization.axis", expectedPayload.quantization.axis, declaredPayload.quantization.axis],
    ["quantization.block_size", expectedPayload.quantization.block_size, declaredPayload.quantization.block_size],
  ];
  for (const [field, expected, declared] of fields) {
    if (canonicalJson(expected) !== canonicalJson(declared)) {
      diffs.push(diff(expectedRow.parameter_id, field, expected, declared, `${base}/${field.replaceAll(".", "/")}`));
    }
  }
  const computedHash = canonicalContractSha256(declaredPayload);
  if (declaredHash && declaredHash !== computedHash) {
    diffs.push(diff(expectedRow.parameter_id, "declared_hash_integrity", computedHash, declaredHash, `${base}/interface_contract_sha256`));
  }
}

function normalizeDeclaredParameter(row, fallbackOrdinal) {
  if (!row || typeof row !== "object" || Array.isArray(row)) return { valid: false, errors: ["Parameter must be an object."] };
  const direction = normalizeDirection(row.direction);
  const ordinal = nonNegativeInteger(row.ordinal ?? fallbackOrdinal);
  const errors = [];
  if (!direction) errors.push("direction must be input or output.");
  if (ordinal == null) errors.push("ordinal must be a non-negative integer.");
  const quant = row.quantization && typeof row.quantization === "object" ? row.quantization : {};
  const hasStructured = ["tensor_name", "name", "dtype", "shape", "quantization"].some((field) => Object.hasOwn(row, field));
  const granularity = normalizeGranularity(quant.granularity || row.granularity || row.quantization_parameterization);
  const scales = validatedNumberArray(quant.scales ?? row.scales, "quantization.scales", errors, { positive: true });
  const zeroPoints = validatedNumberArray(quant.zero_points ?? row.zero_points, "quantization.zero_points", errors, { integer: true });
  const shape = validatedDimensionArray(row.shape, "shape", errors, { required: hasStructured, nonNegative: true });
  const shapeSignature = row.shape_signature == null
    ? null : validatedDimensionArray(row.shape_signature, "shape_signature", errors);
  const interfaceHash = validatedOptionalSha256(row.interface_contract_sha256, "interface_contract_sha256", errors);
  const axis = validatedOptionalInteger(quant.axis ?? row.axis ?? row.quantized_dimension, "quantization.axis", errors);
  const blockSize = validatedOptionalInteger(quant.block_size ?? row.block_size, "quantization.block_size", errors, { positive: true });
  const noAffine = granularity === "not_quantized" || (!scales.length && !zeroPoints.length);
  const status = String(quant.status || row.status || (noAffine ? "not_quantized" : "complete"));
  const scheme = normalizeScheme(quant.scheme || row.scheme, noAffine);
  return {
    valid: errors.length === 0,
    errors,
    value: {
      parameter_id: String(row.parameter_id || `${direction}:${ordinal}`),
      direction,
      ordinal,
      tensor_name: String(row.tensor_name ?? row.name ?? ""),
      dtype: String(row.dtype || row.data_type || "UNKNOWN").toUpperCase(),
      shape,
      shape_signature: shapeSignature,
      quantization: {
        status,
        scheme,
        granularity,
        parameterization: normalizeGranularity(quant.parameterization || row.parameterization || granularity),
        scales,
        zero_points: zeroPoints,
        axis,
        block_size: blockSize,
      },
      interface_contract_sha256: interfaceHash || "",
      has_structured_contract: hasStructured,
    },
  };
}

function contractFromCycloneDx(document) {
  const errors = [];
  const component = document.metadata?.component || (document.components || []).find((row) => row?.type === "machine-learning-model");
  const modelParameters = component?.modelCard?.modelParameters;
  if (!modelParameters) errors.push("CycloneDX modelCard.modelParameters is required.");
  const inputs = modelParameters?.inputs;
  const outputs = modelParameters?.outputs;
  if (inputs != null && !Array.isArray(inputs)) errors.push("CycloneDX modelParameters.inputs must be an array.");
  if (outputs != null && !Array.isArray(outputs)) errors.push("CycloneDX modelParameters.outputs must be an array.");
  const parameters = [
    ...(Array.isArray(inputs) ? inputs : []).map((row, ordinal) => parameterFromCycloneDx(row, "input", ordinal, errors)),
    ...(Array.isArray(outputs) ? outputs : []).map((row, ordinal) => parameterFromCycloneDx(row, "output", ordinal, errors)),
  ];
  const artifactHash = (component?.hashes || []).find((row) => String(row?.alg).toUpperCase().replace("-", "") === "SHA256")?.content;
  const componentProperties = propertyMap(component?.properties, errors, "CycloneDX component.properties");
  return {
    schema: "deepbom.production_interface_contract.from_cyclonedx_2_draft",
    artifact_sha256: artifactHash,
    implementation_sha256: componentProperties.get("deepbom:production:interfaceImplementationSha256") || null,
    parameters,
    extraction_errors: errors,
  };
}

function parameterFromCycloneDx(row, direction, ordinal, errors) {
  const label = `CycloneDX ${direction}[${ordinal}].properties`;
  const props = propertyMap(row?.properties, errors, label);
  const schemeValue = props.get(`${QUANT_PREFIX}scheme`);
  const granularityValue = props.get(`${QUANT_PREFIX}granularity`);
  const scales = jsonNumberArray(props.get(`${QUANT_PREFIX}scale`), errors, `${label}.scale`, { positive: true });
  const zeroPoints = jsonNumberArray(props.get(`${QUANT_PREFIX}zeroPoint`), errors, `${label}.zeroPoint`, { integer: true });
  const blockSize = validatedOptionalInteger(
    props.get(`${QUANT_PREFIX}_undefined:blockSize`) || props.get("deepbom:quantization:blockSize"),
    `${label}.blockSize`,
    errors,
    { positive: true },
  );
  const noAffine = !schemeValue && !granularityValue && !scales.length && !zeroPoints.length;
  const granularity = blockSize ? "blocked" : normalizeGranularity(granularityValue || (noAffine ? "not_quantized" : scales.length > 1 ? "per_axis" : "per_tensor"));
  return {
    direction,
    ordinal,
    name: row?.name || "",
    dtype: row?.format?.dataType || "UNKNOWN",
    shape: row?.shape || [],
    shape_signature: jsonDimensionArray(props.get("deepbom:parameter:shapeSignature"), null, errors, `${label}.shapeSignature`),
    quantization: {
      status: noAffine ? "not_quantized" : "complete",
      scheme: normalizeScheme(schemeValue, noAffine),
      granularity,
      parameterization: granularity,
      scales,
      zero_points: zeroPoints,
      axis: validatedOptionalInteger(props.get(`${QUANT_PREFIX}axis`), `${label}.axis`, errors),
      block_size: blockSize,
    },
    interface_contract_sha256: props.get("deepbom:parameter:interfaceContractSha256") || "",
  };
}

function propertyMap(items, errors = [], label = "properties") {
  if (items != null && !Array.isArray(items)) {
    errors.push(`${label} must be an array.`);
    return new Map();
  }
  const result = new Map();
  for (const [index, item] of (items || []).entries()) {
    const name = String(item?.name || "");
    if (!name || typeof item?.value !== "string") errors.push(`${label}[${index}] must contain a non-empty name and string value.`);
    if (result.has(name)) errors.push(`${label} repeats property ${name}.`);
    result.set(name, String(item?.value ?? ""));
  }
  return result;
}

function normalizeDirection(value) {
  const direction = String(value || "").toLowerCase();
  return direction === "input" || direction === "output" ? direction : "";
}

function normalizeGranularity(value) {
  const normalized = String(value || "invalid_or_incomplete").toLowerCase().replaceAll("-", "_");
  return new Set(["per_tensor", "per_axis", "blocked", "not_quantized", "invalid_or_incomplete"]).has(normalized)
    ? normalized : "invalid_or_incomplete";
}

function normalizeScheme(value, noAffine) {
  if (noAffine) return "none";
  const scheme = String(value || "affine");
  return scheme === "affine_symmetric" || scheme === "affine_asymmetric" || scheme.startsWith("_undefined:") ? "affine" : scheme;
}

function jsonNumberArray(value, errors, label, options = {}) {
  if (value == null || value === "") return [];
  try {
    const parsed = JSON.parse(String(value));
    return validatedNumberArray(Array.isArray(parsed) ? parsed : [parsed], label, errors, options);
  } catch {
    errors.push(`${label} must be a JSON number or array of numbers.`);
    return [];
  }
}

function jsonDimensionArray(value, fallback, errors, label) {
  if (value == null || value === "") return fallback == null ? null : dimensionArray(fallback);
  try {
    const parsed = JSON.parse(String(value));
    if (parsed === null) return null;
    if (!Array.isArray(parsed)) {
      errors.push(`${label} must be a JSON array.`);
      return fallback == null ? null : dimensionArray(fallback);
    }
    return validatedDimensionArray(parsed, label, errors);
  } catch {
    errors.push(`${label} must be a JSON array.`);
    return fallback == null ? null : dimensionArray(fallback);
  }
}

function numericArray(values) {
  return (Array.isArray(values) ? values : []).map(Number).filter(Number.isFinite);
}

function validatedNumberArray(values, label, errors, { integer = false, positive = false } = {}) {
  if (values == null) return [];
  if (!Array.isArray(values)) {
    errors.push(`${label} must be an array.`);
    return [];
  }
  const result = [];
  for (const [index, value] of values.entries()) {
    if (typeof value !== "number" || !Number.isFinite(value)
      || integer && !Number.isSafeInteger(value) || positive && !(value > 0)) {
      errors.push(`${label}[${index}] must be a ${positive ? "finite positive " : "finite "}${integer ? "integer" : "number"}.`);
    } else {
      result.push(value);
    }
  }
  return result;
}

function validatedDimensionArray(values, label, errors, { required = false, nonNegative = false } = {}) {
  if (values == null) {
    if (required) errors.push(`${label} must be an array.`);
    return [];
  }
  if (!Array.isArray(values)) {
    errors.push(`${label} must be an array.`);
    return [];
  }
  return values.map((value, index) => {
    if (value == null) return null;
    if (typeof value !== "number" || !Number.isSafeInteger(value) || nonNegative && value < 0) {
      errors.push(`${label}[${index}] must be ${nonNegative ? "a non-negative integer or null" : "an integer or null"}.`);
      return null;
    }
    return value;
  });
}

function integerArray(values) {
  return (Array.isArray(values) ? values : []).map(Number).filter(Number.isSafeInteger);
}

function dimensionArray(values) {
  return (Array.isArray(values) ? values : []).map((value) => value == null ? null : Number(value));
}

function nonNegativeInteger(value) {
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : null;
}

function normalizeSha256(value) {
  const digest = String(value || "").trim().toLowerCase();
  return SHA256_PATTERN.test(digest) ? digest : "";
}

function validatedOptionalSha256(value, label, errors) {
  if (value == null || value === "") return null;
  const digest = normalizeSha256(value);
  if (!digest) errors.push(`${label} must be exactly 64 lowercase or uppercase hexadecimal characters.`);
  return digest || null;
}

function validatedOptionalInteger(value, label, errors, { positive = false } = {}) {
  if (value == null || value === "") return null;
  const number = typeof value === "number" ? value : Number(value);
  if (!Number.isSafeInteger(number) || positive && number <= 0) {
    errors.push(`${label} must be ${positive ? "a positive" : "an"} integer.`);
    return null;
  }
  return number;
}

function parameterKey(direction, ordinal) {
  return `${normalizeDirection(direction)}:${nonNegativeInteger(ordinal)}`;
}

function diff(parameterId, field, expected, declared, evidencePointer) {
  return { parameter_id: parameterId, field, expected, declared, evidence_pointer: evidencePointer };
}

function escapePointer(value) {
  return String(value).replaceAll("~", "~0").replaceAll("/", "~1");
}
