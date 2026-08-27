import { buildInterfaceQuantizationContractLedger } from "./quantization-contract-summary.js";
import { artifactBomRef, artifactContentVersion, cycloneDxSerialNumber } from "./cyclonedx-identity.js";

export const CYCLONEDX_20_AI_ML_DRAFT = Object.freeze({
  specification_pr: "https://github.com/CycloneDX/specification/pull/990",
  specification_commit: "58a7cc2d04105e7525b0ed369ccf0a4325dc34b2",
  schema: "https://raw.githubusercontent.com/CycloneDX/specification/58a7cc2d04105e7525b0ed369ccf0a4325dc34b2/schema/2.0/cyclonedx-2.0.schema.json",
  taxonomy_pr: "https://github.com/CycloneDX/cyclonedx-property-taxonomy/pull/175",
  taxonomy_commit: "1b380dfae8bf4a83646ae59ea3d3b42d466f3858",
  conformance_status: "NON_CONFORMANT_PROPOSAL_FIXTURE_SCHEMA_CLOSURE_UNAVAILABLE",
});

const PREFIX = "cdx:ai-ml:model:parameter:quantization:";

export function buildCycloneDx20ParameterContractPreview(analysis, options = {}) {
  const ledger = options.interfaceLedger || buildInterfaceQuantizationContractLedger(analysis);
  const sha256 = normalizeSha256(options.hash || analysis?.model_sha256);
  const name = String(analysis?.filename || "model");
  const bomRef = artifactBomRef(sha256, name);
  const contentVersion = artifactContentVersion(sha256, analysis?.metadata_presence?.model_version);
  const generatedAt = options.generatedAt || new Date().toISOString();
  const parameters = {
    inputs: ledger.parameters.filter((row) => row.direction === "input").map(parameterEntry),
    outputs: ledger.parameters.filter((row) => row.direction === "output").map(parameterEntry),
    properties: [
      property("deepbom:interfaceContract:schema", ledger.schema),
      property("deepbom:interfaceContract:ledgerSha256", ledger.ledger_sha256),
    ],
  };
  return {
    $schema: CYCLONEDX_20_AI_ML_DRAFT.schema,
    specFormat: "CycloneDX",
    specVersion: "2.0",
    serialNumber: cycloneDxSerialNumber({ artifactSha256: sha256, generatedAt, profile: "cyclonedx-2.0-parameter-contract-preview" }),
    version: 1,
    metadata: {
      timestamp: generatedAt,
      component: {
        type: "machine-learning-model",
        name,
        ...(contentVersion.version ? { version: contentVersion.version } : {}),
        "bom-ref": bomRef,
        ...(sha256 ? { hashes: [{ alg: "SHA-256", content: sha256 }] } : {}),
        modelCard: {
          "bom-ref": `${bomRef}-model-card`,
          modelParameters: parameters,
          properties: [
            property("deepbom:preview:status", CYCLONEDX_20_AI_ML_DRAFT.conformance_status),
            property("deepbom:preview:specificationCommit", CYCLONEDX_20_AI_ML_DRAFT.specification_commit),
            property("deepbom:preview:taxonomyCommit", CYCLONEDX_20_AI_ML_DRAFT.taxonomy_commit),
            property("deepbom:preview:interpretationBoundary", "Parameter binding follows the pinned CycloneDX 2.0 draft. Generic affine facts are not promoted to symmetric or asymmetric because the artifact does not encode that classification."),
          ],
        },
        properties: [
          property("deepbom:contract:schema", "deepbom.cyclonedx_2_0_parameter_contract_preview.v1"),
          property("deepbom:model:versionBasis", contentVersion.basis),
          property("deepbom:preview:specificationPr", CYCLONEDX_20_AI_ML_DRAFT.specification_pr),
          property("deepbom:preview:taxonomyPr", CYCLONEDX_20_AI_ML_DRAFT.taxonomy_pr),
        ],
      },
    },
  };
}

export function validateCycloneDx20ParameterContractPreview(document) {
  const errors = [];
  if (document?.specFormat !== "CycloneDX" || document?.specVersion !== "2.0") errors.push("Pinned CycloneDX 2.0 proposal identity is required.");
  if (document?.$schema !== CYCLONEDX_20_AI_ML_DRAFT.schema) errors.push("Draft schema source pin is missing or changed.");
  const status = document?.metadata?.component?.modelCard?.properties?.find((item) => item?.name === "deepbom:preview:status")?.value;
  if (status !== CYCLONEDX_20_AI_ML_DRAFT.conformance_status) errors.push("The non-conformant proposal-fixture status is required.");
  const component = document?.metadata?.component;
  if (component?.type !== "machine-learning-model") errors.push("metadata.component must be a machine-learning-model.");
  const modelParameters = component?.modelCard?.modelParameters;
  if (!modelParameters) errors.push("modelCard.modelParameters is required by this preview profile.");
  for (const [direction, rows] of [["inputs", modelParameters?.inputs], ["outputs", modelParameters?.outputs]]) {
    if (rows != null && !Array.isArray(rows)) {
      errors.push(`${direction} must be an array.`);
      continue;
    }
    for (const [ordinal, row] of (rows || []).entries()) validateParameter(row, direction, ordinal, errors);
  }
  return { valid: errors.length === 0, errors };
}

function parameterEntry(parameter) {
  const quantization = parameter.quantization || {};
  const properties = [
    property("deepbom:parameter:direction", parameter.direction),
    property("deepbom:parameter:ordinal", parameter.ordinal),
    property("deepbom:parameter:interfaceContractSha256", parameter.interface_contract_sha256),
    property("deepbom:parameter:shapeSignature", JSON.stringify(parameter.shape_signature ?? null)),
    property("deepbom:parameter:affineMappingStatus", quantization.affine_mapping_status),
  ];
  if (quantization.status === "complete") {
    properties.push(property(`${PREFIX}scheme`, "_undefined:affine"));
    if (quantization.granularity === "blocked") {
      properties.push(property(`${PREFIX}_undefined:granularity`, "blocked"));
      properties.push(property(`${PREFIX}_undefined:blockSize`, quantization.block_size));
    } else {
      properties.push(property(`${PREFIX}granularity`, quantization.granularity.replaceAll("_", "-")));
    }
    properties.push(property(`${PREFIX}scale`, quantization.scales.length === 1
      ? quantization.scales[0] : JSON.stringify(quantization.scales)));
    properties.push(property(`${PREFIX}zeroPoint`, quantization.zero_points.length === 1
      ? quantization.zero_points[0] : JSON.stringify(quantization.zero_points)));
    if (["per_axis", "blocked"].includes(quantization.granularity)) {
      properties.push(property(`${PREFIX}axis`, quantization.axis));
    }
  }
  return {
    name: parameter.tensor_name || `${parameter.direction}_${parameter.ordinal}`,
    format: {
      dataType: String(parameter.dtype || "unknown").toLowerCase(),
      encoding: "raw",
    },
    shape: (parameter.shape || []).map((dimension) => Number.isSafeInteger(Number(dimension)) && Number(dimension) >= 0 ? Number(dimension) : null),
    properties,
  };
}

function validateParameter(row, direction, ordinal, errors) {
  if (!row || typeof row !== "object" || Array.isArray(row)) {
    errors.push(`${direction}[${ordinal}] must be an object.`);
    return;
  }
  if (!String(row.name || "").trim()) errors.push(`${direction}[${ordinal}].name is required.`);
  if (!String(row.format?.dataType || "").trim()) errors.push(`${direction}[${ordinal}].format.dataType is required by this preview profile.`);
  if (!Array.isArray(row.shape)) errors.push(`${direction}[${ordinal}].shape must be an array.`);
  if (!Array.isArray(row.properties)) errors.push(`${direction}[${ordinal}].properties must be an array.`);
  const names = new Set();
  for (const item of row.properties || []) {
    if (!String(item?.name || "").trim() || typeof item?.value !== "string") errors.push(`${direction}[${ordinal}] contains an invalid property.`);
    if (names.has(item?.name)) errors.push(`${direction}[${ordinal}] repeats property ${item?.name}.`);
    names.add(item?.name);
  }
}

function property(name, value) {
  return { name, value: String(value ?? "") };
}

function normalizeSha256(value) {
  const digest = String(value || "").trim().toLowerCase();
  return /^[a-f0-9]{64}$/.test(digest) ? digest : "";
}
