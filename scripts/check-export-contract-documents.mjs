import { readFileSync, writeFileSync } from "node:fs";
import {
  buildCycloneDxEvidenceDocument,
  buildDeploymentContractDocuments,
  buildInterfaceContractLedgerDocument,
  buildMissingProvenanceFieldSpecification,
  buildObservedFormulationDocument,
  buildRuntimeRequirementManifest,
  DEPLOYMENT_CONTRACT_FILES,
} from "../web/lib/report-export-contracts.js";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { validateCycloneDx20ParameterContractPreview } from "../web/lib/cyclonedx-20-preview.js";
import { buildPublicCycloneDxDocuments } from "../web/lib/public-cyclonedx-export.js";
import { analyzerContentVersion } from "../web/lib/cyclonedx-identity.js";
import {
  analyze_tflite_for_target,
  compute_delegation_repair,
  initSync,
} from "../pkg/tflite_wasm_audit.js";
import { assertCycloneDx17, cycloneDx17Validation } from "./cyclonedx-17-schema.mjs";
import { REFERENCE_CONTRACT_ANALYZER_METADATA } from "./reference-contract-provenance.mjs";

const SHA = "a".repeat(64);
const PROFILE_SHA = "b".repeat(64);
const GENERATED_AT = "2026-07-29T00:00:00.000Z";
const analysis = {
  filename: "quant_model.tflite",
  format: "tflite",
  model_sha256: SHA,
  file_size_bytes: 4096,
  operator_count: 3,
  tensor_count: 4,
  quantized_tensors: 4,
  per_channel_tensors: 0,
  quantization_status: {
    classification: "full_integer",
    label: "Full integer quantized",
    full_integer: true,
    compute_ops: 2,
    quantized_compute_ops: 2,
    quantized_compute_mac_percent: 1,
    op_state_counts: [
      { name: "quantized_compute", count: 2 },
      { name: "quantized_data_movement", count: 1 },
    ],
  },
  inputs: [{
    index: 0,
    name: "serving_default_input",
    dtype: "UINT8",
    shape: [1, 16, 16, 3],
    quant_scales: 1,
    quant_zero_points: 1,
    scale_sample: [0.0078125],
    zero_point_sample: [128],
    quantized_dimension: 0,
  }],
  outputs: [{
    index: 3,
    name: "probabilities",
    dtype: "UINT8",
    shape: [1, 2],
    quant_scales: 1,
    quant_zero_points: 1,
    scale_sample: [0.00390625],
    zero_point_sample: [0],
    quantized_dimension: 0,
  }],
  metadata_presence: {
    description: "TOCO Converted.",
    conversion_metadata_status: "not_present",
    conversion_metadata_entry_count: 0,
    metadata_model_version: "7",
    has_signature_defs: false,
    signature_count: 0,
    has_model_metadata: false,
    metadata_author: "",
    metadata_license: "",
    documented_preprocessing: false,
    output_semantics_documented: false,
    output_label_file_count: 0,
    verified_output0_label_file_count: 0,
  },
  runtime_compat: {
    effective_min_runtime_version: "2.2.0",
    runtime_floor_status: "complete_for_observed_builtin_op_versions",
    runtime_floor_evidence_class: "DERIVED",
    runtime_version_basis: "Pinned TFLite op-version map",
  },
  target_profile: {
    id: "android_mid_a55",
    label: "Android Cortex-A55",
    profile_sha256: PROFILE_SHA,
  },
  tensors: [
    {
      index: 0, name: "input", dtype: "UINT8", shape: [1, 16, 16, 3], constant_buffer: false,
      quant_scales: 1, quant_zero_points: 1, scale_sample: [0.0078125], zero_point_sample: [128],
    },
    {
      index: 1, name: "conv_weights", dtype: "UINT8", shape: [2, 1, 1, 3], constant_buffer: true,
      quant_scales: 1, quant_zero_points: 1, scale_sample: [0.01], zero_point_sample: [122],
    },
    {
      index: 2, name: "depthwise_weights", dtype: "UINT8", shape: [1, 1, 1, 2], constant_buffer: true,
      quant_scales: 1, quant_zero_points: 1, scale_sample: [0.02], zero_point_sample: [128],
    },
    {
      index: 3, name: "probabilities", dtype: "UINT8", shape: [1, 2], constant_buffer: false,
      quant_scales: 1, quant_zero_points: 1, scale_sample: [0.00390625], zero_point_sample: [0],
    },
  ],
  ops: [
    { index: 0, name: "CONV_2D", inputs: [0, 1], outputs: [3], xnnpack_chain_id: 0, xnnpack_build_requirement: "--define tflite_with_xnnpack_qu8=true; runtime build configuration not embedded" },
    { index: 1, name: "DEPTHWISE_CONV_2D", inputs: [3, 2], outputs: [3], xnnpack_chain_id: 0, xnnpack_build_requirement: "--define tflite_with_xnnpack_qu8=true; runtime build configuration not embedded" },
    { index: 2, name: "SOFTMAX", inputs: [3], outputs: [3], xnnpack_chain_id: -1, xnnpack_build_requirement: "" },
  ],
  weight_integrity: {
    exact_zero_kernel_slice_count: 11,
    zero_kernel_slice_details: [{ bias_int32_utilization_sample: [0.9756] }],
  },
  channel_vitality: {
    dual_mode_constant_output_channel_count: 14,
    mode_dependent_constant_output_channel_count: 1,
    ops: [{
      op_index: 0,
      op_name: "CONV_2D",
      mode_dependent_constant_output_channel_count: 1,
      mode_dependent_constant_channel_indices: [1],
      vitality_ledger_sha256: "9".repeat(64),
      top_channels: [{
        channel_index: 1,
        default_minimum_output_code: 0,
        default_maximum_output_code: 1,
        single_minimum_output_code: 0,
        single_maximum_output_code: 0,
      }],
    }],
  },
  preprocessing_realizability: {
    status: "assessed",
    evidence_class: "DERIVED",
    assessment_kind: "finite_domain",
    candidate_evaluation_count: 8,
    exact_tensor_realization_candidate_count: 2,
    non_exact_candidate_count: 6,
    exact_contract_ids: ["raw_storage_rgb", "artifact_affine_rgb"],
    best_non_exact_contract_id: "unit_interval_rgb",
    best_non_exact_unrealizable_element_count: 128,
    portfolio_ledger_sha256: "8".repeat(64),
    interpretation_boundary: "Static candidate portfolio.",
  },
  delegation_repair: {
    runtime_build_risks: [{
      id: "xnnpack_required_build_configuration_unbound_0",
      evidence_class: "CONDITIONAL_SOURCE_BACKED_CONFIGURATION_SCENARIO",
      required_build_configuration: "--define tflite_with_xnnpack_qu8=true; runtime build configuration not embedded",
      baseline_conditionally_delegatable_op_count: 2,
      affected_conditionally_delegatable_op_count: 2,
      affected_predicted_delegate_segment_count: 1,
      affected_conditionally_delegatable_mac_ratio: 1,
      absent_condition_remaining_conditionally_delegatable_op_count: 0,
      absent_condition_remaining_predicted_delegate_segment_count: 0,
      interpretation_boundary: "Conditional source-backed assignment consequence; runtime flag is not observed.",
    }],
  },
  requantization_fidelity: {
    assessed_channel_count: 32,
    single_rounding_encoding_divergence_channel_count: 2,
  },
  rounding_equivalence: {
    assessed_channel_count: 32,
    divergent_channel_count: 2,
    divergent_state_count_decimal: "128",
    maximum_absolute_output_delta: 1,
  },
};

const mlBomDocument = {
  $schema: "http://cyclonedx.org/schema/bom-1.7.schema.json",
  bomFormat: "CycloneDX",
  specVersion: "1.7",
  metadata: {
    component: {
      type: "machine-learning-model",
      name: analysis.filename,
    version: "unknown",
      "bom-ref": `model:${SHA.slice(0, 16)}`,
      hashes: [{ alg: "SHA-256", content: SHA }],
      properties: [{ name: "mlbom:model:schemaOrOpset", value: "TFLite schema 3" }],
    },
  },
};
const options = {
  generatedAt: GENERATED_AT,
  hash: SHA,
  fileSizeBytes: 4096,
  mlBomDocument,
};

const set = buildDeploymentContractDocuments(analysis, options);
expectEqual(set.schema, "deepbom.deployment_contract_export_set.v1.4", "export-set schema");
expectEqual(set.generated_at, GENERATED_AT, "stable generation timestamp");
expectEqual(Object.keys(set.documents).length, 7, "document count");
expectEqual(new Set(Object.values(set.files)).size, 7, "unique contract filenames");
expectEqual(set.files.interfaceContracts, DEPLOYMENT_CONTRACT_FILES.interfaceContracts, "interface contract filename");
expectEqual(set.files.runtime, DEPLOYMENT_CONTRACT_FILES.runtime, "runtime contract filename");
expectEqual(JSON.stringify(set), JSON.stringify(buildDeploymentContractDocuments(analysis, options)), "deterministic export document set");
expect(!JSON.stringify(set).includes("NaN") && !JSON.stringify(set).includes("Infinity"), "documents must not serialize non-finite values");
for (const [filename, digest] of Object.entries(set.integrity.member_sha256)) {
  const documentKey = Object.entries(set.files).find(([, value]) => value === filename)?.[0];
  const mappedKey = {
    cyclonedx: "cyclonedx_evidence",
    cyclonedx20Preview: "cyclonedx_2_0_parameter_contract_preview",
    artifactEnvelope: "artifact_evidence_envelope",
    interfaceContracts: "interface_contract_ledger",
    formulation: "observed_formulation",
    runtime: "runtime_requirement_manifest",
    missingFields: "missing_provenance_field_specification",
  }[documentKey];
  expectEqual(digest, sha256TextHex(JSON.stringify(set.documents[mappedKey], null, 2)), `${filename} member digest`);
}

const cycloneDx = set.documents.cyclonedx_evidence;
assertCycloneDx17(cycloneDx, "evidence BOM");
expectEqual(cycloneDx.$schema, "http://cyclonedx.org/schema/bom-1.7.schema.json", "CycloneDX official schema identifier");
expectEqual(cycloneDx.specVersion, "1.7", "CycloneDX specification version");
expectEqual(cycloneDx.metadata.component.type, "machine-learning-model", "CycloneDX component type");
expectEqual(cycloneDx.metadata.component.hashes[0].content, SHA, "CycloneDX artifact hash");
expectEqual(cycloneDx.metadata.component["bom-ref"], `deepbom-model-sha256-${SHA}`, "CycloneDX full-hash subject reference");
expectEqual(cycloneDx.metadata.component.version, "7", "declared model version");
expect(!Array.isArray(cycloneDx.metadata.tools), "CycloneDX 1.7 tools must use the object form.");
expectEqual(
  cycloneDx.metadata.tools.components[0].version,
  analyzerContentVersion(ANALYZER_METADATA.semanticVersion, ANALYZER_METADATA.buildCommit, ANALYZER_METADATA.buildContentSha256),
  "CycloneDX exact analyzer execution version",
);
expectEqual(cycloneDx.citations[0].attributedTo, cycloneDx.metadata.tools.components[0]["bom-ref"], "CycloneDX citation analyzer reference");
expectEqual(cycloneDx.metadata.tools.components[0].hashes[0].content, ANALYZER_METADATA.buildContentSha256, "CycloneDX tool bundle hash");
expectEqual(cycloneDx.metadata.authors[0].name, "Jun-Hwan Kwon", "CycloneDX human author");
expectEqual(cycloneDx.metadata.authors[0]["bom-ref"], "https://orcid.org/0000-0002-6464-3895", "CycloneDX author ORCID reference");
expectEqual(propertyMap(cycloneDx.properties).get("deepbom:documentAuthor:orcid"), "https://orcid.org/0000-0002-6464-3895", "CycloneDX author ORCID property");
expect(/^urn:uuid:[0-9a-f-]{36}$/.test(cycloneDx.serialNumber), "CycloneDX serialNumber must be a UUID URN.");
expectEqual(cycloneDx.metadata.component.modelCard.modelParameters.inputs.length, 1, "CycloneDX modelCard input contract");
const cdxProperties = propertyMap(cycloneDx.metadata.component.properties);
expectEqual(cdxProperties.get("deepbom:model:quantizationClassification"), "full_integer", "CycloneDX quantization classification");
expectEqual(cdxProperties.get("deepbom:model:perAxisQuantizationPresent"), "false", "CycloneDX per-axis presence");
expectEqual(cdxProperties.get("deepbom:model:primaryInputScale"), "0.0078125", "CycloneDX input scale");
expectEqual(cdxProperties.get("deepbom:model:primaryInputZeroPoint"), "128", "CycloneDX input zero-point");
expectEqual(propertyMap(cycloneDx.properties).get("deepbom:artifactEvidenceEnvelopeSha256"), set.documents.artifact_evidence_envelope.envelope_sha256, "CycloneDX envelope binding");
expectEqual(cycloneDx.dependencies[0].ref, cycloneDx.metadata.component["bom-ref"], "CycloneDX subject dependency reference");
expectEqual(cycloneDx.declarations.claims[0].target, cycloneDx.metadata.component["bom-ref"], "CycloneDX declaration target");
expect(cycloneDx.declarations.claims.length > 0, "CycloneDX findings should be projected as claims.");
expectEqual(cdxProperties.get("deepbom:runtime:necessaryFloor"), "2.2.0", "CycloneDX runtime floor");
expectEqual(cdxProperties.get("deepbom:finding:constantOutputChannels"), "14", "CycloneDX constant-output finding");
expectEqual(cdxProperties.get("deepbom:finding:exactZeroKernelSlices"), "11", "CycloneDX exact-zero finding");
expectEqual(cdxProperties.get("deepbom:finding:buildModeDependentChannels"), "1", "CycloneDX build-mode finding");
expectEqual(cdxProperties.get("deepbom:finding:maxInt32UtilizationRatio"), "0.9756", "CycloneDX INT32 utilization finding");
expectEqual(cdxProperties.get("deepbom:conformance:status"), "nonconformant", "CycloneDX current INT8 profile conformance");
expectEqual(cdxProperties.get("deepbom:conformance:legacyQu8Artifact"), "true", "CycloneDX legacy QU8 classification");
expectEqual(
  cdxProperties.get("deepbom:conformance:source"),
  "https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/quantization_spec",
  "CycloneDX current INT8 profile source",
);
expect(
  JSON.parse(cdxProperties.get("deepbom:conformance:violationCodes")).includes("weight_granularity_per_tensor_expected_per_axis"),
  "CycloneDX should expose the current-profile granularity violation.",
);
const modernAnalysis = structuredClone(analysis);
modernAnalysis.per_channel_tensors = 2;
modernAnalysis.metadata_presence.description = "";
modernAnalysis.metadata_presence.conversion_metadata_status = "parsed";
modernAnalysis.metadata_presence.converter_tensorflow_version = "2.21.0";
for (const tensor of modernAnalysis.tensors.filter((item) => [1, 2].includes(item.index))) {
  tensor.dtype = "INT8";
  tensor.quant_scales = 2;
  tensor.quant_zero_points = 2;
  tensor.scale_sample = [0.01, 0.02];
  tensor.zero_point_sample = [0, 0];
}
const modernProperties = propertyMap(
  buildCycloneDxEvidenceDocument(modernAnalysis, options).metadata.component.properties,
);
expectEqual(modernProperties.get("deepbom:conformance:status"), "conformant_for_assessed_operators", "modern per-axis INT8 profile conformance");
expectEqual(
  JSON.parse(cdxProperties.get("deepbom:preprocessing:exactContractIds")).join(","),
  "raw_storage_rgb,artifact_affine_rgb",
  "CycloneDX preprocessing contract candidates",
);
const activationPath = JSON.parse(cdxProperties.get("deepbom:model:activationPath"));
expectEqual(activationPath.all_operator_denominator, 3, "activation-path all-op denominator");
expectEqual(activationPath.compute_scope.denominator, 2, "activation-path compute denominator");
expectEqual(activationPath.compute_scope.quantized_operators, 2, "activation-path quantized compute count");
const inputContracts = set.documents.interface_contract_ledger.parameters
  .filter((row) => row.direction === "input")
  .map((row) => ({ ...row.quantization }));
expectEqual(inputContracts[0].scale_count, 1, "input scale count");
expectEqual(inputContracts[0].scale_values_complete, true, "input scale vector completeness");
expectEqual(inputContracts[0].zero_point_count, 1, "input zero-point count");
expectEqual(inputContracts[0].zero_point_values_complete, true, "input zero-point vector completeness");
expectEqual(inputContracts[0].status, "complete", "input affine contract status");
expectEqual(inputContracts[0].cardinality_status, "valid", "input affine cardinality");
expectEqual(inputContracts[0].scalar_real_code_domain.real_min, -1, "input scalar real domain minimum");
expect(!cdxProperties.has("deepbom:model:inputContracts") && !cdxProperties.has("deepbom:model:outputContracts"), "CycloneDX profile v1.3 should reference the canonical interface ledger instead of duplicating full contract JSON.");
expectEqual(propertyMap(cycloneDx.metadata.component.modelCard.properties).get("deepbom:modelCard:interfaceContractLedgerSha256"), set.documents.interface_contract_ledger.ledger_sha256, "modelCard ledger hash reference");
expectEqual(cdxProperties.get("deepbom:model:completeAffineInterfaceCount"), "2", "complete affine interface count");
expectEqual(cdxProperties.get("deepbom:model:invalidOrIncompleteInterfaceCount"), "0", "invalid affine interface count");
expectExactSet(
  cycloneDx.metadata.component.externalReferences.map((item) => item.type),
  ["evidence", "formulation", "configuration"],
  "CycloneDX external-reference types",
);
expect(
  !cycloneDx.metadata.component.externalReferences.some((item) => ["engineering_evidence.json", "engineering_report.md"].includes(item.url)),
  "Standalone contract export must not contain references to files absent from the contract pack.",
);
for (const reference of cycloneDx.metadata.component.externalReferences) {
  expect(Object.values(set.files).includes(reference.url), `Contract-pack reference ${reference.url} must resolve to a package member.`);
}
for (const reference of cycloneDx.metadata.component.externalReferences.filter((item) =>
  [DEPLOYMENT_CONTRACT_FILES.runtime, DEPLOYMENT_CONTRACT_FILES.missingFields].includes(item.url),
)) expectEqual(reference.hashes.length, 1, `${reference.url} should be content-hash bound`);
expect(
  cycloneDx.metadata.component.properties.every((item) => item.value !== ""),
  "CycloneDX properties must omit unavailable values instead of emitting empty strings.",
);
const noDeclaredVersion = buildCycloneDxEvidenceDocument({
  ...analysis,
  metadata_presence: { ...analysis.metadata_presence, metadata_model_version: "" },
}, {
  ...options,
  mlBomDocument: {
    ...mlBomDocument,
    metadata: {
      component: { ...mlBomDocument.metadata.component, version: "unknown" },
    },
  },
});
expectEqual(noDeclaredVersion.metadata.component.version, `sha256-${SHA}`, "Unknown model versions use the complete artifact hash as a content version.");
expectEqual(propertyMap(noDeclaredVersion.metadata.component.properties).get("deepbom:model:versionBasis"), "full_artifact_sha256", "hash-derived model version basis");
const noFileSizeAnalysis = { ...analysis, file_size_bytes: undefined };
const noFileSizeOptions = { ...options, fileSizeBytes: undefined };
const noFileSizeProperties = propertyMap(
  buildCycloneDxEvidenceDocument(noFileSizeAnalysis, noFileSizeOptions).metadata.component.properties,
);
expect(!noFileSizeProperties.has("deepbom:model:fileSizeBytes"), "Unavailable file size must not be emitted as zero.");

const formulation = set.documents.observed_formulation;
assertCycloneDx17(formulation, "formulation BOM");
expectEqual(formulation.formulation.length, 1, "observed-only formula count");
const observedProperties = propertyMap(formulation.formulation[0].properties);
expectEqual(observedProperties.get("deepbom:formulation:converterFamily"), "tensorflow-toco-legacy", "TOCO family derivation");
expectEqual(observedProperties.get("deepbom:formulation:converterFamilyEvidenceClass"), "CONVERGENT_ARTIFACT_FINGERPRINT", "convergent TOCO fingerprint");
expectEqual(propertyMap(formulation.properties).get("deepbom:formulation:comparisonStatus"), "not_assessable_declared_formulation_not_provided", "missing declaration comparison status");
expect(!propertyMap(formulation.properties).has("deepbom:formulation:mismatchCount"), "Not-assessable comparison must omit mismatchCount.");
expect(!observedProperties.has("deepbom:formulation:converterVersion"), "Unavailable converter version must be omitted.");
expect(formulation.metadata.component.externalReferences.every((item) => item.hashes?.length === 1), "Formulation dependencies should be hash-bound.");
const mismatchFormulation = buildObservedFormulationDocument(analysis, {
  ...options,
  declaredFormulation: { converter_family: "tensorflow-lite-mlir", converter_version: "2.21.0" },
});
expectEqual(mismatchFormulation.formulation.length, 2, "declared and observed formula count");
expectEqual(propertyMap(mismatchFormulation.properties).get("deepbom:formulation:comparisonStatus"), "mismatch", "declared/observed mismatch");
expectEqual(propertyMap(mismatchFormulation.properties).get("deepbom:formulation:mismatchCount"), "1", "comparable mismatch count");

const interfaceContracts = set.documents.interface_contract_ledger;
expectEqual(interfaceContracts.schema, "deepbom.interface_quantization_contracts.v1.3", "interface ledger schema");
expectEqual(interfaceContracts.subject.sha256, SHA, "interface ledger artifact binding");
expectEqual(interfaceContracts.subject.component_bom_ref, `deepbom-model-sha256-${SHA}`, "interface ledger full-hash subject reference");
expectEqual(interfaceContracts.parameter_count, 2, "interface ledger parameter count");
expectEqual(interfaceContracts.quantized_parameter_count, 2, "interface ledger complete affine count");
expectEqual(interfaceContracts.boundary_contract?.status, "fully_affine_quantized", "interface boundary aggregate status");
expectEqual(interfaceContracts.boundary_contract?.inputs?.status, "fully_affine_quantized", "input boundary aggregate status");
expectEqual(interfaceContracts.boundary_contract?.outputs?.status, "fully_affine_quantized", "output boundary aggregate status");
expectEqual(interfaceContracts.source_preprocessing_contract_status, "not_embedded_in_artifact", "normalized source preprocessing status");
expect(
  interfaceContracts.boundary_contract?.not_established_by_this_contract?.includes("resize_interpolation"),
  "interface boundary contract must preserve its preprocessing limitation",
);
expectEqual(
  JSON.stringify(interfaceContracts),
  JSON.stringify(buildInterfaceContractLedgerDocument(analysis, options)),
  "interface ledger deterministic direct builder",
);
const {
  generated_at: _interfaceGeneratedAt,
  subject: _interfaceSubject,
  provenance: _interfaceProvenance,
  ledger_sha256: ledgerSha256,
  ledger_hash: ledgerHash,
  ...ledgerPayload
} = interfaceContracts;
expectEqual(ledgerSha256, sha256TextHex(canonicalJson(ledgerPayload)), "interface ledger independent canonical hash");
expectEqual(ledgerHash.sha256, ledgerSha256, "interface ledger hash descriptor");
expectEqual(ledgerHash.payload_contract.selection, "containing_object_after_exclusions", "ledger hash payload selection");
expect(ledgerHash.payload_contract.excluded_json_pointers.includes("/ledger_sha256"), "ledger hash descriptor excludes its own digest field");
for (const parameter of interfaceContracts.parameters) {
  const quantization = parameter.quantization;
  const independentPayload = {
    direction: parameter.direction,
    ordinal: parameter.ordinal,
    tensor_name: parameter.tensor_name,
    dtype: parameter.dtype,
    shape: parameter.shape,
    shape_signature: parameter.shape_signature,
    quantization: {
      status: quantization.status,
      scheme: quantization.scheme,
      granularity: quantization.granularity,
      parameterization: quantization.parameterization,
      scales: quantization.scales,
      zero_points: quantization.zero_points,
      axis: quantization.axis,
      block_size: quantization.block_size,
    },
  };
  expectEqual(parameter.interface_contract_sha256, sha256TextHex(canonicalJson(independentPayload)), `${parameter.parameter_id} independent ABI hash`);
  expectEqual(parameter.interface_contract_hash.sha256, parameter.interface_contract_sha256, `${parameter.parameter_id} ABI hash descriptor`);
  expect(/^[a-f0-9]{64}$/.test(parameter.interface_contract_hash.legacy_v1_1.sha256), `${parameter.parameter_id} legacy v1.1 ABI hash`);
  expect(!Object.hasOwn(parameter, "legacy_hashes") && !Object.hasOwn(quantization, "legacy_hashes"), `${parameter.parameter_id} legacy hashes have one canonical location`);
}
const serializedInput = interfaceContracts.parameters.find((row) => row.direction === "input");
const serializedOutput = interfaceContracts.parameters.find((row) => row.direction === "output");
expectEqual(serializedInput.shape_signature, null, "absent shape signature is not confused with rank zero");
expectEqual(serializedInput.source_data_to_tensor_preprocessing_status, "not_embedded_in_artifact", "input preprocessing status vocabulary");
expectEqual(serializedOutput.source_data_to_tensor_preprocessing_status, "not_applicable", "output preprocessing applicability");
expectEqual(serializedInput.quantization.axis, null, "per-tensor quantization has no semantic axis");
expectEqual(serializedInput.quantization.axis_applicable, false, "per-tensor axis applicability");
expectEqual(serializedInput.quantization.quantized_dimension, 0, "serialized default quantized dimension remains preserved separately");
expectEqual(JSON.stringify(serializedInput.quantization.scale_tensor_shape), "[1]", "per-tensor scale value-vector shape");
expectEqual(JSON.stringify(serializedInput.quantization.zero_point_tensor_shape), "[1]", "per-tensor zero-point value-vector shape");
const previewValidation = validateCycloneDx20ParameterContractPreview(set.documents.cyclonedx_2_0_parameter_contract_preview);
expect(previewValidation.valid, `CycloneDX 2.0 preview validation: ${previewValidation.errors.join("; ")}`);
expectEqual(
  set.documents.cyclonedx_2_0_parameter_contract_preview.metadata.component.modelCard.modelParameters.inputs[0].name,
  analysis.inputs[0].name,
  "CycloneDX 2.0 named input binding",
);

const runtime = set.documents.runtime_requirement_manifest;
expectEqual(runtime.schema, "deepbom.runtime_requirement_manifest.v1.3", "runtime manifest schema");
expectEqual(runtime.backend_build_requirements.length, 1, "runtime build requirement count");
expectEqual(runtime.backend_build_requirements[0].binding.status, "pending_runtime_evidence_not_imported", "unobserved build configuration status");
expectEqual(runtime.backend_build_requirements[0].conditional_impact.absent_condition_remaining_conditionally_delegatable_ops, 0, "conditional coverage collapse");
expectEqual(runtime.backend_build_requirements[0].conditional_impact.affected_conditionally_delegatable_mac_ratio, 1, "conditionally delegatable MAC ratio unit");
expectEqual(runtime.backend_build_requirements[0].conditional_impact.affected_conditionally_delegatable_mac_percent, 100, "conditionally delegatable MAC percent unit");
expectEqual(runtime.numerical_abi_requirements[0].exact_static_impact.divergent_channels, 2, "rounding divergence count");
expectEqual(runtime.numerical_abi_requirements[0].exact_static_impact.build_mode_dependent_constant_channels, 1, "rounding mode-dependent constant count");
expectEqual(runtime.numerical_abi_requirements[0].exact_static_impact.representative_witness.channel_index, 1, "rounding representative coordinate");
expectEqual(runtime.ci_gate.result, "pending", "unbound runtime evidence should be pending");
expectEqual(runtime.artifact_static_gate.result, "review", "legacy QU8 profile result should enter deterministic review");
expect(
  runtime.artifact_static_gate.items.some((item) => item.code === "current_litert_int8_profile_nonconformance"),
  "runtime manifest should expose the deterministic current-profile nonconformance.",
);
expect(runtime.ci_gate.pending_requirements.some((item) => item.code === "runtime_floor_unbound"), "runtime floor should be pending");
expect(runtime.ci_gate.pending_requirements.some((item) => item.code === "required_build_configuration"), "XNNPACK flag should be pending");
expect(runtime.ci_gate.pending_requirements.some((item) => item.code === "numerical_abi_mode"), "rounding ABI should be pending");
expect(runtime.ci_gate.pending_requirements.some((item) => item.code === "production_preprocessing_binding"), "preprocessing implementation should be pending");
expect(runtime.ci_gate.pending_requirements.some((item) => item.code === "production_interface_contract_binding"), "interface encoder/decoder should be pending");
expectEqual(runtime.interface_contract_requirement.complete_affine_parameter_count, 2, "runtime interface contract count");
expectEqual(runtime.onnx_execution_provider_source_profiles.status, "not_applicable_non_onnx_artifact", "TFLite ONNX EP applicability");
expectEqual(runtime.target_independent_requirements.backend_build_requirements.length, 1, "target-independent build requirement");
expectEqual(runtime.target_specific_impact.backend_build_requirement_impacts.length, 1, "target-specific impact projection");

const boundRuntimeEvidence = {
  schema: "deepbom.runtime_assignment.v1.9",
  evidence_class: "OBSERVED_RUNTIME",
  artifact_sha256: SHA,
  target_profile_sha256: PROFILE_SHA,
  runtime: {
    name: "TensorFlow Lite",
    version: "2.21.0",
    backend: "XNNPACK",
    build: "bazel build --define tflite_with_xnnpack_qu8=true; TFLITE_SINGLE_ROUNDING=1",
    binary_sha256: "c".repeat(64),
  },
  source: { kind: "deepbom_native_runtime_capture" },
  selector_context: {
    build: {
      compile_definitions: [{ name: "TFLITE_SINGLE_ROUNDING", value: "1" }],
    },
  },
};
const productionInterfaceContract = {
  artifact_sha256: SHA,
  implementation_sha256: "6".repeat(64),
  parameters: runtime.interface_contract_requirement.parameters.map((row) => ({
    direction: row.direction,
    ordinal: row.ordinal,
    interface_contract_sha256: row.interface_contract_sha256,
  })),
};
const boundRuntime = buildRuntimeRequirementManifest(analysis, {
  ...options,
  runtimeAssignmentEvidence: boundRuntimeEvidence,
  productionPreprocessingContract: {
    contract_id: "raw_storage_rgb",
    implementation_sha256: "7".repeat(64),
  },
  productionInterfaceContract,
});
expectEqual(boundRuntime.necessary_runtime_floor.coverage_complete, true, "complete runtime-floor map coverage");
expectEqual(boundRuntime.necessary_runtime_floor.observed_floor_satisfied, true, "bound runtime floor");
expectEqual(boundRuntime.backend_build_requirements[0].binding.status, "declared_present", "bound XNNPACK flag");
expectEqual(boundRuntime.numerical_abi_requirements[0].declared_mode, "enabled", "bound rounding mode");
expectEqual(boundRuntime.interface_contract_requirement.production_binding.status, "bound_exact_contract", "bound interface contract");
expectEqual(boundRuntime.runtime_binding_gate.result, "pass", "fully bound runtime-evidence gate");
expectEqual(boundRuntime.ci_gate.result, "review", "bound legacy artifact retains deterministic profile review");
const mismatchedInterface = structuredClone(productionInterfaceContract);
mismatchedInterface.parameters[0].interface_contract_sha256 = "e".repeat(64);
const interfaceMismatchRuntime = buildRuntimeRequirementManifest(analysis, {
  ...options,
  runtimeAssignmentEvidence: boundRuntimeEvidence,
  productionPreprocessingContract: { contract_id: "raw_storage_rgb", implementation_sha256: "7".repeat(64) },
  productionInterfaceContract: mismatchedInterface,
});
expect(
  interfaceMismatchRuntime.ci_gate.blockers.some((item) => item.code === "production_interface_contract_binding"),
  "A production encoder/decoder contract mismatch must block the runtime gate.",
);
const mismatchedRuntime = buildRuntimeRequirementManifest(analysis, {
  ...options,
  runtimeAssignmentEvidence: { ...boundRuntimeEvidence, artifact_sha256: "d".repeat(64) },
  productionPreprocessingContract: {
    contract_id: "raw_storage_rgb",
    implementation_sha256: "7".repeat(64),
  },
  productionInterfaceContract,
});
expectEqual(mismatchedRuntime.ci_gate.result, "block", "runtime evidence artifact mismatch");
expect(
  mismatchedRuntime.ci_gate.blockers.some((item) => item.code === "runtime_evidence_artifact_hash_mismatch"),
  "runtime evidence artifact mismatch should be an explicit blocker.",
);
const incompleteRuntimeEvidence = structuredClone(boundRuntimeEvidence);
delete incompleteRuntimeEvidence.artifact_sha256;
delete incompleteRuntimeEvidence.target_profile_sha256;
delete incompleteRuntimeEvidence.runtime.binary_sha256;
const incompleteBinding = buildRuntimeRequirementManifest(analysis, {
  ...options,
  runtimeAssignmentEvidence: incompleteRuntimeEvidence,
  productionPreprocessingContract: {
    contract_id: "raw_storage_rgb",
    implementation_sha256: "7".repeat(64),
  },
  productionInterfaceContract,
});
expectEqual(incompleteBinding.runtime_binding_gate.result, "pending", "incomplete imported runtime identity");
expectExactSet(
  incompleteBinding.ci_gate.pending_requirements
    .filter((item) => item.code.includes("hash_unbound"))
    .map((item) => item.code),
  [
    "runtime_evidence_artifact_hash_unbound",
    "runtime_evidence_target_profile_hash_unbound",
    "runtime_binary_hash_unbound",
  ],
  "incomplete runtime identity pending requirements",
);
const partialFloorRuntime = buildRuntimeRequirementManifest({
  ...analysis,
  runtime_compat: {
    ...analysis.runtime_compat,
    effective_min_runtime_version: "",
    derived_min_runtime_version: "2.2.0",
    runtime_floor_status: "partial_unmapped_builtin_op_versions",
  },
}, {
  ...options,
  runtimeAssignmentEvidence: boundRuntimeEvidence,
  productionPreprocessingContract: {
    contract_id: "raw_storage_rgb",
    implementation_sha256: "7".repeat(64),
  },
  productionInterfaceContract,
});
expectEqual(partialFloorRuntime.necessary_runtime_floor.coverage_complete, false, "partial runtime-floor map coverage");
expect(
  partialFloorRuntime.ci_gate.blockers.some((item) => item.code === "runtime_floor_incomplete"),
  "A partial runtime-floor map must remain fail-closed even when the observed runtime exceeds its known lower bound.",
);

const missing = set.documents.missing_provenance_field_specification;
expectEqual(missing.schema, "deepbom.missing_provenance_field_specification.v1.1", "missing-field schema");
expectEqual(missing.release_manifest_fields.length, 10, "release-manifest field count");
expectEqual(missing.artifact_embedded_fields.length, 8, "artifact-embedded field count");
expectEqual(missing.fields.length, 18, "total provenance field count");
expectEqual(missing.release_manifest_status_counts.partial, 4, "partial artifact-derived release lineage count");
expectEqual(missing.release_manifest_status_counts.missing, 6, "missing release lineage count");
expectEqual(missing.ci_gate.noncompliant_field_count, missing.status_counts.partial + missing.status_counts.missing, "noncompliant lineage count");
expectEqual(field(missing, "source_checkpoint_sha256").status, "missing", "source checkpoint gap");
expectEqual(field(missing, "quantization_calibration_configuration").status, "partial", "artifact-result versus calibration-recipe distinction");
expectEqual(field(missing, "converter_version").status, "partial", "converter family should survive exact-version absence");
expectEqual(field(missing, "converter_version").evidence_class, "CONVERGENT_ARTIFACT_FINGERPRINT", "converter fingerprint evidence class");
expect(field(missing, "representative_dataset_id").impact.length > 20, "provenance fields should state operational impact");
expect(field(missing, "representative_dataset_id").searched_pointers.length > 0, "provenance fields should record searched pointers");
expectEqual(missing.provenance.rulepack_sha256, ANALYZER_METADATA.rulepackSha256, "missing-field rulepack binding");

const released = buildMissingProvenanceFieldSpecification(analysis, {
  ...options,
  releaseManifest: {
    source_checkpoint_sha256: "d".repeat(64),
    export_framework: { name: "TensorFlow", version: "2.21.0" },
    converter_version: "2.21.0",
    export_configuration: { keepdims: true },
    quantization_configuration: { scheme: "int8" },
    representative_dataset_id: "dataset@sha256:e",
    build_pipeline_id: "pipeline/run/42",
    software_release_id: "mobile-app@7",
    model_requirement_id: "REQ-ML-17",
    previous_artifact_sha256: "f".repeat(64),
  },
});
expectEqual(released.release_manifest_status_counts.missing, 0, "release-context gap closure");
expectEqual(released.release_manifest_status_counts.partial, 0, "release-context partial closure");
expectEqual(released.ci_gate.release_manifest_result, "complete", "complete release-manifest gate");
expectEqual(released.ci_gate.artifact_embedded_result, "incomplete", "artifact-embedded gaps remain separate");
const initialRelease = buildMissingProvenanceFieldSpecification(analysis, {
  ...options,
  releaseManifest: { no_predecessor: true },
});
expectEqual(field(initialRelease, "previous_release_artifact_sha256").status, "not_applicable", "explicit no-predecessor declaration");

const onnx = {
  ...analysis,
  filename: "model.onnx",
  format: "onnx",
  quantized_tensors: 0,
  per_channel_tensors: 0,
  quantization_status: {
    classification: "not_quantized_float",
    label: "Not quantized",
    full_integer: false,
    quantized_compute_mac_percent: 0,
    op_state_counts: [{ name: "none", count: 1 }],
  },
  metadata_presence: { producer_name: "torch", producer_version: "2.8.0", model_version: "1" },
  ort_compatibility_evidence: {
    runtime_floor: { minimum_ort_version: "1.17.0", status: "complete", evidence_class: "DERIVED" },
    execution_providers: [{
      execution_provider: "CPUExecutionProvider",
      source_scope: "pinned ORT kernel registry",
      assessed_op_count: 1,
      source_candidate_after_artifact_precheck_count: 1,
    }],
  },
  delegation_repair: null,
  requantization_fidelity: null,
  rounding_equivalence: null,
  ops: [{ index: 0, name: "Conv" }],
};
const onnxRuntime = buildRuntimeRequirementManifest(onnx, options);
expectEqual(onnxRuntime.necessary_runtime_floor.runtime, "ONNX Runtime", "ONNX runtime family");
expectEqual(onnxRuntime.onnx_execution_provider_source_profiles.status, "assessed", "ONNX EP applicability");
expectEqual(onnxRuntime.onnx_execution_provider_source_profiles.profiles.length, 1, "ONNX EP source profile");
expectEqual(onnxRuntime.numerical_abi_requirements.length, 0, "TFLite numerical ABI suppression for ONNX");
expectEqual(buildMissingProvenanceFieldSpecification(onnx, options).fields.find((item) => item.id === "representative_dataset_id").status, "not_applicable", "float calibration dataset applicability");

for (const document of [
  buildCycloneDxEvidenceDocument(analysis, options),
  buildObservedFormulationDocument(analysis, options),
]) assertCycloneDx17(document, "direct CycloneDX builder");

const licensedAnalysis = structuredClone(analysis);
licensedAnalysis.metadata_presence = {
  ...licensedAnalysis.metadata_presence,
  metadata_model_description: "Hash-bound clinical imaging deployment model.",
  metadata_author: "Artifact Author",
  metadata_license: "Apache-2.0",
  producer_name: "Artifact Producer",
};
const licensedCanonical = buildCycloneDxEvidenceDocument(licensedAnalysis, options);
const licensedPublic = buildPublicCycloneDxDocuments(licensedAnalysis, options).documents.cyclonedx_evidence;
for (const [document, label] of [[licensedCanonical, "canonical"], [licensedPublic, "public"]]) {
  assertCycloneDx17(document, `${label} licensed CycloneDX builder`);
  const component = document.metadata.component;
  expectEqual(component.description, "Hash-bound clinical imaging deployment model.", `${label} description projection`);
  expectEqual(component.authors?.[0]?.name, "Artifact Author", `${label} author projection`);
  expect(!("manufacturer" in component), `${label} producer_name must not be misrepresented as a model manufacturer`);
  expectEqual(component.licenses?.[0]?.license?.name, "Apache-2.0", `${label} license projection`);
}
const undeclaredLicense = buildCycloneDxEvidenceDocument({
  ...analysis,
  metadata_presence: { ...analysis.metadata_presence, metadata_license: "not_declared" },
}, options);
expect(!("licenses" in undeclaredLicense.metadata.component), "Undeclared license placeholders must not be projected as licenses.");

const publicCycloneDx = buildPublicCycloneDxDocuments(analysis, options);
assertCycloneDx17(publicCycloneDx.documents.cyclonedx_evidence, "public standalone CycloneDX 1.7 builder");
const publicPreviewValidation = validateCycloneDx20ParameterContractPreview(
  publicCycloneDx.documents.cyclonedx_2_0_parameter_contract_preview,
);
expect(publicPreviewValidation.valid, `public CycloneDX 2.0 preview validation: ${publicPreviewValidation.errors.join("; ")}`);
const publicProperties = propertyMap(publicCycloneDx.documents.cyclonedx_evidence.metadata.component.properties);
const publicLedger = JSON.parse(publicProperties.get("deepbom:model:interfaceContractLedger"));
expectEqual(publicLedger.ledger_sha256, publicProperties.get("deepbom:model:interfaceContractLedgerSha256"), "public interface ledger binding");
expectEqual(publicLedger.parameter_count, 2, "public external interface cardinality");

initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const sampleFilename = "mobilenet_v2_1.0_224_quant.tflite";
const sampleBytes = new Uint8Array(readFileSync(`web/samples/${sampleFilename}`));
const sampleAnalysis = analyze_tflite_for_target(sampleBytes, sampleFilename, "android_mid_a55");
const sampleRepair = compute_delegation_repair(sampleBytes, sampleFilename, "android_mid_a55");
sampleAnalysis.model_sha256 = sampleRepair.artifact_sha256;
sampleAnalysis.delegation_repair = sampleRepair;
const sampleSet = buildDeploymentContractDocuments(sampleAnalysis, {
  generatedAt: GENERATED_AT,
  hash: sampleRepair.artifact_sha256,
  fileSizeBytes: sampleBytes.byteLength,
  analyzerMetadata: REFERENCE_CONTRACT_ANALYZER_METADATA,
});
const sampleCdx = sampleSet.documents.cyclonedx_evidence;
const sampleCdxProperties = propertyMap(sampleCdx.metadata.component.properties);
const sampleFormulationProperties = propertyMap(sampleSet.documents.observed_formulation.formulation[0].properties);
const sampleRuntime = sampleSet.documents.runtime_requirement_manifest;
assertCycloneDx17(sampleCdx, "actual legacy MobileNet evidence BOM");
assertCycloneDx17(sampleSet.documents.observed_formulation, "actual legacy MobileNet formulation BOM");
const invalidTools = structuredClone(sampleCdx);
invalidTools.metadata.tools = "deepbom";
expect(!cycloneDx17Validation(invalidTools).valid, "official schema must reject a non-object, non-legacy-array tools value");
const invalidModelCard = structuredClone(sampleCdx);
invalidModelCard.metadata.component.modelCard.unrecognized = true;
expect(!cycloneDx17Validation(invalidModelCard).valid, "official schema must reject unknown modelCard fields");
const invalidSpecVersion = structuredClone(sampleCdx);
invalidSpecVersion.specVersion = "1.6";
expect(!cycloneDx17Validation(invalidSpecVersion).valid, "official schema must reject a mismatched specVersion");
expectEqual(sampleCdx.metadata.component.hashes[0].content, sampleRepair.artifact_sha256, "actual sample artifact hash");
expectEqual(sampleCdxProperties.get("deepbom:finding:constantOutputChannels"), "14", "actual sample constant-output channels");
expectEqual(sampleCdxProperties.get("deepbom:finding:exactZeroKernelSlices"), "11", "actual sample exact-zero kernels");
expectEqual(sampleCdxProperties.get("deepbom:finding:buildModeDependentChannels"), "1", "actual sample build-mode channel");
const sampleHighFindingCount = buildFindingsRegister(sampleAnalysis, {
  analyzerMetadata: REFERENCE_CONTRACT_ANALYZER_METADATA,
})
  .filter((item) => item.technical_priority === "High").length;
expectEqual(
  sampleCdxProperties.get("deepbom:finding:highSeverityCount"),
  String(sampleHighFindingCount),
  "actual sample authoritative high-priority findings",
);
expectEqual(sampleCdxProperties.get("deepbom:conformance:status"), "nonconformant", "actual sample current INT8-profile status");
expectExactSet(
  JSON.parse(sampleCdxProperties.get("deepbom:conformance:violationCodes")),
  [
    "weight_dtype_expected_int8",
    "weight_granularity_per_tensor_expected_per_axis",
    "weight_zero_point_nonzero_expected_zero",
  ],
  "actual sample current INT8-profile violations",
);
expectExactSet(
  JSON.parse(sampleCdxProperties.get("deepbom:preprocessing:exactContractIds")),
  ["raw_storage_rgb", "raw_storage_bgr", "artifact_affine_rgb", "center_128_div_128_rgb"],
  "actual sample exact preprocessing candidates",
);
expectEqual(sampleFormulationProperties.get("deepbom:formulation:converterFamily"), "tensorflow-toco-legacy", "actual sample converter family");
expectEqual(sampleFormulationProperties.get("deepbom:formulation:converterFamilyEvidenceClass"), "CONVERGENT_ARTIFACT_FINGERPRINT", "actual sample converter evidence");
expectEqual(sampleRuntime.backend_build_requirements[0].conditional_impact.affected_conditionally_delegatable_ops, 64, "actual sample QU8 affected ops");
expectEqual(sampleRuntime.backend_build_requirements[0].conditional_impact.affected_conditionally_delegatable_mac_ratio, 1, "actual sample QU8 affected MAC ratio");
expectEqual(sampleRuntime.backend_build_requirements[0].conditional_impact.affected_conditionally_delegatable_mac_percent, 100, "actual sample QU8 affected MAC percent");
expectEqual(sampleRuntime.backend_build_requirements[0].conditional_impact.absent_condition_remaining_conditionally_delegatable_ops, 0, "actual sample QU8 coverage collapse");
expectEqual(
  sampleRuntime.ci_gate.pending_requirements.find((item) => item.code === "required_build_configuration")?.severity,
  "critical",
  "actual sample QU8 pending severity",
);
expectEqual(sampleRuntime.ci_gate.result, "pending", "actual sample unbound runtime gate");
for (const document of [
  sampleSet.documents.runtime_requirement_manifest,
  sampleSet.documents.missing_provenance_field_specification,
]) {
  expectEqual(document.provenance.rulepack_sha256, REFERENCE_CONTRACT_ANALYZER_METADATA.rulepackSha256, "reference sample canonical rulepack hash");
  expectEqual(document.provenance.analyzer_semantic_version, REFERENCE_CONTRACT_ANALYZER_METADATA.semanticVersion, "reference sample canonical analyzer version");
}
const exampleDir = "reference/cyclonedx/1.7/examples/mobilenet-v2-quant";
const storedExamples = {};
const updateReferences = process.argv.includes("--update-reference");
for (const [key, filename] of Object.entries(sampleSet.files)) {
  const documentKey = {
    cyclonedx: "cyclonedx_evidence",
    cyclonedx20Preview: "cyclonedx_2_0_parameter_contract_preview",
    artifactEnvelope: "artifact_evidence_envelope",
    interfaceContracts: "interface_contract_ledger",
    formulation: "observed_formulation",
    runtime: "runtime_requirement_manifest",
    missingFields: "missing_provenance_field_specification",
  }[key];
  if (updateReferences) writeFileSync(`${exampleDir}/${filename}`, `${JSON.stringify(sampleSet.documents[documentKey], null, 2)}\n`);
  const stored = JSON.parse(readFileSync(`${exampleDir}/${filename}`, "utf8"));
  storedExamples[filename] = stored;
  const storedSubjectHash = stored.metadata?.component?.hashes?.[0]?.content || stored.subject?.sha256 || stored.identity?.sha256;
  expectEqual(storedSubjectHash, sampleRepair.artifact_sha256, `${filename} public artifact binding`);
  if (filename.endsWith(".cdx.json") && key !== "cyclonedx20Preview") assertCycloneDx17(stored, `${filename} stored public example`);
  if (key === "cyclonedx20Preview") {
    const validation = validateCycloneDx20ParameterContractPreview(stored);
    expect(validation.valid, `${filename} stored preview validation: ${validation.errors.join("; ")}`);
  }
  if (key === "cyclonedx") {
    expectEqual(JSON.stringify(stored.metadata.component.modelCard), JSON.stringify(sampleSet.documents[documentKey].metadata.component.modelCard), `${filename} model-card contracts`);
  }
  expectEqual(canonicalJson(stored), canonicalJson(sampleSet.documents[documentKey]), `${filename} complete canonical reference document`);
}
for (const source of [DEPLOYMENT_CONTRACT_FILES.cyclonedx, DEPLOYMENT_CONTRACT_FILES.formulation]) {
  for (const reference of storedExamples[source].metadata.component.externalReferences) {
    const declaredHash = reference.hashes?.find((item) => item.alg === "SHA-256")?.content;
    if (!declaredHash) continue;
    const sibling = storedExamples[reference.url];
    expect(Boolean(sibling), `stored public example should include hash-bound ${reference.url}`);
    expectEqual(sha256TextHex(JSON.stringify(sibling, null, 2)), declaredHash, `${source} -> ${reference.url} stored sibling hash`);
  }
}

const html = readFileSync("web/index.html", "utf8");
const app = readFileSync("web/app.js", "utf8");
const worker = readFileSync("worker/index.js", "utf8");
const rawEntry = readFileSync("web/lib/report-raw-entry.js", "utf8");
const exportView = readFileSync("web/lib/export-contract-view.js", "utf8");
for (const id of [
  "downloadCycloneDxEvidence",
  "downloadCycloneDx20Preview",
  "downloadObservedFormulation",
  "downloadRuntimeRequirements",
  "downloadMissingProvenance",
  "downloadContractPack",
]) expect(html.includes(`id="${id}"`), `Export tab should include #${id}.`);
expect(html.includes('data-module-tab="export_contracts"') && html.includes('data-module-panel="export_contracts"'), "Export module tab/panel wiring");
expect(app.includes("createExportContractController") && app.includes("buildDeploymentContractDocuments"), "app export controller wiring");
expect(rawEntry.includes('from "./report-export-contracts.js"'), "protected raw entry export");
expect(worker.includes('"/web/lib/report-export-contracts.js"'), "export document generator must remain behind raw-export authorization");
expect(
  exportView.includes("PUBLIC_EXPORT_BUTTONS")
    && exportView.includes("getPublicDocuments")
    && exportView.includes("publicExport ? async () => true"),
  "standalone CycloneDX exports must bypass member authorization without exposing protected companion builders",
);
expect(
  exportView.includes("deepbom.deployment_contract_pack_manifest.v1.4")
    && exportView.includes("exportSet.files.artifactEnvelope")
    && exportView.includes("exportSet.documents.artifact_evidence_envelope")
    && exportView.includes("contract_set_integrity")
    && exportView.includes("analyzer_provenance"),
  "contract pack manifest should preserve document-set integrity and analyzer provenance.",
);

console.log("Export contract documents passed (official CycloneDX 1.7 schema, reproducible public TFLite example, closed companion evidence, protected UI wiring).");

function propertyMap(properties) {
  return new Map((properties || []).map((item) => [item.name, item.value]));
}

function field(document, id) {
  return document.fields.find((item) => item.id === id);
}

function expectExactSet(actual, expected, label) {
  const actualSet = new Set(actual);
  const expectedSet = new Set(expected);
  for (const value of expectedSet) expect(actualSet.has(value), `${label} should include ${value}`);
  for (const value of actualSet) expect(expectedSet.has(value), `${label} includes unexpected ${value}`);
}

function expectEqual(actual, expected, label) {
  if (actual !== expected) throw new Error(`${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

function expect(condition, message) {
  if (!condition) throw new Error(message);
}
