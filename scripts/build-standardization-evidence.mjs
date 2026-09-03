import { createHash } from "node:crypto";
import { readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { gunzipSync } from "node:zlib";

import { GGUF_BACKEND_SOURCE } from "../web/lib/gguf-backend-contract.generated.js";
import { ORT_BUILD_SOURCE } from "../web/lib/ort-build-attestation.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { SAFETENSORS_QUANTIZATION_SOURCES } from "../web/lib/safetensors-quantization-contract.js";
import { TFLITE_DELEGATE_RULEPACK_METADATA } from "../web/lib/tflite-delegate-rulepack-metadata.js";
import { XNNPACK_DELEGATE_RULEPACK_METADATA } from "../web/lib/xnnpack-rulepack-metadata.js";

const OUTPUT = "corpus/standardization-evidence.v1.json";
const PATHS = Object.freeze({
  interfaceProfile: "web/reference/quant-policy-boundary-validation.v1.json",
  interfaceSweep: "corpus/measurements/quantization-interface-public-50-repeat2/quant-policy-boundary-sweep.json.gz",
  quantManifest: "corpus/quant_policy/manifest.v1.json",
  publicSweep: "corpus/measurements/public-model-corpus-v1-final/corpus-sweep.json.gz",
  hfSummary: "corpus/huggingface-community-corpus.v1.summary.json",
  hfSnapshot: "corpus/huggingface-community-corpus.v1.json.gz",
  residual: "corpus/residual-coverage-priorities.v1.json",
  safetensors: "corpus/safetensors-architecture-corpus.v1.json",
  safetensorsQuantization: "corpus/safetensors-quantization-contract-corpus.v1.json",
  gguf: "corpus/gguf-architecture-encoding-corpus.v1.json",
  coreml: "corpus/coreml-mlprogram-contract-corpus.v1.json",
  coremlLegacyQuantization: "corpus/coreml-legacy-quantization-corpus.v1.json",
  onnxExtensions: "corpus/onnx-extension-contract-corpus.v1.json",
  multiformatManifest: "corpus/public-multiformat-corpus.v1.json",
  cyclonedxGeneralization: "corpus/cyclonedx-generalization-evidence.v1.json",
  cyclonedxGeneralizationRecords: "corpus/cyclonedx-generalization-evidence.records.v1.json.gz",
  runtimeSidecar: "web/lib/runtime-evidence-sidecar.js",
  runtimeSidecarCheck: "scripts/check-runtime-evidence-sidecar.mjs",
  cyclonedxQuantization: "corpus/cyclonedx-20-quantization-fixtures/evidence.v1.json",
  ortRulepack: "protected/deepbom_wasm/src/ort_rulepack_generated.rs",
});

const values = Object.fromEntries(await Promise.all(
  Object.entries(PATHS)
    .filter(([key, file]) => file.endsWith(".json") || key === "interfaceSweep" || key === "publicSweep")
    .map(async ([key, file]) => [key, await readJsonSource(file)]),
));
values.cyclonedxGeneralizationRecords = await readJsonSource(PATHS.cyclonedxGeneralizationRecords);
const sources = await Promise.all(Object.entries(PATHS).map(([id, file]) => sourceRecord(id, file)));
const sourceById = Object.fromEntries(sources.map((row) => [row.id, row]));

validateSourceContracts();
const interfacePopulation = deriveInterfacePopulation(values.interfaceSweep, values.quantManifest);
const profileLineage = await deriveProfileLineage(values.interfaceProfile, sources);
const hfFrame = deriveHuggingFaceFrame(values.hfSummary, sourceById.hfSnapshot);
const ortGenerated = await readFile(PATHS.ortRulepack, "utf8");

const evidence = {
  schema: "deepbom.standardization_evidence.v1",
  snapshot_id: `standardization-${sha256(canonicalJson(sources.map(({ id, sha256: digest }) => ({ id, sha256: digest })))).slice(0, 16)}`,
  hash_contract: {
    algorithm: "SHA-256",
    canonicalization: "RFC8785-JCS",
    excluded_pointers: ["/ledger_sha256"],
  },
  aggregation_contract: {
    population_unit: "Artifact counts are unique only within the named population unless that population states another unit.",
    overlap_rule: "Named populations overlap and MUST NOT be summed into an ecosystem-wide artifact denominator.",
    inference_rule: "Exact ratios describe only their predeclared or measured population. No confidence interval or ecosystem prevalence is emitted because no population is a probability sample.",
    zero_rule: "A zero count in a bounded corpus or fixture is not evidence of absence from the ecosystem.",
  },
  taxonomy_coverage_assessment: deriveTaxonomyCoverageAssessment(),
  terminology: [
    term("artifact_identity", "Cryptographic identity of the analyzed model file or complete package content set."),
    term("artifact_set_identity", "Cryptographic identity and dependency binding for a primary model file plus every externally stored payload required to reconstruct its serialized tensors."),
    term("external_parameter_contract", "A named input or output with direction, identity, dtype, shape, and explicit quantization status; affine values are present only when applicable."),
    term("runtime_build_identity", "Runtime family/version plus source commit, build configuration and binary digest when each is available."),
    term("execution_configuration", "Threads, execution mode, providers/backends, delegates and other invocation settings bound to a capture."),
    term("placement_evidence", "Observed assignment only when a runtime record identifies it; source eligibility and static prediction remain separate classes."),
    term("timing_evidence", "Observed timing samples or aggregates bound to one artifact, runtime build and execution configuration."),
    term("memory_evidence", "Observed allocator or process memory when imported; static liveness and arena projections remain separate estimates."),
    term("claim_boundary", "An explicit statement of what the evidence establishes and what remains unassessed."),
    term("field_state", "A machine-readable distinction among observed, derived, externally declared, unavailable, not applicable, runtime required, and out-of-scope values; unavailable and not applicable are never numerical zero."),
  ],
  populations: {
    empirical_measurements: [
      {
        id: "tflite-external-interface-predeclared-50",
        evidence_class: "EXACT_WITHIN_PREDECLARED_POPULATION",
        source_ids: ["interfaceSweep", "quantManifest"],
        unit: "hash-identified TFLite artifact",
        selection: values.quantManifest.selection_policy,
        repeat_count: values.interfaceSweep.repeat_count,
        ...interfacePopulation,
        scope_limit: "This population establishes serialized external parameter dtype, shape, quantization status, and affine values when present. It does not establish color order, source normalization, resize interpolation, labels, deployed preprocessing, task accuracy, clinical performance, or release readiness.",
      },
      {
        id: "tflite-mediapipe-generation-pinned-20",
        evidence_class: "EXACT_WITHIN_PREDECLARED_POPULATION",
        source_ids: ["publicSweep"],
        unit: "hash-identified TFLite artifact",
        requested_artifact_count: values.publicSweep.requested_artifact_count,
        passed_artifact_count: values.publicSweep.passed_artifact_count,
        failed_artifact_count: values.publicSweep.failed_artifact_count,
        deterministic_artifact_count: values.publicSweep.deterministic_artifact_count,
        artifact_class_counts: values.publicSweep.artifact_class_counts,
        overlap_note: "These 20 artifacts are also one subcohort of tflite-external-interface-predeclared-50.",
      },
      measuredResidualPopulation("onnx", values.residual.populations.onnx),
      ...["onnx", "gguf", "safetensors", "coreml"].map((format) => generalizationFormatPopulation(format, values.cyclonedxGeneralization)),
    ],
    metadata_sampling_frames: [hfFrame],
    contract_fixtures: [
      fixturePopulation("safetensors-architecture", values.safetensors, "architecture_class"),
      {
        id: "safetensors-awq-gptq-quantization-contract",
        evidence_class: "PUBLIC_HEADER_AND_CONFIG_CONFORMANCE_ONLY",
        source_ids: ["safetensorsQuantization"],
        format: "safetensors",
        artifact_count: values.safetensorsQuantization.artifact_count,
        strata: countBy(values.safetensorsQuantization.artifacts, (row) => row.method),
        assessed_module_count: values.safetensorsQuantization.artifacts.reduce((sum, row) => sum + row.measurement.valid_module_count, 0),
        invalid_module_count: values.safetensorsQuantization.artifacts.reduce((sum, row) => sum + row.measurement.invalid_module_count, 0),
        scope: values.safetensorsQuantization.purpose,
        non_claim: values.safetensorsQuantization.population_boundary,
      },
      fixturePopulation("gguf-architecture-encoding", values.gguf, "model_family"),
      fixturePopulation("coreml-mlprogram-contract", values.coreml, "contract_class"),
      fixturePopulation("coreml-legacy-per-channel-quantization", values.coremlLegacyQuantization, "semantic_granularity"),
      {
        id: "onnx-extension-contract",
        evidence_class: "CONTRACT_CONFORMANCE_ONLY",
        source_ids: ["onnxExtensions"],
        format: "onnx",
        artifact_count: values.onnxExtensions.summary.artifact_count,
        strata: {
          per_axis_qdq: values.onnxExtensions.summary.per_axis_qdq_artifact_count,
          complete_static_affine: values.onnxExtensions.summary.complete_static_affine_artifact_count,
          runtime_value_unresolved: values.onnxExtensions.summary.runtime_value_unresolved_artifact_count,
          external_custom_domain: values.onnxExtensions.summary.external_custom_domain_artifact_count,
        },
        scope: values.onnxExtensions.purpose,
        non_claim: values.onnxExtensions.population_boundary,
      },
      {
        id: "cross-format-runtime-evidence-sidecar",
        evidence_class: "CONTRACT_CONFORMANCE_ONLY",
        source_ids: ["runtimeSidecar", "runtimeSidecarCheck"],
        covered_formats: ["tflite", "onnx", "gguf", "coreml"],
        schema: "deepbom.runtime_evidence_sidecar.v1",
        guarantees: ["artifact binding", "source evidence digest", "build/placement/timing/memory claim separation", "tamper rejection"],
        non_claim: "This fixture proves normalization and validation behavior, not runtime-provider prevalence or device execution.",
      },
      {
        id: "cyclonedx-2.0-draft-quantization-fixtures",
        evidence_class: "CONTRACT_CONFORMANCE_ONLY",
        source_ids: ["cyclonedxQuantization"],
        upstream_pull_request: values.cyclonedxQuantization.upstream.pull_request,
        upstream_head_commit: values.cyclonedxQuantization.upstream.head_commit,
        candidate_fixture_count: values.cyclonedxQuantization.fixture_results.length,
        candidate_valid_fixture_count: values.cyclonedxQuantization.fixture_results.filter((row) => row.actual === "VALID").length,
        candidate_invalid_fixture_count: values.cyclonedxQuantization.fixture_results.filter((row) => row.actual === "INVALID").length,
        schema_probe_counts: {
          valid: values.cyclonedxQuantization.twelve_case_probe.valid_count,
          invalid: values.cyclonedxQuantization.twelve_case_probe.invalid_count,
        },
        non_claim: "This validates candidate fixtures against one pinned draft PR head. It does not establish acceptance into CycloneDX, final 2.0 semantics, framework prevalence, or per-group behavior in the measured TFLite corpus.",
      },
    ],
    detached_profiles: [profileLineage],
  },
  source_semantics: [
    {
      id: "tflite-xnnpack-delegate",
      evidence_class: "SOURCE_BACKED_RULEPACK",
      repository: "tensorflow/tensorflow",
      commit: XNNPACK_DELEGATE_RULEPACK_METADATA.sourceCommit,
      manifest_sha256: XNNPACK_DELEGATE_RULEPACK_METADATA.manifestSha256,
      operator_counts: { fp32: XNNPACK_DELEGATE_RULEPACK_METADATA.fp32OperatorCount, quantized: XNNPACK_DELEGATE_RULEPACK_METADATA.quantizedOperatorCount },
      constraint_counts: { documented: XNNPACK_DELEGATE_RULEPACK_METADATA.documentedConstraintCount, implemented: XNNPACK_DELEGATE_RULEPACK_METADATA.implementedConstraintCount, unmapped: XNNPACK_DELEGATE_RULEPACK_METADATA.unmappedConstraintCount },
      boundary: "Source-backed artifact eligibility is not selected-build availability, runtime assignment, copy materialization or timing evidence.",
    },
    {
      id: "tflite-gpu-nnapi-delegates",
      evidence_class: "SOURCE_BACKED_RULEPACK",
      repository: "tensorflow/tensorflow",
      commit: TFLITE_DELEGATE_RULEPACK_METADATA.tensorflowCommit,
      manifest_sha256: TFLITE_DELEGATE_RULEPACK_METADATA.manifestSha256,
      profiles: TFLITE_DELEGATE_RULEPACK_METADATA.profiles.map((row) => ({ id: row.id, registered_op_count: row.registeredOpCount })),
      boundary: TFLITE_DELEGATE_RULEPACK_METADATA.interpretationBoundary,
    },
    {
      id: "onnxruntime-execution-providers",
      evidence_class: "SOURCE_BACKED_RULEPACK",
      repository: ORT_BUILD_SOURCE.repository,
      commit: ORT_BUILD_SOURCE.commit,
      runtime_version: ORT_BUILD_SOURCE.runtime_version,
      generated_inventory: {
        provider_source_count: countMacro(ortGenerated, "ep_source"),
        provider_rule_count: countMacro(ortGenerated, "ep_rule"),
        runtime_release_count: countMacro(ortGenerated, "release"),
        contrib_release_count: countMacro(ortGenerated, "contrib_release"),
        schema_history_count: countMacro(ortGenerated, "schema_history"),
      },
      boundary: "Registration and passed artifact predicates establish conditional eligibility only. Selected-build inclusion, optimized-node assignment and execution require build/runtime evidence.",
    },
    {
      id: "gguf-llama-backends",
      evidence_class: "SOURCE_BACKED_RULEPACK",
      repository: GGUF_BACKEND_SOURCE.repository,
      commit: GGUF_BACKEND_SOURCE.source_commit,
      pinned_source_file_count: Object.keys(GGUF_BACKEND_SOURCE.files).length,
      boundary: "Serialized architecture and build-option evidence does not establish the loaded backend, offload plan, scheduler decisions or timing.",
    },
    {
      id: "coreml-mlprogram-format",
      evidence_class: "SOURCE_BACKED_CONTRACT",
      repository: values.coreml.generator_source.repository,
      commit: values.coreml.generator_source.revision,
      pinned_proto_count: 3,
      boundary: "Format semantics and generated fixtures do not establish compute-unit placement or executed device behavior.",
    },
  ],
  standardization_claims: [
    {
      id: "external-affine-contract-requires-parameter-binding",
      status: "SUPPORTED",
      evidence_ids: ["tflite-external-interface-predeclared-50"],
      observation: {
        dtype_shape_signatures_with_multiple_affine_contracts: interfacePopulation.dtype_shape_signatures_with_multiple_affine_contracts,
        complete_affine_parameters: ratio(interfacePopulation.complete_affine_parameter_count, interfacePopulation.external_parameter_count),
      },
      conclusion: "Direction, dtype and shape do not uniquely identify an affine mapping. Exact scale and zero-point values must be bound to a schema-addressable named parameter or another stable tensor identity.",
      generalization_basis: "This is a structural counterexample argument, not a prevalence estimate. One repeated dtype/shape signature with different affine values disproves uniqueness; the predeclared corpus contains six such signatures across multiple workflow strata.",
      non_claim: "The corpus does not estimate how often deployment harnesses contain a mismatched mapping.",
    },
    {
      id: "boundary-contract-must-represent-explicitly-unquantized",
      status: "SUPPORTED",
      evidence_ids: ["tflite-external-interface-predeclared-50"],
      observation: {
        explicitly_unquantized_parameters: ratio(interfacePopulation.explicitly_unquantized_parameter_count, interfacePopulation.external_parameter_count),
        fully_unquantized_artifacts: ratio(interfacePopulation.boundary_status_counts.fully_unquantized, interfacePopulation.public_artifact_count),
        mixed_boundary_artifacts: ratio(interfacePopulation.boundary_status_counts.mixed_quantized_unquantized, interfacePopulation.public_artifact_count),
        valid_explicit_boundary_contracts: ratio(interfacePopulation.valid_explicit_boundary_contract_artifact_count, interfacePopulation.public_artifact_count),
      },
      conclusion: "A boundary contract needs an explicit not-quantized state; absence of affine values must not conflate FLOAT I/O with missing or invalid metadata.",
      generalization_basis: "The semantic distinction applies to any typed model interface. The reported ratios remain exact only for this predeclared TFLite population.",
      non_claim: "An explicit dtype/affine boundary does not establish color order, normalization, resize, labels, task accuracy or clinical validity.",
    },
    {
      id: "model-quantization-requires-scoped-repeatable-representations",
      status: "STRUCTURE_SUPPORTED_VOCABULARY_INCOMPLETE",
      evidence_ids: ["tflite-external-interface-predeclared-50", "public-onnx-generalization", "public-gguf-generalization", "public-safetensors-generalization", "public-coreml-generalization"],
      observation: {
        model_representation_counterexamples: values.cyclonedxGeneralization.cross_format_schema_findings.find((row) => row.id === "model-quantization-multiplicity"),
        tflite_external_affine_parameters: ratio(interfacePopulation.complete_affine_parameter_count, interfacePopulation.external_parameter_count),
        tflite_dtype_shape_signatures_with_multiple_affine_contracts: interfacePopulation.dtype_shape_signatures_with_multiple_affine_contracts,
      },
      conclusion: "A model-level quantization description needs repeatable, explicitly scoped representations when an artifact mixes tensor mappings, storage encodings, axes, granularities, or non-affine block formats.",
      generalization_basis: "The 27/27 result is not used as an ecosystem prevalence estimate. Each of the 27 artifacts is a constructive counterexample to singleton losslessness, so one counterexample is sufficient to establish the repeatable scoped structure. The separate TFLite population supplies positive parameter-level affine contracts and six dtype/shape collisions, showing why model summary and named-parameter contract are distinct levels.",
      non_claim: "This evidence does not establish a complete cross-ecosystem scheme, granularity, bit-width, axis, packing, or zero-point vocabulary. SafeTensors storage dtype is not itself an affine mapping, GGUF block encoding is not recast as one, and runtime lowering and task quality remain outside this evidence.",
    },
    {
      id: "evidence-fields-require-applicability-and-availability-states",
      status: "SUPPORTED_BY_CONSERVED_CROSS_FORMAT_STATES",
      evidence_ids: ["public-onnx-generalization", "public-gguf-generalization", "public-safetensors-generalization", "public-coreml-generalization"],
      observation: values.cyclonedxGeneralization.cross_format_schema_findings.find((row) => row.id === "applicability-availability-and-zero-are-distinct"),
      conclusion: "Portable evidence fields need explicit observed, unavailable, and not-applicable states. Missing assessment and inapplicability must not be serialized as zero or collapsed into one null state.",
      generalization_basis: "The same graph field assumes all three states in hash-identified artifacts: serialized graph present, serialized graph applicable but outside the bounded decoder, and graph not serialized by the container. This is a semantic counterexample, so the need for separate states does not depend on estimating ecosystem frequency.",
      non_claim: "The state counts do not rank formats or imply that an unavailable graph is malformed.",
    },
    {
      id: "external-payloads-require-artifact-set-identity",
      status: "SUPPORTED_BY_ONNX_COUNTEREXAMPLES",
      evidence_ids: ["public-onnx-generalization"],
      observation: values.cyclonedxGeneralization.cross_format_schema_findings.find((row) => row.id === "compound-artifact-identity"),
      conclusion: "When tensor payloads are external, a primary model-file hash is insufficient for a reproducible artifact identity; the record must bind every required external component and its digest.",
      generalization_basis: "Fourteen independently hash-bound ONNX records reconstruct only with verified external payload components. One such artifact is sufficient to disprove universal single-file identity; 14 records demonstrate that the condition is not a synthetic fixture in this frame.",
      non_claim: "The 14/48 ratio is not an estimate of external-data prevalence across ONNX models.",
    },
    {
      id: "artifact-source-and-license-provenance-are-independent",
      status: "SUPPORTED_BY_FIELD_AVAILABILITY_MATRIX",
      evidence_ids: ["public-onnx-generalization", "public-gguf-generalization", "public-safetensors-generalization", "public-coreml-generalization"],
      observation: values.cyclonedxGeneralization.cross_format_schema_findings.find((row) => row.id === "artifact-source-and-license-provenance-separation"),
      conclusion: "Artifact digest, source revision, and externally declared license metadata must remain independently optional and independently sourced fields.",
      generalization_basis: "All 113 records have observed byte identity, but revision and license metadata are available for different subsets and come from external source frames. The coexistence of these states is a direct counterexample to deriving one field from another.",
      non_claim: "External license metadata is not a license grant, legal conclusion, or fact inferred from model bytes.",
    },
    {
      id: "serialized-contract-validity-is-independent-of-readability",
      status: "SUPPORTED_BY_ISSUE_BEARING_ARTIFACTS",
      evidence_ids: ["public-onnx-generalization", "public-gguf-generalization", "public-safetensors-generalization", "public-coreml-generalization"],
      observation: values.cyclonedxGeneralization.cross_format_schema_findings.find((row) => row.id === "validity-is-not-availability-or-runtime-evidence"),
      conclusion: "Parser success, checked serialized-contract validity, external dependency completeness, and runtime compatibility are separate statuses.",
      generalization_basis: "Three readable artifacts retained exact source-rule issue codes while remaining in the measured denominator. Their existence disproves the implication that parse success establishes contract conformance.",
      non_claim: "A clean implemented-rule result is not proof of complete semantic correctness or runtime compatibility.",
    },
    {
      id: "runtime-evidence-needs-cross-format-claim-separation",
      status: "CONTRACT_PROVEN",
      evidence_ids: ["cross-format-runtime-evidence-sidecar", "tflite-xnnpack-delegate", "onnxruntime-execution-providers", "gguf-llama-backends", "coreml-mlprogram-format"],
      conclusion: "A portable runtime evidence index should bind artifact, runtime build and execution configuration while representing placement, timing and memory as separate claims with independent evidence classes.",
      generalization_basis: "The four adapters expose different runtime records but share these identity and claim-boundary invariants. This is cross-format contract conformance, not an empirical ecosystem frequency claim.",
      non_claim: "The normalized sidecar does not upgrade source eligibility or a static plan into observed execution.",
    },
  ],
  contribution_guidance: {
    normative_candidate_terms: ["artifact_set_identity", "external_parameter_contract", "field_state", "runtime_build_identity", "execution_configuration", "placement_evidence", "claim_boundary"],
    fixture_requirements: [
      "Valid fixtures bind affine values to a named input or output.",
      "Valid fixtures distinguish not-quantized from missing or invalid quantization metadata.",
      "Invalid fixtures target constraints that the candidate schema actually rejects; semantic desiderata are labeled separately until schema conditions exist.",
      "Runtime fixtures never equate source eligibility, configured inclusion, assignment and execution.",
      "Graph and numerical fields distinguish an assessed zero from unavailable and not-applicable states.",
      "External-data fixtures bind the primary artifact and every required payload component by digest.",
    ],
    reporting_rule: "Every quoted ratio carries numerator, denominator, population ID, selection rule and claim boundary.",
  },
  source_records: sources,
};
evidence.ledger_sha256 = sha256(canonicalJson(evidence));

const serialized = `${JSON.stringify(evidence, null, 2)}\n`;
if (process.argv.includes("--check")) {
  const observed = await readFile(OUTPUT, "utf8").catch(() => "");
  if (observed !== serialized) throw new Error(`${OUTPUT} is stale; run npm run build:standardization-evidence.`);
  const { ledger_sha256: recordedLedger, ...hashBasis } = JSON.parse(observed);
  assert(recordedLedger === sha256(canonicalJson(hashBasis)), "Standardization evidence ledger SHA-256 does not reconstruct.");
  console.log(`Standardization evidence check passed (${interfacePopulation.public_artifact_count} TFLite artifacts, ${hfFrame.repository_count} repository metadata records, ${sources.length} hash-bound sources).`);
} else {
  await writeFile(OUTPUT, serialized, "utf8");
  console.log(`Wrote ${OUTPUT} (${evidence.snapshot_id}; ${evidence.ledger_sha256}).`);
}

function deriveInterfacePopulation(sweep, manifest) {
  const rows = sweep.rows.filter((row) => row.public_corpus_member === true);
  const counts = { fully_affine_quantized: 0, mixed_quantized_unquantized: 0, fully_unquantized: 0, invalid_or_incomplete: 0 };
  const strata = new Map();
  let parameterCount = 0;
  let complete = 0;
  let unquantized = 0;
  let invalid = 0;
  let anyComplete = 0;
  for (const row of rows) {
    const parameters = row.interface_contracts?.parameters || [];
    const rowComplete = parameters.filter((item) => item.quantization?.status === "complete").length;
    const rowUnquantized = parameters.filter((item) => item.quantization?.status === "not_quantized").length;
    const rowInvalid = parameters.length - rowComplete - rowUnquantized;
    const status = rowInvalid > 0 ? "invalid_or_incomplete"
      : rowComplete === parameters.length ? "fully_affine_quantized"
        : rowUnquantized === parameters.length ? "fully_unquantized" : "mixed_quantized_unquantized";
    parameterCount += parameters.length;
    complete += rowComplete;
    unquantized += rowUnquantized;
    invalid += rowInvalid;
    if (rowComplete > 0) anyComplete += 1;
    counts[status] += 1;
    if (!strata.has(row.subcohort_id)) strata.set(row.subcohort_id, { artifact_count: 0, boundary_status_counts: { fully_affine_quantized: 0, mixed_quantized_unquantized: 0, fully_unquantized: 0, invalid_or_incomplete: 0 } });
    const stratum = strata.get(row.subcohort_id);
    stratum.artifact_count += 1;
    stratum.boundary_status_counts[status] += 1;
  }
  assert(rows.length === manifest.public_artifact_count && sweep.requested_public_artifact_count === rows.length && sweep.passed_public_artifact_count === rows.length && sweep.failed_public_artifact_count === 0, "TFLite interface population denominator is not closed.");
  assert(rows.every((row) => row.status === "passed" && row.deterministic === true && row.repeat_count === sweep.repeat_count), "TFLite interface population is not repeat-deterministic.");
  assert(parameterCount === complete + unquantized + invalid && invalid === 0, "External parameter states do not conserve.");
  const summary = sweep.interface_quantization_contract_summary;
  assert(summary.public_artifact_count === rows.length && summary.parameter_count === parameterCount && summary.complete_quantized_parameter_count === complete && summary.unquantized_parameter_count === unquantized && summary.invalid_or_incomplete_parameter_count === invalid, "Stored interface summary does not reconstruct.");
  return {
    public_artifact_count: rows.length,
    external_parameter_count: parameterCount,
    complete_affine_parameter_count: complete,
    explicitly_unquantized_parameter_count: unquantized,
    invalid_or_incomplete_parameter_count: invalid,
    artifacts_with_any_complete_affine_parameter: anyComplete,
    valid_explicit_boundary_contract_artifact_count: rows.length - counts.invalid_or_incomplete,
    boundary_status_counts: counts,
    subcohorts: Object.fromEntries([...strata.entries()].sort(([left], [right]) => left.localeCompare(right))),
    dtype_shape_signatures_with_multiple_affine_contracts: summary.dtype_shape_signatures_with_multiple_affine_contracts,
  };
}

function deriveTaxonomyCoverageAssessment() {
  const tfliteRows = values.interfaceSweep.rows.filter((row) => row.public_corpus_member === true);
  const tfliteParameters = tfliteRows.flatMap((row) => row.interface_contracts?.parameters || []);
  const records = values.cyclonedxGeneralizationRecords.artifacts || [];
  assert(records.length === values.cyclonedxGeneralization.population.unique_primary_artifact_count,
    "Cross-format taxonomy coverage record denominator changed.");

  const formatRows = ["onnx", "gguf", "safetensors", "coreml"].map((format) => {
    const artifacts = records.filter((row) => row.format === format);
    const profiles = artifacts.flatMap((row) => row.cyclonedx_observation?.model_representations?.representation_profiles || []);
    return {
      format,
      artifact_count: artifacts.length,
      extension_file_counts: countBy(
        values.multiformatManifest.artifacts
          .filter((row) => row.format === format)
          .flatMap((row) => row.files || []),
        (file) => compoundExtension(file.path),
      ),
      observed_scheme_set: uniqueFlat(profiles, "scheme"),
      observed_granularity_set: uniqueFlat(profiles, "granularities"),
      observed_encoding_set: uniqueFlat(profiles, "encoding"),
      observed_group_size_set: uniqueFlat(profiles, "group_sizes"),
      observed_axis_set: uniqueFlat(profiles, "axes"),
      representation_profile_count: profiles.length,
    };
  });
  const byFormat = Object.fromEntries(formatRows.map((row) => [row.format, row]));

  return {
    status: "STRUCTURAL_REQUIREMENTS_SUPPORTED_TAXONOMY_VOCABULARY_INCOMPLETE",
    interpretation: "Artifact counts alone do not establish taxonomy coverage. Coverage is tracked across container extension, representation family, granularity, numeric encoding, scope, axis semantics, and evidence class.",
    measured_extension_matrix: [
      {
        format: "tflite",
        artifact_count: tfliteRows.length,
        extension_file_counts: { ".tflite": tfliteRows.length },
        observed_scheme_set: ["affine", "not_quantized"],
        observed_granularity_set: ["per-tensor external parameter", "per-axis kernel", "per-tensor kernel"],
        observed_encoding_set: [...new Set(tfliteParameters.map((row) => row.dtype).filter(Boolean))].sort(),
        observed_axis_set: [],
        measured_detail: {
          complete_affine_external_parameters: values.interfaceSweep.interface_quantization_contract_summary.complete_quantized_parameter_count,
          explicitly_unquantized_external_parameters: values.interfaceSweep.interface_quantization_contract_summary.unquantized_parameter_count,
          external_per_tensor_parameters: values.interfaceSweep.interface_quantization_contract_summary.per_tensor_parameter_count,
          external_per_axis_parameters: values.interfaceSweep.interface_quantization_contract_summary.per_axis_parameter_count,
          kernel_granularity_artifact_counts: values.interfaceSweep.kernel_quantization_granularity_counts,
        },
      },
      ...formatRows,
    ],
    conformance_extension_matrix: [
      {
        format: "onnx",
        artifact_count: values.onnxExtensions.summary.artifact_count,
        observed_scheme_set: ["affine", "external_custom_domain_identity"],
        observed_granularity_set: ["per-axis"],
        evidence_boundary: values.onnxExtensions.population_boundary,
      },
      {
        format: "safetensors",
        artifact_count: values.safetensorsQuantization.artifact_count,
        observed_scheme_set: values.safetensorsQuantization.artifacts.map((row) => row.method).sort(),
        observed_granularity_set: ["per-group weight"],
        observed_group_size_set: [...new Set(values.safetensorsQuantization.artifacts.map((row) => row.measurement.group_size))].sort(compareScalar),
        assessed_module_count: values.safetensorsQuantization.artifacts.reduce((sum, row) => sum + row.measurement.valid_module_count, 0),
        evidence_boundary: `${values.safetensorsQuantization.population_boundary} Payload values were not scanned.`,
      },
    ],
    source_bound_implementation_matrix: [
      {
        format: "safetensors",
        representation_families: ["awq", "gptq", "hqq", "compressed-tensors"],
        implementation_sources: Object.fromEntries(Object.entries(SAFETENSORS_QUANTIZATION_SOURCES).map(([method, source]) => [method, {
          repository: source.repository, commit: source.commit, path: source.path, sha256: source.sha256,
        }])),
        static_contract_scope: "Exact method-specific metadata, grouping, packed shape, code/storage-bit conservation, and source-defined padding for supported static layouts.",
        evidence_boundary: "AWQ and GPTQ additionally have public revision-bound header/config anchors. HQQ and compressed-tensors currently have source-pinned deterministic regression fixtures, not a public ecosystem prevalence corpus. Dynamic, nonuniform, and unmatched module-target schemes remain not assessed.",
      },
    ],
    structural_conclusions_supported_now: [
      "Model-level numerical representations must be repeatable and explicitly scoped.",
      "Named external input/output affine contracts must remain distinct from model weight/storage summaries.",
      "Affine mappings, graph quantization transforms, block encodings, palette/LUT transforms, and scalar storage precision are distinct representation families.",
      "Observed zero, unavailable, not applicable, externally declared, and runtime-required values require separate machine-readable states.",
      "A portable taxonomy needs an extension mechanism; the measured vocabularies are not a complete closed enum.",
    ],
    vocabulary_coverage_limits: [
      {
        id: "external-per-axis-affine",
        status: "STRUCTURE_AND_STATIC_FIXTURE_POSITIVE_EXTERNAL_VALUES_INCOMPLETE",
        requirement: "The source-derived ONNX fixtures now establish named external tensor shape, schema-default axis, and scale cardinality while values remain runtime inputs; add hash-bound public external-parameter examples whose complete per-axis scale vectors are serialized before claiming external numerical-contract coverage.",
      },
      {
        id: "framework-native-negative-axis",
        status: "SCHEMA_PROBE_ONLY",
        requirement: "Add rank-bound ONNX artifacts and a declared normalization rule before proposing a format-neutral axis constraint.",
      },
      {
        id: "affine-per-group",
        status: "PUBLIC_AWQ_GPTQ_STRUCTURE_POSITIVE_HQQ_COMPRESSED_TENSORS_SOURCE_FIXTURES_POSITIVE_PREVALENCE_UNASSESSED",
        requirement: "Two public revision-bound SafeTensors repositories establish AutoAWQ GEMM and AutoGPTQ packed layout, 322/322 valid module groups, 4-bit codes, group size 128, scale/zero cardinality, and exact packed-capacity conservation. Source-pinned deterministic fixtures now also establish HQQ encoded-state and compressed-tensors pack-quantized shape, grouping, and storage-bit conservation. Add independent public HQQ/compressed-tensors artifacts, sharded packages, and full-payload scale/zero scans before proposing universal affine per-group semantics or ecosystem prevalence. GGUF block size is not automatically recast as affine per-group quantization.",
      },
      {
        id: "safetensors-quantization-sidecars",
        status: "SINGLE_FILE_AWQ_GPTQ_POSITIVE_SHARDED_UNCOVERED",
        requirement: "The public AWQ and GPTQ anchors bind config.json plus quant_config.json or quantize_config.json to immutable model identity. Add manifest-bound sharded quantized repositories and conflicting/missing sidecar invalid fixtures before claiming general SafeTensors package coverage.",
      },
      {
        id: "onnx-quantization-families",
        status: "PARTIAL_POSITIVE_CONFORMANCE",
        requirement: "Per-axis Q/DQ and non-ORT external custom-domain fail-closed behavior now have hash-bound positive conformance artifacts. Positive public application coverage remains needed for per-axis Q/DQ, QLinear, DynamicQuantizeLinear, sub-byte packed tensors, and contrib/custom-domain weight-only operators.",
      },
      {
        id: "coreml-compression-families",
        status: "POSITIVE_PER_CHANNEL_CONFORMANCE_PUBLIC_APPLICATION_INCOMPLETE",
        requirement: "A source-pinned legacy NeuralNetwork fixture now establishes per-output-channel linear scale/bias cardinality, packed INT4 code conservation, and fail-closed axis mismatch. Add public-model application coverage, modern MLProgram per-channel/per-block contracts, and positive sparse/pruned representations before generalizing ecosystem prevalence or all Core ML compression families.",
      },
      {
        id: "symmetry-and-zero-point-semantics",
        status: "PARTIAL",
        requirement: "Add cross-format examples for explicit/implicit zero points, symmetric/asymmetric classification, signed/unsigned domains, and canonical omission rules.",
      },
    ],
    contribution_gate: {
      repeatable_scoped_schema_shape: "SUPPORTED",
      named_external_parameter_binding: "SUPPORTED",
      open_custom_representation_escape_hatch: "REQUIRED_BY_INCOMPLETE_VOCABULARY",
      closed_cross_ecosystem_scheme_enum: "NOT_SUPPORTED",
      closed_cross_ecosystem_granularity_enum: "NOT_SUPPORTED",
      universal_axis_constraint: "NOT_SUPPORTED",
      per_group_normative_semantics: "NOT_SUPPORTED",
    },
    singleton_counterexample_breakdown: {
      onnx: `${byFormat.onnx ? values.cyclonedxGeneralization.format_populations.onnx.singleton_model_quantization_lossy_count : 0}/${values.cyclonedxGeneralization.format_populations.onnx.singleton_model_quantization_applicable_count}`,
      gguf: `${values.cyclonedxGeneralization.format_populations.gguf.singleton_model_quantization_lossy_count}/${values.cyclonedxGeneralization.format_populations.gguf.singleton_model_quantization_applicable_count}`,
      coreml: `${values.cyclonedxGeneralization.format_populations.coreml.singleton_model_quantization_lossy_count}/${values.cyclonedxGeneralization.format_populations.coreml.singleton_model_quantization_applicable_count}`,
      safetensors: `${values.cyclonedxGeneralization.format_populations.safetensors.singleton_model_quantization_lossy_count}/${values.cyclonedxGeneralization.format_populations.safetensors.singleton_model_quantization_applicable_count}`,
      interpretation: "These counterexamples establish the need for a repeatable scoped structure. They do not establish vocabulary completeness or ecosystem prevalence.",
    },
  };
}

async function deriveProfileLineage(profile, sourceRecords) {
  const availableDigests = new Set(sourceRecords.map((row) => row.sha256));
  const declared = profile.source || {};
  const sweepPresent = availableDigests.has(declared.sweep_sha256);
  const reviewPresent = availableDigests.has(declared.review_sha256);
  return {
    id: profile.profile_id,
    evidence_class: "ARCHIVED_VERIFIED_PROFILE",
    source_ids: ["interfaceProfile"],
    verification_status: profile.verification_status,
    verified_at: profile.verified_at,
    declared_source_digests: { sweep_sha256: declared.sweep_sha256, review_sha256: declared.review_sha256, review_ledger_sha256: declared.review_ledger_sha256 },
    declared_source_bytes_present: sweepPresent && reviewPresent,
    lineage_status: sweepPresent && reviewPresent ? "COMPLETE" : "DETACHED_DIGEST_PROFILE",
    available_interface_sweep_sha256: (await sourceRecord("availableInterfaceSweep", PATHS.interfaceSweep)).sha256,
    interpretation: sweepPresent && reviewPresent
      ? "The declared sweep and review bytes are present and hash-bound."
      : "The compact profile is preserved, but its declared sweep/review byte digests are not both present in this repository state. Use the available repeat-2 sweep as a separate measured source; do not claim byte-for-byte reconstruction of the detached profile.",
    boundary: profile.interpretation_boundary,
  };
}

function deriveHuggingFaceFrame(summary, snapshotSource) {
  const row = summary.summary;
  assert(summary.snapshot_sha256 === snapshotSource.sha256 && summary.snapshot_bytes === snapshotSource.byte_length, "Hugging Face metadata snapshot digest does not match its summary.");
  assert(row.known_size_file_count + row.unknown_size_file_count === row.file_count, "Hugging Face file-size denominator does not conserve.");
  assert(Object.values(row.organization_counts).reduce(sum, 0) === row.repository_count, "Hugging Face repository strata do not conserve.");
  return {
    id: "huggingface-community-metadata-frame-v1",
    evidence_class: "METADATA_FRAME_ONLY",
    source_ids: ["hfSummary", "hfSnapshot"],
    repository_count: row.repository_count,
    organization_counts: row.organization_counts,
    file_count: row.file_count,
    model_file_count: row.model_file_count,
    analyzer_supported_file_count: row.analyzer_supported_file_count,
    default_sweep_eligible_file_count: row.default_sweep_eligible_file_count,
    format_counts: row.format_counts,
    non_claim: "Repository metadata and filename classification define a download/sampling frame only. They do not prove parser success, execution, model quality or ecosystem prevalence beyond the enumerated organizations and snapshot.",
  };
}

function measuredResidualPopulation(format, population) {
  return {
    id: `${format}-residual-measured-${population.artifact_count}`,
    evidence_class: population.evidence_class,
    source_ids: ["residual"],
    unit: "unique artifact bytes",
    artifact_count: population.artifact_count,
    path_record_count: population.path_record_count,
    repository_count: population.repository_count,
    measured_counts: {
      node_count: population.node_count,
      node_output_count: population.node_output_count,
      external_custom_domain_node_count: population.external_custom_domain_node_count,
      ort_contrib_node_count: population.ort_contrib_node_count,
      ort_contrib_shape_rule_unsupported_node_count: population.ort_contrib_shape_rule_unsupported_node_count,
      shape_rule_unsupported_node_count: population.shape_rule_unsupported_node_count,
      shape_rule_unresolved_node_count: population.shape_rule_unresolved_node_count,
      shape_contract_known_node_output_count: population.shape_contract_known_node_output_count,
      shape_contract_unknown_node_output_count: population.shape_contract_unknown_node_output_count,
      invalid_node_output_count: population.invalid_node_output_count,
      conditionally_invalid_node_output_count: population.conditionally_invalid_node_output_count,
      conditional_invalid_variant_count: population.conditional_invalid_variant_count,
      conditional_unassessed_variant_count: population.conditional_unassessed_variant_count,
      unresolved_nonconflict_shape_contract_node_output_count: population.unresolved_nonconflict_shape_contract_node_output_count,
      total_macs_unresolved_op_count: population.total_macs_unresolved_op_count,
      total_macs_artifact_contract_conflict_op_count: population.total_macs_artifact_contract_conflict_op_count,
      total_macs_external_binding_required_op_count: population.total_macs_external_binding_required_op_count,
      total_macs_analyzer_or_contract_residual_op_count: population.total_macs_analyzer_or_contract_residual_op_count,
    },
    non_claim: "This bounded measured population ranks observed analyzer residuals and is not an ecosystem prevalence sample.",
  };
}

function generalizationFormatPopulation(format, document) {
  const population = document.format_populations[format];
  return {
    id: `public-${format}-generalization`,
    evidence_class: "MEASURED_CORPUS",
    source_ids: ["multiformatManifest", "cyclonedxGeneralization"],
    unit: "unique primary artifact bytes",
    artifact_count: population.unique_artifact_count,
    path_record_count: population.path_record_count,
    source_count: population.repository_or_catalog_source_count,
    architecture_strata: population.architecture_strata,
    task_strata: population.task_strata,
    precision_strata: population.precision_strata,
    measured_counts: {
      external_parameter_count: population.external_parameter_count,
      affine_external_parameter_count: population.affine_external_parameter_count,
      singleton_model_quantization_lossy_count: population.singleton_model_quantization_lossy_count,
      singleton_model_quantization_applicable_count: population.singleton_model_quantization_applicable_count,
      serialized_contract_status: population.serialized_contract_status,
      serialized_contract_issue_codes: population.serialized_contract_issue_codes,
    },
    non_claim: "This deliberately stratified public-file population validates artifact parsing and representational counterexamples. It is not a probability sample and does not establish runtime placement, model quality, or ecosystem prevalence.",
  };
}

function fixturePopulation(id, value, stratumKey) {
  const strata = countBy(value.artifacts, (row) => row[stratumKey] || row.baseline?.[stratumKey] || "unspecified");
  return {
    id,
    evidence_class: "CONTRACT_CONFORMANCE_ONLY",
    source_ids: [value.format === "safetensors" ? "safetensors" : value.format === "gguf" ? "gguf" : "coreml"],
    format: value.format,
    artifact_count: value.artifact_count,
    strata,
    scope: value.population_scope,
  };
}

function term(name, definition) { return { name, definition }; }
function ratio(numerator, denominator) { return { numerator, denominator, decimal: denominator ? (numerator / denominator).toFixed(6) : null }; }
function countBy(rows, selector) { return Object.fromEntries([...rows.reduce((map, row) => map.set(selector(row), (map.get(selector(row)) || 0) + 1), new Map()).entries()].sort(([left], [right]) => left.localeCompare(right))); }
function uniqueFlat(rows, key) { return [...new Set(rows.flatMap((row) => Array.isArray(row[key]) ? row[key] : row[key] == null ? [] : [row[key]]))].sort(compareScalar); }
function compareScalar(left, right) { return typeof left === "number" && typeof right === "number" ? left - right : String(left).localeCompare(String(right)); }
function compoundExtension(filename) { const lower = String(filename || "").toLowerCase(); if (lower.endsWith(".mlpackage.zip")) return ".mlpackage.zip"; const extension = path.extname(lower); return extension || "(none)"; }
function countMacro(source, name) { return (source.match(new RegExp(`^\\s*${name}!\\(`, "gm")) || []).length; }
function sum(left, right) { return left + right; }
function sha256(value) { return createHash("sha256").update(value).digest("hex"); }
function assert(condition, message) { if (!condition) throw new Error(message); }
async function readJsonSource(file) {
  const bytes = await readFile(file);
  const decoded = file.endsWith(".gz") ? gunzipSync(bytes) : bytes;
  return JSON.parse(decoded.toString("utf8"));
}
async function sourceRecord(id, file) { const bytes = await readFile(file); const info = await stat(file); return { id, path: file.replaceAll("\\", "/"), sha256: sha256(bytes), byte_length: info.size }; }

function validateSourceContracts() {
  assert(values.interfaceProfile.schema === "deepbom.interface_boundary_corpus_validation_profile.v1", "Interface profile schema changed.");
  assert(values.interfaceSweep.schema === "deepbom.quant_policy_boundary_corpus_sweep.v1", "Interface sweep schema changed.");
  assert(values.quantManifest.schema === "deepbom.quant_policy_boundary_corpus.v1", "Quant policy manifest schema changed.");
  assert(values.publicSweep.schema === "deepbom.public_model_corpus_sweep.v1", "Public sweep schema changed.");
  assert(values.hfSummary.schema === "deepbom.huggingface_community_corpus_summary.v1", "Hugging Face summary schema changed.");
  assert(values.residual.schema === "deepbom.residual_coverage_priorities.v1.6", "Residual coverage schema changed.");
  const onnxResidual = values.residual.populations?.onnx;
  assert(onnxResidual
    && onnxResidual.shape_contract_known_node_output_count + onnxResidual.shape_contract_unknown_node_output_count === onnxResidual.node_output_count
    && onnxResidual.invalid_node_output_count + onnxResidual.conditionally_invalid_node_output_count
      + onnxResidual.unresolved_nonconflict_shape_contract_node_output_count === onnxResidual.shape_contract_unknown_node_output_count,
  "Residual coverage shape-contract states do not conserve the measured ONNX output denominator.");
  assert(onnxResidual.total_macs_artifact_contract_conflict_op_count + onnxResidual.total_macs_external_binding_required_op_count
    + onnxResidual.total_macs_analyzer_or_contract_residual_op_count === onnxResidual.total_macs_unresolved_op_count,
  "Residual coverage total-MAC resolution classes do not conserve the measured unresolved denominator.");
  assert(values.safetensors.artifacts.length === values.safetensors.artifact_count, "SafeTensors fixture denominator changed.");
  assert(values.safetensorsQuantization.schema === "deepbom.safetensors_quantization_contract_corpus.v1", "SafeTensors quantization corpus schema changed.");
  assert(values.safetensorsQuantization.artifacts.length === 2
    && values.safetensorsQuantization.artifacts.every((row) => row.measurement.status === "assessed" && row.measurement.invalid_module_count === 0),
  "SafeTensors AWQ/GPTQ contract corpus lost positive structural coverage.");
  assert(values.gguf.artifacts.length === values.gguf.artifact_count, "GGUF fixture denominator changed.");
  assert(values.coreml.artifacts.length === values.coreml.artifact_count, "Core ML fixture denominator changed.");
  assert(values.coremlLegacyQuantization.schema === "deepbom.coreml_legacy_quantization_contract_corpus.v1"
    && values.coremlLegacyQuantization.artifact_count === 1
    && values.coremlLegacyQuantization.artifacts[0]?.baseline?.per_axis_quantized_weight_parameter_count === 1,
  "Core ML legacy per-channel quantization corpus lost positive structural coverage.");
  assert(values.onnxExtensions.schema === "deepbom.onnx_extension_contract_corpus.v1", "ONNX extension contract corpus schema changed.");
  assert(values.onnxExtensions.artifacts.length === values.onnxExtensions.summary.artifact_count, "ONNX extension contract corpus denominator changed.");
  assert(values.onnxExtensions.summary.per_axis_qdq_artifact_count >= 3 && values.onnxExtensions.summary.external_custom_domain_artifact_count >= 1,
    "ONNX extension contract corpus lost positive per-axis or external custom-domain coverage.");
  assert(values.multiformatManifest.schema === "deepbom.public_multiformat_corpus.v1", "Public multiformat manifest schema changed.");
  assert(values.cyclonedxGeneralization.schema === "deepbom.cyclonedx_generalization_evidence.v1", "CycloneDX generalization evidence schema changed.");
  assert(values.cyclonedxGeneralizationRecords.schema === "deepbom.cyclonedx_generalization_records.v1", "CycloneDX generalization record schema changed.");
  assert(values.cyclonedxGeneralizationRecords.ledger_sha256 === values.cyclonedxGeneralization.records.ledger_sha256,
    "CycloneDX generalization record ledger diverged from its index.");
  assert(values.cyclonedxGeneralization.population.unique_primary_artifact_count === values.multiformatManifest.summary.unique_primary_artifact_count,
    "Public multiformat generalization denominator diverged from its manifest.");
  assert(values.cyclonedxQuantization.schema === "deepbom.cyclonedx20_quantization_fixture_evidence.v1", "CycloneDX quantization fixture evidence schema changed.");
  assert(values.cyclonedxQuantization.fixture_results.every((row) => row.expected === row.actual), "CycloneDX candidate fixture result diverged from its declared expectation.");
  assert(values.cyclonedxQuantization.twelve_case_probe.valid_count + values.cyclonedxQuantization.twelve_case_probe.invalid_count === 12, "CycloneDX quantization probe denominator changed.");
}
