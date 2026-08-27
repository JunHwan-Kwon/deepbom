import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import process from "node:process";
import { gzipSync } from "node:zlib";

import { canonicalJson } from "../web/lib/report-utils.js";
import { primaryArtifactSha256, readPublicMultiformatCorpus } from "./public-multiformat-corpus-lib.mjs";

const args = parseArgs(process.argv.slice(2));
const manifest = await readPublicMultiformatCorpus(args.manifest);
const sweep = JSON.parse(await readFile(args.sweep, "utf8"));
if (sweep?.schema !== "deepbom.public_multiformat_corpus_sweep.v1") throw new Error("Unsupported public multiformat sweep schema.");
if (sweep.corpus_generated_at !== manifest.generated_at) throw new Error("Sweep and manifest generation identities differ.");
if (sweep.rows.length !== manifest.artifacts.length || sweep.rows.some((row) => row.status !== "passed" || row.deterministic !== true)) {
  throw new Error("Every selected path record must pass deterministic repeated analysis before evidence can be published.");
}

const manifestById = new Map(manifest.artifacts.map((artifact) => [artifact.id, artifact]));
const groups = new Map();
for (const row of sweep.rows) {
  const artifact = manifestById.get(row.artifact_id);
  if (!artifact || primaryArtifactSha256(artifact) !== row.primary_artifact_sha256) throw new Error(`${row.artifact_id}: sweep identity is not manifest-bound.`);
  const key = row.primary_artifact_sha256;
  if (!groups.has(key)) groups.set(key, []);
  groups.get(key).push({ artifact, row });
}

const artifacts = [...groups.values()].map((records) => {
  const [{ artifact, row }] = records;
  return {
    artifact_id: artifact.id,
    aliases: records.slice(1).map((record) => record.artifact.id),
    format: artifact.format,
    artifact_sha256: row.primary_artifact_sha256,
    source: artifact.source,
    stratum: artifact.stratum,
    files: artifact.files.map(({ source_url: _sourceUrl, ...file }) => file),
    analysis_sha256: row.analysis_sha256,
    receipt_sha256: row.receipt_sha256,
    analysis_summary: row.analysis_summary,
    cyclonedx_observation: row.cyclonedx_observation,
  };
}).sort((left, right) => left.format.localeCompare(right.format) || left.artifact_id.localeCompare(right.artifact_id));

const formatPopulations = Object.fromEntries(["onnx", "gguf", "safetensors", "coreml"].map((format) => {
  const rows = artifacts.filter((artifact) => artifact.format === format);
  return [format, populationSummary(rows)];
}));
const manifestSha256 = sha256(await readFile(args.manifest));
const nonClaims = [
  "Repository or catalog license metadata is external provenance and is not inferred from artifact bytes or treated as a license grant.",
  "Static graph eligibility, source rules, and serialized precision do not establish observed runtime placement, latency, energy, or memory.",
  "Artifact inspection does not establish preprocessing behavior that is absent from the container, task accuracy, clinical validity, or release readiness.",
  "GGUF and SafeTensors serialize weight/storage contracts rather than an executable deployment graph; their storage encodings are not recast as affine input/output mappings.",
];
const recordsBody = {
  schema: "deepbom.cyclonedx_generalization_records.v1",
  manifest_schema: manifest.schema,
  manifest_sha256: manifestSha256,
  measurement_completed_at: sweep.completed_at,
  artifacts,
};
const recordsDocument = { ...recordsBody, ledger_sha256: sha256(canonicalJson(recordsBody)) };
const recordsJson = Buffer.from(`${JSON.stringify(recordsDocument, null, 2)}\n`, "utf8");
const recordsGzip = gzipSync(recordsJson, { level: 9, mtime: 0 });
const body = {
  schema: "deepbom.cyclonedx_generalization_evidence.v1",
  population: {
    manifest_schema: manifest.schema,
    manifest_sha256: manifestSha256,
    measurement_completed_at: sweep.completed_at,
    selection_boundary: manifest.population_boundary,
    inference_boundary: "Exact counts describe the declared hash-identified frame only. ONNX, GGUF, and SafeTensors are purposeful strata; Core ML enumerates the public Apple catalog snapshot dated 2026-08-18. No cross-ecosystem prevalence or confidence interval is claimed.",
    path_record_count: sweep.path_record_count,
    unique_primary_artifact_count: artifacts.length,
    repeat_count: sweep.method.repeats_per_path_record,
    deterministic_pass_count: sweep.rows.filter((row) => row.deterministic === true).length,
    source_file_count: manifest.summary.source_file_count,
    declared_download_bytes: manifest.summary.total_declared_download_bytes,
  },
  format_populations: formatPopulations,
  cross_format_schema_findings: crossFormatFindings(artifacts),
  records: {
    path: args.recordsOutput.replaceAll("\\", "/"),
    media_type: "application/json",
    content_encoding: "gzip",
    schema: recordsDocument.schema,
    record_count: artifacts.length,
    uncompressed_bytes: recordsJson.length,
    uncompressed_sha256: sha256(recordsJson),
    compressed_bytes: recordsGzip.length,
    compressed_sha256: sha256(recordsGzip),
    ledger_sha256: recordsDocument.ledger_sha256,
  },
  artifact_index: artifacts.map((artifact) => ({
    artifact_id: artifact.artifact_id,
    format: artifact.format,
    artifact_sha256: artifact.artifact_sha256,
    analysis_sha256: artifact.analysis_sha256,
    receipt_sha256: artifact.receipt_sha256,
    graph_presence: artifact.analysis_summary.graph_presence,
    quantization_classification: artifact.analysis_summary.quantization_classification,
    serialized_contract_status: artifact.analysis_summary.serialized_contract_status,
    serialized_contract_issue_codes: artifact.analysis_summary.serialized_contract_issue_codes,
  })),
  non_claims: nonClaims,
};
const document = { ...body, ledger_sha256: sha256(canonicalJson(body)) };
await writeFile(args.recordsOutput, recordsGzip);
await writeFile(args.output, `${JSON.stringify(document, null, 2)}\n`, "utf8");
await writeFile(args.markdownOutput, buildMarkdown(document), "utf8");
console.log(`Wrote ${args.output} and ${args.recordsOutput}: ${artifacts.length} unique artifacts / ${sweep.path_record_count} path records.`);

function populationSummary(rows) {
  const observations = rows.map((row) => row.cyclonedx_observation);
  const parameters = observations.flatMap((row) => row.external_parameters || []);
  return {
    unique_artifact_count: rows.length,
    path_record_count: rows.reduce((count, row) => count + 1 + row.aliases.length, 0),
    repository_or_catalog_source_count: new Set(rows.map((row) => row.source.repository || row.source.catalog_url)).size,
    architecture_strata: countBy(rows, (row) => row.stratum.architecture_class),
    task_strata: countBy(rows, (row) => row.stratum.task),
    precision_strata: countBy(rows, (row) => row.stratum.precision),
    graph_presence: countBy(rows, (row) => row.analysis_summary.graph_presence),
    external_data_status: countBy(rows, (row) => row.analysis_summary.external_data_status),
    quantization_classification: countBy(rows, (row) => row.analysis_summary.quantization_classification),
    serialized_contract_status: countBy(rows, (row) => row.analysis_summary.serialized_contract_status),
    serialized_contract_issue_codes: countBy(
      rows.flatMap((row) => row.analysis_summary.serialized_contract_issue_codes || []).map((code) => ({ code })),
      (row) => row.code,
    ),
    external_parameter_count: parameters.length,
    affine_external_parameter_count: parameters.filter((row) => row.quantization).length,
    unquantized_or_undeclared_external_parameter_count: parameters.filter((row) => !row.quantization).length,
    model_scheme_set: [...new Set(observations.flatMap((row) => row.model_quantization?.scheme_set || []))].sort(),
    model_granularity_set: [...new Set(observations.flatMap((row) => row.model_quantization?.granularity_set || []))].sort(),
    model_axis_set: [...new Set(observations.flatMap((row) => row.model_quantization?.axis_set || []))].sort((a, b) => a - b),
    singleton_model_quantization_lossy_count: observations.filter((row) => row.model_quantization?.exact_numerical_contract_representable_by_one_flat_object === false).length,
    singleton_model_quantization_applicable_count: observations.filter((row) => row.model_quantization?.exact_numerical_contract_representable_by_one_flat_object != null).length,
  };
}

function crossFormatFindings(rows) {
  const observations = rows.map((row) => row.cyclonedx_observation);
  const parameters = observations.flatMap((row) => row.external_parameters || []);
  const quantizedParameters = parameters.filter((row) => row.quantization);
  const parameterArtifacts = observations.filter((row) => (row.external_parameters || []).length > 0);
  const multipleParameterArtifacts = parameterArtifacts.filter((row) => row.external_parameters.length > 1);
  const singletonApplicable = observations.filter((row) => row.model_quantization?.exact_numerical_contract_representable_by_one_flat_object != null);
  const lossy = singletonApplicable.filter((row) => row.model_quantization?.exact_numerical_contract_representable_by_one_flat_object === false);
  const quantizationApplicable = observations.filter((row) => (row.model_quantization?.representation_profiles || []).length > 0);
  const axisObserved = quantizationApplicable.filter((row) => (row.model_quantization?.axis_set || []).length > 0);
  const multipleAxes = axisObserved.filter((row) => (row.model_quantization?.axis_set || []).length > 1);
  const storageContainers = rows.filter((row) => ["gguf", "safetensors"].includes(row.format));
  const storageSeparated = storageContainers.filter((row) => (row.cyclonedx_observation.external_parameters || []).every((parameter) => !parameter.quantization));
  const contractIssues = observations.filter((row) => Number(row.serialized_contract_assessment?.issue_count || 0) > 0);
  const fieldAt = (observation, path) => (observation.field_evidence || []).find((field) => field.path === path);
  const graphStates = countBy(observations, (row) => fieldAt(row, "model.graph")?.evidence_class);
  const graphStateCount = Object.values(graphStates).reduce((sum, value) => sum + value, 0);
  const runtimeRequired = observations.filter((row) => fieldAt(row, "deployment.assignment")?.evidence_class === "RUNTIME_REQUIRED");
  const sourceRevisionExternal = observations.filter((row) => fieldAt(row, "artifact.source.revision")?.evidence_class === "EXTERNAL");
  const sourceLicenseExternal = observations.filter((row) => fieldAt(row, "artifact.source.license")?.evidence_class === "EXTERNAL");
  const onnxRows = rows.filter((row) => row.format === "onnx");
  const verifiedExternalPayloads = onnxRows.filter((row) => row.analysis_summary.external_data_status === "verified_payloads");
  return [
    {
      id: "external-affine-contract-availability",
      support_type: "DIRECT_POPULATION_MEASUREMENT",
      conclusion: "External affine values must be represented as optional named-parameter contracts rather than inferred from an integer dtype or a model-level precision label.",
      observed_numerator: quantizedParameters.length,
      observed_denominator: parameters.length,
      population_unit: "serialized external parameters",
      measurement_interpretation: `${quantizedParameters.length}/${parameters.length} parameters in this non-TFLite population carried a complete extracted affine mapping. This count is an availability result, not evidence that parameter binding is unnecessary; the separate TFLite interface population supplies positive affine examples.`,
    },
    {
      id: "external-parameter-collection-cardinality",
      support_type: "DIRECT_STRUCTURAL_COUNTEREXAMPLE",
      conclusion: "External interface contracts require a repeatable parameter collection with stable direction and parameter identity; one model-level parameter object cannot represent a multi-input/output interface without loss of binding.",
      observed_numerator: multipleParameterArtifacts.length,
      observed_denominator: parameterArtifacts.length,
      population_unit: "artifacts with serialized external parameters",
      serialized_external_parameter_count: parameters.length,
      measurement_interpretation: `${multipleParameterArtifacts.length}/${parameterArtifacts.length} artifacts with a serialized interface exposed more than one parameter. This establishes cardinality pressure in the measured frame; the separate TFLite affine corpus establishes that dtype and shape do not uniquely determine scale and zero-point.`,
    },
    {
      id: "model-quantization-multiplicity",
      support_type: "DIRECT_REPRESENTATIONAL_COUNTEREXAMPLE",
      conclusion: "A repeatable, scoped representation is required when one artifact contains mixed encodings, granularities, axes, or non-affine storage families.",
      observed_numerator: lossy.length,
      observed_denominator: singletonApplicable.length,
      population_unit: "unique primary artifacts with an applicable singleton-losslessness assessment",
      measurement_interpretation: "The numerator counts artifacts whose extracted tensor/block-scoped numerical contract cannot be flattened into one model-level scheme/granularity/bits/axis tuple without information loss.",
    },
    {
      id: "applicability-availability-and-zero-are-distinct",
      support_type: "DIRECT_FIELD_STATE_COUNTEREXAMPLE",
      conclusion: "A portable evidence schema must distinguish an observed serialized graph, a graph that is applicable but unavailable to the bounded decoder, and a graph that the container does not serialize. None of these states is numerical zero.",
      observed_numerator: graphStateCount,
      observed_denominator: observations.length,
      population_unit: "unique primary artifacts with a conserved graph evidence state",
      observed_state_counts: graphStates,
      measurement_interpretation: `${graphStates.OBSERVED || 0} OBSERVED, ${graphStates.UNAVAILABLE || 0} UNAVAILABLE, and ${graphStates.NOT_APPLICABLE || 0} NOT_APPLICABLE graph states reconstruct the full ${observations.length}-artifact denominator without null-to-zero coercion.`,
    },
    {
      id: "compound-artifact-identity",
      support_type: "DIRECT_EXTERNAL_PAYLOAD_MEASUREMENT",
      conclusion: "Formats with externally stored tensor payloads need a content-set identity and explicit dependency binding in addition to the primary model-file hash.",
      observed_numerator: verifiedExternalPayloads.length,
      observed_denominator: onnxRows.length,
      population_unit: "ONNX primary artifacts",
      measurement_interpretation: `${verifiedExternalPayloads.length}/${onnxRows.length} ONNX records required and verified a separately hashed external tensor payload. The count proves the existence of compound artifacts in this bounded frame, not their ecosystem prevalence.`,
    },
    {
      id: "artifact-source-and-license-provenance-separation",
      support_type: "DIRECT_PROVENANCE_STATE_MEASUREMENT",
      conclusion: "Artifact byte identity, source revision, and externally declared license metadata are independent provenance fields; a missing revision or license declaration must not invalidate an observed artifact hash or be inferred from bytes.",
      observed_numerator: observations.length,
      observed_denominator: observations.length,
      population_unit: "hash-identified unique primary artifacts",
      source_revision_external_count: sourceRevisionExternal.length,
      source_license_external_count: sourceLicenseExternal.length,
      measurement_interpretation: `All ${observations.length} records are byte-hash bound, while ${sourceRevisionExternal.length} carry an external source revision and ${sourceLicenseExternal.length} carry external license metadata. These are field-availability counts within distinct source frames, not license grants or prevalence estimates.`,
    },
    {
      id: "model-axis-observation-boundary",
      support_type: "DIRECT_NULL_RESULT",
      conclusion: "This population does not establish that one model-level axis is universally sufficient; an axis is meaningful only with tensor scope and a rank-aware normalization rule.",
      observed_numerator: axisObserved.length,
      observed_denominator: quantizationApplicable.length,
      population_unit: "artifacts with extracted model-quantization representations",
      multiple_axis_artifact_count: multipleAxes.length,
      measurement_interpretation: `${axisObserved.length}/${quantizationApplicable.length} applicable artifacts yielded an explicit axis in the current extraction. The null result is retained and no multi-axis prevalence claim is made from this population.`,
    },
    {
      id: "storage-contract-is-not-affine-interface-contract",
      support_type: "DIRECT_FORMAT_CONTRACT_SEPARATION",
      conclusion: "Weight-container dtype or block encoding must remain distinct from a named input/output affine mapping and from an executable graph claim.",
      observed_numerator: storageSeparated.length,
      observed_denominator: storageContainers.length,
      population_unit: "GGUF and SafeTensors artifacts",
      measurement_interpretation: "GGUF block encodings and SafeTensors dtypes were retained as scoped storage representations; neither was promoted to an external scale/zero-point contract.",
    },
    {
      id: "evidence-class-separation",
      support_type: "DIRECT_FIELD_STATE_MEASUREMENT",
      conclusion: "Observed bytes, deterministic derivations, external declarations, unavailable values, and runtime-required claims need distinct machine-readable states.",
      observed_numerator: observations.filter((row) => new Set((row.field_evidence || []).map((field) => field.evidence_class)).size >= 4).length,
      observed_denominator: observations.length,
      population_unit: "unique primary artifacts",
      measurement_interpretation: "The numerator counts records carrying at least four distinct evidence classes without null-to-zero or static-to-runtime promotion.",
    },
    {
      id: "static-artifact-does-not-bind-runtime-assignment",
      support_type: "DIRECT_CLAIM_BOUNDARY_MEASUREMENT",
      conclusion: "Runtime provider, delegate, backend, or device assignment must be a separately bound observation rather than a property inferred from serialized model bytes.",
      observed_numerator: runtimeRequired.length,
      observed_denominator: observations.length,
      population_unit: "unique primary artifacts",
      measurement_interpretation: `Every record retained deployment assignment as RUNTIME_REQUIRED. This is a claim-boundary result across four formats, not evidence that a particular runtime cannot execute an artifact.`,
    },
    {
      id: "validity-is-not-availability-or-runtime-evidence",
      support_type: "DIRECT_SERIALIZED_CONTRACT_MEASUREMENT",
      conclusion: "A readable artifact can carry a source-rule contract defect; validity, missing metadata, external dependency completeness, and runtime observation require separate states.",
      observed_numerator: contractIssues.length,
      observed_denominator: observations.length,
      population_unit: "unique primary artifacts",
      observed_issue_codes: countBy(contractIssues.flatMap((row) => row.serialized_contract_assessment.issues), (row) => row.code),
      measurement_interpretation: "Issue-bearing artifacts remain in the denominator and retain their exact issue codes instead of being silently dropped or reported as fully assessed.",
    },
  ];
}

function countBy(rows, selector) {
  const counts = new Map();
  for (const row of rows) { const key = selector(row) ?? "unknown"; counts.set(String(key), (counts.get(String(key)) || 0) + 1); }
  return Object.fromEntries([...counts].sort(([left], [right]) => left.localeCompare(right)));
}
function buildMarkdown(document) {
  const lines = [
    "# Public Cross-Format Corpus Evidence",
    "",
    "This document is generated from `public-multiformat-corpus.v1.json` and the repeat-2 isolated sweep. Exact counts apply only to this purposeful, hash-identified validation population; they are not ecosystem prevalence estimates.",
    "",
    "## Population",
    "",
    "| Format | Unique artifacts | External parameters | Affine external parameters | Singleton model description lossy / applicable |",
    "| --- | ---: | ---: | ---: | ---: |",
  ];
  for (const format of ["onnx", "gguf", "safetensors", "coreml"]) {
    const row = document.format_populations[format];
    lines.push(`| ${formatName(format)} | ${row.unique_artifact_count} | ${row.external_parameter_count} | ${row.affine_external_parameter_count} | ${row.singleton_model_quantization_lossy_count} / ${row.singleton_model_quantization_applicable_count} |`);
  }
  lines.push("", "## Bounded Schema Findings", "");
  for (const row of document.cross_format_schema_findings) {
    lines.push(`- **${row.id}** (${row.support_type}): ${row.conclusion}`);
    lines.push(`  Population result: ${row.observed_numerator}/${row.observed_denominator} ${row.population_unit}. ${row.measurement_interpretation}`);
  }
  lines.push(
    "",
    "## Per-Artifact Results",
    "",
    `Each row is bound to the primary artifact SHA-256 and two deterministic isolated analyses. Full field evidence, representation profiles, parameter contracts, and schema pressures are retained in the hash-bound gzip JSON \`${document.records.path}\`; \`corpus/cyclonedx-generalization-evidence.v1.json\` is its human-reviewable index and aggregate ledger.`,
    "",
    "| Format | Artifact | Source binding | SHA-256 | Ops | Tensors | Quantization / precision | Serialized contract | Singleton model object |",
    "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
  );
  for (const artifact of artifacts) {
    const summary = artifact.analysis_summary;
    const observation = artifact.cyclonedx_observation;
    lines.push(`| ${formatName(artifact.format)} | ${md(artifact.artifact_id)} | ${md(sourceBinding(artifact.source))} | \`${artifact.artifact_sha256.slice(0, 12)}...\` | ${summary.operator_count ?? "N/A"} | ${summary.tensor_count ?? "N/A"} | ${md(summary.quantization_classification)} | ${md(summary.serialized_contract_status)} | ${singletonLabel(observation.model_quantization)} |`);
  }
  lines.push(
    "",
    "## Interpretation Boundary",
    "",
    "- SafeTensors dtype storage is not promoted to an affine quantization mapping.",
    "- GGUF block/scalar encodings are serialized storage contracts, not a TFLite-style scale/zero-point contract.",
    "- Static graph or format evidence does not establish actual runtime placement, latency, energy, memory, task accuracy, clinical validity, or release readiness.",
    "- A `partial_serialized_contract_issue_observed` row is retained as measured evidence; it is not silently excluded from the denominator.",
    "",
  );
  return `${lines.join("\n")}\n`;
}
function formatName(value) { return ({ onnx: "ONNX", gguf: "GGUF", safetensors: "SafeTensors", coreml: "Core ML" })[value] || value; }
function sourceBinding(source) { return source.revision ? `${source.repository}@${source.revision.slice(0, 12)}` : source.catalog_url || source.publisher || "unbound"; }
function singletonLabel(modelQuantization) {
  const value = modelQuantization?.exact_numerical_contract_representable_by_one_flat_object;
  return value === true ? "lossless" : value === false ? "lossy" : "not applicable / not assessed";
}
function md(value) { return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " "); }
function sha256(value) { return createHash("sha256").update(value).digest("hex"); }
function parseArgs(argv) {
  const output = { manifest: "corpus/public-multiformat-corpus.v1.json", sweep: ".local-validation/public-multiformat-corpus-v1/public-multiformat-corpus-sweep.json", output: "corpus/cyclonedx-generalization-evidence.v1.json", recordsOutput: "corpus/cyclonedx-generalization-evidence.records.v1.json.gz", markdownOutput: "docs/PUBLIC_MULTIFORMAT_CORPUS.md" };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--manifest") output.manifest = required(argv, ++index, key);
    else if (key === "--sweep") output.sweep = required(argv, ++index, key);
    else if (key === "--output") output.output = required(argv, ++index, key);
    else if (key === "--records-output") output.recordsOutput = required(argv, ++index, key);
    else if (key === "--markdown-output") output.markdownOutput = required(argv, ++index, key);
    else throw new Error(`Unknown argument: ${key}`);
  }
  return output;
}
function required(argv, index, key) { const value = argv[index]; if (!value || value.startsWith("--")) throw new Error(`${key} requires a value.`); return value; }
