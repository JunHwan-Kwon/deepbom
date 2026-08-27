import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import { canonicalContractSha256 } from "../web/lib/interface-contract.js";

const args = parseArgs(process.argv.slice(2));
const sweepPath = path.resolve(args.sweep);
const outputPath = path.resolve(
  args.output || path.join(path.dirname(sweepPath), "interface-boundary-corpus-review.json"),
);
const sweepBytes = await readFile(sweepPath);
const sweep = JSON.parse(sweepBytes.toString("utf8"));
const manifestBytes = await readFile("corpus/quant_policy/manifest.v1.json");
const manifest = JSON.parse(manifestBytes.toString("utf8"));
const errors = [];

check(sweep.schema === "deepbom.quant_policy_boundary_corpus_sweep.v1.1", "sweep schema");
check(sweep.corpus_id === manifest.corpus_id, "sweep corpus id matches manifest");
check(sweep.manifest_sha256 === sha256(manifestBytes), "sweep manifest SHA-256");
check(sweep.requested_public_artifact_count === manifest.public_artifact_count, "requested denominator");
check(sweep.passed_public_artifact_count === manifest.public_artifact_count, "passed denominator");
check(sweep.failed_public_artifact_count === 0, "zero failed public artifacts");
check(sweep.repeat_count === 2, "two isolated repeats");

const rows = sweep.rows.filter((row) => row.public_corpus_member);
check(rows.length === manifest.public_artifact_count, "row denominator");
check(rows.every((row) => row.status === "passed"), "every public row passed");
check(rows.every((row) => row.deterministic === true), "every public row deterministic");
check(rows.every((row) => row.repeat_count === sweep.repeat_count), "row repeat counts");
check(rows.every((row) => row.quality?.size_breakdown_conservation_valid === true), "size conservation");
check(rows.every((row) => row.quality?.non_finite_paths?.length === 0), "no non-finite values");
check(rows.every((row) => row.quality?.phantom_reference_count === 0), "no phantom references");
check(new Set(rows.map((row) => row.artifact_sha256)).size === rows.length, "unique artifact SHA-256");
check(rows.every((row) => /^[0-9a-f]{64}$/.test(row.analysis_sha256)), "analysis SHA-256 syntax");

const wasmBytes = await readFile("pkg/tflite_wasm_audit_bg.wasm");
check(sweep.analyzer_artifact_identity?.wasm?.sha256 === sha256(wasmBytes), "current WASM SHA-256");
check(sweep.analyzer_artifact_identity?.wasm?.size === wasmBytes.length, "current WASM size");

const classified = rows.map((row) => {
  const parameters = row.interface_contracts?.parameters || [];
  check(
    row.interface_contracts?.schema === "deepbom.interface_quantization_contracts.v1.3",
    `${row.qualified_id}: interface schema`,
  );
  check(parameters.length > 0, `${row.qualified_id}: external parameters declared`);
  verifyLedger(row.interface_contracts, `${row.qualified_id}: interface ledger`);
  for (const parameter of parameters) verifyParameter(row, parameter);
  const boundary = classifyBoundary(parameters);
  const input = classifyBoundary(parameters.filter((parameter) => parameter.direction === "input"));
  const output = classifyBoundary(parameters.filter((parameter) => parameter.direction === "output"));
  check(
    row.interface_contracts?.boundary_contract?.status === boundary.status,
    `${row.qualified_id}: artifact boundary status`,
  );
  check(
    row.interface_contracts?.boundary_contract?.inputs?.status === input.status,
    `${row.qualified_id}: input boundary status`,
  );
  check(
    row.interface_contracts?.boundary_contract?.outputs?.status === output.status,
    `${row.qualified_id}: output boundary status`,
  );
  return { row, boundary, input, output };
});

const parameters = classified.flatMap(({ row }) => row.interface_contracts.parameters);
const summary = sweep.interface_quantization_contract_summary;
verifyLedger(summary, "corpus interface summary");
check(summary.schema === "deepbom.interface_quantization_contract_corpus_summary.v1.1", "summary schema");

const boundaryCounts = countBy(classified, ({ boundary }) => boundary.status);
const inputCounts = countBy(classified, ({ input }) => input.status);
const outputCounts = countBy(classified, ({ output }) => output.status);
const anyComplete = classified.filter(({ row }) =>
  row.interface_contracts.parameters.some(
    (parameter) => parameter.quantization.status === "complete",
  )).length;
const internalInteger = classified.filter(({ row }) =>
  ["full_integer", "integer_internal_float_io"].includes(
    row.quantization_classification,
  ));
const internalNonFully = internalInteger.filter(
  ({ boundary }) => boundary.status !== "fully_affine_quantized",
);
const internalWithUnquantizedIo = internalInteger.filter(
  ({ boundary }) => boundary.unquantized > 0,
);
const internalExamples = internalNonFully.map(({ row, boundary, input, output }) => ({
  qualified_id: row.qualified_id,
  artifact_sha256: row.artifact_sha256,
  boundary_status: boundary.status,
  input_boundary_status: input.status,
  output_boundary_status: output.status,
  parameters: row.interface_contracts.parameters.map((parameter) => ({
    direction: parameter.direction,
    ordinal: parameter.ordinal,
    dtype: parameter.dtype,
    quantization_status: parameter.quantization.status,
  })),
  quantize_operator_count: Number(row.observed_signals?.operator_histogram?.QUANTIZE || 0),
  dequantize_operator_count: Number(row.observed_signals?.operator_histogram?.DEQUANTIZE || 0),
}));
const subcohorts = Object.fromEntries(
  [...new Set(classified.map(({ row }) => row.subcohort_id))].sort().map((id) => {
    const members = classified.filter(({ row }) => row.subcohort_id === id);
    return [id, { artifact_count: members.length, ...countBy(members, ({ boundary }) => boundary.status) }];
  }),
);

const derived = {
  public_artifact_count: rows.length,
  external_parameter_count: parameters.length,
  complete_affine_parameter_count: parameters.filter(
    (parameter) => parameter.quantization.status === "complete",
  ).length,
  explicitly_unquantized_parameter_count: parameters.filter(
    (parameter) => parameter.quantization.status === "not_quantized",
  ).length,
  invalid_or_incomplete_parameter_count: parameters.filter(
    (parameter) => parameter.quantization.status === "invalid_or_incomplete",
  ).length,
  artifacts_with_any_complete_affine_parameter: anyComplete,
  boundary_status_counts: boundaryCounts,
  input_boundary_status_counts: inputCounts,
  output_boundary_status_counts: outputCounts,
  valid_explicit_boundary_contract_artifact_count: classified.filter(
    ({ boundary }) => !["invalid_or_incomplete", "not_declared"].includes(boundary.status),
  ).length,
  internal_integer_artifact_count: internalInteger.length,
  internal_integer_boundary_status_counts: countBy(
    internalInteger,
    ({ boundary }) => boundary.status,
  ),
  internal_integer_with_any_unquantized_io: internalWithUnquantizedIo.length,
  internal_integer_with_invalid_or_incomplete_interface: internalInteger.filter(
    ({ boundary }) => boundary.status === "invalid_or_incomplete",
  ).length,
  internal_integer_with_no_declared_interface: internalInteger.filter(
    ({ boundary }) => boundary.status === "not_declared",
  ).length,
  internal_integer_non_fully_quantized_with_quantize_op: internalExamples.filter(
    (row) => row.quantize_operator_count > 0,
  ).length,
  internal_integer_non_fully_quantized_with_dequantize_op: internalExamples.filter(
    (row) => row.dequantize_operator_count > 0,
  ).length,
  subcohort_boundary_status_counts: subcohorts,
  internal_integer_non_fully_quantized_examples: internalExamples,
};

compareSummary(summary, derived);

const review = {
  schema: "deepbom.interface_boundary_corpus_review.v1",
  source_sweep: path.relative(process.cwd(), sweepPath).replaceAll("\\", "/"),
  source_sweep_sha256: sha256(sweepBytes),
  source_manifest_sha256: sha256(manifestBytes),
  analyzer_wasm_sha256: sha256(wasmBytes),
  analyzer_wasm_size: wasmBytes.length,
  repeat_count: sweep.repeat_count,
  verification_status: errors.length ? "fail" : "pass",
  derived,
  exact_ratios: {
    fully_quantized_interface_over_public_corpus: ratio(
      boundaryCounts.fully_affine_quantized || 0,
      rows.length,
    ),
    any_affine_quantized_interface_parameter_over_public_corpus: ratio(
      anyComplete,
      rows.length,
    ),
    fully_quantized_interface_over_internal_integer_artifacts: ratio(
      internalInteger.filter(({ boundary }) => boundary.status === "fully_affine_quantized").length,
      internalInteger.length,
    ),
    internal_integer_with_any_unquantized_io: ratio(
      internalWithUnquantizedIo.length,
      internalInteger.length,
    ),
    valid_explicit_boundary_contract_over_public_corpus: ratio(
      derived.valid_explicit_boundary_contract_artifact_count,
      rows.length,
    ),
  },
  proposal_interpretation: {
    affine_only_field_coverage: "A quantization-only interface field is complete for all external parameters in 30/50 artifacts, present for at least one external parameter in 32/50, and absent by design for every external parameter in 18/50.",
    boundary_contract_coverage: "A boundary contract that explicitly records dtype/shape plus affine parameters when applicable represents all 50/50 artifacts without treating FLOAT32 as missing data.",
    scope_limit: "The boundary contract records serialized tensor storage facts only. It does not establish RGB/BGR order, source-value normalization, mean/standard-deviation transforms, resize interpolation, application tensor layout, semantic labels, deployed preprocessing behavior, or task accuracy.",
    sampling_limit: "This is an exact measurement of a predeclared, hash-pinned policy-boundary corpus, not a random estimate of ecosystem prevalence. Subcohort results must accompany the aggregate.",
  },
  errors,
};
const canonicalReview = canonicalJson(review);
review.ledger_sha256 = sha256(Buffer.from(canonicalReview));
await writeFile(outputPath, `${JSON.stringify(review, null, 2)}\n`, "utf8");

console.log(JSON.stringify({
  verification_status: review.verification_status,
  output: path.relative(process.cwd(), outputPath),
  ...derived,
  exact_ratios: review.exact_ratios,
  errors,
}, null, 2));
if (errors.length) process.exitCode = 1;

function verifyParameter(row, parameter) {
  const quantization = parameter.quantization || {};
  check(["input", "output"].includes(parameter.direction), `${row.qualified_id}: parameter direction`);
  check(Number.isSafeInteger(parameter.ordinal) && parameter.ordinal >= 0, `${row.qualified_id}: parameter ordinal`);
  check(typeof parameter.dtype === "string" && parameter.dtype.length > 0, `${row.qualified_id}: parameter dtype`);
  check(Array.isArray(parameter.shape), `${row.qualified_id}: parameter shape`);
  if (quantization.status === "complete") {
    check(quantization.scales?.length > 0, `${row.qualified_id}: complete scales`);
    check(quantization.zero_points?.length > 0, `${row.qualified_id}: complete zero-points`);
    check(quantization.cardinality_status === "valid", `${row.qualified_id}: affine cardinality`);
  } else if (quantization.status === "not_quantized") {
    check(quantization.scales?.length === 0, `${row.qualified_id}: unquantized scales empty`);
    check(quantization.zero_points?.length === 0, `${row.qualified_id}: unquantized zero-points empty`);
  } else {
    check(quantization.status === "invalid_or_incomplete", `${row.qualified_id}: quantization status`);
  }
}

function classifyBoundary(parameters) {
  const complete = parameters.filter((parameter) => parameter.quantization?.status === "complete").length;
  const unquantized = parameters.filter(
    (parameter) => parameter.quantization?.status === "not_quantized",
  ).length;
  const invalid = parameters.length - complete - unquantized;
  let status = "not_declared";
  if (invalid > 0) status = "invalid_or_incomplete";
  else if (parameters.length > 0 && complete === parameters.length) status = "fully_affine_quantized";
  else if (parameters.length > 0 && unquantized === parameters.length) status = "fully_unquantized";
  else if (parameters.length > 0) status = "mixed_quantized_unquantized";
  return { status, parameter_count: parameters.length, complete, unquantized, invalid };
}

function compareSummary(summary, value) {
  equal(summary.public_artifact_count, value.public_artifact_count, "summary artifact count");
  equal(summary.parameter_count, value.external_parameter_count, "summary parameter count");
  equal(summary.complete_quantized_parameter_count, value.complete_affine_parameter_count, "summary complete parameters");
  equal(summary.unquantized_parameter_count, value.explicitly_unquantized_parameter_count, "summary unquantized parameters");
  equal(summary.invalid_or_incomplete_parameter_count, value.invalid_or_incomplete_parameter_count, "summary invalid parameters");
  equal(summary.artifacts_with_any_complete_quantized_parameter, value.artifacts_with_any_complete_affine_parameter, "summary any affine artifact count");
  equal(summary.artifacts_with_fully_quantized_interface, value.boundary_status_counts.fully_affine_quantized || 0, "summary fully quantized artifacts");
  equal(summary.artifacts_with_mixed_quantized_interface, value.boundary_status_counts.mixed_quantized_unquantized || 0, "summary mixed artifacts");
  equal(summary.artifacts_with_fully_unquantized_interface, value.boundary_status_counts.fully_unquantized || 0, "summary fully unquantized artifacts");
  deepEqual(summary.interface_boundary_status_counts, value.boundary_status_counts, "summary boundary crosstab");
  deepEqual(summary.input_boundary_status_counts, value.input_boundary_status_counts, "summary input crosstab");
  deepEqual(summary.output_boundary_status_counts, value.output_boundary_status_counts, "summary output crosstab");
  deepEqual(summary.subcohort_interface_boundary_status_counts, value.subcohort_boundary_status_counts, "summary subcohort crosstab");
  equal(summary.internal_integer_artifact_count, value.internal_integer_artifact_count, "summary internal integer denominator");
  equal(summary.internal_integer_with_any_unquantized_io, value.internal_integer_with_any_unquantized_io, "summary internal integer unquantized I/O");
  equal(summary.internal_integer_with_invalid_or_incomplete_interface, value.internal_integer_with_invalid_or_incomplete_interface, "summary internal integer invalid interface");
  equal(summary.internal_integer_with_no_declared_interface, value.internal_integer_with_no_declared_interface, "summary internal integer undeclared interface");
  equal(summary.internal_integer_non_fully_quantized_interface_with_quantize_op, value.internal_integer_non_fully_quantized_with_quantize_op, "summary QUANTIZE observations");
  equal(summary.internal_integer_non_fully_quantized_interface_with_dequantize_op, value.internal_integer_non_fully_quantized_with_dequantize_op, "summary DEQUANTIZE observations");
}

function verifyLedger(value, label) {
  const { ledger_sha256: recorded, ledger_hash: descriptor, ...payload } = value || {};
  if (value?.schema === "deepbom.interface_quantization_contracts.v1.3") {
    const computed = canonicalContractSha256(payload);
    check(recorded === computed, `${label} canonical SHA-256`);
    check(descriptor?.sha256 === computed, `${label} hash descriptor SHA-256`);
    check(descriptor?.canonicalization === "RFC8785-JCS", `${label} canonicalization`);
  } else {
    check(recorded === sha256(Buffer.from(JSON.stringify(payload))), `${label} SHA-256`);
  }
}

function countBy(values, selector) {
  const counts = new Map();
  for (const value of values) {
    const key = selector(value);
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return Object.fromEntries([...counts].sort(([left], [right]) => left.localeCompare(right)));
}

function ratio(numerator, denominator) {
  return {
    numerator,
    denominator,
    decimal: denominator ? numerator / denominator : null,
    percent: denominator ? numerator * 100 / denominator : null,
  };
}

function check(condition, label) {
  if (!condition) errors.push(label);
}

function equal(actual, expected, label) {
  check(actual === expected, `${label}: expected ${expected}, observed ${actual}`);
}

function deepEqual(actual, expected, label) {
  check(canonicalJson(actual) === canonicalJson(expected), `${label}: values differ`);
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function parseArgs(argv) {
  const result = {
    sweep: ".local-validation/interface-boundary-corpus-2026-08-05/quant-policy-boundary-sweep.json",
    output: "",
  };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--sweep") result.sweep = argv[++index];
    else if (argv[index] === "--output") result.output = argv[++index];
    else throw new Error(`Unknown argument: ${argv[index]}`);
  }
  return result;
}
