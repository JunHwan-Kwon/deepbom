import { sha256Text } from "./google-legacy-corpus-lib.mjs";
import {
  buildInterfaceBoundaryContract,
  buildInterfaceQuantizationContractLedger,
} from "../web/lib/quantization-contract-summary.js";

export function summarizeInterfaceContracts(analysis) {
  return buildInterfaceQuantizationContractLedger(analysis);
}

export function buildInterfaceContractCorpusSummary(rows) {
  const boundaryRows = rows.map((row) => ({
    row,
    boundary: row.interface_contracts?.boundary_contract
      || buildInterfaceBoundaryContract(row.interface_contracts?.parameters || []),
  }));
  const parameters = rows.flatMap((row) =>
    (row.interface_contracts?.parameters || []).map((parameter) => ({
      qualified_id: row.qualified_id,
      artifact_sha256: row.artifact_sha256,
      parameter_id: parameter.parameter_id,
      direction: parameter.direction,
      ordinal: parameter.ordinal,
      tensor_name: parameter.tensor_name,
      dtype: parameter.dtype,
      shape: parameter.shape,
      quantization: parameter.quantization,
    })));
  const complete = parameters.filter(
    (row) => row.quantization?.status === "complete",
  );
  const signatureGroups = new Map();
  for (const parameter of complete) {
    const signature = JSON.stringify({
      direction: parameter.direction,
      dtype: parameter.dtype,
      shape: parameter.shape,
    });
    const group = signatureGroups.get(signature) || {
      signature: JSON.parse(signature),
      artifacts: new Set(),
      contracts: new Map(),
      examples: [],
    };
    group.artifacts.add(parameter.artifact_sha256);
    group.contracts.set(
      parameter.quantization.contract_sha256,
      parameter.quantization,
    );
    if (group.examples.length < 12) {
      group.examples.push({
        qualified_id: parameter.qualified_id,
        artifact_sha256: parameter.artifact_sha256,
        parameter_id: parameter.parameter_id,
        tensor_name: parameter.tensor_name,
        contract_sha256: parameter.quantization.contract_sha256,
        granularity: parameter.quantization.granularity,
        scales: parameter.quantization.scales,
        zero_points: parameter.quantization.zero_points,
        quantized_dimension: parameter.quantization.quantized_dimension,
        scalar_real_code_domain: parameter.quantization.scalar_real_code_domain,
      });
    }
    signatureGroups.set(signature, group);
  }
  const ambiguousSchemaSignatures = [...signatureGroups.values()]
    .filter((group) => group.artifacts.size > 1 && group.contracts.size > 1)
    .map((group) => ({
      ...group.signature,
      artifact_count: group.artifacts.size,
      distinct_affine_contract_count: group.contracts.size,
      examples: group.examples,
    }))
    .sort((left, right) =>
      right.artifact_count - left.artifact_count
      || right.distinct_affine_contract_count - left.distinct_affine_contract_count
      || JSON.stringify(left).localeCompare(JSON.stringify(right)));
  const boundaryStatusCounts = countByBoundaryStatus(boundaryRows);
  const inputBoundaryStatusCounts = countByBoundaryStatus(
    boundaryRows.map(({ row, boundary }) => ({ row, boundary: boundary.inputs })),
  );
  const outputBoundaryStatusCounts = countByBoundaryStatus(
    boundaryRows.map(({ row, boundary }) => ({ row, boundary: boundary.outputs })),
  );
  const integerInternal = boundaryRows.filter(({ row }) =>
    ["full_integer", "integer_internal_float_io"].includes(
      row.quantization_classification,
    ));
  const integerInternalNonFullyQuantized = integerInternal.filter(
    ({ boundary }) => boundary.status !== "fully_affine_quantized",
  );
  const integerInternalWithUnquantizedIo = integerInternal.filter(
    ({ boundary }) => Number(boundary.unquantized_parameter_count || 0) > 0,
  );
  const integerInternalBoundaryRows = integerInternalNonFullyQuantized.map(
    ({ row, boundary }) => {
      const histogram = row.observed_signals?.operator_histogram || {};
      return {
        qualified_id: row.qualified_id,
        artifact_sha256: row.artifact_sha256,
        boundary_status: boundary.status,
        input_boundary_status: boundary.inputs.status,
        output_boundary_status: boundary.outputs.status,
        float32_parameter_count: boundary.float32_parameter_count,
        quantize_operator_count: Number(histogram.QUANTIZE || 0),
        dequantize_operator_count: Number(histogram.DEQUANTIZE || 0),
      };
    },
  );
  const value = {
    schema: "deepbom.interface_quantization_contract_corpus_summary.v1.1",
    public_artifact_count: rows.length,
    parameter_count: parameters.length,
    complete_quantized_parameter_count: complete.length,
    unquantized_parameter_count: parameters.filter(
      (row) => row.quantization?.status === "not_quantized",
    ).length,
    invalid_or_incomplete_parameter_count: parameters.filter(
      (row) => row.quantization?.status === "invalid_or_incomplete",
    ).length,
    per_tensor_parameter_count: complete.filter(
      (row) => row.quantization?.granularity === "per_tensor",
    ).length,
    per_axis_parameter_count: complete.filter(
      (row) => row.quantization?.granularity === "per_axis",
    ).length,
    artifacts_with_any_complete_quantized_parameter: rows.filter(
      (row) => (row.interface_contracts?.quantized_parameter_count || 0) > 0,
    ).length,
    artifacts_with_fully_quantized_interface:
      boundaryStatusCounts.fully_affine_quantized || 0,
    artifacts_with_mixed_quantized_interface:
      boundaryStatusCounts.mixed_quantized_unquantized || 0,
    artifacts_with_fully_unquantized_interface:
      boundaryStatusCounts.fully_unquantized || 0,
    artifacts_with_invalid_or_incomplete_interface:
      boundaryStatusCounts.invalid_or_incomplete || 0,
    artifacts_with_no_declared_interface: boundaryStatusCounts.not_declared || 0,
    interface_boundary_status_counts: boundaryStatusCounts,
    input_boundary_status_counts: inputBoundaryStatusCounts,
    output_boundary_status_counts: outputBoundaryStatusCounts,
    subcohort_interface_boundary_status_counts: subcohortBoundaryCounts(boundaryRows),
    internal_integer_artifact_count: integerInternal.length,
    internal_integer_with_fully_quantized_interface: integerInternal.filter(
      ({ boundary }) => boundary.status === "fully_affine_quantized",
    ).length,
    internal_integer_with_mixed_interface: integerInternal.filter(
      ({ boundary }) => boundary.status === "mixed_quantized_unquantized",
    ).length,
    internal_integer_with_fully_unquantized_interface: integerInternal.filter(
      ({ boundary }) => boundary.status === "fully_unquantized",
    ).length,
    internal_integer_with_any_unquantized_io:
      integerInternalWithUnquantizedIo.length,
    internal_integer_with_invalid_or_incomplete_interface: integerInternal.filter(
      ({ boundary }) => boundary.status === "invalid_or_incomplete",
    ).length,
    internal_integer_with_no_declared_interface: integerInternal.filter(
      ({ boundary }) => boundary.status === "not_declared",
    ).length,
    internal_integer_non_fully_quantized_interface_with_quantize_op:
      integerInternalBoundaryRows.filter((row) => row.quantize_operator_count > 0).length,
    internal_integer_non_fully_quantized_interface_with_dequantize_op:
      integerInternalBoundaryRows.filter((row) => row.dequantize_operator_count > 0).length,
    internal_integer_non_fully_quantized_interface_examples:
      integerInternalBoundaryRows,
    artifacts_with_multiple_complete_interface_contracts: rows.filter(
      (row) =>
        row.interface_contracts?.multiple_complete_quantization_contracts_within_artifact,
    ).length,
    dtype_shape_signatures_with_multiple_affine_contracts:
      ambiguousSchemaSignatures.length,
    dtype_shape_ambiguity_groups: ambiguousSchemaSignatures,
    denominator_definitions: {
      fully_quantized_interface: "Every declared external input and output has a complete, cardinality-valid affine quantization contract.",
      mixed_quantized_interface: "At least one declared external parameter is affine-quantized and at least one is explicitly unquantized.",
      fully_unquantized_interface: "Every declared external input and output is explicitly unquantized; FLOAT32 is a boundary fact rather than missing affine metadata.",
      internal_integer_artifact: "Analyzer quantization_classification is full_integer or integer_internal_float_io.",
      internal_integer_with_any_unquantized_io: "At least one declared external input or output explicitly has no affine quantization parameters. Invalid/incomplete and undeclared boundaries are excluded.",
    },
    interpretation_boundary:
      "The corpus summary records serialized boundary dtype, shape, and affine quantization contracts. It does not establish RGB/BGR order, source-value normalization, mean/standard-deviation transforms, resize interpolation, application tensor layout, deployed harness behavior, or task accuracy. Q/DQ operator counts are graph observations and do not by themselves prove that every observed operator is adjacent to an external boundary.",
  };
  return {
    ...value,
    ledger_sha256: sha256Text(JSON.stringify(value)),
  };
}

function countByBoundaryStatus(rows) {
  return Object.fromEntries(
    [...new Set(rows.map(({ boundary }) => boundary.status))]
      .sort()
      .map((status) => [
        status,
        rows.filter(({ boundary }) => boundary.status === status).length,
      ]),
  );
}

function subcohortBoundaryCounts(rows) {
  return Object.fromEntries(
    [...new Set(rows.map(({ row }) => row.subcohort_id || "unspecified"))]
      .sort()
      .map((subcohort) => {
        const members = rows.filter(
          ({ row }) => (row.subcohort_id || "unspecified") === subcohort,
        );
        return [subcohort, {
          artifact_count: members.length,
          ...countByBoundaryStatus(members),
        }];
      }),
  );
}
