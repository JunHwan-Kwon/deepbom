import { readFile } from "node:fs/promises";

import { createCheck } from "./check-assert.mjs";
import { readCuratedMicroCorpus } from "./curated-micro-corpus-lib.mjs";
import {
  loadResolvedGoogleModernComparators,
  readGoogleModernComparators,
} from "./google-modern-comparator-lib.mjs";
import { readGoogleLegacyManifest } from "./google-legacy-corpus-lib.mjs";
import { readCorpusManifest } from "./public-model-corpus-lib.mjs";
import {
  buildInterfaceContractCorpusSummary,
  summarizeInterfaceContracts,
} from "./interface-quantization-contracts.mjs";

const check = createCheck("quant-policy-corpus");
const manifestPath = "corpus/quant_policy/manifest.v1.json";
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));

check.expectEqual(manifest.schema, "deepbom.quant_policy_boundary_corpus.v1", "schema");
check.expectEqual(manifest.public_artifact_count, 50, "public target count");
check.expectEqual(manifest.case_study_anchor_count, manifest.case_study_anchors.length, "anchor count");
check.expectEqual(manifest.subcohorts.length, 4, "subcohort count");

const artifacts = [];
for (const cohort of manifest.subcohorts) {
  let rows;
  if (cohort.kind === "public_manifest") {
    const source = await readCorpusManifest(cohort.manifest);
    rows = source.models.map((row) => ({ id: row.id, sha256: row.sha256 }));
  } else if (cohort.kind === "curated_micro_manifest") {
    const source = await readCuratedMicroCorpus(cohort.manifest);
    rows = source.artifacts.map((row) => ({ id: row.id, sha256: row.sha256 }));
  } else if (cohort.kind === "google_legacy_manifest") {
    const source = await readGoogleLegacyManifest(cohort.manifest);
    rows = source.artifacts
      .filter((row) => row.cohort === "legacy_quantized")
      .map((row) => ({ id: row.id, sha256: row.member.sha256 }));
  } else if (cohort.kind === "huggingface_pinned_manifest") {
    const source = await readGoogleModernComparators(cohort.manifest);
    const resolved = await loadResolvedGoogleModernComparators(source);
    rows = resolved.map(({ artifact }) => ({ id: artifact.id, sha256: artifact.sha256 }));
  } else {
    throw new Error(`Unsupported quant-policy subcohort kind: ${cohort.kind}.`);
  }
  check.expectEqual(rows.length, cohort.expected_count, `${cohort.id} count`);
  for (const row of rows) artifacts.push({ ...row, cohort_id: cohort.id });
}

check.expectEqual(artifacts.length, manifest.public_artifact_count, "resolved public count");
check.expectEqual(
  new Set(artifacts.map((row) => `${row.cohort_id}/${row.id}`)).size,
  artifacts.length,
  "qualified artifact ids",
);
check.expectEqual(
  new Set(artifacts.map((row) => row.sha256)).size,
  artifacts.length,
  "public artifact SHA-256 identities",
);
for (const row of artifacts) {
  check.expect(/^[0-9a-f]{64}$/.test(row.sha256), `${row.cohort_id}/${row.id} SHA-256`);
}
for (const anchor of manifest.case_study_anchors) {
  check.expect(/^[0-9a-f]{64}$/.test(anchor.sha256), `${anchor.id} SHA-256`);
  check.expect(anchor.redistributable === false, `${anchor.id} redistribution boundary`);
  check.expect(/^[A-Z][A-Z0-9_]+$/.test(anchor.local_path_env), `${anchor.id} local path env`);
}

check.expect(
  manifest.interpretation_boundary.includes("does not observe converter pass execution"),
  "converter-pass interpretation boundary",
);

const interfaceA = summarizeInterfaceContracts(interfaceFixture({
  inputScale: 0.0078125,
  inputZeroPoint: 128,
  outputScale: 0.00390625,
  outputZeroPoint: 0,
}));
const interfaceB = summarizeInterfaceContracts(interfaceFixture({
  inputScale: 0.012566016986966133,
  inputZeroPoint: 131,
  outputScale: 0.09889253973960876,
  outputZeroPoint: 58,
}));
check.expectEqual(interfaceA.parameter_count, 2, "interface parameter count");
check.expectEqual(interfaceA.quantized_parameter_count, 2, "complete interface contracts");
check.expectEqual(
  interfaceA.schema,
  "deepbom.interface_quantization_contracts.v1.3",
  "interface contract schema",
);
check.expectEqual(
  interfaceA.boundary_contract.status,
  "fully_affine_quantized",
  "fully quantized boundary status",
);
check.expectEqual(
  interfaceA.parameters[0].quantization.scalar_real_code_domain.real_min,
  -1,
  "input real-domain minimum",
);
check.expectEqual(
  interfaceA.parameters[0].quantization.scalar_real_code_domain.real_max,
  0.9921875,
  "input real-domain maximum",
);
check.expectEqual(
  summarizeInterfaceContracts(interfaceFixture({
    inputScale: 0.0078125,
    inputZeroPoint: 128,
    outputScale: 0.00390625,
    outputZeroPoint: 0,
  })).ledger_sha256,
  interfaceA.ledger_sha256,
  "interface ledger determinism",
);
const interfaceCorpus = buildInterfaceContractCorpusSummary([
  {
    qualified_id: "fixture/a",
    artifact_sha256: "a".repeat(64),
    interface_contracts: interfaceA,
  },
  {
    qualified_id: "fixture/b",
    artifact_sha256: "b".repeat(64),
    interface_contracts: interfaceB,
  },
]);
check.expectEqual(
  interfaceCorpus.dtype_shape_signatures_with_multiple_affine_contracts,
  2,
  "same dtype/shape signatures with distinct affine contracts",
);
check.expect(
  interfaceCorpus.dtype_shape_ambiguity_groups.every(
    (row) => row.artifact_count === 2 && row.distinct_affine_contract_count === 2,
  ),
  "ambiguity group artifact and contract counts",
);
check.expectEqual(
  interfaceCorpus.artifacts_with_fully_quantized_interface,
  2,
  "fully quantized artifact denominator",
);
check.expectEqual(
  interfaceCorpus.artifacts_with_any_complete_quantized_parameter,
  2,
  "artifacts with any complete affine parameter",
);
const unquantizedInterface = summarizeInterfaceContracts({
  inputs: [floatTensor(0, "input")],
  outputs: [floatTensor(1, "output")],
});
const mixedInterface = summarizeInterfaceContracts({
  inputs: [quantTensor(0, "UINT8", [1, 224, 224, 3], [1 / 255], [0], 0)],
  outputs: [floatTensor(1, "output")],
});
const invalidInterface = summarizeInterfaceContracts({
  inputs: [{
    ...quantTensor(0, "INT8", [-1, 4], [0.25, 0.5], [0, 0], 0),
    quantized_dimension: 0,
  }],
  outputs: [],
});
check.expectEqual(
  invalidInterface.boundary_contract.status,
  "invalid_or_incomplete",
  "A dynamic per-axis cardinality must not be certified complete",
);
const undeclaredInterface = summarizeInterfaceContracts({ inputs: [], outputs: [] });
const boundaryCorpus = buildInterfaceContractCorpusSummary([
  corpusBoundaryFixture("full", interfaceA, "full_integer", {}),
  corpusBoundaryFixture("mixed", mixedInterface, "integer_internal_float_io", {
    QUANTIZE: 1,
    DEQUANTIZE: 1,
  }),
  corpusBoundaryFixture("float", unquantizedInterface, "integer_internal_float_io", {
    QUANTIZE: 1,
    DEQUANTIZE: 1,
  }),
  corpusBoundaryFixture("invalid", invalidInterface, "integer_internal_float_io", {}),
  corpusBoundaryFixture("undeclared", undeclaredInterface, "integer_internal_float_io", {}),
]);
check.expectDeepEqual(
  boundaryCorpus.interface_boundary_status_counts,
  {
    fully_affine_quantized: 1,
    fully_unquantized: 1,
    invalid_or_incomplete: 1,
    mixed_quantized_unquantized: 1,
    not_declared: 1,
  },
  "artifact boundary status counts",
);
check.expectEqual(
  boundaryCorpus.internal_integer_artifact_count,
  5,
  "internal integer artifact denominator",
);
check.expectEqual(
  boundaryCorpus.internal_integer_with_any_unquantized_io,
  2,
  "internal integer artifacts with an unquantized boundary",
);
check.expectEqual(
  boundaryCorpus.internal_integer_with_invalid_or_incomplete_interface,
  1,
  "invalid interface contracts must have a separate denominator",
);
check.expectEqual(
  boundaryCorpus.internal_integer_with_no_declared_interface,
  1,
  "undeclared interface contracts must have a separate denominator",
);
check.expectEqual(
  boundaryCorpus.internal_integer_non_fully_quantized_interface_with_quantize_op,
  2,
  "non-fully-quantized integer artifacts with explicit QUANTIZE",
);
check.expect(
  boundaryCorpus.interpretation_boundary.includes("RGB/BGR"),
  "boundary contract does not overclaim preprocessing coverage",
);
const validPerAxis = summarizeInterfaceContracts({
  inputs: [quantTensor(0, "INT8", [1, 2], [0.25, 0.5], [0, 0], 1)],
  outputs: [],
});
const invalidPerAxis = summarizeInterfaceContracts({
  inputs: [quantTensor(0, "INT8", [1, 2], [0.25, 0.5, 0.75], [0, 0, 0], 1)],
  outputs: [],
});
check.expectEqual(validPerAxis.per_axis_parameter_count, 1, "valid per-axis interface");
check.expectEqual(validPerAxis.quantized_parameter_count, 1, "valid per-axis cardinality");
check.expectEqual(
  invalidPerAxis.invalid_or_incomplete_parameter_count,
  1,
  "invalid per-axis cardinality",
);
check.done();

function interfaceFixture({
  inputScale,
  inputZeroPoint,
  outputScale,
  outputZeroPoint,
}) {
  return {
    inputs: [
      quantTensor(
        0,
        "UINT8",
        [1, 224, 224, 3],
        [inputScale],
        [inputZeroPoint],
        0,
      ),
    ],
    outputs: [
      quantTensor(
        1,
        "UINT8",
        [1, 1001],
        [outputScale],
        [outputZeroPoint],
        0,
      ),
    ],
    metadata_presence: {
      preprocessing_contract_status: "absent_no_model_metadata",
    },
    input_contracts: [{
      tensor_index: 0,
      source_data_to_tensor_preprocessing_status: "not_embedded_in_artifact",
    }],
  };
}

function quantTensor(index, dtype, shape, scales, zeroPoints, quantizedDimension) {
  return {
    index,
    name: `tensor_${index}`,
    dtype,
    shape,
    quant_scales: scales.length,
    quant_zero_points: zeroPoints.length,
    scale_sample: scales,
    zero_point_sample: zeroPoints,
    quantized_dimension: quantizedDimension,
  };
}

function floatTensor(index, name) {
  return {
    index,
    name,
    dtype: "FLOAT32",
    shape: [1, 10],
    quant_scales: 0,
    quant_zero_points: 0,
    scale_sample: [],
    zero_point_sample: [],
    quantized_dimension: 0,
  };
}

function corpusBoundaryFixture(id, interfaceContracts, quantizationClassification, histogram) {
  return {
    qualified_id: `fixture/${id}`,
    subcohort_id: "fixture",
    artifact_sha256: id.padEnd(64, id[0]),
    quantization_classification: quantizationClassification,
    observed_signals: { operator_histogram: histogram },
    interface_contracts: interfaceContracts,
  };
}
