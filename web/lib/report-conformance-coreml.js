export function registerCoreMlSerializedConformance({ staticAnalysis, ops, reportText, check }) {
  const coreml = staticAnalysis.coreml || {};
  const status = staticAnalysis.quantization_status || {};
  const weights = ops.flatMap((op) => Array.isArray(op.coreml_weights) ? op.coreml_weights : []);
  const storedBytes = weights.reduce((sum, weight) => sum + Number(weight.byte_length || 0), 0);
  const preprocessing = Array.isArray(coreml.neural_network?.preprocessing) ? coreml.neural_network.preprocessing : [];
  const inputNames = new Set((staticAnalysis.inputs || []).map((input) => input.name));
  const exactSum = (values) => values.reduce((sum, value) => sum + BigInt(value), 0n);
  const safeMirror = (value) => value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null;
  check("CF-COREML-GRAPH-001", (staticAnalysis.operator_count == null ? ops.length === 0 : Number(staticAnalysis.operator_count) === ops.length)
    && ops.every((op, index) => op.index === index && Array.isArray(op.inputs) && Array.isArray(op.outputs)),
  "Core ML serialized operation count or named-value graph ownership is inconsistent.", ["/evidence/static_analysis/ops"]);
  const tensors = Array.isArray(staticAnalysis.tensors) ? staticAnalysis.tensors : [];
  check("CF-COREML-SHAPE-001", !coreml.neural_network || ops.every((op) => typeof op.shape_status === "string"
    && Array.isArray(op.output_shapes) && op.output_shapes.length === op.outputs.length
    && op.output_shapes.every((shape, index) => JSON.stringify(shape) === JSON.stringify(tensors[op.outputs[index]]?.shape || []))
    && (op.shape_status.startsWith("derived_") ? op.shape_reason == null : typeof op.shape_reason === "string"))
    && reportText.includes("Output shape(s) / contract"),
  "Core ML legacy output shapes, per-operation shape status, tensor rows, or report projection are inconsistent.", ["/evidence/static_analysis/ops", "/evidence/static_analysis/tensors", "/engineering_report.md"]);
  check("CF-COREML-WEIGHT-001", !coreml.neural_network || (storedBytes === Number(status.weight_parameter_bytes)
    && Number(status.scanned_layer_count) === Number(status.layer_count)
    && Number(status.layer_count) === ops.length),
  "Core ML WeightParams bytes or layer scan denominator does not conserve the decoded layer inventory.", ["/evidence/static_analysis/quantization_status", "/evidence/static_analysis/ops"]);
  const quantizedWeights = weights.filter((weight) => ["raw_quantized", "int8_dynamic"].includes(weight.storage));
  const perAxisWeights = quantizedWeights.filter((weight) => Number(weight.quantization?.scale_count || 0) > 1);
  check("CF-COREML-QUANT-001", !coreml.neural_network || (Number(status.quantized_weight_parameter_count || 0) === quantizedWeights.length
    && Number(status.per_axis_quantized_weight_parameter_count || 0) === perAxisWeights.length
    && quantizedWeights.every((weight) => weight.quantization && Number.isInteger(weight.quantization.number_of_bits)
      && weight.quantization.number_of_bits >= 1 && weight.quantization.number_of_bits <= 8)
    && reportText.includes("Legacy quantization granularity")),
  "Core ML quantized WeightParams count, per-axis scale-vector count, bit contract, or report projection is inconsistent.", ["/evidence/static_analysis/quantization_status", "/evidence/static_analysis/ops", "/engineering_report.md"]);
  const weightIntegrity = staticAnalysis.weight_integrity || null;
  check("CF-COREML-MLPROGRAM-WEIGHT-001", coreml.model_type !== "mlProgram" || !weightIntegrity || (Number(weightIntegrity.assessed_parameter_count || 0) <= Number(weightIntegrity.parameter_count || 0)
    && (weightIntegrity.payload_bytes == null || Number(weightIntegrity.assessed_payload_bytes) <= Number(weightIntegrity.payload_bytes))
    && (weightIntegrity.status !== "assessed" || Number(weightIntegrity.assessed_parameter_count) === Number(weightIntegrity.parameter_count))),
  "Core ML ML Program constant/blob assessed counts or bytes exceed, or fail to cover, the declared parameter inventory.", ["/evidence/static_analysis/weight_integrity", "/evidence/static_analysis/coreml_blob_integrity"]);
  const numericalParameters = Array.isArray(weightIntegrity?.parameters) ? weightIntegrity.parameters : [];
  const numericalParameterBytes = numericalParameters.reduce((sum, row) => sum + Number(row.byte_length || 0), 0);
  const assessedNumericalParameters = numericalParameters.filter((row) => row.numerical_integrity?.status?.startsWith("assessed"));
  const assessedNumericalParameterBytes = assessedNumericalParameters.reduce((sum, row) => sum + Number(row.byte_length || 0), 0);
  const completeNumericalCoverage = weightIntegrity?.status === "assessed";
  const classicalOrPipeline = Boolean(coreml.classical_model || coreml.pipeline);
  check("CF-COREML-CLASSICAL-001", !classicalOrPipeline || (weightIntegrity
    && Number(weightIntegrity.parameter_count) === numericalParameters.length
    && Number(weightIntegrity.assessed_parameter_count) === assessedNumericalParameters.length
    && Number(weightIntegrity.assessed_payload_bytes || 0) === assessedNumericalParameterBytes
    && (weightIntegrity.payload_bytes == null || numericalParameterBytes === Number(weightIntegrity.payload_bytes))
    && (!completeNumericalCoverage || assessedNumericalParameters.length === numericalParameters.length
      && Number(weightIntegrity.payload_bytes) === assessedNumericalParameterBytes)
    && numericalParameters.every((row) => typeof row.numerical_integrity?.status === "string"
      && (row.numerical_integrity.schema !== "deepbom.coreml.classical_numerical_integrity.v1"
        || Number(row.byte_length || 0) === Number(row.value_count || 0) * 8))
    && (!coreml.classical_model || ops.length === 1 && reportText.includes("Classical Model Contract"))
    && (!coreml.pipeline || Array.isArray(coreml.pipeline.model_summaries) && coreml.pipeline.model_summaries.length === coreml.pipeline.models.length
      && reportText.includes("Pipeline Stage Contract"))),
  "Core ML classical/pipeline parameter bytes, graph ownership, or report binding is inconsistent.", ["/evidence/static_analysis/coreml/classical_model", "/evidence/static_analysis/coreml/pipeline", "/evidence/static_analysis/weight_integrity", "/engineering_report.md"]);
  check("CF-COREML-PREPROCESS-001", preprocessing.every((item) => inputNames.has(item.feature_name))
    && preprocessing.every((item) => item.kind === "image_scaler"
      ? Number(item.serialized_field_count) > 0
      : item.kind === "mean_image" && Number(item.value_count) * 4 === Number(item.byte_length)),
  "Core ML serialized preprocessing is not bound to a named external input or has inconsistent payload cardinality.", ["/evidence/static_analysis/coreml/neural_network/preprocessing", "/evidence/static_analysis/inputs"]);
  const flexibleScenarios = staticAnalysis.flexible_input_scenarios || coreml.flexible_input_scenarios;
  if (flexibleScenarios?.scenario_count) {
    const rows = flexibleScenarios.scenarios || [];
    const scenarioCount = Number(flexibleScenarios.scenario_count);
    const retainedCount = Number(flexibleScenarios.retained_scenario_count ?? rows.length);
    const evaluatedCount = Number(flexibleScenarios.evaluated_scenario_count ?? scenarioCount);
    const statusCounts = Array.isArray(flexibleScenarios.scenario_status_counts) ? flexibleScenarios.scenario_status_counts : [];
    const countedStatuses = statusCounts.reduce((sum, row) => sum + Number(row.count || 0), 0);
    const assessedStatuses = statusCounts.find((row) => row.status === "assessed")?.count;
    const validRows = rows.length === retainedCount && retainedCount <= scenarioCount && evaluatedCount === scenarioCount
      && (!statusCounts.length
        ? Number(flexibleScenarios.assessed_scenario_count || 0) === rows.filter((row) => row.status === "assessed").length
        : countedStatuses === scenarioCount && Number(flexibleScenarios.assessed_scenario_count || 0) === Number(assessedStatuses || 0))
      && rows.every((row, index) => row.scenario_index === index && Array.isArray(row.input_shapes)
        && row.input_shapes.every((item) => inputNames.has(item.name) && Array.isArray(item.shape)
          && item.shape.every((value) => Number.isSafeInteger(value) && value > 0))
        && (row.status !== "assessed" || /^(?:0|[1-9]\d*)$/.test(String(row.total_macs_decimal))
          && [row.input_logical_payload_bytes, row.output_logical_payload_bytes, row.peak_live_logical_payload_bytes]
            .every((value) => Number.isSafeInteger(value) && value >= 0)));
    check("CF-COREML-FLEX-SCENARIO-001", flexibleScenarios.schema === "deepbom.coreml.flexible_input_scenarios.v1"
      && validRows && reportText.includes("Core ML Flexible Input Scenarios")
      && reportText.includes(`${evaluatedCount.toLocaleString("en-US")}/${scenarioCount.toLocaleString("en-US")}`)
      && reportText.includes("not a proof that an interior point cannot have a larger cost or payload"),
    "Core ML flexible-input scenario count, exact arithmetic, endpoint boundary, or report projection is inconsistent.", ["/evidence/static_analysis/flexible_input_scenarios", "/engineering_report.md"]);
  }
  const macAssessment = staticAnalysis.mac_assessment || {};
  if (macAssessment.assessed_macs_decimal != null) {
    const assessedMacs = exactSum(ops.filter((op) => op.macs_decimal != null && op.macs_status !== "derived_non_mac_operation").map((op) => op.macs_decimal));
    const complete = String(macAssessment.status || "").startsWith("assessed_all_decoded_compute_ops");
    check("CF-COREML-MAC-001", String(macAssessment.assessed_macs_decimal) === assessedMacs.toString()
      && macAssessment.assessed_macs === safeMirror(assessedMacs)
      && macAssessment.complete_macs_decimal === (complete ? assessedMacs.toString() : null)
      && staticAnalysis.total_macs === (complete ? safeMirror(assessedMacs) : null)
      && macAssessment.safe_number_mirror_status === (safeMirror(assessedMacs) == null ? "exact_decimal_only" : "safe_integer_mirror_available"),
    "Core ML assessed MAC subtotal, complete total, and safe-number mirror do not conserve the decoded operation ledger.", ["/evidence/static_analysis/mac_assessment", "/evidence/static_analysis/ops"]);
  }
  const milScopes = coreml.mil_scope_intrinsic_cost;
  if (coreml.model_type === "mlProgram") {
    const scopeRows = milScopes?.scope_rows || [];
    const grouped = new Map();
    for (const op of ops) {
      const rows = grouped.get(op.mil_scope) || [];
      rows.push(op);
      grouped.set(op.mil_scope, rows);
    }
    const validScopes = milScopes?.schema === "deepbom.coreml.mil_scope_intrinsic_cost.v1"
      && milScopes.evidence_class === "SOURCE_PINNED_AND_DERIVED" && scopeRows.length === grouped.size
      && Number(milScopes.scope_count) === scopeRows.length
      && Number(milScopes.nested_scope_count) === scopeRows.filter((row) => row.scope_class === "nested_block").length
      && scopeRows.every((row) => {
        const scoped = grouped.get(row.scope) || [];
        const compute = scoped.filter((op) => op.macs_status !== "derived_non_mac_operation");
        const assessed = compute.filter((op) => op.macs_decimal != null);
        const macs = exactSum(assessed.map((op) => op.macs_decimal));
        const payload = scoped.filter((op) => op.estimated_bytes != null);
        const bytes = exactSum(payload.map((op) => op.estimated_bytes));
        return row.operator_count === scoped.length && row.mac_compute_operator_count === compute.length
          && row.assessed_mac_operator_count === assessed.length && row.residual_mac_operator_count === compute.length - assessed.length
          && row.assessed_nominal_macs_decimal === macs.toString() && row.assessed_nominal_macs === safeMirror(macs)
          && row.complete_nominal_macs_decimal === (assessed.length === compute.length ? macs.toString() : null)
          && row.assessed_output_payload_operator_count === payload.length && row.residual_output_payload_operator_count === scoped.length - payload.length
          && row.assessed_output_payload_bytes_decimal === bytes.toString() && row.assessed_output_payload_bytes === safeMirror(bytes)
          && row.complete_output_payload_bytes_decimal === (payload.length === scoped.length ? bytes.toString() : null)
          && ["assessed", "partial", "not_assessed"].includes(row.scope_local_liveness?.status);
      });
    check("CF-COREML-MIL-SCOPE-001", validScopes && reportText.includes("MIL One-Invocation Scope Ledger")
      && reportText.includes("not multiplied or summed into a model total"),
    "Core ML MIL scope-local MAC, payload, liveness, or report ledgers do not conserve decoded SSA scopes.", ["/evidence/static_analysis/coreml/mil_scope_intrinsic_cost", "/engineering_report.md"]);

    const compression = coreml.mil_compression_contract;
    const compressionOps = ops.filter((op) => op.compression_contract);
    if (compression?.transform_count || compressionOps.length) {
      const rows = Array.isArray(compression?.transforms) ? compression.transforms : [];
      const positiveShape = (shape) => Array.isArray(shape) && shape.length > 0
        && shape.every((value) => Number.isSafeInteger(value) && value > 0);
      const product = (shape) => positiveShape(shape) ? shape.reduce((value, dimension) => value * dimension, 1) : null;
      const validRows = rows.length === compressionOps.length && rows.every((row) => {
        const op = ops[row.op_index];
        if (!op || op.mil_operation_type !== row.op_type || op.compression_contract?.transform !== row.transform
          || !/^[0-9a-f]{64}$/i.test(String(row.source_sha256 || ""))) return false;
        if (["affine_constant_dequantization", "blockwise_affine", "blockwise_lut_palettization", "packed_index_lut_palettization", "unstructured_sparse_bitmask", "packed_unstructured_sparse_bitmask"].includes(row.representation)
          && (!row.payload_integrity || !row.reconstruction?.status)) return false;
        if (row.representation === "affine_constant_dequantization") {
          const outputElements = product(row.output_shape);
          return outputElements === row.logical_output_elements && outputElements === row.quantized_data_elements
            && ["per_tensor", "per_axis"].includes(row.granularity)
            && Number.isSafeInteger(row.normalized_axis) && row.normalized_axis >= 0
            && row.normalized_axis < row.output_shape.length && row.axis_extent === row.output_shape[row.normalized_axis]
            && Number.isSafeInteger(row.serialized_axis)
            && row.normalized_axis === (row.serialized_axis < 0 ? row.serialized_axis + row.output_shape.length : row.serialized_axis)
            && Number.isSafeInteger(row.scale_elements) && Number.isSafeInteger(row.zero_point_elements)
            && [1, row.axis_extent].includes(row.scale_elements) && [1, row.axis_extent].includes(row.zero_point_elements)
            && row.granularity === (row.scale_elements > 1 || row.zero_point_elements > 1 ? "per_axis" : "per_tensor");
        }
        if (row.representation === "blockwise_affine") {
          const outputElements = product(row.output_shape);
          const blockElements = product(row.block_shape);
          return outputElements === row.logical_output_elements && blockElements != null
            && Number.isSafeInteger(row.scale_elements) && row.scale_elements > 0
            && outputElements === row.scale_elements * blockElements;
        }
        if (row.representation === "blockwise_lut_palettization") {
          const indexElements = product(row.index_shape);
          const outputElements = product(row.output_shape);
          return indexElements === row.logical_index_elements && outputElements === row.logical_output_elements
            && Number.isSafeInteger(row.index_bits) && row.palette_count === 2 ** row.index_bits
            && Number.isSafeInteger(row.vector_size) && row.vector_size > 0
            && outputElements === indexElements * row.vector_size
            && positiveShape(row.group_grid_shape) && row.group_grid_shape.length === row.index_shape.length
            && row.index_shape.every((value, index) => value % row.group_grid_shape[index] === 0)
            && (!row.lut_usage?.status?.startsWith("assessed")
              || row.lut_usage.index_count === row.logical_index_elements
                && row.lut_usage.palette_entries_total === row.palette_count
                && row.lut_usage.palette_entries_used <= row.palette_count);
        }
        if (row.representation === "packed_index_lut_palettization") {
          const outputElements = product(row.output_shape);
          return outputElements === row.logical_output_elements
            && Number.isSafeInteger(row.serialized_index_bytes) && row.serialized_index_bytes > 0
            && Number.isSafeInteger(row.index_bits) && row.palette_count === 2 ** row.index_bits;
        }
        if (["unstructured_sparse_bitmask", "packed_unstructured_sparse_bitmask"].includes(row.representation)) {
          const outputElements = product(row.output_shape);
          return outputElements === row.logical_output_elements && Number.isSafeInteger(row.stored_nonzero_elements)
            && row.stored_nonzero_elements >= 0 && row.stored_nonzero_elements <= outputElements
            && ["not_decoded_at_mil_contract_layer", "assessed_exact_immediate_payload", "assessed_exact_bound_blob_payload"].includes(row.mask_population_status)
            && (row.mask_population == null || row.mask_population === row.stored_nonzero_elements);
        }
        return row.status === "not_assessed_source_semantics_not_implemented";
      });
      check("CF-COREML-MIL-COMPRESSION-001", compression?.schema === "deepbom.coreml.mil_compression_ledger.v1"
        && Number(compression.transform_count) === rows.length
        && Number(compression.exact_contract_count) === rows.filter((row) => row.status.startsWith("assessed_exact")).length
        && Number(compression.partial_contract_count) === rows.filter((row) => !row.status.startsWith("assessed_exact")).length
        && validRows && reportText.includes("Core ML Serialized Compression Contracts")
        && reportText.includes("runtime materialization"),
      "Core ML serialized blockwise/LUT/sparse compression contracts, source binding, counts, or report boundary are inconsistent.", ["/evidence/static_analysis/coreml/mil_compression_contract", "/evidence/static_analysis/ops", "/engineering_report.md"]);
    }
  }
  const source = coreml.source_basis || {};
  const classicalSources = Array.isArray(source.classical_model_sources) ? source.classical_model_sources : [];
  check("CF-COREML-SOURCE-001", /^[0-9a-f]{40}$/i.test(String(source.source_commit || ""))
    && /^[0-9a-f]{64}$/i.test(String(source.model_proto_sha256 || ""))
    && (!coreml.neural_network || /^[0-9a-f]{64}$/i.test(String(source.neural_network_proto_sha256 || "")))
    && (coreml.model_type !== "mlProgram" || /^[0-9a-f]{64}$/i.test(String(source.mil_proto_sha256 || "")))
    && (coreml.model_type !== "mlProgram" || /^[0-9a-f]{64}$/i.test(String(source.mil_compression_ios18_definition_sha256 || ""))
      && /^[0-9a-f]{64}$/i.test(String(source.mil_constexpr_ios16_definition_sha256 || "")))
    && (!classicalOrPipeline || classicalSources.length === 10
      && classicalSources.every((row) => typeof row.path === "string" && /^[0-9a-f]{64}$/i.test(String(row.sha256 || ""))))
    && reportText.includes(String(source.source_commit || "")),
  "Core ML model/graph interpretation is not source-commit/content-digest bound.", ["/evidence/static_analysis/coreml/source_basis", "/engineering_report.md"]);
  const floor = coreml.deployment_floor || {};
  check("CF-COREML-FLOOR-001", floor.schema === "deepbom.coreml.deployment_floor.v1"
    && floor.declared_specification_version === coreml.specification_version
    && floor.status !== "invalid_declared_version_below_observed_feature_floor"
    && (floor.status !== "assessed" || floor.declared_load_floor != null)
    && reportText.includes("Declared Core ML load floor"),
  "Core ML specification-to-OS floor is missing, contradictory, or absent from the engineering report.", ["/evidence/static_analysis/coreml/deployment_floor", "/engineering_report.md"]);
}
