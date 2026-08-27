import { buildOnnxRuntimeShapeBinding } from "../web/lib/onnx-runtime-shape-binding.js";
import { createCheck } from "./check-assert.mjs";

const check = createCheck("ONNX runtime shape binding");
const analysis = {
  format: "onnx",
  model_sha256: "a".repeat(64),
  tensors: [
    { index: 0, name: "input", dtype: "FLOAT32", value_kind: "tensor", shape: [-1, 3], shape_declared: true },
    { index: 1, name: "weight", dtype: "FLOAT32", value_kind: "tensor", shape: [3, 4], shape_declared: true },
    { index: 2, name: "output", dtype: "FLOAT32", value_kind: "tensor", shape: [-1, 4], shape_declared: true },
  ],
  ops: [{
    index: 0,
    name: "MatMul",
    domain: "ai.onnx",
    input_names: ["input", "weight"],
    output_names: ["output"],
    inputs: [0, 1],
    outputs: [2],
    onnx_attributes: [],
    macs: null,
    macs_status: "not_assessed",
    macs_reason: "dynamic shape",
  }],
  dynamic_shape_cost_contract: {
    symbols: [{ symbol_id: "D0", occurrences: [{ tensor_index: 0, axis: 0 }, { tensor_index: 2, axis: 0 }] }],
    total_macs_formula: {
      status: "exact_symbolic_integer_polynomial", symbol_ids: ["D0"],
      terms: [{ coefficient_decimal: "12", factors: [{ symbol_id: "D0", exponent: 1 }] }],
    },
  },
};
const runtime = {
  backend: "wasm",
  input_contracts: [{ input_name: "input", executed_shape: [2, 3] }],
  output_contracts: [{ output_name: "output", executed_shape: [2, 4] }],
};
const bound = buildOnnxRuntimeShapeBinding(analysis, runtime);
check.expectEqual(bound.status, "assessed_runtime_io_binding", "complete model-I/O binding status");
check.expectEqual(bound.bound_symbol_count, 1, "bound symbol count");
check.expectEqual(bound.evaluated_total_macs_decimal, "24", "exact evaluated MAC polynomial");

const conflict = buildOnnxRuntimeShapeBinding(analysis, { ...runtime, output_contracts: [{ output_name: "output", executed_shape: [3, 4] }] });
check.expectEqual(conflict.status, "fail", "repeated dim_param mismatch must fail");
check.expect(conflict.conflicts.some((row) => row.reason === "repeated_symbol_runtime_values_conflict"), "repeated symbol conflict reason");

const staticMismatch = buildOnnxRuntimeShapeBinding(analysis, { ...runtime, input_contracts: [{ input_name: "input", executed_shape: [2, 5] }] });
check.expectEqual(staticMismatch.status, "fail", "static dimension mismatch must fail");
check.expect(staticMismatch.conflicts.some((row) => row.reason === "runtime_dimension_conflicts_with_static_contract"), "static mismatch reason");

const internalRuntime = {
  runtime: { backend: "CPUExecutionProvider", graph_optimization_level: "disabled" },
  source: {
    collected_at: "2026-08-24T00:00:00.000Z",
    adapter: {
      schema: "deepbom.ort_profile_adapter.v2.2",
      runtime_tensor_observations: [{
        op_index: 0,
        op_name: "MatMul",
        runtime_node_index: 0,
        runtime_node_name: "matmul",
        sample_count: 1,
        status: "consistent",
        input_type_shapes: [
          { slot: 0, ort_type: "float", dtype: "FLOAT32", shape: [2, 3] },
          { slot: 1, ort_type: "float", dtype: "FLOAT32", shape: [3, 4] },
        ],
        output_type_shapes: [{ slot: 0, ort_type: "float", dtype: "FLOAT32", shape: [2, 4] }],
        output_size_bytes: 32,
        output_size_bytes_decimal: "32",
        activation_size_bytes: 24,
        activation_size_bytes_decimal: "24",
        parameter_size_bytes: 48,
        parameter_size_bytes_decimal: "48",
        observed_contract_variant_count: 1,
      }],
    },
  },
};
const internal = buildOnnxRuntimeShapeBinding(analysis, internalRuntime);
check.expectEqual(internal.status, "assessed_runtime_internal_binding", "complete internal type-shape binding status");
check.expectEqual(internal.observed_internal_tensor_count, 3, "unique internal tensor contracts");
check.expectEqual(internal.runtime_closed_mac_op_count, 1, "runtime shape should close one previously unresolved MatMul");
check.expectEqual(internal.remaining_unassessed_mac_op_count, 0, "runtime shape should leave no supported MAC residual");
check.expectEqual(internal.runtime_bound_complete_macs_decimal, "24", "runtime-bound exact MatMul MAC total");
check.expectEqual(internal.observed_tensor_payload_bytes_decimal, "104", "unique logical tensor payload should be recomputed from concrete shapes and dtypes");

const outputSizeMismatch = structuredClone(internalRuntime);
outputSizeMismatch.source.adapter.runtime_tensor_observations[0].output_size_bytes = 31;
outputSizeMismatch.source.adapter.runtime_tensor_observations[0].output_size_bytes_decimal = "31";
const outputSizeFailure = buildOnnxRuntimeShapeBinding(analysis, outputSizeMismatch);
check.expectEqual(outputSizeFailure.status, "fail", "ORT output-size/type-shape mismatch must fail closed");
check.expect(outputSizeFailure.conflicts.some((row) => row.reason === "runtime_output_size_conflicts_with_type_shape"), "ORT output-size mismatch reason");

const dtypeMismatch = structuredClone(internalRuntime);
dtypeMismatch.source.adapter.runtime_tensor_observations[0].output_type_shapes[0].dtype = "FLOAT16";
dtypeMismatch.source.adapter.runtime_tensor_observations[0].output_type_shapes[0].ort_type = "float16";
dtypeMismatch.source.adapter.runtime_tensor_observations[0].output_size_bytes = 16;
dtypeMismatch.source.adapter.runtime_tensor_observations[0].output_size_bytes_decimal = "16";
const dtypeFailure = buildOnnxRuntimeShapeBinding(analysis, dtypeMismatch);
check.expectEqual(dtypeFailure.status, "fail", "runtime/static dtype mismatch must fail closed");
check.expect(dtypeFailure.conflicts.some((row) => row.reason === "runtime_dtype_conflicts_with_static_contract"), "runtime/static dtype mismatch reason");

const optimizedRuntime = structuredClone(internalRuntime);
optimizedRuntime.runtime.graph_optimization_level = "all";
const optimized = buildOnnxRuntimeShapeBinding(analysis, optimizedRuntime);
check.expectEqual(optimized.observed_internal_tensor_count, 0, "optimized runtime node slots must not be rebound to original tensor identity");
check.expectEqual(optimized.exclusion_count, 1, "optimized runtime shape exclusion must remain explicit");

check.done("hash-bound runtime I/O/internal shapes, exact MAC closure, and conflict paths fail closed");
