import { formatBytes, formatNumber, maxBy, padOp } from "./format.js";
import { buildAnalyzerMetadata, ANALYZER_METADATA } from "./report-metadata.js";
import { collectArtifactIntegrity } from "./report-integrity.js";

export function buildSecurityPostureEvidence({
  analysis,
  model = {},
  fileSizeBytes = 0,
  runtimeWarnings = null,
  runtimeBenchmarkResults = [],
  runtimeBasinResult = null,
  preprocessingConsequenceResult = null,
} = {}) {
  const artifactIntegrity = collectArtifactIntegrity(analysis, model, fileSizeBytes);
  const onnx = String(analysis?.format || "tflite").toLowerCase() === "onnx";
  const customOps = artifactIntegrity.custom_like_ops;
  const dynamicInputs = (analysis?.inputs || []).filter((tensor) =>
    [tensor.shape, tensor.shape_signature].some((shape) => Array.isArray(shape) && shape.some((dim) => dim < 0)),
  );
  const topBytes = maxBy((analysis?.ops || []).filter((op) => op.estimated_bytes != null), (op) => Number(op.estimated_bytes));
  const topMac = maxBy((analysis?.ops || []).filter((op) => op.macs_status !== "not_assessed"), (op) => Number(op.macs || 0));
  const benchmarkSucceeded = (runtimeBenchmarkResults || []).some((item) => item?.ok === true);
  const benchmarkAttempted = (runtimeBenchmarkResults || []).length > 0;
  const runtimeExecuted = Boolean(runtimeBasinResult) || Boolean(preprocessingConsequenceResult) || benchmarkSucceeded;
  const observedRuntimeWarnings = runtimeExecuted
    ? (runtimeWarnings || collectRuntimeWarnings(runtimeBasinResult, runtimeBenchmarkResults))
    : null;
  const domainAnalysis = analysis?.onnx_domain_analysis || null;
  const externalDomains = domainAnalysis?.external_custom_domains || [];
  const customOnnxOps = (domainAnalysis?.nodes || [])
    .filter((node) => node.resolution_class === "external_custom_registry")
    .map((node) => ({ domain: node.domain, op_type: node.op_name, scope: node.scope, op_index: node.top_level_op_index }));
  return {
    schema: ANALYZER_METADATA.schemas.securityPosture,
    analyzer_metadata: buildAnalyzerMetadata(analysis),
    model,
    privacy: {
      raw_model_upload: "not performed by this app workflow",
      generated_tensor_upload: "not performed by this app workflow",
      output_upload: "not performed by this app workflow",
      raw_model_bytes_included: "false",
      boundary: "Application-observed local processing record; not a whole-system no-egress proof.",
    },
    integrity: {
      sha256: artifactIntegrity.sha256,
      file_size_bytes: artifactIntegrity.file_size_bytes,
      format: artifactIntegrity.format,
      schema_or_opset: artifactIntegrity.schema_or_opset,
      onnx_producer: artifactIntegrity.onnx_producer,
      onnx_opsets: artifactIntegrity.onnx_opsets,
      graph_name: artifactIntegrity.graph_name,
      external_domains: artifactIntegrity.external_domains,
      external_data_tensor_count: artifactIntegrity.external_data_tensor_count,
      external_data_verified_tensor_count: artifactIntegrity.external_data_verified_tensor_count,
      external_data_verification_failed_count: artifactIntegrity.external_data_verification_failed_count,
      external_data_incomplete_tensor_count: artifactIntegrity.external_data_incomplete_tensor_count,
      external_data_integrity_status: artifactIntegrity.external_data_integrity_status,
      external_data_summary: artifactIntegrity.external_data_summary,
      tflite_constant_tensor_count: artifactIntegrity.tflite_constant_tensor_count,
      tflite_unique_constant_regions: artifactIntegrity.tflite_unique_constant_regions,
      tflite_constant_buffer_bytes: artifactIntegrity.tflite_constant_buffer_bytes,
      tflite_largest_constant_buffer_bytes: artifactIntegrity.tflite_largest_constant_buffer_bytes,
      tflite_duplicate_constant_regions: artifactIntegrity.tflite_duplicate_constant_regions,
      tflite_out_of_bounds_tensor_buffers: artifactIntegrity.tflite_out_of_bounds_tensor_buffers,
      tflite_buffer_summary: artifactIntegrity.tflite_buffer_summary,
      report_binding: "report references artifact filename, hash, format, and target profile",
      summary: artifactIntegrity.summary,
    },
    execution_surface: {
      format: onnx ? "onnx" : "tflite",
      ...(onnx ? {
        domain_analysis: domainAnalysis,
        external_custom_domains: externalDomains,
        custom_op_types: customOnnxOps,
        local_functions: domainAnalysis
          ? { status: "assessed", value: domainAnalysis.functions || [], reason: "FunctionProto definitions, bodies, overloads, nested graphs, dependencies, duplicate IDs, and recursive cycles were parsed from the artifact." }
          : { status: "not_assessed", value: null, reason: "ONNX domain analysis was not emitted." },
        ort_contrib_nodes: Number(domainAnalysis?.ort_contrib_node_count || 0),
        external_data_references: Number(analysis?.onnx_external_data_tensor_count || 0),
        unknown_attribute_assessment: { status: "not_assessed", value: null, reason: "Unknown-attribute semantic validation requires an opset-aware ONNX schema registry." },
        summary: externalDomains.length
          ? `${externalDomains.length} external custom registry domain(s) observed; model-local functions and ORT contrib nodes are inventoried separately.`
          : `No external custom registry domain was observed; ${domainAnalysis?.model_local_function_call_count || 0} model-local function call(s) and ${domainAnalysis?.ort_contrib_node_count || 0} ORT contrib node(s) were classified separately.`,
      } : {
        custom_like_ops: customOps,
        summary: customOps.length ? customOps.map((item) => `${item.name}:${item.count}`).join(" / ") : "No custom, Flex, or Select TF op family was detected in the TFLite operator histogram.",
      }),
    },
    resource_exhaustion: {
      dynamic_input_count: dynamicInputs.length,
      top_estimated_bytes_op: topBytes ? `#${padOp(topBytes.index)} ${topBytes.name} ${formatBytes(topBytes.estimated_bytes)}` : "not assessed",
      top_mac_op: topMac ? `#${padOp(topMac.index)} ${topMac.name} ${formatNumber(topMac.macs || 0)} MACs` : "-",
      summary: `${dynamicInputs.length} dynamic input(s); largest assessed op traffic ${topBytes ? formatBytes(topBytes.estimated_bytes) : "not assessed"}; largest MAC op ${topMac ? formatNumber(topMac.macs || 0) : "-"}.`,
    },
    execution_integrity: {
      runtime_execution_status: runtimeExecuted ? "executed" : benchmarkAttempted ? "attempted_failed" : "not_run",
      runtime_warnings: observedRuntimeWarnings,
      conclusion: runtimeExecuted ? (observedRuntimeWarnings.length ? "warnings_observed" : "no_warnings_observed") : "not_assessed",
      summary: runtimeExecuted
        ? (observedRuntimeWarnings.length ? observedRuntimeWarnings.slice(0, 3).join(" / ") : "Runtime execution completed and no browser-local runtime warning was observed in this session.")
        : benchmarkAttempted ? "Runtime execution was attempted but no benchmark backend completed; no latency or warning-absence claim is made." : "Runtime execution was not run; warning absence is not asserted.",
    },
  };
}

export function collectRuntimeWarnings(runtimeBasinResult = null, runtimeBenchmarkResults = []) {
  return [
    ...(runtimeBasinResult?.backend_interpretations || [])
      .filter((item) => item.severity === "warn")
      .map((item) => `${item.backend}: ${item.note}`),
    ...(runtimeBenchmarkResults || [])
      .filter((item) => !item.ok)
      .map((item) => `${item.backend}: ${item.error}`),
  ];
}
