import {
  assessedOpLogicalBytes,
  estimateOpBottleneck,
  modelQuantizationStatus,
  rooflineRows,
  stageSummaryText,
  topMacOps,
} from "./analysis.js";
import { insightCard, signalItem, td } from "./dom.js";
import { formatBytes, formatExactInteger, formatNumber, formatPercent, formatUs, humanizeStageKey, padOp } from "./format.js";
import { deriveTfliteBatchOneProjection } from "./dynamic-shape-cost.js";
import { buildQuantResearchCoverage } from "./quant-research-applicability.js";
import { buildTensorInventory, classifyTensorRoles, TENSOR_ROLE_ORDER } from "./tensor-inventory.js";
import { formatEvidenceScope } from "./format-evidence-scope.js";

function formatProfileScalar(value) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(Number(value || 0));
}

function targetCoreText(target) {
  const minimum = Number(target?.core_count_min || 0);
  const maximum = Number(target?.core_count_max || 0);
  if (minimum && maximum) return minimum === maximum ? `${minimum} cores` : `${minimum}-${maximum} cores`;
  return "Unbound";
}

function targetIsaText(target, hardware) {
  const neon = Boolean(hardware?.advanced_simd)
    || /NEON/i.test(`${target?.architecture || ""} ${target?.xnnpack_kernel_family || ""}`);
  if (target?.sve2) {
    return `SVE2 ${target.simd_width_bits || "unbound"}-bit${neon ? " / NEON 128-bit" : ""}`;
  }
  if (neon) return "NEON 128-bit";
  return target?.xnnpack_kernel_family || "Not bound";
}

function roleBreakdown(inventory, field = "total") {
  const rows = Array.isArray(inventory?.rows) ? inventory.rows : [];
  const labels = { kernel: "Kernel", bias: "Bias", activation: "Activation", container_tensor: "Container tensor", metadata: "Metadata/parameter" };
  return TENSOR_ROLE_ORDER.map((role) => {
    const roleRows = rows.filter((row) => row.role === role && Number(row[field] || 0) > 0);
    if (!roleRows.length) return "";
    const count = roleRows.reduce((total, row) => total + Number(row[field] || 0), 0);
    const dtypes = roleRows.map((row) => `${row.dtype} ${formatNumber(row[field])}`).join(", ");
    return `${labels[role] || role} ${formatNumber(count)} (${dtypes})`;
  }).filter(Boolean).join(" / ") || "None";
}

function perTensorKernelFamilySummary(analysis) {
  const kernels = classifyTensorRoles(analysis)
    .filter(({ role, tensor }) => role === "kernel" && Number(tensor?.quant_scales || 0) === 1);
  if (!kernels.length) return "";
  const families = new Map();
  for (const { index } of kernels) {
    const family = (analysis?.ops || []).find((op) => Number(op?.inputs?.[1]) === Number(index))?.name
      || "unresolved consumer";
    families.set(family, (families.get(family) || 0) + 1);
  }
  if (families.size === 1) {
    const [[family, count]] = families;
    return `All ${formatNumber(count)} per-tensor kernels are ${family}`;
  }
  return `Kernel families: ${[...families.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([family, count]) => `${family} ${formatNumber(count)}`)
    .join(", ")}`;
}

function formatVersionEvidence(analysis, format) {
  if (format === "onnx") {
    const opsets = (analysis.opsets || [])
      .map((entry) => `${entry.domain || "ai.onnx"}:${entry.version || "unknown"}`)
      .join(" / ");
    return {
      value: `IR v${analysis.onnx_ir_version || analysis.version || "unknown"}`,
      detail: `Opset ${opsets || "not declared"}; producer ${analysis.producer || "not declared"}`,
      producerValue: analysis.producer
        ? `${analysis.producer}${analysis.metadata_presence?.producer_version ? ` ${analysis.metadata_presence.producer_version}` : ""}`
        : "Not declared",
      producerDetail: "OBSERVED from ONNX ModelProto producer_name/producer_version fields",
      schemaLabel: "IR / opset evidence",
      schemaDetail: "IR and imported opsets are artifact declarations",
      producerLabel: "Producer evidence",
    };
  }
  if (format === "gguf") {
    const gguf = analysis.gguf || {};
    const metadata = analysis.metadata_presence || {};
    return {
      value: `GGUF v${gguf.version ?? "unknown"}`,
      detail: `${gguf.endianness || "unknown"}-endian; architecture ${gguf.architecture || "not declared"}; ${formatNumber(gguf.metadata_kv_count || 0)} metadata entries; ${formatNumber(gguf.alignment || 0)} B tensor alignment; ${formatNumber(gguf.declared_tensor_byte_length || 0)} tensor payload bytes; ${gguf.payload_coverage_status || "coverage not assessed"}`,
      producerValue: metadata.producer_name || "Not declared",
      producerDetail: `GGUF general.* identity fields; producer version ${metadata.producer_version || "not declared"}; parser source is pinned in the exported evidence envelope`,
      schemaLabel: "Container evidence",
      schemaDetail: "OBSERVED header/directory plus full source-pinned scalar/block payload decoding, per-range SHA-256, numerical integrity, and byte conservation",
      producerLabel: "Metadata identity",
    };
  }
  if (format === "safetensors") {
    const safe = analysis.safetensors || {};
    const hf = safe.hf_architecture_contract || {};
    return {
      value: safe.sharded ? `SafeTensors / ${formatNumber(safe.shard_count || 0)} shards` : "SafeTensors header",
      detail: `${formatNumber(safe.tensor_count || 0)} tensor entries; ${formatNumber(safe.header_byte_length || 0)} header bytes; ${formatNumber(safe.payload_byte_length || 0)} payload bytes; ${safe.payload_coverage_status || "coverage not assessed"}; HF architecture ${hf.model_type ? `${hf.model_type} (${hf.status})` : "not bundle-bound"}`,
      producerValue: "Not declared",
      producerDetail: "SafeTensors does not define a mandatory producer field; no producer identity is inferred from tensor names or repository support files",
      schemaLabel: "Container evidence",
      schemaDetail: safe.sharded
        ? `Index binding ${safe.index_binding_status || "not assessed"}; selected shards and declared tensor names are checked bidirectionally, then every supported payload range is numerically decoded and hashed`
        : "OBSERVED from the SafeTensors header and exact payload ranges; supported dtypes are fully decoded with per-range SHA-256, numerical integrity, and byte conservation",
      producerLabel: "Producer metadata",
    };
  }
  if (format === "coreml") {
    const coreml = analysis.coreml || {};
    const metadata = analysis.metadata_presence || {};
    return {
      value: `Core ML specification v${coreml.specification_version ?? "unknown"}`,
      detail: `Model type ${coreml.model_type || "not decoded"}; ${coreml.is_updatable ? "updatable" : "not marked updatable"}; ${coreml.parser_scope || "payload scope not declared"}`,
      producerValue: metadata.producer_name || "Not embedded",
      producerDetail: metadata.producer_name
        ? "OBSERVED from the Core ML user-defined source metadata field"
        : "No Core ML source-producer metadata was embedded; author and license fields are reported separately and are not treated as converter identity",
      schemaLabel: "Core ML evidence",
      schemaDetail: `OBSERVED from the pinned Core ML protobuf schema: ${coreml.neural_network
        ? "legacy NeuralNetwork layer DAG, WeightParams encodings, shapes, MACs, and interfaces"
        : coreml.model_type === "mlProgram"
          ? "MIL SSA graph, typed values, package blob references, explicit compression/quantization transforms, shapes, MACs, and interfaces"
          : coreml.classical_model
            ? "GLM, SVM, or TreeEnsemble structure, FLOAT64 numerical parameters, source-backed invariants, and exact arithmetic where the external feature width closes it"
            : coreml.pipeline
              ? "named nested-stage interfaces, decoded stage graphs, parameter-byte and arithmetic conservation, and explicit unsupported-stage boundaries"
          : "model identity, description, functions, and interface FeatureTypes"}`,
      producerLabel: "Model provenance",
    };
  }
  if (format === "executorch") {
    const pte = analysis.executorch_container === "pte";
    const evidence = pte ? analysis.executorch_program || {} : analysis.executorch_flat_tensor || {};
    const source = evidence.source || {};
    const planCount = Number(analysis.subgraphs || 0);
    const delegateCount = Number(analysis.executorch_program?.delegates?.length || 0);
    return {
      value: `${pte ? "ET12 Program" : "FT01 FlatTensor"} v${analysis.version ?? "unknown"}`,
      detail: pte
        ? `${formatNumber(planCount)} execution plan(s); ${formatNumber(analysis.operator_count || 0)} serialized instruction(s); ${formatNumber(delegateCount)} delegate declaration(s); operator argument direction and nominal MAC semantics remain unbound without the matching operator registry`
        : `${formatNumber(analysis.tensor_count || 0)} named tensor/blob record(s); exact segment ranges and storage conservation assessed; no execution graph is serialized`,
      producerValue: "Not embedded",
      producerDetail: "ExecuTorch ET12/FT01 does not provide a mandatory converter-producer identity field; no producer is inferred from operator or tensor names",
      schemaLabel: "ExecuTorch evidence",
      schemaDetail: `OBSERVED from ${source.repository || "pytorch/executorch"} ${source.release || "pinned schema"} @ ${String(source.commit || "unbound").slice(0, 12)}; runtime/backend availability and executed placement are not inferred`,
      producerLabel: "Producer metadata",
    };
  }
  const runtime = analysis.runtime_compat || {};
  const metadata = analysis.metadata_presence || {};
  const floor = runtime.effective_min_runtime_version || runtime.derived_min_runtime_version;
  const mapped = Number(runtime.mapped_operator_code_count || 0);
  const total = Number(runtime.builtin_operator_code_count || 0);
  const coverage = total ? `; map coverage ${formatNumber(mapped)}/${formatNumber(total)}` : "";
  const floorStatus = runtime.runtime_floor_status || "coverage_not_emitted";
  const converterVersion = String(metadata.converter_tensorflow_version || "");
  const conversionStatus = metadata.conversion_metadata_status || "not_present";
  return {
    value: `FlatBuffer schema v${analysis.version || "unknown"}`,
    detail: `${floor ? `Observed op-version necessary floor >=${floor}` : "Runtime floor not derived"}${coverage}; ${floorStatus === "partial_unmapped_builtin_op_versions" ? "execution compatibility not guaranteed; " : ""}converter version ${converterVersion || "not embedded/determinable"}`,
    producerValue: converterVersion ? `TensorFlow ${converterVersion}` : "Not embedded",
    producerDetail: converterVersion
      ? `OBSERVED from CONVERSION_METADATA; API ${metadata.converter_api_version ?? "not declared"}; source model ${metadata.converter_model_type || "not declared"}`
      : `CONVERSION_METADATA ${conversionStatus}; schema/runtime floor cannot identify the converter build`,
    schemaLabel: "Artifact schema evidence",
    schemaDetail: "Schema version is not the TensorFlow/LiteRT converter version",
    producerLabel: "Converter evidence",
  };
}

function int8SpeedupCard(analysis) {
  const computeOps = (analysis?.ops || []).filter((op) =>
    ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"].includes(op.name),
  );
  const computeMacs = computeOps.reduce((sum, op) => sum + Number(op.macs || 0), 0);
  const quantizedMacs = computeOps
    .filter((op) => op.quantized_compute_path)
    .reduce((sum, op) => sum + Number(op.macs || 0), 0);
  const int8Ratio = computeMacs > 0 ? quantizedMacs / computeMacs : null;
  const estimate = Number(analysis?.estimated_int8_speedup || 1);
  const target = analysis?.target_profile?.label || "selected target";
  const detail = int8Ratio == null
    ? "Conv/FC MAC coverage is unavailable; no compute-kernel opportunity should be inferred."
    : int8Ratio < 0.01
      ? `Current path: ${formatPercent(int8Ratio)} of Conv/FC MACs use INT8. ${estimate.toFixed(2)}x is a full-conversion compute-kernel ceiling for ${target}; runtime share is excluded.`
      : `${formatPercent(int8Ratio)} of Conv/FC MACs use INT8. ${estimate.toFixed(2)}x is a MAC-weighted compute-kernel ceiling for ${target}; runtime share is excluded.`;
  return insightCard(
    int8Ratio != null && int8Ratio < 0.01 ? "INT8 Opportunity" : "INT8 Compute Ceiling",
    `~${estimate.toFixed(2)}x`,
    detail,
    estimate >= 2.5 ? "good" : "neutral",
    "quant",
    null,
    "HEURISTIC COMPUTE-KERNEL CEILING",
  );
}

function prioritizeInsightCards(cards) {
  const labelWeight = new Map([
    ["Quant Risk", 190],
    ["Dynamic Shape Cost", 185],
    ["Cache Row Watchlist", 175],
    ["Fallback Traffic", 165],
    ["XNNPACK Segments", 185],
    ["Weight Packing", 180],
    ["INT8 Opportunity", 150],
    ["INT8 Compute Ceiling", 145],
    ["Channel Alignment", 190],
    ["Input Shape", 135],
  ]);
  const toneWeight = { risk: 400, warn: 300, neutral: 100, good: 40 };
  const ranked = cards.map((card, index) => ({
    card,
    index,
    score: Number(toneWeight[card.dataset.cardTone] || 0)
      + Number(labelWeight.get(card.dataset.cardLabel) || 0),
  })).sort((left, right) => right.score - left.score || left.index - right.index);
  ranked.forEach(({ card }, index) => {
    card.classList.toggle("insight-card-primary", index < 3);
    card.classList.toggle("insight-card-secondary", index >= 3);
    card.dataset.importanceRank = String(index + 1);
  });
  return ranked.map(({ card }) => card);
}

export function summaryMetricCards(analysis) {
  const format = String(analysis?.format || "tflite").toLowerCase();
  const onnx = format === "onnx";
  const graphFormat = ["tflite", "onnx"].includes(format);
  const weightContainer = ["gguf", "safetensors"].includes(format);
  const ggufEncoding = format === "gguf" ? analysis?.quantization_status || {} : null;
  const coreMlNumerics = format === "coreml" ? analysis?.quantization_status || {} : null;
  const coreMlProgram = format === "coreml" && analysis?.coreml?.model_type === "mlProgram";
  const tensorInventoryAssessed = analysis.tensor_count != null;
  const inventory = buildTensorInventory(analysis);
  const version = formatVersionEvidence(analysis, format);
  const batchProjection = deriveTfliteBatchOneProjection(analysis);
  const batchOne = batchProjection.status === "assumption_bound_batch_one"
    && Number.isFinite(Number(batchProjection.projected_total_macs));
  const macs = batchOne ? Number(batchProjection.projected_total_macs) : +analysis.total_macs;
  const ops = +analysis.total_ops;
  const nonMacOps = Math.max(0, ops - 2 * macs);
  const dynamic = analysis?.dynamic_shape_cost_contract;
  const shapeDependent = dynamic
    && dynamic.status !== "not_applicable_static_shapes"
    && !batchOne
    && (Number(dynamic.symbol_count || 0) > 0 || dynamic.total_macs_formula_status === "not_assessed");
  const formulaTerms = Number(dynamic?.total_macs_formula?.terms?.length || 0);
  const formulaGuards = Number(dynamic?.total_macs_formula?.preconditions?.length || 0);
  const dynamicDetail = shapeDependent
    ? `${formatNumber(dynamic.symbol_count || 0)} internal shape symbol(s); ${formulaGuards ? `${formatNumber(formulaGuards)} exact integer guard(s)` : `${formatNumber(formulaTerms)} exact polynomial term(s)`}; bind an approved non-batch shape profile for a numeric total`
    : batchOne
      ? `ASSUMPTION_BOUND: exact polynomial projection at serialized batch N=1; 0 non-batch dynamic input axes; ${formatNumber(batchProjection.internal_symbol_count)} propagated internal batch symbol(s)`
      : "";
  const identityDetail = [
    formatBytes(analysis.file_size ?? analysis.file_size_bytes ?? 0),
    analysis.model_sha256 ? `SHA-256 ${analysis.model_sha256}` : "SHA-256 not bound",
  ].join(" / ");
  const quantResearch = format === "tflite" ? analysis?.quant_research_coverage || buildQuantResearchCoverage(analysis) : null;
  const metrics = [
    ...(!quantResearch ? [] : [[
      "Artifact class / Quant research",
      quantResearch.artifact_class_label,
      `${formatNumber(quantResearch.class_supported_lab_count)}/${formatNumber(quantResearch.lab_count)} labs supported by class; ${formatNumber(quantResearch.artifact_applicable_lab_count)} artifact-applicable; ${formatNumber(quantResearch.assessed_lab_count)} assessed / ${formatNumber(quantResearch.partial_lab_count)} partial / ${formatNumber(quantResearch.not_assessed_lab_count)} not assessed. ${quantResearch.artifact_class_reason_code}`,
      "metric-wide metric-artifact-class",
    ]]),
    ["Model", analysis.filename, identityDetail, "metric-wide metric-artifact-name"],
    ["Format", (analysis.format || "tflite").toUpperCase(), version.detail],
    [
      version.schemaLabel,
      version.value,
      version.schemaDetail,
    ],
    [
      version.producerLabel,
      version.producerValue,
      version.producerDetail,
    ],
    [
      "Operators (layer count)",
      analysis.operator_count == null ? "Not assessed" : formatNumber(analysis.operator_count),
      analysis.operator_count == null
        ? "Model program payload was not decoded by this metadata adapter"
        : weightContainer
          ? "This weight-container format has no serialized graph operator table"
          : "Serialized graph operator count; framework source-layer count may differ",
    ],
    ["Tensors", analysis.tensor_count == null ? "Not assessed" : formatNumber(inventory.tensor_count), analysis.tensor_count == null ? "Model program tensor inventory was not decoded" : roleBreakdown(inventory), "metric-wide"],
    [
      "Quantization",
      tensorInventoryAssessed ? modelQuantizationStatus(analysis).label : "Not assessed",
      tensorInventoryAssessed
        ? ggufEncoding
          ? `${formatNumber(ggufEncoding.block_quantized_tensor_count || 0)}/${formatNumber(inventory.tensor_count)} tensors use source-pinned block-quantized GGML encodings; affine per-axis/per-tensor metadata is not applicable`
          : coreMlNumerics
            ? `${coreMlNumerics.summary || "Core ML numerical payload not assessed"} ${coreMlNumerics.detail || ""}`.trim()
            : `${formatNumber(inventory.quantized_tensors)}/${formatNumber(inventory.tensor_count)} tensors carry quantization parameters`
        : "Model program tensors and weight encodings were not decoded by this metadata adapter",
    ],
    ...(!tensorInventoryAssessed ? [] : ggufEncoding ? [[
      "Block-quantized tensors",
      formatNumber(ggufEncoding.block_quantized_tensor_count || 0),
      `${formatNumber(ggufEncoding.scalar_encoded_tensor_count || 0)} scalar-encoded tensor(s); ${formatNumber(ggufEncoding.unsupported_encoding_tensor_count || 0)} unsupported or invalid encoding tensor(s)`,
      "metric-wide",
    ], [
      "Encoding coverage",
      `${formatNumber(ggufEncoding.encoded_tensor_count || 0)}/${formatNumber(inventory.tensor_count)}`,
      analysis?.gguf?.payload_coverage_status || "payload coverage not assessed",
      "metric-wide",
    ]] : coreMlNumerics ? coreMlProgram ? [[
      "MIL serialized constants",
      formatNumber(analysis.weight_integrity?.parameter_count || 0),
      `${formatNumber(coreMlNumerics.blob_constant_count || 0)} package blob / ${formatNumber(coreMlNumerics.immediate_constant_count || 0)} immediate; ${formatNumber(analysis.weight_integrity?.assessed_parameter_count || 0)} numerical payload contract(s) assessed`,
      "metric-wide",
    ], [
      "Explicit quant/compression transforms",
      `${formatNumber((analysis.ops || []).filter((op) => op.quantization_state === "serialized_quantization_transform").length)}/${formatNumber((analysis.ops || []).length)}`,
      analysis.coreml?.mil_compression_contract
        ? `${formatNumber(analysis.coreml.mil_compression_contract.exact_contract_count)}/${formatNumber(analysis.coreml.mil_compression_contract.transform_count)} exact source-backed serialized contracts; ${formatNumber(analysis.coreml.mil_compression_contract.partial_contract_count)} explicit residual; runtime fusion and backend precision remain unobserved`
        : `${coreMlNumerics.assessment_status || "not assessed"}; serialized MIL transforms only, with runtime fusion and backend precision kept unobserved`,
    ]] : [[
      "Decoded WeightParams",
      formatNumber(coreMlNumerics.weight_parameter_count || 0),
      `${formatNumber(coreMlNumerics.quantized_weight_parameter_count || 0)} quantized / ${formatNumber(coreMlNumerics.fp32_weight_parameter_count || 0)} FP32 / ${formatNumber(coreMlNumerics.fp16_weight_parameter_count || 0)} FP16; counts describe stored parameters, not activation execution precision`,
      "metric-wide",
    ], [
      "Weight field coverage",
      `${formatNumber(coreMlNumerics.scanned_layer_count || 0)}/${formatNumber(coreMlNumerics.layer_count || 0)}`,
      `${coreMlNumerics.assessment_status || "not assessed"}; field, parameter-shape cardinality, and numerical payload contracts are assessed independently`,
    ]] : [[
      "Quantized tensors",
      formatNumber(inventory.quantized_tensors),
      roleBreakdown(inventory, "quantized"),
      "metric-wide",
    ], ["Per-channel", formatNumber(inventory.per_channel_tensors), roleBreakdown(inventory, "per_channel")], [
      "Per-tensor",
      formatNumber(inventory.per_tensor_tensors),
      [roleBreakdown(inventory, "per_tensor"), perTensorKernelFamilySummary(analysis)].filter(Boolean).join(" / "),
      "metric-wide",
    ]]),
    ...(!graphFormat ? [] : [[
      shapeDependent
        ? "MACs / Ops ratio"
        : `${analysis.mac_assessment ? "Assessed " : ""}MACs / Ops ratio 1:${macs ? (ops / macs).toFixed(2) : "N/A"}`,
      shapeDependent ? "Shape-dependent" : formatNumber(macs),
      shapeDependent || batchOne ? dynamicDetail : "",
    ], [
      shapeDependent
        ? "Ops / Non-MAC"
        : `Ops / Non-MAC ${formatNumber(nonMacOps)} (${ops ? (100 * nonMacOps / ops).toFixed(2) : "N/A"}%)`,
      shapeDependent ? "Shape-dependent" : formatNumber(ops),
      shapeDependent
        ? dynamicDetail
        : batchOne
          ? `${dynamicDetail}; Ops and non-MAC counts use the same serialized N=1 graph projection`
        : analysis.mac_assessment
        ? `${analysis.mac_assessment.assessed_compute_ops}/${analysis.mac_assessment.compute_ops} compute operators MAC-assessed`
        : "",
    ]]),
  ];
  return metrics.map(([label, value, detail, className = ""]) => {
    const node = document.createElement("div");
    node.className = `metric ${className}`.trim();
    const small = document.createElement("span");
    small.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = value || "-";
    node.append(small, strong);
    if (detail) {
      const explanation = document.createElement("small");
      explanation.className = "metric-detail";
      explanation.textContent = detail;
      node.append(explanation);
    }
    return node;
  });
}

function targetConditionCard(label, value, detail) {
  const node = document.createElement("div");
  node.className = "target-condition-card";
  const caption = document.createElement("span");
  caption.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value || "-";
  const explanation = document.createElement("small");
  explanation.textContent = detail || "";
  node.append(caption, strong, explanation);
  return node;
}

export function targetConditionCards(analysis) {
  const format = String(analysis?.format || "tflite").toLowerCase();
  const onnx = format === "onnx";
  if (!["tflite", "onnx"].includes(format)) {
    const scope = formatEvidenceScope(format, { analysis });
    const graphState = format === "coreml"
      ? (analysis?.ops || []).length
        ? `${formatNumber(analysis.ops.length)} serialized ops decoded`
        : "Not decoded for this model type"
      : "Not applicable to this container";
    const payload = analysis?.tensor_numerical_integrity || analysis?.weight_integrity || {};
    const assessedCount = payload.assessed_tensor_count ?? payload.assessed_parameter_count;
    const totalCount = payload.tensor_count ?? payload.parameter_count;
    const payloadState = assessedCount == null
      ? payload.status || "Not assessed"
      : `${formatNumber(assessedCount)}/${formatNumber(totalCount || 0)} payload contract(s) assessed`;
    return [
      targetConditionCard("Analysis Depth", scope.depth, scope.evidenceClass),
      targetConditionCard("Assessed Artifact Evidence", scope.label, scope.assessed),
      targetConditionCard("Serialized Graph", graphState, format === "coreml" ? "Decoded only when the selected Core ML model representation carries an addressable program graph" : "GGUF and SafeTensors do not serialize an executable operator graph"),
      targetConditionCard("Payload Coverage", payloadState, payload.byte_conservation_status || analysis?.gguf?.payload_coverage_status || analysis?.safetensors?.payload_coverage_status || "Coverage state is retained in the artifact evidence ledger"),
      targetConditionCard("Runtime / Release", "Not observed / not assessed", `${scope.runtimeBoundary}. ${scope.releaseStatus}.`),
    ];
  }
  const target = analysis?.target_profile || {};
  const hardware = target.hardware_spec || {};
  const cards = [
    targetConditionCard(
      onnx ? "Cache Reference Target" : "Selected Target",
      target.label || "Not bound",
      target.performance_model_evidence_class || "Target-profile evidence not declared",
    ),
    targetConditionCard(
      "CPU / Architecture",
      `${targetCoreText(target)} / ${target.architecture || "not declared"}`,
      `${target.in_order ? "In-order" : "Out-of-order or host-dependent"} execution profile; ${hardware.core_configuration || "implementation not bound"}`,
    ),
    targetConditionCard(
      "ISA",
      targetIsaText(target, hardware),
      `Dot product ${target.dot_product ? "yes" : "no"} / SVE2 ${target.sve2 ? "yes" : "no"} / FP32 lanes ${target.fp32_lanes || "not bound"} / INT8:FP32 planning ratio ${formatProfileScalar(target.int8_speedup_estimate || 1)}x`,
    ),
    targetConditionCard(
      "Cache Conditions",
      `${formatBytes(target.l1_data_bytes)} L1D / ${formatBytes(target.l2_bytes)} L2`,
      hardware.l1_instruction_bytes
        ? `${formatBytes(hardware.l1_instruction_bytes)} L1I; L2 scope ${target.l2_capacity_scope || "unbound"}; ${target.cache_assumption || "cache source bound"}`
        : `L2 scope ${target.l2_capacity_scope || "unbound"}; ${target.cache_assumption || "Planning values; host cache not observed"}`,
    ),
    targetConditionCard(
      onnx ? "Throughput Model" : "Roofline Parameters",
      onnx
        ? "Not applied"
        : `${formatProfileScalar(target.effective_peak_gops)} GOPS / ${formatProfileScalar(target.effective_memory_bandwidth_gbps)} GB/s`,
      onnx
        ? "Selected cache references affect row-working-set watches; no ORT EP throughput model is inferred"
        : `Low/high thresholds ${formatProfileScalar(target.memory_bound_intensity)} / ${formatProfileScalar(target.compute_bound_intensity)} ops/B; compute utilization factor ${formatProfileScalar(target.compute_utilization_factor ?? 1)} (uncalibrated) multiplies configured peak; ${target.performance_model_evidence_class || "evidence not declared"}`,
    ),
  ];
  if (String(target.id || "").includes("zynq")) {
    cards.push(targetConditionCard(
      "Modeled Execution Scope",
      "Cortex-A53 APU CPU only",
      "Programmable logic, DPU/NPU accelerators, DMA overlap, and accelerator memory paths are excluded. The report must not be read as a Zynq PL/DPU performance estimate.",
    ));
  }
  return cards;
}

export function insightDashboardCards(analysis, insights) {
  const l2WatchValue = insights.l2WatchAssessed
    ? `${formatNumber(insights.l2Watch.length)} L2`
    : "L2 N/A";
  const l2RatioValue = insights.l2WatchAssessed
    ? `${Number(insights.maxL2Ratio || 0).toFixed(2)}x L2`
    : "L2 N/A (shared capacity/concurrency unbound)";
  const format = String(analysis?.format || "tflite").toLowerCase();
  if (!["tflite", "onnx"].includes(format)) {
    const integrity = analysis?.tensor_numerical_integrity || analysis?.weight_integrity || {};
    const inventory = buildTensorInventory(analysis);
    if (format === "gguf") {
      const quant = analysis?.quantization_status || {};
      const backend = analysis?.gguf?.backend_compatibility || {};
      return prioritizeInsightCards([
        insightCard("Tensor Inventory", formatNumber(inventory.tensor_count), `${formatBytes(analysis?.gguf?.declared_tensor_byte_length || 0)} declared tensor payload`, "neutral", "quant", null, "OBSERVED / DERIVED"),
        insightCard("Block Encodings", formatNumber(quant.block_quantized_tensor_count || 0), `${formatNumber(quant.unsupported_encoding_tensor_count || 0)} unsupported or invalid encoding tensor(s)`, quant.unsupported_encoding_tensor_count ? "warn" : "good", "quant", null, "OBSERVED / DERIVED"),
        insightCard("Numerical Payload", `${formatNumber(integrity.assessed_tensor_count || 0)}/${formatNumber(integrity.tensor_count || 0)}`, `${formatNumber(integrity.nonfinite_value_count || 0)} non-finite value(s); ${integrity.byte_conservation_status || "byte conservation not assessed"}`, integrity.nonfinite_value_count ? "risk" : "good", "quant", null, "OBSERVED / DERIVED"),
        insightCard("llama.cpp Backend Prerequisites", backend.status || "Not assessed", `${formatNumber(backend.profiles?.length || 0)} source profile(s); architecture ${backend.architecture_registry_match ? "registered" : "not matched"}; selected build and execution not bound`, backend.status === "invalid" ? "risk" : backend.status === "source_candidate" ? "good" : "neutral", "stage", null, "SOURCE-BACKED / NOT EXECUTED"),
      ]);
    }
    if (format === "safetensors") {
      const safe = analysis?.safetensors || {};
      const hf = safe.hf_architecture_contract || {};
      const shardBindingNeedsReview = safe.sharded
        && /missing|invalid|partial|failed|mismatch|unresolved/i.test(String(safe.index_binding_status || "not assessed"));
      const architectureNeedsReview = /invalid|partial/.test(String(hf.status || ""));
      return prioritizeInsightCards([
        insightCard("Checkpoint Inventory", formatNumber(inventory.tensor_count), `${formatBytes(safe.payload_byte_length || 0)} declared tensor payload`, "neutral", "quant", null, "OBSERVED / DERIVED"),
        insightCard("Shard Binding", safe.sharded ? safe.index_binding_status || "Not assessed" : "Single file", safe.sharded ? `${formatNumber(safe.shard_count || 0)} selected shard(s)` : "No shard index is required for this artifact", shardBindingNeedsReview ? "warn" : "good", "quant", null, "OBSERVED / DERIVED"),
        insightCard("Numerical Payload", `${formatNumber(integrity.assessed_tensor_count || 0)}/${formatNumber(integrity.tensor_count || 0)}`, `${formatNumber(integrity.nonfinite_value_count || 0)} non-finite value(s); ${integrity.byte_conservation_status || "byte conservation not assessed"}`, integrity.nonfinite_value_count ? "risk" : "good", "quant", null, "OBSERVED / DERIVED"),
        insightCard("HF Architecture Contract", hf.model_type || "Config not bound", hf.architecture_kind === "hybrid_attention_ssm_moe"
          ? `${hf.fields?.attention_layer_count || "?"} attention + ${hf.fields?.mamba_layer_count || "?"} Mamba layers; ${hf.kv_state_projection?.elements_per_token_per_batch?.decimal || "?"} KV elements/token/batch + ${hf.recurrent_state_projection?.recurrent_state_elements_all_layers_per_batch?.decimal || "?"} recurrent elements/batch`
          : hf.architecture_kind === "ssm_recurrent"
          ? `${hf.recurrent_state_projection?.recurrent_state_elements_all_layers_per_batch?.decimal || "?"} recurrent-state elements/batch; canonical tensors ${formatNumber(hf.tensor_contract?.canonical_tensor_shape_match_count || 0)}/${formatNumber(hf.tensor_contract?.canonical_tensor_check_count || 0)}`
          : hf.architecture_kind === "sparse_moe_decoder"
            ? `${hf.moe_projection?.num_local_experts || hf.moe_projection?.expert_count || hf.fields?.num_local_experts || "?"} total / ${hf.fields?.num_experts_per_tok || "?"} active experts; canonical tensors ${formatNumber(hf.tensor_contract?.canonical_tensor_shape_match_count || 0)}/${formatNumber(hf.tensor_contract?.canonical_tensor_check_count || 0)}`
            : hf.kv_cache_elements_per_token_per_batch_decimal
              ? `${hf.kv_cache_elements_per_token_per_batch_decimal} KV-cache elements/token/batch; canonical tensors ${formatNumber(hf.tensor_contract?.canonical_tensor_shape_match_count || 0)}/${formatNumber(hf.tensor_contract?.canonical_tensor_check_count || 0)}`
              : hf.reason || "Select config.json with the checkpoint repository for registered architecture checks", architectureNeedsReview ? "warn" : hf.status === "assessed" ? "good" : "neutral", "quant", null, hf.evidence_class || "NOT ASSESSED"),
      ]);
    }
    const coreml = analysis?.coreml || {};
    const macs = analysis?.mac_assessment || {};
    const compression = coreml.mil_compression_contract || null;
    return prioritizeInsightCards([
      insightCard("Core ML Model Type", coreml.model_type || "Not decoded", `${formatNumber((analysis?.ops || []).length)} serialized op(s); native compute-unit placement is not observed`, "neutral", "stage", null, "OBSERVED SERIALIZED ARTIFACT"),
      insightCard("MAC Coverage", macs.compute_ops == null ? "Not assessed" : `${formatNumber(macs.assessed_compute_ops || 0)}/${formatNumber(macs.compute_ops)}`, analysis?.total_macs == null ? "No numeric MAC total is claimed" : `${formatNumber(analysis.total_macs)} MACs across assessed serialized compute ops`, macs.compute_ops && macs.assessed_compute_ops !== macs.compute_ops ? "warn" : "good", "stage", null, "DERIVED; EXPLICIT DENOMINATOR"),
      insightCard("Weight Payload", integrity.parameter_count == null ? "Not assessed" : `${formatNumber(integrity.assessed_parameter_count || 0)}/${formatNumber(integrity.parameter_count)}`, `${formatNumber(integrity.nonfinite_value_count || 0)} non-finite value(s); ${integrity.byte_conservation_status || "byte conservation not assessed"}`, integrity.nonfinite_value_count ? "risk" : "good", "quant", null, "OBSERVED / DERIVED"),
      ...(compression?.transform_count ? [insightCard("Serialized Compression", `${formatNumber(compression.exact_contract_count)}/${formatNumber(compression.transform_count)} exact`, `${formatNumber(compression.partial_contract_count)} explicit residual; runtime materialization and device placement remain external`, compression.partial_contract_count ? "warn" : "good", "quant", null, "SOURCE-PINNED / DERIVED")] : []),
    ]);
  }
  if (format === "onnx") {
    return prioritizeInsightCards([
      insightCard(
        "ORT EP Assignment",
        "Not observed",
        `${formatNumber(analysis?.ort_ep_portability_frontier?.execution_provider_count || 0)} source-backed EP rule set(s); import an ORT profile to bind actual node placement`,
        "warn",
        "stage",
        null,
        "SOURCE-BACKED; ASSIGNMENT NOT OBSERVED",
      ),
      insightCard(
        "Cache Row Watchlist",
        insights.maxL1Ratio == null ? "N/A" : `${formatNumber(insights.l1Watch.length)} L1 / ${l2WatchValue}`,
        insights.maxL1Ratio == null
          ? "No ai.onnx Conv row working set was assessable"
          : `Max ${Number(insights.maxL1Ratio).toFixed(2)}x L1 / ${l2RatioValue} (${formatBytes(insights.maxRowWorkingSet)}); ${formatNumber(insights.l1AssessedCount)} op(s) assessed`,
        insights.maxL1Ratio == null ? "neutral" : insights.maxL1Ratio > 3 ? "risk" : insights.maxL1Ratio >= 0.9 ? "warn" : "good",
        "stage",
        null,
        "DERIVED WITH REFERENCE CACHE PROFILE",
      ),
    ]);
  }
  return prioritizeInsightCards([
    int8SpeedupCard(analysis),
    insightCard(
      "Low-intensity Ops",
      `${formatNumber(insights.boundCounts["memory-bound"])} / ${formatNumber(insights.totalOps)}`,
      `${formatPercent(insights.memoryRatio)} memory-traffic candidates`,
      insights.memoryRatio > 0.45 ? "risk" : insights.memoryRatio > 0.25 ? "warn" : "good",
      "roofline",
      null,
      "HEURISTIC TARGET MODEL",
    ),
    insightCard(
      "Cache Row Watchlist",
      `${formatNumber(insights.l1Watch.length)} L1 / ${l2WatchValue}`,
      `Max ${Number(insights.maxL1Ratio || 0).toFixed(2)}x L1 / ${l2RatioValue} (${formatBytes(insights.maxRowWorkingSet)})`,
      insights.maxL1Ratio > 3 ? "risk" : insights.maxL1Ratio >= 0.9 ? "warn" : "good",
      "stage",
      null,
      "DERIVED WITH HEURISTIC TARGET PROFILE",
    ),
    insightCard(
      "Predicted Delegate Segments",
      insights.chainCount === 1 ? "1 segment" : `${formatNumber(insights.chainCount)} segments`,
      insights.effectiveChainBreaks
        ? `${formatNumber(insights.effectiveChainBreaks)} non-structural · ${formatNumber(insights.chainBreaks)} total breaks`
        : `0 breaks · longest ${formatNumber(insights.longestChain)} ops`,
      insights.effectiveChainBreaks ? "warn" : insights.chainBreaks ? "neutral" : "good",
      "xnnpack",
      null,
      "PREDICTED; PINNED RULEPACK",
    ),
    insightCard(
      "Fallback Traffic",
      formatPercent(insights.fallbackByteRatio),
      insights.topFallbackTraffic
        ? `${insights.topFallbackTraffic.name} ${formatBytes(insights.topFallbackTraffic.estimated_bytes)} · ${insights.topFallbackTraffic.count} ops`
        : "No fallback tensor traffic detected",
      insights.fallbackByteRatio >= 0.06 ? "risk" : insights.fallbackByteRatio >= 0.02 ? "warn" : "good",
      "xnnpack",
      null,
      "DERIVED LOGICAL PAYLOAD; PREDICTED PATH",
    ),
    insightCard(
      "Conditionally Delegatable MACs",
      formatPercent(insights.delegatedMacRatio),
      `${formatNumber(analysis.delegated_macs || 0)} MACs · ${formatBytes(analysis.fallback_estimated_bytes || 0)} fallback bytes`,
      insights.delegatedMacRatio >= 0.9 ? "good" : insights.delegatedMacRatio >= 0.6 ? "warn" : "risk",
      "xnnpack",
      null,
      "PREDICTED; EXPLICIT MAC DENOMINATOR",
    ),
    insightCard(
      "Channel Alignment",
      `${formatNumber(insights.misalignedOps)} ops`,
      insights.topMisaligned
        ? `#${padOp(insights.topMisaligned.index)} ${insights.topMisaligned.name} · ${formatPercent(insights.topMisaligned.channel_tail_overhead_percent || 0)} tail`
        : "Channel counts align to target multiple",
      Number(insights.topMisaligned?.channel_tail_overhead_percent || 0) >= 1 ? "risk" : insights.misalignedOps ? "warn" : "good",
      "graph",
      null,
      "DERIVED AGAINST TARGET LANE ASSUMPTION",
    ),
    insightCard(
      "Weight Packing",
      `${formatNumber(insights.packingWarnOps)} ops`,
      insights.topPackingWarn
        ? `#${padOp(insights.topPackingWarn.index)} ${insights.topPackingWarn.name} · ${Number(insights.topPackingWarn.weight_packing_overhead_us || 0).toFixed(1)} µs`
        : "No first-run packing warning",
      insights.packingWarnOps ? "warn" : "good",
      "graph",
      null,
      "HEURISTIC PACKING MODEL",
    ),
    insightCard(
      "Unsupported Break Suspects",
      formatNumber(insights.suspectTotal),
      insights.suspectSummary
        ? `${insights.suspectSummary} Structural zero-MAC breaks remain separately classified in the predicted segment inventory.`
        : "No non-structural suspect op families; structural zero-MAC breaks remain separately classified.",
      insights.suspectTotal ? "warn" : "good",
      "xnnpack",
      null,
      "SOURCE-BACKED PREDICTION",
    ),
  ]);
}

export function insightDashboardSignalItems(insights) {
  const items = insights.signals.length
    ? insights.signals
    : [{ label: "Static warnings", value: "No major static warnings found in this audit pass.", tone: "good" }];
  return items.map((item) => signalItem(item.label, item.value, item.tone));
}

export function insightDashboardRecommendationItems(analysis) {
  const onnx = String(analysis?.format || "tflite").toLowerCase() === "onnx";
  const items = Array.isArray(analysis.recommendations) && analysis.recommendations.length
    ? analysis.recommendations
    : [{ title: "No prioritized action", detail: onnx ? "Profile the selected ONNX Runtime execution provider to add runtime evidence." : "Run target-device profiling to validate static predictions.", tone: "neutral" }];
  const ranked = items
    .map((item, sourceIndex) => ({
      item,
      sourceIndex,
      metadata: recommendationMetadata(analysis, item),
    }))
    .sort((left, right) => left.metadata.axisOrder - right.metadata.axisOrder
      || right.metadata.sortImpactUs - left.metadata.sortImpactUs
      || Number(left.item.priority || left.sourceIndex + 1) - Number(right.item.priority || right.sourceIndex + 1)
      || left.sourceIndex - right.sourceIndex)
    .slice(0, 6);
  const groups = [
    { key: "steady", title: "Steady-State Levers", detail: "Per-inference modeled opportunities", rows: ranked.filter((row) => row.metadata.axisOrder === 0) },
    { key: "cold", title: "Cold-Start Levers", detail: "One-time packing and setup profiles", rows: ranked.filter((row) => row.metadata.axisOrder === 1) },
    { key: "review", title: "Contract And Design Review", detail: "No latency recovery is inferred", rows: ranked.filter((row) => row.metadata.axisOrder > 1) },
  ].filter((group) => group.rows.length);
  return groups.flatMap((group) => {
    const maxImpactUs = Math.max(0, ...group.rows.map((row) => Number(row.metadata.sortImpactUs || 0)));
    const heading = document.createElement("li");
    heading.className = "action-axis-heading";
    const title = document.createElement("strong");
    title.textContent = group.title;
    const detail = document.createElement("span");
    detail.textContent = `${group.detail}${maxImpactUs > 0 ? `; largest modeled component ${formatUs(maxImpactUs)}` : ""}`;
    heading.append(title, detail);
    return [
      heading,
      ...group.rows.map(({ item, metadata }, index) => recommendationItem(
        analysis,
        item,
        `${group.key === "steady" ? "S" : group.key === "cold" ? "C" : "R"}${index + 1}`,
        metadata,
      )),
    ];
  });
}

function recommendationItem(analysis, item, priorityLabel, suppliedMetadata = null) {
  const node = document.createElement("li");
  node.className = `signal-item action-item ${item.tone || "neutral"}`;
  const priority = document.createElement("span");
  priority.textContent = priorityLabel;
  const title = document.createElement("strong");
  title.textContent = item.title || "Engineering review";
  const detail = document.createElement("p");
  detail.textContent = item.detail || "No action detail emitted.";
  const metadata = suppliedMetadata || recommendationMetadata(analysis, item);
  const meta = document.createElement("div");
  meta.className = "action-meta";
  for (const value of [metadata.followUpType, metadata.axis, metadata.impact, metadata.effort]) {
    const badge = document.createElement("span");
    badge.textContent = value;
    meta.append(badge);
  }
  node.append(priority, title, detail, meta);
  return node;
}

function recommendationMetadata(analysis, item) {
  const text = `${item.title || ""} ${item.detail || ""}`.toLowerCase();
  const titleText = String(item.title || "").toLowerCase();
  const interventionId = titleText.includes("fallback")
    ? "predicted_fallback_removed"
    : titleText.includes("packing")
      ? "packing_removed"
      : titleText.includes("chain") || titleText.includes("boundary") || titleText.includes("partition break")
        ? "predicted_boundaries_removed"
        : null;
  const followUpType = recommendationFollowUpType(text, interventionId);
  const squeezeExcitationOps = new Set((analysis?.block_inventory?.blocks || [])
    .filter((block) => block.block_type === "squeeze_excitation")
    .flatMap((block) => block.op_indices || [])
    .map(Number));
  const meanBreaks = (analysis?.ops || []).filter((op) => op.xnnpack_chain_break && op.name === "MEAN");
  const allMeanBreaksMapToSe = meanBreaks.length > 0
    && meanBreaks.every((op) => squeezeExcitationOps.has(Number(op.index)));
  const effort = interventionId === "packing_removed"
    ? "Effort (heuristic): Low / runtime lifecycle"
    : interventionId === "predicted_boundaries_removed"
      ? allMeanBreaksMapToSe
        ? `Effort (heuristic): export/global-pooling review first; all ${meanBreaks.length} MEAN breaks map to squeeze-excitation blocks`
        : "Effort (heuristic): Medium-High / graph or delegate support"
      : interventionId === "predicted_fallback_removed"
        ? "Effort (heuristic): High / graph or kernel change"
        : text.includes("channel")
          ? "Effort (heuristic): High / architecture change"
          : text.includes("quant")
            ? "Effort (heuristic): Medium-High / calibration or QAT"
            : text.includes("fusion")
              ? "Effort (heuristic): Medium / conversion change"
              : text.includes("l1") || text.includes("cache")
                ? "Effort (heuristic): Medium / tiling and runtime validation"
                : "Effort (heuristic): Medium / engineering review";

  const intervention = (analysis?.deployment_frontier?.interventions || [])
    .find((candidate) => candidate.id === interventionId);
  const targetId = analysis?.target_profile?.id;
  const selected = intervention?.per_target?.find((row) => row.target_id === targetId);
  if (titleText.includes("quant")) {
    return {
      followUpType,
      axis: "Axis: numerical/accuracy contract",
      axisOrder: 2,
      sortImpactUs: 0,
      impact: "Impact: latency not inferred from quantization metadata",
      effort,
    };
  }
  if (titleText.includes("channel")) {
    return {
      followUpType,
      axis: "Axis: kernel-shape review",
      axisOrder: 3,
      sortImpactUs: 0,
      impact: "Impact: tail exposure is not separately recoverable in this model",
      effort,
    };
  }
  if (selected) {
    const recoverableUs = Number(selected.recoverable_us || 0);
    if (interventionId === "packing_removed") {
      return {
        followUpType,
        axis: "Axis: cold-start only",
        axisOrder: 1,
        sortImpactUs: recoverableUs,
        impact: `Cold-start component: ${formatUs(recoverableUs)} / ${formatPercent(selected.recoverable_share)} / ${Number(selected.upper_bound_speedup || 1).toFixed(2)}x upper bound`,
        effort,
      };
    }
    if (interventionId === "predicted_boundaries_removed") {
      const breakOps = (analysis?.ops || []).filter((op) => op.xnnpack_chain_break);
      const lowUs = breakOps.reduce((sum, op) => sum + Number(op.chain_break_overhead_us_low || 0), 0);
      const highUs = breakOps.reduce((sum, op) => sum + Number(op.chain_break_overhead_us_high || 0), 0);
      return {
        followUpType,
        axis: "Axis: cold-start setup only",
        axisOrder: 1,
        sortImpactUs: recoverableUs,
        impact: `Cold-start setup midpoint ${formatUs(recoverableUs)}; profile range ${formatUs(lowUs)}-${formatUs(highUs)} / ${formatPercent(selected.recoverable_share)} of cold total / not measured`,
        effort,
      };
    }
    return {
      followUpType,
      axis: "Axis: steady-state heuristic",
      axisOrder: 0,
      sortImpactUs: recoverableUs,
      impact: `Upper bound: ${formatUs(recoverableUs)} / ${formatPercent(selected.recoverable_share)} / ${Number(selected.upper_bound_speedup || 1).toFixed(2)}x`,
      effort,
    };
  }
  if (intervention) {
    const coldOnly = interventionId === "packing_removed" || interventionId === "predicted_boundaries_removed";
    return {
      followUpType,
      axis: coldOnly ? "Axis: cold-start only" : "Axis: steady-state heuristic",
      axisOrder: coldOnly ? 1 : 0,
      sortImpactUs: 0,
      impact: `Upper-bound range: ${formatPercent(intervention.min_recoverable_share)}-${formatPercent(intervention.max_recoverable_share)} / up to ${Number(intervention.max_upper_bound_speedup || 1).toFixed(2)}x`,
      effort,
    };
  }

  const op = Number(item.op_index) >= 0
    ? (analysis?.ops || []).find((candidate) => Number(candidate.index) === Number(item.op_index))
    : null;
  if (op && String(analysis?.format || "tflite").toLowerCase() === "tflite") {
    const estimate = estimateOpBottleneck(op, analysis.target_profile || {});
    const exposureUs = Math.max(Number(estimate.computeUs || 0), Number(estimate.memoryUs || 0))
      + Number(estimate.packingUs || 0)
      + Number(estimate.breakUs || 0)
      + Number(estimate.fallbackUs || 0);
    if (exposureUs > 0) {
      return {
        followUpType,
        axis: "Axis: review",
        axisOrder: 3,
        sortImpactUs: 0,
        impact: `Op exposure: ${formatUs(exposureUs)} / no recoverable share inferred`,
        effort,
      };
    }
  }
  return {
    followUpType,
    axis: "Axis: review",
    axisOrder: 3,
    sortImpactUs: 0,
    impact: "Impact: not modeled",
    effort,
  };
}

function recommendationFollowUpType(text, interventionId) {
  if (/document|metadata|owner|lineage|label map/.test(text)) return "DOCUMENT";
  if (/contract|bind|runtime build|preprocess/.test(text)) return "BIND";
  if (/profile|benchmark|measure|timing|cache|l1|packing/.test(text)) return "BENCHMARK";
  if (/re-export|converter|conversion|fusion/.test(text)) return "RE-EXPORT";
  if (/retrain|qat|ptq|calibrat/.test(text)) return "RETRAIN";
  if (interventionId || /confirm|validate|verify|quant|channel/.test(text)) return "VERIFY";
  return "REVIEW";
}

export function insightDashboardFallbackTrafficItems(analysis) {
  if (String(analysis?.format || "tflite").toLowerCase() === "onnx") {
    return [signalItem("Not assessed", "Execution-provider fallback assignment and materialized boundary traffic require ORT profiling.", "neutral")];
  }
  if (!Array.isArray(analysis.fallback_traffic_by_op_family) || !analysis.fallback_traffic_by_op_family.length) {
    return [signalItem("No fallback traffic family", "Predicted-fallback ops do not dominate static tensor bytes.", "good")];
  }
  return analysis.fallback_traffic_by_op_family.slice(0, 5).map((item) => signalItem(
    item.name,
    `${formatBytes(item.estimated_bytes)} / ${formatPercent(item.byte_percent || 0)} static bytes / ${formatNumber(item.count)} ops`,
    Number(item.byte_percent || 0) >= 0.03 ? "warn" : "neutral",
  ));
}

export function stageCard(stage) {
  const node = document.createElement("div");
  node.className = "stage";
  const title = document.createElement("b");
  title.textContent = `#${stage.index} ${humanizeStageKey(stage.key)}`;
  title.title = `Raw stage key: ${stage.key}`;
  const body = document.createElement("span");
  body.textContent = stageSummaryText(stage);
  node.append(title, body);
  return node;
}

export function histogramRow(item) {
  const tr = document.createElement("tr");
  tr.append(td(item.name), td(formatNumber(item.count), "numeric"));
  return tr;
}

export function topMacRows(analysis, onSelect = null) {
  return topMacOps(analysis).map((op) => {
    const tr = document.createElement("tr");
    if (onSelect) {
      tr.className = "clickable-row";
      tr.title = "Jump to op in Graph Explorer";
      tr.addEventListener("click", () => onSelect(op.index));
    }
    tr.append(
      td(`#${String(op.index).padStart(3, "0")}`),
      td((op.patterns || []).length ? `${op.name} · ${(op.patterns || []).join(" / ")}` : op.name),
      td(JSON.stringify(op.output_shapes), "wrap"),
      td(formatExactInteger(op.macs_decimal, op.macs, "N/A"), `numeric${op.macs_decimal == null && op.macs == null ? " muted-text" : ""}`),
      td(assessedOpLogicalBytes(op) == null ? "N/A" : formatNumber(op.estimated_bytes), `numeric${assessedOpLogicalBytes(op) == null ? " muted-text" : ""}`),
    );
    return tr;
  });
}

export function rooflineTableRows(analysis, onSelect = null) {
  return rooflineRows(analysis).map((op) => {
    const tr = document.createElement("tr");
    if (onSelect) {
      tr.className = "clickable-row";
      tr.title = "Jump to op in Graph Explorer";
      tr.addEventListener("click", () => onSelect(op.index));
    }
    tr.append(
      td(`#${String(op.index).padStart(3, "0")}`),
      td((op.patterns || []).length ? `${op.name} · ${(op.patterns || []).join(" / ")}` : op.name),
      td(op.intensity_ops_per_byte == null ? "N/A" : `${op.intensity_ops_per_byte.toFixed(2)} ops/byte`, `numeric${op.intensity_ops_per_byte == null ? " muted-text" : ""}`),
      td(assessedOpLogicalBytes(op) == null ? "N/A" : formatBytes(op.estimated_bytes), `numeric${assessedOpLogicalBytes(op) == null ? " muted-text" : ""}`),
      td(op.row_working_set_bytes == null ? "N/A" : formatBytes(op.row_working_set_bytes), `numeric${op.row_working_set_bytes == null ? " muted-text" : ""}`),
      td(rooflinePostureLabel(op)),
      td(op.static_action, "wrap"),
    );
    return tr;
  });
}

function rooflinePostureLabel(op) {
  const reason = String(op?.roofline_reason || "").split(":");
  if (reason.length < 5 || !["Q", "F"].includes(reason[1])) return op?.static_bound_guess || "-";
  const intensity = Number(reason[2]);
  const memoryThreshold = Number(reason[3]);
  const computeThreshold = Number(reason[4]);
  const distance = op.static_bound_guess === "mixed" && memoryThreshold > 0
    ? (intensity - memoryThreshold) / memoryThreshold
    : op.static_bound_guess === "compute-bound" && computeThreshold > 0
      ? (intensity - computeThreshold) / computeThreshold
      : null;
  if (distance == null || !Number.isFinite(distance) || Math.abs(distance) > 0.1) {
    return op.static_bound_guess;
  }
  return `${op.static_bound_guess} · ${Math.abs(distance * 100).toFixed(1)}% ${distance >= 0 ? "above" : "below"} threshold`;
}
