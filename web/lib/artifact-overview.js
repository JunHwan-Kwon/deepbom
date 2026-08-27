import {
  formatBytes,
  formatNumber,
  formatPercent,
  formatScientific,
  tensorShapeText,
} from "./format.js";
import { evidenceDisclosure } from "./dom.js";
import { deriveTfliteBatchOneProjection } from "./dynamic-shape-cost.js";
import { deriveGraphTopology } from "./graph-topology.js";
import {
  buildBiasScaleCheck,
  buildInputQuantizationConventionCheck,
  buildInterfaceQuantizationContractLedger,
  buildRepresentableKernelChannelCheck,
} from "./quantization-contract-summary.js";
import { classifyTensorRoles } from "./tensor-inventory.js";
import { safeTensorsQuantizationPanel } from "./safetensors-quantization-view.js";

const MAX_HISTOGRAM_ROWS = 8;
const MAX_INTERFACE_ROWS = 8;
const MAX_KERNEL_ROWS = 5;
const MAX_SCALE_ROWS = 5;

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function panel(title, caption, id, wide = false) {
  const node = element("article", `artifact-evidence-panel${wide ? " panel-wide" : ""}`);
  node.dataset.artifactPanel = id;
  const head = element("div", "artifact-panel-head");
  const copy = element("div");
  copy.append(element("h3", "", title), element("p", "", caption));
  head.append(copy);
  node.append(head);
  return node;
}

function panelStatus(node, value, evidenceClass = "DERIVED") {
  const status = element("span", "artifact-panel-status", `${evidenceClass} / ${value}`);
  node.querySelector(".artifact-panel-head")?.append(status);
}

function emptyState(message) {
  return element("p", "artifact-empty", message);
}

function barRow(label, value, ratio, detail = "", onActivate = null) {
  const row = element("div", "artifact-bar-row");
  const head = element("div", "artifact-bar-head");
  const name = element("strong", "", label);
  name.title = label;
  const amount = element("span", "", value);
  head.append(name, amount);
  const track = element("i", "artifact-bar-track");
  const fill = element("b");
  fill.style.width = `${Math.max(0, Math.min(100, Number(ratio || 0) * 100))}%`;
  track.append(fill);
  row.append(head, track);
  if (detail) row.append(element("small", "", detail));
  if (onActivate) {
    row.classList.add("artifact-bar-action");
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.title = `${detail ? `${detail}. ` : ""}Open the complete scale vector in Quant Evidence.`;
    row.addEventListener("click", onActivate);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onActivate();
      }
    });
  }
  return row;
}

function interfaceRow(direction, ordinal, tensor, contract, inputContract = null, convention = null) {
  const quantization = contract?.quantization || {};
  const statusClass = quantization.status === "complete" ? "contract-complete"
    : quantization.status === "invalid_or_incomplete" ? "contract-invalid" : "contract-unquantized";
  const row = element("div", `artifact-interface-row ${statusClass}`);
  const badge = element("span", `interface-direction direction-${direction}`, direction === "input" ? "IN" : "OUT");
  const identity = element("div", "artifact-interface-identity");
  const name = element("strong", "", tensor?.name || `${direction}_${ordinal}`);
  name.title = tensor?.name || `${direction}_${ordinal}`;
  const dynamicDimensions = Array.isArray(tensor?.shape_signature)
    ? tensor.shape_signature.filter((dimension) => Number(dimension) < 0).length
    : 0;
  identity.append(
    name,
    element(
      "small",
      "",
      [
        `${tensor?.dtype || "UNKNOWN"} ${tensorShapeText(tensor)}`,
        quantization.status === "complete"
          ? `${String(quantization.granularity || "").replace("_", "-")} affine / ${formatNumber(quantization.scale_count)} scale / ${formatNumber(quantization.zero_point_count)} zero-point`
          : quantization.status === "invalid_or_incomplete"
            ? `invalid or incomplete affine contract: ${quantization.cardinality_reason || "serialized values are incomplete"}`
            : "no affine mapping declared",
        quantization.status === "complete"
          ? `scale ${compactInterfaceVector(quantization.scales)} / zero-point ${compactInterfaceVector(quantization.zero_points)}`
          : null,
        quantization.scalar_real_code_domain
          ? `real code domain ${formatScientific(quantization.scalar_real_code_domain.real_min)} to ${formatScientific(quantization.scalar_real_code_domain.real_max)}`
          : ["per_axis", "blocked"].includes(quantization.granularity)
            ? `axis ${formatNumber(quantization.axis)}${quantization.block_size ? ` / block ${formatNumber(quantization.block_size)}` : ""} / ${quantization.cardinality_status}`
            : null,
        quantization.status === "complete" ? `symmetry ${quantization.symmetry_classification || "not encoded"}` : null,
        contract?.interface_contract_sha256 ? `ABI ${contract.interface_contract_sha256.slice(0, 12)}` : null,
        inputContract?.layout
          ? `${inputContract.layout} (${inputContract.layout_evidence_class || "DERIVED"})`
          : null,
        convention?.status === "no_common_full_domain_match"
          ? `no common full-domain preprocessing convention match; range ${convention.real_range.map(formatScientific).join(" to ")}`
          : convention?.matched_convention_ids?.length
            ? `range matches ${convention.matched_convention_ids.join(", ")}`
            : null,
        coreMlPreprocessingText(tensor?.coreml_preprocessing),
        dynamicDimensions ? `${dynamicDimensions} dynamic dim(s)` : null,
      ].filter(Boolean).join(" / "),
    ),
  );
  row.append(badge, identity);
  return row;
}

function coreMlPreprocessingText(preprocessing) {
  if (preprocessing?.kind === "image_scaler") {
    const values = Object.entries(preprocessing.serialized_values || {})
      .filter(([, value]) => value != null)
      .map(([name, value]) => `${name}=${formatScientific(value)}`);
    return `Core ML serialized image scaler: ${values.join(", ") || "no scalar fields explicitly serialized"}`;
  }
  if (preprocessing?.kind === "mean_image") {
    return `Core ML serialized mean image: ${formatNumber(preprocessing.value_count || 0)} FP32 values / range ${formatScientific(preprocessing.value_min)} to ${formatScientific(preprocessing.value_max)}`;
  }
  return null;
}

function operationCompositionPanel(analysis) {
  const node = panel(
    "Operator Composition",
    "Serialized operator families, ranked by layer count.",
    "operator-composition",
    true,
  );
  const rows = [...(analysis?.histogram || [])]
    .map((item) => ({ name: String(item.name || "UNKNOWN"), count: Number(item.count || 0) }))
    .filter((item) => item.count > 0)
    .sort((left, right) => right.count - left.count || left.name.localeCompare(right.name));
  const total = rows.reduce((sum, row) => sum + row.count, 0);
  panelStatus(node, `${formatNumber(rows.length)} families / ${formatNumber(total)} operators`);
  if (!rows.length) {
    node.append(emptyState("No operator histogram was emitted."));
    return node;
  }
  const body = element("div", "artifact-bar-list compact-bars");
  const maximum = Math.max(1, ...rows.map((row) => row.count));
  for (const row of rows.slice(0, MAX_HISTOGRAM_ROWS)) {
    body.append(barRow(
      row.name,
      `${formatNumber(row.count)} / ${formatPercent(row.count / total)}`,
      row.count / maximum,
    ));
  }
  node.append(body);
  if (rows.length > MAX_HISTOGRAM_ROWS) {
    node.append(element("p", "artifact-panel-footnote", `${formatNumber(rows.length - MAX_HISTOGRAM_ROWS)} additional operator families remain in the full Roofline inventory.`));
  }
  return node;
}

function interfacePanel(analysis) {
  const node = panel(
    "Model Interface",
    "Artifact-declared input/output ABI. Runtime-resolved dimensions are not inferred.",
    "model-interface",
    true,
  );
  const inputs = Array.isArray(analysis?.inputs) ? analysis.inputs : [];
  const outputs = Array.isArray(analysis?.outputs) ? analysis.outputs : [];
  const inputContracts = new Map((analysis?.input_contracts || []).map((contract, index) => [
    Number.isInteger(contract?.tensor_index) ? Number(contract.tensor_index) : index,
    contract,
  ]));
  const interfaceLedger = buildInterfaceQuantizationContractLedger(analysis);
  const inputConventions = buildInputQuantizationConventionCheck(analysis);
  const conventionsByOrdinal = new Map((inputConventions.details || [])
    .map((contract) => [Number(contract.ordinal), contract]));
  const interfaceContracts = new Map(interfaceLedger.parameters
    .map((contract) => [`${contract.direction}:${contract.ordinal}`, contract]));
  const rows = [
    ...inputs.map((tensor, index) => ({ direction: "input", index, tensor })),
    ...outputs.map((tensor, index) => ({ direction: "output", index, tensor })),
  ];
  const dynamicCount = rows.reduce((sum, row) => (
    sum + (row.tensor?.shape_signature || []).filter((dimension) => Number(dimension) < 0).length
  ), 0);
  panelStatus(
    node,
    `${formatNumber(interfaceLedger.quantized_parameter_count)} affine / ${formatNumber(interfaceLedger.unquantized_parameter_count)} no-affine / ${formatNumber(interfaceLedger.invalid_or_incomplete_parameter_count)} invalid${dynamicCount ? ` / ${formatNumber(dynamicCount)} dynamic dims` : ""}`,
    "OBSERVED",
  );
  if (!rows.length) {
    node.append(emptyState("No graph input/output tensor contracts were parsed."));
    return node;
  }
  const body = element("div", "artifact-interface-list");
  for (const row of rows.slice(0, MAX_INTERFACE_ROWS)) {
    const contract = row.direction === "input"
      ? inputContracts.get(Number.isInteger(row.tensor?.index) ? Number(row.tensor.index) : row.index)
        || analysis?.input_contracts?.[row.index]
      : null;
    body.append(interfaceRow(
      row.direction,
      row.index,
      row.tensor,
      interfaceContracts.get(`${row.direction}:${row.index}`) || null,
      contract,
      row.direction === "input" ? conventionsByOrdinal.get(row.index) || null : null,
    ));
  }
  node.append(body);
  if (rows.length > MAX_INTERFACE_ROWS) {
    node.append(element("p", "artifact-panel-footnote", `${formatNumber(rows.length - MAX_INTERFACE_ROWS)} additional interface tensors remain in Explorer and the Engineering Report.`));
  }
  return node;
}

function compactInterfaceVector(values) {
  const source = Array.isArray(values) ? values : [];
  if (!source.length) return "[]";
  const sample = source.slice(0, 4).map((value) => formatScientific(value)).join(", ");
  return source.length > 4 ? `[${sample}, ...] (${formatNumber(source.length)})` : `[${sample}]`;
}

function dynamicShapePanel(analysis) {
  const contract = analysis?.dynamic_shape_cost_contract;
  if (!contract || contract.status === "not_applicable_static_shapes") return null;
  const formula = contract.total_macs_formula;
  const symbolCount = Number(formula?.symbol_ids?.length || contract.symbol_count || 0);
  const termCount = Number(formula?.terms?.length || 0);
  const blockers = Number(contract.total_macs_unresolved_op_count || 0)
    + Number(contract.unresolved_dynamic_compute_op_count || 0);
  const batchProjection = deriveTfliteBatchOneProjection(analysis);
  const batchOne = batchProjection.status === "assumption_bound_batch_one";
  const node = panel(
    "Dynamic Shape Cost",
    "Shape-dependent workload summary. Exact symbolic evidence remains collapsed by default.",
    "dynamic-shape-cost",
    true,
  );
  panelStatus(
    node,
    batchOne
      ? `N=1 projection / ${formatNumber(termCount)} terms`
      : `${formatNumber(symbolCount)} internal symbols / ${formatNumber(termCount)} terms`,
    batchOne ? "ASSUMPTION_BOUND" : "DERIVED",
  );
  const ledger = element("div", "artifact-stat-grid");
  const stats = batchOne ? [
    ["External dynamic axes", Number(batchProjection.external_dynamic_axis_count || 0)],
    ["Non-batch free axes", 0],
    ["Propagated batch symbols", Number(batchProjection.internal_symbol_count || 0)],
    ["Dynamic tensors", Number(batchProjection.dynamic_tensor_count || 0)],
  ] : [
    ["Internal symbols", symbolCount],
    ["Exact MAC terms", termCount],
    ["Formula blockers", blockers],
    ["Dynamic tensors", Number(contract.dynamic_tensor_count || 0)],
  ];
  for (const [label, value] of stats) {
    const item = element("div");
    item.append(element("span", "", label), element("strong", "", formatNumber(value)));
    ledger.append(item);
  }
  node.append(
    ledger,
    element(
      "p",
      "artifact-panel-footnote",
      batchOne
        ? `Serialized batch N=1 evaluates to ${formatNumber(batchProjection.projected_total_macs)} MACs${batchProjection.projected_peak_live_payload_bytes != null ? ` and ${formatBytes(batchProjection.projected_peak_live_payload_bytes)} symbolic live-payload peak` : ""}. No spatial or feature input axis is dynamic; N>1 still requires an approved profile.`
        : `Numeric MAC and peak-memory totals require an approved non-batch shape binding. ${contract.liveness?.peak_live_payload_formula?.expression || contract.liveness?.peak_live_payload_max_formula?.expression ? "An exact symbolic live-memory peak expression is available." : "A symbolic live-memory peak was not derived."}`,
    ),
  );
  if (formula?.expression) {
    node.append(evidenceDisclosure(
      `Exact MAC polynomial (${formatNumber(termCount)} terms)`,
      formula.expression,
      { contentLabel: "Exact symbolic MAC polynomial" },
    ));
  }
  return node;
}

function kernelConsumerText(analysis, tensorIndex) {
  const consumers = (analysis?.ops || [])
    .filter((op) => (op.inputs || []).map(Number).includes(Number(tensorIndex)))
    .slice(0, 3)
    .map((op) => `#${String(op.index).padStart(3, "0")} ${op.name}`);
  return consumers.join(" / ") || "consumer not resolved";
}

function largestKernelPanel(analysis) {
  const format = String(analysis?.format || "tflite").toLowerCase();
  const weightContainer = ["gguf", "safetensors"].includes(format);
  const node = panel(
    weightContainer ? "Largest Tensor Payloads" : "Largest Kernel Constants",
    weightContainer
      ? "Top container tensor ranges by descriptor-validated stored byte length."
      : "Top embedded or verified kernel payloads by stored byte length; no shape-based estimate.",
    "largest-kernels",
  );
  const rows = (weightContainer
    ? (analysis?.tensors || []).map((tensor, index) => ({ index: Number(tensor?.index ?? index), tensor }))
    : classifyTensorRoles(analysis).filter(({ role }) => role === "kernel"))
    .filter(({ tensor }) => Number(weightContainer ? tensor?.byte_length : tensor?.buffer_data_length) > 0)
    .map(({ index, tensor }) => ({ index, tensor, bytes: Number(weightContainer ? tensor.byte_length : tensor.buffer_data_length) }))
    .sort((left, right) => right.bytes - left.bytes || left.index - right.index);
  const total = rows.reduce((sum, row) => sum + row.bytes, 0);
  panelStatus(
    node,
    rows.length ? `${formatBytes(total)} across ${formatNumber(rows.length)} ${weightContainer ? "tensors" : "kernels"}` : "not assessable",
    rows.length ? weightContainer ? "OBSERVED + DERIVED" : "OBSERVED" : "NOT_ASSESSABLE",
  );
  if (!rows.length) {
    node.append(emptyState(weightContainer
      ? "Stored tensor byte ranges are unavailable or use an encoding whose source-pinned block layout is not implemented."
      : "Stored kernel byte lengths are unavailable. Unresolved ONNX external data is not substituted with zero."));
    return node;
  }
  const body = element("div", "artifact-ranked-list");
  const maximum = Math.max(1, rows[0].bytes);
  for (const row of rows.slice(0, MAX_KERNEL_ROWS)) {
    const name = row.tensor?.name || `T${row.index}`;
    body.append(barRow(
      name,
      formatBytes(row.bytes),
      row.bytes / maximum,
      weightContainer
        ? `T${row.index} / ${row.tensor?.dtype || "UNKNOWN"} / payload offset ${formatNumber(row.tensor?.data_offset || 0)} B`
        : `T${row.index} / ${row.tensor?.dtype || "UNKNOWN"} / ${kernelConsumerText(analysis, row.index)}`,
    ));
  }
  node.append(body);
  return node;
}

function scaleSpreadPanel(analysis, { onOpenScaleVector = null } = {}) {
  if (!["tflite", "onnx"].includes(String(analysis?.format || "tflite").toLowerCase())) return null;
  const node = panel(
    "Per-axis Scale Spread",
    "Kernel scale vectors ranked once per compute layer. INT32 bias vectors are folded into their source kernel.",
    "scale-spread",
  );
  const biasScale = buildBiasScaleCheck(analysis);
  const representableChannels = buildRepresentableKernelChannelCheck(analysis);
  const biasByKernel = new Map((biasScale.details || []).map((detail) => [
    Number(detail.weight_tensor_index),
    detail,
  ]));
  const vitalityByKernel = new Map((representableChannels.details || []).map((detail) => [
    Number(detail.tensor_index),
    detail,
  ]));
  const rows = classifyTensorRoles(analysis)
    .filter(({ role, tensor }) => role === "kernel"
      && Number(tensor?.quant_scales || 0) > 1
      && tensor?.scale_ratio_meaningful
      && Number.isFinite(Number(tensor.scale_ratio))
      && Number(tensor.scale_ratio) >= 1)
    .map(({ tensor }) => ({
      tensor,
      ratio: Number(tensor.scale_ratio),
      minimum: Number(tensor.scale_min || 0),
      maximum: Number(tensor.scale_max || 0),
      bias: biasByKernel.get(Number(tensor.index)) || null,
      vitality: vitalityByKernel.get(Number(tensor.index)) || null,
    }))
    .sort((left, right) => right.ratio - left.ratio || Number(left.tensor.index) - Number(right.tensor.index));
  panelStatus(node, rows.length ? `${formatNumber(rows.length)} kernel vectors` : "no per-axis kernel vectors", rows.length ? "OBSERVED + DERIVED" : "NOT_APPLICABLE");
  if (!rows.length) {
    node.append(emptyState("No compute kernel carries multiple quantization scales. Per-tensor quantization is summarized above."));
    return node;
  }
  const body = element("div", "artifact-ranked-list");
  const maximumLog = Math.max(1, Math.log10(Math.max(1, rows[0].ratio)));
  for (const row of rows.slice(0, MAX_SCALE_ROWS)) {
    const name = row.tensor?.name || `T${row.tensor?.index}`;
    body.append(barRow(
      name,
      `${formatScientific(row.ratio)}x`,
      Math.log10(Math.max(1, row.ratio)) / maximumLog,
      [
        `T${row.tensor?.index} / min ${formatScientific(row.minimum)} / max ${formatScientific(row.maximum)} / ${formatNumber(row.tensor?.quant_scales)} scales`,
        row.bias
          ? `bias T${row.bias.bias_tensor_index} folded; scale contract ${String(row.bias.status || "not assessed").toUpperCase()}; input scale ${formatScientific(row.bias.declared_input_scale)}`
          : "bias scale contract not applicable",
        Number(row.vitality?.flagged_channel_count || 0) > 0
          ? `${formatNumber(row.vitality.flagged_channel_count)} near-zero representable channel(s)`
          : null,
      ].filter(Boolean).join(" / "),
      onOpenScaleVector ? () => onOpenScaleVector(Number(row.tensor?.index)) : null,
    ));
  }
  node.append(body);
  if (Number(biasScale.checked_groups || 0) > 0) {
    node.append(element(
      "p",
      "artifact-panel-footnote",
      `Bias scales are not separately ranked: ${formatNumber(biasScale.checked_groups)} kernel/bias group(s), ${formatNumber(biasScale.checked_channels)} channel(s) checked; ${formatNumber(biasScale.mismatch_groups)} mismatch group(s).`,
    ));
  }
  node.append(element(
    "p",
    "artifact-panel-footnote",
    `Artifact total: ${formatNumber(representableChannels.flagged_channels || 0)} near-zero representable channel(s) across ${formatNumber(representableChannels.flagged_kernel_tensors || 0)} kernel tensor(s). Per-row counts above are tensor-local. This thresholded scale-domain signal is distinct from exact constant-output proofs and decoded low-norm filters.`,
  ));
  return node;
}

function topologyPanel(analysis) {
  const node = panel(
    "Graph Topology",
    "Topological operator depth and tensor-consumer branch structure.",
    "graph-topology",
  );
  const ops = Array.isArray(analysis?.ops) ? analysis.ops : [];
  const annotated = ops.filter((op) => op?.topo_role || Number.isFinite(Number(op?.topo_depth)));
  if (!annotated.length) {
    panelStatus(node, "not emitted", "NOT_ASSESSABLE");
    node.append(emptyState("Topology annotations were not emitted for this graph."));
    return node;
  }
  const counts = new Map();
  let maxDepth = 0;
  let maxFanOut = 0;
  for (const op of annotated) {
    const role = String(op.topo_role || "through");
    counts.set(role, (counts.get(role) || 0) + 1);
    maxDepth = Math.max(maxDepth, Number(op.topo_depth || 0));
    maxFanOut = Math.max(maxFanOut, Number(op.topo_fan_out_max || 0));
  }
  const tensors = new Map((analysis?.tensors || []).map((tensor, index) => [
    Number.isInteger(Number(tensor?.index)) ? Number(tensor.index) : index,
    tensor,
  ]));
  const derivedPredecessors = new Map(deriveGraphTopology(ops).annotations.map((row) => [
    Number(row.op_index),
    Number(row.predecessor_count || 0),
  ]));
  const addOps = annotated.filter((op) => String(op?.name || "").replace(/[^A-Za-z0-9]+/g, "").toUpperCase() === "ADD");
  const addGraphMerges = addOps.filter((op, position) => {
    const opIndex = Number.isInteger(Number(op?.index)) ? Number(op.index) : position;
    return Number(op?.topo_predecessor_count ?? derivedPredecessors.get(opIndex) ?? 0) > 1;
  }).length;
  const addConstantInputs = addOps.filter((op) => (
    (op.inputs || []).some((index) => tensors.get(Number(index))?.constant_buffer)
  )).length;
  const addNonGraphMerges = Math.max(0, addOps.length - addGraphMerges);
  panelStatus(node, `${formatNumber(annotated.length)} annotated ops`);
  const ledger = element("div", "artifact-stat-grid");
  for (const [label, value] of [
    ["Max op depth", maxDepth],
    ["Branch splits", counts.get("branch-split") || 0],
    ["Branch merges", counts.get("branch-merge") || 0],
    [`ADD graph merges (of ${addOps.length})`, addGraphMerges],
    ["ADD parameter/external", addNonGraphMerges],
    ["Quant boundaries", counts.get("quant-boundary") || 0],
    ["Max tensor fan-out", maxFanOut],
    ["Subgraphs", Number(analysis?.subgraphs || 1)],
  ]) {
    const item = element("div");
    item.append(element("span", "", label), element("strong", "", formatNumber(value)));
    ledger.append(item);
  }
  node.append(
    ledger,
    element(
      "p",
      "artifact-panel-footnote",
      `Depth is longest serialized producer-to-consumer operator depth, not framework source-layer depth. ADDs with fewer than two producer ops are not counted as graph merges${addConstantInputs ? `; ${formatNumber(addConstantInputs)} ADD op(s) consume an observed constant tensor` : ""}.`,
    ),
  );
  return node;
}

function metadataPanel(analysis) {
  const format = String(analysis?.format || "tflite").toLowerCase();
  const onnx = format === "onnx";
  const metadata = analysis?.metadata_presence || {};
  if (format === "gguf") {
    const gguf = analysis.gguf || {};
    const integrity = analysis.tensor_numerical_integrity || {};
    const node = panel("GGUF Metadata", "Container identity and tensor-directory metadata.", "metadata-signatures");
    panelStatus(node, `${formatNumber(gguf.metadata_kv_count || 0)} metadata entries`, "OBSERVED");
    const ledger = element("dl", "artifact-metadata-ledger");
    for (const [label, value] of [
      ["GGUF version", gguf.version ?? "not decoded"], ["Architecture", gguf.architecture || "not declared"],
      ["Endianness", gguf.endianness || "not decoded"], ["Tensor alignment", `${formatNumber(gguf.alignment || 0)} B`],
      ["Tensor payload", `${formatBytes(gguf.declared_tensor_byte_length || 0)} (${formatNumber(gguf.declared_tensor_byte_length || 0)} B)`], ["Payload coverage", gguf.payload_coverage_status || "not assessed"],
      ["Numerical payload scan", `${integrity.status || "not assessed"}; ${formatNumber(integrity.assessed_tensor_count || 0)}/${formatNumber(integrity.tensor_count || 0)} tensors; ${formatNumber(integrity.decoded_value_count || 0)} decoded values`],
      ["Numerical integrity", `${formatNumber(integrity.nonfinite_value_count || 0)} non-finite / ${formatNumber(integrity.all_zero_tensor_count || 0)} all-zero tensor(s); byte conservation ${integrity.byte_conservation_status || "not assessed"}`],
      ["Parser scope", gguf.parser_scope || "not declared"],
    ]) ledger.append(element("dt", "", label), element("dd", "", String(value)));
    node.append(ledger);
    return node;
  }
  if (format === "safetensors") {
    const safe = analysis.safetensors || {};
    const integrity = analysis.tensor_numerical_integrity || {};
    const node = panel("SafeTensors Metadata", "Header conservation and optional shard-index binding.", "metadata-signatures");
    panelStatus(node, safe.sharded ? `${formatNumber(safe.shard_count || 0)} shards` : `${formatNumber(safe.tensor_count || 0)} tensors`, "OBSERVED");
    const ledger = element("dl", "artifact-metadata-ledger");
    for (const [label, value] of [
      ["Tensor entries", formatNumber(safe.tensor_count || 0)], ["Payload coverage", safe.payload_coverage_status || "not assessed"],
      ["Tensor payload", `${formatBytes(safe.payload_byte_length || 0)} (${formatNumber(safe.payload_byte_length || 0)} B)`],
      ["Numerical payload scan", `${integrity.status || "not assessed"}; ${formatNumber(integrity.assessed_tensor_count || 0)}/${formatNumber(integrity.tensor_count || 0)} tensors; ${formatNumber(integrity.decoded_value_count || 0)} decoded values`],
      ["Numerical integrity", `${formatNumber(integrity.nonfinite_value_count || 0)} non-finite / ${formatNumber(integrity.all_zero_tensor_count || 0)} all-zero tensor(s); byte conservation ${integrity.byte_conservation_status || "not assessed"}`],
      ["Duplicate-key validation", safe.duplicate_key_validation || "not assessed"],
      ["Shard index binding", safe.sharded ? safe.index_binding_status || "not assessed" : "not applicable"],
      ["Parser scope", safe.parser_scope || "not declared"],
    ]) ledger.append(element("dt", "", label), element("dd", "", String(value)));
    node.append(ledger);
    return node;
  }
  if (format === "coreml") {
    const coreml = analysis.coreml || {};
    const macs = analysis.mac_assessment || {};
    const integrity = analysis.weight_integrity || {};
    const node = panel("Core ML Model Metadata", "Model description, interface, and package-independent identity.", "metadata-signatures");
    panelStatus(node, `specification v${coreml.specification_version ?? "unknown"}`, "OBSERVED");
    const ledger = element("dl", "artifact-metadata-ledger");
    for (const [label, value] of [
      ["Model type", coreml.model_type || "not decoded"], ["Model version", metadata.metadata_model_version || "not declared"],
      ["Predicted feature", coreml.description?.predicted_feature_name || "not declared"],
      ["Predicted probabilities", coreml.description?.predicted_probabilities_name || "not declared"],
      ["Author", metadata.metadata_author || "not declared"], ["License", metadata.metadata_license || "not declared"],
      ["Preprocessing contract", metadata.preprocessing_contract_status || "not assessed"],
      ["Serialized graph", analysis.operator_count == null ? "not decoded" : `${formatNumber(analysis.operator_count)} ops / ${formatNumber(analysis.tensor_count || 0)} tensors`],
      ["MAC coverage", macs.compute_ops == null ? "not assessed" : `${formatNumber(macs.assessed_compute_ops || 0)}/${formatNumber(macs.compute_ops)} compute ops${analysis.total_macs == null ? "" : ` / ${formatNumber(analysis.total_macs)} MACs`}`],
      ["Weight payload integrity", integrity.parameter_count == null ? "not assessed" : `${integrity.status}; ${formatNumber(integrity.assessed_parameter_count || 0)}/${formatNumber(integrity.parameter_count)} parameters; ${formatNumber(integrity.nonfinite_value_count || 0)} non-finite`],
      ["Weight byte conservation", integrity.payload_bytes == null ? "not assessed" : `${formatNumber(integrity.assessed_payload_bytes || 0)}/${formatNumber(integrity.payload_bytes)} B`],
    ]) ledger.append(element("dt", "", label), element("dd", "", String(value)));
    node.append(ledger);
    return node;
  }
  if (format === "executorch") {
    const program = analysis.executorch_program || {};
    const flatTensor = analysis.executorch_flat_tensor || {};
    const pte = analysis.executorch_container === "pte";
    const node = panel(pte ? "ExecuTorch Program" : "ExecuTorch FlatTensor", pte ? "ET12 execution-plan, delegate, segment, and AOT memory identity." : "FT01 named external tensor/blob identity and segment conservation.", "metadata-signatures");
    panelStatus(node, pte ? `${formatNumber(analysis.subgraphs || 0)} plans / ${formatNumber(analysis.operator_count || 0)} instructions` : `${formatNumber(analysis.tensor_count || 0)} named entries`, "OBSERVED");
    const ledger = element("dl", "artifact-metadata-ledger");
    const rows = pte ? [
      ["Wire identifier", program.identifier || "not decoded"],
      ["Schema version", analysis.version ?? "not decoded"],
      ["Execution plans", formatNumber(analysis.subgraphs || 0)],
      ["Kernel / delegate calls", `${formatNumber(program.kernel_instruction_count || 0)} / ${formatNumber(program.delegate_instruction_count || 0)}`],
      ["Backend delegates", (program.delegates || []).map((item) => item.backend_id).join(", ") || "none serialized"],
      ["Appended segments", `${formatNumber(program.segments?.length || 0)} / ${analysis.size_breakdown?.appended_segment_bytes_decimal || "0"} B`],
      ["Planned non-constant memory", `${analysis.tensor_liveness?.planned_non_const_memory_decimal || "0"} B`],
      ["External PTD binding", program.external_tensor_data?.status || "not applicable"],
    ] : [
      ["Wire identifier", flatTensor.identifier || "not decoded"],
      ["Schema version", analysis.version ?? "not decoded"],
      ["Named entries", formatNumber(analysis.tensor_count || 0)],
      ["Appended segments", `${formatNumber(flatTensor.segments?.length || 0)} / ${analysis.size_breakdown?.appended_segment_bytes_decimal || "0"} B`],
      ["Execution graph", "not applicable to FT01"],
    ];
    for (const [label, value] of rows) ledger.append(element("dt", "", label), element("dd", "", String(value)));
    node.append(ledger);
    return node;
  }
  const node = panel(
    onnx ? "ONNX Graph Metadata" : "TFLite Signatures",
    onnx
      ? "ModelProto identity and graph-interface metadata."
      : "SignatureDef keys and typed model-metadata contract presence.",
    "metadata-signatures",
  );
  const rows = onnx ? [
    ["Producer", metadata.producer_name || analysis?.producer || "not declared"],
    ["Model domain", metadata.model_domain || "not declared"],
    ["Metadata properties", formatNumber(metadata.metadata_property_count || 0)],
    ["Graph name", analysis?.graph_name || "not declared"],
    ["Preprocessing contract", metadata.documented_preprocessing ? "documented" : "not machine-verifiably embedded"],
  ] : [
    ["SignatureDefs", formatNumber(metadata.signature_count || 0)],
    ["Signature keys", (metadata.signature_keys || []).join(", ") || "none embedded"],
    ["Metadata schema", metadata.metadata_schema_identifier || "not embedded"],
    ["Minimum parser", metadata.metadata_min_parser_version || "not declared"],
    ["Preprocessing contract", metadata.documented_preprocessing ? "documented" : "not machine-verifiably embedded"],
  ];
  panelStatus(node, onnx ? `${formatNumber(metadata.metadata_property_count || 0)} metadata properties` : `${formatNumber(metadata.signature_count || 0)} SignatureDefs`, "OBSERVED");
  const ledger = element("dl", "artifact-metadata-ledger");
  for (const [label, value] of rows) {
    ledger.append(element("dt", "", label), element("dd", "", String(value)));
  }
  node.append(ledger);
  return node;
}

function byteIntegrityPanel(analysis) {
  const ledger = analysis?.artifact_byte_integrity;
  if (String(analysis?.format || "").toLowerCase() !== "tflite" || !ledger?.schema) return null;
  const node = panel(
    "Artifact Byte Ownership",
    "Verified FlatBuffer references, terminal metadata ZIP, and unexplained suffix bytes.",
    "artifact-byte-integrity",
    true,
  );
  panelStatus(node, ledger.status || "not assessed", ledger.evidence_class || "DERIVED");
  const stats = element("div", "artifact-stat-grid");
  for (const [label, value] of [
    ["Conservation", ledger.conservation_status || "not assessed"],
    ["FlatBuffer envelope", formatBytes(ledger.flatbuffer_envelope_bytes || 0)],
    ["Metadata ZIP", formatBytes(ledger.metadata_archive_bytes || 0)],
    ["Unowned suffix", formatBytes(ledger.unowned_trailing_bytes || 0)],
  ]) {
    const item = element("div");
    item.append(element("span", "", label), element("strong", "", String(value)));
    stats.append(item);
  }
  node.append(stats, element("p", "artifact-panel-footnote", ledger.detail || "Byte ownership detail was not emitted."));
  if (ledger.issues?.length) {
    node.append(evidenceDisclosure(
      `${formatNumber(ledger.issues.length)} exact integrity issue(s)`,
      ledger.issues.join("\n"),
      { contentLabel: "Artifact byte-integrity issues" },
    ));
  }
  return node;
}

export function artifactOverviewHeader(analysis) {
  const format = String(analysis?.format || "tflite").toLowerCase();
  const graphFormat = ["tflite", "onnx"].includes(format) || format === "executorch" && analysis.executorch_container === "pte";
  const head = element("header", "artifact-overview-head");
  const copy = element("div");
  copy.append(
    element("span", "artifact-overview-kicker", "Artifact Overview"),
    element("h2", "", "Serialized model identity and workload"),
    element("p", "", graphFormat
      ? "Target-independent facts parsed or derived from the selected artifact. Selected-target performance assumptions begin in the dashboard below."
      : "Target-independent metadata parsed from the selected artifact. Runtime performance and graph behavior remain outside this bounded adapter scope."),
  );
  const binding = element("div", "artifact-identity-binding");
  const hash = String(analysis?.model_sha256 || "");
  const hashValue = hash ? `${hash.slice(0, 16)}...${hash.slice(-8)}` : "not bound";
  const code = element("code", "", hashValue);
  code.title = hash || "Artifact SHA-256 not bound";
  binding.append(
    element("span", "", "Artifact binding"),
    code,
    element("strong", "", formatBytes(analysis?.file_size ?? analysis?.file_size_bytes ?? 0)),
  );
  head.append(copy, binding);
  return head;
}

export function artifactOverviewPanels(analysis, options = {}) {
  return [
    operationCompositionPanel(analysis),
    interfacePanel(analysis),
    dynamicShapePanel(analysis),
    largestKernelPanel(analysis),
    scaleSpreadPanel(analysis, options),
    safeTensorsQuantizationPanel(analysis, { element, panel, panelStatus, emptyState }),
    topologyPanel(analysis),
    byteIntegrityPanel(analysis),
    metadataPanel(analysis),
  ].filter(Boolean);
}
