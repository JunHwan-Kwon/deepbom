import {
  buildGraphIndex,
  opDetailRows,
  opSteadyStateUs,
  quantStateLabel,
} from "./analysis.js";
import {
  collectFullGraph,
  layoutFullGraph,
} from "./graph-layout.js";
import { formatBytes, formatExactInteger, formatNumber, formatUs, padOp } from "./format.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const NODE_WIDTH = 216;
const NODE_HEIGHT = 88;
const INTERFACE_WIDTH = 194;
const INTERFACE_HEIGHT = 42;
const MIN_VIEW_WIDTH = 280;
const MIN_VIEW_HEIGHT = 220;

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

function button(label, className = "") {
  const node = element("button", className, label);
  node.type = "button";
  return node;
}

function svgElement(tag, attributes = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  return node;
}

function outputShape(op) {
  const shape = op?.output_shapes?.[0];
  return Array.isArray(shape) && shape.length ? shape.join("x") : "shape unavailable";
}

function interfaceColumnCount(analysis) {
  return Math.max(1, analysis?.input_tensor_indices?.length || 0, analysis?.output_tensor_indices?.length || 0);
}

function providerLabel(analysis, op) {
  const format = String(analysis?.format || "").toLowerCase();
  if (format === "onnx") {
    return op.ort_provider || op.execution_provider || "EP unobserved";
  }
  if (format === "coreml") return "Core ML runtime unobserved";
  return op.xnnpack_chain_id >= 0 ? `XNNPACK P${op.xnnpack_chain_id}` : "CPU / fallback";
}

function quantLabel(op) {
  const name = String(op?.name || "").toUpperCase();
  if (name === "QUANTIZE") return "quantize boundary";
  if (name === "DEQUANTIZE") return "dequantize boundary";
  if (op.quant_hole) return "FP32 island";
  const state = quantStateLabel(op);
  if (state !== "none") return state;
  return op.quant_risk === "risk" ? "quant review" : "not quantized";
}

function nodeRoleLabel(op) {
  const name = String(op?.name || "").toUpperCase();
  if (name === "QUANTIZE") return "Activation precision boundary (quantize)";
  if (name === "DEQUANTIZE") return "Activation precision boundary (dequantize)";
  if (["RESHAPE", "SQUEEZE", "EXPAND_DIMS", "TRANSPOSE"].includes(name)) return "Structural data movement";
  return "Graph operator";
}

function dtypeSummary(analysis, op, direction) {
  const direct = direction === "input" ? op?.input_dtypes : op?.output_dtypes;
  const ids = direction === "input" ? op?.inputs : op?.outputs;
  const values = Array.isArray(direct) && direct.length
    ? direct
    : (ids || []).filter((id) => Number(id) >= 0).map((id) => analysis?.tensors?.[id]?.dtype);
  const normalized = [...new Set(values.map((value) => String(value || "").trim()).filter(Boolean))];
  return normalized.length ? normalized.join(" / ") : "Not assessed";
}

function tensorPortX(op, direction, tensorId, visibleIds = null) {
  const ids = [...new Set((visibleIds || (direction === "input" ? op?.inputs : op?.outputs))
    ?.map(Number)
    .filter((id) => id >= 0) || [])];
  const ordinal = Math.max(0, ids.indexOf(Number(tensorId)));
  return NODE_WIDTH * (ordinal + 1) / (Math.max(1, ids.length) + 1);
}

function nodeEvidenceLabel(analysis, op, overlay) {
  if (overlay === "delegation") return providerLabel(analysis, op);
  if (overlay === "quant") return quantLabel(op);
  if (overlay === "latency") {
    const macLabel = op?.macs_decimal != null || op?.macs != null
      ? `${formatExactInteger(op.macs_decimal, op.macs)} MACs`
      : "MACs N/A";
    return `${formatUs(opSteadyStateUs(op))} / ${macLabel}`;
  }
  const dtype = dtypeSummary(analysis, op, "output");
  return `${dtype} / ${providerLabel(analysis, op)}`;
}

function clampViewBox(viewBox, layout) {
  if (!layout) return viewBox;
  const bounds = layout.bounds;
  const source = viewBox.map(Number);
  let width = Number.isFinite(source[2]) ? source[2] : bounds.width;
  let height = Number.isFinite(source[3]) ? source[3] : bounds.height;
  const minimumFactor = Math.max(MIN_VIEW_WIDTH / Math.max(1, width), MIN_VIEW_HEIGHT / Math.max(1, height));
  const maximumFactor = Math.min(
    Math.max(900, bounds.width * 1.8) / Math.max(1, width),
    Math.max(700, bounds.height * 1.4) / Math.max(1, height),
  );
  const factor = minimumFactor > 1 ? minimumFactor : maximumFactor < 1 ? maximumFactor : 1;
  width *= factor;
  height *= factor;
  const padX = width * 0.16;
  const padY = height * 0.16;
  const minX = bounds.width <= width
    ? bounds.x + (bounds.width - width) / 2
    : bounds.x - padX;
  const maxX = bounds.width <= width
    ? minX
    : bounds.x + bounds.width - width + padX;
  const minY = bounds.height <= height
    ? bounds.y + (bounds.height - height) / 2
    : bounds.y - padY;
  const maxY = bounds.height <= height
    ? minY
    : bounds.y + bounds.height - height + padY;
  return [
    Math.max(minX, Math.min(maxX, Number.isFinite(source[0]) ? source[0] : minX)),
    Math.max(minY, Math.min(maxY, Number.isFinite(source[1]) ? source[1] : minY)),
    width,
    height,
  ];
}

function zoomViewBoxAt(viewBox, factor, anchorX, anchorY, layout) {
  const [x, y, width, height] = viewBox;
  const boundedFactor = Math.max(0.72, Math.min(1.38, Number(factor) || 1));
  const px = (anchorX - x) / Math.max(1, width);
  const py = (anchorY - y) / Math.max(1, height);
  const nextWidth = width * boundedFactor;
  const nextHeight = height * boundedFactor;
  return clampViewBox([
    anchorX - nextWidth * px,
    anchorY - nextHeight * py,
    nextWidth,
    nextHeight,
  ], layout);
}

function zoomPercent(viewBox, layout) {
  if (!layout || !viewBox) return 100;
  const graphArea = Math.max(1, layout.bounds.width * layout.bounds.height);
  const viewArea = Math.max(1, viewBox[2] * viewBox[3]);
  return Math.max(1, Math.min(9999, Math.round(Math.sqrt(graphArea / viewArea) * 100)));
}

function projectionRow(projection, opIndex, index) {
  return index?.ops?.get(Number(opIndex))
    || projection?.op_projections?.find((row) => Number(row.op_index) === Number(opIndex))
    || null;
}

function projectionEdge(projection, edge, index) {
  return index?.edges?.get(`${edge.from}:${edge.to}:${edge.tensorId}`)
    || projection?.propagation_edges?.find((row) =>
    Number(row.producer_op_index) === Number(edge.from)
    && Number(row.consumer_op_index) === Number(edge.to)
    && Number(row.tensor_index) === Number(edge.tensorId),
    )
    || null;
}

function contractShape(contracts = []) {
  const activation = contracts.find((item) => !item.constant) || contracts[0];
  return Array.isArray(activation?.shape) && activation.shape.length
    ? activation.shape.join("x")
    : "shape unavailable";
}

function nodeTone(analysis, op, overlay, projection, projectionIndex) {
  if (overlay === "redesign") {
    const row = projectionRow(projection, op.index, projectionIndex);
    return {
      direct_edit: "nv-node-redesign-direct",
      propagated_contract: "nv-node-redesign-propagated",
      global_projection: "nv-node-redesign-global",
      unchanged: "nv-node-muted",
    }[row?.change_class] || "nv-node-muted";
  }
  if (overlay === "delegation") {
    if (op.xnnpack_chain_break) return "nv-node-risk";
    return op.xnnpack_chain_id >= 0 ? "nv-node-delegated" : "nv-node-muted";
  }
  if (overlay === "quant") {
    if (op.quant_hole || op.quant_risk === "risk") return "nv-node-risk";
    return /int8|uint8/i.test(quantLabel(op)) ? "nv-node-quant" : "nv-node-muted";
  }
  if (overlay === "latency") {
    const values = (analysis?.ops || []).map(opSteadyStateUs);
    const maximum = Math.max(0, ...values);
    const ratio = maximum > 0 ? opSteadyStateUs(op) / maximum : 0;
    return ratio >= 0.5 ? "nv-node-risk" : ratio >= 0.2 ? "nv-node-warn" : "nv-node-cool";
  }
  if (op.topo_role === "merge") return "nv-node-merge";
  if (Number(op.topo_fan_out_max || 0) > 1) return "nv-node-split";
  return "nv-node-default";
}

function sourceConditionSignals(op) {
  const issues = [];
  const watches = [];
  const ratio = Number(op?.row_working_set_ratio);
  const quantRisk = String(op?.quant_risk || "none").toLowerCase();

  if (op?.quant_hole) {
    issues.push(`activation precision boundary${op.quant_hole_class ? ` (${op.quant_hole_class})` : ""}`);
  } else if (quantRisk === "risk") {
    issues.push("quantization integrity risk");
  } else if (quantRisk === "warn" || quantRisk === "warning") {
    watches.push("quantization integrity watch");
  }
  if (op?.quant_zero_point_risk === true || String(op?.quant_zero_point_risk).toLowerCase() === "risk") {
    issues.push("quantized zero-point contract risk");
  }
  if (Number.isFinite(ratio) && ratio > 1) {
    issues.push(`logical row working set ${ratio.toFixed(2)}x L1D`);
  } else if (Number.isFinite(ratio) && ratio >= 0.9) {
    watches.push(`logical row working set ${ratio.toFixed(2)}x L1D`);
  }
  if (op?.xnnpack_chain_break) {
    watches.push(`predicted XNNPACK boundary${op.xnnpack_break_class ? ` (${op.xnnpack_break_class})` : ""}`);
  }
  if (op?.channel_alignment_status === "misaligned") {
    watches.push(`channel alignment tail${op.channel_alignment_multiple ? ` (x${op.channel_alignment_multiple})` : ""}`);
  }
  if (op?.weight_packing_risk === "warn") {
    watches.push("cold-start weight packing watch");
  }
  return { issues, watches };
}

function activationContract(contracts = []) {
  return contracts.find((item) => !item.constant) || contracts[0] || null;
}

function contractChannels(contracts = []) {
  const shape = activationContract(contracts)?.shape;
  if (!Array.isArray(shape) || !shape.length) return null;
  const channels = Number(shape.at(-1));
  return Number.isSafeInteger(channels) && channels > 0 ? channels : null;
}

function projectedConditionSignals(op, row) {
  if (!row) {
    const source = sourceConditionSignals(op);
    return { ...source, resolved: [] };
  }
  const issues = [];
  const watches = [];
  const resolved = [];
  const quantRisk = String(op?.quant_risk || "none").toLowerCase();

  if (op?.quant_hole) {
    issues.push(`activation precision boundary${op.quant_hole_class ? ` (${op.quant_hole_class})` : ""}`);
  } else if (quantRisk === "risk") {
    issues.push("quantization integrity risk retained from the source artifact");
  } else if (quantRisk === "warn" || quantRisk === "warning") {
    watches.push("quantization integrity watch retained from the source artifact");
  }
  if (op?.quant_zero_point_risk === true || String(op?.quant_zero_point_risk).toLowerCase() === "risk") {
    issues.push("quantized zero-point contract risk retained from the source artifact");
  }

  const sourceL1 = Number(row.source_l1_ratio);
  const projectedL1 = Number(row.projected_l1_ratio);
  if (Number.isFinite(projectedL1)) {
    if (projectedL1 > 1) {
      issues.push(`projected logical row working set ${projectedL1.toFixed(2)}x L1D`);
    } else if (projectedL1 >= 0.9) {
      watches.push(`projected logical row working set ${projectedL1.toFixed(2)}x L1D`);
    } else if (Number.isFinite(sourceL1) && sourceL1 >= 0.9) {
      resolved.push(`logical row working set clears the 0.90x L1D watch (${sourceL1.toFixed(2)}x -> ${projectedL1.toFixed(2)}x)`);
    }
  } else if (Number.isFinite(sourceL1)) {
    if (sourceL1 > 1) issues.push(`source logical row working set ${sourceL1.toFixed(2)}x L1D; projection not assessed`);
    else if (sourceL1 >= 0.9) watches.push(`source logical row working set ${sourceL1.toFixed(2)}x L1D; projection not assessed`);
  }

  if (op?.xnnpack_chain_break) {
    watches.push(`predicted XNNPACK boundary retained from the source artifact${op.xnnpack_break_class ? ` (${op.xnnpack_break_class})` : ""}`);
  }
  if (op?.channel_alignment_status === "misaligned") {
    const multiple = Number(op.channel_alignment_multiple);
    const sourceChannels = contractChannels(row.source_outputs);
    const projectedChannels = contractChannels(row.projected_outputs);
    const outputContractWasFlagged = Number.isSafeInteger(multiple)
      && multiple > 1
      && Number.isSafeInteger(sourceChannels)
      && sourceChannels % multiple !== 0;
    if (outputContractWasFlagged && Number.isSafeInteger(projectedChannels)) {
      if (projectedChannels % multiple === 0) {
        resolved.push(`projected output channels satisfy the x${multiple} source-backed alignment condition (${sourceChannels} -> ${projectedChannels})`);
      } else {
        watches.push(`projected output channels ${projectedChannels} retain the x${multiple} alignment tail`);
      }
    } else {
      watches.push(`source channel alignment tail retained${multiple > 1 ? ` (x${multiple})` : ""}`);
    }
  }
  if (op?.weight_packing_risk === "warn") {
    watches.push("cold-start weight packing watch retained from the source artifact");
  }
  return { issues, watches, resolved };
}

function projectionContractState(analysis, projection, op, projectionIndex) {
  const opIndex = Number(op?.index);
  const row = projectionRow(projection, opIndex, projectionIndex);
  const scopedConstraints = (projection?.constraints || []).filter((item) =>
    String(item.scope || "").startsWith(`#${String(opIndex).padStart(3, "0")}`),
  );
  const blocked = projectionIndex?.unresolved?.has(Number(opIndex))
    || scopedConstraints.some((item) =>
      item.severity === "error" && item.code?.startsWith("RD-SHAPE-"),
    );
  if (blocked) return {
    id: "blocked",
    className: "nv-contract-blocked",
    label: "Blocked",
    detail: "A projected tensor contract is inconsistent.",
    reasons: scopedConstraints.filter((item) => item.severity === "error").map((item) => item.detail || item.message || item.code),
    resolved: [],
  };
  const signals = projectedConditionSignals(op, row);
  if (signals.issues.length) return {
    id: "issue",
    className: "nv-contract-issue",
    label: "Issue",
    detail: "A projected or source-retained deployment condition remains unmet.",
    reasons: signals.issues,
    resolved: signals.resolved,
  };
  if (signals.watches.length) return {
    id: "watch",
    className: "nv-contract-watch",
    label: "Watch",
    detail: "A target-dependent, predicted, or source-retained condition requires review.",
    reasons: signals.watches,
    resolved: signals.resolved,
  };
  if (!row || row.shape_rule_status === "not_assessed") return {
    id: "unassessed",
    className: "nv-contract-unassessed",
    label: "Not assessed",
    detail: "No exact shape rule was available for this operator.",
    reasons: [],
    resolved: signals.resolved,
  };
  if (row.shape_rule_status === "serialized_shape_scaling"
    || scopedConstraints.some((item) => item.severity === "warning")) {
    return {
      id: "conditional",
      className: "nv-contract-conditional",
      label: "Conditional",
      detail: "The projected contract uses serialized-shape scaling or retains a warning.",
      reasons: scopedConstraints.filter((item) => item.severity === "warning").map((item) => item.detail || item.message || item.code),
      resolved: signals.resolved,
    };
  }
  return {
    id: "satisfied",
    className: "nv-contract-satisfied",
    label: "Satisfied",
    detail: "The projected input/output shape contract satisfies an exact analyzer rule and no current node-level deployment condition is flagged.",
    reasons: [],
    resolved: signals.resolved,
  };
}

function projectionContractSummary(analysis, projection, projectionIndex) {
  const counts = {
    blocked: 0,
    issue: 0,
    watch: 0,
    conditional: 0,
    unassessed: 0,
    satisfied: 0,
  };
  for (const op of analysis?.ops || []) {
    const state = projectionContractState(analysis, projection, op, projectionIndex);
    counts[state.id] = Number(counts[state.id] || 0) + 1;
  }
  return counts;
}

export function getRedesignContractState(analysis, projection, opIndex) {
  const op = (analysis?.ops || []).find((item) => Number(item.index) === Number(opIndex));
  if (!op) return null;
  const index = indexProjection(projection);
  return projectionContractState(analysis, projection, op, index);
}

export function summarizeRedesignContracts(analysis, projection) {
  return projectionContractSummary(analysis, projection, indexProjection(projection));
}

function edgeTone(analysis, edge, overlay, projection, projectionIndex) {
  if (overlay === "redesign") {
    const row = projectionEdge(projection, edge, projectionIndex);
    if (row?.changed && row.change_class === "direct_edit_output") return "nv-edge-redesign-direct";
    if (row?.changed) return "nv-edge-redesign-propagated";
    return "nv-edge";
  }
  const target = (analysis?.ops || []).find((op) => op.index === edge.to);
  if (overlay === "delegation" && target?.xnnpack_chain_break) return "nv-edge-risk";
  if (overlay === "quant" && target?.quant_hole) return "nv-edge-risk";
  return "nv-edge";
}

function relatedOps(graph, selectedIndex) {
  const upstream = new Set();
  const downstream = new Set();
  for (const edge of graph.edges) {
    if (edge.to === selectedIndex) upstream.add(edge.from);
    if (edge.from === selectedIndex) downstream.add(edge.to);
  }
  return { upstream, downstream };
}

function renderDetail(root, analysis, graph, selectedIndex, actions, projection, projectionIndex, mode) {
  root.replaceChildren();
  const op = (analysis?.ops || []).find((item) => item.index === selectedIndex);
  if (!op) {
    root.append(element("p", "nv-empty", "Select a node to inspect its deployment evidence."));
    return;
  }
  const related = relatedOps(graph, selectedIndex);
  const projected = projectionRow(projection, selectedIndex, projectionIndex);
  const projectedContract = mode === "redesign"
    ? projectionContractState(analysis, projection, op, projectionIndex)
    : null;
  const owningBlock = analysis?.block_inventory?.blocks?.find((block) => block.op_indices?.includes(op.index)) || null;
  const sourceContextRows = [
    ["Node role", nodeRoleLabel(op)],
    ["Input precision", dtypeSummary(analysis, op, "input")],
    ["Output precision", dtypeSummary(analysis, op, "output")],
    ["Static assignment", providerLabel(analysis, op)],
  ];
  const head = element("div", "nv-detail-head");
  head.append(
    element("span", "nv-kicker", `OP #${padOp(op.index)}`),
    element("h3", "", op.name || "UNKNOWN"),
    element("p", "", `${outputShape(op)} / ${op.output_dtypes?.[0] || graph.nodes.find((item) => item.op.index === op.index)?.outputDtype || "dtype unavailable"}`),
  );
  const metrics = element("div", "nv-detail-grid");
  const rows = mode === "redesign" && projected ? [
    ...sourceContextRows,
    ["Change", String(projected.change_class || "unchanged").replaceAll("_", " ")],
    ["Contract", `${projectedContract.label} / ${projectedContract.detail}`],
    ["Remaining signals", projectedContract.reasons?.length ? projectedContract.reasons.join(" / ") : "No flagged projected or retained condition"],
    ["Cleared by projection", projectedContract.resolved?.length ? projectedContract.resolved.join(" / ") : "None"],
    ["Shape evidence", String(projected.shape_rule_status || "not_assessed").replaceAll("_", " ")],
    ["Shape", `${contractShape(projected.source_outputs)} -> ${contractShape(projected.projected_outputs)}`],
    ["Steady", `${formatUs(projected.source_steady_us ?? null)} -> ${formatUs(projected.projected_steady_us ?? null)}`],
    ["Cold", `${formatUs(projected.source_cold_us ?? null)} -> ${formatUs(projected.projected_cold_us ?? null)}`],
    ["MACs", `${formatNumber(projected.source_macs || 0)} -> ${formatNumber(projected.projected_macs || 0)}`],
    ["L1 logical", `${projected.source_l1_ratio == null ? "N/A" : `${Number(projected.source_l1_ratio).toFixed(2)}x`} -> ${projected.projected_l1_ratio == null ? "N/A" : `${Number(projected.projected_l1_ratio).toFixed(2)}x`}`],
  ] : [
    ...sourceContextRows,
    ["Steady estimate", formatUs(opSteadyStateUs(op))],
    ["Cold estimate", formatUs(op.bottleneck_total_us ?? null)],
    ["MACs", formatExactInteger(op.macs_decimal, op.macs)],
    ["Quant state", quantLabel(op)],
    ["L1 logical ratio", op.row_working_set_ratio == null ? "Not assessed" : `${Number(op.row_working_set_ratio).toFixed(2)}x`],
  ];
  for (const [label, value] of rows) {
    const item = element("div");
    item.append(element("span", "", label), element("strong", "", value));
    metrics.append(item);
  }
  const actionsRow = element("div", "nv-detail-actions");
  if (mode === "redesign") {
    const editableBlock = projected?.block_id || owningBlock?.block_id;
    if (editableBlock) {
      const edit = button("Edit owning block", "primary-action");
      edit.addEventListener("click", () => actions.editNode(op.index));
      actionsRow.append(edit);
    } else {
      const stateBadge = element("span", "nv-editability-badge", "Not directly editable");
      stateBadge.title = "This operator has no schema-addressable semantic block. Use model-wide controls or select an owning block explicitly.";
      actionsRow.append(stateBadge);
    }
    if (projected?.direct_edit_fields?.length) {
      actionsRow.append(element("span", "nv-edit-fields", projected.direct_edit_fields.join(" / ")));
    }
  } else {
    const openOps = button("Open in Ops", "primary-action");
    openOps.addEventListener("click", () => actions.openOps(op.index));
    const openQuant = button("Open Quant Evidence", "secondary-action");
    const hasQuantEvidence = actions.canOpenQuant(op.index);
    openQuant.disabled = !hasQuantEvidence;
    openQuant.title = hasQuantEvidence
      ? "Open the per-axis kernel scale evidence linked to this operator."
      : "No per-axis kernel scale evidence is linked to this operator.";
    openQuant.addEventListener("click", () => actions.openQuant(op.index));
    actionsRow.append(openOps, openQuant);
  }
  const returnToGraph = button("Return to graph", "secondary-action nv-return-graph");
  returnToGraph.addEventListener("click", actions.returnToGraph);
  actionsRow.append(returnToGraph);

  const links = element("div", "nv-link-ledger");
  const appendLinks = (label, indices, emptyLabel = "none") => {
    const row = element("div", "nv-link-row");
    row.append(element("span", "", label));
    if (!indices.size) row.append(element("em", "", emptyLabel));
    const ordered = [...indices].sort((a, b) => a - b);
    for (const index of ordered.slice(0, 12)) {
      const linked = (analysis.ops || []).find((item) => item.index === index);
      const chip = button(`#${padOp(index)} ${linked?.name || "UNKNOWN"}`, "nv-link-chip");
      chip.addEventListener("click", () => actions.select(index));
      row.append(chip);
    }
    if (ordered.length > 12) {
      row.append(element("em", "", `+${formatNumber(ordered.length - 12)} more`));
    }
    links.append(row);
  };
  const externalInputSet = new Set((analysis.input_tensor_indices || []).map(Number));
  const externalOutputSet = new Set((analysis.output_tensor_indices || []).map(Number));
  const interfaceLabel = (kind, ids) => ids.length
    ? ids.map((id) => `model ${kind} T${id}`).join(", ")
    : "none";
  const externalInputs = (op.inputs || []).map(Number).filter((id) => externalInputSet.has(id));
  const externalOutputs = (op.outputs || []).map(Number).filter((id) => externalOutputSet.has(id));
  appendLinks("Inputs from", related.upstream, interfaceLabel("input", externalInputs));
  appendLinks("Outputs to", related.downstream, interfaceLabel("output", externalOutputs));
  if (mode === "redesign" && projected) {
    appendLinks("Auto-adjusted", new Set(projected.related_op_indices || []));
  }

  const tensorDetails = document.createElement("details");
  tensorDetails.className = "nv-tensor-contracts";
  tensorDetails.append(element("summary", "", "Tensor contracts"));
  const tensors = element("div", "nv-tensor-ledger");
  for (const [label, ids] of [["Inputs", op.inputs || []], ["Outputs", op.outputs || []]]) {
    const row = element("div");
    row.append(element("strong", "", label));
    for (const id of ids.filter((value) => value >= 0)) {
      const tensor = analysis.tensors?.[id];
      row.append(element(
        "span",
        "",
        `T${id} ${tensor?.name || ""} ${(tensor?.shape || []).join("x") || "shape unavailable"} ${tensor?.dtype || ""}`,
      ));
    }
    tensors.append(row);
  }
  tensorDetails.append(tensors);

  const evidenceDetails = document.createElement("details");
  evidenceDetails.className = "nv-evidence-ledger";
  evidenceDetails.append(element("summary", "", "Operator evidence ledger"));
  const evidenceRows = element("dl", "nv-evidence-rows");
  for (const [label, value] of opDetailRows(op, analysis)) {
    const row = element("div");
    row.append(element("dt", "", label), element("dd", "", value));
    evidenceRows.append(row);
  }
  evidenceDetails.append(evidenceRows);
  root.append(head, actionsRow, metrics, links, tensorDetails, evidenceDetails);
}

function createMinimap(graph, layout, state, onNavigate) {
  const root = element("div", "nv-minimap");
  root.title = "Graph overview; select or drag the viewport window";
  const svg = svgElement("svg", {
    class: "nv-minimap-graph",
    viewBox: `${layout.bounds.x} ${layout.bounds.y} ${layout.bounds.width} ${layout.bounds.height}`,
    preserveAspectRatio: "none",
    role: "img",
    "aria-label": "Graph overview",
  });
  const edgeLayer = svgElement("g", { class: "nv-minimap-edges" });
  for (const edge of graph.edges) {
    const from = layout.positions.get(edge.from);
    const to = layout.positions.get(edge.to);
    if (!from || !to) continue;
    edgeLayer.append(svgElement("line", {
      x1: from.x + NODE_WIDTH / 2,
      y1: from.y + NODE_HEIGHT,
      x2: to.x + NODE_WIDTH / 2,
      y2: to.y,
    }));
  }
  const nodeLayer = svgElement("g", { class: "nv-minimap-nodes" });
  for (const placed of layout.nodes) {
    nodeLayer.append(svgElement("rect", {
      x: placed.x,
      y: placed.y,
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      rx: 3,
      class: placed.op.index === state.selectedOpIndex ? "selected" : "",
    }));
  }
  const windowRect = svgElement("rect", { class: "nv-minimap-window" });
  svg.append(edgeLayer, nodeLayer, windowRect);
  root.append(svg);

  const sync = () => {
    const [x, y, width, height] = state.viewBox;
    const left = Math.max(layout.bounds.x, x);
    const top = Math.max(layout.bounds.y, y);
    const right = Math.min(layout.bounds.x + layout.bounds.width, x + width);
    const bottom = Math.min(layout.bounds.y + layout.bounds.height, y + height);
    windowRect.setAttribute("x", String(left));
    windowRect.setAttribute("y", String(top));
    windowRect.setAttribute("width", String(Math.max(1, right - left)));
    windowRect.setAttribute("height", String(Math.max(1, bottom - top)));
  };
  const moveToPointer = (event) => {
    const rect = svg.getBoundingClientRect();
    const px = Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width)));
    const py = Math.max(0, Math.min(1, (event.clientY - rect.top) / Math.max(1, rect.height)));
    const centerX = layout.bounds.x + layout.bounds.width * px;
    const centerY = layout.bounds.y + layout.bounds.height * py;
    onNavigate([
      centerX - state.viewBox[2] / 2,
      centerY - state.viewBox[3] / 2,
      state.viewBox[2],
      state.viewBox[3],
    ]);
  };
  let dragging = false;
  svg.addEventListener("pointerdown", (event) => {
    dragging = true;
    svg.setPointerCapture(event.pointerId);
    moveToPointer(event);
  });
  svg.addEventListener("pointermove", (event) => {
    if (dragging) moveToPointer(event);
  });
  for (const name of ["pointerup", "pointercancel"]) {
    svg.addEventListener(name, () => { dragging = false; });
  }
  sync();
  return { root, sync };
}

function renderSvg(root, analysis, graph, layout, state, actions) {
  root.replaceChildren();
  const svg = svgElement("svg", {
    class: "nv-graph",
    role: "img",
    "aria-label": "Operator and tensor node graph",
    tabindex: "0",
    viewBox: state.viewBox.join(" "),
    preserveAspectRatio: "xMidYMid meet",
  });
  const defs = svgElement("defs");
  const marker = svgElement("marker", {
    id: "nv-arrow",
    viewBox: "0 0 10 10",
    refX: 9,
    refY: 5,
    markerWidth: 6,
    markerHeight: 6,
    orient: "auto-start-reverse",
  });
  marker.append(svgElement("path", { d: "M 0 0 L 10 5 L 0 10 z", class: "nv-arrow" }));
  defs.append(marker);
  svg.append(defs);
  const selectedRelated = relatedOps(graph, state.selectedOpIndex);

  const interfaceLayer = svgElement("g", { class: "nv-interface-layer" });
  const producerByTensor = new Map();
  const consumersByTensor = new Map();
  for (const op of analysis.ops || []) {
    for (const tensorId of op.outputs || []) producerByTensor.set(Number(tensorId), op.index);
    for (const tensorId of op.inputs || []) {
      const id = Number(tensorId);
      if (id < 0) continue;
      if (!consumersByTensor.has(id)) consumersByTensor.set(id, []);
      consumersByTensor.get(id).push(op.index);
    }
  }
  const inputIds = (analysis.input_tensor_indices || []).map(Number).filter((id) => id >= 0);
  const outputIds = (analysis.output_tensor_indices || []).map(Number).filter((id) => id >= 0);
  const inputPortsByOp = new Map((analysis.ops || []).map((op) => [Number(op.index), new Set()]));
  const outputPortsByOp = new Map((analysis.ops || []).map((op) => [Number(op.index), new Set()]));
  for (const edge of graph.edges) {
    inputPortsByOp.get(Number(edge.to))?.add(Number(edge.tensorId));
    outputPortsByOp.get(Number(edge.from))?.add(Number(edge.tensorId));
  }
  for (const tensorId of inputIds) {
    for (const opIndex of consumersByTensor.get(tensorId) || []) inputPortsByOp.get(Number(opIndex))?.add(tensorId);
  }
  for (const tensorId of outputIds) {
    const opIndex = producerByTensor.get(tensorId);
    outputPortsByOp.get(Number(opIndex))?.add(tensorId);
  }
  const portIds = (opIndex, direction) => [
    ...((direction === "input" ? inputPortsByOp : outputPortsByOp).get(Number(opIndex)) || []),
  ].sort((a, b) => a - b);
  const appendInterface = (tensorId, ordinal, total, kind) => {
    const tensor = analysis.tensors?.[tensorId];
    const x = layout.bounds.x + layout.bounds.width * (ordinal + 1) / (total + 1) - INTERFACE_WIDTH / 2;
    const y = kind === "input" ? 18 : layout.bounds.height - INTERFACE_HEIGHT - 18;
    const centerX = x + INTERFACE_WIDTH / 2;
    const endpoints = kind === "input"
      ? (consumersByTensor.get(tensorId) || []).map((index) => layout.positions.get(index)).filter(Boolean)
      : [layout.positions.get(producerByTensor.get(tensorId))].filter(Boolean);
    for (const endpoint of endpoints) {
      const targetX = endpoint.x + tensorPortX(
        endpoint.op,
        kind === "input" ? "input" : "output",
        tensorId,
        portIds(endpoint.op.index, kind === "input" ? "input" : "output"),
      );
      const fromY = kind === "input" ? y + INTERFACE_HEIGHT : endpoint.y + NODE_HEIGHT;
      const toY = kind === "input" ? endpoint.y : y;
      const curve = Math.max(20, Math.abs(toY - fromY) / 2);
      interfaceLayer.append(svgElement("path", {
        class: "nv-interface-edge",
        d: kind === "input"
          ? `M ${centerX} ${fromY} C ${centerX} ${fromY + curve}, ${targetX} ${toY - curve}, ${targetX} ${toY}`
          : `M ${targetX} ${fromY} C ${targetX} ${fromY + curve}, ${centerX} ${toY - curve}, ${centerX} ${toY}`,
        "marker-end": "url(#nv-arrow)",
      }));
    }
    const group = svgElement("g", { class: `nv-interface-node nv-interface-${kind}`, transform: `translate(${x} ${y})` });
    group.append(svgElement("rect", { width: INTERFACE_WIDTH, height: INTERFACE_HEIGHT, rx: 4 }));
    const title = svgElement("text", { x: 9, y: 15, class: "nv-interface-title" });
    title.textContent = `${kind.toUpperCase()}  T${tensorId}`;
    const detail = svgElement("text", { x: 9, y: 30, class: "nv-interface-meta" });
    detail.textContent = `${Array.isArray(tensor?.shape) ? tensor.shape.join("x") : "shape ?"}  ${tensor?.dtype || "dtype ?"}`;
    group.append(title, detail);
    interfaceLayer.append(group);
  };
  inputIds.forEach((id, index) => appendInterface(id, index, inputIds.length, "input"));
  outputIds.forEach((id, index) => appendInterface(id, index, outputIds.length, "output"));
  svg.append(interfaceLayer);

  const edgeLayer = svgElement("g", { class: "nv-edge-layer" });
  for (const edge of graph.edges) {
    const from = layout.positions.get(edge.from);
    const to = layout.positions.get(edge.to);
    if (!from || !to) continue;
    const x1 = from.x + tensorPortX(from.op, "output", edge.tensorId, portIds(from.op.index, "output"));
    const y1 = from.y + NODE_HEIGHT;
    const x2 = to.x + tensorPortX(to.op, "input", edge.tensorId, portIds(to.op.index, "input"));
    const y2 = to.y;
    const curve = Math.max(24, Math.abs(y2 - y1) * 0.5);
    const selected = edge.from === state.selectedOpIndex || edge.to === state.selectedOpIndex;
    const withinFocus = !state.focusedIndices
      || (state.focusedIndices.has(edge.from) && state.focusedIndices.has(edge.to));
    const path = svgElement("path", {
      d: `M ${x1} ${y1} C ${x1} ${y1 + curve}, ${x2} ${y2 - curve}, ${x2} ${y2}`,
      class: `${edgeTone(analysis, edge, state.overlay, state.projection, state.projectionIndex)}${selected ? " selected" : ""}${withinFocus ? "" : " focus-hidden"}`,
      "marker-end": "url(#nv-arrow)",
      "data-from-op": edge.from,
      "data-to-op": edge.to,
      "data-tensor-index": edge.tensorId,
    });
    const title = svgElement("title");
    const tensor = analysis.tensors?.[edge.tensorId];
    title.textContent = `T${edge.tensorId} ${tensor?.name || ""} / ${(tensor?.shape || []).join("x") || "shape unavailable"} / ${edge.dtype || "dtype unavailable"} / ${formatBytes(edge.bytes || 0)}`;
    path.append(title);
    edgeLayer.append(path);
    const shape = Array.isArray(tensor?.shape) && tensor.shape.length ? tensor.shape.join("x") : "?";
    const labelText = `T${edge.tensorId}  ${shape}  ${edge.dtype || "?"}`;
    const labelWidth = Math.min(180, Math.max(64, labelText.length * 5.4 + 12));
    const labelX = (x1 + x2) / 2;
    const labelY = (y1 + y2) / 2;
    const label = svgElement("g", {
      class: `nv-edge-label${selected ? " selected" : ""}${withinFocus ? "" : " focus-hidden"}`,
      transform: `translate(${labelX} ${labelY})`,
    });
    label.append(
      svgElement("rect", { x: -labelWidth / 2, y: -9, width: labelWidth, height: 18, rx: 3 }),
      svgElement("text", { x: 0, y: 3, "text-anchor": "middle" }),
    );
    label.lastChild.textContent = labelText;
    label.append(title.cloneNode(true));
    edgeLayer.append(label);
  }
  svg.append(edgeLayer);

  const term = state.search.trim().toLowerCase();
  const nodeLayer = svgElement("g", { class: "nv-node-layer" });
  for (const placed of layout.nodes) {
    const op = placed.op;
    const projected = projectionRow(state.projection, op.index, state.projectionIndex);
    const contractState = state.overlay === "redesign"
      ? projectionContractState(analysis, state.projection, op, state.projectionIndex)
      : null;
    const projectedShape = projected ? contractShape(projected.projected_outputs) : "";
    const searchable = `#${op.index} ${op.name || ""} ${outputShape(op)} ${projectedShape} ${providerLabel(analysis, op)}`.toLowerCase();
    const searchMatch = !term || searchable.includes(term);
    const connected = selectedRelated.upstream.has(op.index) || selectedRelated.downstream.has(op.index);
    const withinFocus = !state.focusedIndices || state.focusedIndices.has(op.index);
    const group = svgElement("g", {
      class: [
        "nv-node",
        nodeTone(analysis, op, state.overlay, state.projection, state.projectionIndex),
        contractState?.className || "",
        op.index === state.selectedOpIndex ? "selected" : "",
        connected ? "connected" : "",
        searchMatch ? "" : "search-dim",
        withinFocus ? "" : "focus-hidden",
      ].filter(Boolean).join(" "),
      transform: `translate(${placed.x} ${placed.y})`,
      tabindex: "0",
      role: "button",
      "aria-label": `Operator ${op.index} ${op.name}`,
      "data-op-index": op.index,
      "data-contract-state": contractState?.id || "",
      "data-contract-reasons": contractState?.reasons?.join(" / ") || "",
    });
    group.append(
      svgElement("rect", { x: 0, y: 0, width: NODE_WIDTH, height: NODE_HEIGHT, rx: 5 }),
      svgElement("line", { x1: 1, y1: 43, x2: NODE_WIDTH - 1, y2: 43, class: "nv-node-divider" }),
    );
    for (const tensorId of portIds(op.index, "input")) {
      const port = svgElement("circle", {
        cx: tensorPortX(op, "input", tensorId, portIds(op.index, "input")), cy: 0, r: 4, class: "nv-port nv-port-input",
      });
      const portTitle = svgElement("title");
      portTitle.textContent = `Input T${tensorId}`;
      port.append(portTitle);
      group.append(port);
    }
    for (const tensorId of portIds(op.index, "output")) {
      const port = svgElement("circle", {
        cx: tensorPortX(op, "output", tensorId, portIds(op.index, "output")), cy: NODE_HEIGHT, r: 4, class: "nv-port nv-port-output",
      });
      const portTitle = svgElement("title");
      portTitle.textContent = `Output T${tensorId}`;
      port.append(portTitle);
      group.append(port);
    }
    const indexText = svgElement("text", { x: 10, y: 17, class: "nv-node-index" });
    indexText.textContent = `#${padOp(op.index)}`;
    const nameText = svgElement("text", { x: 10, y: 35, class: "nv-node-name" });
    nameText.textContent = String(op.name || "UNKNOWN").slice(0, 30);
    const metaText = svgElement("text", { x: 10, y: 61, class: "nv-node-meta" });
    metaText.textContent = state.overlay === "redesign" && projected
      ? `${contractShape(projected.source_outputs).slice(0, 12)} -> ${contractShape(projected.projected_outputs).slice(0, 12)}`
      : `${outputShape(op).slice(0, 24)} / ${dtypeSummary(analysis, op, "output").slice(0, 14)}`;
    const evidenceText = svgElement("text", { x: 10, y: 78, class: "nv-node-evidence" });
    evidenceText.textContent = state.overlay === "redesign" && projected
      ? `${String(projected.change_class || "unchanged").replaceAll("_", " ")} / ${formatUs(projected.projected_steady_us ?? null)}`.slice(0, 42)
      : nodeEvidenceLabel(analysis, op, state.overlay).slice(0, 42);
    group.append(indexText, nameText, metaText, evidenceText);
    if (contractState) {
      const statusText = svgElement("text", {
        x: NODE_WIDTH - 10,
        y: 17,
        class: `nv-contract-state-label nv-contract-label-${contractState.id}`,
        "text-anchor": "end",
      });
      statusText.textContent = {
        blocked: "BLOCKED",
        issue: "ISSUE",
        watch: "WATCH",
        conditional: "CONDITIONAL",
        unassessed: "N/A",
        satisfied: "OK",
      }[contractState.id] || contractState.label.toUpperCase();
      group.append(statusText);
    }
    const title = svgElement("title");
    title.textContent = state.overlay === "redesign" && projected
      ? `${op.name} / change path: ${String(projected.change_class || "unchanged").replaceAll("_", " ")} / contract: ${contractState?.label || "Not assessed"}${contractState?.reasons?.length ? ` (${contractState.reasons.join("; ")})` : ""} / shape ${contractShape(projected.source_outputs)} -> ${contractShape(projected.projected_outputs)} / steady ${formatUs(projected.source_steady_us ?? null)} -> ${formatUs(projected.projected_steady_us ?? null)}`
      : `${op.name} / ${providerLabel(analysis, op)} / ${quantLabel(op)} / steady ${formatUs(opSteadyStateUs(op))}`;
    group.append(title);
    const select = () => actions.select(op.index);
    group.addEventListener("click", select);
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select();
      }
    });
    nodeLayer.append(group);
  }
  svg.append(nodeLayer);

  let minimap = null;
  const updateView = (nextViewBox) => {
    state.viewBox = clampViewBox(nextViewBox, layout);
    svg.setAttribute("viewBox", state.viewBox.join(" "));
    const percent = zoomPercent(state.viewBox, layout);
    svg.classList.toggle("overview-scale", percent <= 125);
    const readout = root.closest(".nv-shell")?.querySelector(".nv-zoom-level");
    if (readout) readout.textContent = `${percent}%`;
    minimap?.sync();
  };
  minimap = createMinimap(graph, layout, state, updateView);
  root.append(svg, minimap.root);
  updateView(state.viewBox);

  let drag = null;
  svg.addEventListener("pointerdown", (event) => {
    if (event.target.closest?.(".nv-node")) return;
    svg.setPointerCapture(event.pointerId);
    drag = { x: event.clientX, y: event.clientY, viewBox: [...state.viewBox] };
    svg.classList.add("dragging");
  });
  svg.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const bounds = svg.getBoundingClientRect();
    const dx = (event.clientX - drag.x) / Math.max(1, bounds.width) * drag.viewBox[2];
    const dy = (event.clientY - drag.y) / Math.max(1, bounds.height) * drag.viewBox[3];
    updateView([drag.viewBox[0] - dx, drag.viewBox[1] - dy, drag.viewBox[2], drag.viewBox[3]]);
  });
  const endDrag = () => {
    drag = null;
    svg.classList.remove("dragging");
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const bounds = svg.getBoundingClientRect();
    if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY) * 1.25) {
      const dx = (event.deltaX || event.deltaY) / Math.max(1, bounds.width) * state.viewBox[2];
      const dy = event.shiftKey ? 0 : event.deltaY / Math.max(1, bounds.height) * state.viewBox[3];
      updateView([state.viewBox[0] + dx, state.viewBox[1] + dy, state.viewBox[2], state.viewBox[3]]);
      return;
    }
    const px = Math.max(0, Math.min(1, (event.clientX - bounds.left) / Math.max(1, bounds.width)));
    const py = Math.max(0, Math.min(1, (event.clientY - bounds.top) / Math.max(1, bounds.height)));
    const anchorX = state.viewBox[0] + state.viewBox[2] * px;
    const anchorY = state.viewBox[1] + state.viewBox[3] * py;
    const factor = Math.exp(Math.max(-360, Math.min(360, event.deltaY)) * 0.00135);
    updateView(zoomViewBoxAt(state.viewBox, factor, anchorX, anchorY, layout));
  }, { passive: false });
  svg.addEventListener("dblclick", (event) => {
    if (event.target.closest?.(".nv-node")) return;
    updateView([layout.bounds.x, layout.bounds.y, layout.bounds.width, layout.bounds.height]);
  });
  svg.addEventListener("keydown", (event) => {
    const selected = layout.positions.get(Number(state.selectedOpIndex));
    const anchorX = selected ? selected.x + NODE_WIDTH / 2 : state.viewBox[0] + state.viewBox[2] / 2;
    const anchorY = selected ? selected.y + NODE_HEIGHT / 2 : state.viewBox[1] + state.viewBox[3] / 2;
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      updateView(zoomViewBoxAt(state.viewBox, 0.8, anchorX, anchorY, layout));
    } else if (event.key === "-") {
      event.preventDefault();
      updateView(zoomViewBoxAt(state.viewBox, 1.2, anchorX, anchorY, layout));
    } else if (event.key === "0") {
      event.preventDefault();
      updateView([layout.bounds.x, layout.bounds.y, layout.bounds.width, layout.bounds.height]);
    }
  });
}

export function createNodeViewController({
  root,
  mode = "audit",
  onSelectOp = () => {},
  onOpenOps = () => {},
  onOpenQuant = () => {},
  canOpenQuant = () => false,
  onEditNode = () => {},
} = {}) {
  let mountedRoot = root;
  const state = {
    analysis: null,
    selectedOpIndex: 0,
    overlay: "structure",
    viewScope: "full",
    search: "",
    viewBox: [0, 0, 1000, 600],
    graph: null,
    layout: null,
    projection: null,
    projectionIndex: { ops: new Map(), edges: new Map() },
    focusedIndices: null,
    inspectorOpen: true,
    expanded: false,
  };

  function fit() {
    if (!state.layout) return;
    state.viewBox = clampViewBox([
      state.layout.bounds.x,
      state.layout.bounds.y,
      Math.max(320, state.layout.bounds.width),
      Math.max(240, state.layout.bounds.height),
    ], state.layout);
  }

  function frameTop() {
    if (!state.layout) return;
    const width = Math.max((mountedRoot?.clientWidth || 0) < 600 ? 500 : 720, state.layout.bounds.width);
    const height = Math.min(state.layout.bounds.height, Math.max(560, Math.min(760, width * 0.78)));
    state.viewBox = clampViewBox([
      state.layout.bounds.x + (state.layout.bounds.width - width) / 2,
      state.layout.bounds.y,
      width,
      height,
    ], state.layout);
  }

  function focusIndices(indices) {
    if (!state.layout) return;
    const placed = [...new Set(indices)]
      .map((index) => state.layout.positions.get(Number(index)))
      .filter(Boolean);
    if (!placed.length) {
      fit();
      return;
    }
    const padding = 56;
    const minX = Math.min(...placed.map((item) => item.x));
    const minY = Math.min(...placed.map((item) => item.y));
    const maxX = Math.max(...placed.map((item) => item.x + NODE_WIDTH));
    const maxY = Math.max(...placed.map((item) => item.y + NODE_HEIGHT));
    const width = Math.max(520, maxX - minX + padding * 2);
    const height = Math.max(300, maxY - minY + padding * 2);
    state.viewBox = clampViewBox([
      (minX + maxX) / 2 - width / 2,
      (minY + maxY) / 2 - height / 2,
      width,
      height,
    ], state.layout);
  }

  function frameSelection() {
    if (!state.graph) return;
    const related = relatedOps(state.graph, Number(state.selectedOpIndex));
    focusIndices([state.selectedOpIndex, ...related.upstream, ...related.downstream]);
  }

  function ensureSelectedVisible() {
    const placed = state.layout?.positions?.get(Number(state.selectedOpIndex));
    if (!placed) return;
    const [x, y, width, height] = state.viewBox;
    const paddingX = Math.min(64, width * 0.12);
    const paddingY = Math.min(64, height * 0.12);
    const visible = placed.x >= x + paddingX
      && placed.x + NODE_WIDTH <= x + width - paddingX
      && placed.y >= y + paddingY
      && placed.y + NODE_HEIGHT <= y + height - paddingY;
    if (visible) return;
    state.viewBox = clampViewBox([
      placed.x + NODE_WIDTH / 2 - width / 2,
      placed.y + NODE_HEIGHT / 2 - height / 2,
      width,
      height,
    ], state.layout);
  }

  function selectionIndices() {
    const selected = Number(state.selectedOpIndex);
    const indices = new Set([selected]);
    let frontier = new Set([selected]);
    for (let depth = 0; depth < 2; depth += 1) {
      const next = new Set();
      for (const edge of state.graph?.edges || []) {
        if (frontier.has(edge.from) && !indices.has(edge.to)) next.add(edge.to);
        if (frontier.has(edge.to) && !indices.has(edge.from)) next.add(edge.from);
      }
      for (const index of next) indices.add(index);
      frontier = next;
    }
    const row = projectionRow(state.projection, selected, state.projectionIndex);
    for (const index of row?.propagation_source_op_indices || []) indices.add(Number(index));
    return indices;
  }

  function changedIndices() {
    const changed = (state.projection?.op_projections || [])
      .filter((row) => row.change_class && row.change_class !== "unchanged")
      .map((row) => Number(row.op_index));
    if (changed.length) return new Set(changed);
    const flagged = (state.analysis?.ops || [])
      .filter((op) => {
        const contract = projectionContractState(
          state.analysis,
          state.projection,
          op,
          state.projectionIndex,
        );
        return ["blocked", "issue", "watch"].includes(contract.id);
      })
      .map((op) => Number(op.index));
    return new Set(flagged.length ? flagged : [state.selectedOpIndex]);
  }

  function applyViewScope(scope = state.viewScope) {
    state.viewScope = scope;
    if (scope === "selection") {
      state.focusedIndices = selectionIndices();
      focusIndices(state.focusedIndices);
    } else if (scope === "changes") {
      state.focusedIndices = changedIndices();
      focusIndices(state.focusedIndices);
    } else {
      state.focusedIndices = null;
      if (mode === "redesign") fit();
      else frameTop();
    }
  }

  function zoom(factor) {
    const selected = state.layout?.positions?.get(Number(state.selectedOpIndex));
    const anchorSelection = state.viewScope === "selection" && selected;
    const anchorX = anchorSelection ? selected.x + NODE_WIDTH / 2 : state.viewBox[0] + state.viewBox[2] / 2;
    const anchorY = anchorSelection ? selected.y + NODE_HEIGHT / 2 : state.viewBox[1] + state.viewBox[3] / 2;
    state.viewBox = zoomViewBoxAt(state.viewBox, factor, anchorX, anchorY, state.layout);
  }

  function selectOp(index, { notify = false } = {}) {
    const numeric = Number(index);
    if (!state.analysis?.ops?.some((op) => op.index === numeric)) return;
    state.selectedOpIndex = numeric;
    if (state.viewScope === "selection") applyViewScope("selection");
    else if (notify) ensureSelectedVisible();
    if (notify) onSelectOp(numeric);
    render();
    if (notify && window.matchMedia?.("(max-width: 680px)").matches) {
      requestAnimationFrame(() => mountedRoot?.querySelector(".nv-detail")?.scrollIntoView({ behavior: "smooth", block: "start" }));
    }
  }

  function setAnalysis(analysis) {
    if (state.analysis === analysis && state.graph && state.layout) {
      render();
      return;
    }
    state.analysis = analysis;
    const graphIndex = buildGraphIndex(analysis);
    state.graph = collectFullGraph(analysis, graphIndex);
    state.layout = layoutFullGraph(state.graph.nodes, state.graph.edges, { minimumColumns: interfaceColumnCount(analysis) });
    if (!analysis?.ops?.some((op) => op.index === state.selectedOpIndex)) {
      state.selectedOpIndex = analysis?.ops?.[0]?.index ?? 0;
    }
    applyViewScope();
    render();
  }

  function setProjection(projection) {
    state.projection = projection || null;
    state.projectionIndex = indexProjection(state.projection);
    if (mode === "redesign") state.overlay = "redesign";
    if (mode === "redesign" && state.viewScope !== "full") applyViewScope();
    render();
  }

  function resetInteractionState() {
    state.analysis = null;
    state.selectedOpIndex = 0;
    state.overlay = "structure";
    state.viewScope = "full";
    state.search = "";
    state.viewBox = [0, 0, 1000, 600];
    state.graph = null;
    state.layout = null;
    state.projection = null;
    state.projectionIndex = { ops: new Map(), edges: new Map() };
    state.focusedIndices = null;
    state.inspectorOpen = true;
    state.expanded = false;
    mountedRoot?.replaceChildren();
  }

  function mount(nextRoot) {
    mountedRoot = nextRoot;
    render();
  }

  function sync({
    root: nextRoot,
    analysis,
    projection = null,
    selectedOpIndex = null,
  }) {
    mountedRoot = nextRoot;
    let graphChanged = false;
    if (state.analysis !== analysis || !state.graph || !state.layout) {
      state.analysis = analysis;
      const graphIndex = buildGraphIndex(analysis);
      state.graph = collectFullGraph(analysis, graphIndex);
      state.layout = layoutFullGraph(state.graph.nodes, state.graph.edges, { minimumColumns: interfaceColumnCount(analysis) });
      graphChanged = true;
    }
    state.projection = projection;
    state.projectionIndex = indexProjection(projection);
    if (mode === "redesign") state.overlay = "redesign";
    const numeric = Number(selectedOpIndex);
    const selectionChanged = Number.isFinite(numeric) && numeric !== state.selectedOpIndex;
    if (Number.isFinite(numeric) && analysis?.ops?.some((op) => op.index === numeric)) {
      state.selectedOpIndex = numeric;
    } else if (!analysis?.ops?.some((op) => op.index === state.selectedOpIndex)) {
      state.selectedOpIndex = analysis?.ops?.[0]?.index ?? 0;
    }
    if (graphChanged || (mode === "redesign" && state.viewScope !== "full")) {
      applyViewScope();
    } else if (selectionChanged) {
      ensureSelectedVisible();
    }
    render();
  }

  function render() {
    if (!mountedRoot || !state.analysis || !state.graph || !state.layout) return;
    const shell = element("div", `nv-shell${state.inspectorOpen ? "" : " inspector-collapsed"}${state.expanded ? " expanded" : ""}`);
    const toolbar = element("div", "nv-toolbar");
    const search = document.createElement("input");
    search.type = "search";
    search.value = state.search;
    search.placeholder = "Find op, index, shape, provider";
    search.setAttribute("aria-label", "Search graph nodes");
    search.addEventListener("input", () => {
      state.search = search.value;
      renderSvg(viewport, state.analysis, state.graph, state.layout, state, actions);
    });
    search.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || !search.value.trim()) return;
      const term = search.value.trim().toLowerCase();
      const match = (state.analysis?.ops || []).find((op) => {
        const row = projectionRow(state.projection, op.index, state.projectionIndex);
        return `#${op.index} ${op.name || ""} ${outputShape(op)} ${row ? contractShape(row.projected_outputs) : ""} ${providerLabel(state.analysis, op)}`
          .toLowerCase()
          .includes(term);
      });
      if (match) selectOp(match.index, { notify: true });
    });
    const overlays = element("div", "nv-segments");
    const overlayOptions = mode === "redesign" ? [
      ["redesign", "Contracts + impact"],
      ["structure", "Structure"],
      ["latency", "Source steady"],
    ] : [
      ["structure", "Structure"],
      ["delegation", "Delegation"],
      ["quant", "Quantization"],
      ["latency", "Steady time"],
    ];
    for (const [value, label] of overlayOptions) {
      const control = button(label, state.overlay === value ? "active" : "");
      control.setAttribute("aria-pressed", String(state.overlay === value));
      control.addEventListener("click", () => {
        state.overlay = value;
        render();
      });
      overlays.append(control);
    }
    const viewControls = element("div", "nv-view-controls");
    if (mode === "redesign") {
      for (const [value, label] of [
        ["selection", "Selection"],
        ["changes", "Changes"],
        ["full", "Full"],
      ]) {
        const control = button(label, state.viewScope === value ? "active" : "");
        control.setAttribute("aria-pressed", String(state.viewScope === value));
        control.title = {
          selection: "Fit the selected layer and directly connected layers",
          changes: "Fit every directly edited or automatically propagated layer",
          full: "Fit the complete graph",
        }[value];
        control.addEventListener("click", () => {
          applyViewScope(value);
          render();
        });
        control.dataset.viewScope = value;
        viewControls.append(control);
      }
    } else {
      const topButton = button("Start", "secondary-action");
      topButton.title = "Return to the beginning of the top-down graph";
      topButton.addEventListener("click", () => {
        frameTop();
        renderSvg(viewport, state.analysis, state.graph, state.layout, state, actions);
      });
      const fitButton = button("Overview", "secondary-action");
      fitButton.title = "Fit the complete graph in the viewport";
      fitButton.addEventListener("click", () => {
        fit();
        renderSvg(viewport, state.analysis, state.graph, state.layout, state, actions);
      });
      const selectedButton = button("Selected", "secondary-action");
      selectedButton.title = "Frame the selected operator and its direct tensor neighbors";
      selectedButton.addEventListener("click", () => {
        frameSelection();
        renderSvg(viewport, state.analysis, state.graph, state.layout, state, actions);
      });
      viewControls.append(topButton, fitButton, selectedButton);
    }
    const inspectorButton = button("Inspector", state.inspectorOpen ? "active" : "");
    inspectorButton.title = "Show or hide the operator evidence inspector";
    inspectorButton.setAttribute("aria-pressed", String(state.inspectorOpen));
    inspectorButton.addEventListener("click", () => {
      state.inspectorOpen = !state.inspectorOpen;
      render();
    });
    const expandButton = button(state.expanded ? "Exit" : "Expand", "secondary-action");
    expandButton.title = state.expanded ? "Return the graph to the Explorer" : "Expand the graph workspace";
    expandButton.addEventListener("click", () => {
      state.expanded = !state.expanded;
      render();
    });
    viewControls.append(inspectorButton, expandButton);
    for (const [label, title, factor] of [["-", "Zoom out", 1.2]]) {
      const control = button(label, "nv-zoom-button");
      control.title = title;
      control.setAttribute("aria-label", title);
      control.addEventListener("click", () => {
        zoom(factor);
        renderSvg(viewport, state.analysis, state.graph, state.layout, state, actions);
      });
      viewControls.append(control);
    }
    viewControls.append(element("output", "nv-zoom-level", `${zoomPercent(state.viewBox, state.layout)}%`));
    for (const [label, title, factor] of [["+", "Zoom in", 0.8]]) {
      const control = button(label, "nv-zoom-button");
      control.title = title;
      control.setAttribute("aria-label", title);
      control.addEventListener("click", () => {
        zoom(factor);
        renderSvg(viewport, state.analysis, state.graph, state.layout, state, actions);
      });
      viewControls.append(control);
    }

    const flagControls = element("div", "nv-flag-controls");
    if (mode === "redesign") {
      const flagged = (state.analysis?.ops || []).filter((op) => {
        const contract = projectionContractState(
          state.analysis,
          state.projection,
          op,
          state.projectionIndex,
        );
        return ["blocked", "issue", "watch"].includes(contract.id);
      });
      const selectedFlagIndex = flagged.findIndex((op) => op.index === state.selectedOpIndex);
      const moveFlag = (direction) => {
        if (!flagged.length) return;
        const base = selectedFlagIndex >= 0 ? selectedFlagIndex : direction > 0 ? -1 : 0;
        const next = (base + direction + flagged.length) % flagged.length;
        selectOp(flagged[next].index, { notify: true });
      };
      const previous = button("Previous flag", "secondary-action");
      previous.disabled = !flagged.length;
      previous.addEventListener("click", () => moveFlag(-1));
      const next = button("Next flag", "secondary-action");
      next.disabled = !flagged.length;
      next.addEventListener("click", () => moveFlag(1));
      flagControls.append(previous, next);
    }
    toolbar.append(search, overlays, viewControls, flagControls);

    const heading = element("div", "nv-heading");
    const title = element("div");
    title.append(
      element("span", "nv-kicker", mode === "redesign" ? "NODE REDESIGN" : "MODEL GRAPH"),
      element("h3", "", mode === "redesign" ? "Edit the graph contract in place" : "Operators, tensors, and model flow"),
      element("p", "", mode === "redesign"
        ? "Select a layer, edit its owning semantic block, and inspect every deterministically propagated tensor contract before materializing a new model."
        : "Select a node to inspect its attributes, tensor edges, quantization contract, and deployment evidence."),
    );
    const summary = state.projection?.impact_summary;
    const contractSummary = mode === "redesign"
      ? projectionContractSummary(state.analysis, state.projection, state.projectionIndex)
      : null;
    const changeSummary = summary
      ? `${formatNumber(summary.direct_edit_op_count || 0)} direct / ${formatNumber(summary.propagated_op_count || 0)} auto / ${formatNumber(summary.changed_edge_count || 0)} changed edges`
      : `${formatNumber(state.graph.nodes.length)} nodes / ${formatNumber(state.graph.edges.length)} tensor edges`;
    const conditionSummary = contractSummary
      ? `${formatNumber(contractSummary.issue + contractSummary.blocked)} issue / ${formatNumber(contractSummary.watch)} watch / ${formatNumber(contractSummary.satisfied)} satisfied`
      : "";
    const selectedSummary = state.analysis?.ops?.find((op) => op.index === state.selectedOpIndex);
    const count = element(
      "span",
      "nv-count",
      `${selectedSummary ? `#${padOp(selectedSummary.index)} ${selectedSummary.name} | ` : ""}${conditionSummary ? `${changeSummary} | ${conditionSummary}` : changeSummary}`,
    );
    count.setAttribute("aria-live", "polite");
    heading.append(title, count);

    const layout = element("div", "nv-layout");
    const viewport = element("div", "nv-viewport");
    const detail = element("aside", "nv-detail");
    const actions = {
      select: (index) => selectOp(index, { notify: true }),
      openOps: onOpenOps,
      openQuant: onOpenQuant,
      canOpenQuant,
      editNode: onEditNode,
      returnToGraph: () => viewport.scrollIntoView({ behavior: "smooth", block: "start" }),
    };
    renderSvg(viewport, state.analysis, state.graph, state.layout, state, actions);
    renderDetail(
      detail,
      state.analysis,
      state.graph,
      state.selectedOpIndex,
      actions,
      state.projection,
      state.projectionIndex,
      mode,
    );
    layout.append(viewport, detail);
    shell.append(heading, toolbar, layout);
    mountedRoot.replaceChildren(shell);
  }

  return { setAnalysis, setProjection, mount, sync, render, selectOp, fit, resetInteractionState };
}

function indexProjection(projection) {
  const unresolved = new Set((projection?.constraints || []).filter((item) =>
    item.severity === "error" && item.code?.startsWith("RD-SHAPE-"),
  ).map((item) => Number(/^#(\d+)/.exec(String(item.scope || ""))?.[1])).filter(Number.isFinite));
  return {
    ops: new Map((projection?.op_projections || []).map((row) => [Number(row.op_index), row])),
    edges: new Map((projection?.propagation_edges || []).map((row) => [
      `${row.producer_op_index}:${row.consumer_op_index}:${row.tensor_index}`,
      row,
    ])),
    unresolved,
  };
}
