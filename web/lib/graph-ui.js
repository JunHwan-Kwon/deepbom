import {
  alignmentLabel,
  buildGraphIndex,
  estimateOpBottleneck,
  fusionLabel,
  opDetailRows,
  opInputTensorDetails,
  opKernelLabel,
  opSteadyStateUs,
  opLocalLinks,
  opOutputTensorDetails,
  opPrecisionLabel,
  packingLabel,
  predictedPartitionBoundaryEdgesForOp,
  quantStateLabel,
  quantToneClass,
  xnnpackLabel,
} from "./analysis.js";
import { appendDetailList, detailGrid, evidenceDisclosure, svgEl, td } from "./dom.js";
import { formatExactInteger, formatNumber, formatPercent, formatPercentRange, formatShapes, formatUs, humanizeStageKey } from "./format.js";
import { decodeXnnpReason } from "./reason-codes.js";
import { renderWeightHistogram } from "./weight-hist.js";
import { renderInfluenceCanvas } from "./influence.js";

const GRAPH_NODE_WIDTH = 172;
const GRAPH_NODE_HEIGHT = 60;
const GRAPH_NODE_MID_Y = 30;

export function buildGraphEvidenceMaps(analysis) {
  const opAnnotations = new Map();
  const tensorFanOut = new Map();
  const lowNormStats = new Map();
  const consumers = new Map();
  for (const op of analysis?.ops || []) {
    opAnnotations.set(op.index, {
      role: op.topo_role || "through",
      fanOutMax: op.topo_fan_out_max || 0,
      patterns: new Set(Array.isArray(op.patterns) ? op.patterns : []),
    });
    for (const tensorIndex of op.inputs || []) {
      if (tensorIndex < 0) continue;
      consumers.set(tensorIndex, (consumers.get(tensorIndex) || 0) + 1);
    }
    if (Number.isInteger(op.low_norm_filter_count) && Number.isInteger(op.low_norm_filter_total)) {
      lowNormStats.set(op.index, {
        low_norm: op.low_norm_filter_count,
        total: op.low_norm_filter_total,
      });
    }
  }
  for (const [tensorIndex, count] of consumers) tensorFanOut.set(tensorIndex, count);
  return { opAnnotations, tensorFanOut, lowNormStats };
}

export function graphOpRow(op, { selected = false, onSelect = () => {}, analysis = null, lowNormStat = null } = {}) {
  const tr = document.createElement("tr");
  tr.dataset.opIndex = String(op.index);
  tr.tabIndex = 0;
  tr.setAttribute("role", "button");
  tr.setAttribute("aria-label", `Inspect operator #${op.index} ${op.name}`);
  tr.setAttribute("aria-selected", selected ? "true" : "false");
  if (selected) tr.className = "selected-row";
  tr.addEventListener("click", () => onSelect(op));
  tr.addEventListener("keydown", (event) => {
    if (event.target !== tr || !["Enter", " "].includes(event.key)) return;
    event.preventDefault();
    onSelect(op);
  });

  // Low-norm cell: heuristic ratio for conv-family ops, dash otherwise.
  let lowNormCell;
  if (lowNormStat) {
    const { low_norm: lowNorm, total } = lowNormStat;
    const pct = lowNorm / total;
    lowNormCell = td(`${lowNorm}/${total}`, `op-dead-cell${pct > 0.25 ? " op-dead-high" : pct > 0.1 ? " op-dead-warn" : pct > 0 ? " op-dead-low" : ""}`);
    lowNormCell.title = `${(pct * 100).toFixed(0)}% decoded low-norm filters (L2 < 2% of this layer maximum). This heuristic is distinct from the scale-vector near-zero representable-channel check.`;
  } else {
    lowNormCell = td("—", "muted-text op-dead-cell");
  }

  // Est. µs cell: static per-op latency estimate with dominant-cost breakdown
  const estUs = opSteadyStateUs(op);
  const estCell = td(estUs > 0 ? formatUs(estUs) : "-", `numeric format-tflite${estUs > 0 ? "" : " muted-text"}`);
  if (estUs > 0) {
    const coldUs = Number(op.bottleneck_total_us || 0);
    estCell.title = `Static steady-state estimate — dominant: ${op.bottleneck_dominant || "?"}\n`
      + `compute ${formatUs(op.bottleneck_compute_us || 0)} · memory ${formatUs(op.bottleneck_memory_us || 0)}`
      + `${op.bottleneck_packing_us > 0 ? ` · cold ${formatUs(coldUs)} including one-time packing ${formatUs(op.bottleneck_packing_us)}` : ""}`
      + `${op.bottleneck_break_us > 0 ? ` · cold setup ${formatUs(op.bottleneck_break_us)}` : ""}`
      + `${op.bottleneck_fallback_us > 0 ? ` · fallback ${formatUs(op.bottleneck_fallback_us)}` : ""}`;
  }

  tr.append(
    td(`#${String(op.index).padStart(3, "0")}`),
    td(op.name),
    td(formatShapes(op.output_shapes), "wrap"),
    td(opKernelLabel(op, analysis), "muted-text"),
    td(formatExactInteger(op.macs_decimal, op.macs, "N/A"), `numeric${op.macs_decimal == null && op.macs == null ? " muted-text" : ""}`),
    estCell,
    td(op.static_bound_guess),
    td(xnnpackLabel(op, analysis), `wrap format-tflite ${op.xnnpack_chain_break ? "risk-text" : op.xnnpack_supported ? "good-text" : "muted-text"}`),
    td(quantStateLabel(op), `wrap ${quantToneClass(op)}`),
    td(alignmentLabel(op), `wrap graph-optional-col format-tflite ${op.channel_alignment_status === "misaligned" ? "warn-text" : "muted-text"}`),
    td(packingLabel(op), `wrap graph-optional-col format-tflite ${op.weight_packing_risk === "warn" ? "warn-text" : "muted-text"}`),
    td(fusionLabel(op), "wrap graph-optional-col"),
    lowNormCell,
  );
  return tr;
}

// Build prioritized issue list for the insight callout at the top of op detail.
function buildOpInsights(op, b, analysis) {
  const items = [];
  const isTflite = String(analysis?.format || "tflite").toLowerCase() !== "onnx";

  if (isTflite && op.xnnpack_chain_break) {
    items.push({ tone: "risk", text: `Predicted XNNPACK partition boundary - ${op.xnnpack_break_class || "delegation interrupted"}`, detail: "May add delegate/interface transition overhead. Tensor copy materialization requires runtime evidence." });
  } else if (isTflite && op.xnnpack_chain_id < 0 && op.xnnpack_supported === false) {
    items.push({ tone: "warn", text: "Predicted outside XNNPACK partition", detail: "Static rulepack prediction only; confirm provider assignment with runtime profiling." });
  }

  if (op.quant_hole) {
    items.push({ tone: "risk", text: "Quantization hole — FP32 island in INT8 graph", detail: "Forces FP32 execution inside an otherwise INT8 graph, breaking the delegation chain." });
  } else if (op.quant_risk === "risk") {
    items.push({ tone: "risk", text: `Quant risk: ${op.quant_risk_detail || "scale/zero-point anomaly"}`, detail: "Quantization parameter issue that may degrade accuracy or prevent delegation." });
  }

  if (isTflite && op.channel_alignment_status === "misaligned") {
    const tailPct = ((op.channel_tail_overhead_percent || 0) * 100).toFixed(0);
    const multiples = (op.xnnpack_kernel_alignment_multiples || []).length ? op.xnnpack_kernel_alignment_multiples.map((value) => `x${value}`).join("/") : `x${op.channel_alignment_multiple || "unknown"}`;
    items.push({ tone: "warn", text: `Channel count not tile-aligned - up to ${tailPct}% modeled inactive lanes`, detail: `Channel count is not aligned across the applicable candidate multiples (${multiples}). Runtime kernel selection remains unmeasured.` });
  }

  if (isTflite && op.weight_packing_risk === "warn") {
    items.push({ tone: "warn", text: "Modeled weight-packing warmup", detail: "Packing is generally a compile/first-run cost; cache reuse and repacking frequency depend on the runtime and session lifecycle." });
  }

  if (isTflite && b && b.dominantLabel === "fallback traffic") {
    if (items.length === 0) {
      items.push({ tone: "warn", text: `Bottleneck: ${b.dominantLabel}`, detail: "Dominant cost is delegation overhead, not compute or memory bandwidth." });
    }
  }

  return items.slice(0, 3);
}

export function renderOpDetailPanel(container, analysis, opIndex, { weightHistograms = null, influence = null, outputInfluence = null, runtimeAssignment = null } = {}) {
  const op = analysis.ops.find((item) => item.index === opIndex);
  container.replaceChildren();
  if (!op) {
    const title = document.createElement("h3");
    title.textContent = "Select an op";
    const body = document.createElement("p");
    body.textContent = "Details will appear here.";
    container.append(title, body);
    return;
  }

  const graphIndex = buildGraphIndex(analysis);
  const title = document.createElement("h3");
  title.textContent = `#${String(op.index).padStart(3, "0")} ${op.name}`;
  container.append(title);

  // Static roofline estimate
  const target = analysis.target_profile || {};
  const isTflite = String(analysis?.format || "tflite").toLowerCase() === "tflite";
  const b = isTflite ? estimateOpBottleneck(op, target) : { packingUs: 0, breakUs: 0, fallbackUs: 0, computeUs: 0, memoryUs: 0, dominantLabel: null };
  const coldSetupUs = b.packingUs + b.breakUs;
  const serialUs = coldSetupUs + (b.fallbackUs || 0);
  const parallelUs = Math.max(b.computeUs, b.memoryUs);
  const totalUs = parallelUs + serialUs;
  const steadyUs = parallelUs + (b.fallbackUs || 0);
  if (totalUs > 0) {
    const roofSection = document.createElement("div");
    roofSection.className = "op-detail-roof-section";
    roofSection.append(detailGrid([
      ["Modeled steady",  formatUs(steadyUs)],
      ["Est. cold",    formatUs(totalUs)],
      ["Est. compute", formatUs(b.computeUs)],
      ["Est. memory",  formatUs(b.memoryUs)],
      ["Cold setup", coldSetupUs > 0 ? formatUs(coldSetupUs) : "-"],
      ["Steady fallback", b.fallbackUs > 0 ? formatUs(b.fallbackUs) : "-"],
      ["Dominant",     b.dominantLabel || "-"],
    ]));
    container.append(roofSection);
  }

  // Priority insight callout — shows up to 3 deployment issues above the detail grid
  const insights = buildOpInsights(op, b, analysis);
  if (insights.length > 0) {
    const box = document.createElement("div");
    box.className = "op-insight-box";
    for (const ins of insights) {
      const item = document.createElement("div");
      item.className = `op-insight-item op-insight-${ins.tone}`;
      item.title = ins.detail;
      item.textContent = ins.text;
      box.append(item);
    }
    container.append(box);
  }

  container.append(detailGrid(opDetailRows(op, analysis)));

  const dynamicCost = analysis?.dynamic_shape_cost_contract;
  const dynamicFormula = (dynamicCost?.op_formulas || []).find((row) => Number(row.op_index) === Number(op.index));
  const dynamicUnresolved = (dynamicCost?.unresolved_dynamic_compute_ops || []).find((row) => Number(row.op_index) === Number(op.index));
  const totalBlocker = (dynamicCost?.total_macs_unresolved_ops || []).find((row) => Number(row.op_index) === Number(op.index));
  if (dynamicFormula || dynamicUnresolved || totalBlocker) {
    const section = document.createElement("section");
    section.className = "op-detail-kernel-section";
    const heading = document.createElement("h4");
    heading.textContent = "Dynamic Shape Cost";
    const formula = dynamicFormula?.macs_formula;
    const symbolCount = Number(formula?.symbol_ids?.length || 0);
    const termCount = Number(formula?.terms?.length || 0);
    const guardCount = Number(formula?.preconditions?.length || 0);
    section.append(heading, detailGrid([
      ["Evidence", `${dynamicCost.evidence_class || "DERIVED"}; ${dynamicCost.status || "not assessed"}`],
      ["MAC formula", formula
        ? `Shape-dependent / ${formatNumber(symbolCount)} unbound symbol${symbolCount === 1 ? "" : "s"} / ${formula.status === "exact_guarded_integer_expression" ? `${formatNumber(guardCount)} exact guard${guardCount === 1 ? "" : "s"}` : `${formatNumber(termCount)} exact term${termCount === 1 ? "" : "s"}`}`
        : "not assessed"],
      ["Declared projection", dynamicFormula?.declared_shape_projection_macs == null
        ? dynamicFormula?.declared_shape_projection_status || "not available"
        : `${formatNumber(dynamicFormula.declared_shape_projection_macs)} MACs; ${dynamicFormula.declared_shape_projection_status || "example only"}`],
      ["Formula basis", dynamicFormula?.reason || dynamicUnresolved?.reason || "not emitted"],
      ["Total-MAC contribution", totalBlocker?.reason || "included in the emitted exact total expression"],
      ["Dimension bounds", dynamicCost.dimension_bounds_status || "not assessed"],
      ["Interpretation boundary", dynamicCost.interpretation_boundary || "Runtime dimensions must be bound before numeric evaluation."],
    ]));
    if (formula?.expression) {
      section.append(evidenceDisclosure(
        `Exact op MAC polynomial (${formatNumber(termCount)} terms)`,
        formula.expression,
        { contentLabel: `Exact symbolic MAC polynomial for op ${op.index}` },
      ));
    }
    container.append(section);
  }

  if (String(analysis?.format || "tflite").toLowerCase() === "tflite") {
    const target = analysis.target_profile || {};
    const observed = runtimeAssignment?.assignments?.find((item) => item.op_index === op.index) || null;
    const timingProfile = runtimeAssignment?.source?.adapter?.timing_profile || null;
    const observedTiming = (timingProfile?.execution_nodes || []).find((item) => observed?.delegated
      ? item.node_kind === "delegate_partition" && item.partition_id === observed.partition_id
      : item.node_kind === "original_op" && item.op_index === op.index) || null;
    const compared = runtimeAssignment?.comparison?.op_comparisons?.find((item) => item.op_index === op.index) || null;
    const observedBoundaryEdges = (runtimeAssignment?.comparison?.observed_boundary_inventory?.edges || [])
      .filter((edge) => edge.producer_op_index === op.index || edge.consumer_op_index === op.index);
    const boundaryEdges = predictedPartitionBoundaryEdgesForOp(analysis, op.index);
    const section = document.createElement("section");
    section.className = "op-detail-kernel-section";
    const heading = document.createElement("h4");
    heading.textContent = "Kernel & Delegation Evidence";
    const kernelCandidates = op.xnnpack_kernel_candidates || [];
    const alignmentMultiples = op.xnnpack_kernel_alignment_multiples || [];
    const tailMin = Number(op.channel_tail_overhead_percent_min ?? op.channel_tail_overhead_percent ?? 0);
    const tailMax = Number(op.channel_tail_overhead_percent_max ?? op.channel_tail_overhead_percent ?? 0);
    const selectorFacts = op.selector_artifact_facts || {};
    section.append(heading, detailGrid([
      ["Target binding", `${target.id || "-"} / profile ${(target.profile_sha256 || "").slice(0, 16) || "unbound"}`],
      ["ISA posture", `${target.architecture || "-"}; SIMD ${target.simd_width_bits || "?"}-bit; INT8/FP16/FP32 lanes ${target.int8_lanes || "?"}/${target.fp16_lanes || "?"}/${target.fp32_lanes || "?"}; op precision ${opPrecisionLabel(op, analysis)}`],
      ["Kernel candidates", `${op.xnnpack_kernel_evidence_class || "HEURISTIC_PROFILE"}; ${kernelCandidates.length ? `${kernelCandidates.length} source configuration(s)` : op.xnnpack_kernel_candidate || op.target_microkernel_hint || target.xnnpack_kernel_family || "not emitted"}`],
      ["Artifact selector facts", op.selector_artifact_facts ? `activation ${selectorFacts.activation_dtype || "unavailable"}; kernel area ${selectorFacts.kernel_area_status === "not_applicable" ? "N/A" : `${selectorFacts.kernel_area || 0} (${selectorFacts.kernel_area_status})`}; per-channel weights ${selectorFacts.per_channel_weights ? "yes" : "no"}; output C=${selectorFacts.output_channels || 0} (${selectorFacts.output_channels_status})` : "not emitted by protected selector"],
      ["Unresolved selector dimensions", (op.unresolved_selector_dimensions || []).join(" / ") || "not source-enumerated"],
      ["Selector reason code", op.no_match_reason_code || "not emitted"],
      ["Pinned source", op.xnnpack_kernel_source || "not source-enumerated for this target/op family"],
      ["Tile / tail model", op.output_channels > 0 ? `${op.output_channels} output channels / ${alignmentMultiples.length ? alignmentMultiples.map((value) => `x${value}`).join("/") : `x${op.channel_alignment_multiple || "?"}`}; modeled occupancy ${formatPercentRange(1 - tailMax, 1 - tailMin)}` : "not applicable"],
      ["Selector boundary", op.xnnpack_kernel_selector_status || "runtime kernel selection unconfirmed"],
      ["Build requirement", op.xnnpack_build_requirement || "runtime build configuration not embedded"],
      ["Static assignment", `${op.xnnpack_chain_id >= 0 ? `PREDICTED conditionally delegatable chain C${op.xnnpack_chain_id}` : op.xnnpack_chain_break ? "PREDICTED boundary" : "PREDICTED fallback"}; ${decodeXnnpReason(op.xnnpack_reason || "") || op.xnnpack_reason || "no rule code"}`],
      ["Predicted boundary edges", boundaryEdges.length ? boundaryEdges.map((edge) => `T${edge.tensor_index} ${edge.producer_domain}->${edge.consumer_domain} ${edge.payload_bytes == null ? "payload not assessed" : formatBytes(edge.payload_bytes)}`).join(" / ") : "none for this op"],
      ["Packing evidence", Number(op.weight_packing_overhead_us || 0) > 0 ? `ESTIMATED ${formatUs(op.weight_packing_overhead_us)} planning-profile formula output; not measured latency` : "not applicable"],
      ["Runtime assignment", observed ? `OBSERVED_RUNTIME ${observed.provider}${observed.partition_id != null ? ` / partition ${observed.partition_id}` : ""}${observed.lowering_id ? ` / lowering ${observed.lowering_id}` : ""}${observed.kernel ? ` / ${observed.kernel_id}:${observed.kernel}` : ""}` : "not imported; static prediction is not runtime confirmation"],
      ["Runtime identity mapping", observed?.mapping_method || "not assessed"],
      ["Observed timing", observedTiming
        ? observedTiming.mean_per_run_us == null
          ? `${observedTiming.event_sample_count} event(s) observed; per-run duration withheld because run count is not derivable`
          : observedTiming.node_kind === "delegate_partition"
          ? `${formatUs(observedTiming.mean_per_run_us)} partition total across ${observedTiming.run_count} run(s); not attributed to this original op`
          : `${formatUs(observedTiming.mean_per_run_us)} original execution-node mean across ${observedTiming.run_count} run(s)`
        : timingProfile ? "not mapped; no duration inferred" : "profiling evidence not imported"],
      ["Runtime source binding", runtimeAssignment?.source?.adapter?.schema?.startsWith("deepbom.tflite_runtime_info_adapter.v") ? `source protobuf ${(runtimeAssignment.source.profile_sha256 || "").slice(0, 16)}; active artifact bound by exact original-op topology; source artifact SHA-256 not embedded` : runtimeAssignment ? "assignment source bound to active artifact and target profile" : "not imported"],
      ["Selector dimensions resolved", (observed?.resolved_selector_dimensions || []).join(" / ") || "none"],
      ["Executed microkernel", observed?.kernel ? `${observed.kernel_id}: ${observed.kernel}; ${observed.kernel_source_ref}` : (runtimeAssignment?.source?.adapter?.schema?.startsWith("deepbom.tflite_runtime_info_adapter.v") ? "not exposed by imported TFLite evidence; source candidates remain static" : "not assessed")],
      ["Prediction agreement", compared ? `${compared.matches_prediction ? "MATCH" : "MISMATCH"}; ${compared.classification}` : "not assessed"],
      ["Observed boundary edges", observedBoundaryEdges.length ? observedBoundaryEdges.map((edge) => `T${edge.tensor_index} ${edge.observed_producer_domain}->${edge.observed_consumer_domain} ${edge.payload_bytes == null ? "payload not assessed" : formatBytes(edge.payload_bytes)}`).join(" / ") : "none confirmed for this op"],
    ]));
    if (kernelCandidates.length) {
      const details = document.createElement("details");
      details.className = "kernel-selector-details";
      details.open = true;
      const summary = document.createElement("summary");
      summary.textContent = `Pinned selector matrix (${kernelCandidates.length})`;
      details.append(summary);
      for (const candidate of kernelCandidates) {
        const item = document.createElement("div");
        item.className = "kernel-selector-candidate";
        const family = document.createElement("strong");
        family.textContent = candidate.family;
        const selectors = document.createElement("span");
        selectors.textContent = `Architecture: ${candidate.architecture_condition} | Compile: ${candidate.compile_condition} | Runtime: ${candidate.runtime_condition}`;
        const tail = document.createElement("span");
        tail.className = "kernel-selector-tail";
        const tile = Number(candidate.tile_nr || 0) > 0
          ? `MR<=${candidate.tile_mr || "?"} x NR${candidate.tile_nr}`
          : `${candidate.primary_tile || "?"}p${candidate.channel_tile || "?"}c`;
        tail.textContent = candidate.tail_projection_status === "assessed"
          ? `${tile}; alignment x${candidate.alignment_multiple}; C=${selectorFacts.output_channels} -> padded ${candidate.padded_output_channels}; inactive ${candidate.inactive_output_channels} (${formatPercent(candidate.inactive_lane_ratio)})`
          : `${tile}; tail ${candidate.tail_projection_status || "not assessed"}`;
        const source = document.createElement("span");
        source.textContent = `${candidate.source_ref} | file SHA-256 ${candidate.source_file_sha256}`;
        item.append(family, tail, selectors, source);
        details.append(item);
      }
      section.append(details);
    }
    container.append(section);
  } else {
    const observed = runtimeAssignment?.assignments?.find((item) => item.op_index === op.index) || null;
    const adapter = runtimeAssignment?.source?.adapter || null;
    const sourceRows = (analysis?.ort_compatibility_evidence?.execution_providers || []).flatMap((ep) => {
      const row = (ep.ops || []).find((item) => Number(item.op_index) === Number(op.index));
      return row ? [{ ...row, execution_provider: ep.execution_provider }] : [];
    });
    const section = document.createElement("section");
    section.className = "op-detail-kernel-section";
    const heading = document.createElement("h4");
    heading.textContent = "Runtime Provider Evidence";
    section.append(heading, detailGrid([
      ["Original graph identity", op.graph_node_name ? `${op.graph_node_name} / #${op.index} ${op.name}` : `unnamed / #${op.index} ${op.name}`],
      ["Pinned EP source candidates", sourceRows.length ? sourceRows.map((row) => `${row.execution_provider}: opset ${row.imported_opset ?? "?"} -> schema v${row.resolved_schema_version ?? "?"}; ${row.status}${row.documented_condition ? `; ${row.documented_condition}` : ""}`).join(" / ") : "protected ORT rulepack not loaded"],
      ["Runtime assignment", observed ? `OBSERVED_RUNTIME ${observed.provider}` : "not imported"],
      ["Identity mapping", observed?.mapping_method || "not assessed"],
      ["Runtime node", observed ? `${observed.runtime_node_name || "-"} / index ${observed.runtime_node_index ?? "-"}` : "not assessed"],
      ["Profile duration", observed?.duration_us == null ? "not assessed" : `${formatUs(observed.duration_us)} arithmetic mean / ${observed.sample_count} event(s) / sum ${formatUs(observed.duration_sum_us)}`],
      ["Runtime preparation", runtimeAssignment ? `DECLARED optimization ${runtimeAssignment.runtime?.graph_optimization_level || "not declared"}; execution ${runtimeAssignment.runtime?.execution_mode || "not declared"}` : "not imported"],
      ["Raw profile binding", runtimeAssignment?.source?.profile_sha256 || "not imported"],
      ["Adapter coverage", adapter ? `${adapter.mapped_kernel_event_count}/${adapter.kernel_event_count} kernel event(s); ${runtimeAssignment.assignment_count}/${runtimeAssignment.graph_op_count} original op(s); unresolved ${adapter.unresolved_runtime_node_count}, conflicts ${adapter.conflict_count}` : "not applicable"],
      ["Partition / microkernel", observed?.kernel ? `${observed.kernel_id}: ${observed.kernel}` : observed ? "not exposed by the imported ORT node event; not inferred" : "not assessed"],
    ]));
    container.append(section);
  }

  appendDetailList(
    container,
    "Input Tensors",
    opInputTensorDetails(analysis, op, graphIndex),
  );
  appendDetailList(
    container,
    "Output Tensors",
    opOutputTensorDetails(analysis, op, graphIndex),
  );
  appendDetailList(
    container,
    "Local Links",
    opLocalLinks(op, graphIndex),
  );

  if (influence || outputInfluence) {
    const section = document.createElement("div");
    section.className = "op-detail-weight-section";
    const secTitle = document.createElement("h4");
    secTitle.className = "op-detail-weight-title";
    secTitle.textContent = "Influence Maps";
    section.append(secTitle);

    const row = document.createElement("div");
    row.className = "influence-row";

    const mkInfCol = (res, mode, dirLabel) => {
      if (!res || !res.h || !res.w) return null;           // W5: skip zero-dim results
      const rendered = renderInfluenceCanvas(res, 280, mode);
      if (!rendered) return null;
      const col = document.createElement("div");
      col.className = "influence-col";
      const lbl = document.createElement("div");
      lbl.className = "influence-lbl";
      const chainLen = res.chain_len ?? res.chainLen;
      lbl.append(
        Object.assign(document.createElement("span"), { className: "inf-dir", textContent: dirLabel }),
        Object.assign(document.createElement("span"), { className: "inf-dim", textContent: ` ${res.h}×${res.w}` }),
        Object.assign(document.createElement("span"), { className: "inf-ops", textContent: chainLen != null ? ` · ${chainLen} ops` : "" }),
      );
      col.append(lbl, rendered);
      return col;
    };

    if (influence) {
      const col = mkInfCol(influence, "input", "← Input");
      if (col) row.append(col);
    }
    if (outputInfluence) {
      const col = mkInfCol(outputInfluence, "output", "Output →");
      if (col) row.append(col);
    }

    if (row.children.length > 0) { section.append(row); container.append(section); }
  }

  if (Array.isArray(weightHistograms) && weightHistograms.length > 0) {
    const section = document.createElement("div");
    section.className = "op-detail-weight-section";
    const secTitle = document.createElement("h4");
    secTitle.className = "op-detail-weight-title";
    secTitle.textContent = "Weight Distributions";
    section.append(secTitle);
    const grid = document.createElement("div");
    grid.className = "whist-grid whist-grid-explorer";
    for (const h of weightHistograms) grid.append(renderWeightHistogram(h));
    section.append(grid);
    container.append(section);
  }
}

export function graphSvgText(svg) {
  if (!svg?.children?.length) return "";
  const clone = svg.cloneNode(true);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  return new XMLSerializer().serializeToString(clone);
}

// Stage colors: 8 hues for stage-n CSS classes (deployed via CSS)
const STAGE_COLORS = ["#c7d2fe","#a7f3d0","#fde68a","#fbcfe8","#bae6fd","#d9f99d","#fecaca","#e9d5ff"];

function graphEdgeKey(tensorId, from, to) {
  return `${Number(tensorId)}:${Number(from)}:${Number(to)}`;
}

function boundaryEdgeMaps(boundaryInventory, scenario) {
  const baseline = new Map();
  for (const edge of boundaryInventory?.edges || []) {
    baseline.set(
      graphEdgeKey(edge.tensor_index, edge.producer_op_index, edge.consumer_op_index),
      edge,
    );
  }
  const transitions = new Map();
  for (const edge of scenario?.edgeChanges || []) {
    transitions.set(
      graphEdgeKey(edge.tensor_index, edge.producer_op_index, edge.consumer_op_index),
      edge,
    );
  }
  return { baseline, transitions };
}

function cpuIslandByOp(cpuIslands) {
  const result = new Map();
  for (const island of cpuIslands || []) {
    for (const opIndex of island.op_indices || []) result.set(Number(opIndex), island);
  }
  return result;
}

function appendCpuIslandBands(layer, cpuIslands, positions, scenario) {
  for (const island of cpuIslands || []) {
    const visible = (island.op_indices || [])
      .map((opIndex) => positions.get(Number(opIndex)))
      .filter(Boolean);
    if (!visible.length) continue;
    const minX = Math.min(...visible.map((item) => item.x)) - 12;
    const minY = Math.min(...visible.map((item) => item.y)) - 18;
    const maxX = Math.max(...visible.map((item) => item.x + GRAPH_NODE_WIDTH)) + 12;
    const maxY = Math.max(...visible.map((item) => item.y + GRAPH_NODE_HEIGHT)) + 12;
    const resolved = scenario?.type === "delegation-repair"
      && Number(scenario.islandIndex) === Number(island.island_index);
    const group = svgEl("g", { class: `graph-cpu-island${resolved ? " scenario-resolved" : ""}` });
    group.append(svgEl("rect", {
      x: String(minX),
      y: String(minY),
      width: String(maxX - minX),
      height: String(maxY - minY),
      rx: "8",
    }));
    const label = svgEl("text", { x: String(minX + 7), y: String(minY + 12) });
    label.textContent = resolved
      ? `WHAT-IF island ${island.island_index}: delegated`
      : `CPU island ${island.island_index} / ${island.op_count} op${island.op_count === 1 ? "" : "s"}`;
    group.append(label);
    layer.append(group);
  }
}

export function renderGraphMapContent(container, graphData, layout, {
  fullGraph = false,
  selectedOpIndex = null,
  onSelect = () => {},
  graphMode = "deploy",
  topologyAnnotations = null,
  format = "tflite",
  boundaryInventory = null,
  cpuIslands = [],
  scenario = null,
} = {}) {
  container.append(graphMapDefs(graphMode));

  const islandLayer = svgEl("g", { class: "graph-island-layer" });
  const edgeLayer = svgEl("g");
  const nodeLayer = svgEl("g");
  container.append(islandLayer, edgeLayer, nodeLayer);
  const boundaryMaps = boundaryEdgeMaps(boundaryInventory, scenario);
  const islandByOp = cpuIslandByOp(cpuIslands);
  if (graphMode === "deploy" && format !== "onnx") {
    appendCpuIslandBands(islandLayer, cpuIslands, layout.positions, scenario);
  }

  // Compute max bytes for edge thickness scaling
  const maxBytes = graphData.edges.reduce((m, e) => Math.max(m, e.bytes || 0), 0);

  // Collect ancestor/descendant op indices for the selected node
  const ancestors = new Set();
  const descendants = new Set();
  if (selectedOpIndex != null) {
    const edgesByTo = new Map();
    const edgesByFrom = new Map();
    for (const e of graphData.edges) {
      if (!edgesByTo.has(e.to)) edgesByTo.set(e.to, []);
      edgesByTo.get(e.to).push(e.from);
      if (!edgesByFrom.has(e.from)) edgesByFrom.set(e.from, []);
      edgesByFrom.get(e.from).push(e.to);
    }
    const walkUp = (idx) => {
      for (const prev of (edgesByTo.get(idx) || [])) {
        if (!ancestors.has(prev)) { ancestors.add(prev); walkUp(prev); }
      }
    };
    const walkDown = (idx) => {
      for (const next of (edgesByFrom.get(idx) || [])) {
        if (!descendants.has(next)) { descendants.add(next); walkDown(next); }
      }
    };
    walkUp(selectedOpIndex);
    walkDown(selectedOpIndex);
  }

  for (const edge of graphData.edges) {
    const key = graphEdgeKey(edge.tensorId, edge.from, edge.to);
    const boundary = boundaryMaps.baseline.get(key) || null;
    const scenarioTransition = boundaryMaps.transitions.get(key) || null;
    const isOnPath = selectedOpIndex != null && (
      (edge.from === selectedOpIndex || ancestors.has(edge.from)) &&
      (edge.to === selectedOpIndex || descendants.has(edge.to))
    );
    edgeLayer.append(...graphMapEdgeElements(edge, layout.positions, {
      showLabel: Boolean(boundary || scenarioTransition) || !fullGraph || graphData.edges.length <= 80,
      maxBytes,
      highlighted: isOnPath,
      graphMode,
      boundary,
      scenarioTransition,
    }));
  }

  for (const item of layout.nodes) {
    const pathRole = item.op.index === selectedOpIndex ? "selected"
      : ancestors.has(item.op.index) ? "ancestor"
      : descendants.has(item.op.index) ? "descendant"
      : null;
    nodeLayer.append(graphMapNodeGroup(item, {
      selected: item.op.index === selectedOpIndex,
      pathRole,
      onSelect,
      graphMode,
      topoAnn: topologyAnnotations?.opAnnotations?.get(item.op.index) ?? null,
      format,
      cpuIsland: islandByOp.get(item.op.index) || null,
      scenarioDelegated: scenario?.type === "delegation-repair"
        && (scenario.opIndices || []).includes(item.op.index),
    }));
  }
}

function graphMapDefs(graphMode = "deploy") {
  const defs = svgEl("defs");
  const mkArrow = (id, fill) => {
    const m = svgEl("marker", { id, markerWidth: "10", markerHeight: "10", refX: "9", refY: "3", orient: "auto", markerUnits: "strokeWidth" });
    m.append(svgEl("path", { d: "M0,0 L0,6 L9,3 z", fill }));
    return m;
  };
  defs.append(
    mkArrow("arrow",      "#9aa39d"),
    mkArrow("arrow-hl",   "#4d9e7c"),
    mkArrow("arrow-boundary", "#b45309"),
    mkArrow("arrow-scenario-removed", "#64748b"),
    mkArrow("arrow-scenario-added", "#dc2626"),
    mkArrow("arrow-int8", "#3b82f6"),
    mkArrow("arrow-f32",  "#94a3b8"),
    mkArrow("arrow-quant","#a78bfa"),
  );
  return defs;
}

function graphMapEdgeElements(edge, positions, {
  showLabel = false,
  maxBytes = 0,
  highlighted = false,
  graphMode = "deploy",
  boundary = null,
  scenarioTransition = null,
} = {}) {
  const from = positions.get(edge.from);
  const to = positions.get(edge.to);
  if (!from || !to) return [];

  const sameColumn = Math.abs(from.x - to.x) < GRAPH_NODE_WIDTH / 2;
  const goesRight = to.x > from.x;
  const startX = sameColumn
    ? from.x + GRAPH_NODE_WIDTH / 2
    : goesRight ? from.x + GRAPH_NODE_WIDTH : from.x;
  const startY = sameColumn
    ? from.y + (to.y >= from.y ? GRAPH_NODE_HEIGHT : 0)
    : from.y + GRAPH_NODE_MID_Y;
  const endX = sameColumn
    ? to.x + GRAPH_NODE_WIDTH / 2
    : goesRight ? to.x : to.x + GRAPH_NODE_WIDTH;
  const endY = sameColumn
    ? to.y + (to.y >= from.y ? 0 : GRAPH_NODE_HEIGHT)
    : to.y + GRAPH_NODE_MID_Y;
  const midX = (startX + endX) / 2;
  const midY = (startY + endY) / 2;

  // Scale edge stroke width by tensor bytes (1–5px range)
  const bytePct = maxBytes > 0 ? (edge.bytes || 0) / maxBytes : 0;
  const strokeWidth = (1 + bytePct * 4).toFixed(1);

  // In raw mode: color edge by dtype; otherwise standard highlighted/default
  let edgeClass = "graph-edge";
  let arrowId = "arrow";
  if (highlighted) {
    edgeClass += " path-edge";
    arrowId = "arrow-hl";
  } else if (graphMode === "raw") {
    if (edge.dtype === "INT8" || edge.dtype === "UINT8") {
      edgeClass += " graph-edge--int8"; arrowId = "arrow-int8";
    } else if (edge.dtype === "FLOAT32") {
      edgeClass += " graph-edge--f32"; arrowId = "arrow-f32";
    }
  }
  if (boundary) {
    edgeClass += " graph-edge--boundary";
    arrowId = "arrow-boundary";
  }
  if (scenarioTransition) {
    const transition = String(scenarioTransition.transition || "reclassified");
    edgeClass += ` graph-edge--scenario-${transition}`;
    arrowId = transition === "removed" ? "arrow-scenario-removed" : "arrow-scenario-added";
  }

  const path = svgEl("path", {
    class: edgeClass,
    d: sameColumn
      ? `M ${startX} ${startY} C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}`
      : `M ${startX} ${startY} C ${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}`,
    "marker-end": `url(#${arrowId})`,
    "stroke-width": strokeWidth,
  });
  if (boundary || scenarioTransition) {
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    const payload = scenarioTransition?.payload_bytes ?? boundary?.payload_bytes ?? edge.bytes;
    title.textContent = scenarioTransition
      ? `${String(scenarioTransition.transition || "reclassified").toUpperCase()} boundary T${edge.tensorId}; ${payload == null ? "payload not assessed" : formatBytes(payload)}`
      : `Predicted boundary T${edge.tensorId}; ${payload == null ? "payload not assessed" : formatBytes(payload)}; ${boundary.producer_domain || "unknown"} to ${boundary.consumer_domain || "unknown"}`;
    path.append(title);
  }
  if (!showLabel) return [path];

  const payload = scenarioTransition?.payload_bytes ?? boundary?.payload_bytes ?? edge.bytes;
  const bytes = payload ? ` ${formatBytes(payload)}` : "";
  const label = svgEl("text", {
    class: `graph-edge-label${highlighted ? " path-label" : ""}${boundary ? " boundary-label" : ""}${scenarioTransition ? ` scenario-${scenarioTransition.transition || "reclassified"}-label` : ""}`,
    x: String(midX),
    y: String(midY - 4),
    "text-anchor": "middle",
  });
  if (scenarioTransition) {
    label.textContent = `T${edge.tensorId}${bytes} ${scenarioTransition.transition || "reclassified"}`;
  } else if (boundary) {
    label.textContent = `T${edge.tensorId}${bytes} ${boundary.producer_domain || "?"}>${boundary.consumer_domain || "?"}`;
  } else {
    label.textContent = `T${edge.tensorId}${bytes}`;
  }
  return [path, label];
}

function formatBytes(b) {
  if (b >= 1048576) return `${(b / 1048576).toFixed(1)} MiB`;
  if (b >= 1024) {
    const kib = b / 1024;
    return `${kib.toFixed(kib >= 100 ? 0 : 1)} KiB`;
  }
  return `${b} B`;
}

// Stage background colors (CSS class targets: stage-0 … stage-7)
const STAGE_BG = ["#e0e7ff","#d1fae5","#fef9c3","#fce7f3","#e0f2fe","#f0fdf4","#fee2e2","#f3e8ff"];

function graphMapNodeGroup(item, {
  selected = false,
  pathRole = null,
  onSelect = () => {},
  graphMode = "deploy",
  topoAnn = null,
  format = "tflite",
  cpuIsland = null,
  scenarioDelegated = false,
} = {}) {
  const op = item.op;
  const ann = topoAnn ?? { role: "through", fanOutMax: 0 };

  // Build class list based on mode
  const classParts = ["graph-node"];
  if (graphMode === "deploy") {
    if (op.static_bound_guess) classParts.push(op.static_bound_guess);
    if (op.xnnpack_chain_break) classParts.push("chain-break");
    if (op.quant_hole) classParts.push("quant-hole");
    if (cpuIsland) classParts.push("cpu-island-node");
    if (scenarioDelegated) classParts.push("scenario-delegated");
  } else if (graphMode === "raw") {
    classParts.push("graph-node--raw");
  } else if (graphMode === "stage") {
    classParts.push(`stage-${(op.stage_index ?? 0) % 8}`);
  }
  // Structural topology role (always applied)
  if (ann.role !== "through") classParts.push(`topo-${ann.role}`);
  if (selected) classParts.push("selected");
  if (pathRole === "ancestor") classParts.push("path-ancestor");
  if (pathRole === "descendant") classParts.push("path-descendant");

  const group = svgEl("g", {
    class: classParts.filter(Boolean).join(" "),
    transform: `translate(${item.x}, ${item.y})`,
    tabindex: "0",
    "data-op-index": String(op.index),
    role: "button",
    "aria-label": `Select op ${op.index} ${op.name}`,
  });
  group.addEventListener("pointerdown", (e) => e.stopPropagation());
  group.addEventListener("click", (e) => { e.stopPropagation(); onSelect(op); });
  group.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(op); } });

  group.append(svgEl("rect", { class: "graph-node-hitbox", x: "-5", y: "-5", width: "182", height: "70", rx: "9" }));
  group.append(svgEl("rect", { class: "graph-node-card", width: String(GRAPH_NODE_WIDTH), height: String(GRAPH_NODE_HEIGHT), rx: "7" }));

  // Stage mode: tinted background rect
  if (graphMode === "stage" && op.stage_index != null) {
    const stageRect = svgEl("rect", {
      class: "graph-node-stage-bg",
      x: "1", y: "1",
      width: String(GRAPH_NODE_WIDTH - 2),
      height: String(GRAPH_NODE_HEIGHT - 2),
      rx: "6",
      fill: STAGE_BG[(op.stage_index ?? 0) % 8],
      opacity: "0.65",
    });
    group.append(stageRect);
  }

  // Text lines vary by mode
  const title = svgEl("text", { x: "10", y: "20" });
  title.textContent = `#${String(op.index).padStart(3, "0")} ${op.name}`;

  const detail = svgEl("text", { class: "subtext", x: "10", y: "36" });
  const metric = svgEl("text", { class: "subtext", x: "10", y: "50" });

  if (graphMode === "raw") {
    const dtype = item.outputDtype ?? "";
    detail.textContent = `${formatShapes(op.output_shapes)}${dtype ? ` [${dtype}]` : ""}`;
    metric.textContent = op.input_shapes ? formatShapes(op.input_shapes) : "";
    // Tint card border by dtype
    if (dtype === "INT8" || dtype === "UINT8") {
      group.append(svgEl("rect", { class: "graph-node-dtype-bar graph-node-dtype-int8", x: "0", y: String(GRAPH_NODE_HEIGHT - 3), width: String(GRAPH_NODE_WIDTH), height: "3", rx: "2" }));
    } else if (dtype === "FLOAT32") {
      group.append(svgEl("rect", { class: "graph-node-dtype-bar graph-node-dtype-f32", x: "0", y: String(GRAPH_NODE_HEIGHT - 3), width: String(GRAPH_NODE_WIDTH), height: "3", rx: "2" }));
    }
  } else if (graphMode === "stage") {
    detail.textContent = `Stage ${op.stage_index ?? "?"} / ${humanizeStageKey(op.stage_key)}`;
    detail.setAttribute("title", `Raw stage key: ${op.stage_key || "not emitted"}`);
    metric.textContent = `MACs ${formatExactInteger(op.macs_decimal, op.macs, "N/A")}`;
  } else {
    // deploy (default)
    detail.textContent = `${formatShapes(op.output_shapes)} | ${op.static_bound_guess}`;
    const macText = formatExactInteger(op.macs_decimal, op.macs, "N/A");
    metric.textContent = format === "onnx"
      ? `MACs ${macText} | EP not modeled`
      : `MACs ${macText} | ${scenarioDelegated ? "WHAT-IF delegated" : `XNN ${op.xnnpack_chain_id >= 0 ? `candidate C${op.xnnpack_chain_id}` : "predicted fallback"}`}`;
  }
  group.append(title, detail, metric);

  // Topology role badge (always shown in deploy + stage mode)
  let badgeX = GRAPH_NODE_WIDTH - 8;
  if (graphMode !== "raw") {
    if (ann.role === "branch-merge") {
      const b = svgEl("text", { class: "node-badge topo-merge-badge", x: String(badgeX), y: "14", "text-anchor": "end", role: "img", "aria-label": "Branch merge point" });
      b.textContent = "⋈"; group.append(b); badgeX -= 18;
    } else if (ann.role === "branch-split") {
      const b = svgEl("text", { class: "node-badge topo-split-badge", x: String(badgeX), y: "14", "text-anchor": "end", role: "img", "aria-label": `Branch split, tensor fan-out ${ann.fanOutMax}` });
      b.textContent = ann.fanOutMax > 2 ? `⑂×${ann.fanOutMax}` : "⑂"; group.append(b); badgeX -= 22;
    } else if (ann.role === "quant-boundary") {
      const b = svgEl("text", { class: "node-badge topo-quant-badge", x: String(badgeX), y: "14", "text-anchor": "end", role: "img", "aria-label": "Quantization boundary" });
      b.textContent = "Q↕"; group.append(b); badgeX -= 22;
    }
  }

  // High-level pattern badges — second row at y=28 (right-aligned single-char glyphs)
  // Rendered after text elements so they appear on top in SVG z-order.
  if (ann.patterns?.size > 0 && graphMode !== "raw") {
    const PAT_SPECS = [
      { id: "attention",  sym: "◇", cls: "pattern-badge-attn",     tip: "Attention block — SOFTMAX between MATMUL/FC ops" },
      { id: "qdq-island", sym: "⚡", cls: "pattern-badge-qdq",      tip: "Q/DQ island — FP32 compute between QUANTIZE/DEQUANTIZE ops" },
      { id: "fpn-merge",  sym: "⬦", cls: "pattern-badge-fpn",      tip: "FPN merge — upsampled branch detected (multi-scale feature fusion)" },
      { id: "residual",   sym: "↺", cls: "pattern-badge-residual", tip: "Residual skip connection — branch depth gap ≥3" },
    ];
    let patX = GRAPH_NODE_WIDTH - 8;
    for (const spec of PAT_SPECS) {
      if (!ann.patterns.has(spec.id)) continue;
      const b = svgEl("text", { class: `node-badge ${spec.cls}`, x: String(patX), y: "28", "text-anchor": "end", role: "img", "aria-label": spec.tip });
      const tt = document.createElementNS("http://www.w3.org/2000/svg", "title");
      tt.textContent = spec.tip;
      b.appendChild(tt);
      b.appendChild(document.createTextNode(spec.sym));
      group.append(b);
      patX -= 13;
    }
  }

  // Deploy-mode badges (XNNPACK, quant hole)
  if (graphMode === "deploy") {
    if (!scenarioDelegated && cpuIsland) {
      const badge = svgEl("text", { class: "node-badge cpu-island-badge", x: String(badgeX), y: "14", "text-anchor": "end" });
      badge.textContent = `CPU${cpuIsland.island_index}`;
      group.append(badge);
      badgeX -= 34;
    }
    if (op.quant_hole) {
      const badge = svgEl("text", { class: "node-badge quant-hole-badge", x: String(badgeX), y: "14", "text-anchor": "end" });
      badge.textContent = "Q⚠"; group.append(badge); badgeX -= 24;
    }
    if (op.xnnpack_chain_break && !scenarioDelegated) {
      const badge = svgEl("text", { class: "node-badge break-badge", x: String(badgeX), y: "14", "text-anchor": "end" });
      badge.textContent = "BRK"; group.append(badge);
    }
  }

  return group;
}
