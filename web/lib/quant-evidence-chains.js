import { ensureQuantResearchCoverage } from "./quant-research-applicability.js";
import {
  buildBiasScaleCheck,
  buildRepresentableKernelChannelCheck,
} from "./quantization-contract-summary.js";

const QDQ_VIEW_ID = "qdq-action";
const COVERAGE_VIEW_ID = "coverage";
let openGraphOp = () => {};

const CHAINS = Object.freeze([
  {
    id: "residual-contract",
    title: "A. Lattice & Residual Contracts",
    flow: "Six-family lattice -> ADD-specific migration, step response, and distortion",
    panels: [
      "quantizationLatticePanel",
      "contractMigrationPanel",
      "residualStepResponsePanel",
      "residualContractDistortionPanel",
    ],
  },
  {
    id: "integer-safety",
    title: "B. Integer Arithmetic Safety",
    flow: "Exact accumulator envelope -> constructive extremum -> channel vitality",
    panels: [
      "accumulatorAtlasPanel",
      "kernelWitnessPanel",
      "channelVitalityPanel",
    ],
  },
  {
    id: "numerical-abi",
    title: "C. Numerical ABI And Build Reproducibility",
    flow: "Requantization -> rounding equivalence -> reachability -> propagation -> input witness",
    panels: [
      "requantizationFidelityPanel",
      "roundingEquivalencePanel",
      "accumulatorReachabilityPanel",
      "numericalAbiPropagationPanel",
      "inputCounterexamplePanel",
    ],
  },
  {
    id: "preprocessing",
    title: "D. Preprocessing Contract",
    flow: "Pixel-to-tensor realizability -> local runtime consequence",
    panels: [
      "preprocessingRealizabilityPanel",
      "preprocessingConsequencePanel",
    ],
  },
]);

export function installQuantEvidenceChains(doc = document, { onOpenOp = () => {} } = {}) {
  openGraphOp = onOpenOp;
  const panels = CHAINS.flatMap((chain) => chain.panels.map((id) => doc.getElementById(id)));
  if (panels.some((panel) => !panel)) return false;
  const parent = panels[0].parentElement;
  if (!parent || panels.some((panel) => panel.parentElement !== parent)) return false;
  const marker = doc.createComment("quant-evidence-chains");
  parent.insertBefore(marker, panels[0]);
  const workbench = doc.createElement("section");
  workbench.className = "quant-lab-workbench";
  workbench.dataset.visualScope = "quant-labs";
  const tabs = doc.createElement("div");
  tabs.className = "quant-lab-tabs";
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", "Quantization laboratory views");
  const views = doc.createElement("div");
  views.className = "quant-lab-views";

  const addView = (id, label, detail, tfliteOnly = false) => {
    const tab = doc.createElement("button");
    tab.type = "button";
    tab.dataset.quantLabTab = id;
    tab.dataset.tfliteOnly = tfliteOnly ? "true" : "false";
    tab.setAttribute("role", "tab");
    tab.append(element(doc, "strong", "", label), element(doc, "span", "", detail));
    tab.addEventListener("click", () => activateQuantLabView(doc, id));
    const view = doc.createElement("section");
    view.className = "quant-lab-view";
    view.dataset.quantLabView = id;
    view.hidden = true;
    tabs.append(tab);
    views.append(view);
    return view;
  };

  const qdqView = addView(QDQ_VIEW_ID, "Q/DQ & Action", "boundaries and next experiment");
  qdqView.dataset.quantQdqAction = "";
  const coverage = doc.createElement("section");
  coverage.className = "quant-research-coverage";
  coverage.dataset.quantResearchCoverage = "";
  const coverageView = addView(COVERAGE_VIEW_ID, "Coverage", "applicability and evidence state", true);
  coverageView.append(coverage);
  for (const chain of CHAINS) {
    const view = addView(chain.id, chain.title.replace(/^[A-D]\.\s*/, ""), chain.flow, true);
    view.dataset.quantEvidenceChain = chain.id;
    const synthesis = doc.createElement("section");
    synthesis.className = "quant-chain-synthesis";
    synthesis.dataset.quantChainSynthesis = chain.id;
    synthesis.hidden = true;
    const grid = doc.createElement("div");
    grid.className = "quant-chain-grid";
    for (const panelId of chain.panels) {
      const panel = doc.getElementById(panelId);
      panel.removeAttribute("data-visual-scope");
      panel.removeAttribute("data-format-scope");
      grid.append(panel);
    }
    view.append(synthesis, grid);
  }
  workbench.append(tabs, views);
  parent.insertBefore(workbench, marker);
  marker.remove();
  activateQuantLabView(doc, QDQ_VIEW_ID);
  return true;
}

export function renderQuantEvidenceChains(analysis, doc = document) {
  renderQuantResearchCoverage(analysis, doc);
  for (const chain of CHAINS) {
    const synthesis = doc.querySelector(`[data-quant-chain-synthesis="${chain.id}"]`);
    if (!synthesis) continue;
    synthesis.replaceChildren();
    synthesis.hidden = true;
  }
  renderQdqAndInterventionView(analysis, doc);
  const tflite = analysis && String(analysis.format || "").toLowerCase() === "tflite";
  for (const tab of doc.querySelectorAll("[data-quant-lab-tab][data-tflite-only='true']")) {
    tab.hidden = !tflite;
  }
  if (!tflite) {
    activateQuantLabView(doc, QDQ_VIEW_ID);
    return;
  }
  renderExactChannelConvergence(analysis, doc);
}

export function deriveQuantInterventionPosture(analysis = {}) {
  const status = analysis.quantization_status || {};
  const classification = String(status.classification || "not_assessed");
  const holes = Array.isArray(analysis.quant_holes)
    ? analysis.quant_holes
    : (analysis.ops || []).filter((op) => op.quant_hole);
  const exactZeroChannels = Number(analysis.weight_integrity?.exact_zero_kernel_slice_count || 0);
  const lowGridTensors = Number(analysis.weight_integrity?.low_grid_utilization_tensors || 0);
  const saturatedTensors = Number(analysis.weight_integrity?.saturated_quantized_tensors || 0);
  const bias = buildBiasScaleCheck(analysis);
  const representable = buildRepresentableKernelChannelCheck(analysis);
  const modes = analysis.metadata_presence?.converter_optimization_modes || [];
  const qatDeclared = modes.includes("QUANTIZATION_AWARE_TRAINING");
  const ptqModes = modes.filter((mode) => String(mode).startsWith("PTQ_"));
  const fullInteger = classification === "full_integer";
  const ptqCandidate = [
    "not_quantized_float",
    "float16_weight_storage",
    "dynamic_range_or_weight_only",
    "mixed_quantization",
    "qdq_signals_only",
  ].includes(classification) || holes.length > 0;
  const qatReview = exactZeroChannels > 0 || representable.flagged_channels > 0;
  const contractRepair = bias.status === "fail";
  const lineage = qatDeclared
    ? { label: "QAT", evidence: "DECLARED", detail: "CONVERSION_METADATA declares QUANTIZATION_AWARE_TRAINING; the serialized artifact state is assessed separately." }
    : ptqModes.length
      ? { label: ptqModes.join(" / "), evidence: "DECLARED", detail: "CONVERSION_METADATA declares the listed PTQ optimization mode(s); the serialized artifact state is assessed separately." }
      : { label: "Training path unbound", evidence: "NOT EMBEDDED", detail: "The artifact does not declare whether QAT or PTQ produced this model." };
  const actions = [
    {
      id: "ptq",
      label: "PTQ experiment",
      state: ptqCandidate ? "candidate" : fullInteger ? "not-indicated" : "conditional",
      detail: ptqCandidate
        ? `${holes.length} internal serialized Q/DQ conversion op(s); ${formatPercent(status.quantized_compute_mac_percent || 0)} quantized compute MAC coverage. Test a representative-data PTQ export first where operator support permits.`
        : fullInteger
          ? "The artifact is already full-integer with no internal serialized Q/DQ conversion; another generic PTQ pass is not justified by this evidence."
          : "PTQ suitability depends on target support, representative calibration data, and output acceptance criteria.",
    },
    {
      id: "qat",
      label: "QAT / source review",
      state: qatReview ? "review" : ptqCandidate ? "conditional" : "not-indicated",
      detail: qatReview
        ? `${exactZeroChannels} exact-zero stored kernel channel(s) and ${representable.flagged_channels} near-zero representable channel(s) require source-checkpoint or QAT-graph comparison before changing ranges.`
        : ptqCandidate
          ? "Escalate to QAT only if a controlled PTQ candidate misses the declared output or accuracy requirement."
          : "No static channel-collapse signal currently establishes a QAT intervention.",
    },
    {
      id: "reexport",
      label: "Contract re-export",
      state: contractRepair ? "required" : lowGridTensors || saturatedTensors ? "review" : "conditional",
      detail: contractRepair
        ? `${bias.mismatch_groups} bias-scale group(s) violate bias_scale = input_scale x weight_scale; regenerate the artifact before evaluation.`
        : lowGridTensors || saturatedTensors
          ? `${lowGridTensors} constant tensor(s) use under 25% of the 8-bit grid; ${saturatedTensors} exceed 1% endpoint saturation. Review weight/range generation, not only activation calibration.`
          : "Re-export only when a selected PTQ/QAT experiment changes a serialized quantization contract; preserve hash-bound before/after output evidence.",
    },
  ];
  return {
    schema: "deepbom.quant_intervention_posture.v1",
    evidence_class: "DERIVED_WITH_DECLARED_LIMITS",
    classification,
    lineage,
    q_ops: Number(status.quantize_ops || 0),
    dq_ops: Number(status.dequantize_ops || 0),
    activation_boundary_ops: Number(status.activation_quantize_ops || 0) + Number(status.activation_dequantize_ops || 0),
    constant_conversion_ops: Number(status.constant_precision_conversion_ops || 0),
    holes: holes.length,
    internal_conversion_ops: holes.length,
    maximum_adjacent_mac_percent: Number(analysis.quant_hole_mac_impact || 0),
    actions,
    interpretation_boundary: "This selects the next controlled engineering experiment from artifact evidence. It does not prove that PTQ, QAT, retraining, or a contract rewrite will improve task accuracy.",
  };
}

function activateQuantLabView(doc, id) {
  const workbench = doc.querySelector(".quant-lab-workbench");
  if (!workbench) return;
  const requested = workbench.querySelector(`[data-quant-lab-view="${id}"]`);
  const fallback = workbench.querySelector(`[data-quant-lab-view="${QDQ_VIEW_ID}"]`);
  const target = requested && !workbench.querySelector(`[data-quant-lab-tab="${id}"]`)?.hidden ? requested : fallback;
  for (const tab of workbench.querySelectorAll("[data-quant-lab-tab]")) {
    const active = tab.dataset.quantLabTab === target?.dataset.quantLabView;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  }
  for (const view of workbench.querySelectorAll("[data-quant-lab-view]")) {
    view.hidden = view !== target;
  }
}

function renderQdqAndInterventionView(analysis, doc) {
  const root = doc.querySelector("[data-quant-qdq-action]");
  if (!root) return;
  root.replaceChildren();
  if (!analysis) return;
  const posture = deriveQuantInterventionPosture(analysis);
  const status = analysis.quantization_status || {};
  const ops = analysis.ops || [];
  const boundaries = ops.filter((op) => isQuantBoundaryName(op.name) || op.quant_hole);
  const head = element(doc, "div", "quant-lab-head");
  const copy = element(doc, "div");
  copy.append(
    element(doc, "span", "quant-chain-synthesis-kicker", "Artifact quantization posture"),
    element(doc, "h3", "", status.label || "Quantization not assessed"),
    element(doc, "p", "", status.summary || "No quantization classification was emitted."),
  );
  const evidenceBadges = element(doc, "div", "quant-lab-evidence-badges");
  const artifactState = element(doc, "span", "quant-lab-lineage", `Artifact state / ${posture.evidence_class}`);
  artifactState.title = "Derived from serialized tensor and operator contracts; this is not a training-history observation.";
  const lineage = element(doc, "span", "quant-lab-lineage", `Training lineage: ${posture.lineage.label} / ${posture.lineage.evidence}`);
  lineage.title = posture.lineage.detail;
  evidenceBadges.append(artifactState, lineage);
  head.append(copy, evidenceBadges);

  const metrics = element(doc, "div", "quant-lab-metrics");
  metrics.append(
    labMetric(doc, "Quantized compute MAC share", formatPercent(status.quantized_compute_mac_percent || 0), "", "Numerator: MACs on quantized compute paths. Denominator: all MAC-assessed compute ops."),
    labMetric(doc, "QUANTIZE ops", String(posture.q_ops)),
    labMetric(doc, "DEQUANTIZE ops", String(posture.dq_ops)),
    labMetric(doc, "Mid-graph 8-bit/FP32 boundaries", String(posture.internal_conversion_ops), posture.internal_conversion_ops ? "risk" : "ok", "Count of serialized internal 8-bit activation-to-FP32 or FP32-to-8-bit transition operators. This is not a float-island region count. Constant storage conversions are excluded."),
    labMetric(doc, "Constant precision conversions", String(posture.constant_conversion_ops), "", "Serialized constant storage-to-compute precision conversions; excluded from activation-boundary counts."),
    labMetric(doc, "Max graph-neighbor MAC share", formatPercent(posture.maximum_adjacent_mac_percent), posture.internal_conversion_ops ? "review" : "", "Maximum MAC count among graph-adjacent producer/consumer ops divided by total model MACs."),
  );

  const flow = element(doc, "div", "qdq-flow");
  flow.setAttribute("role", "img");
  flow.setAttribute("aria-label", "Operator quantization and Q/DQ flow");
  for (const op of ops) {
    const cell = element(doc, "button", qdqCellClass(op));
    cell.type = "button";
    cell.title = `#${padOp(op.index)} ${op.name}: ${op.quant_hole ? "mid-graph 8-bit/FP32 boundary" : op.quantization_state || (op.quantized_compute_path ? "quantized" : "floating or structural")}`;
    cell.setAttribute("aria-label", cell.title);
    cell.addEventListener("click", () => openGraphOp(Number(op.index)));
    flow.append(cell);
  }
  const legend = element(doc, "div", "qdq-legend");
  for (const [tone, label] of [["quant", "quantized"], ["q", "quantize"], ["dq", "dequantize"], ["storage", "constant precision"], ["hole", "mid-graph boundary"], ["float", "float / structural"]]) {
    const item = element(doc, "span", "");
    item.append(element(doc, "i", `qdq-cell-${tone}`), doc.createTextNode(label));
    legend.append(item);
  }

  const boundaryList = element(doc, "div", "qdq-boundary-list");
  if (!boundaries.length) {
    boundaryList.append(element(doc, "p", "quant-lab-empty", posture.classification === "full_integer"
      ? "No explicit Q/DQ operators. The serialized path is classified full-integer and contains no mid-graph 8-bit/FP32 boundary operator."
      : "No explicit Q/DQ operator or derived mid-graph 8-bit/FP32 boundary operator was found."));
  } else {
    for (const op of boundaries) {
      const row = element(doc, "button", `qdq-boundary-row${op.quant_hole ? " risk" : ""}`);
      row.type = "button";
      row.append(
        element(doc, "strong", "", `#${padOp(op.index)} ${op.name}`),
        element(doc, "span", "", op.quant_hole_detail || op.quantization_detail || `${shapeLabel(op.input_shapes?.[0])} -> ${shapeLabel(op.output_shapes?.[0])}`),
      );
      row.addEventListener("click", () => openGraphOp(Number(op.index)));
      boundaryList.append(row);
    }
  }

  const actions = element(doc, "div", "quant-action-grid");
  for (const action of posture.actions) {
    const card = element(doc, "article", "quant-action-card");
    card.dataset.state = action.state;
    card.append(
      element(doc, "span", "quant-action-state", action.state.replace("-", " ").toUpperCase()),
      element(doc, "h4", "", action.label),
      element(doc, "p", "", action.detail),
    );
    actions.append(card);
  }
  const boundary = element(doc, "p", "quant-intervention-boundary", posture.interpretation_boundary);
  root.append(head, metrics, element(doc, "h4", "quant-lab-section-title", "Serialized Q/DQ flow"), flow, legend, boundaryList, element(doc, "h4", "quant-lab-section-title", "Next controlled experiment"), actions, boundary);
}

function labMetric(doc, label, value, tone = "", title = "") {
  const item = element(doc, "div", "quant-lab-metric");
  if (tone) item.dataset.tone = tone;
  if (title) item.title = title;
  item.append(element(doc, "span", "", label), element(doc, "strong", "", value));
  return item;
}

function isQuantBoundaryName(name) {
  return ["QUANTIZE", "DEQUANTIZE", "QUANTIZELINEAR", "DEQUANTIZELINEAR"].includes(String(name || "").toUpperCase());
}

function qdqCellClass(op) {
  const name = String(op?.name || "").toUpperCase();
  if (op?.quant_hole) return "qdq-cell qdq-cell-hole";
  if (["float16_constant_expansion", "quantized_constant_expansion", "constant_precision_conversion"].includes(op?.quantization_state)) return "qdq-cell qdq-cell-storage";
  if (["QUANTIZE", "QUANTIZELINEAR"].includes(name)) return "qdq-cell qdq-cell-q";
  if (["DEQUANTIZE", "DEQUANTIZELINEAR"].includes(name)) return "qdq-cell qdq-cell-dq";
  if (op?.quantized_compute_path || String(op?.quantization_state || "").startsWith("quantized")) return "qdq-cell qdq-cell-quant";
  return "qdq-cell qdq-cell-float";
}

function shapeLabel(shape) {
  return Array.isArray(shape) && shape.length ? shape.join("x") : "shape unavailable";
}

export const QUANT_EVIDENCE_CHAIN_COUNT = CHAINS.length;

function renderQuantResearchCoverage(analysis, doc) {
  const root = doc.querySelector("[data-quant-research-coverage]");
  if (!root) return;
  root.replaceChildren();
  if (!analysis || String(analysis.format || "").toLowerCase() !== "tflite") {
    root.hidden = true;
    return;
  }
  root.hidden = false;
  const coverage = ensureQuantResearchCoverage(analysis);
  const head = element(doc, "div", "quant-research-coverage-head");
  const copy = element(doc, "div");
  copy.append(
    element(doc, "span", "quant-chain-synthesis-kicker", "Quant research coverage"),
    element(doc, "h3", "", coverage.artifact_class_label),
    element(doc, "p", "", coverage.artifact_class_detail),
  );
  const badge = element(doc, "span", "quant-research-coverage-badge", `${coverage.class_supported_lab_count}/${coverage.lab_count} class-supported`);
  badge.dataset.tone = coverage.class_supported_lab_count === coverage.lab_count ? "ok" : "review";
  head.append(copy, badge);
  const metrics = element(doc, "div", "quant-research-coverage-metrics");
  metrics.append(
    coverageMetric(doc, "Class envelope", `${coverage.class_supported_lab_count}/${coverage.lab_count}`),
    coverageMetric(doc, "Artifact applicable", `${coverage.artifact_applicable_lab_count}/${coverage.lab_count}`),
    coverageMetric(doc, "Assessed", String(coverage.assessed_lab_count)),
    coverageMetric(doc, "Partial", String(coverage.partial_lab_count)),
    coverageMetric(doc, "Not assessed", String(coverage.not_assessed_lab_count)),
    coverageMetric(doc, "Excluded", String(coverage.not_applicable_lab_count)),
  );
  const applicable = coverage.labs.filter((row) => row.class_supported).map((row) => row.label);
  const boundary = element(doc, "p", "quant-research-coverage-boundary");
  boundary.textContent = applicable.length
    ? `Class-supported analyses: ${applicable.join(", ")}. ${coverage.scan_denominator_policy}`
    : coverage.scan_denominator_policy;
  root.append(head, metrics, boundary);
}

function renderExactChannelConvergence(analysis, doc) {
  const target = doc.querySelector('[data-quant-chain-synthesis="integer-safety"]');
  if (!target) return;
  const proof = deriveExactChannelConvergence(analysis);
  if (!proof) return;
  const {
    accumulator,
    channelIndex,
    outputRange,
    ratio,
    shift,
    termCount,
    topOpIndex,
    zeroWeights,
  } = proof;
  const heading = element(doc, "div", "quant-chain-synthesis-head");
  const copy = element(doc, "div");
  copy.append(
    element(doc, "span", "quant-chain-synthesis-kicker", "Cross-ledger convergence"),
    element(doc, "h4", "", `#${padOp(topOpIndex)} ${accumulator.op_name}, channel ${channelIndex}`),
    element(
      doc,
      "p",
      "",
      "The same operator and channel are joined across independently hashed accumulator, kernel-witness, requantization, and vitality ledgers.",
    ),
  );
  const badge = element(doc, "span", "quant-chain-synthesis-badge", ratio >= 0.9 ? "Critical exact headroom" : "Exact-channel proof");
  badge.dataset.tone = ratio >= 0.9 ? "risk" : "review";
  heading.append(copy, badge);

  const metrics = element(doc, "div", "quant-chain-synthesis-metrics");
  metrics.append(
    synthesisMetric(doc, "INT32 use", formatPercent(ratio)),
    synthesisMetric(doc, "Stored bias", formatInteger(accumulator.worst_channel.bias_decimal)),
    synthesisMetric(doc, "Centered kernel", termCount && zeroWeights === termCount ? `${zeroWeights}/${termCount} weights are zero` : `${zeroWeights}/${termCount} zero weights`),
    synthesisMetric(doc, "Requantization", `shift ${signed(shift)}`),
    synthesisMetric(doc, "Dual-mode output", outputRange ?? "constant by vitality proof"),
  );

  const conclusion = element(doc, "p", "quant-chain-synthesis-conclusion");
  conclusion.textContent = outputRange
    ? `This channel is input-independent at the stored kernel, bias-dominated, and proven constant at output code ${outputRange} under both pinned rounding paths.`
    : "This channel is input-independent at the stored kernel, bias-dominated, and proven constant under both pinned rounding paths.";
  const actions = element(doc, "div", "quant-chain-synthesis-actions");
  [
    ["Accumulator", "accumulatorAtlasPanel"],
    ["Kernel witness", "kernelWitnessPanel"],
    ["Requantization", "requantizationFidelityPanel"],
    ["Vitality", "channelVitalityPanel"],
  ].forEach(([label, panelId]) => {
    const button = element(doc, "button", "", `Open ${label}`);
    button.type = "button";
    button.addEventListener("click", () => {
      const panel = doc.getElementById(panelId);
      const viewId = panel?.closest("[data-quant-lab-view]")?.dataset.quantLabView;
      if (viewId) activateQuantLabView(doc, viewId);
      panel?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    actions.append(button);
  });
  target.append(heading, metrics, conclusion, actions);
  target.hidden = false;
}

export function deriveExactChannelConvergence(analysis) {
  const atlas = analysis?.accumulator_atlas;
  const topOpIndex = atlas?.headroom_ranking_op_indices?.[0];
  const accumulator = findOp(atlas, topOpIndex);
  const channelIndex = accumulator?.worst_channel?.channel_index;
  if (!Number.isInteger(channelIndex)) return null;
  const requantization = findOp(analysis.requantization_fidelity, topOpIndex);
  const kernel = findOp(analysis.kernel_extremum_witness, topOpIndex);
  const vitality = findOp(analysis.channel_vitality, topOpIndex);
  const kernelChannel = findChannel(kernel, channelIndex);
  const vitalityChannel = findChannel(vitality, channelIndex);
  const shift = requantization?.channel_shifts?.[channelIndex];
  const defaultConstant = vitality?.default_constant_channel_indices?.includes(channelIndex);
  const singleConstant = vitality?.single_constant_channel_indices?.includes(channelIndex);
  if (!kernelChannel || !vitalityChannel || !Number.isInteger(shift) || !defaultConstant || !singleConstant) return null;
  const ratio = Number(accumulator.maximum_int32_ratio || 0);
  const zeroWeights = Number(kernelChannel.zero_centered_weight_count || 0);
  const termCount = Number(kernelChannel.term_count || 0);
  return {
    accumulator,
    channelIndex,
    kernelChannel,
    outputRange: exactDualModeOutput(kernelChannel),
    ratio,
    shift,
    termCount,
    topOpIndex,
    vitalityChannel,
    zeroWeights,
  };
}

function findOp(evidence, opIndex) {
  return evidence?.ops?.find((row) => Number(row.op_index) === Number(opIndex) && row.assessment_status === "assessed") || null;
}

function findChannel(row, channelIndex) {
  if (Number(row?.worst_channel?.channel_index) === channelIndex) return row.worst_channel;
  return row?.top_channels?.find((channel) => Number(channel.channel_index) === channelIndex) || null;
}

function exactDualModeOutput(channel) {
  const values = [
    channel?.minimum?.default_output_code,
    channel?.maximum?.default_output_code,
    channel?.minimum?.single_output_code,
    channel?.maximum?.single_output_code,
  ];
  return values.every((value) => Number.isInteger(value)) && new Set(values).size === 1 ? String(values[0]) : null;
}

function synthesisMetric(doc, label, value) {
  const metric = element(doc, "div", "quant-chain-synthesis-metric");
  metric.append(element(doc, "span", "", label), element(doc, "strong", "", value));
  return metric;
}

function coverageMetric(doc, label, value) {
  const metric = element(doc, "div", "quant-research-coverage-metric");
  metric.append(element(doc, "span", "", label), element(doc, "strong", "", value));
  return metric;
}

function element(doc, tag, className = "", text = "") {
  const node = doc.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function padOp(value) {
  return String(Number(value) || 0).padStart(3, "0");
}

function signed(value) {
  const number = Number(value);
  return number > 0 ? `+${number}` : String(number);
}

function formatPercent(value) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "N/A";
}

function formatInteger(value) {
  try {
    return BigInt(value).toLocaleString("en-US");
  } catch {
    return "N/A";
  }
}
