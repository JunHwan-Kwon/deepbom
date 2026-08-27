import { formatNumber, padOp } from "./format.js";
import { roundTiesAway } from "./quantization-math.js";

export const QUANTIZATION_LATTICE_SCHEMA = "deepbom.quantization_lattice.v1.4";

// Distance in output code steps from the unclamped ideal output code to the
// nearer representable end. A negative margin is exactly the escape
// condition, so one axis carries both containment and remaining headroom,
// and the axis is comparable across operators with different output scales.
export const MARGIN_BIN_EDGES = Object.freeze([
  -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64,
]);
const MARGIN_FINE_HALF_WIDTH = 320;

function marginBinIndex(margin) {
  let index = 0;
  while (index < MARGIN_BIN_EDGES.length && margin >= MARGIN_BIN_EDGES[index]) index += 1;
  return index;
}

export function createMarginAccumulator() {
  const fine = new Float64Array(MARGIN_FINE_HALF_WIDTH * 2 + 1);
  let below = 0;
  let above = 0;
  let total = 0;
  let minimum = Infinity;
  let escape = 0;
  let within1 = 0;
  let within2 = 0;
  return {
    observe(margin) {
      total += 1;
      if (margin < minimum) minimum = margin;
      if (margin < 0) escape += 1;
      if (margin < 1) within1 += 1;
      if (margin < 2) within2 += 1;
      const step = Math.floor(margin);
      if (step < -MARGIN_FINE_HALF_WIDTH) below += 1;
      else if (step > MARGIN_FINE_HALF_WIDTH) above += 1;
      else fine[step + MARGIN_FINE_HALF_WIDTH] += 1;
    },
    finish() {
      const percentile = (fraction) => {
        if (!total) return NaN;
        const target = Math.max(1, Math.ceil(fraction * total));
        let seen = below;
        if (seen >= target) return -MARGIN_FINE_HALF_WIDTH - 1;
        for (let index = 0; index < fine.length; index += 1) {
          seen += fine[index];
          if (seen >= target) return index - MARGIN_FINE_HALF_WIDTH;
        }
        return MARGIN_FINE_HALF_WIDTH + 1;
      };
      const bins = new Array(MARGIN_BIN_EDGES.length + 1).fill(0);
      bins[0] += below;
      bins[bins.length - 1] += above;
      for (let index = 0; index < fine.length; index += 1) {
        if (!fine[index]) continue;
        bins[marginBinIndex(index - MARGIN_FINE_HALF_WIDTH)] += fine[index];
      }
      return {
        bins,
        escape,
        within1,
        within2,
        minimum: Number.isFinite(minimum) ? minimum : 0,
        p1: percentile(0.01),
        p5: percentile(0.05),
        median: percentile(0.5),
      };
    },
  };
}

// Enumerable binary elementwise operators. The analyzer derives the same
// combination and legal range; this table exists so the viewer can re-derive
// every row independently rather than trusting the analyzer's own output.
export const BINARY_LATTICE_OPS = Object.freeze({
  ADD: {
    combine: (a, b) => a + b,
    legalRange: (l, r) => [l[0] + r[0], l[1] + r[1]],
    formula: "real=(q0-zp0)*s0+(q1-zp1)*s1; qout=clamp(round_ties_away(real/sout)+zpout)",
  },
  SUB: {
    combine: (a, b) => a - b,
    legalRange: (l, r) => [l[0] - r[1], l[1] - r[0]],
    formula: "real=(q0-zp0)*s0-(q1-zp1)*s1; qout=clamp(round_ties_away(real/sout)+zpout)",
  },
  MUL: {
    combine: (a, b) => a * b,
    // A product's extremes sit at the corners of the input rectangle; sign
    // combinations mean neither is simply low*low or high*high.
    legalRange: (l, r) => {
      const corners = [l[0] * r[0], l[0] * r[1], l[1] * r[0], l[1] * r[1]];
      return [Math.min(...corners), Math.max(...corners)];
    },
    formula: "real=((q0-zp0)*s0)*((q1-zp1)*s1); qout=clamp(round_ties_away(real/sout)+zpout)",
  },
  MAXIMUM: {
    combine: (a, b) => Math.max(a, b),
    legalRange: (l, r) => [Math.max(l[0], r[0]), Math.max(l[1], r[1])],
    formula: "real=max((q0-zp0)*s0,(q1-zp1)*s1); qout=clamp(round_ties_away(real/sout)+zpout)",
  },
  MINIMUM: {
    combine: (a, b) => Math.min(a, b),
    legalRange: (l, r) => [Math.min(l[0], r[0]), Math.min(l[1], r[1])],
    formula: "real=min((q0-zp0)*s0,(q1-zp1)*s1); qout=clamp(round_ties_away(real/sout)+zpout)",
  },
});

export const CONCAT_LATTICE_FORMULA =
  "per input i: real=(q-zp_i)*s_i; qout=clamp(round_ties_away(real/sout)+zpout)";

const METHOD_VERSION = "2026-07-17.3";
const MODES = new Set(["atlas", "projection", "escape", "error", "histogram", "design"]);
const LATTICE_FAMILIES = Object.freeze(["ADD", "SUB", "MUL", "MAXIMUM", "MINIMUM", "CONCATENATION"]);
const TILE_SIZE = 16;
const FLOAT64_BUFFER = new ArrayBuffer(8);
const FLOAT64_VIEW = new DataView(FLOAT64_BUFFER);

export function createQuantizationLatticeController({
  root,
  status,
  summary,
  body,
  downloadButton,
  getAnalysis,
  jumpToGraphOp,
  onDownload,
  residualSelection,
}) {
  let selectedOpIndex = null;
  let activeMode = "atlas";
  let activeFamily = "ADD";
  const selectionSource = "quantization-lattice";

  root?.addEventListener("click", (event) => {
    const familyButton = event.target.closest("[data-lattice-family]");
    if (familyButton && LATTICE_FAMILIES.includes(familyButton.dataset.latticeFamily)) {
      activeFamily = familyButton.dataset.latticeFamily;
      render(getAnalysis());
      return;
    }
    const modeButton = event.target.closest("[data-lattice-mode]");
    if (modeButton && MODES.has(modeButton.dataset.latticeMode)) {
      activeMode = modeButton.dataset.latticeMode;
      render(getAnalysis());
      return;
    }
    const opButton = event.target.closest("[data-lattice-op]");
    if (opButton) {
      selectedOpIndex = Number(opButton.dataset.latticeOp);
      residualSelection?.set({ opIndex: selectedOpIndex }, selectionSource);
      render(getAnalysis());
      return;
    }
    const graphButton = event.target.closest("[data-lattice-open-graph]");
    if (graphButton) jumpToGraphOp?.(Number(graphButton.dataset.latticeOpenGraph));
  });
  root?.addEventListener("keydown", (event) => {
    const familyButton = event.target.closest("[data-lattice-family]");
    if (familyButton && ["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const current = LATTICE_FAMILIES.indexOf(familyButton.dataset.latticeFamily);
      const next = event.key === "Home" ? 0
        : event.key === "End" ? LATTICE_FAMILIES.length - 1
          : (current + (event.key === "ArrowRight" ? 1 : -1) + LATTICE_FAMILIES.length) % LATTICE_FAMILIES.length;
      activeFamily = LATTICE_FAMILIES[next];
      render(getAnalysis());
      root.querySelector(`[data-lattice-family="${activeFamily}"]`)?.focus();
      return;
    }
    if ((event.key !== "Enter" && event.key !== " ") || event.target.matches("button")) return;
    const opRow = event.target.closest("[data-lattice-op]");
    if (!opRow) return;
    event.preventDefault();
    selectedOpIndex = Number(opRow.dataset.latticeOp);
    residualSelection?.set({ opIndex: selectedOpIndex }, selectionSource);
    render(getAnalysis());
  });
  residualSelection?.subscribe((selection, source) => {
    if (source === selectionSource) return;
    const analysis = getAnalysis?.();
    const assessed = (analysis?.quantization_lattice?.residual_adds || []).filter((row) => row.assessment_status === "assessed");
    if (assessed.some((row) => row.op_index === selection.opIndex)) {
      selectedOpIndex = selection.opIndex;
      activeFamily = "ADD";
      render(analysis);
    }
  });
  downloadButton?.addEventListener("click", () => {
    const result = getAnalysis()?.quantization_lattice;
    if (result) onDownload?.(result, "quantization_lattice.json");
  });

  function render(analysis) {
    if (!root || !status || !summary || !body) return;
    const result = analysis?.quantization_lattice;
    if (!result) {
      selectedOpIndex = null;
      status.textContent = analysis ? "not assessed" : "run an audit";
      summary.replaceChildren();
      body.replaceChildren(emptyState("Quantization-lattice evidence is unavailable."));
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    validateQuantizationLattice(result, analysis);
    const assessed = result.residual_adds.filter((row) => row.assessment_status === "assessed");
    const shared = residualSelection?.get();
    if (assessed.some((row) => row.op_index === shared?.opIndex)) selectedOpIndex = shared.opIndex;
    if (!assessed.some((row) => row.op_index === selectedOpIndex)) {
      selectedOpIndex = result.domain_escape_ranking_op_indices[0] ?? assessed[0]?.op_index ?? null;
    }
    residualSelection?.set({ opIndex: selectedOpIndex }, selectionSource);
    status.textContent = `${result.assessed_operator_count}/${result.candidate_operator_count} ops / ${formatNumber(result.total_enumerated_code_pairs)} binary pairs / ${formatNumber(result.total_enumerated_concatenation_codes)} concat codes`;
    summary.replaceChildren(
      metric("Assessed operators", `${result.assessed_operator_count} / ${result.candidate_operator_count}`, "six enumerable operator families"),
      metric("Binary code pairs", formatNumber(result.total_enumerated_code_pairs), "65,536 per assessed binary op"),
      metric("CONCAT codes", formatNumber(result.total_enumerated_concatenation_codes), "256 per assessed input branch"),
      metric("Unassessed operators", formatNumber(result.candidate_operator_count - result.assessed_operator_count), "serialized candidate with incomplete contract"),
    );
    const families = familyTabs(result, activeFamily);
    if (activeFamily !== "ADD") {
      const rows = familyRows(result, activeFamily);
      body.replaceChildren(
        families,
        familyScope(activeFamily),
        ...(rows.length
          ? [activeFamily === "CONCATENATION" ? concatenationTable(rows) : genericBinaryTable(rows, activeFamily)]
          : [familyEmptyState(activeFamily)]),
        methodBoundary(result),
      );
      if (downloadButton) downloadButton.disabled = false;
      return;
    }
    const addSummary = element("div", "lattice-detail-grid lattice-family-summary");
    addSummary.replaceChildren(
      metric("Assessed ADD", `${result.assessed_add_count} / ${result.candidate_add_count}`, `${result.unassessed_add_count} unassessed`),
      metric("Complete domain", `${result.complete_domain_containment_add_count}`, `${result.range_escape_add_count} with endpoint escape`),
      metric("Maximum escape", percent(result.maximum_range_escape_pair_ratio), "uniform legal-code pairs"),
      metric("Containment design", `${result.containment_design_add_count} / ${result.assessed_add_count}`, `${result.global_zero_point_shift_add_count} require zero-point shift`),
    );
    if (!assessed.length) {
      body.replaceChildren(
        families,
        familyScope("ADD"),
        addSummary,
        ...(result.residual_adds.length ? [unassessedTable(result.residual_adds)] : []),
        ...(!result.residual_adds.length ? [familyEmptyState("ADD")] : []),
        methodBoundary(result),
      );
      if (downloadButton) downloadButton.disabled = false;
      return;
    }
    const row = assessed.find((item) => item.op_index === selectedOpIndex) || assessed[0];
    const unassessedCoverage = unassessedCoverageSummary(result, analysis);
    body.replaceChildren(
      families,
      familyScope("ADD"),
      addSummary,
      selector(assessed, row.op_index),
      modeTabs(activeMode),
      activeMode === "atlas" ? marginAtlasView(result, row.op_index)
        : activeMode === "projection" ? marginProjectionView(row)
        : latticeWorkspace(row, activeMode, jumpToGraphOp),
      portfolioTable(result.residual_adds, row.op_index),
      ...(unassessedCoverage ? [unassessedCoverage] : []),
      methodBoundary(result),
    );
    if (downloadButton) downloadButton.disabled = false;
  }

  return { render };
}

function unassessedCoverageSummary(result, analysis) {
  const rows = (result.residual_adds || []).filter((row) => row.assessment_status !== "assessed");
  if (!rows.length) return null;
  const contextsByOp = new Map();
  for (const block of analysis?.block_inventory?.blocks || []) {
    for (const opIndex of block.op_indices || []) {
      const contexts = contextsByOp.get(Number(opIndex)) || new Set();
      contexts.add(block.block_type || "unclassified");
      contextsByOp.set(Number(opIndex), contexts);
    }
  }
  const groups = new Map();
  for (const row of rows) {
    const reason = row.not_assessed_reason || "Reason not emitted";
    const group = groups.get(reason) || { reason, indices: [], contexts: new Set() };
    group.indices.push(row.op_index);
    for (const context of contextsByOp.get(Number(row.op_index)) || ["unclassified"]) group.contexts.add(context);
    groups.set(reason, group);
  }
  const details = element("details", "lattice-method-details");
  details.open = true;
  details.append(element("summary", "", `${formatNumber(rows.length)} ADD exclusion(s): denominator and structural contexts`));
  const table = element("table", "lattice-portfolio-table");
  table.innerHTML = "<thead><tr><th>Exclusion reason</th><th>Count / ops</th><th>Detected block contexts</th></tr></thead>";
  const body = element("tbody");
  for (const group of groups.values()) {
    const row = element("tr");
    row.append(
      element("td", "", group.reason),
      element("td", "numeric", `${formatNumber(group.indices.length)}: ${group.indices.map((index) => `#${padOp(index)}`).join(", ")}`),
      element("td", "", [...group.contexts].join(" / ")),
    );
    body.append(row);
  }
  table.append(body);
  details.append(table, element("p", "lattice-method-line", "Exclusions are grouped by the exact unassessed reason and detected graph context. Context groups may overlap and are not assumed to be squeeze-excitation blocks."));
  return details;
}

export function reconstructConcatLatticeRow(op, tensors) {
  if (!(op.inputs || []).length || !(op.outputs || []).length) {
    return { status: "not_assessed", reason: "CONCATENATION does not expose inputs and one output in the parsed graph." };
  }
  if (op.fused_activation !== "NONE") {
    return { status: "not_assessed", reason: `Fused activation ${op.fused_activation} is not modeled by quantization-lattice v1.` };
  }
  const outputTensor = tensorAt(tensors, op.outputs[0]);
  if (!outputTensor) return { status: "not_assessed", reason: "Output tensor is unavailable." };
  const output = quantContract(outputTensor);
  if (output.error) return { status: "not_assessed", reason: output.error };
  const inputs = [];
  for (const index of op.inputs) {
    const tensor = tensorAt(tensors, index);
    if (!tensor) return { status: "not_assessed", reason: `Input tensor ${index} is unavailable.` };
    const contract = quantContract(tensor);
    if (contract.error) return { status: "not_assessed", reason: contract.error };
    inputs.push(contract);
  }
  const outputRealRange = realRange(output);
  let totalCodes = 0;
  let totalEscapes = 0;
  let totalClamps = 0;
  let errorStepSum = 0;
  let errorStepMax = 0;
  const projections = inputs.map((input, position) => {
    const used = new Set();
    let escapes = 0;
    let clamps = 0;
    let sum = 0;
    let max = 0;
    let count = 0;
    for (let code = input.qmin; code <= input.qmax; code += 1) {
      const real = (code - input.zeroPoint) * input.scale;
      if (real < outputRealRange[0] || real > outputRealRange[1]) escapes += 1;
      const rounded = roundTiesAway(real / output.scale) + output.zeroPoint;
      if (rounded < output.qmin || rounded > output.qmax) clamps += 1;
      const projected = Math.max(output.qmin, Math.min(output.qmax, rounded));
      used.add(projected);
      const steps = Math.abs(real - (projected - output.zeroPoint) * output.scale) / output.scale;
      sum += steps;
      max = Math.max(max, steps);
      count += 1;
    }
    totalCodes += count;
    totalEscapes += escapes;
    totalClamps += clamps;
    errorStepSum += sum;
    errorStepMax = Math.max(errorStepMax, max);
    return {
      input_position: position,
      enumerated_code_count: count,
      range_escape_code_count: escapes,
      rounded_projection_clamp_code_count: clamps,
      distinct_projected_output_code_count: used.size,
      legal_domain_contained: escapes === 0,
    };
  });
  return {
    status: "assessed",
    input_count: inputs.length,
    enumerated_code_count: totalCodes,
    range_escape_code_count: totalEscapes,
    rounded_projection_clamp_code_count: totalClamps,
    complete_legal_domain_contained: totalEscapes === 0,
    maximum_absolute_projection_error_output_steps: errorStepMax,
    projections,
  };
}

/// The new operator families are held to the same standard as ADD: every row is
/// re-derived here and compared, so nothing reaches a report unvalidated.
function validateWidenedLattice(result, analysis) {
  const ops = analysis?.ops || [];
  const binaryCandidates = ops.filter((op) => op.name !== "ADD" && BINARY_LATTICE_OPS[op.name]);
  const rows = result.binary_contracts || [];
  if (rows.length !== binaryCandidates.length || result.candidate_binary_count !== binaryCandidates.length) {
    throw new Error("Quantization-lattice binary-operator coverage is invalid.");
  }
  let assessedBinary = 0;
  let totalBinaryPairs = 0;
  let escapedBinary = 0;
  let containedBinary = 0;
  for (let position = 0; position < binaryCandidates.length; position += 1) {
    const op = binaryCandidates[position];
    const row = rows[position];
    if (row.op_index !== op.index || row.op_name !== op.name) {
      throw new Error(`Quantization-lattice binary op binding is invalid at row ${position}.`);
    }
    if (row.formula !== BINARY_LATTICE_OPS[op.name].formula) {
      throw new Error(`Quantization-lattice formula is invalid for ${op.name} #${op.index}.`);
    }
    const expected = reconstructResidualLatticeRow(op, analysis?.tensors || []);
    if (expected.status !== "assessed") {
      if (row.assessment_status !== "not_assessed" || row.not_assessed_reason !== expected.reason) {
        throw new Error(`Quantization-lattice binary unassessed state is invalid at #${op.index}.`);
      }
      continue;
    }
    validateAssessedRow(row, expected, op.index);
    assessedBinary += 1;
    totalBinaryPairs += row.enumerated_code_pair_count;
    if (row.range_escape_pair_count > 0) escapedBinary += 1;
    if (row.complete_legal_domain_contained) containedBinary += 1;
  }
  if (result.assessed_binary_count !== assessedBinary
    || result.unassessed_binary_count !== binaryCandidates.length - assessedBinary
    || result.binary_status !== familyStatus(binaryCandidates.length, assessedBinary)
    || result.range_escape_binary_count !== escapedBinary
    || result.complete_domain_containment_binary_count !== containedBinary) {
    throw new Error("Quantization-lattice binary counts are invalid.");
  }
  const coverage = result.binary_operator_coverage || [];
  const names = [...new Set(binaryCandidates.map((op) => op.name))].sort();
  if (coverage.length !== names.length || coverage.some((entry, index) => entry.op_name !== names[index])) {
    throw new Error("Quantization-lattice binary coverage summary is invalid.");
  }
  for (const entry of coverage) {
    const candidate = binaryCandidates.filter((op) => op.name === entry.op_name).length;
    const assessed = rows.filter((row) => row.op_name === entry.op_name && row.assessment_status === "assessed").length;
    if (entry.candidate_count !== candidate || entry.assessed_count !== assessed
      || entry.unassessed_count !== candidate - assessed) {
      throw new Error(`Quantization-lattice coverage summary is invalid for ${entry.op_name}.`);
    }
  }

  const concatCandidates = ops.filter((op) => op.name === "CONCATENATION");
  const concatRows = result.concatenation_contracts || [];
  if (concatRows.length !== concatCandidates.length
    || result.candidate_concatenation_count !== concatCandidates.length) {
    throw new Error("Quantization-lattice CONCATENATION coverage is invalid.");
  }
  let assessedConcat = 0;
  let escapeOps = 0;
  let totalConcatCodes = 0;
  for (let position = 0; position < concatCandidates.length; position += 1) {
    const op = concatCandidates[position];
    const row = concatRows[position];
    if (row.op_index !== op.index || row.formula !== CONCAT_LATTICE_FORMULA) {
      throw new Error(`Quantization-lattice CONCATENATION binding is invalid at row ${position}.`);
    }
    const expected = reconstructConcatLatticeRow(op, analysis?.tensors || []);
    if (expected.status !== "assessed") {
      if (row.assessment_status !== "not_assessed" || row.not_assessed_reason !== expected.reason) {
        throw new Error(`Quantization-lattice CONCATENATION unassessed state is invalid at #${op.index}.`);
      }
      continue;
    }
    if (row.input_count !== expected.input_count
      || row.enumerated_code_count !== expected.enumerated_code_count
      || row.range_escape_code_count !== expected.range_escape_code_count
      || row.rounded_projection_clamp_code_count !== expected.rounded_projection_clamp_code_count
      || row.complete_legal_domain_contained !== expected.complete_legal_domain_contained
      || !closeEnough(row.maximum_absolute_projection_error_output_steps, expected.maximum_absolute_projection_error_output_steps)) {
      throw new Error(`Quantization-lattice CONCATENATION projection is invalid at #${op.index}.`);
    }
    if ((row.inputs || []).length !== expected.projections.length) {
      throw new Error(`Quantization-lattice CONCATENATION input coverage is invalid at #${op.index}.`);
    }
    for (let index = 0; index < expected.projections.length; index += 1) {
      const actual = row.inputs[index];
      const want = expected.projections[index];
      if (actual.input_position !== want.input_position
        || actual.enumerated_code_count !== want.enumerated_code_count
        || actual.range_escape_code_count !== want.range_escape_code_count
        || actual.rounded_projection_clamp_code_count !== want.rounded_projection_clamp_code_count
        || actual.distinct_projected_output_code_count !== want.distinct_projected_output_code_count
        || actual.legal_domain_contained !== want.legal_domain_contained) {
        throw new Error(`Quantization-lattice CONCATENATION input ${index} is invalid at #${op.index}.`);
      }
    }
    if (row.range_escape_code_count > 0) escapeOps += 1;
    totalConcatCodes += row.enumerated_code_count;
    assessedConcat += 1;
  }
  if (result.assessed_concatenation_count !== assessedConcat
    || result.unassessed_concatenation_count !== concatCandidates.length - assessedConcat
    || result.concatenation_status !== familyStatus(concatCandidates.length, assessedConcat)
    || result.concatenation_range_escape_count !== escapeOps) {
    throw new Error("Quantization-lattice CONCATENATION counts are invalid.");
  }
  return {
    binaryCandidateCount: binaryCandidates.length,
    assessedBinary,
    totalBinaryPairs,
    concatCandidateCount: concatCandidates.length,
    assessedConcat,
    totalConcatCodes,
    binaryRows: rows.filter((row) => row.assessment_status === "assessed"),
  };
}

function familyStatus(candidateCount, assessedCount) {
  if (!candidateCount) return "not_applicable";
  return assessedCount === candidateCount ? "assessed" : "partial";
}

function closeEnough(actual, expected) {
  const left = Number(actual);
  const right = Number(expected);
  if (!Number.isFinite(left) || !Number.isFinite(right)) return false;
  return Math.abs(left - right) <= Math.max(1e-9, Math.abs(right) * 1e-9);
}

export function validateQuantizationLattice(result, analysis) {
  if (!result || result.schema !== QUANTIZATION_LATTICE_SCHEMA) throw new Error("Quantization-lattice schema is invalid.");
  if (result.method_version !== METHOD_VERSION) throw new Error("Quantization-lattice method version is invalid.");
  if (result.evidence_class !== "DERIVED") throw new Error("Quantization-lattice evidence class is invalid.");
  const candidates = (analysis?.ops || []).filter((op) => op.name === "ADD");
  if (!Array.isArray(result.residual_adds) || result.residual_adds.length !== candidates.length
    || result.candidate_add_count !== candidates.length) throw new Error("Quantization-lattice ADD coverage is invalid.");
  const assessed = [];
  for (let position = 0; position < candidates.length; position += 1) {
    const op = candidates[position];
    const row = result.residual_adds[position];
    if (row.op_index !== op.index || row.op_name !== op.name || row.fused_activation !== op.fused_activation) {
      throw new Error(`Quantization-lattice op binding is invalid at row ${position}.`);
    }
    const expected = reconstructResidualLatticeRow(op, analysis?.tensors || []);
    if (expected.status !== "assessed") {
      if (row.assessment_status !== "not_assessed" || row.not_assessed_reason !== expected.reason
        || row.enumerated_code_pair_count != null) throw new Error(`Quantization-lattice unassessed state is invalid at #${op.index}.`);
      continue;
    }
    validateAssessedRow(row, expected, op.index);
    assessed.push(row);
  }
  const widened = validateWidenedLattice(result, analysis);
  const ranked = [...assessed].sort(latticeComparator);
  if (!sameArray(result.domain_escape_ranking_op_indices, ranked.map((row) => row.op_index))) {
    throw new Error("Quantization-lattice ranking is invalid.");
  }
  ranked.forEach((row, index) => {
    if (row.domain_escape_rank !== index + 1) throw new Error(`Quantization-lattice rank is invalid at #${row.op_index}.`);
  });
  const residualPairs = sum(assessed.map((row) => row.enumerated_code_pair_count));
  const totalPairs = residualPairs + widened.totalBinaryPairs;
  const escapeAdds = assessed.filter((row) => row.range_escape_pair_count > 0).length;
  const contained = assessed.filter((row) => row.complete_legal_domain_contained).length;
  const containmentDesigns = assessed.filter((row) => row.globally_finest_containment).length;
  const fixedDesigns = assessed.filter((row) => row.fixed_zero_point_containment).length;
  const fixedExpansion = assessed.filter((row) => Number(row.fixed_zero_point_containment?.scale_ratio_to_current) > 1).length;
  const globalShift = assessed.filter((row) => Number(row.globally_finest_containment?.absolute_zero_point_shift) > 0).length;
  if (result.assessed_add_count !== assessed.length || result.unassessed_add_count !== candidates.length - assessed.length
    || result.residual_add_status !== familyStatus(candidates.length, assessed.length)
    || result.residual_add_enumerated_code_pairs !== residualPairs
    || result.total_enumerated_code_pairs !== totalPairs
    || result.total_enumerated_concatenation_codes !== widened.totalConcatCodes
    || result.range_escape_add_count !== escapeAdds
    || result.complete_domain_containment_add_count !== contained
    || result.containment_design_add_count !== containmentDesigns
    || result.fixed_zero_point_containment_add_count !== fixedDesigns
    || result.fixed_zero_point_scale_expansion_add_count !== fixedExpansion
    || result.global_zero_point_shift_add_count !== globalShift) throw new Error("Quantization-lattice summary counts are invalid.");
  const allBinaryRows = [...assessed, ...widened.binaryRows];
  assertOptionalNear(result.maximum_range_escape_pair_ratio, maxOptional(allBinaryRows.map((row) => row.range_escape_pair_ratio)), "maximum range escape ratio");
  assertOptionalNear(result.maximum_mean_clamped_projection_error_steps, maxOptional(allBinaryRows.map((row) => row.mean_clamped_projection_error_steps)), "maximum mean projection error");
  assertOptionalNear(result.maximum_fixed_zero_point_scale_ratio, maxOptional(assessed.map((row) => row.fixed_zero_point_containment?.scale_ratio_to_current).filter((value) => value != null)), "maximum fixed-zero-point scale ratio");
  assertOptionalNear(result.maximum_global_finest_scale_ratio, maxOptional(assessed.map((row) => row.globally_finest_containment?.scale_ratio_to_current).filter((value) => value != null)), "maximum global-finest scale ratio");
  const candidateOperatorCount = candidates.length + widened.binaryCandidateCount + widened.concatCandidateCount;
  const assessedOperatorCount = assessed.length + widened.assessedBinary + widened.assessedConcat;
  const expectedStatus = familyStatus(candidateOperatorCount, assessedOperatorCount);
  if (result.candidate_operator_count !== candidateOperatorCount
    || result.assessed_operator_count !== assessedOperatorCount
    || result.unassessed_operator_count !== candidateOperatorCount - assessedOperatorCount
    || result.status !== expectedStatus || !/not over observed activation values/i.test(result.interpretation_boundary || "")
    || !/256x256 input-code Cartesian product/i.test(result.method || "")
    || !/counterfactual output contracts/i.test(result.interpretation_boundary || "")
    || !/min_binary64/.test(result.containment_formula || "")
    || !/ties_away_from_zero/i.test(result.rounding_rule || "")) throw new Error("Quantization-lattice method boundary is incomplete.");
  return true;
}

export function reconstructResidualLatticeRow(op, tensors) {
  const kind = BINARY_LATTICE_OPS[op.name];
  if (!kind) return { status: "not_assessed", reason: `${op.name} is not an enumerable binary operator.` };
  if ((op.inputs || []).length < 2 || !(op.outputs || []).length) {
    return { status: "not_assessed", reason: `${op.name} does not expose two inputs and one output in the parsed graph.` };
  }
  if (op.fused_activation !== "NONE") {
    return { status: "not_assessed", reason: `Fused activation ${op.fused_activation} is not modeled by quantization-lattice v1.` };
  }
  const input0 = tensorAt(tensors, op.inputs[0]);
  const input1 = tensorAt(tensors, op.inputs[1]);
  const output = tensorAt(tensors, op.outputs[0]);
  if (!input0) return { status: "not_assessed", reason: "Input tensor 0 is unavailable." };
  if (!input1) return { status: "not_assessed", reason: "Input tensor 1 is unavailable." };
  if (!output) return { status: "not_assessed", reason: "Output tensor is unavailable." };
  const contracts = [input0, input1, output].map(quantContract);
  const failure = contracts.find((item) => item.error);
  if (failure) return { status: "not_assessed", reason: failure.error };
  return enumerateLattice(contracts[0], contracts[1], contracts[2], kind);
}

function enumerateLattice(input0, input1, output, kind) {
  const marginAccumulator = createMarginAccumulator();
  const outputWidth = output.qmax - output.qmin + 1;
  const grid = outputWidth / TILE_SIZE;
  const histogram = Array(outputWidth).fill(0);
  const tileEscape = Array(grid * grid).fill(0);
  const tileErrorSums = Array(grid * grid).fill(0);
  const outputRealRange = realRange(output);
  let low = 0;
  let high = 0;
  let clamps = 0;
  let inRangeCount = 0;
  let inRangeErrorSum = 0;
  let inRangeErrorMax = 0;
  let projectionErrorSum = 0;
  let projectionErrorMax = 0;
  let worst = null;
  for (let q0 = input0.qmin; q0 <= input0.qmax; q0 += 1) {
    for (let q1 = input1.qmin; q1 <= input1.qmax; q1 += 1) {
      const realSum = kind.combine((q0 - input0.zeroPoint) * input0.scale, (q1 - input1.zeroPoint) * input1.scale);
      let rangeClass = "inside_output_range";
      if (realSum < outputRealRange[0]) { low += 1; rangeClass = "below_output_range"; }
      else if (realSum > outputRealRange[1]) { high += 1; rangeClass = "above_output_range"; }
      else inRangeCount += 1;
      const rounded = roundTiesAway(realSum / output.scale) + output.zeroPoint;
      if (rounded < output.qmin || rounded > output.qmax) clamps += 1;
      const projectedCode = Math.max(output.qmin, Math.min(output.qmax, rounded));
      const projectedReal = (projectedCode - output.zeroPoint) * output.scale;
      const error = Math.abs(realSum - projectedReal);
      const errorSteps = error / output.scale;
      projectionErrorSum += error;
      projectionErrorMax = Math.max(projectionErrorMax, error);
      if (rangeClass === "inside_output_range") {
        inRangeErrorSum += error;
        inRangeErrorMax = Math.max(inRangeErrorMax, error);
      }
      const idealOutputCode = realSum / output.scale + output.zeroPoint;
      marginAccumulator.observe(Math.min(idealOutputCode - output.qmin, output.qmax - idealOutputCode));
      histogram[projectedCode - output.qmin] += 1;
      const tileIndex = Math.floor((q0 - input0.qmin) / TILE_SIZE) * grid + Math.floor((q1 - input1.qmin) / TILE_SIZE);
      if (rangeClass !== "inside_output_range") tileEscape[tileIndex] += 1;
      tileErrorSums[tileIndex] += errorSteps;
      if (!worst || error > worst.absolute_error) {
        worst = {
          input_0_code: q0, input_1_code: q1, real_sum: realSum,
          rounded_unclamped_output_code: rounded, projected_output_code: projectedCode,
          projected_real_value: projectedReal, absolute_error: error,
          absolute_error_output_steps: errorSteps, range_class: rangeClass,
        };
      }
    }
  }
  const pairs = (input0.qmax - input0.qmin + 1) * (input1.qmax - input1.qmin + 1);
  const input0Range = realRange(input0);
  const input1Range = realRange(input1);
  const sumRange = kind.legalRange(input0Range, input1Range);
  const containmentDesign = buildContainmentDesign(input0, input1, output, sumRange, kind);
  const sumWidth = sumRange[1] - sumRange[0];
  const intersection = Math.max(0, Math.min(sumRange[1], outputRealRange[1]) - Math.max(sumRange[0], outputRealRange[0]));
  const escape = low + high;
  const finest = Math.min(input0.scale, input1.scale);
  const coarsest = Math.max(input0.scale, input1.scale);
  return {
    status: "assessed",
    contracts: [input0, input1, output],
    inputRealRanges: [input0Range, input1Range],
    sumRange,
    outputRealRange,
    continuousCoverage: sumWidth > 0 ? intersection / sumWidth : (sumRange[0] >= outputRealRange[0] && sumRange[0] <= outputRealRange[1] ? 1 : 0),
    inputScaleRatio: coarsest / finest,
    outputToFinest: output.scale / finest,
    outputToCoarsest: output.scale / coarsest,
    pairs, low, high, escape, clamps, inRangeCount,
    margin: marginAccumulator.finish(),
    histogram, tileEscape,
    tileMeanError: tileErrorSums.map((value) => value / (TILE_SIZE * TILE_SIZE)),
    distinctOutputCodes: histogram.filter(Boolean).length,
    meanInRangeError: inRangeCount ? inRangeErrorSum / inRangeCount : null,
    maxInRangeError: inRangeCount ? inRangeErrorMax : null,
    meanProjectionError: projectionErrorSum / pairs,
    maxProjectionError: projectionErrorMax,
    worst,
    containmentCandidateCount: containmentDesign.candidateCount,
    containmentFrontier: containmentDesign.frontier,
    fixedZeroPointContainment: containmentDesign.fixedZeroPoint,
    globallyFinestContainment: containmentDesign.globallyFinest,
  };
}

function buildContainmentDesign(input0, input1, currentOutput, legalSumRange, kind) {
  const candidates = [];
  for (let zeroPoint = currentOutput.qmin; zeroPoint <= currentOutput.qmax; zeroPoint += 1) {
    const scale = minimumContainmentScale(legalSumRange, currentOutput.qmin, currentOutput.qmax, zeroPoint);
    if (scale == null) continue;
    const signedDelta = zeroPoint - currentOutput.zeroPoint;
    candidates.push({
      output_zero_point: zeroPoint,
      minimum_output_scale: scale,
      scale_ratio_to_current: scale / currentOutput.scale,
      signed_zero_point_delta: signedDelta,
      absolute_zero_point_shift: Math.abs(signedDelta),
      negative_code_capacity: zeroPoint - currentOutput.qmin,
      positive_code_capacity: currentOutput.qmax - zeroPoint,
    });
  }
  const globallyFinestPoint = [...candidates].sort((left, right) =>
    left.minimum_output_scale - right.minimum_output_scale
    || left.absolute_zero_point_shift - right.absolute_zero_point_shift
    || left.output_zero_point - right.output_zero_point)[0] || null;
  const fixedPoint = candidates.find((candidate) => candidate.output_zero_point === currentOutput.zeroPoint) || null;
  const ordered = [...candidates].sort((left, right) =>
    left.absolute_zero_point_shift - right.absolute_zero_point_shift
    || left.minimum_output_scale - right.minimum_output_scale
    || left.output_zero_point - right.output_zero_point);
  const frontier = [];
  let bestScale = Number.POSITIVE_INFINITY;
  for (const candidate of ordered) {
    if (candidate.minimum_output_scale < bestScale) {
      frontier.push(candidate);
      bestScale = candidate.minimum_output_scale;
    }
  }
  return {
    candidateCount: candidates.length,
    frontier,
    fixedZeroPoint: fixedPoint ? evaluateContainmentCandidate("fixed_zero_point_minimum_containment", fixedPoint, input0, input1, currentOutput, kind) : null,
    globallyFinest: globallyFinestPoint ? evaluateContainmentCandidate("globally_finest_minimum_containment", globallyFinestPoint, input0, input1, currentOutput, kind) : null,
  };
}

function minimumContainmentScale(legalSumRange, qmin, qmax, zeroPoint) {
  const negativeCapacity = zeroPoint - qmin;
  const positiveCapacity = qmax - zeroPoint;
  let lowerRequirement = 0;
  let upperRequirement = 0;
  if (legalSumRange[0] < 0) {
    if (negativeCapacity === 0) return null;
    lowerRequirement = -legalSumRange[0] / negativeCapacity;
  }
  if (legalSumRange[1] > 0) {
    if (positiveCapacity === 0) return null;
    upperRequirement = legalSumRange[1] / positiveCapacity;
  }
  let scale = Math.max(lowerRequirement, upperRequirement);
  if (scale === 0) scale = Number.MIN_VALUE;
  if (!(Number.isFinite(scale) && scale > 0)) return null;
  while (!contractContainsRange(legalSumRange, qmin, qmax, zeroPoint, scale)) {
    const next = nextUpPositive(scale);
    if (!Number.isFinite(next) || next === scale) return null;
    scale = next;
  }
  while (true) {
    const previous = nextDownPositive(scale);
    if (!(previous > 0) || !contractContainsRange(legalSumRange, qmin, qmax, zeroPoint, previous)) break;
    scale = previous;
  }
  return scale;
}

function contractContainsRange(legalSumRange, qmin, qmax, zeroPoint, scale) {
  return (qmin - zeroPoint) * scale <= legalSumRange[0]
    && (qmax - zeroPoint) * scale >= legalSumRange[1];
}

function nextUpPositive(value) {
  FLOAT64_VIEW.setFloat64(0, value, false);
  FLOAT64_VIEW.setBigUint64(0, FLOAT64_VIEW.getBigUint64(0, false) + 1n, false);
  return FLOAT64_VIEW.getFloat64(0, false);
}

function nextDownPositive(value) {
  FLOAT64_VIEW.setFloat64(0, value, false);
  FLOAT64_VIEW.setBigUint64(0, FLOAT64_VIEW.getBigUint64(0, false) - 1n, false);
  return FLOAT64_VIEW.getFloat64(0, false);
}

function evaluateContainmentCandidate(design, point, input0, input1, currentOutput, kind) {
  const candidate = { ...currentOutput, scale: point.minimum_output_scale, zeroPoint: point.output_zero_point };
  const usedCodes = new Set();
  let clampCount = 0;
  let errorSum = 0;
  let maximumError = 0;
  let pairCount = 0;
  for (let q0 = input0.qmin; q0 <= input0.qmax; q0 += 1) {
    for (let q1 = input1.qmin; q1 <= input1.qmax; q1 += 1) {
      const realSum = kind.combine((q0 - input0.zeroPoint) * input0.scale, (q1 - input1.zeroPoint) * input1.scale);
      const rounded = roundTiesAway(realSum / candidate.scale) + candidate.zeroPoint;
      if (rounded < candidate.qmin || rounded > candidate.qmax) clampCount += 1;
      const projectedCode = Math.max(candidate.qmin, Math.min(candidate.qmax, rounded));
      usedCodes.add(projectedCode);
      const projectedReal = (projectedCode - candidate.zeroPoint) * candidate.scale;
      const error = Math.abs(realSum - projectedReal);
      errorSum += error;
      maximumError = Math.max(maximumError, error);
      pairCount += 1;
    }
  }
  const meanError = errorSum / pairCount;
  const outputWidth = candidate.qmax - candidate.qmin + 1;
  return {
    design,
    output_zero_point: candidate.zeroPoint,
    output_scale: candidate.scale,
    scale_ratio_to_current: candidate.scale / currentOutput.scale,
    signed_zero_point_delta: point.signed_zero_point_delta,
    absolute_zero_point_shift: point.absolute_zero_point_shift,
    output_real_range: realRange(candidate),
    rounded_projection_clamp_pair_count: clampCount,
    distinct_projected_output_code_count: usedCodes.size,
    projected_output_code_utilization_ratio: usedCodes.size / outputWidth,
    mean_absolute_projection_error: meanError,
    mean_absolute_projection_error_current_steps: meanError / currentOutput.scale,
    mean_absolute_projection_error_candidate_steps: meanError / candidate.scale,
    maximum_absolute_projection_error: maximumError,
    maximum_absolute_projection_error_current_steps: maximumError / currentOutput.scale,
    maximum_absolute_projection_error_candidate_steps: maximumError / candidate.scale,
  };
}

function sameMarginProfile(actual, expected) {
  if (!actual || !expected) return false;
  if (!sameArray(actual.bin_edges_output_code_steps, MARGIN_BIN_EDGES)) return false;
  if (!sameArray(actual.bin_pair_counts, expected.bins)) return false;
  return actual.escape_pair_count === expected.escape
    && actual.boundary_pressure_1_step_pair_count === expected.within1
    && actual.boundary_pressure_2_step_pair_count === expected.within2
    && closeEnough(actual.minimum_margin_output_code_steps, expected.minimum)
    && actual.percentile_1_margin_output_code_steps === expected.p1
    && actual.percentile_5_margin_output_code_steps === expected.p5
    && actual.median_margin_output_code_steps === expected.median;
}

function validateAssessedRow(row, expected, opIndex) {
  const [input0, input1, output] = expected.contracts;
  if (row.assessment_status !== "assessed" || row.not_assessed_reason !== ""
    || !sameArray(row.input_tensor_indices, [input0.tensorIndex, input1.tensorIndex])
    || row.output_tensor_index !== output.tensorIndex
    || !sameArray(row.input_code_ranges, [[input0.qmin, input0.qmax], [input1.qmin, input1.qmax]])
    || !sameArray(row.output_code_range, [output.qmin, output.qmax])) throw new Error(`Quantization-lattice tensor contract is invalid at #${opIndex}.`);
  assertArrayNear(row.input_scales, [input0.scale, input1.scale], `#${opIndex} input scales`);
  assertNear(row.output_scale, output.scale, `#${opIndex} output scale`);
  if (!sameArray(row.input_zero_points, [input0.zeroPoint, input1.zeroPoint]) || row.output_zero_point !== output.zeroPoint) {
    throw new Error(`Quantization-lattice zero-points are invalid at #${opIndex}.`);
  }
  assertNestedArrayNear(row.input_real_ranges, expected.inputRealRanges, `#${opIndex} input real ranges`);
  assertArrayNear(row.legal_sum_real_range, expected.sumRange, `#${opIndex} legal sum range`);
  assertArrayNear(row.output_real_range, expected.outputRealRange, `#${opIndex} output real range`);
  assertNear(row.continuous_sum_interval_coverage_ratio, expected.continuousCoverage, `#${opIndex} continuous coverage`);
  assertNear(row.input_scale_ratio, expected.inputScaleRatio, `#${opIndex} input scale ratio`);
  assertNear(row.output_to_finest_input_step_ratio, expected.outputToFinest, `#${opIndex} output/finest step`);
  assertNear(row.output_to_coarsest_input_step_ratio, expected.outputToCoarsest, `#${opIndex} output/coarsest step`);
  const fields = {
    enumerated_code_pair_count: expected.pairs,
    range_escape_low_pair_count: expected.low,
    range_escape_high_pair_count: expected.high,
    range_escape_pair_count: expected.escape,
    rounded_projection_clamp_pair_count: expected.clamps,
    distinct_projected_output_code_count: expected.distinctOutputCodes,
  };
  for (const [field, value] of Object.entries(fields)) if (row[field] !== value) throw new Error(`Quantization-lattice ${field} is invalid at #${opIndex}.`);
  assertNear(row.range_escape_pair_ratio, expected.escape / expected.pairs, `#${opIndex} escape ratio`);
  assertNear(row.rounded_projection_clamp_pair_ratio, expected.clamps / expected.pairs, `#${opIndex} clamp ratio`);
  if (row.complete_legal_domain_contained !== (expected.escape === 0)) throw new Error(`Quantization-lattice domain containment is invalid at #${opIndex}.`);
  assertNear(row.projected_output_code_utilization_ratio, expected.distinctOutputCodes / (output.qmax - output.qmin + 1), `#${opIndex} code utilization`);
  assertOptionalNear(row.mean_in_range_rounding_error, expected.meanInRangeError, `#${opIndex} mean in-range error`);
  assertOptionalNear(row.mean_in_range_rounding_error_steps, expected.meanInRangeError == null ? null : expected.meanInRangeError / output.scale, `#${opIndex} mean in-range error steps`);
  assertOptionalNear(row.maximum_in_range_rounding_error, expected.maxInRangeError, `#${opIndex} max in-range error`);
  assertOptionalNear(row.maximum_in_range_rounding_error_steps, expected.maxInRangeError == null ? null : expected.maxInRangeError / output.scale, `#${opIndex} max in-range error steps`);
  assertNear(row.mean_clamped_projection_error, expected.meanProjectionError, `#${opIndex} mean projection error`);
  assertNear(row.mean_clamped_projection_error_steps, expected.meanProjectionError / output.scale, `#${opIndex} mean projection error steps`);
  assertNear(row.maximum_clamped_projection_error, expected.maxProjectionError, `#${opIndex} max projection error`);
  assertNear(row.maximum_clamped_projection_error_steps, expected.maxProjectionError / output.scale, `#${opIndex} max projection error steps`);
  if (row.tile_size_codes !== TILE_SIZE || row.tile_grid_dimension !== 16
    || !sameMarginProfile(row.margin_profile, expected.margin)
    || !sameArray(row.output_code_histogram, expected.histogram)
    || !sameArray(row.tile_range_escape_pair_counts, expected.tileEscape)) throw new Error(`Quantization-lattice exact histogram or tile ledger is invalid at #${opIndex}.`);
  assertArrayNear(row.tile_mean_clamped_projection_error_steps, expected.tileMeanError, `#${opIndex} tile error ledger`);
  validateWorstPair(row.worst_projection_pair, expected.worst, opIndex);
  if (row.containment_candidate_count !== expected.containmentCandidateCount) {
    throw new Error(`Quantization-lattice containment candidate count is invalid at #${opIndex}.`);
  }
  validateContainmentFrontier(row.containment_frontier, expected.containmentFrontier, opIndex);
  validateContainmentCandidate(row.fixed_zero_point_containment, expected.fixedZeroPointContainment, opIndex, "fixed zero-point");
  validateContainmentCandidate(row.globally_finest_containment, expected.globallyFinestContainment, opIndex, "globally finest");
  if (!/min_binary64/.test(row.containment_formula || "")) throw new Error(`Quantization-lattice containment formula is missing at #${opIndex}.`);
}

function quantContract(tensor) {
  const dtype = String(tensor?.dtype || "").toUpperCase();
  const range = dtype === "INT8" ? [-128, 127] : dtype === "UINT8" ? [0, 255] : null;
  if (!range) return { error: `Tensor ${tensor.index} uses ${tensor.dtype}; quantization-lattice v1 requires INT8 or UINT8.` };
  if (Number(tensor.quant_scales) !== 1 || tensor.scale_sample?.length !== 1) return { error: `Tensor ${tensor.index} does not expose exactly one per-tensor quantization scale.` };
  if (Number(tensor.quant_zero_points) !== 1 || tensor.zero_point_sample?.length !== 1) return { error: `Tensor ${tensor.index} does not expose exactly one per-tensor zero-point.` };
  const scale = Number(tensor.scale_sample[0]);
  if (!(Number.isFinite(scale) && scale > 0)) return { error: `Tensor ${tensor.index} has a non-positive or non-finite quantization scale.` };
  const zeroPoint = Number(tensor.zero_point_sample[0]);
  if (!Number.isInteger(zeroPoint) || zeroPoint < range[0] || zeroPoint > range[1]) return { error: `Tensor ${tensor.index} zero-point ${zeroPoint} lies outside [${range[0]}, ${range[1]}].` };
  return { tensorIndex: tensor.index, qmin: range[0], qmax: range[1], scale, zeroPoint };
}

function latticeWorkspace(row, mode, jumpToGraphOp) {
  const workspace = element("div", "lattice-workspace");
  const plot = element("div", "lattice-plot");
  const canvasWrap = element("div", "lattice-canvas-wrap");
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  canvas.className = "lattice-canvas";
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", mode === "histogram" ? `Projected output code histogram for ADD ${row.op_index}` : `${mode} lattice map for ADD ${row.op_index}`);
  const tooltip = element("output", "lattice-tooltip");
  canvasWrap.append(canvas, tooltip);
  const legend = element("div", "lattice-legend");
  drawLatticeCanvas(canvas, row, mode, legend);
  bindCanvasTooltip(canvas, tooltip, row, mode);
  plot.append(canvasWrap, legend);
  const detail = element("div", "lattice-detail");
  const heading = element("div", "lattice-detail-head");
  const title = element("div");
  title.append(element("strong", "", `#${padOp(row.op_index)} ADD`), element("span", "", row.output_tensor_name || `T${row.output_tensor_index}`));
  const graph = element("button", "secondary-action", "Inspect op");
  graph.type = "button";
  graph.dataset.latticeOpenGraph = String(row.op_index);
  graph.disabled = typeof jumpToGraphOp !== "function";
  heading.append(title, graph);
  detail.append(
    heading,
    detailGrid(row),
    tensorContractTable(row),
    containmentDesignTable(row),
    worstPairBlock(row),
  );
  workspace.append(plot, detail);
  return workspace;
}

function drawLatticeCanvas(canvas, row, mode, legend) {
  const context = canvas.getContext("2d", { alpha: false });
  context.fillStyle = "#f8fafc";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const margin = 36;
  const size = canvas.width - margin * 2;
  if (mode === "design") {
    drawContainmentDesign(canvas, context, row, legend, margin, size);
    return;
  }
  if (mode === "histogram") {
    const values = row.output_code_histogram;
    const max = Math.max(...values, 1);
    const width = size / values.length;
    values.forEach((value, index) => {
      const height = (value / max) * size;
      context.fillStyle = index === 0 || index === values.length - 1 ? "#d76a3a" : "#197c78";
      context.fillRect(margin + index * width, margin + size - height, Math.max(1, width), height);
    });
    axes(context, margin, size, "projected qout", "pair count");
    legend.replaceChildren(legendItem("#197c78", "projected codes"), legendItem("#d76a3a", "endpoint bins"));
    return;
  }
  const grid = row.tile_grid_dimension;
  const cell = size / grid;
  const values = mode === "escape"
    ? row.tile_range_escape_pair_counts.map((value) => value / (row.tile_size_codes ** 2))
    : row.tile_mean_clamped_projection_error_steps;
  const max = mode === "escape" ? 1 : Math.max(...values, 1e-12);
  values.forEach((value, index) => {
    const ratio = Math.max(0, Math.min(1, value / max));
    context.fillStyle = mode === "escape" ? escapeColor(ratio) : errorColor(ratio);
    const rowIndex = Math.floor(index / grid);
    const column = index % grid;
    context.fillRect(margin + column * cell, margin + (grid - 1 - rowIndex) * cell, Math.ceil(cell), Math.ceil(cell));
  });
  axes(context, margin, size, "input 1 code", "input 0 code");
  legend.replaceChildren(
    legendItem(mode === "escape" ? "#16766f" : "#28666e", mode === "escape" ? "0% endpoint escape" : "lower mean error"),
    legendItem(mode === "escape" ? "#e15d3a" : "#b43f43", mode === "escape" ? "100% endpoint escape" : `${max.toFixed(2)} output steps`),
  );
}

function bindCanvasTooltip(canvas, tooltip, row, mode) {
  canvas.addEventListener("mousemove", (event) => {
    const bounds = canvas.getBoundingClientRect();
    const x = (event.clientX - bounds.left) * (canvas.width / bounds.width);
    const y = (event.clientY - bounds.top) * (canvas.height / bounds.height);
    const margin = 36;
    const size = canvas.width - margin * 2;
    const hitPadding = mode === "design" ? 10 : 0;
    if (x < margin - hitPadding || x > margin + size + hitPadding || y < margin - hitPadding || y > margin + size + hitPadding) {
      tooltip.hidden = true;
      return;
    }
    if (mode === "histogram") {
      const index = Math.min(255, Math.floor(((x - margin) / size) * 256));
      const qmin = row.output_code_range[0];
      const count = row.output_code_histogram[index];
      tooltip.textContent = `qout ${qmin + index}: ${formatNumber(count)} pairs (${percent(count / row.enumerated_code_pair_count)})`;
    } else if (mode === "design") {
      const nearest = (canvas._containmentDesignPoints || []).map((point) => ({
        ...point,
        distance: Math.hypot(x - point.x, y - point.y),
      })).sort((left, right) => left.distance - right.distance)[0];
      if (!nearest || nearest.distance > 18) {
        tooltip.hidden = true;
        return;
      }
      tooltip.textContent = nearest.label;
    } else {
      const grid = row.tile_grid_dimension;
      const column = Math.min(grid - 1, Math.floor(((x - margin) / size) * grid));
      const displayRow = Math.min(grid - 1, Math.floor(((y - margin) / size) * grid));
      const rowIndex = grid - 1 - displayRow;
      const index = rowIndex * grid + column;
      const q0Start = row.input_code_ranges[0][0] + rowIndex * row.tile_size_codes;
      const q1Start = row.input_code_ranges[1][0] + column * row.tile_size_codes;
      tooltip.textContent = mode === "escape"
        ? `q0 ${q0Start}..${q0Start + 15}, q1 ${q1Start}..${q1Start + 15}: ${row.tile_range_escape_pair_counts[index]} / 256 escape`
        : `q0 ${q0Start}..${q0Start + 15}, q1 ${q1Start}..${q1Start + 15}: mean ${steps(row.tile_mean_clamped_projection_error_steps[index])}`;
    }
    tooltip.hidden = false;
    tooltip.style.left = `${Math.min(bounds.width - 230, Math.max(4, event.clientX - bounds.left + 12))}px`;
    tooltip.style.top = `${Math.max(4, event.clientY - bounds.top - 34)}px`;
  });
  canvas.addEventListener("mouseleave", () => { tooltip.hidden = true; });
  tooltip.hidden = true;
}

function drawContainmentDesign(canvas, context, row, legend, margin, size) {
  const frontier = row.containment_frontier || [];
  const ratios = [1, ...frontier.map((point) => Number(point.scale_ratio_to_current))];
  const maxShift = Math.max(1, ...frontier.map((point) => Number(point.absolute_zero_point_shift)));
  let minimumRatio = Math.min(...ratios);
  let maximumRatio = Math.max(...ratios);
  const span = maximumRatio - minimumRatio;
  const padding = span > 0 ? span * 0.12 : Math.max(0.05, maximumRatio * 0.08);
  minimumRatio = Math.max(0, minimumRatio - padding);
  maximumRatio += padding;
  const pointFor = (shift, ratio) => ({
    x: margin + (Number(shift) / maxShift) * size,
    y: margin + size - ((Number(ratio) - minimumRatio) / (maximumRatio - minimumRatio)) * size,
  });
  context.strokeStyle = "#91a0af";
  context.setLineDash([5, 5]);
  const baselineY = pointFor(0, 1).y;
  context.beginPath();
  context.moveTo(margin, baselineY);
  context.lineTo(margin + size, baselineY);
  context.stroke();
  context.setLineDash([]);
  context.strokeStyle = "#28666e";
  context.lineWidth = 2;
  context.beginPath();
  frontier.forEach((point, index) => {
    const coordinate = pointFor(point.absolute_zero_point_shift, point.scale_ratio_to_current);
    if (index === 0) context.moveTo(coordinate.x, coordinate.y); else context.lineTo(coordinate.x, coordinate.y);
  });
  context.stroke();
  const hitPoints = [];
  frontier.forEach((point) => {
    const coordinate = pointFor(point.absolute_zero_point_shift, point.scale_ratio_to_current);
    context.fillStyle = "#28666e";
    context.beginPath();
    context.arc(coordinate.x, coordinate.y, 5, 0, Math.PI * 2);
    context.fill();
    hitPoints.push({ ...coordinate, label: `zp ${point.output_zero_point} (${signed(point.signed_zero_point_delta)}), shift ${point.absolute_zero_point_shift}, scale ${concise(point.minimum_output_scale)} (${ratio(point.scale_ratio_to_current)})` });
  });
  const current = pointFor(0, 1);
  context.fillStyle = "#b43f43";
  context.fillRect(current.x - 6, current.y - 6, 12, 12);
  hitPoints.push({ ...current, label: `current zp ${row.output_zero_point}, scale ${concise(row.output_scale)} (1.000x)` });
  const global = row.globally_finest_containment;
  if (global) {
    const coordinate = pointFor(global.absolute_zero_point_shift, global.scale_ratio_to_current);
    context.strokeStyle = "#d99a2b";
    context.lineWidth = 3;
    context.beginPath();
    context.arc(coordinate.x, coordinate.y, 9, 0, Math.PI * 2);
    context.stroke();
  }
  canvas._containmentDesignPoints = hitPoints;
  axes(context, margin, size, "absolute zero-point shift", "scale / current");
  context.fillStyle = "#536273";
  context.font = "11px system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`${minimumRatio.toFixed(2)}x`, margin + 4, margin + size - 5);
  context.fillText(`${maximumRatio.toFixed(2)}x`, margin + 4, margin + 13);
  legend.replaceChildren(
    legendItem("#28666e", `${frontier.length} non-dominated contracts`),
    legendItem("#b43f43", "current contract"),
    legendItem("#d99a2b", "globally finest containment"),
  );
}

function axes(context, margin, size, xLabel, yLabel) {
  context.strokeStyle = "#536273";
  context.lineWidth = 1;
  context.strokeRect(margin, margin, size, size);
  context.fillStyle = "#536273";
  context.font = "12px system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText(xLabel, margin + size / 2, margin + size + 24);
  context.save();
  context.translate(13, margin + size / 2);
  context.rotate(-Math.PI / 2);
  context.fillText(yLabel, 0, 0);
  context.restore();
}

function detailGrid(row) {
  const grid = element("div", "lattice-detail-grid");
  grid.append(
    metric("Endpoint escape", `${formatNumber(row.range_escape_pair_count)} / ${formatNumber(row.enumerated_code_pair_count)}`, percent(row.range_escape_pair_ratio)),
    metric("Rounded clamp", formatNumber(row.rounded_projection_clamp_pair_count), percent(row.rounded_projection_clamp_pair_ratio)),
    metric("In-range rounding", steps(row.mean_in_range_rounding_error_steps), `max ${steps(row.maximum_in_range_rounding_error_steps)}`),
    metric("All-pair projection", steps(row.mean_clamped_projection_error_steps), `max ${steps(row.maximum_clamped_projection_error_steps)}`),
    metric("Sum interval coverage", percent(row.continuous_sum_interval_coverage_ratio), interval(row.legal_sum_real_range)),
    metric("Projected codes", `${row.distinct_projected_output_code_count} / 256`, percent(row.projected_output_code_utilization_ratio)),
  );
  return grid;
}

function tensorContractTable(row) {
  const wrap = element("div", "lattice-table-wrap");
  const table = element("table", "lattice-contract-table");
  table.innerHTML = "<thead><tr><th>Tensor</th><th>Dtype</th><th>Scale</th><th>Zero-point</th><th>Real range</th></tr></thead>";
  const body = document.createElement("tbody");
  const items = [0, 1, 2].map((position) => ({
    label: position < 2 ? `Input ${position}` : "Output",
    dtype: row.dtype_triplet[position],
    scale: position < 2 ? row.input_scales[position] : row.output_scale,
    zeroPoint: position < 2 ? row.input_zero_points[position] : row.output_zero_point,
    range: position < 2 ? row.input_real_ranges[position] : row.output_real_range,
  }));
  for (const item of items) {
    const tr = document.createElement("tr");
    [item.label, item.dtype, concise(item.scale), item.zeroPoint, interval(item.range)].forEach((value) => tr.append(element("td", "", String(value))));
    body.append(tr);
  }
  table.append(body);
  wrap.append(table);
  return wrap;
}

function containmentDesignTable(row) {
  const wrap = element("div", "lattice-table-wrap lattice-containment-design");
  const table = element("table", "lattice-contract-table");
  table.innerHTML = "<thead><tr><th>Output contract</th><th>Zero-point</th><th>Scale</th><th>Scale / current</th><th>Real range</th><th>Clamp pairs</th><th>Mean error</th><th>Codes</th></tr></thead>";
  const body = document.createElement("tbody");
  const current = {
    label: "Current artifact",
    output_zero_point: row.output_zero_point,
    output_scale: row.output_scale,
    scale_ratio_to_current: 1,
    output_real_range: row.output_real_range,
    rounded_projection_clamp_pair_count: row.rounded_projection_clamp_pair_count,
    mean_absolute_projection_error_current_steps: row.mean_clamped_projection_error_steps,
    distinct_projected_output_code_count: row.distinct_projected_output_code_count,
  };
  const fixed = row.fixed_zero_point_containment ? { label: "Fixed-zp containment", ...row.fixed_zero_point_containment } : null;
  const global = row.globally_finest_containment ? { label: "Globally finest containment", ...row.globally_finest_containment } : null;
  const equivalentContainment = fixed && global
    && Number(fixed.output_scale) === Number(global.output_scale)
    && Number(fixed.output_zero_point) === Number(global.output_zero_point)
    && Number(fixed.rounded_projection_clamp_pair_count) === Number(global.rounded_projection_clamp_pair_count)
    && Number(fixed.mean_absolute_projection_error_current_steps) === Number(global.mean_absolute_projection_error_current_steps)
    && Number(fixed.distinct_projected_output_code_count) === Number(global.distinct_projected_output_code_count);
  const candidates = [
    current,
    fixed && global && equivalentContainment ? { ...fixed, label: "Fixed-zp = globally finest containment" } : fixed,
    equivalentContainment ? null : global,
  ];
  for (const candidate of candidates) {
    const tr = document.createElement("tr");
    if (!candidate && !equivalentContainment) {
      tr.append(element("td", "", "Fixed-zp containment"));
      const unavailable = element("td", "", "unavailable: current zero-point has no signed endpoint capacity");
      unavailable.colSpan = 7;
      tr.append(unavailable);
    } else if (candidate) {
      [
        candidate.label,
        `${candidate.output_zero_point}${candidate.signed_zero_point_delta == null ? "" : ` (${signed(candidate.signed_zero_point_delta)})`}`,
        concise(candidate.output_scale),
        ratio(candidate.scale_ratio_to_current),
        interval(candidate.output_real_range),
        formatNumber(candidate.rounded_projection_clamp_pair_count),
        steps(candidate.mean_absolute_projection_error_current_steps),
        `${candidate.distinct_projected_output_code_count} / 256`,
      ].forEach((value) => tr.append(element("td", "", String(value))));
    }
    if (tr.childNodes.length) body.append(tr);
  }
  table.append(body);
  wrap.append(table);
  if (equivalentContainment) {
    wrap.append(element("p", "lattice-equivalence-note", "The fixed-zero-point and globally finest searches converge on the same contract and exact projection outcomes; they are one design point, not two independent options."));
  }
  return wrap;
}

function worstPairBlock(row) {
  const worst = row.worst_projection_pair;
  const block = element("div", "lattice-worst");
  block.append(
    element("span", "", "Maximum ideal projection error"),
    element("strong", "", worst ? `q0=${worst.input_0_code}, q1=${worst.input_1_code} -> qout=${worst.projected_output_code}` : "not assessed"),
    element("small", "", worst ? `${concise(worst.real_sum)} -> ${concise(worst.projected_real_value)}; ${steps(worst.absolute_error_output_steps)}; ${labelize(worst.range_class)}` : ""),
  );
  return block;
}

function selector(rows, selected) {
  const wrap = element("div", "lattice-op-selector");
  for (const row of rows) {
    const button = element("button", row.op_index === selected ? "active" : "", `#${padOp(row.op_index)}  ${percent(row.range_escape_pair_ratio)}`);
    button.type = "button";
    button.dataset.latticeOp = String(row.op_index);
    button.title = `${row.output_tensor_name || `T${row.output_tensor_index}`} / mean projection ${steps(row.mean_clamped_projection_error_steps)}`;
    wrap.append(button);
  }
  return wrap;
}


/// Model-wide view: one row per operator projection, in graph order, on the
/// shared output-code-step margin axis. Answers "which operator loses its
/// margin first" without opening any individual lattice.
function marginAtlasView(result, selectedOpIndex) {
  const section = element("section", "lattice-atlas");
  const rows = result.margin_atlas || [];
  if (!rows.length) return emptyState("No operator exposes a margin projection for this artifact.");
  const edges = result.margin_bin_edges_output_code_steps || MARGIN_BIN_EDGES;

  section.append(element("h4", "", "Layerwise quantization margin atlas"));
  section.append(element("p", "lattice-atlas-note",
    "Margin is the distance in output code steps from the unclamped ideal output code to the "
    + "nearer representable end. A negative margin is exactly an escape, so this axis carries "
    + "containment and headroom together. Every enumerated code combination is represented: this "
    + "is the complete projection of the lattice, not a sample of it."));

  const legend = element("div", "lattice-atlas-legend");
  legend.append(element("span", "atlas-zone escape", "escape (m < 0)"));
  legend.append(element("span", "atlas-zone boundary", "boundary (0 to 2)"));
  legend.append(element("span", "atlas-zone interior", "interior (m > 2)"));
  section.append(legend);

  const wrap = element("div", "lattice-table-wrap");
  const table = element("table", "lattice-atlas-table");
  const header = edges.map((edge, index) => {
    const low = index === 0 ? "-inf" : edges[index - 1];
    return `${low}..${edge}`;
  });
  header.push(`${edges[edges.length - 1]}..+inf`);
  table.innerHTML = `<thead><tr><th>Operator</th><th>Escape</th><th>p1</th><th>p5</th>`
    + header.map((label) => `<th class="atlas-bin">${label}</th>`).join("")
    + "</tr></thead>";
  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const total = row.enumerated_count || 1;
    const tr = document.createElement("tr");
    if (row.op_index === selectedOpIndex) tr.classList.add("selected");
    tr.dataset.latticeOp = String(row.op_index);
    tr.tabIndex = 0;
    const name = row.branch_position == null
      ? `#${padOp(row.op_index)} ${row.op_name}`
      : `#${padOp(row.op_index)} ${row.op_name} in${row.branch_position}`;
    tr.append(
      element("td", "", name),
      element("td", "numeric", percent(row.escape_ratio)),
      element("td", "numeric", formatNumber(row.percentile_1_margin_output_code_steps)),
      element("td", "numeric", formatNumber(row.percentile_5_margin_output_code_steps)),
    );
    (row.bin_pair_counts || []).forEach((count, index) => {
      const share = count / total;
      const td = document.createElement("td");
      td.className = `atlas-bin ${index < 7 ? "escape" : index < 10 ? "boundary" : "interior"}`;
      td.style.opacity = String(share > 0 ? Math.min(1, 0.15 + Math.sqrt(share) * 0.85) : 0.05);
      td.title = `${formatNumber(count)} of ${formatNumber(total)} (${percent(share)})`;
      tr.append(td);
    });
    tbody.append(tr);
  }
  table.append(tbody);
  wrap.append(table);
  section.append(wrap);
  return section;
}

/// Per-operator 1D projection: the same enumeration as the 2D lattice, folded
/// onto the margin axis. Nothing is discarded, so the escape count here equals
/// the escape count on the lattice.
function marginProjectionView(row) {
  const profile = row.margin_profile;
  if (!profile) return emptyState("This operator does not expose a margin projection.");
  const section = element("section", "lattice-projection");
  section.append(element("h4", "", `#${padOp(row.op_index)} ${row.op_name} margin projection`));

  const axis = row.sum_projection_axis;
  if (axis) {
    section.append(element("p", "lattice-atlas-note",
      `Projection is aligned to the operator, not to the screen diagonal. Co-activation path slope `
      + `${axis.co_activation_slope_q0_per_q1.toFixed(4)} (both inputs at the same real value, through code `
      + `(${axis.zero_crossing_code[1]}, ${axis.zero_crossing_code[0]})); steepest-increase slope `
      + `${axis.steepest_increase_slope_q0_per_q1.toFixed(4)}; iso-result slope `
      + `${axis.iso_result_slope_q0_per_q1.toFixed(4)}. These are three different lines whenever the input scales differ.`));
  }

  const total = (profile.bin_pair_counts || []).reduce((sum, count) => sum + count, 0) || 1;
  const chart = element("div", "margin-bars");
  const edges = profile.bin_edges_output_code_steps || MARGIN_BIN_EDGES;
  const peak = Math.max(...profile.bin_pair_counts, 1);
  profile.bin_pair_counts.forEach((count, index) => {
    const low = index === 0 ? "-inf" : edges[index - 1];
    const high = index === profile.bin_pair_counts.length - 1 ? "+inf" : edges[index];
    const bar = element("div", `margin-bar ${index < 7 ? "escape" : index < 10 ? "boundary" : "interior"}`);
    bar.style.height = `${Math.max(2, (count / peak) * 100)}%`;
    bar.title = `margin ${low}..${high}: ${formatNumber(count)} (${percent(count / total)})`;
    const column = element("div", "margin-column");
    column.append(bar, element("span", "margin-label", String(low)));
    chart.append(column);
  });
  section.append(chart);

  const stats = element("div", "lattice-projection-stats");
  for (const [label, value] of [
    ["Escape", `${formatNumber(profile.escape_pair_count)} (${percent(profile.escape_pair_count / total)})`],
    ["Within 1 step", `${formatNumber(profile.boundary_pressure_1_step_pair_count)}`],
    ["Within 2 steps", `${formatNumber(profile.boundary_pressure_2_step_pair_count)}`],
    ["Minimum margin", formatNumber(profile.minimum_margin_output_code_steps)],
    ["1st percentile", formatNumber(profile.percentile_1_margin_output_code_steps)],
    ["5th percentile", formatNumber(profile.percentile_5_margin_output_code_steps)],
    ["Median", formatNumber(profile.median_margin_output_code_steps)],
  ]) {
    const item = element("div", "lattice-projection-stat");
    item.append(element("span", "stat-label", label), element("strong", "", String(value)));
    stats.append(item);
  }
  section.append(stats);
  return section;
}

function familyRows(result, family) {
  if (family === "ADD") return result.residual_adds || [];
  if (family === "CONCATENATION") return result.concatenation_contracts || [];
  return (result.binary_contracts || []).filter((row) => row.op_name === family);
}

function familyCounts(result, family) {
  if (family === "ADD") return { candidate: result.candidate_add_count, assessed: result.assessed_add_count };
  if (family === "CONCATENATION") {
    return { candidate: result.candidate_concatenation_count, assessed: result.assessed_concatenation_count };
  }
  const coverage = (result.binary_operator_coverage || []).find((row) => row.op_name === family);
  return { candidate: coverage?.candidate_count || 0, assessed: coverage?.assessed_count || 0 };
}

function familyTabs(result, active) {
  const tabs = element("div", "lattice-family-tabs");
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", "Quantization lattice operator families");
  for (const family of LATTICE_FAMILIES) {
    const counts = familyCounts(result, family);
    const button = element("button", `${family === active ? "active" : ""}${counts.candidate ? " has-candidate" : " no-candidate"}`.trim());
    button.type = "button";
    button.dataset.latticeFamily = family;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(family === active));
    button.tabIndex = family === active ? 0 : -1;
    button.title = counts.candidate
      ? `${counts.assessed} of ${counts.candidate} serialized ${family} operators have complete enumerable contracts.`
      : `The analyzer supports ${family}; this artifact serializes none.`;
    button.append(
      element("strong", "", family),
      element("small", "", counts.candidate ? `${counts.assessed}/${counts.candidate} assessed` : "0 in model"),
    );
    tabs.append(button);
  }
  return tabs;
}

function familyScope(family) {
  const binary = BINARY_LATTICE_OPS[family];
  const section = element("section", "lattice-family-scope");
  section.append(
    element("strong", "", `${family} contract domain`),
    element("span", "", family === "CONCATENATION"
      ? "Each quantized input branch is projected independently over all 256 legal codes."
      : "Both quantized inputs are exhaustively combined over the complete 256 x 256 legal-code domain."),
    element("code", "", binary?.formula || CONCAT_LATTICE_FORMULA),
  );
  return section;
}

function familyEmptyState(family) {
  const condition = family === "CONCATENATION"
    ? "complete per-tensor INT8/UINT8 affine input and output contracts"
    : "two per-tensor INT8/UINT8 inputs, one per-tensor INT8/UINT8 output, legal zero-points, and no fused activation";
  return emptyState(`This artifact serializes 0 ${family} operators. Assessment activates independently when ${condition} are present.`);
}

function modeTabs(active) {
  const tabs = element("div", "lattice-mode-tabs");
  [["atlas", "Model atlas"], ["projection", "Margin projection"], ["escape", "Domain escape"],
    ["error", "Projection error"], ["histogram", "Output codes"], ["design", "Contract design"]].forEach(([mode, label]) => {
    const button = element("button", mode === active ? "active" : "", label);
    button.type = "button";
    button.dataset.latticeMode = mode;
    button.setAttribute("aria-pressed", String(mode === active));
    tabs.append(button);
  });
  return tabs;
}

function portfolioTable(rows, selected) {
  const section = element("section", "lattice-portfolio");
  section.append(element("h4", "", "Residual ADD lattice portfolio"));
  const wrap = element("div", "lattice-table-wrap");
  const table = element("table", "lattice-portfolio-table");
  table.innerHTML = "<thead><tr><th>ADD</th><th>State</th><th>Input steps</th><th>Output step</th><th>Endpoint escape</th><th>Rounded clamp</th><th>Mean projection</th><th>Output codes</th><th>Global containment</th></tr></thead>";
  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    if (row.op_index === selected) tr.className = "selected";
    if (row.assessment_status !== "assessed") {
      tr.append(element("td", "", `#${padOp(row.op_index)} ADD`), element("td", "", "not assessed"));
      const reason = element("td", "", row.not_assessed_reason);
      reason.colSpan = 7;
      tr.append(reason);
    } else {
      const values = [
        `#${padOp(row.op_index)} ADD`,
        row.complete_legal_domain_contained ? "contained" : "endpoint escape",
        row.input_scales.map(concise).join(" / "), concise(row.output_scale),
        percent(row.range_escape_pair_ratio), percent(row.rounded_projection_clamp_pair_ratio),
        steps(row.mean_clamped_projection_error_steps), `${row.distinct_projected_output_code_count} / 256`,
        row.globally_finest_containment
          ? `${ratio(row.globally_finest_containment.scale_ratio_to_current)} / zp ${signed(row.globally_finest_containment.signed_zero_point_delta)}`
          : "unavailable",
      ];
      values.forEach((value) => tr.append(element("td", "", String(value))));
      tr.dataset.latticeOp = String(row.op_index);
      tr.tabIndex = 0;
    }
    tbody.append(tr);
  }
  table.append(tbody);
  wrap.append(table);
  section.append(wrap);
  return section;
}

function genericBinaryTable(rows, family = "binary") {
  const section = element("section", "lattice-portfolio");
  section.append(element("h4", "", `${family} lattice contracts`));
  const wrap = element("div", "lattice-table-wrap");
  const table = element("table", "lattice-portfolio-table");
  table.innerHTML = "<thead><tr><th>Operator</th><th>Assessment</th><th>Input scales</th><th>Output scale</th><th>Endpoint escape</th><th>Rounded clamp</th><th>Mean projection</th><th>Output codes</th><th>Global containment</th></tr></thead>";
  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    if (row.assessment_status !== "assessed") {
      tr.append(element("td", "", `#${padOp(row.op_index)} ${row.op_name}`), element("td", "", "not assessed"));
      const reason = element("td", "", row.not_assessed_reason);
      reason.colSpan = 7;
      tr.append(reason);
    } else {
      [
        `#${padOp(row.op_index)} ${row.op_name}`,
        row.complete_legal_domain_contained ? "contained" : "endpoint escape",
        row.input_scales.map(concise).join(" / "),
        concise(row.output_scale),
        `${formatNumber(row.range_escape_pair_count)} (${percent(row.range_escape_pair_ratio)})`,
        `${formatNumber(row.rounded_projection_clamp_pair_count)} (${percent(row.rounded_projection_clamp_pair_ratio)})`,
        steps(row.mean_clamped_projection_error_steps),
        `${row.distinct_projected_output_code_count} / 256`,
        row.globally_finest_containment
          ? `${ratio(row.globally_finest_containment.scale_ratio_to_current)} / zp ${signed(row.globally_finest_containment.signed_zero_point_delta)}`
          : "unavailable",
      ].forEach((value) => tr.append(element("td", "", String(value))));
    }
    tbody.append(tr);
  }
  table.append(tbody);
  wrap.append(table);
  section.append(wrap);
  return section;
}

function concatenationTable(rows) {
  const section = element("section", "lattice-portfolio");
  section.append(element("h4", "", "CONCATENATION branch contracts"));
  const wrap = element("div", "lattice-table-wrap");
  const table = element("table", "lattice-portfolio-table");
  table.innerHTML = "<thead><tr><th>Operator</th><th>Assessment</th><th>Inputs</th><th>Enumerated codes</th><th>Endpoint escape</th><th>Rounded clamp</th><th>Maximum projection</th></tr></thead>";
  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    if (row.assessment_status !== "assessed") {
      tr.append(element("td", "", `#${padOp(row.op_index)} CONCATENATION`), element("td", "", "not assessed"));
      const reason = element("td", "", row.not_assessed_reason);
      reason.colSpan = 5;
      tr.append(reason);
    } else {
      [
        `#${padOp(row.op_index)} CONCATENATION`,
        row.complete_legal_domain_contained ? "contained" : "endpoint escape",
        formatNumber(row.input_count),
        formatNumber(row.enumerated_code_count),
        formatNumber(row.range_escape_code_count),
        formatNumber(row.rounded_projection_clamp_code_count),
        steps(row.maximum_absolute_projection_error_output_steps),
      ].forEach((value) => tr.append(element("td", "", String(value))));
    }
    tbody.append(tr);
  }
  table.append(tbody);
  wrap.append(table);
  section.append(wrap);
  return section;
}

function unassessedTable(rows) {
  const wrap = element("div", "lattice-table-wrap");
  const table = element("table", "lattice-portfolio-table");
  table.innerHTML = "<thead><tr><th>ADD</th><th>Assessment</th><th>Reason</th></tr></thead>";
  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    [(`#${padOp(row.op_index)} ADD`), row.assessment_status, row.not_assessed_reason].forEach((value) => tr.append(element("td", "", value)));
    tbody.append(tr);
  });
  table.append(tbody);
  wrap.append(table);
  return wrap;
}

function methodBoundary(result) {
  const details = element("details", "lattice-method-details");
  details.append(element("summary", "", `${result.schema} / ${result.evidence_class}`));
  details.append(element("p", "", result.method), element("p", "", result.containment_formula), element("p", "", result.interpretation_boundary));
  return details;
}

function metric(label, value, note) {
  const item = element("div", "lattice-metric");
  item.append(element("span", "", label), element("strong", "", value ?? "-"), element("small", "", note || ""));
  return item;
}

function legendItem(color, label) {
  const item = element("span");
  const swatch = element("i");
  swatch.style.background = color;
  item.append(swatch, document.createTextNode(label));
  return item;
}

function emptyState(message) {
  return element("p", "lattice-empty", message);
}

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

function tensorAt(tensors, index) {
  return Number.isInteger(Number(index)) && Number(index) >= 0 ? tensors[Number(index)] || null : null;
}

function realRange(contract) {
  return [(contract.qmin - contract.zeroPoint) * contract.scale, (contract.qmax - contract.zeroPoint) * contract.scale];
}

function latticeComparator(left, right) {
  return Number(right.range_escape_pair_ratio || 0) - Number(left.range_escape_pair_ratio || 0)
    || Number(right.mean_clamped_projection_error_steps || 0) - Number(left.mean_clamped_projection_error_steps || 0)
    || left.op_index - right.op_index;
}

function validateWorstPair(actual, expected, opIndex) {
  if (!actual || !expected || actual.input_0_code !== expected.input_0_code || actual.input_1_code !== expected.input_1_code
    || actual.rounded_unclamped_output_code !== expected.rounded_unclamped_output_code
    || actual.projected_output_code !== expected.projected_output_code || actual.range_class !== expected.range_class) {
    throw new Error(`Quantization-lattice worst pair is invalid at #${opIndex}.`);
  }
  ["real_sum", "projected_real_value", "absolute_error", "absolute_error_output_steps"].forEach((field) => assertNear(actual[field], expected[field], `#${opIndex} worst ${field}`));
}

function validateContainmentFrontier(actual, expected, opIndex) {
  if (!Array.isArray(actual) || !Array.isArray(expected) || actual.length !== expected.length) {
    throw new Error(`Quantization-lattice containment frontier length is invalid at #${opIndex}.`);
  }
  actual.forEach((point, index) => {
    const reference = expected[index];
    for (const field of ["output_zero_point", "signed_zero_point_delta", "absolute_zero_point_shift", "negative_code_capacity", "positive_code_capacity"]) {
      if (point[field] !== reference[field]) throw new Error(`Quantization-lattice frontier ${field} is invalid at #${opIndex}[${index}].`);
    }
    assertNear(point.minimum_output_scale, reference.minimum_output_scale, `#${opIndex} frontier scale[${index}]`, 0);
    assertNear(point.scale_ratio_to_current, reference.scale_ratio_to_current, `#${opIndex} frontier ratio[${index}]`, 0);
  });
}

function validateContainmentCandidate(actual, expected, opIndex, label) {
  if (actual == null || expected == null) {
    if (!(actual == null && expected == null)) throw new Error(`Quantization-lattice ${label} nullability is invalid at #${opIndex}.`);
    return;
  }
  for (const field of [
    "design", "output_zero_point", "signed_zero_point_delta", "absolute_zero_point_shift",
    "rounded_projection_clamp_pair_count", "distinct_projected_output_code_count",
  ]) {
    if (actual[field] !== expected[field]) throw new Error(`Quantization-lattice ${label} ${field} is invalid at #${opIndex}.`);
  }
  for (const field of [
    "output_scale", "scale_ratio_to_current", "projected_output_code_utilization_ratio",
    "mean_absolute_projection_error", "mean_absolute_projection_error_current_steps",
    "mean_absolute_projection_error_candidate_steps", "maximum_absolute_projection_error",
    "maximum_absolute_projection_error_current_steps", "maximum_absolute_projection_error_candidate_steps",
  ]) assertNear(actual[field], expected[field], `#${opIndex} ${label} ${field}`);
  assertArrayNear(actual.output_real_range, expected.output_real_range, `#${opIndex} ${label} output range`);
}

function assertNear(actual, expected, label, tolerance = 1e-9) {
  const scale = Math.max(1, Math.abs(Number(actual)), Math.abs(Number(expected)));
  if (!Number.isFinite(Number(actual)) || !Number.isFinite(Number(expected)) || Math.abs(Number(actual) - Number(expected)) > tolerance * scale) {
    throw new Error(`Quantization-lattice ${label} mismatch: ${actual} != ${expected}.`);
  }
}

function assertOptionalNear(actual, expected, label) {
  if (actual == null || expected == null) {
    if (!(actual == null && expected == null)) throw new Error(`Quantization-lattice ${label} nullability mismatch.`);
  } else assertNear(actual, expected, label);
}

function assertArrayNear(actual, expected, label) {
  if (!Array.isArray(actual) || !Array.isArray(expected) || actual.length !== expected.length) throw new Error(`Quantization-lattice ${label} length mismatch.`);
  actual.forEach((value, index) => assertNear(value, expected[index], `${label}[${index}]`));
}

function assertNestedArrayNear(actual, expected, label) {
  if (!Array.isArray(actual) || actual.length !== expected.length) throw new Error(`Quantization-lattice ${label} length mismatch.`);
  actual.forEach((value, index) => assertArrayNear(value, expected[index], `${label}[${index}]`));
}

function sameArray(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length
    && left.every((value, index) => Array.isArray(value) ? sameArray(value, right[index]) : value === right[index]);
}

function sum(values) { return values.reduce((total, value) => total + Number(value || 0), 0); }
function maxOptional(values) { return values.length ? Math.max(...values) : null; }
function percent(value) { return value == null ? "-" : `${(Number(value) * 100).toFixed(2)}%`; }
function steps(value) { return value == null ? "-" : `${Number(value).toFixed(3)} steps`; }
function ratio(value) { return value == null ? "-" : `${Number(value).toFixed(3)}x`; }
function signed(value) { return `${Number(value) >= 0 ? "+" : ""}${Number(value)}`; }
function concise(value) { return value == null ? "-" : Number(value).toPrecision(6).replace(/\.0+(?=e|$)/, ""); }
function interval(value) { return Array.isArray(value) ? `[${concise(value[0])}, ${concise(value[1])}]` : "-"; }
function labelize(value) { return String(value || "").replaceAll("_", " "); }

function escapeColor(ratio) {
  return mixColor(ratio < 0.5 ? "#16766f" : "#d99a2b", ratio < 0.5 ? "#d99a2b" : "#e15d3a", ratio < 0.5 ? ratio * 2 : (ratio - 0.5) * 2);
}

function errorColor(ratio) {
  return mixColor(ratio < 0.5 ? "#28666e" : "#d2aa35", ratio < 0.5 ? "#d2aa35" : "#b43f43", ratio < 0.5 ? ratio * 2 : (ratio - 0.5) * 2);
}

function mixColor(from, to, ratio) {
  const parse = (hex) => [1, 3, 5].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16));
  const left = parse(from);
  const right = parse(to);
  return `rgb(${left.map((value, index) => Math.round(value + (right[index] - value) * ratio)).join(",")})`;
}
