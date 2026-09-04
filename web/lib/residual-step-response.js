import { formatNumber, padOp } from "./format.js";
import { browserAssetUrl } from "./browser-asset-url.js";
import { sha256Hex } from "./hash.js";
import { roundTiesAway } from "./quantization-math.js";

export const RESIDUAL_STEP_RESPONSE_SCHEMA = "deepbom.residual_step_response.v1";

const METHOD_VERSION = "2026-07-17.1";
const DESIGNS = [
  "current_artifact_contract",
  "fixed_zero_point_minimum_containment",
  "globally_finest_minimum_containment",
];
const TILE_SIZE = 16;

export function createResidualStepResponseController({
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
  let selectedOp = null;
  let selectedDesign = "globally_finest_minimum_containment";
  let current = null;
  let renderToken = 0;
  let resizeObserver = null;
  let verificationWorker = null;
  const selectionSource = "residual-step-response";

  root?.addEventListener("click", (event) => {
    const opButton = event.target.closest("[data-step-response-op]");
    if (opButton) {
      selectedOp = Number(opButton.dataset.stepResponseOp);
      residualSelection?.set({ opIndex: selectedOp }, selectionSource);
      renderBody();
      return;
    }
    const designButton = event.target.closest("[data-step-response-design]");
    if (designButton && DESIGNS.includes(designButton.dataset.stepResponseDesign)) {
      selectedDesign = designButton.dataset.stepResponseDesign;
      residualSelection?.set({ design: selectedDesign }, selectionSource);
      renderBody();
      return;
    }
    const graphButton = event.target.closest("[data-step-response-open-graph]");
    if (graphButton) jumpToGraphOp?.(Number(graphButton.dataset.stepResponseOpenGraph));
  });
  residualSelection?.subscribe((selection, source) => {
    if (source === selectionSource || !current?.residual_step_response) return;
    const assessed = current.residual_step_response.residual_adds.filter((row) => row.assessment_status === "assessed");
    if (assessed.some((row) => row.op_index === selection.opIndex)) selectedOp = selection.opIndex;
    if (DESIGNS.includes(selection.design)) selectedDesign = selection.design;
    renderBody();
  });
  downloadButton?.addEventListener("click", () => {
    const result = getAnalysis?.()?.residual_step_response;
    if (result) onDownload?.(result, "residual_step_response.json");
  });

  function render(analysis = null) {
    renderToken += 1;
    const token = renderToken;
    current = analysis || getAnalysis?.() || null;
    const result = current?.residual_step_response;
    verificationWorker?.terminate();
    verificationWorker = null;
    resizeObserver?.disconnect();
    resizeObserver = null;
    if (!result || String(current?.format || "").toLowerCase() !== "tflite") {
      if (root) root.hidden = true;
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    if (root) root.hidden = false;
    if (downloadButton) downloadButton.disabled = false;
    try {
      validateResidualStepResponseEnvelope(result, current);
      const assessed = result.residual_adds.filter((row) => row.assessment_status === "assessed");
      const shared = residualSelection?.get();
      if (assessed.some((row) => row.op_index === shared?.opIndex)) selectedOp = shared.opIndex;
      if (DESIGNS.includes(shared?.design)) selectedDesign = shared.design;
      if (!assessed.some((row) => row.op_index === selectedOp)) {
        selectedOp = result.retention_cost_ranking_op_indices[0] ?? assessed[0]?.op_index ?? null;
      }
      residualSelection?.set({ opIndex: selectedOp, design: selectedDesign }, selectionSource);
      renderSummary(summary, result);
      renderBody();
      if (status) {
        status.textContent = "evidence loaded / verification pending";
        status.dataset.tone = "ok";
      }
      void runIndependentVerification(result, current, (worker) => {
        verificationWorker = worker;
      }).then(() => {
        if (token !== renderToken || !status) return;
        status.textContent = "independently verified";
        status.dataset.tone = "ok";
      }).catch((error) => {
        if (token !== renderToken || !status) return;
        status.textContent = `integrity error: ${error.message}`;
        status.dataset.tone = "risk";
      });
    } catch (error) {
      summary?.replaceChildren();
      body?.replaceChildren(messageNode(`Residual step-response evidence rejected: ${error.message}`, "risk"));
      if (status) {
        status.textContent = "evidence rejected";
        status.dataset.tone = "risk";
      }
    }
  }

  function renderBody() {
    if (!body || !current?.residual_step_response) return;
    resizeObserver?.disconnect();
    const result = current.residual_step_response;
    const row = result.residual_adds.find((item) => item.op_index === selectedOp)
      || result.residual_adds.find((item) => item.assessment_status === "assessed");
    if (!row) {
      body.replaceChildren(messageNode("No residual step-response contract is assessable."));
      return;
    }
    const contract = row.contracts.find((item) => item.design === selectedDesign) || row.contracts[0];
    selectedDesign = contract.design;
    const fieldCanvas = element("canvas", "step-response-field-canvas");
    fieldCanvas.setAttribute("aria-label", "Residual branch local distinguishability field");
    const tradeoffCanvas = element("canvas", "step-response-tradeoff-canvas");
    tradeoffCanvas.setAttribute("aria-label", "Residual containment and silent-transition tradeoff plot");
    body.replaceChildren(
      opSelector(result, row.op_index),
      designTabs(row.contracts, contract.design),
      headline(row, contract, jumpToGraphOp),
      element("div", "step-response-canvas-grid", [
        canvasPanel("Local Influence Field", "Each pixel is one exact interior input-code pair, classified by which +1 branch steps remain visible.", fieldCanvas, influenceLegend()),
        canvasPanel("Containment Trade-off", "Clamp pairs removed versus silent branch transitions across all three contracts.", tradeoffCanvas),
      ]),
      branchTable(contract),
      contractTable(row),
      portfolioTable(result, row.op_index),
      methodBoundary(result),
    );
    const draw = () => {
      drawInfluenceField(fieldCanvas, row, contract, current);
      drawTradeoff(tradeoffCanvas, row);
    };
    requestAnimationFrame(draw);
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver(draw);
      resizeObserver.observe(fieldCanvas.parentElement);
      resizeObserver.observe(tradeoffCanvas.parentElement);
    }
  }

  return { render };
}

function validateResidualStepResponseEnvelope(result, analysis) {
  if (!result || result.schema !== RESIDUAL_STEP_RESPONSE_SCHEMA
    || result.method_version !== METHOD_VERSION || result.evidence_class !== "DERIVED") throw new Error("Step-response envelope identity is invalid.");
  const addCount = (analysis?.ops || []).filter((op) => op.name === "ADD").length;
  if (!Array.isArray(result.residual_adds) || result.candidate_add_count !== addCount
    || result.residual_adds.length !== addCount
    || result.residual_adds.filter((row) => row.assessment_status === "assessed").length !== result.assessed_add_count
    || result.residual_adds.flatMap((row) => row.contracts || []).length !== result.contract_response_count
    || !(result.residual_adds || []).flatMap((row) => row.contracts || []).every((contract) => /^[a-f0-9]{64}$/.test(contract.transition_ledger_sha256 || ""))) {
    throw new Error("Step-response envelope coverage is invalid.");
  }
  return true;
}

async function runIndependentVerification(result, analysis, onWorker) {
  if (typeof Worker === "function") {
    const worker = new Worker(browserAssetUrl("./lib/residual-step-response-worker.js", "./residual-step-response-worker.js", import.meta.url), { type: "module" });
    onWorker?.(worker);
    return new Promise((resolve, reject) => {
      worker.onmessage = (event) => {
        worker.terminate();
        if (event.data?.ok) resolve(true);
        else reject(new Error(event.data?.error || "Step-response worker rejected the evidence."));
      };
      worker.onerror = (event) => {
        worker.terminate();
        reject(new Error(event.message || "Step-response worker failed."));
      };
      worker.postMessage({ result, analysis: verificationAnalysis(analysis) });
    });
  }
  await new Promise((resolve) => setTimeout(resolve, 0));
  validateResidualStepResponse(result, analysis);
  return validateResidualStepResponseDigests(result, analysis);
}

function verificationAnalysis(analysis) {
  return {
    format: analysis?.format,
    ops: (analysis?.ops || []).filter((op) => op.name === "ADD").map((op) => ({
      index: op.index,
      name: op.name,
      inputs: op.inputs,
      outputs: op.outputs,
    })),
    tensors: (analysis?.tensors || []).map((tensor) => ({
      index: tensor.index,
      dtype: tensor.dtype,
      scale_sample: tensor.scale_sample,
      zero_point_sample: tensor.zero_point_sample,
    })),
    quantization_lattice: {
      residual_adds: (analysis?.quantization_lattice?.residual_adds || []).map((row) => ({
        op_index: row.op_index,
        fixed_zero_point_containment: row.fixed_zero_point_containment,
        globally_finest_containment: row.globally_finest_containment,
      })),
    },
  };
}

export function validateResidualStepResponse(result, analysis) {
  if (!result || result.schema !== RESIDUAL_STEP_RESPONSE_SCHEMA) throw new Error("Step-response schema is invalid.");
  if (result.method_version !== METHOD_VERSION || result.evidence_class !== "DERIVED") throw new Error("Step-response method identity is invalid.");
  const latticeRows = analysis?.quantization_lattice?.residual_adds || [];
  const addOps = (analysis?.ops || []).filter((op) => op.name === "ADD");
  if (!Array.isArray(result.residual_adds) || result.residual_adds.length !== latticeRows.length
    || result.candidate_add_count !== addOps.length || latticeRows.length !== addOps.length) throw new Error("Step-response ADD coverage is invalid.");
  const reconstructed = [];
  for (let index = 0; index < latticeRows.length; index += 1) {
    const row = result.residual_adds[index];
    const lattice = latticeRows[index];
    const op = addOps[index];
    if (row.op_index !== op.index || lattice.op_index !== op.index || row.op_name !== "ADD") throw new Error(`Step-response op binding is invalid at row ${index}.`);
    const expected = reconstructRow(op, analysis.tensors || [], lattice);
    if (expected.assessment_status !== "assessed") {
      if (row.assessment_status !== "not_assessed" || row.contracts?.length) throw new Error(`Step-response unassessed state is invalid at #${op.index}.`);
      reconstructed.push(expected);
      continue;
    }
    compareRow(row, expected, op.index);
    reconstructed.push(expected);
  }
  const assessed = reconstructed.filter((row) => row.assessment_status === "assessed");
  const ranked = [...assessed].sort((left, right) => right.maximum_containment_silent_ratio_increase - left.maximum_containment_silent_ratio_increase
    || right.maximum_containment_additional_silent_transitions - left.maximum_containment_additional_silent_transitions
    || left.op_index - right.op_index);
  if (!sameArray(result.retention_cost_ranking_op_indices, ranked.map((row) => row.op_index))) throw new Error("Step-response retention-cost ranking is invalid.");
  ranked.forEach((row, index) => {
    const emitted = result.residual_adds.find((item) => item.op_index === row.op_index);
    if (emitted?.retention_cost_rank !== index + 1) throw new Error(`Step-response rank is invalid at #${row.op_index}.`);
  });
  const contracts = assessed.flatMap((row) => row.contracts);
  const current = contracts.filter((contract) => contract.design === DESIGNS[0]);
  const candidates = contracts.filter((contract) => contract.design !== DESIGNS[0]);
  assertEqual(result.assessed_add_count, assessed.length, "assessed ADD count");
  assertEqual(result.unassessed_add_count, result.residual_adds.length - assessed.length, "unassessed ADD count");
  assertEqual(result.contract_response_count, contracts.length, "contract count");
  assertEqual(result.total_transition_count, sum(contracts, "total_transition_count"), "total transition count");
  assertEqual(result.total_joint_interior_cell_count, sum(contracts, "joint_interior_cell_count"), "joint cell count");
  assertEqual(result.current_silent_transition_count, sum(current, "silent_transition_count"), "current silent count");
  assertEqual(result.containment_silent_transition_count, sum(candidates, "silent_transition_count"), "candidate silent count");
  assertEqual(result.containment_additional_silent_transition_count, sum(candidates, "additional_silent_transitions_vs_current"), "additional silent count");
  assertEqual(result.current_rounded_projection_clamp_pair_count, sum(current, "rounded_projection_clamp_pair_count"), "current clamp count");
  assertEqual(result.containment_removed_rounded_clamp_pair_count, sum(candidates, "removed_rounded_clamp_pairs_vs_current"), "removed clamp count");
  if (candidates.length) {
    assertNear(result.maximum_containment_silent_ratio_increase, Math.max(...candidates.map((contract) => -contract.visible_transition_ratio_delta_vs_current)), "maximum silent-ratio increase");
  } else if (result.maximum_containment_silent_ratio_increase != null) {
    throw new Error("Step-response maximum silent-ratio increase must be null without containment contracts.");
  }
  const expectedStatus = result.residual_adds.length === 0 ? "not_applicable"
    : assessed.length === result.residual_adds.length ? "assessed"
      : assessed.length ? "partial" : "not_assessed";
  if (result.status !== expectedStatus
    || !/uniform legal-code-domain local distinguishability/i.test(result.interpretation_boundary || "")
    || !/not an observed activation distribution/i.test(result.interpretation_boundary || "")
    || !/hold the other input code fixed/i.test(result.transition_definition || "")
    || !/both, input-0-only, input-1-only, or neither/i.test(result.joint_cell_definition || "")
    || !/nine signed i64 little-endian fields/i.test(result.transition_ledger_hash_method || "")) throw new Error("Step-response evidence boundary is incomplete.");
  return reconstructed;
}

export async function validateResidualStepResponseDigests(result, analysis) {
  validateResidualStepResponse(result, analysis);
  const addOps = new Map((analysis?.ops || []).filter((op) => op.name === "ADD").map((op) => [op.index, op]));
  const latticeRows = new Map((analysis?.quantization_lattice?.residual_adds || []).map((row) => [row.op_index, row]));
  for (const row of result.residual_adds.filter((item) => item.assessment_status === "assessed")) {
    const op = addOps.get(row.op_index);
    const lattice = latticeRows.get(row.op_index);
    const contracts = contractsFor(op, analysis.tensors || [], lattice);
    for (const emitted of row.contracts) {
      const contract = contracts.find((item) => item.design === emitted.design);
      const ledger = transitionLedgerBytes(contract.input0, contract.input1, contract.output);
      const digest = await sha256Hex(ledger);
      if (digest !== emitted.transition_ledger_sha256) throw new Error(`Transition ledger digest mismatch at #${row.op_index} ${emitted.design}.`);
    }
  }
  return true;
}

function reconstructRow(op, tensors, lattice) {
  let contracts;
  try {
    contracts = contractsFor(op, tensors, lattice);
  } catch (error) {
    return { op_index: op.index, assessment_status: "not_assessed", reason: error.message, contracts: [] };
  }
  const responses = contracts.map((contract) => evaluateContract(contract));
  const current = responses[0];
  for (const response of responses) {
    response.removed_rounded_clamp_pairs_vs_current = current.rounded_projection_clamp_pair_count - response.rounded_projection_clamp_pair_count;
    response.additional_silent_transitions_vs_current = response.silent_transition_count - current.silent_transition_count;
    response.visible_transition_ratio_delta_vs_current = response.visible_transition_ratio - current.visible_transition_ratio;
  }
  const candidates = responses.slice(1);
  return {
    op_index: op.index,
    assessment_status: "assessed",
    contracts: responses,
    maximum_containment_silent_ratio_increase: Math.max(...candidates.map((contract) => -contract.visible_transition_ratio_delta_vs_current)),
    maximum_containment_additional_silent_transitions: Math.max(...candidates.map((contract) => contract.additional_silent_transitions_vs_current)),
    maximum_containment_removed_clamp_pairs: Math.max(...candidates.map((contract) => contract.removed_rounded_clamp_pairs_vs_current)),
  };
}

function contractsFor(op, tensors, lattice) {
  if (!op || op.inputs?.length < 2 || !op.outputs?.length) throw new Error("ADD topology is incomplete.");
  const input0 = quantContract(tensors, op.inputs[0]);
  const input1 = quantContract(tensors, op.inputs[1]);
  const current = quantContract(tensors, op.outputs[0]);
  const fixed = lattice?.fixed_zero_point_containment;
  const global = lattice?.globally_finest_containment;
  if (!fixed || !global) throw new Error("Both containment contracts are required.");
  const candidate = (row) => ({ ...current, scale: Number(row.output_scale), zeroPoint: Number(row.output_zero_point) });
  return [
    { design: DESIGNS[0], input0, input1, output: current, scaleRatio: 1, zeroPointDelta: 0 },
    { design: DESIGNS[1], input0, input1, output: candidate(fixed), scaleRatio: Number(fixed.output_scale) / current.scale, zeroPointDelta: Number(fixed.output_zero_point) - current.zeroPoint },
    { design: DESIGNS[2], input0, input1, output: candidate(global), scaleRatio: Number(global.output_scale) / current.scale, zeroPointDelta: Number(global.output_zero_point) - current.zeroPoint },
  ];
}

function evaluateContract({ design, input0, input1, output, scaleRatio, zeroPointDelta }) {
  const grid = 16;
  const used = Array(256).fill(false);
  const branch = [branchAccumulator(input0), branchAccumulator(input1)];
  const tiles = Array.from({ length: 5 }, () => Array(256).fill(0));
  let clamps = 0;
  let joint = 0;
  const classes = [0, 0, 0, 0];
  for (let q0 = input0.qmin; q0 <= input0.qmax; q0 += 1) {
    for (let q1 = input1.qmin; q1 <= input1.qmax; q1 += 1) {
      const base = project(q0, q1, input0, input1, output);
      used[base.code - output.qmin] = true;
      if (!inRange(base.raw, output)) clamps += 1;
      let delta0 = null;
      let delta1 = null;
      if (q0 < input0.qmax) delta0 = recordBranch(branch[0], q0, q1, base, project(q0 + 1, q1, input0, input1, output), output);
      if (q1 < input1.qmax) delta1 = recordBranch(branch[1], q0, q1, base, project(q0, q1 + 1, input0, input1, output), output);
      if (delta0 != null && delta1 != null) {
        joint += 1;
        const tile = Math.floor((q0 - input0.qmin) / TILE_SIZE) * grid + Math.floor((q1 - input1.qmin) / TILE_SIZE);
        tiles[0][tile] += 1;
        const classIndex = delta0 > 0 ? (delta1 > 0 ? 0 : 1) : (delta1 > 0 ? 2 : 3);
        classes[classIndex] += 1;
        tiles[classIndex + 1][tile] += 1;
      }
    }
  }
  const branches = branch.map((item, index) => finishBranch(item, index));
  const transitions = sum(branches, "transition_count");
  const visible = sum(branches, "visible_transition_count");
  const silent = transitions - visible;
  return {
    design,
    output_scale: output.scale,
    output_zero_point: output.zeroPoint,
    scale_ratio_to_current: scaleRatio,
    signed_zero_point_delta: zeroPointDelta,
    rounded_projection_clamp_pair_count: clamps,
    rounded_projection_clamp_pair_ratio: clamps / 65_536,
    complete_rounded_domain_containment: clamps === 0,
    distinct_projected_output_code_count: used.filter(Boolean).length,
    branch_responses: branches,
    total_transition_count: transitions,
    visible_transition_count: visible,
    silent_transition_count: silent,
    visible_transition_ratio: visible / transitions,
    silent_transition_ratio: silent / transitions,
    joint_interior_cell_count: joint,
    both_branches_visible_cell_count: classes[0],
    input_0_only_visible_cell_count: classes[1],
    input_1_only_visible_cell_count: classes[2],
    neither_branch_visible_cell_count: classes[3],
    both_branches_visible_ratio: classes[0] / joint,
    neither_branch_visible_ratio: classes[3] / joint,
    tile_size_codes: TILE_SIZE,
    tile_grid_dimension: grid,
    tile_joint_cell_counts: tiles[0],
    tile_both_branches_visible_counts: tiles[1],
    tile_input_0_only_visible_counts: tiles[2],
    tile_input_1_only_visible_counts: tiles[3],
    tile_neither_branch_visible_counts: tiles[4],
    removed_rounded_clamp_pairs_vs_current: 0,
    additional_silent_transitions_vs_current: 0,
    visible_transition_ratio_delta_vs_current: 0,
  };
}

function branchAccumulator(input) {
  return { input, transitions: 0, visible: 0, silent: 0, unclipped: 0, unclippedSilent: 0, clamp: 0, clampSilent: 0, multi: 0, deltaSum: 0, maxDelta: 0, errorSum: 0, errorMax: 0, histogram: Array(256).fill(0), worst: null, firstSilent: null };
}

function recordBranch(state, q0, q1, base, adjacent, output) {
  const delta = adjacent.code - base.code;
  const unclipped = inRange(base.raw, output) && inRange(adjacent.raw, output);
  const witness = { base_input_0_code: q0, base_input_1_code: q1, base_output_code: base.code, adjacent_output_code: adjacent.code, output_code_delta: delta, clamp_associated: !unclipped };
  state.transitions += 1;
  state.deltaSum += delta;
  state.histogram[delta] += 1;
  if (delta === 0) {
    state.silent += 1;
    if (unclipped) {
      state.unclippedSilent += 1;
      state.firstSilent ||= witness;
    } else state.clampSilent += 1;
  } else state.visible += 1;
  if (delta > 1) state.multi += 1;
  if (unclipped) state.unclipped += 1;
  else state.clamp += 1;
  if (delta > state.maxDelta) { state.maxDelta = delta; state.worst = witness; }
  const error = Math.abs(delta * output.scale - state.input.scale) / output.scale;
  state.errorSum += error;
  state.errorMax = Math.max(state.errorMax, error);
  return delta;
}

function finishBranch(state, index) {
  let last = state.histogram.length - 1;
  while (last > 0 && state.histogram[last] === 0) last -= 1;
  return {
    branch_index: index,
    input_tensor_index: state.input.index,
    input_scale: state.input.scale,
    transition_count: state.transitions,
    visible_transition_count: state.visible,
    silent_transition_count: state.silent,
    visible_transition_ratio: state.visible / state.transitions,
    silent_transition_ratio: state.silent / state.transitions,
    unclipped_transition_count: state.unclipped,
    unclipped_silent_transition_count: state.unclippedSilent,
    clamp_associated_transition_count: state.clamp,
    clamp_associated_silent_transition_count: state.clampSilent,
    multi_code_jump_transition_count: state.multi,
    mean_output_code_delta: state.deltaSum / state.transitions,
    maximum_output_code_delta: state.maxDelta,
    mean_absolute_step_reproduction_error_output_steps: state.errorSum / state.transitions,
    maximum_absolute_step_reproduction_error_output_steps: state.errorMax,
    output_code_delta_histogram: state.histogram.slice(0, last + 1),
    worst_jump: state.worst,
    first_unclipped_silent: state.firstSilent,
  };
}

function transitionLedgerBytes(input0, input1, output) {
  const transitionCount = (input0.qmax - input0.qmin) * (input1.qmax - input1.qmin + 1)
    + (input1.qmax - input1.qmin) * (input0.qmax - input0.qmin + 1);
  const bytes = new Uint8Array(transitionCount * 9 * 8);
  const view = new DataView(bytes.buffer);
  let offset = 0;
  for (let q0 = input0.qmin; q0 <= input0.qmax; q0 += 1) {
    for (let q1 = input1.qmin; q1 <= input1.qmax; q1 += 1) {
      const base = project(q0, q1, input0, input1, output);
      if (q0 < input0.qmax) offset = writeLedgerRow(view, offset, 0, q0, q1, base, project(q0 + 1, q1, input0, input1, output), output);
      if (q1 < input1.qmax) offset = writeLedgerRow(view, offset, 1, q0, q1, base, project(q0, q1 + 1, input0, input1, output), output);
    }
  }
  if (offset !== bytes.byteLength) throw new Error("Step-response transition ledger byte count is invalid.");
  return bytes;
}

function writeLedgerRow(view, offset, branch, q0, q1, base, adjacent, output) {
  const delta = adjacent.code - base.code;
  const unclipped = Number(inRange(base.raw, output) && inRange(adjacent.raw, output));
  for (const value of [branch, q0, q1, base.raw, adjacent.raw, base.code, adjacent.code, delta, unclipped]) {
    view.setBigInt64(offset, BigInt(value), true);
    offset += 8;
  }
  return offset;
}

function compareRow(actual, expected, opIndex) {
  if (actual.assessment_status !== "assessed" || actual.contracts?.length !== 3) throw new Error(`Step-response contract coverage is invalid at #${opIndex}.`);
  for (let index = 0; index < 3; index += 1) compareContract(actual.contracts[index], expected.contracts[index], opIndex);
  assertNear(actual.maximum_containment_silent_ratio_increase, expected.maximum_containment_silent_ratio_increase, `#${opIndex} silent-ratio increase`);
  assertEqual(actual.maximum_containment_additional_silent_transitions, expected.maximum_containment_additional_silent_transitions, `#${opIndex} additional silent transitions`);
  assertEqual(actual.maximum_containment_removed_clamp_pairs, expected.maximum_containment_removed_clamp_pairs, `#${opIndex} removed clamps`);
}

function compareContract(actual, expected, opIndex) {
  for (const key of ["design", "output_zero_point", "rounded_projection_clamp_pair_count", "complete_rounded_domain_containment", "distinct_projected_output_code_count", "total_transition_count", "visible_transition_count", "silent_transition_count", "joint_interior_cell_count", "both_branches_visible_cell_count", "input_0_only_visible_cell_count", "input_1_only_visible_cell_count", "neither_branch_visible_cell_count", "tile_size_codes", "tile_grid_dimension", "removed_rounded_clamp_pairs_vs_current", "additional_silent_transitions_vs_current"]) {
    assertEqual(actual[key], expected[key], `#${opIndex} ${actual.design} ${key}`);
  }
  for (const key of ["output_scale", "scale_ratio_to_current", "rounded_projection_clamp_pair_ratio", "visible_transition_ratio", "silent_transition_ratio", "both_branches_visible_ratio", "neither_branch_visible_ratio", "visible_transition_ratio_delta_vs_current"]) assertNear(actual[key], expected[key], `#${opIndex} ${actual.design} ${key}`);
  for (const key of ["tile_joint_cell_counts", "tile_both_branches_visible_counts", "tile_input_0_only_visible_counts", "tile_input_1_only_visible_counts", "tile_neither_branch_visible_counts"]) {
    if (!sameArray(actual[key], expected[key])) throw new Error(`Step-response ${key} is invalid at #${opIndex} ${actual.design}.`);
  }
  if (!/^[a-f0-9]{64}$/.test(actual.transition_ledger_sha256 || "") || actual.branch_responses?.length !== 2) throw new Error(`Step-response ledger/branch shape is invalid at #${opIndex}.`);
  actual.branch_responses.forEach((branch, index) => compareBranch(branch, expected.branch_responses[index], opIndex, actual.design));
}

function compareBranch(actual, expected, opIndex, design) {
  for (const key of ["branch_index", "input_tensor_index", "transition_count", "visible_transition_count", "silent_transition_count", "unclipped_transition_count", "unclipped_silent_transition_count", "clamp_associated_transition_count", "clamp_associated_silent_transition_count", "multi_code_jump_transition_count", "maximum_output_code_delta"]) assertEqual(actual[key], expected[key], `#${opIndex} ${design} branch ${actual.branch_index} ${key}`);
  for (const key of ["input_scale", "visible_transition_ratio", "silent_transition_ratio", "mean_output_code_delta", "mean_absolute_step_reproduction_error_output_steps", "maximum_absolute_step_reproduction_error_output_steps"]) assertNear(actual[key], expected[key], `#${opIndex} ${design} branch ${actual.branch_index} ${key}`);
  if (!sameArray(actual.output_code_delta_histogram, expected.output_code_delta_histogram)
    || JSON.stringify(actual.worst_jump ?? null) !== JSON.stringify(expected.worst_jump ?? null)
    || JSON.stringify(actual.first_unclipped_silent ?? null) !== JSON.stringify(expected.first_unclipped_silent ?? null)) throw new Error(`Step-response branch witnesses are invalid at #${opIndex} ${design}.`);
}

function quantContract(tensors, index) {
  const tensor = tensors.find((item) => item.index === index);
  if (!tensor || !["INT8", "UINT8"].includes(tensor.dtype) || tensor.scale_sample?.length !== 1 || tensor.zero_point_sample?.length !== 1) throw new Error(`Tensor T${index} lacks a per-tensor 8-bit contract.`);
  const qmin = tensor.dtype === "INT8" ? -128 : 0;
  const qmax = tensor.dtype === "INT8" ? 127 : 255;
  const scale = Number(tensor.scale_sample[0]);
  const zeroPoint = Number(tensor.zero_point_sample[0]);
  if (!(scale > 0) || !Number.isFinite(scale) || zeroPoint < qmin || zeroPoint > qmax) throw new Error(`Tensor T${index} quantization metadata is invalid.`);
  return { index, qmin, qmax, scale, zeroPoint };
}

function project(q0, q1, input0, input1, output) {
  const real = (q0 - input0.zeroPoint) * input0.scale + (q1 - input1.zeroPoint) * input1.scale;
  const raw = roundTiesAway(real / output.scale) + output.zeroPoint;
  return { raw, code: Math.max(output.qmin, Math.min(output.qmax, raw)) };
}

function inRange(code, contract) { return code >= contract.qmin && code <= contract.qmax; }

function renderSummary(root, result) {
  root?.replaceChildren(
    metric("Exact transitions", formatNumber(result.total_transition_count), `${formatNumber(result.total_joint_interior_cell_count)} joint cells`),
    metric("Current silent", formatNumber(result.current_silent_transition_count), `${formatNumber(result.current_rounded_projection_clamp_pair_count)} clamp pairs`),
    metric("Containment trade", signed(result.containment_additional_silent_transition_count), `${formatNumber(result.containment_removed_rounded_clamp_pair_count)} clamp pairs removed`),
    metric("Worst visibility loss", percent(result.maximum_containment_silent_ratio_increase), `#${padOp(result.retention_cost_ranking_op_indices[0])} ADD`),
  );
}

function opSelector(result, selected) {
  return element("div", "step-response-op-selector", result.residual_adds.filter((row) => row.assessment_status === "assessed").map((row) => {
    const button = element("button", `step-response-op-button${row.op_index === selected ? " active" : ""}`, `#${padOp(row.op_index)}`);
    button.type = "button";
    button.dataset.stepResponseOp = row.op_index;
    button.title = `ADD #${row.op_index}: maximum containment visibility loss ${percent(row.maximum_containment_silent_ratio_increase)}`;
    return button;
  }));
}

function designTabs(contracts, selected) {
  return element("div", "step-response-design-tabs", contracts.map((contract) => {
    const button = element("button", `step-response-design-button${contract.design === selected ? " active" : ""}`, designLabel(contract.design));
    button.type = "button";
    button.dataset.stepResponseDesign = contract.design;
    return button;
  }));
}

function headline(row, contract) {
  const graph = element("button", "icon-button step-response-graph-button", "Graph");
  graph.type = "button";
  graph.dataset.stepResponseOpenGraph = row.op_index;
  graph.title = `Open ADD #${row.op_index} in Graph Explorer`;
  return element("section", "step-response-headline", [
    element("div", "step-response-headline-title", [element("strong", "", `#${padOp(row.op_index)} ADD`), graph]),
    element("div", "step-response-headline-metrics", [
      metric("Output contract", `${number(contract.output_scale)} / zp ${contract.output_zero_point}`, `${number(contract.scale_ratio_to_current)}x current scale`),
      metric("Clamp pairs", formatNumber(contract.rounded_projection_clamp_pair_count), `${signed(contract.removed_rounded_clamp_pairs_vs_current)} removed vs current`),
      metric("Silent transitions", formatNumber(contract.silent_transition_count), `${signed(contract.additional_silent_transitions_vs_current)} vs current`),
      metric("Both visible", percent(contract.both_branches_visible_ratio), `${percent(contract.neither_branch_visible_ratio)} neither`),
    ]),
  ]);
}

function canvasPanel(title, detail, canvas, legend = null) {
  return element("section", "step-response-canvas-panel", [
    element("div", "step-response-canvas-heading", [element("strong", "", title), element("span", "", detail)]),
    canvas,
    legend,
  ].filter(Boolean));
}

function influenceLegend() {
  return element("div", "step-response-legend", [
    legend("both visible", "#5ee0b7"), legend("input 0 only", "#5da9ff"), legend("input 1 only", "#f6b85f"), legend("neither", "#f06d76"),
  ]);
}

function branchTable(contract) {
  return tableSection("Branch Step Ledger", ["Branch", "Input", "Visible", "Silent", "Unclipped silent", "Clamp silent", ">1 jump", "Mean delta", "Mean error"], contract.branch_responses.map((branch) => [
    `input ${branch.branch_index}`,
    `T${branch.input_tensor_index} / s ${number(branch.input_scale)}`,
    `${formatNumber(branch.visible_transition_count)} (${percent(branch.visible_transition_ratio)})`,
    `${formatNumber(branch.silent_transition_count)} (${percent(branch.silent_transition_ratio)})`,
    formatNumber(branch.unclipped_silent_transition_count),
    formatNumber(branch.clamp_associated_silent_transition_count),
    formatNumber(branch.multi_code_jump_transition_count),
    number(branch.mean_output_code_delta),
    `${number(branch.mean_absolute_step_reproduction_error_output_steps)} output steps`,
  ]));
}

function contractTable(row) {
  return tableSection("Contract Comparison", ["Contract", "Scale / zp", "Clamp pairs", "Visible", "Silent", "Both / neither", "Trade vs current", "Ledger SHA-256"], row.contracts.map((contract) => [
    designLabel(contract.design),
    `${number(contract.output_scale)} / ${contract.output_zero_point}`,
    formatNumber(contract.rounded_projection_clamp_pair_count),
    percent(contract.visible_transition_ratio),
    formatNumber(contract.silent_transition_count),
    `${percent(contract.both_branches_visible_ratio)} / ${percent(contract.neither_branch_visible_ratio)}`,
    `${signed(contract.removed_rounded_clamp_pairs_vs_current)} clamps; ${signed(contract.additional_silent_transitions_vs_current)} silent`,
    contract.transition_ledger_sha256,
  ]));
}

function portfolioTable(result, selected) {
  return tableSection("Residual Portfolio", ["Rank", "ADD", "Maximum visibility loss", "Additional silent", "Clamp pairs removed"], result.residual_adds.filter((row) => row.assessment_status === "assessed").sort((a, b) => a.retention_cost_rank - b.retention_cost_rank).map((row) => [
    row.retention_cost_rank,
    row.op_index === selected ? `#${padOp(row.op_index)} selected` : `#${padOp(row.op_index)}`,
    percent(row.maximum_containment_silent_ratio_increase),
    signed(row.maximum_containment_additional_silent_transitions),
    formatNumber(row.maximum_containment_removed_clamp_pairs),
  ]));
}

function methodBoundary(result) {
  return element("section", "step-response-method", [
    element("strong", "", "Method and evidence boundary"),
    element("p", "", result.transition_definition),
    element("p", "", result.joint_cell_definition),
    element("p", "", result.transition_ledger_hash_method),
    element("p", "", result.interpretation_boundary),
  ]);
}

function drawInfluenceField(canvas, row, contract, analysis) {
  const ctx = setupCanvas(canvas, 620, 470);
  if (!ctx) return;
  const { width, height } = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, width, height);
  const left = 56;
  const top = 28;
  const size = Math.min(width - 88, height - 86);
  const classes = reconstructInfluenceClasses(row, contract, analysis);
  const field = document.createElement("canvas");
  field.width = 255;
  field.height = 255;
  const fieldContext = field.getContext("2d");
  const image = fieldContext.createImageData(255, 255);
  const colors = [[94, 224, 183], [93, 169, 255], [246, 184, 95], [240, 109, 118]];
  classes.forEach((classIndex, index) => {
    const color = colors[classIndex];
    const offset = index * 4;
    image.data[offset] = color[0];
    image.data[offset + 1] = color[1];
    image.data[offset + 2] = color[2];
    image.data[offset + 3] = 255;
  });
  fieldContext.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(field, left, top, size, size);
  ctx.strokeStyle = "rgba(219,231,238,.35)";
  ctx.strokeRect(left, top, size, size);
  ctx.fillStyle = "#aebdc7";
  ctx.font = "12px system-ui";
  ctx.fillText("input 1 code ->", left, top + size + 24);
  ctx.save();
  ctx.translate(18, top + size);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("input 0 code ->", 0, 0);
  ctx.restore();
}

function drawTradeoff(canvas, row) {
  const ctx = setupCanvas(canvas, 620, 470);
  if (!ctx) return;
  const { width, height } = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, width, height);
  const margin = { left: 62, right: 28, top: 32, bottom: 58 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  const maxClamp = Math.max(1, ...row.contracts.map((contract) => contract.rounded_projection_clamp_pair_count));
  const maxSilent = Math.max(1, ...row.contracts.map((contract) => contract.silent_transition_count));
  ctx.strokeStyle = "rgba(180,199,210,.28)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = margin.top + h * i / 4;
    ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + w, y); ctx.stroke();
  }
  const points = row.contracts.map((contract) => ({
    contract,
    x: margin.left + contract.rounded_projection_clamp_pair_count / maxClamp * w,
    y: margin.top + h - contract.silent_transition_count / maxSilent * h,
  }));
  ctx.strokeStyle = "#8ea6b5";
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
  ctx.stroke();
  const colors = ["#dbe7ee", "#f6b85f", "#5ee0b7"];
  points.forEach((point, index) => {
    ctx.fillStyle = colors[index];
    ctx.beginPath(); ctx.arc(point.x, point.y, 7, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#dbe7ee";
    ctx.font = "12px system-ui";
    const label = ["current", "fixed-zp", "global"][index];
    const labelY = index === 1 ? point.y - 12 : index === 2 ? point.y + 18 : point.y - 10;
    ctx.fillText(label, Math.min(point.x + 10, width - 70), Math.max(16, labelY));
    ctx.font = "10px system-ui";
    ctx.fillText(formatNumber(point.contract.silent_transition_count), Math.min(point.x + 10, width - 70), Math.max(28, labelY + 13));
  });
  ctx.fillStyle = "#aebdc7";
  ctx.font = "12px system-ui";
  ctx.fillText("rounded clamp pairs ->", margin.left, height - 18);
  ctx.save();
  ctx.translate(18, margin.top + h);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("silent branch transitions ->", 0, 0);
  ctx.restore();
}

function setupCanvas(canvas, logicalWidth, logicalHeight) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(280, rect.width || logicalWidth);
  const height = Math.max(300, rect.height || logicalHeight);
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return ctx;
}

function reconstructInfluenceClasses(row, contract, analysis) {
  const op = (analysis?.ops || []).find((item) => item.index === row.op_index && item.name === "ADD");
  if (!op) throw new Error(`ADD #${row.op_index} is unavailable for influence-field rendering.`);
  const input0 = quantContract(analysis.tensors || [], op.inputs[0]);
  const input1 = quantContract(analysis.tensors || [], op.inputs[1]);
  const currentOutput = quantContract(analysis.tensors || [], op.outputs[0]);
  const output = { ...currentOutput, scale: Number(contract.output_scale), zeroPoint: Number(contract.output_zero_point) };
  const classes = new Uint8Array(255 * 255);
  let index = 0;
  for (let q0 = input0.qmin; q0 < input0.qmax; q0 += 1) {
    for (let q1 = input1.qmin; q1 < input1.qmax; q1 += 1) {
      const base = project(q0, q1, input0, input1, output).code;
      const visible0 = project(q0 + 1, q1, input0, input1, output).code !== base;
      const visible1 = project(q0, q1 + 1, input0, input1, output).code !== base;
      classes[index] = visible0 ? (visible1 ? 0 : 1) : (visible1 ? 2 : 3);
      index += 1;
    }
  }
  return classes;
}

function tableSection(title, headers, rows) {
  const table = element("table", "step-response-table");
  const head = element("thead", "", element("tr", "", headers.map((header) => element("th", "", header))));
  const body = element("tbody", "", rows.map((row) => element("tr", "", row.map((cell) => element("td", "", String(cell))))));
  table.append(head, body);
  return element("section", "step-response-table-section", [element("strong", "", title), element("div", "step-response-table-scroll", table)]);
}

function metric(label, value, detail) {
  return element("div", "step-response-metric", [element("span", "", label), element("strong", "", String(value)), element("small", "", detail)]);
}

function legend(labelText, color) {
  const swatch = element("i", "");
  swatch.style.background = color;
  return element("span", "", [swatch, document.createTextNode(labelText)]);
}

function element(tag, className = "", children = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (children != null) {
    if (Array.isArray(children)) node.append(...children.filter(Boolean).map((child) => child instanceof Node ? child : document.createTextNode(String(child))));
    else if (children instanceof Node) node.append(children);
    else node.textContent = String(children);
  }
  return node;
}

function messageNode(text, tone = "") { return element("p", `step-response-message ${tone}`, text); }
function designLabel(design) { return design === DESIGNS[0] ? "Current" : design === DESIGNS[1] ? "Fixed zero-point" : "Globally finest"; }
function number(value) { return Number(value).toPrecision(7); }
function percent(value) { return value == null ? "N/A" : `${(Number(value) * 100).toFixed(3)}%`; }
function signed(value) { const numberValue = Number(value || 0); return `${numberValue >= 0 ? "+" : ""}${formatNumber(numberValue)}`; }
function sum(rows, key) { return rows.reduce((total, row) => total + Number(row[key] || 0), 0); }
function sameArray(left, right) { return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index]); }
function assertEqual(actual, expected, label) { if (actual !== expected) throw new Error(`${label} mismatch (${actual} != ${expected}).`); }
function assertNear(actual, expected, label) {
  const tolerance = Math.max(1e-12, Math.abs(Number(expected)) * 1e-11);
  if (!Number.isFinite(Number(actual)) || Math.abs(Number(actual) - Number(expected)) > tolerance) throw new Error(`${label} mismatch (${actual} != ${expected}).`);
}
