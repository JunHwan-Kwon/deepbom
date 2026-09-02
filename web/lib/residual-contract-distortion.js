import { formatNumber, padOp } from "./format.js";
import { sha256Hex } from "./hash.js";
import { roundTiesAway } from "./quantization-math.js";

export const RESIDUAL_CONTRACT_DISTORTION_SCHEMA = "deepbom.residual_contract_distortion.v1.1";

const METHOD_VERSION = "2026-07-29.1";
const DESIGNS = ["fixed_zero_point_minimum_containment", "globally_finest_minimum_containment"];
const VIEWS = ["signed_delta", "ideal_error", "clamp_state"];
const TILE_SIZE = 16;
const HISTOGRAM_BINS = 32;

export function createResidualContractDistortionController({ root, status, summary, body, downloadButton, getAnalysis, jumpToGraphOp, onDownload, residualSelection }) {
  let selectedOp = null;
  let selectedDesign = DESIGNS[1];
  let selectedView = VIEWS[0];
  let current = null;
  let renderToken = 0;
  let worker = null;
  let resizeObserver = null;
  const selectionSource = "residual-contract-distortion";

  root?.addEventListener("click", (event) => {
    const op = event.target.closest("[data-distortion-op]");
    const design = event.target.closest("[data-distortion-design]");
    const view = event.target.closest("[data-distortion-view]");
    const graph = event.target.closest("[data-distortion-graph]");
    if (op) {
      selectedOp = Number(op.dataset.distortionOp);
      residualSelection?.set({ opIndex: selectedOp }, selectionSource);
      renderBody();
    }
    else if (design && DESIGNS.includes(design.dataset.distortionDesign)) {
      selectedDesign = design.dataset.distortionDesign;
      residualSelection?.set({ design: selectedDesign }, selectionSource);
      renderBody();
    }
    else if (view && VIEWS.includes(view.dataset.distortionView)) { selectedView = view.dataset.distortionView; renderBody(); }
    else if (graph) jumpToGraphOp?.(Number(graph.dataset.distortionGraph));
  });
  residualSelection?.subscribe((selection, source) => {
    if (source === selectionSource || !current?.residual_contract_distortion) return;
    const assessed = current.residual_contract_distortion.residual_adds.filter((row) => row.assessment_status === "assessed");
    if (assessed.some((row) => row.op_index === selection.opIndex)) selectedOp = selection.opIndex;
    if (DESIGNS.includes(selection.design)) selectedDesign = selection.design;
    renderBody();
  });
  downloadButton?.addEventListener("click", () => {
    const result = getAnalysis?.()?.residual_contract_distortion;
    if (result) onDownload?.(result, "residual_contract_distortion.json");
  });

  function render(analysis = null) {
    const token = ++renderToken;
    current = analysis || getAnalysis?.() || null;
    const result = current?.residual_contract_distortion;
    worker?.terminate();
    worker = null;
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
      validateResidualContractDistortionEnvelope(result, current);
      const assessed = result.residual_adds.filter((row) => row.assessment_status === "assessed");
      const shared = residualSelection?.get();
      if (assessed.some((row) => row.op_index === shared?.opIndex)) selectedOp = shared.opIndex;
      if (DESIGNS.includes(shared?.design)) selectedDesign = shared.design;
      if (!assessed.some((row) => row.op_index === selectedOp)) selectedOp = result.distortion_ranking_op_indices[0] ?? assessed[0]?.op_index ?? null;
      residualSelection?.set({ opIndex: selectedOp, design: selectedDesign }, selectionSource);
      renderSummary(summary, result);
      renderBody();
      setStatus("evidence loaded / verification pending", "ok");
      void runIndependentVerification(result, current, (next) => { worker = next; }).then(() => {
        if (token === renderToken) setStatus("independently verified", "ok");
      }).catch((error) => {
        if (token === renderToken) setStatus(`integrity error: ${error.message}`, "risk");
      });
    } catch (error) {
      summary?.replaceChildren();
      body?.replaceChildren(messageNode(`Residual distortion evidence rejected: ${error.message}`, "risk"));
      setStatus("evidence rejected", "risk");
    }
  }

  function setStatus(text, tone) {
    if (!status) return;
    status.textContent = text;
    status.dataset.tone = tone;
  }

  function renderBody() {
    if (!body || !current?.residual_contract_distortion) return;
    resizeObserver?.disconnect();
    resizeObserver = null;
    const result = current.residual_contract_distortion;
    const row = result.residual_adds.find((item) => item.op_index === selectedOp)
      || result.residual_adds.find((item) => item.assessment_status === "assessed");
    if (!row) { body.replaceChildren(messageNode("No residual contract-distortion scenario is assessable.")); return; }
    const scenario = row.scenarios.find((item) => item.design === selectedDesign) || row.scenarios[0];
    selectedDesign = scenario.design;
    const viewMode = selectedView;
    const renderKey = `${row.op_index}|${scenario.design}|${viewMode}`;
    const field = element("canvas", "step-response-field-canvas distortion-field-canvas");
    field.setAttribute("aria-label", "Exact residual output-contract distortion field");
    const histogram = element("canvas", "step-response-tradeoff-canvas distortion-histogram-canvas");
    histogram.setAttribute("aria-label", "Residual contract distortion histogram and quantiles");
    body.replaceChildren(
      opSelector(result, row.op_index),
      candidateTabs(row.scenarios, scenario.design),
      equivalentScenarioNote(row.scenarios),
      viewTabs(viewMode),
      headline(row, scenario),
      element("div", "step-response-canvas-grid", [
        canvasPanel("Exact Counterfactual Field", viewDescription(viewMode), field, fieldLegend(viewMode)),
        canvasPanel("Displacement Distribution", "All 65,536 pair displacements in current output steps; bar height uses log pair count.", histogram),
      ]),
      scenarioTable(row),
      witnessTable(scenario),
      portfolioTable(result, row.op_index),
      methodBoundary(result),
    );
    const draw = () => {
      drawDistortionField(field, row, scenario, current, viewMode);
      drawHistogram(histogram, scenario);
      field.dataset.distortionRenderKey = renderKey;
      histogram.dataset.distortionRenderKey = renderKey;
    };
    requestAnimationFrame(draw);
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver(draw);
      resizeObserver.observe(field.parentElement);
      resizeObserver.observe(histogram.parentElement);
    }
  }

  return { render };
}

function validateResidualContractDistortionEnvelope(result, analysis) {
  if (!result || result.schema !== RESIDUAL_CONTRACT_DISTORTION_SCHEMA || result.method_version !== METHOD_VERSION || result.evidence_class !== "DERIVED") throw new Error("Distortion envelope identity is invalid.");
  const addCount = (analysis?.ops || []).filter((op) => op.name === "ADD").length;
  const scenarios = (result.residual_adds || []).flatMap((row) => row.scenarios || []);
  if (!Array.isArray(result.residual_adds) || result.candidate_add_count !== addCount || result.residual_adds.length !== addCount
    || scenarios.length !== result.scenario_count || !scenarios.every((scenario) => /^[a-f0-9]{64}$/.test(scenario.pair_ledger_sha256 || ""))) throw new Error("Distortion envelope coverage is invalid.");
  return true;
}

async function runIndependentVerification(result, analysis, onWorker) {
  if (typeof Worker === "function") {
    const worker = new Worker(new URL("./residual-contract-distortion-worker.js", import.meta.url), { type: "module" });
    onWorker?.(worker);
    return new Promise((resolve, reject) => {
      worker.onmessage = (event) => {
        worker.terminate();
        if (event.data?.ok) resolve(true);
        else reject(new Error(event.data?.error || "Distortion worker rejected the evidence."));
      };
      worker.onerror = (event) => { worker.terminate(); reject(new Error(event.message || "Distortion worker failed.")); };
      worker.postMessage({ result, analysis: verificationAnalysis(analysis) });
    });
  }
  await new Promise((resolve) => setTimeout(resolve, 0));
  return validateResidualContractDistortionDigests(result, analysis);
}

function verificationAnalysis(analysis) {
  return {
    format: analysis?.format,
    ops: (analysis?.ops || []).filter((op) => op.name === "ADD").map((op) => ({ index: op.index, name: op.name, inputs: op.inputs, outputs: op.outputs })),
    tensors: (analysis?.tensors || []).map((tensor) => ({ index: tensor.index, dtype: tensor.dtype, scale_sample: tensor.scale_sample, zero_point_sample: tensor.zero_point_sample })),
    quantization_lattice: { residual_adds: (analysis?.quantization_lattice?.residual_adds || []).map((row) => ({ op_index: row.op_index, fixed_zero_point_containment: row.fixed_zero_point_containment, globally_finest_containment: row.globally_finest_containment })) },
  };
}

export function validateResidualContractDistortion(result, analysis) {
  validateResidualContractDistortionEnvelope(result, analysis);
  const addOps = (analysis?.ops || []).filter((op) => op.name === "ADD");
  const latticeRows = analysis?.quantization_lattice?.residual_adds || [];
  if (latticeRows.length !== addOps.length) throw new Error("Distortion lattice coverage is invalid.");
  const reconstructed = [];
  for (let index = 0; index < addOps.length; index += 1) {
    const op = addOps[index];
    const lattice = latticeRows[index];
    const emitted = result.residual_adds[index];
    if (op.index !== lattice.op_index || emitted.op_index !== op.index) throw new Error(`Distortion op binding is invalid at row ${index}.`);
    let expected;
    try { expected = reconstructRow(op, analysis.tensors || [], lattice); }
    catch (error) { expected = { op_index: op.index, assessment_status: "not_assessed", reason: error.message, scenarios: [] }; }
    if (expected.assessment_status !== "assessed") {
      if (emitted.assessment_status !== "not_assessed" || emitted.scenarios?.length) throw new Error(`Distortion unassessed state is invalid at #${op.index}.`);
    } else compareRow(emitted, expected, op.index);
    reconstructed.push(expected);
  }
  const assessed = reconstructed.filter((row) => row.assessment_status === "assessed");
  const scenarios = assessed.flatMap((row) => row.scenarios);
  const ranked = [...assessed].sort((left, right) => right.maximum_rms_contract_delta_current_steps - left.maximum_rms_contract_delta_current_steps
    || right.maximum_p99_contract_delta_current_steps - left.maximum_p99_contract_delta_current_steps || left.op_index - right.op_index);
  if (!sameArray(result.distortion_ranking_op_indices, ranked.map((row) => row.op_index))) throw new Error("Distortion ranking is invalid.");
  ranked.forEach((row, index) => assertEqual(result.residual_adds.find((item) => item.op_index === row.op_index)?.distortion_rank, index + 1, `#${row.op_index} rank`));
  const aggregateKeys = [
    "total_enumerated_pair_count", "scenario_current_clamped_pair_instance_count", "candidate_clamped_pair_count",
    "rescued_current_clamp_pair_instance_count", "changed_represented_value_pair_count", "ideal_error_improved_pair_count",
    "ideal_error_worsened_pair_count", "ideal_error_equal_within_tolerance_pair_count", "sign_class_changed_pair_count",
  ];
  const scenarioKeys = [
    "enumerated_pair_count", "current_clamped_pair_count", "candidate_clamped_pair_count", "rescued_current_clamp_pair_count",
    "changed_represented_value_pair_count", "ideal_error_improved_pair_count", "ideal_error_worsened_pair_count",
    "ideal_error_equal_within_tolerance_pair_count", "sign_class_changed_pair_count",
  ];
  aggregateKeys.forEach((key, index) => assertEqual(result[key], sum(scenarios, scenarioKeys[index]), key));
  assertEqual(
    result.current_clamped_pair_instance_count,
    sum(assessed.map((row) => row.scenarios[0]).filter(Boolean), "current_clamped_pair_count"),
    "unique current clamp count",
  );
  assertEqual(result.assessed_add_count, assessed.length, "assessed ADD count");
  assertEqual(result.unassessed_add_count, result.residual_adds.length - assessed.length, "unassessed ADD count");
  assertEqual(result.scenario_count, scenarios.length, "scenario count");
  if (scenarios.length) {
    assertNear(result.maximum_rms_contract_delta_current_steps, Math.max(...scenarios.map((row) => row.root_mean_square_contract_delta_current_steps)), "maximum RMS");
    assertNear(result.maximum_p99_contract_delta_current_steps, Math.max(...scenarios.map((row) => row.p99_absolute_contract_delta_current_steps)), "maximum p99");
  }
  if (!/uniform legal-code-domain counterfactual/i.test(result.interpretation_boundary || "")
    || !/not an observed activation distribution/i.test(result.interpretation_boundary || "")
    || !/nine signed i64/i.test(result.pair_ledger_hash_method || "")
    || !/six IEEE-754 binary64/i.test(result.pair_ledger_hash_method || "")) throw new Error("Distortion evidence boundary is incomplete.");
  return reconstructed;
}

export async function validateResidualContractDistortionDigests(result, analysis) {
  validateResidualContractDistortion(result, analysis);
  const ops = new Map((analysis?.ops || []).filter((op) => op.name === "ADD").map((op) => [op.index, op]));
  const lattices = new Map((analysis?.quantization_lattice?.residual_adds || []).map((row) => [row.op_index, row]));
  for (const row of result.residual_adds.filter((item) => item.assessment_status === "assessed")) {
    const contracts = contractsFor(ops.get(row.op_index), analysis.tensors || [], lattices.get(row.op_index));
    for (const emitted of row.scenarios) {
      const contract = contracts.find((item) => item.design === emitted.design);
      const digest = await sha256Hex(pairLedgerBytes(contract));
      if (digest !== emitted.pair_ledger_sha256) throw new Error(`Distortion pair-ledger digest mismatch at #${row.op_index} ${emitted.design}.`);
    }
  }
  return true;
}

function reconstructRow(op, tensors, lattice) {
  const scenarios = contractsFor(op, tensors, lattice).map(evaluateScenario);
  return {
    op_index: op.index,
    assessment_status: "assessed",
    scenarios,
    maximum_rms_contract_delta_current_steps: Math.max(...scenarios.map((row) => row.root_mean_square_contract_delta_current_steps)),
    maximum_p99_contract_delta_current_steps: Math.max(...scenarios.map((row) => row.p99_absolute_contract_delta_current_steps)),
    maximum_rescued_current_clamp_pair_count: Math.max(...scenarios.map((row) => row.rescued_current_clamp_pair_count)),
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
  return [[DESIGNS[0], fixed], [DESIGNS[1], global]].map(([design, candidate]) => ({
    design, input0, input1, current,
    candidate: { ...current, scale: Number(candidate.output_scale), zeroPoint: Number(candidate.output_zero_point) },
  }));
}

function evaluateScenario(contract) {
  const { design, input0, input1, current, candidate } = contract;
  const pairCount = 65_536;
  const tolerance = Math.max(current.scale, candidate.scale) * 2 ** -40;
  const tiles = Array.from({ length: 9 }, () => Array(256).fill(0));
  const absoluteDeltas = [];
  let currentClamped = 0; let candidateClamped = 0; let rescued = 0; let persistent = 0; let introduced = 0;
  let same = 0; let signChanged = 0; let improved = 0; let worsened = 0; let equal = 0;
  let signedSum = 0; let absoluteSum = 0; let squaredSum = 0; let currentErrorSum = 0; let candidateErrorSum = 0;
  let withinHalf = 0; let withinOne = 0; let withinTwo = 0; let worst = null;
  for (let q0 = input0.qmin; q0 <= input0.qmax; q0 += 1) {
    for (let q1 = input1.qmin; q1 <= input1.qmax; q1 += 1) {
      const pair = projectPair(q0, q1, input0, input1, current, candidate, tolerance);
      const { currentProjection: a, candidateProjection: b, deltaReal, deltaSteps, absoluteDeltaSteps, relation } = pair;
      currentClamped += Number(a.clipped); candidateClamped += Number(b.clipped);
      rescued += Number(a.clipped && !b.clipped); persistent += Number(a.clipped && b.clipped); introduced += Number(!a.clipped && b.clipped);
      same += Number(a.real === b.real); signChanged += Number(signClass(a.real) !== signClass(b.real));
      if (relation < 0) improved += 1; else if (relation > 0) worsened += 1; else equal += 1;
      signedSum += deltaReal; absoluteSum += Math.abs(deltaReal); squaredSum += deltaReal * deltaReal;
      currentErrorSum += a.error; candidateErrorSum += b.error;
      withinHalf += Number(absoluteDeltaSteps <= 0.5); withinOne += Number(absoluteDeltaSteps <= 1); withinTwo += Number(absoluteDeltaSteps <= 2);
      absoluteDeltas.push(absoluteDeltaSteps);
      const tile = Math.floor((q0 - input0.qmin) / TILE_SIZE) * 16 + Math.floor((q1 - input1.qmin) / TILE_SIZE);
      tiles[0][tile] += 1; tiles[1][tile] += deltaSteps; tiles[2][tile] += absoluteDeltaSteps; tiles[3][tile] = Math.max(tiles[3][tile], absoluteDeltaSteps);
      tiles[4][tile] += Number(a.clipped && !b.clipped); tiles[5][tile] += Number(relation < 0); tiles[6][tile] += Number(relation > 0); tiles[7][tile] += Number(relation === 0); tiles[8][tile] += Number(signClass(a.real) !== signClass(b.real));
      if (!worst || absoluteDeltaSteps > worst.absolute_contract_delta_current_steps) worst = witness(q0, q1, pair);
    }
  }
  absoluteDeltas.sort((left, right) => left - right);
  const maximum = absoluteDeltas.at(-1) || 0;
  const width = maximum > 0 ? maximum / HISTOGRAM_BINS : 1;
  const histogram = Array(HISTOGRAM_BINS).fill(0);
  absoluteDeltas.forEach((value) => { histogram[Math.min(HISTOGRAM_BINS - 1, Math.floor(value / width))] += 1; });
  const tileMean = (sums) => sums.map((sum, index) => sum / Math.max(1, tiles[0][index]));
  const meanCurrentError = currentErrorSum / pairCount;
  const meanCandidateError = candidateErrorSum / pairCount;
  return {
    design,
    candidate_output_scale: candidate.scale,
    candidate_output_zero_point: candidate.zeroPoint,
    candidate_scale_ratio_to_current: candidate.scale / current.scale,
    candidate_signed_zero_point_delta: candidate.zeroPoint - current.zeroPoint,
    enumerated_pair_count: pairCount,
    current_clamped_pair_count: currentClamped,
    candidate_clamped_pair_count: candidateClamped,
    rescued_current_clamp_pair_count: rescued,
    persistent_clamp_pair_count: persistent,
    introduced_clamp_pair_count: introduced,
    same_represented_value_pair_count: same,
    changed_represented_value_pair_count: pairCount - same,
    sign_class_changed_pair_count: signChanged,
    ideal_error_improved_pair_count: improved,
    ideal_error_worsened_pair_count: worsened,
    ideal_error_equal_within_tolerance_pair_count: equal,
    error_comparison_tolerance_real: tolerance,
    mean_signed_contract_delta_real: signedSum / pairCount,
    mean_absolute_contract_delta_real: absoluteSum / pairCount,
    root_mean_square_contract_delta_real: Math.sqrt(squaredSum / pairCount),
    mean_signed_contract_delta_current_steps: signedSum / pairCount / current.scale,
    mean_absolute_contract_delta_current_steps: absoluteSum / pairCount / current.scale,
    root_mean_square_contract_delta_current_steps: Math.sqrt(squaredSum / pairCount) / current.scale,
    maximum_absolute_contract_delta_current_steps: maximum,
    p50_absolute_contract_delta_current_steps: quantile(absoluteDeltas, 0.50),
    p90_absolute_contract_delta_current_steps: quantile(absoluteDeltas, 0.90),
    p99_absolute_contract_delta_current_steps: quantile(absoluteDeltas, 0.99),
    within_half_current_step_pair_count: withinHalf,
    within_one_current_step_pair_count: withinOne,
    within_two_current_steps_pair_count: withinTwo,
    mean_absolute_ideal_error_current: meanCurrentError,
    mean_absolute_ideal_error_candidate: meanCandidateError,
    signed_mean_absolute_ideal_error_delta: meanCandidateError - meanCurrentError,
    absolute_delta_histogram_bin_width_current_steps: width,
    absolute_delta_histogram_counts: histogram,
    tile_size_codes: TILE_SIZE,
    tile_grid_dimension: 16,
    tile_pair_counts: tiles[0],
    tile_mean_signed_delta_current_steps: tileMean(tiles[1]),
    tile_mean_absolute_delta_current_steps: tileMean(tiles[2]),
    tile_maximum_absolute_delta_current_steps: tiles[3],
    tile_rescued_current_clamp_pair_counts: tiles[4],
    tile_ideal_error_improved_pair_counts: tiles[5],
    tile_ideal_error_worsened_pair_counts: tiles[6],
    tile_ideal_error_equal_pair_counts: tiles[7],
    tile_sign_class_changed_pair_counts: tiles[8],
    worst_absolute_contract_delta_pair: worst,
  };
}

function pairLedgerBytes(contract) {
  const bytes = new Uint8Array(65_536 * 15 * 8);
  const view = new DataView(bytes.buffer);
  let offset = 0;
  const tolerance = Math.max(contract.current.scale, contract.candidate.scale) * 2 ** -40;
  for (let q0 = contract.input0.qmin; q0 <= contract.input0.qmax; q0 += 1) {
    for (let q1 = contract.input1.qmin; q1 <= contract.input1.qmax; q1 += 1) {
      const pair = projectPair(q0, q1, contract.input0, contract.input1, contract.current, contract.candidate, tolerance);
      const a = pair.currentProjection; const b = pair.candidateProjection;
      for (const value of [q0, q1, a.raw, a.code, b.raw, b.code, Number(a.clipped), Number(b.clipped), pair.relation]) {
        view.setBigInt64(offset, BigInt(value), true); offset += 8;
      }
      for (const value of [pair.ideal, a.real, b.real, pair.deltaSteps, a.error, b.error]) {
        view.setFloat64(offset, value, true); offset += 8;
      }
    }
  }
  if (offset !== bytes.byteLength) throw new Error("Distortion pair-ledger byte count is invalid.");
  return bytes;
}

function projectPair(q0, q1, input0, input1, current, candidate, tolerance) {
  const ideal = (q0 - input0.zeroPoint) * input0.scale + (q1 - input1.zeroPoint) * input1.scale;
  const currentProjection = project(ideal, current);
  const candidateProjection = project(ideal, candidate);
  const deltaReal = candidateProjection.real - currentProjection.real;
  const deltaSteps = deltaReal / current.scale;
  return { ideal, currentProjection, candidateProjection, deltaReal, deltaSteps, absoluteDeltaSteps: Math.abs(deltaSteps), relation: errorRelation(currentProjection.error, candidateProjection.error, tolerance) };
}

function project(ideal, output) {
  const raw = roundTiesAway(ideal / output.scale) + output.zeroPoint;
  const code = Math.max(output.qmin, Math.min(output.qmax, raw));
  const real = (code - output.zeroPoint) * output.scale;
  return { raw, code, real, error: Math.abs(real - ideal), clipped: raw < output.qmin || raw > output.qmax };
}

function witness(q0, q1, pair) {
  const a = pair.currentProjection; const b = pair.candidateProjection;
  return { input_0_code: q0, input_1_code: q1, ideal_real_sum: pair.ideal, current_raw_code: a.raw, current_projected_code: a.code, current_represented_real: a.real, current_absolute_ideal_error: a.error, current_clipped: a.clipped, candidate_raw_code: b.raw, candidate_projected_code: b.code, candidate_represented_real: b.real, candidate_absolute_ideal_error: b.error, candidate_clipped: b.clipped, signed_contract_delta_real: pair.deltaReal, signed_contract_delta_current_steps: pair.deltaSteps, absolute_contract_delta_current_steps: pair.absoluteDeltaSteps };
}

function compareRow(actual, expected, opIndex) {
  if (actual.assessment_status !== "assessed" || actual.scenarios?.length !== 2) throw new Error(`Distortion scenario coverage is invalid at #${opIndex}.`);
  actual.scenarios.forEach((scenario, index) => compareScenario(scenario, expected.scenarios[index], opIndex));
  for (const key of ["maximum_rms_contract_delta_current_steps", "maximum_p99_contract_delta_current_steps"]) assertNear(actual[key], expected[key], `#${opIndex} ${key}`);
  assertEqual(actual.maximum_rescued_current_clamp_pair_count, expected.maximum_rescued_current_clamp_pair_count, `#${opIndex} rescued clamp maximum`);
}

function compareScenario(actual, expected, opIndex) {
  const integerKeys = ["design", "candidate_output_zero_point", "candidate_signed_zero_point_delta", "enumerated_pair_count", "current_clamped_pair_count", "candidate_clamped_pair_count", "rescued_current_clamp_pair_count", "persistent_clamp_pair_count", "introduced_clamp_pair_count", "same_represented_value_pair_count", "changed_represented_value_pair_count", "sign_class_changed_pair_count", "ideal_error_improved_pair_count", "ideal_error_worsened_pair_count", "ideal_error_equal_within_tolerance_pair_count", "within_half_current_step_pair_count", "within_one_current_step_pair_count", "within_two_current_steps_pair_count", "tile_size_codes", "tile_grid_dimension"];
  integerKeys.forEach((key) => assertEqual(actual[key], expected[key], `#${opIndex} ${actual.design} ${key}`));
  const floatKeys = ["candidate_output_scale", "candidate_scale_ratio_to_current", "error_comparison_tolerance_real", "mean_signed_contract_delta_real", "mean_absolute_contract_delta_real", "root_mean_square_contract_delta_real", "mean_signed_contract_delta_current_steps", "mean_absolute_contract_delta_current_steps", "root_mean_square_contract_delta_current_steps", "maximum_absolute_contract_delta_current_steps", "p50_absolute_contract_delta_current_steps", "p90_absolute_contract_delta_current_steps", "p99_absolute_contract_delta_current_steps", "mean_absolute_ideal_error_current", "mean_absolute_ideal_error_candidate", "signed_mean_absolute_ideal_error_delta", "absolute_delta_histogram_bin_width_current_steps"];
  floatKeys.forEach((key) => assertNear(actual[key], expected[key], `#${opIndex} ${actual.design} ${key}`));
  const integerArrays = ["absolute_delta_histogram_counts", "tile_pair_counts", "tile_rescued_current_clamp_pair_counts", "tile_ideal_error_improved_pair_counts", "tile_ideal_error_worsened_pair_counts", "tile_ideal_error_equal_pair_counts", "tile_sign_class_changed_pair_counts"];
  integerArrays.forEach((key) => { if (!sameArray(actual[key], expected[key])) throw new Error(`Distortion ${key} is invalid at #${opIndex} ${actual.design}.`); });
  const floatArrays = ["tile_mean_signed_delta_current_steps", "tile_mean_absolute_delta_current_steps", "tile_maximum_absolute_delta_current_steps"];
  floatArrays.forEach((key) => compareFloatArray(actual[key], expected[key], `#${opIndex} ${actual.design} ${key}`));
  compareObjectNumbers(actual.worst_absolute_contract_delta_pair, expected.worst_absolute_contract_delta_pair, `#${opIndex} worst witness`);
  if (!/^[a-f0-9]{64}$/.test(actual.pair_ledger_sha256 || "")) throw new Error(`Distortion ledger is invalid at #${opIndex}.`);
}

function renderSummary(container, result) {
  if (!container) return;
  container.replaceChildren(
    metric("Exact pair scenarios", formatNumber(result.total_enumerated_pair_count), `${result.scenario_count} candidate contracts`),
    metric("Clamp pairs rescued", formatNumber(result.rescued_current_clamp_pair_instance_count), `${formatNumber(result.candidate_clamped_pair_count)} candidate clamps`),
    metric("Ideal-error direction", `${formatNumber(result.ideal_error_improved_pair_count)} / ${formatNumber(result.ideal_error_worsened_pair_count)}`, "improved / worsened code pairs"),
    metric("Worst p99 displacement", `${number(result.maximum_p99_contract_delta_current_steps)} steps`, `#${padOp(result.distortion_ranking_op_indices[0] ?? 0)} ADD`),
  );
}

function opSelector(result, selected) {
  return element("div", "step-response-op-selector", result.distortion_ranking_op_indices.map((opIndex) => {
    const button = element("button", opIndex === selected ? "active" : "", `#${padOp(opIndex)}`);
    button.type = "button"; button.dataset.distortionOp = opIndex; return button;
  }));
}

function candidateTabs(scenarios, selected) {
  return element("div", "step-response-design-tabs", scenarioGroups(scenarios).map((group) => {
    const selectedScenario = group.find((scenario) => scenario.design === selected) || group[0];
    const label = group.length > 1
      ? `${group.map((scenario) => designLabel(scenario.design)).join(" = ")}`
      : designLabel(selectedScenario.design);
    const button = element("button", group.some((scenario) => scenario.design === selected) ? "active" : "", label);
    button.type = "button"; button.dataset.distortionDesign = selectedScenario.design; return button;
  }));
}

function equivalentScenarioNote(scenarios) {
  const duplicateGroups = scenarioGroups(scenarios).filter((group) => group.length > 1);
  return duplicateGroups.length
    ? element("p", "step-response-equivalence-note", duplicateGroups.map((group) =>
        `${group.map((scenario) => designLabel(scenario.design)).join(" and ")} produce one identical output contract and exact pair ledger; one row is shown.`).join(" "))
    : document.createTextNode("");
}

function viewTabs(selected) {
  return element("div", "step-response-design-tabs distortion-view-tabs", VIEWS.map((view) => {
    const button = element("button", view === selected ? "active" : "", view === VIEWS[0] ? "Signed displacement" : view === VIEWS[1] ? "Ideal-error direction" : "Clamp state");
    button.type = "button"; button.dataset.distortionView = view; return button;
  }));
}

function headline(row, scenario) {
  const graph = element("button", "secondary-action", "Graph"); graph.type = "button"; graph.dataset.distortionGraph = row.op_index;
  return element("section", "step-response-headline", [
    element("div", "step-response-headline-title", [element("strong", "", `#${padOp(row.op_index)} ADD`), graph]),
    element("div", "step-response-headline-metrics", [
      metric("Candidate contract", `${number(scenario.candidate_output_scale)} / zp ${scenario.candidate_output_zero_point}`, `${number(scenario.candidate_scale_ratio_to_current)}x current scale`),
      metric("RMS displacement", `${number(scenario.root_mean_square_contract_delta_current_steps)} steps`, `p99 ${number(scenario.p99_absolute_contract_delta_current_steps)}`),
      metric("Clamp rescue", formatNumber(scenario.rescued_current_clamp_pair_count), `${formatNumber(scenario.candidate_clamped_pair_count)} candidate clamps`),
      metric("Ideal error", signedNumber(scenario.signed_mean_absolute_ideal_error_delta), `${formatNumber(scenario.ideal_error_improved_pair_count)} improve / ${formatNumber(scenario.ideal_error_worsened_pair_count)} worsen`),
    ]),
  ]);
}

function canvasPanel(title, description, canvas, legend = null) {
  return element("section", "step-response-canvas-panel", [element("strong", "", title), element("p", "", description), canvas, legend]);
}

function fieldLegend(view) {
  const rows = view === VIEWS[0] ? [["negative", "#5da9ff"], ["near zero", "#243343"], ["positive", "#f6b85f"]]
    : view === VIEWS[1] ? [["improved", "#5ee0b7"], ["equal", "#647687"], ["worsened", "#f06d76"]]
      : [["rescued", "#5ee0b7"], ["unchanged", "#243343"], ["persistent/new", "#f6b85f"]];
  return element("div", "step-response-legend", rows.map(([label, color]) => legend(label, color)));
}

function scenarioTable(row) {
  return tableSection("Candidate Comparison", ["Candidate", "Scale / zp", "RMS / p99 steps", "Clamp rescue", "Error improve / worsen", "Sign changes", "Ledger SHA-256"], scenarioGroups(row.scenarios).map((group) => {
    const scenario = group[0];
    return [
    group.map((item) => designLabel(item.design)).join(" = "), `${number(scenario.candidate_output_scale)} / ${scenario.candidate_output_zero_point}`,
    `${number(scenario.root_mean_square_contract_delta_current_steps)} / ${number(scenario.p99_absolute_contract_delta_current_steps)}`,
    formatNumber(scenario.rescued_current_clamp_pair_count), `${formatNumber(scenario.ideal_error_improved_pair_count)} / ${formatNumber(scenario.ideal_error_worsened_pair_count)}`,
    formatNumber(scenario.sign_class_changed_pair_count), scenario.pair_ledger_sha256,
  ]; }));
}

function scenarioGroups(scenarios) {
  const groups = new Map();
  for (const scenario of scenarios || []) {
    const key = [
      Number(scenario.candidate_output_scale).toPrecision(17),
      scenario.candidate_output_zero_point,
      Number(scenario.root_mean_square_contract_delta_current_steps).toPrecision(17),
      Number(scenario.p99_absolute_contract_delta_current_steps).toPrecision(17),
      scenario.pair_ledger_sha256,
    ].join("|");
    const group = groups.get(key) || [];
    group.push(scenario);
    groups.set(key, group);
  }
  return [...groups.values()];
}

function witnessTable(scenario) {
  const witness = scenario.worst_absolute_contract_delta_pair;
  return tableSection("Worst Exact Pair Witness", ["Input pair", "Ideal sum", "Current raw -> code / real", "Candidate raw -> code / real", "Displacement", "Ideal error current -> candidate"], [[
    `${witness.input_0_code}, ${witness.input_1_code}`, number(witness.ideal_real_sum),
    `${witness.current_raw_code} -> ${witness.current_projected_code} / ${number(witness.current_represented_real)}`,
    `${witness.candidate_raw_code} -> ${witness.candidate_projected_code} / ${number(witness.candidate_represented_real)}`,
    `${signedNumber(witness.signed_contract_delta_current_steps)} current steps`,
    `${number(witness.current_absolute_ideal_error)} -> ${number(witness.candidate_absolute_ideal_error)}`,
  ]]);
}

function portfolioTable(result, selected) {
  const rows = result.residual_adds.filter((row) => row.assessment_status === "assessed").sort((a, b) => a.distortion_rank - b.distortion_rank);
  return tableSection("Residual Distortion Portfolio", ["Rank", "ADD", "Maximum RMS", "Maximum p99", "Clamp pairs rescued"], rows.map((row) => [
    row.distortion_rank, row.op_index === selected ? `#${padOp(row.op_index)} selected` : `#${padOp(row.op_index)}`,
    `${number(row.maximum_rms_contract_delta_current_steps)} steps`, `${number(row.maximum_p99_contract_delta_current_steps)} steps`, formatNumber(row.maximum_rescued_current_clamp_pair_count),
  ]));
}

function methodBoundary(result) {
  return element("section", "step-response-method", [element("strong", "", "Method and evidence boundary"), element("p", "", result.projection_definition), element("p", "", result.error_comparison_definition), element("p", "", result.pair_ledger_hash_method), element("p", "", result.interpretation_boundary)]);
}

function drawDistortionField(canvas, row, scenario, analysis, viewMode) {
  const ctx = setupCanvas(canvas, 620, 470); if (!ctx) return;
  const rect = canvas.getBoundingClientRect(); const width = rect.width; const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  const op = (analysis.ops || []).find((item) => item.index === row.op_index && item.name === "ADD");
  const lattice = (analysis.quantization_lattice?.residual_adds || []).find((item) => item.op_index === row.op_index);
  const contract = contractsFor(op, analysis.tensors || [], lattice).find((item) => item.design === scenario.design);
  const field = document.createElement("canvas"); field.width = 256; field.height = 256;
  const fieldContext = field.getContext("2d"); const image = fieldContext.createImageData(256, 256);
  const tolerance = Math.max(contract.current.scale, contract.candidate.scale) * 2 ** -40;
  const clip = Math.max(1e-12, scenario.p99_absolute_contract_delta_current_steps);
  let pixel = 0;
  for (let q0 = contract.input0.qmin; q0 <= contract.input0.qmax; q0 += 1) {
    for (let q1 = contract.input1.qmin; q1 <= contract.input1.qmax; q1 += 1) {
      const pair = projectPair(q0, q1, contract.input0, contract.input1, contract.current, contract.candidate, tolerance);
      const color = fieldColor(pair, viewMode, clip); const offset = pixel * 4;
      image.data[offset] = color[0]; image.data[offset + 1] = color[1]; image.data[offset + 2] = color[2]; image.data[offset + 3] = 255; pixel += 1;
    }
  }
  fieldContext.putImageData(image, 0, 0);
  const left = 56; const top = 28; const size = Math.min(width - 88, height - 86);
  ctx.imageSmoothingEnabled = false; ctx.drawImage(field, left, top, size, size);
  ctx.strokeStyle = "rgba(219,231,238,.35)"; ctx.strokeRect(left, top, size, size);
  ctx.fillStyle = "#aebdc7"; ctx.font = "12px system-ui"; ctx.fillText("input 1 code ->", left, top + size + 24);
  ctx.save(); ctx.translate(18, top + size); ctx.rotate(-Math.PI / 2); ctx.fillText("input 0 code ->", 0, 0); ctx.restore();
}

function drawHistogram(canvas, scenario) {
  const ctx = setupCanvas(canvas, 620, 470); if (!ctx) return;
  const { width, height } = canvas.getBoundingClientRect(); ctx.clearRect(0, 0, width, height);
  const margin = { left: 58, right: 24, top: 32, bottom: 58 }; const w = width - margin.left - margin.right; const h = height - margin.top - margin.bottom;
  const counts = scenario.absolute_delta_histogram_counts; const max = Math.max(...counts, 1); const gap = 2; const barWidth = w / counts.length;
  counts.forEach((count, index) => {
    const normalized = Math.log1p(count) / Math.log1p(max); const barHeight = normalized * h;
    ctx.fillStyle = index < counts.length * 0.25 ? "#5da9ff" : index < counts.length * 0.7 ? "#5ee0b7" : "#f6b85f";
    ctx.fillRect(margin.left + index * barWidth + gap / 2, margin.top + h - barHeight, Math.max(1, barWidth - gap), barHeight);
  });
  ctx.strokeStyle = "rgba(180,199,210,.35)"; ctx.strokeRect(margin.left, margin.top, w, h);
  const maxSteps = scenario.absolute_delta_histogram_bin_width_current_steps * counts.length;
  [["p50", scenario.p50_absolute_contract_delta_current_steps, "#dbe7ee"], ["p90", scenario.p90_absolute_contract_delta_current_steps, "#f6b85f"], ["p99", scenario.p99_absolute_contract_delta_current_steps, "#f06d76"]].forEach(([label, value, color], index) => {
    const x = margin.left + Math.min(1, value / maxSteps) * w; ctx.strokeStyle = color; ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + h); ctx.stroke();
    ctx.fillStyle = color; ctx.font = "11px system-ui"; ctx.fillText(`${label} ${number(value)}`, Math.min(x + 4, width - 92), margin.top + 14 + index * 15);
  });
  ctx.fillStyle = "#aebdc7"; ctx.font = "12px system-ui"; ctx.fillText("absolute displacement in current output steps ->", margin.left, height - 18);
  ctx.save(); ctx.translate(18, margin.top + h); ctx.rotate(-Math.PI / 2); ctx.fillText("log pair count ->", 0, 0); ctx.restore();
}

function fieldColor(pair, mode, clip) {
  if (mode === VIEWS[1]) return pair.relation < 0 ? [94, 224, 183] : pair.relation > 0 ? [240, 109, 118] : [100, 118, 135];
  if (mode === VIEWS[2]) {
    if (pair.currentProjection.clipped && !pair.candidateProjection.clipped) return [94, 224, 183];
    if (!pair.currentProjection.clipped && pair.candidateProjection.clipped) return [240, 109, 118];
    if (pair.currentProjection.clipped && pair.candidateProjection.clipped) return [246, 184, 95];
    return [36, 51, 67];
  }
  const magnitude = Math.min(1, Math.abs(pair.deltaSteps) / clip) ** 0.55;
  return pair.deltaSteps < 0 ? mix([36, 51, 67], [93, 169, 255], magnitude) : mix([36, 51, 67], [246, 184, 95], magnitude);
}

function setupCanvas(canvas, logicalWidth, logicalHeight) {
  const rect = canvas.getBoundingClientRect(); const width = Math.max(280, rect.width || logicalWidth); const height = Math.max(300, rect.height || logicalHeight); const ratio = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); return ctx;
}

function quantContract(tensors, index) {
  const tensor = tensors.find((item) => item.index === index);
  if (!tensor || !["INT8", "UINT8"].includes(tensor.dtype) || tensor.scale_sample?.length !== 1 || tensor.zero_point_sample?.length !== 1) throw new Error(`Tensor T${index} lacks a per-tensor 8-bit contract.`);
  const qmin = tensor.dtype === "INT8" ? -128 : 0; const qmax = tensor.dtype === "INT8" ? 127 : 255; const scale = Number(tensor.scale_sample[0]); const zeroPoint = Number(tensor.zero_point_sample[0]);
  if (!(scale > 0) || !Number.isFinite(scale) || zeroPoint < qmin || zeroPoint > qmax) throw new Error(`Tensor T${index} quantization metadata is invalid.`);
  return { index, qmin, qmax, scale, zeroPoint };
}

function errorRelation(current, candidate, tolerance) { return candidate + tolerance < current ? -1 : current + tolerance < candidate ? 1 : 0; }
function signClass(value) { return value > 0 ? 1 : value < 0 ? -1 : 0; }
function quantile(sorted, probability) { return sorted[Math.min(sorted.length, Math.max(1, Math.ceil(probability * sorted.length))) - 1] || 0; }
function viewDescription(view) { return view === VIEWS[0] ? "Candidate minus current represented value, normalized to the current output step and clipped at scenario p99 for color only." : view === VIEWS[1] ? "Candidate ideal-projection error compared with current using the declared binary64 equality tolerance." : "Exact current/candidate rounded-clamp state for every legal input-code pair."; }
function designLabel(design) { return design === DESIGNS[0] ? "Fixed zero-point" : "Globally finest"; }
function number(value) { return Number(value).toPrecision(7); }
function signedNumber(value) { const n = Number(value); return `${n >= 0 ? "+" : ""}${number(n)}`; }
function sum(rows, key) { return rows.reduce((total, row) => total + Number(row[key] || 0), 0); }
function sameArray(left, right) { return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index]); }
function compareFloatArray(left, right, label) { if (!Array.isArray(left) || left.length !== right.length) throw new Error(`${label} length mismatch.`); left.forEach((value, index) => assertNear(value, right[index], `${label}[${index}]`)); }
function compareObjectNumbers(left, right, label) { for (const key of Object.keys(right || {})) { if (typeof right[key] === "number") assertNear(left?.[key], right[key], `${label}.${key}`); else assertEqual(left?.[key], right[key], `${label}.${key}`); } }
function assertEqual(actual, expected, label) { if (actual !== expected) throw new Error(`${label} mismatch (${actual} != ${expected}).`); }
function assertNear(actual, expected, label) { const tolerance = Math.max(1e-12, Math.abs(Number(expected)) * 1e-11); if (!Number.isFinite(Number(actual)) || Math.abs(Number(actual) - Number(expected)) > tolerance) throw new Error(`${label} mismatch (${actual} != ${expected}).`); }
function mix(a, b, t) { return a.map((value, index) => Math.round(value + (b[index] - value) * t)); }

function tableSection(title, headers, rows) {
  const table = element("table", "step-response-table");
  table.append(element("thead", "", element("tr", "", headers.map((header) => element("th", "", header)))), element("tbody", "", rows.map((row) => element("tr", "", row.map((cell) => element("td", "", String(cell)))))));
  return element("section", "step-response-table-section", [element("strong", "", title), element("div", "step-response-table-scroll", table)]);
}
function metric(label, value, detail) { return element("div", "step-response-metric", [element("span", "", label), element("strong", "", String(value)), element("small", "", detail)]); }
function legend(label, color) { const swatch = element("i", ""); swatch.style.background = color; return element("span", "", [swatch, document.createTextNode(label)]); }
function messageNode(text, tone = "") { return element("p", `step-response-message ${tone}`, text); }
function element(tag, className = "", children = null) {
  const node = document.createElement(tag); if (className) node.className = className;
  if (children != null) { if (Array.isArray(children)) node.append(...children.filter(Boolean).map((child) => child instanceof Node ? child : document.createTextNode(String(child)))); else if (children instanceof Node) node.append(children); else node.textContent = String(children); }
  return node;
}
