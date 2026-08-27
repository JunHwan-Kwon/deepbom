import { formatNumber, padOp } from "./format.js";
import { sha256Hex } from "./hash.js";
import { quantizeMultiplier, roundTiesAway } from "./quantization-math.js";

export const CONTRACT_MIGRATION_SCHEMA = "deepbom.contract_migration.v1";

const METHOD_VERSION = "2026-07-17.1";
const SOURCE_COMMIT = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const DESIGNS = new Set(["fixed_zero_point_minimum_containment", "globally_finest_minimum_containment"]);
const KERNELS = new Set(["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"]);
const SOURCE_SHA256 = Object.freeze({
  "tensorflow/lite/kernels/add.cc": "436dbd27aba268d8828b07ce1447e6c8a979324925667a0a3c8987d9185b6947",
  "tensorflow/lite/kernels/internal/quantization_util.cc": "22e46f15663437c407298f5230545600faa2f6b2f1b46488e20c97ff3a5c96f9",
  "tensorflow/lite/kernels/kernel_util.cc": "fb03b532b1f510ccf5d7d169eeebcc408791677c97cbce235893560b4379da49",
});

export function createContractMigrationController({
  root,
  status,
  summary,
  body,
  downloadButton,
  getContext,
  jumpToGraphOp,
  onDownload,
  residualSelection,
}) {
  let analysis = null;
  let migration = null;
  let reconstructed = null;
  let selectedAdd = null;
  let selectedDesign = "globally_finest_minimum_containment";
  let renderToken = 0;
  let resizeObserver = null;
  const selectionSource = "contract-migration";

  root?.addEventListener("click", (event) => {
    const addButton = event.target.closest("[data-migration-add]");
    if (addButton) {
      selectedAdd = Number(addButton.dataset.migrationAdd);
      residualSelection?.set({ opIndex: selectedAdd }, selectionSource);
      renderBody();
      return;
    }
    const designButton = event.target.closest("[data-migration-design]");
    if (designButton && DESIGNS.has(designButton.dataset.migrationDesign)) {
      selectedDesign = designButton.dataset.migrationDesign;
      residualSelection?.set({ design: selectedDesign }, selectionSource);
      renderBody();
      return;
    }
    const graphButton = event.target.closest("[data-migration-open-graph]");
    if (graphButton) jumpToGraphOp?.(Number(graphButton.dataset.migrationOpenGraph));
  });
  residualSelection?.subscribe((selection, source) => {
    if (source === selectionSource || !migration) return;
    if (migration.migrations.some((row) => row.source_add_op_index === selection.opIndex)) selectedAdd = selection.opIndex;
    if (DESIGNS.has(selection.design)) selectedDesign = selection.design;
    renderBody();
  });
  downloadButton?.addEventListener("click", () => {
    if (migration) onDownload?.(migration, "contract_migration_impact.json");
  });

  function render(explicitAnalysis = null) {
    renderToken += 1;
    const token = renderToken;
    const context = getContext?.() || {};
    analysis = explicitAnalysis || context.analysis || null;
    migration = analysis?.contract_migration || null;
    const bytes = context.modelBytes || null;
    resizeObserver?.disconnect();
    resizeObserver = null;
    if (!migration || !bytes || String(analysis?.format || "").toLowerCase() !== "tflite") {
      selectedAdd = null;
      reconstructed = null;
      if (root) root.hidden = true;
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    if (root) root.hidden = false;
    if (downloadButton) downloadButton.disabled = false;
    try {
      reconstructed = validateContractMigration(analysis, bytes);
      const preferred = analysis.quantization_lattice?.domain_escape_ranking_op_indices?.[0];
      const shared = residualSelection?.get();
      if (migration.migrations.some((row) => row.source_add_op_index === shared?.opIndex)) selectedAdd = shared.opIndex;
      if (DESIGNS.has(shared?.design)) selectedDesign = shared.design;
      if (!migration.migrations.some((row) => row.source_add_op_index === selectedAdd)) {
        selectedAdd = preferred ?? migration.migrations[0]?.source_add_op_index ?? null;
      }
      residualSelection?.set({ opIndex: selectedAdd, design: selectedDesign }, selectionSource);
      renderMigrationSummary(summary, migration);
      renderBody();
      if (status) {
        status.textContent = "arithmetic verified / digest pending";
        status.dataset.tone = migration.bias_int32_overflow_channel_scenario_count ? "risk" : "ok";
      }
      void validateContractMigrationDigests(analysis, bytes).then(() => {
        if (token !== renderToken || !status) return;
        status.textContent = "independently verified";
        status.dataset.tone = migration.bias_int32_overflow_channel_scenario_count ? "risk" : "ok";
      }).catch((error) => {
        if (token !== renderToken || !status) return;
        status.textContent = `integrity error: ${error.message}`;
        status.dataset.tone = "risk";
      });
    } catch (error) {
      reconstructed = null;
      summary?.replaceChildren();
      body?.replaceChildren(messageNode(`Contract-migration evidence rejected: ${error.message}`, "risk"));
      if (status) {
        status.textContent = "evidence rejected";
        status.dataset.tone = "risk";
      }
    }
  }

  function renderBody() {
    if (!body || !migration) return;
    resizeObserver?.disconnect();
    const row = migration.migrations.find((item) => item.source_add_op_index === selectedAdd)
      || migration.migrations[0];
    if (!row) {
      body.replaceChildren(messageNode("No residual containment migration is assessable."));
      return;
    }
    const scenario = row.scenarios.find((item) => item.design === selectedDesign)
      || row.scenarios[0];
    selectedDesign = scenario.design;
    const channelCanvas = element("canvas", "migration-channel-canvas");
    channelCanvas.setAttribute("aria-label", "Channel multiplier shift and bias rebasing error plot");
    const radiusCanvas = element("canvas", "migration-radius-canvas");
    radiusCanvas.setAttribute("aria-label", "Reachable downstream operation depth histogram");
    body.replaceChildren(
      migrationSelector(migration.migrations, row.source_add_op_index),
      designTabs(row.scenarios, scenario.design),
      scenarioHeadline(row, scenario, jumpToGraphOp),
      plotBand(channelCanvas, "Channel parameter migration", "Q0.31 shift delta and bias rebasing error in current bias steps"),
      consumerLedger(scenario, jumpToGraphOp),
      plotBand(radiusCanvas, "Structural behavior radius", "Reachable ops by minimum graph-edge depth; only depth 1 is a parameter-regeneration boundary"),
      topBiasTable(scenario),
      addParameterTable(scenario),
      methodBoundary(migration),
    );
    const redraw = () => {
      drawChannelMigration(channelCanvas, scenario);
      drawImpactRadius(radiusCanvas, row);
    };
    redraw();
    resizeObserver = new ResizeObserver(redraw);
    resizeObserver.observe(channelCanvas.parentElement);
    resizeObserver.observe(radiusCanvas.parentElement);
  }

  return { render, getReconstructed: () => reconstructed };
}

export function validateContractMigration(analysis, modelBytes) {
  const actual = analysis?.contract_migration;
  assert(actual?.schema === CONTRACT_MIGRATION_SCHEMA, "Contract-migration schema mismatch.");
  assert(actual.method_version === METHOD_VERSION, "Contract-migration method version mismatch.");
  assert(actual.evidence_class === "DERIVED", "Contract-migration evidence class must be DERIVED.");
  assert(actual.source_commit === SOURCE_COMMIT, "Contract-migration source commit mismatch.");
  validateSources(actual.source_references);
  assert(String(actual.interpretation_boundary || "").includes("counterfactual re-export impact analysis"), "Migration re-export boundary is incomplete.");
  assert(String(actual.interpretation_boundary || "").includes("structural behavior-impact radius only"), "Migration reachability boundary is incomplete.");
  assert(String(actual.method || "").includes("every stored INT32 bias"), "Migration bias method is incomplete.");
  const expected = reconstructContractMigration(analysis, modelBytes);
  compareComputedAnalysis(actual, expected);
  return expected;
}

export async function validateContractMigrationDigests(analysis, modelBytes) {
  const expected = validateContractMigration(analysis, modelBytes);
  for (const [key, ledger] of expected.ledgerTexts) {
    const digest = await sha256Hex(new TextEncoder().encode(ledger));
    assert(digest === expected.ledgerDigests.get(key), `Contract-migration ledger SHA-256 mismatch at ${key}.`);
  }
  return expected;
}

export function reconstructContractMigration(analysis, modelBytes) {
  assert(modelBytes instanceof Uint8Array, "Contract migration requires original model bytes.");
  const ops = new Map((analysis?.ops || []).map((op) => [Number(op.index), op]));
  const tensors = new Map((analysis?.tensors || []).map((tensor) => [Number(tensor.index), tensor]));
  const accumulator = new Map((analysis?.accumulator_atlas?.ops || []).map((row) => [Number(row.op_index), row]));
  const consumers = consumerMap(analysis?.ops || []);
  const ledgerTexts = new Map();
  const ledgerDigests = new Map();
  const migrations = [];
  const downstreamUnion = new Set();
  for (const lattice of analysis?.quantization_lattice?.residual_adds || []) {
    if (lattice.output_tensor_index == null || lattice.output_scale == null || lattice.output_zero_point == null) continue;
    const source = tensors.get(Number(lattice.output_tensor_index));
    if (!source) continue;
    const consumerIndices = consumers.get(Number(lattice.output_tensor_index)) || [];
    const directConsumers = consumerIndices.map((opIndex) => {
      const op = ops.get(opIndex);
      return {
        op_index: op.index,
        op_name: op.name,
        input_slots: matchingInputSlots(op, Number(lattice.output_tensor_index)),
        migration_class: migrationClass(op.name),
      };
    });
    const affectedOps = downstreamOps(Number(lattice.output_tensor_index), ops, consumers);
    affectedOps.forEach((item) => downstreamUnion.add(item.op_index));
    const candidates = [lattice.fixed_zero_point_containment, lattice.globally_finest_containment].filter(Boolean);
    const actualMigration = analysis?.contract_migration?.migrations?.find((row) => row.source_add_op_index === lattice.op_index);
    const scenarios = candidates.map((candidate) => reconstructScenario({
      candidate,
      source,
      currentScale: Number(lattice.output_scale),
      currentZeroPoint: Number(lattice.output_zero_point),
      consumerIndices,
      ops,
      tensors,
      accumulator,
      modelBytes,
      ledgerTexts,
      ledgerDigests,
      actualScenario: actualMigration?.scenarios?.find((row) => row.design === candidate.design),
    }));
    migrations.push({
      source_add_op_index: lattice.op_index,
      output_tensor_index: lattice.output_tensor_index,
      output_tensor_name: lattice.output_tensor_name,
      current_output_scale: lattice.output_scale,
      current_output_zero_point: lattice.output_zero_point,
      direct_consumer_count: directConsumers.length,
      direct_consumer_edge_count: sum(directConsumers.map((consumer) => consumer.input_slots.length)),
      direct_consumers: directConsumers,
      reachable_downstream_op_count: affectedOps.length,
      maximum_downstream_edge_depth: maxOrZero(affectedOps.map((item) => item.minimum_edge_depth)),
      affected_ops: affectedOps,
      scenarios,
    });
  }
  const scenarios = migrations.flatMap((row) => row.scenarios);
  const identities = migrations.flatMap((row) => row.direct_consumers);
  const summary = {
    status: migrations.length === 0 ? "not_applicable" : scenarios.some((item) => item.unassessed_consumer_count) ? "partial" : "assessed",
    residual_contract_count: migrations.length,
    candidate_scenario_count: scenarios.length,
    direct_consumer_count: identities.length,
    direct_consumer_edge_count: sum(identities.map((item) => item.input_slots.length)),
    kernel_consumer_count: identities.filter((item) => item.migration_class === "integer_kernel").length,
    add_consumer_count: identities.filter((item) => item.migration_class === "quantized_add").length,
    other_consumer_count: identities.filter((item) => item.migration_class === "not_modeled").length,
    assessed_consumer_scenario_count: sum(scenarios.map((item) => item.assessed_consumer_count)),
    unassessed_consumer_scenario_count: sum(scenarios.map((item) => item.unassessed_consumer_count)),
    assessed_kernel_channel_scenario_count: sum(scenarios.map((item) => item.assessed_kernel_channel_count)),
    multiplier_encoding_changed_channel_scenario_count: sum(scenarios.map((item) => item.multiplier_encoding_changed_channel_count)),
    multiplier_shift_changed_channel_scenario_count: sum(scenarios.map((item) => item.multiplier_shift_changed_channel_count)),
    bias_code_changed_channel_scenario_count: sum(scenarios.map((item) => item.bias_code_changed_channel_count)),
    bias_int32_overflow_channel_scenario_count: sum(scenarios.map((item) => item.bias_int32_overflow_channel_count)),
    add_parameter_encoding_changed_scenario_count: sum(scenarios.map((item) => item.add_parameter_encoding_changed_count)),
    reachable_downstream_op_union_count: downstreamUnion.size,
    maximum_downstream_edge_depth: maxOrZero(migrations.map((item) => item.maximum_downstream_edge_depth)),
  };
  return { summary, migrations, ledgerTexts, ledgerDigests };
}

function reconstructScenario(context) {
  const kernelConsumers = [];
  const addConsumers = [];
  const unassessedConsumers = [];
  for (const opIndex of context.consumerIndices) {
    const op = context.ops.get(opIndex);
    const slots = matchingInputSlots(op, Number(context.source.index));
    if (KERNELS.has(op.name)) {
      const row = reconstructKernelConsumer(op, slots, context);
      kernelConsumers.push(row.serialized);
      if (row.ledgerText != null) {
        const key = `ADD #${context.source.index}/${context.candidate.design}/op #${op.index}`;
        context.ledgerTexts.set(key, row.ledgerText);
        context.ledgerDigests.set(key, row.serialized.channel_ledger_sha256);
      }
    } else if (op.name === "ADD") {
      addConsumers.push(reconstructAddConsumer(op, slots, context));
    } else {
      unassessedConsumers.push({
        op_index: op.index,
        op_name: op.name,
        input_slots: slots,
        reason: "Direct consumer has no source-backed migration rule in contract_migration.v1.",
      });
    }
  }
  const assessedKernelChannelCount = sum(kernelConsumers.map((row) => row.assessed_channel_count));
  const unassessed = unassessedConsumers.length
    + kernelConsumers.filter((row) => row.assessment_status !== "assessed").length
    + addConsumers.filter((row) => row.assessment_status !== "assessed").length;
  return {
    design: context.candidate.design,
    candidate_output_scale: context.candidate.output_scale,
    candidate_output_zero_point: context.candidate.output_zero_point,
    scale_ratio_to_current: context.candidate.output_scale / context.currentScale,
    signed_zero_point_delta: context.candidate.output_zero_point - context.currentZeroPoint,
    assessed_consumer_count: kernelConsumers.filter((row) => row.assessment_status === "assessed").length
      + addConsumers.filter((row) => row.assessment_status === "assessed").length,
    unassessed_consumer_count: unassessed,
    assessed_kernel_channel_count: assessedKernelChannelCount,
    multiplier_encoding_changed_channel_count: sum(kernelConsumers.map((row) => row.multiplier_encoding_changed_channel_count)),
    multiplier_shift_changed_channel_count: sum(kernelConsumers.map((row) => row.multiplier_shift_changed_channel_count)),
    bias_code_changed_channel_count: sum(kernelConsumers.map((row) => row.bias_code_changed_channel_count)),
    bias_int32_overflow_channel_count: sum(kernelConsumers.map((row) => row.bias_int32_overflow_channel_count)),
    add_parameter_encoding_changed_count: sum(addConsumers.map((row) => row.changed_multiplier_encoding_count)),
    kernel_consumers: kernelConsumers,
    add_consumers: addConsumers,
    unassessed_consumers: unassessedConsumers,
  };
}

function reconstructKernelConsumer(op, slots, context) {
  const fail = (reason) => ({ serialized: notAssessedKernel(op, context.source, slots, context, reason), ledgerText: null });
  if (slots.length !== 1 || slots[0] !== 0) return fail("Kernel migration requires the changed activation at input slot 0.");
  const weight = context.tensors.get(Number(op.inputs?.[1]));
  const output = context.tensors.get(Number(op.outputs?.[0]));
  const accumulator = context.accumulator.get(Number(op.index));
  if (!weight) return fail("Weight tensor is unavailable.");
  if (!output) return fail("Output tensor is unavailable.");
  if (!accumulator || accumulator.assessment_status !== "assessed") return fail("A channel-complete accumulator row is unavailable.");
  const channels = Number(accumulator.output_channel_count || 0);
  if (!channels) return fail("Output-channel cardinality is unavailable.");
  if (output.scale_sample?.length !== 1 || !validScale(Number(output.scale_sample[0]))) return fail("Kernel output requires one finite positive scale.");
  if (![1, channels].includes(weight.scale_sample?.length)
    || weight.scale_sample.some((scale) => !validScale(Number(scale)))) return fail(`Weight scale cardinality must be 1 or ${channels}.`);
  let biasDecoded;
  try {
    biasDecoded = decodeBias(op, context.tensors, context.modelBytes, channels);
  } catch (error) {
    return fail(error.message);
  }
  const outputScale = Number(output.scale_sample[0]);
  const witnesses = [];
  let ledgerText = "";
  for (let channel = 0; channel < channels; channel += 1) {
    const weightScale = Number(weight.scale_sample[weight.scale_sample.length === 1 ? 0 : channel]);
    const currentReal = context.currentScale * weightScale / outputScale;
    const candidateReal = Number(context.candidate.output_scale) * weightScale / outputScale;
    const currentEncoding = quantizeMultiplier(currentReal, false);
    const candidateEncoding = quantizeMultiplier(candidateReal, false);
    const currentBias = biasDecoded.values[channel];
    const candidateBias = roundTiesAway(currentBias * context.currentScale / Number(context.candidate.output_scale));
    const overflow = candidateBias < -2147483648 || candidateBias > 2147483647;
    const currentBiasScale = context.currentScale * weightScale;
    const candidateBiasScale = Number(context.candidate.output_scale) * weightScale;
    const preservedBias = currentBias * currentBiasScale;
    const candidateBiasReal = overflow ? null : candidateBias * candidateBiasScale;
    const absoluteError = overflow ? null : Math.abs(candidateBiasReal - preservedBias);
    const currentSteps = absoluteError == null ? null : absoluteError / currentBiasScale;
    const candidateSteps = absoluteError == null ? null : absoluteError / candidateBiasScale;
    const witness = {
      channel_index: channel,
      weight_scale: weightScale,
      current_real_multiplier: currentReal,
      candidate_real_multiplier: candidateReal,
      current_quantized_multiplier: currentEncoding.multiplier,
      candidate_quantized_multiplier: candidateEncoding.multiplier,
      current_shift: currentEncoding.shift,
      candidate_shift: candidateEncoding.shift,
      current_bias_code: currentBias,
      candidate_bias_code_decimal: String(candidateBias),
      bias_code_changed: candidateBias !== currentBias,
      bias_int32_overflow: overflow,
      preserved_bias_real_value: preservedBias,
      candidate_bias_real_value: candidateBiasReal,
      absolute_bias_rebase_error: absoluteError,
      absolute_bias_rebase_error_current_steps: currentSteps,
      absolute_bias_rebase_error_candidate_steps: candidateSteps,
    };
    ledgerText += kernelLedgerRow(op.index, witness);
    witnesses.push(witness);
  }
  const encodingChanged = witnesses.filter((row) => row.current_quantized_multiplier !== row.candidate_quantized_multiplier
    || row.current_shift !== row.candidate_shift).length;
  const shiftChanged = witnesses.filter((row) => row.current_shift !== row.candidate_shift).length;
  const biasChanged = witnesses.filter((row) => row.bias_code_changed).length;
  const overflow = witnesses.filter((row) => row.bias_int32_overflow).length;
  const sorted = [...witnesses].sort((left, right) => nullableDesc(
    left.absolute_bias_rebase_error_current_steps,
    right.absolute_bias_rebase_error_current_steps,
  ) || left.channel_index - right.channel_index);
  return {
    ledgerText,
    serialized: {
      op_index: op.index,
      op_name: op.name,
      assessment_status: "assessed",
      not_assessed_reason: "",
      input_slots: slots,
      input_tensor_index: context.source.index,
      input_tensor_name: context.source.name,
      weight_tensor_index: weight.index,
      weight_tensor_name: weight.name,
      bias_tensor_index: biasDecoded.tensor?.index ?? null,
      bias_tensor_name: biasDecoded.tensor?.name || "",
      output_tensor_index: output.index,
      output_tensor_name: output.name,
      current_input_scale: context.currentScale,
      candidate_input_scale: context.candidate.output_scale,
      input_scale_ratio: context.candidate.output_scale / context.currentScale,
      current_input_zero_point: context.currentZeroPoint,
      candidate_input_zero_point: context.candidate.output_zero_point,
      input_zero_point_delta: context.candidate.output_zero_point - context.currentZeroPoint,
      output_scale: outputScale,
      weight_scale_mode: weight.scale_sample.length === 1 ? "per_tensor" : "per_output_channel",
      bias_status: biasDecoded.status,
      assessed_channel_count: channels,
      multiplier_encoding_changed_channel_count: encodingChanged,
      multiplier_shift_changed_channel_count: shiftChanged,
      bias_code_changed_channel_count: biasChanged,
      bias_int32_overflow_channel_count: overflow,
      maximum_absolute_bias_rebase_error: maxOptional(witnesses.map((row) => row.absolute_bias_rebase_error)),
      maximum_absolute_bias_rebase_error_current_steps: maxOptional(witnesses.map((row) => row.absolute_bias_rebase_error_current_steps)),
      maximum_absolute_bias_rebase_error_candidate_steps: maxOptional(witnesses.map((row) => row.absolute_bias_rebase_error_candidate_steps)),
      channel_current_quantized_multipliers: witnesses.map((row) => row.current_quantized_multiplier),
      channel_candidate_quantized_multipliers: witnesses.map((row) => row.candidate_quantized_multiplier),
      channel_current_shifts: witnesses.map((row) => row.current_shift),
      channel_candidate_shifts: witnesses.map((row) => row.candidate_shift),
      channel_current_bias_codes: witnesses.map((row) => row.current_bias_code),
      channel_candidate_bias_code_decimals: witnesses.map((row) => row.candidate_bias_code_decimal),
      channel_bias_rebase_error_current_steps: witnesses.map((row) => row.absolute_bias_rebase_error_current_steps),
      top_channels: sorted.slice(0, 8),
      channel_ledger_sha256: contextActualKernel(context, op.index)?.channel_ledger_sha256 || "",
      ledger_hash_method: contextActualKernel(context, op.index)?.ledger_hash_method || "",
    },
  };
}

function contextActualKernel(context, opIndex) {
  return context.actualScenario?.kernel_consumers?.find((row) => row.op_index === opIndex);
}

function notAssessedKernel(op, source, slots, context, reason) {
  return {
    op_index: op.index,
    op_name: op.name,
    assessment_status: "not_assessed",
    not_assessed_reason: reason,
    input_slots: slots,
    input_tensor_index: source.index,
    input_tensor_name: source.name,
    weight_tensor_index: Number(op.inputs?.[1]) >= 0 ? Number(op.inputs[1]) : null,
    weight_tensor_name: "",
    bias_tensor_index: Number(op.inputs?.[2]) >= 0 ? Number(op.inputs[2]) : null,
    bias_tensor_name: "",
    output_tensor_index: Number(op.outputs?.[0]) >= 0 ? Number(op.outputs[0]) : null,
    output_tensor_name: "",
    current_input_scale: context.currentScale,
    candidate_input_scale: context.candidate.output_scale,
    input_scale_ratio: context.candidate.output_scale / context.currentScale,
    current_input_zero_point: context.currentZeroPoint,
    candidate_input_zero_point: context.candidate.output_zero_point,
    input_zero_point_delta: context.candidate.output_zero_point - context.currentZeroPoint,
    output_scale: null,
    weight_scale_mode: "",
    bias_status: "",
    assessed_channel_count: 0,
    multiplier_encoding_changed_channel_count: 0,
    multiplier_shift_changed_channel_count: 0,
    bias_code_changed_channel_count: 0,
    bias_int32_overflow_channel_count: 0,
    maximum_absolute_bias_rebase_error: null,
    maximum_absolute_bias_rebase_error_current_steps: null,
    maximum_absolute_bias_rebase_error_candidate_steps: null,
    channel_current_quantized_multipliers: [],
    channel_candidate_quantized_multipliers: [],
    channel_current_shifts: [],
    channel_candidate_shifts: [],
    channel_current_bias_codes: [],
    channel_candidate_bias_code_decimals: [],
    channel_bias_rebase_error_current_steps: [],
    top_channels: [],
    channel_ledger_sha256: "",
    ledger_hash_method: "SHA-256 over UTF-8 rows op=<index>;channel=<index>;weight_scale=<f64hex>;current_real=<f64hex>;candidate_real=<f64hex>;current_q=<i32>;candidate_q=<i32>;current_shift=<i32>;candidate_shift=<i32>;current_bias=<i32>;candidate_bias=<decimal>;overflow=<0|1>;preserved_bias=<f64hex>;candidate_bias_real=<f64hex|na>;bias_error=<f64hex|na>;current_steps=<f64hex|na>;candidate_steps=<f64hex|na>\\n",
  };
}

function reconstructAddConsumer(op, slots, context) {
  const inputs = (op.inputs || []).slice(0, 2).map(Number);
  const outputIndex = Number(op.outputs?.[0]);
  const base = {
    op_index: op.index,
    op_name: op.name,
    assessment_status: "not_assessed",
    not_assessed_reason: "",
    changed_input_slots: slots,
    input_tensor_indices: inputs,
    input_tensor_names: inputs.map((index) => context.tensors.get(index)?.name).filter((name) => name != null),
    output_tensor_index: outputIndex >= 0 ? outputIndex : null,
    output_tensor_name: context.tensors.get(outputIndex)?.name || "",
    current_parameters: null,
    candidate_parameters: null,
    changed_offset_count: 0,
    changed_multiplier_encoding_count: 0,
    changed_shift_count: 0,
  };
  if (inputs.length < 2 || !(outputIndex >= 0)) return { ...base, not_assessed_reason: "ADD does not expose two inputs and one output." };
  const tensors = [context.tensors.get(inputs[0]), context.tensors.get(inputs[1]), context.tensors.get(outputIndex)];
  if (!tensors[0]) return { ...base, not_assessed_reason: "ADD input 0 is unavailable." };
  if (!tensors[1]) return { ...base, not_assessed_reason: "ADD input 1 is unavailable." };
  if (!tensors[2]) return { ...base, not_assessed_reason: "ADD output is unavailable." };
  const contracts = addContracts(...tensors);
  if (!contracts) return { ...base, not_assessed_reason: "ADD requires per-tensor INT8/UINT8 input and output contracts." };
  if (Number(context.source.scale_sample?.[0]) !== context.currentScale
    || Number(context.source.zero_point_sample?.[0]) !== context.currentZeroPoint) {
    return { ...base, not_assessed_reason: "Lattice source contract does not match the parsed consumer input." };
  }
  const candidateContracts = structuredClone(contracts);
  slots.filter((slot) => slot < 2).forEach((slot) => {
    candidateContracts.inputScales[slot] = Number(context.candidate.output_scale);
    candidateContracts.inputZeroPoints[slot] = Number(context.candidate.output_zero_point);
  });
  const currentParameters = deriveAddParameters(contracts);
  const candidateParameters = deriveAddParameters(candidateContracts);
  if (!currentParameters) return { ...base, not_assessed_reason: "Current ADD multiplier domain is invalid." };
  if (!candidateParameters) return { ...base, not_assessed_reason: "Candidate ADD multiplier domain is invalid." };
  const currentEncodings = [...currentParameters.input_multipliers, currentParameters.output_multiplier];
  const candidateEncodings = [...candidateParameters.input_multipliers, candidateParameters.output_multiplier];
  return {
    ...base,
    assessment_status: "assessed",
    current_parameters: currentParameters,
    candidate_parameters: candidateParameters,
    changed_offset_count: currentParameters.input_offsets.filter((value, index) => value !== candidateParameters.input_offsets[index]).length,
    changed_multiplier_encoding_count: currentEncodings.filter((value, index) => value.quantized_multiplier !== candidateEncodings[index].quantized_multiplier
      || value.shift !== candidateEncodings[index].shift).length,
    changed_shift_count: currentEncodings.filter((value, index) => value.shift !== candidateEncodings[index].shift).length,
  };
}

function addContracts(input0, input1, output) {
  if (!["INT8", "UINT8"].includes(input0.dtype) || input1.dtype !== input0.dtype || output.dtype !== input0.dtype
    || input0.scale_sample?.length !== 1 || input1.scale_sample?.length !== 1 || output.scale_sample?.length !== 1
    || input0.zero_point_sample?.length !== 1 || input1.zero_point_sample?.length !== 1 || output.zero_point_sample?.length !== 1) return null;
  const inputScales = [Number(input0.scale_sample[0]), Number(input1.scale_sample[0])];
  const outputScale = Number(output.scale_sample[0]);
  if (![...inputScales, outputScale].every(validScale)) return null;
  return {
    inputScales,
    inputZeroPoints: [Number(input0.zero_point_sample[0]), Number(input1.zero_point_sample[0])],
    outputScale,
    outputZeroPoint: Number(output.zero_point_sample[0]),
  };
}

function deriveAddParameters(contract) {
  const leftShift = 20;
  const twiceMax = 2 * Math.max(...contract.inputScales);
  const real = [
    contract.inputScales[0] / twiceMax,
    contract.inputScales[1] / twiceMax,
    twiceMax / ((2 ** leftShift) * contract.outputScale),
  ];
  if (real.some((value) => !validScale(value) || value >= 1)) return null;
  const encode = (value) => {
    const encoding = quantizeMultiplier(value, false);
    return {
      real_multiplier: value,
      quantized_multiplier: encoding.multiplier,
      shift: encoding.shift,
      represented_multiplier: encoding.represented,
    };
  };
  return {
    left_shift: leftShift,
    twice_max_input_scale: twiceMax,
    input_offsets: contract.inputZeroPoints.map((value) => -value),
    output_offset: contract.outputZeroPoint,
    input_multipliers: [encode(real[0]), encode(real[1])],
    output_multiplier: encode(real[2]),
  };
}

function decodeBias(op, tensors, modelBytes, channels) {
  const index = Number(op.inputs?.[2]);
  if (!(index >= 0)) return { values: Array(channels).fill(0), tensor: null, status: "absent_zero_bias" };
  const tensor = tensors.get(index);
  if (!tensor) throw new Error(`Bias tensor ${index} is unavailable.`);
  if (tensor.dtype !== "INT32") throw new Error(`Bias tensor ${index} uses ${tensor.dtype}; INT32 is required for integer accumulation.`);
  const offset = Number(tensor.buffer_data_offset);
  const length = Number(tensor.buffer_data_length);
  if (!(offset >= 0) || length !== channels * 4 || offset + length > modelBytes.byteLength) {
    throw new Error(`Bias tensor ${index} exposes ${length} byte(s); ${channels} output channels require ${channels * 4} INT32 byte(s).`);
  }
  const view = new DataView(modelBytes.buffer, modelBytes.byteOffset + offset, length);
  return {
    values: Array.from({ length: channels }, (_, channel) => view.getInt32(channel * 4, true)),
    tensor,
    status: "stored_int32_bias",
  };
}

function compareComputedAnalysis(actual, expected) {
  for (const [key, value] of Object.entries(expected.summary)) assertDeep(actual[key], value, `Contract-migration ${key}`);
  assertDeep(actual.migrations, expected.migrations, "Contract-migration rows");
}

function consumerMap(ops) {
  const map = new Map();
  for (const op of ops) {
    for (const input of new Set((op.inputs || []).map(Number).filter((value) => value >= 0))) {
      if (!map.has(input)) map.set(input, []);
      map.get(input).push(Number(op.index));
    }
  }
  return map;
}

function downstreamOps(sourceTensor, ops, consumers) {
  const depths = new Map();
  const queue = [[sourceTensor, 0]];
  const visitedTensors = new Set([sourceTensor]);
  while (queue.length) {
    const [tensor, tensorDepth] = queue.shift();
    for (const opIndex of consumers.get(tensor) || []) {
      const depth = tensorDepth + 1;
      depths.set(opIndex, Math.min(depths.get(opIndex) ?? Infinity, depth));
      const op = ops.get(opIndex);
      for (const output of (op?.outputs || []).map(Number).filter((value) => value >= 0)) {
        if (!visitedTensors.has(output)) {
          visitedTensors.add(output);
          queue.push([output, depth]);
        }
      }
    }
  }
  return [...depths].sort((left, right) => left[0] - right[0]).map(([opIndex, depth]) => ({
    op_index: opIndex,
    op_name: ops.get(opIndex).name,
    minimum_edge_depth: depth,
    direct_consumer: depth === 1,
  }));
}

function matchingInputSlots(op, tensorIndex) {
  return (op.inputs || []).map(Number).flatMap((value, index) => value === tensorIndex ? [index] : []);
}

function migrationClass(name) {
  return KERNELS.has(name) ? "integer_kernel" : name === "ADD" ? "quantized_add" : "not_modeled";
}

function kernelLedgerRow(opIndex, witness) {
  return `op=${opIndex};channel=${witness.channel_index};weight_scale=${f64Bits(witness.weight_scale)};current_real=${f64Bits(witness.current_real_multiplier)};candidate_real=${f64Bits(witness.candidate_real_multiplier)};current_q=${witness.current_quantized_multiplier};candidate_q=${witness.candidate_quantized_multiplier};current_shift=${witness.current_shift};candidate_shift=${witness.candidate_shift};current_bias=${witness.current_bias_code};candidate_bias=${witness.candidate_bias_code_decimal};overflow=${Number(witness.bias_int32_overflow)};preserved_bias=${f64Bits(witness.preserved_bias_real_value)};candidate_bias_real=${optionalF64Bits(witness.candidate_bias_real_value)};bias_error=${optionalF64Bits(witness.absolute_bias_rebase_error)};current_steps=${optionalF64Bits(witness.absolute_bias_rebase_error_current_steps)};candidate_steps=${optionalF64Bits(witness.absolute_bias_rebase_error_candidate_steps)}\n`;
}

function f64Bits(value) {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value, false);
  return view.getBigUint64(0, false).toString(16).padStart(16, "0");
}

function optionalF64Bits(value) {
  return value == null ? "na" : f64Bits(value);
}

function validateSources(sources) {
  assert(Array.isArray(sources) && sources.length === 3, "Contract-migration source coverage mismatch.");
  for (const source of sources) {
    assert(source.sha256 === SOURCE_SHA256[source.file], `Contract-migration source digest mismatch for ${source.file}.`);
    assert(source.url === `https://github.com/tensorflow/tensorflow/blob/${SOURCE_COMMIT}/${source.file}`, `Contract-migration source URL mismatch for ${source.file}.`);
  }
}

function renderMigrationSummary(root, result) {
  root?.replaceChildren(
    metric("Direct boundary", `${result.direct_consumer_count} ops`, `${result.kernel_consumer_count} kernels / ${result.add_consumer_count} ADDs`),
    metric("Kernel channels", formatNumber(result.assessed_kernel_channel_scenario_count), `${formatNumber(result.multiplier_shift_changed_channel_scenario_count)} shift changes`),
    metric("Bias regeneration", formatNumber(result.bias_code_changed_channel_scenario_count), `${result.bias_int32_overflow_channel_scenario_count} INT32 overflow`),
    metric("ADD regeneration", formatNumber(result.add_parameter_encoding_changed_scenario_count), `${result.reachable_downstream_op_union_count} reachable ops`),
  );
}

function migrationSelector(rows, selected) {
  const wrap = element("div", "migration-selector");
  rows.forEach((row) => {
    const button = element("button", row.source_add_op_index === selected ? "active" : "", `#${padOp(row.source_add_op_index)} ADD`);
    button.type = "button";
    button.dataset.migrationAdd = String(row.source_add_op_index);
    button.title = `${row.direct_consumer_count} direct consumers / ${row.reachable_downstream_op_count} reachable ops`;
    wrap.append(button);
  });
  return wrap;
}

function designTabs(scenarios, selected) {
  const tabs = element("div", "migration-design-tabs");
  scenarios.forEach((scenario) => {
    const label = scenario.design.startsWith("fixed") ? "Keep zero-point" : "Finest containment";
    const button = element("button", scenario.design === selected ? "active" : "", label);
    button.type = "button";
    button.dataset.migrationDesign = scenario.design;
    button.setAttribute("aria-pressed", String(scenario.design === selected));
    tabs.append(button);
  });
  return tabs;
}

function scenarioHeadline(row, scenario) {
  const section = element("section", "migration-headline");
  const title = element("div", "");
  title.append(
    element("span", "", `#${padOp(row.source_add_op_index)} output contract`),
    element("strong", "", `${concise(row.current_output_scale)} / zp ${row.current_output_zero_point}  ->  ${concise(scenario.candidate_output_scale)} / zp ${scenario.candidate_output_zero_point}`),
    element("small", "", `${scenario.scale_ratio_to_current.toFixed(4)}x scale / zp ${signed(scenario.signed_zero_point_delta)} / ${row.direct_consumer_count} direct consumers`),
  );
  const metrics = element("div", "migration-inline-metrics");
  metrics.append(
    compactMetric("Multiplier", `${formatNumber(scenario.multiplier_encoding_changed_channel_count)} changed`),
    compactMetric("Shift", `${formatNumber(scenario.multiplier_shift_changed_channel_count)} changed`),
    compactMetric("Bias", `${formatNumber(scenario.bias_code_changed_channel_count)} codes`),
    compactMetric("Overflow", String(scenario.bias_int32_overflow_channel_count)),
  );
  section.append(title, metrics);
  return section;
}

function consumerLedger(scenario) {
  const section = element("section", "migration-consumers");
  section.append(element("h4", "", "Direct parameter-regeneration boundary"));
  for (const row of scenario.kernel_consumers) {
    const item = element("div", "migration-consumer-row");
    item.append(
      graphButton(row.op_index, row.op_name),
      compactMetric("Channels", formatNumber(row.assessed_channel_count)),
      compactMetric("Q0.31", `${formatNumber(row.multiplier_encoding_changed_channel_count)} changed`),
      compactMetric("Shift", `${formatNumber(row.multiplier_shift_changed_channel_count)} changed`),
      compactMetric("Bias", `${formatNumber(row.bias_code_changed_channel_count)} changed`),
      compactMetric("Max error", `${concise(row.maximum_absolute_bias_rebase_error_current_steps)} old steps`),
      statusPill(row.bias_int32_overflow_channel_count ? `${row.bias_int32_overflow_channel_count} overflow` : "INT32 safe", row.bias_int32_overflow_channel_count ? "risk" : "ok"),
    );
    section.append(item);
  }
  for (const row of scenario.add_consumers) {
    const item = element("div", "migration-consumer-row");
    item.append(
      graphButton(row.op_index, row.op_name),
      compactMetric("Q0.31", `${row.changed_multiplier_encoding_count} / 3 changed`),
      compactMetric("Shift", `${row.changed_shift_count} / 3 changed`),
      compactMetric("Offsets", `${row.changed_offset_count} / 2 changed`),
      compactMetric("Left shift", String(row.candidate_parameters?.left_shift ?? "n/a")),
      statusPill(row.assessment_status === "assessed" ? "source-backed" : "not assessed", row.assessment_status === "assessed" ? "ok" : "risk"),
    );
    section.append(item);
  }
  return section;
}

function topBiasTable(scenario) {
  const section = element("section", "migration-table-section");
  section.append(element("h4", "", "Highest bias rebasing error"));
  const wrap = element("div", "migration-table-wrap");
  const table = element("table", "migration-table");
  table.innerHTML = "<thead><tr><th>Consumer</th><th>Channel</th><th>Weight scale</th><th>Multiplier q / shift</th><th>Bias code</th><th>Error, old steps</th><th>INT32</th></tr></thead>";
  const tbody = document.createElement("tbody");
  const rows = scenario.kernel_consumers.flatMap((consumer) => consumer.top_channels.map((channel) => ({ consumer, channel })))
    .sort((left, right) => (right.channel.absolute_bias_rebase_error_current_steps ?? Infinity)
      - (left.channel.absolute_bias_rebase_error_current_steps ?? Infinity)).slice(0, 12);
  if (!rows.length) {
    section.append(messageNode("This candidate requires no kernel multiplier, shift, or INT32 bias regeneration on the direct consumer boundary."));
    return section;
  }
  rows.forEach(({ consumer, channel }) => {
    const tr = document.createElement("tr");
    [
      `#${padOp(consumer.op_index)} ${consumer.op_name}`,
      channel.channel_index,
      concise(channel.weight_scale),
      `${channel.current_quantized_multiplier} / ${channel.current_shift} -> ${channel.candidate_quantized_multiplier} / ${channel.candidate_shift}`,
      `${channel.current_bias_code} -> ${channel.candidate_bias_code_decimal}`,
      concise(channel.absolute_bias_rebase_error_current_steps),
      channel.bias_int32_overflow ? "overflow" : "safe",
    ].forEach((value) => tr.append(element("td", "", String(value))));
    tbody.append(tr);
  });
  table.append(tbody);
  wrap.append(table);
  section.append(wrap);
  return section;
}

function addParameterTable(scenario) {
  const section = element("section", "migration-table-section");
  section.append(element("h4", "", "Direct ADD prepare-time parameters"));
  if (!scenario.add_consumers.length) {
    section.append(messageNode("The selected residual output has no direct ADD consumer."));
    return section;
  }
  const wrap = element("div", "migration-table-wrap");
  const table = element("table", "migration-table");
  table.innerHTML = "<thead><tr><th>ADD</th><th>Parameter</th><th>Current real</th><th>Current q / shift</th><th>Candidate real</th><th>Candidate q / shift</th></tr></thead>";
  const tbody = document.createElement("tbody");
  scenario.add_consumers.forEach((consumer) => {
    if (consumer.assessment_status !== "assessed") return;
    const current = [...consumer.current_parameters.input_multipliers, consumer.current_parameters.output_multiplier];
    const candidate = [...consumer.candidate_parameters.input_multipliers, consumer.candidate_parameters.output_multiplier];
    ["input 0", "input 1", "output"].forEach((label, index) => {
      const tr = document.createElement("tr");
      [
        `#${padOp(consumer.op_index)}`,
        label,
        concise(current[index].real_multiplier),
        `${current[index].quantized_multiplier} / ${current[index].shift}`,
        concise(candidate[index].real_multiplier),
        `${candidate[index].quantized_multiplier} / ${candidate[index].shift}`,
      ].forEach((value) => tr.append(element("td", "", String(value))));
      tbody.append(tr);
    });
  });
  table.append(tbody);
  wrap.append(table);
  section.append(wrap);
  return section;
}

function plotBand(canvas, title, subtitle) {
  const section = element("section", "migration-plot-band");
  const head = element("div", "migration-plot-head");
  head.append(element("h4", "", title), element("span", "", subtitle));
  section.append(head, canvas);
  return section;
}

export function drawChannelMigration(canvas, scenario, logicalWidth = null, logicalHeight = 260) {
  const channels = scenario.kernel_consumers.flatMap((consumer) => consumer.channel_current_shifts.map((currentShift, index) => ({
    currentShift,
    candidateShift: consumer.channel_candidate_shifts[index],
    error: consumer.channel_bias_rebase_error_current_steps[index] ?? 0,
  })));
  const width = logicalWidth || Math.max(280, canvas.parentElement?.clientWidth || 760);
  const height = logicalHeight;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const margin = { left: 44, right: 18, top: 20, bottom: 34 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  context.fillStyle = "#f5f7f9";
  context.fillRect(margin.left, margin.top, plotWidth, plotHeight);
  if (!channels.length) {
    context.fillStyle = "#66717f";
    context.font = "12px system-ui";
    context.fillText("No assessed kernel channels", margin.left + 12, margin.top + 24);
    return;
  }
  const bins = Math.max(1, Math.min(Math.floor(plotWidth), channels.length));
  const binSize = channels.length / bins;
  const maxError = Math.max(1, ...channels.map((row) => row.error));
  const maxShiftDelta = Math.max(1, ...channels.map((row) => Math.abs(row.candidateShift - row.currentShift)));
  for (let bin = 0; bin < bins; bin += 1) {
    const start = Math.floor(bin * binSize);
    const end = Math.max(start + 1, Math.floor((bin + 1) * binSize));
    const slice = channels.slice(start, end);
    const shiftDelta = slice.reduce((max, row) => Math.max(max, Math.abs(row.candidateShift - row.currentShift)), 0);
    const error = slice.reduce((max, row) => Math.max(max, row.error), 0);
    const x = margin.left + bin / bins * plotWidth;
    const barWidth = Math.max(1, plotWidth / bins);
    context.fillStyle = `rgba(180,63,67,${0.12 + 0.78 * error / maxError})`;
    context.fillRect(x, margin.top + plotHeight / 2, barWidth, plotHeight / 2);
    if (shiftDelta) {
      const h = shiftDelta / maxShiftDelta * (plotHeight / 2 - 4);
      context.fillStyle = "#28666e";
      context.fillRect(x, margin.top + plotHeight / 2 - h, barWidth, h);
    }
  }
  drawAxis(context, margin, plotWidth, plotHeight, `${formatNumber(channels.length)} channels`, `max bias ${maxError.toFixed(3)} steps`);
}

export function drawImpactRadius(canvas, row, logicalWidth = null, logicalHeight = 220) {
  const width = logicalWidth || Math.max(280, canvas.parentElement?.clientWidth || 760);
  const height = logicalHeight;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const counts = new Map();
  row.affected_ops.forEach((op) => counts.set(op.minimum_edge_depth, (counts.get(op.minimum_edge_depth) || 0) + 1));
  const maxDepth = Math.max(1, ...counts.keys());
  const maxCount = Math.max(1, ...counts.values());
  const margin = { left: 42, right: 18, top: 18, bottom: 34 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  context.fillStyle = "#f5f7f9";
  context.fillRect(margin.left, margin.top, plotWidth, plotHeight);
  const slot = plotWidth / maxDepth;
  for (let depth = 1; depth <= maxDepth; depth += 1) {
    const count = counts.get(depth) || 0;
    const barHeight = count / maxCount * (plotHeight - 8);
    context.fillStyle = depth === 1 ? "#b43f43" : "#28666e";
    context.fillRect(margin.left + (depth - 1) * slot + 2, margin.top + plotHeight - barHeight, Math.max(2, slot - 4), barHeight);
  }
  drawAxis(context, margin, plotWidth, plotHeight, `${maxDepth} edge depths`, `${row.reachable_downstream_op_count} reachable ops`);
}

function drawAxis(context, margin, width, height, xLabel, note) {
  context.strokeStyle = "#6c7785";
  context.lineWidth = 1;
  context.strokeRect(margin.left, margin.top, width, height);
  context.fillStyle = "#5c6876";
  context.font = "11px system-ui";
  context.textAlign = "left";
  context.fillText(note, margin.left + 6, margin.top + 14);
  context.textAlign = "center";
  context.fillText(xLabel, margin.left + width / 2, margin.top + height + 24);
}

function methodBoundary(result) {
  const section = element("section", "migration-method");
  section.append(
    element("strong", "", `${result.schema} / ${result.method_version}`),
    element("p", "", result.interpretation_boundary),
    element("code", "", result.bias_rebase_formula),
  );
  const sources = element("div", "migration-sources");
  result.source_references.forEach((source) => {
    const link = element("a", "", source.file.split("/").at(-1));
    link.href = source.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.title = `${source.role} / SHA-256 ${source.sha256}`;
    sources.append(link);
  });
  section.append(sources);
  return section;
}

function graphButton(opIndex, opName) {
  const button = element("button", "migration-graph-button", `#${padOp(opIndex)} ${opName}`);
  button.type = "button";
  button.dataset.migrationOpenGraph = String(opIndex);
  button.title = `Open op #${opIndex} in graph explorer`;
  return button;
}

function metric(label, value, detail) {
  const node = element("div", "migration-metric");
  node.append(element("span", "", label), element("strong", "", value), element("small", "", detail));
  return node;
}

function compactMetric(label, value) {
  const node = element("div", "migration-compact-metric");
  node.append(element("span", "", label), element("strong", "", value));
  return node;
}

function statusPill(text, tone) {
  return element("span", `migration-status ${tone}`, text);
}

function messageNode(text, tone = "muted") {
  return element("p", `migration-message ${tone}`, text);
}

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

function assertDeep(actual, expected, label) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`${label} mismatch.`);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function validScale(value) {
  return Number.isFinite(value) && value > 0;
}

function maxOptional(values) {
  const finite = values.filter((value) => value != null);
  return finite.length ? Math.max(...finite) : null;
}

function maxOrZero(values) {
  return values.length ? Math.max(...values) : 0;
}

function sum(values) {
  return values.reduce((total, value) => total + Number(value || 0), 0);
}

function nullableDesc(left, right) {
  return (right == null ? Infinity : right) - (left == null ? Infinity : left);
}

function concise(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "n/a";
  if (number === 0) return "0";
  if (Math.abs(number) >= 1000 || Math.abs(number) < 0.001) return number.toExponential(4);
  return number.toPrecision(6).replace(/\.?0+$/, "");
}

function signed(value) {
  return Number(value) >= 0 ? `+${value}` : String(value);
}
