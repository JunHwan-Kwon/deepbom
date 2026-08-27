import { sha256Hex } from "./hash.js";

export const ACCUMULATOR_ATLAS_SCHEMA = "deepbom.accumulator_atlas.v1.3";
const SOURCE_COMMIT = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const INT32_MIN = -2147483648n;
const INT32_MAX = 2147483647n;
const INT32_HALF_RANGE_MAX = INT32_MAX / 2n;
const BIAS_HALF_RANGE_FLOAT32_TOLERANCE_CODES = 129n;
const HISTOGRAM_BINS = 129;
const CANDIDATE_OPS = new Set(["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"]);

export function createAccumulatorAtlasController({
  root,
  status,
  summary,
  body,
  downloadButton,
  getContext,
  jumpToGraphOp,
  onDownload,
}) {
  let analysis = null;
  let bytes = null;
  let atlas = null;
  let reconstructed = null;
  let selectedOpIndex = null;
  let mode = "bits";
  let renderToken = 0;
  let resizeObserver = null;

  downloadButton?.addEventListener("click", () => {
    if (atlas) onDownload?.(atlas, "accumulator_headroom_atlas.json");
  });

  function render(explicitAnalysis = null) {
    renderToken += 1;
    const token = renderToken;
    const context = getContext?.() || {};
    analysis = explicitAnalysis || context.analysis || null;
    bytes = context.modelBytes || null;
    atlas = analysis?.accumulator_atlas || null;
    resizeObserver?.disconnect();
    resizeObserver = null;
    if (!atlas || !bytes || String(analysis?.format || "").toLowerCase() !== "tflite") {
      selectedOpIndex = null;
      reconstructed = null;
      if (root) root.hidden = true;
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    if (root) root.hidden = false;
    if (downloadButton) downloadButton.disabled = false;
    try {
      reconstructed = validateAccumulatorAtlas(analysis, bytes);
      const assessed = atlas.ops.filter((row) => row.assessment_status === "assessed");
      if (!assessed.some((row) => row.op_index === selectedOpIndex)) {
        selectedOpIndex = atlas.headroom_ranking_op_indices?.[0] ?? assessed[0]?.op_index ?? null;
      }
      renderSummary(summary, atlas);
      renderBody();
      if (status) {
        status.textContent = atlas.assessed_op_count
          ? "arithmetic verified · digest pending"
          : atlas.status.replaceAll("_", " ");
        status.dataset.tone = headroomTone(atlas);
      }
      if (atlas.assessed_op_count) {
        void validateAccumulatorAtlasDigests(analysis, bytes).then(() => {
          if (token !== renderToken || !status) return;
          status.textContent = Number(atlas.maximum_int32_ratio || 0) >= 0.9
            ? "independently verified · critical headroom"
            : "independently verified";
          status.dataset.tone = headroomTone(atlas);
        }).catch((error) => {
          if (token !== renderToken || !status) return;
          status.textContent = `integrity error: ${error.message}`;
          status.dataset.tone = "risk";
        });
      }
    } catch (error) {
      reconstructed = null;
      if (summary) summary.replaceChildren();
      if (body) body.replaceChildren(messageNode(`Accumulator evidence rejected: ${error.message}`, "risk"));
      if (status) {
        status.textContent = "evidence rejected";
        status.dataset.tone = "risk";
      }
    }
  }

  function renderBody() {
    if (!body || !atlas) return;
    resizeObserver?.disconnect();
    const assessed = atlas.ops.filter((row) => row.assessment_status === "assessed");
    if (!assessed.length) {
      body.replaceChildren(
        messageNode("No constant 8-bit Conv, Depthwise Conv, or rank-2 Fully Connected accumulator was assessable."),
        unassessedTable(atlas.ops),
      );
      return;
    }
    const row = assessed.find((item) => item.op_index === selectedOpIndex) || assessed[0];
    selectedOpIndex = row.op_index;
    const toolbar = document.createElement("div");
    toolbar.className = "accumulator-toolbar";
    const select = document.createElement("select");
    select.className = "accumulator-op-select";
    select.setAttribute("aria-label", "Accumulator operator");
    for (const candidate of assessed) {
      const option = new Option(
        `#${candidate.op_index} ${candidate.op_name} · ${formatNumber(candidate.assessed_channel_count)} ch · ${candidate.maximum_required_signed_bits} bits`,
        String(candidate.op_index),
      );
      option.selected = candidate.op_index === row.op_index;
      select.append(option);
    }
    select.addEventListener("change", () => {
      selectedOpIndex = Number(select.value);
      renderBody();
    });
    toolbar.append(select, modeControl());

    const main = document.createElement("div");
    main.className = "accumulator-main";
    const canvasWrap = document.createElement("div");
    canvasWrap.className = "accumulator-canvas-wrap";
    const canvas = document.createElement("canvas");
    canvas.className = "accumulator-canvas";
    canvas.tabIndex = 0;
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `Per-channel accumulator map for operator ${row.op_index}`);
    const tooltip = document.createElement("div");
    tooltip.className = "accumulator-tooltip";
    tooltip.hidden = true;
    canvasWrap.append(canvas, tooltip);
    installCanvasInteraction(canvas, tooltip, row);
    const details = document.createElement("div");
    details.className = "accumulator-details";
    details.append(
      definitionTable([
        ["Input contract", `${row.input_dtype} [${row.input_code_range.join(", ")}], zp ${row.input_zero_point}`],
        ["Kernel", `${row.weight_dtype} ${shapeText(row.weight_shape)} · ${row.weight_zero_point_mode.replaceAll("_", " ")}`],
        ["Channel equation", `${formatNumber(row.assessed_channel_count)} channels × ${formatNumber(row.accumulation_terms_per_channel)} terms`],
        ["Bias", row.bias_status.replaceAll("_", " ")],
        ["Pinned source", sourceLink(row)],
        ["Channel ledger", row.channel_ledger_sha256],
      ]),
      metricTable(row),
      witnessTable(row.worst_channel),
    );
    main.append(canvasWrap, details);

    const histogram = requiredBitsHistogram(row);
    const ranking = rankingTable(atlas.ops, row.op_index, (opIndex) => {
      selectedOpIndex = opIndex;
      renderBody();
    }, jumpToGraphOp);
    const boundary = document.createElement("p");
    boundary.className = "accumulator-boundary";
    boundary.textContent = atlas.interpretation_boundary;
    const pressure = accumulatorPressureNotice(row);
    const biasPressure = biasPolicyNotice(row);
    body.replaceChildren(
      ...(biasPressure ? [biasPressure] : []),
      ...(pressure ? [pressure] : []),
      toolbar,
      main,
      histogram,
      ranking,
      boundary,
    );
    requestAnimationFrame(() => drawAccumulatorAtlasCanvas(canvas, row, mode));
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver(() => drawAccumulatorAtlasCanvas(canvas, row, mode));
      resizeObserver.observe(canvasWrap);
    }
  }

  function modeControl() {
    const control = document.createElement("div");
    control.className = "accumulator-mode-control";
    control.setAttribute("role", "tablist");
    control.setAttribute("aria-label", "Accumulator map mode");
    for (const [id, label] of [["bits", "Required bits"], ["ratio", "INT32 use"], ["polarity", "Signed reach"]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.dataset.accumulatorMode = id;
      button.classList.toggle("active", mode === id);
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", String(mode === id));
      button.addEventListener("click", () => {
        mode = id;
        renderBody();
      });
      control.append(button);
    }
    return control;
  }

  return { render, getReconstructed: () => reconstructed };
}

export function validateAccumulatorAtlas(analysis, modelBytes) {
  const atlas = analysis?.accumulator_atlas;
  assert(atlas && atlas.schema === ACCUMULATOR_ATLAS_SCHEMA, "Accumulator schema mismatch.");
  assert(atlas.method_version === "2026-07-30.4", "Accumulator method version mismatch.");
  assert(atlas.evidence_class === "DERIVED", "Accumulator evidence class must be DERIVED.");
  assert(atlas.source_commit === SOURCE_COMMIT, "Accumulator source commit mismatch.");
  assert(String(atlas.interpretation_boundary || "").includes("not an observed activation distribution"), "Accumulator interpretation boundary is incomplete.");
  assert(String(atlas.interpretation_boundary || "").includes("executed delegate/microkernel trace"), "Accumulator runtime boundary is incomplete.");
  const reconstructed = reconstructAccumulatorAtlas(analysis, modelBytes);
  assert(atlas.ops.length === reconstructed.ops.length, "Accumulator candidate row count mismatch.");
  for (let index = 0; index < reconstructed.ops.length; index += 1) {
    compareOpRow(atlas.ops[index], reconstructed.ops[index]);
  }
  compareScalar(atlas, reconstructed, "candidate_op_count");
  compareScalar(atlas, reconstructed, "assessed_op_count");
  compareScalar(atlas, reconstructed, "unassessed_op_count");
  compareScalar(atlas, reconstructed, "assessed_channel_count");
  compareScalar(atlas, reconstructed, "stored_bias_channel_count");
  compareScalar(atlas, reconstructed, "int32_safe_channel_count");
  compareScalar(atlas, reconstructed, "int32_overflow_channel_count");
  compareScalar(atlas, reconstructed, "overflow_op_count");
  compareOptional(atlas, reconstructed, "maximum_absolute_bias_decimal");
  compareOptionalNear(atlas, reconstructed, "maximum_bias_int32_ratio");
  for (const key of [
    "bias_half_range_exceedance_channel_count",
    "bias_half_range_exceedance_op_count",
    "bias_half_range_guard_adjacent_channel_count",
    "bias_half_range_material_exceedance_channel_count",
    "bias_half_range_material_exceedance_op_count",
    "exact_zero_kernel_channel_count",
    "exact_zero_bias_half_range_exceedance_channel_count",
    "exact_zero_bias_half_range_material_exceedance_channel_count",
  ]) compareScalar(atlas, reconstructed, key);
  compareOptional(atlas, reconstructed, "maximum_absolute_accumulator_decimal");
  compareOptionalNear(atlas, reconstructed, "maximum_int32_ratio");
  compareOptional(atlas, reconstructed, "maximum_required_signed_bits");
  compareOptional(atlas, reconstructed, "minimum_int32_headroom_bits");
  assert(equalArray(atlas.headroom_ranking_op_indices, reconstructed.headroom_ranking_op_indices), "Accumulator ranking mismatch.");
  assert(equalArray(atlas.required_signed_bits_histogram, reconstructed.required_signed_bits_histogram), "Accumulator global bit histogram mismatch.");
  assert(atlas.status === reconstructed.status, "Accumulator aggregate status mismatch.");
  return reconstructed;
}

export async function validateAccumulatorAtlasDigests(analysis, modelBytes) {
  const reconstructed = validateAccumulatorAtlas(analysis, modelBytes);
  const rows = analysis.accumulator_atlas.ops;
  for (let index = 0; index < rows.length; index += 1) {
    if (rows[index].assessment_status !== "assessed") continue;
    const digest = await sha256Hex(new TextEncoder().encode(reconstructed.ops[index].ledger_text));
    assert(digest === rows[index].channel_ledger_sha256, `Accumulator ledger SHA-256 mismatch at op #${rows[index].op_index}.`);
  }
  return reconstructed;
}

export function reconstructAccumulatorAtlas(analysis, modelBytes) {
  const bytes = asBytes(modelBytes);
  const tensors = new Map((analysis?.tensors || []).map((tensor) => [Number(tensor.index), tensor]));
  const candidates = (analysis?.ops || []).filter((op) => CANDIDATE_OPS.has(String(op.name)));
  const ops = candidates.map((op) => {
    try {
      return reconstructOp(op, tensors, bytes);
    } catch (error) {
      return {
        op_index: Number(op.index),
        op_name: String(op.name),
        assessment_status: "not_assessed",
        not_assessed_reason: error.message,
      };
    }
  });
  const assessed = ops.filter((row) => row.assessment_status === "assessed");
  const ranking = [...assessed].sort((left, right) => right.maximum_int32_ratio - left.maximum_int32_ratio || left.op_index - right.op_index);
  const histogram = Array(HISTOGRAM_BINS).fill(0);
  for (const row of assessed) row.required_signed_bits_histogram.forEach((count, index) => { histogram[index] += count; });
  const top = ranking[0] || null;
  const maximumBiasRow = [...assessed]
    .filter((row) => row.maximum_bias_int32_ratio != null)
    .sort((left, right) => right.maximum_bias_int32_ratio - left.maximum_bias_int32_ratio)[0] || null;
  return {
    status: !ops.length ? "not_applicable" : assessed.length === ops.length ? "assessed" : "partial",
    candidate_op_count: ops.length,
    assessed_op_count: assessed.length,
    unassessed_op_count: ops.length - assessed.length,
    assessed_channel_count: sum(assessed.map((row) => row.assessed_channel_count)),
    stored_bias_channel_count: sum(assessed.map((row) => row.stored_bias_channel_count)),
    int32_safe_channel_count: sum(assessed.map((row) => row.int32_safe_channel_count)),
    int32_overflow_channel_count: sum(assessed.map((row) => row.int32_overflow_channel_count)),
    overflow_op_count: assessed.filter((row) => row.int32_overflow_channel_count > 0).length,
    maximum_absolute_bias_decimal: maximumBiasRow?.maximum_absolute_bias_decimal ?? null,
    maximum_bias_int32_ratio: maximumBiasRow?.maximum_bias_int32_ratio ?? null,
    bias_half_range_exceedance_channel_count: sum(assessed.map((row) => row.bias_half_range_exceedance_channel_count)),
    bias_half_range_exceedance_op_count: assessed.filter((row) => row.bias_half_range_exceedance_channel_count > 0).length,
    bias_half_range_guard_adjacent_channel_count: sum(assessed.map((row) => row.bias_half_range_guard_adjacent_channel_count)),
    bias_half_range_material_exceedance_channel_count: sum(assessed.map((row) => row.bias_half_range_material_exceedance_channel_count)),
    bias_half_range_material_exceedance_op_count: assessed.filter((row) => row.bias_half_range_material_exceedance_channel_count > 0).length,
    exact_zero_kernel_channel_count: sum(assessed.map((row) => row.exact_zero_kernel_channel_count)),
    exact_zero_bias_half_range_exceedance_channel_count: sum(assessed.map((row) => row.exact_zero_bias_half_range_exceedance_channel_count)),
    exact_zero_bias_half_range_material_exceedance_channel_count: sum(assessed.map((row) => row.exact_zero_bias_half_range_material_exceedance_channel_count)),
    maximum_absolute_accumulator_decimal: top?.maximum_absolute_accumulator_decimal ?? null,
    maximum_int32_ratio: top?.maximum_int32_ratio ?? null,
    maximum_required_signed_bits: assessed.length ? Math.max(...assessed.map((row) => row.maximum_required_signed_bits)) : null,
    minimum_int32_headroom_bits: assessed.length ? Math.min(...assessed.map((row) => row.minimum_int32_headroom_bits)) : null,
    headroom_ranking_op_indices: ranking.map((row) => row.op_index),
    required_signed_bits_histogram: histogram,
    ops,
  };
}

function reconstructOp(op, tensors, bytes) {
  const input = requiredTensor(tensors, op.inputs?.[0], "Input tensor is unavailable.");
  const weight = requiredTensor(tensors, op.inputs?.[1], "Weight tensor is unavailable.");
  const [qmin, qmax] = codeRange(input.dtype, `Input tensor ${input.index} uses ${input.dtype}; INT8 or UINT8 is required.`);
  if (Number(input.quant_scales) !== 1 || input.scale_sample?.length !== 1) throw new Error(`Input tensor ${input.index} does not expose one per-tensor scale.`);
  if (!(Number(input.scale_sample[0]) > 0) || !Number.isFinite(Number(input.scale_sample[0]))) throw new Error(`Input tensor ${input.index} has an invalid quantization scale.`);
  if (Number(input.quant_zero_points) !== 1 || input.zero_point_sample?.length !== 1) throw new Error(`Input tensor ${input.index} does not expose one per-tensor zero-point.`);
  const inputZeroPoint = Number(input.zero_point_sample[0]);
  if (inputZeroPoint < qmin || inputZeroPoint > qmax) throw new Error(`Input zero-point ${inputZeroPoint} lies outside [${qmin}, ${qmax}].`);
  const [weightQmin, weightQmax] = codeRange(weight.dtype, `Weight tensor ${weight.index} uses ${weight.dtype}; INT8 or UINT8 is required.`);
  const layout = weightLayout(op, weight);
  const channels = layout.channels;
  const terms = layout.terms;
  const scales = weight.scale_sample || [];
  if (![1, channels].includes(scales.length) || scales.some((value) => !(Number(value) > 0) || !Number.isFinite(Number(value)))) {
    throw new Error(`Weight tensor ${weight.index} scale cardinality is not 1 or output-channel count ${channels}.`);
  }
  if (scales.length > 1 && Number(weight.quantized_dimension) !== layout.axis) {
    throw new Error(`Weight tensor ${weight.index} per-axis dimension ${weight.quantized_dimension} does not match output-channel axis ${layout.axis}.`);
  }
  const zeroPoints = expandedZeroPoints(weight, channels, weightQmin, weightQmax);
  const rawWeights = tensorBytes(bytes, weight, `Weight tensor ${weight.index} constant bytes are unavailable.`);
  if (rawWeights.length !== channels * terms) throw new Error(`Weight tensor ${weight.index} exposes ${rawWeights.length} byte(s); shape requires ${channels * terms} 8-bit element(s).`);
  const biasResult = decodeBias(op, tensors, bytes, channels);
  const xmin = BigInt(qmin - inputZeroPoint);
  const xmax = BigInt(qmax - inputZeroPoint);
  const channelMins = [];
  const channelMaxs = [];
  const postBiasMins = [];
  const postBiasMaxs = [];
  const channelBits = [];
  const histogram = Array(HISTOGRAM_BINS).fill(0);
  const overflow = [];
  const halfRangeExceedance = [];
  const materialHalfRangeExceedance = [];
  const witnesses = [];
  let guardAdjacentCount = 0;
  let exactZeroKernelCount = 0;
  let exactZeroHalfRangeExceedanceCount = 0;
  let exactZeroMaterialHalfRangeExceedanceCount = 0;
  let maxLegalWeightMagnitude = 0n;
  let maxBiasMagnitude = 0n;
  let ledger = "";
  for (let channel = 0; channel < channels; channel += 1) {
    const weightZeroPoint = BigInt(zeroPoints[channel]);
    maxLegalWeightMagnitude = maxBigInt(maxLegalWeightMagnitude, absBigInt(BigInt(weightQmin) - weightZeroPoint), absBigInt(BigInt(weightQmax) - weightZeroPoint));
    let positive = 0n;
    let negative = 0n;
    for (let term = 0; term < terms; term += 1) {
      const centered = BigInt(rawCode(rawWeights[layout.rawIndex(channel, term)], weight.dtype)) - weightZeroPoint;
      if (centered >= 0n) positive += centered;
      else negative += centered;
    }
    const dotMin = positive * xmin + negative * xmax;
    const dotMax = positive * xmax + negative * xmin;
    const bias = BigInt(biasResult.values[channel]);
    const exactZeroCenteredKernel = positive === 0n && negative === 0n;
    if (exactZeroCenteredKernel) exactZeroKernelCount += 1;
    const biasExcess = biasResult.status === "stored_int32_bias"
      ? maxBigInt(0n, absBigInt(bias) - INT32_HALF_RANGE_MAX)
      : 0n;
    const biasClass = classifyBiasHalfRangeExcess(biasExcess);
    if (biasExcess > 0n) {
      halfRangeExceedance.push(channel);
      if (exactZeroCenteredKernel) exactZeroHalfRangeExceedanceCount += 1;
    }
    if (biasClass === "float32_guard_adjacent") {
      guardAdjacentCount += 1;
    } else if (biasClass === "material_exceedance") {
      materialHalfRangeExceedance.push(channel);
      if (exactZeroCenteredKernel) exactZeroMaterialHalfRangeExceedanceCount += 1;
    }
    maxBiasMagnitude = maxBigInt(maxBiasMagnitude, absBigInt(bias));
    const postMin = dotMin + bias;
    const postMax = dotMax + bias;
    const envelopeMin = minBigInt(0n, dotMin, postMin);
    const envelopeMax = maxBigInt(0n, dotMax, postMax);
    const maxAbs = maxBigInt(absBigInt(envelopeMin), absBigInt(envelopeMax));
    const bits = requiredSignedBits(envelopeMin, envelopeMax);
    const fits = envelopeMin >= INT32_MIN && envelopeMax <= INT32_MAX;
    if (!fits) overflow.push(channel);
    histogram[Math.min(bits, HISTOGRAM_BINS - 1)] += 1;
    const witness = {
      channel_index: channel,
      positive_centered_weight_sum_decimal: String(positive),
      negative_centered_weight_sum_decimal: String(negative),
      bias_decimal: String(bias),
      dot_product_min_decimal: String(dotMin),
      dot_product_max_decimal: String(dotMax),
      post_bias_min_decimal: String(postMin),
      post_bias_max_decimal: String(postMax),
      accumulator_envelope_min_decimal: String(envelopeMin),
      accumulator_envelope_max_decimal: String(envelopeMax),
      maximum_absolute_accumulator_decimal: String(maxAbs),
      required_signed_bits: bits,
      int32_ratio: Number(maxAbs) / Number(INT32_MAX),
      bias_int32_ratio: Number(absBigInt(bias)) / Number(INT32_MAX),
      bias_exceeds_half_range: biasExcess > 0n,
      bias_half_range_excess_decimal: String(biasExcess),
      bias_half_range_classification: biasClass,
      exact_zero_centered_kernel: exactZeroCenteredKernel,
      fits_int32: fits,
    };
    channelMins.push(String(envelopeMin));
    channelMaxs.push(String(envelopeMax));
    postBiasMins.push(String(postMin));
    postBiasMaxs.push(String(postMax));
    channelBits.push(bits);
    witnesses.push(witness);
    ledger += `op=${op.index};channel=${channel};post_min=${postMin};post_max=${postMax};min=${envelopeMin};max=${envelopeMax};bits=${bits}\n`;
  }
  witnesses.sort((left, right) => compareBigIntDesc(left.maximum_absolute_accumulator_decimal, right.maximum_absolute_accumulator_decimal) || left.channel_index - right.channel_index);
  const worst = witnesses[0];
  const maximumAbsolute = BigInt(worst.maximum_absolute_accumulator_decimal);
  const maximumBits = Math.max(...channelBits);
  const inputMagnitude = maxBigInt(absBigInt(xmin), absBigInt(xmax));
  const metadataBound = BigInt(terms) * inputMagnitude * maxLegalWeightMagnitude + maxBiasMagnitude;
  return {
    op_index: Number(op.index),
    op_name: String(op.name),
    assessment_status: "assessed",
    not_assessed_reason: "",
    input_code_range: [qmin, qmax],
    input_zero_point: inputZeroPoint,
    output_channel_axis: layout.axis,
    output_channel_count: channels,
    accumulation_terms_per_channel: terms,
    assessed_channel_count: channels,
    stored_bias_channel_count: biasResult.status === "stored_int32_bias" ? channels : 0,
    int32_safe_channel_count: channels - overflow.length,
    int32_overflow_channel_count: overflow.length,
    overflow_channel_indices: overflow,
    maximum_absolute_bias_decimal: String(maxBiasMagnitude),
    maximum_bias_int32_ratio: Number(maxBiasMagnitude) / Number(INT32_MAX),
    bias_half_range_exceedance_channel_count: halfRangeExceedance.length,
    bias_half_range_exceedance_channel_indices: halfRangeExceedance,
    bias_half_range_guard_adjacent_channel_count: guardAdjacentCount,
    bias_half_range_material_exceedance_channel_count: materialHalfRangeExceedance.length,
    bias_half_range_material_exceedance_channel_indices: materialHalfRangeExceedance,
    exact_zero_kernel_channel_count: exactZeroKernelCount,
    exact_zero_bias_half_range_exceedance_channel_count: exactZeroHalfRangeExceedanceCount,
    exact_zero_bias_half_range_material_exceedance_channel_count:
      exactZeroMaterialHalfRangeExceedanceCount,
    maximum_absolute_accumulator_decimal: String(maximumAbsolute),
    maximum_int32_ratio: Number(maximumAbsolute) / Number(INT32_MAX),
    maximum_required_signed_bits: maximumBits,
    minimum_int32_headroom_bits: 32 - maximumBits,
    metadata_only_magnitude_bound_decimal: String(metadataBound),
    metadata_only_int32_ratio: Number(metadataBound) / Number(INT32_MAX),
    exact_tightening_factor: maximumAbsolute > 0n ? Number(metadataBound) / Number(maximumAbsolute) : null,
    channel_accumulator_envelope_min_decimals: channelMins,
    channel_accumulator_envelope_max_decimals: channelMaxs,
    channel_post_bias_min_decimals: postBiasMins,
    channel_post_bias_max_decimals: postBiasMaxs,
    channel_required_signed_bits: channelBits,
    required_signed_bits_histogram: histogram,
    top_channels: witnesses.slice(0, 8),
    worst_channel: worst,
    ledger_text: ledger,
    weight_zero_point_mode: weight.zero_point_sample.length === 1 ? "per_tensor" : "per_output_channel",
    bias_status: biasResult.status,
  };
}

function compareOpRow(actual, expected) {
  assert(Number(actual.op_index) === expected.op_index && actual.op_name === expected.op_name, "Accumulator op binding mismatch.");
  assert(actual.assessment_status === expected.assessment_status, `Accumulator assessment mismatch at op #${expected.op_index}.`);
  assert(actual.not_assessed_reason === expected.not_assessed_reason, `Accumulator reason mismatch at op #${expected.op_index}.`);
  if (expected.assessment_status !== "assessed") return;
  for (const key of ["input_zero_point", "output_channel_axis", "output_channel_count", "accumulation_terms_per_channel", "assessed_channel_count", "stored_bias_channel_count", "int32_safe_channel_count", "int32_overflow_channel_count", "bias_half_range_exceedance_channel_count", "bias_half_range_guard_adjacent_channel_count", "bias_half_range_material_exceedance_channel_count", "exact_zero_kernel_channel_count", "exact_zero_bias_half_range_exceedance_channel_count", "exact_zero_bias_half_range_material_exceedance_channel_count", "maximum_required_signed_bits", "minimum_int32_headroom_bits", "weight_zero_point_mode", "bias_status"]) compareScalar(actual, expected, key, expected.op_index);
  for (const key of ["maximum_absolute_bias_decimal", "maximum_absolute_accumulator_decimal", "metadata_only_magnitude_bound_decimal"]) compareOptional(actual, expected, key, expected.op_index);
  for (const key of ["maximum_bias_int32_ratio", "maximum_int32_ratio", "metadata_only_int32_ratio", "exact_tightening_factor"]) compareOptionalNear(actual, expected, key, expected.op_index);
  for (const key of ["input_code_range", "overflow_channel_indices", "bias_half_range_exceedance_channel_indices", "bias_half_range_material_exceedance_channel_indices", "channel_accumulator_envelope_min_decimals", "channel_accumulator_envelope_max_decimals", "channel_post_bias_min_decimals", "channel_post_bias_max_decimals", "channel_required_signed_bits", "required_signed_bits_histogram"]) {
    assert(equalArray(actual[key], expected[key]), `Accumulator ${key} mismatch at op #${expected.op_index}.`);
  }
  assert(Array.isArray(actual.top_channels) && actual.top_channels.length === expected.top_channels.length, `Accumulator top-channel count mismatch at op #${expected.op_index}.`);
  actual.top_channels.forEach((item, index) => compareWitness(item, expected.top_channels[index], expected.op_index));
  compareWitness(actual.worst_channel, expected.worst_channel, expected.op_index);
  assert(/^[0-9a-f]{64}$/.test(actual.channel_ledger_sha256 || ""), `Accumulator ledger digest syntax mismatch at op #${expected.op_index}.`);
}

function compareWitness(actual, expected, opIndex) {
  assert(actual && expected, `Accumulator witness missing at op #${opIndex}.`);
  for (const key of ["channel_index", "positive_centered_weight_sum_decimal", "negative_centered_weight_sum_decimal", "bias_decimal", "dot_product_min_decimal", "dot_product_max_decimal", "post_bias_min_decimal", "post_bias_max_decimal", "accumulator_envelope_min_decimal", "accumulator_envelope_max_decimal", "maximum_absolute_accumulator_decimal", "required_signed_bits", "bias_exceeds_half_range", "bias_half_range_excess_decimal", "bias_half_range_classification", "exact_zero_centered_kernel", "fits_int32"]) {
    assert(actual[key] === expected[key], `Accumulator witness ${key} mismatch at op #${opIndex}.`);
  }
  assertNear(actual.int32_ratio, expected.int32_ratio, `Accumulator witness ratio mismatch at op #${opIndex}.`);
  assertNear(actual.bias_int32_ratio, expected.bias_int32_ratio, `Accumulator witness bias ratio mismatch at op #${opIndex}.`);
}

export function drawAccumulatorAtlasCanvas(canvas, row, mode = "bits", logicalWidth = null, logicalHeight = null) {
  if (!canvas || !row || row.assessment_status !== "assessed") return null;
  const width = logicalWidth || Math.max(260, Math.floor(canvas.parentElement?.clientWidth || canvas.clientWidth || 720));
  const height = logicalHeight || Math.max(160, Math.round(width * 0.56));
  const dpr = Math.max(1, Math.min(2, globalThis.devicePixelRatio || 1));
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#10171a";
  context.fillRect(0, 0, width, height);
  const count = row.assessed_channel_count;
  const columns = Math.min(64, Math.max(4, Math.ceil(Math.sqrt(count * 1.75))));
  const rows = Math.ceil(count / columns);
  const gap = count > 512 ? 1 : 2;
  const pad = 18;
  const cellWidth = (width - pad * 2 - gap * (columns - 1)) / columns;
  const cellHeight = (height - pad * 2 - gap * (rows - 1)) / rows;
  for (let channel = 0; channel < count; channel += 1) {
    const column = channel % columns;
    const rowIndex = Math.floor(channel / columns);
    const x = pad + column * (cellWidth + gap);
    const y = pad + rowIndex * (cellHeight + gap);
    context.fillStyle = accumulatorColor(row, channel, mode);
    context.fillRect(x, y, Math.max(1, cellWidth), Math.max(1, cellHeight));
    if (row.bias_half_range_material_exceedance_channel_indices?.includes(channel)) {
      context.strokeStyle = "#ff5a66";
      context.lineWidth = Math.max(1, Math.min(2, cellWidth / 4));
      context.strokeRect(x, y, Math.max(1, cellWidth), Math.max(1, cellHeight));
    } else if (row.bias_half_range_exceedance_channel_indices?.includes(channel)) {
      context.strokeStyle = "#e4b640";
      context.lineWidth = 1;
      context.strokeRect(x, y, Math.max(1, cellWidth), Math.max(1, cellHeight));
    }
  }
  context.strokeStyle = "rgba(255,255,255,.15)";
  context.strokeRect(pad - 1, pad - 1, width - pad * 2 + 2, height - pad * 2 + 2);
  canvas._accumulatorGeometry = { width, height, pad, gap, columns, rows, cellWidth, cellHeight, count };
  canvas.dataset.pixelSignature = `${mode}:${row.op_index}:${row.maximum_absolute_accumulator_decimal}`;
  return canvas._accumulatorGeometry;
}

function installCanvasInteraction(canvas, tooltip, row) {
  const show = (event) => {
    const geometry = canvas._accumulatorGeometry;
    if (!geometry) return;
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * geometry.width / rect.width - geometry.pad;
    const y = (event.clientY - rect.top) * geometry.height / rect.height - geometry.pad;
    const column = Math.floor(x / (geometry.cellWidth + geometry.gap));
    const rowIndex = Math.floor(y / (geometry.cellHeight + geometry.gap));
    const channel = rowIndex * geometry.columns + column;
    const inside = column >= 0 && column < geometry.columns && rowIndex >= 0 && rowIndex < geometry.rows
      && x % (geometry.cellWidth + geometry.gap) <= geometry.cellWidth
      && y % (geometry.cellHeight + geometry.gap) <= geometry.cellHeight
      && channel < geometry.count;
    if (!inside) { tooltip.hidden = true; return; }
    const minimum = row.channel_accumulator_envelope_min_decimals[channel];
    const maximum = row.channel_accumulator_envelope_max_decimals[channel];
    const ratio = Number(maxBigInt(absBigInt(BigInt(minimum)), absBigInt(BigInt(maximum)))) / Number(INT32_MAX);
    tooltip.textContent = `ch ${channel} · [${formatInteger(minimum)}, ${formatInteger(maximum)}] · ${row.channel_required_signed_bits[channel]} bits · ${formatPercent(ratio)} INT32`;
    tooltip.hidden = false;
    tooltip.style.left = `${Math.min(rect.width - 24, Math.max(12, event.clientX - rect.left))}px`;
    tooltip.style.top = `${Math.max(12, event.clientY - rect.top)}px`;
  };
  canvas.addEventListener("pointermove", show);
  canvas.addEventListener("pointerleave", () => { tooltip.hidden = true; });
}

function accumulatorColor(row, channel, mode) {
  const minimum = BigInt(row.channel_accumulator_envelope_min_decimals[channel]);
  const maximum = BigInt(row.channel_accumulator_envelope_max_decimals[channel]);
  const bits = Number(row.channel_required_signed_bits[channel]);
  if (bits > 32) return "#d63c52";
  if (mode === "polarity") {
    const negative = Math.log2(Number(absBigInt(minimum)) + 1);
    const positive = Math.log2(Number(absBigInt(maximum)) + 1);
    const delta = Math.max(-1, Math.min(1, (positive - negative) / Math.max(1, positive, negative)));
    if (delta > 0.08) return mixColor([96, 128, 134], [225, 91, 62], delta);
    if (delta < -0.08) return mixColor([96, 128, 134], [55, 135, 183], -delta);
    return "#668086";
  }
  const utilization = mode === "ratio"
    ? Math.max(0, Math.min(1, Math.log2(Number(maxBigInt(absBigInt(minimum), absBigInt(maximum))) + 1) / 31))
    : Math.max(0, Math.min(1, bits / 32));
  if (utilization < 0.68) return mixColor([31, 137, 119], [221, 177, 54], utilization / 0.68);
  return mixColor([221, 177, 54], [219, 75, 71], (utilization - 0.68) / 0.32);
}

function mixColor(left, right, amount) {
  const values = left.map((value, index) => Math.round(value + (right[index] - value) * amount));
  return `rgb(${values.join(",")})`;
}

function renderSummary(container, atlas) {
  if (!container) return;
  const tone = headroomTone(atlas);
  container.replaceChildren(
    summaryMetric("Assessed ops", `${formatNumber(atlas.assessed_op_count)} / ${formatNumber(atlas.candidate_op_count)}`),
    summaryMetric("Exact channels", formatNumber(atlas.assessed_channel_count)),
    summaryMetric(
      "Bias half-range material",
      formatNumber(atlas.bias_half_range_material_exceedance_channel_count),
      atlas.bias_half_range_material_exceedance_channel_count ? "warn" : "ok",
    ),
    summaryMetric("Maximum INT32 use", atlas.maximum_int32_ratio == null ? "N/A" : formatPercent(atlas.maximum_int32_ratio), tone),
    summaryMetric("Minimum headroom", atlas.minimum_int32_headroom_bits == null ? "N/A" : `${atlas.minimum_int32_headroom_bits} bits`, tone),
  );
}

function headroomTone(atlas) {
  if (!atlas?.assessed_op_count) return "muted";
  const ratio = Number(atlas.maximum_int32_ratio || 0);
  if (atlas.int32_overflow_channel_count || ratio >= 0.9) return "risk";
  if (ratio >= 0.5) return "warn";
  return "ok";
}

function accumulatorPressureNotice(row) {
  if (Number(row?.maximum_int32_ratio || 0) < 0.9 || !row.worst_channel) return null;
  const witness = row.worst_channel;
  const bias = absBigInt(BigInt(witness.bias_decimal));
  const dot = maxBigInt(absBigInt(BigInt(witness.dot_product_min_decimal)), absBigInt(BigInt(witness.dot_product_max_decimal)));
  const envelope = absBigInt(BigInt(witness.maximum_absolute_accumulator_decimal));
  const biasShare = envelope > 0n ? Number(bias) / Number(envelope) : 0;
  const dominance = dot > 0n ? Number(bias) / Number(dot) : Number.POSITIVE_INFINITY;
  return messageNode(
    `Critical exact headroom at #${row.op_index} channel ${witness.channel_index}: ${formatPercent(row.maximum_int32_ratio)} of INT32 and ${row.minimum_int32_headroom_bits} headroom bits. Stored bias |${formatInteger(witness.bias_decimal)}| contributes ${formatPercent(biasShare)} of the maximum envelope${Number.isFinite(dominance) ? ` and is ${dominance.toFixed(1)}x the maximum dot-product magnitude` : ""}. Bias is added on every inference; validate the deployed kernel's accumulator ordering and arithmetic mode.`,
    "risk",
  );
}

function biasPolicyNotice(row) {
  const material = Number(row?.bias_half_range_material_exceedance_channel_count || 0);
  if (!material) return null;
  return messageNode(
    `Stored-bias policy signal at #${row.op_index}: ${formatNumber(material)} channel(s) exceed the source half-range reference beyond the 129-code float32 guard-adjacent tolerance; maximum stored-bias utilization is ${formatPercent(row.maximum_bias_int32_ratio)}. This is artifact-derived pressure relative to a source-backed converter threshold, not proof that a converter pass ran or was skipped.`,
    "warn",
  );
}

function metricTable(row) {
  return definitionTable([
    ["Exact envelope maximum", formatInteger(row.maximum_absolute_accumulator_decimal)],
    ["INT32 utilization", formatPercent(row.maximum_int32_ratio)],
    ["Required width / headroom", `${row.maximum_required_signed_bits} bits / ${row.minimum_int32_headroom_bits} bits`],
    ["Full-domain INT32 exceedance", `${formatNumber(row.int32_overflow_channel_count)} / ${formatNumber(row.assessed_channel_count)}`],
    ["Stored bias maximum", `${formatInteger(row.maximum_absolute_bias_decimal)} (${formatPercent(row.maximum_bias_int32_ratio)})`],
    ["Bias half-range strict / adjacent / material", `${formatNumber(row.bias_half_range_exceedance_channel_count)} / ${formatNumber(row.bias_half_range_guard_adjacent_channel_count)} / ${formatNumber(row.bias_half_range_material_exceedance_channel_count)}`],
    ["Metadata-only bound", `${formatInteger(row.metadata_only_magnitude_bound_decimal)} (${formatPercent(row.metadata_only_int32_ratio)})`],
    ["Bound tightening", `${Number(row.exact_tightening_factor).toFixed(2)}×`],
  ], "Accumulator metrics");
}

function witnessTable(witness) {
  if (!witness) return messageNode("No channel witness emitted.");
  return definitionTable([
    ["Worst channel", String(witness.channel_index)],
    ["Centered weight sums", `+${formatInteger(witness.positive_centered_weight_sum_decimal)} / ${formatInteger(witness.negative_centered_weight_sum_decimal)}`],
    ["Dot range", `[${formatInteger(witness.dot_product_min_decimal)}, ${formatInteger(witness.dot_product_max_decimal)}]`],
    ["Stored bias", formatInteger(witness.bias_decimal)],
    ["Post-bias range", `[${formatInteger(witness.post_bias_min_decimal)}, ${formatInteger(witness.post_bias_max_decimal)}]`],
  ], "Worst-channel witness");
}

function requiredBitsHistogram(row) {
  const section = document.createElement("section");
  section.className = "accumulator-histogram-section";
  const title = document.createElement("h4");
  title.textContent = "Required signed-bit distribution";
  const chart = document.createElement("div");
  chart.className = "accumulator-bit-histogram";
  const populated = row.required_signed_bits_histogram.map((count, bits) => ({ count, bits })).filter((item) => item.count > 0);
  const maximum = Math.max(...populated.map((item) => item.count), 1);
  for (const item of populated) {
    const bar = document.createElement("div");
    bar.className = "accumulator-bit-bar";
    bar.style.setProperty("--bar-height", `${Math.max(4, item.count / maximum * 100)}%`);
    bar.dataset.tone = item.bits > 32 ? "risk" : item.bits >= 28 ? "warn" : "good";
    bar.title = `${item.bits} bits: ${formatNumber(item.count)} channel(s)`;
    const value = document.createElement("span");
    value.textContent = formatNumber(item.count);
    const label = document.createElement("b");
    label.textContent = String(item.bits);
    bar.append(value, label);
    chart.append(bar);
  }
  section.append(title, chart);
  return section;
}

function rankingTable(rows, selected, selectOp, jumpToGraphOp) {
  const section = document.createElement("section");
  section.className = "accumulator-ranking-section";
  const title = document.createElement("h4");
  title.textContent = "Operator headroom portfolio";
  const assessed = rows.filter((row) => row.assessment_status === "assessed")
    .sort((left, right) => right.maximum_int32_ratio - left.maximum_int32_ratio || left.op_index - right.op_index);
  const table = document.createElement("table");
  table.className = "accumulator-ranking-table";
  table.innerHTML = "<thead><tr><th>Rank</th><th>Operator</th><th>Channels × terms</th><th>INT32 use</th><th>Width</th><th>Headroom</th><th>Tightening</th><th></th></tr></thead>";
  const tbody = document.createElement("tbody");
  assessed.forEach((row, index) => {
    const tr = document.createElement("tr");
    tr.classList.toggle("selected", row.op_index === selected);
    tr.tabIndex = 0;
    tr.innerHTML = `<td>${index + 1}</td><td>#${row.op_index} ${escapeHtml(row.op_name)}</td><td>${formatNumber(row.assessed_channel_count)} × ${formatNumber(row.accumulation_terms_per_channel)}</td><td>${formatPercent(row.maximum_int32_ratio)}</td><td>${row.maximum_required_signed_bits} bits</td><td>${row.minimum_int32_headroom_bits} bits</td><td>${Number(row.exact_tightening_factor).toFixed(2)}×</td>`;
    const action = document.createElement("td");
    const graph = document.createElement("button");
    graph.type = "button";
    graph.className = "accumulator-graph-button";
    graph.textContent = "Graph";
    graph.title = `Inspect operator ${row.op_index} in Graph Explorer`;
    graph.addEventListener("click", (event) => { event.stopPropagation(); jumpToGraphOp?.(row.op_index); });
    action.append(graph);
    tr.append(action);
    const choose = () => selectOp(row.op_index);
    tr.addEventListener("click", choose);
    tr.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(); } });
    tbody.append(tr);
  });
  table.append(tbody);
  const wrap = document.createElement("div");
  wrap.className = "accumulator-ranking-wrap";
  wrap.append(table);
  section.append(title, wrap);
  return section;
}

function unassessedTable(rows) {
  const table = document.createElement("table");
  table.className = "accumulator-ranking-table";
  table.innerHTML = "<thead><tr><th>Operator</th><th>Reason</th></tr></thead>";
  const tbody = document.createElement("tbody");
  for (const row of rows.filter((item) => item.assessment_status !== "assessed")) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>#${row.op_index} ${escapeHtml(row.op_name)}</td><td>${escapeHtml(row.not_assessed_reason)}</td>`;
    tbody.append(tr);
  }
  table.append(tbody);
  return table;
}

function definitionTable(entries, label = "Artifact contract") {
  const section = document.createElement("section");
  section.className = "accumulator-definition";
  const title = document.createElement("h4");
  title.textContent = label;
  const dl = document.createElement("dl");
  for (const [term, value] of entries) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    if (value instanceof Node) dd.append(value);
    else dd.textContent = value;
    dl.append(dt, dd);
  }
  section.append(title, dl);
  return section;
}

function sourceLink(row) {
  const link = document.createElement("a");
  link.href = row.source_url;
  link.target = "_blank";
  link.rel = "noreferrer";
  link.textContent = row.source_file.split("/").at(-1) || "source";
  return link;
}

function summaryMetric(label, value, tone = "neutral") {
  const item = document.createElement("div");
  item.className = "accumulator-summary-metric";
  item.dataset.tone = tone;
  const strong = document.createElement("strong");
  strong.textContent = value;
  const span = document.createElement("span");
  span.textContent = label;
  item.append(strong, span);
  return item;
}

function messageNode(text, tone = "muted") {
  const node = document.createElement("p");
  node.className = "accumulator-message";
  node.dataset.tone = tone;
  node.textContent = text;
  return node;
}

function weightLayout(op, tensor) {
  const shape = (tensor.shape || []).map(Number);
  if (shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0)) throw new Error(`Weight tensor ${tensor.index} has a dynamic or non-positive shape.`);
  if (op.name === "CONV_2D" && shape.length === 4) return { channels: shape[0], terms: product(shape.slice(1)), axis: 0, rawIndex: (channel, term) => channel * product(shape.slice(1)) + term };
  if (op.name === "DEPTHWISE_CONV_2D" && shape.length === 4 && shape[0] === 1) return { channels: shape[3], terms: shape[1] * shape[2], axis: 3, rawIndex: (channel, term) => term * shape[3] + channel };
  if (op.name === "FULLY_CONNECTED" && shape.length === 2) return { channels: shape[0], terms: shape[1], axis: 0, rawIndex: (channel, term) => channel * shape[1] + term };
  if (op.name === "CONV_2D") throw new Error(`CONV_2D weight tensor ${tensor.index} is not OHWI rank 4.`);
  if (op.name === "DEPTHWISE_CONV_2D") throw new Error(`DEPTHWISE_CONV_2D weight tensor ${tensor.index} is not [1,H,W,O].`);
  throw new Error(`FULLY_CONNECTED weight tensor ${tensor.index} is not rank 2.`);
}

function decodeBias(op, tensors, bytes, channels) {
  const index = Number(op.inputs?.[2]);
  if (!Number.isInteger(index) || index < 0) return { values: Array(channels).fill(0), status: "absent_zero_bias" };
  const tensor = tensors.get(index);
  if (!tensor) throw new Error(`Bias tensor ${index} is unavailable.`);
  if (tensor.dtype !== "INT32") throw new Error(`Bias tensor ${tensor.index} uses ${tensor.dtype}; INT32 is required for integer accumulation.`);
  const raw = tensorBytes(bytes, tensor, `Bias tensor ${tensor.index} constant bytes are unavailable.`);
  if (raw.length !== channels * 4) throw new Error(`Bias tensor ${tensor.index} exposes ${raw.length} byte(s); ${channels} output channels require ${channels * 4} INT32 byte(s).`);
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  return { values: Array.from({ length: channels }, (_, channel) => view.getInt32(channel * 4, true)), status: "stored_int32_bias" };
}

function expandedZeroPoints(tensor, channels, qmin, qmax) {
  const values = (tensor.zero_point_sample || []).map(Number);
  if (![1, channels].includes(values.length)) throw new Error(`Weight tensor ${tensor.index} zero-point cardinality is not 1 or output-channel count ${channels}.`);
  if (values.some((value) => value < qmin || value > qmax)) throw new Error(`Weight tensor ${tensor.index} contains a zero-point outside [${qmin}, ${qmax}].`);
  return values.length === 1 ? Array(channels).fill(values[0]) : values;
}

function tensorBytes(bytes, tensor, message) {
  const offset = Number(tensor.buffer_data_offset);
  const length = Number(tensor.buffer_data_length);
  if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || length <= 0 || offset < 0 || offset + length > bytes.length) throw new Error(message);
  return bytes.subarray(offset, offset + length);
}

function codeRange(dtype, message) {
  if (dtype === "INT8") return [-128, 127];
  if (dtype === "UINT8") return [0, 255];
  throw new Error(message);
}

function rawCode(byte, dtype) { return dtype === "INT8" ? (byte << 24) >> 24 : byte; }
function requiredTensor(tensors, index, message) { const tensor = tensors.get(Number(index)); if (!tensor) throw new Error(message); return tensor; }
function requiredSignedBits(minimum, maximum) { for (let bits = 1; bits < 128; bits += 1) { const magnitude = 1n << BigInt(bits - 1); if (minimum >= -magnitude && maximum <= magnitude - 1n) return bits; } return 128; }
function asBytes(value) { if (value instanceof Uint8Array) return value; if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength); if (value instanceof ArrayBuffer) return new Uint8Array(value); throw new Error("Accumulator verification requires model bytes."); }
function absBigInt(value) { return value < 0n ? -value : value; }
function classifyBiasHalfRangeExcess(excess) { if (excess <= 0n) return "within_source_threshold"; if (excess <= BIAS_HALF_RANGE_FLOAT32_TOLERANCE_CODES) return "float32_guard_adjacent"; return "material_exceedance"; }
function minBigInt(...values) { return values.reduce((left, right) => left < right ? left : right); }
function maxBigInt(...values) { return values.reduce((left, right) => left > right ? left : right); }
function compareBigIntDesc(left, right) { const a = BigInt(left); const b = BigInt(right); return a === b ? 0 : a > b ? -1 : 1; }
function product(values) { return values.reduce((total, value) => total * value, 1); }
function sum(values) { return values.reduce((total, value) => total + Number(value || 0), 0); }
function equalArray(left, right) { return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index]); }
function compareScalar(actual, expected, key, opIndex = "summary") { assert(actual[key] === expected[key], `Accumulator ${key} mismatch at ${opIndex}.`); }
function compareOptional(actual, expected, key, opIndex = "summary") { const left = actual[key] ?? null; const right = expected[key] ?? null; assert(left === right, `Accumulator ${key} mismatch at ${opIndex}.`); }
function compareOptionalNear(actual, expected, key, opIndex = "summary") { const left = actual[key]; const right = expected[key]; if (left == null || right == null) { assert(left == null && right == null, `Accumulator ${key} missing mismatch at ${opIndex}.`); return; } assertNear(left, right, `Accumulator ${key} mismatch at ${opIndex}.`); }
function assertNear(actual, expected, message) { const tolerance = Math.max(1e-12, Math.abs(Number(expected)) * 1e-12); assert(Number.isFinite(Number(actual)) && Math.abs(Number(actual) - Number(expected)) <= tolerance, message); }
function assert(condition, message) { if (!condition) throw new Error(message); }
function formatNumber(value) { return Number(value || 0).toLocaleString("en-US"); }
function formatPercent(value) { return `${(Number(value || 0) * 100).toFixed(Number(value) < 0.001 ? 4 : 2)}%`; }
function formatInteger(value) { try { return BigInt(value).toLocaleString("en-US"); } catch { return String(value ?? "N/A"); } }
function shapeText(shape) { return `[${(shape || []).join(" × ")}]`; }
function escapeHtml(value) { return String(value ?? "").replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[character]); }
