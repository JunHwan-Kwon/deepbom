import { argmax, clampInteger, formatDrift } from "./format.js";
import {
  driftSeverity,
  statusForCosineDistance,
  statusForMaxDrift,
  statusForRmsDrift,
  statusForTop1Flip,
} from "./status.js";

function defaultOutputDriftProfile() {
  return {
    dtype: "FLOAT32",
    unit: "raw output value",
    rmsOk: 0.0005,
    rmsWarn: 0.005,
    maxOk: 0.001,
    maxWarn: 0.01,
    // Relative thresholds kick in when baseline magnitude > 1 (e.g. unnormalized logits)
    relRmsOk: 0.005,
    relRmsWarn: 0.05,
    relMaxOk: 0.01,
    relMaxWarn: 0.10,
  };
}

export function outputDriftProfileForAnalysis(analysis) {
  const dtype = String(analysis?.outputs?.[0]?.dtype || analysis?.inputs?.[0]?.dtype || "FLOAT32").toUpperCase();
  const integer = dtype.includes("INT8") || dtype.includes("UINT8");
  // Integer models: thresholds in LSB units; no relative mode (values already dimensionless)
  return integer
    ? {
        dtype,
        unit: "raw output LSB",
        rmsOk: 0.5,
        rmsWarn: 2,
        maxOk: 1,
        maxWarn: 4,
      }
    : defaultOutputDriftProfile();
}

export function summarizeProbeResult(probe) {
  if (!probe) return null;
  return {
    backend: probe.backend || null,
    compile_ms: roundMetric(probe.compileMs),
    run_ms: roundMetric(probe.runMs),
    output_count: Number(probe.outputCount || probe.outputs?.length || 0),
    input_perturbation: probe.inputPerturbation || null,
  };
}

export function summarizeProjectedHessians(hessians = []) {
  const values = hessians
    .map((hessian) => hessian?.lambdaMax)
    .filter((value) => value != null)
    .map(Number)
    .filter(Number.isFinite);
  if (!values.length) {
    return { assessedCount: 0, lambdaMean: null, lambdaStd: null, directionalLambdaMaxCv: null };
  }
  const lambdaMean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const lambdaStd = Math.sqrt(values.reduce((sum, value) => sum + (value - lambdaMean) ** 2, 0) / values.length);
  const directionalLambdaMaxCv = Math.abs(lambdaMean) > 1e-12
    ? lambdaStd / Math.abs(lambdaMean)
    : null;
  return { assessedCount: values.length, lambdaMean, lambdaStd, directionalLambdaMaxCv };
}

export function outputDriftGeometryProtocol({ gridSize, radius } = {}) {
  return {
    evidenceClass: "MEASURED_SYNTHETIC_PROXY",
    scalarFunctional: "F(alpha,beta)=RMS(y(W+alpha*d1+beta*d2;x_syn)-y(W;x_syn)) over concatenated final outputs",
    inputBasis: "Deterministic sinusoidal HWC tensor; batch 1; spatial dimensions capped at 32x32; no dataset, labels, or training objective",
    perturbationNormalization: "Independent Gaussian directions normalized per stored filter to the dequantized filter L2 norm",
    grid: `${gridSize}x${gridSize} over [-${radius},+${radius}] on both direction axes`,
    stencil: "Centered second differences at the grid origin, including the four-corner mixed partial",
    directionSeeds: "projection k uses seeds (2000+11*k)|1 and (7000+17*k)|1",
    includedConstants: "Primary-subgraph dense FLOAT32 and INT8 constant tensors accepted by the research weight reader",
    quantizedWeightPolicy: "INT8 values are dequantized with tensor/per-channel scale and zero point; this f64 ensemble perturbs in float domain without requantization",
    interpretationBoundary: "Descriptive local output-drift geometry only; not training loss, a full-space Hessian, a 3D reconstruction, accuracy, robustness, or release-readiness evidence",
  };
}

export function outputDriftProjectionCopy({ seeds, gridSize, radius } = {}) {
  return {
    title: `Output-drift perturbation surface — ${seeds} seeds × ${gridSize}×${gridSize} grid (r=${radius}) · INT8 vs f64`,
    method: "Scalar functional: RMS change in the model output relative to the unperturbed artifact under a deterministic synthetic input, not training loss. Directions are filter-normalized Gaussian draws (Li et al. 2018 normalization only). INT8 uses requantized LiteRT.js inference; f64 uses dequantized weights in the WASM synthetic forward path. Projected curvature uses the centered finite-difference stencil; bands and ± values use sample SEM (s/√n).",
  };
}

export function outputDriftEnsembleCopy({ numProjections, assessedCount } = {}) {
  return {
    title: "Multi-Projection Output-Drift Geometry",
    method: `${numProjections} independently seeded 2D slices summarize local variation of synthetic-output RMS drift; projected curvature was assessed for ${assessedCount}. Each slice uses a distinct filter-normalized direction pair. Directional λ_max CV is the raw population coefficient of variation across finite slices. This is not training loss, a full-space Hessian, or a reconstruction of 3D weight geometry.`,
  };
}

export function summarizeOutputDriftProjectionEnsemble(raw, { numProjections, gridSize, radius } = {}) {
  const axes = Array.from(raw?.axes || []);
  const flatGrids = Array.from(raw?.grids || []);
  const grids = flatGrids.map((flat) => Array.from({ length: gridSize }, (_, row) => (
    Array.from(flat.slice(row * gridSize, row * gridSize + gridSize))
  )));
  const dmGrids = grids.map((grid) => subtractCenter(grid, gridSize));
  const summarizeCell = (row, column, spread) => {
    const values = dmGrids.map((grid) => grid[row][column]).filter(Number.isFinite);
    if (!values.length) return Number.NaN;
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    if (!spread) return mean;
    if (values.length < 2) return 0;
    return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
  };
  const matrix = (spread) => Array.from({ length: gridSize }, (_, row) => (
    Array.from({ length: gridSize }, (_, column) => summarizeCell(row, column, spread))
  ));
  const hessianAll = dmGrids.map((grid) => computeHessian2D(grid, axes, gridSize));
  const hessians = hessianAll.filter(Boolean);
  const lambdas = hessianAll.map((hessian) => hessian?.lambdaMax ?? null);
  const summary = summarizeProjectedHessians(hessians);
  return {
    dmGrids,
    meanGrid: matrix(false),
    varGrid: matrix(true),
    axes,
    G: gridSize,
    hessians,
    lambdas,
    lambdaMean: summary.lambdaMean,
    lambdaStd: summary.lambdaStd,
    directionalLambdaMaxCv: summary.directionalLambdaMaxCv,
    hessianAssessedCount: summary.assessedCount,
    numProjections,
    protocol: outputDriftGeometryProtocol({ gridSize, radius }),
  };
}

export function applyLandscapePatch(modelBytes, d1, d2, alpha, beta, metas) {
  assertTypedBytes(modelBytes, "landscape model bytes");
  assertFinite(alpha, "landscape alpha");
  assertFinite(beta, "landscape beta");
  if (!Array.isArray(metas)) throw new TypeError("Landscape weight metadata must be an array.");
  const totalParams = metas.reduce((sum, meta) => sum + integerField(meta?.elem_count, "landscape element count"), 0);
  if (!d1 || !d2 || d1.length !== totalParams || d2.length !== totalParams) {
    throw new RangeError(`Landscape directions must each contain exactly ${totalParams} values.`);
  }

  const buffer = modelBytes.buffer.slice(modelBytes.byteOffset, modelBytes.byteOffset + modelBytes.byteLength);
  const result = new Uint8Array(buffer);
  const view = new DataView(buffer);
  let directionOffset = 0;
  for (const meta of metas) {
    const byteOffset = integerField(meta?.buf_offset, "landscape weight byte offset");
    const elementCount = integerField(meta?.elem_count, "landscape element count");
    const dtype = String(meta?.dtype || "");
    const bytesPerElement = dtype === "FLOAT32" ? 4 : dtype === "INT8" ? 1 : 0;
    if (!bytesPerElement) throw new TypeError(`Unsupported landscape weight dtype ${dtype || "missing"}.`);
    if (byteOffset + elementCount * bytesPerElement > result.byteLength) {
      throw new RangeError(`Landscape weight span ${byteOffset}+${elementCount * bytesPerElement} exceeds ${result.byteLength} model bytes.`);
    }

    if (dtype === "FLOAT32") {
      for (let index = 0; index < elementCount; index += 1) {
        const directionIndex = directionOffset + index;
        const delta = directionDelta(d1, d2, directionIndex, alpha, beta);
        const valueOffset = byteOffset + index * 4;
        view.setFloat32(valueOffset, view.getFloat32(valueOffset, true) + delta, true);
      }
    } else {
      const outputChannels = positiveInteger(meta?.oc, "landscape output channel count");
      const filterSize = positiveInteger(meta?.filter_size, "landscape filter size");
      const scales = Array.from(meta?.scales || [], Number);
      const zeroPoints = Array.from(meta?.zps || [], Number);
      if (!scales.length) throw new RangeError("INT8 landscape metadata has no quantization scale.");
      const perChannel = scales.length > 1;
      for (let index = 0; index < elementCount; index += 1) {
        const channel = Math.min(Math.floor(index / filterSize), outputChannels - 1);
        const scale = scales[perChannel ? channel : 0];
        const zeroPoint = zeroPoints[perChannel ? channel : 0] ?? zeroPoints[0] ?? 0;
        if (!Number.isFinite(scale) || scale <= 0) throw new RangeError(`Invalid INT8 landscape scale at channel ${channel}.`);
        if (!Number.isInteger(zeroPoint)) throw new RangeError(`Invalid INT8 landscape zero point at channel ${channel}.`);
        const rawByte = result[byteOffset + index];
        const rawValue = rawByte > 127 ? rawByte - 256 : rawByte;
        const realValue = (rawValue - zeroPoint) * scale;
        const delta = directionDelta(d1, d2, directionOffset + index, alpha, beta);
        const quantized = roundTiesAwayFromZero((realValue + delta) / scale + zeroPoint);
        result[byteOffset + index] = Math.max(-128, Math.min(127, quantized)) & 0xff;
      }
    }
    directionOffset += elementCount;
  }
  return result;
}

export function linspaceArr(lower, upper, count) {
  assertFinite(lower, "linspace lower bound");
  assertFinite(upper, "linspace upper bound");
  const size = positiveInteger(count, "linspace count");
  if (size === 1) return [lower];
  return Array.from({ length: size }, (_, index) => lower + index * (upper - lower) / (size - 1));
}

export function aggregateGrids(grids, size) {
  const gridSize = positiveInteger(size, "landscape grid size");
  if (!Array.isArray(grids)) throw new TypeError("Landscape grids must be an array.");
  const mean = Array.from({ length: gridSize }, () => new Float64Array(gridSize).fill(NaN));
  const sem = Array.from({ length: gridSize }, () => new Float64Array(gridSize));
  for (let row = 0; row < gridSize; row += 1) {
    for (let column = 0; column < gridSize; column += 1) {
      const values = grids.map((grid) => Number(grid?.[row]?.[column])).filter(Number.isFinite);
      if (!values.length) continue;
      mean[row][column] = arithmeticMean(values);
      sem[row][column] = sampleSem(values);
    }
  }
  const center = Math.floor(gridSize / 2);
  const centerLoss = Number.isFinite(mean[center][center]) ? mean[center][center] : 0;
  const dmean = mean.map((row) => Array.from(row, (value) => Number.isFinite(value) ? value - centerLoss : NaN));
  const finiteDeltas = dmean.flat().filter(Number.isFinite);
  const nonzeroSems = sem.flatMap((row) => Array.from(row)).filter((value) => value > 0);
  return {
    mean: mean.map((row) => Array.from(row)),
    sem: sem.map((row) => Array.from(row)),
    dmean,
    centerLoss,
    maxDmean: finiteDeltas.length ? Math.max(...finiteDeltas) : 0,
    meanSem: nonzeroSems.length ? arithmeticMean(nonzeroSems) : 0,
  };
}

export function subtractCenter(grid, size) {
  const gridSize = positiveInteger(size, "landscape grid size");
  const center = Math.floor(gridSize / 2);
  const centerValue = Number(grid?.[center]?.[center]);
  const baseline = Number.isFinite(centerValue) ? centerValue : 0;
  return Array.from({ length: gridSize }, (_, row) => Array.from({ length: gridSize }, (_, column) => {
    const value = Number(grid?.[row]?.[column]);
    return Number.isFinite(value) ? value - baseline : NaN;
  }));
}

export function computeRadialProfileSEM(perSeedDmeans, axes, size, binCount = 9) {
  const gridSize = positiveInteger(size, "radial grid size");
  const bins = positiveInteger(binCount, "radial bin count");
  if (!Array.isArray(perSeedDmeans) || !perSeedDmeans.length) return { rc: [], mu: [], sem: [] };
  validateAxes(axes, gridSize);
  const profiles = perSeedDmeans.map((grid) => radialMeanOneSeed(grid, axes, gridSize, bins));
  const rc = profiles[0].rc;
  const mu = rc.map((_, index) => arithmeticMean(profiles.map((profile) => profile.mu[index] ?? 0)));
  const sem = rc.map((_, index) => sampleSem(profiles.map((profile) => profile.mu[index] ?? 0)));
  return { rc, mu, sem };
}

export function computeHessian2D(grid, axes, size) {
  const gridSize = positiveInteger(size, "Hessian grid size");
  if (gridSize < 3 || !Array.isArray(axes) || axes.length < gridSize) return null;
  validateAxes(axes, gridSize);
  const center = Math.floor(gridSize / 2);
  const step = Number(axes[center + 1]) - Number(axes[center]);
  const previousStep = Number(axes[center]) - Number(axes[center - 1]);
  const tolerance = Math.max(Math.abs(step), Math.abs(previousStep), 1) * 1e-12;
  if (!Number.isFinite(step) || Math.abs(step) < 1e-12 || Math.abs(step - previousStep) > tolerance) return null;
  const value = (row, column) => {
    const item = Number(grid?.[row]?.[column]);
    return Number.isFinite(item) ? item : null;
  };
  const centerValue = value(center, center);
  const positiveA = value(center, center + 1);
  const negativeA = value(center, center - 1);
  const positiveB = value(center + 1, center);
  const negativeB = value(center - 1, center);
  const positivePositive = value(center + 1, center + 1);
  const positiveNegative = value(center + 1, center - 1);
  const negativePositive = value(center - 1, center + 1);
  const negativeNegative = value(center - 1, center - 1);
  const stencil = [centerValue, positiveA, negativeA, positiveB, negativeB, positivePositive, positiveNegative, negativePositive, negativeNegative];
  if (stencil.some((item) => item == null)) return null;
  const stepSquared = step * step;
  const Haa = (positiveA - 2 * centerValue + negativeA) / stepSquared;
  const Hbb = (positiveB - 2 * centerValue + negativeB) / stepSquared;
  const Hab = (positivePositive - positiveNegative - negativePositive + negativeNegative) / (4 * stepSquared);
  const trace = Haa + Hbb;
  const discriminant = Math.sqrt(Math.max(0, ((Haa - Hbb) / 2) ** 2 + Hab ** 2));
  return { Haa, Hbb, Hab, trace, lambdaMax: trace / 2 + discriminant };
}

export function aggregateHessian(hessians) {
  if (!Array.isArray(hessians) || !hessians.length) return null;
  const lambdaValues = hessians.map((item) => Number(item?.lambdaMax));
  const traceValues = hessians.map((item) => Number(item?.trace));
  if (lambdaValues.some((value) => !Number.isFinite(value)) || traceValues.some((value) => !Number.isFinite(value))) {
    throw new TypeError("Hessian aggregation requires finite lambdaMax and trace values.");
  }
  return {
    lambdaMax_mean: arithmeticMean(lambdaValues),
    lambdaMax_sem: sampleSem(lambdaValues),
    trace_mean: arithmeticMean(traceValues),
    trace_sem: sampleSem(traceValues),
    per_seed: hessians,
  };
}

export function requantRatio(int8Dmean, f64Dmean, size) {
  const gridSize = positiveInteger(size, "requantization-ratio grid size");
  let crossProduct = 0;
  let floatEnergy = 0;
  let count = 0;
  for (let row = 0; row < gridSize; row += 1) {
    for (let column = 0; column < gridSize; column += 1) {
      const quantized = Number(int8Dmean?.[row]?.[column]);
      const floating = Number(f64Dmean?.[row]?.[column]);
      if (Number.isFinite(quantized) && Number.isFinite(floating) && Math.abs(floating) > 1e-10) {
        crossProduct += floating * quantized;
        floatEnergy += floating * floating;
        count += 1;
      }
    }
  }
  return count && floatEnergy > 1e-20 ? crossProduct / floatEnergy : null;
}

function radialMeanOneSeed(grid, axes, size, binCount) {
  const axisRadius = Math.max(...Array.from(axes, (value) => Math.abs(Number(value))).filter(Number.isFinite), 0);
  const radius = Math.sqrt(2) * (axisRadius || 0.4);
  const boundaries = linspaceArr(0, radius, binCount + 1);
  const rc = boundaries.slice(0, -1).map((lower, index) => (lower + boundaries[index + 1]) / 2);
  const sums = new Float64Array(binCount);
  const counts = new Uint32Array(binCount);
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      const radial = Math.hypot(Number(axes[column]), Number(axes[row]));
      const value = Number(grid?.[row]?.[column]);
      if (!Number.isFinite(value)) continue;
      let bin = Math.floor(radial / radius * binCount);
      if (bin === binCount && Math.abs(radial - radius) <= radius * 1e-12) bin = binCount - 1;
      if (bin >= 0 && bin < binCount) {
        sums[bin] += value;
        counts[bin] += 1;
      }
    }
  }
  return { rc, mu: Array.from(sums, (value, index) => counts[index] ? value / counts[index] : 0) };
}

function sampleSem(values) {
  if (values.length < 2) return 0;
  const mean = arithmeticMean(values);
  const sampleVariance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
  return Math.sqrt(sampleVariance / values.length);
}

function arithmeticMean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function directionDelta(d1, d2, index, alpha, beta) {
  const left = Number(d1[index]);
  const right = Number(d2[index]);
  if (!Number.isFinite(left) || !Number.isFinite(right)) throw new TypeError(`Landscape direction ${index} is not finite.`);
  return alpha * left + beta * right;
}

function roundTiesAwayFromZero(value) {
  return value < 0 ? -Math.floor(-value + 0.5) : Math.floor(value + 0.5);
}

function validateAxes(axes, size) {
  if (!Array.isArray(axes) && !ArrayBuffer.isView(axes)) throw new TypeError("Landscape axes must be an array.");
  if (axes.length < size || Array.from(axes.slice(0, size), Number).some((value) => !Number.isFinite(value))) {
    throw new RangeError(`Landscape axes must contain ${size} finite values.`);
  }
}

function assertTypedBytes(value, label) {
  if (!ArrayBuffer.isView(value) || typeof value.byteOffset !== "number" || typeof value.byteLength !== "number") {
    throw new TypeError(`${label} must be a typed-array view.`);
  }
}

function integerField(value, label) {
  const result = Number(value);
  if (!Number.isSafeInteger(result) || result < 0) throw new RangeError(`${label} must be a non-negative safe integer.`);
  return result;
}

function positiveInteger(value, label) {
  const result = integerField(value, label);
  if (!result) throw new RangeError(`${label} must be positive.`);
  return result;
}

function assertFinite(value, label) {
  if (!Number.isFinite(Number(value))) throw new TypeError(`${label} must be finite.`);
}

export function sanitizeDrift(drift, profile = defaultOutputDriftProfile()) {
  if (!drift) return null;
  const leftRms = Number(drift.leftRms || 0);
  const severity = driftSeverity(drift, profile);
  return {
    count: Number(drift.count || 0),
    mean_abs: Number(drift.meanAbs || 0),
    rms: Number(drift.rms || 0),
    max_abs: Number(drift.maxAbs || 0),
    cosine_distance: Number(drift.cosineDistance || 0),
    top1_flip: Boolean(drift.top1Flip),
    left_rms: leftRms,
    rms_severity: statusForRmsDrift(drift.rms || 0, profile, leftRms).label,
    max_abs_severity: statusForMaxDrift(drift.maxAbs || 0, profile, leftRms).label,
    cosine_severity: statusForCosineDistance(drift.cosineDistance || 0).label,
    top1_severity: statusForTop1Flip(Boolean(drift.top1Flip)).label,
    severity: severity.label,
    severity_criteria: severity.criteria,
  };
}

export function sanitizeRuntimeAttempt(item, profile = defaultOutputDriftProfile(), referenceAttempt = null, analysis = null) {
  const interpretation = runtimeAttemptInterpretation(item, referenceAttempt, analysis);
  return {
    backend: item.backend,
    ok: Boolean(item.ok),
    error: item.ok ? null : item.error,
    compile_ms: item.ok ? roundMetric(item.result?.compileMs) : null,
    run_ms: item.ok ? roundMetric(item.result?.runMs) : null,
    run_ratio_to_reference: interpretation.run_ratio_to_reference,
    output_count: item.ok ? Number(item.result?.outputCount || item.result?.outputs?.length || 0) : 0,
    drift: item.drift ? sanitizeDrift(item.drift, profile) : null,
    interpretation: interpretation.note,
    interpretation_severity: interpretation.severity,
  };
}

export function runtimeAttemptInterpretation(item, referenceAttempt = null, analysis = null) {
  if (!item?.ok) {
    return {
      severity: "failed",
      run_ratio_to_reference: null,
      note: item?.error || "Backend did not return outputs.",
    };
  }
  const referenceRun = Number(referenceAttempt?.result?.runMs || 0);
  const runMs = Number(item.result?.runMs || 0);
  const ratio = referenceRun > 0 ? runMs / referenceRun : null;
  const backend = String(item.backend || "").toLowerCase();
  const quantized = modelHas8BitRuntimeContract(analysis);
  if (backend === "webgpu" && quantized && ratio && ratio >= 3) {
    return {
      severity: "warn",
      run_ratio_to_reference: Number(ratio.toFixed(3)),
      note: "WebGPU completed but is much slower than the reference backend on an 8-bit model. Browser WebGPU/LiteRT paths may dequantize, emulate, copy through GPU buffers, or miss INT8 kernels; treat this as backend-path mismatch until measured on the deployment target.",
    };
  }
  if (backend === "webgpu" && quantized) {
    return {
      severity: "info",
      run_ratio_to_reference: ratio ? Number(ratio.toFixed(3)) : null,
      note: "8-bit TFLite on browser WebGPU can differ from native NPU/GPU delegates; validate with target runtime before interpreting latency.",
    };
  }
  return {
    severity: "info",
    run_ratio_to_reference: ratio ? Number(ratio.toFixed(3)) : null,
    note: "Backend completed locally; single-run timing is diagnostic only.",
  };
}

export function modelHas8BitRuntimeContract(analysis) {
  const tensors = [
    ...(analysis?.inputs || []),
    ...(analysis?.outputs || []),
    ...(analysis?.tensors || []),
  ];
  return tensors.some((tensor) => /^(U?INT8)$/i.test(String(tensor?.dtype || ""))) ||
    ["full_integer", "integer_internal_float_io", "mixed_quantization", "dynamic_range_or_weight_only"].includes(analysis?.quantization_status?.classification || "");
}

export function sanitizeCurvature(curvature) {
  if (!curvature) return null;
  return {
    count: Number(curvature.count || 0),
    raw_rms: Number(curvature.rawRms || 0),
    max_abs: Number(curvature.maxAbs || 0),
    normalized_rms: Number(curvature.normalizedRms || 0),
    local_lipschitz: Number(curvature.localLipschitz || 0),
    metric_basis: "local deploy-function finite difference around the current synthetic/prepared input; relative across same model, dtype, input scale, and backend only",
    normalized_rms_interpretation: curvatureMetricInterpretation(curvature.normalizedRms, "normalized_rms"),
    local_lipschitz_interpretation: curvatureMetricInterpretation(curvature.localLipschitz, "local_lipschitz"),
  };
}

export function curvatureMetricInterpretation(value, metric) {
  const number = Number(value || 0);
  const label = metric === "local_lipschitz" ? "Local Lipschitz" : "Normalized curvature RMS";
  return `${label}=${formatDrift(number)} is a relative deploy-domain sensitivity signal, not a universal pass/fail threshold. Compare it against previous builds of the same model, same input contract, same perturbation scale, and same backend; lower is calmer under those controlled conditions.`;
}

export function sanitizeTimingVariance(variance) {
  return variance
    ? {
        label: variance.label,
        detail: variance.detail,
        status: variance.status?.label || "",
        criteria: variance.status?.criteria || "",
      }
    : null;
}

export function timingContextNote() {
  return "Single-run module timings are diagnostic only. Compile time can differ across modules because LiteRT/browser runtime state, WASM JIT, and weight packing may be cold or warm depending on execution order; use Runtime Benchmark p50/p90/p95/p99 for latency claims.";
}

export function perturbationProtocolStatus(inputExecuted, weightExecuted, layerExecuted, haarExecuted) {
  return [
    { id: "input_perturbation", status: inputExecuted ? "executed" : "not_run" },
    { id: "output_drift", status: inputExecuted ? "executed" : "not_run" },
    { id: "weight_perturbation", status: weightExecuted ? "executed" : "not_run" },
    { id: "layer_wise_robustness", status: layerExecuted ? "executed" : "not_run" },
    { id: "haar_pattern_sweep", status: haarExecuted ? "executed" : "not_run" },
  ];
}

export function runtimeBasinProtocolStatus(preprocessingExecuted = false) {
  return [
    { id: "backend_availability", status: "executed" },
    { id: "backend_output_drift", status: "executed" },
    { id: "preprocessing_drift", status: preprocessingExecuted ? "executed" : "not_available" },
  ];
}

export function selectWeightPerturbationCandidates(analysis, limit = 12) {
  const tensors = tensorMapByIndex(analysis);
  const inputSet = new Set((analysis?.input_tensor_indices || []).map(Number));
  return (analysis?.ops || [])
    .map((op) => {
      const tensorIds = weightTensorIdsForOp(op, tensors, inputSet);
      return {
        op,
        tensorIds,
        score: Number(op.macs || 0) + Number(op.weight_bytes || 0) * 20 + Number(op.estimated_bytes || 0),
      };
    })
    .filter((item) => item.tensorIds.length)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

export function tensorMapByIndex(analysis) {
  const map = new Map();
  for (const tensor of analysis?.tensors || []) map.set(Number(tensor.index), tensor);
  return map;
}

export function weightTensorIdsForOp(op, tensors, inputSet) {
  const ids = [];
  const inputs = Array.isArray(op.inputs) ? op.inputs.map(Number) : [];
  for (const tensorId of inputs.slice(1)) {
    const tensor = tensors.get(tensorId);
    if (!tensor || inputSet.has(tensorId)) continue;
    if (!Number(tensor.buffer_data_length || 0)) continue;
    if (!["INT8", "UINT8", "FLOAT32"].includes(String(tensor.dtype || "").toUpperCase())) continue;
    ids.push(tensorId);
  }
  return ids.slice(0, 1);
}

export function perturbModelWeightBytes(sourceBytes, analysis, candidates, options = {}) {
  const bytes = new Uint8Array(sourceBytes);
  const tensors = tensorMapByIndex(analysis);
  const seen = new Set();
  let touchedTensors = 0;
  let touchedValues = 0;
  for (const candidate of candidates) {
    for (const tensorId of candidate.tensorIds || []) {
      const tensor = tensors.get(Number(tensorId));
      if (!tensor || seen.has(Number(tensorId))) continue;
      seen.add(Number(tensorId));
      const touched = perturbTensorBuffer(bytes, tensor, options);
      if (touched) {
        touchedTensors += 1;
        touchedValues += touched;
      }
    }
  }
  return { bytes, touchedTensors, touchedValues };
}

export function perturbTensorBuffer(bytes, tensor, options = {}) {
  const start = Number(tensor.buffer_data_offset || 0);
  const length = Number(tensor.buffer_data_length || 0);
  if (!start || !length || start + length > bytes.length) return 0;
  const dtype = String(tensor.dtype || "").toUpperCase();
  const direction = Number(options.direction || 1);
  const amplitude = Math.max(1, Number(options.amplitude || 1));
  if (dtype === "INT8" || dtype === "UINT8") {
    const step = Math.max(1, Math.floor(length / 512));
    let touched = 0;
    for (let offset = start; offset < start + length; offset += step) {
      if (dtype === "UINT8") {
        bytes[offset] = clampInteger(bytes[offset] + direction * amplitude, 0, 255);
      } else {
        const signed = bytes[offset] > 127 ? bytes[offset] - 256 : bytes[offset];
        const next = clampInteger(signed + direction * amplitude, -128, 127);
        bytes[offset] = next < 0 ? next + 256 : next;
      }
      touched += 1;
    }
    return touched;
  }
  if (dtype === "FLOAT32") {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const values = Math.floor(length / 4);
    const step = Math.max(1, Math.floor(values / 512));
    let touched = 0;
    for (let index = 0; index < values; index += step) {
      const pos = start + index * 4;
      const value = view.getFloat32(pos, true);
      if (!Number.isFinite(value)) continue;
      const epsilon = Math.max(1e-5, Math.abs(value) * 1e-3);
      view.setFloat32(pos, value + direction * epsilon, true);
      touched += 1;
    }
    return touched;
  }
  return 0;
}

export function robustnessScoreFromDrift(drift, profile) {
  const severity = driftSeverity(drift, profile);
  const penalty = severity.label === "risk" ? 45 : severity.label === "warn" ? 22 : 5;
  const magnitudePenalty = Math.min(35, Number(drift?.maxAbs || 0) / Math.max(profile.maxWarn, 1e-9) * 20);
  return Math.max(0, Math.min(100, 100 - penalty - magnitudePenalty));
}

export function roundMetric(value) {
  const number = Number(value || 0);
  return Math.round(number * 1000) / 1000;
}

export function normalizedZeroPointForDtype(staticTensor, dtype) {
  const d = (dtype || "").toLowerCase();  // WASM returns uppercase; normalise here
  const raw = Number(staticTensor?.zero_point_sample?.[0] ?? 0);
  if (d === "uint8") return clampInteger(Number.isFinite(raw) ? raw : 0, 0, 255);
  if (d === "int8") {
    if (Number.isFinite(raw) && raw > 127 && raw <= 255) return clampInteger(raw - 256, -128, 127);
    return clampInteger(Number.isFinite(raw) ? raw : 0, -128, 127);
  }
  if (d === "int32") return Number.isFinite(raw) ? Math.trunc(raw) : 0;
  return 0;
}

export function createResearchInputData(dtype, size, staticTensor) {
  if (dtype === "float32") return new Float32Array(size);
  if (dtype === "int32") return new Int32Array(size);
  if (dtype === "uint8") {
    const data = new Uint8Array(size);
    data.fill(normalizedZeroPointForDtype(staticTensor, dtype));
    return data;
  }
  if (dtype === "int8") {
    const data = new Int8Array(size);
    data.fill(normalizedZeroPointForDtype(staticTensor, dtype));
    return data;
  }
  if (dtype === "float16") return new Uint16Array(size); // 0x0000 = 0.0 in float16
  throw new Error(`Input dtype ${dtype} is not supported by this research runner`);
}

export function perturbTypedArray(data, dtype, staticTensor, options = {}) {
  const step = Math.max(1, Math.floor(data.length / 2048));
  const zeroPoint = normalizedZeroPointForDtype(staticTensor, dtype);
  const direction = Number(options.direction || 1);
  const amplitude = Math.max(1, Number(options.amplitude || 1));
  for (let i = 0; i < data.length; i += step) {
    const ordinal = Math.floor(i / step);
    const patternSign = options.pattern === "wide_positive"
      ? 1
      : options.pattern === "triad"
        ? (ordinal % 3 === 0 ? 1 : ordinal % 3 === 1 ? 0 : -1)
        : (ordinal % 2 === 0 ? 1 : -1);
    const sign = patternSign * direction * amplitude;
    if (!sign) continue;
    if (dtype === "float32") {
      data[i] += sign * 0.01;
    } else if (dtype === "uint8") {
      data[i] = clampInteger((Number.isFinite(zeroPoint) ? zeroPoint : data[i]) + sign, 0, 255);
    } else if (dtype === "int8") {
      data[i] = clampInteger((Number.isFinite(zeroPoint) ? zeroPoint : data[i]) + sign, -128, 127);
    } else if (dtype === "int32") {
      data[i] += sign;
    }
  }
}

export function isPerturbationMode(mode) {
  return ["perturb", "curve_plus", "curve_minus", "curve_wide"].includes(mode);
}

export function perturbationOptions(mode) {
  if (mode === "curve_minus") return { direction: -1, amplitude: 1 };
  if (mode === "curve_wide") return { direction: 1, amplitude: 2, pattern: "wide_positive" };
  return { direction: 1, amplitude: 1 };
}

// ── Haar pattern sweep ─────────────────────────────────────────────────────

export const HAAR_BANK_SCALES = [2, 4, 8, 16, 32];

export const HAAR_PATTERN_SPECS = [
  { id: "dc",         label: "DC uniform",   freqBand: "dc" },
  { id: "h_edge",     label: "Horiz edge",   freqBand: "very_low" },
  { id: "v_edge",     label: "Vert edge",    freqBand: "very_low" },
  { id: "diag",       label: "Diagonal",     freqBand: "low" },
  { id: "h_line",     label: "Horiz ridge",  freqBand: "low" },
  { id: "v_line",     label: "Vert ridge",   freqBand: "low" },
  { id: "checker_16", label: "Checker 16px", freqBand: "mid" },
  { id: "checker_8",  label: "Checker 8px",  freqBand: "mid_high" },
  { id: "checker_4",  label: "Checker 4px",  freqBand: "high" },
  { id: "checker_2",  label: "Checker 2px",  freqBand: "very_high" },
  { id: "diag_sv",    label: "Diag shift V", freqBand: "low" },
  { id: "diag_sh",    label: "Diag shift H", freqBand: "low" },
  // Multi-scale Haar wavelet bank: LH=horiz-edge, HL=vert-edge, HH=diagonal tiled at each scale
  ...["LH", "HL", "HH"].flatMap((type) =>
    HAAR_BANK_SCALES.map((s) => ({
      id:        `haar_${type}_${s}`,
      label:     `${type}@${s}px`,
      freqBand:  s <= 2 ? "very_high" : s <= 4 ? "high" : s <= 8 ? "mid_high" : s <= 16 ? "mid" : "low",
      haarType:  type,
      haarScale: s,
    }))
  ),
  // Fractal/multiscale Haar fields (1/f-weighted superposition across all scales)
  { id: "fractal_full", label: "Fractal (LH+HL+HH)", freqBand: "multiscale", fractal: true, fractalType: "full" },
  { id: "fractal_lh",   label: "Fractal LH",          freqBand: "multiscale", fractal: true, fractalType: "lh" },
  { id: "fractal_hl",   label: "Fractal HL",          freqBand: "multiscale", fractal: true, fractalType: "hl" },
  // COCO natural-image prior probe (LH:HL:HH weighted by COCO Haar statistics)
  { id: "coco_prior",   label: "COCO prior",           freqBand: "natural",   cocoPrior: true },
];

// Sweep config: which patterns / parameters to use for extended probes
export const HAAR_SWEEP_CONFIG = {
  translationPatternId: "haar_HL_8",
  translationGridN:     3,
  rotationPatternIds:   ["haar_LH_8", "haar_HL_8"],
  phasePatternIds:      ["haar_HH_4", "haar_HH_8"],
  phaseOffsets:         [0, 1, 2, 4],
  amplitudePatternId:   "haar_HH_4",
  polarityPatternIds:   ["haar_HL_8", "haar_LH_8", "haar_HH_4"],
  channelPatternId:     "haar_HL_8",
  channelModes:         ["all", "r", "g", "b"],
};

// opts: { phaseX, phaseY, rotation, channelMode, patchGrid, amplitude, polarity, baseline }
export function createHaarPatternData(dtype, shape, patternId, staticTensor, opts = {}) {
  const {
    phaseX      = 0,
    phaseY      = 0,
    rotation    = 0,
    channelMode = "all",
    patchGrid   = null,   // { gridRow, gridCol, gridN } for localized patch
    amplitude:  ampOverride = null,
    polarity    = 1,
    baseline    = "zero_point",
  } = opts;

  // Delegate fractal and COCO-prior patterns to their own generators
  if (patternId === "coco_prior") {
    return createCocoPriorProbe(dtype, shape, staticTensor,
      { amplitude: ampOverride, baseline });
  }
  if (patternId.startsWith("fractal_")) {
    const fractalType = patternId.replace("fractal_", ""); // "full"|"lh"|"hl"
    return createFractalHaarField(dtype, shape, staticTensor,
      { haarType: fractalType, amplitude: ampOverride, baseline });
  }

  const { h, w, c, layout } = resolveHaarDims(shape);
  const zp  = haarBaselineValue(dtype, staticTensor, baseline);
  const amp = ampOverride !== null ? ampOverride : haarAmplitudeForDtype(dtype, staticTensor);
  const size = shape.reduce((acc, v) => acc * Math.max(1, v), 1);
  const data = allocTypedArrayForDtype(dtype, size);
  data.fill(clampForDtype(dtype, zp));

  if (h < 2 || w < 2) {
    for (let i = 0; i < size; i++)
      data[i] = clampForDtype(dtype, zp + (i % 2 === 0 ? amp * polarity : -amp * polarity));
    return data;
  }

  // Localized patch bounds (defaults to full image)
  const rS = patchGrid ? Math.floor(patchGrid.gridRow * h / patchGrid.gridN) : 0;
  const rE = patchGrid ? Math.floor((patchGrid.gridRow + 1) * h / patchGrid.gridN) : h;
  const cS = patchGrid ? Math.floor(patchGrid.gridCol * w / patchGrid.gridN) : 0;
  const cE = patchGrid ? Math.floor((patchGrid.gridCol + 1) * w / patchGrid.gridN) : w;

  // Effective spatial dims after rotation (90°/270° swap h↔w)
  const effH = (rotation === 90 || rotation === 270) ? w : h;
  const effW = (rotation === 90 || rotation === 270) ? h : w;

  for (let row = rS; row < rE; row++) {
    for (let col = cS; col < cE; col++) {
      let pr = row, pc = col;
      if (rotation === 90)  { pr = col;           pc = h - 1 - row; }
      if (rotation === 180) { pr = h - 1 - row;   pc = w - 1 - col; }
      if (rotation === 270) { pr = w - 1 - col;   pc = row; }

      const sign = haarSignForPixel(patternId, pr, pc, effH, effW, { phaseX, phaseY }) * polarity;
      const val  = clampForDtype(dtype, zp + sign * amp);

      for (let ch = 0; ch < c; ch++) {
        const idx = layout === "nhwc"
          ? (row * w + col) * c + ch
          : ch * h * w + row * w + col;
        let v = val;
        if      (channelMode === "r")     v = ch === 0 ? val : clampForDtype(dtype, zp);
        else if (channelMode === "g")     v = ch === 1 ? val : clampForDtype(dtype, zp);
        else if (channelMode === "b")     v = ch === 2 ? val : clampForDtype(dtype, zp);
        else if (channelMode === "chroma")
          v = ch === 1 ? clampForDtype(dtype, zp - sign * amp) : val; // R+G-B+ opponent
        data[idx] = v;
      }
    }
  }
  return data;
}

// Per-element RMS of deviation from baseline zp; use for response_gain = output_rms / input_l2
export function computePatternInputStats(data, dtype, zp = 0) {
  const n = data.length;
  if (!n) return { l2: 0, linf: 0, mean: 0, nonzero_ratio: 0, is_zero_mean: true, n };
  let sumSq = 0, sum = 0, maxAbs = 0, nz = 0;
  for (let i = 0; i < n; i++) {
    const v = Number(data[i]) - zp;
    sumSq += v * v;
    sum   += v;
    if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    if (v !== 0) nz++;
  }
  const l2   = Math.sqrt(sumSq / n);
  const mean = sum / n;
  return { l2, linf: maxAbs, mean, nonzero_ratio: nz / n, is_zero_mean: Math.abs(mean) < l2 * 0.05 + 1e-9, n };
}

export function haarAmplitudeSweepLevels(dtype) {
  if (dtype === "float32") return [0.02, 0.05, 0.1, 0.25, 0.45];
  if (dtype === "int8" || dtype === "uint8") return [1, 2, 4, 8, 16, 32];
  return [1];
}

export function haarTranslationProfile(results) {
  const REGIONS = ["top-left","top-center","top-right","center-left","center","center-right","bottom-left","bottom-center","bottom-right"];
  const ok = results.filter((r) => r.drift && !r.error && typeof r.gridRow === "number");
  if (!ok.length) return null;
  const rmsList  = ok.map((r) => Number(r.drift?.rms || 0));
  const mean     = rmsList.reduce((a, v) => a + v, 0) / rmsList.length;
  const std      = Math.sqrt(rmsList.reduce((a, v) => a + (v - mean) ** 2, 0) / rmsList.length);
  const sensitivity = mean > 1e-9 ? std / mean : 0;
  const sorted   = [...ok].sort((a, b) => Number(b.drift?.rms || 0) - Number(a.drift?.rms || 0));
  const regionOf = (r) => REGIONS[r.gridRow * 3 + r.gridCol] || `(${r.gridRow},${r.gridCol})`;
  return {
    translation_sensitivity: Math.round(sensitivity * 1000) / 1000,
    max_shift_delta:  Math.max(...rmsList) - Math.min(...rmsList),
    peak_region:      regionOf(sorted[0]),
    trough_region:    regionOf(sorted[sorted.length - 1]),
    heatmap:          ok.map((r) => ({ row: r.gridRow, col: r.gridCol, region: regionOf(r), rms: Number(r.drift?.rms || 0) })),
    label: sensitivity < 0.1 ? "translation_stable" : sensitivity < 0.3 ? "moderate_position_sensitive" : "highly_position_sensitive",
  };
}

export function haarRotationProfile(results) {
  const ok = results.filter((r) => r.drift && !r.error && typeof r.rotation === "number");
  if (!ok.length) return null;
  const byRot = {};
  for (const r of ok) byRot[r.rotation] = Number(r.drift?.rms || 0);
  const vals  = Object.values(byRot);
  const maxR  = Math.max(...vals, 1e-9);
  const minR  = Math.min(...vals.filter((v) => v > 0), 1e-9);
  const asymmetry = maxR / minR;
  const h0  = (byRot[0] || 0) + (byRot[180] || 0);
  const h90 = (byRot[90] || 0) + (byRot[270] || 0);
  const denom = Math.max(h0 + h90, 1e-9);
  return {
    rotation_asymmetry: Math.round(asymmetry * 100) / 100,
    orientation_bias: Math.abs(h0 - h90) / denom < 0.1 ? "isotropic" : h0 > h90 ? "axial_dominant" : "transverse_dominant",
    by_rotation: byRot,
    label: asymmetry < 1.2 ? "rotation_stable" : asymmetry < 2.0 ? "moderate_rotation_sensitive" : "highly_rotation_sensitive",
  };
}

export function haarPhaseProfile(results) {
  const ok = results.filter((r) => r.drift && !r.error);
  if (!ok.length) return null;
  const rmsList = ok.map((r) => Number(r.drift?.rms || 0));
  const mean    = rmsList.reduce((a, v) => a + v, 0) / rmsList.length;
  const std     = Math.sqrt(rmsList.reduce((a, v) => a + (v - mean) ** 2, 0) / rmsList.length);
  const jitter  = mean > 1e-9 ? std / mean : 0;
  return {
    phase_jitter:  Math.round(jitter * 1000) / 1000,
    mean_rms:      mean,
    label:         jitter < 0.05 ? "phase_stable" : jitter < 0.2 ? "moderate_phase_jitter" : "high_phase_jitter",
    aliasing_risk: jitter > 0.2 ? "elevated" : "low",
  };
}

export function haarAmplitudeSweepProfile(results) {
  const ok = results.filter((r) => r.drift && !r.error && (r.inputStats?.l2 || 0) > 1e-9);
  if (ok.length < 2) return null;
  ok.sort((a, b) => a.amplitude - b.amplitude);
  const gains    = ok.map((r) => Number(r.drift?.rms || 0) / r.inputStats.l2);
  const meanGain = gains.reduce((a, v) => a + v, 0) / gains.length;
  const gainStd  = Math.sqrt(gains.reduce((a, v) => a + (v - meanGain) ** 2, 0) / gains.length);
  const nonlinearity = meanGain > 1e-9 ? gainStd / meanGain : 0;
  // saturation: if high-amp gain < low-amp gain → response is flattening
  const saturation   = gains[0] > 1e-9 ? 1 - Math.min(1, gains[gains.length - 1] / gains[0]) : 0;
  return {
    low_gain:           Math.round(gains[0] * 1000) / 1000,
    high_gain:          Math.round(gains[gains.length - 1] * 1000) / 1000,
    saturation_ratio:   Math.round(saturation * 1000) / 1000,
    nonlinearity_index: Math.round(nonlinearity * 1000) / 1000,
    amplitudes:         ok.map((r) => ({ amplitude: r.amplitude, rms: Number(r.drift?.rms || 0), gain: Number(r.drift?.rms || 0) / r.inputStats.l2 })),
    label:              nonlinearity < 0.1 ? "linear_response" : nonlinearity < 0.3 ? "mildly_nonlinear" : "highly_nonlinear",
  };
}

export function haarPolarityProfile(pairs) {
  const ok = pairs.filter((p) => p.plus?.drift && p.minus?.drift);
  if (!ok.length) return null;
  return ok.map(({ patternId, plus, minus }) => {
    const pr  = Number(plus.drift?.rms || 0);
    const mr  = Number(minus.drift?.rms || 0);
    const mean = (pr + mr) / 2;
    const asym = mean > 1e-9 ? Math.abs(pr - mr) / mean : 0;
    return {
      pattern_id:          patternId,
      plus_rms:            pr,
      minus_rms:           mr,
      polarity_asymmetry:  Math.round(asym * 1000) / 1000,
      dominant:            pr >= mr ? "positive" : "negative",
      label:               asym < 0.1 ? "symmetric" : asym < 0.3 ? "mild_asymmetry" : "strong_asymmetry",
    };
  });
}

export function haarSpatialPatchProfile(results, gridN = 3) {
  const ok = results.filter((r) => r.drift && !r.error && typeof r.gridRow === "number");
  if (!ok.length) return null;
  const REGIONS = ["top-left","top-center","top-right","center-left","center","center-right","bottom-left","bottom-center","bottom-right"];
  const heatmap = [];
  for (let gr = 0; gr < gridN; gr++)
    for (let gc = 0; gc < gridN; gc++) {
      const match = ok.find((r) => r.gridRow === gr && r.gridCol === gc);
      heatmap.push({ row: gr, col: gc, region: REGIONS[gr * gridN + gc] || `${gr},${gc}`, rms: match ? Number(match.drift?.rms || 0) : null });
    }
  const valid  = heatmap.filter((c) => c.rms !== null);
  const sorted = [...valid].sort((a, b) => b.rms - a.rms);
  const vals   = valid.map((c) => c.rms);
  const mean   = vals.reduce((a, v) => a + v, 0) / Math.max(vals.length, 1);
  const sv     = mean > 1e-9 ? Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / vals.length) / mean : 0;
  return {
    heatmap,
    peak_cell:        sorted[0] || null,
    trough_cell:      sorted[sorted.length - 1] || null,
    spatial_variance: Math.round(sv * 1000) / 1000,
  };
}

export function haarSensitivityProfile(results) {
  const ok = results.filter((r) => !r.error && r.drift);
  if (!ok.length) return { trend: "unavailable" };

  const byBand = {};
  for (const r of ok) {
    const b = r.freqBand;
    if (!byBand[b]) byBand[b] = [];
    byBand[b].push(Number(r.drift?.rms || 0));
  }
  const avgByBand = Object.fromEntries(
    Object.entries(byBand).map(([b, vals]) => [b, vals.reduce((a, v) => a + v, 0) / vals.length]),
  );

  const get = (id) => ok.find((r) => r.id === id)?.drift?.rms || 0;
  const hEdge  = get("h_edge");
  const vEdge  = get("v_edge");
  const fine   = get("checker_2");
  const mid4   = get("checker_4");
  const mid8   = get("checker_8");
  const coarse = get("checker_16");
  const diag   = get("diag");
  const diagSv = get("diag_sv");
  const diagSh = get("diag_sh");

  const orientationBias = (() => {
    const diff   = Math.abs(hEdge - vEdge);
    const maxEdge = Math.max(hEdge, vEdge, 1e-9);
    if (diff / maxEdge < 0.15) return "isotropic";
    return hEdge > vEdge ? "horizontal_dominant" : "vertical_dominant";
  })();

  const checkerCandidates = [
    { id: "checker_2",  label: "2px",  rms: fine },
    { id: "checker_4",  label: "4px",  rms: mid4 },
    { id: "checker_8",  label: "8px",  rms: mid8 },
    { id: "checker_16", label: "16px", rms: coarse },
  ];
  const peakChecker = checkerCandidates.reduce((a, b) => b.rms > a.rms ? b : a);

  const freqRatio = coarse > 1e-9 ? fine / coarse : null;
  const frequencyTrend = (() => {
    if (freqRatio === null) return "indeterminate";
    if (freqRatio > 2.5) return peakChecker.id === "checker_2" ? "high_freq_sensitive" : "mid_high_freq_peak";
    if (freqRatio < 0.4) return peakChecker.id === "checker_16" ? "low_freq_sensitive" : "mid_low_freq_peak";
    return "broadband";
  })();

  const positionSensitivity = (() => {
    const ref    = diag || 1e-9;
    const vShift = Math.abs(diagSv - diag) / ref;
    const hShift = Math.abs(diagSh - diag) / ref;
    if (vShift < 0.1 && hShift < 0.1) return "translation_invariant";
    const maxShift  = Math.max(vShift, hShift);
    const dominance = maxShift > 1e-9 ? Math.abs(vShift - hShift) / maxShift : 0;
    if (dominance < 0.10) return "position_sensitive";
    return vShift > hShift ? "vertical_position_sensitive" : "horizontal_position_sensitive";
  })();

  // Haar bank: find peak scale per sub-band type
  const peakByType = {};
  for (const type of ["LH", "HL", "HH"]) {
    let best = null;
    for (const r of ok.filter((r) => r.haarType === type))
      if (!best || Number(r.drift?.rms || 0) > Number(best.drift?.rms || 0)) best = r;
    if (best) peakByType[type] = { id: best.id, scale: best.haarScale, rms: Number(best.drift?.rms || 0) };
  }
  const lhRms = peakByType["LH"]?.rms || 0;
  const hlRms = peakByType["HL"]?.rms || 0;
  const bankBias = Math.abs(lhRms - hlRms) / Math.max(lhRms + hlRms, 1e-9) < 0.1
    ? "isotropic" : lhRms > hlRms ? "horizontal_dominant" : "vertical_dominant";

  const mostSensitive  = ok.reduce((a, b) => (b.drift?.rms || 0) > (a.drift?.rms || 0) ? b : a);
  const leastSensitive = ok.reduce((a, b) => (b.drift?.rms || 0) < (a.drift?.rms || 0) ? b : a);

  return {
    orientation_bias:        orientationBias,
    frequency_trend:         frequencyTrend,
    freq_ratio_fine_coarse:  freqRatio !== null ? Math.round(freqRatio * 100) / 100 : null,
    peak_checker:            peakChecker.id,
    peak_checker_label:      peakChecker.label,
    position_sensitivity:    positionSensitivity,
    most_sensitive_pattern:  mostSensitive.label,
    most_sensitive_rms:      mostSensitive.drift?.rms,
    least_sensitive_pattern: leastSensitive.label,
    avg_drift_by_freq_band:  avgByBand,
    peak_by_haar_type:       peakByType,
    bank_orientation_bias:   bankBias,
  };
}

function haarBaselineValue(dtype, staticTensor, baseline) {
  if (baseline === "zero") return 0;
  if (baseline === "mid")  return dtype === "uint8" ? 128 : dtype === "float32" ? 0.5 : 0;
  return normalizedZeroPointForDtype(staticTensor, dtype);
}

function haarSignForPixel(id, row, col, h, w, opts = {}) {
  const { phaseX = 0, phaseY = 0 } = opts;

  // Multi-scale Haar bank: haar_LH_S (horiz), haar_HL_S (vert), haar_HH_S (diagonal)
  const haarM = id.match(/^haar_(LH|HL|HH)_(\d+)$/);
  if (haarM) {
    const type = haarM[1];
    const S    = parseInt(haarM[2]);
    const tile = 2 * S;
    const r = ((row + phaseY) % tile + tile) % tile;
    const c = ((col + phaseX) % tile + tile) % tile;
    if (type === "LH") return r < S ? 1 : -1;
    if (type === "HL") return c < S ? 1 : -1;
    return (r < S) === (c < S) ? 1 : -1;  // HH
  }

  switch (id) {
    case "dc":         return 1;
    case "h_edge":     return row < h / 2 ? 1 : -1;
    case "v_edge":     return col < w / 2 ? 1 : -1;
    case "diag":       return ((row < h / 2) !== (col < w / 2)) ? 1 : -1;
    case "h_line":     return row < h / 3 ? 1 : row < (2 * h) / 3 ? -1 : 1;
    case "v_line":     return col < w / 3 ? 1 : col < (2 * w) / 3 ? -1 : 1;
    case "checker_2":  return ((Math.floor(row / 2)  + Math.floor(col / 2))  % 2 === 0) ? 1 : -1;
    case "checker_4":  return ((Math.floor(row / 4)  + Math.floor(col / 4))  % 2 === 0) ? 1 : -1;
    case "checker_8":  return ((Math.floor(row / 8)  + Math.floor(col / 8))  % 2 === 0) ? 1 : -1;
    case "checker_16": return ((Math.floor(row / 16) + Math.floor(col / 16)) % 2 === 0) ? 1 : -1;
    case "diag_sv": {
      const sr = (row + Math.floor(h / 4)) % h;
      return ((sr < h / 2) !== (col < w / 2)) ? 1 : -1;
    }
    case "diag_sh": {
      const sc = (col + Math.floor(w / 4)) % w;
      return ((row < h / 2) !== (sc < w / 2)) ? 1 : -1;
    }
    default: return 0;
  }
}

function resolveHaarDims(shape) {
  const s = shape.map((v) => Math.max(1, v));
  if (s.length === 4) {
    const [, d1, d2, d3] = s;
    // NHWC: last dim is small (channels), mid dims are large (spatial)
    if (d3 <= 4 && d1 > 4 && d2 > 4) return { h: d1, w: d2, c: d3, layout: "nhwc" };
    if (d1 <= 4 && d2 > 4 && d3 > 4) return { h: d2, w: d3, c: d1, layout: "nchw" };
    return { h: d1, w: d2, c: d3, layout: "nhwc" };
  }
  // Non-4D: synthesize spatial dims — only meaningful for image-like inputs
  const total = s.reduce((a, v) => a * v, 1);
  const side  = Math.ceil(Math.sqrt(total));
  return { h: side, w: Math.ceil(total / side), c: 1, layout: "nhwc", synthetic_spatial: true };
}

function haarAmplitudeForDtype(dtype, staticTensor) {
  const d = (dtype || "").toLowerCase();  // accept both "FLOAT32" and "float32"
  const zp = normalizedZeroPointForDtype(staticTensor, d);
  if (d === "float32" || d === "float16") return 0.45;
  if (d === "int8")  return Math.max(1, Math.min(60, Math.min(127 - zp, zp + 128)));
  if (d === "uint8") return Math.max(1, Math.min(60, zp > 0 ? Math.min(255 - zp, zp) : 30));
  if (d === "int32") return 1000;
  return 1;
}

function allocTypedArrayForDtype(dtype, size) {
  if (dtype === "float32") return new Float32Array(size);
  if (dtype === "int8")    return new Int8Array(size);
  if (dtype === "uint8")   return new Uint8Array(size);
  if (dtype === "int32")   return new Int32Array(size);
  return new Float32Array(size);
}

function clampForDtype(dtype, value) {
  if (dtype === "float32") return value;
  if (dtype === "int8")    return clampInteger(value, -128, 127);
  if (dtype === "uint8")   return clampInteger(value, 0, 255);
  if (dtype === "int32")   return Math.trunc(value);
  return value;
}

export function perturbationStatsFromMode(mode) {
  const options = perturbationOptions(mode);
  return {
    active: isPerturbationMode(mode),
    mode,
    touched: 0,
    total: 0,
    epsilon: 0,
    epsilonLabel: "0",
    l2: 0,
    linf: 0,
    direction: options.direction,
    amplitude: options.amplitude,
    pattern: options.pattern || "alternating",
  };
}

export function compareOutputArrays(left, right) {
  let count = 0;
  let sumAbs = 0;
  let sumSq = 0;
  let maxAbs = 0;
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let i = 0; i < Math.min(left.length, right.length); i++) {
    const a = left[i];
    const b = right[i];
    const n = Math.min(a.length, b.length);
    for (let j = 0; j < n; j++) {
      const av = Number(a[j]);
      const bv = Number(b[j]);
      const delta = bv - av;
      const abs = Math.abs(delta);
      count += 1;
      sumAbs += abs;
      sumSq += delta * delta;
      maxAbs = Math.max(maxAbs, abs);
      dot += av * bv;
      leftNorm += av * av;
      rightNorm += bv * bv;
    }
  }
  const cosine = leftNorm && rightNorm ? dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm)) : 1;
  return {
    count,
    meanAbs: count ? sumAbs / count : 0,
    rms: count ? Math.sqrt(sumSq / count) : 0,
    maxAbs,
    cosineDistance: Math.max(0, 1 - cosine),
    top1Flip: argmax(left[0]) !== argmax(right[0]),
    leftRms: count ? Math.sqrt(leftNorm / count) : 0, // baseline output magnitude for relative thresholding
  };
}

export function driftLooksIdentical(left, right) {
  if (!left || !right) return false;
  return [
    "count",
    "meanAbs",
    "rms",
    "maxAbs",
    "cosineDistance",
    "top1Flip",
  ].every((key) => {
    const a = left[key];
    const b = right[key];
    if (typeof a === "boolean" || typeof b === "boolean") return Boolean(a) === Boolean(b);
    return Math.abs(Number(a || 0) - Number(b || 0)) <= 1e-12;
  });
}

export function driftDeltaSummary(left, right) {
  if (!left || !right) return null;
  return {
    mean_abs_delta: Number(left.meanAbs || 0) - Number(right.meanAbs || 0),
    rms_delta: Number(left.rms || 0) - Number(right.rms || 0),
    max_abs_delta: Number(left.maxAbs || 0) - Number(right.maxAbs || 0),
    cosine_distance_delta: Number(left.cosineDistance || 0) - Number(right.cosineDistance || 0),
    top1_flip_changed: Boolean(left.top1Flip) !== Boolean(right.top1Flip),
  };
}

export function deployProbeConsistencyWarning(widePlusIdentical, wideMinusIdentical) {
  if (widePlusIdentical && wideMinusIdentical) {
    return "The 2eps drift summary is identical to both +eps and -eps. This usually means the local probe is in a quantized plateau, saturated output region, or runtime path with coarse output resolution; validate with prepared/calibration inputs before interpreting the tested rank-stability radius.";
  }
  if (widePlusIdentical) {
    return "The 2eps drift summary is identical to +eps even though the input perturbation amplitude doubled. For INT8 models this can be legitimate plateau/saturation behavior, but it should be reported explicitly and checked with representative calibration inputs.";
  }
  if (wideMinusIdentical) {
    return "The 2eps drift summary is identical to -eps. This can indicate output saturation, quantized plateaus, or a runtime/input perturbation path that should be checked with representative calibration data.";
  }
  return "";
}

export function computeDirectionalCurvature(baseline, plus, minus, inputPerturbation) {
  let count = 0;
  let sumSq = 0;
  let maxAbs = 0;
  for (let i = 0; i < Math.min(baseline.length, plus.length, minus.length); i++) {
    const base = baseline[i];
    const p = plus[i];
    const m = minus[i];
    const n = Math.min(base.length, p.length, m.length);
    for (let j = 0; j < n; j++) {
      const second = Number(p[j]) - 2 * Number(base[j]) + Number(m[j]);
      const abs = Math.abs(second);
      count += 1;
      sumSq += second * second;
      maxAbs = Math.max(maxAbs, abs);
    }
  }
  const rawRms = count ? Math.sqrt(sumSq / count) : 0;
  const l2 = Math.max(Number(inputPerturbation?.l2 || 0), 1e-9);
  const plusDrift = compareOutputArrays(baseline, plus);
  const minusDrift = compareOutputArrays(baseline, minus);
  return {
    count,
    rawRms,
    maxAbs,
    normalizedRms: rawRms / (l2 * l2),
    localLipschitz: Math.max(plusDrift.rms, minusDrift.rms) / l2,
    plusDrift,
    minusDrift,
  };
}

export function computeDeployBasinProxy(plusDrift, minusDrift, wideDrift, curvature, margin) {
  const smallFlip = Boolean(plusDrift.top1Flip || minusDrift.top1Flip);
  const wideFlip = Boolean(wideDrift.top1Flip);
  const radiusLabel = smallFlip ? "<1 epsilon" : wideFlip ? "1 epsilon" : ">=2 epsilon";
  const curvaturePenalty = Math.min(30, Math.log10(1 + Math.max(0, curvature.normalizedRms)) * 12);
  const driftPenalty = Math.min(25, Math.max(plusDrift.cosineDistance, minusDrift.cosineDistance, wideDrift.cosineDistance) * 500);
  const flipPenalty = smallFlip ? 35 : wideFlip ? 18 : 0;
  const marginPenalty = margin.ready && margin.margin <= Math.max(wideDrift.maxAbs, 1e-9) ? 12 : 0;
  const score = Math.max(0, Math.min(100, 100 - curvaturePenalty - driftPenalty - flipPenalty - marginPenalty));
  return {
    score,
    radiusLabel,
    top1Stable: !smallFlip && !wideFlip,
    score_basis: "unvalidated fixed-weight experimental composite over deploy-domain finite-difference probes; not a loss-landscape, accuracy, latency, robustness, or release metric",
  };
}

export function decisionMargin(values) {
  if (!values || values.length < 2) {
    return {
      ready: false,
      margin: 0,
      status: "unavailable",
      detail: "At least two output values are required to compute a top-1/top-2 margin.",
    };
  }
  let first = -Infinity;
  let second = -Infinity;
  for (const value of values) {
    const number = Number(value);
    if (number > first) {
      second = first;
      first = number;
    } else if (number > second) {
      second = number;
    }
  }
  const ready = Number.isFinite(first) && Number.isFinite(second);
  const margin = ready ? first - second : 0;
  return {
    ready,
    margin,
    status: !ready ? "unavailable" : margin <= 0 ? "tie_or_ambiguous" : "measured",
    detail: !ready
      ? "At least two finite output values are required."
      : margin <= 0
        ? "The first output tensor has no positive top-1/top-2 separation under this raw output contract; treat rank stability as boundary-sensitive."
        : "Compare this margin with observed max output drift; margin below drift can indicate rank instability.",
  };
}

// ── Natural-Haar Prior (COCO-derived) ─────────────────────────────────────────
//
// Empirical Haar energy distribution from 4 COCO val2017 images (256×256 luma).
// Reference: coco_haar_decomposition_observation.md
export const COCO_HAAR_PRIOR = {
  // Mean energy fraction per subband (one-level 2D Haar, 256×256)
  mean_ll: 0.984,   // ~98.4% in LL
  mean_lh: 0.0048,  // ~0.48% horizontal-edge detail
  mean_hl: 0.0041,  // ~0.41% vertical-edge detail
  mean_hh: 0.0019,  // ~0.19% diagonal/checker detail
  // Sparsity: fraction of coefficients needed for 95% energy
  sparsity_95: { min: 21, max: 1229, mean: 491 },
  // Detail-band relative weights (LH:HL:HH = 2.5 : 2.2 : 1.0 from COCO means)
  detail_weight_lh: 2.53,
  detail_weight_hl: 2.16,
  detail_weight_hh: 1.00,
};

// Generate a probe input whose Haar detail-band energy ratios match COCO statistics.
// Uses COCO LH:HL:HH ratios to mix three Haar atoms at the given scale.
// scale defaults to 8 (mid-freq band most relevant for object boundaries).
export function createCocoPriorProbe(dtype, shape, staticTensor, opts = {}) {
  const { scale = 8, amplitude = null, baseline = "zero_point" } = opts;
  const zp = baseline === "zero_point" ? normalizedZeroPointForDtype(staticTensor, dtype) : 0;

  let h, w, c;
  if (shape.length === 4) {
    [, h, w, c] = shape;
  } else if (shape.length === 3) {
    [h, w, c] = shape;
  } else {
    h = w = 1; c = shape[shape.length - 1] || 1;
  }
  const n = shape.reduce((a, b) => a * b, 1);
  const isFloat = (dtype || "").toUpperCase().includes("FLOAT");
  const isInt8  = (dtype || "").toUpperCase() === "INT8";
  const isUint8 = (dtype || "").toUpperCase() === "UINT8";

  // Default amplitude: derive from dtype range and zero-point so INT8/UINT8 get proper stimulus
  let amp = amplitude;
  if (amp == null) amp = haarAmplitudeForDtype(dtype, staticTensor);

  const tile = 2 * scale;
  const { detail_weight_lh: wLH, detail_weight_hl: wHL, detail_weight_hh: wHH } = COCO_HAAR_PRIOR;
  const total_w = wLH + wHL + wHH;
  const aLH = amp * (wLH / total_w);
  const aHL = amp * (wHL / total_w);
  const aHH = amp * (wHH / total_w);

  // Build flat array
  const out = isFloat ? new Float32Array(n) : new Int32Array(n);
  const spatial_h = h || 1;
  const spatial_w = w || 1;

  let idx = 0;
  for (let row = 0; row < spatial_h; row++) {
    const r = row % tile;
    for (let col = 0; col < spatial_w; col++) {
      const cc = col % tile;
      const lhSign = r < scale ? 1 : -1;
      const hlSign = cc < scale ? 1 : -1;
      const hhSign = (r < scale) === (cc < scale) ? 1 : -1;
      const val = aLH * lhSign + aHL * hlSign + aHH * hhSign;
      for (let ch = 0; ch < (c || 1); ch++) {
        if (isFloat) {
          out[idx++] = val;
        } else {
          out[idx++] = Math.round(zp + val);
        }
      }
    }
  }

  if (isFloat) return out;
  // C4: buffer reinterpret (new Int8Array(Int32Array.buffer)) gives 4× wrong length.
  //     Use typed mapping instead, matching createFractalHaarField's correct pattern.
  if (isInt8)  return Int8Array.from(out, v => Math.max(-128, Math.min(127, Math.round(v))));
  if (isUint8) return new Uint8Array(out.map(v => Math.max(0, Math.min(255, v))));
  return out;
}

// Generate a fractal/multiscale Haar field: 1/sqrt(scale)-weighted superposition
// of LH+HL+HH atoms across scales [2,4,8,16,32]. Approximates a 1/f (pink noise)
// amplitude spectrum, which is characteristic of natural images.
export function createFractalHaarField(dtype, shape, staticTensor, opts = {}) {
  const {
    haarType  = "full",   // "lh" | "hl" | "hh" | "full"
    amplitude = null,
    baseline  = "zero_point",
    scaleExponent = 0.5,  // amplitude ∝ scale^scaleExponent (0.5 = pink noise approx)
  } = opts;
  const zp = baseline === "zero_point" ? normalizedZeroPointForDtype(staticTensor, dtype) : 0;

  let h, w, c;
  if (shape.length === 4)      { [, h, w, c] = shape; }
  else if (shape.length === 3) { [h, w, c] = shape; }
  else                          { h = w = 1; c = shape[shape.length - 1] || 1; }

  const n = shape.reduce((a, b) => a * b, 1);
  const isFloat = (dtype || "").toUpperCase().includes("FLOAT");
  const isInt8  = (dtype || "").toUpperCase() === "INT8";
  const isUint8 = (dtype || "").toUpperCase() === "UINT8";

  let baseAmp = amplitude;
  if (baseAmp == null) baseAmp = haarAmplitudeForDtype(dtype, staticTensor);

  const scales = HAAR_BANK_SCALES; // [2,4,8,16,32]
  const maxScale = Math.max(...scales);

  // Precompute per-scale amplitudes (1/f weighting: coarser scale = higher amplitude)
  const scaleAmps = scales.map(s => baseAmp * Math.pow(s / maxScale, scaleExponent));
  // Normalize so RMS across the sum is approximately baseAmp
  const rmsNorm = Math.sqrt(scaleAmps.reduce((a, v) => a + v * v, 0)) / baseAmp;
  const normAmps = scaleAmps.map(a => a / Math.max(rmsNorm, 1e-9));

  const spatial_h = h || 1;
  const spatial_w = w || 1;
  const out = new Float64Array(n);

  let idx = 0;
  for (let row = 0; row < spatial_h; row++) {
    for (let col = 0; col < spatial_w; col++) {
      let val = 0;
      for (let si = 0; si < scales.length; si++) {
        const s = scales[si];
        const tile = 2 * s;
        const r = row % tile;
        const cc = col % tile;
        const a = normAmps[si];
        if (haarType === "lh" || haarType === "full") val += a * (r < s ? 1 : -1);
        if (haarType === "hl" || haarType === "full") val += a * (cc < s ? 1 : -1);
        if (haarType === "hh" || haarType === "full") val += a * ((r < s) === (cc < s) ? 1 : -1);
      }
      if (haarType === "full") val /= Math.sqrt(3);
      for (let ch = 0; ch < (c || 1); ch++) out[idx++] = val;
    }
  }

  if (isFloat) return Float32Array.from(out);
  if (isInt8) {
    return Int8Array.from(out.map(v => Math.max(-128, Math.min(127, Math.round(zp + v)))));
  }
  if (isUint8) {
    return Uint8Array.from(out.map(v => Math.max(0, Math.min(255, Math.round(zp + v)))));
  }
  return Int32Array.from(out.map(v => Math.round(zp + v)));
}

// ── Static-Runtime Alignment ──────────────────────────────────────────────────
//
// Compares kernel Haar decomposition (what filters HAVE learned)
// with the Haar probe sweep (what the model IS SENSITIVE to at runtime).
//
// kernelHaarResult: output of compute_kernel_haar_decomposition() WASM call
// haarPatterns:    array of { patternId, response_gain } from runHaarPatternSweep
// haarProfile:     output of haarSensitivityProfile()
//
// Returns alignment score and per-op breakdown.
export function computeStaticRuntimeAlignment(kernelHaarResult, haarPatterns, haarProfile) {
  if (!kernelHaarResult?.ops?.length || !haarPatterns?.length) {
    return { score: null, detail: "insufficient_data", per_op: [] };
  }

  // Build runtime sensitivity map: type → best scale → gain
  const runtimeBest = { LH: null, HL: null, HH: null, LL: null };
  for (const p of haarPatterns) {
    const m = (p.patternId || "").match(/^haar_(LH|HL|HH)_(\d+)$/);
    if (!m) continue;
    const type = m[1];
    const gain = Number(p.response_gain ?? p.drift?.rms ?? 0);
    if (!runtimeBest[type] || gain > runtimeBest[type].gain) {
      runtimeBest[type] = { scale: parseInt(m[2]), gain };
    }
  }
  // DC pattern
  const dcPat = haarPatterns.find(p => p.patternId === "dc");
  runtimeBest.LL = { scale: 0, gain: Number(dcPat?.response_gain ?? dcPat?.drift?.rms ?? 0) };

  // Guard: if all patterns failed (all non-LL entries still null and dc gain=0), no usable data
  const hasUsableData = runtimeBest.LL.gain > 0 || Object.entries(runtimeBest).some(([k, v]) => k !== "LL" && v != null);
  if (!hasUsableData) return { score: null, detail: "insufficient_data", per_op: [] };

  // Global runtime dominant (by gain)
  const runtimeDominant = Object.entries(runtimeBest)
    .filter(([, v]) => v != null)
    .sort((a, b) => (b[1].gain ?? 0) - (a[1].gain ?? 0))[0]?.[0]?.toLowerCase() ?? "ll";

  // W1: pre-compute median runtime gain so per-op alignment uses a data-driven threshold
  // rather than a single global dominant band (which would unfairly mark all non-dominant
  // ops as misaligned even when their band has substantial runtime sensitivity).
  const allGains = Object.values(runtimeBest)
    .filter(v => v != null)
    .map(v => v.gain ?? 0)
    .sort((a, b) => a - b);
  const gainMedian = allGains.length ? allGains[Math.floor(allGains.length / 2)] : 0;

  // Per-op alignment: kernel dominant band has runtime gain at or above the median.
  const per_op = kernelHaarResult.ops.map(op => {
    const kDom = op.dominant; // "ll"|"lh"|"hl"|"hh"
    const typeKey = kDom.toUpperCase();
    const rtGain = runtimeBest[typeKey]?.gain ?? 0;
    // Aligned = the band the kernel learned is among the runtime-sensitive bands
    const aligned = rtGain >= gainMedian;

    // Fraction of this op's kernel energy concentrated in its dominant band
    const kFrac = op.mean_energy[kDom] ?? 0;

    return {
      op_index: op.op_index,
      op_name:  op.op_name,
      kernel_dominant: kDom,
      runtime_dominant: runtimeDominant,  // global reference kept for context
      aligned,
      kernel_energy_in_dominant: kFrac,
      runtime_gain_for_dominant: rtGain,
    };
  });

  const aligned_count = per_op.filter(o => o.aligned).length;
  const total = per_op.length;
  const score = total > 0 ? aligned_count / total : null;

  // Orientation match: does kernel orientation bias agree with runtime LH vs HL gain?
  const kernelBias = kernelHaarResult.summary.orientation_bias; // >0 = LH (H-edge) dominant
  const runtimeLhGain = runtimeBest.LH?.gain ?? 0;
  const runtimeHlGain = runtimeBest.HL?.gain ?? 0;
  const runtimeOrientBias = runtimeLhGain - runtimeHlGain;
  const orientMatch = (kernelBias > 0) === (runtimeOrientBias > 0);

  return {
    score,
    aligned_count,
    total_conv_ops: total,
    kernel_global_dominant: kernelHaarResult.summary.global_dominant,
    runtime_dominant: runtimeDominant,
    runtime_best: runtimeBest,
    orientation_match: orientMatch,
    kernel_orientation_bias: kernelBias,
    runtime_orientation_bias: runtimeOrientBias,
    per_op,
    interpretation: alignmentInterpretation(score, orientMatch, kernelHaarResult.summary),
  };
}

function alignmentInterpretation(score, orientMatch, summary) {
  if (score == null) return "Insufficient data for alignment analysis.";
  const pct = Math.round((score ?? 0) * 100);
  const orient = orientMatch
    ? "Kernel orientation bias agrees with runtime probe sensitivity."
    : "Kernel orientation bias disagrees with runtime probe sensitivity — possible filter redundancy or activation cancellation.";
  if (score >= 0.7) {
    return `High static-runtime alignment (${pct}% of conv ops). ${orient} Learned filters and activation sensitivity are consistent, supporting probe validity.`;
  } else if (score >= 0.4) {
    return `Moderate alignment (${pct}% of conv ops). ${orient} Some layers show divergence between learned components and runtime sensitivity — examine misaligned ops for potential dead or overparameterized filters.`;
  } else {
    return `Low alignment (${pct}% of conv ops). ${orient} Learned filter structures do not match runtime activation sensitivity. This may indicate that ${summary.edge_heavy_ops > summary.dc_heavy_ops ? "edge-oriented filters are not activated by the probe suite" : "DC-dominant filters dominate despite edge-sensitive probes"}.`;
  }
}

export function timingVarianceSummary(baselineMs, perturbedMs) {
  const base = Number(baselineMs || 0);
  const next = Number(perturbedMs || 0);
  const delta = next - base;
  const denom = Math.max(base, next, 1e-9);
  const ratio = Math.abs(delta) / denom;
  const label = `${delta >= 0 ? "+" : ""}${delta.toFixed(2)} ms (${(ratio * 100).toFixed(1)}%)`;
  const detail = delta < 0
    ? "Perturbed single-run timing was faster than baseline. This is normal browser/runtime variance; cold vs warm compile/runtime cache can also change module-to-module timings unless repeated p50/p90/p95 data confirms it."
    : "Perturbed single-run timing was slower than baseline. Treat as variance; cold vs warm compile/runtime cache can also change module-to-module timings unless repeated p50/p90/p95 data confirms it.";
  if (ratio <= 0.15) {
    return { label, detail, status: { tone: "info", label: "variance", criteria: "single-run timing delta <= 15% is treated as runtime/browser variance." } };
  }
  return { label, detail, status: { tone: "warn", label: "watch", criteria: "single-run timing delta > 15%; rerun Runtime Benchmark p50/p90/p95 before making latency claims." } };
}
