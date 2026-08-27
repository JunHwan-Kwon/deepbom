import { deploymentFrontierTargetIds } from "./app-config.js";
import {
  assessedOpLogicalBytes,
  bottleneckDistributionData,
  boundTone,
  estimateOpBottleneck,
  macDistributionData,
  quantLabel,
  quantStateLabel,
  quantTileTone,
  summarizeFrontierTargetComparison,
  summarizeTargetComparison,
} from "./analysis.js";
import {
  renderVisualEmpty,
  targetCompareMessage,
  targetMetric,
  targetStack,
  visualListItem,
} from "./dom.js";
import {
  formatBytes,
  formatNumber,
  formatPercent,
  formatUs,
  humanizeStageKey,
  maxBy,
  padOp,
  shortError,
  sumNumbers,
} from "./format.js";
import { performanceVisualCopyFor } from "./workflow-copy.js";
import { serializedTensorPresentation } from "./serialized-tensor-view.js";
import { renderOnDeviceLlmView } from "./on-device-llm-view.js";
import {
  buildQuantizationExposurePresentation,
  renderEvidenceTreemap,
} from "./evidence-treemap.js";

// ── Floating tooltip ──────────────────────────────────────────────────────────
let _tip = null;
let _tipSuppressedUntil = 0;
function getTip() {
  if (!_tip) {
    _tip = document.createElement("div");
    _tip.className = "flame-tooltip";
    _tip.hidden = true;
    document.body.appendChild(_tip);
    window.addEventListener("scroll", hideTip, { capture: true, passive: true });
    window.addEventListener("resize", hideTip, { passive: true });
  }
  return _tip;
}
function hideTip() { getTip().hidden = true; }
function suppressTip() {
  _tipSuppressedUntil = performance.now() + 700;
  hideTip();
}
function moveTip(tip, e) {
  const margin = 8;
  const rect = tip.getBoundingClientRect();
  let x = e.clientX + 14;
  let y = e.clientY - 10;
  if (x + rect.width > window.innerWidth - margin) x = e.clientX - rect.width - 14;
  if (y + rect.height > window.innerHeight - margin) y = e.clientY - rect.height - 14;
  x = Math.max(margin, Math.min(x, window.innerWidth - rect.width - margin));
  y = Math.max(margin, Math.min(y, window.innerHeight - rect.height - margin));
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
function attachTip(node, lines) {
  node.addEventListener("mouseenter", (e) => {
    if (performance.now() < _tipSuppressedUntil) return;
    const tip = getTip();
    tip.replaceChildren(...lines.map(([label, value]) => {
      const row = document.createElement("div");
      row.className = "tip-row";
      if (label) {
        const l = document.createElement("span");
        l.className = "tip-label";
        l.textContent = label + ":";
        row.append(l);
      }
      const v = document.createElement("span");
      v.textContent = value;
      row.append(v);
      return row;
    }));
    tip.hidden = false;
    moveTip(tip, e);
  });
  node.addEventListener("mousemove", (e) => { if (!getTip().hidden) moveTip(getTip(), e); });
  node.addEventListener("mouseleave", hideTip);
  node.addEventListener("pointerdown", suppressTip);
  node.addEventListener("click", suppressTip);
}

export function quantSummaryEvidence(analysis) {
  const ops = Array.isArray(analysis?.ops) ? analysis.ops : [];
  const tensors = Array.isArray(analysis?.tensors) ? analysis.tensors : [];
  const tensorByIndex = new Map(tensors.map((tensor) => [Number(tensor.index), tensor]));
  const opSignals = new Map();
  const addSignal = (index, label, tone = "warn") => {
    if (!Number.isInteger(index)) return;
    const current = opSignals.get(index) || { labels: [], tone: "warn" };
    if (!current.labels.includes(label)) current.labels.push(label);
    if (tone === "risk") current.tone = "risk";
    opSignals.set(index, current);
  };
  const localRiskOps = ops.filter((op) => op.quant_risk === "risk" || op.quant_risk === "warn");
  for (const op of localRiskOps) addSignal(Number(op.index), "op-local scale / zero-point", op.quant_risk);
  const kernelOps = ops.map((op) => ({ op, weight: tensorByIndex.get(Number(op.inputs?.[1])) }))
    .filter(({ op, weight }) => ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"].includes(op.name) && weight?.constant_buffer);
  const asymmetricOps = kernelOps.filter(({ weight }) => weight.dtype === "UINT8"
    && Number(weight.quant_zero_points) === 1 && Number(weight.zero_point_sample?.[0]) !== 0).map(({ op }) => op);
  const perTensorDepthwiseOps = kernelOps.filter(({ op, weight }) => op.name === "DEPTHWISE_CONV_2D"
    && ["INT8", "UINT8"].includes(weight.dtype) && Number(weight.quant_scales) === 1).map(({ op }) => op);
  for (const op of asymmetricOps) addSignal(Number(op.index), "asymmetric UINT8 kernel");
  for (const op of perTensorDepthwiseOps) addSignal(Number(op.index), "per-tensor depthwise weights");
  const integrity = analysis?.weight_integrity || {};
  const zeroDetails = Array.isArray(integrity.zero_kernel_slice_details) ? integrity.zero_kernel_slice_details : [];
  const zeroOpIndices = [...new Set(zeroDetails.flatMap((detail) => (detail.consumer_ops || []).map((value) => Number(String(value).match(/#(\d+)/)?.[1])).filter(Number.isInteger)))];
  for (const index of zeroOpIndices) addSignal(index, "exact/near-zero kernel channel");
  const quantizedConstants = tensors.filter((tensor) => tensor.constant_buffer
    && ["INT8", "UINT8"].includes(tensor.dtype) && Number(tensor.quant_scales) > 0);
  const riskCount = localRiskOps.filter((op) => op.quant_risk === "risk").length;
  const warnCount = localRiskOps.length - riskCount;
  const opList = (items) => items.slice(0, 6).map((op) => `#${padOp(op.index)} ${op.name}`).join(", ") || "none";
  const categories = [{
    label: "Operator-local scale / zero-point",
    evidence: `${riskCount} risk / ${warnCount} warn across ${ops.length} graph ops`,
    affected: opList(localRiskOps), tone: riskCount ? "risk" : warnCount ? "review" : "pass",
    firstOpIndex: Number(localRiskOps[0]?.index),
  }];
  if (asymmetricOps.length) categories.push({ label: "Asymmetric UINT8 kernels", evidence: `${asymmetricOps.length} kernel op(s), non-zero weight zero-point`, affected: opList(asymmetricOps), tone: "review", firstOpIndex: Number(asymmetricOps[0].index) });
  if (perTensorDepthwiseOps.length) categories.push({ label: "Per-tensor depthwise weights", evidence: `${perTensorDepthwiseOps.length} op(s), one weight scale across channels`, affected: opList(perTensorDepthwiseOps), tone: "review", firstOpIndex: Number(perTensorDepthwiseOps[0].index) });
  const exactZero = Number(integrity.exact_zero_kernel_slice_count || 0);
  const nearZero = Number(integrity.zero_kernel_slice_count || 0);
  if (exactZero || nearZero) categories.push({ label: "Stored kernel-channel integrity", evidence: `${exactZero} exact-zero / ${nearZero} near-zero slice(s) in ${Number(integrity.zero_kernel_slice_tensors || 0)} tensor(s); inactivity is not inferred`, affected: zeroOpIndices.slice(0, 6).map((index) => `#${padOp(index)}`).join(", ") || "consumer unavailable", tone: "review", firstOpIndex: zeroOpIndices[0] });
  const lowGrid = Number(integrity.low_grid_utilization_tensors || 0);
  const saturated = Number(integrity.saturated_quantized_tensors || 0);
  if (lowGrid || saturated) categories.push({ label: "8-bit constant grid", evidence: `${lowGrid}/${Number(integrity.threshold_eligible_quantized_constant_tensors || quantizedConstants.length)} below heuristic 25% level use; ${saturated} above 1% saturation`, affected: `minimum ${formatPercent(integrity.min_threshold_eligible_grid_utilization ?? integrity.min_grid_utilization)} / endpoint max ${formatPercent(integrity.max_saturation_percent)}`, tone: "review" });
  return { categories, opSignals, quantizedConstants, perAxisConstants: quantizedConstants.filter((tensor) => Number(tensor.quant_scales) > 1) };
}

export function perAxisScaleContractEvidence(tensors = []) {
  const contracts = tensors.map((tensor) => {
    const axis = Number(tensor.quantized_dimension);
    const shape = Array.isArray(tensor.shape) ? tensor.shape.map(Number) : [];
    const axisDimension = Number.isSafeInteger(axis) && axis >= 0 && axis < shape.length && shape[axis] > 0
      ? shape[axis] : null;
    const scaleCount = Number(tensor.quant_scales || 0);
    const decodedScales = Array.isArray(tensor.scale_sample) ? tensor.scale_sample.map(Number) : [];
    return { axis, axisDimension, scaleCount, decodedScales };
  });
  const resolvable = contracts.filter((row) => row.axisDimension != null);
  const matched = resolvable.filter((row) => row.scaleCount === row.axisDimension).length;
  const decodedScales = contracts.flatMap((row) => row.decodedScales);
  const nonPositive = decodedScales.filter((value) => !Number.isFinite(value) || value <= 0).length;
  const axes = [...new Set(contracts.map((row) => Number.isSafeInteger(row.axis) ? row.axis : null).filter((value) => value != null))];
  const maximumSpread = Math.max(0, ...tensors.map((tensor) => Number(tensor.scale_ratio || 0)).filter(Number.isFinite));
  return { tensorCount: tensors.length, resolvableCount: resolvable.length, matchedCount: matched, decodedScaleCount: decodedScales.length, nonPositiveCount: nonPositive, axes, maximumSpread };
}

function perAxisScaleContractSummary(tensors) {
  if (!tensors.length) return null;
  const contract = perAxisScaleContractEvidence(tensors);
  const summary = document.createElement("div");
  summary.className = "target-metrics scale-contract-summary";
  summary.append(
    targetMetric("Per-axis tensors", formatNumber(contract.tensorCount), "DERIVED from serialized constant quantization vectors."),
    targetMetric("Quantized axes", contract.axes.length ? contract.axes.join(", ") : "Unresolved", "Axis values are serialized tensor dimensions, not channel-order guesses."),
    targetMetric("Scale cardinality", contract.resolvableCount ? `${contract.matchedCount}/${contract.resolvableCount} matched` : "Not statically resolved", "Matched means scale_count equals the static shape dimension selected by quantized_dimension."),
    targetMetric("Scale validity", `${contract.nonPositiveCount}/${contract.decodedScaleCount} non-positive`, "Non-positive count uses every decoded serialized scale value."),
    targetMetric("Maximum spread", contract.maximumSpread > 0 ? `${contract.maximumSpread.toExponential(3)}x` : "N/A", "Exact maximum max(scale)/min(positive scale) across assessed per-axis tensors."),
  );
  return summary;
}

function progressiveScaleRows(rows, limit = 10) {
  const wrap = document.createElement("div");
  wrap.className = "scale-scatter-rows";
  wrap.append(...rows.slice(0, limit));
  if (rows.length <= limit) return wrap;
  const more = document.createElement("details");
  more.className = "scale-scatter-more";
  const summary = document.createElement("summary");
  summary.textContent = `View remaining ${formatNumber(rows.length - limit)} operators`;
  const remainder = document.createElement("div");
  remainder.className = "scale-scatter-rows";
  remainder.append(...rows.slice(limit));
  more.append(summary, remainder);
  wrap.append(more);
  return wrap;
}

function nonGraphQuantEvidence(analysis = {}) {
  const format = String(analysis.format || "").toLowerCase();
  if (format === "gguf") {
    const status = analysis.quantization_status || {};
    const total = Number(analysis.tensor_count || 0);
    const block = Number(status.block_quantized_tensor_count || 0);
    const unsupported = Number(status.unsupported_encoding_tensor_count || 0);
    const version = analysis.gguf?.quantization_version;
    return [{
      label: "Serialized GGML tensor encodings",
      evidence: `${block}/${total} tensors use source-pinned block layouts`,
      affected: `${formatNumber(analysis.gguf?.declared_tensor_byte_length || 0)} declared tensor bytes`,
      tone: unsupported ? "risk" : "pass",
    }, {
      label: "Payload range conservation",
      evidence: analysis.gguf?.payload_coverage_status || "not assessed",
      affected: `${unsupported} unsupported encoding tensor(s)`,
      tone: unsupported ? "risk" : "pass",
    }, {
      label: "Quantizer provenance",
      evidence: version == null ? "general.quantization_version is not declared" : `general.quantization_version = ${version}`,
      affected: "A quantization version identifies the encoding generation, not the calibration corpus or full quantizer recipe.",
      tone: block && version == null ? "review" : "pass",
    }];
  }
  if (format === "safetensors") {
    return [{
      label: "SafeTensors dtype contract",
      evidence: `${formatNumber(analysis.tensor_count || 0)} tensor descriptors with exact shape, dtype, and byte cardinality`,
      affected: `${formatNumber(analysis.safetensors?.payload_byte_length || 0)} payload bytes`,
      tone: "pass",
    }, {
      label: "Payload range conservation",
      evidence: analysis.safetensors?.payload_coverage_status || "not assessed",
      affected: `duplicate-key validation: ${analysis.safetensors?.duplicate_key_validation || "not assessed"}`,
      tone: analysis.safetensors?.payload_coverage_status === "complete_without_gaps_or_overlaps" ? "pass" : "risk",
    }, {
      label: "Execution quantization contract",
      evidence: "Not serialized by the SafeTensors container format",
      affected: "Integer or low-bit storage dtype alone does not prove quantized execution, scale, zero-point, or Q/DQ placement.",
      tone: "pass",
    }];
  }
  if (format === "coreml") {
    const status = analysis.quantization_status || {};
    const assessed = status.assessment_status === "assessed";
    return [{
      label: "Core ML numerical payload",
      evidence: assessed ? status.summary || "Weight encodings assessed" : "Not fully assessed for this model representation",
      affected: analysis.coreml?.parser_scope || "Core ML model payload scope is unavailable.",
      tone: assessed ? "pass" : "review",
    }];
  }
  return null;
}

function appendTableCells(row, cells) {
  for (const [value, className = ""] of cells) {
    const cell = document.createElement("td");
    if (className) cell.className = className;
    cell.textContent = String(value ?? "");
    row.append(cell);
  }
}

function boundDisplayLabel(value) {
  if (value === "compute-bound") return "high-intensity";
  if (value === "memory-bound") return "low-intensity";
  if (value === "mixed") return "mixed-intensity";
  return value || "?";
}

export function createPerformanceVisualController({
  elements,
  getContext,
  analyzeForTarget,
  jumpToGraphOp,
}) {
  let targetComparisonCache = null;
  const quantExposureOptions = { metric: "macs", groupBy: "stage" };

  // ── Per-panel inline detail card ─────────────────────────────────────────
  function showPanelDetail(triggerEl, rows, opIdx) {
    const panel = triggerEl.closest(".perf-panel");
    if (!panel) return;
    const prev = panel.querySelector(".panel-detail-card");
    const isSame = prev && prev.dataset.opIdx === String(opIdx ?? "stage");
    panel.querySelectorAll(".panel-detail-active").forEach((el) => el.classList.remove("panel-detail-active"));
    if (prev) prev.remove();
    if (isSame) return;

    triggerEl.classList.add("panel-detail-active");

    const card = document.createElement("div");
    card.className = "panel-detail-card";
    card.dataset.opIdx = String(opIdx ?? "stage");

    const head = document.createElement("div");
    head.className = "panel-detail-head";
    const closeBtn = document.createElement("button");
    closeBtn.className = "panel-detail-close";
    closeBtn.textContent = "×";
    closeBtn.addEventListener("click", () => {
      card.remove();
      triggerEl.classList.remove("panel-detail-active");
    });
    head.append(closeBtn);

    const grid = document.createElement("div");
    grid.className = "panel-detail-grid";
    for (const [label, value, tone] of rows) {
      const lEl = document.createElement("span");
      lEl.className = "panel-detail-label";
      lEl.textContent = label;
      const vEl = document.createElement("span");
      vEl.className = `panel-detail-value${tone ? ` panel-detail-${tone}` : ""}`;
      vEl.textContent = value;
      grid.append(lEl, vEl);
    }
    card.append(head, grid);

    if (opIdx != null) {
      const footer = document.createElement("div");
      footer.className = "panel-detail-footer";
      const link = document.createElement("button");
      link.className = "panel-detail-graph-link";
      link.textContent = "→ Open in Graph Explorer";
      link.addEventListener("click", () => { card.remove(); jumpToGraphOp(opIdx); });
      footer.append(link);
      card.append(footer);
    }

    panel.append(card);
    if (window.matchMedia?.("(max-width: 720px)").matches) {
      requestAnimationFrame(() => card.scrollIntoView({ behavior: "smooth", block: "nearest" }));
    }
  }

  function render(analysis) {
    renderOnDeviceLlmView(elements.llmEvidencePanel, analysis);
    renderMacDistribution(analysis);
    renderBottleneckContribution(analysis);
    renderTargetComparison(analysis);
    renderXnnpackChainFlow(analysis);
    renderXnnpackFallbackMap(analysis);
    renderXnnpackBreakTable(analysis);
    renderQuantRiskHeatmap(analysis);
    renderQuantizationExposureMap(analysis);
    renderQuantScaleScatter(analysis);
    renderQuantRiskTable(analysis);
    renderQuantHoles(analysis);
    renderRooflineChart(analysis);
    renderStageMemoryMix(analysis);
    renderPerfTimeline(analysis);
    updateVisibility();
  }

  function renderQuantizationExposureMap(analysis) {
    if (!elements.quantExposureMap) return;
    const serialized = serializedTensorPresentation(analysis);
    const evidence = quantSummaryEvidence(analysis);
    const presentation = buildQuantizationExposurePresentation(
      analysis,
      quantExposureOptions,
      serialized,
      evidence.opSignals,
    );
    Object.assign(quantExposureOptions, {
      metric: presentation.metricId,
      groupBy: presentation.groupBy,
    });
    renderEvidenceTreemap(elements.quantExposureMap, presentation, {
      onControl: (name, value) => {
        quantExposureOptions[name] = value;
        renderQuantizationExposureMap(analysis);
      },
      onSelect: (item, trigger) => {
        if (item.kind === "op") {
          showPanelDetail(trigger, [
            ["Operator", item.label],
            [presentation.metricLabel, formatMetricForExposure(item.value, presentation.unit)],
            ["Group", item.groupLabel],
            ["Quantization evidence", item.detail],
            ["Area share", presentation.total ? formatPercent(item.value / presentation.total) : "N/A"],
          ], item.index);
        } else {
          showPanelDetail(trigger, [
            ["Tensor", item.label],
            [presentation.metricLabel, formatMetricForExposure(item.value, presentation.unit)],
            ["Group", item.groupLabel],
            ["Storage evidence", item.detail],
            ["Area share", presentation.total ? formatPercent(item.value / presentation.total) : "N/A"],
          ]);
        }
      },
    });
  }

  function formatMetricForExposure(value, unit) {
    if (unit === "B") return `${formatBytes(value)} (${formatNumber(value)} B)`;
    if (unit === "us") return formatUs(value);
    return `${formatNumber(value)} ${unit}`;
  }

  function resetTargetComparisonCache() {
    targetComparisonCache = null;
  }

  function buildTargetComparisonRows(analysis) {
    const context = getContext();
    if ((analysis.format || "tflite") !== "tflite" || !context.currentModelBytes?.length) {
      return [summarizeTargetComparison(analysis)];
    }
    return getTargetComparisonRows(analysis, availableTargetComparisonIds());
  }

  function updateVisibility() {
    const { activeAuditTab, current } = getContext();
    const format = String(current?.format || "tflite").toLowerCase();
    const onnx = format === "onnx";
    const copy = performanceVisualCopyFor(activeAuditTab, format);
    elements.perfVisualTitle.textContent = onnx && activeAuditTab === "overview" ? "Static Structure Visuals" : copy.title;
    elements.perfVisualSubtitle.textContent = onnx && activeAuditTab === "overview"
      ? "Assessed MAC concentration, static intensity posture, quantization signals, and stage memory estimates."
      : copy.subtitle;
    elements.perfVisualStatus.textContent = onnx && activeAuditTab === "overview" ? "ONNX static assessment" : copy.status;
    for (const panel of elements.visualPanels) {
      const scopes = String(panel.dataset.visualScope || "")
        .split(/\s+/)
        .filter(Boolean);
      const formatScopes = String(panel.dataset.formatScope || "")
        .split(/\s+/)
        .filter(Boolean);
      panel.hidden = !scopes.includes(activeAuditTab) || Boolean(formatScopes.length && !formatScopes.includes(format));
      if (panel.id === "performanceMacPanel") panel.classList.toggle("scope-wide", activeAuditTab === "stage");
    }
  }

  function renderMacDistribution(analysis) {
    const { ops, totalMacs, top, otherMacs } = macDistributionData(analysis);
    if (!ops.length || !totalMacs) {
      renderVisualEmpty(elements.macFlame, "No MAC-heavy operators were detected for this model.");
      elements.macTopList.replaceChildren(visualListItem("MAC distribution", "N/A", "neutral"));
      return;
    }
    const segments = top.map((op) => ({
      label: `#${padOp(op.index)}`,
      detail: `${op.name} / ${formatPercent(Number(op.macs || 0) / totalMacs)} MACs`,
      value: Number(op.macs || 0),
      tone: boundTone(op.static_bound_guess),
      opIndex: op.index,
    }));
    if (otherMacs > 0) {
      segments.push({
        label: "Other",
        detail: `${formatPercent(otherMacs / totalMacs)} remaining MACs`,
        value: otherMacs,
        tone: "neutral",
      });
    }
    renderFlameStrip(elements.macFlame, segments, totalMacs);
    elements.macTopList.replaceChildren(
      ...top.slice(0, 5).map((op) =>
        visualListItem(
          `#${padOp(op.index)} ${op.name}`,
          `${formatNumber(op.macs)} MACs / ${formatPercent(Number(op.macs || 0) / totalMacs)}`,
          boundTone(op.static_bound_guess),
        ),
      ),
    );
  }

  function renderBottleneckContribution(analysis) {
    const { rows: allEstimates, top: estimates, totalUs: total, otherUs } = bottleneckDistributionData(
      analysis,
      analysis.target_profile || getContext().selectedTargetProfile,
    );
    if (!allEstimates.length || !total) {
      renderVisualEmpty(elements.bottleneckFlame, "No static bottleneck estimate is available for this model.");
      elements.bottleneckList.replaceChildren(visualListItem("Modeled bottleneck", "N/A", "neutral"));
      return;
    }
    const segments = estimates.map(({ op, estimate }) => ({
      label: `#${padOp(op.index)}`,
      detail: `${op.name} / ${formatPercent(estimate.totalUs / total)} modeled contribution / ${formatUs(estimate.totalUs)} estimated / ${estimate.dominantLabel}`,
      value: estimate.totalUs,
      tone: estimate.dominantTone,
      opIndex: op.index,
    }));
    if (otherUs > 0) {
      segments.push({
        label: "Other",
        detail: `${formatUs(otherUs)} remaining static estimate`,
        value: otherUs,
        tone: "neutral",
      });
    }
    renderFlameStrip(elements.bottleneckFlame, segments, total);
    elements.bottleneckList.replaceChildren(
      ...estimates.slice(0, 5).map(({ op, estimate }) =>
        visualListItem(
          `#${padOp(op.index)} ${op.name}`,
          `${formatPercent(estimate.totalUs / total)} modeled contribution · ${formatUs(estimate.totalUs)} estimated · ${estimate.dominantLabel}`,
          estimate.dominantTone,
        ),
      ),
    );
  }

  function renderTargetComparison(analysis) {
    const context = getContext();
    if ((analysis.format || "tflite") !== "tflite") {
      elements.targetCompareGrid.replaceChildren(targetCompareMessage("Target comparison is available for TFLite artifacts."));
      return;
    }
    const availableIds = availableTargetComparisonIds();
    if (!availableIds.length) {
      elements.targetCompareGrid.replaceChildren(targetCompareMessage("Target profiles are not loaded."));
      return;
    }
    if (!context.currentModelBytes?.length) {
      // No raw bytes — show only the current-target card
      const row = summarizeTargetComparison(analysis);
      elements.targetCompareGrid.replaceChildren(targetCompareCard(row));
      return;
    }
    elements.targetCompareGrid.replaceChildren(...getTargetComparisonRows(analysis, availableIds).map((row) => targetCompareCard(row)));
  }

  function availableTargetComparisonIds() {
    const context = getContext();
    return deploymentFrontierTargetIds(context.targetProfiles, context.selectedTargetId);
  }

  function getTargetComparisonRows(analysis, availableIds = availableTargetComparisonIds()) {
    if (!availableIds.length) return [];
    const frontierTargets = analysis?.deployment_frontier?.targets;
    if (Array.isArray(frontierTargets)
      && availableIds.every((targetId) => frontierTargets.some((target) => target.target_id === targetId))) {
      const rows = availableIds.map((targetId) => targetComparisonRowFromFrontier(
        frontierTargets.find((target) => target.target_id === targetId),
      ));
      const changedBoundOps = (analysis?.deployment_frontier?.ops || [])
        .filter((op) => new Set(op.bound_classes || []).size > 1).length;
      return rows.map((row) => ({
        ...row,
        intensitySensitivityDetail: changedBoundOps
          ? `${changedBoundOps} op(s) change roofline class across the compared target thresholds.`
          : `0 ops change roofline class across ridges ${rows.map((item) => formatNumber(item.ridge)).join(" / ")} ops/B; identical low-intensity ratios are a deterministic threshold-gap result.`,
      }));
    }
    const cacheKey = targetComparisonCacheKey(analysis, availableIds);
    if (!targetComparisonCache || targetComparisonCache.key !== cacheKey) {
      targetComparisonCache = {
        key: cacheKey,
        rows: availableIds.map((targetId) => summarizeTargetForComparison(analysis, targetId)),
      };
    }
    return targetComparisonCache.rows;
  }

  function targetComparisonRowFromFrontier(target) {
    return summarizeFrontierTargetComparison(target);
  }

  function targetComparisonCacheKey(analysis, availableIds) {
    const context = getContext();
    return [
      context.current?.model_sha256 || context.currentFilename || analysis.filename || "model",
      context.currentModelBytes?.length || 0,
      analysis.target_profile?.id || context.selectedTargetId,
      availableIds.join(","),
    ].join(":");
  }

  function summarizeTargetForComparison(analysis, targetId) {
    const context = getContext();
    try {
      const compared = targetId === (analysis.target_profile?.id || context.selectedTargetId)
        ? analysis
        : analyzeForTarget(context.currentModelBytes, context.currentFilename || analysis.filename || "model.tflite", targetId);
      return summarizeTargetComparison(compared);
    } catch (error) {
      const profile = context.targetProfiles.find((item) => item.id === targetId) || { id: targetId, label: targetId };
      return { id: targetId, label: profile.label || targetId, error: shortError(error) };
    }
  }

  function chainClassToModifier(chainClass) {
    if (["high-MAC-share candidate segment", "high-compute delegated chain"].includes(chainClass)) return "high-compute";
    if (["zero-MAC candidate segment", "zero-mac structural chain"].includes(chainClass)) return "zero-mac";
    return "low-compute";
  }

  function xnnBreakClassLabel(value) {
    return ({
      "high-adjacent-mac-exposure": "high adjacent-MAC exposure",
      "memory-traffic": "memory-traffic",
      "structural-zero-mac": "structural zero-MAC",
      "zero-modeled-mac-nonstructural": "zero-modeled-MAC non-structural",
      "low-impact-nonstructural": "low-impact non-structural",
    })[value] || value || "unclassified";
  }

  function xnnBreakClassDisplayLabel(value) {
    return ({
      "high-adjacent-mac-exposure": "high-exposure break",
      "memory-traffic": "memory-traffic break",
      "structural-zero-mac": "structural zero-MAC break",
      "zero-modeled-mac-nonstructural": "zero-MAC non-structural break",
      "low-impact-nonstructural": "low-impact non-structural break",
    })[value] || `${value || "unclassified"} break`;
  }

  function isStructuralViewBreak(op) {
    return ["RESHAPE", "SQUEEZE", "EXPAND_DIMS", "SHAPE"].includes(String(op?.name || ""));
  }

  function xnnBreakLabels(op) {
    const labels = [xnnBreakClassDisplayLabel(op?.xnnpack_break_class)];
    if (isStructuralViewBreak(op) && op?.xnnpack_break_class !== "structural-zero-mac") labels.push("structural zero-MAC break");
    if (!isStructuralViewBreak(op) && Number(op?.macs || 0) === 0 && op?.xnnpack_break_class !== "zero-modeled-mac-nonstructural") labels.push("zero-MAC non-structural break");
    return labels;
  }

  function renderXnnpackChainFlow(analysis) {
    const chains = Array.isArray(analysis.xnnpack_chains) ? analysis.xnnpack_chains : [];
    const ops = Array.isArray(analysis.ops) ? analysis.ops : [];
    const breaks = ops.filter((op) => op.xnnpack_chain_break);
    if (!chains.length) {
      elements.chainFlow.replaceChildren(targetCompareMessage("No conditionally delegatable XNNPACK segment was predicted for this target."));
      return;
    }

    const breakCount = (value) => breaks.filter((op) => op.xnnpack_break_class === value).length;
    const highBreaks = breakCount("high-adjacent-mac-exposure");
    const memBreaks = breakCount("memory-traffic");
    const structBreaks = breaks.filter(isStructuralViewBreak).length;
    const zeroMacNonstructuralBreaks = breaks.filter((op) => !isStructuralViewBreak(op) && Number(op.macs || 0) === 0).length;
    const lowImpactNonstructuralBreaks = breakCount("low-impact-nonstructural");
    const buildRisks = analysis.delegation_repair?.runtime_build_risks || [];
    const summary = document.createElement("div");
    summary.className = "chain-summary-row";
    const summaryBadge = (text, className = "", title = "") => {
      const badge = document.createElement("span");
      if (className) badge.className = className;
      badge.textContent = text;
      if (title) badge.title = title;
      return badge;
    };
    summary.append(summaryBadge(`${chains.length} predicted delegate segment${chains.length === 1 ? "" : "s"}`));
    if (highBreaks) summary.append(summaryBadge(`${highBreaks} high-exposure break${highBreaks === 1 ? "" : "s"}`, "break-badge risk", "High adjacent-MAC exposure: the break is adjacent to at least one predicted candidate segment with high MAC share; the break op itself is not classified as compute-heavy."));
    if (memBreaks) summary.append(summaryBadge(`${memBreaks} memory-traffic break${memBreaks === 1 ? "" : "s"}`, "break-badge warn"));
    if (structBreaks) summary.append(summaryBadge(`${structBreaks} structural zero-MAC break${structBreaks === 1 ? "" : "s"}`, "break-badge neutral", "Operator anatomy count; independent of adjacent-segment exposure."));
    if (zeroMacNonstructuralBreaks) summary.append(summaryBadge(`${zeroMacNonstructuralBreaks} zero-MAC non-structural break${zeroMacNonstructuralBreaks === 1 ? "" : "s"}`, "break-badge neutral", "Operator anatomy count; independent of adjacent-segment exposure."));
    if (lowImpactNonstructuralBreaks) summary.append(summaryBadge(`${lowImpactNonstructuralBreaks} low-impact non-structural break${lowImpactNonstructuralBreaks === 1 ? "" : "s"}`, "break-badge neutral"));
    if (buildRisks.length) summary.append(summaryBadge(`${buildRisks.length} runtime build condition${buildRisks.length === 1 ? "" : "s"} not artifact-bound`, "break-badge risk"));
    if (buildRisks.length) {
      summary.title = `Static compatibility is conditional on: ${buildRisks.map((risk) => risk.required_build_configuration).join(" / ")}. These build settings are not embedded in the model artifact.`;
    }

    const lane = document.createElement("div");
    lane.className = "chain-lane";
    const sortedChains = [...chains].sort((a, b) => Number(a.first_op || 0) - Number(b.first_op || 0));
    sortedChains.forEach((chain, index) => {
      const block = document.createElement("button");
      block.type = "button";
      const mod = chainClassToModifier(chain.chain_class || "");
      block.className = `chain-block chain-${mod}`;
      const macShare = Number(chain.mac_percent || 0);
      block.style.flexGrow = String(Math.max(0.4, macShare * 12));
      const inner = document.createElement("strong");
      inner.textContent = `C${chain.id}`;
      const opsSpan = document.createElement("span");
      const opCount = chain.op_count || (isFinite(Number(chain.last_op) - Number(chain.first_op)) ? Number(chain.last_op) - Number(chain.first_op) + 1 : "?");
      opsSpan.textContent = `${opCount} op${opCount === 1 ? "" : "s"}`;
      const macSpan = document.createElement("em");
      macSpan.textContent = formatPercent(macShare) + " MACs";
      block.append(inner, opsSpan, macSpan);
      attachTip(block, [
        ["Predicted segment", `C${chain.id}: ops ${chain.first_op}–${chain.last_op}`],
        ["Class", chain.chain_class || "candidate segment"],
        ["MACs", `${formatPercent(macShare)} (${formatNumber(chain.macs || 0)})`],
        ["Ops", String(chain.op_count || "?")],
        ["Kernel", chain.target_hint || "-"],
      ]);
      const firstOp = ops.find((op) => op.index === Number(chain.first_op));
      block.addEventListener("click", (e) => showPanelDetail(e.currentTarget, [
        ["Predicted segment", `C${chain.id}: ops ${chain.first_op}–${chain.last_op}`],
        ["Class",   chain.chain_class || "candidate segment", chain.chain_class === "fallback" ? "fallback" : chain.chain_class === "structural" ? "overhead" : chain.chain_class === "mixed" ? "memory" : "compute"],
        ["MACs",    `${formatPercent(macShare)} (${formatNumber(chain.macs || 0)})`],
        ["Ops",     String(chain.op_count || "?")],
        ["Kernel",  chain.target_hint || "-"],
      ], firstOp?.index));
      lane.append(block);
      const nextChain = sortedChains[index + 1];
      if (nextChain) {
        const between = breaks.filter((op) => op.index > Number(chain.last_op) && op.index < Number(nextChain.first_op));
        lane.append(chainBreakMarker(between));
      }
    });
    const finalBreaks = breaks.filter((op) => op.index > Number(sortedChains.at(-1)?.last_op || -1));
    if (finalBreaks.length) lane.append(chainBreakMarker(finalBreaks));

    // Reason breakdown table
    const reasonMap = new Map();
    for (const op of breaks) {
      const addReason = (cls) => {
        const key = `${op.name}|${cls}`;
        const entry = reasonMap.get(key) || { name: op.name, cls, count: 0 };
        entry.count++;
        reasonMap.set(key, entry);
      };
      addReason(op.xnnpack_break_class || "break");
      if (isStructuralViewBreak(op) && op.xnnpack_break_class !== "structural-zero-mac") addReason("structural-zero-mac");
      if (!isStructuralViewBreak(op) && Number(op.macs || 0) === 0 && op.xnnpack_break_class !== "zero-modeled-mac-nonstructural") addReason("zero-modeled-mac-nonstructural");
    }
    let reasonBlock = null;
    if (reasonMap.size) {
      reasonBlock = document.createElement("div");
      reasonBlock.className = "chain-reason-list";
      const sorted = [...reasonMap.values()].sort((a, b) => b.count - a.count);
      for (const r of sorted.slice(0, 8)) {
        const row = document.createElement("div");
        row.className = `chain-reason-row ${r.cls === "high-adjacent-mac-exposure" ? "risk-text" : r.cls === "memory-traffic" ? "warn-text" : "muted-text"}`;
        row.textContent = `${r.name} — ${xnnBreakClassDisplayLabel(r.cls)} (×${r.count})`;
        row.title = r.cls === "high-adjacent-mac-exposure"
          ? "High adjacent-MAC exposure describes the neighboring predicted candidate segment, not the break operator's own MAC count."
          : xnnBreakClassLabel(r.cls);
        reasonBlock.append(row);
      }
    }

    const buildWarning = document.createElement("div");
    buildWarning.className = "chain-build-warning";
    buildWarning.hidden = buildRisks.length === 0;
    if (buildRisks.length) {
      const baseline = Number(buildRisks[0]?.baseline_conditionally_delegatable_op_count || 0);
      const conditions = [...new Set(buildRisks.map((risk) => risk.required_build_configuration).filter(Boolean))];
      buildWarning.textContent = `Conditional assignment: ${baseline}/${ops.length} XNNPACK candidates depend on an unbound selected-runtime build condition (${conditions.join("; ")}). This is source-pinned eligibility, not executed placement.`;
    }
    elements.chainFlow.replaceChildren(buildWarning, summary, lane, ...(reasonBlock ? [reasonBlock] : []));
  }

  // ── XNNPACK: Delegation Map ────────────────────────────────────────────────
  function xnnOpTone(op) {
    if (op.xnnpack_chain_break) {
      if (op.xnnpack_break_class === "high-adjacent-mac-exposure") return "xnn-compbreak";
      if (op.xnnpack_break_class === "memory-traffic") return "xnn-membreak";
      if (op.xnnpack_break_class === "structural-zero-mac") return "xnn-structural-break";
      return "xnn-nonstructural-break";
    }
    if (op.xnnpack_chain_id >= 0) {
      return op.xnnpack_chain_role === "delegated" && op.macs === 0 ? "xnn-structural" : "xnn-delegated";
    }
    return "fallback";
  }

  function renderXnnpackFallbackMap(analysis) {
    const ops = Array.isArray(analysis.ops) ? analysis.ops : [];
    const buildRisks = analysis.delegation_repair?.runtime_build_risks || [];
    if (!ops.length) {
      if (elements.xnnpackFallbackMap) elements.xnnpackFallbackMap.replaceChildren(targetCompareMessage("No operators available."));
      return;
    }
    const counts = { "xnn-delegated": 0, "xnn-structural": 0, "xnn-structural-break": 0, "xnn-nonstructural-break": 0, "xnn-membreak": 0, "xnn-compbreak": 0, fallback: 0 };
    const tiles = ops.map((op) => {
      const tone = xnnOpTone(op);
      counts[tone] = (counts[tone] || 0) + 1;
      const tile = document.createElement("button");
      tile.type = "button";
      tile.className = `quant-tile ${tone}`;
      tile.textContent = String(op.index);
      attachTip(tile, [
        ["Op", `#${padOp(op.index)} ${op.name}`],
        ["XNNPACK", op.xnnpack_supported ? "supported" : "not supported"],
        ["Status", op.xnnpack_chain_break ? xnnBreakLabels(op).join("; ") : op.xnnpack_chain_id != null ? `predicted segment C${op.xnnpack_chain_id}` : "fallback"],
        ["Role", op.xnnpack_chain_role || "-"],
      ]);
      tile.addEventListener("click", (e) => showPanelDetail(e.currentTarget, [
        ["Op",      `#${padOp(op.index)} ${op.name}`],
        ["XNNPACK", op.xnnpack_supported ? "supported" : "not supported", op.xnnpack_supported ? "compute" : "fallback"],
        ["Predicted segment", op.xnnpack_chain_id >= 0 ? `C${op.xnnpack_chain_id}` : "fallback"],
        ["Role",    op.xnnpack_chain_role || "-"],
        ["Break",   op.xnnpack_chain_break ? `yes — ${xnnBreakLabels(op).join("; ")}` : "no", op.xnnpack_chain_break ? "overhead" : ""],
        ["MACs",    formatNumber(op.macs)],
      ], op.index));
      return tile;
    });
    if (elements.xnnpackFallbackMap) elements.xnnpackFallbackMap.replaceChildren(...tiles);
    if (elements.xnnpackFallbackCount) {
      const delegated = ops.filter((op) => Number(op.xnnpack_chain_id) >= 0).length;
      const breakOps = ops.filter((op) => op.xnnpack_chain_break);
      const structuralBreaks = breakOps.filter(isStructuralViewBreak).length;
      const highExposureBreaks = breakOps.filter((op) => op.xnnpack_break_class === "high-adjacent-mac-exposure").length;
      const zeroMacNonstructuralBreaks = breakOps.filter((op) => !isStructuralViewBreak(op) && Number(op.macs || 0) === 0).length;
      const memoryTrafficBreaks = breakOps.filter((op) => op.xnnpack_break_class === "memory-traffic").length;
      const lowImpactBreaks = breakOps.filter((op) => op.xnnpack_break_class === "low-impact-nonstructural").length;
      const otherNonstructuralBreaks = breakOps.length - structuralBreaks - zeroMacNonstructuralBreaks;
      const boundaryEdges = Number(analysis.predicted_partition_boundaries?.edge_count || 0);
      const exposure = [
        [highExposureBreaks, "high adjacent-MAC"],
        [memoryTrafficBreaks, "memory-traffic"],
        [lowImpactBreaks, "low-impact"],
      ].filter(([count]) => count).map(([count, label]) => `${count} ${label}`).join(" / ") || "0 flagged";
      const anatomy = `${zeroMacNonstructuralBreaks} zero-MAC non-structural / ${structuralBreaks} structural / ${otherNonstructuralBreaks} other non-structural`;
      const eligibility = buildRisks.length
        ? "CONDITIONALLY DELEGATABLE UNDER THE STATED XNNPACK BUILD CONDITION"
        : "CONDITIONALLY DELEGATABLE";
      elements.xnnpackFallbackCount.textContent = `${delegated}/${ops.length} ${eligibility} · ${breakOps.length} BREAK OP${breakOps.length === 1 ? "" : "S"} · EXPOSURE ${exposure} · OPERATOR ANATOMY ${anatomy} · ${boundaryEdges} TENSOR EDGES`;
    }
  }

  // ── XNNPACK: Break Cost Table ──────────────────────────────────────────────
  function renderXnnpackBreakTable(analysis) {
    const ops = Array.isArray(analysis.ops) ? analysis.ops : [];
    const boundaryEdges = analysis.predicted_partition_boundaries?.edges || [];
    const incidentEvidence = (op) => {
      const edges = boundaryEdges.filter((edge) => Number(edge.producer_op_index) === Number(op.index) || Number(edge.consumer_op_index) === Number(op.index));
      const assessed = edges.filter((edge) => Number.isFinite(Number(edge.payload_bytes)));
      return {
        edgeCount: edges.length,
        assessedCount: assessed.length,
        bytes: assessed.reduce((sum, edge) => sum + Number(edge.payload_bytes), 0),
        complete: edges.length === assessed.length,
      };
    };
    const breakOps = ops
      .filter((op) => op.xnnpack_chain_break)
      .map((op) => ({ op, incident: incidentEvidence(op) }))
      .sort((left, right) =>
        Number(right.op.chain_break_impact_mac_percent || 0) - Number(left.op.chain_break_impact_mac_percent || 0)
        || Number(right.incident.complete ? right.incident.bytes : -1) - Number(left.incident.complete ? left.incident.bytes : -1)
        || Number(left.op.index) - Number(right.op.index));

    if (!elements.xnnpackBreakTable) return;
    if (!breakOps.length) {
      elements.xnnpackBreakTable.replaceChildren(targetCompareMessage("No predicted partition breaks detected for this target."));
      return;
    }

    const table = document.createElement("table");
    table.className = "break-cost-table";
    const thead = document.createElement("thead");
    thead.innerHTML = "<tr><th>#</th><th>Op</th><th>Exposure / operator anatomy</th><th>Adjacent candidate-segment MAC share</th><th>Incident logical payload</th><th>Boundary tensor edges</th></tr>";
    table.append(thead);
    const tbody = document.createElement("tbody");
    for (const item of breakOps) {
      const { op, incident } = item;
      const tr = document.createElement("tr");
      tr.className = "clickable-row";
      const macPct = Number(op.chain_break_impact_mac_percent || 0);
      const cls = op.xnnpack_break_class || "unknown";
      const clsTone = cls === "high-adjacent-mac-exposure" ? "risk-text" : cls === "memory-traffic" ? "warn-text" : "muted-text";
      appendTableCells(tr, [
        [`#${padOp(op.index)}`],
        [op.name],
        [xnnBreakLabels(op).join("; "), clsTone],
        [macPct > 0 ? formatPercent(macPct) : "-", "numeric"],
        [incident.complete ? formatBytes(incident.bytes) : `partial ${formatBytes(incident.bytes)}`, "numeric"],
        [formatNumber(incident.edgeCount), "numeric"],
      ]);
      tr.addEventListener("click", () => jumpToGraphOp(op.index));
      tbody.append(tr);
    }
    table.append(tbody);
    const note = document.createElement("p");
    note.className = "repair-method-line";
    note.textContent = "Ranked by graph-derived adjacent candidate-segment MAC share, then complete incident logical boundary payload. The repeated partition-planning setup range is an unmeasured cold-start profile prior and is not used for ranking.";
    elements.xnnpackBreakTable.replaceChildren(table, note);
  }

  function renderQuantScaleScatter(analysis) {
    const format = String(analysis.format || "").toLowerCase();
    if (format === "gguf" || format === "safetensors") {
      const message = format === "gguf"
        ? "Not applicable: GGUF block encodings use type-specific block layouts. They do not expose affine per-axis scale vectors through the GGUF tensor directory."
        : "Not applicable: SafeTensors descriptors contain dtype, shape, and byte ranges, but no standardized affine scale or zero-point fields.";
      elements.quantScaleScatter?.replaceChildren(targetCompareMessage(message));
      return;
    }
    if (format === "coreml") {
      if (analysis.coreml?.model_type === "mlProgram") {
        const transforms = (analysis.ops || []).filter((op) => op.quantization_state === "serialized_quantization_transform");
        const perAxis = [];
        for (const op of transforms) {
          const scaleBinding = op.mil_input_bindings?.scale?.find((item) => Number.isSafeInteger(item.tensor_index));
          const tensor = scaleBinding ? analysis.tensors?.[scaleBinding.tensor_index] : null;
          const integrity = tensor?.numerical_integrity || {};
          const count = Number(integrity.decoded_value_count || 0);
          const minimum = Number(integrity.finite_min);
          const maximum = Number(integrity.finite_max);
          if (count > 1 && Number.isFinite(minimum) && minimum > 0 && Number.isFinite(maximum) && maximum >= minimum) {
            perAxis.push({ op, tensor, count, ratio: maximum / minimum, minimum, maximum });
          }
        }
        if (!transforms.length) {
          elements.quantScaleScatter?.replaceChildren(targetCompareMessage("Not applicable: the decoded MIL SSA graph contains no explicit quantize, dequantize, or constexpr compression transform."));
        } else if (!perAxis.length) {
          const unbound = transforms.filter((op) => op.mil_input_bindings?.scale?.some((item) => Number.isSafeInteger(item.tensor_index)
            && !analysis.tensors?.[item.tensor_index]?.numerical_integrity?.status?.startsWith("assessed"))).length;
          elements.quantScaleScatter?.replaceChildren(targetCompareMessage(
            `No assessed multi-value scale vector is serialized for ${transforms.length} explicit MIL quantization/compression transform(s). ${unbound ? `${unbound} transform scale payload(s) remain package-unbound or unsupported.` : "Scalar scales and transforms without affine scale inputs do not define per-axis spread."}`,
          ));
        } else {
          const rows = [];
          const maxRatio = Math.max(1, ...perAxis.map((row) => row.ratio));
          for (const item of perAxis) {
            const row = document.createElement("div");
            row.className = "ss-row";
            const label = document.createElement("span");
            label.className = "ss-label";
            label.textContent = `#${padOp(item.op.index)} ${item.op.mil_operation_type}`;
            const barWrap = document.createElement("div");
            barWrap.className = "ss-bar-wrap";
            const bar = document.createElement("div");
            bar.className = "ss-bar mixed";
            bar.style.width = `${Math.max(2, item.ratio / maxRatio * 100).toFixed(1)}%`;
            barWrap.append(bar);
            const value = document.createElement("span");
            value.className = "ss-value";
            value.textContent = `${item.ratio.toFixed(3)}x / ${formatNumber(item.count)} scales`;
            attachTip(row, [["Scale tensor", item.tensor.name], ["Finite range", `${item.minimum} to ${item.maximum}`], ["Source", item.tensor.numerical_integrity.payload_sha256 ? "package blob payload" : "MIL immediate value"]]);
            row.append(label, barWrap, value);
            rows.push(row);
          }
          elements.quantScaleScatter?.replaceChildren(progressiveScaleRows(rows));
        }
        return;
      }
      const weights = (analysis.coreml?.neural_network?.layers || []).flatMap((layer) =>
        (layer.weights || []).map((weight) => ({ ...weight, layer })));
      const quantized = weights.filter((weight) => weight.quantization);
      const perAxis = quantized.filter((weight) => Number(weight.quantization.scale_count || 0) > 1);
      if (analysis.quantization_status?.assessment_status !== "assessed") {
        elements.quantScaleScatter?.replaceChildren(targetCompareMessage("Not completely assessed: one or more Core ML layer types lack a decoded WeightParams field map."));
      } else if (!quantized.length) {
        elements.quantScaleScatter?.replaceChildren(targetCompareMessage(`Not applicable: 0/${weights.length} decoded Core ML WeightParams use quantized storage.`));
      } else if (!perAxis.length) {
        elements.quantScaleScatter?.replaceChildren(targetCompareMessage(`No per-axis spread: all ${quantized.length} quantized Core ML WeightParams use one linear scale or a lookup-table scheme.`));
      } else {
        const rows = [];
        const maximum = Math.max(1, ...perAxis.map((weight) => Number(weight.quantization.scale_max) / Number(weight.quantization.scale_min)));
        for (const weight of perAxis) {
          const ratio = Number(weight.quantization.scale_max) / Number(weight.quantization.scale_min);
          const row = document.createElement("div");
          row.className = "ss-row";
          const label = document.createElement("span");
          label.className = "ss-label";
          label.textContent = `#${padOp(weight.layer.index)} ${weight.layer.type} / ${weight.role}`;
          const barWrap = document.createElement("div");
          barWrap.className = "ss-bar-wrap";
          const bar = document.createElement("div");
          bar.className = "ss-bar mixed";
          bar.style.width = `${Math.max(2, ratio / maximum * 100).toFixed(1)}%`;
          barWrap.append(bar);
          const value = document.createElement("span");
          value.className = "ss-value";
          value.textContent = `${ratio.toFixed(3)}x / ${weight.quantization.scale_count} scales`;
          row.append(label, barWrap, value);
          rows.push(row);
        }
        elements.quantScaleScatter?.replaceChildren(progressiveScaleRows(rows));
      }
      return;
    }
    const ops = Array.isArray(analysis.ops) ? analysis.ops : [];
    const evidence = quantSummaryEvidence(analysis);
    const contractSummary = perAxisScaleContractSummary(evidence.perAxisConstants);
    const meaningful = ops
      .filter((op) => op.quant_scale_ratio_meaningful && Number(op.quant_scale_ratio || 0) > 1)
      .sort((a, b) => Number(b.quant_scale_ratio || 0) - Number(a.quant_scale_ratio || 0));

    if (!elements.quantScaleScatter) return;
    if (!meaningful.length) {
      const count = evidence.quantizedConstants.length;
      const message = targetCompareMessage(count && !evidence.perAxisConstants.length
        ? `Not applicable: all ${count} decoded 8-bit constants use one scale per tensor. No per-axis spread exists; analysis is complete for this contract.`
        : evidence.perAxisConstants.length
          ? "Per-axis tensor contracts were decoded, but no operator-local scale-spread projection is available. The contract summary above remains authoritative."
          : "No assessed per-axis scale spread is available for this artifact.");
      elements.quantScaleScatter.replaceChildren(...(contractSummary ? [contractSummary] : []), message);
      return;
    }

    const maxLog = Math.log10(Math.max(...meaningful.map((op) => Number(op.quant_scale_ratio || 1))));
    const RISK_LOG = Math.log10(1e6);
    const WARN_LOG = Math.log10(1e3);

    const rows = [];

    // B3/W7: cap threshold markers to [0,100%] when maxLog is tiny
    const riskPct = maxLog > 0 ? Math.min(100, RISK_LOG / maxLog * 100).toFixed(1) : "100";
    const warnPct = maxLog > 0 ? Math.min(100, WARN_LOG / maxLog * 100).toFixed(1) : "100";

    for (const op of meaningful) {
      const ratio = Number(op.quant_scale_ratio || 1);
      const weight = analysis.tensors?.[Number(op.inputs?.[1])];
      const axis = Number(weight?.quantized_dimension);
      const axisDimension = Number.isSafeInteger(axis) && axis >= 0 && axis < (weight?.shape?.length || 0)
        ? Number(weight.shape[axis]) : null;
      const cardinality = Number(weight?.quant_scales || 0);
      const logVal = Math.log10(Math.max(1, ratio));
      const pct = Math.min(100, (logVal / maxLog) * 100);
      const tone = quantTileTone(op);
      const cv = op.quant_scale_cv != null ? Number(op.quant_scale_cv).toFixed(2) : null;

      const row = document.createElement("div");
      row.className = "ss-row clickable-row";
      row.addEventListener("click", (e) => showPanelDetail(e.currentTarget, [
        ["Op",         `#${padOp(op.index)} ${op.name}`],
        ["Scale ratio", ratio.toFixed(3), ratio >= 1e6 ? "fallback" : ratio >= 1e3 ? "overhead" : ""],
        ["Scale CV",   cv ?? "-", cv && Number(cv) > 0.5 ? "overhead" : ""],
        ["Quantized axis", Number.isSafeInteger(axis) ? String(axis) : "unresolved"],
        ["Scale cardinality", axisDimension > 0 ? `${cardinality}/${axisDimension} ${cardinality === axisDimension ? "matched" : "mismatch"}` : `${cardinality || "?"}/dynamic`],
        ["Quant state", quantStateLabel(op)],
        ["Risk",       quantLabel(op), tone === "risk" ? "fallback" : tone === "warn" ? "overhead" : ""],
        ["MACs",       formatNumber(op.macs)],
      ], op.index));

      const lbl = document.createElement("span");
      lbl.className = "ss-label";
      lbl.textContent = `#${padOp(op.index)} ${op.name}`;

      const barWrap = document.createElement("div");
      barWrap.className = "ss-bar-wrap";

      // Threshold markers on the bar
      const warnMark = document.createElement("i");
      warnMark.className = "ss-threshold warn";
      warnMark.style.left = `${warnPct}%`;
      const riskMark = document.createElement("i");
      riskMark.className = "ss-threshold risk";
      riskMark.style.left = `${riskPct}%`;

      const bar = document.createElement("div");
      bar.className = `ss-bar ${tone}`;
      bar.style.width = `${pct.toFixed(1)}%`;

      barWrap.append(warnMark, riskMark, bar);

      const val = document.createElement("span");
      val.className = "ss-value";
      val.textContent = ratio >= 1e6 ? `${(ratio / 1e6).toFixed(1)}M` : ratio >= 1e3 ? `${(ratio / 1e3).toFixed(1)}k` : ratio.toFixed(0);
      if (cv) val.title = `CV: ${cv}`;

      row.append(lbl, barWrap, val);
      attachTip(row, [
        ["Op", `#${padOp(op.index)} ${op.name}`],
        ["Scale ratio", `${ratio.toExponential(2)} (max/min per-channel)`],
        ["Scale CV", cv ? `${cv} (spread across channels)` : "-"],
        ["Axis / cardinality", Number.isSafeInteger(axis) ? `${axis}; ${axisDimension > 0 ? `${cardinality}/${axisDimension}` : `${cardinality || "?"}/dynamic`}` : "unresolved"],
        ["ZP status", op.quant_zero_point_status || "-"],
        ["Risk", op.quant_risk || "-"],
      ]);
      rows.push(row);
    }

    // Legend below
    const legend = document.createElement("div");
    legend.className = "ss-legend";
    legend.innerHTML = `<span class="warn-text">│ 10³ warn</span><span class="risk-text">│ 10⁶ risk</span><span class="muted-text">(log scale · top ${meaningful.length} ops by ratio)</span>`;
    elements.quantScaleScatter.replaceChildren(...(contractSummary ? [contractSummary] : []), progressiveScaleRows(rows), legend);
  }

  function renderQuantRiskTable(analysis) {
    if (!elements.quantRiskTable) return;
    const special = nonGraphQuantEvidence(analysis);
    const evidence = special ? { categories: special } : quantSummaryEvidence(analysis);
    const anomalous = evidence.categories.filter((row) => row.tone !== "pass");
    if (elements.quantRiskCount) {
      elements.quantRiskCount.textContent = `${anomalous.filter((row) => row.tone === "risk").length} risk · ${anomalous.filter((row) => row.tone === "review").length} review categories`;
    }
    const table = document.createElement("table");
    table.className = "quant-risk-detail-table";
    const thead = document.createElement("thead");
    thead.innerHTML = "<tr><th>Scope</th><th>Evidence</th><th>Affected</th><th>Status</th></tr>";
    table.append(thead);
    const tbody = document.createElement("tbody");
    for (const item of evidence.categories) {
      const tr = document.createElement("tr");
      if (Number.isInteger(item.firstOpIndex)) {
        tr.className = "clickable-row";
        tr.addEventListener("click", () => jumpToGraphOp(item.firstOpIndex));
      }
      appendTableCells(tr, [
        [item.label], [item.evidence], [item.affected, "muted-text"],
        [item.tone, `${item.tone === "risk" ? "risk-text" : item.tone === "review" ? "warn-text" : "good-text"} bold`],
      ]);
      tbody.append(tr);
    }
    table.append(tbody);
    elements.quantRiskTable.replaceChildren(table);
  }

  function renderQuantHoles(analysis) {
    if (!elements.quantHoleList) return;
    const format = String(analysis.format || "").toLowerCase();
    if (["gguf", "safetensors"].includes(format)) {
      if (elements.quantHoleCount) {
        elements.quantHoleCount.textContent = "not applicable to weight containers";
        elements.quantHoleCount.className = "muted-text";
      }
      elements.quantHoleList.replaceChildren(targetCompareMessage(format === "gguf"
        ? "GGUF does not serialize an operator graph, so Q/DQ boundary placement cannot be derived from the artifact."
        : "SafeTensors does not serialize an operator graph, so Q/DQ boundary placement cannot be derived from the artifact."));
      return;
    }
    if (format === "coreml") {
      if (analysis.coreml?.model_type === "mlProgram") {
        const boundaries = (analysis.ops || []).filter((op) => ["quantize", "dequantize"].includes(String(op.mil_operation_type || "").toLowerCase()));
        const constantTransforms = (analysis.ops || []).filter((op) => String(op.mil_operation_type || "").toLowerCase().startsWith("constexpr_")
          && op.quantization_state === "serialized_quantization_transform");
        if (elements.quantHoleCount) {
          elements.quantHoleCount.textContent = `${boundaries.length} serialized Q/DQ ${boundaries.length === 1 ? "boundary" : "boundaries"}`;
          elements.quantHoleCount.className = boundaries.length ? "badge-warn" : "badge-ok";
        }
        if (!boundaries.length) {
          elements.quantHoleList.replaceChildren(targetCompareMessage(
            `No explicit MIL quantize/dequantize operation was found. ${constantTransforms.length} constexpr compression transform(s) decode stored constants and are not counted as activation precision boundaries. Runtime lowering and fusion remain unobserved.`,
          ));
          return;
        }
        const rows = boundaries.map((op) => {
          const row = document.createElement("button");
          row.type = "button";
          row.className = `quant-hole-row hole-${String(op.mil_operation_type).toLowerCase() === "dequantize" ? "dequant" : "requant"}`;
          const input = analysis.tensors?.[op.inputs?.[0]];
          const output = analysis.tensors?.[op.outputs?.[0]];
          const index = document.createElement("span");
          index.className = "hole-index";
          index.textContent = `#${String(op.index).padStart(3, "0")}`;
          const kind = document.createElement("span");
          kind.className = "hole-op";
          kind.textContent = String(op.mil_operation_type).toUpperCase();
          const flow = document.createElement("span");
          flow.className = "hole-flow";
          flow.textContent = `${input?.dtype || "UNKNOWN"} to ${output?.dtype || "UNKNOWN"}`;
          const context = document.createElement("span");
          context.className = "hole-context";
          context.textContent = `${output?.shape?.join("x") || "dynamic shape"}; ${op.estimated_bytes == null ? "payload not assessed" : formatBytes(op.estimated_bytes)}`;
          row.append(index, kind, flow, context);
          attachTip(row, [["MIL scope", op.mil_scope || "main"], ["Input", input?.name || "not bound"], ["Output", output?.name || "not bound"], ["Evidence", "serialized MIL operation type and ValueType"]]);
          row.addEventListener("click", () => jumpToGraphOp(op.index));
          return row;
        });
        const note = document.createElement("p");
        note.className = "hole-intro";
        note.textContent = `${boundaries.length} explicit activation precision boundary operation(s) are serialized. Their presence is observed; whether Core ML fuses them or where ANE/GPU/CPU lowering places them requires runtime evidence.`;
        elements.quantHoleList.replaceChildren(note, ...rows);
        return;
      }
      if (elements.quantHoleCount) {
        elements.quantHoleCount.textContent = analysis.coreml?.neural_network
          ? "not serialized by legacy NeuralNetwork" : "not assessed";
        elements.quantHoleCount.className = "muted-text";
      }
      elements.quantHoleList.replaceChildren(targetCompareMessage(analysis.coreml?.neural_network
        ? "Legacy Core ML NeuralNetwork weight quantization is stored in WeightParams rather than explicit graph Q/DQ operators. No Q/DQ boundary count is inferred."
        : "Core ML quantization boundaries are not reported without a decoded ML Program numerical contract."));
      return;
    }
    const holes = Array.isArray(analysis.quant_holes) ? analysis.quant_holes : [];
    const count = Number(analysis.quant_hole_count || 0);
    const status = analysis.quantization_status || {};
    const activationBoundaries = Number(status.activation_quantize_ops || 0) + Number(status.activation_dequantize_ops || 0);
    const constantConversions = Number(status.constant_precision_conversion_ops || 0);
    const fp16Expansions = Number(status.float16_constant_expansion_ops || 0);

    if (elements.quantHoleCount) {
      if (!count) {
        const context = [
          activationBoundaries ? `${activationBoundaries} activation Q/DQ` : "",
          constantConversions ? `${constantConversions} constant conversion${constantConversions === 1 ? "" : "s"}` : "",
        ].filter(Boolean).join("; ");
        elements.quantHoleCount.textContent = `0 mid-graph 8-bit/FP32 boundaries${context ? `; ${context}` : ""}`;
        elements.quantHoleCount.className = "badge-ok";
      } else {
        elements.quantHoleCount.textContent = `${count} mid-graph 8-bit/FP32 boundary op${count === 1 ? "" : "s"}`;
        elements.quantHoleCount.className = count >= 3 ? "badge-risk" : "badge-warn";
      }
    }

    if (!holes.length) {
      elements.quantHoleList.replaceChildren(targetCompareMessage(
        count === 0
          ? fp16Expansions
            ? `No 8-bit/FP32 activation boundary operator was found. ${fp16Expansions} serialized FP16 constant${fp16Expansions === 1 ? " is" : "s are"} expanded to FP32 compute precision and excluded from the activation-boundary denominator.`
            : `No serialized mid-graph 8-bit/FP32 activation boundary operator was found.${activationBoundaries ? ` ${activationBoundaries} serialized activation boundary conversion${activationBoundaries === 1 ? " is" : "s are"} outside the internal boundary set.` : ""} Weight integrity, grid quality, and task accuracy are separate.`
          : "Activation precision boundary operators are reported, but detail is unavailable."
      ));
      return;
    }

    const rows = holes.map((hole) => {
      const row = document.createElement("div");
      row.className = `quant-hole-row hole-${hole.hole_class === "int8-to-fp32" ? "dequant" : hole.hole_class === "fp32-to-int8" ? "requant" : "mixed"}`;

      const indexBadge = document.createElement("span");
      indexBadge.className = "hole-index";
      indexBadge.textContent = `#${String(hole.op_index).padStart(3, "0")}`;

      const opBadge = document.createElement("span");
      opBadge.className = "hole-op";
      opBadge.textContent = hole.op_name;

      const flow = document.createElement("span");
      flow.className = "hole-flow";
      const fromDtype = document.createElement("span");
      fromDtype.className = `hole-dtype ${hole.from_dtype === "FLOAT32" ? "dtype-fp32" : "dtype-int8"}`;
      fromDtype.textContent = hole.from_dtype;
      const arrow = document.createElement("span");
      arrow.className = "hole-arrow";
      arrow.textContent = "to";
      const toDtype = document.createElement("span");
      toDtype.className = `hole-dtype ${hole.to_dtype === "FLOAT32" ? "dtype-fp32" : "dtype-int8"}`;
      toDtype.textContent = hole.to_dtype;
      flow.append(fromDtype, arrow, toDtype);

      const context = document.createElement("span");
      context.className = "hole-context";
      context.textContent = `${hole.prev_op_name} → [${hole.op_name}] → ${hole.next_op_name}`;

      const impact = document.createElement("span");
      impact.className = "hole-impact";
      const macPct = Number(hole.adjacent_mac_percent || 0);
      impact.textContent = macPct > 0.001 ? `adj. ${formatPercent(macPct)} MACs` : "low-MAC context";

      row.append(indexBadge, opBadge, flow, context, impact);

      attachTip(row, [
        ["Op", `#${hole.op_index} ${hole.op_name}`],
        ["Transition", `${hole.from_dtype} → ${hole.to_dtype}`],
        ["Class", hole.hole_class],
        ["Context", `${hole.prev_op_name} → ${hole.next_op_name}`],
        ["Adjacent MACs", macPct > 0 ? formatPercent(macPct) : "< 0.1%"],
      ]);

      row.style.cursor = "pointer";
      row.addEventListener("click", () => jumpToGraphOp(hole.op_index));

      return row;
    });

    const intro = document.createElement("p");
    intro.className = "hole-intro";
    intro.textContent = holes.length === 1
      ? "1 serialized mid-graph 8-bit/FP32 boundary operator was found. This is one transition, not a complete float-island count; runtime fusion and placement remain unobserved."
      : `${holes.length} serialized mid-graph 8-bit/FP32 boundary operators were found. Each row is one transition operator, not one float island; region extent, runtime fusion, and placement remain unobserved.`;

    elements.quantHoleList.replaceChildren(intro, ...rows);
  }

  function configureQuantMap(title, legend) {
    const panel = elements.quantHeatmap?.closest(".perf-panel");
    const heading = panel?.querySelector(".mini-panel-head h3");
    const legendNode = panel?.querySelector(".quant-legend");
    if (heading) heading.textContent = title;
    if (legendNode) {
      legendNode.replaceChildren(...legend.map(([tone, label]) => {
        const item = document.createElement("span");
        item.className = `ql-item ${tone}`;
        item.textContent = label;
        return item;
      }));
    }
  }

  function quantStateRow(text, ratio, tone) {
    const row = document.createElement("div");
    row.className = "quant-state-row";
    const bar = document.createElement("div");
    bar.className = `qs-bar ${tone}`;
    bar.style.width = `${Math.max(2, Math.min(100, ratio * 100)).toFixed(1)}%`;
    const label = document.createElement("span");
    label.className = "qs-label";
    label.textContent = text;
    row.append(bar, label);
    return row;
  }

  function renderSerializedTensorMap(presentation) {
    configureQuantMap(presentation.title, presentation.legend);
    const cap = 500;
    const displayed = presentation.tiles.slice(0, cap);
    const tiles = displayed.map((tensor) => {
      const tile = document.createElement("button");
      tile.type = "button";
      tile.className = `quant-tile ${tensor.tone}`;
      tile.textContent = String(tensor.index);
      const shape = Array.isArray(tensor.shape) ? tensor.shape.join("x") : "unknown";
      const bitText = Number.isFinite(tensor.bits_per_element)
        ? `${Number(tensor.bits_per_element).toFixed(3)} stored bits/element`
        : "not assessed";
      const details = [
        ["Tensor", `#${tensor.index} ${tensor.name || "unnamed"}`],
        ["Shape", shape || "scalar"],
        ["Encoding", tensor.dtype || "UNKNOWN"],
        ["Storage class", String(tensor.encoding_class || "unknown").replace(/_/g, " ")],
        ["Payload", Number.isSafeInteger(tensor.byte_length) ? `${formatBytes(tensor.byte_length)} (${formatNumber(tensor.byte_length)} B)` : "not assessed"],
        ["Density", bitText],
      ];
      if (presentation.format === "gguf") {
        details.push(["Block cardinality", tensor.block_elements ? `${tensor.block_elements} elements / ${tensor.block_bytes} B` : "not assessed"]);
      }
      const integrity = tensor.numerical_integrity || {};
      details.push(
        ["Numerical scan", integrity.status || "not assessed"],
        ["Decoded values", Number.isSafeInteger(integrity.value_count) ? formatNumber(integrity.value_count) : "not assessed"],
        ["Non-finite", Number.isSafeInteger(integrity.nan_value_count) ? `${formatNumber(integrity.nan_value_count)} NaN / ${formatNumber(Number(integrity.positive_infinity_value_count || 0) + Number(integrity.negative_infinity_value_count || 0))} Inf` : "not assessed"],
        ["Exact zeros", Number.isSafeInteger(integrity.zero_value_count) ? `${formatNumber(integrity.zero_value_count)} (${formatPercent(integrity.exact_zero_fraction || 0)})` : "not assessed"],
        ["Finite range", integrity.minimum_finite != null || integrity.minimum_finite_decimal != null ? `${integrity.minimum_finite ?? integrity.minimum_finite_decimal} to ${integrity.maximum_finite ?? integrity.maximum_finite_decimal}` : "not assessed"],
        ["Finite mean / RMS", integrity.arithmetic_mean_finite == null ? integrity.moment_status || "not assessed" : `${integrity.arithmetic_mean_finite} / ${integrity.rms_finite}`],
        ["Stored grid utilization", integrity.encoded_codebook_entries_legal != null
          ? `${integrity.encoded_codebook_name}: ${formatNumber(integrity.encoded_codebook_entries_used)}/${formatNumber(integrity.encoded_codebook_entries_legal)} (${formatPercent(integrity.encoded_codebook_utilization || 0)})`
          : integrity.quantization_code_levels_legal != null
            ? `${formatNumber(integrity.quantization_code_levels_used)}/${formatNumber(integrity.quantization_code_levels_legal)} scalar codes (${formatPercent(integrity.quantization_code_utilization || 0)})`
            : "not applicable"],
        ["Encoded blocks / non-finite scales", integrity.encoded_block_count == null ? "not applicable" : `${formatNumber(integrity.encoded_block_count)} / ${formatNumber(integrity.nonfinite_scale_block_count || 0)}`],
        ["Payload SHA-256", integrity.payload_sha256 || "not assessed"],
      );
      if (integrity.reason) details.push(["Unassessed reason", integrity.reason]);
      attachTip(tile, details);
      tile.addEventListener("click", (event) => showPanelDetail(event.currentTarget, details));
      return tile;
    });
    elements.quantHeatmap.replaceChildren(...tiles);
    const truncated = presentation.tiles.length - displayed.length;
    if (elements.quantHeatmapCount) {
      elements.quantHeatmapCount.textContent = truncated
        ? `${displayed.length} of ${presentation.tiles.length} tensors shown (${truncated} truncated)`
        : presentation.count_label;
    }
    const rows = presentation.summary_rows.map((row) => {
      const tensorRatio = row.denominator ? row.count / row.denominator : 0;
      const byteText = Number.isSafeInteger(row.byte_length) && row.byte_denominator
        ? `; ${formatNumber(row.byte_length)}/${formatNumber(row.byte_denominator)} B (${formatPercent(row.byte_length / row.byte_denominator)})`
        : "";
      return quantStateRow(`${row.label} - ${row.count}/${row.denominator} tensors (${formatPercent(tensorRatio)})${byteText}`, tensorRatio, row.tone);
    });
    const coverage = document.createElement("p");
    coverage.className = "repair-method-line";
    coverage.textContent = `Payload coverage: ${presentation.coverage_status}. ${presentation.scope}`;
    const breakdown = document.createElement("div");
    breakdown.className = "quant-state-rows";
    breakdown.append(...rows, coverage);
    elements.quantStateBreakdown.replaceChildren(breakdown);
  }

  function renderQuantRiskHeatmap(analysis) {
    const serialized = serializedTensorPresentation(analysis);
    elements.quantHeatmap.replaceChildren();
    elements.quantStateBreakdown?.replaceChildren();
    if (serialized) {
      renderSerializedTensorMap(serialized);
      return;
    }
    configureQuantMap("Quantization State & Risk Map", [
      ["good", "state: 8-bit arithmetic"],
      ["mixed", "state: boundary / weight-only"],
      ["signal-warn", "border: warn signal"],
      ["signal-risk", "border: risk signal"],
      ["neutral", "state: unquantized"],
    ]);
    const ops = Array.isArray(analysis.ops) ? analysis.ops : [];
    const quantOps = ops.filter((op) => op.quantization_state && op.quantization_state !== "none");
    if (!quantOps.length) {
      if (elements.quantHeatmapCount) elements.quantHeatmapCount.textContent = `${ops.length} graph op${ops.length === 1 ? "" : "s"}`;
      const coreMlUnassessed = String(analysis.format || "").toLowerCase() === "coreml"
        && analysis.quantization_status?.assessment_status !== "assessed";
      elements.quantHeatmap.replaceChildren(targetCompareMessage(
        coreMlUnassessed
          ? "Core ML operator quantization is not assessed for this model representation. No FP32 or quantized classification is inferred."
          : ops.length ? "No quantized operators were found in the assessed graph contract." : "No serialized operator graph is available for quantization assessment."
      ));
      elements.quantStateBreakdown?.replaceChildren(targetCompareMessage(coreMlUnassessed
        ? analysis.coreml?.parser_scope || "Core ML numerical payload was not decoded."
        : String(analysis.format || "").toLowerCase() === "coreml"
          ? `${analysis.quantization_status?.summary || "No quantized WeightParams were found."} ${analysis.quantization_status?.scanned_layer_count || 0}/${analysis.quantization_status?.layer_count || ops.length} layer WeightParams field scans complete; ${analysis.weight_integrity?.assessed_parameter_count || 0}/${analysis.weight_integrity?.parameter_count || 0} parameter payload cardinalities and numerical contracts assessed.`
          : "No graph-op quantization denominator is available."));
      return;
    }
    const evidence = quantSummaryEvidence(analysis);
    const CAP = 500;
    const displayed = ops.slice(0, CAP);
    const truncated = ops.length - displayed.length;

    const tiles = displayed.map((op) => {
      const tile = document.createElement("button");
      tile.type = "button";
      const signal = evidence.opSignals.get(Number(op.index));
      const executionTone = op.quantization_state === "quantized_compute"
        ? "good"
        : op.quantization_state && op.quantization_state !== "none" ? "mixed" : "neutral";
      tile.className = `quant-tile ${executionTone}${signal ? ` signal-${signal.tone}` : ""}`;
      tile.textContent = String(op.index);
      attachTip(tile, [
        ["Op", `#${padOp(op.index)} ${op.name}`],
        ["State", quantStateLabel(op)],
        ["Risk", quantLabel(op)],
        ["Review signals", signal?.labels.join(" / ") || "none"],
        ["Zero-pt", op.quant_zero_point_status || "-"],
        ["Scale CV", op.quant_scale_cv != null ? Number(op.quant_scale_cv).toFixed(3) : "-"],
      ]);
      tile.addEventListener("click", (e) => showPanelDetail(e.currentTarget, [
        ["Op",       `#${padOp(op.index)} ${op.name}`],
        ["State",    quantStateLabel(op)],
        ["Risk",     quantLabel(op), op.quant_risk === "risk" ? "fallback" : op.quant_risk === "warn" ? "overhead" : ""],
        ["Review signals", signal?.labels.join(" / ") || "none", signal ? "overhead" : ""],
        ["Zero-pt",  op.quant_zero_point_status || "-", op.quant_zero_point_status === "out-of-range" ? "fallback" : op.quant_zero_point_status === "near-boundary" ? "overhead" : ""],
        ["Scale CV", op.quant_scale_cv != null ? Number(op.quant_scale_cv).toFixed(3) : "-"],
        ["MACs",     formatNumber(op.macs)],
      ], op.index));
      return tile;
    });
    elements.quantHeatmap.replaceChildren(...tiles);

    // Update count label
    if (elements.quantHeatmapCount) {
      elements.quantHeatmapCount.textContent = truncated
        ? `${displayed.length} of ${ops.length} graph ops shown (${truncated} truncated)`
        : `${ops.length} graph op${ops.length === 1 ? "" : "s"}`;
    }

    // State breakdown panel
    if (elements.quantStateBreakdown) {
      const stateCounts = new Map();
      for (const op of ops) {
        const key = op.quantization_state || "none";
        stateCounts.set(key, (stateCounts.get(key) || 0) + 1);
      }
      const sorted = [...stateCounts.entries()].sort((a, b) => b[1] - a[1]);
      const breakdown = document.createElement("div");
      breakdown.className = "quant-state-rows";
      const addRow = (text, ratio, tone) => breakdown.append(quantStateRow(text, ratio, tone));
      const status = analysis.quantization_status || {};
      const macOps = ops.filter((op) => Number(op.macs || 0) > 0);
      const quantMacOps = macOps.filter((op) => op.quantization_state === "quantized_compute");
      const totalMacs = sumNumbers(macOps.map((op) => op.macs));
      const quantMacs = sumNumbers(quantMacOps.map((op) => op.macs));
      const computeOps = Number(status.compute_ops ?? macOps.length);
      const quantizedComputeOps = Number(status.quantized_compute_ops ?? quantMacOps.length);
      const macRatio = totalMacs ? quantMacs / totalMacs : 0;
      if (computeOps) addRow(`MAC-bearing compute coverage — ${quantizedComputeOps}/${computeOps} ops; ${formatNumber(quantMacs)}/${formatNumber(totalMacs)} MACs (${formatPercent(macRatio)})`, macRatio, "good");
      const stateLabels = { quantized_compute: "8-bit arithmetic state", quantized_data_movement: "8-bit data movement", quant_boundary: "activation Q/DQ boundary", integer_requantization: "integer-domain requantization", float16_constant_expansion: "FP16 constant to FP32 compute", quantized_constant_expansion: "8-bit constant to float compute", constant_precision_conversion: "constant precision conversion", precision_boundary: "non-8-bit precision boundary", weight_only_or_dynamic_range: "weight-only / dynamic-range", mixed_or_hybrid_compute: "mixed / hybrid compute", serialized_quantization_transform: "serialized quant/compression transform", none: "unquantized" };
      for (const [state, count] of sorted) {
        const tone = state === "quantized_compute" ? "good"
          : ["weight_only_or_dynamic_range", "quant_boundary", "integer_requantization", "float16_constant_expansion", "quantized_constant_expansion", "constant_precision_conversion"].includes(state) ? "mixed"
          : state === "none" || state === "unquantized" ? "neutral"
          : "warn";
        addRow(`${stateLabels[state] || state.replace(/_/g, " ")} — ${count}/${ops.length} graph ops (${formatPercent(count / ops.length)})`, count / ops.length, tone);
      }
      elements.quantStateBreakdown.replaceChildren(breakdown);
    }
  }

  // ── Roofline Scatter Chart ────────────────────────────────────────────────
  function renderRooflineChart(analysis) {
    if (!elements.rooflineChart) return;
    const ops = Array.isArray(analysis.ops) ? analysis.ops : [];
    const target = analysis.target_profile || {};
    const peakGops = Number(target.effective_peak_gops || 0);
    const bwGbps = Number(target.effective_memory_bandwidth_gbps || 0);
    const ridge = Number(target.ridge_point_ops_per_byte || 0);
    const memThr = Number(target.memory_bound_intensity || 0);
    const compThr = Number(target.compute_bound_intensity || 0);

    if (!peakGops || !bwGbps || !ops.length) {
      elements.rooflineChart.replaceChildren(targetCompareMessage("No roofline data available."));
      return;
    }

    const NS = "http://www.w3.org/2000/svg";
    const W = 600, H = 300, MT = 24, MR = 16, MB = 44, ML = 58;
    const PW = W - ML - MR, PH = H - MT - MB;

    // Determine axis ranges (log scale)
    const trafficOps = ops.filter((op) => assessedOpLogicalBytes(op) != null && assessedOpLogicalBytes(op) > 0);
    const macOps = trafficOps.filter((op) => Number(op.macs || 0) > 0 && op.intensity_ops_per_byte != null);
    if (!macOps.length) { // B4: empty state when no MAC ops
      elements.rooflineChart.replaceChildren(targetCompareMessage(ops.some((op) => Number(op.macs || 0) > 0)
        ? "MAC-bearing ops exist, but logical traffic or intensity is not assessed."
        : "No MAC-contributing ops found for roofline chart."));
      return;
    }
    const maxIntensity = macOps.reduce((m, op) => Math.max(m, Number(op.intensity_ops_per_byte || 0)), 0.1);
    const xMinVal = 0.05, xMaxVal = Math.max(maxIntensity * 2.5, compThr * 2, 200);
    const maxBytes = trafficOps.reduce((m, op) => Math.max(m, assessedOpLogicalBytes(op)), 1);
    const yMinVal = 100, yMaxVal = Math.max(maxBytes * 3, 1e8);

    const logScale = (v, vMin, vMax, pxSize) => {
      const l = Math.log10(Math.max(v, vMin));
      const lMin = Math.log10(vMin), lMax = Math.log10(vMax);
      return ((l - lMin) / (lMax - lMin)) * pxSize;
    };
    const toX = (v) => ML + logScale(v, xMinVal, xMaxVal, PW);
    const toY = (v) => MT + PH - logScale(v, yMinVal, yMaxVal, PH);

    const el = (tag, attrs = {}) => {
      const node = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
      return node;
    };
    const txt = (content, attrs = {}) => {
      const node = el("text", { "font-size": "10", "fill": "var(--muted, #888)", ...attrs });
      node.textContent = content;
      return node;
    };

    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", style: "max-width:600px;display:block;" });

    // Background zones
    const xMemThr = toX(memThr), xCompThr = toX(compThr);
    svg.append(
      el("rect", { x: ML, y: MT, width: Math.max(0, xMemThr - ML), height: PH, fill: "rgba(220,38,38,0.07)" }),
      el("rect", { x: xMemThr, y: MT, width: Math.max(0, xCompThr - xMemThr), height: PH, fill: "rgba(234,179,8,0.07)" }),
      el("rect", { x: xCompThr, y: MT, width: Math.max(0, ML + PW - xCompThr), height: PH, fill: "rgba(15,118,110,0.07)" }),
    );

    // Threshold vertical lines
    const dashLine = (x, color, label) => {
      const g = el("g");
      g.append(el("line", { x1: x, y1: MT, x2: x, y2: MT + PH, stroke: color, "stroke-width": "1", "stroke-dasharray": "4 3", opacity: "0.7" }));
      g.append(txt(label, { x, y: MT - 4, "text-anchor": "middle", fill: color, "font-size": "9", "font-weight": "600" }));
      return g;
    };
    if (memThr > xMinVal) svg.append(dashLine(xMemThr, "#dc2626", `mem ${memThr}`));
    if (compThr > xMinVal) svg.append(dashLine(xCompThr, "#0f766e", `cmp ${compThr}`));
    if (ridge > xMinVal) {
      const xR = toX(ridge);
      svg.append(el("line", { x1: xR, y1: MT, x2: xR, y2: MT + PH, stroke: "#7c3aed", "stroke-width": "1.5", "stroke-dasharray": "6 3", opacity: "0.6" }));
      svg.append(txt(`ridge ${ridge}`, { x: xR, y: MT - 4, "text-anchor": "middle", fill: "#7c3aed", "font-size": "9", "font-weight": "600" }));
    }

    // X axis ticks
    const xTicks = [0.1, 1, 10, 100, 1000].filter((v) => v >= xMinVal * 0.9 && v <= xMaxVal * 1.1);
    for (const v of xTicks) {
      const x = toX(v);
      svg.append(el("line", { x1: x, y1: MT + PH, x2: x, y2: MT + PH + 4, stroke: "var(--line,#ddd)" }));
      svg.append(txt(v >= 1 ? String(v) : v.toFixed(1), { x, y: MT + PH + 14, "text-anchor": "middle" }));
    }

    // Y axis ticks
    const yTicks = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9].filter((v) => v >= yMinVal * 0.9 && v <= yMaxVal * 1.1);
    for (const v of yTicks) {
      const y = toY(v);
      svg.append(el("line", { x1: ML - 4, y1: y, x2: ML, y2: y, stroke: "var(--line,#ddd)" }));
      svg.append(txt(v >= 1e9 ? `${v / 1e9}GB` : v >= 1e6 ? `${v / 1e6}MB` : v >= 1e3 ? `${v / 1e3}KB` : String(v),
        { x: ML - 7, y: y + 3.5, "text-anchor": "end" }));
    }

    // Axis lines + labels
    svg.append(el("line", { x1: ML, y1: MT, x2: ML, y2: MT + PH, stroke: "var(--line,#ddd)", "stroke-width": "1" }));
    svg.append(el("line", { x1: ML, y1: MT + PH, x2: ML + PW, y2: MT + PH, stroke: "var(--line,#ddd)", "stroke-width": "1" }));
    svg.append(txt("Arithmetic Intensity (ops/byte)", { x: ML + PW / 2, y: H - 4, "text-anchor": "middle", "font-size": "10" }));
    const yLabel = el("text", { transform: `rotate(-90)`, x: -(MT + PH / 2), y: 12, "text-anchor": "middle", "font-size": "10", fill: "var(--muted,#888)" });
    yLabel.textContent = "Assessed Memory Traffic (bytes)";
    svg.append(yLabel);

    // Op bubbles
    const BOUND_COLOR = { "compute-bound": "#0f766e", "mixed": "#d97706", "memory-bound": "#dc2626" };
    for (const op of macOps) {
      const intensity = Number(op.intensity_ops_per_byte);
      const bytes = assessedOpLogicalBytes(op);
      const macs = Number(op.macs || 0);
      if (bytes <= 0) continue;
      const cx = toX(Math.max(intensity, xMinVal * 0.6));
      const cy = toY(Math.max(bytes, yMinVal));
      const r = macs > 0 ? Math.max(3, Math.min(12, 2 + Math.log10(macs + 1) * 2.2)) : 2.5;
      const color = BOUND_COLOR[op.static_bound_guess] || "#888";
      const circle = el("circle", {
        cx, cy, r,
        fill: color,
        "fill-opacity": macs > 0 ? "0.72" : "0.35",
        stroke: color,
        "stroke-width": macs > 0 ? "0" : "1",
      });
      // Attach tooltip and click via a foreignObject trick isn't ideal — use pointer-events + title
      circle.style.cursor = "pointer";
      const tipLines = [
        ["Op", `#${String(op.index).padStart(3, "0")} ${op.name}`],
        ["Intensity", `${intensity.toFixed(2)} ops/byte`],
        ["Bound", op.static_bound_guess],
        ["Traffic", bytes >= 1e6 ? `${(bytes / 1e6).toFixed(2)} MB` : bytes >= 1e3 ? `${(bytes / 1e3).toFixed(1)} KB` : `${bytes} B`],
        ["MACs", macs > 0 ? formatNumber(macs) : "zero-MAC"],
      ];
      attachTip(circle, tipLines);
      circle.addEventListener("click", () => jumpToGraphOp(op.index));
      svg.append(circle);
    }

    elements.rooflineChart.replaceChildren(svg);

    if (elements.rooflineChartLegend) {
      const compCount = ops.filter((op) => op.static_bound_guess === "compute-bound").length;
      const mixCount = ops.filter((op) => op.static_bound_guess === "mixed").length;
      const memCount = ops.filter((op) => op.static_bound_guess === "memory-bound").length;
      const macBearingOps = ops.filter((op) => Number(op.macs || 0) > 0);
      elements.rooflineChartLegend.innerHTML =
        `<span style="color:#dc2626">■</span> mem ${memCount} &nbsp;`
        + `<span style="color:#d97706">■</span> mixed ${mixCount} &nbsp;`
        + `<span style="color:#0f766e">■</span> compute ${compCount} &nbsp; posture ${compCount + mixCount + memCount}/${ops.length} classified &nbsp; MAC-bearing intensity ${macOps.length}/${macBearingOps.length} assessed &nbsp; logical traffic ${trafficOps.length}/${ops.length} assessed`;
    }
  }

  function renderStageMemoryMix(analysis) {
    const ops = Array.isArray(analysis.ops) ? analysis.ops : [];
    const blockStages = analysis?.block_inventory?.status === "assessed"
      ? analysis.block_inventory.stages || []
      : [];
    const stages = blockStages.length
      ? blockStages.map((stage) => {
          const stageOps = new Set(stage.op_indices || []);
          const assessed = ops.filter((op) => stageOps.has(op.index) && op.macs_status !== "not_assessed").length;
          return {
            index: stage.index,
            key: stage.display_name || `stage-${stage.index}`,
            op_indices: stage.op_indices || [],
            mac_percent: stage.aggregates?.mac_percent ?? null,
            mac_assessed_ops: assessed,
            mac_not_assessed_ops: (stage.op_indices || []).length - assessed,
            xnnpack_chain_breaks: stage.aggregates?.predicted_break_count || 0,
            l1_max_ratio: stage.aggregates?.l1_max_ratio ?? null,
            l1_watch_count: stage.aggregates?.l1_watch_count || 0,
            logical_traffic_bytes: stage.aggregates?.logical_traffic_bytes || 0,
          };
        })
      : Array.isArray(analysis.stages) ? analysis.stages : [];
    if (!stages.length || !ops.length) {
      elements.stageMemoryMix.replaceChildren(targetCompareMessage("No stage information is available for this model."));
      return;
    }
    const isOverview = getContext().activeAuditTab === "overview";
    const OVERVIEW_CAP = 6;
    const visible = isOverview ? stages.slice(0, OVERVIEW_CAP) : stages;
    const hidden = isOverview ? stages.slice(OVERVIEW_CAP) : [];
    const rows = visible.map((stage) => {
      const opIndices = Array.isArray(stage.op_indices) ? new Set(stage.op_indices) : null;
      const stageOps = opIndices
        ? ops.filter((op) => opIndices.has(op.index))
        : ops.filter((op) => op.index >= Number(stage.first_op) && op.index <= Number(stage.last_op));
      return stageMemoryMixRow(stage, stageOps, String(analysis?.format || "tflite").toLowerCase() === "onnx");
    });
    if (hidden.length) {
      const more = document.createElement("button");
      more.className = "stage-mix-more";
      more.textContent = `+ ${hidden.length} more semantic stage${hidden.length === 1 ? "" : "s"} — open Blocks for the complete inventory`;
      rows.push(more);
    }
    elements.stageMemoryMix.replaceChildren(...rows);
  }

  function renderFlameStrip(container, segments, total) {
    if (!segments.length || !total) {
      renderVisualEmpty(container, "No data available.");
      return;
    }
    container.replaceChildren(
      ...segments.map((segment) => {
        const share = Number(segment.value || 0) / Math.max(1e-9, total);
        const node = document.createElement("button");
        node.type = "button";
        node.className = `flame-segment ${segment.tone || "neutral"}`;
        node.style.flexBasis = `${Math.max(3, share * 100)}%`;
        const label = document.createElement("strong");
        label.textContent = share >= 0.045 ? segment.label : "";
        const detail = document.createElement("span");
        detail.textContent = share >= 0.09 ? formatPercent(share) : "";
        node.append(label, detail);
        // Rich hover tooltip replaces native title
        const tipLines = [[null, segment.label]];
        if (segment.detail) tipLines.push([null, segment.detail]);
        tipLines.push(["Share", formatPercent(share)]);
        attachTip(node, tipLines);
        if (segment.opIndex != null) {
          node.addEventListener("click", (e) => showPanelDetail(e.currentTarget, [
            ["Op",    segment.label, ""],
            ...(segment.detail ? [["Detail", segment.detail, ""]] : []),
            ["Share", formatPercent(share), ""],
          ], segment.opIndex));
        }
        return node;
      }),
    );
  }

  function targetCompareCard(row) {
    const card = document.createElement("article");
    card.className = row.error ? "target-card risk" : "target-card";
    const title = document.createElement("div");
    title.className = "target-card-title";
    const strong = document.createElement("strong");
    strong.textContent = row.label;
    const span = document.createElement("span");
    span.textContent = row.error ? "analysis failed" : `ridge ~${formatNumber(row.ridge)} ops/B`;
    title.append(strong, span);
    card.append(title);
    if (row.error) {
      const error = document.createElement("p");
      error.textContent = row.error;
      card.append(error);
      return card;
    }
    card.append(targetStack(row.totals));
    const metrics = document.createElement("div");
    metrics.className = "target-metrics";
    metrics.append(
      targetMetric("Low-intensity", formatPercent(row.memoryRatio), row.intensitySensitivityDetail || ""),
      targetMetric("Breaks", formatNumber(row.chainBreaks)),
      targetMetric(row.isFp32 ? "INT8 potential" : "INT8", `~${row.speedup.toFixed(2)}x`),
      targetMetric(
        "Steady / cold",
        `${formatUs(row.totalUs)} / ${formatUs(row.coldStartUs)}`,
        `Steady range ${formatUs(row.steadyStateLowUs ?? row.totalUs)}-${formatUs(row.steadyStateHighUs ?? row.totalUs)}; cold range ${formatUs(row.coldStartLowUs ?? row.coldStartUs)}-${formatUs(row.coldStartHighUs ?? row.coldStartUs)}. Boundary profile range only; point uses the midpoint.`,
      ),
      targetMetric("Top op", row.topOp ? `#${padOp(row.topOp.index)} ${formatUs(row.topUs)}` : "-"),
    );
    card.append(metrics);
    return card;
  }

  function chainBreakMarker(breaks) {
    const marker = document.createElement("button");
    marker.type = "button";
    marker.className = breaks.length ? "chain-break-chip" : "chain-break-chip quiet";
    if (!breaks.length) {
      marker.textContent = "break";
      marker.title = "Delegate partition boundary";
      return marker;
    }
    const effective = breaks.filter((op) => (op.xnnpack_break_class || "") !== "structural-terminal");
    const top = maxBy(breaks, (op) => Number(op.estimated_bytes || 0) + Number(op.chain_break_impact_mac_percent || 0) * 1e6);
    marker.textContent = `${breaks.length} break${breaks.length === 1 ? "" : "s"}`;
    marker.title = breaks
      .slice(0, 5)
      .map((op) => `#${padOp(op.index)} ${op.name} / ${op.xnnpack_break_class || "break"}`)
      .join("\n");
    marker.classList.toggle("risk", effective.length > 1);
    marker.addEventListener("click", () => {
      if (top) jumpToGraphOp(top.index);
    });
    return marker;
  }

  function stageMemoryMixRow(stage, ops, onnx = false) {
    const row = document.createElement("div");
    row.className = "stage-mix-row";
    const totalMacs = sumNumbers(ops.map((op) => op.macs));
    const postureOps = ops.filter((op) => op.static_bound_guess !== "not-assessed");
    const denominator = totalMacs || Math.max(1, postureOps.length);
    const counts = { "compute-bound": 0, mixed: 0, "memory-bound": 0 };
    const assessedBytesByPosture = { "compute-bound": 0, mixed: 0, "memory-bound": 0 };
    const unassessedBytesByPosture = { "compute-bound": 0, mixed: 0, "memory-bound": 0 };
    for (const op of postureOps) {
      const key = op.static_bound_guess === "compute-bound" || op.static_bound_guess === "mixed"
        ? op.static_bound_guess : "memory-bound";
      counts[key] += totalMacs ? Number(op.macs || 0) : 1;
      const bytes = assessedOpLogicalBytes(op);
      if (bytes == null) unassessedBytesByPosture[key] += 1;
      else assessedBytesByPosture[key] += bytes;
    }

    // Extra metrics
    const fallbackOps = onnx ? 0 : ops.filter((op) => !op.xnnpack_supported).length;
    const quantOps = ops.filter((op) => op.quantized_compute_path).length;
    const holeOps = ops.filter((op) => op.quant_hole).length;
    const assessedByteOps = ops.filter((op) => assessedOpLogicalBytes(op) != null);
    const totalBytes = sumNumbers(assessedByteOps.map(assessedOpLogicalBytes));
    const unassessedByteOps = ops.length - assessedByteOps.length;

    const label = document.createElement("div");
    label.className = "stage-mix-label";
    const title = document.createElement("strong");
    title.textContent = `#${stage.index} ${humanizeStageKey(stage.key)}`;
    title.title = `Raw stage key: ${stage.key}`;
    const detail = document.createElement("span");
    const stageMacShare = stage.mac_percent == null ? "MAC share N/A" : `${formatPercent(stage.mac_percent)} MACs`;
    const assessedStageOps = Number(stage.mac_assessed_ops || 0);
    const unassessedStageOps = Number(stage.mac_not_assessed_ops || 0);
    const stageCoverage = assessedStageOps + unassessedStageOps > 0
      ? `MAC coverage ${assessedStageOps}/${assessedStageOps + unassessedStageOps}`
      : "MAC assessment N/A (non-compute stage)";
    const metaParts = [
      stageMacShare,
      onnx ? stageCoverage : `breaks ${stage.xnnpack_chain_breaks || 0}`,
      !onnx && fallbackOps > 0 ? `fallback ${fallbackOps}` : null,
      quantOps > 0 ? `INT8 ${quantOps}/${ops.length}` : null,
      holeOps > 0 ? `⚠Q-boundary ${holeOps}` : null,
      stage.l1_max_ratio == null ? "L1 N/A" : `L1 max ${Number(stage.l1_max_ratio).toFixed(2)}x / ${formatNumber(stage.l1_watch_count || 0)} watch`,
    ].filter(Boolean).join(" / ");
    detail.textContent = metaParts;
    label.append(title, detail);

    const bar = document.createElement("div");
    bar.className = "stage-mix-bar";
    for (const key of ["compute-bound", "mixed", "memory-bound"]) {
      const share = counts[key] / denominator;
      const segment = document.createElement("i");
      segment.className = key;
      segment.style.flexBasis = `${Math.max(share > 0 ? 3 : 0, share * 100)}%`;
      attachTip(segment, [
        ["Bound", boundDisplayLabel(key)],
        ["Share", formatPercent(share)],
        ["Bytes", `${formatBytes(assessedBytesByPosture[key])} assessed${unassessedBytesByPosture[key] ? `; ${unassessedBytesByPosture[key]} op(s) N/A` : ""}`],
        ["Denominator", totalMacs ? "weighted MACs" : "op count"],
      ]);
      bar.append(segment);
    }

    const meta = document.createElement("span");
    meta.className = "stage-mix-meta";
    const memPct = formatPercent(counts["memory-bound"] / denominator);
    const memoryShareBasis = totalMacs ? "memory-bound MAC share" : "memory-bound op share";
    meta.textContent = `${memPct} ${memoryShareBasis} / ${formatBytes(totalBytes)} logical bytes assessed${unassessedByteOps ? ` / ${unassessedByteOps} op(s) N/A` : ""}`;
    row.append(label, bar, meta);

    const topOp = [...ops].sort((a, b) => Number(b.macs || 0) - Number(a.macs || 0))[0];
    row.style.cursor = "pointer";
    row.addEventListener("click", (e) => {
      showPanelDetail(e.currentTarget, [
        ["Stage",       humanizeStageKey(stage.key)],
        ["Raw stage key", stage.key],
        ["Ops",         String(ops.length)],
        ["MACs",        stage.mac_percent == null ? "N/A" : formatPercent(stage.mac_percent)],
        [totalMacs ? "Low-intensity MAC share" : "Low-intensity op share", formatPercent(counts["memory-bound"] / denominator), counts["memory-bound"] / denominator > 0.5 ? "overhead" : ""],
        ["Logical bytes",  `${formatBytes(totalBytes)} assessed${unassessedByteOps ? `; ${unassessedByteOps} op(s) N/A` : ""}`],
        ["L1 row watch", stage.l1_max_ratio == null ? "N/A" : `${formatNumber(stage.l1_watch_count || 0)} op(s); max ${Number(stage.l1_max_ratio).toFixed(2)}x`],
        ...(!onnx ? [["XNN breaks", String(stage.xnnpack_chain_breaks || 0), stage.xnnpack_chain_breaks > 0 ? "overhead" : ""]] : []),
        ...(!onnx && fallbackOps > 0 ? [["CPU fallback", String(fallbackOps), "fallback"]] : []),
        ...(quantOps > 0  ? [["INT8 ops",    `${quantOps}/${ops.length}`, "compute"]] : []),
        ...(holeOps > 0   ? [["Precision boundaries", String(holeOps), "overhead"]] : []),
      ], topOp?.index);
    });

    return row;
  }

  let _perfTimelineAC = null;
  function renderPerfTimeline(analysis) {
    const container = elements.perfTimeline;
    if (!container) return;
    if (_perfTimelineAC) { _perfTimelineAC.abort(); }
    _perfTimelineAC = new AbortController();
    const { signal } = _perfTimelineAC;
    container.replaceChildren();

    const target = analysis.target_profile || {};
    const allOps = analysis.ops || [];

    // Estimate time per op; keep zero-time ops for structure but mark them
    const allOpData = allOps.map((op) => {
      const b = estimateOpBottleneck(op, target);
      const serialUs = b.packingUs + b.breakUs + (b.fallbackUs || 0);
      const parallelUs = Math.max(b.computeUs, b.memoryUs);
      const totalUs = parallelUs + serialUs;
      let tone;
      if (!op.xnnpack_supported) tone = "fallback";
      else if (serialUs > parallelUs * 0.6) tone = "overhead";
      else if (b.computeUs >= b.memoryUs) tone = "compute";
      else tone = "memory";
      return { op, b, parallelUs, serialUs, totalUs, tone };
    });
    const opData = allOpData.filter((d) => d.totalUs > 0);
    const zeroTimeCount = allOpData.length - opData.length;

    if (!opData.length) {
      const msg = document.createElement("p");
      msg.style.cssText = "color:var(--muted);font-size:12px;padding:12px 14px;margin:0;";
      msg.textContent = "Insufficient timing data for this target.";
      container.append(msg);
      return;
    }

    const totalUs = opData.reduce((s, d) => s + d.totalUs, 0);
    const packingUs = opData.reduce((s, d) => s + d.b.packingUs, 0);
    const boundarySetupUs = opData.reduce((s, d) => s + d.b.breakUs, 0);
    const steadyUs = Math.max(0, totalUs - packingUs - boundarySetupUs);

    // Group by XNN chain id (sorted: 0,1,2...  fallback=-1 last)
    const chainMap = new Map();
    for (const d of opData) {
      const id = d.op.xnnpack_chain_id ?? -1;
      if (!chainMap.has(id)) chainMap.set(id, []);
      chainMap.get(id).push(d);
    }
    const chains = [...chainMap.entries()]
      .sort(([a], [b]) => (a === -1 ? 1 : b === -1 ? -1 : a - b));

    if (elements.perfTimelineSubtitle) {
      const delegatedOps = allOps.filter((op) =>
        op.xnnpack_chain_id >= 0 && op.xnnpack_supported !== false && !op.xnnpack_chain_break,
      ).length;
      const delegatedPct = allOps.length > 0 ? delegatedOps / allOps.length * 100 : 0;
      const delegatedMacText = analysis.delegated_mac_percent == null
        ? "MAC share N/A"
        : `${(Number(analysis.delegated_mac_percent) * 100).toFixed(1)}% MACs`;
      const zeroNote = zeroTimeCount > 0 ? ` / +${zeroTimeCount} zero-cost` : "";
      elements.perfTimelineSubtitle.textContent =
        `${opData.length} timed ops${zeroNote} / ${formatUs(steadyUs)} steady / ${formatUs(totalUs)} cold / ${delegatedOps}/${allOps.length} conditionally delegatable (${delegatedPct.toFixed(1)}% ops; ${delegatedMacText})`;
    }

    // Shared floating tooltip
    const tip = document.createElement("div");
    tip.className = "fc-tooltip";
    tip.hidden = true;
    container.append(tip);

    function showTip(e, lines) {
      tip.replaceChildren(...lines.map((line) => {
        const item = document.createElement("span");
        item.textContent = line;
        return item;
      }));
      tip.hidden = false;
      moveFcTip(e);
    }
    function moveFcTip(e) {
      const cr = container.getBoundingClientRect();
      const x = e.clientX - cr.left + 14;
      const y = e.clientY - cr.top + 14;
      tip.style.left = `${Math.min(x, cr.width - 240)}px`;
      tip.style.top = `${Math.min(y, cr.height - 70)}px`;
    }
    container.addEventListener("pointermove", (e) => {
      if (!tip.hidden) moveFcTip(e);
    }, { signal });

    // Shared detail card (one per chart, shown below clicked op block)
    const detailCard = document.createElement("div");
    detailCard.className = "fc-detail-card";
    detailCard.hidden = true;
    let activeBlock = null;

    function closeDetail() {
      detailCard.hidden = true;
      if (activeBlock) { activeBlock.classList.remove("fc-block-active"); activeBlock = null; }
    }

    function showDetail(opData) {
      const { op, b, serialUs, totalUs: opTotalUs, tone } = opData;
      const xnnLabel = op.xnnpack_supported
        ? (op.xnnpack_chain_id >= 0 ? `XNN Chain ${op.xnnpack_chain_id}` : "XNN (no chain)")
        : "CPU fallback";

      detailCard.innerHTML = "";

      const head = document.createElement("div");
      head.className = "fc-detail-head";
      const title = document.createElement("strong");
      title.textContent = `#${padOp(op.index)} ${op.name}`;
      const closeBtn = document.createElement("button");
      closeBtn.className = "fc-detail-close";
      closeBtn.textContent = "×";
      closeBtn.addEventListener("click", closeDetail);
      head.append(title, closeBtn);

      const grid = document.createElement("div");
      grid.className = "fc-detail-grid";

      const rows = [
        ["Cold total", formatUs(opTotalUs), tone],
        ["Steady",   formatUs(Math.max(0, opTotalUs - b.packingUs - b.breakUs)), ""],
        ["Compute",  formatUs(b.computeUs), b.computeUs >= b.memoryUs && b.computeUs > 0 ? "compute" : ""],
        ["Memory",   formatUs(b.memoryUs),  b.memoryUs > b.computeUs ? "memory" : ""],
        ["Cold setup", formatUs(b.packingUs + b.breakUs), b.packingUs + b.breakUs > 0 ? "overhead" : ""],
        ["Fallback", formatUs(b.fallbackUs), b.fallbackUs > 0 ? "fallback" : ""],
        ["Bound",    boundDisplayLabel(op.static_bound_guess), ""],
        ["MACs",     formatNumber(op.macs), ""],
        ["Bytes",    op.estimated_bytes == null ? "N/A" : formatBytes(op.estimated_bytes), ""],
        ["Delegate", xnnLabel, op.xnnpack_supported ? "compute" : "fallback"],
      ];
      if (op.quant_hole) rows.push(["⚠ Precision boundary", "8-bit/FP32 transition op", "overhead"]);

      for (const [label, value, rowTone] of rows) {
        const lEl = document.createElement("span");
        lEl.className = "fc-detail-label";
        lEl.textContent = label;
        const vEl = document.createElement("span");
        vEl.className = `fc-detail-value${rowTone ? ` fc-detail-${rowTone}` : ""}`;
        vEl.textContent = value;
        grid.append(lEl, vEl);
      }

      const footer = document.createElement("div");
      footer.className = "fc-detail-footer";
      const graphLink = document.createElement("button");
      graphLink.className = "fc-detail-graph-link";
      graphLink.textContent = "→ Open in Graph Explorer";
      graphLink.addEventListener("click", () => { closeDetail(); jumpToGraphOp(op.index); });
      footer.append(graphLink);

      detailCard.append(head, grid, footer);
      detailCard.hidden = false;
    }

    function makeBlock(cls, pct, lines, opDataRef) {
      const el = document.createElement("div");
      el.className = `fc-block ${cls}`;
      el.style.width = `${pct.toFixed(4)}%`;
      const accessibleLabel = lines.filter(Boolean).join(" / ");
      el.title = accessibleLabel;
      el.setAttribute("aria-label", accessibleLabel);
      el.addEventListener("pointerenter", (e) => { e.stopPropagation(); showTip(e, lines); });
      el.addEventListener("pointerleave", () => { tip.hidden = true; });
      if (opDataRef != null) {
        el.setAttribute("role", "button");
        el.tabIndex = 0;
        const activate = () => {
          tip.hidden = true;
          if (activeBlock === el) { closeDetail(); return; }
          if (activeBlock) activeBlock.classList.remove("fc-block-active");
          activeBlock = el;
          el.classList.add("fc-block-active");
          showDetail(opDataRef);
        };
        el.addEventListener("click", (e) => {
          e.stopPropagation();
          activate();
        });
        el.addEventListener("keydown", (e) => {
          if (e.key !== "Enter" && e.key !== " ") return;
          e.preventDefault();
          e.stopPropagation();
          activate();
        });
      }
      return el;
    }

    const chart = document.createElement("div");
    chart.className = "fc-chart";
    const labeledRow = (row, label, detail) => {
      const shell = document.createElement("div");
      shell.className = "fc-level-row";
      const rowLabel = document.createElement("div");
      rowLabel.className = "fc-level-label";
      const name = document.createElement("strong");
      name.textContent = label;
      const description = document.createElement("span");
      description.textContent = detail;
      rowLabel.append(name, description);
      shell.append(rowLabel, row);
      return shell;
    };

    // ── Level 0: Total ──────────────────────────────────────────────────────
    const row0 = document.createElement("div");
    row0.className = "fc-row fc-row-total";
    const totalBlock = makeBlock("fc-total", 100, [
      `Cold-start estimated: ${formatUs(totalUs)}; steady-state ${formatUs(steadyUs)}; one-time packing ${formatUs(packingUs)}; partition-planning setup ${formatUs(boundarySetupUs)}`,
      `${chains.length} partition(s) / ${opData.length} ops`,
    ]);
    totalBlock.textContent = `${formatUs(totalUs)} cold`;
    row0.append(totalBlock);

    // ── Level 1: XNN chains ─────────────────────────────────────────────────
    const row1 = document.createElement("div");
    row1.className = "fc-row fc-row-chain";

    // ── Level 2: Op types within each chain ────────────────────────────────
    const row2 = document.createElement("div");
    row2.className = "fc-row fc-row-type";

    // ── Level 3: Individual ops ─────────────────────────────────────────────
    const row3 = document.createElement("div");
    row3.className = "fc-row fc-row-op";

    for (const [chainId, chainOps] of chains) {
      const chainUs = chainOps.reduce((s, d) => s + d.totalUs, 0);
      const chainPct = (chainUs / totalUs) * 100;
      const isDelegate = chainId !== -1;
      const chainLabel = isDelegate ? `XNN C${chainId}` : "CPU";
      const chainCls = isDelegate ? "fc-chain-delegated" : "fc-chain-fallback";

      const chainBlock = makeBlock(chainCls, chainPct, [
        isDelegate ? `XNNPACK Chain ${chainId}` : "CPU Fallback",
        `${formatUs(chainUs)}  (${(chainUs / totalUs * 100).toFixed(1)}%)`,
        `${chainOps.length} ops`,
      ]);
      chainBlock.textContent = chainPct >= 6 ? chainLabel : "";
      row1.append(chainBlock);

      // Sub-group by op name within chain
      const typeMap = new Map();
      for (const d of chainOps) {
        if (!typeMap.has(d.op.name)) typeMap.set(d.op.name, []);
        typeMap.get(d.op.name).push(d);
      }

      for (const [typeName, typeOps] of typeMap) {
        const typeUs = typeOps.reduce((s, d) => s + d.totalUs, 0);
        const typePct = (typeUs / totalUs) * 100;
        const tone = typeOps[0].tone;
        const typeBlock = makeBlock(`fc-type fc-tone-${tone}`, typePct, [
          typeName,
          `${formatUs(typeUs)}  (${(typeUs / totalUs * 100).toFixed(1)}%)`,
          `${typeOps.length} op(s) / ${tone}`,
        ]);
        typeBlock.textContent = typePct >= 7 ? typeName : "";
        row2.append(typeBlock);

        for (const d of typeOps) {
          const opPct = (d.totalUs / totalUs) * 100;
          const opBlock = makeBlock(`fc-op fc-tone-${d.tone}${d.op.quant_hole ? " fc-quant-hole" : ""}`, opPct, [
            `#${padOp(d.op.index)} ${d.op.name}`,
            `${formatUs(d.totalUs)}  (${(d.totalUs / totalUs * 100).toFixed(1)}%)`,
            `compute ${formatUs(d.b.computeUs)}  mem ${formatUs(d.b.memoryUs)}`,
            `overhead ${formatUs(d.serialUs)}  bound: ${d.op.static_bound_guess || "?"}`,
            d.op.quant_hole ? "⚠ activation precision boundary" : "",
          ].filter(Boolean), d);
          opBlock.textContent = opPct >= 4 ? `#${padOp(d.op.index)}` : "";
          row3.append(opBlock);
        }
      }
    }

    chart.append(
      labeledRow(row0, "Total", "estimated latency"),
      labeledRow(row1, "Partitions", "delegate and CPU regions"),
      labeledRow(row2, "Op families", "grouped operator types"),
      labeledRow(row3, "Operators", "individual graph nodes"),
    );
    container.append(chart, detailCard);

    // Close detail when clicking outside chart rows
    container.addEventListener("click", (e) => {
      if (!e.target.closest(".fc-block") && !e.target.closest(".fc-detail-card")) closeDetail();
    }, { signal });
  }

  return {
    render,
    buildTargetComparisonRows,
    resetTargetComparisonCache,
    updateVisibility,
    renderPerfTimeline,
  };
}
