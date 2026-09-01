import { analyzeDeepBomInWorker } from "./deepbom-worker-client.js";

export function indexQuantRiskOps(rows = []) {
  return new Map(rows.map((risk) => [Number(risk.op_index), risk]));
}

export function quantRiskForOp(riskMap, op, position) {
  return riskMap.get(Number(op?.index ?? position));
}

export function createDeepBomWorkspace(ctx) {
  const {
    applyProtectedOrtCompatibilityEvidence,
    applyProtectedTfliteDelegateCompatibilityEvidence,
    applyProtectedXnnpackSelectorEvidence,
    assessedOpLogicalBytes,
    compute_model_tomography,
    deepBomCaveats,
    deepBomGrid,
    deepBomMetric,
    deepBomNotes,
    deepBomPanel,
    deepBomProtocols,
    deepBomStatus,
    deploymentFrontierController,
    downloadDeepBom,
    drawMvDepth,
    drawMvFilter,
    deploymentSensitivityGrid,
    deploymentSensitivityNotes,
    deploymentSensitivityProtocols,
    deploymentSensitivityResultPanel,
    deploymentSensitivityStatus,
    formatBytes,
    formatNumber,
    formatPercent1,
    layer_landscape_grid,
    mk,
    modelIdentity,
    modelSupportsCapability,
    modelViewerPanel,
    nextPaint,
    perturbationGrid,
    perturbationNotes,
    perturbationResultPanel,
    perturbationStatus,
    protocolBlock,
    renderAuditClaimBoundary,
    renderCurrentKernelInspector,
    renderGraphOpRows,
    renderOpDetail,
    runDeepBom,
    runtimeBasinGrid,
    runtimeBasinNotes,
    runtimeBasinResultPanel,
    runtimeBasinStatus,
    score100,
    selectedTargetId,
    setStatus,
    shortError,
    statusForEntropy,
    updateModuleAccessState,
    updateWorkflowState,
  } = ctx;

async function runDeepBomAnalysis(manifest, { activateWorkflow = true } = {}) {
  runDeepBom.disabled = true;
  deepBomPanel.hidden = false;
  deepBomStatus.textContent = "Loading…";
  deepBomNotes.textContent = "The advanced module is fetched on demand after capability authorization.";
  deepBomProtocols.replaceChildren();
  if (deepBomCaveats) { deepBomCaveats.replaceChildren(); deepBomCaveats.hidden = true; }
  // Skeleton cards stay visible while loading; mark them as "running" to pulse differently
  deepBomGrid.querySelectorAll(".deepbom-metric.skeleton").forEach(c => c.classList.add("running"));
  try {
    deepBomStatus.textContent = "Analyzing…";
    await nextPaint();
    ctx.deepBomResult = await analyzeDeepBomInWorker({
      manifest,
      modelBytes: ctx.currentModelBytes,
      analysis: ctx.current,
      onStatus: (phase) => { deepBomStatus.textContent = phase; },
    });
    const protectedFormat = String(ctx.current?.format || "").toLowerCase();
    if (protectedFormat === "onnx" && ctx.deepBomResult?.ort_compatibility_evidence) {
      applyProtectedOrtCompatibilityEvidence(ctx.current, ctx.deepBomResult.ort_compatibility_evidence);
      deploymentFrontierController.render(ctx.current);
    }
    if (protectedFormat === "tflite" && ctx.deepBomResult?.xnnpack_selector_evidence) {
      applyProtectedXnnpackSelectorEvidence(ctx.current, ctx.deepBomResult.xnnpack_selector_evidence);
      renderGraphOpRows(ctx.current);
    }
    if (protectedFormat === "tflite" && ctx.deepBomResult?.tflite_delegate_compatibility_evidence) {
      applyProtectedTfliteDelegateCompatibilityEvidence(
        ctx.current,
        ctx.deepBomResult.tflite_delegate_compatibility_evidence,
      );
      deploymentFrontierController.render(ctx.current);
    }
    renderCurrentKernelInspector(true);
    renderAuditClaimBoundary(ctx.current?.format, ctx.current);
    renderDeepBomResult(ctx.deepBomResult, { updateWorkflow: activateWorkflow });
    if (ctx.current && ctx.selectedOpIndex != null) renderOpDetail(ctx.current, ctx.selectedOpIndex);
    setStatus("DEEPBOM complete", "ok");
    return ctx.deepBomResult;
  } catch (error) {
    console.error(error);
    ctx.deepBomResult = {
      schema: "deepbom.v1",
      generated_at: new Date().toISOString(),
      model: modelIdentity(),
      status: "failed",
      error: shortError(error),
    };
    deepBomStatus.textContent = "Failed";
    deepBomGrid.replaceChildren(deepBomMetric("Analysis", "Failed", shortError(error), { tone: "risk", label: "risk", criteria: "WASM module did not complete. Check that the model file is valid." }));
    deepBomNotes.textContent = "DEEPBOM did not complete. The module encountered an error before producing proxy evidence.";
    setStatus("DEEPBOM failed", "error");
    return ctx.deepBomResult;
  } finally {
    runDeepBom.disabled = false;
  }
}

function resetDeepBomPanel() {
  deepBomPanel.hidden = true;
  deepBomStatus.textContent = "Not loaded";
  deepBomGrid.replaceChildren();
  deepBomProtocols.replaceChildren();
  if (deepBomCaveats) { deepBomCaveats.replaceChildren(); deepBomCaveats.hidden = true; }
  deepBomNotes.textContent = "";
  downloadDeepBom.hidden = true;
}

function renderDeepBomSkeleton() {
  deepBomPanel.hidden = false;
  deepBomStatus.textContent = "Ready";
  deepBomNotes.textContent = "Run Artifact Geometry to analyze weight, entropy, and topology proxy signals.";
  deepBomProtocols.replaceChildren();
  if (deepBomCaveats) { deepBomCaveats.replaceChildren(); deepBomCaveats.hidden = true; }
  downloadDeepBom.hidden = true;

  function skelCard(label, hint) {
    const card = document.createElement("div");
    card.className = "deepbom-metric skeleton";
    const lbl = document.createElement("span"); lbl.textContent = label;
    const val = document.createElement("strong");
    const sv = document.createElement("span"); sv.className = "skel-val";
    val.append(sv);
    const p = document.createElement("p"); p.textContent = hint;
    card.append(lbl, val, p);
    return card;
  }

  function skelSection(title, metrics) {
    const wrap = document.createElement("div");
    const h = document.createElement("h3"); h.className = "deepbom-section-title"; h.textContent = title;
    const grid = document.createElement("div"); grid.className = "deepbom-section-grid";
    grid.append(...metrics.map(m => skelCard(m.label, m.hint)));
    wrap.append(h, grid);
    return wrap;
  }

  deepBomGrid.replaceChildren(
    skelSection("Experimental Composites", [
      { label: "Artifact Composite", hint: "Unvalidated fixed-weight summary of artifact descriptors; not an accuracy or stability score." },
      { label: "Quant Composite", hint: "Unvalidated combination of scale CV, quant-risk count, metadata coverage, and byte repetition." },
      { label: "Topology Composite", hint: "Unvalidated combination of low-intensity mix, chain breaks, fallback bytes, and byte entropy." },
    ]),
    skelSection("XNNPACK Selector Evidence", [
      { label: "Selector Coverage", hint: "Eligible TFLite compute ops assessed by the protected pinned-source rulepack." },
      { label: "Source Configurations", hint: "Configurations remaining after artifact-visible and planning-profile selectors." },
      { label: "No-match Ops", hint: "Signatures with no match in the enumerated pinned source paths." },
    ]),
    skelSection("File Fingerprint", [
      { label: "Byte Entropy", hint: "Raw model-byte diversity in bits per byte." },
      { label: "Quantized Tensors", hint: "Fraction of tensors declared INT8/UINT8." },
      { label: "Zero Byte Ratio", hint: "Proportion of zero bytes — may indicate padding or sparsity." },
      { label: "High Bit Ratio", hint: "Proportion of bytes ≥ 128." },
      { label: "Repeated Byte Ratio", hint: "Max single-byte frequency." },
      { label: "Per-channel Ratio", hint: "Fraction of quantized tensors with per-axis scale metadata." },
    ]),
    skelSection("Weight Tensors", [
      { label: "W.Sparsity", hint: "Mean fraction of near-zero values across weight tensors." },
      { label: "W.Range Util.", hint: "Mean quantization range utilization across INT8 tensors." },
    ]),
  );
}

function renderDriftSkeleton() {
  function skelCard(label, hint) {
    const card = document.createElement("div");
    card.className = "deepbom-metric skeleton";
    const lbl = document.createElement("span"); lbl.textContent = label;
    const val = document.createElement("strong");
    const sv = document.createElement("span"); sv.className = "skel-val";
    val.append(sv);
    const p = document.createElement("p"); p.textContent = hint;
    card.append(lbl, val, p);
    return card;
  }

  // Perturbation panel skeleton
  perturbationResultPanel.hidden = false;
  perturbationStatus.textContent = "Ready";
  perturbationNotes.textContent = "Click Run Drift Analysis to probe input/output sensitivity and weight perturbation response.";
  perturbationGrid.replaceChildren(
    skelCard("RMS Drift", "Root-mean-square output change after local input perturbation."),
    skelCard("Max Abs Drift", "Largest absolute output change observed under perturbation."),
    skelCard("Cosine Distance", "1 − cosine similarity across flattened outputs."),
    skelCard("Top-1 Flip", "Whether argmax of the first output tensor changed under perturbation."),
    skelCard("Weight Perturbation", "Max drift when weight bytes are locally mutated."),
    skelCard("Layer Robustness", "Worst-case layer under per-layer weight perturbation sweep."),
    skelCard("Haar Sweep", "Spatial frequency sensitivity across standard Haar patterns."),
    skelCard("Freq Profile", "Coarse vs fine frequency dominance in model sensitivity."),
    skelCard("Position Sensitivity", "Translation-invariance vs position-sensitivity from stride-shifted patterns."),
    skelCard("Run Variance", "Timing stability between baseline and perturbed local runs."),
  );

  // Backend consistency panel skeleton
  runtimeBasinResultPanel.hidden = false;
  runtimeBasinStatus.textContent = "Ready";
  runtimeBasinNotes.textContent = "Click Run Drift Analysis to check backend availability and cross-backend output drift.";
  runtimeBasinGrid.replaceChildren(
    skelCard("Backends OK", "How many browser runtime backends completed successfully."),
    skelCard("Reference", "First successful backend used as the drift reference."),
    skelCard("Max Backend Drift", "Largest absolute drift vs the reference backend across all paths."),
    skelCard("Mean Run", "Average single local run time across successful backends."),
    skelCard("Failures", "Number of backends that did not complete."),
    skelCard("Backend Caveat", "Per-backend interpretation note (slow path, dtype mismatch, etc)."),
  );
}

function renderDeploymentSensitivitySkeleton() {
  function skelCard(label, hint) {
    const card = document.createElement("div");
    card.className = "deepbom-metric skeleton";
    const lbl = document.createElement("span"); lbl.textContent = label;
    const val = document.createElement("strong");
    const sv = document.createElement("span"); sv.className = "skel-val";
    val.append(sv);
    const p = document.createElement("p"); p.textContent = hint;
    card.append(lbl, val, p);
    return card;
  }

  deploymentSensitivityResultPanel.hidden = false;
  deploymentSensitivityStatus.textContent = "Ready";
  deploymentSensitivityNotes.textContent = "Run Deployment Sensitivity to collect local finite-difference output observations and an explicitly unvalidated composite.";
  deploymentSensitivityProtocols.replaceChildren();
  deploymentSensitivityGrid.replaceChildren(
    skelCard("Deploy Curvature", "Finite-difference probes executed on the deployed TFLite/LiteRT function in this browser."),
    skelCard("Directional Curvature", "Central finite-difference RMS normalized by input perturbation L2² along one local input direction."),
    skelCard("Raw 2nd Diff RMS", "RMS of y(x+eps) − 2y(x) + y(x−eps) across returned output tensors."),
    skelCard("Local Lipschitz", "Max first-order RMS output drift divided by input perturbation L2 norm."),
    skelCard("Tested Rank-Stability Radius", "Largest tested epsilon band that preserved first-output argmax for this synthetic local direction."),
    skelCard("Experimental Stability Composite", "Unvalidated fixed-weight summary; inspect curvature, drift, margin, and rank-change components instead."),
    skelCard("Decision Margin", "Top-1 minus top-2 margin on the first output tensor."),
    skelCard("Top-1 Stability", "Whether the first-output argmax survives ±eps and 2eps probes."),
    skelCard("Probe Consistency", "Whether the 2eps probe stays distinct from the ±eps drift summaries."),
    skelCard("Input Probe", "Perturbed value count, epsilon, and L2 norm of the synthetic input probe."),
  );
}

function deepBomSignalValue(result, label, fallback) {
  const signal = Array.isArray(result?.signals)
    ? result.signals.find((item) => String(item.label || "").toLowerCase() === label.toLowerCase())
    : null;
  return signal?.value || fallback;
}

function renderDeepBomResult(result, { updateWorkflow = true } = {}) {
  if (updateWorkflow) updateWorkflowState("module");
  deepBomStatus.textContent = result.posture || "Complete";
  const scoresAssessed = result.score_assessment?.status === "ASSESSED";
  const compositeStatus = scoresAssessed
    ? { tone: "info", label: "experimental", criteria: "Deterministic composite with disclosed fixed weights; no validated accuracy, latency, robustness, or release threshold." }
    : { tone: "info", label: "not assessable" };
  const entropyStatus = statusForEntropy(result.byte_entropy_bits_per_byte || 0);

  // Score strings with heuristic sensitivity band (±10% input perturbation low/high)
  function scoreWithSensitivity(mid01, low01, high01) {
    if (mid01 == null) return "NOT_ASSESSABLE";
    const midStr = score100(mid01);
    if (low01 == null || high01 == null || Math.abs(high01 - low01) < 0.001) return midStr;
    return `${midStr} (${score100(low01)}–${score100(high01)})`;
  }
  const basinMid = result.basin_proxy_score;
  const quantMid = result.quant_stress_score;
  const topoMid = result.topology_stress_score;
  const basinScore = deepBomSignalValue(result, "Artifact composite",
    scoreWithSensitivity(basinMid, result.basin_proxy_sensitivity_low, result.basin_proxy_sensitivity_high));
  const quantScore = deepBomSignalValue(result, "Quant composite",
    scoreWithSensitivity(quantMid, result.quant_stress_sensitivity_low, result.quant_stress_sensitivity_high));
  const topologyScore = deepBomSignalValue(result, "Topology composite",
    scoreWithSensitivity(topoMid, result.topology_stress_sensitivity_low, result.topology_stress_sensitivity_high));

  // Weight tensor stats (from actual weight bytes)
  const ws = result.weight_tensor_stats || {};
  const hasWeightStats = (ws.analyzed_count || 0) > 0;
  const sparsityTone = (ws.mean_exact_zero_ratio || 0) > 0.5 ? "warn" : "info";
  const rangeUtilTone = ws.has_quantized && (ws.mean_range_utilization || 0) > 0 && (ws.mean_range_utilization || 0) < 0.4 ? "warn" : "info";

  // New fields from WASM rebuild
  const maxScaleCv = result.max_quant_scale_cv || 0;
  const quantizedRatio = result.quantized_tensor_ratio || 0;
  const quantRiskOps = Array.isArray(result.quant_risk_ops) ? result.quant_risk_ops : [];
  const memBoundCount = result.memory_bound_op_count || 0;
  const computeBoundCount = result.compute_bound_op_count || 0;

  // Helper: creates a labeled section of metric cards within the grid
  function deepBomSection(title, items, extra = null) {
    const wrap = document.createElement("div");
    const h = document.createElement("h3");
    h.className = "deepbom-section-title";
    h.textContent = title;
    const grid = document.createElement("div");
    grid.className = "deepbom-section-grid";
    grid.append(...items.filter(Boolean));
    wrap.append(h, grid);
    if (extra) wrap.append(extra);
    return wrap;
  }

  const proxySection = deepBomSection("Experimental Composites", [
    deepBomMetric("Artifact Composite", basinScore, scoresAssessed ? "Fixed-weight artifact summary. The range is a deterministic +/-10% assumption sensitivity envelope, not a confidence interval or measured stability." : result.score_assessment?.reason || "Required score inputs were not assessed.", { ...compositeStatus, ...(basinMid == null ? {} : { score01: basinMid }) }),
    deepBomMetric("Quant Composite", quantScore, scoresAssessed ? "Fixed-weight combination of scale coefficient-of-variation, quant-risk count, metadata coverage, and byte repetition. No validated decision threshold." : result.score_assessment?.reason || "Required score inputs were not assessed.", { ...compositeStatus, ...(quantMid == null ? {} : { score01: quantMid }) }),
    deepBomMetric("Topology Composite", topologyScore, scoresAssessed ? "Fixed-weight combination of low-intensity mix, predicted chain breaks, fallback bytes, and byte entropy. It is not a latency or accuracy estimate." : result.score_assessment?.reason || "Required score inputs were not assessed.", { ...compositeStatus, ...(topoMid == null ? {} : { score01: topoMid }) }),
  ]);

  const fingerprintSection = deepBomSection("File Fingerprint", [
    deepBomMetric("Byte Entropy", `${Number(result.byte_entropy_bits_per_byte || 0).toFixed(2)} bits/B`, "Raw model-byte diversity. Watch < 4.5 bits/B may indicate atypical artifact structure.", entropyStatus),
    deepBomMetric("Quantized Tensors", formatPercent1(quantizedRatio), "Fraction of all tensors declared quantized (INT8/UINT8).", { tone: "info", label: "info", criteria: "Coverage is descriptive. Inspect tensor contracts and quantization findings before drawing a deployment conclusion." }),
    deepBomMetric("Zero Byte Ratio", formatPercent1(result.zero_byte_ratio || 0), "Proportion of zero bytes. Elevated ratios may indicate padding or sparsity.", { tone: "info", label: "info", criteria: "Interpret with model format and sparsity/packing context." }),
    deepBomMetric("High Bit Ratio", formatPercent1(result.high_bit_ratio || 0), "Proportion of bytes ≥ 128. Complements zero-byte ratio for artifact fingerprinting.", { tone: "info", label: "info", criteria: "Interpret alongside byte entropy and zero-byte ratio." }),
    deepBomMetric("Repeated Byte Ratio", formatPercent1(result.repeated_byte_ratio || 0), "Max single-byte frequency. High values indicate sparse or padding-heavy artifacts.", { tone: "info", label: "info", criteria: "Ratio >20% may indicate padding, sparsity, or aggressive quantization." }),
    deepBomMetric("Per-channel Ratio", formatPercent1(result.per_channel_tensor_ratio || 0), "Fraction of quantized tensors with per-axis scale metadata.", { tone: "info", label: "info", criteria: "Higher can help INT8 conv accuracy; model topology determines relevance." }),
  ]);

  const selector = result.xnnpack_selector_evidence || null;
  let selectorSection = null;
  if (selector) {
    const complete = selector.assessment_status === "complete";
    const selectorActions = complete ? document.createElement("div") : null;
    if (selectorActions) {
      selectorActions.className = "deepbom-selector-actions";
      const inspect = document.createElement("button");
      inspect.type = "button";
      inspect.className = "secondary-action";
      inspect.textContent = "Open Kernel Inspector";
      inspect.addEventListener("click", () => {
        document.querySelector('[data-workflow-step="graph"]')?.click();
        document.querySelector('[data-explorer-tab="kernels"]')?.click();
        document.querySelector('[data-kernel-filter="selector"]')?.click();
        requestAnimationFrame(() => document.getElementById("kernelInspectorPanel")?.scrollIntoView({ behavior: "auto", block: "start" }));
      });
      selectorActions.append(inspect);
    }
    const assessed = Number(selector.assessed_op_count || 0);
    const candidateOps = Number(selector.candidate_op_count || 0);
    const configurations = Number(selector.candidate_configuration_count || 0);
    const uniqueOps = Number(selector.unique_candidate_op_count || 0);
    const ambiguousOps = Number(selector.ambiguous_candidate_op_count || 0);
    const noMatch = Number(selector.no_match_op_count || 0);
    const worstTailOps = (selector.worst_case_tail_op_indices || []).map((index) => `#${String(index).padStart(3, "0")}`).join(", ") || "none";
    const unresolvedOps = Number(selector.unresolved_selector_op_count || 0);
    const unresolvedDimensions = Number(selector.unresolved_selector_dimension_count || 0);
    selectorSection = deepBomSection("XNNPACK Selector Evidence", [
      deepBomMetric(
        "Selector Coverage",
        complete ? `${assessed}/${assessed} ops` : "Not applicable",
        complete ? "Every eligible TFLite CONV_2D, DEPTHWISE_CONV_2D, and FULLY_CONNECTED op received a source-enumerated candidate set or explicit no-match result." : "The pinned TFLite XNNPACK selector rulepack is not applicable to this format or planning profile.",
        complete ? { tone: "good", label: "complete", criteria: selector.evidence_boundary } : { tone: "info", label: "n/a" },
      ),
      deepBomMetric(
        "Candidate Ops",
        complete ? `${candidateOps} ops` : "Not applicable",
        complete ? "Ops with one or more configurations remaining after artifact-visible and planning-profile selectors." : "No TFLite selector assessment was executed.",
        { tone: "info", label: complete ? "source" : "n/a" },
      ),
      deepBomMetric(
        "Unique / Ambiguous",
        complete ? `${uniqueOps} / ${ambiguousOps}` : "Not applicable",
        complete ? "Unique means one source configuration remains; ambiguous means compile, host, lowering, or runtime dispatch selectors still prevent a single microkernel claim." : "No TFLite selector assessment was executed.",
        ambiguousOps ? { tone: "warn", label: "unresolved" } : { tone: complete ? "good" : "info", label: complete ? "unique" : "n/a" },
      ),
      deepBomMetric(
        "Source Configurations",
        complete ? `${configurations}` : "Not applicable",
        complete ? `Enumerated from google/XNNPACK@${selector.xnnpack_source_commit || "unbound"}; this count does not identify the executed runtime microkernel.` : "No source configuration set was emitted.",
        { tone: "info", label: complete ? "pinned" : "n/a" },
      ),
      deepBomMetric(
        "No-match Ops",
        complete ? `${noMatch}` : "Not applicable",
        complete ? "No-match means the protected GEMM/DWCONV source paths had no configuration for the visible signature; it does not prove delegate rejection through every lowering." : "No source paths were assessed.",
        noMatch ? { tone: "warn", label: "review" } : { tone: complete ? "good" : "info", label: complete ? "none" : "n/a" },
      ),
      deepBomMetric(
        "Worst Candidate Tail",
        complete ? `${formatPercent1(selector.worst_case_tail_ratio || 0)} at ${worstTailOps}` : "Not applicable",
        complete ? "Maximum inactive output-channel lane ratio across each op's remaining source candidates. This is deterministic tile arithmetic, not measured latency." : "No candidate tail projection was emitted.",
        Number(selector.worst_case_tail_ratio || 0) > 0 ? { tone: "warn", label: "inspect" } : { tone: complete ? "good" : "info", label: complete ? "aligned" : "n/a" },
      ),
      deepBomMetric(
        "Unresolved Selector Gates",
        complete ? `${unresolvedDimensions} across ${unresolvedOps} ops` : "Not applicable",
        complete ? "Counts architecture identity, compile configuration, lowering shape, runtime dispatch, or unenumerated lowering-path gates retained in the decision ledger." : "No source selector gates were assessed.",
        unresolvedDimensions ? { tone: "warn", label: "explicit" } : { tone: complete ? "good" : "info", label: complete ? "none" : "n/a" },
      ),
    ], selectorActions);
  }

  // Op Map flame chart — interactive drill-down (groups overview → group detail → all ops)
  function buildOpsFlame() {
    const ops = Array.isArray(ctx.current?.ops) ? ctx.current.ops : [];
    if (!ops.length) return null;

    const riskMap = indexQuantRiskOps(quantRiskOps);
    const riskForOp = (op, position) => quantRiskForOp(riskMap, op, position);
    const opWS = Array.isArray(result.op_weight_stats) ? result.op_weight_stats : [];
    const assessedTrafficOps = ops.filter((op) => assessedOpLogicalBytes(op) != null);
    const trafficCoverageComplete = assessedTrafficOps.length === ops.length;
    const flameWeight = (op) => trafficCoverageComplete ? Math.max(1, assessedOpLogicalBytes(op)) : 1;
    const totalBytes = Math.max(1, ops.reduce((sum, op) => sum + flameWeight(op), 0));
    const maxCv = Math.max(0.01, ...ops.map(op => op.quant_scale_cv || 0));

    // Layer group extraction from TFLite op name paths
    const SKIP_PREFIX = new Set(["sequential", "model", "keras_layer", "module", "tf", "serving_default"]);
    function layerGroup(name) {
      const first = (name || "").split(";")[0];
      const parts = first.split("/").filter(Boolean);
      if (parts.length <= 1) return parts[0] || name;
      let i = 0;
      while (i < parts.length - 1 && SKIP_PREFIX.has(parts[i].toLowerCase())) i++;
      return parts[Math.min(i, parts.length - 2)] || parts[0];
    }

    const PARAMETRIC_RE = /conv|fc|fully.connected|dense|lstm|embed|linear|matmul|prelu/i;
    const groupSpans = [];
    let curGroup = null;
    ops.forEach((op, i) => {
      const g = layerGroup(op.name);
      const risk = riskForOp(op, i);
      const rl = risk?.quant_risk === "risk" ? 2 : risk?.quant_risk === "warn" ? 1 : 0;
      const opWsEntry = opWS[i] || {};
      const isParametric = opWsEntry.has_weights || PARAMETRIC_RE.test(op.name || "");
      if (!curGroup || g !== curGroup.name) {
        if (curGroup) groupSpans.push(curGroup);
        curGroup = { name: g, start: i, end: i, flex: 0, worstRisk: 0, riskCount: 0, paramCount: 0 };
      }
      curGroup.end = i;
      curGroup.flex += flameWeight(op) / totalBytes * 1000;
      curGroup.worstRisk = Math.max(curGroup.worstRisk, rl);
      if (rl > 0) curGroup.riskCount++;
      if (isParametric) curGroup.paramCount++;
    });
    if (curGroup) groupSpans.push(curGroup);

    // Scaffold
    const wrap = document.createElement("div");
    const h = document.createElement("h3");
    h.className = "deepbom-section-title";
    h.textContent = "Op Map";
    wrap.append(h);

    const metaEl = document.createElement("p");
    metaEl.className = "flame-meta";
    const memCount = ops.filter(op => op.static_bound_guess === "memory-bound").length;
    metaEl.textContent = `${ops.length} ops · ${groupSpans.length} groups · ${quantRiskOps.length} quant-risk · ${memCount}/${ops.length} low-intensity · max CV ${maxCv.toFixed(2)} · width ${trafficCoverageComplete ? "assessed logical bytes" : `op count (traffic ${assessedTrafficOps.length}/${ops.length} assessed)`}`;
    wrap.append(metaEl);

    const legend = document.createElement("div");
    legend.className = "flame-legend";
    for (const [cls, label] of [["flame-ok","ok"],["flame-warn","warn"],["flame-risk","risk"],["flame-mem","low-intensity"],["flame-comp","high-intensity"],["flame-wt-ok","range ok"],["flame-wt-fp32","float"],["flame-other","none"]]) {
      const item = document.createElement("span");
      item.className = "flame-legend-item";
      const dot = document.createElement("i");
      dot.className = cls;
      item.append(dot, document.createTextNode(label));
      legend.append(item);
    }
    wrap.append(legend);

    // Floating tooltip
    const tooltip = document.createElement("div");
    tooltip.className = "flame-tooltip";
    tooltip.hidden = true;
    document.body.append(tooltip);
    const obs = new MutationObserver(() => {
      if (!wrap.isConnected) { tooltip.remove(); obs.disconnect(); }
    });
    obs.observe(deepBomGrid, { childList: true });

    function tipRow(k, v) {
      const d = document.createElement("div");
      d.className = "tip-row";
      const key = document.createElement("span"); key.className = "tip-key"; key.textContent = k;
      const val = document.createElement("span"); val.className = "tip-val"; val.textContent = v;
      d.append(key, val);
      tooltip.append(d);
    }

    function showOpTooltip(i) {
      const op = ops[i];
      const risk = riskForOp(op, i);
      const ws = opWS[i] || {};
      tooltip.innerHTML = "";
      const head = document.createElement("div"); head.className = "tip-head";
      const nm = document.createElement("strong"); nm.textContent = `#${i + 1} ${op.type || op.name}`;
      head.append(nm);
      if (risk) {
        const badge = document.createElement("em");
        badge.className = `tip-badge tip-${risk.quant_risk}`;
        badge.textContent = risk.quant_risk;
        head.append(badge);
      }
      tooltip.append(head);
      tipRow("Group", layerGroup(op.name));
      tipRow("Logical traffic", assessedOpLogicalBytes(op) == null ? `N/A (${op.estimated_bytes_reason || "shape or dtype unavailable"})` : formatBytes(op.estimated_bytes));
      const cv = op.quant_scale_cv || 0;
      const scaleMode = op.quant_scale_mode || "none";
      if (cv > 0) {
        tipRow("Scale CV", cv.toFixed(2));
      } else if (scaleMode === "per-axis") {
        tipRow("Scale CV", "0.00 (uniform channels)");
      } else if (scaleMode === "per-tensor") {
        tipRow("Scale CV", "N/A (per-tensor)");
      } else {
        tipRow("Scale CV", "N/A (no quant)");
      }
      tipRow("Bound", op.static_bound_guess || "unknown");
      if (!ws || !ws.has_weights) {
        tipRow("Weight", "activation op — no weight tensors");
      } else if ((ws.mean_range_util || 0) > 0) {
        tipRow("Range util", `${(ws.mean_range_util * 100).toFixed(0)}%${ws.mean_range_util < 0.4 ? " ⚠" : ""}`);
        if ((ws.mean_exact_zero_ratio || 0) > 0.01) tipRow("Exact-zero", `${(ws.mean_exact_zero_ratio * 100).toFixed(0)}%`);
        tipRow("Tensors", `${ws.weight_count || 1} weight tensor${(ws.weight_count || 1) !== 1 ? "s" : ""}`);
      } else {
        tipRow("Weight", `float32 (${ws.weight_count || 1} tensor${(ws.weight_count || 1) !== 1 ? "s" : ""})`);
      }
      if (risk?.detail) {
        const det = document.createElement("div"); det.className = "tip-detail"; det.textContent = risk.detail;
        tooltip.append(det);
      }
    }

    function showGroupTooltip(gi) {
      const g = groupSpans[gi];
      tooltip.innerHTML = "";
      const head = document.createElement("div"); head.className = "tip-head";
      const nm = document.createElement("strong"); nm.textContent = g.name;
      head.append(nm);
      if (g.worstRisk > 0) {
        const rl = g.worstRisk === 2 ? "risk" : "warn";
        const badge = document.createElement("em");
        badge.className = `tip-badge tip-${rl}`; badge.textContent = rl;
        head.append(badge);
      }
      tooltip.append(head);
      tipRow("Ops", `${g.end - g.start + 1} (op ${g.start + 1}–${g.end + 1})`);
      const groupTotal = g.end - g.start + 1;
      if (g.paramCount > 0) {
        tipRow("Parametric", `${g.paramCount}/${groupTotal} (have weights)`);
      } else {
        tipRow("Parametric", "none — activation ops only");
      }
      if (g.riskCount > 0) tipRow("Risk ops", `${g.riskCount}`);
      const hint = document.createElement("div"); hint.className = "tip-detail"; hint.textContent = "Click to drill in";
      tooltip.append(hint);
    }

    function positionTooltip(e) {
      const r = tooltip.getBoundingClientRect();
      let x = e.clientX + 16, y = e.clientY - 8;
      if (x + r.width > window.innerWidth - 12) x = e.clientX - r.width - 12;
      if (y + r.height > window.innerHeight - 12) y = e.clientY - r.height - 4;
      tooltip.style.left = `${x}px`; tooltip.style.top = `${y}px`;
    }

    // Controls + tracks container
    const controlsEl = document.createElement("div");
    controlsEl.className = "flame-controls";
    const tracksEl = document.createElement("div");
    wrap.append(controlsEl, tracksEl);

    // Tooltip event delegation on tracksEl (single set, permanent)
    tracksEl.addEventListener("mouseover", e => {
      const bar = e.target.closest("[data-oi],[data-gi]");
      if (!bar) { tooltip.hidden = true; return; }
      if (bar.dataset.gi !== undefined) showGroupTooltip(Number(bar.dataset.gi));
      else showOpTooltip(Number(bar.dataset.oi));
      tooltip.hidden = false;
    });
    tracksEl.addEventListener("mousemove", positionTooltip);
    tracksEl.addEventListener("mouseleave", () => { tooltip.hidden = true; });
    tracksEl.addEventListener("click", e => {
      const bar = e.target.closest("[data-oi]");
      if (!bar) return;
      tooltip.hidden = true;
    });

    // Helpers
    function makeTrackRow(label, cls = "") {
      const row = document.createElement("div"); row.className = "flame-row";
      const lbl = document.createElement("span"); lbl.className = "flame-label"; lbl.textContent = label;
      const track = document.createElement("div");
      track.className = `flame-track${cls ? " " + cls : ""}`;
      row.append(lbl, track);
      return { row, track };
    }

    function mkBar(flex, cls) {
      const b = document.createElement("div");
      b.className = `flame-bar ${cls}`;
      b.style.flexGrow = flex.toFixed(3);
      b.style.flexBasis = "0";
      return b;
    }

    function ctrlBtn(label, onClick, active = false) {
      const btn = document.createElement("button");
      btn.className = `flame-ctrl-btn${active ? " active" : ""}`;
      btn.textContent = label;
      btn.addEventListener("click", () => { tooltip.hidden = true; onClick(); });
      return btn;
    }

    // Build signal tracks for a given set of op indices, with proportional flex
    function buildDetailTracks(opIndices, refBytes) {
      const rawMaxCv = Math.max(0, ...opIndices.map(i => ops[i].quant_scale_cv || 0));
      const localMaxCv = Math.max(0.01, rawMaxCv);
      const rawMaxRangeGap = Math.max(0, ...opIndices.map(i => {
        const ws = opWS[i] || {};
        return (ws.has_weights && (ws.mean_range_util || 0) > 0) ? 1 - ws.mean_range_util : 0;
      }));
      const localMaxRangeGap = Math.max(0.01, rawMaxRangeGap);
      const hasRangeData = rawMaxRangeGap > 0 || opIndices.some(i => {
        const ws = opWS[i] || {};
        return ws.has_weights && (ws.mean_range_util || 0) > 0;
      });
      const { row: r1, track: t1 } = makeTrackRow("Quant");
      const { row: r3, track: t3 } = makeTrackRow("Bound");
      const { row: r4, track: t4 } = makeTrackRow("W.Range", "cv-track");

      // W.Range: note when no range data
      if (!hasRangeData) {
        const hasAnyWeights = opIndices.some(i => (opWS[i] || {}).has_weights);
        const rangeAbsent = document.createElement("span");
        rangeAbsent.className = "flame-cv-absent";
        if (hasAnyWeights) {
          rangeAbsent.textContent = "float32 weights — no INT8 range data";
        } else {
          rangeAbsent.textContent = "activation ops (ADD, RELU…) — no weight tensors";
        }
        t4.append(rangeAbsent);
      }

      // Determine Scale CV track state
      const hasCvData = rawMaxCv > 0.001;
      const { row: r2, track: t2 } = makeTrackRow("Scale CV", "cv-track");
      if (!hasCvData) {
        // Determine why CV is absent from the first op's scale mode
        const sampleMode = opIndices.length > 0 ? (ops[opIndices[0]].quant_scale_mode || "none") : "none";
        const cvAbsent = document.createElement("span");
        cvAbsent.className = "flame-cv-absent";
        cvAbsent.textContent = sampleMode === "per-axis" ? "uniform channels (CV ≈ 0)"
          : sampleMode === "per-tensor" ? "per-tensor quantization — no per-channel CV"
          : "no quantization metadata";
        t2.append(cvAbsent);
      }

      for (const i of opIndices) {
        const op = ops[i];
        const bytes = flameWeight(op);
        const flex = bytes / refBytes * 1000;
        const cv = op.quant_scale_cv || 0;
        const risk = riskForOp(op, i);
        const ws = opWS[i] || { has_weights: false, mean_exact_zero_ratio: 0, mean_near_zero_ratio: 0, mean_range_util: 0 };

        const b1 = mkBar(flex, risk ? `flame-${risk.quant_risk}` : "flame-ok");
        b1.style.height = "100%"; b1.dataset.oi = i;
        if (bytes / refBytes > 0.04) {
          const s = document.createElement("span"); s.textContent = op.type || ""; b1.append(s);
        }
        t1.append(b1);

        if (hasCvData) {
          const b2 = mkBar(flex, cv > 3 ? "flame-warn" : "flame-cv-low");
          b2.style.height = `${Math.max(5, Math.round(cv / localMaxCv * 100))}%`;
          b2.dataset.oi = i;
          t2.append(b2);
        }

        const bndCls = op.static_bound_guess === "memory-bound" ? "flame-mem"
          : op.static_bound_guess === "compute-bound" ? "flame-comp" : "flame-other";
        const b3 = mkBar(flex, bndCls);
        b3.style.height = "100%"; b3.dataset.oi = i;
        t3.append(b3);

        // W.Range: bar height = 1 - range_util (tall = poor coverage = bad)
        const sp = ws.mean_exact_zero_ratio || 0, ru = ws.mean_range_util || 0;
        const hasQuantWts = ws.has_weights && ru > 0;
        const hasFP32Wts = ws.has_weights && ru === 0;
        const rangeGap = hasQuantWts ? 1 - ru : 0;
        const wCls = !ws.has_weights ? "flame-other"
          : hasFP32Wts ? "flame-wt-fp32"
          : sp > 0.5 || ru < 0.4 ? "flame-warn" : "flame-wt-ok";
        const b4 = mkBar(flex, wCls);
        b4.style.height = hasQuantWts
          ? `${Math.max(5, Math.round(rangeGap / localMaxRangeGap * 100))}%`
          : (hasFP32Wts ? "40%" : "5%");
        b4.dataset.oi = i;
        t4.append(b4);
      }
      return [r1, r2, r3, r4];
    }

    // View mode state: "overview" | "all" | number (group index)
    let viewMode = "overview";

    function render() {
      controlsEl.replaceChildren();
      tracksEl.replaceChildren();
      if (viewMode === "overview") {
        // Controls: All ops button
        controlsEl.append(ctrlBtn("All ops", () => { viewMode = "all"; render(); }));

        // Groups track — each bar is clickable
        const { row: gRow, track: gTrack } = makeTrackRow("Groups", "group-track");
        for (let gi = 0; gi < groupSpans.length; gi++) {
          const g = groupSpans[gi];
          const cls = g.worstRisk === 2 ? "flame-risk" : g.worstRisk === 1 ? "flame-warn" : "flame-ok";
          const b = mkBar(g.flex, `flame-group ${cls}`);
          b.style.height = "100%"; b.style.cursor = "pointer";
          b.dataset.gi = gi;
          b.setAttribute("role", "button");
          b.tabIndex = 0;
          b.setAttribute("aria-label", `Open ${g.name} group, ops ${g.start + 1} through ${g.end + 1}`);
          if (g.flex > 12) {
            const s = document.createElement("span"); s.textContent = g.name; b.append(s);
          }
          b.addEventListener("click", () => { viewMode = gi; render(); });
          b.addEventListener("keydown", (event) => {
            if (event.key !== "Enter" && event.key !== " ") return;
            event.preventDefault();
            viewMode = gi;
            render();
          });
          gTrack.append(b);
        }
        tracksEl.append(gRow);

      } else if (viewMode === "all") {
        controlsEl.append(ctrlBtn("← Groups", () => { viewMode = "overview"; render(); }, false));
        buildDetailTracks(ops.map((_, i) => i), totalBytes).forEach(r => tracksEl.append(r));

      } else {
        // Drilled into a specific group
        const gi = viewMode;
        const g = groupSpans[gi];

        // Controls: back, all ops, group name breadcrumb, prev/next
        controlsEl.append(ctrlBtn("← Groups", () => { viewMode = "overview"; render(); }));
        controlsEl.append(ctrlBtn("All ops", () => { viewMode = "all"; render(); }));
        const crumb = document.createElement("span");
        crumb.className = "flame-crumb";
        crumb.textContent = `${g.name}  (op ${g.start + 1}–${g.end + 1}, ${g.end - g.start + 1} ops)`;
        controlsEl.append(crumb);
        if (gi > 0) {
          const prev = ctrlBtn("‹", () => { viewMode = gi - 1; render(); });
          prev.title = groupSpans[gi - 1].name;
          controlsEl.append(prev);
        }
        if (gi < groupSpans.length - 1) {
          const next = ctrlBtn("›", () => { viewMode = gi + 1; render(); });
          next.title = groupSpans[gi + 1].name;
          controlsEl.append(next);
        }

        // Build detail tracks for this group's ops only, scaled to group bytes
        const groupIdx = [];
        for (let i = g.start; i <= g.end; i++) groupIdx.push(i);
        const groupBytes = groupIdx.reduce((sum, i) => sum + flameWeight(ops[i]), 0);
        buildDetailTracks(groupIdx, groupBytes).forEach(r => tracksEl.append(r));
      }
    }

    render();

    // Risk ops collapsible
    if (quantRiskOps.length > 0) {
      const det = document.createElement("details");
      det.className = "deepbom-riskops-details";
      const sum = document.createElement("summary");
      sum.textContent = `${quantRiskOps.length} quant-risk op${quantRiskOps.length !== 1 ? "s" : ""}`;
      det.append(sum);
      const ul = document.createElement("ul");
      ul.className = "deepbom-riskops-list";
      for (const op of quantRiskOps) {
        const li = document.createElement("li");
        li.className = `riskop-${op.quant_risk}`;
        const name = document.createElement("strong"); name.textContent = op.name;
        const badge = document.createElement("em");
        badge.textContent = op.quant_risk;
        badge.className = `metric-status ${op.quant_risk === "risk" ? "warn" : "watch"}`;
        const cv = document.createElement("span");
        cv.textContent = op.quant_scale_cv > 0 ? ` CV ${op.quant_scale_cv.toFixed(2)}` : "";
        const detail = document.createElement("p"); detail.textContent = op.detail || "";
        li.append(name, badge, cv, detail);
        ul.append(li);
      }
      det.append(ul);
      wrap.append(det);
    }

    return wrap;
  }

  const sections = [proxySection, selectorSection, fingerprintSection].filter(Boolean);
  const opsFlame = buildOpsFlame();
  if (opsFlame) sections.push(opsFlame);

  // Weight Analysis with per-tensor table
  if (hasWeightStats) {
    const weightItems = [
      deepBomMetric(
        "Exact-zero Ratio",
        formatPercent1(ws.mean_exact_zero_ratio || 0),
        `Mean exact numerical-zero fraction across ${ws.analyzed_count} constant tensor(s). Quantized tensors use their declared zero point.`,
        { tone: sparsityTone, label: sparsityTone === "warn" ? "watch" : "ok", score01: ws.mean_exact_zero_ratio || 0, criteria: "High exact-zero ratios (>50%) may indicate sparse weights or padding; near-zero values are reported separately." },
      ),
      ws.has_quantized && (ws.mean_range_utilization || 0) > 0 ? deepBomMetric(
        "Range Utilization",
        formatPercent1(ws.mean_range_utilization || 0),
        "Mean fraction of available INT8/UINT8 range used by weight tensors.",
        { tone: rangeUtilTone, label: rangeUtilTone === "warn" ? "low" : "ok", score01: ws.mean_range_utilization || 0, criteria: "Below 40% suggests per-tensor quantization is imprecise." },
      ) : null,
      deepBomMetric(
        "Weight Entropy",
        `${Number(ws.mean_entropy_bits || 0).toFixed(2)} bits`,
        "Mean per-tensor Shannon entropy. Low = uniform/constant weights.",
        { tone: "info", label: "info" },
      ),
      deepBomMetric(
        "Weight L2 Norm",
        Number(ws.mean_l2_norm || 0).toFixed(3),
        "Mean RMS weight magnitude. Relative fingerprint; compare across versions.",
        { tone: "info", label: "info" },
      ),
      deepBomMetric(
        "Weight Anomalies",
        ws.anomaly_count ? `${ws.anomaly_count} tensor(s)` : "None",
        `Flagged for narrow range (<15%) or very high sparsity (>85%). ${ws.analyzed_count} tensor(s) examined.`,
        ws.anomaly_count
          ? { tone: "warn", label: "warn" }
          : { tone: "good", label: "ok" },
      ),
    ];

    let perTensorExtra = null;
    const perTensorRecords = Array.isArray(ws.per_tensor) ? ws.per_tensor : [];
    if (perTensorRecords.length > 0) {
      const sorted = [...perTensorRecords].sort((a, b) => (b.flagged ? 1 : 0) - (a.flagged ? 1 : 0));
      const det = document.createElement("details");
      det.className = "per-tensor-details";
      const sum = document.createElement("summary");
      sum.textContent = `Per-tensor (${sorted.length})`;
      det.append(sum);
      const table = document.createElement("table");
      table.className = "per-tensor-table";
      const thead = document.createElement("thead");
      const hr = document.createElement("tr");
      for (const col of ["Tensor", "Type", "Elements", "Bytes", "Exact zero", "Near zero", "Range", "Non-finite", "Entropy", "Status"]) {
        const th = document.createElement("th");
        th.textContent = col;
        hr.append(th);
      }
      thead.append(hr);
      table.append(thead);
      const tbody = document.createElement("tbody");
      for (const r of sorted) {
        const tr = document.createElement("tr");
        if (r.flagged) tr.className = "flagged-row";
        const shortName = r.name.includes("/") ? r.name.split("/").pop() : r.name;
        const cells = [
          [shortName || r.name || "—", r.name || ""],
          [r.dtype, ""],
          [formatNumber(r.element_count), ""],
          [formatNumber(r.storage_byte_count), r.element_count_source || ""],
          [formatPercent1(r.exact_zero_ratio), ""],
          [formatPercent1(r.near_zero_ratio), "epsilon=1e-6 for FP32; exact zero for integer tensors"],
          [r.range_utilization > 0 ? formatPercent1(r.range_utilization) : "—", ""],
          [formatNumber(r.non_finite_count || 0), r.quantized_code_domain || ""],
          [`${r.entropy_bits.toFixed(2)}`, ""],
          [r.flagged ? `⚑ ${r.flag_reason}` : "ok", ""],
        ];
        for (const [text, title] of cells) {
          const td = document.createElement("td");
          td.textContent = text;
          if (title) td.title = title;
          if (text.startsWith("⚑")) td.className = "flag-warn";
          else if (text === "ok") td.className = "flag-ok";
          tr.append(td);
        }
        tbody.append(tr);
      }
      table.append(tbody);
      det.append(table);
      perTensorExtra = det;
    }
    sections.push(deepBomSection("Weight Analysis", weightItems, perTensorExtra));
  }

  deepBomGrid.replaceChildren(...sections);
  deepBomGrid.querySelectorAll(".deepbom-metric").forEach((card, i) => {
    card.classList.add("filling");
    card.style.animationDelay = `${i * 35}ms`;
    card.addEventListener("animationend", () => { card.classList.remove("filling"); card.style.animationDelay = ""; }, { once: true });
  });
  deepBomProtocols.replaceChildren();

  // Only show anomalies (no scope/caveats boilerplate)
  if (deepBomCaveats) {
    const anomalies = Array.isArray(result.anomalies) ? result.anomalies : [];
    if (anomalies.length > 0) {
      deepBomCaveats.hidden = false;
      const aLabel = document.createElement("p");
      aLabel.className = "deepbom-caveats-label deepbom-anomaly-label";
      aLabel.textContent = "Cross-validation anomalies";
      const aList = document.createElement("ul");
      aList.className = "deepbom-caveats-list";
      for (const a of anomalies) {
        const li = document.createElement("li");
        li.textContent = a;
        aList.append(li);
      }
      deepBomCaveats.replaceChildren(aLabel, aList);
    } else {
      deepBomCaveats.hidden = true;
    }
  }

  // One-line scope summary. Composite values are intentionally not graded.
  const compositeNote = scoresAssessed ? "Experimental composites; inspect components" : "Artifact composites not assessable";
  const cvNote = maxScaleCv > 0 ? ` (scale CV ${maxScaleCv.toFixed(2)})` : "";
  const riskNote = quantRiskOps.length > 0 ? ` · ${quantRiskOps.length} quant-risk ops` : "";
  const memNote = memBoundCount > 0 ? ` · ${memBoundCount}/${memBoundCount + computeBoundCount} low-intensity` : "";
  const selectorNote = selector?.assessment_status === "complete" ? ` · selector ${selector.assessed_op_count || 0} ops / ${selector.candidate_configuration_count || 0} configurations` : "";
  deepBomNotes.textContent = `${compositeNote}${cvNote}${riskNote}${memNote}${selectorNote}`;

  downloadDeepBom.hidden = false;
  updateModuleAccessState();
}

function renderProtocolGroups(container, groups) {
  container?.replaceChildren(...groups.map((group) => protocolBlock(group.title, group.items)));
}


let modelTomography = null;
const mvState = { layerIdx: 0, ocIdx: 0, metric: "l2" };

function activeThemeColor(token, fallback) {
  return getComputedStyle(document.documentElement).getPropertyValue(token).trim() || fallback;
}

window.addEventListener("deepbom:themechange", () => {
  if (modelTomography?.length && modelViewerPanel && !modelViewerPanel.hidden) renderModelViewer(modelTomography);
});

function renderModelViewer(tomo) {
  if (!modelViewerPanel) return;
  modelViewerPanel.replaceChildren();
  if (!tomo?.length) { modelViewerPanel.hidden = true; return; }
  modelViewerPanel.hidden = false;

  const N = tomo.length;

  const explorer = mk("div");
  explorer.style.cssText = "border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--viz-panel);display:flex;flex-direction:column";

  const headerBar = mk("div");
  headerBar.style.cssText = "display:flex;align-items:center;gap:10px;padding:9px 14px;border-bottom:1px solid var(--line);background:var(--viz-panel-strong);flex-wrap:wrap";
  const titleEl = mk("span");
  titleEl.style.cssText = "font-size:13px;font-weight:600;color:var(--ink)";
  titleEl.textContent = `Model Explorer — ${N} weight layers`;
  const metricSel = document.createElement("select");
  metricSel.style.cssText = "margin-left:auto;background:var(--surface);color:var(--ink);border:1px solid var(--line-strong);border-radius:4px;padding:3px 8px;font-size:11px;cursor:pointer";
  [["l2","Weight RMS"],["haar_ll","Haar LL"],["haar_lh","Haar LH"],["haar_hl","Haar HL"],["haar_hh","Haar HH"],["snr","Quant SNR (dB)"]].forEach(([v,l]) => {
    const o = document.createElement("option"); o.value = v; o.textContent = l;
    if (v === mvState.metric) o.selected = true;
    metricSel.append(o);
  });
  headerBar.append(titleEl, metricSel);

  const body = mk("div");
  body.style.cssText = "display:flex;height:480px;overflow:hidden";

  const sidebar = mk("div");
  sidebar.style.cssText = "width:172px;min-width:172px;overflow-y:auto;border-right:1px solid var(--line);background:var(--viz-canvas);scrollbar-width:thin;scrollbar-color:var(--line) var(--viz-canvas);flex-shrink:0";

  const detail = mk("div");
  detail.style.cssText = "flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px;background:var(--viz-panel);scrollbar-width:thin;scrollbar-color:var(--line) var(--viz-panel)";

  body.append(sidebar, detail);
  explorer.append(headerBar, body);
  modelViewerPanel.append(explorer);

  function getMetricVal(entry) {
    switch (mvState.metric) {
      case "haar_ll": return entry.haar_ll ?? 0;
      case "haar_lh": return entry.haar_lh ?? 0;
      case "haar_hl": return entry.haar_hl ?? 0;
      case "haar_hh": return entry.haar_hh ?? 0;
      // SNR: low = most int8-quantization distortion, so invert (60 dB ceiling)
      // to keep hot color = needs attention. Already-quantized layers → 0.
      case "snr": return entry.quant_snr_valid ? Math.max(0, 60 - entry.quant_snr_db) : 0;
      default: {
        const v = entry.l2_per_oc;
        return v?.length ? v.reduce((a,b) => a+b,0) / v.length : 0;
      }
    }
  }

  // Sidebar item color from metric value 0–1
  function metricColor(t) {
    const r = Math.round(40 + 215 * t), g = Math.round(120 - 80 * t), b = Math.round(220 - 140 * t);
    return `rgb(${r},${g},${b})`;
  }

  function rebuildSidebar() {
    sidebar.replaceChildren();
    const allVals = tomo.map(getMetricVal);
    const vMax = Math.max(...allVals, 1e-10);

    tomo.forEach((entry, idx) => {
      const v = allVals[idx];
      const t = v / vMax;
      const col = metricColor(t);
      const isSelected = idx === mvState.layerIdx;

      const item = mk("div");
      item.dataset.idx = idx;
      item.style.cssText = `display:flex;align-items:center;gap:6px;padding:7px 8px 7px 10px;cursor:pointer;border-left:3px solid ${isSelected ? "var(--accent)" : "transparent"};border-bottom:1px solid var(--line);background:${isSelected ? "var(--accent-soft)" : "transparent"};transition:background 0.1s`;

      const dot = mk("span");
      dot.style.cssText = `width:7px;height:7px;border-radius:50%;flex-shrink:0;background:${col}`;

      const nameWrap = mk("div");
      nameWrap.style.cssText = "flex:1;min-width:0";
      const nameEl = mk("div");
      nameEl.style.cssText = "font-size:11px;color:var(--ink-soft);font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
      nameEl.textContent = entry.op_name.replace("DEPTHWISE_CONV_2D","DW_CONV").replace("FULLY_CONNECTED","FC");
      const badge = mk("div");
      badge.style.cssText = "font-size:9px;color:var(--muted);margin-top:1px";
      badge.textContent = `${entry.oc}ch · ${entry.kh}×${entry.kw}`;
      nameWrap.append(nameEl, badge);

      const barWrap = mk("div");
      barWrap.style.cssText = "width:24px;height:22px;position:relative;flex-shrink:0;display:flex;align-items:flex-end";
      const bar = mk("div");
      bar.style.cssText = `width:100%;height:${Math.round(Math.max(t,0.04)*18+2)}px;background:${col};border-radius:2px 2px 0 0;opacity:0.65`;
      barWrap.append(bar);

      item.append(dot, nameWrap, barWrap);

      item.addEventListener("mouseenter", () => { if (idx !== mvState.layerIdx) item.style.background = "var(--surface-hover)"; });
      item.addEventListener("mouseleave", () => { if (idx !== mvState.layerIdx) item.style.background = "transparent"; });
      item.addEventListener("click", () => { mvState.layerIdx = idx; updateDetail(); rebuildSidebar(); });

      sidebar.append(item);
    });
  }

  function updateDetail() {
    detail.replaceChildren();
    const entry = tomo[mvState.layerIdx];
    if (!entry) return;

    // Info bar
    const infoBar = mk("div");
    infoBar.style.cssText = "padding:7px 10px;background:var(--viz-panel-strong);border-radius:6px;font-size:11px;color:var(--ink-soft);font-family:monospace;line-height:1.7;flex-shrink:0";
    const opIdentity = mk("span");
    opIdentity.style.cssText = "color:var(--ink);font-weight:600";
    opIdentity.textContent = `#${entry.op_index} ${entry.op_name}`;
    const dimensions = document.createTextNode(`  OC=${entry.oc} / IC=${entry.ic} / ${entry.kh}x${entry.kw} / ${entry.dtype} / `);
    const layerType = mk("span");
    layerType.style.color = "var(--warn)";
    layerType.textContent = entry.layer_type;
    infoBar.append(opIdentity, dimensions, layerType);
    if (entry.quant_snr_valid) {
      const separator = document.createTextNode(" / ");
      const snr = mk("span");
      snr.style.color = entry.quant_snr_db < 30 ? "var(--risk)" : entry.quant_snr_db < 40 ? "var(--warn)" : "var(--good)";
      snr.title = "Simulated per-tensor symmetric int8 weight-quantization SNR; weight-distortion proxy, not an accuracy claim";
      snr.textContent = `int8 SNR ${entry.quant_snr_db.toFixed(1)} dB`;
      infoBar.append(separator, snr);
    }
    detail.append(infoBar);

    // Top row: filter + landscape
    const topRow = mk("div");
    topRow.style.cssText = "display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start;flex-shrink:0";

    // Filter panel
    const filterBox = mk("div");
    filterBox.style.cssText = "display:flex;flex-direction:column;gap:4px";
    const filterLbl = mk("div"); filterLbl.style.cssText = "font-size:10px;color:var(--muted)"; filterLbl.textContent = "Filter [kH×kW] · IC-avg";

    // OC selector
    const ocRow = mk("div"); ocRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:3px";
    const ocLabel = mk("span"); ocLabel.style.cssText = "font-size:10px;color:var(--muted)"; ocLabel.textContent = "ch:";
    const ocInput = document.createElement("input");
    ocInput.type = "range"; ocInput.min = 0; ocInput.max = entry.oc - 1; ocInput.value = Math.min(mvState.ocIdx, entry.oc - 1);
    ocInput.style.cssText = "width:80px;accent-color:var(--accent);height:14px";
    const ocVal = mk("span"); ocVal.style.cssText = "font-size:10px;color:var(--ink-soft);font-family:monospace;width:28px";
    ocVal.textContent = ocInput.value;
    ocRow.append(ocLabel, ocInput, ocVal);

    const filterCv = document.createElement("canvas");
    filterCv.width = 140; filterCv.height = 140;
    filterCv.style.cssText = "border:1px solid var(--line);border-radius:4px;image-rendering:pixelated;display:block";

    const drawFilter = () => {
      const oc = +ocInput.value;
      mvState.ocIdx = oc;
      ocVal.textContent = oc;
      drawMvFilter(filterCv, entry, oc);
    };
    ocInput.addEventListener("input", drawFilter);
    drawFilter();

    const filterNote = mk("div"); filterNote.style.cssText = "font-size:9px;color:var(--muted)"; filterNote.textContent = "Blue−  White0  Red+";
    filterBox.append(filterLbl, ocRow, filterCv, filterNote);

    // Landscape panel (lazy)
    const lscBox = mk("div");
    lscBox.style.cssText = "display:flex;flex-direction:column;gap:4px";
    const lscLbl = mk("div"); lscLbl.style.cssText = "font-size:10px;color:var(--muted)"; lscLbl.textContent = "Layer landscape (on-demand)";

    const lscCv = document.createElement("canvas");
    lscCv.width = 140; lscCv.height = 140;
    lscCv.style.cssText = "border:1px solid var(--line);border-radius:4px;image-rendering:pixelated;display:block;cursor:pointer";
    const lscCtx = lscCv.getContext("2d");
    const drawLscPlaceholder = () => {
      lscCtx.fillStyle = activeThemeColor("--viz-canvas", "#0d1420"); lscCtx.fillRect(0, 0, 140, 140);
      lscCtx.fillStyle = activeThemeColor("--viz-meta", "#8a9b93"); lscCtx.font = "10px monospace"; lscCtx.textAlign = "center";
      lscCtx.fillText("Click to compute", 70, 63); lscCtx.fillText("landscape", 70, 78);
    };
    drawLscPlaceholder();

    const computeBtn = mk("button");
    computeBtn.textContent = "Compute landscape";
    computeBtn.style.cssText = "padding:4px 8px;background:var(--surface);color:var(--ink-soft);border:1px solid var(--line-strong);border-radius:4px;font-size:10px;cursor:pointer;width:100%";

    const doCompute = async () => {
      computeBtn.disabled = true; computeBtn.textContent = "Computing…";
      lscCtx.fillStyle = activeThemeColor("--viz-canvas", "#0d1420"); lscCtx.fillRect(0, 0, 140, 140);
      lscCtx.fillStyle = activeThemeColor("--viz-meta", "#8a9b93"); lscCtx.font = "10px monospace"; lscCtx.textAlign = "center";
      lscCtx.fillText("Running…", 70, 74);
      try {
        const G = 13;
        const raw = layer_landscape_grid(ctx.currentModelBytes, entry.op_index, 1234, 5678, G, 0.4);
        const { grid: flat } = raw;
        const ci = Math.floor(G / 2);
        const center = flat[ci * G + ci] ?? 0;
        const dm = flat.map(v => isNaN(v) ? 0 : v - center);
        const vMax = Math.max(...dm.map(Math.abs), 1e-12);
        const CW = 140;
        const cX = i => Math.floor((i / G) * CW); // float-based, no gap
        lscCtx.clearRect(0, 0, CW, CW);
        for (let bi = 0; bi < G; bi++) {
          const y0 = cX(G - 1 - bi), y1 = cX(G - bi); // y-flip: negative β at bottom
          for (let ai = 0; ai < G; ai++) {
            const x0 = cX(ai), x1 = cX(ai + 1);
            const v = dm[bi * G + ai] ?? 0;
            const t = Math.max(-1, Math.min(1, v / vMax));
            lscCtx.fillStyle = t >= 0
              ? `rgb(${Math.round(255*t)},${Math.round(40*(1-t))},${Math.round(40*(1-t))})`
              : `rgb(${Math.round(40*(1+t))},${Math.round(40*(1+t))},${Math.round(-255*t)})`;
            lscCtx.fillRect(x0, y0, x1 - x0, y1 - y0);
          }
        }
        // crosshair
        lscCtx.strokeStyle = "rgba(245,158,11,0.85)"; lscCtx.lineWidth = 1.5;
        const cx = (cX(ci) + cX(ci + 1)) / 2, cy = (cX(G - 1 - ci) + cX(G - ci)) / 2;
        lscCtx.beginPath(); lscCtx.moveTo(cx-8,cy); lscCtx.lineTo(cx+8,cy); lscCtx.stroke();
        lscCtx.beginPath(); lscCtx.moveTo(cx,cy-8); lscCtx.lineTo(cx,cy+8); lscCtx.stroke();
        computeBtn.textContent = "Recompute";
      } catch(e) {
        const reason = e?.message || String(e) || "unknown error";
        lscCtx.fillStyle = activeThemeColor("--viz-canvas", "#0d1420"); lscCtx.fillRect(0, 0, 140, 140);
        lscCtx.fillStyle = activeThemeColor("--risk", "#e18488"); lscCtx.font = "9px monospace"; lscCtx.textAlign = "center";
        const words = reason.length > 60 ? reason.slice(0, 57) + "…" : reason;
        lscCtx.fillText(words, 70, 63);
        lscCtx.fillStyle = activeThemeColor("--viz-meta", "#8a9b93"); lscCtx.fillText("click to retry", 70, 80);
        computeBtn.textContent = "Retry";
        console.warn("layer_landscape_grid error:", e);
      } finally { computeBtn.disabled = false; }
    };
    lscCv.addEventListener("click", doCompute);
    computeBtn.addEventListener("click", doCompute);
    lscBox.append(lscLbl, lscCv, computeBtn);

    topRow.append(filterBox, lscBox);
    detail.append(topRow);

    // L2 per OC bar chart
    const barBox = mk("div");
    barBox.style.cssText = "display:flex;flex-direction:column;gap:4px;flex-shrink:0";
    const barLbl = mk("div"); barLbl.style.cssText = "font-size:10px;color:var(--muted)";
    barLbl.textContent = `L2 per output channel — ${entry.oc} channels`;
    const barScroll = mk("div");
    barScroll.style.cssText = "max-height:108px;overflow-y:auto;scrollbar-width:thin;background:var(--viz-canvas);border:1px solid var(--line);border-radius:4px;padding:4px 6px";
    const l2vals = entry.l2_per_oc ?? [];
    const l2max = Math.max(...l2vals, 1e-10);
    l2vals.forEach((v, i) => {
      const row = mk("div");
      row.style.cssText = "display:flex;align-items:center;gap:4px;margin-bottom:1px";
      const lbl = mk("span"); lbl.style.cssText = "width:26px;font-size:9px;color:var(--muted);text-align:right;flex-shrink:0;font-family:monospace";
      lbl.textContent = i;
      const bg = mk("div"); bg.style.cssText = "flex:1;height:5px;background:var(--viz-node-muted);border-radius:2px;overflow:hidden";
      const fill = mk("div");
      const t2 = v / l2max;
      fill.style.cssText = `height:100%;width:${(t2*100).toFixed(1)}%;background:${metricColor(t2)};border-radius:2px`;
      bg.append(fill);
      const vl = mk("span"); vl.style.cssText = "width:46px;font-size:9px;color:var(--muted);font-family:monospace";
      vl.textContent = v.toFixed(4);
      row.append(lbl, bg, vl);
      barScroll.append(row);
    });
    barBox.append(barLbl, barScroll);
    detail.append(barBox);

    // Depth profile
    const dpBox = mk("div");
    dpBox.style.cssText = "display:flex;flex-direction:column;gap:4px;flex-shrink:0";
    const dpLbl = mk("div"); dpLbl.style.cssText = "font-size:10px;color:var(--muted)";
    dpLbl.textContent = `Depth profile — ${mvState.metric} · gold = selected layer`;
    const dpCv = document.createElement("canvas");
    dpCv.width = 500; dpCv.height = 90;
    dpCv.style.cssText = "width:100%;border:1px solid var(--line);border-radius:4px;display:block";
    dpBox.append(dpLbl, dpCv);
    detail.append(dpBox);
    drawMvDepth(dpCv, tomo, mvState);

    // Scroll selected sidebar item into view
    sidebar.querySelector(`[data-idx="${mvState.layerIdx}"]`)?.scrollIntoView({ block: "nearest" });
  }

  metricSel.addEventListener("change", () => {
    mvState.metric = metricSel.value;
    rebuildSidebar();
    updateDetail();
  });

  rebuildSidebar();
  updateDetail();
}

async function runModelViewer() {
  if (!ctx.currentModelBytes || !modelSupportsCapability(ctx.current?.format, "model_tomography")) return;
  if (!modelViewerPanel) return;
  try {
    const tomo = compute_model_tomography(ctx.currentModelBytes, ctx.currentFilename || "model.tflite", selectedTargetId());
    if (!Array.isArray(tomo) || !tomo.length) {
      modelViewerPanel.hidden = false;
      modelViewerPanel.textContent = "No weight layers found — Model Explorer requires at least one CONV/DW_CONV op with a constant weight tensor.";
      modelViewerPanel.style.cssText = "padding:12px 16px;font-size:12px;color:var(--muted);background:var(--viz-canvas);border-radius:6px;border:1px solid var(--line);margin:16px 0";
      modelViewerPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
      return;
    }
    modelViewerPanel.style.cssText = "margin:16px 0";
    modelTomography = tomo;
    mvState.layerIdx = 0; mvState.ocIdx = 0;
    renderModelViewer(tomo);
    modelViewerPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (e) {
    console.warn("Model viewer failed:", e);
    modelViewerPanel.hidden = false;
    modelViewerPanel.textContent = `Artifact Viewer failed: ${e?.message || String(e)}`;
    modelViewerPanel.style.cssText = "padding:12px 16px;font-size:12px;color:var(--risk);background:var(--viz-canvas);border-radius:6px;border:1px solid var(--risk);margin:16px 0";
  }
}


  return {
    runDeepBomAnalysis,
    resetDeepBomPanel,
    renderDeepBomSkeleton,
    renderDriftSkeleton,
    renderDeploymentSensitivitySkeleton,
    deepBomSignalValue,
    renderDeepBomResult,
    renderProtocolGroups,
    activeThemeColor,
    renderModelViewer,
    runModelViewer,
    hasModelTomography: () => Boolean(modelTomography?.length),
  };
}
