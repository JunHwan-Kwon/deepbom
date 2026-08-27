import {
  buildResourceMapPresentation,
  renderEvidenceTreemap,
} from "./evidence-treemap.js";

export function createGraphWorkspace(workspace) {
  const {
    TEXT_EXPORT_ARTIFACTS,
    appendBenchmarkRow,
    artifactOverviewHeader,
    artifactOverviewPanels,
    assessedOpLogicalBytes,
    backendCandidates,
    backendSelect,
    benchmarkBody,
    benchmarkErrorStatus,
    benchmarkWrap,
    blocksExplorerPanel,
    buildGraphEvidenceMaps,
    buildGraphIndex,
    buildOnnxRuntimeShapeBinding,
    buildRepresentableKernelChannelCheck,
    buildTensorInventory,
    cacheExplorerPanel,
    canvasToPngBytes,
    clampInt,
    classifyTensorRoles,
    tensorQuantizationMode,
    clearRuntimeAssignment,
    collectFullGraph,
    collectNeighborhood,
    compute_input_influence,
    compute_output_influence,
    compute_static_runtime_calibration,
    compute_weight_histogram,
    downloadTextArtifact,
    ensureLiteRtRuntime,
    explorerDecisionController,
    explorerExecutionPlacementPanel,
    explorerRedesignController,
    explorerTabs,
    findingsBody,
    formatBytes,
    formatNumber,
    formatPercent,
    formatUs,
    graphDepth,
    graphDetailLayout,
    graphExplorer,
    resourceMapPanel,
    graphMapStatus,
    graphMapSvg,
    graphModeHint,
    graphOpBody,
    graphOpHead,
    graphOpRow,
    graphOpsView,
    graphScenarioBanner,
    graphScenarioDetail,
    graphScenarioLabel,
    graphSearch,
    graphStats,
    graphSvgText,
    histogramBody,
    histogramRow,
    kernelBoundaryInventory,
    kernelInspectorBody,
    kernelInspectorPanel,
    kernelInspectorSearch,
    kernelInspectorSummary,
    layeredViewPanel,
    layoutFoldedGraph,
    layoutNeighborhood,
    loadOnnxBenchmark,
    loadProtectedSourceAnalysis,
    loadTfliteBenchmark,
    mk,
    modelFormatAdapter,
    modelSupportsCapability,
    nodeViewController,
    nodeViewPanel,
    opDetail,
    opFilterCount,
    opLogicalL1Ratio,
    opLogicalRowPayloadBytes,
    opMatchesSearch,
    opNavLabel,
    opNavNext,
    opNavPrev,
    opParetoBar,
    opSteadyStateUs,
    opTimeline,
    p99EvidenceForSampleCount,
    padOp,
    performanceVisualController,
    preprocessingConsequenceController,
    quantEvidenceController,
    quantEvidencePanel,
    renderExecutionPlacementView,
    renderFindingsCalibration,
    renderGraphMapContent,
    renderInsightDashboardView,
    renderKernelInspector,
    renderOpDetailPanel,
    renderRuntimeEvidenceClosure,
    renderTensorArenaViewer,
    rooflineBody,
    rooflineTableRows,
    runInference,
    runsInput,
    runtimeAssignmentComparison,
    runtimeAssignmentStatus,
    runtimeEvidenceClosure,
    runtimeNotes,
    runtimeReadinessSignals,
    runtimeStatus,
    selectWasmCalibrationResult,
    selectedTargetId,
    selectedTargetProfile,
    setActiveAuditTab,
    setActiveWorkspace,
    shortError,
    stageCard,
    stageCount,
    stageStrip,
    submitBenchmarkTelemetry,
    summary,
    summaryMetricCards,
    syncTabSelection,
    tensorBody,
    tensorExplorerPanel,
    tensorMemoryTimeline,
    tensorStatsBar,
    textExportOptions,
    topMacBody,
    topMacRows,
    updateBenchmarkRow,
    updateWorkflowState,
    visualPngSpecs,
    warmupInput,
    xnnSegmentBar,
    zipBinaryFile,
    zipTextFile,
  } = workspace;

const resourceMapOptions = { metric: "macs", groupBy: "stage", colorBy: "intensity" };

function renderSummary(analysis) {
  const metrics = document.createElement("div");
  metrics.className = "artifact-metric-grid";
  metrics.append(...summaryMetricCards(analysis));

  const evidence = document.createElement("div");
  evidence.className = "artifact-evidence-grid";
  evidence.append(...artifactOverviewPanels(analysis, {
    onOpenScaleVector: (tensorIndex) => {
      quantEvidenceController.selectTensor(tensorIndex);
      setActiveWorkspace("graph", { force: true });
      switchExplorerTab("quant");
    },
  }));

  summary.replaceChildren(
    artifactOverviewHeader(analysis),
    metrics,
    evidence,
  );
}

function adaptInsightsForUI(analysis) {
  const ins = analysis.insights || {};
  const target = analysis.target_profile || {};
  const ops = Array.isArray(analysis.ops) ? analysis.ops : [];
  const l1Bytes = Number(target.l1_data_bytes || 32768);
  const l2Bytes = Number(target.l2_bytes || 0);

  // Simple lookups: find specific op objects for UI display (not scoring — scoring is in WASM)
  const topMisaligned = ops.filter(op => op.channel_alignment_status === "misaligned")
    .reduce((best, op) => (!best || (op.channel_tail_overhead_percent || 0) > (best.channel_tail_overhead_percent || 0)) ? op : best, null);
  const topPackingWarn = ops.filter(op => op.weight_packing_risk === "warn")
    .reduce((best, op) => (!best || (op.weight_packing_overhead_us || 0) > (best.weight_packing_overhead_us || 0)) ? op : best, null);
  const l1AssessedOps = ops.filter((op) => opLogicalRowPayloadBytes(op) != null);
  const l1Watch = l1AssessedOps.filter((op) => Number(opLogicalL1Ratio(op, l1Bytes) || 0) >= 0.9);
  const l2WatchAssessed = String(target.l2_capacity_scope || "").startsWith("private_per_core");
  const l2Watch = l2WatchAssessed && l2Bytes > 0
    ? l1AssessedOps.filter((op) => Number(opLogicalRowPayloadBytes(op) || 0) / l2Bytes >= 0.9)
    : [];
  const maxRowWorkingSet = l1AssessedOps.length ? Math.max(...l1AssessedOps.map((op) => Number(opLogicalRowPayloadBytes(op) || 0))) : null;
  const derivedMaxL1Ratio = l1AssessedOps.length ? Math.max(...l1AssessedOps.map((op) => Number(opLogicalL1Ratio(op, l1Bytes) || 0))) : null;
  const maxL2Ratio = l2WatchAssessed && maxRowWorkingSet != null && l2Bytes > 0
    ? maxRowWorkingSet / l2Bytes
    : null;
  const representableChannels = buildRepresentableKernelChannelCheck(analysis);
  const flaggedChannels = Number(representableChannels.flagged_channels || 0);
  const exactZeroKernelSlices = Number(analysis?.weight_integrity?.exact_zero_kernel_slice_count
    ?? ins.exact_zero_kernel_slices
    ?? 0);
  const headlineLabel = exactZeroKernelSlices > 0
    ? `${formatNumber(exactZeroKernelSlices)} exact-zero stored kernel channel(s) require review`
    : flaggedChannels > 0
    ? `${formatNumber(flaggedChannels)} near-zero kernel channel(s) require review`
    : ins.label || "";
  const headlineRationale = exactZeroKernelSlices > 0
    ? `${formatNumber(exactZeroKernelSlices)} kernel output slice(s) are exact-zero in stored centered-code space. Review the emitted bias, fixed-point channel proof, downstream path, and representative outputs before concluding functional inactivity. ${ins.rationale || ""}`
    : flaggedChannels > 0
    ? `${formatNumber(flaggedChannels)} per-axis kernel channel(s) meet the disclosed representable-range and relative-scale review thresholds. This is a quantization design signal, not proof of functional inactivity. ${ins.rationale || ""}`
    : ins.rationale || "";

  return {
    // From analysis.insights computed by the Rust/WASM core.
    score: ins.score ?? 0,
    scoreEvidenceClass: ins.score_evidence_class || "HEURISTIC",
    scoreMethod: ins.score_method || "",
    scoreBreakdown: ins.score_breakdown || null,
    tone: exactZeroKernelSlices > 0 || flaggedChannels > 0 ? "warn" : ins.tone || "neutral",
    label: headlineLabel,
    rationale: headlineRationale,
    boundCounts: { "compute-bound": ins.bound_compute || 0, mixed: ins.bound_mixed || 0, "memory-bound": ins.bound_memory || 0 },
    totalOps: ops.length || 1,
    memoryRatio: ins.memory_ratio || 0,
    maxL1Ratio: ins.max_l1_ratio ?? derivedMaxL1Ratio,
    chainBreaks: ins.chain_breaks || 0,
    effectiveChainBreaks: ins.effective_chain_breaks || 0,
    chainCount: (analysis.xnnpack_chains || []).length,
    longestChain: analysis.xnnpack_longest_chain || 0,
    fallbackByteRatio: ins.fallback_byte_ratio || 0,
    delegatedMacRatio: ins.delegated_mac_ratio || 0,
    misalignedOps: ins.misaligned_ops || 0,
    packingWarnOps: ins.packing_warn_ops || 0,
    suspectTotal: ins.suspect_total || 0,
    suspectSummary: ins.suspect_summary || "",
    signals: Array.isArray(ins.signals) ? ins.signals : [],
    // From analysis directly (target profile + traffic data)
    targetL1Bytes: l1Bytes,
    targetL2Bytes: l2Bytes,
    topFallbackTraffic: (analysis.fallback_traffic_by_op_family || [])[0] || null,
    // Presentation lookups: find specific op objects for display (simple filter, not scoring)
    l1Watch, l2Watch, l2WatchAssessed, maxL2Ratio, l1AssessedCount: l1AssessedOps.length, maxRowWorkingSet,
    topMisaligned, topPackingWarn,
  };
}

function renderInsightDashboard(analysis) {
  const insights = adaptInsightsForUI(analysis);
  renderInsightDashboardView({
    analysis,
    insights,
    runtimeEvidence: workspace.runtimeAssignmentEvidence,
    setActiveWorkspace,
    setActiveAuditTab,
  });
}

async function buildVisualPngFiles() {
  if (!workspace.current) return [];
  const specs = visualPngSpecs({
    analysis: workspace.current,
    filename: workspace.current.filename || workspace.currentFilename,
    targetProfile: workspace.current.target_profile || selectedTargetProfile(),
    targetComparisonRows: performanceVisualController.buildTargetComparisonRows(workspace.current),
    preprocessingConsequenceResult: workspace.preprocessingConsequenceResult,
    preprocessingConsequenceCapture: preprocessingConsequenceController.getCapture(),
  });
  const files = [];
  for (const [name, renderer] of specs) {
    try {
      const canvas = renderer();
      const png = await canvasToPngBytes(canvas);
      files.push(zipBinaryFile(name, png));
    } catch (error) {
      files.push(zipTextFile(name.replace(/\.png$/i, ".error.txt"), shortError(error)));
    }
  }
  return files;
}

function jumpToGraphOp(opIndex) {
  if (!workspace.current) return;
  switchExplorerTab("ops");
  setActiveWorkspace("graph", { force: true });
  requestAnimationFrame(() => selectGraphOp(workspace.current, Number(opIndex), { scrollTable: true }));
}

function graphScenarioMatchesAnalysis(scenario, analysis) {
  return Boolean(scenario && analysis
    && scenario.artifactSha256 === analysis.model_sha256
    && scenario.targetProfileSha256 === analysis.target_profile?.profile_sha256);
}

function renderGraphScenarioState() {
  if (!graphScenarioBanner) return;
  const active = graphScenarioMatchesAnalysis(workspace.activeGraphScenario, workspace.current);
  graphScenarioBanner.hidden = !active;
  if (!active) {
    graphScenarioLabel.textContent = "No scenario";
    graphScenarioDetail.textContent = "Viewer-only counterfactual; the bound analysis and report remain unchanged.";
    return;
  }
  graphScenarioLabel.textContent = workspace.activeGraphScenario.label;
  graphScenarioDetail.textContent = `${formatNumber(workspace.activeGraphScenario.boundaryEdgeReductionCount)} predicted boundary edge(s) and ${formatBytes(workspace.activeGraphScenario.boundaryPayloadReductionBytes)} payload removed. Latency is unchanged; runtime support remains unobserved.`;
}

function previewDelegationScenario(scenario) {
  if (!graphScenarioMatchesAnalysis(scenario, workspace.current)) return;
  workspace.activeGraphScenario = structuredClone(scenario);
  explorerDecisionController.setScenario(workspace.activeGraphScenario);
  renderGraphScenarioState();
  setActiveWorkspace("graph", { force: true });
  switchExplorerTab("ops");
  switchGraphMode("deploy");
  requestAnimationFrame(() => {
    if (!workspace.current) return;
    selectGraphOp(workspace.current, Number(workspace.activeGraphScenario.firstOpIndex), { scrollTable: false });
    requestAnimationFrame(() => {
      const scenarioNode = graphMapSvg?.querySelector(".graph-node.scenario-delegated .graph-node-card");
      (scenarioNode || graphScenarioBanner)?.scrollIntoView({ block: "center", inline: "center", behavior: "auto" });
    });
  });
}

function clearGraphScenarioPreview() {
  workspace.activeGraphScenario = null;
  explorerDecisionController.setScenario(null);
  renderGraphScenarioState();
  if (workspace.current) deferGraphMap(workspace.current, workspace.selectedOpIndex);
}

function renderInferencePanel(analysis) {
  benchmarkWrap.hidden = true;
  benchmarkBody.replaceChildren();
  runtimeStatus.textContent = "Ready";
  const executable = modelSupportsCapability(analysis?.format, "runtime_execution") && Boolean(workspace.currentModelPayloadLoaded);
  runInference.disabled = !executable;
  runInference.title = executable ? "Run a local browser runtime benchmark." : `${modelFormatAdapter(analysis?.format).label} static artifact analysis is available, but no local browser runtime benchmark is implemented.`;
  const warmup = clampInt(warmupInput.value, 0, 100, 1);
  const runs = clampInt(runsInput.value, 1, 500, 50);
  runtimeNotes.replaceChildren(...runtimeReadinessSignals(analysis, {
    backendValue: backendSelect.value,
    navigatorLike: navigator,
    warmup,
    runs,
  }));
}

function renderGraphExplorer(analysis) {
  workspace.selectedOpIndex = analysis.ops.length ? 0 : null;
  graphStats.textContent = `${analysis.ops.length} ops / ${analysis.tensors.length} tensors`;
  updateGraphModeHint(workspace.currentGraphMode);

  const graphEvidence = buildGraphEvidenceMaps(analysis);
  workspace.currentLowNormStatMap = graphEvidence.lowNormStats;
  workspace.currentTopologyAnnotations = graphEvidence;

  renderOpTimeline(analysis);
  renderXnnSegmentBar(analysis);
  renderOpParetoBar(analysis);
  renderResourceMap(analysis);
  renderGraphOpRows(analysis);
  renderOpDetail(analysis, workspace.selectedOpIndex);
  deferGraphMap(analysis, workspace.selectedOpIndex);
  renderTensorExplorer(analysis);
  quantEvidenceController.setAnalysis(analysis);
  nodeViewController.setAnalysis(analysis);
  renderCurrentKernelInspector(true);
  layeredViewStale = true;
  if (layeredViewPanel && !layeredViewPanel.hidden) {
    renderLayeredView(analysis);
  } else {
    layeredViewPanel?.replaceChildren(); // free any previous model's canvas now
  }
}

function renderResourceMap(analysis) {
  if (!resourceMapPanel) return;
  const presentation = buildResourceMapPresentation(analysis, resourceMapOptions);
  Object.assign(resourceMapOptions, {
    metric: presentation.metricId,
    groupBy: presentation.groupBy,
    colorBy: presentation.colorBy,
  });
  renderEvidenceTreemap(resourceMapPanel, presentation, {
    onControl: (name, value) => {
      resourceMapOptions[name] = value;
      renderResourceMap(analysis);
    },
    onSelect: (item) => {
      if (item.kind !== "op") return;
      switchExplorerTab("ops");
      selectGraphOp(analysis, item.index, { scrollTable: true });
    },
  });
}


const OP_TIMELINE_TONES = {
  compute: "#3b82f6", memory: "#f59e0b", packing: "#8b5cf6",
  break: "#dc2626", fallback: "#64748b",
};
const PROFILE_LANE_DEFS = [
  ["traffic", "Traffic", "Estimated bytes moved per op"],
  ["cache", "L1 pressure", "Row working set vs L1 size (line = 1×L1)"],
  ["quant", "Quant", "Quantization state strip: green ok · amber warn · red risk · violet FP32 island"],
  ["intensity", "Intensity", "Arithmetic intensity (ops/byte, log scale), colored by bound type"],
  ["lowNorm", "Low norm", "L2 below 2% of the layer maximum; heuristic only"],
];
const profileLaneState = { traffic: true, cache: true, quant: true, intensity: false, lowNorm: false };

function renderOpTimeline(analysis) {
  if (!opTimeline) return;
  opTimeline.replaceChildren();
  const ops = analysis?.ops || [];
  const N = ops.length;
  if (!N) return;

  const us = new Float64Array(N);
  const cum = new Float64Array(N);
  let total = 0, peakUs = 0, peakIdx = 0;
  for (const op of ops) {
    const v = opSteadyStateUs(op);
    us[op.index] = v;
    total += v;
    if (v > peakUs) { peakUs = v; peakIdx = op.index; }
  }
  if (total <= 0) return;
  for (let i = 0, acc = 0; i < N; i++) { acc += us[i]; cum[i] = acc; }

  const label = mk("div", "insight-strip-label");
  const peakOp = ops.find((o) => o.index === peakIdx);
  label.textContent = `Modeled steady-state profile — ${formatUs(total)} total · peak #${peakIdx} · not device-calibrated`
    + `${peakOp ? ` ${peakOp.name}` : ""} ${formatUs(peakUs)} (one-time packing excluded; bar color = dominant cost)`;

  const W = Math.max(480, Math.round(opTimeline.clientWidth || graphExplorer?.clientWidth || 1200));
  const dpr = window.devicePixelRatio || 1;
  const PL = 6;
  const toX = (i) => PL + (i / (N - 1 || 1)) * (W - PL * 2);
  const barW = Math.max(1, (W - PL * 2) / N);
  const lanes = []; // { canvas, ctx, H, baseFrame }

  function makeLane(H, cls = "memory-timeline-canvas profile-lane") {
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    canvas.className = cls;
    canvas.style.height = `${H}px`;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    return { canvas, ctx, H };
  }
  function laneLabel(ctx, text) {
    ctx.fillStyle = "rgba(148,163,184,0.75)";
    ctx.font = "9px system-ui";
    ctx.fillText(text, PL + 2, 9);
  }

  // Main lane: est time bars
  const main = makeLane(84, "memory-timeline-canvas");
  {
    const { ctx, H } = main;
    const PB = 14;
    const toY = (v) => (H - PB) - (v / peakUs) * (H - PB - 4);
    for (const op of ops) {
      const v = us[op.index];
      if (v <= 0) continue;
      ctx.fillStyle = OP_TIMELINE_TONES[op.bottleneck_dominant] || "#3b82f6";
      ctx.fillRect(toX(op.index) - barW / 2, toY(v), barW, (H - PB) - toY(v));
    }
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px system-ui";
    ctx.fillText("op 0", PL, H - 3);
    ctx.textAlign = "right";
    ctx.fillText(`op ${N - 1}`, W - PL, H - 3);
    ctx.textAlign = "left";
  }
  lanes.push(main);

  // Optional thin lanes
  const opIntensity = (op) => op.intensity_status === "not_assessed" || op.intensity_ops_per_byte == null
    ? null : Number(op.intensity_ops_per_byte);
  const opL1Ratio = (op) => opLogicalL1Ratio(op, analysis?.target_profile?.l1_data_bytes);
  const maxBytes = ops.reduce((m, op) => Math.max(m, assessedOpLogicalBytes(op) ?? 0), 0);
  const maxIntensity = ops.reduce((m, op) => Math.max(m, opIntensity(op) ?? 0), 0);
  const maxLowNormPct = ops.reduce((m, op) => {
    const d = workspace.currentLowNormStatMap.get(op.index);
    return d?.total ? Math.max(m, d.low_norm / d.total) : m;
  }, 0);

  if (profileLaneState.traffic && maxBytes > 0) {
    const lane = makeLane(26);
    for (const op of ops) {
      const b = assessedOpLogicalBytes(op) ?? 0;
      if (b <= 0) continue;
      lane.ctx.fillStyle = "#0ea5e9";
      const h = Math.max(1, (b / maxBytes) * (lane.H - 11));
      lane.ctx.fillRect(toX(op.index) - barW / 2, lane.H - h, barW, h);
    }
      laneLabel(lane.ctx, "assessed traffic (bytes)");
    lanes.push(lane);
  }

  if (profileLaneState.cache) {
    const lane = makeLane(26);
    const CAP = 4; // display cap: 4×L1
    let any = false;
    for (const op of ops) {
      const r = opL1Ratio(op) ?? 0;
      if (r <= 0) continue;
      any = true;
      lane.ctx.fillStyle = r >= 2 ? "#dc2626" : r >= 1 ? "#f59e0b" : "#475569";
      const h = Math.max(1, (Math.min(r, CAP) / CAP) * (lane.H - 11));
      lane.ctx.fillRect(toX(op.index) - barW / 2, lane.H - h, barW, h);
    }
    if (any) {
      const y1 = lane.H - (1 / CAP) * (lane.H - 11); // 1×L1 baseline
      lane.ctx.strokeStyle = "rgba(226,232,240,0.35)";
      lane.ctx.setLineDash([3, 3]);
      lane.ctx.beginPath(); lane.ctx.moveTo(PL, y1); lane.ctx.lineTo(W - PL, y1); lane.ctx.stroke();
      lane.ctx.setLineDash([]);
      laneLabel(lane.ctx, "L1 pressure (line = 1×L1)");
      lanes.push(lane);
    }
  }

  if (profileLaneState.quant) {
    const lane = makeLane(16);
    for (const op of ops) {
      lane.ctx.fillStyle = op.quant_hole ? "#8b5cf6"
        : op.quant_risk === "risk" ? "#dc2626"
        : op.quant_risk === "warn" ? "#f59e0b"
        : op.quantized_compute_path ? "#15803d" : "#1e293b";
      lane.ctx.fillRect(toX(op.index) - barW / 2, 11, barW, lane.H - 12);
    }
    laneLabel(lane.ctx, "quant state");
    lanes.push(lane);
  }

  if (profileLaneState.intensity && maxIntensity > 0) {
    const lane = makeLane(26);
    const logMax = Math.log10(1 + maxIntensity);
    const BOUND_TONES = { "compute-bound": "#3b82f6", "memory-bound": "#f59e0b", mixed: "#94a3b8" };
    for (const op of ops) {
      const v = opIntensity(op) ?? 0;
      if (v <= 0) continue;
      lane.ctx.fillStyle = BOUND_TONES[op.static_bound_guess] || "#64748b";
      const h = Math.max(1, (Math.log10(1 + v) / logMax) * (lane.H - 11));
      lane.ctx.fillRect(toX(op.index) - barW / 2, lane.H - h, barW, h);
    }
    laneLabel(lane.ctx, "intensity (ops/byte, log)");
    lanes.push(lane);
  }

  if (profileLaneState.lowNorm && maxLowNormPct > 0) {
    const lane = makeLane(26);
    for (const op of ops) {
      const d = workspace.currentLowNormStatMap.get(op.index);
      if (!d?.total || d.low_norm <= 0) continue;
      lane.ctx.fillStyle = "#f472b6";
      const h = Math.max(1, ((d.low_norm / d.total) / maxLowNormPct) * (lane.H - 11));
      lane.ctx.fillRect(toX(op.index) - barW / 2, lane.H - h, barW, h);
    }
    laneLabel(lane.ctx, "low-norm filters");
    lanes.push(lane);
  }

  // Shared hover / click across every lane
  for (const lane of lanes) lane.baseFrame = lane.ctx.getImageData(0, 0, W, lane.H);
  const laneStack = mk("div", "profile-lane-stack");
  for (const lane of lanes) laneStack.append(lane.canvas);

  const tip = mk("div", "layered-tooltip");
  tip.hidden = true;
  laneStack.title = "Hover to inspect per-op signals · click to select the op";

  const stepFromEvent = (e) => {
    const r = laneStack.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    return Math.max(0, Math.min(N - 1, Math.round(((x - PL) / (W - PL * 2)) * (N - 1))));
  };
  const restoreAll = () => {
    for (const lane of lanes) {
      lane.ctx.save();
      lane.ctx.setTransform(1, 0, 0, 1, 0, 0);
      lane.ctx.putImageData(lane.baseFrame, 0, 0);
      lane.ctx.restore();
    }
  };

  laneStack.addEventListener("mousemove", (e) => {
    const i = stepFromEvent(e);
    restoreAll();
    for (const lane of lanes) {
      lane.ctx.beginPath();
      lane.ctx.moveTo(toX(i), 0);
      lane.ctx.lineTo(toX(i), lane.H);
      lane.ctx.strokeStyle = "#e2e8f0";
      lane.ctx.lineWidth = 1;
      lane.ctx.stroke();
    }
    const op = ops.find((o) => o.index === i);
    const parts = [`#${i}${op ? ` ${op.name}` : ""}`, `${formatUs(us[i])}${op?.bottleneck_dominant ? ` (${op.bottleneck_dominant})` : ""}`];
    const traffic = assessedOpLogicalBytes(op);
    const l1Ratio = opL1Ratio(op);
    const intensity = opIntensity(op);
    if (profileLaneState.traffic) parts.push(traffic == null ? "traffic N/A" : `${formatBytes(traffic)} traffic`);
    if (profileLaneState.cache) parts.push(l1Ratio == null ? "WS N/A" : `WS ${l1Ratio.toFixed(1)}×L1`);
    if (profileLaneState.quant) parts.push(op?.quant_hole ? "FP32 island" : `quant ${op?.quant_risk || "none"}`);
    if (profileLaneState.intensity) parts.push(intensity == null ? "intensity N/A" : `${intensity.toFixed(1)} ops/B`);
    const lowNorm = profileLaneState.lowNorm ? workspace.currentLowNormStatMap.get(i) : null;
    if (lowNorm?.total) parts.push(`low norm ${lowNorm.low_norm}/${lowNorm.total}`);
    parts.push(`cum ${formatPercent(cum[i] / total)}`);
    tip.textContent = parts.join(" · ");
    const wrapR = opTimeline.getBoundingClientRect();
    const sr = laneStack.getBoundingClientRect();
    tip.style.left = `${Math.min(e.clientX - wrapR.left + 12, wrapR.width - 260)}px`;
    tip.style.top = `${sr.top - wrapR.top + 4}px`;
    tip.hidden = false;
  });
  laneStack.addEventListener("mouseleave", () => { restoreAll(); tip.hidden = true; });
  laneStack.addEventListener("click", (e) => {
    if (!workspace.current) return;
    selectGraphOp(workspace.current, stepFromEvent(e), { scrollTable: true });
  });

  // Lane toggles
  const toggles = mk("div", "profile-lane-toggles");
  for (const [key, name, tipText] of PROFILE_LANE_DEFS) {
    const btn = mk("button", `filter-chip${profileLaneState[key] ? " active" : ""}`);
    btn.type = "button";
    btn.textContent = name;
    btn.title = tipText;
    btn.addEventListener("click", () => {
      profileLaneState[key] = !profileLaneState[key];
      if (workspace.current) renderOpTimeline(workspace.current);
    });
    toggles.append(btn);
  }

  opTimeline.append(label, laneStack, tip, toggles);
}


function renderXnnSegmentBar(analysis) {
  if (!xnnSegmentBar) return;
  xnnSegmentBar.replaceChildren();
  xnnSegmentBar.hidden = String(analysis?.format || "tflite").toLowerCase() === "onnx";
  if (xnnSegmentBar.hidden) return;
  const ops = analysis?.ops || [];
  if (!ops.length) return;

  const totalMacs = ops.reduce((s, op) => s + (op.macs || 0), 0);
  const segments = [];
  for (const op of ops) {
    const kind = op.xnnpack_chain_break ? "break" : (op.xnnpack_chain_id >= 0 ? `chain-${op.xnnpack_chain_id}` : "fallback");
    const last = segments[segments.length - 1];
    if (last && last.kind === kind && kind !== "break") {
      last.end = op.index;
      last.macs += op.macs || 0;
      last.count += 1;
    } else {
      segments.push({ kind, start: op.index, end: op.index, macs: op.macs || 0, count: 1 });
    }
  }

  const delegatedSegs = segments.filter((s) => s.kind.startsWith("chain-")).length;
  const breaks = segments.filter((s) => s.kind === "break").length;

  const label = mk("div", "insight-strip-label");
  label.textContent = `XNNPACK delegation map — ${delegatedSegs} delegated segment${delegatedSegs === 1 ? "" : "s"}, `
    + `${breaks} break${breaks === 1 ? "" : "s"}, delegated MACs ${formatPercent(analysis.delegated_mac_percent || 0)}`;

  const bar = mk("div", "segment-bar");
  let chainTone = 0;
  for (const seg of segments) {
    const block = mk("button", "segment-block");
    block.type = "button";
    const share = totalMacs > 0 ? seg.macs / totalMacs : seg.count / ops.length;
    block.style.flexGrow = String(Math.max(share, 0.006) * 1000);
    if (seg.kind === "break") {
      block.classList.add("segment-break");
      const op = ops.find((o) => o.index === seg.start);
      block.title = `#${seg.start} ${op?.name || ""} — chain break (${op?.xnnpack_break_class || "delegation interrupted"})`;
    } else if (seg.kind === "fallback") {
      block.classList.add("segment-fallback");
      block.title = `#${seg.start}–#${seg.end} · ${seg.count} op${seg.count === 1 ? "" : "s"} · CPU fallback · MACs ${(share * 100).toFixed(1)}%`;
    } else {
      block.classList.add(chainTone % 2 === 0 ? "segment-delegated" : "segment-delegated-alt");
      chainTone += 1;
      block.title = `#${seg.start}–#${seg.end} · ${seg.count} op${seg.count === 1 ? "" : "s"} · XNNPACK ${seg.kind} · MACs ${(share * 100).toFixed(1)}%`;
    }
    block.addEventListener("click", () => selectGraphOp(analysis, seg.start, { scrollTable: true }));
    bar.append(block);
  }
  xnnSegmentBar.append(label, bar);
}

function renderOpParetoBar(analysis) {
  if (!opParetoBar) return;
  opParetoBar.replaceChildren();
  const ops = (analysis?.ops || []).filter((op) => opSteadyStateUs(op) > 0);
  const totalUs = ops.reduce((sum, op) => sum + opSteadyStateUs(op), 0);
  if (!ops.length || totalUs <= 0) return;

  const top = [...ops].sort((a, b) => opSteadyStateUs(b) - opSteadyStateUs(a)).slice(0, 5);
  const topShare = top.reduce((sum, op) => sum + opSteadyStateUs(op), 0) / totalUs;

  const label = mk("div", "insight-strip-label");
  label.textContent = `Modeled steady-state Pareto — top ${top.length} ops ≈ ${(topShare * 100).toFixed(0)}% of `
    + `${formatUs(totalUs)} total (one-time packing excluded)`;

  const bar = mk("div", "segment-bar");
  const TONES = ["pareto-1", "pareto-2", "pareto-3", "pareto-4", "pareto-5"];
  top.forEach((op, i) => {
    const steadyUs = opSteadyStateUs(op);
    const share = steadyUs / totalUs;
    const block = mk("button", `segment-block ${TONES[i]}`);
    block.type = "button";
    block.style.flexGrow = String(Math.max(share, 0.01) * 1000);
    block.title = `#${op.index} ${op.name} — ${formatUs(steadyUs)} steady (${(share * 100).toFixed(1)}%) · dominant: ${op.bottleneck_dominant || "?"}`;
    block.textContent = share > 0.08 ? `#${op.index}` : "";
    block.addEventListener("click", () => selectGraphOp(analysis, op.index, { scrollTable: true }));
    bar.append(block);
  });
  const rest = mk("div", "segment-block pareto-rest");
  rest.style.flexGrow = String(Math.max(1 - topShare, 0.01) * 1000);
  rest.title = `Remaining ${ops.length - top.length} ops — ${formatUs(totalUs * (1 - topShare))} (${((1 - topShare) * 100).toFixed(1)}%)`;
  bar.append(rest);
  opParetoBar.append(label, bar);
}

function renderGraphOpRows(analysis) {
  const term = graphSearch.value.trim().toLowerCase();
  let matched = analysis.ops.filter((op) => opMatchesSearch(analysis, op, term));

  // Apply active filters
  if (workspace.opFilterBound) matched = matched.filter((op) => op.static_bound_guess === workspace.opFilterBound);
  if (workspace.opFilterXnn === "delegated") matched = matched.filter((op) => Number(op.xnnpack_chain_id) >= 0);
  if (workspace.opFilterXnn === "break") matched = matched.filter((op) => op.xnnpack_chain_break);
  if (workspace.opFilterXnn === "fallback") matched = matched.filter((op) => Number(op.xnnpack_chain_id) < 0 && !op.xnnpack_chain_break);
  if (workspace.opFilterQuant === "compute") matched = matched.filter((op) => op.quantization_state === "quantized_compute");
  if (workspace.opFilterQuant === "warn") matched = matched.filter((op) => op.quant_risk === "risk" || op.quant_risk === "warn");
  if (workspace.opFilterQuant === "risk") matched = matched.filter((op) => op.quant_risk === "risk");

  // Apply sort (default: natural index order when sortKey is "")
  if (workspace.opTableSortKey) {
    const key = workspace.opTableSortKey;
    const dir = workspace.opTableSortDir;
    const QUANT_RISK_ORDER = { none: 0, warn: 1, risk: 2 };
    matched = [...matched].sort((a, b) => {
      const av = a[key] ?? 0;
      const bv = b[key] ?? 0;
      if (key === "quant_risk") {
        return dir * ((QUANT_RISK_ORDER[av] ?? 0) - (QUANT_RISK_ORDER[bv] ?? 0));
      }
      if (typeof av === "string") return dir * av.localeCompare(bv);
      return dir * (Number(av) - Number(bv));
    });
  }

  const total = matched.length;
  const CAP = 600;
  const rows = matched.slice(0, CAP);
  graphOpBody.replaceChildren(
    ...rows.map((op) => graphOpRow(op, {
      selected: op.index === workspace.selectedOpIndex,
      onSelect: (item) => selectGraphOp(analysis, item.index),
      analysis,
      lowNormStat: workspace.currentLowNormStatMap.get(op.index) ?? null,
    })),
  );

  // Update count badge
  if (opFilterCount) {
    const showing = Math.min(total, CAP);
    const isFiltered = total !== analysis.ops.length;
    const isCapped = showing < total;
    if (isFiltered || isCapped) {
      opFilterCount.textContent = `${showing}/${analysis.ops.length}${isCapped && !isFiltered ? " (capped)" : " shown"}`;
    } else {
      opFilterCount.textContent = "";
    }
  }

  // Update sort indicators on column headers
  if (graphOpHead) {
    for (const th of graphOpHead.querySelectorAll("th[data-sort-key]")) {
      th.classList.toggle("sort-asc", th.dataset.sortKey === workspace.opTableSortKey && workspace.opTableSortDir === 1);
      th.classList.toggle("sort-desc", th.dataset.sortKey === workspace.opTableSortKey && workspace.opTableSortDir === -1);
    }
  }
}



function buildTensorConsumers(analysis) {
  const consumers = new Map();
  for (const op of analysis?.ops || []) {
    for (const id of op.inputs || []) {
      if (id < 0) continue;
      if (!consumers.has(id)) consumers.set(id, []);
      consumers.get(id).push(op.index);
    }
  }
  return consumers;
}

function buildDuplicateWeightGroups(analysis) {
  const byKey = new Map();
  for (const t of analysis?.tensors || []) {
    if (!t?.constant_buffer || !t.buffer_hash || !(t.buffer_data_length > 0)) continue;
    const key = `${t.buffer_hash}:${t.buffer_data_length}`;
    if (!byKey.has(key)) byKey.set(key, []);
    byKey.get(key).push(t.index);
  }
  for (const [key, list] of byKey) {
    if (list.length < 2) byKey.delete(key);
  }
  return byKey;
}

function renderTensorExplorer(analysis) {
  if (!tensorBody) return;
  const { tensorFanOut } = workspace.currentTopologyAnnotations ?? { tensorFanOut: new Map() };
  const classifiedRoles = classifyTensorRoles(analysis);
  const roleByTensor = new Map(classifiedRoles.map(({ index, role }) => [index, role]));
  const tensorInventory = buildTensorInventory(analysis);
  const consumersMap = buildTensorConsumers(analysis);
  const dupGroups = buildDuplicateWeightGroups(analysis);
  const dupTensorToGroup = new Map();
  for (const [key, list] of dupGroups) {
    for (const idx of list) dupTensorToGroup.set(idx, list);
  }

  renderTensorMemoryTimeline(analysis, consumersMap);

  // Compute stats for header bar (always across all tensors, not filtered)
  if (tensorStatsBar) {
    const roleCount = (role) => classifiedRoles.filter((item) => item.role === role).length;
    const highFanOut = [...tensorFanOut.values()].filter(n => n > 1).length;
    const int8Acts   = analysis.tensors.filter(t => t && !t.constant_buffer && (t.dtype === "INT8" || t.dtype === "UINT8")).length;
    let dupWasted = 0;
    for (const [key, list] of dupGroups) {
      const len = Number(key.split(":")[1] || 0);
      dupWasted += (list.length - 1) * len;
    }

    const stat = (label, value, cls = "", tip = "") => {
      const s = document.createElement("span");
      s.className = `tstat ${cls}`;
      if (tip) s.title = tip;
      const b = document.createElement("b");
      b.textContent = String(value);
      s.append(b, " " + label);
      return s;
    };
    tensorStatsBar.replaceChildren(
      ...([
        stat("kernels", roleCount("kernel"), "", "Constants consumed at format-specific kernel input positions"),
        stat("biases", roleCount("bias"), "", "Constants consumed at format-specific bias input positions"),
        stat("activations", roleCount("activation"), "", "Non-constant graph tensors"),
        stat("metadata", roleCount("metadata"), "", "Constant tensors not uniquely classified as a kernel or bias"),
        highFanOut > 0 ? stat("fan-out >1", highFanOut, "tstat-warn", "Tensors consumed by 2+ ops — residual/FPN branch points") : null,
        tensorInventory.quantized_tensors > 0 ? stat("quantized", tensorInventory.quantized_tensors, "tstat-info", "Tensors carrying quantization scale/zero-point metadata") : null,
        int8Acts > 0 ? stat("INT8 acts", int8Acts, "tstat-quant", "INT8/UINT8 runtime tensors — quant boundary locations") : null,
        dupGroups.size > 0 ? stat("dup weights", `${dupGroups.size} × (${formatBytes(dupWasted)})`, "tstat-warn",
          "Constant tensors with byte-identical buffer data (same FNV-64 fingerprint + length) — duplicated storage; value is the wasted bytes beyond one copy per group") : null,
      ].filter(Boolean))
    );
  }

  const term = workspace.currentTensorFilter.toLowerCase();
  const roleFilter = workspace.currentTensorRoleFilter;

  const rows = [];
  for (const t of analysis.tensors) {
    if (!t) continue;
    const isConstant = !!t.constant_buffer;
    const fanOut = tensorFanOut.get(t.index) ?? 0;
    const quantMode = tensorQuantizationMode(t);
    const hasQuant = quantMode !== "none";
    const tensorRole = roleByTensor.get(t.index) || (isConstant ? "metadata" : "activation");
    const isModelInput = (analysis.input_tensor_indices ?? []).includes(t.index);
    const isModelOutput = (analysis.output_tensor_indices ?? []).includes(t.index);

    // Role filter
    if (["kernel", "bias", "activation", "metadata"].includes(roleFilter) && tensorRole !== roleFilter) continue;
    if (roleFilter === "fanout" && fanOut <= 1) continue;
    if (roleFilter === "quant" && !hasQuant) continue;

    // Search filter
    if (term) {
      const matchStr = [
        String(t.index), `t${t.index}`, t.name || "",
        t.dtype || "", (t.shape || []).join("x"),
      ].join(" ").toLowerCase();
      if (!matchStr.includes(term)) continue;
    }

    rows.push({ t, tensorRole, fanOut, hasQuant, quantMode, isModelInput, isModelOutput });
  }

  const { producers } = buildGraphIndex(analysis);
  tensorBody.replaceChildren(
    ...rows.map(({ t, tensorRole, fanOut, hasQuant, quantMode, isModelInput, isModelOutput }) => {
      const tr = document.createElement("tr");
      tr.className = "tensor-row";
      tr.tabIndex = 0;
      tr.setAttribute("role", "button");
      const role = tensorRole;
      const roleLabel = isModelInput
        ? `${role} / model input`
        : isModelOutput
          ? `${role} / model output`
          : role;
      const producerOpIdx = producers.get(t.index);
      const shapeStr = (t.shape || []).join("×") || "-";
      const quantStr = hasQuant
        ? `${quantMode.replace("_", "-")}${t.scale_sample?.length ? ` · ${Number(t.scale_sample[0]).toPrecision(5)}` : ""}`
        : "none";

      const mkTd = (cls, text, title) => {
        const td = document.createElement("td");
        td.className = cls;
        td.textContent = text ?? "";
        if (title != null) td.title = title;
        return td;
      };
      const dtypeCls = t.dtype === "INT8" || t.dtype === "UINT8" ? "tensor-int8"
        : t.dtype === "FLOAT32" ? "tensor-f32" : "";
      const fanoutCls = fanOut > 2 ? " tensor-fanout-high" : fanOut > 1 ? " tensor-fanout-multi" : "";

      // Name cell with duplicate-weight marker
      const dupGroup = dupTensorToGroup.get(t.index);
      const nameTd = mkTd("tensor-name", t.name ? t.name.slice(-32) : "(unnamed)", t.name || "");
      if (dupGroup) {
        nameTd.textContent += " ⧉";
        nameTd.title = `${t.name || "(unnamed)"}\nByte-identical weight data shared with: ${dupGroup.filter((i) => i !== t.index).map((i) => `T${i}`).join(", ")}`;
      }

      // Consumers cell: clickable op chips
      const consumerIdxs = consumersMap.get(t.index) || [];
      const consTd = document.createElement("td");
      consTd.className = "tensor-consumers";
      if (!consumerIdxs.length) {
        consTd.textContent = "—";
      } else {
        const shown = consumerIdxs.slice(0, 3);
        for (const ci of shown) {
          const chip = document.createElement("button");
          chip.type = "button";
          chip.className = "consumer-chip";
          chip.textContent = `#${ci}`;
          chip.title = `Jump to consumer op #${ci}`;
          chip.addEventListener("click", (e) => {
            e.stopPropagation();
            if (!workspace.current) return;
            switchExplorerTab("ops");
            selectGraphOp(workspace.current, ci, { scrollTable: true });
          });
          consTd.append(chip);
        }
        if (consumerIdxs.length > shown.length) {
          const more = document.createElement("span");
          more.className = "consumer-more";
          more.textContent = `+${consumerIdxs.length - shown.length}`;
          more.title = `All consumers: ${consumerIdxs.map((i) => `#${i}`).join(", ")}`;
          consTd.append(more);
        }
      }

      tr.append(
        mkTd("tensor-id", `T${t.index}`),
        nameTd,
        mkTd("tensor-shape numeric", shapeStr),
        mkTd(`tensor-dtype${dtypeCls ? " " + dtypeCls : ""}`, t.dtype || "?"),
        mkTd(`tensor-role role-${role}`, roleLabel),
        mkTd("tensor-producer", producerOpIdx != null ? `#${String(producerOpIdx).padStart(3, "0")}` : "—"),
        mkTd(`tensor-fanout${fanoutCls}`, String(fanOut)),
        consTd,
        mkTd("tensor-quant", quantStr),
      );
      // Click → jump to producer op
      const navigateToProducer = () => {
        if (producerOpIdx != null && workspace.current) {
          // Switch to ops tab and select this op
          switchExplorerTab("ops");
          selectGraphOp(workspace.current, producerOpIdx, { scrollTable: true });
        }
      };
      tr.addEventListener("click", navigateToProducer);
      tr.addEventListener("keydown", (event) => {
        if (event.target !== tr || !["Enter", " "].includes(event.key)) return;
        event.preventDefault();
        navigateToProducer();
      });
      if (producerOpIdx != null) {
        tr.title = `Open producer op #${producerOpIdx}`;
        tr.setAttribute("aria-label", `Tensor T${t.index}; open producer op #${producerOpIdx}`);
      } else {
        tr.setAttribute("aria-label", `Tensor T${t.index}; no producer operator`);
      }
      return tr;
    }),
  );
}


const TENSOR_DTYPE_BYTES = {
  FLOAT64: 8, INT64: 8, UINT64: 8, COMPLEX64: 8,
  FLOAT32: 4, INT32: 4, UINT32: 4,
  FLOAT16: 2, INT16: 2, UINT16: 2,
  INT8: 1, UINT8: 1, BOOL: 1, INT4: 1,
};

function renderTensorMemoryTimeline(analysis, consumersMap) {
  if (!tensorMemoryTimeline) return;
  if (analysis?.tensor_arena_plan || analysis?.tensor_liveness) {
    renderTensorArenaViewer(tensorMemoryTimeline, analysis, {
      formatBytes,
      runtimeEvidence: workspace.runtimeAssignmentEvidence,
      onSelectOp: (opIndex) => {
        if (!workspace.current) return;
        switchExplorerTab("ops");
        selectGraphOp(workspace.current, opIndex, { scrollTable: true });
      },
    });
    return;
  }
  tensorMemoryTimeline.replaceChildren();
  const ops = analysis?.ops || [];
  const N = ops.length;
  if (!N) return;

  const { producers } = buildGraphIndex(analysis);
  const graphInputs = new Set(analysis.input_tensor_indices ?? []);
  const graphOutputs = new Set(analysis.output_tensor_indices ?? []);

  let hasDynamicDims = false;
  const spans = [];
  for (const t of analysis.tensors) {
    if (!t || t.constant_buffer) continue;
    const cons = consumersMap.get(t.index) || [];
    const prod = producers.get(t.index);
    const isIn = graphInputs.has(t.index);
    const isOut = graphOutputs.has(t.index);
    if (prod == null && !cons.length && !isIn && !isOut) continue; // detached tensor
    const birth = isIn ? 0 : (prod ?? 0);
    const death = isOut ? N - 1 : (cons.length ? Math.max(...cons) : birth);
    let elems = 1;
    for (const d of t.shape || []) {
      if (d > 0) elems *= d;
      else hasDynamicDims = true; // dynamic dim counted as 1 — underestimates
    }
    spans.push({ t, birth: Math.max(0, birth), death: Math.min(N - 1, Math.max(birth, death)), bytes: elems * (TENSOR_DTYPE_BYTES[t.dtype] ?? 4) });
  }
  if (!spans.length) return;

  const delta = new Float64Array(N + 1);
  for (const s of spans) {
    delta[s.birth] += s.bytes;
    delta[s.death + 1] -= s.bytes;
  }
  let acc = 0, peak = 0, peakIdx = 0;
  const curve = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    acc += delta[i];
    curve[i] = acc;
    if (acc > peak) { peak = acc; peakIdx = i; }
  }
  if (peak <= 0) return;

  const label = mk("div", "insight-strip-label");
  const peakOp = ops.find((o) => o.index === peakIdx);
  label.textContent = `Est. activation memory — peak ${formatBytes(peak)} at op #${peakIdx}${peakOp ? ` ${peakOp.name}` : ""} `
    + `(static upper bound, no arena reuse${hasDynamicDims ? "; dynamic dims counted as 1" : ""})`;

  const canvas = document.createElement("canvas");
  const W = Math.max(480, Math.round(tensorMemoryTimeline.clientWidth || graphExplorer?.clientWidth || 1200));
  const H = 84, PL = 6, PB = 14;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  canvas.style.height = `${H}px`;
  canvas.className = "memory-timeline-canvas";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const toX = (i) => PL + (i / (N - 1 || 1)) * (W - PL * 2);
  const toY = (v) => (H - PB) - (v / peak) * (H - PB - 4);

  ctx.beginPath();
  ctx.moveTo(toX(0), H - PB);
  for (let i = 0; i < N; i++) ctx.lineTo(toX(i), toY(curve[i]));
  ctx.lineTo(toX(N - 1), H - PB);
  ctx.closePath();
  ctx.fillStyle = "rgba(59,130,246,0.25)";
  ctx.fill();
  ctx.beginPath();
  for (let i = 0; i < N; i++) { const x = toX(i), y = toY(curve[i]); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
  ctx.strokeStyle = "#3b82f6";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  // Peak marker
  ctx.beginPath();
  ctx.moveTo(toX(peakIdx), toY(peak));
  ctx.lineTo(toX(peakIdx), H - PB);
  ctx.strokeStyle = "#f59e0b";
  ctx.setLineDash([3, 3]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "10px system-ui";
  ctx.fillText("op 0", PL, H - 3);
  ctx.textAlign = "right";
  ctx.fillText(`op ${N - 1}`, W - PL, H - 3);
  ctx.textAlign = "left";

  // Top tensors alive at the peak step
  const atPeak = spans.filter((s) => s.birth <= peakIdx && peakIdx <= s.death)
    .sort((a, b) => b.bytes - a.bytes).slice(0, 5);
  const peakList = mk("div", "memory-peak-list");
  for (const s of atPeak) {
    const chip = mk("span", "memory-peak-chip");
    chip.textContent = `T${s.t.index} ${formatBytes(s.bytes)}`;
    chip.title = `${s.t.name || "(unnamed)"} · ${(s.t.shape || []).join("×")} ${s.t.dtype} · alive #${s.birth}–#${s.death}`;
    peakList.append(chip);
  }

  const countDelta = new Int32Array(N + 1);
  for (const s of spans) { countDelta[s.birth] += 1; countDelta[s.death + 1] -= 1; }
  const aliveCounts = new Int32Array(N);
  for (let i = 0, c = 0; i < N; i++) { c += countDelta[i]; aliveCounts[i] = c; }

  const baseFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const tip = mk("div", "layered-tooltip");
  tip.hidden = true;
  canvas.title = "Hover to inspect per-op memory · click to jump to the op";

  const stepFromEvent = (e) => {
    const r = canvas.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W; // canvas is CSS-stretched to 100%
    return Math.max(0, Math.min(N - 1, Math.round(((x - PL) / (W - PL * 2)) * (N - 1))));
  };

  const restoreBase = () => {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.putImageData(baseFrame, 0, 0);
    ctx.restore();
  };
  canvas.addEventListener("mousemove", (e) => {
    const i = stepFromEvent(e);
    restoreBase();
    ctx.beginPath();
    ctx.moveTo(toX(i), 4);
    ctx.lineTo(toX(i), H - PB);
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(toX(i), toY(curve[i]), 2.5, 0, Math.PI * 2);
    ctx.fillStyle = "#e2e8f0";
    ctx.fill();
    const op = ops.find((o) => o.index === i);
    tip.textContent = `#${i}${op ? ` ${op.name}` : ""} · ${formatBytes(curve[i])} · ${aliveCounts[i]} tensors alive`;
    const r = canvas.getBoundingClientRect();
    const wrapR = tensorMemoryTimeline.getBoundingClientRect();
    tip.style.left = `${Math.min(e.clientX - wrapR.left + 12, wrapR.width - 200)}px`;
    tip.style.top = `${r.top - wrapR.top + 4}px`;
    tip.hidden = false;
  });
  canvas.addEventListener("mouseleave", () => {
    restoreBase();
    tip.hidden = true;
  });
  canvas.addEventListener("click", (e) => {
    if (!workspace.current) return;
    const i = stepFromEvent(e);
    switchExplorerTab("ops");
    selectGraphOp(workspace.current, i, { scrollTable: true });
  });

  tensorMemoryTimeline.append(label, canvas, tip, peakList);
}


const LAYERED_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"];
const LAYERED_SCALE_Z = 0.1, LAYERED_MIN_Z = 10, LAYERED_MAX_Z = 400;
const LAYERED_SCALE_XY = 1, LAYERED_MIN_XY = 10, LAYERED_MAX_XY = 512;
const LAYERED_SPACING = 10, LAYERED_PADDING = 10, LAYERED_SHADE = 10, LAYERED_OP_CAP = 400;
const LAYERED_MAX_DEV_W = 16000;        // device px (~half the hard 32,767 limit)
const LAYERED_MAX_DEV_AREA = 12_000_000; // device px² ≈ 48 MB of pixel buffer
let layeredViewStale = true;             // render lazily: only when the tab is visible

function layeredFade(hex, amount) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.max(0, (n >> 16) - amount), g = Math.max(0, ((n >> 8) & 255) - amount), b = Math.max(0, (n & 255) - amount);
  return `rgb(${r},${g},${b})`;
}

function layeredBoxSize(shape) {
  const dims = (shape || []).slice(1).map((d) => Math.max(1, d)); // drop batch
  let h = 1, w = 1, c = 1;
  if (dims.length >= 3) {
    h = dims[0]; w = dims[1];
    c = dims.slice(2).reduce((a, b) => a * b, 1);
  } else if (dims.length > 0) {
    // 1D/2D: no real spatial/channel structure — last value on the z axis
    c = dims[dims.length - 1];
  }
  const x = Math.min(Math.max(h * LAYERED_SCALE_XY, LAYERED_MIN_XY), LAYERED_MAX_XY); // H → extrusion
  const y = Math.min(Math.max(w * LAYERED_SCALE_XY, LAYERED_MIN_XY), LAYERED_MAX_XY); // W → screen height
  const z = Math.min(Math.max(Math.round(c * LAYERED_SCALE_Z), LAYERED_MIN_Z), LAYERED_MAX_Z); // C → screen width
  return { width: z, height: y, de: Math.round(x / 3) };
}

function renderLayeredView(analysis) {
  if (!layeredViewPanel) return;
  layeredViewPanel.replaceChildren();
  const allOps = analysis?.ops || [];
  if (!allOps.length) return;
  const ops = allOps.slice(0, LAYERED_OP_CAP);

  // Color wheel: per op type, first-seen order (visualtorch ColorWheel)
  const typeColor = new Map();
  const colorFor = (name) => {
    if (!typeColor.has(name)) typeColor.set(name, LAYERED_PALETTE[typeColor.size % LAYERED_PALETTE.length]);
    return typeColor.get(name);
  };

  // Layout: linear columns (layout_columns with one box per column)
  const boxes = ops.map((op) => {
    const { width, height, de } = layeredBoxSize((op.output_shapes || [])[0]);
    return { op, width, height, de, fill: colorFor(op.name), x1: 0, y1: 0 };
  });
  let cursor = LAYERED_PADDING;
  const xOff = Math.max(0, Math.round(boxes[0].de / 2));
  let maxRight = 0, bandH = 0, maxDe = 0;
  for (const b of boxes) {
    b.x1 = cursor - Math.round(b.de / 2) + xOff; // x_shift = -de/2 (+global x_off)
    maxRight = Math.max(maxRight, b.x1 + b.width + b.de);
    bandH = Math.max(bandH, b.height);
    maxDe = Math.max(maxDe, b.de);
    cursor += b.width + LAYERED_SPACING;
  }
  const topPad = maxDe + 4; // headroom so tall extrusions never clip
  for (const b of boxes) b.y1 = topPad + (bandH - b.height) / 2; // vertical centering

  layeredViewStale = false;
  const cssW = Math.ceil(maxRight + LAYERED_PADDING);
  const cssH = Math.ceil(topPad + bandH + 18);

  const dpr = window.devicePixelRatio || 1;
  const fit = Math.min(
    1,
    LAYERED_MAX_DEV_W / (cssW * dpr),
    Math.sqrt(LAYERED_MAX_DEV_AREA / (cssW * cssH * dpr * dpr)),
  );

  const note = mk("div", "insight-strip-label");
  note.textContent = `Legacy layered overview (VisualTorch-derived layout concepts) — box width ∝ channels ×${LAYERED_SCALE_Z}, `
    + `height ∝ spatial W, depth ∝ H/3 · ${ops.length}${allOps.length > ops.length ? ` of ${allOps.length}` : ""} ops`
    + `${allOps.length > ops.length ? " (truncated)" : ""}`
    + `${fit < 1 ? ` · scaled ×${fit.toFixed(2)} to fit canvas memory budget` : ""}`
    + ` · click a box to inspect the op · `;
  const noticeLink = document.createElement("a");
  noticeLink.href = "./NOTICE.txt";
  noticeLink.target = "_blank";
  noticeLink.rel = "license";
  noticeLink.textContent = "third-party notice";
  note.append(noticeLink);

  const scroll = mk("div", "layered-scroll");
  const canvas = document.createElement("canvas");
  const px = dpr * fit;
  canvas.width = Math.max(1, Math.round(cssW * px));
  canvas.height = Math.max(1, Math.round(cssH * px));
  canvas.style.width = `${Math.round(cssW * fit)}px`;
  canvas.style.height = `${Math.round(cssH * fit)}px`;
  canvas.className = "layered-canvas";
  const ctx = canvas.getContext("2d");
  ctx.scale(px, px);
  ctx.lineWidth = 1;

  const OUTLINE = "#0b1220";        // visualtorch uses black-on-white; dark-slate for our theme
  const FUNNEL = "rgba(148,163,184,0.55)";

  function drawFunnel(s, e) { // _draw_funnel: 4 tapered lines, drawn before the target box
    const sy1 = s.y1, sy2 = s.y1 + s.height, ey1 = e.y1, ey2 = e.y1 + e.height;
    const sx2 = s.x1 + s.width, ex1 = e.x1;
    ctx.strokeStyle = FUNNEL;
    ctx.beginPath();
    ctx.moveTo(sx2 + s.de, sy1 - s.de); ctx.lineTo(ex1 + e.de, ey1 - e.de);
    ctx.moveTo(sx2 + s.de, sy2 - s.de); ctx.lineTo(ex1 + e.de, ey2 - e.de);
    ctx.moveTo(sx2, sy2); ctx.lineTo(ex1, ey2);
    ctx.moveTo(sx2, sy1); ctx.lineTo(ex1, ey1);
    ctx.stroke();
  }

  function drawBox(b, selected) { // Box.draw: hidden edges, top face, right face, front rect
    const { x1, y1, width, height, de, fill } = b;
    const x2 = x1 + width, y2 = y1 + height;
    ctx.strokeStyle = selected ? "#f59e0b" : OUTLINE;
    if (de > 0) {
      ctx.beginPath(); // hidden back-left edges
      ctx.moveTo(x1 + de, y1 - de); ctx.lineTo(x1 + de, y2 - de);
      ctx.lineTo(x1, y2);
      ctx.moveTo(x1 + de, y2 - de); ctx.lineTo(x2 + de, y2 - de);
      ctx.stroke();
      ctx.beginPath(); // top face (shade ×1)
      ctx.moveTo(x1, y1); ctx.lineTo(x1 + de, y1 - de); ctx.lineTo(x2 + de, y1 - de); ctx.lineTo(x2, y1); ctx.closePath();
      ctx.fillStyle = layeredFade(fill, LAYERED_SHADE);
      ctx.fill(); ctx.stroke();
      ctx.beginPath(); // right face (shade ×2)
      ctx.moveTo(x2 + de, y1 - de); ctx.lineTo(x2, y1); ctx.lineTo(x2, y2); ctx.lineTo(x2 + de, y2 - de); ctx.closePath();
      ctx.fillStyle = layeredFade(fill, 2 * LAYERED_SHADE);
      ctx.fill(); ctx.stroke();
    }
    ctx.fillStyle = fill; // front face
    ctx.fillRect(x1, y1, width, height);
    ctx.strokeRect(x1, y1, width, height);
    if (selected) {
      ctx.save();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#f59e0b";
      ctx.strokeRect(x1 - 1, y1 - 1, width + 2, height + 2);
      ctx.restore();
    }
  }

  boxes.forEach((b, i) => {
    if (i > 0) drawFunnel(boxes[i - 1], b);
    drawBox(b, b.op.index === workspace.selectedOpIndex);
  });

  // Sparse op-index labels along the bottom
  ctx.fillStyle = "#94a3b8";
  ctx.font = "9px system-ui";
  ctx.textAlign = "center";
  const labelEvery = Math.max(1, Math.ceil(boxes.length / 30));
  boxes.forEach((b, i) => {
    if (i % labelEvery === 0) ctx.fillText(`#${b.op.index}`, b.x1 + b.width / 2, topPad + bandH + 12);
  });

  // Hover tooltip + click-to-select
  const tip = mk("div", "layered-tooltip");
  tip.hidden = true;
  // Mouse coords arrive in displayed (scaled) pixels; divide by fit for layout space
  const hit = (mx, my) => boxes.find((b) => mx >= b.x1 && mx <= b.x1 + b.width + b.de && my >= b.y1 - b.de && my <= b.y1 + b.height);
  canvas.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect();
    const b = hit((e.clientX - r.left) / fit, (e.clientY - r.top) / fit);
    if (!b) { tip.hidden = true; canvas.style.cursor = "default"; return; }
    canvas.style.cursor = "pointer";
    const op = b.op;
    const shape = ((op.output_shapes || [])[0] || []).join("×");
    tip.textContent = `#${op.index} ${op.name} · out ${shape || "?"}`
      + `${op.macs > 0 ? ` · ${formatNumber(op.macs)} MACs` : ""}`
      + `${opSteadyStateUs(op) > 0 ? ` · steady ${formatUs(opSteadyStateUs(op))}` : ""}`;
    tip.style.left = `${e.clientX - r.left + scroll.scrollLeft + 12}px`;
    tip.style.top = `${e.clientY - r.top - 6}px`;
    tip.hidden = false;
  });
  canvas.addEventListener("mouseleave", () => { tip.hidden = true; });
  canvas.addEventListener("click", (e) => {
    const r = canvas.getBoundingClientRect();
    const b = hit((e.clientX - r.left) / fit, (e.clientY - r.top) / fit);
    if (!b || !workspace.current) return;
    selectGraphOp(workspace.current, b.op.index, { scrollTable: true });
    renderLayeredView(workspace.current); // re-render for the selection highlight
  });

  scroll.append(canvas, tip);

  // Legend: op type → color, first-seen order (visualtorch legend)
  const legend = mk("div", "layered-legend");
  for (const [name, color] of typeColor) {
    const chip = mk("span", "layered-legend-chip");
    const swatch = mk("span", "layered-legend-swatch");
    swatch.style.background = color;
    chip.append(swatch, document.createTextNode(name));
    legend.append(chip);
  }

  layeredViewPanel.append(note, scroll, legend);
}


function switchGraphMode(mode) {
  workspace.currentGraphMode = mode;
  updateGraphModeHint(mode);
  // Re-render graph map
  if (workspace.current) deferGraphMap(workspace.current, workspace.selectedOpIndex);
}

function updateGraphModeHint(mode) {
  const HINTS = {
    raw:    "Raw structure: op names, shapes, tensor dtype (blue=INT8, grey=FP32)",
    deploy: workspace.current?.format === "onnx"
      ? "Deployment overlay: static bound posture and quantization state; execution-provider assignment is not modeled"
      : "Deployment overlay: bound type, predicted XNNPACK partition, quant risk",
    stage:  "Stage view: ops colored by processing stage group",
  };
  if (graphModeHint) graphModeHint.textContent = HINTS[mode] ?? "";
  const deployButton = document.querySelector('[data-graph-mode="deploy"]');
  if (deployButton) deployButton.title = HINTS.deploy;
}


function switchExplorerTab(tab) {
  const opsPanel = graphOpBody?.closest(".graph-list-wrap");
  if (!opsPanel || !tensorExplorerPanel) return;
  const showBlocks = tab === "blocks";
  const showResource = tab === "resource";
  const showQuant = tab === "quant";
  const showNode = tab === "node";
  const showCache = tab === "cache";
  if (blocksExplorerPanel) {
    blocksExplorerPanel.hidden = !showBlocks;
    if (showBlocks && workspace.current) explorerRedesignController.renderBlocks();
  }
  if (resourceMapPanel) {
    resourceMapPanel.hidden = !showResource;
    if (showResource && workspace.current) renderResourceMap(workspace.current);
  }
  if (cacheExplorerPanel) {
    cacheExplorerPanel.hidden = !showCache;
    if (showCache && workspace.current) explorerRedesignController.renderCache();
  }
  if (quantEvidencePanel) {
    quantEvidencePanel.hidden = !showQuant;
    if (showQuant && workspace.current) quantEvidenceController.render();
  }
  if (nodeViewPanel) {
    nodeViewPanel.hidden = !showNode;
    if (showNode && workspace.current) nodeViewController.render();
  }
  if (graphDetailLayout) graphDetailLayout.hidden = showBlocks || showResource || showQuant || showNode || showCache;
  const showOps = tab === "ops";
  opsPanel.hidden = !showOps;
  // Search/filters/view-mode/graph-map are op-centric — they belong to the Ops tab
  if (graphOpsView) graphOpsView.hidden = !showOps;
  if (showOps && workspace.current) fitGraphMap(); // re-fit: the map measures 0 while hidden
  const showTensors = tab === "tensors";
  const tensorsWasHidden = tensorExplorerPanel.hidden;
  tensorExplorerPanel.hidden = !showTensors;
  if (showTensors && tensorsWasHidden && workspace.current) renderTensorExplorer(workspace.current); // re-measure canvas width now that it is visible
  const showKernels = tab === "kernels";
  if (kernelInspectorPanel) {
    kernelInspectorPanel.hidden = !showKernels;
    if (showKernels && workspace.current) renderCurrentKernelInspector();
  }
  if (layeredViewPanel) {
    layeredViewPanel.hidden = tab !== "layered";
    if (tab === "layered" && layeredViewStale && workspace.current) renderLayeredView(workspace.current);
  }
  // Update tab active state
  for (const btn of document.querySelectorAll(".explorer-tab")) {
    btn.classList.toggle("active", btn.dataset.explorerTab === tab);
  }
  syncTabSelection(explorerTabs, (button) => button.dataset.explorerTab === tab);
}

function renderCurrentKernelInspector(placement = false) {
  if (!workspace.current || !kernelInspectorPanel) return;
  if (placement) renderExecutionPlacementView(explorerExecutionPlacementPanel, workspace.current, workspace.runtimeAssignmentEvidence);
  renderKernelInspector({
    analysis: workspace.current,
    body: kernelInspectorBody,
    summary: kernelInspectorSummary,
    comparisonPanel: runtimeAssignmentComparison,
    boundaryList: kernelBoundaryInventory,
    status: runtimeAssignmentStatus,
    query: kernelInspectorSearch?.value || "",
    filter: workspace.currentKernelFilter,
    runtimeEvidence: workspace.runtimeAssignmentEvidence,
    onSelect: (opIndex) => selectGraphOp(workspace.current, opIndex, { scrollTable: true }),
    onLoadSourceEvidence: loadProtectedSourceAnalysis,
  });
  renderRuntimeEvidenceClosure(runtimeEvidenceClosure, workspace.current, workspace.runtimeAssignmentEvidence);
  if (clearRuntimeAssignment) clearRuntimeAssignment.hidden = !workspace.runtimeAssignmentEvidence;
}

function selectGraphOp(analysis, opIndex, options = {}) {
  const op = analysis.ops.find((item) => item.index === opIndex);
  if (!op) return;
  if (options.fromGraph && graphSearch.value.trim() && !opMatchesSearch(analysis, op, graphSearch.value.trim().toLowerCase())) {
    graphSearch.value = "";
  }
  workspace.selectedOpIndex = opIndex;
  nodeViewController.selectOp(opIndex);
  quantEvidenceController.selectOp(opIndex);
  renderGraphOpRows(analysis);
  renderOpDetail(analysis, opIndex);
  deferGraphMap(analysis, opIndex);
  updateOpNav(analysis);
  if (options.scrollTable) {
    requestAnimationFrame(() => scrollGraphTableToOp(opIndex));
  }
}

function updateOpNav(analysis) {
  if (!opNavPrev || !opNavNext || !opNavLabel) return;
  const ops = analysis.ops;
  const idx = ops.findIndex((op) => op.index === workspace.selectedOpIndex);
  if (idx < 0) { opNavLabel.textContent = "—"; return; }
  opNavLabel.textContent = `${idx + 1} / ${ops.length}`;
  opNavPrev.disabled = idx === 0;
  opNavNext.disabled = idx === ops.length - 1;
  opNavPrev.onclick = () => selectGraphOp(analysis, ops[idx - 1].index, { scrollTable: true });
  opNavNext.onclick = () => selectGraphOp(analysis, ops[idx + 1].index, { scrollTable: true });
}

function selectGraphNodeFromMap(analysis, opIndex) {
  selectGraphOp(analysis, Number(opIndex), { fromGraph: true, scrollTable: true });
  graphMapStatus.textContent = `Selected #${padOp(opIndex)} from graph map`;
}

function scrollGraphTableToOp(opIndex) {
  const row = graphOpBody.querySelector(`tr[data-op-index="${opIndex}"]`);
  if (!row) return;
  const wrap = row.closest(".table-wrap");
  if (wrap) {
    const rowRect = row.getBoundingClientRect();
    const wrapRect = wrap.getBoundingClientRect();
    wrap.scrollTo({
      top: Math.max(0, wrap.scrollTop + rowRect.top - wrapRect.top - wrap.clientHeight / 2 + rowRect.height / 2),
      behavior: "smooth",
    });
  } else {
    row.scrollIntoView({ block: "center", behavior: "smooth" });
  }
  row.classList.add("jumped-row");
  window.setTimeout(() => row.classList.remove("jumped-row"), 900);
}

function closestGraphNode(target) {
  return target?.closest?.(".graph-node") || null;
}

function deferGraphMap(analysis, opIndex) {
  const token = ++workspace.graphRenderToken;
  graphMapSvg.replaceChildren();
  graphMapStatus.textContent = "Preparing graph map";
  requestAnimationFrame(() => {
    if (token !== workspace.graphRenderToken || workspace.current !== analysis) return;
    try {
      renderGraphMap(analysis, opIndex);
    } catch (error) {
      console.error(error);
      graphMapStatus.textContent = "Graph map failed";
    }
  });
}

function renderOpDetail(analysis, opIndex) {
  const op = analysis.ops.find(o => o.index === opIndex);
  let weightHistograms = null;
  let influence = null;
  let outputInfluence = null;
  if (op && workspace.currentModelBytes) {
    // Determine if this op consumes a model input tensor (first-layer detection)
    const inputTensorSet = new Set(analysis.input_tensor_indices ?? []);
    const isInputLayer = op.inputs.some(idx => idx >= 0 && inputTensorSet.has(idx));

    // Weight histograms computed in the Rust/WASM core.
    const hists = op.inputs
      .filter(idx => idx >= 0 && analysis.tensors[idx]?.constant_buffer)
      .map(idx => {
        try {
          const h = compute_weight_histogram(workspace.currentModelBytes, analysis.filename, idx, selectedTargetId());
          if (h) h.isInputLayer = isInputLayer;
          return h;
        } catch { return null; }
      })
      .filter(Boolean);
    if (hists.length > 0) weightHistograms = hists;
    // Influence computation in Rust/WASM using spatial BFS with kernel-weighted propagation.
    try { influence = compute_input_influence(workspace.currentModelBytes, analysis.filename, opIndex, selectedTargetId()) || null; } catch { influence = null; }
    try { outputInfluence = compute_output_influence(workspace.currentModelBytes, analysis.filename, opIndex, selectedTargetId()) || null; } catch { outputInfluence = null; }
  }
  renderOpDetailPanel(opDetail, analysis, opIndex, { weightHistograms, influence, outputInfluence, runtimeAssignment: workspace.runtimeAssignmentEvidence });
}

function renderGraphMap(analysis, opIndex) {
  graphMapSvg.replaceChildren();
  const fullGraph = graphDepth.value === "all";
  graphMapSvg.classList.toggle("full-graph", fullGraph);
  if (opIndex == null && !fullGraph) {
    graphMapStatus.textContent = "No graph";
    return;
  }
  const graphIndex = buildGraphIndex(analysis);
  const depth = fullGraph ? null : clampInt(graphDepth.value, 1, 4, 2);
  const graphData = fullGraph
    ? collectFullGraph(analysis, graphIndex)
    : collectNeighborhood(analysis, graphIndex, opIndex, depth);
  const layout = fullGraph ? layoutFoldedGraph(graphData.nodes) : layoutNeighborhood(graphData.nodes);
  workspace.graphMapBounds = layout.bounds;
  graphMapStatus.textContent = fullGraph
    ? `${layout.nodes.length} nodes / ${graphData.edges.length} edges / full graph`
    : `${layout.nodes.length} nodes / ${graphData.edges.length} edges / ${depth}-hop`;

  renderGraphMapContent(graphMapSvg, graphData, layout, {
    fullGraph,
    selectedOpIndex: workspace.selectedOpIndex,
    onSelect: (op) => selectGraphNodeFromMap(analysis, op.index),
    graphMode: workspace.currentGraphMode,
    topologyAnnotations: workspace.currentTopologyAnnotations,
    format: analysis.format,
    boundaryInventory: analysis.predicted_partition_boundaries || null,
    cpuIslands: analysis.delegation_repair?.cpu_islands || [],
    scenario: graphScenarioMatchesAnalysis(workspace.activeGraphScenario, analysis) ? workspace.activeGraphScenario : null,
  });

  fitGraphMap();
}

function fitGraphMap() {
  if (!workspace.graphMapBounds) return;
  workspace.graphViewBox = {
    x: workspace.graphMapBounds.x,
    y: workspace.graphMapBounds.y,
    width: workspace.graphMapBounds.width,
    height: workspace.graphMapBounds.height,
  };
  applyGraphViewBox();
}

function zoomGraphMap(factor) {
  if (!workspace.graphViewBox) return;
  const centerX = workspace.graphViewBox.x + workspace.graphViewBox.width / 2;
  const centerY = workspace.graphViewBox.y + workspace.graphViewBox.height / 2;
  const width = workspace.graphViewBox.width * factor;
  const height = workspace.graphViewBox.height * factor;
  workspace.graphViewBox = {
    x: centerX - width / 2,
    y: centerY - height / 2,
    width,
    height,
  };
  applyGraphViewBox();
}

function applyGraphViewBox() {
  if (!workspace.graphViewBox) return;
  graphMapSvg.setAttribute(
    "viewBox",
    `${workspace.graphViewBox.x} ${workspace.graphViewBox.y} ${workspace.graphViewBox.width} ${workspace.graphViewBox.height}`,
  );
}

async function downloadCurrentGraphSvg() {
  if (!workspace.current || !graphMapSvg.children.length) return;
  await downloadTextArtifact({
    artifact: TEXT_EXPORT_ARTIFACTS.graphSvg,
    buildText: () => graphSvgText(graphMapSvg),
    ...textExportOptions,
  });
}

async function runInferenceBenchmark() {
  const selected = backendSelect.value;
  const warmup = clampInt(warmupInput.value, 0, 100, 1);
  const runs = clampInt(runsInput.value, 1, 500, 50);
  const candidates = backendCandidates(selected, workspace.current?.format, navigator);

  runInference.disabled = true;
  benchmarkWrap.hidden = false;
  benchmarkBody.replaceChildren();
  workspace.runtimeBenchmarkResults = [];
  runtimeStatus.textContent = "Benchmarking";

  try {
    for (const backend of candidates) {
      const row = appendBenchmarkRow(benchmarkBody, backend, "running");
      try {
        const result = workspace.current?.format === "onnx"
          ? await loadOnnxBenchmark().then(({ benchmarkOnnxModel }) => benchmarkOnnxModel({ modelBytes: workspace.currentModelBytes, analysis: workspace.current, backend, warmup, runs, externalDataFiles: workspace.currentOnnxExternalDataFiles }))
          : await loadTfliteBenchmark().then(({ benchmarkTfliteModel }) => benchmarkTfliteModel({ modelBytes: workspace.currentModelBytes, analysis: workspace.current, backend, warmup, runs, ensureRuntime: ensureLiteRtRuntime }));
        const inputBasis = [...new Set((result.inputContracts || []).map((item) => item.basis).filter(Boolean))].join(" / ") || "not_recorded";
        const inputSummary = (result.inputContracts || []).map((item) => `${item.input_name}:${item.runtime_dtype}[${item.executed_shape.join("x") || "scalar"}]`).join(" / ");
        const outputSummary = (result.outputContracts || []).map((item) => `${item.output_name}:${item.runtime_dtype}[${item.executed_shape.join("x") || "scalar"}]`).join(" / ");
        const generatedAt = new Date().toISOString();
        const onnxRuntimeShapeBinding = workspace.current?.format === "onnx"
          ? buildOnnxRuntimeShapeBinding(workspace.current, {
            backend,
            generated_at: generatedAt,
            input_contracts: result.inputContracts || [],
            output_contracts: result.outputContracts || [],
          }) : null;
        if (onnxRuntimeShapeBinding) workspace.current.onnx_runtime_shape_binding = onnxRuntimeShapeBinding;
        const shapeBindingSummary = onnxRuntimeShapeBinding
          ? ` / shape binding ${onnxRuntimeShapeBinding.bound_symbol_count}/${onnxRuntimeShapeBinding.symbol_count}${onnxRuntimeShapeBinding.evaluated_total_macs_decimal ? ` / ${onnxRuntimeShapeBinding.evaluated_total_macs_decimal} MACs` : ""}` : "";
        const p99Evidence = p99EvidenceForSampleCount(result.timings.length);
        updateBenchmarkRow(row, {
          backend,
          compileMs: result.compileMs,
          firstRunMs: result.firstRunMs,
          timings: result.timings,
          stats: result.stats,
          steadyStats: result.steadyStats,
          status: `ok / outputs ${result.outputCount}${outputSummary ? ` ${outputSummary}` : ""}${inputSummary ? ` / inputs ${inputSummary}` : ""}${shapeBindingSummary}${result.outputDigest ? ` / digest ${result.outputDigest.slice(0, 16)}` : ""}`,
        });
        workspace.runtimeBenchmarkResults.push({
          backend,
          ok: true,
          compile_ms: result.compileMs,
          first_run_ms: result.firstRunMs,
          stats: result.stats,
          steady_stats: result.steadyStats,
          measured_samples_ms: [...result.timings],
          statistics_method: result.statisticsMethod,
          noise_method: result.noiseMethod,
          noise_diagnostics: result.noiseDiagnostics,
          p99_evidence: p99Evidence,
          warmup,
          runs,
          output_count: result.outputCount,
          output_digest: result.outputDigest || "",
          input_basis: inputBasis,
          input_contracts: result.inputContracts || [],
          output_contracts: result.outputContracts || [],
          external_data_runtime_binding: result.externalDataContract || null,
          onnx_runtime_shape_binding: onnxRuntimeShapeBinding,
          timing_method: result.timingMethod,
          phase_counts: result.phaseCounts,
          generated_at: generatedAt,
        });
        submitBenchmarkTelemetry({
          backend,
          compileMs: result.compileMs,
          firstRunMs: result.firstRunMs,
          stats: result.stats,
          steadyStats: result.steadyStats,
          warmup,
          runs,
          outputCount: result.outputCount,
          outputDigest: result.outputDigest || "",
          timingMethod: result.timingMethod,
          phaseCounts: result.phaseCounts,
        }).catch((error) => console.warn("Benchmark telemetry skipped", error));
      } catch (error) {
        workspace.runtimeBenchmarkResults.push({
          backend,
          ok: false,
          error: benchmarkErrorStatus(error, backend),
          warmup,
          runs,
          input_basis: "not_executed",
          generated_at: new Date().toISOString(),
        });
        updateBenchmarkRow(row, {
          backend,
          compileMs: null,
          stats: null,
          status: benchmarkErrorStatus(error, backend),
        });
      }
    }
  } finally {
    runtimeStatus.textContent = "Ready";
    runInference.disabled = false;
    updateWorkflowState("runtime");
    // Enrich findings with calibration computed by the Rust/WASM core.
    const wasmResult = selectWasmCalibrationResult(workspace.runtimeBenchmarkResults);
    if (wasmResult && workspace.current && workspace.currentModelBytes && workspace.currentFilename) {
      try {
        const measuredMs = wasmResult.steady_stats?.p50 ?? wasmResult.stats?.p50 ?? 0;
        if (measuredMs > 0) {
          const cal = compute_static_runtime_calibration(
            workspace.currentModelBytes, workspace.currentFilename, workspace.activeTargetId || "android_mid_a55", measuredMs
          );
          if (cal && findingsBody) renderFindingsCalibration(findingsBody, cal);
        }
      } catch (e) { console.warn("[calibration]", e); }
    }
  }
}

function renderStages(analysis) {
  const semanticStages = analysis?.block_inventory?.status === "assessed"
    ? analysis.block_inventory.stages || []
    : [];
  if (semanticStages.length) {
    stageCount.textContent = `${semanticStages.length} semantic stages`;
    stageStrip.replaceChildren(
      ...semanticStages.map((stage) => {
        const card = document.createElement("div");
        card.className = "stage stage-clickable";
        const title = document.createElement("b");
        title.textContent = `#${stage.index} ${stage.display_name || "operator group"}`;
        const body = document.createElement("span");
        const aggregate = stage.aggregates || {};
        body.textContent = `${formatNumber((stage.block_ids || []).length)} blocks / ${formatNumber((stage.op_indices || []).length)} ops / ${formatNumber(aggregate.macs || 0)} MACs (${formatPercent(aggregate.mac_percent || 0)}) / max L1 ${aggregate.l1_max_ratio == null ? "N/A" : `${Number(aggregate.l1_max_ratio).toFixed(2)}x`} / ${formatNumber(aggregate.l1_watch_count || 0)} L1 watch / ${formatNumber(aggregate.predicted_break_count || 0)} predicted breaks`;
        card.append(title, body);
        card.title = "Open the shared semantic stage inventory in Blocks";
        card.addEventListener("click", () => {
          setActiveWorkspace("graph", { force: true });
          switchExplorerTab("blocks");
        });
        return card;
      }),
    );
    return;
  }
  const structuralStages = analysis?.stages || [];
  stageCount.textContent = `${structuralStages.length} structural stages`;
  stageStrip.replaceChildren(
    ...structuralStages.map((stage) => {
      const card = stageCard(stage);
      card.classList.add("stage-clickable");
      card.title = `Click to explore Stage #${stage.index} in Graph Explorer`;
      card.addEventListener("click", () => jumpToStage(analysis, stage));
      return card;
    }),
  );
}

function jumpToStage(analysis, stage) {
  if (!workspace.current) return;
  setActiveWorkspace("graph", { force: true });
  requestAnimationFrame(() => {
    // Switch to Stage mode to show stage grouping colors
    workspace.currentGraphMode = "stage";
    for (const btn of document.querySelectorAll("[data-graph-mode]")) {
      btn.classList.toggle("active", btn.dataset.graphMode === "stage");
    }
    if (graphModeHint) graphModeHint.textContent = "Stage view: ops colored by processing stage group";

    // Filter op table to this stage by setting the search term
    if (graphSearch) {
      graphSearch.value = stage.key;
      renderGraphOpRows(analysis);
    }

    // Navigate to the first op of this stage
    const firstOp = analysis.ops.find(op => op.stage_index === stage.index);
    if (firstOp) {
      selectGraphOp(analysis, firstOp.index, { scrollTable: true });
    } else {
      deferGraphMap(analysis, workspace.selectedOpIndex);
    }
  });
}

function renderHistogram(analysis) {
  histogramBody.replaceChildren(
    ...(analysis?.histogram || []).map((item) => histogramRow(item)),
  );
}

function renderTopMacs(analysis) {
  topMacBody.replaceChildren(...topMacRows(analysis, (opIndex) => jumpToGraphOp(opIndex)));
}

function renderRoofline(analysis) {
  rooflineBody.replaceChildren(...rooflineTableRows(analysis, (opIndex) => jumpToGraphOp(opIndex)));
}


  return {
    renderSummary,
    adaptInsightsForUI,
    renderInsightDashboard,
    buildVisualPngFiles,
    jumpToGraphOp,
    graphScenarioMatchesAnalysis,
    renderGraphScenarioState,
    previewDelegationScenario,
    clearGraphScenarioPreview,
    renderInferencePanel,
    renderGraphExplorer,
    renderOpTimeline,
    renderXnnSegmentBar,
    renderOpParetoBar,
    renderResourceMap,
    renderGraphOpRows,
    buildTensorConsumers,
    buildDuplicateWeightGroups,
    renderTensorExplorer,
    renderTensorMemoryTimeline,
    renderLayeredView,
    switchGraphMode,
    updateGraphModeHint,
    switchExplorerTab,
    renderCurrentKernelInspector,
    selectGraphOp,
    updateOpNav,
    selectGraphNodeFromMap,
    scrollGraphTableToOp,
    closestGraphNode,
    deferGraphMap,
    renderOpDetail,
    renderGraphMap,
    fitGraphMap,
    zoomGraphMap,
    applyGraphViewBox,
    downloadCurrentGraphSvg,
    runInferenceBenchmark,
    renderStages,
    jumpToStage,
    renderHistogram,
    renderTopMacs,
    renderRoofline,
  };
}
