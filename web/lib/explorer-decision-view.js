import { formatBytes, formatPercent, formatUs, padOp } from "./format.js";
import {
  buildModelAtGlance,
  CACHE_WATCH_RATIO,
} from "./model-glance.js";
import { renderExplorerQuestionEntry } from "./explorer-question-view.js";

const VIEWS = new Set(["summary", "cache", "actions"]);
const CACHE_OPTIONS = [16, 32, 64, 128].map((kib) => ({
  id: `${kib}k`,
  label: `${kib} KiB`,
  bytes: kib * 1024,
}));
const PENALTY_LABELS = {
  quantization_coverage_penalty: "Quantization coverage",
  graph_runtime_pressure_penalty: "Correlated graph/runtime pressure",
  memory_posture_penalty: "Low-intensity posture",
  l1_watch_penalty: "L1 near-capacity watch",
  suspect_op_penalty: "Unsupported non-structural break suspects",
  predicted_boundary_penalty: "Predicted boundaries",
  fallback_byte_penalty: "Fallback traffic",
  quantization_risk_penalty: "Quantization risk",
  copy_like_op_penalty: "Copy-like movement",
  dynamic_input_penalty: "Dynamic input",
};

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function button(text, className = "") {
  const node = element("button", className, text);
  node.type = "button";
  return node;
}

function metricRow(label, value, detail = "", evidence = "") {
  const row = element("div", "glance-ledger-row");
  const labelNode = element("span", "glance-ledger-label", label);
  const valueWrap = element("div", "glance-ledger-value");
  valueWrap.append(element("strong", "", value));
  if (detail) valueWrap.append(element("small", "", detail));
  row.append(labelNode, valueWrap);
  if (evidence) row.append(element("em", "glance-evidence", evidence));
  return row;
}

function ratioText(value) {
  return value == null ? "N/A" : `${Number(value).toFixed(2)}x`;
}

function counted(count, singular, plural = `${singular}s`) {
  return `${count} ${Number(count) === 1 ? singular : plural}`;
}

function latencyComposition(glance) {
  const totals = glance.latency.totals;
  return [
    { id: "compute", label: "Compute", value: totals.activeComputeUs },
    { id: "memory", label: "Memory", value: totals.activeMemoryUs },
    { id: "mixed", label: "Roofline tie", value: totals.mixedRooflineUs },
    { id: "fallback", label: "Fallback", value: totals.fallbackUs },
    { id: "other", label: "Other", value: totals.otherUs },
  ].filter((item) => item.value > 0);
}

function compositionBar(items, total, className = "") {
  const wrap = element("div", `glance-composition ${className}`.trim());
  for (const item of items) {
    const segment = element("span", `glance-component component-${item.id}`);
    segment.style.width = `${Math.max(0, total > 0 ? item.value / total * 100 : 0)}%`;
    segment.title = `${item.label}: ${formatUs(item.value)} (${formatPercent(total > 0 ? item.value / total : 0)})`;
    wrap.append(segment);
  }
  return wrap;
}

function componentLegend(items, total) {
  const legend = element("div", "glance-component-legend");
  for (const item of items) {
    const label = element("span", `component-${item.id}`);
    label.append(element("i"), document.createTextNode(`${item.label} ${formatPercent(total > 0 ? item.value / total : 0)}`));
    legend.append(label);
  }
  return legend;
}

function frontierText(glance) {
  if (!glance.frontierTargets.length) return "Not assessed";
  return glance.frontierTargets
    .map((target) => `${target.label}: ${formatUs(target.steadyStateUs)} steady / ${formatUs(target.coldStartUs)} cold`)
    .join(" / ");
}

function delegationText(glance) {
  if (glance.format === "onnx") return "ORT EP runtime assignment not observed";
  const delegation = glance.delegation;
  const delegatedMacs = delegation.delegatedMacRatio == null
    ? "MAC share N/A (shape unbound)"
    : `${formatPercent(delegation.delegatedMacRatio)} MACs`;
  return `${delegation.delegatedOpCount}/${delegation.opCount} ops; ${delegatedMacs}; ${counted(delegation.boundaryEdgeCount, "boundary", "boundaries")}; ${counted(delegation.cpuIslandCount, "CPU island")}`;
}

function memoryText(glance) {
  const memory = glance.memory;
  const cache = memory.cache;
  return [
    memory.peakLiveBytes == null ? "peak live N/A" : `peak live ${formatBytes(memory.peakLiveBytes)}`,
    memory.arenaBytes == null ? "arena N/A" : `arena ${formatBytes(memory.arenaBytes)}`,
    memory.artifactPlusArenaBytes == null
      ? "artifact + arena N/A"
      : `artifact + arena ${formatBytes(memory.artifactPlusArenaBytes)}`,
    `L1 ${ratioText(cache.l1.maxRatio)} (${cache.l1.watchCount} watch)`,
    cache.l2.assessmentStatus === "assessed"
      ? `L2 ${ratioText(cache.l2.maxRatio)} (${cache.l2.watchCount} watch)`
      : `L2 watch N/A (${String(cache.l2.assessmentStatus || "unbound").replaceAll("_", " ")})`,
  ].join("; ");
}

function boundaryPayloadText(glance) {
  const bytes = formatBytes(glance.delegation.boundaryPayloadBytes || 0);
  const payload = glance.delegation.boundaryPayloadStatus === "complete"
    ? `${bytes} boundary payload`
    : `PARTIAL: ${bytes} assessed boundary payload`;
  const overhead = glance.delegation.boundaryOverhead;
  const motif = glance.delegation.motifAttribution?.summary;
  return overhead?.highUs > 0
    ? `${payload}; cold-start partition-planning setup range ${formatUs(overhead.lowUs)}-${formatUs(overhead.highUs)} (midpoint ${formatUs(overhead.midpointUs)}). This unmeasured profile constant is independent of logical payload and is excluded from steady state.${motif ? ` ${motif}` : ""}`
    : `${payload}${motif ? `; ${motif}` : ""}`;
}

function summaryView(glance, onSelectOp) {
  const body = element("div", "glance-summary-layout");
  const ledger = element("div", "glance-ledger");
  const shapeBindingRequired = ["requires_explicit_shape_binding", "batch_one_projection_formula_incomplete"]
    .includes(glance.artifact.dynamicShapeProjection?.status);
  ledger.append(
    metricRow(
      "Intensity",
      shapeBindingRequired
        ? "Not assessed"
        : `High ${glance.intensity.highCount} / mixed ${glance.intensity.mixedCount} / low ${glance.intensity.lowCount}`,
      shapeBindingRequired
        ? `Not assessed until non-batch dynamic axes are bound; ${glance.intensity.opCount} serialized ops remain inventoried.`
        : `${glance.intensity.assessedOpCount}/${glance.intensity.opCount} ops assessed${glance.intensity.ridgeOpsPerByte == null ? "" : `; target thresholds low <${glance.target.memoryBoundIntensity} / high >=${glance.target.computeBoundIntensity} ops/B (ridge ${glance.intensity.ridgeOpsPerByte})`}`,
      glance.format === "onnx" ? "DERIVED" : "ESTIMATED",
    ),
    metricRow("Delegation", delegationText(glance), glance.format === "onnx" ? "Source compatibility is not runtime placement." : boundaryPayloadText(glance), glance.format === "onnx" ? "NOT OBSERVED" : "PREDICTED"),
    metricRow("Memory", memoryText(glance), `${formatBytes(glance.memory.cache.l1.bytes)} L1D / ${formatBytes(glance.memory.cache.l2.bytes)} L2; watch >=${CACHE_WATCH_RATIO.toFixed(2)}x`, `DERIVED FROM ${glance.memory.performanceModelEvidenceClass} PROFILE`),
    metricRow("Target totals", frontierText(glance), glance.format === "onnx" ? "Cross-host latency model not assessed." : "Steady excludes one-time packing and partition-planning setup; cold adds both once. No cache-miss penalty inferred.", glance.format === "onnx" ? "NOT ASSESSED" : "HEURISTIC MODEL"),
  );

  const side = element("div", "glance-side");
  const composition = latencyComposition(glance);
  const batchProjection = glance.artifact.dynamicShapeProjection;
  const projectionNote = batchProjection?.status === "assumption_bound_batch_one"
    ? " Dynamic-shape arithmetic is evaluated exactly at the serialized N=1 projection; N>1 is not bounded."
    : "";
  if (glance.latency.totals.steadyStateUs > 0) {
    const block = element("section", "glance-composition-block");
    const heading = element("div", "glance-subhead");
    const latencyRange = glance.latency.range;
    const steadyHeading = latencyRange?.status === "cold_setup_profile_range"
      ? `${formatUs(latencyRange.steadyLowUs)}-${formatUs(latencyRange.steadyHighUs)}`
      : formatUs(glance.latency.totals.steadyStateUs);
    heading.append(element("strong", "", "Steady-state modeled composition"), element("span", "", steadyHeading));
    block.append(
      heading,
      compositionBar(composition, glance.latency.totals.steadyStateUs),
      componentLegend(composition, glance.latency.totals.steadyStateUs),
      element("p", "glance-boundary-note", latencyRange?.status === "cold_setup_profile_range"
        ? `Steady point ${formatUs(latencyRange.steadyPointUs)}; PAD-folding candidate low ${formatUs(latencyRange.steadyLowUs)}. Cold range ${formatUs(latencyRange.coldLowUs)}-${formatUs(latencyRange.coldHighUs)} = steady + one-time packing ${formatUs(glance.latency.totals.packingUs)} + partition-planning setup ${formatUs(glance.delegation.boundaryOverhead.lowUs)}-${formatUs(glance.delegation.boundaryOverhead.highUs)}. ${glance.delegation.padFusion?.directConvolutionCandidateCount || 0}/${glance.delegation.padFusion?.candidateCount || 0} PAD node(s) have exactly one convolution-family consumer. Low bound removes ${formatUs(latencyRange.padFusionRecoverableUpperBoundUs || 0)} only as a runtime-unobserved candidate upper bound; point/high retain explicit PAD materialization. Conservation: ${glance.latency.conservationStatus}. Roofline uses max(compute, memory).${projectionNote}`
        : `Cold start ${formatUs(glance.latency.totals.coldStartUs)} = steady ${formatUs(glance.latency.totals.steadyStateUs)} + one-time packing ${formatUs(glance.latency.totals.packingUs)} + partition-planning setup ${formatUs(glance.latency.totals.boundaryUs)}. Conservation: ${glance.latency.conservationStatus}. Roofline uses max(compute, memory).${projectionNote}`),
    );
    side.append(block);
  }

  const score = element("section", "glance-score-breakdown");
  const scoreHead = element("div", "glance-subhead");
  scoreHead.append(
    element("strong", "", "Component triage signals"),
    element("span", "", glance.score.penalties.length ? `${glance.score.penalties.length} review` : "No flagged component"),
  );
  score.append(scoreHead);
  if (glance.score.value == null) {
    score.append(element("p", "glance-empty", "Not assessed for this format."));
  } else if (!glance.score.penalties.length) {
    score.append(element("p", "glance-empty", "No configured component signal was triggered."));
  } else {
    for (const penalty of glance.score.penalties) {
      const row = element("div", "glance-penalty-row");
      row.append(
        element("span", "", PENALTY_LABELS[penalty.key] || penalty.key.replaceAll("_", " ")),
        element("strong", "", "Review"),
      );
      score.append(row);
    }
  }
  score.append(element("small", "glance-score-method", "HEURISTIC components; no context-free 0–100 score. Deployment requirements, runtime evidence, and task-output acceptance evidence are required for release decisions."));
  side.append(score);
  body.append(ledger, side);

  if (glance.latency.rows.length) {
    const hotspot = element("section", "glance-hotspots");
    const head = element("div", "glance-subhead");
    head.append(element("strong", "", "Top modeled time contributors"), element("span", "", "Click an op to inspect"));
    hotspot.append(head);
    const maximum = Math.max(...glance.latency.rows.map((row) => row.steadyStateUs), 1);
    for (const row of glance.latency.rows.slice(0, 6)) {
      const item = button("", "glance-hotspot-row");
      item.dataset.opIndex = String(row.opIndex);
      item.addEventListener("click", () => onSelectOp?.(row.opIndex));
      const identity = element("span", "glance-hotspot-id");
      identity.append(
        element("strong", "", `#${padOp(row.opIndex)} ${row.opName}`),
        element("small", "", `${formatUs(row.steadyStateUs)} steady${row.packingUs + row.boundaryUs > 0 ? ` / +${formatUs(row.packingUs + row.boundaryUs)} cold setup` : ""}`),
      );
      const outer = element("span", "glance-hotspot-scale");
      const inner = element("span", "glance-hotspot-track");
      inner.style.width = `${row.steadyStateUs / maximum * 100}%`;
      const components = [
        { id: "compute", label: "Compute", value: row.activeComputeUs },
        { id: "memory", label: "Memory", value: row.activeMemoryUs },
        { id: "mixed", label: "Roofline tie", value: row.mixedRooflineUs },
        { id: "fallback", label: "Fallback", value: row.fallbackUs },
        { id: "other", label: "Other", value: row.otherUs },
      ].filter((component) => component.value > 0);
      inner.append(...compositionBar(components, row.steadyStateUs, "compact").children);
      outer.append(inner);
      item.append(identity, outer);
      hotspot.append(item);
    }
    body.append(hotspot);
  }
  return body;
}

function cacheView(glance, boundL1Bytes, cacheOverrideBytes, setCacheOverride, onSelectOp) {
  const body = element("div", "glance-cache-view");
  const toolbar = element("div", "glance-cache-toolbar");
  toolbar.append(element("strong", "", "L1D denominator"));
  const bound = button(`Bound ${formatBytes(boundL1Bytes)}`, cacheOverrideBytes == null ? "active" : "");
  bound.dataset.cacheBytes = "bound";
  bound.addEventListener("click", () => setCacheOverride(null));
  toolbar.append(bound);
  for (const option of CACHE_OPTIONS) {
    const choice = button(option.label, cacheOverrideBytes === option.bytes ? "active" : "");
    choice.dataset.cacheBytes = String(option.bytes);
    choice.addEventListener("click", () => setCacheOverride(option.bytes));
    toolbar.append(choice);
  }
  const state = element(
    "span",
    cacheOverrideBytes == null ? "glance-cache-state bound" : "glance-cache-state what-if",
    cacheOverrideBytes == null ? "PROFILE-BOUND" : "UNBOUND WHAT-IF",
  );
  toolbar.append(state);
  body.append(toolbar);

  const cache = glance.memory.cache;
  body.append(metricRow(
    "Cache pressure",
    `Max L1 ${ratioText(cache.l1.maxRatio)} / ${cache.l1.watchCount} watch; ${cache.l2.assessmentStatus === "assessed" ? `max L2 ${ratioText(cache.l2.maxRatio)} / ${cache.l2.watchCount} watch` : "L2 watch N/A"}`,
    `${cache.assessedOpCount} ops assessed against ${formatBytes(cache.l1.bytes)} L1D and ${formatBytes(cache.l2.bytes)} L2`,
    `DERIVED FROM ${glance.memory.performanceModelEvidenceClass} PROFILE`,
  ));
  const note = element(
    "p",
    "glance-boundary-note",
    cacheOverrideBytes == null
      ? [
          glance.memory.cacheAssumption || "Cache topology was not observed from the executing device.",
          /shared/i.test(glance.memory.cacheAssumption || "")
            ? "Shared L2 is not divided into a fabricated per-thread capacity; active thread count, co-runners, and cache occupancy are unbound."
            : null,
          glance.memory.hardwareSpec
            ? `${glance.memory.hardwareSpec.evidence_class || "SOURCE_BACKED"}: ${glance.memory.hardwareSpec.sources?.map((source) => `${source.document} ${source.revision}; pages ${source.pages}; ${source.sha256 ? `SHA-256 ${source.sha256}` : "content digest not embedded"}`).join(" / ") || "source record embedded"}.`
            : null,
        ].filter(Boolean).join(" ")
      : "Viewer-only denominator. It does not mutate the target profile, report findings, profile hash, or modeled latency.",
  );
  body.append(note);

  if (!cache.rows.length) {
    body.append(element("p", "glance-empty", "No row working set was assessable."));
    return body;
  }
  const maximumRatio = Math.max(1.15, ...cache.rows.map((row) => Number(row.l1Ratio || 0)));
  const chart = element("div", "glance-cache-chart");
  const axis = element("div", "glance-cache-axis");
  const limit = element("i");
  limit.style.left = `${100 / maximumRatio}%`;
  limit.append(element("span", "", "1.00x L1D"));
  axis.append(limit);
  chart.append(axis);
  for (const row of cache.rows.slice(0, 12)) {
    const item = button("", `glance-cache-row${Number(row.l1Ratio || 0) >= 1 ? " exceeds" : Number(row.l1Ratio || 0) >= CACHE_WATCH_RATIO ? " watch" : ""}`);
    item.addEventListener("click", () => onSelectOp?.(row.opIndex));
    const identity = element("span", "glance-cache-id");
    identity.append(
      element("strong", "", `#${padOp(row.opIndex)} ${row.opName}`),
      element("small", "", `${formatBytes(row.rowWorkingSetBytes)} / ${ratioText(row.l1Ratio)}`),
    );
    const track = element("span", "glance-cache-track");
    const fill = element("i");
    fill.style.width = `${Math.min(100, Number(row.l1Ratio || 0) / maximumRatio * 100)}%`;
    track.append(fill);
    item.append(identity, track);
    chart.append(item);
  }
  body.append(chart);
  return body;
}

function actionCard(title, impact, {
  detail = "",
  evidence = "",
  effort = "",
  action = null,
} = {}) {
  const card = element("article", "glance-action-card");
  const head = element("div", "glance-action-head");
  head.append(element("strong", "", title), element("span", "", impact));
  card.append(head);
  if (detail) card.append(element("p", "", detail));
  const meta = element("div", "glance-action-meta");
  if (evidence) meta.append(element("em", "", evidence));
  if (effort) meta.append(element("span", "", effort));
  if (action) meta.append(action);
  card.append(meta);
  return card;
}

function actionsView(glance, analysis, scenario, onPreviewScenario, onClearScenario, onOpenCache) {
  const body = element("div", "glance-actions-view");
  const repair = analysis?.delegation_repair;
  const islands = [...(repair?.cpu_islands || [])].sort((left, right) => left.portfolio_rank - right.portfolio_rank);
  const best = islands[0];
  if (best) {
    const motif = glance.delegation.motifAttribution;
    const motifDetail = motif?.summary
      ? ` Motif attribution: ${motif.summary} Runtime assignment and actual boundary materialization remain unobserved.`
      : " Runtime support remains unobserved.";
    const active = scenario?.type === "delegation-repair"
      && scenario.targetProfileSha256 === analysis?.target_profile?.profile_sha256
      && scenario.islandIndex === best.island_index;
    const preview = button(active ? "Clear graph preview" : "Preview in graph", active ? "secondary-action" : "");
    preview.addEventListener("click", () => {
      if (active) {
        onClearScenario?.();
        return;
      }
      onPreviewScenario?.({
        type: "delegation-repair",
        artifactSha256: String(analysis?.model_sha256 || ""),
        targetProfileSha256: String(analysis?.target_profile?.profile_sha256 || ""),
        islandIndex: best.island_index,
        opIndices: [...best.op_indices],
        firstOpIndex: best.first_op_index,
        lastOpIndex: best.last_op_index,
        label: `CPU island ${best.island_index}: #${padOp(best.first_op_index)}-#${padOp(best.last_op_index)}`,
        edgeChanges: structuredClone(best.edge_changes || []),
        boundaryEdgeReductionCount: best.boundary_edge_reduction_count,
        boundaryPayloadReductionBytes: best.boundary_payload_reduction_bytes,
      });
    });
    body.append(actionCard(
      `Repair CPU island ${best.island_index}`,
      `${counted(best.boundary_edge_reduction_count, "boundary", "boundaries")} / ${formatBytes(best.boundary_payload_reduction_bytes)} removable`,
      {
        detail: `Complete counterfactual delegates ${counted(best.op_count, "op")}, from #${padOp(best.first_op_index)} to #${padOp(best.last_op_index)}.${motifDetail}`,
        evidence: "DERIVED BOUNDARY COUNTERFACTUAL",
        effort: motif?.meanBreakCount > 0 && motif.meanBreakCount === motif.meanBreaksInSqueezeExcitationCount
          ? "Export/global-pooling review first; runtime/build validation second"
          : "Runtime/build dependent",
        action: preview,
      },
    ));
  }

  if (glance.memory.cache.l1.watchCount > 0) {
    const open = button("Open cache what-if", "secondary-action");
    open.addEventListener("click", onOpenCache);
    body.append(actionCard(
      "Validate L1 residency",
      `${counted(glance.memory.cache.l1.watchCount, "op")} at or above ${CACHE_WATCH_RATIO.toFixed(2)}x`,
      {
        detail: `Maximum row working set is ${ratioText(glance.memory.cache.l1.maxRatio)} of the bound ${formatBytes(glance.memory.cache.l1.bytes)} L1D reference. Latency effect is not inferred.`,
        evidence: "DERIVED RATIO",
        effort: "Low to inspect; device counters required",
        action: open,
      },
    ));
  }

  if (glance.quantization.quantizedTensorCount > 0 && glance.quantization.perChannelTensorCount === 0) {
    body.append(actionCard(
      "Evaluate per-channel weight quantization",
      "0 per-channel tensors",
      {
        detail: "A transformed artifact and accuracy validation are required before any quality or latency effect can be claimed.",
        evidence: "OBSERVED METADATA",
        effort: "Converter and validation work",
      },
    ));
  }

  if (!body.children.length) {
    body.append(element("p", "glance-empty", "No artifact-visible action has a quantified static counterfactual."));
  }
  return body;
}

export function createExplorerDecisionView({
  root,
  onSelectOp = () => {},
  onPreviewScenario = () => {},
  onClearScenario = () => {},
  getRuntimeEvidence = () => ({}),
  onOpenAuditTab = () => {},
} = {}) {
  let analysis = null;
  let activeView = "summary";
  let cacheOverrideBytes = null;
  let analysisKey = "";
  let scenario = null;

  function renderView() {
    if (!root || !analysis) return;
    const boundL1Bytes = Number(analysis?.target_profile?.l1_data_bytes || 0);
    const glance = buildModelAtGlance(analysis, {
      l1DataBytes: cacheOverrideBytes ?? boundL1Bytes,
      l2Bytes: analysis?.target_profile?.l2_bytes,
    });
    root.hidden = false;
    root.dataset.glanceSchema = glance.schema;
    root.dataset.l1WatchCount = String(glance.memory.cache.l1.watchCount);
    root.dataset.maxL1Ratio = glance.memory.cache.l1.maxRatio == null ? "" : String(glance.memory.cache.l1.maxRatio);
    root.dataset.cacheState = cacheOverrideBytes == null ? "profile_bound" : "unbound_what_if";
    root.dataset.latencyConservation = glance.latency.conservationStatus;
    root.replaceChildren();
    const head = element("div", "glance-head");
    const title = element("div");
    title.append(
      element("span", "", "Selected-target decision summary"),
      element("h3", "", "Target At A Glance"),
      element("p", "", `${analysis.filename || "Artifact"} / ${glance.target.label || glance.format.toUpperCase()}`),
    );
    const evidence = element("div", "glance-head-evidence");
    evidence.append(
      element("strong", "", glance.format.toUpperCase()),
      element("span", "", glance.target.id || "no target profile"),
    );
    head.append(title, evidence);

    const tabs = element("div", "glance-tabs");
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", "Model decision summary views");
    for (const [id, label] of [["summary", "Summary"], ["cache", "Cache"], ["actions", "Actions"]]) {
      const tab = button(label, id === activeView ? "active" : "");
      tab.dataset.glanceView = id;
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", String(id === activeView));
      tab.addEventListener("click", () => {
        activeView = id;
        if (id !== "cache") cacheOverrideBytes = null;
        renderView();
      });
      tabs.append(tab);
    }
    const content = activeView === "cache"
      ? cacheView(glance, boundL1Bytes, cacheOverrideBytes, (bytes) => {
        cacheOverrideBytes = bytes;
        renderView();
      }, onSelectOp)
      : activeView === "actions"
        ? actionsView(glance, analysis, scenario, onPreviewScenario, onClearScenario, () => {
          activeView = "cache";
          renderView();
        })
        : summaryView(glance, onSelectOp);
    const questions = renderExplorerQuestionEntry({
      analysis,
      glance,
      runtimeEvidence: getRuntimeEvidence() || {},
      onOpen: (item) => {
        if (item.id === "fallback" || item.id === "runtime") activeView = "summary";
        if (item.id === "memory") activeView = "cache";
        if (item.auditTab) onOpenAuditTab(item.auditTab);
        renderView();
      },
    });
    root.append(head, questions, tabs, content);
  }

  return {
    render(nextAnalysis, { activeScenario = null } = {}) {
      analysis = nextAnalysis || null;
      scenario = activeScenario;
      if (!analysis) {
        if (root) {
          root.hidden = true;
          root.replaceChildren();
        }
        analysisKey = "";
        cacheOverrideBytes = null;
        return;
      }
      const nextKey = `${analysis.model_sha256 || analysis.filename || ""}:${analysis.target_profile?.profile_sha256 || analysis.target_profile?.id || ""}`;
      if (analysisKey !== nextKey) {
        analysisKey = nextKey;
        activeView = "summary";
        cacheOverrideBytes = null;
      }
      renderView();
    },
    setScenario(nextScenario) {
      scenario = nextScenario;
      renderView();
    },
    setView(view) {
      if (!VIEWS.has(view)) return;
      activeView = view;
      if (view !== "cache") cacheOverrideBytes = null;
      renderView();
    },
  };
}
