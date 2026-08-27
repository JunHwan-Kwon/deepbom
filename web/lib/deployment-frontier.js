import { formatBytes, formatNumber, formatPercent, formatUs, padOp } from "./format.js";

export const DEPLOYMENT_FRONTIER_SCHEMA = "deepbom.deployment_frontier.v1.6";
export const ORT_EP_PORTABILITY_FRONTIER_SCHEMA = "deepbom.ort_ep_portability_frontier.v2";

const FRONTIER_VIEWS = new Set(["frontier", "levers", "divergence"]);
const DEFAULT_FRONTIER_ROW_COUNT = 12;
const COMPONENT_FIELDS = Object.freeze({
  packing: "packing_us",
  boundary: "boundary_us",
  fallback: "fallback_us",
});

export function createDeploymentFrontierController({
  root,
  status,
  summary,
  body,
  downloadButton,
  getContext,
  jumpToGraphOp,
  onDownload,
}) {
  let activeView = "frontier";
  let expandedRobustSet = false;
  let renderedArtifactKey = "";
  let activePairKey = "";

  root?.addEventListener("click", (event) => {
    const expand = event.target.closest("[data-frontier-expand]");
    if (expand) {
      expandedRobustSet = !expandedRobustSet;
      render(getContext().analysis);
      return;
    }
    const pair = event.target.closest("[data-frontier-pair]");
    if (pair) {
      activePairKey = pair.dataset.frontierPair;
      render(getContext().analysis);
      return;
    }
    const tab = event.target.closest("[data-frontier-view]");
    if (!tab || !FRONTIER_VIEWS.has(tab.dataset.frontierView)) return;
    activeView = tab.dataset.frontierView;
    syncTabs(root, activeView);
    render(getContext().analysis);
  });
  downloadButton?.addEventListener("click", () => {
    const analysis = getContext().analysis;
    const frontier = frontierForAnalysis(analysis);
    if (frontier) onDownload(frontier, analysis?.format === "onnx" ? "ort_ep_portability_frontier.json" : "deployment_frontier.json");
  });

  function render(analysis) {
    if (!root || !summary || !body || !status) return;
    const onnx = String(analysis?.format || "tflite").toLowerCase() === "onnx";
    const artifactKey = `${onnx ? "onnx" : "tflite"}:${analysis?.model_sha256 || "none"}`;
    if (artifactKey !== renderedArtifactKey) {
      renderedArtifactKey = artifactKey;
      expandedRobustSet = false;
      activePairKey = "";
    }
    root.dataset.frontierFormat = onnx ? "onnx" : "tflite";
    syncTabs(root, activeView);
    if (!analysis) {
      status.textContent = "not assessed";
      summary.replaceChildren();
      body.replaceChildren(emptyState("Run an audit to build the deployment frontier."));
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    try {
      if (onnx) renderOnnx(analysis);
      else renderTflite(analysis);
    } catch (error) {
      console.error("[deploymentFrontier]", error);
      status.textContent = "unavailable";
      summary.replaceChildren();
      body.replaceChildren(emptyState(`Frontier validation failed: ${error?.message || String(error)}`));
      if (downloadButton) downloadButton.disabled = true;
    }
  }

  function renderTflite(analysis) {
    const frontier = analysis.deployment_frontier;
    if (!frontier) {
      status.textContent = "not assessed";
      summary.replaceChildren();
      body.replaceChildren(emptyState("Cross-target estimates are unavailable for this artifact."));
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    validateDeploymentFrontier(frontier, analysis);
    status.textContent = `${frontier.target_count} targets / HEURISTIC COST MODEL`;
    if (downloadButton) downloadButton.disabled = false;
    const bestSteadyLever = [...(frontier.interventions || [])]
      .filter((item) => !["packing", "boundary"].includes(item.removed_component))
      .sort((left, right) => Number(right.max_recoverable_share || 0) - Number(left.max_recoverable_share || 0))[0];
    const packingLever = (frontier.interventions || [])
      .find((item) => item.removed_component === "packing");
    const boundarySetupLever = (frontier.interventions || [])
      .find((item) => item.removed_component === "boundary");
    const maximumLeverTarget = (lever) => [...(lever?.per_target || [])]
      .sort((left, right) => Number(right.recoverable_share || 0) - Number(left.recoverable_share || 0))[0];
    const packingTarget = maximumLeverTarget(packingLever);
    const packingTargetProfile = frontier.targets.find((target) => target.target_id === packingTarget?.target_id);
    const boundaryTarget = maximumLeverTarget(boundarySetupLever);
    const boundaryTargetProfile = frontier.targets.find((target) => target.target_id === boundaryTarget?.target_id);
    summary.replaceChildren(
      summaryMetric("Steady-state robust set", `${frontier.robust_coverage.selected_op_count} / ${frontier.op_count}`, "80% steady-state prefix union"),
      summaryMetric("Worst target coverage", formatPercent(frontier.robust_coverage.minimum_union_coverage), "union coverage"),
      summaryMetric(
        "Target divergence",
        frontier.target_divergence.mean_normalized_jensen_shannon_divergence.toFixed(3),
        `mean normalized JSD; max rank span ${Math.max(0, ...(frontier.ops || []).map((op) => Number(op.rank_span || 0)))}`,
      ),
      summaryMetric(
        "Largest steady lever",
        bestSteadyLever ? formatPercent(bestSteadyLever.max_recoverable_share) : "0%",
        bestSteadyLever ? `${bestSteadyLever.label}; steady-state heuristic` : "none",
      ),
      summaryMetric(
        "Cold-start packing lever",
        packingLever ? formatPercent(packingLever.max_recoverable_share) : "0%",
        packingTarget
          ? `${formatUs(packingTarget.recoverable_us)} / ${formatUs(packingTargetProfile?.cold_start_us || 0)} cold on ${shortTargetLabel(packingTargetProfile?.target_label || packingTarget.target_id)}; maximum across profiles`
          : "none",
      ),
      summaryMetric(
        "Cold-start setup lever",
        boundarySetupLever ? formatPercent(boundarySetupLever.max_recoverable_share) : "0%",
        boundaryTarget
          ? `${formatUs(boundaryTarget.recoverable_us)} / ${formatUs(boundaryTargetProfile?.cold_start_us || 0)} cold on ${shortTargetLabel(boundaryTargetProfile?.target_label || boundaryTarget.target_id)}; unmeasured profile constant`
          : "none",
      ),
    );
    if (activeView === "levers") body.replaceChildren(renderTfliteLevers(frontier));
    else if (activeView === "divergence") body.replaceChildren(renderTfliteDivergence(frontier));
    else body.replaceChildren(renderTfliteFrontier(frontier));
  }

  function renderOnnx(analysis) {
    const frontier = analysis.ort_ep_portability_frontier;
    if (!frontier) {
      status.textContent = "source portfolio not loaded";
      summary.replaceChildren();
      body.replaceChildren(emptyState("Run the source-backed ORT analysis to build the EP portability frontier."));
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    validateOrtEpPortabilityFrontier(frontier, analysis);
    const runtime = getContext().runtimeEvidence;
    const observed = Boolean(runtime);
    status.textContent = `${frontier.execution_provider_count} EP rules / ${observed ? "runtime bound" : "assignment unobserved"}`;
    if (downloadButton) downloadButton.disabled = false;
    summary.replaceChildren(
      summaryMetric("Narrowed common candidates", `${frontier.all_ep_artifact_precheck_candidate_op_count} / ${frontier.op_count}`, "after definite artifact exclusions"),
      summaryMetric("Narrowed assessed MACs", frontier.all_ep_artifact_precheck_candidate_mac_ratio == null ? "N/A" : formatPercent(frontier.all_ep_artifact_precheck_candidate_mac_ratio), "still not support or assignment"),
      summaryMetric("EP portfolio", String(frontier.execution_provider_count), "CPU / WebGPU / WebNN"),
      summaryMetric("Runtime assignment", observed ? "Observed" : "Unobserved", observed ? runtimeEvidenceLabel(runtime) : "source rules only"),
    );
    if (activeView === "levers") body.replaceChildren(renderOnnxGaps(frontier));
    else if (activeView === "divergence") body.replaceChildren(renderOnnxDivergence(frontier));
    else body.replaceChildren(renderOnnxFrontier(frontier));
  }

  return { render };

  function renderTfliteFrontier(frontier) {
    const wrap = document.createElement("div");
    wrap.className = "frontier-table-wrap";
    const cacheTable = document.createElement("table");
    cacheTable.className = "frontier-matrix frontier-cache-table";
    cacheTable.append(tableHead(["Target cache reference", "L1D / L2", "Max row-WS L1 / L2", "Watch L1 / L2"]));
    const cacheBody = document.createElement("tbody");
    for (const target of frontier.targets) {
      const row = document.createElement("tr");
      const label = cell("td", shortTargetLabel(target.target_label));
      label.title = [
        target.cache_assumption || "Planning cache assumption",
        `hardware ${target.hardware_spec_evidence_class || "HEURISTIC_PROFILE"}`,
        `performance ${target.performance_model_evidence_class || "HEURISTIC"}`,
      ].join(" | ");
      row.append(
        label,
        cell("td", `${formatBytes(target.l1_data_bytes)} / ${formatBytes(target.l2_bytes)}`),
        cell("td", `${Number(target.max_l1_ratio || 0).toFixed(2)}x / ${target.max_l2_ratio == null ? "N/A" : `${Number(target.max_l2_ratio).toFixed(2)}x`}`),
        cell("td", `${target.l1_watch_count} / ${target.l2_watch_count == null ? `N/A (${String(target.l2_capacity_scope || "scope unbound").replaceAll("_", " ")})` : target.l2_watch_count} (>=${Number(frontier.cache_watch_ratio || 0.9).toFixed(2)}x)`),
      );
      cacheBody.append(row);
    }
    cacheTable.append(cacheBody);
    const table = document.createElement("table");
    table.className = "frontier-matrix";
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    headRow.append(cell("th", "Op"));
    for (const target of frontier.targets) headRow.append(cell("th", shortTargetLabel(target.target_label)));
    headRow.append(cell("th", "Rank span"));
    head.append(headRow);
    const tableBody = document.createElement("tbody");
    const selected = (frontier.ops || []).filter((op) => op.in_robust_coverage_union);
    const visible = expandedRobustSet ? selected : selected.slice(0, DEFAULT_FRONTIER_ROW_COUNT);
    for (const op of visible) {
      const row = document.createElement("tr");
      row.className = "frontier-op-row";
      row.tabIndex = 0;
      row.append(opCell(op));
      for (const target of frontier.targets) {
        const estimate = op.target_estimates.find((item) => item.target_id === target.target_id);
        row.append(contributionCell(estimate));
      }
      row.append(cell("td", op.rank_span ? `${op.best_rank}-${op.worst_rank}` : `#${op.best_rank}`, op.rank_span ? "frontier-volatile" : ""));
      row.addEventListener("click", () => jumpToGraphOp(op.op_index));
      row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") jumpToGraphOp(op.op_index); });
      tableBody.append(row);
    }
    table.append(head, tableBody);
    if (selected.length > DEFAULT_FRONTIER_ROW_COUNT) {
      const expand = document.createElement("button");
      expand.type = "button";
      expand.className = "frontier-expand-control secondary-action";
      expand.dataset.frontierExpand = "true";
      expand.setAttribute("aria-expanded", String(expandedRobustSet));
      expand.textContent = expandedRobustSet
        ? `Show top ${DEFAULT_FRONTIER_ROW_COUNT}`
        : `Show all ${selected.length} robust hotspots`;
      wrap.append(cacheTable, table, expand);
    } else {
      wrap.append(cacheTable, table);
    }
    const footer = document.createElement("div");
    footer.className = "frontier-method-line";
    footer.textContent = `${frontier.robust_coverage.method} Evidence: ${frontier.evidence_class}.`;
    wrap.append(footer);
    return wrap;
  }

  function renderTfliteLevers(frontier) {
    const wrap = document.createElement("div");
    wrap.className = "frontier-levers";
    for (const intervention of frontier.interventions || []) {
      const row = document.createElement("section");
      row.className = "frontier-lever-row";
      const heading = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = intervention.label;
      const range = document.createElement("span");
      const coldAxis = ["packing", "boundary"].includes(intervention.removed_component);
      range.textContent = `${formatPercent(intervention.min_recoverable_share)}-${formatPercent(intervention.max_recoverable_share)} ${coldAxis ? "cold-start" : "steady-state heuristic"} contribution / up to ${intervention.max_upper_bound_speedup.toFixed(2)}x`;
      heading.append(title, range);
      const targets = document.createElement("div");
      targets.className = "frontier-lever-targets";
      for (const target of intervention.per_target || []) {
        const label = frontier.targets.find((item) => item.target_id === target.target_id)?.target_label || target.target_id;
        targets.append(labeledBar(shortTargetLabel(label), target.recoverable_share, `${formatUs(target.recoverable_us)} / ${target.upper_bound_speedup.toFixed(2)}x`));
      }
      const boundary = document.createElement("p");
      boundary.textContent = `${intervention.evidence_class}: ${intervention.interpretation_boundary}`;
      row.append(heading, targets, boundary);
      wrap.append(row);
    }
    const queue = document.createElement("div");
    queue.className = "frontier-evidence-queue";
    const queueTitle = document.createElement("strong");
    queueTitle.textContent = "Evidence upgrade queue";
    queue.append(queueTitle);
    for (const item of frontier.evidence_queue || []) {
      const line = document.createElement("button");
      line.type = "button";
      line.className = "frontier-evidence-row";
      line.append(
        cell("span", item.evidence_needed),
        cell("strong", `${item.decision_exposed_op_count} ops / peak ${formatPercent(item.max_single_target_contribution_share)}`),
      );
      line.title = item.evidence_boundary;
      if (item.decision_exposed_op_indices?.length) line.addEventListener("click", () => jumpToGraphOp(item.decision_exposed_op_indices[0]));
      queue.append(line);
    }
    wrap.append(queue);
    return wrap;
  }

  function renderTfliteDivergence(frontier) {
    const wrap = document.createElement("div");
    wrap.className = "frontier-divergence";
    const pairs = [...(frontier.target_divergence.pairs || [])];
    if (!pairs.length) return emptyState("No target pair was available for divergence attribution.");
    const defaultPair = [...pairs].sort((left, right) => right.normalized_jensen_shannon_divergence - left.normalized_jensen_shannon_divergence)[0];
    let selected = pairs.find((pair) => targetPairKey(pair) === activePairKey) || defaultPair;
    activePairKey = targetPairKey(selected);
    const pairSelector = document.createElement("div");
    pairSelector.className = "frontier-pair-selector";
    for (const pair of pairs) {
      const left = frontier.targets.find((item) => item.target_id === pair.left_target_id)?.target_label || pair.left_target_id;
      const right = frontier.targets.find((item) => item.target_id === pair.right_target_id)?.target_label || pair.right_target_id;
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.frontierPair = targetPairKey(pair);
      button.className = targetPairKey(pair) === activePairKey ? "active" : "";
      button.append(
        cell("strong", `${shortTargetLabel(left)} / ${shortTargetLabel(right)}`),
        cell("span", `JSD ${pair.normalized_jensen_shannon_divergence.toFixed(4)} / overlap ${formatPercent(pair.coverage_prefix_jaccard)}`),
      );
      pairSelector.append(button);
    }
    const selectedLeft = frontier.targets.find((item) => item.target_id === selected.left_target_id)?.target_label || selected.left_target_id;
    const selectedRight = frontier.targets.find((item) => item.target_id === selected.right_target_id)?.target_label || selected.right_target_id;
    const context = document.createElement("div");
    context.className = "frontier-driver-context";
    context.append(
      summaryMetric("Attribution pair", `${shortTargetLabel(selectedLeft)} / ${shortTargetLabel(selectedRight)}`, `normalized JSD ${selected.normalized_jensen_shannon_divergence.toFixed(4)}`),
      summaryMetric("80% explanation", `${selected.attribution_prefix_op_count} ops`, `${formatPercent(selected.attribution_prefix_coverage)} of pair divergence`),
      summaryMetric("Top driver", selected.top_driver_op_index == null ? "none" : `#${padOp(selected.top_driver_op_index)} ${selected.top_driver_op_name}`, formatPercent(selected.top_driver_attribution_share)),
      summaryMetric("Transitions", `${selected.bound_transition_op_count} bound / ${selected.dominant_component_transition_op_count} component`, "all graph ops"),
    );
    const tableWrap = document.createElement("div");
    tableWrap.className = "frontier-table-wrap";
    const table = document.createElement("table");
    table.className = "frontier-driver-table";
    table.append(tableHead(["Op", "JSD attribution", "Modeled contribution", "Rank", "Largest component shift"]));
    const tableBody = document.createElement("tbody");
    for (const driver of (selected.drivers || []).filter((item) => item.normalized_js_contribution > 0).slice(0, 12)) {
      const row = document.createElement("tr");
      row.className = "frontier-op-row";
      row.append(
        opCell(driver),
        cell("td", formatPercent(driver.attribution_share)),
        cell("td", `${formatPercent(driver.left_contribution_share)} -> ${formatPercent(driver.right_contribution_share)}`),
        cell("td", `#${driver.left_rank} -> #${driver.right_rank}`),
        driverShiftCell(driver),
      );
      row.addEventListener("click", () => jumpToGraphOp(driver.op_index));
      tableBody.append(row);
    }
    table.append(tableBody);
    tableWrap.append(table);
    wrap.append(
      pairSelector,
      context,
      tableWrap,
      methodDetails(`Attribution conservation: ${selected.attribution_sum.toFixed(8)} = normalized JSD ${selected.normalized_jensen_shannon_divergence.toFixed(8)}. ${frontier.target_divergence.method}`),
    );
    return wrap;
  }

  function renderOnnxFrontier(frontier) {
    const wrap = document.createElement("div");
    wrap.className = "frontier-table-wrap";
    const table = document.createElement("table");
    table.className = "frontier-matrix frontier-onnx-matrix";
    table.append(tableHead(["Op", ...(frontier.providers || []).map((provider) => provider.execution_provider), "Assessed MACs"]));
    const tableBody = document.createElement("tbody");
    const rows = [...(frontier.ops || [])].sort((left, right) => (right.assessed_macs ?? -1) - (left.assessed_macs ?? -1) || left.op_index - right.op_index);
    for (const op of rows) {
      const row = document.createElement("tr");
      row.className = "frontier-op-row";
      row.append(opCell(op));
      for (const provider of frontier.providers || []) {
        const ep = provider.execution_provider;
        const sourceMatch = op.source_match_eps.includes(ep);
        const excluded = op.artifact_precheck_definite_exclusion_eps.includes(ep);
        const unresolved = op.artifact_precheck_unresolved_eps.includes(ep);
        const label = !sourceMatch ? "SOURCE GAP" : excluded ? "EXCLUDED" : unresolved ? "UNRESOLVED" : "CANDIDATE";
        row.append(cell("td", label, !sourceMatch || excluded ? "frontier-gap" : unresolved ? "frontier-volatile" : "frontier-match"));
      }
      row.append(cell("td", op.assessed_macs == null ? "N/A" : formatNumber(op.assessed_macs)));
      row.addEventListener("click", () => jumpToGraphOp(op.op_index));
      tableBody.append(row);
    }
    table.append(tableBody);
    wrap.append(table, methodLine(frontier.evidence_boundary));
    return wrap;
  }

  function renderOnnxGaps(frontier) {
    const wrap = document.createElement("div");
    wrap.className = "frontier-onnx-gaps";
    for (const provider of frontier.providers || []) {
      const section = document.createElement("section");
      const heading = document.createElement("div");
      heading.className = "frontier-provider-heading";
      heading.append(
        cell("strong", provider.execution_provider),
        cell("span", `${provider.artifact_precheck_candidate_op_count}/${frontier.op_count} narrowed candidates / ${provider.artifact_precheck_candidate_assessed_mac_ratio == null ? "MAC N/A" : `${formatPercent(provider.artifact_precheck_candidate_assessed_mac_ratio)} assessed MACs`} / ${provider.artifact_precheck_definite_fail_op_count} definite exclusion(s)`),
      );
      const list = document.createElement("div");
      list.className = "frontier-gap-list";
      if (!provider.top_gap_ops.length) list.append(emptyState("No source-version gaps or definite artifact-condition exclusions."));
      for (const gap of provider.top_gap_ops) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "frontier-gap-row";
        button.append(
          cell("strong", `#${padOp(gap.op_index)} ${gap.op_name}`),
          cell("span", `${gap.gap_class}: ${gap.status}`),
          cell("span", gap.assessed_macs == null ? "MAC N/A" : formatNumber(gap.assessed_macs)),
        );
        button.addEventListener("click", () => jumpToGraphOp(gap.op_index));
        list.append(button);
      }
      section.append(heading, list);
      wrap.append(section);
    }
    return wrap;
  }

  function renderOnnxDivergence(frontier) {
    const wrap = document.createElement("div");
    wrap.className = "frontier-divergence";
    for (const pair of frontier.provider_pairs || []) {
      wrap.append(labeledBar(
        `${pair.left_execution_provider} / ${pair.right_execution_provider}`,
        pair.artifact_precheck_candidate_set_jaccard,
        `narrowed candidate overlap ${formatPercent(pair.artifact_precheck_candidate_set_jaccard)} / source-only ${formatPercent(pair.source_match_set_jaccard)}`,
      ));
    }
    wrap.append(methodLine(frontier.method), methodLine(frontier.evidence_boundary));
    return wrap;
  }
}

export function validateDeploymentFrontier(frontier, analysis) {
  if (!frontier || frontier.schema !== DEPLOYMENT_FRONTIER_SCHEMA) throw new Error("Deployment frontier schema is unsupported.");
  if (frontier.artifact_sha256 !== analysis.model_sha256) throw new Error(`Deployment frontier artifact SHA binding is invalid (${frontier.artifact_sha256 || "missing"} != ${analysis.model_sha256 || "missing"}; keys=${Object.keys(frontier).slice(0, 8).join(",") || "none"}).`);
  if (frontier.op_count !== analysis.ops.length) throw new Error(`Deployment frontier op-count binding is invalid (${frontier.op_count} != ${analysis.ops.length}).`);
  if (!Array.isArray(frontier.targets) || frontier.targets.length !== frontier.target_count || frontier.target_count < 2) throw new Error("Deployment frontier target coverage is invalid.");
  const targetIds = frontier.targets.map((target) => target.target_id);
  if (new Set(targetIds).size !== targetIds.length || frontier.targets.some((target) => !/^[a-f0-9]{64}$/.test(target.target_profile_sha256 || ""))) throw new Error("Deployment frontier target identity is invalid.");
  for (const target of frontier.targets) {
    const l2Assessed = String(target.l2_capacity_scope || "").startsWith("private_per_core");
    if (!(target.effective_peak_gops > 0)
      || !(target.effective_memory_bandwidth_gbps > 0)
      || !(target.weight_packing_bandwidth_gbps > 0)
      || !(target.compute_utilization_factor > 0 && target.compute_utilization_factor <= 1)
      || !(target.l1_data_bytes > 0) || !(target.l2_bytes > 0)
      || !Number.isInteger(target.l1_watch_count)
      || !(target.max_l1_ratio >= 0)
      || (l2Assessed && (!Number.isInteger(target.l2_watch_count) || !(target.max_l2_ratio >= 0)))
      || (!l2Assessed && (target.l2_watch_count != null || target.max_l2_ratio != null))) throw new Error(`Deployment frontier profile-constant or cache ledger is invalid for ${target.target_id}.`);
    assertNear(target.total_us, componentTotal(target.components), `target ${target.target_id} total`);
    assertNear(target.cold_start_us, target.total_us, `target ${target.target_id} cold-start alias`);
    assertNear(target.steady_state_us, target.total_us
      - Number(target.components?.packing_us || 0)
      - Number(target.components?.boundary_us || 0), `target ${target.target_id} steady total`);
    assertNear(target.boundary_setup_midpoint_us, Number(target.components?.boundary_us || 0), `target ${target.target_id} setup midpoint`);
    if (!(target.boundary_setup_low_us >= 0
      && target.boundary_setup_low_us <= target.boundary_setup_midpoint_us
      && target.boundary_setup_midpoint_us <= target.boundary_setup_high_us)) throw new Error(`Deployment frontier setup range is invalid for ${target.target_id}.`);
    if (!(target.steady_state_low_us <= target.steady_state_us
      && target.steady_state_us <= target.steady_state_high_us
      && target.cold_start_low_us <= target.cold_start_us
      && target.cold_start_us <= target.cold_start_high_us)) throw new Error(`Deployment frontier latency range is invalid for ${target.target_id}.`);
    assertNear(target.cold_start_low_us, target.steady_state_low_us
      + Number(target.components?.packing_us || 0)
      + Number(target.boundary_setup_low_us || 0), `target ${target.target_id} low cold-start conservation`);
    assertNear(target.cold_start_us, target.steady_state_us
      + Number(target.components?.packing_us || 0)
      + Number(target.boundary_setup_midpoint_us || 0), `target ${target.target_id} point cold-start conservation`);
    assertNear(target.cold_start_high_us, target.steady_state_high_us
      + Number(target.components?.packing_us || 0)
      + Number(target.boundary_setup_high_us || 0), `target ${target.target_id} high cold-start conservation`);
  }
  if (!(frontier.cache_watch_ratio > 0 && frontier.cache_watch_ratio <= 1)) throw new Error("Deployment frontier cache-watch threshold is invalid.");
  const selectedTarget = frontier.targets.find((target) => target.target_id === analysis.target_profile?.id);
  if (!selectedTarget) throw new Error("Deployment frontier selected target is unbound.");
  if (selectedTarget.l1_data_bytes !== analysis.target_profile?.l1_data_bytes || selectedTarget.l2_bytes !== analysis.target_profile?.l2_bytes
    || selectedTarget.l1_watch_count !== Number(analysis.insights?.l1_watch_count || 0)) throw new Error("Deployment frontier selected-target cache binding is invalid.");
  const selectedBottleneckTotal = (analysis.ops || []).reduce((sum, op) => sum + Number(op.bottleneck_total_us || 0), 0);
  assertNear(selectedTarget.total_us, selectedBottleneckTotal, "selected target bottleneck-total binding");
  if (!Array.isArray(frontier.ops) || frontier.ops.length !== analysis.ops.length) throw new Error("Deployment frontier op coverage is invalid.");
  const distributions = new Map(targetIds.map((id) => [id, []]));
  const estimatesByTarget = new Map(targetIds.map((id) => [id, new Map()]));
  for (const row of frontier.ops) {
    const op = analysis.ops.find((item) => Number(item.index) === Number(row.op_index));
    if (!op || row.op_name !== op.name || row.target_estimates.length !== targetIds.length) throw new Error(`Deployment frontier op identity is invalid at #${row.op_index}.`);
    for (const estimate of row.target_estimates) {
      const target = frontier.targets.find((item) => item.target_id === estimate.target_id);
      if (!target) throw new Error(`Deployment frontier target estimate is unbound at #${row.op_index}.`);
      assertNear(estimate.total_us, componentTotal(estimate.components), `op #${row.op_index} ${estimate.target_id} total`);
      assertNear(estimate.cold_start_us, estimate.total_us, `op #${row.op_index} ${estimate.target_id} cold-start alias`);
      assertNear(estimate.steady_state_us, estimate.total_us
        - Number(estimate.components?.packing_us || 0)
        - Number(estimate.components?.boundary_us || 0), `op #${row.op_index} ${estimate.target_id} steady total`);
      if (estimate.target_id === analysis.target_profile?.id) {
        assertNear(estimate.cold_start_us, Number(op.bottleneck_total_us || 0), `op #${row.op_index} selected-target bottleneck binding`);
        assertNear(estimate.steady_state_us, Math.max(
          0,
          Number(op.bottleneck_total_us || 0)
            - Number(op.bottleneck_packing_us || 0)
            - Number(op.bottleneck_break_us || 0),
        ), `op #${row.op_index} selected-target steady binding`);
      }
      assertNear(estimate.contribution_share, target.steady_state_us > 0 ? estimate.steady_state_us / target.steady_state_us : 0, `op #${row.op_index} ${estimate.target_id} share`);
      distributions.get(estimate.target_id).push(estimate.contribution_share);
      estimatesByTarget.get(estimate.target_id).set(row.op_index, estimate);
    }
  }
  for (const [targetId, distribution] of distributions) {
    const target = frontier.targets.find((item) => item.target_id === targetId);
    assertNear(distribution.reduce((sum, value) => sum + value, 0), target.steady_state_us > 0 ? 1 : 0, `${targetId} distribution`);
  }
  const selected = new Set(frontier.robust_coverage.selected_op_indices || []);
  const flagged = new Set(frontier.ops.filter((op) => op.in_robust_coverage_union).map((op) => op.op_index));
  if (JSON.stringify([...selected].sort((a, b) => a - b)) !== JSON.stringify([...flagged].sort((a, b) => a - b))) throw new Error("Deployment frontier robust set flags disagree.");
  const minimumCoverage = Math.min(...frontier.robust_coverage.per_target.map((target) => target.union_coverage));
  assertNear(frontier.robust_coverage.minimum_union_coverage, minimumCoverage, "deployment frontier minimum union coverage");
  if (minimumCoverage + 1e-12 < frontier.robust_coverage.threshold) throw new Error("Deployment frontier robust set does not reach its declared threshold.");
  const pairCount = frontier.target_count * (frontier.target_count - 1) / 2;
  if (frontier.target_divergence.pairs.length !== pairCount) throw new Error("Deployment frontier target-pair coverage is incomplete.");
  for (const pair of frontier.target_divergence.pairs) {
    const expected = normalizedJsDivergence(distributions.get(pair.left_target_id), distributions.get(pair.right_target_id));
    assertNear(pair.normalized_jensen_shannon_divergence, expected, "deployment frontier normalized JSD");
    const leftTarget = frontier.targets.find((target) => target.target_id === pair.left_target_id);
    const rightTarget = frontier.targets.find((target) => target.target_id === pair.right_target_id);
    if (!leftTarget || !rightTarget || !Array.isArray(pair.drivers) || pair.drivers.length !== frontier.op_count) throw new Error("Deployment frontier divergence attribution coverage is invalid.");
    const expectedDrivers = frontier.ops.map((op) => {
      const left = estimatesByTarget.get(pair.left_target_id).get(op.op_index);
      const right = estimatesByTarget.get(pair.right_target_id).get(op.op_index);
      const contribution = normalizedJsContribution(left.contribution_share, right.contribution_share);
      return {
        op,
        left,
        right,
        contribution,
        absoluteDelta: Math.abs(right.contribution_share - left.contribution_share),
        componentDelta: componentContributionDelta(left, right, leftTarget, rightTarget),
      };
    }).sort((left, right) => right.contribution - left.contribution || right.absoluteDelta - left.absoluteDelta || left.op.op_index - right.op.op_index);
    let attributionSum = 0;
    let boundTransitions = 0;
    let componentTransitions = 0;
    for (let index = 0; index < expectedDrivers.length; index += 1) {
      const actual = pair.drivers[index];
      const item = expectedDrivers[index];
      if (actual.op_index !== item.op.op_index || actual.op_name !== item.op.op_name) throw new Error(`Deployment frontier divergence driver order is invalid at pair ${pair.left_target_id}/${pair.right_target_id}.`);
      assertNear(actual.normalized_js_contribution, item.contribution, `op #${actual.op_index} JSD contribution`);
      assertNear(actual.attribution_share, expected > 0 ? item.contribution / expected : 0, `op #${actual.op_index} JSD attribution share`);
      assertNear(actual.left_contribution_share, item.left.contribution_share, `op #${actual.op_index} left contribution`);
      assertNear(actual.right_contribution_share, item.right.contribution_share, `op #${actual.op_index} right contribution`);
      assertNear(actual.signed_contribution_share_delta, item.right.contribution_share - item.left.contribution_share, `op #${actual.op_index} signed contribution delta`);
      assertNear(actual.absolute_contribution_share_delta, item.absoluteDelta, `op #${actual.op_index} absolute contribution delta`);
      if (actual.left_rank !== item.left.rank || actual.right_rank !== item.right.rank || actual.rank_delta !== Math.abs(item.left.rank - item.right.rank)) throw new Error(`Deployment frontier divergence rank attribution is invalid at #${actual.op_index}.`);
      if (actual.left_bound !== item.left.bound || actual.right_bound !== item.right.bound || actual.bound_transition !== (item.left.bound !== item.right.bound)) throw new Error(`Deployment frontier bound transition is invalid at #${actual.op_index}.`);
      if (actual.left_dominant_component !== item.left.dominant_component || actual.right_dominant_component !== item.right.dominant_component || actual.dominant_component_transition !== (item.left.dominant_component !== item.right.dominant_component)) throw new Error(`Deployment frontier component transition is invalid at #${actual.op_index}.`);
      for (const field of ["compute", "memory", "packing", "boundary", "fallback", "largest_absolute_delta"]) assertNear(actual.component_contribution_delta[field], item.componentDelta[field], `op #${actual.op_index} ${field} component delta`);
      if (actual.component_contribution_delta.largest_absolute_component !== item.componentDelta.largest_absolute_component) throw new Error(`Deployment frontier largest component delta is invalid at #${actual.op_index}.`);
      attributionSum += item.contribution;
      if (actual.bound_transition) boundTransitions += 1;
      if (actual.dominant_component_transition) componentTransitions += 1;
    }
    assertNear(pair.attribution_sum, attributionSum, "deployment frontier JSD attribution sum");
    assertNear(pair.attribution_sum, expected, "deployment frontier JSD attribution conservation");
    if (pair.attribution_prefix_threshold !== 0.8) throw new Error("Deployment frontier divergence attribution threshold is invalid.");
    let prefixSum = 0;
    let prefixCount = 0;
    if (expected > 0) {
      for (const driver of expectedDrivers) {
        if (prefixSum / expected >= pair.attribution_prefix_threshold || driver.contribution <= 0) break;
        prefixSum += driver.contribution;
        prefixCount += 1;
      }
    }
    const prefixCoverage = expected > 0 ? prefixSum / expected : 1;
    if (pair.attribution_prefix_op_count !== prefixCount) throw new Error("Deployment frontier divergence attribution prefix count is invalid.");
    assertNear(pair.attribution_prefix_coverage, prefixCoverage, "deployment frontier divergence attribution prefix coverage");
    const top = expectedDrivers.find((driver) => driver.contribution > 0);
    if (pair.top_driver_op_index !== (top?.op.op_index ?? null) || pair.top_driver_op_name !== (top?.op.op_name ?? null)) throw new Error("Deployment frontier top divergence driver identity is invalid.");
    assertNear(pair.top_driver_attribution_share, top && expected > 0 ? top.contribution / expected : 0, "deployment frontier top divergence driver share");
    if (pair.bound_transition_op_count !== boundTransitions || pair.dominant_component_transition_op_count !== componentTransitions) throw new Error("Deployment frontier divergence transition counts are invalid.");
    const leftPrefixCount = frontier.robust_coverage.per_target.find((target) => target.target_id === pair.left_target_id)?.selected_prefix_op_count || 0;
    const rightPrefixCount = frontier.robust_coverage.per_target.find((target) => target.target_id === pair.right_target_id)?.selected_prefix_op_count || 0;
    const leftPrefix = new Set([...estimatesByTarget.get(pair.left_target_id)].filter(([, estimate]) => estimate.rank <= leftPrefixCount).map(([opIndex]) => opIndex));
    const rightPrefix = new Set([...estimatesByTarget.get(pair.right_target_id)].filter(([, estimate]) => estimate.rank <= rightPrefixCount).map(([opIndex]) => opIndex));
    assertNear(pair.coverage_prefix_jaccard, setJaccard(leftPrefix, rightPrefix), "deployment frontier pair prefix Jaccard");
  }
  for (const intervention of frontier.interventions || []) {
    const field = COMPONENT_FIELDS[intervention.removed_component];
    if (!field || intervention.evidence_class !== "ESTIMATED_COUNTERFACTUAL_UPPER_BOUND") throw new Error("Deployment frontier intervention identity is invalid.");
    for (const targetRow of intervention.per_target || []) {
      const target = frontier.targets.find((item) => item.target_id === targetRow.target_id);
      const recoverable = Number(target.components?.[field] || 0);
      assertNear(targetRow.recoverable_us, recoverable, `${intervention.id} recoverable time`);
      const denominator = ["packing", "boundary"].includes(intervention.removed_component)
        ? target.cold_start_us
        : target.steady_state_us;
      assertNear(targetRow.recoverable_share, denominator > 0 ? recoverable / denominator : 0, `${intervention.id} recoverable share`);
      const remaining = Math.max(0, denominator - recoverable);
      const speedup = denominator > 0 && remaining > 0 ? denominator / remaining : 1;
      assertNear(targetRow.upper_bound_speedup, speedup, `${intervention.id} upper-bound speedup`);
    }
  }
  return true;
}

export function validateOrtEpPortabilityFrontier(frontier, analysis) {
  if (!frontier || frontier.schema !== ORT_EP_PORTABILITY_FRONTIER_SCHEMA) throw new Error("ORT EP portability frontier schema is unsupported.");
  if (frontier.op_count !== analysis.ops.length || frontier.ops.length !== analysis.ops.length) throw new Error("ORT EP portability frontier op coverage is invalid.");
  if (frontier.execution_provider_count !== frontier.providers.length
    || frontier.evidence_class !== "DERIVED_FROM_PINNED_SOURCE_AND_ARTIFACT_VISIBLE_DEFINITE_EXCLUSIONS") throw new Error("ORT EP portability frontier provider coverage is invalid.");
  const providerIds = frontier.providers.map((provider) => provider.execution_provider);
  for (const row of frontier.ops) {
    const op = analysis.ops.find((item) => Number(item.index) === Number(row.op_index));
    if (!op || row.op_name !== op.name || row.source_match_ep_count !== row.source_match_eps.length) throw new Error(`ORT EP portability frontier row is invalid at #${row.op_index}.`);
    const partition = new Set([...(row.source_match_eps || []), ...(row.source_gap_eps || [])]);
    if (partition.size !== providerIds.length || providerIds.some((id) => !partition.has(id))) throw new Error(`ORT EP portability provider partition is invalid at #${row.op_index}.`);
    if (row.artifact_precheck_candidate_ep_count !== row.artifact_precheck_candidate_eps.length
      || row.artifact_precheck_candidate_eps.some((id) => !row.source_match_eps.includes(id))
      || row.artifact_precheck_definite_exclusion_eps.some((id) => !row.source_match_eps.includes(id))
      || row.artifact_precheck_unresolved_eps.some((id) => !row.artifact_precheck_candidate_eps.includes(id))) throw new Error(`ORT EP artifact-precheck partition is invalid at #${row.op_index}.`);
  }
  if (!String(frontier.evidence_boundary || "").includes("not support or assignment")) throw new Error("ORT EP portability evidence boundary is missing.");
  return true;
}

function frontierForAnalysis(analysis) {
  return String(analysis?.format || "tflite").toLowerCase() === "onnx"
    ? analysis?.ort_ep_portability_frontier
    : analysis?.deployment_frontier;
}

function componentTotal(components = {}) {
  return ["compute_us", "memory_us", "packing_us", "boundary_us", "fallback_us"]
    .reduce((sum, field) => sum + Number(components[field] || 0), 0);
}

function normalizedJsDivergence(left = [], right = []) {
  const value = left.reduce((sum, p, index) => sum + normalizedJsContribution(p, right[index] || 0), 0);
  return Math.min(1, Math.max(0, value));
}

function normalizedJsContribution(left, right) {
  const middle = (left + right) * 0.5;
  const leftTerm = left > 0 && middle > 0 ? left * Math.log(left / middle) : 0;
  const rightTerm = right > 0 && middle > 0 ? right * Math.log(right / middle) : 0;
  return Math.max(0, 0.5 * (leftTerm + rightTerm) / Math.LN2);
}

function componentContributionDelta(left, right, leftTarget, rightTarget) {
  const fields = [
    ["compute", "compute_us"],
    ["memory", "memory_us"],
    ["packing", "packing_us"],
    ["boundary", "boundary_us"],
    ["fallback", "fallback_us"],
  ];
  const result = {};
  let largestName = "none";
  let largestDelta = 0;
  for (const [name, field] of fields) {
    const delta = name === "packing" || name === "boundary"
      ? 0
      : Number(right.components?.[field] || 0) / Number(rightTarget.steady_state_us || 1)
        - Number(left.components?.[field] || 0) / Number(leftTarget.steady_state_us || 1);
    result[name] = delta;
    if (Math.abs(delta) >= largestDelta) {
      largestName = name;
      largestDelta = Math.abs(delta);
    }
  }
  result.largest_absolute_component = largestName;
  result.largest_absolute_delta = largestDelta;
  return result;
}

function setJaccard(left, right) {
  const union = new Set([...left, ...right]);
  if (!union.size) return 1;
  let intersection = 0;
  for (const value of left) if (right.has(value)) intersection += 1;
  return intersection / union.size;
}

function assertNear(actual, expected, label) {
  const tolerance = Math.max(1e-10, Math.abs(Number(expected)) * 1e-10);
  if (!Number.isFinite(Number(actual)) || Math.abs(Number(actual) - Number(expected)) > tolerance) throw new Error(`${label} invariant failed.`);
}

function syncTabs(root, activeView) {
  for (const button of root.querySelectorAll("[data-frontier-view]")) {
    const active = button.dataset.frontierView === activeView;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  }
}

function summaryMetric(label, value, detail) {
  const node = document.createElement("div");
  node.className = "frontier-summary-metric";
  node.append(cell("span", label), cell("strong", value), cell("small", detail));
  return node;
}

function tableHead(labels) {
  const head = document.createElement("thead");
  const row = document.createElement("tr");
  for (const label of labels) row.append(cell("th", label));
  head.append(row);
  return head;
}

function opCell(op) {
  const node = document.createElement("td");
  node.className = "frontier-op-cell";
  node.append(cell("strong", `#${padOp(op.op_index)} ${op.op_name}`));
  if (op.dominant_components?.length) node.append(cell("span", op.dominant_components.join(" / ")));
  return node;
}

function contributionCell(estimate) {
  if (!estimate) return cell("td", "N/A");
  const node = document.createElement("td");
  node.className = "frontier-contribution-cell";
  const track = document.createElement("i");
  track.style.setProperty("--frontier-share", `${Math.min(100, Math.max(0, estimate.contribution_share * 100))}%`);
  const value = document.createElement("strong");
  value.textContent = formatPercent(estimate.contribution_share);
  const detail = document.createElement("span");
  detail.textContent = `#${estimate.rank} / ${formatUs(estimate.steady_state_us)} steady`;
  node.append(track, value, detail);
  return node;
}

function driverShiftCell(driver) {
  const node = document.createElement("td");
  node.className = "frontier-driver-shift";
  const component = driver.component_contribution_delta?.largest_absolute_component || "none";
  const delta = Number(driver.component_contribution_delta?.[component] || 0);
  const primary = document.createElement("strong");
  primary.textContent = `${component} ${formatSignedPercentagePoints(delta)}`;
  const detail = document.createElement("span");
  const transitions = [];
  if (driver.bound_transition) transitions.push(`${driver.left_bound} -> ${driver.right_bound}`);
  if (driver.dominant_component_transition) transitions.push(`${driver.left_dominant_component} -> ${driver.right_dominant_component}`);
  detail.textContent = transitions.join(" / ") || "classification stable";
  node.append(primary, detail);
  return node;
}

function formatSignedPercentagePoints(value) {
  const points = Number(value || 0) * 100;
  return `${points > 0 ? "+" : ""}${points.toFixed(2)} pp`;
}

function targetPairKey(pair) {
  return `${pair.left_target_id}::${pair.right_target_id}`;
}

function labeledBar(label, ratio, detail) {
  const row = document.createElement("div");
  row.className = "frontier-bar-row";
  const head = document.createElement("div");
  head.append(cell("strong", label), cell("span", detail));
  const track = document.createElement("div");
  track.className = "frontier-bar-track";
  const fill = document.createElement("i");
  fill.style.width = `${Math.min(100, Math.max(0, Number(ratio || 0) * 100))}%`;
  track.append(fill);
  row.append(head, track);
  return row;
}

function methodLine(text) {
  const node = document.createElement("p");
  node.className = "frontier-method-line";
  node.textContent = text;
  return node;
}

function methodDetails(text) {
  const details = document.createElement("details");
  details.className = "frontier-method-details";
  const summary = document.createElement("summary");
  summary.textContent = "Method and evidence";
  details.append(summary, methodLine(text));
  return details;
}

function emptyState(text) {
  const node = document.createElement("div");
  node.className = "visual-empty";
  node.textContent = text;
  return node;
}

function cell(tag, text, className = "") {
  const node = document.createElement(tag);
  node.textContent = String(text ?? "");
  if (className) node.className = className;
  return node;
}

function shortTargetLabel(label) {
  return String(label || "Target")
    .replace("Android mid-range / ", "")
    .replace("Raspberry Pi 4 / ", "")
    .replace("Browser / ", "")
    .replace("Desktop / ", "");
}

function runtimeEvidenceLabel(runtime) {
  const native = runtime?.source?.adapter?.native_capture_evidence;
  if (native) return "pinned native capture";
  return runtime?.source?.adapter?.schema ? "bound imported profile" : "canonical assignment";
}
