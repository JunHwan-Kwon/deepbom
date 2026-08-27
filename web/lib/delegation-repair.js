import { formatBytes, formatNumber, padOp } from "./format.js";

export const DELEGATION_REPAIR_SCHEMA = "deepbom.delegation_repair.v1.3";

const VIEWS = new Set(["portfolio", "repair", "islands", "fragility", "edges"]);
const DTYPE_BYTES = Object.freeze({
  BOOL: 1, INT8: 1, UINT8: 1,
  FLOAT16: 2, BFLOAT16: 2, INT16: 2, UINT16: 2,
  FLOAT32: 4, INT32: 4, UINT32: 4,
  FLOAT64: 8, INT64: 8, UINT64: 8,
  COMPLEX64: 8, COMPLEX128: 16,
});

export function createDelegationRepairController({
  root,
  status,
  summary,
  body,
  downloadButton,
  getAnalysis,
  jumpToGraphOp,
  onPreviewScenario,
  onDownload,
}) {
  let activeView = "repair";
  let selectedOpIndex = null;
  let selectedIslandIndex = null;
  let renderedKey = "";

  root?.addEventListener("click", (event) => {
    const view = event.target.closest("[data-repair-view]");
    if (view && VIEWS.has(view.dataset.repairView)) {
      activeView = view.dataset.repairView;
      render(getAnalysis());
      return;
    }
    const island = event.target.closest("[data-repair-island]");
    if (island) {
      selectedIslandIndex = Number(island.dataset.repairIsland);
      activeView = "islands";
      render(getAnalysis());
      return;
    }
    const row = event.target.closest("[data-repair-op]");
    if (!row) return;
    selectedOpIndex = Number(row.dataset.repairOp);
    if (event.target.closest("[data-repair-open-edges]")) activeView = "edges";
    render(getAnalysis());
  });
  root?.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const island = event.target.closest("[data-repair-island]");
    if (island) {
      event.preventDefault();
      selectedIslandIndex = Number(island.dataset.repairIsland);
      activeView = "islands";
      render(getAnalysis());
      return;
    }
    const row = event.target.closest("[data-repair-op]");
    if (!row) return;
    event.preventDefault();
    selectedOpIndex = Number(row.dataset.repairOp);
    activeView = "edges";
    render(getAnalysis());
  });
  downloadButton?.addEventListener("click", () => {
    const result = getAnalysis()?.delegation_repair;
    if (result) onDownload?.(result, "delegation_repair.json");
  });

  function render(analysis) {
    if (!root || !status || !summary || !body) return;
    const result = analysis?.delegation_repair;
    if (!result) {
      renderedKey = "";
      selectedOpIndex = null;
      selectedIslandIndex = null;
      status.textContent = analysis?.delegation_repair_error ? "not assessed" : "run an audit";
      summary.replaceChildren();
      body.replaceChildren(emptyState(analysis?.delegation_repair_error || "Delegation counterfactuals are unavailable."));
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    validateDelegationRepair(result, analysis);
    const key = `${result.artifact_sha256}:${result.target_profile_sha256}`;
    if (key !== renderedKey) {
      renderedKey = key;
      selectedOpIndex = result.repair_ranking_op_indices[0]
        ?? result.fragility_ranking_op_indices[0]
        ?? result.toggles[0]?.op_index
        ?? null;
      selectedIslandIndex = result.cpu_island_ranking_indices?.[0] ?? null;
      activeView = (result.export_interventions?.length || result.runtime_build_risks?.length) ? "portfolio" : "repair";
    }
    status.textContent = `${result.repair_opportunity_count} repair / ${result.group_only_repair_count} group-only / ${result.fragmentation_risk_count} fragility`;
    if (downloadButton) downloadButton.disabled = false;
    summary.replaceChildren(
      metric("Predicted delegate segments", formatNumber(result.baseline.delegate_segment_count), result.target_label),
      metric("Boundary edges", formatNumber(result.baseline.boundary_edge_count), `${result.graph_edge_count} graph edges assessed`),
      metric("Boundary payload", optionalBytes(result.baseline.summed_edge_payload_bytes), `${result.baseline.assessed_payload_edge_count}/${result.baseline.boundary_edge_count} edge payloads`),
      metric("Actionable scenarios", `${formatNumber(result.export_interventions?.length || 0)} export / ${formatNumber(result.runtime_build_risks?.length || 0)} build / ${formatNumber(result.singleton_delegate_segments?.length || 0)} runtime review`, "Export/build interventions plus singleton-segment measurement reviews"),
    );
    body.replaceChildren(renderTabs(), renderActive(result, analysis));
  }

  function renderTabs() {
    const tabs = document.createElement("div");
    tabs.className = "repair-tabs";
    tabs.setAttribute("role", "tablist");
    for (const [id, label] of [["portfolio", "Actionable scenarios"], ["repair", "Merge candidates"], ["islands", "CPU islands"], ["fragility", "Fragmentation"], ["edges", "Edge ledger"]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.repairView = id;
      button.className = id === activeView ? "active" : "";
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", String(id === activeView));
      button.textContent = label;
      tabs.append(button);
    }
    return tabs;
  }

  function renderActive(result, analysis) {
    if (activeView === "portfolio") return renderActionableScenarios(result);
    if (activeView === "islands") return renderCpuIslands(result);
    if (activeView === "fragility") return renderFragilityPortfolio(result, analysis);
    if (activeView === "edges") return renderEdgeLedger(result);
    return renderRankedTable(result, false);
  }

  function renderActionableScenarios(result) {
    const section = document.createElement("section");
    section.className = "repair-section repair-scenario-section";
    const heading = document.createElement("div");
    heading.className = "repair-section-heading";
    heading.append(
      cell("h4", "Model export and runtime binding scenarios"),
      cell("span", "ACTIONABLE_CAUSE_GROUPS"),
    );
    section.append(heading);
    for (const intervention of result.export_interventions || []) {
      const card = document.createElement("article");
      card.className = "repair-scenario-card";
      card.append(
        identityBlock(intervention.title, `${intervention.evidence_class} · ${intervention.block_count} matched SE block(s)`),
        scenarioMetrics([
          ["Pattern", intervention.pattern],
          ["Derived reduction axes", `[${(intervention.derived_reduction_axes || []).join(", ")}] (NHWC H/W)`],
          ["Matched MEAN / EXPAND_DIMS", `${formatNumber(intervention.mean_op_indices?.length || 0)} / ${formatNumber(intervention.expand_dims_op_indices?.length || 0)}`],
          ["Delegate segment delta", formatSigned(intervention.signed_delegate_segment_count)],
          ["CPU-island delta", formatSigned(intervention.signed_cpu_segment_count)],
          ["Boundary edge delta", formatSigned(intervention.signed_boundary_edge_count)],
          ["Logical boundary payload delta", optionalSignedBytes(intervention.signed_boundary_payload_bytes)],
          ["Independent single-toggle edge sum", formatSigned(intervention.independent_single_toggle_signed_boundary_edge_count_sum)],
          ["Combined interaction edge term", formatSigned(intervention.interaction_signed_boundary_edge_count)],
          ["Independent single-toggle payload sum", optionalSignedBytes(intervention.independent_single_toggle_signed_boundary_payload_bytes_sum)],
          ["Combined interaction payload term", optionalSignedBytes(intervention.interaction_signed_boundary_payload_bytes)],
          ["Exported graph hypothesis", `${formatNumber(intervention.hypothesized_removed_op_count)} EXPAND_DIMS node(s) removed`],
          ["Excluded rank4 MEAN", (intervention.unmatched_rank4_mean_op_indices || []).length
            ? intervention.unmatched_rank4_mean_op_indices.map((index) => `#${padOp(index)}`).join(", ")
            : "none"],
        ]),
        cell("p", intervention.action, "repair-scenario-action"),
        (intervention.unmatched_rank4_mean_op_indices || []).length
          ? methodLine(`${intervention.unmatched_rank4_mean_reason}. These ops remain separate repair candidates and are not counted in the six-block keepdims portfolio.`)
          : document.createTextNode(""),
        methodLine(intervention.interpretation_boundary),
      );
      section.append(card);
    }
    for (const risk of result.runtime_build_risks || []) {
      const card = document.createElement("article");
      card.className = "repair-scenario-card repair-scenario-risk";
      card.append(
        identityBlock("Runtime build configuration is not artifact-bound", risk.evidence_class),
        scenarioMetrics([
          ["Required configuration", risk.required_build_configuration],
          ["Conditionally delegatable ops affected", `${formatNumber(risk.affected_conditionally_delegatable_op_count)}/${formatNumber(risk.baseline_conditionally_delegatable_op_count)}`],
          ["Affected predicted delegate segments", formatNumber(risk.affected_predicted_delegate_segment_count)],
          ["Affected modeled MAC share", `${(Number(risk.affected_conditionally_delegatable_mac_ratio || 0) * 100).toFixed(1)}%`],
          ["If the build condition is absent: remaining conditionally delegatable ops", formatNumber(risk.absent_condition_remaining_conditionally_delegatable_op_count)],
          ["If the build condition is absent: remaining predicted delegate segments", formatNumber(risk.absent_condition_remaining_predicted_delegate_segment_count)],
        ]),
        methodLine(risk.interpretation_boundary),
      );
      section.append(card);
    }
    if (result.singleton_delegate_segments?.length) {
      const singleton = document.createElement("article");
      singleton.className = "repair-scenario-card";
      singleton.append(
        identityBlock("Singleton delegate segment", "DERIVED_ASSIGNMENT_STRUCTURE"),
        ...result.singleton_delegate_segments.map((row) =>
          cell("p", `C${row.segment_id}: #${padOp(row.op_index)} ${row.op_name} · ${formatNumber(row.macs)} MACs. ${row.interpretation}`, "repair-method-line")),
      );
      section.append(singleton);
    }
    if (!(result.export_interventions?.length || result.runtime_build_risks?.length || result.singleton_delegate_segments?.length)) {
      section.append(emptyState("No grouped export, runtime-build, or singleton-segment scenario was derived."));
    }
    return section;
  }

  function renderFragilityPortfolio(result, analysis) {
    const rows = (result.fragility_ranking_op_indices || [])
      .map((index) => result.toggles.find((row) => row.op_index === index))
      .filter(Boolean);
    if (!rows.length) return emptyState("No delegated op produced a single-op fragmentation increase under this static model.");
    const section = document.createElement("section");
    section.className = "repair-section";
    const heading = document.createElement("div");
    heading.className = "repair-section-heading";
    heading.append(
      cell("h4", "Support-loss sensitivity by operator family"),
      cell("span", "SENSITIVITY_WITHOUT_FAILURE_PROBABILITY"),
    );
    const blockByOp = new Map();
    for (const block of analysis?.block_inventory?.blocks || []) {
      for (const opIndex of block.op_indices || []) blockByOp.set(Number(opIndex), block.block_type || "unclassified");
    }
    const groups = new Map();
    for (const row of rows) {
      const family = row.op_name;
      const entry = groups.get(family) || {
        family,
        blocks: new Set(),
        rows: [],
        macs: 0,
        maxSegments: 0,
        maxSegmentsOp: null,
        maxEdges: 0,
        maxEdgesOp: null,
        maxPayload: null,
        maxPayloadOp: null,
      };
      entry.blocks.add(blockByOp.get(row.op_index) || "unclassified");
      entry.rows.push(row);
      entry.macs += Number(row.macs || 0);
      if (Number(row.signed_delegate_segment_count || 0) > entry.maxSegments) {
        entry.maxSegments = Number(row.signed_delegate_segment_count);
        entry.maxSegmentsOp = row.op_index;
      }
      if (Number(row.signed_boundary_edge_count || 0) > entry.maxEdges) {
        entry.maxEdges = Number(row.signed_boundary_edge_count);
        entry.maxEdgesOp = row.op_index;
      }
      if (row.signed_boundary_payload_bytes != null && Number(row.signed_boundary_payload_bytes) > Number(entry.maxPayload ?? -Infinity)) {
        entry.maxPayload = Number(row.signed_boundary_payload_bytes);
        entry.maxPayloadOp = row.op_index;
      }
      groups.set(family, entry);
    }
    const table = document.createElement("table");
    table.className = "repair-table repair-fragility-group-table";
    table.append(tableHead(["Operator family", "Affected ops", "Block contexts", "Summed MACs", "Max segment delta", "Max edge delta", "Max payload delta"]));
    const tbody = document.createElement("tbody");
    for (const group of [...groups.values()].sort((left, right) => right.rows.length - left.rows.length || right.macs - left.macs || left.family.localeCompare(right.family))) {
      const row = document.createElement("tr");
      row.append(
        cell("td", group.family),
        cell("td", formatNumber(group.rows.length)),
        cell("td", [...group.blocks].join(" / ")),
        cell("td", formatNumber(group.macs)),
        cell("td", `${formatSigned(group.maxSegments)}${group.maxSegmentsOp == null ? "" : ` (#${padOp(group.maxSegmentsOp)})`}`, group.maxSegments > 0 ? "repair-risk" : ""),
        cell("td", `${formatSigned(group.maxEdges)}${group.maxEdgesOp == null ? "" : ` (#${padOp(group.maxEdgesOp)})`}`, group.maxEdges > 0 ? "repair-risk" : ""),
        cell("td", `${optionalSignedBytes(group.maxPayload)}${group.maxPayloadOp == null ? "" : ` (#${padOp(group.maxPayloadOp)})`}`, Number(group.maxPayload || 0) > 0 ? "repair-risk" : ""),
      );
      tbody.append(row);
    }
    table.append(tbody);
    const wrap = document.createElement("div");
    wrap.className = "repair-table-wrap";
    wrap.append(table);
    const details = document.createElement("details");
    details.className = "repair-method-details";
    details.append(cell("summary", `Open the complete ${formatNumber(rows.length)}-row single-op ledger`), renderRankedTable(result, true));
    const padDirectionNote = groups.has("PAD")
      ? " PAD support loss models current predicted support disappearing; PAD folding models current standalone PAD materialization disappearing. They are opposite counterfactual directions and are not additive."
      : "";
    section.append(heading, wrap, methodLine(`This is deterministic support-loss sensitivity grouped by operator family and block context. No failure probability is assigned; the runtime-build scenario above captures the common configuration cause.${padDirectionNote}`), details);
    return section;
  }

  function renderRankedTable(result, fragility) {
    const ranked = fragility ? result.fragility_ranking_op_indices : result.repair_ranking_op_indices;
    const rows = ranked.map((index) => result.toggles.find((row) => row.op_index === index)).filter(Boolean);
    if (!rows.length) return emptyState(fragility
      ? "No delegated op produced a single-op fragmentation increase under this static model."
      : "No predicted CPU op reduced a static boundary when toggled to delegated.");
    const section = document.createElement("section");
    section.className = "repair-section";
    const heading = document.createElement("div");
    heading.className = "repair-section-heading";
    heading.append(
      cell("h4", fragility ? "Support-loss fragmentation ledger" : "Support-extension merge ledger"),
      cell("span", fragility ? "PREDICTED_SUPPORT_LOSS" : "HYPOTHETICAL_SUPPORT_EXTENSION"),
    );
    const tableWrap = document.createElement("div");
    tableWrap.className = "repair-table-wrap";
    const table = document.createElement("table");
    table.className = "repair-table";
    table.append(tableHead(["Rank / op", "Current rule", "Outcome", "Delegate segment delta", "Boundary edge delta", "Boundary payload delta", "Changed tensor edges"]));
    const tableBody = document.createElement("tbody");
    for (const rowData of rows.slice(0, fragility ? rows.length : 16)) {
      const row = document.createElement("tr");
      row.dataset.repairOp = String(rowData.op_index);
      row.tabIndex = 0;
      if (rowData.op_index === selectedOpIndex) row.className = "selected";
      const payload = rowData.signed_boundary_payload_bytes;
      const edges = rowData.signed_boundary_edge_count;
      const segments = rowData.signed_delegate_segment_count;
      row.append(
        identityCell(`#${fragility ? rowData.fragility_rank : rowData.repair_rank} / #${padOp(rowData.op_index)} ${rowData.op_name}`, `${formatNumber(rowData.macs)} MACs`),
        identityCell(rowData.baseline_assignment.replace("predicted_", ""), rowData.baseline_xnnpack_reason),
        identityCell(
          humanize(rowData.outcome_class),
          `${rowData.delegated_neighbor_count}/${rowData.dataflow_neighbor_count} delegated adjacent ops; gross incident boundary ${optionalBytes(rowData.baseline_incident_boundary_payload_bytes)}; net delta ${optionalSignedBytes(rowData.signed_boundary_payload_bytes)}`,
        ),
        signedCell(segments, segments > 0),
        signedCell(edges, edges > 0),
        cell("td", optionalSignedBytes(payload), Number(payload || 0) > 0 ? "repair-risk" : "repair-gain"),
        edgeActionCell(rowData),
      );
      tableBody.append(row);
    }
    table.append(tableBody);
    tableWrap.append(table);
    section.append(heading, tableWrap, methodLine(result.ranking_basis));
    return section;
  }

  function renderCpuIslands(result) {
    const ranked = (result.cpu_island_ranking_indices || [])
      .map((index) => result.cpu_islands.find((island) => island.island_index === index))
      .filter(Boolean);
    if (!ranked.length) return emptyState("No predicted CPU segment exists for this target profile.");
    const selected = ranked.find((island) => island.island_index === selectedIslandIndex) || ranked[0];
    selectedIslandIndex = selected.island_index;
    const section = document.createElement("section");
    section.className = "repair-section";
    const heading = document.createElement("div");
    heading.className = "repair-section-heading";
    heading.append(
      cell("h4", "Predicted CPU-island assignment portfolio"),
      cell("span", "FULL_CONTIGUOUS_CPU_RUN_ASSIGNMENT_PROXY"),
    );
    const tableWrap = document.createElement("div");
    tableWrap.className = "repair-table-wrap";
    const table = document.createElement("table");
    table.className = "repair-table repair-island-table";
    table.append(tableHead(["Rank / island", "Execution range", "Members", "Incident boundaries", "Full-set outcome", "Boundary edge delta", "Payload delta", "Beyond best single"]));
    const tableBody = document.createElement("tbody");
    for (const island of ranked) {
      const row = document.createElement("tr");
      row.dataset.repairIsland = String(island.island_index);
      row.tabIndex = 0;
      if (island.island_index === selected.island_index) row.className = "selected";
      const range = island.first_op_index === island.last_op_index
        ? `#${padOp(island.first_op_index)}`
        : `#${padOp(island.first_op_index)}-#${padOp(island.last_op_index)}`;
      const bestSingle = island.best_single_op_index == null
        ? "none"
        : `#${padOp(island.best_single_op_index)}`;
      row.append(
        identityCell(`#${island.portfolio_rank} / island ${island.island_index}`, island.group_only_repair ? "GROUP_ONLY_REPAIR" : "complete-run toggle"),
        identityCell(range, `${island.execution_position_start}-${island.execution_position_end} execution positions`),
        identityCell(`${formatNumber(island.op_count)} op(s)`, island.op_names.join(" / ")),
        identityCell(formatNumber(island.baseline_incident_boundary_edge_count), optionalBytes(island.baseline_incident_boundary_payload_bytes)),
        identityCell(humanize(island.outcome_class), `${formatSigned(island.signed_delegate_segment_count)} delegate segments`),
        cell("td", formatSigned(island.signed_boundary_edge_count), island.signed_boundary_edge_count > 0 ? "repair-risk" : "repair-gain"),
        cell("td", optionalSignedBytes(island.signed_boundary_payload_bytes), Number(island.signed_boundary_payload_bytes || 0) > 0 ? "repair-risk" : "repair-gain"),
        identityCell(`${formatSigned(island.additional_edge_reduction_over_best_single)} edges`, `${optionalSignedBytes(island.additional_payload_reduction_over_best_single)} vs ${bestSingle}`),
      );
      tableBody.append(row);
    }
    table.append(tableBody);
    tableWrap.append(table);

    const context = document.createElement("div");
    context.className = "repair-context repair-island-context";
    context.append(
      metric("Selected island", `#${selected.island_index}: #${padOp(selected.first_op_index)}-#${padOp(selected.last_op_index)}`, `${selected.op_count} complete member toggle(s)`),
      metric("Baseline incident", `${selected.baseline_incident_boundary_edge_count} edges`, optionalBytes(selected.baseline_incident_boundary_payload_bytes)),
      metric("Full-set delta", `${formatSigned(selected.signed_boundary_edge_count)} edges`, optionalSignedBytes(selected.signed_boundary_payload_bytes)),
      metric("Beyond best single", `${formatSigned(selected.additional_edge_reduction_over_best_single)} edges`, `${optionalSignedBytes(selected.additional_payload_reduction_over_best_single)} payload`),
    );
    const graphButton = document.createElement("button");
    graphButton.type = "button";
    graphButton.className = "secondary-action repair-graph-action";
    graphButton.textContent = `Inspect first member #${padOp(selected.first_op_index)} in graph`;
    graphButton.addEventListener("click", () => jumpToGraphOp?.(selected.first_op_index));
    const previewButton = document.createElement("button");
    previewButton.type = "button";
    previewButton.className = "repair-graph-action";
    previewButton.textContent = "Preview complete island repair";
    previewButton.addEventListener("click", () => onPreviewScenario?.({
      type: "delegation-repair",
      artifactSha256: result.artifact_sha256,
      targetProfileSha256: result.target_profile_sha256,
      islandIndex: selected.island_index,
      opIndices: [...selected.op_indices],
      firstOpIndex: selected.first_op_index,
      lastOpIndex: selected.last_op_index,
      label: `CPU island ${selected.island_index}: #${padOp(selected.first_op_index)}-#${padOp(selected.last_op_index)}`,
      edgeChanges: structuredClone(selected.edge_changes || []),
      boundaryEdgeReductionCount: selected.boundary_edge_reduction_count,
      boundaryPayloadReductionBytes: selected.boundary_payload_reduction_bytes,
    }));
    const graphActions = document.createElement("div");
    graphActions.className = "repair-graph-actions";
    graphActions.append(previewButton, graphButton);
    const edgeWrap = document.createElement("div");
    edgeWrap.className = "repair-table-wrap repair-island-edge-wrap";
    const edgeTable = document.createElement("table");
    edgeTable.className = "repair-table repair-edge-table";
    edgeTable.append(tableHead(["Change", "Tensor", "Producer", "Consumer", "Payload", "Direction"]));
    const edgeBody = document.createElement("tbody");
    appendEdgeRows(edgeBody, selected.edge_changes);
    edgeTable.append(edgeBody);
    edgeWrap.append(edgeTable);
    section.append(heading, tableWrap, context, graphActions, edgeWrap, methodLine(result.island_ranking_basis));
    return section;
  }

  function renderEdgeLedger(result) {
    const selected = result.toggles.find((row) => row.op_index === selectedOpIndex)
      || result.toggles.find((row) => row.repair_opportunity)
      || result.toggles[0];
    if (!selected) return emptyState("No operator counterfactual is available.");
    selectedOpIndex = selected.op_index;
    const section = document.createElement("section");
    section.className = "repair-section";
    const selector = document.createElement("div");
    selector.className = "repair-op-selector";
    const selectableIndices = new Set([
      ...result.repair_ranking_op_indices.slice(0, 16),
      ...result.fragility_ranking_op_indices.slice(0, 24),
    ]);
    const interesting = result.toggles
      .filter((row) => selectableIndices.has(row.op_index))
      .sort((left, right) => left.op_index - right.op_index);
    for (const row of interesting) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.repairOp = String(row.op_index);
      button.className = row.op_index === selected.op_index ? "active" : "";
      button.textContent = `#${padOp(row.op_index)} ${row.op_name}`;
      selector.append(button);
    }
    const context = document.createElement("div");
    context.className = "repair-context";
    context.append(
      metric("Counterfactual", `#${padOp(selected.op_index)} ${selected.op_name}`, `${selected.baseline_assignment} -> ${selected.counterfactual_assignment}`),
      metric("Outcome", humanize(selected.outcome_class), selected.counterfactual_class),
      metric("Segment delta", formatSigned(selected.signed_delegate_segment_count), `${result.baseline.delegate_segment_count} -> ${selected.counterfactual.delegate_segment_count}`),
      metric("Gross incident boundary", optionalBytes(selected.baseline_incident_boundary_payload_bytes), `${selected.baseline_incident_boundary_edge_count} baseline edge(s)`),
      metric("Net payload delta", optionalSignedBytes(selected.signed_boundary_payload_bytes, true), `${optionalBytes(selected.removed_boundary_payload_bytes)} removed / ${optionalBytes(selected.reclassified_boundary_payload_bytes)} reclassified`),
    );
    const graphButton = document.createElement("button");
    graphButton.type = "button";
    graphButton.className = "secondary-action repair-graph-action";
    graphButton.textContent = `Inspect #${padOp(selected.op_index)} in graph`;
    graphButton.addEventListener("click", () => jumpToGraphOp?.(selected.op_index));
    const tableWrap = document.createElement("div");
    tableWrap.className = "repair-table-wrap";
    const table = document.createElement("table");
    table.className = "repair-table repair-edge-table";
    table.append(tableHead(["Change", "Tensor", "Producer", "Consumer", "Payload", "Direction"]));
    const tableBody = document.createElement("tbody");
    appendEdgeRows(tableBody, selected.edge_changes, "No boundary edge changed under this single-op counterfactual.");
    table.append(tableBody);
    tableWrap.append(table);
    const details = document.createElement("details");
    details.className = "repair-method-details";
    const detailSummary = document.createElement("summary");
    detailSummary.textContent = "Method and evidence boundary";
    details.append(detailSummary, methodLine(result.method), methodLine(result.interpretation_boundary));
    section.append(selector, context, graphButton, tableWrap, details);
    return section;
  }

  return { render };
}

export function validateDelegationRepair(result, analysis) {
  if (!result || result.schema !== DELEGATION_REPAIR_SCHEMA) throw new Error("Delegation repair schema is unsupported.");
  if (!analysis || result.artifact_sha256 !== analysis.model_sha256 || result.format !== "tflite") throw new Error("Delegation repair artifact binding is invalid.");
  if (result.target_id !== analysis.target_profile?.id || result.target_profile_sha256 !== analysis.target_profile?.profile_sha256) throw new Error("Delegation repair target binding is invalid.");
  if (!Array.isArray(result.toggles) || result.operator_count !== analysis.ops.length || result.toggles.length !== analysis.ops.length) throw new Error("Delegation repair op coverage is invalid.");
    const graph = buildGraphLedger(analysis);
    if (result.graph_edge_count !== graph.edges.length) throw new Error("Delegation repair graph-edge coverage is invalid.");
    validateBaseline(result.baseline, analysis.predicted_partition_boundaries);
    const baselineSegments = assignSegments(graph.baselineAssignments);
    const baselineBoundaries = boundaryMap(graph.edges, graph.baselineAssignments, baselineSegments);
  const rowByIndex = new Map();
  for (let position = 0; position < result.toggles.length; position += 1) {
    const row = result.toggles[position];
    const op = analysis.ops[position];
    if (row.op_index !== op.index || row.op_name !== op.name || rowByIndex.has(row.op_index)) throw new Error(`Delegation repair op identity is invalid at position ${position}.`);
    rowByIndex.set(row.op_index, row);
    const baselineDelegated = Number(op.xnnpack_chain_id) >= 0;
    if (row.baseline_assignment !== assignmentLabel(baselineDelegated)
      || row.counterfactual_assignment !== assignmentLabel(!baselineDelegated)
      || row.baseline_xnnpack_supported !== Boolean(op.xnnpack_supported)
      || row.baseline_xnnpack_reason !== op.xnnpack_reason) throw new Error(`Delegation repair assignment binding is invalid at #${row.op_index}.`);
    assertNear(row.macs, Number(op.macs || 0), `#${row.op_index} MAC binding`);
    assertNear(row.estimated_bytes, Number(op.estimated_bytes || 0), `#${row.op_index} byte binding`);
    const incidentEdges = graph.edges.filter((edge) => edge.producerPosition === position || edge.consumerPosition === position);
    const adjacentPositions = new Set(incidentEdges.map((edge) => edge.producerPosition === position ? edge.consumerPosition : edge.producerPosition));
    const delegatedIncidentEdges = incidentEdges.filter((edge) => {
      const neighbor = edge.producerPosition === position ? edge.consumerPosition : edge.producerPosition;
      return graph.baselineAssignments[neighbor];
    });
    const delegatedAdjacentPositions = new Set(delegatedIncidentEdges.map((edge) => edge.producerPosition === position ? edge.consumerPosition : edge.producerPosition));
    const incidentBoundaries = [...baselineBoundaries.values()].filter((edge) => edge.producerPosition === position || edge.consumerPosition === position);
    if (row.dataflow_neighbor_count !== adjacentPositions.size
      || row.dataflow_incident_tensor_edge_count !== incidentEdges.length
      || row.delegated_neighbor_count !== delegatedAdjacentPositions.size
      || row.delegated_incident_tensor_edge_count !== delegatedIncidentEdges.length
      || row.baseline_incident_boundary_edge_count !== incidentBoundaries.length) {
      throw new Error(`Delegation repair adjacency binding is invalid at #${row.op_index}.`);
    }
    assertNullableNear(row.baseline_incident_boundary_payload_bytes, completeSum(incidentBoundaries.map((edge) => edge.payloadBytes)), `#${row.op_index} gross incident boundary payload`);
    const counterfactual = recalculateCounterfactual(graph, position);
    validateSummary(row.counterfactual, counterfactual.summary, `#${row.op_index} counterfactual`);
    assertNear(row.signed_delegate_segment_count, row.counterfactual.delegate_segment_count - result.baseline.delegate_segment_count, `#${row.op_index} delegate-segment delta`);
    assertNear(row.signed_cpu_segment_count, row.counterfactual.cpu_segment_count - result.baseline.cpu_segment_count, `#${row.op_index} CPU-segment delta`);
    assertNear(row.signed_boundary_edge_count, row.counterfactual.boundary_edge_count - result.baseline.boundary_edge_count, `#${row.op_index} boundary-edge delta`);
    assertNullableNear(row.signed_boundary_payload_bytes, nullableDifference(row.counterfactual.summed_edge_payload_bytes, result.baseline.summed_edge_payload_bytes), `#${row.op_index} payload delta`);
    assertNear(row.boundary_edge_reduction_count, -row.signed_boundary_edge_count, `#${row.op_index} edge reduction`);
    assertNullableNear(row.boundary_payload_reduction_bytes, row.signed_boundary_payload_bytes == null ? null : -row.signed_boundary_payload_bytes, `#${row.op_index} payload reduction`);
    validateEdgeChanges(row, counterfactual.changes, analysis);
    const expectedOutcome = outcomeClass(baselineDelegated, row.signed_delegate_segment_count, row.signed_boundary_edge_count, row.signed_boundary_payload_bytes);
    if (row.outcome_class !== expectedOutcome) throw new Error(`Delegation repair outcome classification is invalid at #${row.op_index}.`);
    const repair = !baselineDelegated && (row.signed_boundary_edge_count < 0 || row.signed_delegate_segment_count < 0 || Number(row.boundary_payload_reduction_bytes || 0) > 0);
    const fragility = baselineDelegated && (row.signed_boundary_edge_count > 0 || row.signed_delegate_segment_count > 0 || Number(row.signed_boundary_payload_bytes || 0) > 0);
    if (row.repair_opportunity !== repair || row.fragmentation_risk !== fragility) throw new Error(`Delegation repair risk flags are invalid at #${row.op_index}.`);
  }
  const repairRows = result.toggles.filter((row) => row.repair_opportunity).sort(repairComparator(false));
  const fragilityRows = result.toggles.filter((row) => row.fragmentation_risk).sort(repairComparator(true));
  const repairIndices = repairRows.map((row) => row.op_index);
  const fragilityIndices = fragilityRows.map((row) => row.op_index);
  if (!sameArray(result.repair_ranking_op_indices, repairIndices) || !sameArray(result.fragility_ranking_op_indices, fragilityIndices)) throw new Error("Delegation repair ranking is invalid.");
  for (let index = 0; index < repairRows.length; index += 1) if (repairRows[index].repair_rank !== index + 1) throw new Error("Delegation repair rank numbering is invalid.");
  for (let index = 0; index < fragilityRows.length; index += 1) if (fragilityRows[index].fragility_rank !== index + 1) throw new Error("Delegation fragility rank numbering is invalid.");
  if (result.repair_opportunity_count !== repairRows.length || result.fragmentation_risk_count !== fragilityRows.length
    || result.no_static_effect_count !== result.toggles.filter((row) => row.outcome_class === "no_static_boundary_effect").length) throw new Error("Delegation repair summary counts are invalid.");
  validateCpuIslands(result, graph, rowByIndex, analysis);
  validateExportInterventions(result, graph, analysis);
  validateRuntimeBuildRisks(result, graph, analysis);
  validateSingletonSegments(result, graph, analysis);
  if (!String(result.interpretation_boundary || "").includes("not proof")
    || !/toggle exactly one op/i.test(String(result.method || ""))
    || !String(result.method || "").includes("complete run")) throw new Error("Delegation repair evidence boundary is incomplete.");
  return true;
}

function validateExportInterventions(result, graph, analysis) {
  if (!Array.isArray(result.export_interventions)) throw new Error("Delegation repair export interventions are missing.");
  for (const intervention of result.export_interventions) {
    if (intervention.id !== "se_global_pool_keepdims") throw new Error(`Delegation repair export intervention ${intervention.id} is unsupported.`);
    const meanIndices = intervention.mean_op_indices || [];
    const expandIndices = intervention.expand_dims_op_indices || [];
    const toggleIndices = [...meanIndices, ...expandIndices].sort((left, right) => left - right);
    const emittedToggleIndices = [...(intervention.assignment_toggle_op_indices || [])].sort((left, right) => left - right);
    if (!sameArray(toggleIndices, emittedToggleIndices)
      || intervention.block_count !== meanIndices.length
      || intervention.hypothesized_removed_op_count !== expandIndices.length
      || !sameArray(intervention.derived_reduction_axes, [1, 2])
      || !String(intervention.action || "").includes("keepdims=True")) {
      throw new Error("Delegation repair SE export intervention identity is invalid.");
    }
    for (const opIndex of meanIndices) {
      const op = analysis.ops.find((candidate) => candidate.index === opIndex);
      const input = analysis.tensors[op?.inputs?.[0]];
      const output = analysis.tensors[op?.outputs?.[0]];
      if (op?.name !== "MEAN" || !String(op.xnnpack_reason || "").includes("primary_io_quant8_rank4")
        || input?.shape?.length !== 4 || output?.shape?.length !== 2
        || Number(input.shape[0]) !== Number(output.shape[0])
        || Number(input.shape.at(-1)) !== Number(output.shape.at(-1))) {
        throw new Error(`Delegation repair SE MEAN evidence is invalid at #${opIndex}.`);
      }
    }
    if (!expandIndices.every((opIndex) => analysis.ops.find((op) => op.index === opIndex)?.name === "EXPAND_DIMS")) {
      throw new Error("Delegation repair SE EXPAND_DIMS evidence is invalid.");
    }
    const positions = emittedToggleIndices.map((opIndex) => analysis.ops.findIndex((op) => op.index === opIndex));
    if (positions.some((position) => position < 0)) throw new Error("Delegation repair SE assignment coordinates are invalid.");
    const proxy = recalculateCounterfactual(graph, positions, true);
    validateSummary(intervention.assignment_proxy, proxy.summary, "SE assignment proxy");
    assertNear(intervention.signed_delegate_segment_count, proxy.summary.delegate_segment_count - result.baseline.delegate_segment_count, "SE delegate-segment delta");
    assertNear(intervention.signed_cpu_segment_count, proxy.summary.cpu_segment_count - result.baseline.cpu_segment_count, "SE CPU-segment delta");
    assertNear(intervention.signed_boundary_edge_count, proxy.summary.boundary_edge_count - result.baseline.boundary_edge_count, "SE boundary-edge delta");
    assertNullableNear(intervention.signed_boundary_payload_bytes, nullableDifference(proxy.summary.summed_edge_payload_bytes, result.baseline.summed_edge_payload_bytes), "SE boundary-payload delta");
    const singleProxies = positions.map((position) => recalculateCounterfactual(graph, position, true));
    const independentDelegateDelta = singleProxies.reduce((sum, item) => sum + item.summary.delegate_segment_count - result.baseline.delegate_segment_count, 0);
    const independentCpuDelta = singleProxies.reduce((sum, item) => sum + item.summary.cpu_segment_count - result.baseline.cpu_segment_count, 0);
    const independentEdgeDelta = singleProxies.reduce((sum, item) => sum + item.summary.boundary_edge_count - result.baseline.boundary_edge_count, 0);
    const independentPayloadDeltas = singleProxies.map((item) => nullableDifference(item.summary.summed_edge_payload_bytes, result.baseline.summed_edge_payload_bytes));
    const independentPayloadDelta = independentPayloadDeltas.some((value) => value == null)
      ? null
      : independentPayloadDeltas.reduce((sum, value) => sum + value, 0);
    assertNear(intervention.independent_single_toggle_signed_delegate_segment_count_sum, independentDelegateDelta, "SE independent delegate-segment delta sum");
    assertNear(intervention.independent_single_toggle_signed_cpu_segment_count_sum, independentCpuDelta, "SE independent CPU-segment delta sum");
    assertNear(intervention.independent_single_toggle_signed_boundary_edge_count_sum, independentEdgeDelta, "SE independent boundary-edge delta sum");
    assertNullableNear(intervention.independent_single_toggle_signed_boundary_payload_bytes_sum, independentPayloadDelta, "SE independent boundary-payload delta sum");
    assertNear(intervention.interaction_signed_delegate_segment_count, intervention.signed_delegate_segment_count - independentDelegateDelta, "SE delegate-segment interaction");
    assertNear(intervention.interaction_signed_cpu_segment_count, intervention.signed_cpu_segment_count - independentCpuDelta, "SE CPU-segment interaction");
    assertNear(intervention.interaction_signed_boundary_edge_count, intervention.signed_boundary_edge_count - independentEdgeDelta, "SE boundary-edge interaction");
    assertNullableNear(
      intervention.interaction_signed_boundary_payload_bytes,
      intervention.signed_boundary_payload_bytes == null || independentPayloadDelta == null
        ? null
        : intervention.signed_boundary_payload_bytes - independentPayloadDelta,
      "SE boundary-payload interaction",
    );
    const matchedMeans = new Set(meanIndices);
    const expectedUnmatchedMeans = analysis.ops.filter((op, position) => {
      if (matchedMeans.has(op.index) || graph.baselineAssignments[position] || op.name !== "MEAN"
        || !String(op.xnnpack_reason || "").includes("primary_io_quant8_rank4")) return false;
      const input = analysis.tensors[op.inputs?.[0]];
      const output = analysis.tensors[op.outputs?.[0]];
      return input?.shape?.length === 4 && output?.shape?.length !== 4;
    }).map((op) => op.index);
    if (!sameArray(intervention.unmatched_rank4_mean_op_indices || [], expectedUnmatchedMeans)
      || !String(intervention.unmatched_rank4_mean_reason || "").includes("outside the exact six-op")) {
      throw new Error("Delegation repair unmatched rank4 MEAN inventory is invalid.");
    }
    validateScenarioEdgeChanges(intervention, proxy.changes, analysis, "SE");
  }
}

function validateRuntimeBuildRisks(result, graph, analysis) {
  if (!Array.isArray(result.runtime_build_risks)) throw new Error("Delegation repair runtime-build risks are missing.");
  const delegated = analysis.ops.filter((op) => Number(op.xnnpack_chain_id) >= 0);
  for (const risk of result.runtime_build_risks) {
    const affected = delegated.filter((op) => op.xnnpack_build_requirement === risk.required_build_configuration);
    const positions = affected.map((op) => analysis.ops.findIndex((candidate) => candidate.index === op.index));
    const assignments = [...graph.baselineAssignments];
    positions.forEach((position) => { assignments[position] = false; });
    const segments = assignSegments(assignments);
    const affectedSegments = new Set(positions.map((position) => assignSegments(graph.baselineAssignments)[position]).filter((value) => value != null));
    const affectedMacs = affected.reduce((sum, op) => sum + Number(op.macs || 0), 0);
    const totalMacs = analysis.ops.reduce((sum, op) => sum + Number(op.macs || 0), 0);
    if (risk.configuration_binding_status !== "not_embedded_in_model_artifact"
      || risk.baseline_conditionally_delegatable_op_count !== delegated.length
      || risk.affected_conditionally_delegatable_op_count !== affected.length
      || risk.affected_predicted_delegate_segment_count !== affectedSegments.size
      || !sameArray(risk.affected_op_indices, affected.map((op) => op.index))
      || risk.absent_condition_remaining_conditionally_delegatable_op_count !== assignments.filter(Boolean).length
      || risk.absent_condition_remaining_predicted_delegate_segment_count !== new Set(segments.filter((value) => value != null)).size) {
      throw new Error(`Delegation repair runtime-build scenario is invalid for ${risk.required_build_configuration}.`);
    }
    assertNear(risk.affected_conditionally_delegatable_macs, affectedMacs, "runtime-build affected MACs");
    assertNear(risk.affected_conditionally_delegatable_mac_ratio, totalMacs > 0 ? affectedMacs / totalMacs : 0, "runtime-build affected MAC share");
  }
}

function validateSingletonSegments(result, graph, analysis) {
  if (!Array.isArray(result.singleton_delegate_segments)) throw new Error("Delegation repair singleton-segment inventory is missing.");
  const segments = assignSegments(graph.baselineAssignments);
  const members = new Map();
  segments.forEach((segment, position) => {
    if (segment == null) return;
    const rows = members.get(segment) || [];
    rows.push(position);
    members.set(segment, rows);
  });
  const expected = [...members.entries()]
    .filter(([, positions]) => positions.length === 1)
    .map(([segmentId, positions]) => ({ segmentId, op: analysis.ops[positions[0]] }));
  if (result.singleton_delegate_segments.length !== expected.length) throw new Error("Delegation repair singleton-segment count is invalid.");
  expected.forEach((item, index) => {
    const actual = result.singleton_delegate_segments[index];
    if (actual.segment_id !== item.segmentId || actual.op_index !== item.op.index || actual.op_name !== item.op.name) {
      throw new Error(`Delegation repair singleton-segment identity is invalid at ${index}.`);
    }
  });
}

function validateCpuIslands(result, graph, rowByIndex, analysis) {
  const ranges = cpuIslandRanges(graph.baselineAssignments);
  if (!Array.isArray(result.cpu_islands) || result.cpu_island_count !== ranges.length || result.cpu_islands.length !== ranges.length) throw new Error("Delegation repair CPU-island coverage is invalid.");
  const baselineSegments = assignSegments(graph.baselineAssignments);
  const baselineBoundaries = boundaryMap(graph.edges, graph.baselineAssignments, baselineSegments);
  for (let position = 0; position < ranges.length; position += 1) {
    const [start, end] = ranges[position];
    const island = result.cpu_islands[position];
    const members = analysis.ops.slice(start, end + 1);
    if (island.island_index !== position + 1 || island.execution_position_start !== start || island.execution_position_end !== end
      || island.first_op_index !== members[0]?.index || island.last_op_index !== members.at(-1)?.index
      || island.op_count !== members.length || !sameArray(island.op_indices, members.map((op) => op.index))
      || !sameArray(island.op_names, members.map((op) => op.name))) throw new Error(`Delegation repair CPU-island identity is invalid at island ${position + 1}.`);
    assertNear(island.total_macs, members.reduce((sum, op) => sum + Number(op.macs || 0), 0), `island ${island.island_index} MAC total`);
    assertNear(island.summed_estimated_bytes, members.reduce((sum, op) => sum + Number(op.estimated_bytes || 0), 0), `island ${island.island_index} byte total`);
    const incidentEdges = [...baselineBoundaries.values()].filter((edge) => (edge.producerPosition >= start && edge.producerPosition <= end) || (edge.consumerPosition >= start && edge.consumerPosition <= end));
    if (island.baseline_incident_boundary_edge_count !== incidentEdges.length) throw new Error(`Delegation repair CPU-island incident-edge count is invalid at island ${island.island_index}.`);
    assertNullableNear(island.baseline_incident_boundary_payload_bytes, completeSum(incidentEdges.map((edge) => edge.payloadBytes)), `island ${island.island_index} incident payload`);
    const memberPositions = Array.from({ length: end - start + 1 }, (_, offset) => start + offset);
    const counterfactual = recalculateCounterfactual(graph, memberPositions, true);
    validateSummary(island.counterfactual, counterfactual.summary, `island ${island.island_index} counterfactual`);
    const delegateDelta = island.counterfactual.delegate_segment_count - result.baseline.delegate_segment_count;
    const cpuDelta = island.counterfactual.cpu_segment_count - result.baseline.cpu_segment_count;
    const edgeDelta = island.counterfactual.boundary_edge_count - result.baseline.boundary_edge_count;
    const payloadDelta = nullableDifference(island.counterfactual.summed_edge_payload_bytes, result.baseline.summed_edge_payload_bytes);
    assertNear(island.signed_delegate_segment_count, delegateDelta, `island ${island.island_index} delegate-segment delta`);
    assertNear(island.signed_cpu_segment_count, cpuDelta, `island ${island.island_index} CPU-segment delta`);
    assertNear(island.signed_boundary_edge_count, edgeDelta, `island ${island.island_index} boundary-edge delta`);
    assertNullableNear(island.signed_boundary_payload_bytes, payloadDelta, `island ${island.island_index} payload delta`);
    assertNear(island.boundary_edge_reduction_count, -edgeDelta, `island ${island.island_index} edge reduction`);
    assertNullableNear(island.boundary_payload_reduction_bytes, payloadDelta == null ? null : -payloadDelta, `island ${island.island_index} payload reduction`);
    validateEdgeChanges(island, counterfactual.changes, analysis);
    const repairMembers = memberPositions.map((memberPosition) => rowByIndex.get(analysis.ops[memberPosition].index)).filter((row) => row?.repair_opportunity).sort(repairComparator(false));
    const bestSingle = repairMembers[0] || null;
    const bestEdgeReduction = bestSingle?.boundary_edge_reduction_count ?? 0;
    const bestPayloadReduction = bestSingle ? (bestSingle.boundary_payload_reduction_bytes ?? null) : 0;
    const fullRepair = island.boundary_edge_reduction_count > 0 || island.signed_delegate_segment_count < 0 || Number(island.boundary_payload_reduction_bytes || 0) > 0;
    if (island.member_single_repair_count !== repairMembers.length || (island.best_single_op_index ?? null) !== (bestSingle?.op_index ?? null)
      || island.best_single_boundary_edge_reduction_count !== bestEdgeReduction) throw new Error(`Delegation repair CPU-island best-single binding is invalid at island ${island.island_index}.`);
    assertNullableNear(island.best_single_boundary_payload_reduction_bytes, bestPayloadReduction, `island ${island.island_index} best-single payload`);
    assertNear(island.additional_edge_reduction_over_best_single, island.boundary_edge_reduction_count - bestEdgeReduction, `island ${island.island_index} additional edge reduction`);
    assertNullableNear(island.additional_payload_reduction_over_best_single,
      island.boundary_payload_reduction_bytes == null || bestPayloadReduction == null ? null : island.boundary_payload_reduction_bytes - bestPayloadReduction,
      `island ${island.island_index} additional payload reduction`);
    if (island.full_segment_repair !== fullRepair || island.group_only_repair !== (fullRepair && repairMembers.length === 0)
      || island.outcome_class !== cpuIslandOutcomeClass(delegateDelta, edgeDelta, payloadDelta)) throw new Error(`Delegation repair CPU-island classification is invalid at island ${island.island_index}.`);
  }
  const ranked = [...result.cpu_islands].sort(cpuIslandComparator);
  if (!sameArray(result.cpu_island_ranking_indices, ranked.map((island) => island.island_index))) throw new Error("Delegation repair CPU-island ranking is invalid.");
  for (let position = 0; position < ranked.length; position += 1) if (ranked[position].portfolio_rank !== position + 1) throw new Error("Delegation repair CPU-island rank numbering is invalid.");
  if (result.full_segment_repair_count !== result.cpu_islands.filter((island) => island.full_segment_repair).length
    || result.group_only_repair_count !== result.cpu_islands.filter((island) => island.group_only_repair).length) throw new Error("Delegation repair CPU-island summary counts are invalid.");
}

function buildGraphLedger(analysis) {
  const producers = new Map();
  analysis.ops.forEach((op, position) => (op.outputs || []).forEach((tensorIndex) => {
    if (tensorIndex >= 0) producers.set(tensorIndex, position);
  }));
  const edges = [];
  analysis.ops.forEach((consumer, consumerPosition) => {
    const seen = new Set();
    for (const tensorIndex of consumer.inputs || []) {
      if (tensorIndex < 0 || seen.has(tensorIndex) || !producers.has(tensorIndex)) continue;
      seen.add(tensorIndex);
      const tensor = analysis.tensors[tensorIndex];
      if (!tensor || tensor.constant_buffer) continue;
      const payload = tensorPayloadAssessment(tensor);
      const producerPosition = producers.get(tensorIndex);
      edges.push({
        key: `${analysis.ops[producerPosition].index}:${consumer.index}:${tensorIndex}`,
        producerPosition,
        consumerPosition,
        producerIndex: analysis.ops[producerPosition].index,
        consumerIndex: consumer.index,
        tensorIndex,
        payloadBytes: payload.bytes,
        payloadStatus: payload.status,
        payloadBinding: payload.binding,
      });
    }
  });
  edges.sort((left, right) => left.producerIndex - right.producerIndex || left.consumerIndex - right.consumerIndex || left.tensorIndex - right.tensorIndex);
  return {
    analysis,
    edges,
    baselineAssignments: analysis.ops.map((op) => Number(op.xnnpack_chain_id) >= 0),
  };
}

function recalculateCounterfactual(graph, togglePosition, forceDelegated = false) {
  const baselineSegments = assignSegments(graph.baselineAssignments);
  const baselineEdges = boundaryMap(graph.edges, graph.baselineAssignments, baselineSegments);
  const assignments = [...graph.baselineAssignments];
  const positions = Array.isArray(togglePosition) ? togglePosition : [togglePosition];
  for (const position of positions) assignments[position] = forceDelegated ? true : !assignments[position];
  const segments = assignSegments(assignments);
  const boundaries = boundaryMap(graph.edges, assignments, segments);
  const changes = edgeChanges(baselineEdges, boundaries);
  return { summary: boundarySummary(assignments, segments, boundaries), changes };
}

function cpuIslandRanges(assignments) {
  const ranges = [];
  let start = null;
  assignments.forEach((delegated, position) => {
    if (!delegated && start == null) start = position;
    if (delegated && start != null) {
      ranges.push([start, position - 1]);
      start = null;
    }
  });
  if (start != null) ranges.push([start, assignments.length - 1]);
  return ranges;
}

function assignSegments(assignments) {
  const segments = Array(assignments.length).fill(null);
  let next = 0;
  let active = null;
  assignments.forEach((delegated, position) => {
    if (!delegated) {
      active = null;
      return;
    }
    if (active == null) active = next++;
    segments[position] = active;
  });
  return segments;
}

function boundaryMap(edges, assignments, segments) {
  const boundaries = new Map();
  for (const edge of edges) {
    const producer = assignments[edge.producerPosition];
    const consumer = assignments[edge.consumerPosition];
    const boundary = producer && consumer
      ? segments[edge.producerPosition] !== segments[edge.consumerPosition]
      : producer !== consumer;
    if (boundary) boundaries.set(edge.key, { ...edge, direction: boundaryDirection(producer, consumer) });
  }
  return boundaries;
}

function boundarySummary(assignments, segments, boundaries) {
  const payloads = [...boundaries.values()].map((edge) => edge.payloadBytes);
  const assessed = payloads.filter((value) => value != null);
  return {
    delegate_segment_count: new Set(segments.filter((value) => value != null)).size,
    cpu_segment_count: contiguousCount(assignments, false),
    boundary_edge_count: boundaries.size,
    assessed_payload_edge_count: assessed.length,
    unassessed_payload_edge_count: payloads.length - assessed.length,
    assessed_edge_payload_bytes: assessed.reduce((sum, value) => sum + value, 0),
    summed_edge_payload_bytes: assessed.length === payloads.length ? assessed.reduce((sum, value) => sum + value, 0) : null,
  };
}

function edgeChanges(baseline, counterfactual) {
  const keys = [...new Set([...baseline.keys(), ...counterfactual.keys()])].sort();
  const changes = [];
  for (const key of keys) {
    const left = baseline.get(key);
    const right = counterfactual.get(key);
    if (left && !right) changes.push({ transition: "removed", ...left, baselineDirection: left.direction, counterfactualDirection: null });
    else if (!left && right) changes.push({ transition: "added", ...right, baselineDirection: null, counterfactualDirection: right.direction });
    else if (left?.direction !== right?.direction) changes.push({ transition: "reclassified", ...left, baselineDirection: left.direction, counterfactualDirection: right.direction });
  }
  return changes.sort((left, right) => transitionRank(left.transition) - transitionRank(right.transition) || left.producerIndex - right.producerIndex || left.consumerIndex - right.consumerIndex || left.tensorIndex - right.tensorIndex);
}

function validateBaseline(actual, inventory) {
  const expected = {
    boundary_edge_count: Number(inventory?.edge_count || 0),
    assessed_payload_edge_count: Number(inventory?.assessed_payload_edge_count || 0),
    unassessed_payload_edge_count: Number(inventory?.unassessed_payload_edge_count || 0),
    assessed_edge_payload_bytes: Number(inventory?.assessed_edge_payload_bytes || 0),
    summed_edge_payload_bytes: inventory?.summed_edge_payload_bytes ?? null,
  };
  for (const [field, value] of Object.entries(expected)) assertNullableNear(actual?.[field], value, `baseline ${field}`);
}

function validateSummary(actual, expected, label) {
  for (const field of ["delegate_segment_count", "cpu_segment_count", "boundary_edge_count", "assessed_payload_edge_count", "unassessed_payload_edge_count", "assessed_edge_payload_bytes"]) assertNear(actual?.[field], expected[field], `${label} ${field}`);
  assertNullableNear(actual?.summed_edge_payload_bytes, expected.summed_edge_payload_bytes, `${label} summed payload`);
}

function validateEdgeChanges(row, expected, analysis) {
  if (!Array.isArray(row.edge_changes) || row.edge_changes.length !== expected.length) throw new Error(`Delegation repair edge-change coverage is invalid at #${row.op_index}.`);
  for (let index = 0; index < expected.length; index += 1) {
    const actual = row.edge_changes[index];
    const item = expected[index];
    if (actual.transition !== item.transition || actual.producer_op_index !== item.producerIndex || actual.consumer_op_index !== item.consumerIndex || actual.tensor_index !== item.tensorIndex
      || (actual.baseline_direction ?? null) !== item.baselineDirection || (actual.counterfactual_direction ?? null) !== item.counterfactualDirection) throw new Error(`Delegation repair edge-change identity is invalid at #${row.op_index}/${index}.`);
    const tensor = analysis.tensors[item.tensorIndex];
    if (actual.tensor_name !== tensor.name || actual.tensor_dtype !== tensor.dtype || !sameArray(actual.tensor_shape, tensor.shape)) throw new Error(`Delegation repair tensor binding is invalid at #${row.op_index}/${index}.`);
    assertNullableNear(actual.payload_bytes, item.payloadBytes, `#${row.op_index}/${index} edge payload`);
    if (actual.payload_status !== item.payloadStatus || actual.payload_binding !== item.payloadBinding) throw new Error(`Delegation repair payload binding is invalid at #${row.op_index}/${index}.`);
  }
  const removed = expected.filter((edge) => edge.transition === "removed");
  const added = expected.filter((edge) => edge.transition === "added");
  const reclassified = expected.filter((edge) => edge.transition === "reclassified");
  if (row.removed_boundary_edge_count !== removed.length || row.added_boundary_edge_count !== added.length || row.reclassified_boundary_edge_count !== reclassified.length) throw new Error(`Delegation repair edge-change counts are invalid at #${row.op_index}.`);
  assertNullableNear(row.removed_boundary_payload_bytes, completeSum(removed.map((edge) => edge.payloadBytes)), `#${row.op_index} removed payload`);
  assertNullableNear(row.added_boundary_payload_bytes, completeSum(added.map((edge) => edge.payloadBytes)), `#${row.op_index} added payload`);
  if (Object.hasOwn(row, "reclassified_boundary_payload_bytes")) {
    assertNullableNear(row.reclassified_boundary_payload_bytes, completeSum(reclassified.map((edge) => edge.payloadBytes)), `#${row.op_index} reclassified payload`);
  }
}

function validateScenarioEdgeChanges(row, expected, analysis, label) {
  if (!Array.isArray(row.edge_changes) || row.edge_changes.length !== expected.length) throw new Error(`Delegation repair ${label} edge-change coverage is invalid.`);
  const proxy = {
    op_index: label,
    edge_changes: row.edge_changes,
    removed_boundary_edge_count: row.removed_boundary_edge_count,
    added_boundary_edge_count: row.added_boundary_edge_count,
    reclassified_boundary_edge_count: expected.filter((edge) => edge.transition === "reclassified").length,
    removed_boundary_payload_bytes: row.removed_boundary_payload_bytes,
    added_boundary_payload_bytes: row.added_boundary_payload_bytes,
  };
  validateEdgeChanges(proxy, expected, analysis);
}

function repairComparator(fragility) {
  return (left, right) => compareOptionalDesc(
    fragility ? left.signed_boundary_payload_bytes : left.boundary_payload_reduction_bytes,
    fragility ? right.signed_boundary_payload_bytes : right.boundary_payload_reduction_bytes,
  ) || Number(fragility ? right.signed_boundary_edge_count : right.boundary_edge_reduction_count) - Number(fragility ? left.signed_boundary_edge_count : left.boundary_edge_reduction_count)
    || Number(fragility ? right.signed_delegate_segment_count : -right.signed_delegate_segment_count) - Number(fragility ? left.signed_delegate_segment_count : -left.signed_delegate_segment_count)
    || left.op_index - right.op_index;
}

function cpuIslandComparator(left, right) {
  return compareOptionalDesc(left.boundary_payload_reduction_bytes, right.boundary_payload_reduction_bytes)
    || Number(right.boundary_edge_reduction_count) - Number(left.boundary_edge_reduction_count)
    || Number(-right.signed_delegate_segment_count) - Number(-left.signed_delegate_segment_count)
    || Number(left.op_count) - Number(right.op_count)
    || Number(left.first_op_index) - Number(right.first_op_index);
}

function cpuIslandOutcomeClass(segmentDelta, edgeDelta, payloadDelta) {
  if (segmentDelta < 0 && (edgeDelta < 0 || Number(payloadDelta || 0) < 0)) return "eliminates_cpu_island_and_merges_delegate_segments";
  if (edgeDelta < 0 || Number(payloadDelta || 0) < 0) return "eliminates_cpu_island_and_reduces_boundaries";
  if (edgeDelta > 0 || Number(payloadDelta || 0) > 0) return "eliminates_cpu_island_but_increases_boundaries";
  if (segmentDelta < 0) return "eliminates_cpu_island_and_merges_execution_segments";
  return "eliminates_cpu_island_without_boundary_reduction";
}

function outcomeClass(baselineDelegated, segmentDelta, edgeDelta, payloadDelta) {
  if (!baselineDelegated) {
    if (segmentDelta < 0) return "bridge_merges_delegate_segments";
    if (segmentDelta > 0) return "creates_delegate_island";
    if (edgeDelta < 0 || Number(payloadDelta || 0) < 0) return "extends_delegate_coverage";
    if (edgeDelta > 0 || Number(payloadDelta || 0) > 0) return "support_extension_increases_boundaries";
    return "no_static_boundary_effect";
  }
  if (segmentDelta > 0) return "splits_delegate_segment";
  if (segmentDelta < 0) return "removes_singleton_delegate_segment";
  if (edgeDelta > 0 || Number(payloadDelta || 0) > 0) return "support_loss_increases_boundaries";
  if (edgeDelta < 0 || Number(payloadDelta || 0) < 0) return "support_loss_reduces_boundaries";
  return "no_static_boundary_effect";
}

function tensorPayloadBytes(tensor) {
  return tensorPayloadAssessment(tensor).bytes;
}

function tensorPayloadAssessment(tensor) {
  if (!Array.isArray(tensor?.shape) || tensor.shape.some((dim) => !Number.isInteger(Number(dim)) || Number(dim) < 0)) return { bytes: null, status: "not_assessed", binding: "unbound" };
  const signature = Array.isArray(tensor.shape_signature) ? tensor.shape_signature : [];
  const staticSignature = !signature.length
    || (signature.length === tensor.shape.length
      && signature.every((dim, index) => Number.isInteger(Number(dim)) && Number(dim) >= 0 && Number(dim) === Number(tensor.shape[index])));
  const serializedBatchOne = signature.length === tensor.shape.length
    && Number(signature[0]) === -1
    && Number(tensor.shape[0]) === 1
    && signature.slice(1).every((dim, index) => Number.isInteger(Number(dim)) && Number(dim) >= 0 && Number(dim) === Number(tensor.shape[index + 1]));
  if (!staticSignature && !serializedBatchOne) return { bytes: null, status: "not_assessed", binding: "unbound" };
  const width = DTYPE_BYTES[String(tensor.dtype || "").toUpperCase()];
  if (!width) return { bytes: null, status: "not_assessed", binding: "unbound" };
  const elements = tensor.shape.reduce((product, dim) => product * Number(dim), 1);
  if (!Number.isSafeInteger(elements) || !Number.isSafeInteger(elements * width)) return { bytes: null, status: "not_assessed", binding: "unbound" };
  return {
    bytes: elements * width,
    status: serializedBatchOne ? "assessed_serialized_batch1" : "assessed_static",
    binding: serializedBatchOne ? "serialized_batch1_projection" : "static",
  };
}

function contiguousCount(assignments, value) {
  return assignments.filter((item, index) => item === value && (index === 0 || assignments[index - 1] !== value)).length;
}

function boundaryDirection(producer, consumer) {
  if (producer && !consumer) return "delegate_to_cpu";
  if (!producer && consumer) return "cpu_to_delegate";
  if (producer && consumer) return "delegate_partition_to_delegate_partition";
  return "cpu_to_cpu";
}

function assignmentLabel(delegated) {
  return delegated ? "predicted_delegate" : "predicted_cpu";
}

function transitionRank(value) {
  return value === "removed" ? 0 : value === "added" ? 1 : 2;
}

function completeSum(values) {
  return values.some((value) => value == null) ? null : values.reduce((sum, value) => sum + value, 0);
}

function nullableDifference(candidate, baseline) {
  return candidate == null || baseline == null ? null : candidate - baseline;
}

function compareOptionalDesc(left, right) {
  if (left != null && right == null) return -1;
  if (left == null && right != null) return 1;
  return Number(right || 0) - Number(left || 0);
}

function assertNear(actual, expected, label) {
  const tolerance = Math.max(1e-9, Math.abs(Number(expected)) * 1e-10);
  if (!Number.isFinite(Number(actual)) || Math.abs(Number(actual) - Number(expected)) > tolerance) throw new Error(`Delegation repair ${label} invariant failed.`);
}

function assertNullableNear(actual, expected, label) {
  if (expected == null) {
    if (actual != null) throw new Error(`Delegation repair ${label} should be unavailable.`);
    return;
  }
  assertNear(actual, expected, label);
}

function sameArray(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function metric(label, value, detail) {
  const node = document.createElement("div");
  node.className = "repair-metric";
  node.append(cell("span", label), cell("strong", value), cell("small", detail));
  return node;
}

function identityBlock(primary, detail) {
  const node = document.createElement("div");
  node.className = "repair-scenario-identity";
  node.append(cell("strong", primary), cell("span", detail));
  return node;
}

function scenarioMetrics(rows) {
  const node = document.createElement("dl");
  node.className = "repair-scenario-metrics";
  for (const [label, value] of rows) {
    node.append(cell("dt", label), cell("dd", value));
  }
  return node;
}

function identityCell(primary, detail) {
  const node = document.createElement("td");
  node.className = "repair-identity-cell";
  node.append(cell("strong", primary));
  if (detail) node.append(cell("span", detail));
  return node;
}

function signedCell(value, risk) {
  return cell("td", formatSigned(value), risk ? "repair-risk" : "repair-gain");
}

function edgeActionCell(row) {
  const node = document.createElement("td");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "repair-edge-action";
  button.dataset.repairOpenEdges = "true";
  button.textContent = `${row.removed_boundary_edge_count} removed / ${row.added_boundary_edge_count} added / ${row.reclassified_boundary_edge_count} reclassified`;
  node.append(button);
  return node;
}

function appendEdgeRows(body, edges, emptyText = "No boundary edge changed under this complete CPU-island counterfactual.") {
  for (const edge of edges || []) {
    const row = document.createElement("tr");
    row.append(
      cell("td", edge.transition, `repair-edge-${edge.transition}`),
      identityCell(`T${edge.tensor_index} ${edge.tensor_name}`, `${edge.tensor_dtype} [${edge.tensor_shape.join("x")}]`),
      cell("td", `#${padOp(edge.producer_op_index)} ${edge.producer_op_name}`),
      cell("td", `#${padOp(edge.consumer_op_index)} ${edge.consumer_op_name}`),
      cell("td", optionalBytes(edge.payload_bytes)),
      identityCell(edge.baseline_direction || "none", edge.counterfactual_direction ? `-> ${edge.counterfactual_direction}` : "-> none"),
    );
    body.append(row);
  }
  if (!(edges || []).length) {
    const row = document.createElement("tr");
    const empty = cell("td", emptyText);
    empty.colSpan = 6;
    row.append(empty);
    body.append(row);
  }
}

function tableHead(labels) {
  const head = document.createElement("thead");
  const row = document.createElement("tr");
  for (const label of labels) row.append(cell("th", label));
  head.append(row);
  return head;
}

function optionalBytes(value) {
  return value == null ? "N/A" : formatBytes(value);
}

function optionalSignedBytes(value) {
  if (value == null) return "N/A";
  const number = Number(value);
  return `${number > 0 ? "+" : number < 0 ? "-" : ""}${formatBytes(Math.abs(number))}`;
}

function formatSigned(value) {
  const number = Number(value || 0);
  return `${number > 0 ? "+" : ""}${formatNumber(number)}`;
}

function humanize(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase())
    .replaceAll("Cpu", "CPU")
    .replaceAll("Xnnpack", "XNNPACK");
}

function methodLine(text) {
  return cell("p", text, "repair-method-line");
}

function emptyState(text) {
  return cell("div", text, "visual-empty");
}

function cell(tag, text, className = "") {
  const node = document.createElement(tag);
  node.textContent = String(text ?? "");
  if (className) node.className = className;
  return node;
}
