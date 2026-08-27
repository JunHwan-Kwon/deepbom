import { formatBytes, formatNumber, padOp } from "./format.js";
import {
  validateNumericalAbiPropagationAnalysis,
  validateNumericalAbiPropagationShape,
} from "./numerical-abi-propagation.js";

export function createNumericalAbiPropagationController({
  root,
  status,
  summary,
  body,
  downloadButton,
  getAnalysis,
  jumpToGraphOp,
  onDownload,
}) {
  let evidence = null;
  let analysis = null;
  let selectedOpIndex = null;
  let selectedFacet = "exact";
  let renderToken = 0;
  let resizeObserver = null;

  downloadButton?.addEventListener("click", () => {
    if (evidence) onDownload?.(evidence, "numerical_abi_propagation.json");
  });

  function render(explicitAnalysis = null) {
    renderToken += 1;
    const token = renderToken;
    resizeObserver?.disconnect();
    resizeObserver = null;
    analysis = explicitAnalysis || getAnalysis?.() || null;
    evidence = analysis?.numerical_abi_propagation || null;
    if (!evidence || String(analysis?.format || "").toLowerCase() !== "tflite") {
      selectedOpIndex = null;
      if (root) root.hidden = true;
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    if (root) root.hidden = false;
    if (downloadButton) downloadButton.disabled = false;
    try {
      validateNumericalAbiPropagationShape(evidence);
      const propagating = evidence.sources.filter((source) => source.assessment_status === "propagates_structurally");
      if (!propagating.some((source) => source.op_index === selectedOpIndex)) selectedOpIndex = evidence.propagation_ranking_op_indices?.[0] ?? propagating[0]?.op_index ?? null;
      renderSummary(summary, evidence);
      renderBody();
      setStatus(status, propagating.length ? "graph reconstruction / digest verification pending" : humanize(evidence.status), propagating.length ? "watch" : "ok");
      if (propagating.length) {
        validateNumericalAbiPropagationAnalysis(analysis).then(() => {
          if (token === renderToken) setStatus(status, "independently verified", "ok");
        }).catch((error) => {
          if (token === renderToken) setStatus(status, `integrity error: ${error.message}`, "risk");
        });
      }
    } catch (error) {
      summary?.replaceChildren();
      body?.replaceChildren(message(`Numerical ABI propagation evidence rejected: ${error.message}`, "risk"));
      setStatus(status, "evidence rejected", "risk");
    }
  }

  function renderBody() {
    if (!body || !evidence) return;
    resizeObserver?.disconnect();
    const propagating = evidence.sources.filter((source) => source.assessment_status === "propagates_structurally");
    if (!propagating.length) {
      body.replaceChildren(message("No build-mode-divergent source corridor was present."));
      return;
    }
    const visibleSources = propagating.filter((source) => sourceMatchesFacet(source, selectedFacet));
    const source = visibleSources.find((candidate) => candidate.op_index === selectedOpIndex) || visibleSources[0] || propagating[0];
    selectedOpIndex = source.op_index;
    const toolbar = element("div", "abi-propagation-toolbar");
    const facets = element("div", "abi-propagation-facets");
    facets.setAttribute("role", "group");
    facets.setAttribute("aria-label", "Numerical ABI source evidence facet");
    for (const [value, labelText] of [["exact", "Exact local"], ["unresolved", "Unresolved"], ["excluded", "Residue"], ["all", "All interval"]]) {
      const count = propagating.filter((candidate) => sourceMatchesFacet(candidate, value)).length;
      const button = element("button", `abi-propagation-facet${selectedFacet === value ? " selected" : ""}`, `${labelText} ${formatNumber(count)}`);
      button.type = "button";
      button.setAttribute("aria-pressed", String(selectedFacet === value));
      button.addEventListener("click", () => {
        selectedFacet = value;
        renderBody();
      });
      facets.append(button);
    }
    const label = element("label", "abi-propagation-select-label", "Divergence source");
    const select = element("select", "abi-propagation-source-select");
    select.setAttribute("aria-label", "Numerical ABI divergence source operator");
    for (const candidate of visibleSources.length ? visibleSources : propagating) {
      const option = new Option(`#${padOp(candidate.op_index)} ${candidate.op_name} / ${formatNumber(candidate.exact_reachable_divergent_state_count_decimal)} exact / ${formatNumber(candidate.unresolved_divergent_state_count_decimal)} unresolved`, String(candidate.op_index));
      option.selected = candidate.op_index === source.op_index;
      select.append(option);
    }
    select.addEventListener("change", () => {
      selectedOpIndex = Number(select.value);
      renderBody();
    });
    label.append(select);
    const actions = element("div", "abi-propagation-actions");
    actions.append(
      commandButton("Source certificate", "Download the selected source corridor and referenced graph edges", () => onDownload?.(
        selectedSourceCertificate(evidence, source),
        `numerical_abi_source_op_${source.op_index}.json`,
      )),
      commandButton("Graph source", "Open the divergence source in the graph workspace", () => jumpToGraphOp?.(source.op_index)),
    );
    toolbar.append(facets, label, actions);

    const matrixSection = element("section", "abi-propagation-band");
    matrixSection.append(sectionHead("Portfolio propagation matrix", `${humanize(selectedFacet)} evidence facet x artifact ops`));
    const matrixWrap = element("div", "abi-propagation-canvas-wrap matrix");
    const matrix = element("canvas", "abi-propagation-matrix");
    matrix.tabIndex = 0;
    matrix.setAttribute("role", "img");
    matrix.setAttribute("aria-label", "Numerical ABI source-to-operator propagation matrix");
    const matrixTooltip = element("div", "abi-propagation-tooltip");
    matrixTooltip.hidden = true;
    matrixWrap.append(matrix, matrixTooltip);
    installMatrixInteraction(matrix, matrixTooltip, evidence, (opIndex) => {
      selectedOpIndex = opIndex;
      renderBody();
    }, jumpToGraphOp);
    matrixSection.append(matrixWrap, legend());

    const corridorSection = element("section", "abi-propagation-band");
    corridorSection.append(sectionHead("Selected output corridor", "shortest path with execution-domain lanes"));
    const corridorWrap = element("div", "abi-propagation-canvas-wrap corridor");
    const corridor = element("canvas", "abi-propagation-corridor");
    corridor.tabIndex = 0;
    corridor.setAttribute("role", "img");
    corridor.setAttribute("aria-label", `Shortest model-output corridor from operator ${source.op_index}`);
    const corridorTooltip = element("div", "abi-propagation-tooltip corridor");
    corridorTooltip.hidden = true;
    corridorWrap.append(corridor, corridorTooltip);
    installCorridorInteraction(corridor, corridorTooltip, evidence, source, jumpToGraphOp);
    corridorSection.append(corridorWrap);

    const definition = element("section", "abi-propagation-band definition");
    const firstPath = source.model_output_paths[0];
    definition.append(
      sectionHead("Source certificate", `#${padOp(source.op_index)} ${source.op_name}`),
      definitionTable([
        ["Rounding divergence", `${formatNumber(source.divergent_channel_count)} / ${formatNumber(source.assessed_channel_count)} channels; ${formatNumber(source.divergent_state_count_decimal)} / ${formatNumber(source.interval_state_count_decimal)} interval states (${formatPercentPrecise(source.divergent_state_ratio)})`],
        ["Kernel-local reachability", `${humanize(source.local_reachability_status)}; ${formatNumber(source.exact_reachable_divergent_state_count_decimal)} exact / ${formatNumber(source.provably_unreachable_divergent_state_count_decimal)} residue-excluded / ${formatNumber(source.unresolved_divergent_state_count_decimal)} unresolved states`],
        ["Structural corridor", `${formatNumber(source.reachable_op_count)} ops, ${formatNumber(source.reachable_tensor_count)} tensors, ${formatNumber(source.corridor_edge_count)} unique graph edges`],
        ["Model-output contract", firstPath ? `tensor #${firstPath.output_tensor_index} ${firstPath.output_tensor_name}; ${firstPath.shortest_op_hops} edge hops (${firstPath.shortest_path_op_indices.length} ops including source)` : "no declared output reachable"],
        ["Exact route multiplicity", source.exact_model_output_graph_route_count_decimal == null ? source.route_count_status : `${formatNumber(source.exact_model_output_graph_route_count_decimal)} distinct graph routes`],
        ["Merge surface", `${formatNumber(source.reconvergence_op_count)} reconvergences; ${formatNumber(source.single_branch_merge_op_count)} single-branch merges`],
        ["Predicted execution boundaries", `${formatNumber(source.predicted_boundary_edge_count)} edges; ${formatBytes(source.assessed_boundary_logical_payload_bytes)} unique within this source corridor`],
        ["Maximum output-code delta at source", `${formatNumber(source.maximum_absolute_output_delta)} code`],
        ["Source equivalence ledger", source.source_equivalence_ledger_sha256],
        ["Source reachability ledger", source.source_reachability_ledger_sha256],
        ["Propagation ledger", source.propagation_ledger_sha256],
      ]),
    );

    body.replaceChildren(
      toolbar,
      matrixSection,
      corridorSection,
      definition,
      outputPathTable(source, jumpToGraphOp),
      mergeTable(source, jumpToGraphOp),
      boundaryEdgeTable(evidence, source, jumpToGraphOp),
      rankingTable(evidence, source.op_index, selectedFacet, (opIndex) => {
        selectedOpIndex = opIndex;
        renderBody();
      }, jumpToGraphOp),
      message(evidence.interpretation_boundary, "boundary"),
    );
    requestAnimationFrame(() => {
      drawNumericalAbiPropagationMatrix(matrix, evidence, source.op_index, selectedFacet);
      drawNumericalAbiCorridor(corridor, evidence, source);
    });
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver(() => {
        drawNumericalAbiPropagationMatrix(matrix, evidence, source.op_index, selectedFacet);
        drawNumericalAbiCorridor(corridor, evidence, source);
      });
      resizeObserver.observe(body);
    }
  }

  return { render };
}

export function drawNumericalAbiPropagationMatrix(canvas, evidence, selectedOpIndex = null, facet = "all") {
  if (!canvas || !evidence) return;
  const rows = evidence.propagation_ranking_op_indices.map((index) => evidence.sources.find((source) => source.op_index === index)).filter((source) => source && sourceMatchesFacet(source, facet));
  const opCount = Math.max(1, ...evidence.graph_edges.flatMap((edge) => [edge.producer_op_index + 1, edge.consumer_op_index + 1]));
  const width = 1320;
  const height = 690;
  const left = 188;
  const top = 64;
  const right = 34;
  const bottom = 62;
  const cellWidth = (width - left - right) / opCount;
  const cellHeight = (height - top - bottom) / Math.max(1, rows.length);
  prepareCanvas(canvas, width, height);
  const context = canvas.getContext("2d");
  context.fillStyle = "#101820";
  context.fillRect(0, 0, width, height);
  context.font = "12px ui-monospace, SFMono-Regular, Consolas, monospace";
  context.textBaseline = "middle";
  for (let opIndex = 0; opIndex < opCount; opIndex += 1) {
    if (opIndex % 5 !== 0) continue;
    const x = left + (opIndex + 0.5) * cellWidth;
    context.fillStyle = "#82919f";
    context.textAlign = "center";
    context.fillText(String(opIndex), x, 32);
    context.strokeStyle = "rgba(130,145,159,.12)";
    context.beginPath();
    context.moveTo(x, top - 9);
    context.lineTo(x, height - bottom + 5);
    context.stroke();
  }
  const edgeByIndex = evidence.graph_edges;
  rows.forEach((source, rowIndex) => {
    const y = top + rowIndex * cellHeight;
    const reachable = new Set(source.reachable_op_indices);
    const merges = new Map(source.merge_points.map((point) => [point.op_index, point.merge_class]));
    const boundaryConsumers = new Set(source.corridor_edge_indices.filter((index) => edgeByIndex[index]?.predicted_boundary).map((index) => edgeByIndex[index].consumer_op_index));
    if (source.op_index === selectedOpIndex) {
      context.fillStyle = "rgba(83,185,167,.10)";
      context.fillRect(0, y, width, cellHeight);
    }
    context.fillStyle = source.op_index === selectedOpIndex ? "#e8f5f2" : "#a9b8c4";
    context.textAlign = "right";
    context.font = `${source.op_index === selectedOpIndex ? "600 " : ""}11px ui-monospace, SFMono-Regular, Consolas, monospace`;
    context.fillText(`#${padOp(source.op_index)} ${source.op_name.slice(0, 17)}`, left - 12, y + cellHeight / 2);
    for (let opIndex = 0; opIndex < opCount; opIndex += 1) {
      let fill = "#18242e";
      if (reachable.has(opIndex)) fill = "#286a67";
      if (merges.get(opIndex) === "single_branch_merge") fill = "#9b7740";
      if (merges.get(opIndex) === "reconvergence") fill = "#d19b47";
      if (boundaryConsumers.has(opIndex)) fill = "#d9634f";
      if (opIndex === source.op_index) fill = "#63d4c0";
      context.fillStyle = fill;
      context.fillRect(left + opIndex * cellWidth + 0.7, y + 0.7, Math.max(1, cellWidth - 1.4), Math.max(1, cellHeight - 1.4));
    }
  });
  context.fillStyle = "#8395a3";
  context.textAlign = "left";
  context.font = "12px Inter, system-ui, sans-serif";
  context.fillText("Divergence source", 18, 32);
  context.textAlign = "center";
  context.fillText("Downstream artifact operator index", left + (width - left - right) / 2, height - 25);
  canvas.__abiMatrix = { width, height, left, top, cellWidth, cellHeight, rows, opCount };
}

export function drawNumericalAbiCorridor(canvas, evidence, source) {
  if (!canvas || !evidence || !source) return;
  const path = source.model_output_paths[0];
  const width = 1320;
  const height = 390;
  prepareCanvas(canvas, width, height);
  const context = canvas.getContext("2d");
  context.fillStyle = "#101820";
  context.fillRect(0, 0, width, height);
  if (!path) {
    context.fillStyle = "#a9b8c4";
    context.font = "16px Inter, system-ui, sans-serif";
    context.fillText("No declared model output is structurally reachable.", 40, 60);
    return;
  }
  const edges = path.shortest_path_edge_indices.map((index) => evidence.graph_edges[index]);
  const sourceDomain = edges[0]?.producer_domain || "source";
  const domains = sortedDomains([sourceDomain, ...edges.flatMap((edge) => [edge.producer_domain, edge.consumer_domain])]);
  const yByDomain = new Map(domains.map((domain, index) => [domain, 92 + index * (190 / Math.max(1, domains.length - 1))]));
  const nodes = path.shortest_path_op_indices;
  const left = 112;
  const right = 58;
  const step = (width - left - right) / Math.max(1, nodes.length - 1);
  const mergeByOp = new Map(source.merge_points.map((point) => [point.op_index, point]));
  domains.forEach((domain) => {
    const y = yByDomain.get(domain);
    context.strokeStyle = "rgba(132,151,166,.18)";
    context.beginPath();
    context.moveTo(left - 24, y);
    context.lineTo(width - right + 16, y);
    context.stroke();
    context.fillStyle = domain === "TFLITE_CPU" ? "#e4a25c" : "#63bfb0";
    context.textAlign = "right";
    context.font = "12px ui-monospace, SFMono-Regular, Consolas, monospace";
    context.fillText(domain, left - 32, y);
  });
  const nodeDomains = [sourceDomain, ...edges.map((edge) => edge.consumer_domain)];
  for (let index = 0; index < nodes.length - 1; index += 1) {
    const x1 = left + index * step;
    const x2 = left + (index + 1) * step;
    const y1 = yByDomain.get(nodeDomains[index]);
    const y2 = yByDomain.get(nodeDomains[index + 1]);
    const edge = edges[index];
    context.strokeStyle = edge?.predicted_boundary ? "#ef7059" : "#4a9c91";
    context.lineWidth = edge?.predicted_boundary ? 4 : 2;
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
  }
  const geometryNodes = [];
  nodes.forEach((opIndex, index) => {
    const x = left + index * step;
    const y = yByDomain.get(nodeDomains[index]);
    const merge = mergeByOp.get(opIndex);
    context.fillStyle = index === 0 ? "#6de0cb" : index === nodes.length - 1 ? "#f0c16f" : "#d8e2e8";
    context.beginPath();
    context.arc(x, y, merge ? 8 : 5, 0, Math.PI * 2);
    context.fill();
    if (merge) {
      context.strokeStyle = merge.merge_class === "reconvergence" ? "#f2b85b" : "#c99b55";
      context.lineWidth = 3;
      context.beginPath();
      context.arc(x, y, 12, 0, Math.PI * 2);
      context.stroke();
    }
    if (index === 0 || index === nodes.length - 1 || merge || index % 5 === 0 || edges[index - 1]?.predicted_boundary || edges[index]?.predicted_boundary) {
      context.save();
      context.translate(x, y + 20);
      context.rotate(-Math.PI / 3);
      context.fillStyle = "#aebdc8";
      context.textAlign = "right";
      context.font = "11px ui-monospace, SFMono-Regular, Consolas, monospace";
      context.fillText(`#${opIndex}`, 0, 0);
      context.restore();
    }
    geometryNodes.push({ x, y, opIndex, domain: nodeDomains[index], merge });
  });
  context.fillStyle = "#dbe7ec";
  context.textAlign = "left";
  context.font = "600 14px Inter, system-ui, sans-serif";
  context.fillText(`${formatNumber(source.exact_model_output_graph_route_count_decimal)} exact graph routes`, 24, 28);
  context.fillStyle = "#8fa1ae";
  context.font = "12px Inter, system-ui, sans-serif";
  context.fillText(`${path.shortest_op_hops} edge hops (${path.shortest_path_op_indices.length} ops incl. source) / ${source.reconvergence_op_count} reconvergences / ${source.predicted_boundary_edge_count} predicted boundaries`, 24, 50);
  canvas.__abiCorridor = { width, height, nodes: geometryNodes, edges, step };
}

export function renderNumericalAbiPropagationCanvas(analysis, filename = "") {
  const evidence = analysis?.numerical_abi_propagation;
  if (!evidence || typeof document === "undefined") return null;
  const selected = evidence.propagation_ranking_op_indices
    .map((index) => evidence.sources.find((source) => source.op_index === index))
    .find((source) => sourceMatchesFacet(source, "exact"));
  const canvas = document.createElement("canvas");
  canvas.width = 2360;
  canvas.height = 1600;
  const context = canvas.getContext("2d");
  context.fillStyle = "#0f171f";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#eef5f7";
  context.font = "700 56px Inter, system-ui, sans-serif";
  context.fillText("Numerical ABI Propagation Atlas", 92, 105);
  context.fillStyle = "#8fa3b1";
  context.font = "25px Inter, system-ui, sans-serif";
  context.fillText(filename || analysis.filename || "TFLite artifact", 94, 148);
  const metrics = [
    ["EXACT LOCAL SOURCES", `${formatNumber(evidence.exact_local_counterexample_source_op_count)} / ${formatNumber(evidence.divergent_source_op_count)}`],
    ["REACHABLE DIVERGENCE", formatNumber(evidence.exact_local_divergent_state_count_decimal)],
    ["EXACT CORRIDOR INSTANCES", formatNumber(evidence.exact_source_corridor_edge_instance_count)],
    ["EXACT BOUNDARIES", `${formatNumber(evidence.exact_unique_predicted_boundary_edge_count)} / ${formatBytes(evidence.exact_unique_predicted_boundary_logical_payload_bytes)}`],
    ["MAX GRAPH ROUTES", formatNumber(evidence.maximum_model_output_graph_route_count_decimal)],
  ];
  metrics.forEach(([label, value], index) => {
    const x = 94 + index * 445;
    context.fillStyle = "#8ea2af";
    context.font = "18px Inter, system-ui, sans-serif";
    context.fillText(label, x, 218);
    context.fillStyle = index === 4 ? "#f0bd68" : "#e8f3f4";
    context.font = "700 33px ui-monospace, SFMono-Regular, Consolas, monospace";
    context.fillText(value, x, 260);
  });
  const matrix = document.createElement("canvas");
  drawNumericalAbiPropagationMatrix(matrix, evidence, selected?.op_index, "exact");
  context.drawImage(matrix, 90, 310, 2180, 830);
  if (selected) {
    const corridor = document.createElement("canvas");
    drawNumericalAbiCorridor(corridor, evidence, selected);
    context.drawImage(corridor, 90, 1175, 2180, 360);
  }
  context.fillStyle = "#617684";
  context.font = "17px Inter, system-ui, sans-serif";
  context.fillText("Exact kernel-local counterexamples joined to structural corridors; full-model output effects and runtime execution remain unobserved.", 94, 1570);
  return canvas;
}

function renderSummary(root, evidence) {
  root?.replaceChildren(
    metric("Exact local sources", `${formatNumber(evidence.exact_local_counterexample_source_op_count)} / ${formatNumber(evidence.divergent_source_op_count)}`, `${formatNumber(evidence.unresolved_divergence_source_op_count)} unresolved facets / ${formatNumber(evidence.residue_excluded_divergence_source_op_count)} residue facets`),
    metric("Reachable divergence", formatNumber(evidence.exact_local_divergent_state_count_decimal), `${formatPercentPrecise(Number(evidence.exact_local_divergent_state_count_decimal || 0) / Math.max(1, Number(evidence.interval_divergent_state_count_decimal || 0)))} of interval divergence`),
    metric("Exact output-reachable", formatNumber(evidence.exact_output_reachable_source_op_count), `${formatNumber(evidence.output_isolated_source_op_count)} structurally isolated interval sources`),
    metric("Exact corridor union", `${formatNumber(evidence.exact_unique_reachable_op_count)} ops`, `${formatNumber(evidence.exact_unique_reachable_tensor_count)} tensors / ${formatNumber(evidence.exact_source_corridor_edge_instance_count)} repeated source-corridor edge instances`),
    metric("Exact boundaries", formatNumber(evidence.exact_unique_predicted_boundary_edge_count), evidence.exact_unique_predicted_boundary_logical_payload_bytes == null ? "payload not fully assessed" : `${formatBytes(evidence.exact_unique_predicted_boundary_logical_payload_bytes)} / max ${formatNumber(evidence.maximum_model_output_graph_route_count_decimal)} routes`),
  );
}

function outputPathTable(source, jumpToGraphOp) {
  const section = element("section", "abi-propagation-band table-band");
  section.append(sectionHead("Declared output paths", `${source.model_output_paths.length} reached`));
  section.append(table(["Output tensor", "Shortest hops", "Exact graph routes", "Predicted boundaries", "Boundary payload", "Shortest op path"], source.model_output_paths.map((path) => [
    `#${path.output_tensor_index} ${path.output_tensor_name}`,
    formatNumber(path.shortest_op_hops),
    path.exact_graph_route_count_decimal == null ? path.route_count_status : formatNumber(path.exact_graph_route_count_decimal),
    formatNumber(path.shortest_path_predicted_boundary_count),
    path.shortest_path_boundary_logical_payload_bytes == null ? "not fully assessed" : formatBytes(path.shortest_path_boundary_logical_payload_bytes),
    opPath(path.shortest_path_op_indices, jumpToGraphOp),
  ])));
  return section;
}

function mergeTable(source, jumpToGraphOp) {
  const section = element("section", "abi-propagation-band table-band");
  section.append(sectionHead("Merge ledger", `${source.reconvergence_op_count} reconvergence / ${source.single_branch_merge_op_count} single-branch`));
  if (!source.merge_points.length) {
    section.append(message("The selected corridor contains no multi-input graph merge."));
    return section;
  }
  section.append(table(["Hop / op", "Class", "Influenced inputs", "Uninfluenced inputs", "Predicted domain"], source.merge_points.map((point) => [
    opButton(`#${point.minimum_op_hops} / #${padOp(point.op_index)} ${point.op_name}`, point.op_index, jumpToGraphOp),
    humanize(point.merge_class),
    point.influenced_input_tensor_indices.map((index) => `#${index}`).join(", "),
    point.uninfluenced_input_tensor_indices.length ? point.uninfluenced_input_tensor_indices.map((index) => `#${index}`).join(", ") : "none",
    point.predicted_execution_domain,
  ])));
  return section;
}

function boundaryEdgeTable(evidence, source, jumpToGraphOp) {
  const edges = source.corridor_edge_indices.map((index) => evidence.graph_edges[index]).filter((edge) => edge.predicted_boundary);
  const section = element("section", "abi-propagation-band table-band");
  section.append(sectionHead("Predicted execution-boundary ledger", `${edges.length} corridor edges`));
  if (!edges.length) {
    section.append(message("The selected shortest downstream corridor remains within one predicted execution domain."));
    return section;
  }
  section.append(table(["Edge", "Tensor", "Producer", "Consumer", "Direction", "Logical payload"], edges.map((edge) => [
    `#${edge.edge_index}`,
    `#${edge.tensor_index} ${edge.tensor_name} ${shapeText(edge.tensor_shape)} ${edge.tensor_dtype}`,
    opButton(`#${padOp(edge.producer_op_index)} ${edge.producer_op_name}`, edge.producer_op_index, jumpToGraphOp),
    opButton(`#${padOp(edge.consumer_op_index)} ${edge.consumer_op_name}`, edge.consumer_op_index, jumpToGraphOp),
    `${edge.producer_domain} -> ${edge.consumer_domain}`,
    edge.logical_payload_bytes == null ? edge.payload_status : formatBytes(edge.logical_payload_bytes),
  ])));
  return section;
}

function rankingTable(evidence, selectedOpIndex, facet, onSelect, jumpToGraphOp) {
  const byIndex = new Map(evidence.sources.map((source) => [source.op_index, source]));
  const rows = evidence.propagation_ranking_op_indices.map((index) => byIndex.get(index)).filter((source) => source && sourceMatchesFacet(source, facet)).slice(0, 20);
  const section = element("section", "abi-propagation-band table-band");
  section.append(sectionHead("Reachability-qualified exposure ranking", "exact lexicographic order, no synthetic score"));
  const node = table(["Rank / source", "Exact / excluded / unresolved", "Output routes", "Reachable ops", "Merge R / S", "Boundaries", "Ledger"], rows.map((source, rank) => [
    sourceSelectButton(`#${rank + 1} / #${padOp(source.op_index)} ${source.op_name}`, source.op_index, selectedOpIndex, onSelect),
    `${formatNumber(source.exact_reachable_divergent_state_count_decimal)} / ${formatNumber(source.provably_unreachable_divergent_state_count_decimal)} / ${formatNumber(source.unresolved_divergent_state_count_decimal)}`,
    formatNumber(source.exact_model_output_graph_route_count_decimal),
    formatNumber(source.reachable_op_count),
    `${source.reconvergence_op_count} / ${source.single_branch_merge_op_count}`,
    `${source.predicted_boundary_edge_count} / ${formatBytes(source.assessed_boundary_logical_payload_bytes)}`,
    ledgerCell(source, jumpToGraphOp),
  ]));
  section.append(node);
  return section;
}

function installMatrixInteraction(canvas, tooltip, evidence, onSelect, jumpToGraphOp) {
  const update = (event, activate = false) => {
    const geometry = canvas.__abiMatrix;
    if (!geometry) return;
    const point = canvasPoint(canvas, event);
    const row = Math.floor((point.y - geometry.top) / geometry.cellHeight);
    const opIndex = Math.floor((point.x - geometry.left) / geometry.cellWidth);
    const source = geometry.rows[row];
    if (!source || opIndex < 0 || opIndex >= geometry.opCount) return hideTooltip(tooltip);
    const reachable = source.reachable_op_indices.includes(opIndex);
    const merge = source.merge_points.find((point) => point.op_index === opIndex);
    const boundary = source.corridor_edge_indices.map((index) => evidence.graph_edges[index]).find((edge) => edge.consumer_op_index === opIndex && edge.predicted_boundary);
    tooltip.textContent = `source #${source.op_index} -> op #${opIndex}: ${opIndex === source.op_index ? "source" : reachable ? merge?.merge_class || boundary?.predicted_boundary_direction || "reachable" : "outside corridor"}; ${formatNumber(source.exact_reachable_divergent_state_count_decimal)} exact / ${formatNumber(source.provably_unreachable_divergent_state_count_decimal)} excluded / ${formatNumber(source.unresolved_divergent_state_count_decimal)} unresolved; ${formatNumber(source.exact_model_output_graph_route_count_decimal)} output routes`;
    showTooltip(tooltip, canvas, event);
    if (activate) {
      if (event.shiftKey && reachable) jumpToGraphOp?.(opIndex);
      else onSelect?.(source.op_index);
    }
  };
  canvas.addEventListener("pointermove", (event) => update(event));
  canvas.addEventListener("pointerleave", () => hideTooltip(tooltip));
  canvas.addEventListener("click", (event) => update(event, true));
}

function installCorridorInteraction(canvas, tooltip, evidence, source, jumpToGraphOp) {
  const update = (event, activate = false) => {
    const geometry = canvas.__abiCorridor;
    if (!geometry) return;
    const point = canvasPoint(canvas, event);
    const node = geometry.nodes.reduce((best, candidate) => {
      const distance = Math.hypot(point.x - candidate.x, point.y - candidate.y);
      return !best || distance < best.distance ? { candidate, distance } : best;
    }, null);
    if (!node || node.distance > 28) return hideTooltip(tooltip);
    const op = node.candidate;
    const edge = geometry.edges[geometry.nodes.indexOf(op) - 1];
    tooltip.textContent = `#${op.opIndex} ${op.domain}${op.merge ? ` / ${humanize(op.merge.merge_class)}` : ""}${edge?.predicted_boundary ? ` / ${edge.predicted_boundary_direction} / ${formatBytes(edge.logical_payload_bytes)}` : ""}`;
    showTooltip(tooltip, canvas, event);
    if (activate) jumpToGraphOp?.(op.opIndex);
  };
  canvas.addEventListener("pointermove", (event) => update(event));
  canvas.addEventListener("pointerleave", () => hideTooltip(tooltip));
  canvas.addEventListener("click", (event) => update(event, true));
}

function selectedSourceCertificate(evidence, source) {
  return {
    schema: "deepbom.numerical_abi_propagation_source.v1.1",
    method_version: evidence.method_version,
    graph_ledger_sha256: evidence.graph_ledger_sha256,
    route_definition: evidence.route_definition,
    source,
    referenced_graph_edges: source.corridor_edge_indices.map((index) => evidence.graph_edges[index]),
    interpretation_boundary: evidence.interpretation_boundary,
  };
}

function legend() {
  const node = element("div", "abi-propagation-legend");
  for (const [className, text] of [["source", "exact local source"], ["reachable", "structurally reachable"], ["merge", "reconvergence"], ["single", "single-branch merge"], ["boundary", "predicted domain crossing"]]) {
    const item = element("span", "abi-propagation-legend-item");
    item.append(element("i", className), document.createTextNode(text));
    node.append(item);
  }
  return node;
}

function metric(label, value, detail) { const node = element("div", "abi-propagation-metric"); node.append(element("span", "abi-propagation-metric-label", label), element("strong", "", value), element("small", "", detail)); return node; }
function sectionHead(title, detail) { const node = element("div", "abi-propagation-section-head"); node.append(element("h4", "", title), element("span", "", detail)); return node; }
function definitionTable(rows) { const node = element("dl", "abi-propagation-definition"); for (const [term, value] of rows) node.append(element("dt", "", term), element("dd", "", value)); return node; }
function table(headers, rows) { const wrap = element("div", "abi-propagation-table-scroll"); const node = element("table", "abi-propagation-table"); const head = element("thead"); const headRow = element("tr"); headers.forEach((header) => headRow.append(element("th", "", header))); head.append(headRow); const body = element("tbody"); for (const values of rows) { const row = element("tr"); for (const value of values) { const cell = element("td"); if (value instanceof Node) cell.append(value); else cell.textContent = String(value ?? ""); row.append(cell); } body.append(row); } node.append(head, body); wrap.append(node); return wrap; }
function opPath(indices, jumpToGraphOp) { const node = element("div", "abi-propagation-op-path"); indices.forEach((index, position) => { if (position) node.append(document.createTextNode(" -> ")); node.append(opButton(`#${index}`, index, jumpToGraphOp)); }); return node; }
function opButton(text, index, jumpToGraphOp) { const button = element("button", "abi-propagation-link", text); button.type = "button"; button.addEventListener("click", () => jumpToGraphOp?.(index)); return button; }
function sourceSelectButton(text, index, selected, onSelect) { const button = element("button", `abi-propagation-link source${index === selected ? " selected" : ""}`, text); button.type = "button"; button.addEventListener("click", () => onSelect?.(index)); return button; }
function ledgerCell(source, jumpToGraphOp) { const node = element("div", "abi-propagation-ledger-cell"); node.append(element("code", "", source.propagation_ledger_sha256.slice(0, 12)), opButton("open", source.op_index, jumpToGraphOp)); return node; }
function commandButton(text, title, action) { const button = element("button", "secondary-action", text); button.type = "button"; button.title = title; button.addEventListener("click", action); return button; }
function message(text, className = "") { return element("p", `abi-propagation-message ${className}`.trim(), text); }
function element(tag, className = "", text = "") { const node = document.createElement(tag); if (className) node.className = className; if (text !== "") node.textContent = text; return node; }
function setStatus(node, text, tone) { if (!node) return; node.textContent = text; node.dataset.tone = tone; }
function humanize(value) { return String(value || "").replaceAll("_", " "); }
function shapeText(shape) { return `[${(shape || []).join("x")}]`; }
function formatPercentPrecise(value) { return `${(Number(value || 0) * 100).toFixed(6)}%`; }
function sourceMatchesFacet(source, facet) {
  if (!source) return false;
  if (facet === "exact") return BigInt(source.exact_reachable_divergent_state_count_decimal || "0") > 0n;
  if (facet === "unresolved") return BigInt(source.unresolved_divergent_state_count_decimal || "0") > 0n;
  if (facet === "excluded") return BigInt(source.provably_unreachable_divergent_state_count_decimal || "0") > 0n;
  return source.assessment_status === "propagates_structurally";
}
function sortedDomains(domains) { const priority = (value) => value === "TFLITE_CPU" ? 1 : value.includes(":C0") ? 0 : 2; return [...new Set(domains)].sort((left, right) => priority(left) - priority(right) || left.localeCompare(right)); }
function prepareCanvas(canvas, width, height) { const ratio = Math.max(1, Math.min(2, window.devicePixelRatio || 1)); canvas.width = width * ratio; canvas.height = height * ratio; canvas.style.aspectRatio = `${width} / ${height}`; canvas.getContext("2d").setTransform(ratio, 0, 0, ratio, 0, 0); }
function canvasPoint(canvas, event) { const rect = canvas.getBoundingClientRect(); const geometry = canvas.__abiMatrix || canvas.__abiCorridor; return { x: (event.clientX - rect.left) * geometry.width / rect.width, y: (event.clientY - rect.top) * geometry.height / rect.height }; }
function showTooltip(tooltip, canvas, event) { const rect = canvas.getBoundingClientRect(); tooltip.hidden = false; tooltip.style.left = `${Math.min(rect.width - 280, Math.max(8, event.clientX - rect.left + 12))}px`; tooltip.style.top = `${Math.max(8, event.clientY - rect.top - 42)}px`; }
function hideTooltip(tooltip) { tooltip.hidden = true; }
