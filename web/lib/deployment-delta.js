import { formatBytes, formatNumber, formatPercent, formatUs, padOp } from "./format.js";

export const DEPLOYMENT_DELTA_SCHEMA = "deepbom.deployment_delta.v1.1";

const VIEWS = new Set(["overview", "targets", "drivers", "alignment"]);
const COMPONENTS = Object.freeze([
  ["Compute", "compute_us"],
  ["Memory", "memory_us"],
  ["Packing", "packing_us"],
  ["Planning setup", "boundary_us"],
  ["Fallback", "fallback_us"],
]);

export function createDeploymentDeltaController({
  root,
  status,
  summary,
  body,
  downloadButton,
  getDelta,
  getBaseline,
  jumpToGraphOp,
  onDownload,
}) {
  let activeView = "overview";
  let activeTargetId = "";
  let alignmentFilter = "all";
  let renderedPair = "";

  root?.addEventListener("click", (event) => {
    const view = event.target.closest("[data-delta-view]");
    if (view && VIEWS.has(view.dataset.deltaView)) {
      activeView = view.dataset.deltaView;
      render(getDelta());
      return;
    }
    const target = event.target.closest("[data-delta-target]");
    if (target) {
      activeTargetId = target.dataset.deltaTarget;
      render(getDelta());
      return;
    }
    const filter = event.target.closest("[data-delta-filter]");
    if (filter) {
      alignmentFilter = filter.dataset.deltaFilter;
      render(getDelta());
      return;
    }
    const graph = event.target.closest("[data-delta-candidate-op]");
    if (graph) jumpToGraphOp?.(Number(graph.dataset.deltaCandidateOp));
  });
  root?.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const graph = event.target.closest("[data-delta-candidate-op]");
    if (!graph) return;
    event.preventDefault();
    jumpToGraphOp?.(Number(graph.dataset.deltaCandidateOp));
  });
  downloadButton?.addEventListener("click", () => {
    const delta = getDelta();
    if (delta) onDownload?.(delta, "deployment_delta.json");
  });

  function render(delta, candidateAnalysis = null) {
    if (!root || !status || !summary || !body) return;
    if (!delta) {
      renderedPair = "";
      activeTargetId = "";
      const baseline = getBaseline?.();
      status.textContent = baseline ? `baseline pinned / ${shortHash(baseline.sha256)}` : "baseline not pinned";
      summary.replaceChildren();
      body.replaceChildren(emptyState(baseline
        ? "Audit a different TFLite artifact to calculate the deployment delta."
        : "Pin the current TFLite audit as the comparison baseline."));
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    validateDeploymentDelta(delta, candidateAnalysis);
    const pair = `${delta.baseline.sha256}:${delta.candidate.sha256}`;
    if (pair !== renderedPair) {
      renderedPair = pair;
      activeTargetId = delta.worst_relative_delta_target_id || delta.target_deltas[0]?.target_id || "";
      alignmentFilter = "all";
    }
    status.textContent = `${delta.target_count} targets / DERIVED + ESTIMATED`;
    if (downloadButton) downloadButton.disabled = false;
    const consistent = delta.cross_target_drivers.filter((row) => row.consistent_regression).length;
    const worstTarget = delta.target_deltas.find((target) => target.target_id === delta.worst_relative_delta_target_id);
    summary.replaceChildren(
      metric("Alignment", `${delta.alignment.matched_op_count} matched`, `${delta.alignment.added_op_count} added / ${delta.alignment.removed_op_count} removed`),
      metric("Worst modeled delta", formatSignedPercent(delta.worst_relative_delta), shortTarget(worstTarget?.target_label || delta.worst_relative_delta_target_id)),
      metric("Cross-target regressions", formatNumber(consistent), `of ${delta.cross_target_drivers.length} aligned entities`),
      metric("Contract change", delta.graph_delta.input_contract_changed || delta.graph_delta.output_contract_changed ? "Detected" : "None", `${delta.baseline.input_contracts.join(", ")} -> ${delta.candidate.input_contracts.join(", ")}`),
    );
    body.replaceChildren(renderTabs(), renderActiveView(delta));
  }

  function renderTabs() {
    const tabs = document.createElement("div");
    tabs.className = "delta-tabs";
    tabs.setAttribute("role", "tablist");
    for (const [id, label] of [["overview", "Overview"], ["targets", "Targets"], ["drivers", "Drivers"], ["alignment", "Alignment"]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.deltaView = id;
      button.className = id === activeView ? "active" : "";
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", String(id === activeView));
      button.textContent = label;
      tabs.append(button);
    }
    return tabs;
  }

  function renderActiveView(delta) {
    if (activeView === "targets") return renderTargets(delta);
    if (activeView === "drivers") return renderDrivers(delta);
    if (activeView === "alignment") return renderAlignment(delta);
    return renderOverview(delta);
  }

  function renderOverview(delta) {
    const wrap = document.createElement("div");
    wrap.className = "delta-overview";
    wrap.append(
      comparisonTable(delta),
      graphChanges(delta),
      methodDetails(delta),
    );
    return wrap;
  }

  function comparisonTable(delta) {
    const section = document.createElement("section");
    section.className = "delta-section";
    section.append(cell("h4", "Artifact comparison"));
    const tableWrap = document.createElement("div");
    tableWrap.className = "delta-table-wrap";
    const table = document.createElement("table");
    table.className = "delta-table";
    table.append(tableHead(["Property", "Baseline", "Candidate", "Delta"]));
    const body = document.createElement("tbody");
    const rows = [
      ["Artifact", delta.baseline.filename, delta.candidate.filename, delta.alignment.artifact_relation],
      ["SHA-256", shortHash(delta.baseline.sha256), shortHash(delta.candidate.sha256), delta.baseline.sha256 === delta.candidate.sha256 ? "identical" : "different"],
      ["File size", formatBytes(delta.baseline.file_size), formatBytes(delta.candidate.file_size), formatSignedBytes(delta.graph_delta.signed_file_size_bytes)],
      ["Operators", formatNumber(delta.baseline.operator_count), formatNumber(delta.candidate.operator_count), formatSignedInteger(delta.graph_delta.signed_operator_count)],
      ["Tensors", formatNumber(delta.baseline.tensor_count), formatNumber(delta.candidate.tensor_count), formatSignedInteger(delta.graph_delta.signed_tensor_count)],
      ["MACs", formatNumber(delta.baseline.total_macs), formatNumber(delta.candidate.total_macs), `${formatSignedNumber(delta.graph_delta.signed_total_macs)} (${formatSignedPercent(delta.graph_delta.relative_total_macs_delta)})`],
      ["Quantized compute", formatPercent(delta.baseline.quantized_compute_mac_ratio), formatPercent(delta.candidate.quantized_compute_mac_ratio), formatSignedPoints(delta.graph_delta.signed_quantized_compute_mac_ratio)],
      ["All-zero kernel slices", formatNumber(delta.baseline.zero_kernel_slice_count), formatNumber(delta.candidate.zero_kernel_slice_count), formatSignedInteger(delta.graph_delta.signed_zero_kernel_slice_count)],
    ];
    for (const values of rows) {
      const row = document.createElement("tr");
      values.forEach((value, index) => row.append(cell(index ? "td" : "th", value)));
      body.append(row);
    }
    table.append(body);
    tableWrap.append(table);
    section.append(tableWrap);
    return section;
  }

  function graphChanges(delta) {
    const section = document.createElement("section");
    section.className = "delta-section";
    section.append(cell("h4", "Deterministic graph ledger"));
    const ledger = document.createElement("div");
    ledger.className = "delta-ledger";
    ledger.append(
      metric("Exact structure", formatNumber(delta.alignment.exact_structural_match_count), "dtype / shape / constant role / quant presence"),
      metric("Sequence coordinate", formatNumber(delta.alignment.op_sequence_match_count), "op type only inside exact-anchor gaps"),
      metric("Baseline coverage", formatPercent(delta.alignment.baseline_match_ratio), `${delta.alignment.removed_op_count} explicit removals`),
      metric("Candidate coverage", formatPercent(delta.alignment.candidate_match_ratio), `${delta.alignment.added_op_count} explicit additions`),
    );
    const contracts = document.createElement("div");
    contracts.className = "delta-contracts";
    contracts.append(
      contractLine("Input", delta.baseline.input_contracts, delta.candidate.input_contracts, delta.graph_delta.input_contract_changed),
      contractLine("Output", delta.baseline.output_contracts, delta.candidate.output_contracts, delta.graph_delta.output_contract_changed),
    );
    section.append(ledger, contracts);
    return section;
  }

  function renderTargets(delta) {
    const section = document.createElement("section");
    section.className = "delta-section";
    section.append(cell("h4", "Target-specific modeled change"));
    const tableWrap = document.createElement("div");
    tableWrap.className = "delta-table-wrap";
    const table = document.createElement("table");
    table.className = "delta-table delta-target-table";
    table.append(tableHead(["Target", "Cold baseline", "Cold candidate", "Cold delta", ...COMPONENTS.map(([label]) => label)]));
    const body = document.createElement("tbody");
    for (const target of delta.target_deltas) {
      const row = document.createElement("tr");
      row.append(
        identityCell(shortTarget(target.target_label || target.target_id), shortHash(target.target_profile_sha256)),
        cell("td", formatUs(target.baseline_total_us)),
        cell("td", formatUs(target.candidate_total_us)),
        signedTimeCell(target.signed_delta_us, target.relative_delta),
      );
      for (const [, field] of COMPONENTS) row.append(cell("td", formatSignedUs(target.component_delta[field]), signedClass(target.component_delta[field])));
      body.append(row);
    }
    table.append(body);
    tableWrap.append(table);
    const note = cell("p", "Every target row is a cold-start planning-profile total. It independently conserves candidate minus baseline across steady compute, memory, and fallback plus one-time packing and unmeasured partition-planning setup; setup is not charged to steady inference.", "delta-method-line");
    section.append(tableWrap, note);
    return section;
  }

  function renderDrivers(delta) {
    const section = document.createElement("section");
    section.className = "delta-section";
    const heading = document.createElement("div");
    heading.className = "delta-section-heading";
    heading.append(cell("h4", "Regression and improvement drivers"), targetSelector(delta, activeTargetId));
    section.append(heading);
    const target = delta.target_deltas.find((item) => item.target_id === activeTargetId) || delta.target_deltas[0];
    activeTargetId = target.target_id;
    const tableWrap = document.createElement("div");
    tableWrap.className = "delta-table-wrap";
    const table = document.createElement("table");
    table.className = "delta-table delta-driver-table";
    table.append(tableHead(["Entity", "Baseline -> candidate", "Relation", "Baseline", "Candidate", "Signed delta", "Largest component"]));
    const body = document.createElement("tbody");
    for (const driver of target.drivers.filter((row) => Math.abs(row.signed_delta_us) > 1e-12).slice(0, 18)) {
      const row = document.createElement("tr");
      if (driver.candidate_op_index != null) {
        row.dataset.deltaCandidateOp = String(driver.candidate_op_index);
        row.tabIndex = 0;
      }
      const component = largestComponent(driver.component_delta);
      row.append(
        identityCell(driver.entity_id, `${driver.match_class.replaceAll("_", " ")}`),
        opTransitionCell(driver),
        cell("td", driver.relation),
        cell("td", formatUs(driver.baseline_us)),
        cell("td", formatUs(driver.candidate_us)),
        cell("td", formatSignedUs(driver.signed_delta_us), signedClass(driver.signed_delta_us)),
        identityCell(component.label, formatSignedUs(driver.component_delta[component.field])),
      );
      body.append(row);
    }
    table.append(body);
    tableWrap.append(table);
    const cross = document.createElement("div");
    cross.className = "delta-cross-target";
    cross.append(cell("h5", "Cross-target consistency"));
    for (const driver of delta.cross_target_drivers.filter((row) => row.consistent_regression || row.consistent_improvement).slice(0, 8)) {
      const line = document.createElement("button");
      line.type = "button";
      line.className = "delta-cross-row";
      if (driver.candidate_op_index != null) line.dataset.deltaCandidateOp = String(driver.candidate_op_index);
      line.append(
        identityCell(driver.entity_id, opLabel(driver.candidate_op_index, driver.candidate_op_name) || opLabel(driver.baseline_op_index, driver.baseline_op_name)),
        cell("strong", driver.consistent_regression ? `${driver.regression_target_count}/${delta.target_count} regressions` : `${driver.improvement_target_count}/${delta.target_count} improvements`, driver.consistent_regression ? "delta-regression" : "delta-improvement"),
        cell("span", `${formatSignedUs(driver.min_delta_us)} to ${formatSignedUs(driver.max_delta_us)}`),
      );
      cross.append(line);
    }
    section.append(tableWrap, cross);
    return section;
  }

  function renderAlignment(delta) {
    const section = document.createElement("section");
    section.className = "delta-section";
    const heading = document.createElement("div");
    heading.className = "delta-section-heading";
    heading.append(cell("h4", "Graph alignment ledger"), alignmentFilters(delta, alignmentFilter));
    section.append(heading);
    const rows = delta.alignment_rows.filter((row) => alignmentFilter === "all" || row.relation === alignmentFilter);
    const tableWrap = document.createElement("div");
    tableWrap.className = "delta-table-wrap";
    const table = document.createElement("table");
    table.className = "delta-table delta-alignment-table";
    table.append(tableHead(["Entity", "Baseline", "Candidate", "Match basis", "Shape", "Quantization", "Assignment", "MAC delta"]));
    const body = document.createElement("tbody");
    for (const item of rows) {
      const row = document.createElement("tr");
      if (item.candidate_op_index != null) {
        row.dataset.deltaCandidateOp = String(item.candidate_op_index);
        row.tabIndex = 0;
      }
      row.append(
        identityCell(item.entity_id, item.relation),
        identityCell(opLabel(item.baseline_op_index, item.baseline_op_name) || "-", shapesLabel(item.baseline_output_shapes)),
        identityCell(opLabel(item.candidate_op_index, item.candidate_op_name) || "-", shapesLabel(item.candidate_output_shapes)),
        cell("td", item.match_class.replaceAll("_", " ")),
        transitionCell(item.output_shape_changed, item.output_shape_changed == null ? "not comparable" : item.output_shape_changed ? "changed" : "stable"),
        transitionCell(item.quantization_transition, `${item.baseline_quantization_state || "-"} -> ${item.candidate_quantization_state || "-"}`),
        transitionCell(item.static_assignment_transition, `${item.baseline_static_assignment || "-"} -> ${item.candidate_static_assignment || "-"}`),
        cell("td", formatSignedNumber(item.signed_macs_delta), signedClass(item.signed_macs_delta)),
      );
      body.append(row);
    }
    table.append(body);
    tableWrap.append(table);
    section.append(tableWrap, cell("p", delta.alignment.semantic_identity_conclusion, "delta-method-line"));
    return section;
  }

  return { render };
}

export function validateDeploymentDelta(delta, candidateAnalysis = null) {
  if (!delta || delta.schema !== DEPLOYMENT_DELTA_SCHEMA) throw new Error("Deployment delta schema is unsupported.");
  validateArtifact(delta.baseline, "baseline");
  validateArtifact(delta.candidate, "candidate");
  if (candidateAnalysis) {
    if (delta.candidate.sha256 !== candidateAnalysis.model_sha256) throw new Error("Deployment delta candidate SHA binding is invalid.");
    if (delta.candidate.operator_count !== candidateAnalysis.ops?.length || delta.candidate.tensor_count !== candidateAnalysis.tensor_count) throw new Error("Deployment delta candidate graph binding is invalid.");
    const selectedTarget = delta.target_deltas?.find((target) => target.target_id === candidateAnalysis.target_profile?.id);
    if (!selectedTarget) throw new Error("Deployment delta selected candidate target is unbound.");
    const candidateBottleneckTotal = (candidateAnalysis.ops || []).reduce((sum, op) => sum + Number(op.bottleneck_total_us || 0), 0);
    assertNear(selectedTarget.candidate_total_us, candidateBottleneckTotal, "candidate bottleneck-total binding");
  }
  const graph = delta.graph_delta || {};
  assertNear(graph.signed_file_size_bytes, delta.candidate.file_size - delta.baseline.file_size, "file-size delta");
  assertNear(graph.signed_operator_count, delta.candidate.operator_count - delta.baseline.operator_count, "operator-count delta");
  assertNear(graph.signed_tensor_count, delta.candidate.tensor_count - delta.baseline.tensor_count, "tensor-count delta");
  assertNear(graph.signed_total_macs, delta.candidate.total_macs - delta.baseline.total_macs, "MAC delta");
  assertNullableNear(graph.relative_total_macs_delta, relativeDelta(delta.candidate.total_macs, delta.baseline.total_macs), "relative MAC delta");
  assertNear(graph.signed_quantized_compute_mac_ratio, delta.candidate.quantized_compute_mac_ratio - delta.baseline.quantized_compute_mac_ratio, "quantized-compute delta");
  assertNear(graph.signed_predicted_effective_chain_breaks, delta.candidate.predicted_effective_chain_breaks - delta.baseline.predicted_effective_chain_breaks, "chain-break delta");
  assertNear(graph.signed_delegated_mac_ratio, delta.candidate.delegated_mac_ratio - delta.baseline.delegated_mac_ratio, "delegated-MAC delta");
  assertNear(graph.signed_zero_kernel_slice_count, delta.candidate.zero_kernel_slice_count - delta.baseline.zero_kernel_slice_count, "dead-slice delta");
  if (graph.input_contract_changed !== !sameArray(delta.baseline.input_contracts, delta.candidate.input_contracts)
    || graph.output_contract_changed !== !sameArray(delta.baseline.output_contracts, delta.candidate.output_contracts)) throw new Error("Deployment delta I/O contract flags are invalid.");

  const rows = delta.alignment_rows;
  if (!Array.isArray(rows) || !delta.alignment) throw new Error("Deployment delta alignment ledger is missing.");
  const entityIds = rows.map((row) => row.entity_id);
  if (new Set(entityIds).size !== rows.length) throw new Error("Deployment delta alignment entity IDs are not unique.");
  const baselineIndices = [];
  const candidateIndices = [];
  const counts = { matched: 0, added: 0, removed: 0, exact: 0, sequence: 0 };
  let previousBaselineIndex = -1;
  let previousCandidateIndex = -1;
  const candidateOps = new Map((candidateAnalysis?.ops || []).map((op) => [Number(op.index), op]));
  for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
    const row = rows[rowIndex];
    if (row.entity_id !== `E${String(rowIndex).padStart(4, "0")}`) throw new Error(`Deployment delta entity ordering is invalid at ${row.entity_id}.`);
    if (!new Set(["matched", "added", "removed"]).has(row.relation)) throw new Error(`Deployment delta relation is invalid at ${row.entity_id}.`);
    counts[row.relation] += 1;
    if (row.baseline_op_index != null) {
      if (row.baseline_op_index <= previousBaselineIndex) throw new Error(`Deployment delta baseline sequence is not monotonic at ${row.entity_id}.`);
      previousBaselineIndex = row.baseline_op_index;
      baselineIndices.push(row.baseline_op_index);
    }
    if (row.candidate_op_index != null) {
      if (row.candidate_op_index <= previousCandidateIndex) throw new Error(`Deployment delta candidate sequence is not monotonic at ${row.entity_id}.`);
      previousCandidateIndex = row.candidate_op_index;
      candidateIndices.push(row.candidate_op_index);
      if (candidateAnalysis) {
        const op = candidateOps.get(Number(row.candidate_op_index));
        if (!op || row.candidate_op_name !== op.name
          || !sameArray(row.candidate_output_shapes, op.output_shapes)
          || row.candidate_quantization_state !== op.quantization_state
          || row.candidate_static_assignment !== staticAssignment(op)) throw new Error(`Deployment delta candidate op binding is invalid at ${row.entity_id}.`);
        assertNear(row.candidate_macs, Number(op.macs || 0), `${row.entity_id} candidate MAC binding`);
      }
    }
    if (row.relation === "matched") {
      if (row.baseline_op_index == null || row.candidate_op_index == null) throw new Error(`Deployment delta matched entity is incomplete at ${row.entity_id}.`);
      if (row.match_class === "exact_structural_signature") counts.exact += 1;
      else if (row.match_class === "op_type_sequence_alignment") counts.sequence += 1;
      else throw new Error(`Deployment delta match class is invalid at ${row.entity_id}.`);
    } else if (row.relation === "added" && (row.baseline_op_index != null || row.candidate_op_index == null)) throw new Error(`Deployment delta added entity is invalid at ${row.entity_id}.`);
    else if (row.relation === "removed" && (row.baseline_op_index == null || row.candidate_op_index != null)) throw new Error(`Deployment delta removed entity is invalid at ${row.entity_id}.`);
  }
  assertIndexCoverage(baselineIndices, delta.baseline.operator_count, "baseline");
  assertIndexCoverage(candidateIndices, delta.candidate.operator_count, "candidate");
  if (counts.matched !== delta.alignment.matched_op_count || counts.added !== delta.alignment.added_op_count || counts.removed !== delta.alignment.removed_op_count
    || counts.exact !== delta.alignment.exact_structural_match_count || counts.sequence !== delta.alignment.op_sequence_match_count) throw new Error("Deployment delta alignment summary is invalid.");
  assertNear(delta.alignment.baseline_match_ratio, ratio(counts.matched, delta.baseline.operator_count), "baseline match ratio");
  assertNear(delta.alignment.candidate_match_ratio, ratio(counts.matched, delta.candidate.operator_count), "candidate match ratio");
  const expectedRelation = delta.baseline.sha256 === delta.candidate.sha256 ? "identical_bytes" : "different_artifacts_lineage_unproven";
  if (delta.alignment.artifact_relation !== expectedRelation || !String(delta.alignment.semantic_identity_conclusion || "").includes("NOT_CONCLUDED")) throw new Error("Deployment delta lineage boundary is invalid.");

  if (!Array.isArray(delta.target_deltas) || delta.target_count < 2 || delta.target_deltas.length !== delta.target_count) throw new Error("Deployment delta target coverage is invalid.");
  const targetIds = delta.target_deltas.map((target) => target.target_id);
  if (new Set(targetIds).size !== targetIds.length) throw new Error("Deployment delta target IDs are not unique.");
  for (const target of delta.target_deltas) validateTargetDelta(target, rows);
  validateCrossTarget(delta, rows);
  const worst = delta.target_deltas.filter((target) => target.relative_delta != null)
    .sort((left, right) => right.relative_delta - left.relative_delta)[0] || null;
  if ((delta.worst_relative_delta_target_id ?? null) !== (worst?.target_id ?? null)) throw new Error("Deployment delta worst-target identity is invalid.");
  assertNullableNear(delta.worst_relative_delta, worst?.relative_delta ?? null, "worst relative delta");
  if (!String(delta.interpretation_boundary || "").includes("not device measurements")
    || !String(delta.method || "").includes("driver sum must equal")
    || !String(delta.method || "").includes("max(compute, memory) exactly once")) throw new Error("Deployment delta evidence boundary is incomplete.");
  return true;
}

function validateArtifact(artifact, role) {
  if (!artifact || artifact.role !== role || artifact.format !== "tflite" || !/^[a-f0-9]{64}$/.test(artifact.sha256 || "")) throw new Error(`Deployment delta ${role} artifact identity is invalid.`);
  for (const field of ["file_size", "operator_count", "tensor_count", "total_macs", "quantized_compute_mac_ratio", "delegated_mac_ratio"]) {
    if (!Number.isFinite(Number(artifact[field])) || Number(artifact[field]) < 0) throw new Error(`Deployment delta ${role} ${field} is invalid.`);
  }
  if (!Array.isArray(artifact.input_contracts) || !Array.isArray(artifact.output_contracts)) throw new Error(`Deployment delta ${role} I/O contracts are invalid.`);
}

function validateTargetDelta(target, alignmentRows) {
  const alignmentByEntity = new Map(alignmentRows.map((row) => [row.entity_id, row]));
  if (!/^[a-f0-9]{64}$/.test(target.target_profile_sha256 || "")) throw new Error(`Deployment delta target digest is invalid for ${target.target_id}.`);
  assertNear(target.baseline_total_us, componentTotal(target.baseline_components), `${target.target_id} baseline total`);
  assertNear(target.candidate_total_us, componentTotal(target.candidate_components), `${target.target_id} candidate total`);
  for (const [, field] of COMPONENTS) assertNear(target.component_delta[field], Number(target.candidate_components[field]) - Number(target.baseline_components[field]), `${target.target_id} ${field} delta`);
  assertNear(target.component_delta.total_us, componentTotal(target.component_delta), `${target.target_id} component delta total`);
  assertNear(target.signed_delta_us, target.candidate_total_us - target.baseline_total_us, `${target.target_id} signed total`);
  assertNear(target.component_delta.total_us, target.signed_delta_us, `${target.target_id} component conservation`);
  assertNullableNear(target.relative_delta, relativeDelta(target.candidate_total_us, target.baseline_total_us), `${target.target_id} relative delta`);
  if (!Array.isArray(target.drivers) || target.drivers.length !== alignmentRows.length || new Set(target.drivers.map((driver) => driver.entity_id)).size !== alignmentRows.length
    || target.drivers.some((driver) => !alignmentByEntity.has(driver.entity_id))) throw new Error(`Deployment delta driver coverage is invalid for ${target.target_id}.`);
  let positive = 0;
  let negative = 0;
  let absolute = 0;
  let total = 0;
  for (const driver of target.drivers) {
    const alignment = alignmentByEntity.get(driver.entity_id);
    if (driver.relation !== alignment.relation || driver.match_class !== alignment.match_class
      || driver.baseline_op_index !== alignment.baseline_op_index || driver.candidate_op_index !== alignment.candidate_op_index
      || driver.baseline_op_name !== alignment.baseline_op_name || driver.candidate_op_name !== alignment.candidate_op_name) throw new Error(`Deployment delta driver identity is invalid for ${target.target_id}/${driver.entity_id}.`);
    assertNear(driver.baseline_us, componentTotal(driver.baseline_components), `${target.target_id}/${driver.entity_id} baseline component binding`);
    assertNear(driver.candidate_us, componentTotal(driver.candidate_components), `${target.target_id}/${driver.entity_id} candidate component binding`);
    for (const [, field] of COMPONENTS) assertNear(driver.component_delta[field], Number(driver.candidate_components[field]) - Number(driver.baseline_components[field]), `${target.target_id}/${driver.entity_id} ${field} delta`);
    assertNear(driver.signed_delta_us, driver.candidate_us - driver.baseline_us, `${target.target_id}/${driver.entity_id} signed delta`);
    assertNear(driver.component_delta.total_us, componentTotal(driver.component_delta), `${target.target_id}/${driver.entity_id} component total`);
    assertNear(driver.component_delta.total_us, driver.signed_delta_us, `${target.target_id}/${driver.entity_id} component conservation`);
    assertNullableNear(driver.relative_delta, relativeDelta(driver.candidate_us, driver.baseline_us), `${target.target_id}/${driver.entity_id} relative delta`);
    positive += Math.max(0, driver.signed_delta_us);
    negative += Math.min(0, driver.signed_delta_us);
    absolute += Math.abs(driver.signed_delta_us);
    total += driver.signed_delta_us;
  }
  assertNear(target.positive_regression_us, positive, `${target.target_id} positive ledger`);
  assertNear(target.negative_improvement_us, negative, `${target.target_id} negative ledger`);
  assertNear(target.absolute_change_us, absolute, `${target.target_id} absolute ledger`);
  assertNear(target.driver_delta_sum_us, total, `${target.target_id} driver sum`);
  assertNear(target.conservation_error_us, total - target.signed_delta_us, `${target.target_id} conservation error`);
  assertNear(target.conservation_error_us, 0, `${target.target_id} exact conservation`);
  for (const driver of target.drivers) {
    assertNear(driver.absolute_change_share, ratio(Math.abs(driver.signed_delta_us), absolute), `${target.target_id}/${driver.entity_id} absolute share`);
    assertNear(driver.positive_regression_share, ratio(Math.max(0, driver.signed_delta_us), positive), `${target.target_id}/${driver.entity_id} positive share`);
    assertNear(driver.negative_improvement_share, ratio(Math.max(0, -driver.signed_delta_us), -negative), `${target.target_id}/${driver.entity_id} improvement share`);
    assertNear(driver.baseline_contribution_share, ratio(driver.baseline_us, target.baseline_total_us), `${target.target_id}/${driver.entity_id} baseline contribution`);
    assertNear(driver.candidate_contribution_share, ratio(driver.candidate_us, target.candidate_total_us), `${target.target_id}/${driver.entity_id} candidate contribution`);
  }
  const ordered = [...target.drivers].sort((left, right) => Math.abs(right.signed_delta_us) - Math.abs(left.signed_delta_us) || left.entity_id.localeCompare(right.entity_id));
  if (ordered.some((driver, index) => driver.entity_id !== target.drivers[index].entity_id)) throw new Error(`Deployment delta driver order is invalid for ${target.target_id}.`);
  validateDriverRanks(target, "baseline");
  validateDriverRanks(target, "candidate");
  const topRegression = target.drivers.filter((driver) => driver.signed_delta_us > 0).reduce((best, driver) => !best || driver.signed_delta_us >= best.signed_delta_us ? driver : best, null);
  const topImprovement = target.drivers.filter((driver) => driver.signed_delta_us < 0).reduce((best, driver) => !best || driver.signed_delta_us <= best.signed_delta_us ? driver : best, null);
  if ((target.top_regression_entity_id ?? null) !== (topRegression?.entity_id ?? null)
    || (target.top_improvement_entity_id ?? null) !== (topImprovement?.entity_id ?? null)) throw new Error(`Deployment delta top-driver identity is invalid for ${target.target_id}.`);
  assertNear(target.top_regression_delta_us, topRegression?.signed_delta_us || 0, `${target.target_id} top regression`);
  assertNear(target.top_improvement_delta_us, topImprovement?.signed_delta_us || 0, `${target.target_id} top improvement`);
}

function validateCrossTarget(delta, alignmentRows) {
  const alignmentByEntity = new Map(alignmentRows.map((row) => [row.entity_id, row]));
  if (!Array.isArray(delta.cross_target_drivers) || delta.cross_target_drivers.length !== alignmentRows.length
    || new Set(delta.cross_target_drivers.map((row) => row.entity_id)).size !== alignmentRows.length) throw new Error("Deployment delta cross-target driver coverage is invalid.");
  for (const row of delta.cross_target_drivers) {
    const alignment = alignmentByEntity.get(row.entity_id);
    if (!alignment || row.relation !== alignment.relation || row.match_class !== alignment.match_class
      || row.baseline_op_index !== alignment.baseline_op_index || row.candidate_op_index !== alignment.candidate_op_index
      || row.baseline_op_name !== alignment.baseline_op_name || row.candidate_op_name !== alignment.candidate_op_name) throw new Error(`Deployment delta cross-target identity is invalid at ${row.entity_id}.`);
    const drivers = delta.target_deltas.map((target) => target.drivers.find((driver) => driver.entity_id === row.entity_id));
    const values = drivers.map((driver) => driver.signed_delta_us);
    const regressions = values.filter((value) => value > 1e-12).length;
    const improvements = values.filter((value) => value < -1e-12).length;
    const unchanged = values.length - regressions - improvements;
    if (row.regression_target_count !== regressions || row.improvement_target_count !== improvements || row.unchanged_target_count !== unchanged
      || row.consistent_regression !== (regressions === delta.target_count) || row.consistent_improvement !== (improvements === delta.target_count)) throw new Error(`Deployment delta cross-target classification is invalid at ${row.entity_id}.`);
    assertNear(row.min_delta_us, Math.min(...values), `${row.entity_id} minimum delta`);
    assertNear(row.max_delta_us, Math.max(...values), `${row.entity_id} maximum delta`);
    assertNear(row.max_absolute_delta_us, Math.max(...values.map(Math.abs)), `${row.entity_id} maximum absolute delta`);
    assertNear(row.max_absolute_change_share, Math.max(...drivers.map((driver) => driver.absolute_change_share)), `${row.entity_id} maximum absolute share`);
  }
}

function validateDriverRanks(target, side) {
  const indexField = `${side}_op_index`;
  const valueField = `${side}_us`;
  const rankField = `${side}_rank`;
  const ranked = target.drivers.filter((driver) => driver[indexField] != null)
    .sort((left, right) => Number(right[valueField]) - Number(left[valueField]) || Number(left[indexField]) - Number(right[indexField]));
  const expected = new Map(ranked.map((driver, index) => [driver.entity_id, index + 1]));
  for (const driver of target.drivers) {
    if ((driver[rankField] ?? null) !== (expected.get(driver.entity_id) ?? null)) throw new Error(`Deployment delta ${side} rank is invalid for ${target.target_id}/${driver.entity_id}.`);
  }
}

function staticAssignment(op) {
  return op?.xnnpack_supported && !op?.xnnpack_chain_break ? "predicted_delegate" : "predicted_cpu";
}

function componentTotal(components = {}) {
  return COMPONENTS.reduce((sum, [, field]) => sum + Number(components[field] || 0), 0);
}

function relativeDelta(candidate, baseline) {
  return Number(baseline) > 0 ? (Number(candidate) - Number(baseline)) / Number(baseline) : null;
}

function ratio(numerator, denominator) {
  return Number(denominator) > 0 ? Number(numerator) / Number(denominator) : 0;
}

function assertNear(actual, expected, label) {
  const tolerance = Math.max(1e-9, Math.abs(Number(expected)) * 1e-10);
  if (!Number.isFinite(Number(actual)) || Math.abs(Number(actual) - Number(expected)) > tolerance) throw new Error(`Deployment delta ${label} invariant failed.`);
}

function assertNullableNear(actual, expected, label) {
  if (expected == null) {
    if (actual != null) throw new Error(`Deployment delta ${label} should be unavailable.`);
    return;
  }
  assertNear(actual, expected, label);
}

function assertIndexCoverage(indices, count, label) {
  const ordered = [...indices].sort((left, right) => left - right);
  if (ordered.length !== count || ordered.some((value, index) => value !== index)) throw new Error(`Deployment delta ${label} op coverage is invalid.`);
}

function sameArray(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function targetSelector(delta, activeTargetId) {
  const wrap = document.createElement("div");
  wrap.className = "delta-target-selector";
  for (const target of delta.target_deltas) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.deltaTarget = target.target_id;
    button.className = target.target_id === activeTargetId ? "active" : "";
    button.textContent = shortTarget(target.target_label || target.target_id);
    wrap.append(button);
  }
  return wrap;
}

function alignmentFilters(delta, alignmentFilter) {
  const wrap = document.createElement("div");
  wrap.className = "delta-target-selector";
  for (const [id, label, count] of [["all", "All", delta.alignment_rows.length], ["matched", "Matched", delta.alignment.matched_op_count], ["added", "Added", delta.alignment.added_op_count], ["removed", "Removed", delta.alignment.removed_op_count]]) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.deltaFilter = id;
    button.className = id === alignmentFilter ? "active" : "";
    button.textContent = `${label} ${count}`;
    wrap.append(button);
  }
  return wrap;
}

function methodDetails(delta) {
  const details = document.createElement("details");
  details.className = "delta-method-details";
  const summary = document.createElement("summary");
  summary.textContent = "Method and evidence boundary";
  details.append(summary, cell("p", delta.alignment.method), cell("p", delta.method), cell("p", delta.interpretation_boundary));
  return details;
}

function contractLine(label, baseline, candidate, changed) {
  const line = document.createElement("div");
  line.className = `delta-contract-line ${changed ? "changed" : "stable"}`;
  line.append(cell("strong", label), cell("span", baseline.join(", ") || "none"), cell("i", "->"), cell("span", candidate.join(", ") || "none"), cell("b", changed ? "changed" : "stable"));
  return line;
}

function transitionCell(changed, text) {
  return cell("td", text, changed ? "delta-transition" : "");
}

function opTransitionCell(driver) {
  return identityCell(opLabel(driver.baseline_op_index, driver.baseline_op_name) || "-", opLabel(driver.candidate_op_index, driver.candidate_op_name) || "-");
}

function opLabel(index, name) {
  return index == null ? "" : `#${padOp(index)} ${name || "UNKNOWN"}`;
}

function shapesLabel(shapes) {
  return Array.isArray(shapes) ? shapes.map((shape) => `[${shape.join("x")}]`).join(" ") : "";
}

function largestComponent(components = {}) {
  return COMPONENTS.map(([label, field]) => ({ label, field, value: Math.abs(Number(components[field] || 0)) }))
    .sort((left, right) => right.value - left.value || left.field.localeCompare(right.field))[0];
}

function metric(label, value, detail) {
  const node = document.createElement("div");
  node.className = "delta-metric";
  node.append(cell("span", label), cell("strong", value), cell("small", detail));
  return node;
}

function identityCell(primary, detail) {
  const node = document.createElement("td");
  node.className = "delta-identity-cell";
  node.append(cell("strong", primary));
  if (detail) node.append(cell("span", detail));
  return node;
}

function signedTimeCell(value, relative) {
  return identityCell(formatSignedUs(value), formatSignedPercent(relative));
}

function tableHead(labels) {
  const head = document.createElement("thead");
  const row = document.createElement("tr");
  for (const label of labels) row.append(cell("th", label));
  head.append(row);
  return head;
}

function formatSignedPercent(value) {
  if (value == null) return "N/A";
  const number = Number(value);
  return `${number > 0 ? "+" : ""}${formatPercent(number)}`;
}

function formatSignedPoints(value) {
  const number = Number(value || 0) * 100;
  return `${number > 0 ? "+" : ""}${number.toFixed(1)} pp`;
}

function formatSignedUs(value) {
  const number = Number(value || 0);
  return `${number > 0 ? "+" : number < 0 ? "-" : ""}${formatUs(Math.abs(number))}`;
}

function formatSignedBytes(value) {
  const number = Number(value || 0);
  return `${number > 0 ? "+" : number < 0 ? "-" : ""}${formatBytes(Math.abs(number))}`;
}

function formatSignedInteger(value) {
  const number = Number(value || 0);
  return `${number > 0 ? "+" : ""}${formatNumber(number)}`;
}

function formatSignedNumber(value) {
  const number = Number(value || 0);
  return `${number > 0 ? "+" : ""}${formatNumber(number)}`;
}

function signedClass(value) {
  return Number(value) > 1e-12 ? "delta-regression" : Number(value) < -1e-12 ? "delta-improvement" : "";
}

function shortTarget(label) {
  return String(label || "-")
    .replace("Android mid-range / ", "")
    .replace("Raspberry Pi 4 / ", "")
    .replace("Browser / ", "")
    .replace("Desktop / ", "");
}

function shortHash(value) {
  const text = String(value || "");
  return text.length > 16 ? `${text.slice(0, 8)}...${text.slice(-8)}` : text;
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
