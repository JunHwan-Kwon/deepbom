import { buildExecutionPlacementEvidence } from "./execution-placement-evidence.js";
import { formatBytes, formatNumber } from "./format.js";

export function renderExecutionPlacementView(root, analysis, runtimeEvidence = null, { doc = globalThis.document } = {}) {
  if (!root) return;
  if (!analysis) {
    root.hidden = true;
    root.replaceChildren();
    root.onclick = null;
    return;
  }
  root.hidden = false;
  try {
    const evidence = buildExecutionPlacementEvidence(analysis, runtimeEvidence);
    root.dataset.placementFormat = evidence.format;
    root.dataset.placementState = evidence.runtime_observation.status;
    root.replaceChildren(
      placementHeader(doc, evidence),
      ...(evidence.banner ? [placementBanner(doc, evidence.banner)] : []),
      placementTopology(doc, evidence),
      placementBody(doc, evidence),
      placementActions(doc, actionsFor(evidence, analysis)),
    );
    root.onclick = (event) => handleAction(event, doc);
  } catch (error) {
    root.dataset.placementFormat = String(analysis.format || "unknown").toLowerCase();
    root.dataset.placementState = "invalid";
    root.replaceChildren(
      placementHeader(doc, {
        title: "Execution Placement Evidence",
        subtitle: "Evidence rejected",
        state: "INVALID",
        tone: "missing",
      }),
      placementBanner(doc, error instanceof Error ? error.message : String(error)),
    );
    root.onclick = null;
  }
}

function placementHeader(doc, evidence) {
  const head = el(doc, "div", "mini-panel-head execution-placement-head");
  const title = el(doc, "div");
  title.append(el(doc, "h3", "", evidence.title), el(doc, "span", "", evidence.subtitle));
  head.append(title, el(doc, "em", `placement-state ${evidence.tone}`, evidence.state));
  return head;
}

function placementBanner(doc, text) {
  return el(doc, "p", "execution-placement-banner", text);
}

function placementTopology(doc, evidence) {
  const root = el(doc, "section", "execution-placement-topology");
  root.setAttribute("aria-label", "Deployment evidence topology");
  const heading = el(doc, "div", "execution-placement-topology-head");
  const title = el(doc, "div");
  title.append(
    el(doc, "strong", "", "Deployment evidence topology"),
    el(doc, "span", "", "Claim progression, not a physical-routing observation"),
  );
  heading.append(title);

  const track = el(doc, "div", "execution-placement-levels");
  const relations = ["source assessment", "configuration binding", "runtime observation"];
  for (const [index, item] of evidence.levels.entries()) {
    if (index) track.append(placementRelation(doc, relations[index - 1], index === 3));
    const node = doc.createElement("details");
    node.className = `execution-placement-level ${item.tone}`;
    node.open = true;
    node.dataset.placementLevel = item.id;
    const summary = doc.createElement("summary");
    const identity = el(doc, "div", "execution-placement-level-identity");
    identity.append(
      el(doc, "span", "execution-placement-level-index", String(index + 1).padStart(2, "0")),
      el(doc, "span", "execution-placement-level-label", item.label),
    );
    summary.append(identity, el(doc, "strong", "", item.state));
    node.append(
      summary,
      el(doc, "small", "execution-placement-class", item.evidence_class),
      el(doc, "p", "", item.detail),
    );
    track.append(node);
  }
  root.append(heading, track);
  return root;
}

function placementRelation(doc, label, externalBoundary) {
  const relation = el(doc, "div", `execution-placement-relation${externalBoundary ? " external" : ""}`);
  relation.setAttribute("aria-hidden", "true");
  relation.append(el(doc, "i"), el(doc, "span", "", label));
  return relation;
}

function placementBody(doc, evidence) {
  const root = el(doc, "div", "execution-placement-body");
  root.append(el(doc, "strong", "execution-placement-flow-label", evidence.flow.label));
  if (evidence.flow.segments.length) {
    const flow = el(doc, "div", "execution-placement-flow");
    flow.tabIndex = 0;
    flow.setAttribute("role", "region");
    flow.setAttribute("aria-label", `${evidence.flow.label}; horizontal scroll region for contiguous segments`);
    for (const segment of evidence.flow.segments) {
      const node = el(doc, "div", `execution-placement-segment ${segment.tone}`);
      node.style.setProperty("--placement-share", String(Math.max(1, segment.item_count)));
      node.title = `${segment.label}: ${segment.item_count} item${segment.item_count === 1 ? "" : "s"}; ${segment.start_item_id} to ${segment.end_item_id}`;
      node.append(
        el(doc, "strong", "", segment.label),
        el(doc, "span", "", `${segment.item_count} item${segment.item_count === 1 ? "" : "s"} / #${segment.start_item_id}${segment.start_item_id === segment.end_item_id ? "" : `-#${segment.end_item_id}`}`),
      );
      flow.append(node);
    }
    root.append(flow, el(doc, "small", "execution-placement-scroll-hint", "Horizontal scrolling remains available when the segment flow extends beyond the panel"));
  } else {
    root.append(el(doc, "p", "execution-placement-empty", evidence.interpretation_boundary));
  }
  if (evidence.portfolios.length) {
    const portfolios = el(doc, "div", "execution-placement-portfolios");
    for (const item of evidence.portfolios) {
      const node = el(doc, "div", `execution-placement-portfolio ${item.tone}`);
      node.append(
        el(doc, "span", "", item.label),
        el(doc, "strong", "", `${item.candidate_count}/${item.total_count}`),
        el(doc, "small", "", `${item.detail} - ${item.evidence_class}`),
        placementPortfolioBar(doc, item),
      );
      portfolios.append(node);
    }
    root.append(portfolios);
  }
  if (evidence.static_profiles.length) root.append(staticProfileExplorer(doc, evidence.static_profiles));
  if (evidence.configuration_preflights.length) root.append(configurationPreflightView(doc, evidence.configuration_preflights));
  if (evidence.flow.segments.length) root.append(el(doc, "p", "execution-placement-note", evidence.interpretation_boundary));
  return root;
}

function configurationPreflightView(doc, preflights) {
  const root = el(doc, "section", "placement-configuration-preflights");
  for (const preflight of preflights) {
    const details = doc.createElement("details");
    details.className = "placement-configuration-preflight";
    const summary = doc.createElement("summary");
    const identity = el(doc, "span", "");
    identity.append(el(doc, "strong", "", preflight.label), el(doc, "small", "", preflight.execution_path || "configuration unbound"));
    summary.append(identity, el(doc, "em", "", preflight.status));
    details.append(summary);
    const metrics = el(doc, "div", "placement-profile-metrics placement-preflight-metrics");
    metrics.append(
      placementMetric(doc, "Blocking", preflight.blocking_issue_count, preflight.blocking_issue_count ? "excluded" : "candidate"),
      placementMetric(doc, "Unresolved", preflight.unresolved_issue_count, preflight.unresolved_issue_count ? "unresolved" : "candidate"),
      placementMetric(doc, "Build profile", preflight.build_profile_sha256 ? preflight.build_profile_sha256.slice(0, 12) : "unbound", "neutral"),
      placementMetric(doc, "Shape-cost points", preflight.profile_cost_scenario_count, preflight.profile_cost_status === "assessed" ? "candidate" : "unresolved"),
      placementMetric(doc, "Plan in browser", "prohibited", "excluded"),
    );
    details.append(metrics);
    if (preflight.optimization_profile_cost?.scenarios?.length) {
      const table = el(doc, "div", "placement-preflight-issues");
      for (const scenario of preflight.optimization_profile_cost.scenarios) {
        const row = el(doc, "div", "placement-preflight-issue");
        row.append(
          el(doc, "strong", "", `${scenario.profile_id} / ${scenario.profile_point}`),
          el(doc, "span", "", `MACs ${scenario.total_macs_decimal ?? "not assessed"}; live payload ${scenario.peak_live_payload_bytes == null ? scenario.peak_live_payload_bytes_decimal ?? "not assessed" : formatBytes(scenario.peak_live_payload_bytes)}`),
        );
        table.append(row);
      }
      details.append(table);
    }
    if (preflight.issues.length) {
      const list = el(doc, "div", "placement-preflight-issues");
      for (const issue of preflight.issues) {
        const row = el(doc, "div", `placement-preflight-issue ${String(issue.severity || "").toLowerCase()}`);
        row.append(
          el(doc, "strong", "", `${issue.severity} / ${issue.id}`),
          el(doc, "span", "", issue.observation),
          el(doc, "small", "", issue.action),
        );
        list.append(row);
      }
      details.append(list);
    }
    details.append(el(doc, "p", "execution-placement-note", preflight.interpretation_boundary));
    root.append(details);
  }
  return root;
}

function staticProfileExplorer(doc, profiles) {
  const root = el(doc, "section", "placement-profile-explorer");
  const comparison = placementProfileComparison(doc, profiles);
  if (comparison) root.append(comparison);
  const head = el(doc, "div", "placement-profile-head");
  const title = el(doc, "div");
  title.append(
    el(doc, "strong", "", "Detailed independent projection"),
    el(doc, "span", "", "Inspect one source-backed backend ledger without combining provider priority"),
  );
  const field = el(doc, "label", "placement-profile-select placement-profile-detail-select");
  field.append(el(doc, "span", "", "Backend"));
  const select = doc.createElement("select");
  select.setAttribute("aria-label", "Static backend eligibility profile");
  for (const profile of profiles) {
    const option = doc.createElement("option");
    option.value = profile.profile_id;
    option.textContent = profile.label;
    select.append(option);
  }
  const preferred = preferredProfile(profiles);
  select.value = preferred.profile_id;
  field.append(select);
  head.append(title, field);
  const body = el(doc, "div", "placement-profile-detail");
  const render = () => {
    const profile = profiles.find((item) => item.profile_id === select.value) || profiles[0];
    renderStaticProfile(doc, body, profile);
  };
  select.addEventListener("change", render);
  render();
  root.append(head, body);
  return root;
}

function placementProfileComparison(doc, profiles) {
  const acceleratorProfiles = profiles.filter((profile) => profileClass(profile) === "accelerator");
  const cpuProfiles = profiles.filter((profile) => profileClass(profile) === "cpu");
  if (!acceleratorProfiles.length || !cpuProfiles.length) return null;

  const root = el(doc, "section", "placement-profile-comparison");
  const head = el(doc, "div", "placement-profile-comparison-head");
  head.append(
    el(doc, "strong", "", "Accelerator and CPU eligibility comparison"),
    el(doc, "span", "", "Independent source projections over one graph and tensor ledger; no provider priority or timing is inferred"),
  );
  const grid = el(doc, "div", "placement-profile-comparison-grid");
  grid.append(
    placementComparisonCard(doc, "Accelerator backend", acceleratorProfiles, preferredProfile(acceleratorProfiles)),
    placementComparisonCard(doc, "CPU baseline", cpuProfiles, preferredCpuProfile(cpuProfiles)),
  );
  root.append(head, grid, el(doc, "p", "execution-placement-note", "Logical boundary exposure is a serialized-shape edge sum, not observed CPU-to-accelerator transfer. GPU roofline, generated shader or kernel, occupancy, memory plan, and latency remain NOT ASSESSED."));
  return root;
}

function placementComparisonCard(doc, label, profiles, preferred) {
  const root = el(doc, "article", "placement-profile-comparison-card");
  root.dataset.profileClass = label.startsWith("CPU") ? "cpu" : "accelerator";
  const field = el(doc, "label", "placement-profile-select placement-profile-comparison-select");
  field.append(el(doc, "span", "", label));
  const select = doc.createElement("select");
  select.setAttribute("aria-label", `${label} eligibility profile`);
  for (const profile of profiles) {
    const option = doc.createElement("option");
    option.value = profile.profile_id;
    option.textContent = profile.label;
    select.append(option);
  }
  select.value = preferred.profile_id;
  field.append(select);
  const summary = el(doc, "dl", "placement-profile-comparison-summary");
  const render = () => renderPlacementComparisonSummary(doc, summary,
    profiles.find((profile) => profile.profile_id === select.value) || profiles[0]);
  select.addEventListener("change", render);
  render();
  root.append(field, summary);
  return root;
}

function renderPlacementComparisonSummary(doc, root, profile) {
  const counts = profile.state_counts;
  const payload = profile.boundary_payload;
  const precision = Object.entries(profile.workload_envelope?.total?.output_dtype_reference_counts || {})
    .map(([dtype, count]) => `${dtype} ${formatNumber(count)}`)
    .join(" / ") || "No serialized output dtype references";
  const predicateCount = new Set(profile.rows.flatMap((row) => row.unresolved_predicates || [])).size;
  root.replaceChildren(
    comparisonDatum(doc, "Conditionally eligible", `${formatNumber(counts.CONDITIONALLY_ELIGIBLE)}/${formatNumber(profile.op_count)} ops`),
    comparisonDatum(doc, "Definite / unresolved", `${formatNumber(counts.DEFINITE_EXCLUSION)} / ${formatNumber(counts.UNRESOLVED)}`),
    comparisonDatum(doc, "Independent segments", formatNumber(profile.segment_count)),
    comparisonDatum(doc, "Boundary exposure", payload.summed_edge_payload_bytes == null
      ? `${formatBytes(payload.assessed_edge_payload_bytes)} assessed; ${formatNumber(payload.unassessed_edge_count)} unassessed edges`
      : `${formatBytes(payload.summed_edge_payload_bytes)} logical`),
    comparisonDatum(doc, "Artifact output dtypes", precision),
    comparisonDatum(doc, "Unresolved condition types", formatNumber(predicateCount)),
  );
}

function comparisonDatum(doc, label, value) {
  const row = doc.createElement("div");
  row.append(el(doc, "dt", "", label), el(doc, "dd", "", value));
  return row;
}

function renderStaticProfile(doc, root, profile) {
  const counts = profile.state_counts;
  const payload = profile.boundary_payload;
  const metrics = el(doc, "div", "placement-profile-metrics");
  metrics.append(
    placementMetric(doc, "Conditionally eligible", counts.CONDITIONALLY_ELIGIBLE, "candidate"),
    placementMetric(doc, "Definite exclusion", counts.DEFINITE_EXCLUSION, "excluded"),
    placementMetric(doc, "Unresolved", counts.UNRESOLVED, "unresolved"),
    placementMetric(doc, "State boundaries", profile.boundary_edge_count, "boundary"),
    placementMetric(doc, "Logical exposure", payload.summed_edge_payload_bytes == null
      ? `${formatBytes(payload.assessed_edge_payload_bytes)} assessed`
      : formatBytes(payload.summed_edge_payload_bytes), payload.unassessed_edge_count ? "unresolved" : "neutral"),
  );
  const flow = el(doc, "div", "execution-placement-flow placement-profile-flow");
  flow.tabIndex = 0;
  flow.setAttribute("role", "region");
  flow.setAttribute("aria-label", `${profile.label} independent static segment projection`);
  for (const segment of profile.segments) {
    const tone = segment.state === "CONDITIONALLY_ELIGIBLE" ? "candidate"
      : segment.state === "DEFINITE_EXCLUSION" ? "fallback" : "missing";
    const node = el(doc, "div", `execution-placement-segment ${tone}`);
    node.style.setProperty("--placement-share", String(Math.max(1, segment.op_count)));
    node.title = `${segment.state}; ${segment.op_count} op(s); #${segment.start_op_index} to #${segment.end_op_index}`;
    node.append(
      el(doc, "strong", "", stateLabel(segment.state)),
      el(doc, "span", "", `${segment.op_count} op${segment.op_count === 1 ? "" : "s"} / #${segment.start_op_index}${segment.start_op_index === segment.end_op_index ? "" : `-#${segment.end_op_index}`}`),
    );
    flow.append(node);
  }
  const details = doc.createElement("details");
  details.className = "placement-boundary-ledger";
  const summary = doc.createElement("summary");
  summary.textContent = `${formatNumber(profile.boundary_edge_count)} state-boundary tensor edge${profile.boundary_edge_count === 1 ? "" : "s"}`;
  details.append(summary);
  if (profile.boundary_edges.length) {
    const tableWrap = el(doc, "div", "placement-boundary-table-wrap");
    const table = doc.createElement("table");
    const thead = doc.createElement("thead");
    const header = doc.createElement("tr");
    for (const label of ["Tensor", "Producer", "Consumer", "Transition", "Logical payload"]) header.append(el(doc, "th", "", label));
    thead.append(header);
    const tbody = doc.createElement("tbody");
    for (const edge of profile.boundary_edges) {
      const row = doc.createElement("tr");
      const values = [
        `T${edge.tensor_index}${edge.tensor_name ? ` ${edge.tensor_name}` : ""}`,
        `#${edge.producer_op_index}`,
        `#${edge.consumer_op_index}`,
        `${stateLabel(edge.producer_state)} -> ${stateLabel(edge.consumer_state)}`,
        edge.logical_payload_bytes == null ? `Not assessed: ${edge.logical_payload_reason}` : formatBytes(edge.logical_payload_bytes),
      ];
      for (const value of values) row.append(el(doc, "td", "", value));
      tbody.append(row);
    }
    table.append(thead, tbody);
    tableWrap.append(table);
    details.append(tableWrap);
  } else {
    details.append(el(doc, "p", "execution-placement-empty", "No cross-state tensor edge is present in this profile."));
  }
  const workload = workloadEnvelopeView(doc, profile.workload_envelope);
  const conditions = conditionLedgerView(doc, profile);
  root.replaceChildren(
    metrics,
    el(doc, "strong", "execution-placement-flow-label", "Independent contiguous eligibility segments"),
    flow,
    workload,
    ...(conditions ? [conditions] : []),
    details,
    el(doc, "p", "execution-placement-note", profile.interpretation_boundary),
  );
}

function conditionLedgerView(doc, profile) {
  const grouped = new Map();
  for (const row of profile.rows || []) {
    for (const [kind, values] of [
      ["Unresolved condition", row.unresolved_predicates],
      ["Definite exclusion", row.reason_codes],
    ]) {
      for (const value of values || []) {
        const code = String(value || "").trim();
        if (!code) continue;
        const key = `${kind}:${code}`;
        const entry = grouped.get(key) || { kind, code, op_indices: [] };
        entry.op_indices.push(row.op_index);
        grouped.set(key, entry);
      }
    }
  }
  const rows = [...grouped.values()].sort((left, right) => {
    const kindOrder = left.kind === right.kind ? 0 : left.kind === "Unresolved condition" ? -1 : 1;
    return kindOrder || right.op_indices.length - left.op_indices.length || left.code.localeCompare(right.code);
  });
  if (!rows.length) return null;

  const unresolvedCount = rows.filter((row) => row.kind === "Unresolved condition").length;
  const exclusionCount = rows.length - unresolvedCount;
  const details = doc.createElement("details");
  details.className = "placement-condition-ledger";
  const summary = doc.createElement("summary");
  summary.textContent = `${formatNumber(unresolvedCount)} unresolved condition type${unresolvedCount === 1 ? "" : "s"} / ${formatNumber(exclusionCount)} exclusion reason type${exclusionCount === 1 ? "" : "s"}`;
  details.append(summary);

  const tableWrap = el(doc, "div", "placement-boundary-table-wrap placement-condition-table-wrap");
  const table = doc.createElement("table");
  const thead = doc.createElement("thead");
  const header = doc.createElement("tr");
  for (const label of ["Evidence role", "Condition or reason", "Affected ops", "Operator indices"]) header.append(el(doc, "th", "", label));
  thead.append(header);
  const tbody = doc.createElement("tbody");
  for (const item of rows) {
    const row = doc.createElement("tr");
    const indices = item.op_indices.slice(0, 12).map((index) => `#${index}`).join(", ");
    const suffix = item.op_indices.length > 12 ? ` +${item.op_indices.length - 12} more` : "";
    for (const value of [item.kind, item.code, formatNumber(item.op_indices.length), `${indices}${suffix}`]) {
      row.append(el(doc, "td", "", value));
    }
    tbody.append(row);
  }
  table.append(thead, tbody);
  tableWrap.append(table);
  details.append(
    tableWrap,
    el(doc, "p", "execution-placement-note", "Conditions are grouped from the complete per-op static ledger. An unresolved condition is not support, exclusion, selected-build acceptance, or runtime assignment."),
  );
  return details;
}

function workloadEnvelopeView(doc, envelope) {
  const details = doc.createElement("details");
  details.className = "placement-workload-envelope";
  details.open = true;
  const summary = doc.createElement("summary");
  summary.textContent = "Artifact workload envelope";
  details.append(summary);
  const wrap = el(doc, "div", "placement-boundary-table-wrap");
  const table = doc.createElement("table");
  const thead = doc.createElement("thead");
  const header = doc.createElement("tr");
  for (const label of ["Eligibility state", "Ops", "MACs", "Logical op bytes", "MAC-equivalent ops/B", "Output dtypes"]) header.append(el(doc, "th", "", label));
  thead.append(header);
  const tbody = doc.createElement("tbody");
  for (const state of ["CONDITIONALLY_ELIGIBLE", "DEFINITE_EXCLUSION", "UNRESOLVED"]) {
    const row = envelope.by_state[state];
    const tr = doc.createElement("tr");
    const values = [
      stateLabel(state),
      formatNumber(row.op_count),
      completeOrAssessed(row.complete_macs, row.assessed_macs, row.assessed_mac_op_count, row.op_count),
      completeOrAssessed(row.complete_logical_bytes, row.assessed_logical_bytes, row.assessed_logical_byte_op_count, row.op_count, true),
      row.mac_equivalent_ops_per_logical_byte_decimal == null ? `Not assessed: ${row.intensity_status}` : row.mac_equivalent_ops_per_logical_byte_decimal,
      Object.entries(row.output_dtype_reference_counts).map(([dtype, count]) => `${dtype} ${count}`).join(" / ") || "none",
    ];
    for (const value of values) tr.append(el(doc, "td", "", value));
    tbody.append(tr);
  }
  table.append(thead, tbody);
  wrap.append(table);
  details.append(
    wrap,
    el(doc, "p", "execution-placement-note", `${envelope.interpretation_boundary} Backend cost model: ${envelope.backend_cost_model.reason}`),
  );
  return details;
}

function completeOrAssessed(complete, assessed, assessedCount, totalCount, bytes = false) {
  const value = bytes ? formatBytes : formatNumber;
  return complete == null
    ? `${value(assessed)} assessed (${formatNumber(assessedCount)}/${formatNumber(totalCount)} ops)`
    : value(complete);
}

function preferredProfile(profiles) {
  const rank = ["tflite_gpu", "webgpu", "directml", "qnn", "coreml", "nnapi", "tflite_nnapi", "xnnpack", "wasm_cpu"];
  return [...profiles].sort((left, right) => {
    const leftRank = rank.indexOf(String(left.profile_id).toLowerCase());
    const rightRank = rank.indexOf(String(right.profile_id).toLowerCase());
    return (leftRank < 0 ? rank.length : leftRank) - (rightRank < 0 ? rank.length : rightRank);
  })[0];
}

function preferredCpuProfile(profiles) {
  const rank = ["xnnpack_cpu", "xnnpack", "wasm_cpu", "cpuexecutionprovider"];
  return [...profiles].sort((left, right) => {
    const leftRank = rank.indexOf(String(left.profile_id).toLowerCase());
    const rightRank = rank.indexOf(String(right.profile_id).toLowerCase());
    return (leftRank < 0 ? rank.length : leftRank) - (rightRank < 0 ? rank.length : rightRank);
  })[0];
}

function profileClass(profile) {
  const id = `${profile?.profile_id || ""} ${profile?.label || ""}`.toLowerCase();
  if (/(gpu|directml|webgpu|webnn|nnapi|qnn|coreml|tensorrt|cuda|rocm|metal|vulkan|opencl)/.test(id)) return "accelerator";
  if (/(cpu|xnnpack|wasm)/.test(id)) return "cpu";
  return "other";
}

function placementMetric(doc, label, value, tone) {
  const node = el(doc, "div", `placement-profile-metric ${tone}`);
  node.append(el(doc, "span", "", label), el(doc, "strong", "", typeof value === "number" ? formatNumber(value) : value));
  return node;
}

function stateLabel(state) {
  if (state === "CONDITIONALLY_ELIGIBLE") return "Conditionally eligible";
  if (state === "DEFINITE_EXCLUSION") return "Definite exclusion";
  return "Unresolved";
}

function placementPortfolioBar(doc, item) {
  const track = el(doc, "span", "execution-placement-portfolio-track");
  const fill = el(doc, "i");
  const ratio = item.total_count > 0 ? item.candidate_count / item.total_count : 0;
  fill.style.setProperty("--placement-ratio", String(Math.max(0, Math.min(1, ratio))));
  track.append(fill);
  track.setAttribute("aria-hidden", "true");
  return track;
}

function placementActions(doc, actions) {
  const root = el(doc, "div", "execution-placement-actions");
  for (const item of actions) {
    const button = el(doc, "button", "secondary-action", item.label);
    button.type = "button";
    button.dataset.placementAction = item.id;
    root.append(button);
  }
  return root;
}

function actionsFor(evidence, analysis) {
  if (evidence.format === "tflite") return [
    action("Open detailed flow", "tflite-detail"),
    !analysis.tflite_delegate_compatibility_evidence && action("Load GPU / NNAPI source ledger", "source"),
    action("Import runtime evidence", "runtime"),
  ].filter(Boolean);
  if (evidence.format === "onnx") return [
    action("Open EP evidence", "graph-detail"),
    !analysis.ort_compatibility_evidence && action("Load ORT source ledger", "source"),
    action("Import ORT profile", "runtime"),
  ].filter(Boolean);
  if (evidence.format === "coreml") return [
    action("Open placement evidence", "graph-detail"),
    action(evidence.flow.evidence_basis === "ANTICIPATED_MLCOMPUTEPLAN" ? "Replace compute plan" : "Import compute plan", "runtime"),
  ];
  if (evidence.format === "gguf") return [
    action("Open capability boundary", "capability"),
    action(evidence.levels[2].state === "BOUND" ? "Replace runtime manifest" : "Import runtime manifest", "runtime"),
  ];
  return [action("Open capability boundary", "capability")];
}

function handleAction(event, doc) {
  const actionId = event.target.closest("[data-placement-action]")?.dataset.placementAction;
  if (!actionId) return;
  if (actionId === "source") return doc?.getElementById("runDeepBom")?.click();
  if (actionId === "runtime") return doc?.getElementById("runtimeAssignmentInput")?.click();
  if (actionId === "tflite-detail") return doc?.querySelector('[data-evidence-stage="deployment"] button')?.click();
  if (actionId === "graph-detail") {
    doc?.querySelector('[data-workflow-step="graph"]')?.click();
    globalThis.setTimeout(() => doc?.querySelector('[data-explorer-tab="kernels"]')?.click(), 0);
    return;
  }
  const capability = doc?.getElementById("formatCapabilityPanel");
  if (capability) {
    capability.open = true;
    capability.scrollIntoView({ block: "start", behavior: "smooth" });
  }
}

function action(label, id) { return { label, id }; }
function el(doc, tag, className = "", text = null) {
  const node = doc.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = String(text);
  return node;
}
