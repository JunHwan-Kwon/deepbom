function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function number(value, digits = 1) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toLocaleString(undefined, { maximumFractionDigits: digits }) : "-";
}

function duration(us) {
  const value = Number(us);
  if (!Number.isFinite(value)) return "-";
  if (value >= 1000) return `${number(value / 1000, 2)} ms`;
  return `${number(value, 1)} us`;
}

function allocationLabel(row) {
  return row?.allocation_class === "exclusive_isolation_candidate"
    ? "Isolation candidate"
    : "Full-set baseline";
}

function runtimePartition(runtimeEvidence) {
  return runtimeEvidence?.selector_context?.invocation?.resource_partition || null;
}

function evidenceSummary(partition, row) {
  if (!partition) return {
    label: "STATIC SCENARIO",
    tone: "estimated",
    detail: "No imported affinity or cpuset observation",
  };
  const requested = Array.isArray(partition.requested_cpu_ids) ? partition.requested_cpu_ids : [];
  const online = Array.isArray(partition.online_cpu_ids) ? partition.online_cpu_ids : [];
  const selectedAiCores = Number(row?.ai_assigned_core_count);
  const selectedSystemCores = Number(row?.system_core_count);
  const exclusive = partition.exclusive_isolation_status === "observed_cgroup_v2_isolated_partition";
  const affinity = partition.affinity_status === "observed_all_sampled_threads_within_requested_set";
  const mismatches = [];
  if (requested.length !== selectedAiCores) {
    mismatches.push(`selected scenario allows ${selectedAiCores} AI core${selectedAiCores === 1 ? "" : "s"}, imported mask has ${requested.length}`);
  }
  if (online.length && online.length !== selectedSystemCores) {
    mismatches.push(`selected ${selectedSystemCores}-core system, imported host reports ${online.length} online CPUs`);
  }
  if (mismatches.length) return {
    label: "IMPORTED CPU SET DOES NOT MATCH SCENARIO",
    tone: "warn",
    detail: `${exclusive ? "Observed isolated cpuset" : affinity ? "Observed affinity" : "Declared CPU set"} [${requested.join(", ") || "unresolved"}]; ${mismatches.join("; ")}`,
  };
  if (exclusive) return {
    label: "OBSERVED ISOLATED CPUSET",
    tone: "observed",
    detail: `CPU set ${(partition.observed_effective_cpu_ids || partition.requested_cpu_ids || []).join(", ") || "unresolved"}`,
  };
  if (affinity) return {
    label: "OBSERVED AFFINITY",
    tone: "observed",
    detail: `Threads remained within CPUs ${(partition.requested_cpu_ids || []).join(", ") || "unresolved"}; exclusivity not established`,
  };
  return {
    label: "DECLARED CPU SET",
    tone: "warn",
    detail: "A runtime CPU-set request exists, but sampled-thread affinity was not closed",
  };
}

export function createCoreIsolationController({ elements, getContext }) {
  let selectedSystemCores = null;
  let selectedAiCores = null;

  function selectScenario(systemCores, aiCores) {
    selectedSystemCores = Number(systemCores);
    selectedAiCores = Number(aiCores);
    render();
  }

  function renderControls(options, rows) {
    const root = elements.controls;
    root.replaceChildren();
    const systemGroup = element("label", "core-isolation-field");
    systemGroup.append(element("span", null, "System variant"));
    const select = element("select", "core-isolation-system-select");
    select.setAttribute("aria-label", "System core-count variant");
    for (const count of options) select.append(new Option(`${count} cores`, String(count)));
    select.value = String(selectedSystemCores);
    select.disabled = options.length < 2;
    select.addEventListener("change", () => {
      const system = Number(select.value);
      const candidates = rows.filter((row) => Number(row.system_core_count) === system);
      const preferred = Math.max(1, Math.floor(system / 2));
      selectScenario(system, candidates.some((row) => Number(row.ai_assigned_core_count) === preferred) ? preferred : 1);
    });
    systemGroup.append(select);

    const aiGroup = element("fieldset", "core-isolation-field core-isolation-ai-field");
    aiGroup.append(element("legend", null, "AI-assigned cores"));
    const segmented = element("div", "core-isolation-segments");
    for (const row of rows.filter((candidate) => Number(candidate.system_core_count) === selectedSystemCores)) {
      const assigned = Number(row.ai_assigned_core_count);
      const button = element("button", assigned === selectedAiCores ? "active" : "", String(assigned));
      button.type = "button";
      button.dataset.aiCores = String(assigned);
      button.setAttribute("aria-pressed", String(assigned === selectedAiCores));
      button.title = `${assigned} AI core${assigned === 1 ? "" : "s"}; ${row.housekeeping_core_count} housekeeping`;
      button.addEventListener("click", () => selectScenario(selectedSystemCores, assigned));
      segmented.append(button);
    }
    aiGroup.append(segmented);
    root.append(systemGroup, aiGroup);
  }

  function renderSummary(row, analysis, runtimeEvidence) {
    const root = elements.summary;
    root.replaceChildren();
    const evidence = evidenceSummary(runtimePartition(runtimeEvidence), row);
    const metrics = [
      ["AI / housekeeping", `${row.ai_assigned_core_count} / ${row.housekeeping_core_count}`],
      ["Allocation", allocationLabel(row)],
      ["INT8 / FP32 ceiling", `${number(row.int8_issue_ceiling_gops)} / ${number(row.fp32_issue_ceiling_gops)} GOPS`],
      ["Shared BW ceiling", `${number(row.shared_memory_bandwidth_ceiling_gbps)} GB/s`],
      ["Theoretical floor", duration(row.theoretical_roofline_floor_us)],
      ["Steady estimate", duration(row.steady_state_estimate_us)],
      ["Cold estimate", duration(row.cold_start_estimate_us)],
      ["Static ratio vs 1 core", `${number(row.estimate_ratio_vs_one_ai_core, 2)}x`],
    ];
    for (const [label, value] of metrics) {
      const item = element("div", "core-isolation-metric");
      item.append(element("span", null, label), element("strong", null, value));
      root.append(item);
    }
    const evidenceCard = element("div", `core-isolation-evidence ${evidence.tone}`);
    evidenceCard.append(element("strong", null, evidence.label), element("span", null, evidence.detail));
    root.append(evidenceCard);
    elements.status.textContent = analysis.evidence_class || "ESTIMATED";
    elements.status.dataset.tone = evidence.tone;
  }

  function renderChart(rows) {
    const root = elements.chart;
    root.replaceChildren();
    const selectedRows = rows.filter((row) => Number(row.system_core_count) === selectedSystemCores);
    const maximum = Math.max(...selectedRows.map((row) => Number(row.steady_state_estimate_us) || 0), 1);
    for (const row of selectedRows) {
      const line = element("button", Number(row.ai_assigned_core_count) === selectedAiCores ? "core-isolation-bar active" : "core-isolation-bar");
      line.type = "button";
      line.setAttribute("aria-label", `${row.ai_assigned_core_count} AI cores, steady estimate ${duration(row.steady_state_estimate_us)}`);
      line.addEventListener("click", () => selectScenario(selectedSystemCores, row.ai_assigned_core_count));
      const label = element("span", "core-isolation-bar-label", `${row.ai_assigned_core_count}C`);
      const track = element("span", "core-isolation-bar-track");
      const fill = element("span", "core-isolation-bar-fill");
      fill.style.width = `${Math.max(2, (Number(row.steady_state_estimate_us) / maximum) * 100)}%`;
      track.append(fill);
      line.append(label, track, element("strong", null, duration(row.steady_state_estimate_us)));
      root.append(line);
    }
  }

  function renderTable(rows) {
    elements.body.replaceChildren();
    for (const row of rows.filter((candidate) => Number(candidate.system_core_count) === selectedSystemCores)) {
      const tr = element("tr", Number(row.ai_assigned_core_count) === selectedAiCores ? "selected" : "");
      tr.tabIndex = 0;
      tr.addEventListener("click", () => selectScenario(selectedSystemCores, row.ai_assigned_core_count));
      tr.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectScenario(selectedSystemCores, row.ai_assigned_core_count);
        }
      });
      const cells = [
        row.ai_assigned_core_count,
        row.housekeeping_core_count,
        allocationLabel(row),
        `${number(row.int8_issue_ceiling_gops)} / ${number(row.fp32_issue_ceiling_gops)} GOPS`,
        duration(row.theoretical_roofline_floor_us),
        duration(row.steady_state_estimate_us),
        duration(row.cold_start_estimate_us),
        `${number(row.estimate_ratio_vs_one_ai_core, 2)}x`,
      ];
      for (const value of cells) tr.append(element("td", null, value));
      elements.body.append(tr);
    }
  }

  function render() {
    const { analysis, runtimeEvidence } = getContext();
    const result = analysis?.core_isolation_analysis;
    const rows = Array.isArray(result?.scenarios) ? result.scenarios : [];
    const options = Array.isArray(result?.system_core_count_options) ? result.system_core_count_options.map(Number) : [];
    if (!result || result.status !== "assessed" || !rows.length || !options.length) {
      elements.controls.replaceChildren();
      elements.summary.replaceChildren(element("p", "core-isolation-empty", result?.unavailable_reason || "Core allocation analysis is not applicable to this artifact."));
      elements.chart.replaceChildren();
      elements.body.replaceChildren();
      elements.boundary.textContent = result?.isolation_evidence_boundary || "";
      elements.status.textContent = result?.status === "not_assessed" ? "not assessed" : "not applicable";
      elements.status.dataset.tone = "neutral";
      return;
    }

    if (!options.includes(Number(selectedSystemCores))) selectedSystemCores = options[options.length - 1];
    const systemRows = rows.filter((row) => Number(row.system_core_count) === selectedSystemCores);
    const preferred = Math.max(1, Math.floor(selectedSystemCores / 2));
    if (!systemRows.some((row) => Number(row.ai_assigned_core_count) === Number(selectedAiCores))) {
      selectedAiCores = systemRows.some((row) => Number(row.ai_assigned_core_count) === preferred) ? preferred : 1;
    }
    const selected = systemRows.find((row) => Number(row.ai_assigned_core_count) === selectedAiCores) || systemRows[0];
    renderControls(options, rows);
    renderSummary(selected, result, runtimeEvidence);
    renderChart(rows);
    renderTable(rows);
    elements.boundary.textContent = `${result.method} Evidence boundary: ${result.isolation_evidence_boundary}`;
  }

  return { render };
}
