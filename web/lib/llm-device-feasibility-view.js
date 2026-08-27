import { compareLlmMemoryCapacity, LLM_MEMORY_CAPACITY_TIERS } from "./llm-memory-feasibility.js";

function decimal(value) {
  const text = value && typeof value === "object" ? value.decimal : value;
  return /^\d+$/.test(String(text || "")) ? String(text) : null;
}

function bytesText(value) {
  const text = decimal(value);
  if (!text) return "not derived";
  const bytes = BigInt(text);
  const units = [["TiB", 1024n ** 4n], ["GiB", 1024n ** 3n], ["MiB", 1024n ** 2n], ["KiB", 1024n]];
  const unit = units.find(([, size]) => bytes >= size);
  if (!unit) return `${bytes.toLocaleString("en-US")} B`;
  const hundredths = bytes * 100n / unit[1];
  return `${hundredths / 100n}.${String(hundredths % 100n).padStart(2, "0")} ${unit[0]}`;
}

function scenarioLabel(row) {
  if (row.state_kind === "hybrid_kv_ssm") return `Hybrid KV + SSM / ctx ${row.context_length ?? "?"} / batch ${row.batch_size ?? "?"}`;
  if (row.state_kind === "ssm_recurrent") return `SSM recurrent / batch ${row.batch_size ?? "?"}`;
  if (row.state_kind === "state_contract_unbound") return "Serialized weights only";
  return `KV ctx ${row.context_length ?? "?"} / batch ${row.batch_size ?? "?"} / ${row.storage_bits ?? "?"}-bit`;
}

export function buildLlmDeviceFeasibilityPresentation(memory = {}, capacityValue) {
  const scenarios = Array.isArray(memory.static_scenarios) ? memory.static_scenarios : [];
  const rows = scenarios.map((scenario) => ({
    label: scenarioLabel(scenario),
    required: scenario.static_lower_bound_bytes,
    comparison: compareLlmMemoryCapacity(scenario.static_lower_bound_bytes, capacityValue),
  }));
  const exceeds = rows.filter((row) => row.comparison.status === "lower_bound_exceeds_capacity");
  const unresolvedFit = rows.filter((row) => row.comparison.status === "lower_bound_at_or_below_capacity_fit_unresolved");
  const largestDeficit = exceeds.map((row) => decimal(row.comparison.deficit_bytes)).filter(Boolean)
    .map(BigInt).reduce((max, value) => value > max ? value : max, 0n);
  const smallestHeadroom = unresolvedFit.map((row) => decimal(row.comparison.headroom_after_lower_bound_bytes)).filter(Boolean)
    .map(BigInt).reduce((min, value) => min == null || value < min ? value : min, null);
  return {
    capacity: capacityValue,
    scenarioCount: rows.length,
    exceedsCount: exceeds.length,
    status: !rows.length ? "not_assessable"
      : exceeds.length ? "insufficient_under_assumption"
        : "lower_bounds_do_not_exceed_fit_unresolved",
    largestDeficit: largestDeficit > 0n ? { decimal: String(largestDeficit) } : null,
    smallestHeadroom: smallestHeadroom == null ? null : { decimal: String(smallestHeadroom) },
    rows,
  };
}

export function renderLlmDeviceFeasibilityView(doc, memory = {}, runtime = {}) {
  const section = doc.createElement("section");
  section.className = "llm-device-feasibility";
  const head = doc.createElement("header");
  const title = doc.createElement("div");
  const eyebrow = doc.createElement("span");
  eyebrow.textContent = "DEVICE FEASIBILITY";
  const heading = doc.createElement("h4");
  heading.textContent = "Compare deterministic memory lower bounds";
  title.append(eyebrow, heading);
  const label = doc.createElement("label");
  label.textContent = "Aggregate memory budget";
  const select = doc.createElement("select");
  select.setAttribute("aria-label", "Aggregate device memory budget");
  const tiers = Array.isArray(memory.reference_capacity_tiers) && memory.reference_capacity_tiers.length
    ? memory.reference_capacity_tiers
    : LLM_MEMORY_CAPACITY_TIERS.map((tier) => ({ label: tier.label, bytes: { decimal: tier.bytes } }));
  for (const tier of tiers) {
    const option = doc.createElement("option");
    option.value = decimal(tier.bytes) || "";
    option.textContent = tier.label;
    option.selected = tier.label === "8 GiB";
    select.append(option);
  }
  label.append(select);
  head.append(title, label);
  const result = doc.createElement("div");
  result.className = "llm-device-feasibility-result";

  const render = () => {
    const presentation = buildLlmDeviceFeasibilityPresentation(memory, select.value);
    section.dataset.state = presentation.status;
    const verdict = doc.createElement("strong");
    verdict.textContent = presentation.status === "insufficient_under_assumption"
      ? `${presentation.exceedsCount}/${presentation.scenarioCount} lower-bound scenarios exceed ${select.selectedOptions[0]?.textContent || "the budget"}`
      : presentation.status === "lower_bounds_do_not_exceed_fit_unresolved"
        ? `No lower bound exceeds ${select.selectedOptions[0]?.textContent || "the budget"}; fit remains unresolved`
        : "Memory lower bounds are not assessable";
    const metrics = doc.createElement("dl");
    metrics.append(
      term(doc, "Lower-bound range", `${bytesText(memory.minimum_static_lower_bound_bytes)} to ${bytesText(memory.maximum_static_lower_bound_bytes)}`),
      term(doc, presentation.exceedsCount ? "Largest deficit" : "Smallest remaining headroom", bytesText(presentation.exceedsCount ? presentation.largestDeficit : presentation.smallestHeadroom)),
      term(doc, "Scenarios assessed", String(presentation.scenarioCount)),
      term(doc, "Evidence", memory.evidence_class || "NOT_ASSESSABLE"),
    );
    const boundary = doc.createElement("p");
    boundary.textContent = presentation.exceedsCount
      ? "Insufficient only under the stated simultaneous-residency assumption."
        : "A lower bound below capacity is necessary evidence, not proof of fit. Runtime workspaces, packing, allocator overhead, application memory, and OS reserve remain unbound.";
    result.replaceChildren(verdict, metrics, runtimePlacementSummary(doc, runtime), boundary);
  };
  select.addEventListener("change", render);
  section.append(head, result);
  render();
  return section;
}

function runtimePlacementSummary(doc, runtime) {
  const summary = doc.createElement("p");
  summary.className = "llm-device-feasibility-runtime";
  const bound = ["artifact_bound_declared_runtime", "artifact_bound_observed_runtime"].includes(runtime?.status);
  if (!bound) {
    summary.textContent = "Offload manifest: not artifact-bound. Exact CPU/accelerator weight residency, layer placement, and state paging require an identity-bound runtime manifest.";
    return summary;
  }
  const weights = runtime.weight_residency || {};
  const layers = runtime.layer_placement || {};
  const cache = runtime.state_cache || {};
  summary.textContent = `Offload manifest: ${runtime.status}; weights CPU / accelerator / unresident ${bytesText(weights.cpu_bytes)} / ${bytesText(weights.accelerator_bytes)} / ${bytesText(weights.unresident_bytes)}; layers ${layers.cpu_layer_count ?? "?"} / ${layers.accelerator_layer_count ?? "?"} / ${layers.unresident_layer_count ?? "?"}; state resident / allocated ${bytesText(cache.resident_bytes)} / ${bytesText(cache.allocated_bytes)}.`;
  return summary;
}

function term(doc, label, value) {
  const wrapper = doc.createElement("div");
  const dt = doc.createElement("dt");
  dt.textContent = label;
  const dd = doc.createElement("dd");
  dd.textContent = value;
  wrapper.append(dt, dd);
  return wrapper;
}
