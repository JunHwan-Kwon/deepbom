import { formatBytes, formatPercent } from "./format.js";

export const EXPLORER_QUESTION_SCHEMA = "deepbom.explorer_question_entry.v1";

const QUANT_FINDING_PATTERN = /quant|scale|zero.?point|dead channel|weight integrity|requant/i;

function finiteNonNegative(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

function findingText(finding) {
  return [finding?.id, finding?.title, finding?.class, finding?.category, finding?.observation]
    .filter(Boolean)
    .join(" ");
}

function deviceCapacityBytes(analysis) {
  const profile = analysis?.target_profile || {};
  for (const value of [
    profile.device_memory_capacity_bytes,
    profile.memory_capacity_bytes,
    profile.ram_bytes,
    profile.vram_bytes,
  ]) {
    const capacity = finiteNonNegative(value);
    if (capacity != null && capacity > 0) return capacity;
  }
  return null;
}

function exactNumber(value) {
  if (value && typeof value === "object" && /^\d+$/.test(String(value.decimal || ""))) {
    const number = Number(value.decimal);
    return Number.isSafeInteger(number) ? number : null;
  }
  return finiteNonNegative(value);
}

function memoryQuestion(analysis, glance) {
  const llm = analysis?.on_device_llm?.memory_feasibility || null;
  const llmLowerBound = exactNumber(llm?.minimum_static_lower_bound_bytes);
  const required = llmLowerBound ?? finiteNonNegative(glance?.memory?.artifactPlusArenaBytes);
  const capacity = deviceCapacityBytes(analysis);
  const source = llmLowerBound != null
    ? "serialized weights plus conditional KV/SSM state lower bound"
    : required != null
      ? "artifact plus deterministic arena plan"
      : "required residency";
  if (required == null) return {
    id: "memory",
    question: "Does it fit this device's memory?",
    state: "unassessed",
    answer: "Required residency is not fully assessable.",
    detail: "Bind dynamic shapes or an LLM state scenario before comparing capacity.",
    evidence: "NOT ASSESSED",
    auditTab: analysis?.on_device_llm ? "llm" : "roofline",
  };
  if (capacity == null) return {
    id: "memory",
    question: "Does it fit this device's memory?",
    state: "unassessed",
    answer: `${formatBytes(required)} lower bound; device capacity is unbound.`,
    detail: `${source}; backend workspaces, application memory, and OS reserve remain excluded.`,
    evidence: llmLowerBound != null ? "OBSERVED / DERIVED LOWER BOUND" : "OBSERVED / DERIVED",
    auditTab: analysis?.on_device_llm ? "llm" : "roofline",
  };
  const ratio = required / capacity;
  return {
    id: "memory",
    question: "Does it fit this device's memory?",
    state: required > capacity ? "issue" : "observed",
    answer: required > capacity
      ? `${formatBytes(required)} lower bound exceeds ${formatBytes(capacity)} capacity.`
      : `${formatBytes(required)} lower bound is ${formatPercent(ratio)} of ${formatBytes(capacity)} capacity.`,
    detail: required > capacity
      ? `Insufficient under the stated ${source} assumption.`
      : "This is necessary capacity evidence, not a sufficient fit claim; runtime-private allocations remain unbound.",
    evidence: llmLowerBound != null ? "OBSERVED / DERIVED LOWER BOUND" : "OBSERVED / DERIVED",
    auditTab: analysis?.on_device_llm ? "llm" : "roofline",
  };
}

function fallbackQuestion(analysis, glance) {
  if (glance?.format === "onnx") {
    const frontier = analysis?.ort_ep_portability_frontier;
    const profileCount = finiteNonNegative(frontier?.execution_provider_count) || 0;
    return {
      id: "fallback",
      question: "Why could execution fall back to CPU?",
      state: "unassessed",
      answer: profileCount
        ? `${profileCount} source-backed EP profiles are available; runtime assignment is not observed.`
        : "A selected EP source assessment and runtime assignment are not both bound.",
      detail: "Static source eligibility can identify definite exclusions and unresolved conditions, but only imported GetCapability or profiler evidence establishes provider assignment.",
      evidence: profileCount ? "SOURCE-BACKED PRECHECK" : "NOT ASSESSED",
      auditTab: "accelerator",
    };
  }
  const breaks = (analysis?.ops || []).filter((op) => op?.xnnpack_chain_break);
  const reasonCounts = [...breaks.reduce((counts, op) => {
    const reason = String(op.xnnpack_break_class || op.xnnpack_reason || op.name || "unclassified");
    counts.set(reason, (counts.get(reason) || 0) + 1);
    return counts;
  }, new Map())].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]));
  return {
    id: "fallback",
    question: "Why could execution fall back to CPU?",
    state: breaks.length ? "issue" : "observed",
    answer: breaks.length
      ? `${breaks.length} predicted break op(s) form ${glance?.delegation?.cpuIslandCount || 0} CPU island(s).`
      : "No predicted XNNPACK break op is present under the bound rulepack.",
    detail: reasonCounts.length
      ? `Leading serialized reason: ${reasonCounts.slice(0, 2).map(([reason, count]) => `${reason} (${count})`).join("; ")}. Runtime assignment remains unobserved.`
      : "This is conditional source-rule evidence, not observed delegate placement.",
    evidence: "PREDICTED",
    auditTab: "xnnpack",
  };
}

function quantQuestion(analysis, glance) {
  const quantized = finiteNonNegative(glance?.quantization?.quantizedTensorCount) || 0;
  const findings = (analysis?.findings || []).filter((finding) => QUANT_FINDING_PATTERN.test(findingText(finding)));
  const actionable = findings.filter((finding) => !/^(informational|info)$/i.test(String(finding?.priority || finding?.severity || "")));
  if (!quantized) return {
    id: "quantization",
    question: "Where is quantization loss risk visible?",
    state: "unassessed",
    answer: "No serialized quantized tensor contract is available for this artifact.",
    detail: "Quantization-specific loss cannot be inferred from a floating-point or storage-only contract.",
    evidence: "NOT APPLICABLE / NOT ASSESSED",
    auditTab: "quant",
  };
  return {
    id: "quantization",
    question: "Where is quantization loss risk visible?",
    state: actionable.length ? "issue" : "observed",
    answer: actionable.length
      ? `${actionable.length} actionable quantization finding(s) require review.`
      : `${quantized} quantized tensor(s); no configured artifact-visible risk signal was emitted.`,
    detail: actionable.length
      ? findings.slice(0, 2).map((finding) => finding.title || finding.id).filter(Boolean).join("; ")
      : "Task accuracy, calibration representativeness, and runtime numerical parity remain separate evidence.",
    evidence: actionable.length ? "OBSERVED / DERIVED FINDINGS" : "OBSERVED METADATA",
    auditTab: "quant",
  };
}

function runtimeQuestion(analysis, runtimeEvidence) {
  const assignment = runtimeEvidence?.runtimeAssignmentEvidence || runtimeEvidence?.runtime_assignment || null;
  if (!assignment) return {
    id: "runtime",
    question: "Which runtime evidence is still missing?",
    state: "unassessed",
    answer: "No artifact-bound runtime assignment has been imported.",
    detail: "Provider/delegate assignment, executed lowering or kernel identity, measured boundary materialization, and device timing remain unobserved.",
    evidence: "STATIC ONLY",
    auditTab: String(analysis?.format || "").toLowerCase() === "tflite" ? "xnnpack" : "accelerator",
  };
  const mapped = finiteNonNegative(assignment.mapped_op_count ?? assignment.assignment_count);
  const total = finiteNonNegative(analysis?.operator_count ?? analysis?.ops?.length);
  return {
    id: "runtime",
    question: "Which runtime evidence is still missing?",
    state: mapped != null && total != null && mapped < total ? "issue" : "observed",
    answer: mapped == null
      ? "Runtime evidence is imported; assignment coverage is not reported."
      : `${mapped}/${total ?? "?"} graph op(s) have imported assignment evidence.`,
    detail: "Imported evidence does not establish task accuracy or release readiness; inspect unresolved assignments and build/device binding.",
    evidence: String(assignment.evidence_class || assignment.assignment_evidence_class || "OBSERVED_RUNTIME"),
    auditTab: String(analysis?.format || "").toLowerCase() === "tflite" ? "xnnpack" : "accelerator",
  };
}

export function buildExplorerQuestionSummary(analysis, glance, runtimeEvidence = {}) {
  return {
    schema: EXPLORER_QUESTION_SCHEMA,
    items: [
      fallbackQuestion(analysis, glance),
      quantQuestion(analysis, glance),
      memoryQuestion(analysis, glance),
      runtimeQuestion(analysis, runtimeEvidence),
    ],
  };
}

export function renderExplorerQuestionEntry({
  analysis,
  glance,
  runtimeEvidence = {},
  onOpen = () => {},
} = {}) {
  const summary = buildExplorerQuestionSummary(analysis, glance, runtimeEvidence);
  const section = document.createElement("section");
  section.className = "explorer-question-entry";
  section.dataset.schema = summary.schema;
  const head = document.createElement("header");
  const title = document.createElement("strong");
  title.textContent = "Answer the deployment question first";
  const note = document.createElement("span");
  note.textContent = "Open the underlying ledger without losing evidence detail.";
  head.append(title, note);
  const grid = document.createElement("div");
  grid.className = "explorer-question-grid";
  for (const item of summary.items) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `explorer-question-card state-${item.state}`;
    card.dataset.questionId = item.id;
    card.dataset.targetAuditTab = item.auditTab;
    card.addEventListener("click", () => onOpen(item));
    const question = document.createElement("span");
    question.className = "explorer-question-label";
    question.textContent = item.question;
    const answer = document.createElement("strong");
    answer.textContent = item.answer;
    const detail = document.createElement("small");
    detail.textContent = item.detail;
    const evidence = document.createElement("em");
    evidence.textContent = item.evidence;
    card.append(question, answer, detail, evidence);
    grid.append(card);
  }
  section.append(head, grid);
  return section;
}
