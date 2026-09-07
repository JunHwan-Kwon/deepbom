import { quantizationScopeExplanation } from "./analysis.js";
import { buildExecutionPlacementEvidence } from "./execution-placement-evidence.js";
import { APPLICABILITY_STATUS, auditTabApplicability } from "./evidence-applicability.js";
import { formatBytes, formatNumber, formatPercent } from "./format.js";

// Overview 는 각 도메인의 판정만 얕게 보여주고, 상세는 해당 도메인 탭이 소유한다.
// 여기서 계산을 새로 하지 않는다 — 도메인 탭이 읽는 것과 동일한 분석 필드만 투영한다.
const DOMAINS = Object.freeze([
  { id: "numerical", tab: "quant", label: "Numerical contracts", figures: numericalFigures },
  { id: "placement", tab: "accelerator", label: "Execution placement", figures: placementFigures },
  { id: "memory", tab: "roofline", label: "Memory & cost", figures: memoryFigures },
  { id: "architecture", tab: "stage", label: "Architecture", figures: architectureFigures },
]);

export function buildOverviewDomainMap(analysis, { format = "", onOpen = null } = {}) {
  const section = document.createElement("section");
  section.className = "overview-domain-map";
  section.setAttribute("aria-label", "Evidence domain map");

  const heading = document.createElement("h3");
  heading.textContent = "Evidence domains";
  const note = document.createElement("p");
  note.className = "overview-domain-map-note";
  note.textContent = "Each row states what this artifact shows for one domain. Open a row for its full ledger.";
  section.append(heading, note);

  if (!analysis || typeof analysis !== "object") return section;
  const applicability = auditTabApplicability(format || analysis.format, analysis);

  const list = document.createElement("ul");
  list.className = "overview-domain-list";
  for (const domain of DOMAINS) {
    list.append(domainRow(domain, analysis, applicability[domain.tab], onOpen));
  }
  section.append(list);
  return section;
}

function domainRow(domain, analysis, record, onOpen) {
  const item = document.createElement("li");
  item.className = "overview-domain-row";
  item.dataset.overviewDomain = domain.id;

  const status = record?.applicability_status || APPLICABILITY_STATUS.NOT_ASSESSED_YET;
  const applicable = status === APPLICABILITY_STATUS.APPLICABLE;
  const reasonText = record?.reason_text || "Applicability was not resolved.";
  item.dataset.applicabilityStatus = status;
  // 상태를 선언하는 노드는 이유도 함께 들고 있어야 한다 (evidence applicability 계약).
  item.dataset.applicabilityReasonCode = record?.reason_code || "APPLICABILITY_NOT_RESOLVED";
  item.dataset.applicabilityReason = reasonText;
  if (record?.required_evidence) item.dataset.applicabilityRequired = record.required_evidence;

  const control = document.createElement(applicable && onOpen ? "button" : "div");
  control.className = "overview-domain-control";
  if (control.tagName === "BUTTON") {
    control.type = "button";
    control.addEventListener("click", () => onOpen(domain.tab));
    control.setAttribute("aria-label", `Open ${domain.label}`);
  }

  const name = document.createElement("strong");
  name.textContent = domain.label;

  const detail = document.createElement("span");
  detail.className = "overview-domain-detail";
  if (applicable) {
    const figures = domain.figures(analysis) || [];
    if (figures.length) {
      for (const figure of figures) {
        const chip = document.createElement("em");
        chip.textContent = figure;
        detail.append(chip);
      }
    } else {
      detail.textContent = "No figure was emitted for this domain.";
    }
  } else {
    detail.classList.add("overview-domain-reason");
    detail.textContent = reasonText;
  }

  const state = document.createElement("small");
  state.className = "overview-domain-state";
  state.textContent = applicable ? "Open" : statusLabel(status);

  control.append(name, detail, state);
  item.append(control);
  return item;
}

function statusLabel(status) {
  if (status === APPLICABILITY_STATUS.NOT_APPLICABLE) return "Not applicable";
  if (status === APPLICABILITY_STATUS.NOT_ASSESSABLE) return "Not assessable";
  return "Not assessed yet";
}

function numericalFigures(analysis) {
  const out = [];
  const label = analysis?.quantization_status?.label || analysis?.quantization_status?.classification;
  if (label) out.push(String(label));
  const scope = safeScope(analysis);
  if (finite(scope?.compute_ops_denominator) && finite(scope?.quantized_compute_ops) != null) {
    out.push(`${formatNumber(scope.quantized_compute_ops)}/${formatNumber(scope.compute_ops_denominator)} compute ops quantized`);
  }
  if (finite(analysis?.per_channel_tensors) != null) out.push(`${formatNumber(analysis.per_channel_tensors)} per-axis tensors`);
  if (finite(analysis?.quant_hole_count) != null) out.push(`${formatNumber(analysis.quant_hole_count)} mid-graph holes`);
  return out;
}

function placementFigures(analysis) {
  const out = [];
  // 포맷 중립 경로를 먼저 쓴다. Execution Placement 탭이 읽는 것과 같은 산출물이다.
  const evidence = safePlacement(analysis);
  const flow = evidence?.flow;
  if (finite(flow?.scope_item_count) != null) {
    out.push(`${formatNumber(flow.covered_item_count)}/${formatNumber(flow.scope_item_count)} items with placement evidence`);
  }
  const segments = Array.isArray(flow?.segments) ? flow.segments.length : null;
  if (segments != null) out.push(`${formatNumber(segments)} placement segments`);
  // TFLite 전용 예측 수치는 있을 때만 덧붙인다.
  if (finite(analysis?.delegated_mac_percent) != null) {
    out.push(`${formatPercent(analysis.delegated_mac_percent)} MACs conditionally delegatable`);
  }
  if (finite(analysis?.xnnpack_chain_breaks) != null) out.push(`${formatNumber(analysis.xnnpack_chain_breaks)} predicted breaks`);
  return out;
}

function memoryFigures(analysis) {
  const out = [];
  const arena = finite(analysis?.tensor_arena_plan?.combined_arena_bytes);
  if (arena != null) out.push(`${formatBytes(arena)} projected arena`);
  const liveness = analysis?.tensor_liveness || {};
  if (finite(liveness.peak_bytes) != null) {
    const at = liveness.peak_at_op_name || (finite(liveness.peak_at_op) != null ? `op ${liveness.peak_at_op}` : null);
    out.push(at ? `peak ${formatBytes(liveness.peak_bytes)} at ${at}` : `peak ${formatBytes(liveness.peak_bytes)}`);
  }
  return out;
}

function architectureFigures(analysis) {
  const out = [];
  if (finite(analysis?.operator_count) != null) out.push(`${formatNumber(analysis.operator_count)} operators`);
  if (finite(analysis?.tensor_count) != null) out.push(`${formatNumber(analysis.tensor_count)} tensors`);
  const nested = finite(analysis?.artifact_ir_nested_scope_count);
  if (nested != null && nested > 0) out.push(`${formatNumber(nested)} nested scopes`);
  return out;
}

// 정적 배치 근거만 쓴다. 런타임 오버레이는 Overview 의 관심사가 아니다.
function safePlacement(analysis) {
  try {
    return buildExecutionPlacementEvidence(analysis, null) || null;
  } catch {
    return null;
  }
}

// 도메인 탭이 쓰는 것과 같은 헬퍼를 통과시킨다. 여기서 다시 세지 않는다.
function safeScope(analysis) {
  try {
    return quantizationScopeExplanation(analysis) || null;
  } catch {
    return null;
  }
}

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}
