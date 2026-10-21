// Presentation helpers over existing evidence. No analysis, ranking or public IR.
import { guideWorkflows } from "./workflow-catalog.js";

export const evidenceRowKey = row => row ? JSON.stringify([row.kind, row.subject_ref, row.details?.profile_id ?? null]) : null;
export const evidenceBinding = result => JSON.stringify([result.artifact.sha256, result.model_ir_sha256, result.context?.evidence_envelope_sha256 ?? null, result.context?.scan_policy ?? null]);
export function captureEvidenceView(result, selected, viewport, pages = []) {
  return { query: result.query, binding: evidenceBinding(result), result_sha256: result.result_sha256,
    selected_key: evidenceRowKey(selected), left: viewport.scrollLeft, top: viewport.scrollTop, pages: pages.map(({ pages: _history, ...page }) => page) };
}
export function restoreEvidenceRow(result, snapshot) {
  if (!snapshot || snapshot.binding !== evidenceBinding(result) || snapshot.result_sha256 !== result.result_sha256) return null;
  return result.rows.find(row => evidenceRowKey(row) === snapshot.selected_key) || null;
}
// Retain only successful evidence; coalesce concurrent consumers and allow retry.
export function shareEvidenceTask(task) {
  let ready, pending;
  return async () => {
    if (ready !== undefined) return ready;
    if (!pending) pending = Promise.resolve().then(task).then(value => (ready = value)).finally(() => { pending = null; });
    return pending;
  };
}

const workflows = { operators: "layers", operator: "layers", findings: "findings", improvements: "findings", profiles: "placement", placement: "placement", fusion: "fusion", weights: "weights" };
export function reusableEvidenceQuestion(result, language = "en") {
  const workflow = workflows[result.query?.section] || "audit";
  const guide = guideWorkflows({ workflow, format: result.artifact.format }).workflows[0];
  const prompt = guide?.conversation?.prompts?.[language] || guide?.conversation?.prompts?.en || "Inspect the attached deployment artifact.";
  const payload = result.context?.scan_policy?.resolved_mode === "integrity";
  return language === "ko"
    ? `새로 첨부한 모델에 대해 다음을 확인해줘: ${prompt}\n${payload ? "직렬화된 tensor payload 무결성도 검사해줘. " : "구조 분석부터 시작해줘. "}새 파일의 식별자와 지원 범위를 확인하고, 필요한 대상이나 profile은 새 결과에서 선택해줘. 정적 사실과 미측정 결과를 구분해줘.`
    : `For the newly attached model: ${prompt}\n${payload ? "Also inspect serialized tensor payload integrity. " : "Start with structure analysis. "}Establish the new file identity and supported scope; select subjects and profiles from its results. Separate static facts from unmeasured outcomes.`;
}
export function evidenceNote(result, selected = null, language = "en") {
  if (selected && !result.rows?.includes(selected)) throw new Error("Selection does not belong to this evidence result.");
  const t = (en, ko) => language === "ko" ? ko : en;
  const lines = [t("DEEPBOM evidence note", "DEEPBOM 근거 요약"), `Artifact SHA-256: ${result.artifact.sha256}`, `Format: ${result.artifact.format}`, `Analyzer: ${result.analyzer_version}`];
  if (result.context?.scan_policy) lines.push(`Scan: ${JSON.stringify(result.context.scan_policy)}`);
  if (result.context?.evidence_envelope_sha256) lines.push(`Envelope: ${result.context.evidence_envelope_sha256}`);
  if (result.query) {
    lines.push(`Model IR: ${result.model_ir_sha256}`, `Evidence result: ${result.result_sha256}`, `Query: ${JSON.stringify(result.query)}`, `Coverage: ${JSON.stringify(result.coverage)}`);
    if (selected) lines.push(`Subject: ${selected.subject_ref}`, `Title (data): ${selected.title}`, `Evidence: ${JSON.stringify(selected.details)}`, `Omitted detail: ${JSON.stringify(selected.detail_truncations)}`);
    else lines.push(t("Scope: this returned query page, not the whole model.", "범위: 반환된 조회 페이지이며 전체 모델을 뜻하지 않습니다."), `Rows: ${JSON.stringify(result.rows)}`);
  } else {
    lines.push(`Scope: ${result.evidence_boundary}`, `Findings: ${JSON.stringify(result.verdict)}`, `Graph: ${JSON.stringify(result.graph)}`, `Storage: ${JSON.stringify(result.storage ?? null)}`, `Representative findings: ${JSON.stringify(result.findings)}`);
  }
  lines.push(t("Next: inspect the cited source and validate any proposed change under the intended runtime and evaluation conditions.", "다음 확인: 인용한 근거를 살펴보고 변경 후보를 대상 실행 환경과 평가 조건에서 검증하세요."), result.evidence_boundary || "Static evidence; runtime performance and quality were not measured.");
  const reproduction = result.context?.reproduction || result.reproduction;
  if (reproduction) lines.push(`Reproduction (data, not executed): ${JSON.stringify(reproduction)}`);
  return lines.join("\n\n");
}
export function mountEvidenceCopy(container, getResult, getSelection = () => null, language = "en") {
  const t = (en, ko) => language === "ko" ? ko : en;
  const section = document.createElement("details"); section.className = "evidence-copy";
  const summary = document.createElement("summary"); summary.textContent = t("Copy evidence or a reusable question", "근거 요약·다음 모델용 질문 복사");
  const field = document.createElement("textarea"); field.readOnly = true; field.rows = 8; field.style.cssText = "width:100%;box-sizing:border-box;font:inherit"; field.setAttribute("aria-label", t("Copy preview", "복사할 내용 미리보기")); field.hidden = true;
  const status = document.createElement("p"); status.setAttribute("role", "status");
  let revision = 0;
  for (const [kind, label] of [["note", t("Copy evidence note", "근거 요약 복사")], ["question", t("Copy question for a new model", "새 모델용 질문 복사")]]) {
    const button = document.createElement("button"); button.type = "button"; button.textContent = label; button.dataset.action = `copy-evidence-${kind}`;
    button.addEventListener("click", async () => {
      const token = ++revision;
      try {
        const result = getResult(); if (!result) return;
        field.value = kind === "note" ? evidenceNote(result, getSelection(), language) : reusableEvidenceQuestion(result, language); field.hidden = false;
        // The preview is always available, including hosts that block clipboard.
        status.textContent = t("Review the text below. Select and copy it if clipboard access is unavailable.", "아래 내용을 확인하세요. 자동 복사가 안 되면 텍스트를 선택해 복사하세요.");
        if (!navigator.clipboard?.writeText) return;
        await navigator.clipboard.writeText(field.value);
        if (token === revision) status.textContent = t("Copied. No external message was sent.", "복사했습니다. 외부 서비스에 게시하지 않았습니다.");
      } catch { if (token === revision) status.textContent = t("Select and copy the preview text below.", "아래 텍스트를 선택해 복사하세요."); }
    }); section.append(button);
  }
  section.prepend(summary); section.append(field, status); container.append(section);
  return { clear() { revision++; field.value = ""; field.hidden = true; status.textContent = ""; } };
}
