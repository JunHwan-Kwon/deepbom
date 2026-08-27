import { formatUs, shortError } from "./format.js";
import { previewOrtProfileMapping } from "./runtime-profile-adapter.js";
import { previewTfliteRuntimeInfoMapping } from "./tflite-runtime-info-adapter.js";
import { previewTfliteBenchmarkProfileMapping } from "./tflite-profile-info-adapter.js";
import { closeModal, installModalKeyboard, openModal } from "./modal-accessibility.js";

export function createRuntimeProfileModal({ elements, getPending, clearPending, getAnalysis, getAssignmentEvidence }) {
  const {
    backdrop, closeButton, cancelButton, title, preview, version, backend, optimization,
    executionMode, collectedAt, captureLabel, capture, binarySha, build, status, importButton,
    fallbackFocus,
  } = elements;

  const renderPreview = () => {
    const pending = getPending();
    const analysis = getAnalysis();
    if (!pending || !preview || !analysis) return;
    try {
      const tflitePlan = pending.profile.kind === "tflite_model_runtime_info";
      const tfliteTiming = pending.profile.kind === "tflite_benchmark_profile";
      const mapping = tfliteTiming
        ? previewTfliteBenchmarkProfileMapping(pending.profile, getAssignmentEvidence(), analysis)
        : tflitePlan
          ? previewTfliteRuntimeInfoMapping(pending.profile, analysis)
          : previewOrtProfileMapping(pending.profile, analysis, {
            graphOptimizationLevel: optimization.value,
            executionMode: executionMode.value,
          });
      const metrics = previewMetrics(mapping, tflitePlan, tfliteTiming);
      preview.replaceChildren(...metrics.map(([label, value]) => metric(label, value)));
      const importable = tfliteTiming
        ? mapping.mapped_execution_node_count > 0 || mapping.delegate_internal_event_count > 0
        : mapping.assignment_count > 0;
      importButton.disabled = !importable;
      status.textContent = previewStatus(mapping, tflitePlan, tfliteTiming, importable);
      status.dataset.tone = importable ? "neutral" : "error";
    } catch (error) {
      preview.replaceChildren();
      importButton.disabled = true;
      status.textContent = shortError(error);
      status.dataset.tone = "error";
    }
  };

  const close = () => {
    clearPending();
    closeModal(backdrop, { fallbackFocus });
  };

  const open = () => {
    const pending = getPending();
    if (!pending) return;
    const tflitePlan = pending.profile.kind === "tflite_model_runtime_info";
    const tfliteTiming = pending.profile.kind === "tflite_benchmark_profile";
    const tfliteEvidence = tflitePlan || tfliteTiming;
    const verified = pending.verifiedMetadata || null;
    title.textContent = tfliteTiming ? "TFLite execution profile" : tflitePlan ? "TFLite runtime plan" : verified ? "Pinned native ONNX Runtime profile" : "ONNX Runtime profile";
    version.value = verified?.runtimeVersion || "";
    version.placeholder = tflitePlan ? "Runtime version or commit" : "1.26.0";
    backend.value = verified?.backend || (tfliteTiming ? getAssignmentEvidence()?.runtime?.backend || "" : pending.profile.providers.join(" + "));
    backend.title = backend.value;
    optimization.value = verified?.graphOptimizationLevel || "unknown";
    executionMode.value = verified?.executionMode || "unknown";
    for (const input of [version, backend, build, binarySha]) input.disabled = tfliteTiming;
    for (const input of [optimization, executionMode]) input.disabled = tfliteEvidence || Boolean(verified);
    for (const input of [version, backend, build, binarySha]) input.closest("label").hidden = tfliteTiming;
    for (const input of [optimization, executionMode]) input.closest("label").hidden = tfliteEvidence;
    for (const input of [version, build, binarySha, collectedAt, capture]) input.readOnly = Boolean(verified);
    collectedAt.value = verified ? toLocalDateTimeValue(verified.collectedAt) : "";
    binarySha.value = verified?.binarySha256 || "";
    build.value = verified?.runtimeBuild || "";
    captureLabel.hidden = !tfliteEvidence && !verified;
    capture.required = tfliteEvidence || Boolean(verified);
    capture.value = verified?.captureId || "";
    capture.placeholder = tfliteTiming ? getAssignmentEvidence()?.source?.capture_id || "Matching runtime-plan capture ID" : "device-run-2026-07-16-001";
    status.textContent = "";
    status.dataset.tone = "";
    renderPreview();
    openModal(backdrop, { focus: tfliteTiming ? capture : verified ? importButton : version });
  };

  optimization?.addEventListener("change", renderPreview);
  executionMode?.addEventListener("change", renderPreview);
  closeButton?.addEventListener("click", close);
  cancelButton?.addEventListener("click", close);
  backdrop?.addEventListener("click", (event) => { if (event.target === backdrop) close(); });
  installModalKeyboard(backdrop, close);
  return { open, close, renderPreview };
}

function toLocalDateTimeValue(value) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "";
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60_000);
  return local.toISOString().slice(0, 16);
}

function previewMetrics(mapping, tflitePlan, tfliteTiming) {
  if (tfliteTiming) return [
    ["Execution nodes", `${mapping.mapped_execution_node_count}/${mapping.execution_plan_node_count}`],
    ["CPU node timings", String(mapping.original_execution_node_timing_count)],
    ["Partition timings", String(mapping.delegate_partition_timing_count)],
    ["Delegate internal", String(mapping.delegate_internal_event_count)],
    ["Graph total", mapping.graph_total_available ? formatUs(mapping.graph_total_us) : "Withheld"],
  ];
  if (tflitePlan) return [
    ["Original ops", `${mapping.topology_match_count}/${mapping.graph_op_count}`],
    ["Delegate partitions", String(mapping.delegate_partition_count)],
    ["Delegated ops", String(mapping.delegated_op_count)],
    ["Execution nodes", String(mapping.execution_plan_node_count)],
    ["Binding", "Exact topology"],
  ];
  return [
    ["Mapped ops", `${mapping.assignment_count}/${mapping.graph_op_count}`],
    ["Kernel events", `${mapping.mapped_kernel_event_count}/${mapping.kernel_event_count}`],
    ["Unresolved", String(mapping.unresolved_runtime_node_count)],
    ["Conflicts", String(mapping.conflict_count)],
    ["Timing", mapping.durations_additive ? "Additive" : "Non-additive"],
  ];
}

function previewStatus(mapping, tflitePlan, tfliteTiming, importable) {
  if (!importable) return "No runtime node has a deterministic original-op mapping.";
  if (tfliteTiming) return mapping.graph_total_available
    ? "All execution-plan nodes are covered; partition durations are counted once."
    : "Partial execution-node coverage; delegate-profiled event classes remain unassigned and no graph total will be emitted.";
  if (tflitePlan) return "Execution plan and delegate replacement maps are symmetric; runtime identity remains declared.";
  return `${mapping.mapping_method_counts.exact_graph_node_name_and_op_type || 0} exact-name / ${mapping.mapping_method_counts.optimization_disabled_unnamed_node_index_and_op_type || 0} strict index mapping`;
}

function metric(label, value) {
  const item = document.createElement("div");
  item.className = "runtime-profile-preview-item";
  const name = document.createElement("span");
  name.textContent = label;
  const number = document.createElement("strong");
  number.textContent = value;
  item.append(name, number);
  return item;
}
