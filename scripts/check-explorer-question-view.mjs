import { buildExplorerQuestionSummary } from "../web/lib/explorer-question-view.js";
import { readFileSync } from "node:fs";

const questionViewSource = readFileSync("web/lib/explorer-question-view.js", "utf8");
expect(!questionViewSource.includes("card.dataset.auditTab"), "Question cards must not reuse the audit-tab identity attribute.");
expect(questionViewSource.includes("card.dataset.targetAuditTab"), "Question cards need a separate navigation target attribute.");

const baseGlance = {
  format: "tflite",
  quantization: { quantizedTensorCount: 4 },
  delegation: { cpuIslandCount: 1 },
  memory: { artifactPlusArenaBytes: 8 * 1024 * 1024 },
};

const tflite = {
  format: "tflite",
  operator_count: 2,
  target_profile: {},
  ops: [
    { index: 0, name: "CONV_2D", xnnpack_chain_break: false },
    { index: 1, name: "MEAN", xnnpack_chain_break: true, xnnpack_break_class: "unsupported_non_structural" },
  ],
  findings: [{ id: "EA-QNT-TEST", title: "Quantized weight dead channel", priority: "High" }],
};
const tfliteSummary = buildExplorerQuestionSummary(tflite, baseGlance);
expect(tfliteSummary.items.length === 4, "Question entry must expose exactly four stable deployment questions.");
expect(item(tfliteSummary, "fallback").state === "issue", "Predicted delegate breaks must be surfaced first.");
expect(item(tfliteSummary, "fallback").answer.includes("1 predicted break"), "Fallback answer must preserve the break denominator.");
expect(item(tfliteSummary, "quantization").state === "issue", "Actionable quantization findings must be surfaced.");
expect(item(tfliteSummary, "memory").state === "unassessed"
  && item(tfliteSummary, "memory").answer.includes("capacity is unbound"), "Memory fit must fail closed when capacity is absent.");
expect(item(tfliteSummary, "runtime").evidence === "STATIC ONLY", "Missing runtime assignment must not be promoted to observed evidence.");

const capacityBound = structuredClone(tflite);
capacityBound.target_profile.device_memory_capacity_bytes = 16 * 1024 * 1024;
const capacitySummary = buildExplorerQuestionSummary(capacityBound, baseGlance);
expect(item(capacitySummary, "memory").state === "observed", "A lower bound below capacity may be reported as an observed comparison.");
expect(item(capacitySummary, "memory").detail.includes("not a sufficient fit claim"), "A lower-bound comparison must not claim sufficient device fit.");

const onnx = {
  format: "onnx",
  operator_count: 5,
  ops: Array.from({ length: 5 }, (_, index) => ({ index, name: "Add" })),
  ort_ep_portability_frontier: { execution_provider_count: 8 },
};
const onnxGlance = {
  ...baseGlance,
  format: "onnx",
  quantization: { quantizedTensorCount: 0 },
  memory: { artifactPlusArenaBytes: null },
};
const onnxSummary = buildExplorerQuestionSummary(onnx, onnxGlance, {
  runtimeAssignmentEvidence: { evidence_class: "OBSERVED_RUNTIME", mapped_op_count: 3 },
});
expect(item(onnxSummary, "fallback").answer.includes("8 source-backed EP profiles"), "ONNX fallback entry must distinguish source profiles from assignment.");
expect(item(onnxSummary, "runtime").state === "issue"
  && item(onnxSummary, "runtime").answer.includes("3/5"), "Partial runtime assignment coverage must retain its exact denominator.");

console.log("Explorer question entry check passed (4 questions, fail-closed memory, static/runtime evidence separation).");

function item(summary, id) {
  return summary.items.find((entry) => entry.id === id);
}

function expect(condition, message) {
  if (!condition) throw new Error(message);
}
