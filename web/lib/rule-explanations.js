import { ONNX_OPERATION_COST_SOURCE } from "../onnx.js";

const TENSORFLOW_COMMIT = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";

const RULES = Object.freeze({
  "onnx.conv.output-shape": Object.freeze({
    title: "ONNX Conv output shape and nominal MAC count",
    question: "What output extent and multiply-accumulate count are serialized or statically derivable for an ONNX Conv node?",
    method: "Apply the pinned ONNX Conv shape contract, preserve unresolved dimensions symbolically, then multiply output cardinality by the effective kernel reduction cardinality.",
    evidence_class: "SOURCE_BACKED",
    source: ONNX_OPERATION_COST_SOURCE,
    implementation: "web/onnx.js",
    limitations: [
      "Nominal MACs do not establish runtime latency, fusion, provider assignment, or physical data movement.",
      "A malformed or internally contradictory shape contract remains unresolved instead of being guessed.",
    ],
  }),
  "tflite.conv2d.macs": Object.freeze({
    title: "TFLite Conv2D nominal MAC count",
    question: "How many nominal multiply-accumulate operations does a serialized TFLite Conv2D require?",
    method: "Use serialized tensor extents and the TensorFlow Lite Conv2D operator contract; reject inconsistent filter and channel cardinalities.",
    evidence_class: "SOURCE_BACKED",
    source: {
      repository: "tensorflow/tensorflow",
      commit: TENSORFLOW_COMMIT,
      documents: [{
        role: "tflite_conv2d_contract",
        path: "tensorflow/lite/kernels/conv.cc",
        source_ref: `https://raw.githubusercontent.com/tensorflow/tensorflow/${TENSORFLOW_COMMIT}/tensorflow/lite/kernels/conv.cc`,
      }],
    },
    implementation: "src/lib.rs",
    limitations: [
      "Nominal MACs do not establish the selected kernel, delegate assignment, latency, or energy use.",
    ],
  }),
  "finding.gate.defects": Object.freeze({
    title: "Artifact-defect gate",
    question: "Should this static audit block an automation gate?",
    method: "Block only when at least one canonical finding has finding_kind artifact_defect. Cautions and evidence gaps remain visible but do not block by default.",
    evidence_class: "DERIVED",
    source: {
      repository: "JunHwan-Kwon/deepbom",
      documents: [
        { role: "finding_contract", path: "web/lib/finding-contract.js" },
        { role: "gate_evaluator", path: "bin/deepbom-automation.mjs" },
      ],
    },
    implementation: "bin/deepbom-automation.mjs",
    limitations: [
      "This gate is not a release, safety, security, clinical, or regulatory decision.",
      "Organizations can bind stricter evidence requirements through an explicit review policy.",
    ],
  }),
});

export function explainRule(ruleId) {
  const id = String(ruleId || "").trim().toLowerCase();
  const rule = RULES[id];
  if (!rule) throw new Error(`Unknown rule: ${ruleId}. Use deepbom explain-rule --list.`);
  return {
    schema: "deepbom.rule_explanation.v1",
    rule_id: id,
    ...rule,
  };
}

export function listRuleExplanations() {
  return Object.entries(RULES).map(([rule_id, rule]) => ({ rule_id, title: rule.title }));
}
