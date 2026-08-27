import { createCheck } from "./check-assert.mjs";
import {
  assessOrtReducedOperatorConfig,
  parseOrtReducedOperatorConfig,
} from "../web/lib/ort-reduced-operator-config.js";

const { done, expect, expectEqual, expectThrows } = createCheck("ORT reduced-operator config check");

const config = parseOrtReducedOperatorConfig(`
#domain;opset;operators
ai.onnx;13;Add,Cast{"inputs":{"0":["float","int32_t"]},"outputs":{"0":["float"]}}
ai.onnx;13;Conv
com.microsoft;1;Attention
`);
expectEqual(config.status, "explicit_operator_inventory", "Explicit config status should be retained.");
expectEqual(config.domain_opset_row_count, 2, "Repeated domain/opset rows should merge once.");
expectEqual(config.operator_identity_count, 4, "Merged operator identity count should conserve all unique operators.");
expectEqual(config.type_reduction_entry_count, 1, "Per-operator type JSON should be counted once.");
expectEqual(config.entries[0].operators.map((row) => row.name).join(","), "Add,Cast,Conv", "Operators should be canonicalized by name.");

const analysis = {
  format: "onnx",
  opsets: [{ domain: "", version: 13 }, { domain: "com.microsoft", version: 1 }],
  onnx_domain_analysis: {
    nodes: [
      { scope: "main_graph", top_level_op_index: 0, domain: "ai.onnx", imported_opset: 13, op_name: "Conv" },
      { scope: "main_graph/node:0/attribute:body", top_level_op_index: null, domain: "ai.onnx", imported_opset: 13, op_name: "Add" },
      { scope: "function:com.microsoft::Block", top_level_op_index: null, domain: "com.microsoft", imported_opset: 1, op_name: "Attention" },
    ],
  },
};
const assessment = assessOrtReducedOperatorConfig(analysis, config);
expectEqual(assessment.assessed_node_count, 3, "Main, nested, and function-body nodes should all be assessed.");
expectEqual(assessment.included_node_count, 3, "Every exact domain/imported-opset identity should be included.");
expectEqual(assessment.status, "compatible_operator_identity", "Complete operator identities without active type reductions should be compatible.");

const typeAssessment = assessOrtReducedOperatorConfig({
  ...analysis,
  onnx_domain_analysis: { nodes: [{ scope: "main_graph", top_level_op_index: 0, domain: "ai.onnx", imported_opset: 13, op_name: "Cast" }] },
}, config);
expectEqual(typeAssessment.status, "partial", "A present type-reduction clause must remain unresolved without generated type inventory binding.");
expectEqual(typeAssessment.type_reduction_unresolved_node_count, 1, "Type-reduction uncertainty should be counted explicitly.");

const missing = assessOrtReducedOperatorConfig({
  ...analysis,
  onnx_domain_analysis: { nodes: [{ scope: "main_graph", top_level_op_index: 0, domain: "ai.onnx", imported_opset: 13, op_name: "MatMul" }] },
}, config);
expectEqual(missing.status, "incompatible_missing_operator_identity", "A missing exact operator identity should be a definite config incompatibility.");

const alternateOpset = assessOrtReducedOperatorConfig({
  ...analysis,
  onnx_domain_analysis: { nodes: [{ scope: "main_graph", top_level_op_index: 0, domain: "ai.onnx", imported_opset: 14, op_name: "Conv" }] },
}, config);
expectEqual(alternateOpset.unresolved_node_count, 1, "An operator listed only at another opset must not be called included or excluded without registration-range binding.");

const all = parseOrtReducedOperatorConfig("!no_ops_specified_means_all_ops_are_required\n");
expectEqual(assessOrtReducedOperatorConfig(analysis, all).included_node_count, 3, "The official all-operators directive should include every artifact node.");
expectEqual(parseOrtReducedOperatorConfig("# no operators\n").status, "no_operators_required", "An empty config without the directive should require no operators.");

expectThrows(() => parseOrtReducedOperatorConfig("ai.onnx;13;Add,Add"), "duplicates one", "Duplicate operator declarations in one row should fail closed.");
expectThrows(() => parseOrtReducedOperatorConfig("!globally_allowed_types;float\nai.onnx;13;Cast{\"inputs\":{\"0\":[\"float\"]}}"), "cannot combine", "Global and per-op type reduction should be mutually exclusive.");
expectThrows(() => parseOrtReducedOperatorConfig("ai.onnx;13;Cast{\"inputs\":{}"), "unmatched", "Unbalanced type JSON should fail closed.");
expect(config.source_documents.every((row) => /^[a-f0-9]{64}$/.test(row.sha256)), "Official parser and build-contract sources should be SHA-256 pinned.");

done();
