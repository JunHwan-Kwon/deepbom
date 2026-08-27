import { buildOnnxDomainAnalysis } from "../web/lib/onnx-domain-analysis.js";
import { inferOnnxShapesWithReachableScopes } from "../web/lib/onnx-extended-shape-inference.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildEngineeringReportArtifacts } from "../web/lib/report-engineering.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { buildEngineeringBundleArtifactFiles } from "../web/lib/report.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { makeOnnxOptionalType, makeOnnxSequenceType, makeOnnxTensorType, onnxValueDescriptorFromType } from "../web/lib/onnx-type-proto.js";
import { readFileSync } from "node:fs";

const standardOpsets = [{ domain: "", version: 13 }];

const transposeFunction = {
  name: "TransposeBy",
  domain: "local.test",
  overload: "",
  inputs: ["formal_x"],
  outputs: ["formal_y"],
  attributes: ["perm"],
  attributeProtos: [],
  valueInfo: [],
  metadataProps: [],
  opsets: standardOpsets,
  nodes: [node("Transpose", ["formal_x"], ["formal_y"], attrs([
    "perm", { name: "perm", type: 7, refAttrName: "perm", valueTypesPresent: [], ints: [], graphs: [], tensors: [] },
  ]))],
};
const thenGraph = graph(
  [node("Identity", ["function_out"], ["then_out"])],
  [],
  [value("then_out")],
);
const elseGraph = graph(
  [node("Identity", ["function_out"], ["else_out"])],
  [],
  [value("else_out")],
);
const functionAndIfGraph = graph([
  node("TransposeBy", ["x"], ["function_out"], attrs([
    "perm", intsAttr("perm", [1, 0]),
  ]), "local.test"),
  node("If", ["cond"], ["y"], attrs(
    ["then_branch", graphAttr("then_branch", thenGraph)],
    ["else_branch", graphAttr("else_branch", elseGraph)],
  )),
], [value("x", "FLOAT32", [2, 3]), value("cond", "BOOL", [])], [value("y")]);
const functionAndIfModel = model(functionAndIfGraph, [...standardOpsets, { domain: "local.test", version: 1 }], [transposeFunction]);
const functionAndIf = run(functionAndIfModel);
expectEqual(JSON.stringify(functionAndIf.tensors.get("function_out").shape), JSON.stringify([3, 2]), "FunctionProto ref_attr_name binding");
expectEqual(JSON.stringify(functionAndIf.tensors.get("y").shape), JSON.stringify([3, 2]), "If branch union shape");
expectEqual(functionAndIf.evidence.extended_scope_inference.local_function_call_pass_count, 1, "local function call pass count");
expectEqual(functionAndIf.evidence.extended_scope_inference.control_flow_pass_count, 1, "If pass count");
expectEqual(functionAndIf.evidence.shape_scope.executed_reachable_scope_count, 3, "function and two branch scopes executed");
expectEqual(functionAndIf.evidence.shape_scope.unassessed_reachable_node_count, 0, "fully assessed reachable scope nodes");
expectEqual(functionAndIf.evidence.shape_scope.status, "assessed_reachable_scope", "reachable scope status");

const rankVariantThen = graph([node("Identity", ["matrix"], ["then_out"])], [], [value("then_out")]);
const rankVariantElse = graph([node("Identity", ["vector"], ["else_out"])], [], [value("else_out")]);
const rankVariantGraph = graph([
  node("If", ["cond"], ["rank_variant"], attrs(
    ["then_branch", graphAttr("then_branch", rankVariantThen)],
    ["else_branch", graphAttr("else_branch", rankVariantElse)],
  )),
  node("Transpose", ["rank_variant"], ["rank_variant_transposed"]),
], [value("cond", "BOOL", []), value("matrix", "FLOAT32", [2, 3]), value("vector", "FLOAT32", [7])], [value("rank_variant_transposed")]);
const rankVariant = run(model(rankVariantGraph, standardOpsets));
expectEqual(rankVariant.tensors.get("rank_variant").shapeDeclared, false, "ONNX's ordinary If union must not invent one rank when branches have different ranks.");
expectEqual(rankVariant.tensors.get("rank_variant").conditionalShapeContract?.status, "assessed_complete", "A finite rank-varying If union must retain a complete conditional shape contract.");
expectEqual(rankVariant.tensors.get("rank_variant").conditionalShapeVariants?.length, 2, "Both reachable If branch shape variants must be preserved.");
expectEqual(rankVariant.tensors.get("rank_variant_transposed").conditionalShapeContract?.status, "assessed_complete", "Conditional shapes must propagate through downstream source-backed rules.");
expectEqual(rankVariant.evidence.conditional_shape_contract_node_output_count, 2, "Conditional shape coverage must be counted separately from unconditional symbolic contracts.");
expectEqual(rankVariant.evidence.shape_contract_unknown_node_output_count, 0, "A complete finite branch variant set must not be mislabeled as an unknown shape contract.");

const selectedThenGraph = graph([node("Identity", ["x"], ["then_out"])], [], [value("then_out")]);
const selectedElseGraph = graph([node("Identity", ["z"], ["else_out"])], [], [value("else_out")]);
const selectedIfGraph = graph([
  node("If", ["cond"], ["selected"], attrs(
    ["then_branch", graphAttr("then_branch", selectedThenGraph)],
    ["else_branch", graphAttr("else_branch", selectedElseGraph)],
  )),
], [value("x", "FLOAT32", [2, 3]), value("z", "FLOAT32", [5, 7])], [value("selected")], [], [{
  name: "cond", dtype: "BOOL", shape: [1], shapeDeclared: true,
  staticValuesStatus: "complete", staticValuesComplete: true, staticValues: [true], staticValuesSource: "test_initializer",
}]);
const selectedIf = run(model(selectedIfGraph, standardOpsets));
const selectedIfRow = selectedIf.evidence.extended_scope_inference.control_flow_rows[0];
expectEqual(JSON.stringify(selectedIf.tensors.get("selected").shape), JSON.stringify([2, 3]), "Static singleton If condition selects the reachable branch shape");
expectEqual(selectedIfRow.condition_status, "assessed_static_single_bool", "Static singleton If condition status");
expectEqual(selectedIfRow.selected_branch, "then_branch", "Static singleton If selected branch");
expectEqual(selectedIf.evidence.shape_scope.scope_execution_rows.find((row) => row.scope.endsWith("/attribute:else_branch"))?.execution_count, 0, "Unselected If branch is validated without being counted as executed");

const selectedValidThenGraph = graph([node("Identity", ["selected_valid_x"], ["then_out"])], [], [value("then_out")]);
const unselectedSemanticFailureGraph = graph([node("Expand", ["unselected_bad_data", "unselected_bad_target"], ["else_out"])], [], [value("else_out")]);
const selectedWithInvalidRuntimeElseGraph = graph([
  node("If", ["selected_runtime_cond"], ["selected_runtime_out"], attrs(
    ["then_branch", graphAttr("then_branch", selectedValidThenGraph)],
    ["else_branch", graphAttr("else_branch", unselectedSemanticFailureGraph)],
  )),
], [
  value("selected_valid_x", "FLOAT32", [2, 3]),
  value("unselected_bad_data", "FLOAT32", [2, 4]),
], [value("selected_runtime_out")], [], [
  {
    name: "selected_runtime_cond", dtype: "BOOL", shape: [], shapeDeclared: true,
    staticValuesStatus: "complete", staticValuesComplete: true, staticValues: [true], staticValuesSource: "test_initializer",
  },
  {
    name: "unselected_bad_target", dtype: "INT64", shape: [2], shapeDeclared: true,
    staticValuesStatus: "complete", staticValuesComplete: true, staticValues: [2, 3], staticValuesSource: "test_initializer",
  },
]);
const selectedWithInvalidRuntimeElse = run(model(selectedWithInvalidRuntimeElseGraph, standardOpsets));
const selectedWithInvalidRuntimeElseRow = selectedWithInvalidRuntimeElse.evidence.extended_scope_inference.control_flow_rows[0];
expectEqual(JSON.stringify(selectedWithInvalidRuntimeElse.tensors.get("selected_runtime_out").shape), JSON.stringify([2, 3]), "A static If must preserve the selected valid branch even when an unselected branch has a condition-bound runtime semantic failure.");
expectEqual(selectedWithInvalidRuntimeElseRow.status, "partial", "An unselected runtime-semantic failure must remain condition-bound partial evidence, not invalidate the selected execution path.");
expectEqual(selectedWithInvalidRuntimeElseRow.reason_codes.some((reason) => reason.startsWith("if_unselected_branch_inference_failed:")), true, "The unselected branch failure reason must remain auditable.");
expectEqual(selectedWithInvalidRuntimeElseRow.reason_codes.some((reason) => reason.includes("Expand:expand_target_not_broadcast_compatible")), true, "Nested semantic conflict evidence must preserve the source op and reason at the parent If boundary.");

const defaultIdentityFunction = {
  name: "DefaultIdentity",
  domain: "local.test",
  overload: "",
  inputs: ["identity_x"],
  outputs: ["identity_y"],
  attributes: [], attributeProtos: [], valueInfo: [], metadataProps: [], opsets: standardOpsets,
  nodes: [node("Identity", ["identity_x"], ["identity_y"])],
};
const defaultThenGraph = graph([node("DefaultIdentity", ["formal_x"], ["then_out"], new Map(), "local.test")], [], [value("then_out")]);
const defaultElseGraphA = graph([node("Identity", ["formal_x"], ["else_a_out"])], [], [value("else_a_out")]);
const defaultElseGraphB = graph([node("Identity", ["formal_x"], ["else_b_out"])], [], [value("else_b_out")]);
const defaultGraphFunction = {
  name: "SelectDefault",
  domain: "local.test",
  overload: "",
  inputs: ["formal_cond", "formal_x"],
  outputs: ["formal_y"],
  attributes: [],
  attributeProtos: [graphAttr("then_branch", defaultThenGraph)],
  valueInfo: [], metadataProps: [], opsets: standardOpsets,
  nodes: [node("If", ["formal_cond"], ["intermediate"], attrs(
    ["then_branch", { name: "then_branch", type: 5, refAttrName: "then_branch", valueTypesPresent: [], graphs: [], tensors: [] }],
    ["else_branch", graphAttr("else_branch", defaultElseGraphA)],
  )), node("If", ["formal_cond"], ["formal_y"], attrs(
    ["then_branch", { name: "then_branch", type: 5, refAttrName: "then_branch", valueTypesPresent: [], graphs: [], tensors: [] }],
    ["else_branch", graphAttr("else_branch", defaultElseGraphB)],
  ))],
};
const defaultGraphModel = model(graph([
  node("SelectDefault", ["cond" , "x"], ["y"], new Map(), "local.test"),
], [value("cond", "BOOL", []), value("x", "FLOAT32", [2, 3])], [value("y")]), [
  ...standardOpsets, { domain: "local.test", version: 1 },
], [defaultGraphFunction, defaultIdentityFunction]);
const defaultGraphResult = run(defaultGraphModel);
expectEqual(JSON.stringify(defaultGraphResult.tensors.get("y").shape), JSON.stringify([2, 3]), "FunctionProto default GraphProto binding");
expectEqual(defaultGraphResult.evidence.shape_scope.function_default_graph_count, 1, "default GraphProto definition inventory");
expectEqual(defaultGraphResult.evidence.shape_scope.nested_graph_count, 3, "one default definition and two concrete branch definitions");
expectEqual(defaultGraphResult.evidence.shape_scope.reachable_nested_graph_count, 4, "one default definition bound at two invocation sites plus two concrete branch execution scopes");
expectEqual(defaultGraphResult.evidence.shape_scope.reachable_local_function_definition_count, 2, "execution-only dependency reached from a bound default graph enters reachable function coverage");
expectEqual(defaultGraphResult.evidence.shape_scope.reachable_scope_count, 6, "nested-graph and local-function scope rows conserve independently of definition cardinality");
expectEqual(defaultGraphResult.evidence.shape_scope.scope_execution_rows.some((row) => row.scope.endsWith("/attribute:then_branch") && row.execution_count === 1), true, "Bound default GraphProto should enter the invocation-scope execution ledger.");
expectEqual(defaultGraphResult.evidence.shape_scope.scope_execution_rows.find((row) => row.scope === "function:local.test::DefaultIdentity::")?.execution_count, 2, "A function reached only through a bound default graph preserves both executions in one definition-scoped row.");
expectEqual(defaultGraphResult.evidence.shape_scope.unassessed_reachable_node_count, 0, "bound default GraphProto residual count");

const loopBody = graph([
  node("Identity", ["cond_in"], ["cond_out"]),
  node("Identity", ["state_in"], ["state_out"]),
], [value("iter", "INT64", []), value("cond_in", "BOOL", []), value("state_in", "FLOAT32")], [
  value("cond_out", "BOOL", []), value("state_out", "FLOAT32"),
]);
const loopGraph = graph([
  node("Loop", ["trip", "cond", "state"], ["final_state"], attrs(["body", graphAttr("body", loopBody)])),
], [value("trip", "INT64", []), value("cond", "BOOL", []), value("state", "FLOAT32", [2, 3])], [value("final_state")]);
const loop = run(model(loopGraph, standardOpsets));
expectEqual(loop.evidence.extended_scope_inference.control_flow_partial_count, 1, "Loop shape-changing state remains partial");
expectEqual(loop.evidence.shape_scope.unassessed_reachable_node_count, 0, "Loop body rules executed despite unresolved state shape");
expectEqual(loop.evidence.shape_scope.reachable_scope_unresolved_output_count, 1, "Loop body unresolved output preserved separately");
expectEqual(loop.tensors.get("final_state").dtype, "FLOAT32", "Loop state dtype propagation");
expectEqual(loop.tensors.get("final_state").shapeDeclared, false, "Loop state shape is not overclaimed");

const scanBody = graph([
  node("Identity", ["state_in"], ["state_out"]),
  node("Identity", ["scan_in"], ["scan_out"]),
], [value("state_in", "FLOAT32"), value("scan_in", "FLOAT32")], [value("state_out"), value("scan_out")]);
const scanGraph = graph([
  node("Scan", ["state", "sequence"], ["final_state", "sequence_out"], attrs(
    ["body", graphAttr("body", scanBody)],
    ["num_scan_inputs", intAttr("num_scan_inputs", 1)],
  )),
], [value("state", "FLOAT32", [2, 3]), value("sequence", "FLOAT32", [5, 2, 3])], [value("final_state"), value("sequence_out")]);
const scan = run(model(scanGraph, standardOpsets));
expectEqual(JSON.stringify(scan.tensors.get("final_state").shape), JSON.stringify([2, 3]), "Scan state shape");
expectEqual(JSON.stringify(scan.tensors.get("sequence_out").shape), JSON.stringify([5, 2, 3]), "Scan output sequence axis");
expectEqual(scan.evidence.extended_scope_inference.control_flow_pass_count, 1, "Scan pass count");

const scanLastAxisGraph = graph([
  node("Scan", ["state", "sequence"], ["final_state", "sequence_out"], attrs(
    ["body", graphAttr("body", scanBody)],
    ["num_scan_inputs", intAttr("num_scan_inputs", 1)],
    ["scan_output_axes", intsAttr("scan_output_axes", [-1])],
  )),
], [value("state", "FLOAT32", [2, 3]), value("sequence", "FLOAT32", [5, 2, 3])], [value("final_state"), value("sequence_out")]);
const scanLastAxis = run(model(scanLastAxisGraph, standardOpsets));
expectEqual(JSON.stringify(scanLastAxis.tensors.get("sequence_out").shape), JSON.stringify([2, 3, 5]), "Scan negative output axis normalization");

const scan8Body = graph([
  node("Identity", ["state_in"], ["state_out"]),
  node("Identity", ["scan_in"], ["scan_out"]),
], [value("state_in", "FLOAT32"), value("scan_in", "FLOAT32")], [value("state_out"), value("scan_out")]);
const scan8Graph = graph([
  node("Scan", ["", "state", "sequence"], ["final_state", "sequence_out"], attrs(
    ["body", graphAttr("body", scan8Body)],
    ["num_scan_inputs", intAttr("num_scan_inputs", 1)],
  )),
], [value("state", "FLOAT32", [4, 2, 3]), value("sequence", "FLOAT32", [4, 5, 2, 3])], [value("final_state"), value("sequence_out")]);
const scan8 = run(model(scan8Graph, [{ domain: "", version: 8 }]));
expectEqual(JSON.stringify(scan8.tensors.get("final_state").shape), JSON.stringify([4, 2, 3]), "Scan-8 batch state reconstruction");
expectEqual(JSON.stringify(scan8.tensors.get("sequence_out").shape), JSON.stringify([4, 5, 2, 3]), "Scan-8 batch and sequence reconstruction");

const float23Type = makeOnnxTensorType("FLOAT32", [2, 3], true);
const float32Type = makeOnnxTensorType("FLOAT32", [3, 2], true);
const sequence23Type = makeOnnxSequenceType(float23Type);
const exactSequenceLoopBody = graph([
  node("Identity", ["loop_cond_in"], ["loop_cond_out"]),
  node("Identity", ["loop_sequence_in"], ["loop_sequence_out"]),
], [
  value("loop_iter", "INT64", []), value("loop_cond_in", "BOOL", []), typedValue("loop_sequence_in", sequence23Type),
], [value("loop_cond_out", "BOOL", []), typedValue("loop_sequence_out", sequence23Type)]);
const exactSequenceLoopGraph = graph([
  node("Loop", ["trip_2", "cond_true", "sequence_state"], ["sequence_final"], attrs(["body", graphAttr("body", exactSequenceLoopBody)])),
], [typedValue("sequence_state", sequence23Type, sequenceState(2, [float23Type, float23Type]))], [typedValue("sequence_final", sequence23Type)], [], [
  constantScalar("trip_2", "INT64", 2), constantScalar("cond_true", "BOOL", 1),
]);
const exactSequenceLoop = run(model(exactSequenceLoopGraph, [{ domain: "", version: 13 }]));
const exactSequenceLoopRow = exactSequenceLoop.evidence.extended_scope_inference.control_flow_rows[0];
expectEqual(exactSequenceLoopRow.status, "pass", "Loop-13 exact sequence state pass");
expectEqual(exactSequenceLoopRow.exact_expansion_status, "assessed", "Loop exact expansion status");
expectEqual(exactSequenceLoopRow.exact_iteration_count, 2, "Loop exact iteration count");
expectEqual(exactSequenceLoopRow.exact_body_node_evaluation_count, 4, "Loop exact body work");
expectEqual(exactSequenceLoopRow.exact_iteration_state_contracts.length, 2, "Loop exact expansion must expose one state-contract ledger row per reached iteration.");
expectEqual(exactSequenceLoopRow.exact_final_state_contracts[0]?.sequence_length, 2, "Loop exact final-state evidence must preserve the final sequence cardinality.");
expectEqual(exactSequenceLoopRow.exact_iteration_state_contracts[0]?.states[0]?.sequence_element_types?.[0], "tensor<FLOAT32[2,3]>", "Loop iteration evidence must preserve canonical exact sequence element types.");
expectEqual(exactSequenceLoop.tensors.get("sequence_final").sequenceLength, 2, "Loop exact sequence length preservation");
expectEqual(exactSequenceLoop.tensors.get("sequence_final").sequenceElementTypes.length, 2, "Loop exact sequence inventory preservation");
expectEqual(exactSequenceLoop.evidence.extended_scope_inference.loop_exact_expansion_count, 1, "Loop exact aggregate count");
expectEqual(exactSequenceLoop.evidence.extended_scope_inference.loop_non_dense_state_variable_count, 1, "Loop non-dense state aggregate");
expectEqual(exactSequenceLoop.evidence.extended_scope_inference.scope_execution_count, 1, "Exact Loop expansion must not inflate the public scope execution ledger");
expectEqual(buildFindingsRegister({ format: "onnx", ...exactSequenceLoop.evidence, onnx_shape_inference: exactSequenceLoop.evidence }).some((finding) => finding.finding_id === "EA-ONX-0008"), false, "Fully exact Loop inference should not create a recursive residual finding");

const optionalSequence23Type = makeOnnxOptionalType(sequence23Type);
const exactOptionalLoopBody = graph([
  node("Identity", ["loop_cond_in"], ["loop_cond_out"]),
  node("Identity", ["loop_optional_in"], ["loop_optional_out"]),
], [
  value("loop_iter", "INT64", []), value("loop_cond_in", "BOOL", []), typedValue("loop_optional_in", optionalSequence23Type),
], [value("loop_cond_out", "BOOL", []), typedValue("loop_optional_out", optionalSequence23Type)]);
const exactOptionalLoopGraph = graph([
  node("Loop", ["trip_2", "cond_true", "optional_state"], ["optional_final"], attrs(["body", graphAttr("body", exactOptionalLoopBody)])),
], [typedValue("optional_state", optionalSequence23Type, { optionalPresenceStatus: "assessed_exact", optionalPresence: true })], [typedValue("optional_final", optionalSequence23Type)], [], [
  constantScalar("trip_2", "INT64", 2), constantScalar("cond_true", "BOOL", 1),
]);
const exactOptionalLoop = run(model(exactOptionalLoopGraph, [{ domain: "", version: 16 }]));
expectEqual(exactOptionalLoop.evidence.extended_scope_inference.control_flow_rows[0].status, "pass", "Loop-16 exact optional state pass");
expectEqual(exactOptionalLoop.tensors.get("optional_final").optionalPresence, true, "Loop exact Optional presence preservation");

const invalidSequenceLoop = run(model(exactSequenceLoopGraph, [{ domain: "", version: 11 }]));
expectEqual(invalidSequenceLoop.evidence.extended_scope_inference.control_flow_fail_count, 1, "Loop-11 must reject sequence state");
expectIncludes(invalidSequenceLoop.evidence.extended_scope_inference.control_flow_rows[0].reason_codes, "loop_sequence_state_requires_opset_13:0", "Loop sequence state version reason");

const invalidOptionalLoop = run(model(exactOptionalLoopGraph, [{ domain: "", version: 13 }]));
expectEqual(invalidOptionalLoop.evidence.extended_scope_inference.control_flow_fail_count, 1, "Loop-13 must reject optional state");
expectIncludes(invalidOptionalLoop.evidence.extended_scope_inference.control_flow_rows[0].reason_codes, "loop_optional_state_requires_opset_16:0", "Loop Optional state version reason");

const reachedInvalidLoopBody = graph([
  node("Identity", ["reached_cond_in"], ["reached_cond_out"]),
  node("Concat", ["reached_state_in", "reached_int"], ["reached_state_out"], attrs(["axis", intAttr("axis", 0)])),
], [value("reached_iter", "INT64", []), value("reached_cond_in", "BOOL", []), value("reached_state_in", "FLOAT32", [1])], [
  value("reached_cond_out", "BOOL", []), value("reached_state_out", "FLOAT32", [2]),
]);
const reachedInvalidLoopGraph = graph([
  node("Loop", ["trip_2", "cond_true", "reached_state"], ["reached_final"], attrs(["body", graphAttr("body", reachedInvalidLoopBody)])),
], [value("reached_state", "FLOAT32", [1])], [value("reached_final")], [], [
  constantScalar("trip_2", "INT64", 2),
  constantScalar("cond_true", "BOOL", 1),
  {
    name: "reached_int", dtype: "INT64", shape: [1], shapeDeclared: true,
    staticValuesStatus: "complete", staticValuesComplete: true, staticValues: [1], staticValuesSource: "test_initializer",
  },
]);
const reachedInvalidLoop = run(model(reachedInvalidLoopGraph, [{ domain: "", version: 13 }]));
const reachedInvalidLoopRow = reachedInvalidLoop.evidence.extended_scope_inference.control_flow_rows.find((row) => row.op_name === "Loop");
expectEqual(reachedInvalidLoopRow.exact_expansion_status, "fail", "An exact Loop iteration that reaches a deterministic semantic violation must fail, not degrade to a runtime residual.");
expectEqual(reachedInvalidLoopRow.reason_codes.some((reason) => reason.includes("Concat:concat_input_dtype_mismatch")), true, "Reached Loop failure evidence must retain the source op and semantic reason.");
expectEqual(reachedInvalidLoop.tensors.get("reached_final").contractStatus, "invalid", "A deterministic nested Loop contract violation must invalidate the parent output instead of becoming an unresolved shape.");
expectEqual(reachedInvalidLoop.evidence.semantic_contract_conflicts[0]?.op_name, "Loop", "A nested Loop contract violation must enter the main semantic-conflict ledger.");

const invalidIterationBindingBody = graph([
  node("Identity", ["binding_cond_in"], ["binding_cond_out"]),
  node("Identity", ["binding_state_in"], ["binding_state_out"]),
], [value("binding_iter", "INT64", [1]), value("binding_cond_in", "BOOL", []), value("binding_state_in", "FLOAT32", [1])], [
  value("binding_cond_out", "BOOL", []), value("binding_state_out", "FLOAT32", [1]),
]);
const invalidIterationBindingGraph = graph([
  node("Loop", ["trip_2", "cond_true", "binding_state"], ["binding_final"], attrs(["body", graphAttr("body", invalidIterationBindingBody)])),
], [value("binding_state", "FLOAT32", [1])], [value("binding_final")], [], [constantScalar("trip_2", "INT64", 2), constantScalar("cond_true", "BOOL", 1)]);
const invalidIterationBinding = run(model(invalidIterationBindingGraph, [{ domain: "", version: 13 }]));
expectEqual(invalidIterationBinding.tensors.get("binding_final").contractStatus, "invalid", "A Loop body that declares the scalar iteration counter with rank one must fail as a serialized binding conflict.");
expectEqual(invalidIterationBinding.evidence.semantic_contract_conflicts[0]?.reason, "graph_input_binding_conflict:0:rank", "Loop binding conflicts must retain their exact input index and field.");

const nonDenseScanBody = graph([
  node("Identity", ["scan_sequence_state_in"], ["scan_sequence_state_out"]),
  node("Identity", ["scan_tensor_in"], ["scan_tensor_out"]),
], [typedValue("scan_sequence_state_in", sequence23Type), typedValue("scan_tensor_in", float23Type)], [
  typedValue("scan_sequence_state_out", sequence23Type), typedValue("scan_tensor_out", float23Type),
]);
const nonDenseScanGraph = graph([
  node("Scan", ["sequence_state", "scan_tensor"], ["sequence_final", "scan_final"], attrs(
    ["body", graphAttr("body", nonDenseScanBody)],
    ["num_scan_inputs", intAttr("num_scan_inputs", 1)],
  )),
], [typedValue("sequence_state", sequence23Type), value("scan_tensor", "FLOAT32", [2, 2, 3])], [typedValue("sequence_final", sequence23Type), value("scan_final")]);
const nonDenseScan = run(model(nonDenseScanGraph, [{ domain: "", version: 13 }]));
expectEqual(nonDenseScan.evidence.extended_scope_inference.control_flow_fail_count, 1, "Scan state remains tensor-only under the pinned schema");
expectIncludes(nonDenseScan.evidence.extended_scope_inference.control_flow_rows[0].reason_codes, "scan_state_input_non_tensor_value:0", "Scan tensor-only state reason");

const dynamicSequenceLoopGraph = graph([
  node("Loop", ["runtime_trip", "cond_true", "sequence_state"], ["sequence_final"], attrs(["body", graphAttr("body", exactSequenceLoopBody)])),
], [value("runtime_trip", "INT64", []), typedValue("sequence_state", sequence23Type, sequenceState(2, [float23Type, float23Type]))], [typedValue("sequence_final", sequence23Type)], [], [
  constantScalar("cond_true", "BOOL", 1),
]);
const dynamicSequenceLoop = run(model(dynamicSequenceLoopGraph, [{ domain: "", version: 13 }]));
expectEqual(dynamicSequenceLoop.evidence.extended_scope_inference.control_flow_partial_count, 1, "Dynamic-trip Loop remains partial");
expectEqual(dynamicSequenceLoop.tensors.get("sequence_final").sequenceLength, null, "Dynamic-trip Loop must not retain an initial sequence length as a final fact");
expectIncludes(dynamicSequenceLoop.evidence.extended_scope_inference.control_flow_rows[0].reason_codes, "loop_trip_count_runtime_unknown", "Dynamic Loop trip-count reason");
expectIncludes(buildFindingsRegister({ format: "onnx", ...dynamicSequenceLoop.evidence, onnx_shape_inference: dynamicSequenceLoop.evidence }).map((finding) => finding.finding_id), "EA-ONX-0008", "A partial Loop must enter the action queue even when its non-dense TypeProto is known");

const exactScanLoopBody = graph([
  node("Identity", ["loop_cond_in"], ["loop_cond_out"]),
  node("Identity", ["loop_state_in"], ["loop_state_out"]),
  node("Identity", ["loop_state_in"], ["loop_scan_out"]),
], [
  value("loop_iter", "INT64", []), value("loop_cond_in", "BOOL", []), value("loop_state_in", "FLOAT32", [2, 3]),
], [value("loop_cond_out", "BOOL", []), value("loop_state_out", "FLOAT32", [2, 3]), value("loop_scan_out", "FLOAT32", [2, 3])]);
const exactScanLoopGraph = graph([
  node("Loop", ["trip_2", "cond_true", "state"], ["state_final", "scan_final"], attrs(["body", graphAttr("body", exactScanLoopBody)])),
], [value("state", "FLOAT32", [2, 3])], [value("state_final", "FLOAT32", [2, 3]), value("scan_final", "FLOAT32", [2, 2, 3])], [], [
  constantScalar("trip_2", "INT64", 2), constantScalar("cond_true", "BOOL", 1),
]);
const exactScanLoop = run(model(exactScanLoopGraph, [{ domain: "", version: 13 }]));
expectEqual(JSON.stringify(exactScanLoop.tensors.get("scan_final").shape), JSON.stringify([2, 2, 3]), "Exact Loop scan leading dimension");

const zeroIterationLoopGraph = graph([
  node("Loop", ["trip_2", "cond_false", "sequence_state"], ["sequence_final"], attrs(["body", graphAttr("body", exactSequenceLoopBody)])),
], [typedValue("sequence_state", sequence23Type, sequenceState(2, [float23Type, float23Type]))], [typedValue("sequence_final", sequence23Type)], [], [
  constantScalar("trip_2", "INT64", 2), constantScalar("cond_false", "BOOL", 0),
]);
const zeroIterationLoop = run(model(zeroIterationLoopGraph, [{ domain: "", version: 13 }]));
expectEqual(zeroIterationLoop.evidence.extended_scope_inference.control_flow_rows[0].exact_iteration_count, 0, "False initial Loop condition gives zero iterations");
expectEqual(zeroIterationLoop.tensors.get("sequence_final").sequenceLength, 2, "Zero-iteration Loop returns the exact initial state");

const unknownConditionLoopBody = graph([
  node("Identity", ["runtime_condition"], ["loop_cond_out"]),
  node("Identity", ["loop_sequence_in"], ["loop_sequence_out"]),
], [
  value("loop_iter", "INT64", []), value("loop_cond_in", "BOOL", []), typedValue("loop_sequence_in", sequence23Type),
], [value("loop_cond_out", "BOOL", []), typedValue("loop_sequence_out", sequence23Type)]);
const unknownConditionLoopGraph = graph([
  node("Loop", ["trip_2", "cond_true", "sequence_state"], ["sequence_final"], attrs(["body", graphAttr("body", unknownConditionLoopBody)])),
], [value("runtime_condition", "BOOL", []), typedValue("sequence_state", sequence23Type, sequenceState(2, [float23Type, float23Type]))], [typedValue("sequence_final", sequence23Type)], [], [
  constantScalar("trip_2", "INT64", 2), constantScalar("cond_true", "BOOL", 1),
]);
const unknownConditionLoop = run(model(unknownConditionLoopGraph, [{ domain: "", version: 13 }]));
expectEqual(unknownConditionLoop.evidence.extended_scope_inference.control_flow_partial_count, 1, "Loop with a runtime body condition remains partial after the last proven iteration");
expectEqual(unknownConditionLoop.evidence.extended_scope_inference.control_flow_rows[0].exact_iteration_count, 1, "Loop discloses the proven iteration prefix");
expectIncludes(unknownConditionLoop.evidence.extended_scope_inference.control_flow_rows[0].reason_codes, "loop_body_condition_runtime_unknown_after_iteration:0", "Loop runtime body-condition reason");

const earlyExitLargeTripLoopBody = graph([
  node("Identity", ["cond_false"], ["loop_cond_out"]),
  node("Identity", ["loop_sequence_in"], ["loop_sequence_out"]),
], [
  value("loop_iter", "INT64", []), value("loop_cond_in", "BOOL", []), typedValue("loop_sequence_in", sequence23Type),
], [value("loop_cond_out", "BOOL", []), typedValue("loop_sequence_out", sequence23Type)]);
const earlyExitLargeTripLoopGraph = graph([
  node("Loop", ["trip_5000", "cond_true", "sequence_state"], ["sequence_final"], attrs(["body", graphAttr("body", earlyExitLargeTripLoopBody)])),
], [typedValue("sequence_state", sequence23Type, sequenceState(2, [float23Type, float23Type]))], [typedValue("sequence_final", sequence23Type)], [], [
  constantScalar("trip_5000", "INT64", 5000), constantScalar("cond_true", "BOOL", 1), constantScalar("cond_false", "BOOL", 0),
]);
const earlyExitLargeTripLoop = run(model(earlyExitLargeTripLoopGraph, [{ domain: "", version: 13 }]));
expectEqual(earlyExitLargeTripLoop.evidence.extended_scope_inference.control_flow_rows[0].exact_expansion_status, "assessed", "A large trip bound remains exactly assessable when an artifact-known body condition proves early termination");
expectEqual(earlyExitLargeTripLoop.evidence.extended_scope_inference.control_flow_rows[0].exact_iteration_count, 1, "Large-bound Loop exact early-exit iteration count");

const sequenceMapBody = graph([
  node("Transpose", ["item"], ["mapped_item"], attrs(["perm", intsAttr("perm", [1, 0])])),
], [typedValue("item", float23Type)], [typedValue("mapped_item", float32Type)]);
const sequenceMapGraph = graph([
  node("SequenceMap", ["items"], ["mapped"], attrs(["body", graphAttr("body", sequenceMapBody)])),
], [typedValue("items", sequence23Type, sequenceState(2, [float23Type, float23Type]))], [typedValue("mapped", makeOnnxSequenceType(float32Type))]);
const sequenceMap = run(model(sequenceMapGraph, [{ domain: "", version: 18 }]));
expectEqual(sequenceMap.evidence.extended_scope_inference.sequence_map_pass_count, 1, "SequenceMap exact element expansion pass count");
expectEqual(sequenceMap.tensors.get("mapped").sequenceLength, 2, "SequenceMap should preserve the exact first-input sequence length.");
expectEqual(sequenceMap.tensors.get("mapped").sequenceElementTypes.length, 2, "SequenceMap should materialize a bounded exact output element inventory.");
expectEqual(JSON.stringify(sequenceMap.tensors.get("mapped").sequenceElementTypes[0].shape), JSON.stringify([3, 2]), "SequenceMap should infer each exact mapped element type through its body graph.");
expectEqual(sequenceMap.evidence.extended_scope_inference.sequence_map_rows[0].element_node_evaluation_count, 2, "SequenceMap should disclose exact element/body work.");
expectEqual(sequenceMap.evidence.extended_scope_inference.status, "assessed", "Exact SequenceMap and body scopes should produce an assessed recursive-engine status.");

const unknownSequenceMapGraph = graph([
  node("SequenceMap", ["items"], ["mapped"], attrs(["body", graphAttr("body", sequenceMapBody)])),
], [typedValue("items", sequence23Type)], [typedValue("mapped", makeOnnxSequenceType(float32Type))]);
const unknownSequenceMap = run(model(unknownSequenceMapGraph, [{ domain: "", version: 18 }]));
expectEqual(unknownSequenceMap.evidence.extended_scope_inference.sequence_map_partial_count, 1, "SequenceMap without an exact input inventory remains partial.");
expectEqual(unknownSequenceMap.evidence.extended_scope_inference.status, "partial", "A partial SequenceMap must lower the recursive-engine status even when its body scope is assessed.");
expectIncludes(unknownSequenceMap.evidence.extended_scope_inference.sequence_map_rows[0].reason_codes, "sequence_map_input_length_runtime_unknown", "SequenceMap runtime-length reason");

const secondSequenceMapBody = graph([
  node("Identity", ["first_item"], ["mapped_item"]),
], [typedValue("first_item", float23Type), typedValue("second_item", float23Type)], [typedValue("mapped_item", float23Type)]);
const mismatchedSequenceMapGraph = graph([
  node("SequenceMap", ["items2", "items3"], ["mapped"], attrs(["body", graphAttr("body", secondSequenceMapBody)])),
], [
  typedValue("items2", sequence23Type, sequenceState(2, [float23Type, float23Type])),
  typedValue("items3", sequence23Type, sequenceState(3, [float23Type, float23Type, float23Type])),
], [typedValue("mapped", sequence23Type)]);
const mismatchedSequenceMap = run(model(mismatchedSequenceMapGraph, [{ domain: "", version: 18 }]));
expectEqual(mismatchedSequenceMap.evidence.extended_scope_inference.sequence_map_fail_count, 1, "Mismatched SequenceMap sequence lengths fail deterministically.");
expectEqual(mismatchedSequenceMap.evidence.extended_scope_inference.status, "fail", "A failed SequenceMap must fail the recursive-engine status.");
expectIncludes(mismatchedSequenceMap.evidence.extended_scope_inference.sequence_map_rows[0].reason_codes, "sequence_map_sequence_length_mismatch:1", "SequenceMap length mismatch reason");

const sequence2 = typedValue("sequence2", sequence23Type, sequenceState(2, [float23Type, float23Type]));
const sequence3 = typedValue("sequence3", sequence23Type, sequenceState(3, [float23Type, float23Type, float23Type]));
const thenSequenceGraph = graph([node("Identity", ["sequence2"], ["then_sequence"])], [], [typedValue("then_sequence", sequence23Type)]);
const elseSequenceGraph = graph([node("Identity", ["sequence3"], ["else_sequence"])], [], [typedValue("else_sequence", sequence23Type)]);
const nonDenseIfGraph = graph([
  node("If", ["cond"], ["conditional_sequence"], attrs(
    ["then_branch", graphAttr("then_branch", thenSequenceGraph)],
    ["else_branch", graphAttr("else_branch", elseSequenceGraph)],
  )),
], [value("cond", "BOOL", []), sequence2, sequence3], [typedValue("conditional_sequence", sequence23Type)]);
const nonDenseIf = run(model(nonDenseIfGraph, [{ domain: "", version: 18 }]));
expectEqual(nonDenseIf.evidence.extended_scope_inference.control_flow_pass_count, 1, "If should union compatible non-dense branch TypeProto contracts.");
expectEqual(nonDenseIf.tensors.get("conditional_sequence").valueKind, "sequence", "If output should remain a sequence instead of being coerced to a tensor.");
expectEqual(nonDenseIf.tensors.get("conditional_sequence").sequenceLength, null, "Branch-dependent sequence length must remain explicitly unresolved.");
expectEqual(nonDenseIf.tensors.get("conditional_sequence").sequenceLengthStatus, "not_assessed_branch_dependent", "If should label branch-dependent sequence length precisely.");

const optional23Type = makeOnnxOptionalType(float23Type);
const presentOptional = typedValue("present_optional", optional23Type, { optionalPresenceStatus: "assessed_exact", optionalPresence: true });
const emptyOptional = typedValue("empty_optional", optional23Type, { optionalPresenceStatus: "assessed_exact", optionalPresence: false });
const thenOptionalGraph = graph([node("Identity", ["present_optional"], ["then_optional"])], [], [typedValue("then_optional", optional23Type)]);
const elseOptionalGraph = graph([node("Identity", ["empty_optional"], ["else_optional"])], [], [typedValue("else_optional", optional23Type)]);
const optionalIfGraph = graph([
  node("If", ["cond"], ["conditional_optional"], attrs(
    ["then_branch", graphAttr("then_branch", thenOptionalGraph)],
    ["else_branch", graphAttr("else_branch", elseOptionalGraph)],
  )),
], [value("cond", "BOOL", []), presentOptional, emptyOptional], [typedValue("conditional_optional", optional23Type)]);
const optionalIf = run(model(optionalIfGraph, [{ domain: "", version: 18 }]));
expectEqual(optionalIf.tensors.get("conditional_optional").optionalPresence, null, "Branch-dependent Optional presence must not be guessed.");
expectEqual(optionalIf.tensors.get("conditional_optional").optionalPresenceStatus, "not_assessed_branch_dependent", "If should label branch-dependent Optional presence precisely.");

const missingAttributeGraph = graph([
  node("TransposeBy", ["x"], ["y"], new Map(), "local.test"),
], [value("x", "FLOAT32", [2, 3])], [value("y")]);
const missingAttribute = run(model(missingAttributeGraph, [...standardOpsets, { domain: "local.test", version: 1 }], [transposeFunction]));
expectEqual(missingAttribute.evidence.status, "fail", "invalid local function call fails the recursive shape contract");
expectEqual(missingAttribute.evidence.extended_scope_inference.local_function_call_fail_count, 1, "required function attribute failure");
expectIncludes(missingAttribute.evidence.extended_scope_inference.function_call_rows[0].reason_codes, "function_required_attribute_missing:perm", "required function attribute reason");

const parsedFixture = analyzeOnnxModel(
  new Uint8Array(readFileSync("scripts/fixtures/onnx_recursive_scope.onnx")),
  "onnx_recursive_scope.onnx",
);
const parsedTensor = (name) => parsedFixture.tensors.find((tensor) => tensor.name === name);
expectEqual(JSON.stringify(parsedTensor("function_out")?.shape), JSON.stringify([3, 2]), "serialized FunctionProto attribute reference");
expectEqual(JSON.stringify(parsedTensor("if_out")?.shape), JSON.stringify([3, 2]), "serialized If branch union");
expectEqual(parsedTensor("loop_final")?.shape_declared, true, "serialized Loop output retains declared symbolic rank");
expectEqual(JSON.stringify(parsedTensor("scan_final")?.shape), JSON.stringify([2, 3]), "serialized Scan state output");
expectEqual(JSON.stringify(parsedTensor("scan_sequence_out")?.shape), JSON.stringify([5, 2, 3]), "serialized Scan sequence output");
expectEqual(parsedFixture.onnx_shape_inference.extended_scope_inference.local_function_call_pass_count, 1, "serialized FunctionProto call count");
expectEqual(parsedFixture.onnx_shape_inference.extended_scope_inference.control_flow_pass_count, 2, "serialized If and Scan pass count");
expectEqual(parsedFixture.onnx_shape_inference.extended_scope_inference.control_flow_partial_count, 1, "serialized Loop partial count");
expectEqual(parsedFixture.onnx_shape_inference.extended_scope_inference.status, "partial", "serialized partial Loop lowers recursive-engine status");
expectEqual(parsedFixture.onnx_shape_inference.extended_scope_inference.schema, "deepbom.onnx_extended_shape_inference.v1.6", "serialized recursive engine schema with intrinsic-cost, finite conditional-shape variants, and deterministic nested-conflict propagation");
const parsedIntrinsic = parsedFixture.onnx_shape_inference.extended_scope_inference;
expectEqual(parsedIntrinsic.main_graph_intrinsic_cost.schema, "deepbom.onnx_scope_intrinsic_cost.v1", "main-graph intrinsic-cost schema");
expectEqual(parsedIntrinsic.main_graph_intrinsic_cost.status, "partial", "main-graph unknown Loop payload remains a residual instead of becoming zero");
expectEqual(parsedIntrinsic.main_graph_intrinsic_cost.assessed_operator_io_payload_bytes_decimal, "361", "main-graph assessed logical operator-I/O subtotal");
expectEqual(parsedIntrinsic.intrinsic_cost_variant_count, 5, "unique nested-scope intrinsic-cost variant count");
expectEqual(parsedIntrinsic.intrinsic_cost_variant_overflow_count, 0, "intrinsic-cost variant overflow count");
expectEqual(parsedIntrinsic.intrinsic_cost_unassessed_execution_count, 0, "intrinsic-cost unassessed execution count");
expectEqual(parsedIntrinsic.scope_rows.reduce((sum, row) => sum + row.intrinsic_cost_variants.reduce((count, cost) => count + cost.observation_count, 0)
  + row.intrinsic_cost_variant_overflow_count + row.intrinsic_cost_unassessed_execution_count, 0), parsedIntrinsic.scope_execution_count, "intrinsic-cost observations conserve recursive scope executions");
expectEqual(parsedIntrinsic.scope_rows.find((row) => row.scope.endsWith("node:3/attribute:body"))?.intrinsic_cost_variants[0]?.complete_operator_io_payload_bytes_decimal, "96", "Scan-body one-invocation logical operator-I/O payload");
expectEqual(parsedFixture.onnx_shape_inference.shape_scope.unassessed_reachable_node_count, 0, "serialized recursive scope residual count");
expectEqual(parsedFixture.onnx_shape_inference.shape_scope.reachable_scope_unresolved_output_count, 1, "serialized recursive scope unresolved output count");
const parsedFindingIds = buildFindingsRegister(parsedFixture).map((finding) => finding.finding_id);
expectIncludes(parsedFindingIds, "EA-ONX-0008", "serialized unresolved Loop scope finding");
expectEqual(parsedFindingIds.includes("EA-ONX-0009"), false, "valid serialized recursive contracts should not emit a failure finding");
const parsedReport = buildEngineeringReportArtifacts(parsedFixture, {
  identity: { filename: parsedFixture.filename, format: "onnx" },
  generatedAt: "2026-07-22T00:00:00.000Z",
}).report;
expectIncludes(parsedReport, "### Recursive Shape Scope Assessment", "recursive scope report table");
expectIncludes(parsedReport, "### FunctionProto Call Contracts", "FunctionProto report table");
expectIncludes(parsedReport, "### If / Loop / Scan Shape Contracts", "control-flow report table");
expectIncludes(parsedReport, "Bounded exact Loop expansion", "exact Loop report summary");
expectIncludes(parsedReport, "Exact Loop expansion", "exact Loop report row contract");
expectIncludes(parsedReport, "### Recursive Engine Execution Ledger", "recursive-engine execution ledger");
expectIncludes(parsedReport, "One-invocation intrinsic cost variants", "recursive-engine intrinsic-cost variants");
expectIncludes(parsedReport, "361 assessed subtotal", "main-graph partial logical operator-I/O subtotal");
expectIncludes(parsedReport, "96 complete", "Scan-body exact one-invocation payload");
expectIncludes(parsedReport, "not summed across FunctionProto calls", "intrinsic-cost no-double-count boundary");
expectIncludes(parsedReport, "0428224a3cb2b5aabf87dab3dfca94988c3a913d73b6f39fa295980060b97594", "matrix-operation cost source hash in report");
expectIncludes(parsedReport, "All-kind output contract coverage", "output contract coverage wording");
expectIncludes(parsedReport, "48ced14e52a8c2d9a8e230f1be3c6428c6bd074e923d035f962a0626215d3d33", "control-flow source hash in report");
expectIncludes(parsedReport, "67d03c30742c96fae1f79831b28d2409d5dcb4921ea77f694a4172f66ceebeff", "current control-flow schema source hash in report");
expectIncludes(parsedReport, "de85b9527008a725e718c593313c38c7aeea30c8781027ddea546a5dd80f5283", "historical control-flow schema source hash in report");
const parsedMlBom = buildMlBomDocument(parsedFixture, { hash: "fixture" });
const parsedProperties = new Map([
  ...(parsedMlBom.properties || []),
  ...(parsedMlBom.metadata?.component?.properties || []),
].map((property) => [property.name, property.value]));
expectEqual(parsedProperties.get("deepbom:compatibility:profile"), "deepbom.compact_mlbom_compatibility.v2", "compact ML-BOM profile");
expectEqual(parsedProperties.get("deepbom:compatibility:detailLocation"), "engineering_evidence.json#/evidence/static_analysis", "compact ML-BOM detail pointer");
for (const omittedProperty of [
  "deepbom:model:onnxLocalFunctionCalls",
  "deepbom:model:onnxControlFlowShapeNodes",
  "deepbom:model:onnxShapeReachableScopeUnresolvedOutputs",
  "deepbom:model:onnxRecursiveScopeExecutions",
  "deepbom:model:onnxRecursiveScopeDefinitions",
  "deepbom:model:onnxRecursiveResidualUnresolvedOutputs",
  "deepbom:model:onnxLoopNodes",
  "deepbom:model:onnxLoopExactExpansions",
  "deepbom:model:onnxLoopExactIterations",
  "deepbom:model:onnxLoopExactBodyNodeEvaluations",
  "deepbom:model:onnxLoopNonDenseStateVariables",
]) {
  expectEqual(parsedProperties.has(omittedProperty), false, `compact ML-BOM omits ${omittedProperty}`);
}
expectEqual((parsedMlBom.metadata.component.externalReferences || []).some((item) => item.type === "evidence" && item.url === "engineering_evidence.json"), true, "compact ML-BOM evidence reference");
const parsedBundle = buildEngineeringBundleArtifactFiles(parsedFixture, {
  reportContext: { identity: { filename: parsedFixture.filename, format: "onnx" }, generatedAt: "2026-07-22T00:00:00.000Z" },
  rawEvidenceContext: { identity: { filename: parsedFixture.filename, format: "onnx" } },
  mlBomDocument: parsedMlBom,
});
const parsedEvidence = JSON.parse(parsedBundle.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
expectEqual(parsedEvidence.evidence?.conformance_report?.status, "pass", "serialized recursive-scope Engineering Bundle conformance");

const tamperedStatusFixture = structuredClone(parsedFixture);
tamperedStatusFixture.onnx_shape_inference.extended_scope_inference.status = "assessed";
const tamperedMlBom = buildMlBomDocument(tamperedStatusFixture, { hash: "fixture" });
expectThrows(() => buildEngineeringBundleArtifactFiles(tamperedStatusFixture, {
  reportContext: { identity: { filename: tamperedStatusFixture.filename, format: "onnx" }, generatedAt: "2026-07-22T00:00:00.000Z" },
  rawEvidenceContext: { identity: { filename: tamperedStatusFixture.filename, format: "onnx" } },
  mlBomDocument: tamperedMlBom,
}), "CF-SHAPE-001", "tampered recursive-engine status must fail closed");

console.log("ONNX FunctionProto, If, Loop, Scan-8/9+, SequenceMap, and non-dense If recursive inference");

function run(inputModel) {
  const tensors = seedTensorMap(inputModel.graph);
  const domain = buildOnnxDomainAnalysis(inputModel);
  const evidence = inferOnnxShapesWithReachableScopes(inputModel.graph, tensors, inputModel, (value) => String(value), domain);
  return { tensors, evidence };
}

function seedTensorMap(inputGraph) {
  const tensors = new Map();
  for (const item of [...inputGraph.inputs, ...inputGraph.outputs, ...inputGraph.valueInfo]) {
    tensors.set(item.name, { ...item, shape: [...item.shape] });
  }
  for (const item of inputGraph.initializers) tensors.set(item.name, { ...item, shapeDeclared: true });
  for (const item of inputGraph.nodes) {
    for (const name of [...item.inputs, ...item.outputs]) {
      if (name && !tensors.has(name)) tensors.set(name, { name, dtype: "UNKNOWN", shape: [], shapeDeclared: false });
    }
  }
  return tensors;
}

function model(inputGraph, opsets, functions = []) {
  return { graph: inputGraph, opsets, functions };
}

function graph(nodes = [], inputs = [], outputs = [], valueInfo = [], initializers = []) {
  return { name: "", nodes, inputs, outputs, valueInfo, initializers };
}

function node(opType, inputs, outputs, attributes = new Map(), domain = "") {
  return { name: "", opType, domain, overload: "", inputs, outputs, attributes, duplicateAttributeNames: [] };
}

function value(name, dtype = "UNKNOWN", shape = null) {
  return { name, dtype, shape: shape == null ? [] : [...shape], shapeDeclared: shape != null };
}

function typedValue(name, typeProto, state = {}) {
  return { name, ...onnxValueDescriptorFromType(typeProto, state) };
}

function sequenceState(length, inventory) {
  return {
    sequenceLengthStatus: "assessed_exact",
    sequenceLength: length,
    sequenceElementInventoryStatus: "assessed_exact",
    sequenceElementTypes: inventory,
  };
}

function constantScalar(name, dtype, item) {
  return {
    name,
    dtype,
    shape: [],
    shapeDeclared: true,
    staticValuesStatus: "complete",
    staticValuesComplete: true,
    staticValues: [item],
    staticValuesSource: "test_initializer",
  };
}

function attrs(...entries) {
  const normalized = entries.length === 1 && Array.isArray(entries[0]) && typeof entries[0][0] === "string" ? [entries[0]] : entries;
  return new Map(normalized);
}

function graphAttr(name, nested) {
  return { name, type: 5, graph: nested, graphs: [], valueTypesPresent: [5], refAttrName: "" };
}

function intAttr(name, value) {
  return { name, type: 2, i: value, iExactDecimal: String(value), valueTypesPresent: [2], refAttrName: "" };
}

function intsAttr(name, values) {
  return { name, type: 7, ints: [...values], intExactDecimals: values.map(String), valueTypesPresent: [7], refAttrName: "" };
}

function expectEqual(actual, expected, label) {
  if (actual !== expected) throw new Error(`${label}: expected ${JSON.stringify(expected)}, received ${JSON.stringify(actual)}`);
}

function expectIncludes(values, expected, label) {
  if (!values.includes(expected)) throw new Error(`${label}: missing ${JSON.stringify(expected)} in ${JSON.stringify(values)}`);
}

function expectThrows(callback, expected, label) {
  try {
    callback();
  } catch (error) {
    if (String(error?.message || error).includes(expected)) return;
    throw new Error(`${label}: expected ${JSON.stringify(expected)}, received ${JSON.stringify(error?.message || error)}`);
  }
  throw new Error(`${label}: expected an error containing ${JSON.stringify(expected)}`);
}
