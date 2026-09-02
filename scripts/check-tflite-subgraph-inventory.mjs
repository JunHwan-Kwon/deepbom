import { readFileSync } from "node:fs";
import { Builder } from "flatbuffers";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { buildEngineeringBundleArtifactFiles, buildMlBomDocument } from "../web/lib/report.js";

const expect = (condition, message) => { if (!condition) throw new Error(message); };
const expectEqual = (actual, expected, message) => {
  if (actual !== expected) throw new Error(`${message}: expected ${expected}, got ${actual}`);
};

function int32Vector(builder, values) {
  builder.startVector(4, values.length, 4);
  for (let index = values.length - 1; index >= 0; index -= 1) builder.addInt32(values[index]);
  return builder.endVector();
}

function offsetVector(builder, values) {
  builder.startVector(4, values.length, 4);
  for (let index = values.length - 1; index >= 0; index -= 1) builder.addOffset(values[index]);
  return builder.endVector();
}

function tensor(builder, nameValue, dtype, shapeValue = [1], shapeSignature = null) {
  const name = builder.createString(nameValue);
  const shape = int32Vector(builder, shapeValue);
  const signature = shapeSignature ? int32Vector(builder, shapeSignature) : 0;
  builder.startObject(10);
  builder.addFieldOffset(0, shape, 0);
  builder.addFieldInt8(1, dtype, 0);
  builder.addFieldOffset(3, name, 0);
  if (signature) builder.addFieldOffset(7, signature, 0);
  return builder.endObject();
}

function controlFlowOperator(builder, options, optionsType, inputsValue, outputsValue) {
  const inputs = int32Vector(builder, inputsValue);
  const outputs = int32Vector(builder, outputsValue);
  builder.startObject(14);
  builder.addFieldInt32(0, 0, 0);
  builder.addFieldOffset(1, inputs, 0);
  builder.addFieldOffset(2, outputs, 0);
  builder.addFieldInt8(3, optionsType, 0);
  builder.addFieldOffset(4, options, 0);
  return builder.endObject();
}

function plainOperator(builder, opcodeIndex, inputsValue, outputsValue) {
  const inputs = int32Vector(builder, inputsValue);
  const outputs = int32Vector(builder, outputsValue);
  builder.startObject(14);
  builder.addFieldInt32(0, opcodeIndex, 0);
  builder.addFieldOffset(1, inputs, 0);
  builder.addFieldOffset(2, outputs, 0);
  return builder.endObject();
}

function ifOperator(builder, thenSubgraph, elseSubgraph, inputs = [0, 1], outputs = [2]) {
  builder.startObject(2);
  builder.addFieldInt32(0, thenSubgraph, 0);
  builder.addFieldInt32(1, elseSubgraph, 0);
  const options = builder.endObject();
  return controlFlowOperator(builder, options, 92, inputs, outputs); // BuiltinOptions.IfOptions
}

function whileOperator(builder, conditionSubgraph, bodySubgraph, inputs = [0], outputs = [1]) {
  builder.startObject(2);
  builder.addFieldInt32(0, conditionSubgraph, 0);
  builder.addFieldInt32(1, bodySubgraph, 0);
  const options = builder.endObject();
  return controlFlowOperator(builder, options, 93, inputs, outputs); // BuiltinOptions.WhileOptions
}

function callOnceOperator(builder, initSubgraph, inputs = [], outputs = []) {
  builder.startObject(1);
  builder.addFieldInt32(0, initSubgraph, 0);
  const options = builder.endObject();
  return controlFlowOperator(builder, options, 103, inputs, outputs); // BuiltinOptions.CallOnceOptions
}

function subgraph(builder, { nameValue, tensors, inputs, outputs, operators = [] }) {
  const name = builder.createString(nameValue);
  const tensorVector = offsetVector(builder, tensors);
  const inputVector = int32Vector(builder, inputs);
  const outputVector = int32Vector(builder, outputs);
  const operatorVector = operators.length ? offsetVector(builder, operators) : 0;
  builder.startObject(6);
  builder.addFieldOffset(0, tensorVector, 0);
  builder.addFieldOffset(1, inputVector, 0);
  builder.addFieldOffset(2, outputVector, 0);
  if (operatorVector) builder.addFieldOffset(3, operatorVector, 0);
  builder.addFieldOffset(4, name, 0);
  return builder.endObject();
}

function finishModel(builder, subgraphValues, builtinCodes) {
  const subgraphs = offsetVector(builder, subgraphValues);
  const opcodeValues = builtinCodes.map((builtinCode) => {
    builder.startObject(4);
    builder.addFieldInt8(0, builtinCode, 0);
    builder.addFieldInt32(2, 1, 1);
    builder.addFieldInt32(3, builtinCode, 0);
    return builder.endObject();
  });
  const opcodes = offsetVector(builder, opcodeValues);
  builder.startObject(3);
  const emptyBuffer = builder.endObject();
  const buffers = offsetVector(builder, [emptyBuffer]);
  builder.startObject(8);
  builder.addFieldInt32(0, 3, 0);
  builder.addFieldOffset(1, opcodes, 0);
  builder.addFieldOffset(2, subgraphs, 0);
  builder.addFieldOffset(4, buffers, 0);
  const model = builder.endObject();
  builder.finish(model, "TFL3");
  return builder.asUint8Array();
}

function finishSingleOpcodeModel(builder, subgraphValues, builtinCode) {
  return finishModel(builder, subgraphValues, [builtinCode]);
}

function makeIfFixture({
  thenSubgraph = 1,
  branchOutput = 0,
  conditionDtype = 6,
  conditionShape = [1],
  conditionShapeSignature = null,
  thenInputs = [0],
  elseInputs = [0],
} = {}) {
  const builder = new Builder(2048);
  const op = ifOperator(builder, thenSubgraph, 2);
  const primary = subgraph(builder, {
    nameValue: "main",
    tensors: [tensor(builder, "condition", conditionDtype, conditionShape, conditionShapeSignature), tensor(builder, "value", 0), tensor(builder, "result", 0)],
    inputs: [0, 1],
    outputs: [2],
    operators: [op],
  });
  const thenBranch = subgraph(builder, {
    nameValue: "then_branch",
    tensors: [tensor(builder, "then_value", 0)],
    inputs: thenInputs,
    outputs: [branchOutput],
  });
  const elseBranch = subgraph(builder, {
    nameValue: "else_branch",
    tensors: [tensor(builder, "else_value", 0)],
    inputs: elseInputs,
    outputs: [0],
  });
  return finishSingleOpcodeModel(builder, [primary, thenBranch, elseBranch], 118); // BuiltinOperator.IF
}

function makeWhileFixture({
  conditionDtype = 6,
  conditionShape = [1],
  conditionShapeSignature = null,
  bodyOutputDtype = 0,
  sourceOutputs = [1],
  conditionSubgraph = 1,
  bodySubgraph = 2,
} = {}) {
  const builder = new Builder(2048);
  const op = whileOperator(builder, conditionSubgraph, bodySubgraph, [0], sourceOutputs);
  const primary = subgraph(builder, {
    nameValue: "main",
    tensors: [tensor(builder, "loop_input", 0), tensor(builder, "loop_output", 0)],
    inputs: [0],
    outputs: sourceOutputs,
    operators: [op],
  });
  const condition = subgraph(builder, {
    nameValue: "condition",
    tensors: [tensor(builder, "condition_input", 0), tensor(builder, "condition_output", conditionDtype, conditionShape, conditionShapeSignature)],
    inputs: [0],
    outputs: [1],
  });
  const body = subgraph(builder, {
    nameValue: "body",
    tensors: [tensor(builder, "body_input", 0), tensor(builder, "body_output", bodyOutputDtype)],
    inputs: [0],
    outputs: [1],
  });
  return finishSingleOpcodeModel(builder, [primary, condition, body], 119); // BuiltinOperator.WHILE
}

function makeIfComputeFixture() {
  const builder = new Builder(4096);
  const primaryOp = ifOperator(builder, 1, 2);
  const primary = subgraph(builder, {
    nameValue: "main",
    tensors: [
      tensor(builder, "condition", 6, [1]),
      tensor(builder, "value", 0, [1, 2, 2, 1]),
      tensor(builder, "result", 0, [1, 2, 2, 1]),
    ],
    inputs: [0, 1],
    outputs: [2],
    operators: [primaryOp],
  });
  const thenConv = plainOperator(builder, 1, [0, 1], [2]);
  const thenBranch = subgraph(builder, {
    nameValue: "then_conv",
    tensors: [
      tensor(builder, "then_input", 0, [1, 2, 2, 1]),
      tensor(builder, "then_filter", 0, [1, 1, 1, 1]),
      tensor(builder, "then_output", 0, [1, 2, 2, 1]),
    ],
    inputs: [0],
    outputs: [2],
    operators: [thenConv],
  });
  const elseConv = plainOperator(builder, 1, [0, 1], [2]);
  const elseBranch = subgraph(builder, {
    nameValue: "else_conv",
    tensors: [
      tensor(builder, "else_input", 0, [1, 2, 2, 1]),
      tensor(builder, "else_filter", 0, [1, 1, 1, 1]),
      tensor(builder, "else_output", 0, [1, 2, 2, 1]),
    ],
    inputs: [0],
    outputs: [2],
    operators: [elseConv],
  });
  return finishModel(builder, [primary, thenBranch, elseBranch], [118, 3]);
}

function makeConv3dFixture({ filterShape = [1, 1, 1, 2, 3] } = {}) {
  const builder = new Builder(2048);
  const op = plainOperator(builder, 0, [0, 1], [2]);
  const primary = subgraph(builder, {
    nameValue: "conv3d_main",
    tensors: [
      tensor(builder, "input", 0, [1, 2, 2, 2, 2]),
      tensor(builder, "filter", 0, filterShape),
      tensor(builder, "output", 0, [1, 2, 2, 2, 3]),
    ],
    inputs: [0],
    outputs: [2],
    operators: [op],
  });
  return finishSingleOpcodeModel(builder, [primary], 132); // BuiltinOperator.CONV_3D
}

function makeCallOnceFixture({ initSubgraph = 1, sourceInputs = [], sourceOutputs = [], initInputs = [], initOutputs = [] } = {}) {
  const builder = new Builder(1024);
  const op = callOnceOperator(builder, initSubgraph, sourceInputs, sourceOutputs);
  const primary = subgraph(builder, {
    nameValue: "main",
    tensors: sourceInputs.length || sourceOutputs.length ? [tensor(builder, "source_value", 0)] : [],
    inputs: sourceInputs,
    outputs: sourceOutputs,
    operators: [op],
  });
  const initialization = subgraph(builder, {
    nameValue: "initialization",
    tensors: initInputs.length || initOutputs.length ? [tensor(builder, "init_value", 0)] : [],
    inputs: initInputs,
    outputs: initOutputs,
  });
  return finishSingleOpcodeModel(builder, [primary, initialization], 129); // BuiltinOperator.CALL_ONCE
}

function expectRejected(bytes, fragment, label) {
  try {
    analyze_tflite_for_target(bytes, `${label}.tflite`, "android_mid_a55");
  } catch (error) {
    expect(String(error).includes(fragment), `${label} rejection should mention ${fragment}: ${error}`);
    return;
  }
  throw new Error(`${label} was accepted`);
}

initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const ifFixture = makeIfFixture();
const result = analyze_tflite_for_target(ifFixture, "if-subgraphs.tflite", "android_mid_a55");
result.model_sha256 = "b".repeat(64);
const inventory = result.tflite_subgraph_inventory;
expectEqual(inventory.schema, "deepbom.tflite_subgraph_inventory.v1.3", "Subgraph schema");
expectEqual(inventory.status, "assessed", "Subgraph status");
expectEqual(inventory.parsed_subgraph_count, 3, "Parsed subgraph count");
expectEqual(inventory.primary_operator_count, 1, "Primary operator count");
expectEqual(inventory.serialized_operator_count, 1, "Serialized operator count");
expectEqual(inventory.primary_tensor_count, 3, "Primary tensor count");
expectEqual(inventory.serialized_tensor_count, 5, "Serialized tensor count");
expectEqual(inventory.nested_tensor_count, 2, "Nested tensor count");
expectEqual(inventory.control_flow_reference_count, 2, "IF reference count");
expectEqual(inventory.control_flow_contract_count, 1, "IF contract count");
expectEqual(inventory.assessed_control_flow_contract_count, 1, "Assessed IF contract count");
expectEqual(inventory.partial_control_flow_contract_count, 0, "Partial IF contract count");
expectEqual(inventory.reachable_subgraph_count, 3, "Reachable subgraph count");
expectEqual(inventory.unreachable_subgraph_indices.length, 0, "Unreachable subgraph count");
expectEqual(inventory.rows[0].operator_histogram[0].name, "IF", "Primary histogram op");
expect(inventory.references.some((row) => row.role === "then" && row.target_subgraph_index === 1), "Then branch reference");
expect(inventory.references.some((row) => row.role === "else" && row.target_subgraph_index === 2), "Else branch reference");
expectEqual(inventory.control_flow_contracts[0].condition_contract_status, "assessed_static_single_bool", "IF BOOL singleton condition contract");
expectEqual(inventory.control_flow_sources.length, 4, "Pinned control-flow source count");
expect(inventory.control_flow_sources.some((row) => row.path.endsWith("control_flow_common.h")), "Pinned control-flow tensor propagation source");
expectEqual(inventory.nominal_mac_sources.length, 4, "Pinned nominal-MAC source count");
expect(inventory.nominal_mac_sources.some((row) => row.role === "conv_3d_dhwio_ndhwc_contract"
  && row.sha256 === "7dfd75d047b7d22f76c365d48ecb1facad4656897ed3d58a661afcb0ad503b36"), "Pinned Conv3D DHWIO contract source");
expectEqual(result.operator_count, 1, "Primary execution operator total must remain separate");
const ifArtifactIr = getArtifactIrContext(result, { filename: result.filename, format: "tflite", sha256: result.model_sha256, size: ifFixture.length }).artifact_ir;
expectEqual(ifArtifactIr.graph.totals.scope_count, 3, "Artifact IR serialized IF scope count");
expectEqual(ifArtifactIr.graph.totals.materialized_scope_count, 3, "Artifact IR materialized IF scope count");
expectEqual(ifArtifactIr.graph.totals.scope_relationship_count, 2, "Artifact IR IF scope-reference count");
expectEqual(ifArtifactIr.graph.totals.operator_count, 1, "Artifact IR simple IF all-scope operator count");
expectEqual(ifArtifactIr.graph.totals.value_count, 5, "Artifact IR simple IF all-scope value count");
expectEqual(ifArtifactIr.graph.completeness, "all_serialized_scopes_materialized", "Artifact IR IF scope completeness");
expectEqual(ifArtifactIr.overlays.static.flatMap((overlay) => overlay.rows).length, 1, "Static placement must remain primary-scope only");
expect(ifArtifactIr.overlays.static.flatMap((overlay) => overlay.rows).every((row) => row.subject_ref.includes("scope:tflite:subgraph:0")), "Static placement must not be copied onto nested subgraphs");
const deep = result.tflite_subgraph_deep_analysis;
expectEqual(deep.schema, "deepbom.tflite_subgraph_deep_analysis.v1", "Deep-scope schema");
expectEqual(deep.status, "assessed_all_serialized_subgraphs", "Deep-scope status");
expectEqual(deep.assessed_subgraph_count, 3, "All IF scopes must be deep-assessed");
expectEqual(deep.rows.length, 3, "Deep-scope row count");
expect(deep.rows.every((row, index) => row.subgraph_index === index && row.operator_count === inventory.rows[index].operator_count
  && row.tensor_count === inventory.rows[index].tensor_count), "Deep-scope identity and structural denominators must match inventory");
expectEqual(deep.rows[0].advanced_numerical_storage, "referenced_top_level_without_duplication", "Primary deep evidence storage");
expect(deep.rows[0].advanced_numerical_evidence == null && deep.rows[0].advanced_numerical_evidence_pointers.length === 18,
  "Primary deep evidence must use top-level pointers without duplication");
expect(deep.rows.slice(1).every((row) => row.advanced_numerical_storage === "embedded_in_scope_row"
  && row.advanced_numerical_evidence && row.advanced_numerical_evidence_pointers.length === 0),
"Nested deep evidence must be embedded in its own scope row");

const mlBomDocument = buildMlBomDocument(result, { hash: result.model_sha256, targetId: result.target_profile.id });
const bundle = buildEngineeringBundleArtifactFiles(result, {
  reportContext: { identity: { filename: result.filename, format: "tflite", sha256: result.model_sha256 } },
  rawEvidenceContext: { identity: { filename: result.filename, format: "tflite", sha256: result.model_sha256 } },
  mlBomDocument,
});
const report = bundle.find((file) => file.name === "engineering_report.md")?.data || "";
const evidence = JSON.parse(bundle.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
expect(report.includes("## TFLite Subgraph And Control-flow Inventory"), "Engineering Report subgraph section");
expect(report.includes("then_branch") && report.includes("else_branch"), "Engineering Report branch names");
expect(report.includes("Pinned Control-flow Prepare Contracts") && report.includes("tensorflow/lite/kernels/if.cc"), "Engineering Report control-flow contract and source");
expect(report.includes("nested serialized operators"), "Engineering Report execution-count boundary");
expect(report.includes("## TFLite Per-subgraph Deep Analysis"), "Engineering Report deep-scope section");
expect(report.includes("no cross-control-flow execution total") || report.includes("Rows are never summed"), "Engineering Report deep-scope aggregation boundary");
expectEqual(evidence.evidence?.conformance_report?.status, "pass", "Subgraph bundle conformance");

const computeFixture = makeIfComputeFixture();
const computeResult = analyze_tflite_for_target(computeFixture, "if-compute-subgraphs.tflite", "android_mid_a55");
computeResult.model_sha256 = "c".repeat(64);
const computeInventory = computeResult.tflite_subgraph_inventory;
const computeDeep = computeResult.tflite_subgraph_deep_analysis;
expectEqual(computeResult.operator_count, 1, "Primary operator count must exclude branch operators");
expectEqual(computeInventory.serialized_operator_count, 3, "All serialized compute/control operators");
expectEqual(computeInventory.nested_operator_count, 2, "Nested branch operator count");
const computeArtifactIr = getArtifactIrContext(computeResult, { filename: computeResult.filename, format: "tflite", sha256: computeResult.model_sha256, size: computeFixture.length }).artifact_ir;
expectEqual(computeArtifactIr.graph.totals.operator_count, 3, "Artifact IR all serialized branch operators");
expectEqual(computeArtifactIr.graph.totals.assessed_macs.decimal, "0", "Artifact IR primary entrypoint MAC subtotal");
expectEqual(computeArtifactIr.graph.totals.serialized_scope_assessed_macs.decimal, "8", "Artifact IR independent serialized-scope nominal subtotal");
expectEqual(computeArtifactIr.overlays.static.flatMap((overlay) => overlay.rows).length, 1, "Nested branch operators must not inherit primary static placement");
expectEqual(computeInventory.rows[0].intrinsic_cost.status, "assessed_no_mac_compute", "Primary IF intrinsic status");
expect(computeInventory.rows[0].intrinsic_cost.complete_nominal_macs == null, "Control-flow op has no MAC total");
for (const [rowIndex, role] of [[1, "then"], [2, "else"]]) {
  const row = computeInventory.rows[rowIndex];
  expectEqual(row.intrinsic_cost.status, "assessed", `${role} branch intrinsic status`);
  expectEqual(row.intrinsic_cost.complete_nominal_macs, 4, `${role} branch nominal MACs`);
  expectEqual(row.intrinsic_cost.complete_nominal_macs_decimal, "4", `${role} branch lossless MAC decimal`);
  expectEqual(row.intrinsic_cost.assessed_nominal_mac_operator_count, 1, `${role} branch MAC coverage numerator`);
  expectEqual(row.intrinsic_cost.mac_compute_operator_count, 1, `${role} branch MAC coverage denominator`);
  expectEqual(row.intrinsic_cost.logical_tensor_payload_bytes, 36, `${role} branch tensor payload`);
  expectEqual(row.intrinsic_cost.logical_operator_io_payload_bytes, 36, `${role} branch operator I/O payload`);
  expectEqual(row.intrinsic_cost.graph_input_payload_bytes, 16, `${role} branch graph input payload`);
  expectEqual(row.intrinsic_cost.graph_output_payload_bytes, 16, `${role} branch graph output payload`);
  expectEqual(row.operator_intrinsics[0].mac_assessment_status, "assessed_nominal", `${role} branch MAC evidence class`);
  expect(row.invocation_semantics.includes("conditional branch"), `${role} branch invocation boundary`);
  const deepRow = computeDeep.rows[rowIndex];
  expectEqual(deepRow.total_macs, 4, `${role} deep branch nominal MACs`);
  expectEqual(deepRow.operator_evidence.length, 1, `${role} deep op evidence count`);
  expectEqual(deepRow.delegate.assessed_operator_count, 1, `${role} delegate assessment denominator`);
  expectEqual(deepRow.delegate.predicted_delegated_operator_count + deepRow.delegate.predicted_fallback_operator_count, 1, `${role} delegate count conservation`);
  expect(typeof deepRow.tensor_liveness.status === "string" && typeof deepRow.tensor_arena_plan.status === "string", `${role} memory evidence status`);
  expect(deepRow.advanced_numerical_evidence?.accumulator_atlas && deepRow.advanced_numerical_evidence?.requantization_fidelity,
    `${role} embedded numerical evidence: ${JSON.stringify(deepRow.advanced_numerical_evidence)}`);
}
const computeMlBom = buildMlBomDocument(computeResult, { hash: computeResult.model_sha256, targetId: computeResult.target_profile.id });
const computeBundle = buildEngineeringBundleArtifactFiles(computeResult, {
  reportContext: { identity: { filename: computeResult.filename, format: "tflite", sha256: computeResult.model_sha256 } },
  rawEvidenceContext: { identity: { filename: computeResult.filename, format: "tflite", sha256: computeResult.model_sha256 } },
  mlBomDocument: computeMlBom,
});
const computeReport = computeBundle.find((file) => file.name === "engineering_report.md")?.data || "";
const computeEvidence = JSON.parse(computeBundle.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
expect(computeReport.includes("### Per-invocation Intrinsic Cost Ledger"), "Compute fixture intrinsic report section");
expect(computeReport.includes("4 nominal MACs"), "Compute fixture exact branch MAC report value");
expect(computeReport.includes("36 B"), "Compute fixture branch payload report value");
expect(computeReport.includes("Independent Scope Evidence"), "Compute fixture deep-scope report table");
expectEqual(computeEvidence.evidence?.conformance_report?.status, "pass", "Compute subgraph bundle conformance");

const conv3dResult = analyze_tflite_for_target(makeConv3dFixture(), "conv3d-dhwio.tflite", "android_mid_a55");
const conv3dRow = conv3dResult.tflite_subgraph_inventory.rows[0];
expectEqual(conv3dRow.intrinsic_cost.status, "assessed", "Conv3D DHWIO intrinsic status");
expectEqual(conv3dRow.intrinsic_cost.complete_nominal_macs, 48, "Conv3D DHWIO nominal MACs");
expectEqual(conv3dRow.operator_intrinsics[0].mac_assessment_status, "assessed_nominal", "Conv3D DHWIO formula classification");
expect(conv3dRow.operator_intrinsics[0].mac_assessment_reason.includes("shape"), "Conv3D exact result should disclose its shape-contract basis");

const legacyConv3dResult = analyze_tflite_for_target(makeConv3dFixture({ filterShape: [3, 1, 1, 1, 2] }), "conv3d-legacy-wrong-layout.tflite", "android_mid_a55");
const legacyConv3dRow = legacyConv3dResult.tflite_subgraph_inventory.rows[0];
expectEqual(legacyConv3dRow.intrinsic_cost.status, "partial", "Wrong Conv3D layout must remain partial");
expectEqual(legacyConv3dRow.operator_intrinsics[0].mac_assessment_status, "not_assessed", "Wrong Conv3D layout must not emit exact MACs");
expect(legacyConv3dRow.operator_intrinsics[0].mac_assessment_reason.includes("DHWIO/NDHWC"), "Wrong Conv3D layout should identify the pinned tensor contract");

expectRejected(makeIfFixture({ thenSubgraph: 3 }), "targets missing subgraph 3", "invalid-subgraph-target");
expectRejected(makeIfFixture({ branchOutput: 1 }), "out-of-range tensor index 1", "invalid-nested-tensor-index");
expectRejected(makeIfFixture({ conditionDtype: 0 }), "pinned runtime requires BOOL", "invalid-if-condition-dtype");
expectRejected(makeIfFixture({ conditionShape: [2] }), "static cardinality 2", "invalid-if-condition-cardinality");
expectRejected(makeIfFixture({ thenInputs: [] }), "interface is 0/1 input/output tensor(s)", "invalid-if-branch-interface");

const whileResult = analyze_tflite_for_target(makeWhileFixture(), "while-subgraphs.tflite", "android_mid_a55");
const whileInventory = whileResult.tflite_subgraph_inventory;
expectEqual(whileInventory.control_flow_reference_count, 2, "WHILE reference count");
expectEqual(whileInventory.control_flow_contract_count, 1, "WHILE contract count");
expectEqual(whileInventory.control_flow_contracts[0].status, "assessed", "WHILE contract status");
expectEqual(whileInventory.control_flow_contracts[0].condition_contract_status, "assessed_static_single_bool", "WHILE condition status");
expect(whileInventory.references.some((row) => row.role === "condition") && whileInventory.references.some((row) => row.role === "body"), "WHILE condition/body references");
expectRejected(makeWhileFixture({ conditionDtype: 0 }), "pinned runtime requires BOOL", "invalid-while-condition-dtype");
expectRejected(makeWhileFixture({ conditionShape: [1, 1] }), "requires a scalar or one-dimensional [1] BOOL", "invalid-while-condition-shape");
expectRejected(makeWhileFixture({ bodyOutputDtype: 6 }), "loop-carried value 0 dtype contract differs", "invalid-while-body-output-dtype");
expectRejected(makeWhileFixture({ sourceOutputs: [] }), "pinned Prepare requires equal counts", "invalid-while-source-interface");

const dynamicWhile = analyze_tflite_for_target(makeWhileFixture({ conditionShapeSignature: [-1] }), "dynamic-while-subgraphs.tflite", "android_mid_a55");
expectEqual(dynamicWhile.tflite_subgraph_inventory.control_flow_contracts[0].status, "partial", "Dynamic WHILE contract status");
expectEqual(dynamicWhile.tflite_subgraph_inventory.control_flow_contracts[0].condition_contract_status, "partial_dynamic_cardinality", "Dynamic WHILE condition status");

const callOnceResult = analyze_tflite_for_target(makeCallOnceFixture(), "call-once-subgraphs.tflite", "android_mid_a55");
const callOnceInventory = callOnceResult.tflite_subgraph_inventory;
expectEqual(callOnceInventory.control_flow_reference_count, 1, "CALL_ONCE reference count");
expectEqual(callOnceInventory.control_flow_contracts[0].status, "assessed", "CALL_ONCE contract status");
expectEqual(callOnceInventory.control_flow_contracts[0].target_subgraph_indices[0], 1, "CALL_ONCE target");
expectRejected(makeCallOnceFixture({ initSubgraph: 0 }), "must target a distinct zero-input/zero-output", "invalid-call-once-recursion");
expectRejected(makeCallOnceFixture({ initInputs: [0] }), "must target a distinct zero-input/zero-output", "invalid-call-once-init-interface");
expectRejected(makeCallOnceFixture({ sourceInputs: [0] }), "must target a distinct zero-input/zero-output", "invalid-call-once-source-interface");

console.log("TFLite subgraph inventory checks passed (all-scope deep analysis, per-branch intrinsic MAC/payload, pinned Conv3D DHWIO, IF/WHILE/CALL_ONCE, and 15 fail-closed/partial fixtures).");
