import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";

import {
  analyze_tflite_for_target,
  explore_tflite_redesign_pareto,
  initSync,
  project_tflite_redesign,
  target_profiles,
} from "../pkg/tflite_wasm_audit.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import {
  buildRedesignImplementationFiles,
  buildRedesignScenarioSet,
  scenarioFingerprint,
  verifyRedesignScenarioSet,
} from "../web/lib/redesign-codegen.js";
import { createZipBlob } from "../web/lib/zip.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Explorer + Redesign contract check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
const targetId = "android_mid_a55";
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const sha256 = createHash("sha256").update(bytes).digest("hex");

initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const analysis = analyze_tflite_for_target(bytes, filename, targetId);
const sourceAnalysisDigest = createHash("sha256").update(JSON.stringify(analysis)).digest("hex");

expectEqual(analysis.block_inventory?.schema, "deepbom.block_inventory.v1.1", "Block inventory schema should be pinned.");
expectEqual(analysis.block_inventory?.status, "assessed", "MobileNetV2 block inventory should be assessed.");
expect((analysis.block_inventory?.semantic_block_count || 0) > 0, "MobileNetV2 should expose graph-semantic blocks.");
expect((analysis.block_inventory?.blocks || []).some((block) => block.block_type === "inverted_bottleneck"), "MobileNetV2 should contain an inverted-bottleneck block.");
expectEqual(
  analysis.block_inventory.blocks.filter((block) => block.block_type === "inverted_bottleneck").length,
  10,
  "MobileNetV2 should preserve the ten residual MBConv graph motifs.",
);
expectEqual(
  analysis.block_inventory.blocks.filter((block) => block.block_type === "depthwise_separable").length,
  7,
  "MobileNetV2 should preserve the seven non-residual depthwise-separable motifs.",
);

const claimedOps = (analysis.block_inventory?.blocks || []).flatMap((block) => block.op_indices || []);
const uniqueClaimedOps = new Set(claimedOps);
expectEqual(uniqueClaimedOps.size, analysis.ops.length, "Block inventory should cover every serialized operator exactly once.");
expectEqual(claimedOps.length, uniqueClaimedOps.size, "No serialized operator should have ambiguous block ownership.");

for (const op of analysis.ops) {
  const payload = op.cache_payload;
  if (payload?.status !== "assessed") continue;
  expectEqual(
    payload.logical_row_payload_bytes,
    payload.input_strip_bytes + payload.output_row_bytes,
    `#${op.index} logical row payload should conserve input and output components.`,
  );
  expectEqual(
    op.row_working_set_bytes,
    payload.logical_row_payload_bytes,
    `#${op.index} row working set should equal the complete logical row payload.`,
  );
  if (op.weight_packing_risk === "warn") {
    expect(
      Number(op.weight_packing_overhead_us || 0) >= 10,
      `#${op.index} should not emit a packing warning below the single 10 us threshold.`,
    );
  }
}

const input = analysis.inputs.find((tensor) => tensor.shape?.length === 4);
const noOpRequest = {
  schema: "deepbom.redesign_request.v1",
  source_sha256: sha256,
  input_height: input.shape[1],
  input_width: input.shape[2],
  width_multiplier: 1,
  activation_dtype: "source",
  block_edits: [],
};
const noOp = project_tflite_redesign(bytes, filename, targetId, noOpRequest);

expectEqual(noOp.schema, "deepbom.redesign_projection.v1.1", "Projection schema should be pinned.");
expectEqual(noOp.projection_status, "PROJECTED_UNTRAINED", "Projection state should remain explicit.");
expectEqual(noOp.source.sha256_before, sha256, "Projection should bind the source hash.");
expectEqual(noOp.source.sha256_after, sha256, "Projection should re-hash the same loaded source bytes.");
expectEqual(noOp.source.loaded_source_bytes_unchanged, true, "Projection should verify loaded-byte immutability.");
expectEqual(noOp.request.block_edit_count, 0, "Viewing a block should not create a structural edit.");
expectEqual(noOp.op_projections.length, analysis.ops.length, "Projection should emit one node ledger row per source operator.");
expectEqual(
  noOp.impact_summary.unchanged_op_count,
  analysis.ops.length,
  "No-op projection should classify every operator as unchanged.",
);
expectEqual(noOp.impact_summary.changed_edge_count, 0, "No-op projection should not invent changed tensor edges.");
expectEqual(noOp.implementation_plan?.schema, "deepbom.redesign_implementation_plan.v1", "Implementation-plan schema should be pinned.");
expectEqual(noOp.implementation_plan?.nodes?.length, analysis.ops.length, "Implementation plan should conserve the projected operator ledger.");
expectEqual(noOp.implementation_plan?.unsupported_codegen_op_count, 0, "The verified MobileNet sample should have complete weight-free scaffold coverage.");
expectEqual(noOp.implementation_plan?.mapped_source_layer_count, analysis.ops.length, "Every verified MobileNet op should retain an artifact source-like path.");
expect(
  noOp.implementation_plan.nodes.every((node) => node.source_layer_evidence_class.startsWith("ARTIFACT_")),
  "Source-like layer mappings should be explicitly artifact-derived.",
);

const structureFiles = buildRedesignImplementationFiles({ analysis, projection: noOp, request: noOpRequest });
for (const expected of [
  "manifest.json",
  "implementation_plan.json",
  "pytorch/model.py",
  "pytorch/smoke_test.py",
  "keras/model.py",
  "keras/convert_litert.py",
]) {
  expect(structureFiles.some((file) => file.name === expected), `Structure package should contain ${expected}.`);
}
expect(
  !structureFiles.some((file) => /weight/i.test(file.name) && !/weight-free/i.test(file.name)),
  "Weight-free structure package should not emit a weight payload.",
);
for (const file of structureFiles.filter((item) => item.name.endsWith(".py"))) {
  const parsed = spawnSync("python", ["-c", "import ast,sys; ast.parse(sys.stdin.read())"], {
    input: file.data,
    encoding: "utf8",
  });
  expectEqual(parsed.status, 0, `${file.name} should parse as Python: ${parsed.stderr || ""}`);
}
const deterministicZipTimestamp = new Date(1980, 0, 1, 0, 0, 0);
const structureZipA = new Uint8Array(await createZipBlob(structureFiles, { timestamp: deterministicZipTimestamp }).arrayBuffer());
const structureZipB = new Uint8Array(await createZipBlob(structureFiles, { timestamp: deterministicZipTimestamp }).arrayBuffer());
expectEqual(
  createHash("sha256").update(structureZipA).digest("hex"),
  createHash("sha256").update(structureZipB).digest("hex"),
  "Structure package ZIP bytes should be deterministic for the same bound scenario.",
);
expectEqual(
  scenarioFingerprint(noOpRequest),
  scenarioFingerprint({ ...noOpRequest, block_edits: [...noOpRequest.block_edits].reverse() }),
  "Scenario identity should be canonical and insensitive to block-edit input ordering.",
);

const pareto = explore_tflite_redesign_pareto(bytes, filename, targetId, noOpRequest);
const paretoRepeat = explore_tflite_redesign_pareto(bytes, filename, targetId, noOpRequest);
expectEqual(pareto.schema, "deepbom.redesign_pareto.v1", "Pareto schema should be pinned.");
expectEqual(pareto.evaluated_candidate_count, 25, "Default Pareto grid should evaluate five resolutions by five width multipliers.");
expectEqual(pareto.accepted_candidate_count, pareto.candidates.length, "Accepted Pareto count should conserve emitted candidate rows.");
expect(pareto.frontier_candidate_count > 0, "Pareto search should identify at least one non-dominated candidate.");
expect(
  pareto.candidates.every((candidate) => candidate.request.source_sha256 === sha256),
  "Every Pareto candidate should remain bound to the loaded source SHA-256.",
);
expectEqual(
  createHash("sha256").update(JSON.stringify(pareto)).digest("hex"),
  createHash("sha256").update(JSON.stringify(paretoRepeat)).digest("hex"),
  "Pareto search should be deterministic for an identical artifact, target, and base request.",
);
const scenarioSet = buildRedesignScenarioSet({
  analysis,
  savedScenarios: [{
    scenarioId: scenarioFingerprint(noOpRequest),
    label: "baseline",
    request: noOpRequest,
    projection: noOp,
  }],
  pareto,
});
expectEqual(scenarioSet.schema, "deepbom.redesign_scenario_set.v1.1", "Scenario-set schema should include the provenance handoff revision.");
expectEqual(scenarioSet.scenario_count, 1, "Scenario-set count should conserve saved scenarios.");
expectEqual(scenarioSet.scenarios[0].source_mapping.mapped_source_layer_count, analysis.ops.length, "Scenario-set should retain source-like mapping coverage.");
expectEqual(scenarioSet.scenarios[0].regeneration.unsupported_codegen_op_count, 0, "Scenario-set should retain regeneration coverage.");
expectEqual(scenarioSet.pareto_search.frontier_candidate_count, pareto.frontier_candidate_count, "Scenario-set should retain the complete Pareto denominator.");
expectEqual(verifyRedesignScenarioSet(scenarioSet), true, "Scenario-set hashes and conservation should reconstruct.");
const scenarioSetRepeat = buildRedesignScenarioSet({
  analysis,
  savedScenarios: [{ scenarioId: scenarioFingerprint(noOpRequest), label: "baseline", request: noOpRequest, projection: noOp }],
  pareto: paretoRepeat,
});
expectEqual(JSON.stringify(scenarioSet), JSON.stringify(scenarioSetRepeat), "Scenario-set export should be deterministic.");
const tamperedScenarioSet = structuredClone(scenarioSet);
tamperedScenarioSet.scenarios[0].projection.metrics.projected.macs += 1;
expectThrows(() => verifyRedesignScenarioSet(tamperedScenarioSet), "projection", "Scenario-set verification should reject a modified projection.");

for (const key of ["operator_count", "macs", "operations", "parameter_elements", "serialized_parameter_bytes", "l1_watch_count"]) {
  expectEqual(
    noOp.metrics.projected[key],
    noOp.metrics.source[key],
    `No-op projection should preserve ${key}.`,
  );
}
expectEqual(
  noOp.metrics.projected.predicted_break_count,
  noOp.metrics.source.predicted_break_count,
  "No-op projection should preserve the source predicted break count.",
);

const edited = project_tflite_redesign(bytes, filename, targetId, {
  ...noOpRequest,
  input_height: Math.floor(input.shape[1] / 2),
  input_width: Math.floor(input.shape[2] / 2),
  width_multiplier: 0.75,
});
expect(edited.metrics.projected.macs < edited.metrics.source.macs, "Resolution and width reduction should reduce projected MACs.");
expect(edited.metrics.projected.parameter_elements < edited.metrics.source.parameter_elements, "Width reduction should reduce projected parameter elements.");
expect(
  edited.metrics.projected.predicted_break_count == null,
  "Structural projection should suppress delegate-break claims until a transformed artifact is re-audited.",
);
expectEqual(
  edited.metrics.projected.delegation_evidence_class,
  "NOT_ASSESSABLE_HYPOTHETICAL_ARTIFACT_NOT_MATERIALIZED",
  "Structural projection delegation evidence should fail closed.",
);
expectEqual(
  createHash("sha256").update(JSON.stringify(analysis)).digest("hex"),
  sourceAnalysisDigest,
  "Projection should not mutate the source analysis object.",
);

const editableBlock = analysis.block_inventory.blocks.find((block) =>
  block.block_type === "depthwise_separable"
  && Number(block.channels?.output || 0) > 0
  && !block.residual,
);
expect(editableBlock, "The sample should expose a non-residual channel-editable block.");
const channelRequest = {
  ...noOpRequest,
  block_edits: [{
    block_id: editableBlock.block_id,
    output_channels: Number(editableBlock.channels.output) + 8,
  }],
};
const channelEdited = project_tflite_redesign(bytes, filename, targetId, channelRequest);
expect(
  channelEdited.impact_summary.direct_edit_op_count > 0,
  "A block channel edit should identify the exact directly edited operator.",
);
expect(
  channelEdited.impact_summary.propagated_op_count > 0,
  "A block channel edit should identify downstream operators whose tensor contract changed automatically.",
);
expect(
  channelEdited.impact_summary.changed_edge_count > 0,
  "A block channel edit should expose changed producer-tensor-consumer edges.",
);
expect(
  channelEdited.op_projections.some((row) =>
    row.change_class === "direct_edit"
    && row.direct_edit_fields.includes("output_channels")
    && row.block_id === editableBlock.block_id,
  ),
  "The node ledger should bind output_channels to the exact edited block operator.",
);
expect(
  channelEdited.op_projections.some((row) =>
    row.change_class === "propagated_contract"
    && row.propagation_source_op_indices.length > 0
    && row.related_op_indices.length > 0,
  ),
  "Automatically adjusted nodes should retain their originating edit and related-node ledger.",
);
for (const edge of channelEdited.propagation_edges.filter((row) => row.changed)) {
  const producer = channelEdited.op_projections.find((row) => row.op_index === edge.producer_op_index);
  const consumer = channelEdited.op_projections.find((row) => row.op_index === edge.consumer_op_index);
  const producerTensor = producer?.projected_outputs.find((tensor) => tensor.tensor_index === edge.tensor_index);
  const consumerTensor = consumer?.projected_inputs.find((tensor) => tensor.tensor_index === edge.tensor_index);
  expectEqual(
    JSON.stringify(producerTensor?.shape),
    JSON.stringify(edge.projected_shape),
    `T${edge.tensor_index} producer contract should match the propagation edge.`,
  );
  expectEqual(
    JSON.stringify(consumerTensor?.shape),
    JSON.stringify(edge.projected_shape),
    `T${edge.tensor_index} consumer contract should match the propagation edge.`,
  );
}
expectEqual(
  createHash("sha256").update(JSON.stringify(
    project_tflite_redesign(bytes, filename, targetId, channelRequest),
  )).digest("hex"),
  createHash("sha256").update(JSON.stringify(channelEdited)).digest("hex"),
  "Identical redesign requests should produce a deterministic node propagation ledger.",
);

const residualBlock = analysis.block_inventory.blocks.find((block) =>
  block.block_type === "inverted_bottleneck"
  && block.residual
  && Number(block.channels?.output || 0) > 0,
);
expect(residualBlock, "The sample should expose a residual block for merge-contract validation.");
const incompatibleResidual = project_tflite_redesign(bytes, filename, targetId, {
  ...noOpRequest,
  block_edits: [{
    block_id: residualBlock.block_id,
    output_channels: Number(residualBlock.channels.output) + 8,
  }],
});
expectEqual(
  incompatibleResidual.status,
  "blocked",
  "A residual output-channel edit should fail closed when its identity branch cannot be inferred.",
);
expect(
  incompatibleResidual.constraints.some((row) => row.code === "RD-SHAPE-MERGE-001"),
  "An incompatible residual edit should emit the exact projected merge-contract violation.",
);
expect(
  incompatibleResidual.impact_summary.unresolved_contract_count > 0,
  "The redesign impact summary should count unresolved projected tensor contracts.",
);

const repeated = project_tflite_redesign(bytes, filename, targetId, {
  ...noOpRequest,
  block_edits: [{ block_id: residualBlock.block_id, repeat: 2 }],
});
expectEqual(
  repeated.implementation_plan.exportable,
  false,
  "Code export should fail closed while repeat topology is not materialized as projected tensor nodes.",
);
expectEqual(
  repeated.implementation_plan.non_materialized_repeat_edit_count,
  1,
  "Implementation plan should count the non-materialized repeat edit.",
);
expectThrows(
  () => buildRedesignImplementationFiles({
    analysis,
    projection: repeated,
    request: { ...noOpRequest, block_edits: [{ block_id: residualBlock.block_id, repeat: 2 }] },
  }),
  "not materialized",
  "Repeat-edit code export should fail closed.",
);

const spatialOnly = project_tflite_redesign(bytes, filename, targetId, {
  ...noOpRequest,
  input_height: Math.floor(input.shape[1] / 2),
  input_width: Math.floor(input.shape[2] / 2),
});
for (const point of spatialOnly.cache_points || []) {
  if (!(point.source_width > 0) || !(point.projected_width > 0)) continue;
  expectEqual(
    point.projected_width,
    Math.ceil(point.source_width / 2),
    `#${point.op_index} stride-chain width should preserve the serialized ceil-downsampling relation.`,
  );
}

const floatFilename = "mobilenet_v1_025_224_float.tflite";
const floatBytes = new Uint8Array(readFileSync(`web/samples/${floatFilename}`));
const floatSha = createHash("sha256").update(floatBytes).digest("hex");
const floatAnalysis = analyze_tflite_for_target(floatBytes, floatFilename, targetId);
const floatInput = floatAnalysis.inputs.find((tensor) => tensor.shape?.length === 4);
const int8Storage = project_tflite_redesign(floatBytes, floatFilename, targetId, {
  schema: "deepbom.redesign_request.v1",
  source_sha256: floatSha,
  input_height: floatInput.shape[1],
  input_width: floatInput.shape[2],
  width_multiplier: 1,
  activation_dtype: "int8",
  block_edits: [],
});
for (const point of int8Storage.cache_points || []) {
  if (!(point.source_logical_row_payload_bytes > 0) || !(point.projected_logical_row_payload_bytes > 0)) continue;
  expectEqual(
    point.projected_logical_row_payload_bytes,
    point.source_logical_row_payload_bytes / 4,
    `#${point.op_index} FLOAT32-to-INT8 logical row payload should be exactly one quarter at unchanged shape.`,
  );
}

const onnxBytes = new Uint8Array(readFileSync("web/samples/sample_cnn_float.onnx"));
const target = target_profiles().find((profile) => profile.id === targetId);
const onnx = analyzeOnnxModel(onnxBytes, "sample_cnn_float.onnx", target);
const onnxCacheOps = onnx.ops.filter((op) => op.cache_payload?.status === "assessed");
expect(onnxCacheOps.length > 0, "The ONNX sample should expose at least one deterministically assessed Conv cache payload.");
for (const op of onnxCacheOps) {
  expectEqual(
    op.cache_payload.logical_row_payload_bytes,
    op.cache_payload.input_strip_bytes + op.cache_payload.output_row_bytes,
    `ONNX #${op.index} cache payload should conserve input and output components.`,
  );
  expectEqual(
    op.row_working_set_bytes,
    op.cache_payload.logical_row_payload_bytes,
    `ONNX #${op.index} row working set should equal the complete logical row payload.`,
  );
}

expectThrows(
  () => project_tflite_redesign(bytes, filename, targetId, { ...noOpRequest, source_sha256: "0".repeat(64) }),
  "does not match",
  "Projection should reject a source hash mismatch.",
);

done(
  `Explorer + Redesign contract passed: ${analysis.block_inventory.block_count} blocks, `
  + `${analysis.ops.filter((op) => op.cache_payload?.status === "assessed").length} cache rows, `
  + `${onnxCacheOps.length} ONNX cache rows, ${noOp.projection_coverage.status} no-op coverage.`,
);
