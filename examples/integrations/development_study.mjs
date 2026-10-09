/** Checkout example: reuse the workbench's common engines, not a new SDK API. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { initSync, analyze_tflite_for_target, project_tflite_redesign } from "../../pkg/tflite_wasm_audit.js";
import { getArtifactIrContext } from "../../web/lib/artifact-ir-context.js";
import { collectWeightAnalysis } from "../../web/lib/weight-analysis.js";
import { prepareMagnitudePruning, simulateMagnitudePruning } from "../../web/lib/weight-pruning.js";
import { buildWeightQueryVisual } from "../../web/lib/weight-query-visual.js";
import { buildRedesignImplementationFiles } from "../../web/lib/redesign-codegen.js";
import { escapeXml } from "../../web/lib/weight-visuals.js";

const root = fileURLToPath(new URL("../../", import.meta.url));
const [inputArgument, outputArgument, target = "android_mid_a55"] = process.argv.slice(2);
assert(inputArgument && outputArgument, "Usage: node development_study.mjs MODEL.tflite NEW_OUTPUT_DIRECTORY [TARGET]");
const input = path.resolve(inputArgument), output = path.resolve(outputArgument);
const emitted = [];
assert.equal(path.extname(input).toLowerCase(), ".tflite", "This redesign example supports standalone TFLite only");
await mkdir(output, { recursive: false }); // Never mix evidence from prior studies.
const bytes = new Uint8Array(await readFile(input));
const sha256 = digest(bytes), filename = path.basename(input);
const wasm = await readFile(path.join(root, "pkg/tflite_wasm_audit_bg.wasm"));
const packageVersion = JSON.parse(await readFile(path.join(root, "package.json"), "utf8")).version;
initSync({ module: wasm });
const analysis = analyze_tflite_for_target(bytes, filename, target);
analysis.model_sha256 = sha256;
const context = getArtifactIrContext(analysis, { filename, format: "tflite", sha256, size: bytes.length });
assert(context?.model_ir, "Model IR unavailable");
const model = context.model_ir;
const evidence = await collectWeightAnalysis(model, analysis, bytes);
await json("source/model-ir.json", model);
await json("source/weight-ir.json", evidence.weight_ir);
await json("source/weight-analysis.json", evidence.weight_analysis);
await json("source/block-inventory.json", analysis.block_inventory);

// Explicit example heuristic: prioritize an internal pointwise convolution
// with observed similar filters and a substantial source MAC contribution.
// Similarity nominates an experiment; it does not authorize channel removal.
const bindings = new Map(model.weight_bindings.bindings.map(row => [row.id, row]));
const operations = new Map(model.program.operations.map(row => [row.id, row]));
const tensors = new Map(evidence.weight_ir.tensors.map(row => [row.id, row]));
const eligible = [];
for (const row of evidence.weight_analysis.tensors) {
  if (row.status !== "assessed" || row.axes?.channel_meaning !== "output_channel"
      || row.shape.length !== 4 || row.shape[1] !== 1 || row.shape[2] !== 1
      || row.shape[0] < 64 || row.similarity.status !== "assessed" || !row.similarity.pairs.length) continue;
  const refs = tensors.get(row.weight_ref).binding_refs;
  const linked = [...new Set(refs.map(ref => bindings.get(ref)?.operation_ref).filter(Boolean))];
  if (linked.length !== 1) continue; // Shared storage needs a different policy.
  const operation = operations.get(linked[0]);
  if (operation?.native_op.name !== "CONV_2D") continue;
  const owners = analysis.block_inventory.blocks.filter(block => block.op_indices.includes(operation.native_index));
  if (owners.length !== 1) continue;
  const block = owners[0];
  if (block.block_type !== "depthwise_separable" || block.residual
      || block.op_indices.at(-1) !== operation.native_index || block.channels.output !== row.shape[0]) continue;
  eligible.push({ weight_ref: row.weight_ref, operation_ref: operation.id, op_index: operation.native_index,
    block_id: block.block_id, source_channels: row.shape[0], source_block_macs: block.aggregates.macs,
    similarity_pair_count: row.similarity.pairs.length, similarity_threshold: row.similarity.threshold,
    observed_pairs: row.similarity.pairs });
}
eligible.sort((a, b) => b.source_block_macs - a.source_block_macs || a.op_index - b.op_index);
assert(eligible.length, "No supported internal block meets this example's evidence filter; no proposal was fabricated");
const selected = eligible[0];
const focused = await collectWeightAnalysis(model, analysis, bytes, {
  weightIr: evidence.weight_ir, options: { tensor_ids: [selected.weight_ref] },
});
await json("source/focused-weight-analysis.json", focused.weight_analysis);
const row = focused.weight_analysis.tensors.find(item => item.weight_ref === selected.weight_ref);
assert.equal(row.status, "assessed");
const reasons = { example_policy: "local-channel-experiment-v1", selected, eligible,
  selection_rule: "Among uniquely bound internal depthwise-separable pointwise weights with >=64 output channels and at least one signed-cosine pair >=0.95, choose the largest source block MAC count; break ties by native op index.",
  experiment_grid: "Reduce the selected block output width by 8 and 16 channels. These are declared experiment sizes, not an inferred safe pruning amount or a target alignment rule.",
  boundary: "Filter similarity is not functional redundancy. SVD is the weight unfolding, not the full convolution operator. No candidate weights or measured improvement exist yet." };
await json("source/proposal-basis.json", reasons);
for (const view of ["distribution", "channels", "similarity", "spectrum", "sparsity"]) {
  const visual = buildWeightQueryVisual(focused, selected.weight_ref, view);
  if (visual) await save(`source/plots/${view}.svg`, visual.svg);
}
const decoded = focused.decoded.get(selected.weight_ref);
const pruning = prepareMagnitudePruning({ matrix: decoded.matrix, row,
  source: focused.weight_analysis.source, weight_ir_sha256: evidence.weight_ir.weight_ir_sha256 });
await json("source/pruning-preview-50.json", simulateMagnitudePruning(pruning, { target_percent: 50 }));

const sourceInput = analysis.inputs.find(tensor => tensor.shape?.length === 4);
assert(sourceInput && sourceInput.shape.every(value => Number.isInteger(value) && value > 0), "Static rank-4 input required");
const baseRequest = { schema: "deepbom.redesign_request.v1", source_sha256: sha256,
  input_height: sourceInput.shape[1], input_width: sourceInput.shape[2],
  width_multiplier: 1, activation_dtype: "source", block_edits: [] };
const runs = [];
let baselinePlan;
for (const [index, channels] of [selected.source_channels, selected.source_channels - 8, selected.source_channels - 16].entries()) {
  const name = index === 0 ? "baseline" : `local-channels-${channels}`;
  const request = { ...baseRequest, block_edits: index === 0 ? [] : [{ block_id: selected.block_id, output_channels: channels }] };
  const projection = project_tflite_redesign(bytes, filename, target, request);
  assert.equal(projection.source.sha256_before, sha256);
  assert.equal(projection.source.sha256_after, sha256);
  assert.equal(projection.source.loaded_source_bytes_unchanged, true);
  assert.equal(projection.projection_status, "PROJECTED_UNTRAINED");
  assert(!projection.constraints.some(item => item.severity === "error"), `${name}: invalid projected contracts`);
  const plan = projection.implementation_plan;
  baselinePlan ||= plan;
  assert.deepEqual(plan.model_inputs, baselinePlan.model_inputs, "External input contract changed");
  assert.deepEqual(plan.model_outputs, baselinePlan.model_outputs, "External output contract changed");
  const skeleton = value => value.nodes.map(node => ({ index: node.op_index, name: node.op_name,
    block: node.block_id, inputs: node.activation_inputs, outputs: node.outputs }));
  assert.deepEqual(skeleton(plan), skeleton(baselinePlan), "Operator topology changed");
  assert.equal(projection.metrics.projected.operator_count, analysis.operator_count);
  assert.equal(projection.projection_coverage.unassessed_op_count, 0);
  // A no-op must reconstruct the observed baseline; it is not an optimized model.
  if (index === 0) assert.equal(projection.metrics.projected.macs, analysis.total_macs);
  await json(`${name}/request.json`, request);
  await json(`${name}/projection.json`, projection);
  if (index > 0) {
    for (const file of buildRedesignImplementationFiles({ analysis, projection, request })) {
      await save(`${name}/structure/${file.name}`, file.data);
    }
  }
  const metrics = projection.metrics.projected;
  runs.push({ name, kind: index === 0 ? "observed_baseline_with_noop_projection" : "untrained_structural_proposal",
    proposed_output_channels: channels, source_artifact_sha256: sha256,
    request_sha256: digest(await readFile(path.join(output, name, "request.json"))),
    projection_sha256: digest(await readFile(path.join(output, name, "projection.json"))),
    state: index === 0 ? "development_reference" : "ready_for_training_experiment",
    constraints_checked: { same_external_inputs: true, same_external_outputs: true, same_operator_topology: true },
    metrics: { "projection.macs": metrics.macs, "projection.serialized_parameter_elements": metrics.parameter_elements,
      "projection.peak_live_activation_bytes": metrics.peak_live_activation_bytes,
      "projection.modeled_latency_ms": metrics.modeled_latency_ms,
      "projection.exact_shape_ops": projection.projection_coverage.exact_shape_rule_op_count,
      "projection.scaled_shape_ops": projection.projection_coverage.scaled_shape_fallback_op_count,
      "projection.unassessed_ops": projection.projection_coverage.unassessed_op_count },
    delta: projection.metrics.delta, coverage: projection.projection_coverage,
    generated_code: index > 0 ? { weights_included: false, exact_ops: plan.exact_codegen_op_count,
      scaffold_ops: plan.scaffold_codegen_op_count, unsupported_ops: plan.unsupported_codegen_op_count } : null });
}
assert.equal(digest(await readFile(input)), sha256, "Input file changed during the study");
const summary = { scenario: "weight-guided-local-development", engine_version: packageVersion,
  engine_wasm_sha256: digest(wasm), source: { filename, sha256, target,
    target_profile_sha256: analysis.target_profile.profile_sha256, model_ir_sha256: model.model_ir_sha256,
    weight_ir_sha256: evidence.weight_ir.weight_ir_sha256 },
  weight_coverage: evidence.weight_analysis.coverage, selected, runs,
  next_step: "Implement the chosen local channel experiment in the original training code; train or fine-tune with a compatible weight mapping; export, re-audit, and bind measured quality/runtime evidence to the new artifact SHA-256.",
  boundary: "Development hypotheses only. Projections include disclosed serialized-shape fallback rules. No optimized model bytes, trained candidate, measured latency, task quality, runtime delegate placement or clinical claim is produced." };
await json("study.json", summary);
await save("index.html", report(summary));
const artifacts = [];
// An explicit file inventory lets the MLflow adapter verify every upload.
for (const name of emitted) artifacts.push({ file: name, sha256: digest(await readFile(path.join(output, name))) });
await json("completed.json", { status: "complete", artifacts });
console.log(JSON.stringify({ output, selected: selected.block_id,
  runs: runs.map(run => ({ name: run.name, metrics: run.metrics, coverage: run.coverage.status })) }, null, 2));

function digest(value) { return createHash("sha256").update(value).digest("hex"); }
async function save(name, text) {
  assert(!path.isAbsolute(name) && !name.split(/[\\/]/).includes(".."), "Unsafe generated path");
  const destination = path.join(output, name);
  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, text, { flag: "wx" });
  emitted.push(name);
}
async function json(name, value) { await save(name, `${JSON.stringify(value, null, 2)}\n`); }
function report(study) {
  const fmt = value => value == null ? "Not assessed" : Number(value).toLocaleString("en-US", { maximumFractionDigits: 3 });
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Weight-guided development study</title>
<style>body{font:16px/1.6 system-ui,sans-serif;color:#173e36;background:#f4f7f5;margin:0}main{max-width:1120px;margin:40px auto;padding:32px;background:white}h1{font-size:32px}h2{font-size:22px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:12px;border-bottom:1px solid #d4dfda}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}.plots img{width:100%}code{overflow-wrap:anywhere}a{color:#166757}.note{padding:16px;background:#edf4f0}small{color:#4f665f} @media(max-width:600px){main{margin:0;padding:16px}.plots{display:block}table{font-size:12px}th,td{padding:5px}}</style>
<main><small>DEEPBOM × MLflow · DEVELOPMENT EXPERIMENT</small><h1>Keep the backbone. Investigate one block.</h1>
<p>Weight IR → source-bound block selection → local channel proposals → training and evaluation.</p>
<p>Selected <b>${escapeXml(selected.block_id)}</b>, op ${selected.op_index}, ${escapeXml(selected.weight_ref)}: ${selected.similarity_pair_count} filter pair(s) exceed signed cosine ${selected.similarity_threshold}. This nominates an experiment, not redundant computation.</p>
<p class="note">The operator topology and external input/output contracts are checked unchanged. Width changes propagate into adjacent tensors. Candidates contain no trained weights. Serialized shape fallback: ${runs[1].coverage.scaled_shape_fallback_op_count}/${runs[1].coverage.op_count} operations.</p>
<table><thead><tr><th>Experiment</th><th>Block channels</th><th>Projected MACs</th><th>Serialized parameter elements</th><th>Peak live activation bytes</th></tr></thead><tbody>
${runs.map(run => `<tr><td><a href="${run.name}/projection.json">${run.name}</a></td><td>${run.proposed_output_channels}</td><td>${fmt(run.metrics["projection.macs"])}</td><td>${fmt(run.metrics["projection.serialized_parameter_elements"])}</td><td>${fmt(run.metrics["projection.peak_live_activation_bytes"])}</td></tr>`).join("")}</tbody></table>
<p>These are structural projections, not measured speed or memory consumption. Accuracy remains unmeasured.</p>
<h2>Inspect the original weight evidence</h2><div class="plots">${["distribution", "channels", "similarity", "spectrum", "sparsity"].filter(view => emitted.includes(`source/plots/${view}.svg`)).map(view => `<a href="source/plots/${view}.svg"><img src="source/plots/${view}.svg" alt="${view}"></a>`).join("")}</div>
<h2>Continue the experiment</h2><p>${escapeXml(study.next_step)}</p><ul>${runs.slice(1).map(run => `<li>${run.name}: <a href="${run.name}/structure/pytorch/model.py">PyTorch structure scaffold</a> · <a href="${run.name}/structure/keras/model.py">Keras scaffold</a> · <a href="${run.name}/structure/README.md">limitations</a></li>`).join("")}</ul>
<p><a href="source/pruning-preview-50.json">Separate 50% magnitude-mask simulation</a> · <a href="source/proposal-basis.json">Selection rule and evidence</a> · <a href="study.json">Full study</a></p>
<p><small>Source SHA-256: <code>${sha256}</code><br>Target: ${escapeXml(target)} (declared planning profile, not detected hardware).</small></p></main></html>`;
}
