import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright";
import { finalizeNvidiaAcceleratorProfile } from "../web/lib/nvidia-accelerator-profile.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";
import { createTensorRtBuildProfile, TENSORRT_PARSER_OBSERVATION_SCHEMA } from "../web/lib/tensorrt-static-preflight.js";
import { buildCanonicalGraphIr } from "../web/lib/graph-ir.js";
import { exportGraphPng } from "../bin/graph-png-export.mjs";

const root = path.resolve(".");
const output = path.join(root, ".local-validation", "cli-graph-export");
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });

const onnxJson = run(["graph", "web/samples/sample_cnn_float.onnx", "--format", "json", "--compact"]);
const graph = JSON.parse(onnxJson);
assert.equal(graph.graph_ir.schema, "deepbom.graph_ir.v1");
assert.equal(graph.graph_ir.totals.node_count, 9);
assert.equal(graph.graph_ir.totals.tensor_count, 16);
assert.equal(graph.graph_ir.totals.macs.decimal, "6488384");
assert.equal(graph.graph_ir.nodes.reduce((sum, node) => sum + BigInt(node.macs?.decimal || "0"), 0n), 6488384n);
assert.equal(graph.visualization_manifest.rendered_node_count, 9);
assert.match(graph.graph_ir.artifact.artifact_set_sha256, /^[a-f0-9]{64}$/);
assert.equal(graph.graph_ir.projection.placement_evidence.profile_id, "tensorrt_unbound");
assert.equal(graph.graph_ir.projection.placement_evidence.original_op_engine_selection_claim, false);
assert.equal(graph.graph_ir.nodes.every((node) => node.placement.status === "NOT_ASSESSABLE"), true,
  "unbound TensorRT configuration must not become a GPU placement claim");

const tensorRtConfig = {
  execution_path: "native_tensorrt",
  expected_tensorrt_version: "10.14.1",
  expected_cuda_version: "13.0",
  device_id: 0,
  device_compute_capability: "8.9",
  precision: { tf32: true, fp16: true, bf16: false, int8: false, fp8: false },
  workspace_limit_bytes: 1_073_741_824,
  builder_optimization_level: 3,
  dla_core: null,
  allow_gpu_fallback: false,
  calibration_cache_sha256: null,
  plugins: [],
  optimization_profiles: [],
};
const tensorRtProfile = createTensorRtBuildProfile(tensorRtConfig);
const parserEvidencePath = path.join(output, "tensorrt-parser-evidence.json");
await writeFile(parserEvidencePath, JSON.stringify({
  schema: TENSORRT_PARSER_OBSERVATION_SCHEMA,
  artifact_sha256: graph.graph_ir.artifact.sha256,
  build_profile_sha256: tensorRtProfile.profile_sha256,
  build_profile_file_sha256: sha256TextHex(`${canonicalJson(tensorRtProfile)}\n`),
  build_profile: tensorRtProfile,
  execution_path: "native_tensorrt",
  tensorrt_version: "10.14.1",
  cuda_version: "13.0",
  device_id: 0,
  device_compute_capability: "8.9",
  device_identity: "Graph export fixture / CC 8.9",
  api_method: "supportsModelV2",
  subgraph_support_semantics: "per_subgraph_api_flag",
  parser_returned: true,
  collector: { binary_sha256: "b".repeat(64), source_set_sha256: "c".repeat(64), git_commit: "fixture", git_state: "clean" },
  plugins: [],
  subgraphs: [{ subgraph_index: 0, supported: true, sdk_reported_flag: true, node_indices: graph.graph_ir.nodes.map((node) => node.index) }],
  errors: [],
}), "utf8");
const observedPlacement = JSON.parse(run([
  "graph", "web/samples/sample_cnn_float.onnx", "--view", "placement", "--format", "json", "--compact",
  "--tensorrt-parser-evidence", parserEvidencePath,
]));
assert.equal(observedPlacement.graph_ir.projection.placement_evidence.parser_observation_status, "complete");
assert.equal(observedPlacement.graph_ir.nodes.every((node) => node.placement.status === "CONDITIONALLY_ELIGIBLE"), true);
assert.equal(observedPlacement.graph_ir.nodes.every((node) => node.placement.evidence_state === "ARTIFACT_ELIGIBLE"), true);

const partialMacGraph = buildCanonicalGraphIr({
  format: "onnx",
  input_tensor_indices: [0, 1],
  output_tensor_indices: [2],
  ops: [{ index: 0, name: "MatMul", domain: "ai.onnx", inputs: [0, 1], outputs: [2], output_shapes: [[-1, 8]], macs: null, macs_decimal: null }],
  tensors: [
    { index: 0, name: "a", dtype: "FLOAT32", shape: [-1, 8] },
    { index: 1, name: "b", dtype: "FLOAT32", shape: [8, 8] },
    { index: 2, name: "c", dtype: "FLOAT32", shape: [-1, 8] },
  ],
  mac_assessment: {
    status: "not_assessed", total_assessed_macs: 0, total_assessed_macs_decimal: "0",
    compute_ops: 1, assessed_compute_ops: 0, not_assessed_compute_ops: 1,
    metric_scope: "nominal tensor-contraction MACs",
  },
}, { filename: "dynamic.onnx", format: "onnx", sha256: "a".repeat(64), size: 128 });
assert.equal(partialMacGraph.totals.macs, null, "unassessed compute must not be reported as zero total MACs");
assert.equal(partialMacGraph.totals.assessed_macs.decimal, "0");
assert.equal(partialMacGraph.totals.mac_assessment.unassessed_compute_op_count, 1);

const svgArgs = ["graph", "web/samples/mobilenet_v1_025_224_float.tflite", "--view", "placement", "--format", "svg"];
const svgA = run(svgArgs), svgB = run(svgArgs);
assert.equal(svgA, svgB, "same-input SVG bytes must be deterministic");
assert.match(svgA, /^<\?xml version="1\.0"/);
assert.equal(/\b(?:NaN|Infinity|undefined)\b/.test(svgA), false);
assert.match(svgA, /CONDITIONALLY_DELEGATABLE/);
assert.equal((svgA.match(/class="node /g) || []).length, 31, "TFLite SVG node count");
const pngPathA = path.join(output, "onnx-a.png"), pngPathB = path.join(output, "onnx-b.png");
run(["graph", "web/samples/sample_cnn_float.onnx", "--format", "png", "--output", pngPathA]);
run(["graph", "web/samples/sample_cnn_float.onnx", "--format", "png", "--output", pngPathB]);
const pngA = await readFile(pngPathA), pngB = await readFile(pngPathB);
assert.equal(pngA.subarray(0, 8).toString("hex"), "89504e470d0a1a0a", "PNG signature");
assert.deepEqual(pngA, pngB, "same-input PNG bytes must be deterministic");
assert.equal(pngA.readUInt32BE(16), 760, "PNG fixed scene width");
assert.equal(pngA.readUInt32BE(20), 1462, "PNG fixed scene height");

const posterAnalysis = {
  format: "onnx",
  input_tensor_indices: [0], output_tensor_indices: [230],
  ops: Array.from({ length: 230 }, (_, index) => ({ index, name: "Identity", domain: "ai.onnx", inputs: [index], outputs: [index + 1], macs: 0 })),
  tensors: Array.from({ length: 231 }, (_, index) => ({ index, name: `tensor_${index}`, dtype: "FLOAT32", shape: [1, 8] })),
  mac_assessment: { status: "assessed", total_assessed_macs: 0, total_assessed_macs_decimal: "0", compute_ops: 0, assessed_compute_ops: 0, not_assessed_compute_ops: 0 },
};
const posterGraph = buildCanonicalGraphIr(posterAnalysis, { filename: "large.onnx", format: "onnx", sha256: "d".repeat(64), size: 4096 });
const poster = exportGraphPng(posterGraph, { view: "placement" });
assert.equal(poster.manifest.raster_layout, "readable_row_major_poster");
assert.equal(poster.manifest.raster_scale, 1);
assert.equal(poster.manifest.dimensions.width >= 3000 && poster.manifest.dimensions.height >= 1500, true,
  `large PNG must retain readable node dimensions: ${JSON.stringify(poster.manifest.dimensions)}`);
assert.equal(poster.bytes.readUInt32BE(16), poster.manifest.dimensions.width);
assert.equal(poster.bytes.readUInt32BE(20), poster.manifest.dimensions.height);

const ggufJson = JSON.parse(run(["graph", "web/samples/tinymqa1m.Q4_0.gguf", "--view", "architecture", "--format", "json", "--compact"]));
assert.equal(ggufJson.graph_ir.projection.executable_dag_claim, false, "GGUF must not be promoted to an executable DAG");
assert.equal(ggufJson.graph_ir.projection.kind, "llm_layer_storage_architecture_projection");
assert.equal(ggufJson.graph_ir.nodes.length, 4);

const profilePath = path.join(output, "nvidia-profile.json");
await writeFile(profilePath, JSON.stringify(finalizeNvidiaAcceleratorProfile(testNvidiaProfile())), "utf8");
const placementJson = JSON.parse(run([
  "graph", "web/samples/tinymqa1m.Q4_0.gguf", "--view", "placement", "--format", "json", "--compact",
  "--context", "4096", "--batch", "1", "--state-bits", "16", "--accelerator-profile", profilePath,
]));
assert.equal(placementJson.graph_ir.projection.placement_scenario.source, "cli_declared");
assert.equal(placementJson.graph_ir.nodes.filter((node) => node.placement.backend === "nvidia_accelerator").length, 1,
  "exact serialized lower-bound placement must retain only the highest-index layer under the test capacity");
assert.equal(placementJson.graph_ir.nodes.find((node) => node.index === 3).placement.backend, "nvidia_accelerator");
assert.equal(placementJson.graph_ir.nodes.some((node) => node.placement.status.includes("FIT")), false, "placement view must not claim fit");

const htmlPath = path.join(output, "gguf-architecture.html");
run(["graph", "web/samples/tinymqa1m.Q4_0.gguf", "--view", "architecture", "--format", "html", "--output", htmlPath]);
const html = await readFile(htmlPath, "utf8");
assert.match(html, /Decoder layer 0/);
assert.match(html, /do not serialize an executable operator DAG/);
assert.equal(/https?:\/\/(?!www\.w3\.org)/.test(html), false, "interactive HTML must not load external assets");

const mermaid = run(["graph", "web/samples/sample_cnn_float.onnx", "--format", "mermaid"]);
const dot = run(["graph", "web/samples/sample_cnn_float.onnx", "--format", "dot"]);
assert.match(mermaid, /^%% deepbom\.visualization_manifest\.v1/);
assert.match(mermaid, /flowchart TD/);
assert.match(dot, /^digraph deepbom/);
const formatConflict = runFailure(["graph", "web/samples/sample_cnn_float.onnx", "--format", "svg", "--compact"]);
assert.match(formatConflict.stderr, /conflicts with an explicit non-JSON graph --format/);

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
  await page.goto(pathToFileURL(htmlPath).href, { waitUntil: "load" });
  const initial = await page.locator("#scene").evaluate((node) => getComputedStyle(node).transform);
  await page.locator("#plus").click();
  const zoomed = await page.locator("#scene").evaluate((node) => getComputedStyle(node).transform);
  assert.notEqual(zoomed, initial, "zoom button must change the scene transform");
  await page.locator("#search").fill("decoder layer 2");
  assert.equal(await page.locator(".node.match").count(), 1, "search must identify one architecture node");
  const mobile = await page.evaluate(() => ({
    bodyOverflow: document.documentElement.scrollWidth - innerWidth,
    headerOverflow: document.querySelector("header").scrollWidth - document.querySelector("header").clientWidth,
    viewportHeight: document.querySelector("#viewport").clientHeight,
  }));
  assert.equal(mobile.bodyOverflow <= 0, true, JSON.stringify(mobile));
  assert.equal(mobile.headerOverflow <= 0, true, JSON.stringify(mobile));
  assert.equal(mobile.viewportHeight > 500, true, JSON.stringify(mobile));
  await page.screenshot({ path: path.join(output, "gguf-architecture-mobile.png") });
} finally { await browser.close(); }

console.log("CLI graph export checks passed (IR conservation, deterministic SVG/PNG, conditional LLM placement, GGUF non-DAG boundary, HTML interaction/mobile layout, Mermaid, and DOT). ");

function run(args) {
  const result = spawnSync(process.execPath, ["bin/deepbom.mjs", ...args], { cwd: root, encoding: "utf8", maxBuffer: 256 * 1024 * 1024 });
  if (result.status !== 0) throw new Error(`${args.join(" ")} failed\n${result.stdout}\n${result.stderr}`);
  return result.stdout;
}

function runFailure(args) {
  const result = spawnSync(process.execPath, ["bin/deepbom.mjs", ...args], { cwd: root, encoding: "utf8" });
  assert.notEqual(result.status, 0, `${args.join(" ")} should fail`);
  return result;
}

function testNvidiaProfile() {
  const digest = (character) => character.repeat(64);
  return {
    schema: "deepbom.accelerator_profile.v1",
    evidence_class: "OBSERVED_HOST_TOOLING",
    collection: {
      collected_at: "2026-08-30T00:00:00.000Z", platform: "linux", architecture: "x64",
      collector: "deepbom", collector_version: "test",
      tools: [{ role: "nvidia_smi", status: "observed", executable_name: "nvidia-smi", executable_sha256: digest("a"), observation_sha256: digest("b") }],
    },
    devices: [{
      index: 0, name: "NVIDIA Test GPU", compute_capability: "8.9", driver_version: "581.86",
      memory_total_bytes: { decimal: "2400000", number: 2400000 }, uuid_sha256: digest("c"), pci_bus_id_sha256: digest("d"),
      maximum_sm_clock_mhz: null, maximum_memory_clock_mhz: null,
      unexposed_fields: ["maximum_sm_clock_mhz", "maximum_memory_clock_mhz"],
    }],
    software: {
      nvidia_driver: { status: "observed", version: "581.86" }, cuda_driver_api: { status: "not_exposed", version: null },
      cuda_toolkit: { status: "not_installed", version: null }, tensorrt: { status: "not_installed", version: null },
    },
    roofline_contract: {
      status: "not_assessable_missing_exact_hardware_contract", theoretical_compute_ceiling: null, theoretical_memory_bandwidth: null,
      missing_fields: ["sm_count", "memory_bus_width_bits"],
    },
    interpretation_boundary: "Test fixture proving only the same strict host-profile boundary used by production collection.",
  };
}
