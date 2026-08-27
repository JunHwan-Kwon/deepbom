#!/usr/bin/env node

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { initSync, analyze_tflite_for_target } from "../pkg/tflite_wasm_audit.js";

const rootDir = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const defaultOutDir = path.join(rootDir, "reports", "visualizer_parity");
const defaultNetronDir = path.join(rootDir, "reports", "reference_tools", "netron_wheel", "netron");

const args = parseArgs(process.argv.slice(2));
if (!args.models.length) {
  fail(
    [
      "Usage:",
      "  node scripts/visualizer-parity.mjs --target android_mid_a55 path/to/model.tflite [more.tflite]",
      "",
      "Optional:",
      "  --out reports/visualizer_parity",
      "  --netron-dir reports/reference_tools/netron_wheel/netron",
      "  --model-explorer-json path/to/model-explorer-export.json",
    ].join("\n"),
  );
}

const wasmBytes = fs.readFileSync(path.join(rootDir, "pkg", "tflite_wasm_audit_bg.wasm"));
initSync({ module: wasmBytes });

await fsp.mkdir(args.outDir, { recursive: true });

const netron = await loadNetron(args.netronDir);
const modelExplorerExport = args.modelExplorerJson ? await loadJson(args.modelExplorerJson) : null;
const reports = [];

for (let modelIndex = 0; modelIndex < args.models.length; modelIndex += 1) {
  const modelPath = path.resolve(args.models[modelIndex]);
  const filename = path.basename(modelPath);
  const bytes = fs.readFileSync(modelPath);
  const ours = analyze_tflite_for_target(new Uint8Array(bytes), filename, args.target);
  const netronSummary = await analyzeWithNetron(netron, modelPath);
  const modelExplorerSummary =
    modelIndex === 0 && modelExplorerExport ? compareModelExplorerExport(ours, modelExplorerExport) : skippedModelExplorerSummary();
  const report = compareModel(filename, modelPath, ours, netronSummary, modelExplorerSummary);
  reports.push(report);
}

const markdown = buildMarkdown(reports, args);
const json = {
  generated_at: new Date().toISOString(),
  target: args.target,
  netron_dir: path.relative(rootDir, args.netronDir),
  model_explorer_json: args.modelExplorerJson ? path.relative(rootDir, path.resolve(args.modelExplorerJson)) : null,
  reports,
};

const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
const jsonPath = path.join(args.outDir, `visualizer_parity_${stamp}.json`);
const mdPath = path.join(args.outDir, `visualizer_parity_${stamp}.md`);
await fsp.writeFile(jsonPath, `${JSON.stringify(json, null, 2)}\n`);
await fsp.writeFile(mdPath, markdown);

const failing = reports.filter((report) => report.verdict !== "pass");
console.log(`Wrote ${path.relative(rootDir, mdPath)}`);
console.log(`Wrote ${path.relative(rootDir, jsonPath)}`);
if (failing.length) {
  console.log(`Visualizer parity completed with ${failing.length} non-pass model(s).`);
  process.exitCode = 1;
} else {
  console.log("Visualizer parity passed.");
}

function parseArgs(argv) {
  const parsed = {
    target: "android_mid_a55",
    outDir: defaultOutDir,
    netronDir: defaultNetronDir,
    modelExplorerJson: "",
    models: [],
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--target") {
      parsed.target = requireValue(argv, ++i, arg);
    } else if (arg === "--out") {
      parsed.outDir = path.resolve(requireValue(argv, ++i, arg));
    } else if (arg === "--netron-dir") {
      parsed.netronDir = path.resolve(requireValue(argv, ++i, arg));
    } else if (arg === "--model-explorer-json") {
      parsed.modelExplorerJson = path.resolve(requireValue(argv, ++i, arg));
    } else if (arg.startsWith("--")) {
      fail(`Unknown option: ${arg}`);
    } else {
      parsed.models.push(path.resolve(arg));
    }
  }
  return parsed;
}

function requireValue(argv, index, option) {
  if (index >= argv.length || argv[index].startsWith("--")) {
    fail(`Missing value for ${option}`);
  }
  return argv[index];
}

function fail(message) {
  console.error(message);
  process.exit(2);
}

async function loadNetron(netronDir) {
  const required = ["node.js", "flatbuffers.js", "tflite.js", "tflite-schema.js"];
  const missing = required.filter((file) => !fs.existsSync(path.join(netronDir, file)));
  if (missing.length) {
    fail(
      [
        `Netron parser files not found under ${netronDir}.`,
        `Missing: ${missing.join(", ")}`,
        "",
        "Fetch the official wheel first:",
        "  python -m pip download netron==9.1.1 --no-deps -d reports/reference_tools",
        "  mkdir reports\\reference_tools\\netron_wheel",
        "  tar -xf reports\\reference_tools\\netron-9.1.1-py3-none-any.whl -C reports\\reference_tools\\netron_wheel",
      ].join("\n"),
    );
  }
  const moduleUrl = (file) => pathToFileURL(path.join(netronDir, file)).href;
  const [{ FileStream }, flatbuffers, { ModelFactory }, tfliteSchema] = await Promise.all([
    import(moduleUrl("node.js")),
    import(moduleUrl("flatbuffers.js")),
    import(moduleUrl("tflite.js")),
    import(moduleUrl("tflite-schema.js")),
  ]);
  return { netronDir, FileStream, flatbuffers, ModelFactory, tfliteSchema };
}

async function analyzeWithNetron(netron, modelPath) {
  const stat = fs.statSync(modelPath);
  const stream = new netron.FileStream(modelPath, 0, stat.size, stat.mtimeMs);
  const context = {
    identifier: path.basename(modelPath),
    stream,
    type: "",
    value: null,
    async peek(kind) {
      if (kind === "flatbuffers.binary") {
        return netron.flatbuffers.BinaryReader.open(stream);
      }
      if (kind === "json") {
        return null;
      }
      return null;
    },
    set(type, value) {
      this.type = type;
      this.value = value;
      return this;
    },
    async require(id) {
      if (id === "./tflite-schema") {
        return netron.tfliteSchema;
      }
      const specifier = id.endsWith(".js") ? id : `${id}.js`;
      return import(pathToFileURL(path.join(netron.netronDir, specifier)).href);
    },
    async metadata() {
      return dummyNetronMetadata();
    },
    async read() {
      return null;
    },
  };

  const factory = new netron.ModelFactory();
  const matched = await factory.match(context);
  if (!matched) {
    throw new Error(`Netron did not recognize ${modelPath} as TensorFlow Lite.`);
  }
  const model = await factory.open(context);
  const graph = model.modules[0];
  const ops = graph.nodes.map((node, fallbackIndex) => summarizeNetronNode(node, fallbackIndex));
  const tensors = summarizeNetronTensors(graph, ops);
  return {
    format: model.format,
    graph_name: graph.name || "",
    op_count: ops.length,
    tensor_count_referenced: tensors.length,
    inputs: graph.inputs.map((arg) => summarizeNetronArgument(arg)),
    outputs: graph.outputs.map((arg) => summarizeNetronArgument(arg)),
    histogram: histogram(ops.map((op) => op.name_normalized)),
    ops,
    tensors,
    edges: graphEdges(ops),
  };
}

function dummyNetronMetadata() {
  return {
    type(name) {
      return { name };
    },
    attribute() {
      return {};
    },
  };
}

function summarizeNetronNode(node, fallbackIndex) {
  const index = Number.isFinite(Number(node.identifier)) ? Number(node.identifier) : fallbackIndex;
  const inputs = node.inputs.flatMap((arg) => valuesOf(arg)).map((value) => Number(value.identifier));
  const outputs = node.outputs.flatMap((arg) => valuesOf(arg)).map((value) => Number(value.identifier));
  return {
    index,
    name: node.type?.name || "",
    name_normalized: normalizeOpName(node.type?.name || ""),
    inputs,
    outputs,
    input_tensors: node.inputs.map((arg) => summarizeNetronArgument(arg)),
    output_tensors: node.outputs.map((arg) => summarizeNetronArgument(arg)),
  };
}

function summarizeNetronArgument(arg) {
  return {
    name: arg.name || "",
    tensors: valuesOf(arg).map((value) => summarizeNetronValue(value)),
  };
}

function summarizeNetronTensors(graph, ops) {
  const byId = new Map();
  const addArg = (arg) => {
    for (const value of valuesOf(arg)) {
      byId.set(Number(value.identifier), summarizeNetronValue(value));
    }
  };
  for (const arg of graph.inputs) addArg(arg);
  for (const arg of graph.outputs) addArg(arg);
  for (const op of ops) {
    for (const arg of op.input_tensors) {
      for (const tensor of arg.tensors) byId.set(tensor.index, tensor);
    }
    for (const arg of op.output_tensors) {
      for (const tensor of arg.tensors) byId.set(tensor.index, tensor);
    }
  }
  return [...byId.values()].sort((a, b) => a.index - b.index);
}

function summarizeNetronValue(value) {
  const quant = value.quantization || {};
  return {
    index: Number(value.identifier),
    name: String(value.name || "").replace(/\n\d+$/, ""),
    dtype: normalizeDtype(value.type?.dataType || ""),
    shape: Array.isArray(value.type?.shape?.dimensions) ? [...value.type.shape.dimensions] : [],
    quant_scales: Array.isArray(quant.scale) ? quant.scale.length : quant.scale?.length || 0,
    quant_zero_points: Array.isArray(quant.offset) ? quant.offset.length : quant.offset?.length || 0,
    quantized_dimension: Number.isFinite(Number(quant.dimension)) ? Number(quant.dimension) : 0,
  };
}

function valuesOf(arg) {
  return Array.isArray(arg?.value) ? arg.value.filter(Boolean) : [];
}

function compareModel(filename, modelPath, ours, netron, modelExplorerSummary) {
  const ourOps = ours.ops.map((op) => ({
    index: op.index,
    name: op.name,
    name_normalized: normalizeOpName(op.name),
    inputs: op.inputs.filter((id) => id >= 0),
    outputs: op.outputs.filter((id) => id >= 0),
  }));
  const checks = [];
  checks.push(checkEqual("op count", ours.operator_count, netron.op_count));
  checks.push(checkOpSequence(ourOps, netron.ops));
  checks.push(checkOpTensorIds(ourOps, netron.ops));
  checks.push(checkEdges(graphEdges(ourOps), netron.edges));
  checks.push(checkTensorMetadata(ours.tensors, netron.tensors));
  checks.push(checkQuantizationFacts(ours, netron));
  checks.push(modelExplorerSummary.check);

  const hardFailures = checks.filter((check) => check.status === "fail");
  const warnings = checks.filter((check) => check.status === "warn");
  const verdict = hardFailures.length ? "fail" : warnings.length ? "warn" : "pass";

  return {
    filename,
    model_path: modelPath,
    verdict,
    ours: {
      op_count: ours.operator_count,
      tensor_count: ours.tensor_count,
      input_tensors: ours.inputs.map(summarizeOurTensor),
      output_tensors: ours.outputs.map(summarizeOurTensor),
      histogram: histogram(ourOps.map((op) => op.name_normalized)),
      quantization_status: ours.quantization_status || null,
    },
    netron: {
      format: netron.format,
      graph_name: netron.graph_name,
      op_count: netron.op_count,
      tensor_count_referenced: netron.tensor_count_referenced,
      input_tensors: netron.inputs,
      output_tensors: netron.outputs,
      histogram: netron.histogram,
      quantized_referenced_tensors: netron.tensors.filter((tensor) => Number(tensor.quant_scales || 0) > 0).length,
    },
    model_explorer: modelExplorerSummary.summary,
    checks,
  };
}

function summarizeOurTensor(tensor) {
  return {
    index: tensor.index,
    name: tensor.name,
    dtype: tensor.dtype,
    shape: tensor.shape,
    shape_signature: tensor.shape_signature || [],
    quant_scales: tensor.quant_scales,
    quant_zero_points: tensor.quant_zero_points,
    quantized_dimension: tensor.quantized_dimension,
  };
}

function checkEqual(label, left, right) {
  return {
    label,
    status: left === right ? "pass" : "fail",
    detail: `${left} vs ${right}`,
  };
}

function checkOpSequence(ours, netronOps) {
  const mismatches = [];
  const count = Math.min(ours.length, netronOps.length);
  for (let i = 0; i < count; i += 1) {
    if (ours[i].name_normalized !== netronOps[i].name_normalized) {
      mismatches.push(`#${i} ours=${ours[i].name} netron=${netronOps[i].name}`);
    }
  }
  if (ours.length !== netronOps.length) {
    mismatches.push(`length ours=${ours.length} netron=${netronOps.length}`);
  }
  return {
    label: "op sequence",
    status: mismatches.length ? "fail" : "pass",
    detail: mismatches.length ? mismatches.slice(0, 12).join(" | ") : "normalized op names match by index",
    mismatch_count: mismatches.length,
  };
}

function checkOpTensorIds(ours, netronOps) {
  const mismatches = [];
  const count = Math.min(ours.length, netronOps.length);
  for (let i = 0; i < count; i += 1) {
    const inputA = ours[i].inputs.join(",");
    const inputB = netronOps[i].inputs.join(",");
    const outputA = ours[i].outputs.join(",");
    const outputB = netronOps[i].outputs.join(",");
    if (inputA !== inputB || outputA !== outputB) {
      mismatches.push(`#${i} inputs ${inputA} vs ${inputB}; outputs ${outputA} vs ${outputB}`);
    }
  }
  return {
    label: "op tensor ids",
    status: mismatches.length ? "fail" : "pass",
    detail: mismatches.length ? mismatches.slice(0, 12).join(" | ") : "input/output tensor ids match by op index",
    mismatch_count: mismatches.length,
  };
}

function checkEdges(ours, netronEdges) {
  const oursOnly = [...ours].filter((edge) => !netronEdges.has(edge));
  const netronOnly = [...netronEdges].filter((edge) => !ours.has(edge));
  return {
    label: "producer/consumer edges",
    status: oursOnly.length || netronOnly.length ? "fail" : "pass",
    detail:
      oursOnly.length || netronOnly.length
        ? `ours-only ${oursOnly.slice(0, 8).join(" ")} / netron-only ${netronOnly.slice(0, 8).join(" ")}`
        : `${ours.size} graph edges match`,
    ours_edge_count: ours.size,
    netron_edge_count: netronEdges.size,
  };
}

function checkTensorMetadata(ourTensors, netronTensors) {
  const mismatches = [];
  const notes = [];
  for (const netronTensor of netronTensors) {
    const ours = ourTensors[netronTensor.index];
    if (!ours) {
      mismatches.push(`T${netronTensor.index} missing in ours`);
      continue;
    }
    const visibleShape = Array.isArray(ours.shape_signature) && ours.shape_signature.length ? ours.shape_signature : ours.shape;
    if (netronTensor.dtype !== ours.dtype) {
      mismatches.push(`T${netronTensor.index} dtype ${ours.dtype} vs ${netronTensor.dtype}`);
    }
    if (shapeKey(visibleShape) !== shapeKey(netronTensor.shape)) {
      mismatches.push(`T${netronTensor.index} visible shape ${shapeKey(visibleShape)} vs ${shapeKey(netronTensor.shape)}`);
    }
    if (shapeKey(ours.shape) !== shapeKey(netronTensor.shape) && shapeKey(visibleShape) === shapeKey(netronTensor.shape)) {
      notes.push(`T${netronTensor.index} Netron uses shape_signature ${shapeKey(netronTensor.shape)}; static shape is ${shapeKey(ours.shape)}`);
    }
    if (Number(ours.quant_scales || 0) !== Number(netronTensor.quant_scales || 0)) {
      mismatches.push(`T${netronTensor.index} quant scales ${ours.quant_scales} vs ${netronTensor.quant_scales}`);
    }
    if (Number(ours.quant_zero_points || 0) !== Number(netronTensor.quant_zero_points || 0)) {
      mismatches.push(`T${netronTensor.index} zero-points ${ours.quant_zero_points} vs ${netronTensor.quant_zero_points}`);
    }
    if (Number(ours.quantized_dimension || 0) !== Number(netronTensor.quantized_dimension || 0)) {
      mismatches.push(`T${netronTensor.index} qdim ${ours.quantized_dimension} vs ${netronTensor.quantized_dimension}`);
    }
  }
  return {
    label: "referenced tensor metadata",
    status: mismatches.length ? "fail" : "pass",
    detail: mismatches.length ? mismatches.slice(0, 12).join(" | ") : `${netronTensors.length} referenced tensors match`,
    mismatch_count: mismatches.length,
    notes: notes.slice(0, 20),
  };
}

function checkQuantizationFacts(ours, netron) {
  const ourQuantTensorIds = new Set(
    (ours.tensors || [])
      .filter((tensor) => Number(tensor.quant_scales || 0) > 0 || tensor.dtype === "INT8" || tensor.dtype === "UINT8")
      .map((tensor) => Number(tensor.index)),
  );
  const netronQuantTensorIds = new Set(
    (netron.tensors || [])
      .filter((tensor) => Number(tensor.quant_scales || 0) > 0 || tensor.dtype === "INT8" || tensor.dtype === "UINT8")
      .map((tensor) => Number(tensor.index)),
  );
  const oursOnly = [...ourQuantTensorIds].filter((id) => !netronQuantTensorIds.has(id));
  const netronOnly = [...netronQuantTensorIds].filter((id) => !ourQuantTensorIds.has(id));
  return {
    label: "raw quantization facts",
    status: oursOnly.length || netronOnly.length ? "fail" : "pass",
    detail:
      oursOnly.length || netronOnly.length
        ? `ours-only T${oursOnly.slice(0, 12).join(",T")} / netron-only T${netronOnly.slice(0, 12).join(",T")}`
        : `${ourQuantTensorIds.size} tensors with quant dtype/metadata match Netron referenced values`,
    ours_quant_tensor_count: ourQuantTensorIds.size,
    netron_quant_tensor_count: netronQuantTensorIds.size,
    note:
      "This validates raw dtype/quant metadata parity. Per-op quantization_state is a derived analyzer classification and is not expected to be emitted by Netron.",
  };
}

function graphEdges(ops) {
  const producers = new Map();
  for (const op of ops) {
    for (const output of op.outputs) {
      producers.set(output, op.index);
    }
  }
  const edges = new Set();
  for (const op of ops) {
    for (const input of op.inputs) {
      if (producers.has(input)) {
        edges.add(`${producers.get(input)}->${op.index}:T${input}`);
      }
    }
  }
  return edges;
}

function compareModelExplorerExport(ours, exportedJson) {
  const graphs = extractModelExplorerGraphs(exportedJson);
  if (!graphs.length) {
    return {
      check: {
        label: "Model Explorer export",
        status: "warn",
        detail: "JSON was provided, but no graphCollections/graphs were recognized.",
      },
      summary: { status: "unrecognized" },
    };
  }
  const graph = graphs.reduce((best, item) => ((item.nodes?.length || 0) > (best.nodes?.length || 0) ? item : best), graphs[0]);
  const labels = (graph.nodes || []).map((node) => normalizeOpName(node.label || "")).filter(Boolean);
  const ourHistogram = histogram(ours.ops.map((op) => normalizeOpName(op.name)));
  const exportedHistogram = histogram(labels);
  const missing = [];
  for (const [name, count] of Object.entries(ourHistogram)) {
    const exportedCount = exportedHistogram[name] || 0;
    if (exportedCount < count) {
      missing.push(`${name} ours=${count} modelExplorer=${exportedCount}`);
    }
  }
  return {
    check: {
      label: "Model Explorer export histogram",
      status: missing.length ? "warn" : "pass",
      detail: missing.length
        ? `Model Explorer processed graph may include grouped/layer labels; missing normalized labels: ${missing.slice(0, 12).join(" | ")}`
        : "normalized label histogram covers our op histogram",
    },
    summary: {
      status: "provided",
      graph_count: graphs.length,
      selected_graph_id: graph.id || "",
      selected_graph_nodes: graph.nodes?.length || 0,
    },
  };
}

function skippedModelExplorerSummary() {
  return {
    check: {
      label: "Model Explorer export",
      status: "skip",
      detail:
        "No processed JSON export was provided. Open the model in Model Explorer and use the graph selector download button, then pass --model-explorer-json.",
    },
    summary: {
      status: "skipped",
      reason:
        "The official TFLite flatbuffer adapter depends on ai-edge-model-explorer-adapter, which is not distributed for win32 in the package metadata. JSON export comparison is the reliable Windows path.",
    },
  };
}

function extractModelExplorerGraphs(value) {
  if (!value) return [];
  if (Array.isArray(value.graphCollections)) {
    return value.graphCollections.flatMap((collection) => collection.graphs || []);
  }
  if (Array.isArray(value.graphs)) return value.graphs;
  if (Array.isArray(value)) {
    return value.flatMap((item) => extractModelExplorerGraphs(item));
  }
  if (Array.isArray(value.subgraphs)) return value.subgraphs;
  return [];
}

async function loadJson(file) {
  return JSON.parse(await fsp.readFile(file, "utf8"));
}

function normalizeOpName(name) {
  return String(name || "")
    .replace(/[^A-Za-z0-9]+/g, "")
    .toUpperCase();
}

function normalizeDtype(dtype) {
  const text = String(dtype || "").toUpperCase();
  const aliases = {
    BOOLEAN: "BOOL",
    "COMPLEX<FLOAT32>": "COMPLEX64",
    "COMPLEX<FLOAT64>": "COMPLEX128",
  };
  return aliases[text] || text;
}

function histogram(items) {
  const out = {};
  for (const item of items) {
    out[item] = (out[item] || 0) + 1;
  }
  return Object.fromEntries(Object.entries(out).sort(([a], [b]) => a.localeCompare(b)));
}

function shapeKey(shape) {
  return `[${(Array.isArray(shape) ? shape : []).join(",")}]`;
}

function buildMarkdown(reports, args) {
  const lines = [];
  lines.push("# Visualizer Parity Report");
  lines.push("");
  lines.push(`- Generated: ${new Date().toISOString()}`);
  lines.push(`- Target: \`${args.target}\``);
  lines.push(`- Netron parser: \`${path.relative(rootDir, args.netronDir)}\``);
  lines.push(`- Model Explorer JSON: \`${args.modelExplorerJson ? path.relative(rootDir, path.resolve(args.modelExplorerJson)) : "not provided"}\``);
  lines.push("");
  lines.push("## Summary");
  lines.push("");
  lines.push("| Model | Verdict | Ours ops | Netron ops | Tensor check | Quant facts | Edge check | Model Explorer |");
  lines.push("|---|---:|---:|---:|---|---|---|---|");
  for (const report of reports) {
    const tensorCheck = report.checks.find((check) => check.label === "referenced tensor metadata");
    const quantCheck = report.checks.find((check) => check.label === "raw quantization facts");
    const edgeCheck = report.checks.find((check) => check.label === "producer/consumer edges");
    const modelExplorerCheck = report.checks.find((check) => check.label.startsWith("Model Explorer"));
    lines.push(
      `| ${report.filename} | ${report.verdict} | ${report.ours.op_count} | ${report.netron.op_count} | ${tensorCheck?.status || "-"} | ${quantCheck?.status || "-"} | ${edgeCheck?.status || "-"} | ${modelExplorerCheck?.status || "-"} |`,
    );
  }
  for (const report of reports) {
    lines.push("");
    lines.push(`## ${report.filename}`);
    lines.push("");
    lines.push(`- Verdict: \`${report.verdict}\``);
    lines.push(`- Netron format: \`${report.netron.format}\``);
    lines.push(`- Ours tensors: \`${report.ours.tensor_count}\`; Netron referenced tensors: \`${report.netron.tensor_count_referenced}\``);
    lines.push("");
    lines.push("| Check | Status | Detail |");
    lines.push("|---|---|---|");
    for (const check of report.checks) {
      lines.push(`| ${check.label} | ${check.status} | ${escapeTable(check.detail || "")} |`);
      if (Array.isArray(check.notes) && check.notes.length) {
        lines.push(`| ${check.label} notes | info | ${escapeTable(check.notes.join(" / "))} |`);
      }
      if (check.note) {
        lines.push(`| ${check.label} note | info | ${escapeTable(check.note)} |`);
      }
    }
  }
  lines.push("");
  lines.push("## Interpretation");
  lines.push("");
  lines.push("- `pass` means the raw TFLite graph facts match Netron at op/tensor/edge level.");
  lines.push("- Raw graph parity scope: op count/order, tensor ids, producer/consumer edges, dtype, visible shape/signature, quant scale count, zero-point count, and quantized dimension.");
  lines.push("- Quantization parity scope is raw tensor dtype/metadata parity. Per-layer `quantization_state` is a derived analyzer classification built from those raw facts.");
  lines.push("- `warn` on Model Explorer usually means no processed JSON was supplied, or Model Explorer grouped/layered labels need manual review.");
  lines.push("- Shape comparisons use `shape_signature` when present, because Netron displays signature shape while this analyzer also keeps the static execution shape for benchmarking.");
  lines.push("- Roofline, XNNPACK chain prediction, packing estimates, and quantization risk are DEEPBOM-derived analyses and are not expected to appear in Netron or Model Explorer.");
  lines.push("");
  return `${lines.join("\n")}\n`;
}

function escapeTable(value) {
  return String(value).replace(/\|/g, "\\|").replace(/\n/g, " ");
}
