import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

const PACKAGE_SCHEMA = "deepbom.redesign_structure_package.v1";
export const REDESIGN_SCENARIO_SET_SCHEMA = "deepbom.redesign_scenario_set.v1.1";

export function scenarioFingerprint(request) {
  const normalized = canonicalScenario(request);
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(JSON.stringify(normalized))) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, "0");
}

export function canonicalScenario(request) {
  return {
    schema: "deepbom.redesign_request.v1",
    source_sha256: String(request?.source_sha256 || ""),
    input_height: finiteInteger(request?.input_height),
    input_width: finiteInteger(request?.input_width),
    width_multiplier: finiteNumber(request?.width_multiplier, 1),
    activation_dtype: String(request?.activation_dtype || "source"),
    block_edits: [...(request?.block_edits || [])]
      .map((edit) => Object.fromEntries(Object.entries(edit)
        .filter(([, value]) => value != null)
        .sort(([left], [right]) => left.localeCompare(right))))
      .sort((left, right) => String(left.block_id || "").localeCompare(String(right.block_id || ""))),
  };
}

export function buildRedesignScenarioSet({ analysis, savedScenarios, pareto = null }) {
  const firstSaved = savedScenarios?.[0];
  const sourceSha256 = requiredSha(
    analysis?.model_sha256 || firstSaved?.request?.source_sha256 || firstSaved?.projection?.source?.sha256_before,
    "Redesign source SHA-256",
  );
  for (const candidate of [analysis?.model_sha256, firstSaved?.request?.source_sha256, firstSaved?.projection?.source?.sha256_before].filter(Boolean)) {
    if (requiredSha(candidate, "Redesign source SHA-256") !== sourceSha256) {
      throw new Error("Redesign analysis, request, and projection source identities disagree.");
    }
  }
  const targetProfileSha256 = requiredSha(analysis?.target_profile?.profile_sha256, "Redesign target-profile SHA-256");
  const scenarios = [...(savedScenarios || [])].map((record) => {
    const request = canonicalScenario(record?.request);
    const projection = jsonClone(record?.projection, "Redesign projection");
    const scenarioId = scenarioFingerprint(request);
    if (request.source_sha256 !== sourceSha256 || record?.scenarioId !== scenarioId) {
      throw new Error("Saved redesign scenario identity does not match the bound source artifact.");
    }
    verifyProjectionSource(projection, sourceSha256);
    const plan = projection?.implementation_plan;
    if (!plan || plan.schema !== "deepbom.redesign_implementation_plan.v1") {
      throw new Error("Saved redesign scenario is missing its WASM implementation plan.");
    }
    const implementationPlanSha256 = sha256TextHex(canonicalJson(plan));
    const projectionSha256 = sha256TextHex(canonicalJson(projection));
    return {
      scenario_id: scenarioId,
      label: String(record?.label || scenarioId),
      request,
      projection,
      projection_sha256: projectionSha256,
      implementation_plan_sha256: implementationPlanSha256,
      source_mapping: sourceMappingSummary(plan),
      regeneration: regenerationSummary(plan),
    };
  }).sort((left, right) => left.scenario_id.localeCompare(right.scenario_id));
  if (!scenarios.length || new Set(scenarios.map((row) => row.scenario_id)).size !== scenarios.length) {
    throw new Error("Redesign scenario export requires one or more unique saved scenarios.");
  }
  const paretoSearch = pareto == null ? null : jsonClone(pareto, "Redesign Pareto search");
  if (paretoSearch) verifyParetoSource(paretoSearch, sourceSha256);
  const value = {
    schema: REDESIGN_SCENARIO_SET_SCHEMA,
    hash_contract: { algorithm: "SHA-256", canonicalization: "RFC8785-JCS", excluded_pointers: ["/ledger_sha256"] },
    source: {
      artifact_sha256: sourceSha256,
      format: String(analysis?.format || "tflite"),
      target_profile_id: String(analysis?.target_profile?.id || analysis?.target_profile?.target_id || ""),
      target_profile_sha256: targetProfileSha256,
    },
    scenario_count: scenarios.length,
    scenarios,
    pareto_search: paretoSearch == null ? null : {
      search_sha256: sha256TextHex(canonicalJson(paretoSearch)),
      evaluated_candidate_count: paretoSearch.evaluated_candidate_count,
      accepted_candidate_count: paretoSearch.accepted_candidate_count,
      frontier_candidate_count: paretoSearch.frontier_candidate_count,
      search: paretoSearch,
    },
    handoff: {
      contains_weights: false,
      generated_frameworks: [...new Set(scenarios.flatMap((row) => row.regeneration.framework_targets))].sort(),
      implementation_plan_embedded_per_scenario: true,
      source_mapping_embedded_per_scenario: true,
    },
    interpretation_boundary: "Every projection is a deterministic PROJECTED_UNTRAINED structural scenario bound to one source artifact and target profile. Pareto retention is a structure proxy, not accuracy, calibration, clinical, runtime, or release evidence.",
  };
  value.ledger_sha256 = sha256TextHex(canonicalJson(value));
  verifyRedesignScenarioSet(value);
  return value;
}

export function verifyRedesignScenarioSet(value) {
  if (!value || value.schema !== REDESIGN_SCENARIO_SET_SCHEMA) throw new Error("Redesign scenario-set schema is invalid.");
  const sourceSha256 = requiredSha(value.source?.artifact_sha256, "Scenario-set source SHA-256");
  requiredSha(value.source?.target_profile_sha256, "Scenario-set target-profile SHA-256");
  const scenarios = value.scenarios;
  if (!Array.isArray(scenarios) || !scenarios.length || value.scenario_count !== scenarios.length
    || new Set(scenarios.map((row) => row.scenario_id)).size !== scenarios.length) {
    throw new Error("Redesign scenario-set count or identity conservation failed.");
  }
  for (const row of scenarios) {
    const request = canonicalScenario(row.request);
    if (row.scenario_id !== scenarioFingerprint(request) || request.source_sha256 !== sourceSha256) {
      throw new Error("Redesign scenario request identity does not reconstruct.");
    }
    verifyProjectionSource(row.projection, sourceSha256);
    const plan = row.projection?.implementation_plan;
    if (requiredSha(row.projection_sha256, "Scenario projection SHA-256") !== sha256TextHex(canonicalJson(row.projection))
      || requiredSha(row.implementation_plan_sha256, "Implementation-plan SHA-256") !== sha256TextHex(canonicalJson(plan))
      || canonicalJson(row.source_mapping) !== canonicalJson(sourceMappingSummary(plan))
      || canonicalJson(row.regeneration) !== canonicalJson(regenerationSummary(plan))) {
      throw new Error("Redesign scenario projection, mapping, or regeneration summary does not reconstruct.");
    }
  }
  if (value.pareto_search != null) {
    const search = value.pareto_search.search;
    verifyParetoSource(search, sourceSha256);
    if (requiredSha(value.pareto_search.search_sha256, "Pareto search SHA-256") !== sha256TextHex(canonicalJson(search))
      || value.pareto_search.evaluated_candidate_count !== search.evaluated_candidate_count
      || value.pareto_search.accepted_candidate_count !== search.accepted_candidate_count
      || value.pareto_search.frontier_candidate_count !== search.frontier_candidate_count) {
      throw new Error("Redesign Pareto search summary does not reconstruct.");
    }
  }
  const { ledger_sha256: recorded, ...hashBasis } = value;
  if (requiredSha(recorded, "Scenario-set ledger SHA-256") !== sha256TextHex(canonicalJson(hashBasis))) {
    throw new Error("Redesign scenario-set ledger SHA-256 does not reconstruct.");
  }
  return true;
}

function sourceMappingSummary(plan) {
  const evidenceCounts = {};
  for (const node of plan?.nodes || []) {
    const key = String(node.source_layer_evidence_class || "NOT_MAPPED");
    evidenceCounts[key] = (evidenceCounts[key] || 0) + 1;
  }
  return {
    node_count: plan?.nodes?.length || 0,
    mapped_source_layer_count: plan?.mapped_source_layer_count || 0,
    evidence_counts: Object.fromEntries(Object.entries(evidenceCounts).sort(([left], [right]) => left.localeCompare(right))),
  };
}

function regenerationSummary(plan) {
  return {
    exportable: plan?.exportable === true,
    framework_targets: [...(plan?.framework_targets || [])].sort(),
    exact_codegen_op_count: plan?.exact_codegen_op_count || 0,
    scaffold_codegen_op_count: plan?.scaffold_codegen_op_count || 0,
    unsupported_codegen_op_count: plan?.unsupported_codegen_op_count || 0,
    non_materialized_repeat_edit_count: plan?.non_materialized_repeat_edit_count || 0,
  };
}

function verifyProjectionSource(projection, sourceSha256) {
  if (!projection || projection.projection_status !== "PROJECTED_UNTRAINED"
    || projection.source?.sha256_before !== sourceSha256
    || projection.source?.sha256_after !== sourceSha256
    || projection.source?.loaded_source_bytes_unchanged !== true) {
    throw new Error("Redesign projection is not immutably bound to the scenario-set source artifact.");
  }
}

function verifyParetoSource(search, sourceSha256) {
  if (!search || search.schema !== "deepbom.redesign_pareto.v1"
    || search.accepted_candidate_count !== search.candidates?.length
    || search.frontier_candidate_count !== search.candidates.filter((row) => row.pareto_optimal).length
    || search.candidates.some((row) => canonicalScenario(row.request).source_sha256 !== sourceSha256)) {
    throw new Error("Redesign Pareto search is not conserved or source-bound.");
  }
}

function requiredSha(value, label) {
  const text = String(value || "").toLowerCase();
  if (!/^[a-f0-9]{64}$/.test(text)) throw new Error(`${label} is required.`);
  return text;
}

function jsonClone(value, label) {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    throw new Error(`${label} is not JSON-serializable.`);
  }
}

export function buildRedesignImplementationFiles({ analysis, projection, request }) {
  const plan = projection?.implementation_plan;
  if (!plan || plan.schema !== "deepbom.redesign_implementation_plan.v1") {
    throw new Error("The WASM projection does not contain a reviewed implementation plan.");
  }
  if (!plan.exportable) {
    throw new Error("Structure code export is blocked because the projected repeat topology was not materialized in the WASM tensor graph.");
  }
  const scenario = canonicalScenario(request);
  const manifest = {
    schema: PACKAGE_SCHEMA,
    status: plan.status,
    source: projection.source,
    scenario_id: scenarioFingerprint(scenario),
    implementation_plan_schema: plan.schema,
    frameworks: plan.framework_targets,
    contains_weights: false,
    generated_files: [
      "manifest.json",
      "README.md",
      "implementation_plan.md",
      "implementation_plan.json",
      "scenario.json",
      "projection.json",
      "pytorch/model.py",
      "pytorch/smoke_test.py",
      "keras/model.py",
      "keras/convert_litert.py",
    ],
    verification_boundary: plan.interpretation_boundary,
  };
  return [
    textFile("README.md", packageReadme(analysis, projection, scenario)),
    textFile("implementation_plan.md", implementationMarkdown(projection, scenario)),
    textFile("implementation_plan.json", json(plan)),
    textFile("scenario.json", json(scenario)),
    textFile("projection.json", json(projection)),
    textFile("manifest.json", json(manifest)),
    textFile("pytorch/model.py", renderPytorch(plan)),
    textFile("pytorch/smoke_test.py", renderPytorchSmokeTest(plan)),
    textFile("keras/model.py", renderKeras(plan)),
    textFile("keras/convert_litert.py", renderLiteRtConverter()),
  ];
}

function renderPytorch(plan) {
  const lines = [
    '"""Weight-free architecture scaffold derived from a DEEPBOM WASM projection.',
    "",
    "This is not recovered source code. Parameters are randomly initialized and must",
    "be trained and validated before conversion or deployment.",
    '"""',
    "from __future__ import annotations",
    "",
    "import torch",
    "from torch import nn",
    "from torch.nn import functional as F",
    "",
    "",
    "class SamePadConv2d(nn.Module):",
    "    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):",
    "        super().__init__()",
    "        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)",
    "        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)",
    "        self.conv = nn.Conv2d(in_channels, out_channels, self.kernel_size, self.stride, padding=0, groups=groups, bias=True)",
    "",
    "    def forward(self, x):",
    "        in_h, in_w = x.shape[-2:]",
    "        out_h = (in_h + self.stride[0] - 1) // self.stride[0]",
    "        out_w = (in_w + self.stride[1] - 1) // self.stride[1]",
    "        pad_h = max((out_h - 1) * self.stride[0] + self.kernel_size[0] - in_h, 0)",
    "        pad_w = max((out_w - 1) * self.stride[1] + self.kernel_size[1] - in_w, 0)",
    "        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))",
    "        return self.conv(x)",
    "",
    "",
    "class ReconstructedModel(nn.Module):",
    "    def __init__(self):",
    "        super().__init__()",
  ];
  const declarations = plan.nodes.flatMap((node) => pytorchDeclaration(node));
  lines.push(...(declarations.length ? declarations : ["        pass"]));
  lines.push("", `    def forward(self, ${pytorchArguments(plan)}):`, "        values = {}", ...pytorchInputs(plan));
  for (const node of plan.nodes) lines.push(...pytorchForwardNode(node));
  lines.push(...pytorchOutputs(plan), "");
  return `${lines.join("\n")}\n`;
}

function pytorchDeclaration(node) {
  const prefix = `        # ${sourceComment(node)}`;
  const symbol = `self.${node.generated_symbol}`;
  if (node.module_kind === "conv2d" || node.module_kind === "depthwise_conv2d") {
    if (![node.input_channels, node.output_channels, node.kernel_height, node.stride_height].every(Number.isFinite)) return [];
    const groups = node.module_kind === "depthwise_conv2d" ? node.input_channels : 1;
    return [
      prefix,
      `        ${symbol} = SamePadConv2d(${node.input_channels}, ${node.output_channels}, (${node.kernel_height}, ${node.kernel_width || node.kernel_height}), stride=(${node.stride_height}, ${node.stride_width || node.stride_height}), groups=${groups})`,
    ];
  }
  if (node.module_kind === "linear") {
    const inputFeatures = featureCount(node.input_shapes?.[0]);
    const outputFeatures = featureCount(node.output_shapes?.[0]);
    if (!inputFeatures || !outputFeatures) return [];
    return [prefix, `        ${symbol} = nn.Linear(${inputFeatures}, ${outputFeatures})`];
  }
  return [];
}

function pytorchForwardNode(node) {
  const output = node.outputs?.[0];
  const inputs = node.activation_inputs || [];
  const target = Number.isInteger(output) ? `values[${output}]` : "_unused";
  const args = inputs.map((index) => `values[${index}]`);
  const comment = `        # ${sourceComment(node)} | ${node.codegen_status}`;
  if (!Number.isInteger(output)) return [comment, `        raise NotImplementedError(${pythonString(`Operator #${node.op_index} has no addressable output.`)})`];
  let expression = "";
  switch (node.module_kind) {
    case "conv2d":
    case "depthwise_conv2d": expression = `self.${node.generated_symbol}(${args[0]})`; break;
    case "linear": expression = `self.${node.generated_symbol}(torch.flatten(${args[0]}, 1))`; break;
    case "add": expression = `${args[0]} + ${args[1]}`; break;
    case "subtract": expression = `${args[0]} - ${args[1]}`; break;
    case "multiply": expression = `${args[0]} * ${args[1]}`; break;
    case "maximum": expression = `torch.maximum(${args[0]}, ${args[1]})`; break;
    case "minimum": expression = `torch.minimum(${args[0]}, ${args[1]})`; break;
    case "concatenate": expression = `torch.cat([${args.join(", ")}], dim=${pytorchAxis(node.concatenation_axis_nhwc, node.output_shapes?.[0]?.length)})`; break;
    case "average_pool2d": expression = `F.adaptive_avg_pool2d(${args[0]}, ${pytorchSpatial(node.output_shapes?.[0])})`; break;
    case "max_pool2d": expression = `F.adaptive_max_pool2d(${args[0]}, ${pytorchSpatial(node.output_shapes?.[0])})`; break;
    case "reshape": expression = pytorchReshape(args[0], node.output_shapes?.[0]); break;
    case "squeeze": expression = pytorchReshape(args[0], node.output_shapes?.[0]); break;
    case "relu": expression = `F.relu(${args[0]})`; break;
    case "relu6": expression = `F.relu6(${args[0]})`; break;
    case "sigmoid": expression = `torch.sigmoid(${args[0]})`; break;
    case "softmax": expression = `F.softmax(${args[0]}, dim=${node.output_shapes?.[0]?.length === 4 ? 1 : -1})`; break;
    case "precision_boundary": expression = args[0]; break;
    default:
      return [comment, `        raise NotImplementedError(${pythonString(`No reviewed emitter for #${node.op_index} ${node.op_name}`)})`];
  }
  const lines = [comment, `        ${target} = ${expression}`];
  return lines.concat(pytorchActivation(target, node.fused_activation));
}

function pytorchActivation(target, activation) {
  switch (String(activation || "NONE").toUpperCase()) {
    case "RELU": return [`        ${target} = F.relu(${target})`];
    case "RELU6": return [`        ${target} = F.relu6(${target})`];
    case "TANH": return [`        ${target} = torch.tanh(${target})`];
    default: return [];
  }
}

function pytorchArguments(plan) {
  return plan.model_inputs.map((_, index) => `input_${index}`).join(", ") || "input_0";
}

function pytorchInputs(plan) {
  if (!plan.model_inputs.length) return ["        values[0] = input_0"];
  return plan.model_inputs.map((tensor, index) => tensor.projected_shape?.length === 4
    ? `        values[${tensor.tensor_index}] = input_${index}.permute(0, 3, 1, 2).contiguous()`
    : `        values[${tensor.tensor_index}] = input_${index}`);
}

function pytorchOutputs(plan) {
  const values = plan.model_outputs.map((tensor) => tensor.projected_shape?.length === 4
    ? `values[${tensor.tensor_index}].permute(0, 2, 3, 1).contiguous()`
    : `values[${tensor.tensor_index}]`);
  if (!values.length) return ["        raise RuntimeError(\"No projected model output is available.\")"];
  return [`        return ${values.length === 1 ? values[0] : `(${values.join(", ")})`}`];
}

function renderPytorchSmokeTest(plan) {
  const inputs = plan.model_inputs.map((tensor) => `torch.randn(${pythonShape(tensor.projected_shape)})`).join(", ");
  return `"""Shape-only smoke test. This does not establish inference equivalence or accuracy."""\nimport torch\nfrom model import ReconstructedModel\n\nmodel = ReconstructedModel().eval()\nwith torch.no_grad():\n    output = model(${inputs || "torch.randn(1, 1)"})\nprint([tuple(item.shape) for item in output] if isinstance(output, tuple) else tuple(output.shape))\n`;
}

function renderKeras(plan) {
  const lines = [
    '"""Weight-free Keras architecture scaffold derived from a DEEPBOM WASM projection."""',
    "from __future__ import annotations",
    "",
    "import tensorflow as tf",
    "from tensorflow import keras",
    "from tensorflow.keras import layers",
    "",
    "",
    "def build_model():",
    "    values = {}",
  ];
  if (!plan.model_inputs.length) {
    lines.push('    raise RuntimeError("No projected model input is available.")');
  } else {
    plan.model_inputs.forEach((tensor, index) => {
      const shape = (tensor.projected_shape || []).slice(1);
      lines.push(`    input_${index} = keras.Input(shape=${pythonTuple(shape)}, name=${pythonString(`input_${index}`)})`);
      lines.push(`    values[${tensor.tensor_index}] = input_${index}`);
    });
    for (const node of plan.nodes) lines.push(...kerasForwardNode(node));
    const outputs = plan.model_outputs.map((tensor) => `values[${tensor.tensor_index}]`);
    const inputs = plan.model_inputs.map((_, index) => `input_${index}`);
    lines.push(`    return keras.Model(inputs=[${inputs.join(", ")}], outputs=[${outputs.join(", ")}], name="deepbom_reconstruction")`);
  }
  lines.push("", "", 'if __name__ == "__main__":', "    build_model().summary()", "");
  return `${lines.join("\n")}\n`;
}

function kerasForwardNode(node) {
  const output = node.outputs?.[0];
  const inputs = (node.activation_inputs || []).map((index) => `values[${index}]`);
  const target = `values[${output}]`;
  const name = pythonString(node.generated_symbol);
  const comment = `    # ${sourceComment(node)} | ${node.codegen_status}`;
  if (!Number.isInteger(output)) return [comment, `    raise NotImplementedError(${pythonString(`Operator #${node.op_index} has no output.`)})`];
  let expression = "";
  switch (node.module_kind) {
    case "conv2d":
      expression = `layers.Conv2D(${node.output_channels}, (${node.kernel_height}, ${node.kernel_width || node.kernel_height}), strides=(${node.stride_height}, ${node.stride_width || node.stride_height}), padding="same", name=${name})(${inputs[0]})`;
      break;
    case "depthwise_conv2d":
      expression = `layers.DepthwiseConv2D((${node.kernel_height}, ${node.kernel_width || node.kernel_height}), strides=(${node.stride_height}, ${node.stride_width || node.stride_height}), padding="same", depth_multiplier=${Math.max(1, Math.round((node.output_channels || 1) / (node.input_channels || 1)))}, name=${name})(${inputs[0]})`;
      break;
    case "linear": expression = `layers.Dense(${featureCount(node.output_shapes?.[0])}, name=${name})(layers.Flatten()(${inputs[0]}))`; break;
    case "add": expression = `layers.Add(name=${name})([${inputs.join(", ")}])`; break;
    case "subtract": expression = `layers.Subtract(name=${name})([${inputs.join(", ")}])`; break;
    case "multiply": expression = `layers.Multiply(name=${name})([${inputs.join(", ")}])`; break;
    case "maximum": expression = `layers.Maximum(name=${name})([${inputs.join(", ")}])`; break;
    case "minimum": expression = `layers.Minimum(name=${name})([${inputs.join(", ")}])`; break;
    case "concatenate": expression = `layers.Concatenate(axis=${node.concatenation_axis_nhwc}, name=${name})([${inputs.join(", ")}])`; break;
    case "average_pool2d": expression = `layers.GlobalAveragePooling2D(keepdims=${node.output_shapes?.[0]?.length === 4 ? "True" : "False"}, name=${name})(${inputs[0]})`; break;
    case "max_pool2d": expression = `layers.GlobalMaxPooling2D(keepdims=${node.output_shapes?.[0]?.length === 4 ? "True" : "False"}, name=${name})(${inputs[0]})`; break;
    case "reshape":
    case "squeeze": expression = `layers.Reshape(${pythonTuple((node.output_shapes?.[0] || []).slice(1))}, name=${name})(${inputs[0]})`; break;
    case "relu": expression = `layers.ReLU(name=${name})(${inputs[0]})`; break;
    case "relu6": expression = `layers.ReLU(max_value=6.0, name=${name})(${inputs[0]})`; break;
    case "sigmoid": expression = `layers.Activation("sigmoid", name=${name})(${inputs[0]})`; break;
    case "softmax": expression = `layers.Softmax(axis=-1, name=${name})(${inputs[0]})`; break;
    case "precision_boundary": expression = `layers.Activation("linear", name=${name})(${inputs[0]})`; break;
    default: return [comment, `    raise NotImplementedError(${pythonString(`No reviewed emitter for #${node.op_index} ${node.op_name}`)})`];
  }
  const lines = [comment, `    ${target} = ${expression}`];
  return lines.concat(kerasActivation(target, node));
}

function kerasActivation(target, node) {
  const name = pythonString(`${node.generated_symbol}_activation`);
  switch (String(node.fused_activation || "NONE").toUpperCase()) {
    case "RELU": return [`    ${target} = layers.ReLU(name=${name})(${target})`];
    case "RELU6": return [`    ${target} = layers.ReLU(max_value=6.0, name=${name})(${target})`];
    case "TANH": return [`    ${target} = layers.Activation("tanh", name=${name})(${target})`];
    default: return [];
  }
}

function renderLiteRtConverter() {
  return `"""Convert the untrained Keras structure scaffold to a float LiteRT artifact.\n\nNo PTQ/QAT setting is enabled because calibration and training data are not encoded\nin the source deployment artifact.\n"""\nfrom pathlib import Path\nimport tensorflow as tf\nfrom model import build_model\n\nmodel = build_model()\nconverter = tf.lite.TFLiteConverter.from_keras_model(model)\nartifact = converter.convert()\nPath("reconstructed_float.tflite").write_bytes(artifact)\nprint(f"wrote {len(artifact)} bytes")\n`;
}

function packageReadme(analysis, projection, scenario) {
  const plan = projection.implementation_plan;
  return `# DEEPBOM Weight-Free Reconstruction Package\n\n- Source artifact: \`${projection.source?.filename || analysis?.filename || "unknown"}\`\n- Source SHA-256: \`${projection.source?.sha256_before || "unknown"}\`\n- Target profile: \`${projection.source?.target_id || "unknown"}\`\n- Scenario: \`${scenarioFingerprint(scenario)}\`\n- Projection status: \`${projection.status}\` / \`${projection.projection_status}\`\n- Codegen coverage: ${plan.exact_codegen_op_count} exact-structure, ${plan.scaffold_codegen_op_count} scaffold, ${plan.unsupported_codegen_op_count} unsupported\n- Weights included: **no**\n\n## What this package is\n\nThe files provide PyTorch and Keras implementations of the projected operator/tensor structure. They are generated from the same WASM projection ledger used by the Redesign metrics. Artifact tensor paths are included only as source-like mapping evidence.\n\n## What this package is not\n\nIt is not recovered source code, a trained model, an accuracy forecast, or a deployable replacement. Randomly initialized parameters must be trained. Dataset, preprocessing, labels, loss, optimizer, calibration, and clinical validation remain external contracts.\n\n## Required gates\n\n1. Review every scaffold or unsupported operator in \`implementation_plan.md\`.\n2. Run each framework smoke test and verify projected output shapes.\n3. Bind governed preprocessing, dataset, labels, loss, and optimizer.\n4. Train or fine-tune and evaluate task, calibration, robustness, and subgroup metrics.\n5. Convert a new deployment artifact and rerun the complete DEEPBOM static and runtime audit.\n`;
}

function implementationMarkdown(projection, scenario) {
  const plan = projection.implementation_plan;
  const lines = [
    "# Redesign Implementation Plan",
    "",
    `- Scenario: \`${scenarioFingerprint(scenario)}\``,
    `- Plan status: \`${plan.status}\``,
    `- Evidence class: \`${plan.evidence_class}\``,
    `- Source-like mappings: ${plan.mapped_source_layer_count}/${plan.nodes.length}`,
    `- Codegen: ${plan.exact_codegen_op_count} exact-structure / ${plan.scaffold_codegen_op_count} scaffold / ${plan.unsupported_codegen_op_count} unsupported`,
    "",
    "| Op | Block | Generated symbol | Source-like reference | Mapping evidence | Codegen |",
    "|---:|---|---|---|---|---|",
  ];
  for (const node of plan.nodes) {
    lines.push(`| #${String(node.op_index).padStart(3, "0")} ${md(node.op_name)} | ${md(node.block_id || "unbound")} | \`${md(node.generated_symbol)}\` | ${md(node.source_layer_ref || "not available")} | ${md(node.source_layer_evidence_class)} | ${md(node.codegen_status)} |`);
  }
  lines.push("", "## Interpretation boundary", "", plan.interpretation_boundary, "");
  return `${lines.join("\n")}\n`;
}

function sourceComment(node) {
  return `#${String(node.op_index).padStart(3, "0")} ${node.op_name} | ${node.block_id || "unbound"} | ${String(node.source_layer_ref || "artifact path unavailable").replace(/[\r\n]+/g, " ")}`;
}

function pytorchReshape(value, shape) {
  if (!Array.isArray(shape) || shape.length < 2) return value;
  if (shape.length === 2) return `torch.flatten(${value}, 1)`;
  const nchw = shape.length === 4 ? [shape[3], shape[1], shape[2]] : shape.slice(1);
  return `${value}.reshape(${value}.shape[0], ${nchw.map((item) => Number(item) > 0 ? Number(item) : -1).join(", ")})`;
}

function pytorchSpatial(shape) {
  return Array.isArray(shape) && shape.length === 4 ? `(${Math.max(1, Number(shape[1]) || 1)}, ${Math.max(1, Number(shape[2]) || 1)})` : "(1, 1)";
}

function pytorchAxis(nhwcAxis, rank) {
  if (!Number.isInteger(nhwcAxis) || !Number.isInteger(rank)) return -1;
  if (rank !== 4) return nhwcAxis;
  return [0, 2, 3, 1][nhwcAxis] ?? -1;
}

function featureCount(shape) {
  if (!Array.isArray(shape) || shape.length < 2) return 0;
  return shape.slice(1).reduce((product, value) => product * Math.max(1, Number(value) || 1), 1);
}

function pythonShape(shape) {
  return (shape || []).map((value) => Math.max(1, Number(value) || 1)).join(", ");
}

function pythonTuple(values) {
  const items = (values || []).map((value) => Math.max(1, Number(value) || 1));
  return `(${items.join(", ")}${items.length === 1 ? "," : ""})`;
}

function pythonString(value) {
  return JSON.stringify(String(value ?? ""));
}

function finiteInteger(value) {
  const number = Number(value);
  return Number.isInteger(number) ? number : null;
}

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function md(value) {
  return String(value ?? "").replaceAll("|", "\\|").replace(/[\r\n]+/g, " ");
}

function json(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function textFile(name, data) {
  return { name, data };
}
