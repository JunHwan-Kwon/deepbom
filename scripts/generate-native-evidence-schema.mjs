import { readFile, writeFile } from "node:fs/promises";
import { NATIVE_SCHEMAS } from "../web/lib/native-evidence.js";
const base = "https://deepbom.org/schemas/";
const string = { type: "string" },
  hash = { type: "string", pattern: "^[a-f0-9]{64}$" },
  count = { type: "string", pattern: "^(0|[1-9][0-9]*)$" },
  nullable = (x) => ({ anyOf: [x, { type: "null" }] }),
  array = (x) => ({ type: "array", items: x }),
  obj = (properties) => ({
    type: "object",
    additionalProperties: false,
    required: Object.keys(properties),
    properties,
  });
const json = {
  description:
    "Explicit framework/user metadata, never inferred domain semantics.",
  type: "object",
};
const identity = obj({
  id: string,
  name: string,
  role: { enum: ["parameter", "buffer", "variable", "activation", "gradient"] },
  trainable: { type: "boolean" },
  aliases: { ...array(string), uniqueItems: true },
  dtype: {
    enum: [
      "F16",
      "BF16",
      "F32",
      "F64",
      "I8",
      "U8",
      "I16",
      "U16",
      "I32",
      "U32",
      "I64",
      "U64",
      "BOOL",
    ],
  },
  shape: array({
    type: "integer",
    minimum: 0,
    maximum: Number.MAX_SAFE_INTEGER,
  }),
  byte_order: { const: "little" },
  element_count: count,
  byte_length: count,
  payload_sha256: hash,
});
const definition = obj({
  framework: obj({
    name: { enum: ["pytorch", "tensorflow"] },
    version: string,
    backend: string,
  }),
  nodes: array(
    obj({
      id: string,
      kind: string,
      config: json,
      state_refs: array(string),
      mode: { enum: ["train", "eval", "call_dependent"] },
    }),
  ),
  relationships: array(
    obj({
      from: string,
      to: string,
      kind: { enum: ["contains", "data_dependency"] },
    }),
  ),
  inputs: json,
  scope: { enum: ["module_tree", "functional_graph", "fx_graph"] },
  limitations: array(string),
});
const source = (numerical = false) =>
  obj({
    kind: { const: "model_state_snapshot" },
    snapshot_sha256: hash,
    ...(numerical ? { model_ir_sha256: hash } : {}),
  });
const row = (activation = false) =>
  obj({
    identity,
    status: { enum: ["assessed", "not_assessed"] },
    reason: nullable(string),
    statistics: nullable({
      $ref: base + "deepbom-numerical-ir-v1.schema.json#/$defs/statistics",
    }),
    distribution: nullable(json),
    features: nullable(
      obj({
        axis: { const: 0 },
        axis_meaning: {
          const: "native_storage_axis_not_inferred_output_channel",
        },
        channels: json,
        spectrum: json,
        sparsity: json,
        similarity: json,
        projection: json,
      }),
    ),
    ...(activation ? { subject_ref: string, invocation: string } : {}),
  });
const header = (schema) => ({
  schema: { const: schema },
  method_version: { const: "1.0.0" },
  boundary: string,
});
const event = obj({
  id: string,
  sequence: count,
  kind: string,
  at: json,
  payload: json,
});
const reference = obj({ schema: string, sha256: hash });
const rawTensor = obj({
  ...Object.fromEntries(
    Object.entries(identity.properties).filter(
      ([k]) => !["element_count", "byte_length", "payload_sha256"].includes(k),
    ),
  ),
  data_base64: { type: "string", contentEncoding: "base64" },
});
const totals = obj({
  state_elements: count,
  state_bytes: count,
  trainable_elements: count,
});
const defs = {
  input: obj({
    schema: { const: "deepbom.native_evidence_input.v1" },
    definition,
    tensors: array(rawTensor),
    capture: nullable(
      obj({
        context: json,
        values: array(
          obj({ subject_ref: string, invocation: string, tensor: rawTensor }),
        ),
      }),
    ),
    options: obj({
      max_values: { type: "integer", minimum: 0, maximum: 100000000 },
      advanced: { type: "boolean" },
    }),
  }),
  training_input: obj({
    schema: { const: "deepbom.training_input.v1" },
    run_id: string,
    segment_id: string,
    lifecycle: {
      enum: ["running", "ended", "failed", "interrupted", "collector_closed"],
    },
    events: array(event),
    previous: nullable({ $ref: "#/$defs/training" }),
  }),
  candidate: obj({
    schema: { const: "deepbom.model_candidate_manifest.v1" },
    snapshot_sha256: hash,
    baseline_sha256: hash,
    parent_sha256: nullable(hash),
    evidence: obj({
      snapshot: reference,
      model_ir: reference,
      weight_ir: reference,
    }),
    files: { type: "object", additionalProperties: hash },
    optimizer_state: { const: "not_restored" },
    manifest_sha256: hash,
  }),
  comparison: obj({
    schema: { const: "deepbom.native_model_comparison.v1" },
    baseline: hash,
    candidate: hash,
    tensors: array(
      obj({
        id: string,
        status: {
          enum: [
            "added",
            "removed",
            "aligned_native_id",
            "shape_or_dtype_changed",
          ],
        },
        matching_basis: { const: "native_id_not_semantic_equivalence" },
        payload_equal: nullable({ type: "boolean" }),
        distribution: nullable(json),
      }),
    ),
    nodes: array(
      obj({
        id: string,
        status: { enum: ["added", "removed", "unchanged", "changed"] },
      }),
    ),
    totals: obj({ baseline: totals, candidate: totals }),
    structure_equal: { type: "boolean" },
    boundary: string,
    comparison_sha256: hash,
  }),
  result: obj({
    ...header("deepbom.training_result_manifest.v1"),
    training_ir_sha256: hash,
    snapshot_sha256: hash,
    selected_event_id: string,
    selection_rule: { const: "latest_recorded_capture_of_requested_state" },
    evidence: obj({
      snapshot: reference,
      model_ir: reference,
      weight_ir: reference,
    }),
    training_lifecycle: {
      enum: ["running", "ended", "failed", "interrupted", "collector_closed"],
    },
    training_result_manifest_sha256: hash,
  }),
  snapshot: obj({
    ...header(NATIVE_SCHEMAS.snapshot),
    definition,
    tensors: array(identity),
    completeness: { const: "adapter_declared_scope" },
    snapshot_sha256: hash,
  }),
  model: obj({
    ...header(NATIVE_SCHEMAS.model),
    source: source(),
    program: definition,
    storage: array(identity),
    model_ir_sha256: hash,
  }),
  weight: obj({
    ...header(NATIVE_SCHEMAS.weight),
    source: source(true),
    tensors: array(row()),
    coverage: obj({
      total: { type: "integer", minimum: 0 },
      assessed: { type: "integer", minimum: 0 },
    }),
    weight_ir_sha256: hash,
  }),
  activation: obj({
    ...header(NATIVE_SCHEMAS.activation),
    source: source(true),
    context: {
      type: "object",
      required: ["mode", "input_identity", "state_boundary"],
      properties: { mode: string, input_identity: {}, state_boundary: string },
    },
    tensors: array(row(true)),
    activation_ir_sha256: hash,
  }),
  training: obj({
    ...header(NATIVE_SCHEMAS.training),
    run_id: string,
    segment_id: string,
    lifecycle: {
      enum: ["running", "ended", "failed", "interrupted", "collector_closed"],
    },
    event_count: count,
    chunks: array(obj({ sha256: hash, offset: count, count })),
    previous_ir_sha256: nullable(hash),
    training_ir_sha256: hash,
  }),
  chunk: obj({
    schema: { const: "deepbom.training_chunk.v1" },
    run_id: string,
    segment_id: string,
    events: array(event),
    chunk_sha256: hash,
  }),
};
const schema = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: base + "deepbom-native-evidence-v1.schema.json",
  title: "DEEPBOM native state and training evidence",
  description:
    "Additive native source contracts. Not aliases or replacements for artifact-based v1 model/numerical IR. Semantic validators check digests, counts and exact bindings. Config, context and event payload retain declared metadata; schema validity is not authenticity.",
  $defs: defs,
  oneOf: Object.keys(defs).map((k) => ({ $ref: "#/$defs/" + k })),
};
const family = {
  $schema: schema.$schema,
  $id: base + "deepbom-evidence-ir-v2.schema.json",
  title: "DEEPBOM Evidence IR family with native state and training",
  oneOf: [
    { $ref: base + "deepbom-evidence-ir-v1.schema.json" },
    ...["model", "weight", "activation", "training"].map((k) => ({
      $ref: schema.$id + "#/$defs/" + k,
    })),
  ],
};
// Output projections are registered from their own schemas, not hand-appended
// to generated JSON. Regeneration must preserve the full native catalog.
const projectionContracts = {};
for (const [key, filename, definition] of [
  ["optimization_report", "deepbom-optimization-report-v1.schema.json"],
  ["optimization_diff", "deepbom-optimization-diff-v1.schema.json"],
  ["optimization_report_query", "deepbom-optimization-report-access-v1.schema.json", "query"],
  ["optimization_report_export", "deepbom-optimization-report-access-v1.schema.json", "export"],
]) {
  const document = JSON.parse(await readFile("docs/schemas/" + filename, "utf8"));
  const contract = definition ? document.$defs?.[definition] : document;
  const identity = contract?.properties?.schema?.const;
  if (typeof identity !== "string" || document.$id !== base + filename)
    throw Error("Invalid native output contract: " + filename);
  projectionContracts[key] = {
    schema: identity,
    json_schema: document.$id + (definition ? "#/$defs/" + definition : ""),
  };
}
const catalog = {
  schema: "deepbom.native_evidence_catalog.v1",
  status: "experimental",
  catalog_version: "0.3.0",
  family_schema: family.$id,
  json_schema: schema.$id,
  contracts: { ...Object.fromEntries(
    Object.entries(defs).map(([k, v]) => [
      k,
      {
        schema: v.properties.schema.const,
        json_schema: schema.$id + "#/$defs/" + k,
      },
    ]),
  ), ...projectionContracts },
  scope:
    "Qualified CPU native adapters. Snapshot and chunk are supporting documents, not extra IR layers. Artifact-based family v1 is preserved.",
};
for (const [path, value] of [
  ["docs/schemas/deepbom-native-evidence-v1.schema.json", schema],
  ["docs/schemas/deepbom-evidence-ir-v2.schema.json", family],
  ["docs/evidence-ir/native-catalog.json", catalog],
]) {
  const content = JSON.stringify(value, null, 2) + "\n";
  if (process.argv.includes("--check")) {
    if ((await readFile(path, "utf8")) !== content)
      throw Error("Stale " + path);
  } else await writeFile(path, content);
}
