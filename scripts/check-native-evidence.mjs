import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import Ajv from "ajv/dist/2020.js";
import {
  buildNativeEvidence,
  analyzeNativeTensor,
  validateNativeBundle,
  validateNativeDocument,
  buildTrainingIr,
  validateTrainingIr,
  nativeDigest,
} from "../web/lib/native-evidence.js";
const input = {
  schema: "deepbom.native_evidence_input.v1",
  definition: {
    framework: { name: "pytorch", version: "test", backend: "torch" },
    nodes: [
      {
        id: "linear",
        kind: "test.Linear",
        config: {},
        state_refs: ["w"],
        mode: "train",
      },
    ],
    relationships: [],
    inputs: {},
    scope: "module_tree",
    limitations: [],
  },
  tensors: [
    {
      id: "w",
      name: "w",
      role: "parameter",
      trainable: true,
      aliases: [],
      dtype: "F32",
      shape: [2, 2],
      byte_order: "little",
      data_base64: Buffer.from(new Float32Array([1, 0, 0, 2]).buffer).toString(
        "base64",
      ),
    },
  ],
  capture: null,
  options: { max_values: 4, advanced: true },
};
const b = await buildNativeEvidence(input);
validateNativeBundle(b);
// Independent native transport width oracle: production widths come from the
// common scalar registry, while this fixture must detect incorrect aliases.
for (const [dtype, bytes] of [
  ["F16", 2],
  ["BF16", 2],
  ["F32", 4],
  ["F64", 8],
  ["I8", 1],
  ["U8", 1],
  ["I16", 2],
  ["U16", 2],
  ["I32", 4],
  ["U32", 4],
  ["I64", 8],
  ["U64", 8],
  ["BOOL", 1],
]) {
  const tensor = await analyzeNativeTensor({
    ...input.tensors[0],
    dtype,
    shape: [2],
    data_base64: Buffer.alloc(2 * bytes).toString("base64"),
  });
  assert.equal(tensor.identity.byte_length, String(2 * bytes), dtype);
  assert.equal(tensor.statistics.zero_count, "2", dtype);
}
assert.equal(b.weight_ir.tensors[0].statistics.mean, 0.75);
assert.equal(b.weight_ir.tensors[0].statistics.zero_count, "2");
assert.deepEqual(
  b.weight_ir.tensors[0].features.spectrum.singular_values,
  [2, 1],
);
const ajv = new Ajv({ strict: true, allErrors: true });
for (const name of [
  "artifact-ir-v2",
  "model-ir-v1",
  "numerical-ir-v1",
  "provenance-ir-v1",
  "evidence-ir-v1",
  "native-evidence-v1",
  "evidence-ir-v2",
])
  ajv.addSchema(
    JSON.parse(
      await readFile(`docs/schemas/deepbom-${name}.schema.json`, "utf8"),
    ),
  );
const valid = ajv.getSchema(
  "https://deepbom.org/schemas/deepbom-native-evidence-v1.schema.json",
);
for (const doc of Object.values(b).filter(Boolean))
  assert(valid(doc), JSON.stringify(valid.errors));
function alter(doc, key, edit) {
  const bad = structuredClone(doc);
  edit(bad);
  delete bad[key];
  bad[key] = nativeDigest(bad);
  return bad;
}
assert.throws(
  () =>
    validateNativeDocument(
      alter(b.weight_ir, "weight_ir_sha256", (d) => (d.coverage.assessed = 0)),
    ),
  /coverage/,
);
assert.throws(
  () =>
    validateNativeDocument(
      alter(
        b.snapshot,
        "snapshot_sha256",
        (d) => (d.tensors[0].element_count = "5"),
      ),
    ),
  /counts/,
);
assert.throws(
  () =>
    validateNativeDocument(
      alter(
        b.model_ir,
        "model_ir_sha256",
        (d) => (d.program.nodes[0].state_refs = ["foreign"]),
      ),
    ),
  /reference/,
);
assert.throws(
  () =>
    validateNativeBundle({
      ...b,
      weight_ir: alter(
        b.weight_ir,
        "weight_ir_sha256",
        (d) => (d.source.snapshot_sha256 = "0".repeat(64)),
      ),
    }),
  /source/,
);
const truncated = await buildNativeEvidence({
  ...input,
  options: { max_values: 3, advanced: true },
});
assert.equal(truncated.weight_ir.tensors[0].status, "not_assessed");
assert.equal(truncated.weight_ir.tensors[0].statistics, null);
validateNativeBundle(truncated);
const capture = {
  context: {
    mode: "train",
    input_identity: "declared-fixture",
    state_boundary: "before_forward",
  },
  values: [
    {
      subject_ref: "linear",
      invocation: "0",
      tensor: {
        ...input.tensors[0],
        id: "capture:0",
        name: "capture:0",
        role: "activation",
        trainable: false,
      },
    },
  ],
};
const active = await buildNativeEvidence({ ...input, capture });
validateNativeBundle(active);
assert(valid(active.activation_ir), JSON.stringify(valid.errors));
const first = buildTrainingIr({
  schema: "deepbom.training_input.v1",
  run_id: "r",
  segment_id: "s",
  lifecycle: "running",
  events: [
    {
      id: "s:0",
      sequence: "0",
      kind: "metrics",
      at: { step: 0 },
      payload: { values: { loss: 1 } },
    },
  ],
  previous: null,
});
for (const d of Object.values(first))
  assert(valid(d), JSON.stringify(valid.errors));
const next = buildTrainingIr({
  schema: "deepbom.training_input.v1",
  run_id: "r",
  segment_id: "s",
  lifecycle: "collector_closed",
  events: [
    { id: "s:1", sequence: "1", kind: "collector_closed", at: {}, payload: {} },
  ],
  previous: first.document,
});
assert.equal(next.document.event_count, "2");
assert.throws(
  () =>
    validateTrainingIr(
      alter(
        next.document,
        "training_ir_sha256",
        (d) => (d.chunks[1].offset = "0"),
      ),
    ),
  /range/,
);
assert.throws(
  () =>
    buildTrainingIr({
      schema: "deepbom.training_input.v1",
      run_id: "other",
      segment_id: "s",
      lifecycle: "running",
      events: [],
      previous: first.document,
    }),
  /another run/,
);
assert.throws(
  () => validateNativeDocument({ ...b.weight_ir, extra: true }),
  /mismatch/,
);
console.log(
  "Native schemas, exact statistics, SVD, budgets, source/subject binding and training continuity passed.",
);

// Multi-megabyte transport must not exhaust the RegExp interpreter stack.
const large = await analyzeNativeTensor(
  {
    ...input.tensors[0],
    shape: [1000000],
    data_base64: Buffer.alloc(4000000).toString("base64"),
  },
  { maxValues: 0 },
);
assert.equal(large.identity.element_count, "1000000");
assert.equal(large.status, "not_assessed");
for (const bad of ["====", "A===", "AA=A", "!!!!", "A"])
  await assert.rejects(
    analyzeNativeTensor({ ...input.tensors[0], data_base64: bad }),
    /base64/,
  );
console.log(
  "Large native tensor transport and malformed base64 rejection passed.",
);

const oldFamily = ajv.getSchema(
    "https://deepbom.org/schemas/deepbom-evidence-ir-v1.schema.json",
  ),
  newFamily = ajv.getSchema(
    "https://deepbom.org/schemas/deepbom-evidence-ir-v2.schema.json",
  );
for (const document of [
  b.model_ir,
  b.weight_ir,
  active.activation_ir,
  first.document,
]) {
  assert.equal(oldFamily(document), false);
  assert(newFamily(document), JSON.stringify(newFamily.errors));
}
assert(valid(input), JSON.stringify(valid.errors));
console.log(
  "Native documents require the additive family; legacy family acceptance was not widened.",
);
