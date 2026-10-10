// Common owner for native snapshot evidence. Existing artifact-based v1 IRs
// remain immutable contracts. Framework adapters supply facts, never statistics.
import { canonicalJson } from "./report-utils.js";
import { sha256BytesHex } from "./sha256-sync.js";
import {
  exactKeys,
  requireCondition as need,
  shapeCount,
  SHA256,
} from "./numerical-ir/common.js";
import {
  TensorStatistics,
  validateStatistics,
} from "./numerical-ir/statistics.js";
import {
  describeDistribution,
  compareDistributions,
} from "./numerical-ir/distribution.js";
import {
  matrixView,
  channelStatistics,
  spectrumAnalysis,
  sparsityAnalysis,
  similarityAnalysis,
  tileProjection,
  notAssessed,
} from "./numerical-ir/weight-math.js";
import { validateWeightMathFeatures } from "./weight-analysis.js";
import { visitDenseTensorValues } from "./tensor-numerical-integrity.js";
import { scalarDtypeBytes } from "./tensor-size.js";
import { canonicalEvidenceDigest, sealDocument } from "./evidence-identity.js";

export const NATIVE_SCHEMAS = Object.freeze({
  snapshot: "deepbom.model_state_snapshot.v1",
  model: "deepbom.model_ir.v2",
  weight: "deepbom.weight_ir.v2",
  activation: "deepbom.activation_ir.v2",
  training: "deepbom.training_ir.v1",
});
export const nativeDigest = canonicalEvidenceDigest;
export function sealNative(body, field) {
  return sealDocument(body, field);
}
function text(x, label) {
  need(
    typeof x === "string" && x.length > 0 && x.length <= 4096,
    `invalid ${label}`,
  );
}
function record(x, label) {
  need(x && typeof x === "object" && !Array.isArray(x), `invalid ${label}`);
  canonicalJson(x);
}
function digest(x) {
  need(typeof x === "string" && SHA256.test(x), "invalid native digest");
}
function unique(rows, label) {
  need(
    Array.isArray(rows) && new Set(rows.map((r) => r.id)).size === rows.length,
    `duplicate ${label}`,
  );
  rows.forEach((r) => text(r.id, label));
}
function checkSeal(doc, field) {
  const { [field]: hash, ...body } = doc;
  digest(hash);
  need(nativeDigest(body) === hash, `${field} mismatch`);
}
const roles = new Set([
  "parameter",
  "buffer",
  "variable",
  "activation",
  "gradient",
]);
const nativeDtypes = new Set([
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
]);
export async function analyzeNativeTensor(
  row,
  { maxValues = 1_000_000, advanced = false } = {},
) {
  need(
    Number.isSafeInteger(maxValues) &&
      maxValues >= 0 &&
      maxValues <= 100_000_000,
    "invalid value budget",
  );
  need(typeof advanced === "boolean", "invalid advanced option");
  exactKeys(
    row,
    [
      "id",
      "name",
      "role",
      "trainable",
      "aliases",
      "dtype",
      "shape",
      "byte_order",
      "data_base64",
    ],
    "native tensor",
  );
  text(row.id, "tensor id");
  text(row.name, "tensor name");
  need(roles.has(row.role), "invalid native tensor role");
  need(typeof row.trainable === "boolean", "invalid trainability");
  need(
    Array.isArray(row.aliases) &&
      new Set(row.aliases).size === row.aliases.length &&
      row.aliases.every((x) => typeof x === "string"),
    "invalid aliases",
  );
  need(nativeDtypes.has(row.dtype), "unsupported native dtype");
  need(
    Array.isArray(row.shape) &&
      row.shape.every((d) => Number.isSafeInteger(d) && d >= 0),
    "native dimensions must be exact nonnegative integers",
  );
  const count = shapeCount(row.shape);
  need(count <= 100_000_000n, "native tensor exceeds transport budget");
  need(row.byte_order === "little", "native byte order must be little");
  need(
    typeof row.data_base64 === "string" &&
      row.data_base64.length <= 128 * 1024 * 1024 &&
      row.data_base64.length % 4 === 0 &&
      !/[^A-Za-z0-9+/=]/.test(row.data_base64) &&
      (row.data_base64.indexOf("=") === -1 ||
        (row.data_base64.indexOf("=") >= row.data_base64.length - 2 &&
          ["=", "=="].includes(
            row.data_base64.slice(row.data_base64.indexOf("=")),
          ))),
    "invalid native base64 payload",
  );
  const raw = Uint8Array.from(atob(row.data_base64), (c) => c.charCodeAt(0));
  need(
    BigInt(raw.length) === count * BigInt(scalarDtypeBytes(row.dtype)),
    "native shape/byte count mismatch",
  );
  const { data_base64, ...descriptor } = row;
  const identity = {
    ...descriptor,
    element_count: String(count),
    byte_length: String(raw.length),
    payload_sha256: sha256BytesHex(raw),
  };
  const stats = new TensorStatistics(),
    values = [];
  const result = await visitDenseTensorValues(
    raw,
    row,
    (value) => {
      stats.add(value);
      if (advanced) values.push(value);
    },
    { maxValues },
  );
  const statistics = result.status === "assessed" ? stats.finish() : null;
  let features = null;
  if (
    advanced &&
    statistics &&
    statistics.unsafe_integer_count === "0" &&
    statistics.finite_count === statistics.value_count &&
    count > 0n
  ) {
    const numbers = values.map(Number),
      matrix = matrixView(numbers, row.shape, 0, "last_axis_fastest");
    features = {
      axis: 0,
      axis_meaning: "native_storage_axis_not_inferred_output_channel",
      channels:
        matrix.rows <= 4096
          ? channelStatistics(matrix)
          : notAssessed("channel_budget_exceeded"),
      spectrum: spectrumAnalysis(matrix),
      sparsity: sparsityAnalysis(numbers, row.shape, "last_axis_fastest", 0),
      similarity: similarityAnalysis(matrix, 64, 0.99),
      projection: tileProjection(matrix),
    };
  }
  return {
    identity,
    status: result.status,
    reason: result.reason || null,
    statistics,
    distribution: statistics ? describeDistribution(statistics) : null,
    features,
  };
}
function validateDefinition(definition, stateIds) {
  exactKeys(
    definition,
    ["framework", "nodes", "relationships", "inputs", "scope", "limitations"],
    "native definition",
  );
  exactKeys(definition.framework, ["name", "version", "backend"], "framework");
  need(
    ["pytorch", "tensorflow"].includes(definition.framework.name),
    "unsupported native framework",
  );
  text(definition.framework.version, "framework version");
  text(definition.framework.backend, "framework backend");
  need(
    ["module_tree", "functional_graph", "fx_graph"].includes(definition.scope),
    "invalid graph scope",
  );
  unique(definition.nodes, "native nodes");
  const ids = new Set(definition.nodes.map((n) => n.id));
  for (const n of definition.nodes) {
    exactKeys(n, ["id", "kind", "config", "state_refs", "mode"], "native node");
    text(n.kind, "node kind");
    record(n.config, "node config");
    need(
      ["train", "eval", "call_dependent"].includes(n.mode),
      "invalid module mode",
    );
    need(
      Array.isArray(n.state_refs) &&
        n.state_refs.every((ref) => stateIds.has(ref)),
      "unknown native state reference",
    );
  }
  need(Array.isArray(definition.relationships), "invalid relationships");
  for (const r of definition.relationships) {
    exactKeys(r, ["from", "to", "kind"], "native relationship");
    need(ids.has(r.from) && ids.has(r.to), "dangling native relationship");
    need(
      ["contains", "data_dependency"].includes(r.kind),
      "invalid native relationship kind",
    );
  }
  record(definition.inputs, "input spec");
  need(
    Array.isArray(definition.limitations) &&
      definition.limitations.every((s) => typeof s === "string"),
    "invalid native limitations",
  );
}
export async function buildNativeEvidence(input) {
  exactKeys(
    input,
    ["schema", "definition", "tensors", "capture", "options"],
    "native evidence input",
  );
  need(
    input.schema === "deepbom.native_evidence_input.v1",
    "unknown native input schema",
  );
  exactKeys(input.options, ["max_values", "advanced"], "native options");
  need(
    Number.isSafeInteger(input.options.max_values) &&
      input.options.max_values >= 0 &&
      input.options.max_values <= 100_000_000,
    "invalid value budget",
  );
  need(typeof input.options.advanced === "boolean", "invalid advanced option");
  unique(input.tensors, "native tensors");
  validateDefinition(input.definition, new Set(input.tensors.map((t) => t.id)));
  need(
    input.tensors.every((t) =>
      ["parameter", "buffer", "variable"].includes(t.role),
    ),
    "snapshot contains non-state tensor",
  );
  const rows = [];
  for (const tensor of input.tensors)
    rows.push(
      await analyzeNativeTensor(tensor, {
        maxValues: input.options.max_values,
        advanced: input.options.advanced,
      }),
    );
  const snapshot = sealNative(
    {
      schema: NATIVE_SCHEMAS.snapshot,
      method_version: "1.0.0",
      definition: input.definition,
      tensors: rows.map((r) => r.identity),
      completeness: "adapter_declared_scope",
      boundary:
        "Consistent copied state within the adapter scope. A digest does not attest the source or training history; no optimizer/RNG resume state.",
    },
    "snapshot_sha256",
  );
  const source = {
    kind: "model_state_snapshot",
    snapshot_sha256: snapshot.snapshot_sha256,
  };
  const model = sealNative(
    {
      schema: NATIVE_SCHEMAS.model,
      method_version: "1.0.0",
      source,
      program: input.definition,
      storage: rows.map((r) => r.identity),
      boundary:
        "Native module/call structure, not a serialized deployment program. Graph scope and unsupported state remain explicit.",
    },
    "model_ir_sha256",
  );
  const numericalSource = { ...source, model_ir_sha256: model.model_ir_sha256 };
  const weight = sealNative(
    {
      schema: NATIVE_SCHEMAS.weight,
      method_version: "1.0.0",
      source: numericalSource,
      tensors: rows,
      coverage: {
        total: rows.length,
        assessed: rows.filter((r) => r.status === "assessed").length,
      },
      boundary:
        "Native stored state values; trainability is supplied by the framework adapter. Storage is not peak runtime memory.",
    },
    "weight_ir_sha256",
  );
  let activation = null;
  if (input.capture !== null) {
    exactKeys(
      input.capture,
      ["context", "values"],
      "native activation capture",
    );
    record(input.capture.context, "activation context");
    need(Array.isArray(input.capture.values), "invalid activation occurrences");
    unique(
      input.capture.values.map((v) => v.tensor),
      "activation occurrences",
    );
    const nodes = new Set(input.definition.nodes.map((n) => n.id)),
      captures = [];
    for (const value of input.capture.values) {
      exactKeys(
        value,
        ["subject_ref", "invocation", "tensor"],
        "native capture value",
      );
      need(
        nodes.has(value.subject_ref),
        "activation subject not bound to native structure",
      );
      text(value.invocation, "activation invocation");
      need(value.tensor.role === "activation", "invalid activation role");
      captures.push({
        subject_ref: value.subject_ref,
        invocation: value.invocation,
        ...(await analyzeNativeTensor(value.tensor, {
          maxValues: input.options.max_values,
        })),
      });
    }
    activation = sealNative(
      {
        schema: NATIVE_SCHEMAS.activation,
        method_version: "1.0.0",
        source: numericalSource,
        context: input.capture.context,
        tensors: captures,
        boundary:
          "Explicit native invocation observations. State binding requires an adapter-declared consistent boundary; not replay, runtime attestation, or task validation.",
      },
      "activation_ir_sha256",
    );
  }
  return validateNativeBundle({
    snapshot,
    model_ir: model,
    weight_ir: weight,
    activation_ir: activation,
  });
}
function validateNativeIdentity(row) {
  exactKeys(
    row,
    [
      "id",
      "name",
      "role",
      "trainable",
      "aliases",
      "dtype",
      "shape",
      "byte_order",
      "element_count",
      "byte_length",
      "payload_sha256",
    ],
    "tensor identity",
  );
  text(row.id, "tensor id");
  text(row.name, "tensor name");
  need(
    roles.has(row.role) && typeof row.trainable === "boolean",
    "invalid native role",
  );
  need(
    Array.isArray(row.aliases) &&
      row.aliases.every((a) => typeof a === "string" && a !== row.id) &&
      new Set(row.aliases).size === row.aliases.length,
    "invalid aliases",
  );
  need(
    nativeDtypes.has(row.dtype) && row.byte_order === "little",
    "invalid tensor layout",
  );
  need(
    Array.isArray(row.shape) &&
      row.shape.every((d) => Number.isSafeInteger(d) && d >= 0),
    "invalid native shape",
  );
  need(
    row.element_count === String(shapeCount(row.shape)) &&
      row.byte_length ===
        String(shapeCount(row.shape) * BigInt(scalarDtypeBytes(row.dtype))),
    "identity counts do not conserve",
  );
  digest(row.payload_sha256);
}
function validateNativeTensorMetrics(row) {
  validateNativeIdentity(row.identity);
  need(
    ["assessed", "not_assessed"].includes(row.status),
    "invalid assessment status",
  );
  if (row.status === "assessed") {
    validateStatistics(row.statistics);
    need(
      row.statistics.value_count === row.identity.element_count,
      "partial statistics presented as complete",
    );
    need(
      canonicalJson(row.distribution) ===
        canonicalJson(describeDistribution(row.statistics)),
      "distribution contradicts statistics",
    );
    if (row.features !== null) {
      exactKeys(
        row.features,
        [
          "axis",
          "axis_meaning",
          "channels",
          "spectrum",
          "sparsity",
          "similarity",
          "projection",
        ],
        "native features",
      );
      need(
        row.features.axis === 0 &&
          row.features.axis_meaning ===
            "native_storage_axis_not_inferred_output_channel",
        "invalid native axis interpretation",
      );
      for (const key of [
        "channels",
        "spectrum",
        "sparsity",
        "similarity",
        "projection",
      ]) {
        const f = row.features[key];
        need(
          f && ["assessed", "not_assessed"].includes(f.status),
          "invalid feature",
        );
        if (f.status === "not_assessed") {
          exactKeys(f, ["status", "reason"], "unassessed feature");
          text(f.reason, "feature reason");
        }
      }
      validateWeightMathFeatures({
        ...row.features,
        statistics: row.statistics,
        value_count: row.identity.element_count,
      });
    }
  } else
    need(
      row.statistics === null &&
        row.distribution === null &&
        row.features === null &&
        typeof row.reason === "string",
      "unassessed evidence has invented values",
    );
}
export function validateNativeDocument(doc) {
  record(doc, "native document");
  const fields = {
    [NATIVE_SCHEMAS.snapshot]: "snapshot_sha256",
    [NATIVE_SCHEMAS.model]: "model_ir_sha256",
    [NATIVE_SCHEMAS.weight]: "weight_ir_sha256",
    [NATIVE_SCHEMAS.activation]: "activation_ir_sha256",
  };
  const field = fields[doc.schema];
  need(field, "unsupported native document");
  checkSeal(doc, field);
  const common = ["schema", "method_version", "boundary", field];
  need(doc.method_version === "1.0.0", "unsupported native method");
  text(doc.boundary, "boundary");

  if (
    doc.schema === NATIVE_SCHEMAS.snapshot ||
    doc.schema === NATIVE_SCHEMAS.model
  ) {
    const isSnapshot = doc.schema === NATIVE_SCHEMAS.snapshot;
    exactKeys(
      doc,
      [
        ...common,
        ...(isSnapshot
          ? ["definition", "tensors", "completeness"]
          : ["source", "program", "storage"]),
      ],
      "native structural document",
    );
    if (isSnapshot)
      need(
        doc.completeness === "adapter_declared_scope",
        "invalid snapshot scope",
      );
    const rows = isSnapshot ? doc.tensors : doc.storage;
    unique(rows, "state identity");
    rows.forEach(validateNativeIdentity);
    need(
      rows.every((r) => ["parameter", "buffer", "variable"].includes(r.role)),
      "non-state tensor in snapshot",
    );
    const names = rows.flatMap((r) => [r.id, ...r.aliases]);
    need(new Set(names).size === names.length, "ambiguous state aliases");
    validateDefinition(
      isSnapshot ? doc.definition : doc.program,
      new Set(rows.map((r) => r.id)),
    );
  } else {
    const weight = doc.schema === NATIVE_SCHEMAS.weight;
    exactKeys(
      doc,
      [...common, "source", "tensors", weight ? "coverage" : "context"],
      "native numerical document",
    );
    need(Array.isArray(doc.tensors), "invalid tensors");
    unique(
      doc.tensors.map((r) => r.identity),
      "numerical identities",
    );
    for (const row of doc.tensors) {
      exactKeys(
        row,
        [
          "identity",
          "status",
          "reason",
          "statistics",
          "distribution",
          "features",
          ...(weight ? [] : ["subject_ref", "invocation"]),
        ],
        "native numerical row",
      );
      need(
        weight
          ? ["parameter", "buffer", "variable"].includes(row.identity.role)
          : row.identity.role === "activation",
        "wrong numerical role",
      );
      validateNativeTensorMetrics(row);
      if (!weight) {
        text(row.subject_ref, "activation subject");
        text(row.invocation, "invocation");
      }
    }
    if (weight) {
      exactKeys(doc.coverage, ["total", "assessed"], "native coverage");
      need(
        doc.coverage.total === doc.tensors.length &&
          doc.coverage.assessed ===
            doc.tensors.filter((r) => r.status === "assessed").length,
        "native coverage mismatch",
      );
    } else {
      record(doc.context, "activation context");
      need(
        ["mode", "input_identity", "state_boundary"].every((k) =>
          Object.hasOwn(doc.context, k),
        ),
        "missing activation context",
      );
      for (const key of ["mode", "input_identity", "state_boundary"]) {
        need(doc.context[key] != null && doc.context[key] !== "", `activation ${key} needs explicit context or an explicit unknown declaration`);
      }
    }
  }
  if (doc.source) {
    exactKeys(
      doc.source,
      [
        "kind",
        "snapshot_sha256",
        ...(doc.schema === NATIVE_SCHEMAS.model ? [] : ["model_ir_sha256"]),
      ],
      "native source",
    );
    digest(doc.source.snapshot_sha256);
    need(
      doc.source.kind === "model_state_snapshot",
      "invalid native source kind",
    );
    if (doc.source.model_ir_sha256) digest(doc.source.model_ir_sha256);
  }
  return doc;
}
export function validateNativeBundle(bundle) {
  exactKeys(
    bundle,
    ["snapshot", "model_ir", "weight_ir", "activation_ir"],
    "native bundle",
  );
  for (const name of ["snapshot", "model_ir", "weight_ir", "activation_ir"])
    if (bundle[name]) validateNativeDocument(bundle[name]);
  const {
    snapshot,
    model_ir: model,
    weight_ir: weight,
    activation_ir: activation,
  } = bundle;
  need(snapshot && model && weight, "incomplete native bundle");
  need(
    model.source.snapshot_sha256 === snapshot.snapshot_sha256 &&
      canonicalJson(model.program) === canonicalJson(snapshot.definition) &&
      canonicalJson(model.storage) === canonicalJson(snapshot.tensors),
    "snapshot/model binding mismatch",
  );
  for (const doc of [weight, activation].filter(Boolean))
    need(
      doc.source.snapshot_sha256 === snapshot.snapshot_sha256 &&
        doc.source.model_ir_sha256 === model.model_ir_sha256,
      "numerical source mismatch",
    );
  need(
    canonicalJson(weight.tensors.map((r) => r.identity)) ===
      canonicalJson(snapshot.tensors),
    "weight state identities mismatch",
  );
  if (activation) {
    const nodes = new Set(model.program.nodes.map((n) => n.id));
    need(
      activation.tensors.every((t) => nodes.has(t.subject_ref)),
      "activation references missing node",
    );
  }
  return bundle;
}
export function compareNativeNodes(before, after) {
  const index = rows => {
    need(Array.isArray(rows), "invalid native nodes");
    const map = new Map(rows.map(n => [n.id, n]));
    need(map.size === rows.length, "duplicate native node");
    return map;
  };
  const a = index(before), b = index(after);
  return [...new Set([...a.keys(), ...b.keys()])].sort().map(id => ({
    id, status: !a.has(id) ? "added" : !b.has(id) ? "removed"
      : canonicalJson(a.get(id)) === canonicalJson(b.get(id)) ? "unchanged" : "changed",
  }));
}

// Shared by bundle generation and saved-report ingestion. Correspondence can
// be verified from recorded state identities without inventing missing values.
export function compareNativeTensorIdentities(before, after) {
  const index = rows => {
    need(Array.isArray(rows), "invalid native tensor inventory");
    rows.forEach(validateNativeIdentity);
    const map = new Map(rows.map(t => [t.id, t]));
    need(map.size === rows.length, "duplicate native tensor identity");
    return map;
  };
  const a = index(before), b = index(after);
  return [...new Set([...a.keys(), ...b.keys()])].sort().map((id) => {
    const x = a.get(id),
      y = b.get(id),
      aligned =
        x &&
        y &&
        x.dtype === y.dtype &&
        canonicalJson(x.shape) === canonicalJson(y.shape);
    return {
      id,
      status: !x
        ? "added"
        : !y
          ? "removed"
          : aligned
            ? "aligned_native_id"
            : "shape_or_dtype_changed",
      matching_basis: "native_id_not_semantic_equivalence",
      payload_equal: aligned
        ? x.payload_sha256 === y.payload_sha256
        : null,
    };
  });
}

export function validateNativeTensorComparison(tensors, before, after) {
  need(Array.isArray(tensors), "missing tensor comparison inventory");
  const expected = compareNativeTensorIdentities(before, after);
  const recorded = tensors.map(({ distribution, ...identity }) => identity);
  need(canonicalJson(recorded) === canonicalJson(expected), "tensor comparison disagrees with state identities");
  need(tensors.every(t => t.status === "aligned_native_id" || t.distribution === null), "unaligned tensor has a distribution comparison");
  return tensors;
}

export function compareNativeEvidence(baseline, candidate) {
  for (const set of [baseline, candidate]) validateNativeBundle(set);
  const a = new Map(baseline.weight_ir.tensors.map(t => [t.identity.id, t])),
    b = new Map(candidate.weight_ir.tensors.map(t => [t.identity.id, t]));
  const tensors = compareNativeTensorIdentities(baseline.snapshot.tensors, candidate.snapshot.tensors).map(row => ({
    ...row,
    distribution: row.status === "aligned_native_id" && a.get(row.id).statistics && b.get(row.id).statistics
      ? compareDistributions(a.get(row.id).statistics, b.get(row.id).statistics) : null,
  }));
  const nodes = compareNativeNodes(baseline.model_ir.program.nodes, candidate.model_ir.program.nodes);
  return sealNative(
    {
      schema: "deepbom.native_model_comparison.v1",
      baseline: baseline.snapshot.snapshot_sha256,
      candidate: candidate.snapshot.snapshot_sha256,
      tensors,
      nodes,
      totals: { baseline: nativeStateTotals(baseline.snapshot.tensors), candidate: nativeStateTotals(candidate.snapshot.tensors) },
      structure_equal:
        canonicalJson(baseline.model_ir.program) ===
        canonicalJson(candidate.model_ir.program),
      boundary:
        "Identity and native-name alignment, not quality or numerical equivalence. State bytes are not runtime peak memory.",
    },
    "comparison_sha256",
  );
}

export function nativeStateTotals(tensors) {
  tensors.forEach(validateNativeIdentity);
  return {
    state_elements: String(
      tensors.reduce((n, t) => n + BigInt(t.element_count), 0n),
    ),
    state_bytes: String(
      tensors.reduce((n, t) => n + BigInt(t.byte_length), 0n),
    ),
    trainable_elements: String(
      tensors
        .filter((t) => t.trainable)
        .reduce((n, t) => n + BigInt(t.element_count), 0n),
    ),
  };
}
export function buildTrainingIr(input) {
  exactKeys(
    input,
    ["schema", "run_id", "segment_id", "lifecycle", "events", "previous"],
    "training input",
  );
  need(input.schema === "deepbom.training_input.v1", "unknown training input");
  text(input.run_id, "run id");
  text(input.segment_id, "segment id");
  need(
    ["running", "ended", "failed", "interrupted", "collector_closed"].includes(
      input.lifecycle,
    ),
    "invalid training lifecycle",
  );
  const prior = input.previous;
  if (prior) {
    validateTrainingIr(prior);
    need(
      prior.run_id === input.run_id && prior.segment_id === input.segment_id,
      "training predecessor belongs to another run",
    );
  }
  need(
    Array.isArray(input.events) && input.events.length <= 10000,
    "training chunk exceeds budget",
  );
  const offset = BigInt(prior?.event_count || "0");
  for (const [i, event] of input.events.entries()) {
    exactKeys(
      event,
      ["id", "sequence", "kind", "at", "payload"],
      "training event",
    );
    need(
      event.sequence === String(offset + BigInt(i)) &&
        event.id === `${input.segment_id}:${event.sequence}`,
      "training event order or identity mismatch",
    );
    text(event.kind, "event kind");
    record(event.at, "event coordinates");
    record(event.payload, "event payload");
  }
  const chunk = sealNative(
    {
      schema: "deepbom.training_chunk.v1",
      run_id: input.run_id,
      segment_id: input.segment_id,
      events: input.events,
    },
    "chunk_sha256",
  );
  const chunks = [
    ...(prior?.chunks || []),
    {
      sha256: chunk.chunk_sha256,
      offset: String(offset),
      count: String(input.events.length),
    },
  ];
  const document = sealNative(
    {
      schema: NATIVE_SCHEMAS.training,
      method_version: "1.0.0",
      run_id: input.run_id,
      segment_id: input.segment_id,
      lifecycle: input.lifecycle,
      event_count: String(offset + BigInt(input.events.length)),
      chunks,
      previous_ir_sha256: prior?.training_ir_sha256 || null,
      boundary:
        "Observed training events only. Collection closure is not proof of training completion; missing state binding and metric semantics remain explicit.",
    },
    "training_ir_sha256",
  );
  validateTrainingIr(document);
  return { document, chunk };
}
export function validateTrainingIr(doc) {
  exactKeys(
    doc,
    [
      "schema",
      "method_version",
      "run_id",
      "segment_id",
      "lifecycle",
      "event_count",
      "chunks",
      "previous_ir_sha256",
      "boundary",
      "training_ir_sha256",
    ],
    "Training IR",
  );
  need(
    doc.schema === NATIVE_SCHEMAS.training && doc.method_version === "1.0.0",
    "unsupported Training IR",
  );
  checkSeal(doc, "training_ir_sha256");
  text(doc.run_id, "run id");
  text(doc.segment_id, "segment id");
  need(
    ["running", "ended", "failed", "interrupted", "collector_closed"].includes(
      doc.lifecycle,
    ),
    "invalid training lifecycle",
  );
  need(Array.isArray(doc.chunks), "invalid training chunks");
  text(doc.boundary, "training boundary");
  let count = 0n;
  const hashes = new Set();
  for (const chunk of doc.chunks) {
    exactKeys(chunk, ["sha256", "offset", "count"], "chunk ref");
    digest(chunk.sha256);
    need(!hashes.has(chunk.sha256), "duplicate training chunk");
    hashes.add(chunk.sha256);
    need(
      typeof chunk.count === "string" &&
        /^(0|[1-9][0-9]*)$/.test(chunk.count) &&
        chunk.offset === String(count),
      "invalid training chunk range",
    );
    count += BigInt(chunk.count);
  }
  need(doc.event_count === String(count), "training counts do not conserve");
  if (doc.previous_ir_sha256 !== null) digest(doc.previous_ir_sha256);
  return doc;
}

export function validateTrainingChunk(chunk) {
  exactKeys(chunk,["schema","run_id","segment_id","events","chunk_sha256"],"training chunk");
  need(chunk.schema === "deepbom.training_chunk.v1", "unknown training chunk");
  text(chunk.run_id,"training run");text(chunk.segment_id,"training segment");
  need(Array.isArray(chunk.events),"training events must be an array");
  for(const event of chunk.events){
    exactKeys(event,["id","sequence","kind","at","payload"],"training event");
    text(event.id,"training event ID");text(event.kind,"training event kind");
    need(typeof event.sequence === "string" && /^(0|[1-9][0-9]*)$/.test(event.sequence),"invalid event sequence");
    record(event.at,"event coordinates");record(event.payload,"event payload");
  }
  checkSeal(chunk,"chunk_sha256");return chunk;
}

export function validateTrainingBundle(bundle) {
  exactKeys(bundle, ["training", "events", "objects"], "training bundle");
  validateTrainingIr(bundle.training);
  record(bundle.objects, "training objects");
  const doc = bundle.training,
    events = [],
    pending = new Set(),
    seenTokens = new Set();
  let terminal = null;
  for (const ref of doc.chunks) {
    const chunk = bundle.objects[ref.sha256];
    need(chunk, "missing training chunk");
    validateTrainingChunk(chunk);
    need(
      chunk.schema === "deepbom.training_chunk.v1" &&
        chunk.run_id === doc.run_id &&
        chunk.segment_id === doc.segment_id &&
        chunk.chunk_sha256 === ref.sha256,
      "foreign chunk",
    );
    need(
      Array.isArray(chunk.events) && String(chunk.events.length) === ref.count,
      "chunk size mismatch",
    );
    for (const event of chunk.events) {
      exactKeys(
        event,
        ["id", "sequence", "kind", "at", "payload"],
        "training event",
      );
      need(
        event.sequence === String(events.length) &&
          event.id === `${doc.segment_id}:${event.sequence}`,
        "event continuity mismatch",
      );
      text(event.kind, "event kind");
      record(event.at, "event coordinates");
      record(event.payload, "event payload");
      events.push(event);
      const get = (reference) => {
        exactKeys(reference, ["schema", "sha256"], "evidence reference");
        digest(reference.sha256);
        const obj = bundle.objects[reference.sha256];
        need(
          obj && obj.schema === reference.schema,
          "missing or foreign referenced evidence",
        );
        validateNativeDocument(obj);
        const key = {
          snapshot: "snapshot_sha256",
          model: "model_ir_sha256",
          weight: "weight_ir_sha256",
          activation: "activation_ir_sha256",
        }[
          Object.entries(NATIVE_SCHEMAS).find(([, s]) => s === obj.schema)?.[0]
        ];
        need(obj[key] === reference.sha256, "reference digest mismatch");
        return obj;
      };
      if (terminal)
        need(
          event.kind === "collector_closed",
          "training observation after terminal status",
        );
      if (event.kind === "update_attempt") {
        const p = event.payload;
        text(p.token, "update token");
        need(!seenTokens.has(p.token), "duplicate update token");
        seenTokens.add(p.token);
        pending.add(p.token);
        if (p.gradients) {
          need(Array.isArray(p.gradients), "invalid gradient inventory");
          const indices = new Set();
          for (const g of p.gradients) {
            need(
              Number.isSafeInteger(g.index) &&
                g.index >= 0 &&
                !indices.has(g.index),
              "duplicate gradient index",
            );
            indices.add(g.index);
            text(g.native_state_ref, "gradient state reference");
            need(
              g.snapshot_binding === null,
              "unsupported implicit gradient state binding",
            );
            if (g.status === "missing_gradient") {
              exactKeys(
                g,
                ["index", "native_state_ref", "snapshot_binding", "status"],
                "missing gradient",
              );
              continue;
            }
            exactKeys(
              g,
              [
                "index",
                "native_state_ref",
                "snapshot_binding",
                "status",
                "scope",
                "indices",
                "dense_shape",
                "evidence",
              ],
              "gradient",
            );
            need(g.status === "observed", "invalid gradient status");
            need(
              [
                "dense_tensor",
                "sparse_stored_values_not_dense_gradient",
                "indexed_slices_values_not_dense_gradient",
              ].includes(g.scope),
              "invalid gradient scope",
            );
            need(
              Array.isArray(g.dense_shape) &&
                g.dense_shape.every((d) => Number.isSafeInteger(d) && d >= 0),
              "invalid dense shape",
            );
            exactKeys(
              g.evidence,
              [
                "identity",
                "status",
                "reason",
                "statistics",
                "distribution",
                "features",
              ],
              "gradient evidence",
            );
            need(
              g.evidence.identity.role === "gradient",
              "invalid gradient role",
            );
            validateNativeTensorMetrics(g.evidence);
            if (g.scope === "dense_tensor")
              need(
                g.indices === null &&
                  canonicalJson(g.dense_shape) ===
                    canonicalJson(g.evidence.identity.shape),
                "dense gradient shape mismatch",
              );
            else {
              need(Array.isArray(g.indices), "sparse gradient indices missing");
              const shape = g.evidence.identity.shape,
                nnz = shape[0];
              const validIndex = (x, axis) =>
                Number.isSafeInteger(x) && x >= 0 && x < g.dense_shape[axis];
              if (g.scope === "indexed_slices_values_not_dense_gradient")
                need(
                  g.indices.length === nnz &&
                    g.indices.every((x) => validIndex(x, 0)) &&
                    canonicalJson(shape.slice(1)) ===
                      canonicalJson(g.dense_shape.slice(1)),
                  "indexed slices shape/index mismatch",
                );
              else
                need(
                  g.indices.length > 0 &&
                    g.indices.length <= g.dense_shape.length &&
                    g.indices.every(
                      (row, axis) =>
                        Array.isArray(row) &&
                        row.length === nnz &&
                        row.every((x) => validIndex(x, axis)),
                    ) &&
                    canonicalJson(shape.slice(1)) ===
                      canonicalJson(g.dense_shape.slice(g.indices.length)),
                  "sparse gradient shape/index mismatch",
                );
            }
          }
        }
      } else if (event.kind === "update_returned") {
        need(
          pending.has(event.payload.token),
          "unmatched or reused update return",
        );
        pending.delete(event.payload.token);
        need(
          [
            "returned_effect_unverified",
            "skipped_declared",
            "failed_declared",
          ].includes(event.payload.outcome),
          "invalid update outcome",
        );
      } else if (event.kind === "training_status") {
        terminal = event.payload.state;
        need(
          ["ended", "failed", "interrupted"].includes(terminal) &&
            doc.lifecycle === terminal,
          "training lifecycle contradiction",
        );
      } else if (event.kind === "collector_closed") {
        need(
          canonicalJson(event.payload.unfinished_update_tokens) ===
            canonicalJson([...pending]),
          "unfinished update token ledger mismatch",
        );
      } else if (event.kind === "model_snapshot") {
        const refs = event.payload.evidence;
        exactKeys(
          refs,
          ["snapshot", "model_ir", "weight_ir"],
          "snapshot event references",
        );
        validateNativeBundle({
          ...Object.fromEntries(
            Object.entries(refs).map(([k, r]) => [k, get(r)]),
          ),
          activation_ir: null,
        });
      } else if (event.kind === "activation_capture") {
        const a = get(event.payload.evidence);
        need(
          a.schema === NATIVE_SCHEMAS.activation &&
            a.source.snapshot_sha256 === event.payload.snapshot_sha256,
          "activation event binding mismatch",
        );
        need(canonicalJson(a.context.at ?? {}) === canonicalJson(event.at), "activation event coordinates mismatch");
        const model = bundle.objects[a.source.model_ir_sha256];
        need(
          model && model.source.snapshot_sha256 === a.source.snapshot_sha256,
          "activation model missing",
        );
        need(
          a.tensors.every((t) =>
            model.program.nodes.some((n) => n.id === t.subject_ref),
          ),
          "activation subject missing",
        );
      }
    }
  }
  need(
    String(events.length) === doc.event_count &&
      canonicalJson(events) === canonicalJson(bundle.events),
    "training event aggregate mismatch",
  );
  return bundle;
}

export function selectTrainingResult(input) {
  exactKeys(input, ["bundle", "snapshot_sha256"], "training result selection");
  validateTrainingBundle(input.bundle);
  digest(input.snapshot_sha256);
  const { training, events, objects } = input.bundle;
  const matches = events.filter(
    (e) =>
      e.kind === "model_snapshot" &&
      e.payload.evidence.snapshot.sha256 === input.snapshot_sha256,
  );
  need(
    matches.length,
    "selected state was not recorded in this training snapshot",
  );
  const event = matches.at(-1);
  return sealNative(
    {
      schema: "deepbom.training_result_manifest.v1",
      method_version: "1.0.0",
      training_ir_sha256: training.training_ir_sha256,
      snapshot_sha256: input.snapshot_sha256,
      selected_event_id: event.id,
      selection_rule: "latest_recorded_capture_of_requested_state",
      evidence: event.payload.evidence,
      training_lifecycle: training.lifecycle,
      boundary:
        "Selected model evidence, not training completion, optimizer restoration, quality acceptance or deployment approval.",
    },
    "training_result_manifest_sha256",
  );
}
