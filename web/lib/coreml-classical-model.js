import { Sha256Accumulator } from "./sha256-sync.js";

const MAX_ROWS = 1_000_000;
const MAX_VALUES = 20_000_000;
const MAX_PIPELINE_MODELS = 4096;

export const COREML_CLASSICAL_SOURCE = Object.freeze({
  repository: "apple/coremltools",
  release: "9.0",
  source_commit: "428d4b2658dfc44194f27f4f36870751be402ff7",
  files: Object.freeze([
    ["mlmodel/format/DataStructures.proto", "4d70f805399523654346e06de97777febc5a06910784513d8cdc952832bcf7e8"],
    ["mlmodel/format/GLMRegressor.proto", "bd3b678e1620e9a739bfd23df9726d6ac59e716ec65507315eed188311993df7"],
    ["mlmodel/format/GLMClassifier.proto", "f232a040af8c1f78813688892fcc5c22cc1f5993829d721c2bdd0bdf33bf630b"],
    ["mlmodel/format/SVM.proto", "ed552d080f46eb1e5f0ba1bcfab2b8c5e4efc643fdb2b2a991317efcdec98fd6"],
    ["mlmodel/format/TreeEnsemble.proto", "346c8e862048beb693e847727f7f307efd9796123951575d9ea44d47262c5641"],
    ["mlmodel/src/Validation/LinearModelValidator.cpp", "da6ebcad55e1b5ea17fd973a7c62cd681f038fc61732df0dc6c778dcb928a7cc"],
    ["mlmodel/src/Validation/SVMValidator.cpp", "d26bdaec6ad1dca079c31fa120f739dfa6f09560f6f3e2f888828009da9b7c92"],
    ["mlmodel/src/Validation/TreeEnsembleValidator.cpp", "6c6a8461106e83d6e66d34bfeab4785066414173e11ccb44d68bb7be203cd2aa"],
    ["mlmodel/src/TreeEnsembleCommon.cpp", "a9c430465ede077f9954c820ce9dd824b22db5680a6793fb178ba440ec96dbca"],
    ["mlmodel/src/Validation/PipelineValidator.cpp", "e18a340d71f8b21d87aa0a41fde8e58bb86bb13fb7fae4d117ddfe0c1f9bc2dc"],
  ].map(([path, sha256]) => Object.freeze({ path, sha256 }))),
  interpretation_boundary: "The parser reproduces serialized cardinality and structural invariants from the pinned protobuf and validator sources. It does not claim Core ML runtime scheduling, floating-point reduction order, or task accuracy.",
});

export const COREML_CLASSICAL_FIELDS = new Set([300, 301, 302, 400, 401, 402]);
export const COREML_PIPELINE_FIELDS = new Set([200, 201, 202]);

function boundedPush(rows, value, label, limit = MAX_ROWS) {
  if (rows.length >= limit) throw new Error(`${label} exceeds ${limit} entries`);
  rows.push(value);
}

function readText(reader, wire, label) {
  const bytes = reader.bytesField(wire, label);
  try { return new TextDecoder("utf-8", { fatal: true }).decode(bytes); }
  catch { throw new Error(`${label} is not valid UTF-8`); }
}

function uint64(reader, wire, singular, field, label) {
  reader.requireWire(wire, 0, label);
  if (singular?.has(field)) throw new Error(`${label} is repeated`);
  singular?.add(field);
  const value = reader.rawVarint();
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${label} exceeds the safe integer range`);
  return Number(value);
}

function repeatedInt32(reader, wire, label) {
  const values = [];
  const read = (source) => {
    const raw = source.rawVarint();
    const signed = BigInt.asIntN(32, raw);
    if (signed < -2147483648n || signed > 2147483647n) throw new Error(`${label} contains an invalid int32`);
    values.push(Number(signed));
  };
  if (wire === 0) read(reader);
  else if (wire === 2) {
    const packed = reader.message(wire, label);
    while (!packed.done) read(packed);
  } else throw new Error(`${label} has unsupported wire type ${wire}`);
  return values;
}

function repeatedInt64Strings(reader, wire, label) {
  const values = [];
  const read = (source) => values.push(BigInt.asIntN(64, source.rawVarint()).toString());
  if (wire === 0) read(reader);
  else if (wire === 2) {
    const packed = reader.message(wire, label);
    while (!packed.done) read(packed);
  } else throw new Error(`${label} has unsupported wire type ${wire}`);
  return values;
}

function emptyStats() {
  return { count: 0, finite_count: 0, zero_count: 0, negative_zero_count: 0, nan_count: 0, positive_infinity_count: 0, negative_infinity_count: 0, min: null, max: null };
}

function addNumber(stats, value) {
  if (++stats.count > MAX_VALUES) throw new Error(`Core ML numerical payload exceeds ${MAX_VALUES} values`);
  if (Number.isNaN(value)) { stats.nan_count += 1; return; }
  if (value === Number.POSITIVE_INFINITY) { stats.positive_infinity_count += 1; return; }
  if (value === Number.NEGATIVE_INFINITY) { stats.negative_infinity_count += 1; return; }
  stats.finite_count += 1;
  if (value === 0) { stats.zero_count += 1; if (Object.is(value, -0)) stats.negative_zero_count += 1; }
  stats.min = stats.min == null ? value : Math.min(stats.min, value);
  stats.max = stats.max == null ? value : Math.max(stats.max, value);
}

class DoubleLedger {
  constructor(role) { this.role = role; this.stats = emptyStats(); this.digest = new Sha256Accumulator(); }
  read(reader, wire, label, retain = null) {
    const consume = (bytes) => {
      if (bytes.length % 8 !== 0) throw new Error(`${label} packed double payload is not 8-byte aligned`);
      this.digest.update(bytes);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (let offset = 0; offset < bytes.length; offset += 8) {
        const value = view.getFloat64(offset, true);
        addNumber(this.stats, value);
        retain?.push(value);
      }
    };
    if (wire === 2) consume(reader.bytesField(wire, label));
    else if (wire === 1) {
      const start = reader.position;
      reader.advance(8);
      consume(reader.bytes.subarray(start, start + 8));
    } else throw new Error(`${label} has unsupported wire type ${wire}`);
  }
  parameter(extra = {}) {
    const stats = this.stats;
    const nonfinite = stats.nan_count + stats.positive_infinity_count + stats.negative_infinity_count;
    return {
      role: this.role,
      storage: "float64",
      byte_length: stats.count * 8,
      value_count: stats.count,
      numerical_integrity: {
        schema: "deepbom.coreml.classical_numerical_integrity.v1",
        status: "assessed",
        payload_sha256: stats.count ? this.digest.digestHex() : null,
        decoded_value_count: stats.count,
        finite_count: stats.finite_count,
        zero_count: stats.zero_count,
        negative_zero_count: stats.negative_zero_count,
        nan_count: stats.nan_count,
        positive_infinity_count: stats.positive_infinity_count,
        negative_infinity_count: stats.negative_infinity_count,
        nonfinite_count: nonfinite,
        finite_min: stats.min,
        finite_max: stats.max,
        all_zero: stats.count > 0 && stats.zero_count === stats.count,
        constant: stats.count > 0 && nonfinite === 0 && stats.min === stats.max,
      },
      ...extra,
    };
  }
}

function readFiniteDouble(reader, wire, singular, field, label, ledger) {
  if (singular.has(field)) throw new Error(`${label} is repeated`);
  singular.add(field);
  const values = [];
  ledger.read(reader, wire, label, values);
  if (values.length !== 1 || !Number.isFinite(values[0])) throw new Error(`${label} must contain one finite double`);
  return values[0];
}

function parseStringVector(reader, label) {
  const values = [];
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) boundedPush(values, readText(reader, wire, label), label);
    else reader.skip(wire);
  }
  return values;
}

function parseInt64Vector(reader, label) {
  const values = [];
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) {
      for (const value of repeatedInt64Strings(reader, wire, label)) boundedPush(values, value, label);
    } else reader.skip(wire);
  }
  return values;
}

function parseLabels(reader, field, wire, current, label) {
  if (current) throw new Error(`${label} contains multiple class-label encodings`);
  const values = field === 100
    ? parseStringVector(reader.message(wire, `${label}.stringClassLabels`), `${label} string labels`)
    : parseInt64Vector(reader.message(wire, `${label}.int64ClassLabels`), `${label} int64 labels`);
  if (!values.length) throw new Error(`${label} class labels are empty`);
  return { kind: field === 100 ? "string" : "int64", values };
}

function parseDoubleArray(reader, role) {
  const ledger = new DoubleLedger(role);
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) ledger.read(reader, wire, `${role}.value`);
    else reader.skip(wire);
  }
  return ledger.parameter();
}

function validateEqualWidths(parameters, label, requireNonempty) {
  if (!parameters.length) {
    if (requireNonempty) throw new Error(`${label} has no coefficient rows`);
    return 0;
  }
  const width = parameters[0].value_count;
  if (requireNonempty && width === 0) throw new Error(`${label} coefficient rows are empty`);
  if (parameters.some((row) => row.value_count !== width)) throw new Error(`${label} coefficient rows have inconsistent widths`);
  return width;
}

function parseGlm(reader, classifier) {
  const weights = [];
  const offsets = new DoubleLedger("offsets");
  const singular = new Set();
  let transform = 0;
  let encoding = 0;
  let labels = null;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) boundedPush(weights, parseDoubleArray(reader.message(wire, "CoreML.GLM.weights"), `weights_${weights.length}`), "Core ML GLM weights");
    else if (field === 2) offsets.read(reader, wire, "CoreML.GLM.offset");
    else if (field === 3) transform = reader.intField(wire, singular, field, "postEvaluationTransform");
    else if (classifier && field === 4) encoding = reader.intField(wire, singular, field, "classEncoding");
    else if (classifier && (field === 100 || field === 101)) labels = parseLabels(reader, field, wire, labels, "Core ML GLM classifier");
    else reader.skip(wire);
  }
  const width = validateEqualWidths(weights, "Core ML GLM", classifier);
  const offsetParameter = offsets.parameter();
  if (weights.length !== offsetParameter.value_count) throw new Error("Core ML GLM weights and offsets must have equal cardinality");
  if (classifier) {
    if (!labels) throw new Error("Core ML GLM classifier is missing class labels");
    if (![0, 1].includes(transform) || ![0, 1].includes(encoding)) throw new Error("Core ML GLM classifier has an invalid transform or class encoding");
    const classes = labels.values.length;
    const expected = encoding === 0 ? classes - 1 : classes === 2 ? 1 : classes;
    if (weights.length !== expected) throw new Error(`Core ML GLM classifier coefficient rows ${weights.length} do not match class encoding expectation ${expected}`);
    if (transform === 1 && classes > 2) throw new Error("Core ML GLM probit transform is only valid for binary classification");
  } else if (![0, 1, 2].includes(transform)) throw new Error("Core ML GLM regressor has an invalid post-evaluation transform");
  return {
    schema: "deepbom.coreml.classical_model.v1",
    kind: classifier ? "glmClassifier" : "glmRegressor",
    coefficient_row_count: weights.length,
    coefficient_width: width,
    offset_count: offsetParameter.value_count,
    post_evaluation_transform: transform,
    class_encoding: classifier ? encoding : null,
    class_labels: labels,
    parameters: [...weights, offsetParameter],
    source_validation: "pinned_LinearModelValidator",
  };
}

function parseKernel(reader, parameterLedgers) {
  let kernel = null;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (![1, 2, 3, 4].includes(field)) { reader.skip(wire); continue; }
    if (kernel) throw new Error("Core ML SVM contains multiple kernel definitions");
    const payload = reader.message(wire, `CoreML.SVM.Kernel.${field}`);
    const singular = new Set();
    const ledger = new DoubleLedger(`kernel_${field}_parameters`);
    if (field === 1) {
      while (!payload.done) { const item = payload.key(); payload.skip(item.wire); }
      kernel = { kind: "linear" };
    } else if (field === 2) {
      let gamma = 0;
      while (!payload.done) {
        const item = payload.key();
        if (item.field === 1) gamma = readFiniteDouble(payload, item.wire, singular, 1, "Core ML RBF gamma", ledger);
        else payload.skip(item.wire);
      }
      if (gamma < 0) throw new Error("Core ML RBF gamma must be non-negative");
      kernel = { kind: "rbf", gamma };
    } else if (field === 3) {
      let degree = 0; let c = 0; let gamma = 0;
      while (!payload.done) {
        const item = payload.key();
        if (item.field === 1) degree = payload.intField(item.wire, singular, 1, "degree");
        else if (item.field === 2) c = readFiniteDouble(payload, item.wire, singular, 2, "Core ML polynomial c", ledger);
        else if (item.field === 3) gamma = readFiniteDouble(payload, item.wire, singular, 3, "Core ML polynomial gamma", ledger);
        else payload.skip(item.wire);
      }
      if (gamma < 0) throw new Error("Core ML polynomial gamma must be non-negative");
      kernel = { kind: "polynomial", degree, c, gamma };
    } else {
      let gamma = 0; let c = 0;
      while (!payload.done) {
        const item = payload.key();
        if (item.field === 1) gamma = readFiniteDouble(payload, item.wire, singular, 1, "Core ML sigmoid gamma", ledger);
        else if (item.field === 2) c = readFiniteDouble(payload, item.wire, singular, 2, "Core ML sigmoid c", ledger);
        else payload.skip(item.wire);
      }
      if (gamma < 0) throw new Error("Core ML sigmoid gamma must be non-negative");
      kernel = { kind: "sigmoid", gamma, c };
    }
    const parameter = ledger.parameter();
    if (parameter.value_count) parameterLedgers.push(parameter);
  }
  if (!kernel) throw new Error("Core ML SVM is missing a supported kernel");
  return kernel;
}

function parseDenseSupportVectors(reader, ledger) {
  const widths = [];
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field !== 1) { reader.skip(wire); continue; }
    const vector = reader.message(wire, "CoreML.DenseVector");
    const before = ledger.stats.count;
    while (!vector.done) {
      const item = vector.key();
      if (item.field === 1) ledger.read(vector, item.wire, "CoreML.DenseVector.values");
      else vector.skip(item.wire);
    }
    boundedPush(widths, ledger.stats.count - before, "Core ML dense support vectors");
  }
  const width = widths[0] || 0;
  if (!widths.length || width === 0 || widths.some((value) => value !== width)) throw new Error("Core ML dense support vectors must be nonempty and have equal widths");
  return { kind: "dense", count: widths.length, width, nonzero_entry_count: null };
}

function parseSparseSupportVectors(reader, ledger) {
  const widths = [];
  let entries = 0;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field !== 1) { reader.skip(wire); continue; }
    const vector = reader.message(wire, "CoreML.SparseVector");
    const indices = new Set();
    let maximum = 0;
    while (!vector.done) {
      const item = vector.key();
      if (item.field !== 1) { vector.skip(item.wire); continue; }
      const node = vector.message(item.wire, "CoreML.SparseNode");
      const singular = new Set();
      let index = 0; let valueSeen = false;
      while (!node.done) {
        const entry = node.key();
        if (entry.field === 1) index = node.intField(entry.wire, singular, 1, "index");
        else if (entry.field === 2) { readFiniteDouble(node, entry.wire, singular, 2, "Core ML sparse node value", ledger); valueSeen = true; }
        else node.skip(entry.wire);
      }
      if (index <= 0 || indices.has(index) || !valueSeen) throw new Error("Core ML sparse support vector has a missing, zero, or duplicate 1-based index");
      indices.add(index); maximum = Math.max(maximum, index); entries += 1;
    }
    boundedPush(widths, maximum, "Core ML sparse support vectors");
  }
  if (!widths.length) throw new Error("Core ML sparse support vectors are empty");
  return { kind: "sparse", count: widths.length, width: Math.max(...widths), vector_max_indices: widths, nonzero_entry_count: entries };
}

function parseCoefficients(reader, ledger) {
  const before = ledger.stats.count;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) ledger.read(reader, wire, "CoreML.SVM.coefficients.alpha");
    else reader.skip(wire);
  }
  return ledger.stats.count - before;
}

function parseSvm(reader, classifier) {
  const parameters = [];
  const supportLedger = new DoubleLedger("support_vectors");
  const coefficientLedger = new DoubleLedger("coefficients");
  const rhoLedger = new DoubleLedger("rho");
  const probALedger = new DoubleLedger("probA");
  const probBLedger = new DoubleLedger("probB");
  const singular = new Set();
  let kernel = null; let support = null; let labels = null;
  const perClass = [];
  const coefficientRows = [];
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) {
      if (kernel) throw new Error("Core ML SVM repeats Kernel");
      kernel = parseKernel(reader.message(wire, "CoreML.SVM.Kernel"), parameters);
    } else if (classifier && field === 2) {
      for (const value of repeatedInt32(reader, wire, "CoreML.SVM.numberOfSupportVectorsPerClass")) boundedPush(perClass, value, "Core ML SVM class support counts");
    } else if ((!classifier && (field === 2 || field === 3)) || classifier && (field === 3 || field === 4)) {
      if (support) throw new Error("Core ML SVM contains multiple support-vector encodings");
      const sparse = classifier ? field === 3 : field === 2;
      support = sparse
        ? parseSparseSupportVectors(reader.message(wire, "CoreML.SparseSupportVectors"), supportLedger)
        : parseDenseSupportVectors(reader.message(wire, "CoreML.DenseSupportVectors"), supportLedger);
    } else if (!classifier && field === 4) {
      if (coefficientRows.length) throw new Error("Core ML SVM regressor repeats coefficients");
      coefficientRows.push(parseCoefficients(reader.message(wire, "CoreML.SVM.Coefficients"), coefficientLedger));
    } else if (classifier && field === 5) {
      boundedPush(coefficientRows, parseCoefficients(reader.message(wire, "CoreML.SVM.Coefficients"), coefficientLedger), "Core ML SVM coefficient rows");
    } else if ((!classifier && field === 5) || classifier && field === 6) rhoLedger.read(reader, wire, "CoreML.SVM.rho");
    else if (classifier && field === 7) probALedger.read(reader, wire, "CoreML.SVM.probA");
    else if (classifier && field === 8) probBLedger.read(reader, wire, "CoreML.SVM.probB");
    else if (classifier && (field === 100 || field === 101)) labels = parseLabels(reader, field, wire, labels, "Core ML SVM classifier");
    else reader.skip(wire);
  }
  if (!kernel || !support) throw new Error("Core ML SVM is missing kernel or support vectors");
  if (coefficientRows.some((value) => value !== support.count)) throw new Error("Core ML SVM coefficient width does not match support-vector count");
  if (classifier) {
    if (!labels || labels.values.length < 2) throw new Error("Core ML SVM classifier requires at least two class labels");
    const classes = labels.values.length;
    if (coefficientRows.length !== classes - 1) throw new Error("Core ML SVM coefficient row count must equal class count minus one");
    if (perClass.length !== classes || perClass.some((value) => value < 0)
      || perClass.reduce((sum, value) => sum + value, 0) !== support.count) throw new Error("Core ML SVM per-class support-vector counts are inconsistent");
    const pairs = classes * (classes - 1) / 2;
    if (rhoLedger.stats.count !== pairs) throw new Error("Core ML SVM rho cardinality does not match class-pair count");
    if (probALedger.stats.count || probBLedger.stats.count) {
      if (probALedger.stats.count !== pairs || probBLedger.stats.count !== pairs) throw new Error("Core ML SVM probability calibration cardinality does not match class-pair count");
    }
  } else if (coefficientRows.length !== 1 || rhoLedger.stats.count !== 1) {
    throw new Error("Core ML SVM regressor requires one coefficient vector and one rho value");
  }
  for (const ledger of [supportLedger, coefficientLedger, rhoLedger, probALedger, probBLedger]) {
    const parameter = ledger.parameter();
    if (parameter.value_count) parameters.push(parameter);
  }
  return {
    schema: "deepbom.coreml.classical_model.v1",
    kind: classifier ? "supportVectorClassifier" : "supportVectorRegressor",
    kernel,
    support_vectors: support,
    coefficient_row_count: coefficientRows.length,
    coefficient_count: coefficientRows.reduce((sum, value) => sum + value, 0),
    rho_count: rhoLedger.stats.count,
    probability_pair_count: probALedger.stats.count,
    class_labels: labels,
    support_vectors_per_class: classifier ? perClass : null,
    parameters,
    source_validation: "pinned_SVMValidator",
  };
}

function parseEvaluationInfo(reader, ledger) {
  const singular = new Set();
  let index = 0; let value = 0; let valueSeen = false;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) index = uint64(reader, wire, singular, 1, "Core ML tree evaluationIndex");
    else if (field === 2) { value = readFiniteDouble(reader, wire, singular, 2, "Core ML tree evaluationValue", ledger); valueSeen = true; }
    else reader.skip(wire);
  }
  if (!valueSeen) throw new Error("Core ML tree leaf evaluation is missing its value");
  return { index, value };
}

function parseTreeNode(reader, ledgers) {
  const singular = new Set();
  const node = { tree_id: 0, node_id: 0, behavior: 0, feature_index: 0, branch_value: 0, true_child: 0, false_child: 0, missing_tracks_true: false, evaluations: [], relative_hit_rate: 0 };
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) node.tree_id = uint64(reader, wire, singular, field, "Core ML treeId");
    else if (field === 2) node.node_id = uint64(reader, wire, singular, field, "Core ML nodeId");
    else if (field === 3) node.behavior = reader.intField(wire, singular, field, "nodeBehavior");
    else if (field === 10) node.feature_index = uint64(reader, wire, singular, field, "Core ML branchFeatureIndex");
    else if (field === 11) node.branch_value = readFiniteDouble(reader, wire, singular, field, "Core ML branchFeatureValue", ledgers.branch);
    else if (field === 12) node.true_child = uint64(reader, wire, singular, field, "Core ML trueChildNodeId");
    else if (field === 13) node.false_child = uint64(reader, wire, singular, field, "Core ML falseChildNodeId");
    else if (field === 14) {
      const value = reader.intField(wire, singular, field, "missingValueTracksTrueChild");
      if (value > 1) throw new Error("Core ML missingValueTracksTrueChild is not a bool");
      node.missing_tracks_true = value !== 0;
    } else if (field === 20) boundedPush(node.evaluations, parseEvaluationInfo(reader.message(wire, "CoreML.Tree.EvaluationInfo"), ledgers.evaluation), "Core ML tree leaf evaluations");
    else if (field === 30) node.relative_hit_rate = readFiniteDouble(reader, wire, singular, field, "Core ML relativeHitRate", ledgers.hitRate);
    else reader.skip(wire);
  }
  if (node.behavior > 6) throw new Error("Core ML tree node has an unknown behavior");
  return node;
}

function validateTrees(nodes, dimensions) {
  if (!nodes.length || !Number.isSafeInteger(dimensions) || dimensions <= 0) throw new Error("Core ML tree ensemble requires nodes and a positive prediction dimension");
  const trees = new Map();
  for (const node of nodes) {
    if (!trees.has(node.tree_id)) trees.set(node.tree_id, new Map());
    const tree = trees.get(node.tree_id);
    if (tree.has(node.node_id)) throw new Error(`Core ML tree ${node.tree_id} repeats node ${node.node_id}`);
    tree.set(node.node_id, node);
    if (node.behavior === 6) {
      if (!node.evaluations.length || node.evaluations.some((entry) => entry.index >= dimensions)) throw new Error("Core ML tree leaf evaluation index is missing or outside prediction dimensions");
      const indices = new Set(node.evaluations.map((entry) => entry.index));
      if (indices.size !== node.evaluations.length) throw new Error("Core ML tree leaf repeats an evaluation index");
    } else if (node.evaluations.length) throw new Error("Core ML branch node contains leaf evaluation values");
  }
  let leafCount = 0; let branchCount = 0; let maxDepth = 0;
  for (const [treeId, tree] of trees) {
    const parents = new Map();
    for (const node of tree.values()) {
      if (node.behavior === 6) { leafCount += 1; continue; }
      branchCount += 1;
      for (const childId of [node.true_child, node.false_child]) {
        if (!tree.has(childId)) throw new Error(`Core ML tree ${treeId} branch ${node.node_id} references missing child ${childId}`);
        if (parents.has(childId)) throw new Error(`Core ML tree ${treeId} node ${childId} has multiple parents`);
        parents.set(childId, node.node_id);
      }
    }
    const roots = [...tree.keys()].filter((id) => !parents.has(id));
    if (roots.length !== 1) throw new Error(`Core ML tree ${treeId} must contain exactly one root`);
    const visited = new Set();
    const active = new Set();
    const walk = (id, depth) => {
      if (active.has(id)) throw new Error(`Core ML tree ${treeId} contains a cycle`);
      if (visited.has(id)) return;
      active.add(id); visited.add(id); maxDepth = Math.max(maxDepth, depth);
      const node = tree.get(id);
      if (node.behavior !== 6) { walk(node.true_child, depth + 1); walk(node.false_child, depth + 1); }
      active.delete(id);
    };
    walk(roots[0], 0);
    if (visited.size !== tree.size) throw new Error(`Core ML tree ${treeId} contains unreachable nodes`);
  }
  return { tree_count: trees.size, branch_node_count: branchCount, leaf_node_count: leafCount, maximum_depth: maxDepth };
}

function parseTreeParameters(reader) {
  const ledgers = { branch: new DoubleLedger("branch_values"), evaluation: new DoubleLedger("evaluation_values"), hitRate: new DoubleLedger("relative_hit_rates"), base: new DoubleLedger("base_prediction_values") };
  const nodes = [];
  const singular = new Set();
  let dimensions = 0;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) boundedPush(nodes, parseTreeNode(reader.message(wire, "CoreML.TreeNode"), ledgers), "Core ML tree nodes");
    else if (field === 2) dimensions = uint64(reader, wire, singular, field, "Core ML numPredictionDimensions");
    else if (field === 3) ledgers.base.read(reader, wire, "CoreML.Tree.basePredictionValue");
    else reader.skip(wire);
  }
  const structure = validateTrees(nodes, dimensions);
  if (ledgers.base.stats.count && ledgers.base.stats.count !== dimensions) throw new Error("Core ML tree base prediction cardinality does not match prediction dimensions");
  const parameters = Object.values(ledgers).map((ledger) => ledger.parameter()).filter((row) => row.value_count);
  return { dimensions, nodes, structure, parameters, base_prediction_count: ledgers.base.stats.count };
}

function parseTree(reader, classifier) {
  const singular = new Set();
  let ensemble = null; let transform = 0; let labels = null;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) {
      if (ensemble) throw new Error("Core ML tree ensemble repeats treeEnsemble");
      ensemble = parseTreeParameters(reader.message(wire, "CoreML.TreeEnsembleParameters"));
    } else if (field === 2) transform = reader.intField(wire, singular, field, "postEvaluationTransform");
    else if (classifier && (field === 100 || field === 101)) labels = parseLabels(reader, field, wire, labels, "Core ML tree classifier");
    else reader.skip(wire);
  }
  if (!ensemble || transform > 3) throw new Error("Core ML tree ensemble is missing parameters or has an invalid transform");
  if (classifier && labels) {
    const count = labels.values.length;
    if (!((ensemble.dimensions === 1 && count === 2) || (ensemble.dimensions >= 2 && count === ensemble.dimensions))) {
      throw new Error("Core ML tree classifier labels do not match prediction dimensions");
    }
  }
  return {
    schema: "deepbom.coreml.classical_model.v1",
    kind: classifier ? "treeEnsembleClassifier" : "treeEnsembleRegressor",
    prediction_dimension_count: ensemble.dimensions,
    post_evaluation_transform: transform,
    class_labels: labels,
    ...ensemble.structure,
    base_prediction_count: ensemble.base_prediction_count,
    nodes: ensemble.nodes,
    parameters: ensemble.parameters,
    source_validation: "pinned_TreeEnsembleValidator_and_TreeEnsembleCommon",
  };
}

export function parseCoreMlClassicalModel(field, reader) {
  if (field === 300) return parseGlm(reader, false);
  if (field === 400) return parseGlm(reader, true);
  if (field === 301) return parseSvm(reader, false);
  if (field === 401) return parseSvm(reader, true);
  if (field === 302) return parseTree(reader, false);
  if (field === 402) return parseTree(reader, true);
  throw new Error(`Core ML classical model field ${field} is not supported`);
}

function parsePipelineMessage(reader, parseNestedModel, depth) {
  const models = [];
  const names = [];
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) {
      if (models.length >= MAX_PIPELINE_MODELS) throw new Error(`Core ML pipeline exceeds ${MAX_PIPELINE_MODELS} nested models`);
      models.push(parseNestedModel(reader.bytesField(wire, "CoreML.Pipeline.models"), depth + 1));
    } else if (field === 2) boundedPush(names, readText(reader, wire, "CoreML.Pipeline.names"), "Core ML pipeline names", MAX_PIPELINE_MODELS);
    else reader.skip(wire);
  }
  if (!models.length) throw new Error("Core ML pipeline must contain at least one model");
  if (names.length && names.length !== models.length) throw new Error("Core ML pipeline model-name count does not match model count");
  if (new Set(names).size !== names.length) throw new Error("Core ML pipeline model names are not unique");
  return { schema: "deepbom.coreml.pipeline.v1", models, names, source_validation: "pinned_PipelineValidator" };
}

export function parseCoreMlPipeline(field, reader, parseNestedModel, depth) {
  if (field === 202) return parsePipelineMessage(reader, parseNestedModel, depth);
  let pipeline = null;
  while (!reader.done) {
    const { field: nestedField, wire } = reader.key();
    if (nestedField === 1) {
      if (pipeline) throw new Error("Core ML pipeline classifier/regressor repeats Pipeline");
      pipeline = parsePipelineMessage(reader.message(wire, "CoreML.Pipeline"), parseNestedModel, depth);
    } else reader.skip(wire);
  }
  if (!pipeline) throw new Error("Core ML pipeline classifier/regressor is missing Pipeline");
  return pipeline;
}
