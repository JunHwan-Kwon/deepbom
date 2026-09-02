import { isCanonicalEvidenceClass } from "./evidence-class.js";

export const NODE_EDGE_EVIDENCE_OVERLAY_SCHEMA = "deepbom.node_edge_evidence_overlay.v1";
const MAX_ROWS = 100000;
const MAX_METRICS = 64;

export function buildNodeEdgeEvidenceOverlayTemplate(analysis) {
  const firstOp = analysis?.ops?.[0];
  const firstEdge = serializedEdges(analysis)[0];
  return {
    schema: NODE_EDGE_EVIDENCE_OVERLAY_SCHEMA,
    artifact_sha256: String(analysis?.model_sha256 || ""),
    source: {
      label: "external evidence source",
      sha256: null,
      collected_at: null,
    },
    nodes: firstOp ? [{
      op_index: Number(firstOp.index),
      metrics: [{ key: "latency", value: null, unit: "us", evidence_class: "NOT_ASSESSABLE", source_pointer: null }],
    }] : [],
    edges: firstEdge ? [{
      producer_op_index: firstEdge.producer_op_index,
      consumer_op_index: firstEdge.consumer_op_index,
      tensor_index: firstEdge.tensor_index,
      metrics: [{ key: "transfer", value: null, unit: "bytes", evidence_class: "NOT_ASSESSABLE", source_pointer: null }],
    }] : [],
    interpretation_boundary: "JSON data only. DEEPBOM does not execute overlay code. Every node and edge identity must exist in the hash-bound source graph.",
  };
}

export function validateNodeEdgeEvidenceOverlay(value, analysis) {
  if (!plainObject(value) || value.schema !== NODE_EDGE_EVIDENCE_OVERLAY_SCHEMA) throw new Error("Node/edge evidence overlay schema is invalid.");
  const artifactSha = sha(value.artifact_sha256, "artifact_sha256");
  const expectedSha = sha(analysis?.model_sha256, "analyzed artifact SHA-256");
  if (artifactSha !== expectedSha) throw new Error("Node/edge evidence overlay is bound to a different artifact SHA-256.");
  const ops = new Set((analysis?.ops || []).map((op) => Number(op.index)).filter(Number.isSafeInteger));
  const graphEdges = new Set(serializedEdges(analysis).map(edgeKey));
  const nodes = array(value.nodes, "nodes");
  const edges = array(value.edges, "edges");
  if (nodes.length + edges.length > MAX_ROWS) throw new Error("Node/edge evidence overlay exceeds the bounded row limit.");
  const nodeKeys = new Set();
  const normalizedNodes = nodes.map((row, index) => {
    if (!plainObject(row)) throw new Error(`nodes[${index}] must be an object.`);
    const opIndex = exactIndex(row.op_index, `nodes[${index}].op_index`);
    if (!ops.has(opIndex)) throw new Error(`nodes[${index}] references an unknown op_index.`);
    if (nodeKeys.has(opIndex)) throw new Error(`nodes[${index}] duplicates op_index ${opIndex}.`);
    nodeKeys.add(opIndex);
    return { op_index: opIndex, metrics: metrics(row.metrics, `nodes[${index}].metrics`) };
  });
  const edgeKeys = new Set();
  const normalizedEdges = edges.map((row, index) => {
    if (!plainObject(row)) throw new Error(`edges[${index}] must be an object.`);
    const normalized = {
      producer_op_index: exactIndex(row.producer_op_index, `edges[${index}].producer_op_index`),
      consumer_op_index: exactIndex(row.consumer_op_index, `edges[${index}].consumer_op_index`),
      tensor_index: exactIndex(row.tensor_index, `edges[${index}].tensor_index`),
      metrics: metrics(row.metrics, `edges[${index}].metrics`),
    };
    const key = edgeKey(normalized);
    if (!graphEdges.has(key)) throw new Error(`edges[${index}] does not identify a serialized source-graph edge.`);
    if (edgeKeys.has(key)) throw new Error(`edges[${index}] duplicates edge ${key}.`);
    edgeKeys.add(key);
    return normalized;
  });
  return {
    schema: NODE_EDGE_EVIDENCE_OVERLAY_SCHEMA,
    artifact_sha256: artifactSha,
    source: source(value.source),
    nodes: normalizedNodes,
    edges: normalizedEdges,
    interpretation_boundary: "Imported JSON evidence is displayed only on exact hash-bound node and edge identities. It is not executed and cannot establish runtime facts beyond its declared evidence class and source.",
  };
}

export function indexNodeEdgeEvidenceOverlay(overlay) {
  return {
    nodes: new Map((overlay?.nodes || []).map((row) => [Number(row.op_index), row])),
    edges: new Map((overlay?.edges || []).map((row) => [edgeKey(row), row])),
  };
}

function metrics(value, pointer) {
  const rows = array(value, pointer);
  if (!rows.length || rows.length > MAX_METRICS) throw new Error(`${pointer} must contain 1-${MAX_METRICS} metrics.`);
  const keys = new Set();
  return rows.map((metric, index) => {
    if (!plainObject(metric)) throw new Error(`${pointer}[${index}] must be an object.`);
    const key = boundedText(metric.key, `${pointer}[${index}].key`, 96);
    if (keys.has(key)) throw new Error(`${pointer}[${index}] duplicates metric key ${key}.`);
    keys.add(key);
    const evidenceClass = boundedText(metric.evidence_class, `${pointer}[${index}].evidence_class`, 16).toUpperCase();
    if (!isCanonicalEvidenceClass(evidenceClass)) throw new Error(`${pointer}[${index}].evidence_class is invalid.`);
    const numeric = typeof metric.value === "number";
    if (numeric && !Number.isFinite(metric.value)) throw new Error(`${pointer}[${index}].value must be finite.`);
    if (!numeric && metric.value !== null && typeof metric.value !== "string" && typeof metric.value !== "boolean") {
      throw new Error(`${pointer}[${index}].value must be a finite number, string, boolean, or null.`);
    }
    if ((metric.value === null) !== (evidenceClass === "NOT_ASSESSABLE")) {
      throw new Error(`${pointer}[${index}] must use NOT_ASSESSABLE exactly when value is null.`);
    }
    return {
      key,
      value: typeof metric.value === "string" ? boundedText(metric.value, `${pointer}[${index}].value`, 512) : metric.value,
      unit: metric.unit == null ? null : boundedText(metric.unit, `${pointer}[${index}].unit`, 48),
      evidence_class: evidenceClass,
      source_pointer: metric.source_pointer == null ? null : boundedText(metric.source_pointer, `${pointer}[${index}].source_pointer`, 512),
    };
  });
}

function source(value) {
  if (value == null) return { label: "external evidence", sha256: null, collected_at: null };
  if (!plainObject(value)) throw new Error("source must be an object.");
  return {
    label: boundedText(value.label || "external evidence", "source.label", 160),
    sha256: value.sha256 == null ? null : sha(value.sha256, "source.sha256"),
    collected_at: value.collected_at == null ? null : timestamp(value.collected_at, "source.collected_at"),
  };
}

function serializedEdges(analysis) {
  const producer = new Map();
  const consumers = new Map();
  for (const op of analysis?.ops || []) {
    for (const tensorIndex of indices(op.outputs)) if (!producer.has(tensorIndex)) producer.set(tensorIndex, Number(op.index));
    for (const tensorIndex of indices(op.inputs)) {
      if (!consumers.has(tensorIndex)) consumers.set(tensorIndex, new Set());
      consumers.get(tensorIndex).add(Number(op.index));
    }
  }
  const rows = [];
  for (const [tensorIndex, targets] of consumers) {
    const sourceOp = producer.get(tensorIndex);
    if (!Number.isSafeInteger(sourceOp)) continue;
    for (const targetOp of targets) rows.push({ producer_op_index: sourceOp, consumer_op_index: targetOp, tensor_index: tensorIndex });
  }
  return rows;
}

function edgeKey(row) { return `${Number(row.producer_op_index)}:${Number(row.consumer_op_index)}:${Number(row.tensor_index)}`; }
function indices(value) { return (Array.isArray(value) ? value : []).map(Number).filter((item) => Number.isSafeInteger(item) && item >= 0); }
function exactIndex(value, pointer) { const number = Number(value); if (!Number.isSafeInteger(number) || number < 0) throw new Error(`${pointer} must be a non-negative integer.`); return number; }
function array(value, pointer) { if (!Array.isArray(value)) throw new Error(`${pointer} must be an array.`); return value; }
function sha(value, pointer) { const text = String(value || "").toLowerCase(); if (!/^[0-9a-f]{64}$/.test(text)) throw new Error(`${pointer} must be a SHA-256 hex digest.`); return text; }
function boundedText(value, pointer, maximum) { const text = String(value || "").trim(); if (!text || text.length > maximum) throw new Error(`${pointer} must contain 1-${maximum} characters.`); return text; }
function timestamp(value, pointer) { const text = boundedText(value, pointer, 64); if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/.test(text) || !Number.isFinite(Date.parse(text))) throw new Error(`${pointer} must be an RFC 3339 UTC timestamp.`); return text; }
function plainObject(value) { return Boolean(value) && typeof value === "object" && !Array.isArray(value) && Object.getPrototypeOf(value) === Object.prototype; }
