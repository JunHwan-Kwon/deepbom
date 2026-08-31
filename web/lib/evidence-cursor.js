export const EVIDENCE_CURSOR_SCHEMA = "deepbom.evidence_cursor.v1";

const EMPTY = Object.freeze({
  schema: EVIDENCE_CURSOR_SCHEMA,
  artifact_sha256: null,
  finding_id: null,
  op_index: null,
  tensor_index: null,
  runtime_node_id: null,
  report_anchor: null,
  source: "initial",
  revision: 0,
});

export function createEvidenceCursor(initial = null) {
  let state = Object.freeze(normalize({ ...EMPTY, ...(initial || {}) }, 0));
  const listeners = new Set();
  return Object.freeze({
    get: () => state,
    subscribe(listener) {
      if (typeof listener !== "function") throw new Error("Evidence cursor subscriber must be a function.");
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    select(selection, { source = "unknown" } = {}) {
      const next = normalize({
        ...EMPTY,
        artifact_sha256: selection?.artifact_sha256 ?? state.artifact_sha256,
        ...(selection || {}),
        source,
      }, state.revision + 1);
      if (sameSelection(state, next)) return state;
      const previous = state;
      state = Object.freeze(next);
      for (const listener of listeners) listener(state, previous);
      return state;
    },
    reset(artifactSha256 = null, { source = "reset" } = {}) {
      const previous = state;
      state = Object.freeze(normalize({ ...EMPTY, artifact_sha256: artifactSha256, source }, previous.revision + 1));
      for (const listener of listeners) listener(state, previous);
      return state;
    },
  });
}

export function validateEvidenceCursor(value) {
  try {
    normalize(value, value?.revision);
    return { valid: true, issues: [] };
  } catch (error) {
    return { valid: false, issues: [String(error?.message || error)] };
  }
}

function normalize(value, revision) {
  if (!value || value.schema !== EVIDENCE_CURSOR_SCHEMA) throw new Error("Evidence cursor schema is invalid.");
  if (!Number.isSafeInteger(revision) || revision < 0) throw new Error("Evidence cursor revision must be a non-negative safe integer.");
  return {
    schema: EVIDENCE_CURSOR_SCHEMA,
    artifact_sha256: optionalSha(value.artifact_sha256),
    finding_id: optionalText(value.finding_id, 200, "finding_id"),
    op_index: optionalIndex(value.op_index, "op_index"),
    tensor_index: optionalIndex(value.tensor_index, "tensor_index"),
    runtime_node_id: optionalText(value.runtime_node_id, 500, "runtime_node_id"),
    report_anchor: optionalText(value.report_anchor, 500, "report_anchor"),
    source: optionalText(value.source, 100, "source") || "unknown",
    revision,
  };
}

function sameSelection(left, right) {
  return ["artifact_sha256", "finding_id", "op_index", "tensor_index", "runtime_node_id", "report_anchor", "source"]
    .every((key) => left[key] === right[key]);
}

function optionalIndex(value, label) {
  if (value == null || value === "") return null;
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number < 0) throw new Error(`Evidence cursor ${label} is invalid.`);
  return number;
}

function optionalSha(value) {
  if (value == null || value === "") return null;
  const text = String(value).toLowerCase();
  if (!/^[a-f0-9]{64}$/.test(text)) throw new Error("Evidence cursor artifact SHA-256 is invalid.");
  return text;
}

function optionalText(value, maximum, label) {
  if (value == null || value === "") return null;
  const text = String(value).trim();
  if (!text || text.length > maximum) throw new Error(`Evidence cursor ${label} is invalid.`);
  return text;
}
