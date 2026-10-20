import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

// Only the adapter reads native analysis. The hierarchy engine consumes this
// allow-listed, artifact/Model-bound record; renderers never reparse ONNX.
export function collectModelSourceAnnotations(analysis, model) {
  const mainRegion = model.program.regions.find(row => row.parent_region_ref === null && row.kind !== "function")?.id;
  const primary = new Map(model.program.operations.filter(row => row.region_ref === mainRegion).map(row => [row.native_index, row]));
  const records = [];
  for (const [position, native] of (analysis?.ops || []).entries()) {
    const op = primary.get(native.index ?? position);
    if (op && native.source_annotation?.basis === 'serialized_keras_config') {
      const entry=native.source_annotation;
      const valid=[entry.scope_names,entry.class_names].every(a=>Array.isArray(a)&&a.length>0&&a.length<=128&&a.every(s=>typeof s==='string'&&s.length<=4096&&s.isWellFormed()&&!/[\u0000-\u001f\u007f]/.test(s)))&&entry.scope_names.length===entry.class_names.length;
      records.push({subject_ref:op.id,scope_names:valid?entry.scope_names:[],class_names:valid?entry.class_names:[],status:valid?'observed_exporter_annotation':'invalid_annotation',reason:valid?null:'invalid_serialized_source_annotation',source_pointer:`/ops/${position}/source_annotation`,metadata_sha256:sha256TextHex(canonicalJson(entry)),call_identity_status:'exporter_annotation_not_runtime_invocation'});
      continue;
    }
    const entries = native.onnx_metadata_properties;
    if (!op || !Array.isArray(entries) || !entries.length) continue;
    const keys = new Map(), duplicates = new Set();
    for (const row of entries) {
      if (keys.has(row.key)) duplicates.add(row.key);
      keys.set(row.key, row.value);
    }
    const relevant = ["pkg.torch.onnx.name_scopes", "pkg.torch.onnx.class_hierarchy"];
    if (!relevant.some(key => keys.has(key))) continue;
    let scopes = [], classes = [], status = "observed_exporter_annotation", reason = null;
    try {
      if (relevant.some(key => duplicates.has(key))) throw new Error("duplicate_source_metadata_key");
      scopes = parseSourceStringList(keys.get(relevant[0]));
      classes = keys.has(relevant[1]) ? parseSourceStringList(keys.get(relevant[1])) : [];
      if (!scopes.length) throw new Error("empty_source_scope_list");
      if (classes.length && classes.length !== scopes.length) throw new Error("source_scope_class_length_mismatch");
    } catch (error) {
      status = "invalid_annotation"; reason = error.message; scopes = []; classes = [];
    }
    records.push({ subject_ref: op.id, scope_names: scopes, class_names: classes, status, reason,
      source_pointer: `/ops/${position}/onnx_metadata_properties`,
      metadata_sha256: sha256TextHex(canonicalJson(entries)),
      call_identity_status: "exporter_annotation_not_runtime_invocation",
    });
  }
  return records;
}

// PyTorch metadata uses a repr(list[str]), rather than necessarily JSON. This
// bounded grammar accepts string lists only: no Python evaluator or code paths.
export function parseSourceStringList(input) {
  if (typeof input !== "string" || input.length > 65536) throw new Error("source_list_missing_or_oversized");
  let i = 0;
  const ws = () => { while (/\s/.test(input[i] || "!") && i < input.length) i++; };
  ws(); if (input[i++] !== "[") throw new Error("invalid_source_list");
  const result = [];
  ws();
  while (input[i] !== "]") {
    if (result.length >= 128) throw new Error("source_scope_depth_exceeded");
    const quote = input[i++];
    if (quote !== "'" && quote !== '"') throw new Error("invalid_source_list_string");
    let value = "", closed = false;
    while (i < input.length) {
      let c = input[i++];
      if (c === quote) { closed = true; break; }
      if (c === "\\") {
        const escaped = input[i++];
        if (["\\", "'", '"', "/"].includes(escaped)) c = escaped;
        else if (escaped === "u" || escaped === "x" || escaped === "U") {
          const width = escaped === "u" ? 4 : escaped === "x" ? 2 : 8;
          const hex = input.slice(i, i + width);
          if (hex.length !== width || !/^[a-f0-9]+$/i.test(hex) || parseInt(hex, 16) > 0x10ffff) throw new Error("invalid_source_escape");
          c = String.fromCodePoint(parseInt(hex, 16)); i += width;
        } else if (Object.hasOwn({ n: "\n", r: "\r", t: "\t" }, escaped)) c = { n: "\n", r: "\r", t: "\t" }[escaped];
        else throw new Error("unsupported_source_escape");
      }
      value += c;
      if (value.length > 4096) throw new Error("source_name_too_long");
    }
    if (!closed || /[\u0000-\u001f\u007f]/.test(value) || !value.isWellFormed()) throw new Error("invalid_source_name");
    result.push(value); ws();
    if (input[i] === "]") break;
    if (input[i++] !== ",") throw new Error("invalid_source_list_separator");
    ws();
  }
  if (input[i++] !== "]") throw new Error("invalid_source_list_end");
  ws(); if (i !== input.length) throw new Error("invalid_source_list_suffix");
  return result;
}
