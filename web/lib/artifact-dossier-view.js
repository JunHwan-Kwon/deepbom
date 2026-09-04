import { formatBytes } from "./format.js";

export function renderArtifactDossier(root, context, analysis = {}) {
  if (!root) return;
  const ir = context?.artifact_ir;
  root.toggleAttribute("data-ready", Boolean(ir));
  if (!ir) return;
  const artifact = ir.artifact;
  const graph = ir.graph;
  const storage = ir.storage_topology;
  const quantization = ir.quantization_contracts;
  const runtimeCount = ir.overlays?.runtime?.length || 0;
  const artifactSet = analysis.artifact_set;
  const fileCount = Array.isArray(artifactSet?.files) ? artifactSet.files.length : 1;
  const put = (key, value, title = value) => {
    const node = root.querySelector(`[data-artifact-field="${key}"]`);
    if (!node) return;
    node.textContent = String(value);
    if (title) node.title = String(title);
  };

  put("name", artifact.filename);
  put("format", String(artifact.format || "unknown").toUpperCase());
  put("size", artifact.byte_length ? formatBytes(artifact.byte_length.number ?? artifact.byte_length.decimal) : "Not exposed by parser");
  put("files", `${fileCount} hash-bound ${fileCount === 1 ? "file" : "files"} / ${sourceLabel(artifactSet?.source?.kind)}`);
  put("ir-schema", `${ir.schema} / method ${ir.method_version}`);
  put("sha256", artifact.sha256);
  put("artifact-set-sha256", artifact.artifact_set_sha256 || "Not separately bound");
  put("ir-sha256", ir.artifact_ir_sha256);

  if (graph.status === "serialized") {
    put("graph-value", `${count(graph.totals.operator_count, "operator")} / ${count(graph.totals.value_count, "value")} / ${count(graph.totals.scope_count, "scope")}`);
    put("graph-note", graph.completeness === "all_serialized_scopes_materialized" || graph.completeness === "serialized_scope_materialized"
      ? "Serialized operators, values, ports, and available graph scopes are materialized."
      : "The primary graph is materialized; additional serialized scope inventory is preserved with explicit completeness.");
  } else {
    put("graph-value", "Not serialized by this format");
    put("graph-note", "No executable edge is inferred from tensor names or architecture order.");
  }

  const storageCount = storage.totals?.object_count || 0;
  put("storage-value", storageCount ? `${count(storageCount, "object")} / ${formatBytes(storage.totals.serialized_object_bytes_sum?.number ?? storage.totals.serialized_object_bytes_sum?.decimal)}` : "No serialized payload ledger");
  put("storage-note", storageCount ? `${storage.totals.exact_range_count || 0} exact byte ranges; storage bytes are not runtime allocation.` : "This state does not imply that runtime-created weights or buffers are absent.");

  const quantCount = quantization.totals?.record_count || 0;
  put("quantization-value", quantCount ? `${count(quantCount, "scoped contract")} / ${quantization.totals.complete_record_count || 0} complete` : "Not applicable or not serialized");
  put("quantization-note", quantCount ? `${quantization.totals.partial_record_count || 0} partial contracts remain explicitly separate.` : "No quantization contract is synthesized when the artifact does not encode one.");

  put("runtime-value", runtimeCount ? count(runtimeCount, "imported record") : "Not imported");
  put("runtime-note", runtimeCount ? "Hash-bound runtime evidence is attached as an overlay and does not rewrite artifact facts." : "Static analysis does not claim observed backend assignment or measured execution.");
}

function count(value, unit) {
  const number = Number(value || 0);
  return `${number.toLocaleString("en-US")} ${unit}${number === 1 ? "" : "s"}`;
}

function sourceLabel(kind) {
  return ({ local: "local browser selection", huggingface: "Hugging Face snapshot", https: "HTTPS source", gcs: "Google Cloud Storage" })[kind] || "local browser selection";
}
