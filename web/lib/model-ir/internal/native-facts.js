import { canonicalJson } from "../../report-utils.js";
import { sha256TextHex } from "../../sha256-sync.js";
import { clone, list } from "./shared.js";

export const MODEL_IR_NATIVE_FACT_LEDGER_SCHEMA = "deepbom.model_ir_native_fact_ledger.v1";

export function buildModelIrNativeFactLedger(analysis, artifactIr) {
  const operatorByNativeIndex = new Map(artifactIr.graph.operators.filter(row => row.scope_ref === artifactIr.graph.primary_scope_ref).map((row) => [row.native_index, row.id]));
  const graphdef = analysis?.graphdef;
  const controlDependencies = list(graphdef?.control_dependencies).map((row, index) => ({
    id: `native-control:${index}`,
    source_operation_ref: operatorByNativeIndex.get(row.source_op_index) || null,
    target_operation_ref: operatorByNativeIndex.get(row.target_op_index) || null,
    source_node_name: String(row.source_node_name || ""),
    target_node_name: String(row.target_node_name || ""),
    status: operatorByNativeIndex.has(row.source_op_index) && operatorByNativeIndex.has(row.target_op_index) ? "resolved" : "unresolved",
    native_source: { schema: graphdef?.schema || null, pointer: `/graphdef/control_dependencies/${index}` },
  }));
  const facts = {
    control_dependencies: controlDependencies,
    function_summaries: list(graphdef?.functions).map((row, index) => ({
      index: Number.isSafeInteger(row?.index) ? row.index : index,
      name: String(row?.name || `function_${index}`),
      node_count: Number(row?.node_count || 0),
      return_entry_count: Number(row?.return_entry_count || 0),
      control_return_entry_count: Number(row?.control_return_entry_count || 0),
      materialization_status: "inventory_only_body_not_materialized",
    })),
    savedmodel_meta_graphs: list(analysis?.savedmodel?.additional_meta_graphs).map((row) => clone(row)),
  };
  const material = Object.values(facts).some((rows) => rows.length > 0);
  const body = {
    schema: MODEL_IR_NATIVE_FACT_LEDGER_SCHEMA,
    status: material ? "bounded_supplementary_facts_embedded" : "no_supplementary_facts",
    artifact_ir_sha256: artifactIr.artifact_ir_sha256,
    source_format: artifactIr.artifact.format,
    facts,
    interpretation_boundary: "Only explicitly allow-listed serialized facts omitted by the frozen Artifact IR v2 compatibility contract are embedded. This ledger does not authorize arbitrary native-analysis access and does not synthesize absent program structure.",
  };
  return { ...body, ledger_sha256: sha256TextHex(canonicalJson(body)) };
}

export function validateModelIrNativeFactLedger(input, artifactIrSha256) {
  canonicalJson(input);
  const value = clone(input);
  if (value?.schema !== MODEL_IR_NATIVE_FACT_LEDGER_SCHEMA) throw new Error("Model IR native fact ledger schema is invalid.");
  const digest = String(value.ledger_sha256 || "").toLowerCase();
  delete value.ledger_sha256;
  if (value.artifact_ir_sha256 !== artifactIrSha256 || digest !== sha256TextHex(canonicalJson(value))) {
    throw new Error("Model IR native fact ledger binding is invalid.");
  }
  for (const dependency of list(value.facts?.control_dependencies)) {
    if (!dependency.id || !["resolved", "unresolved"].includes(dependency.status)) throw new Error("Model IR native control dependency is invalid.");
  }
  return { ...value, ledger_sha256: digest };
}
