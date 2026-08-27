import { formatBytes, formatNumber } from "./format.js";
import { code, markdownTable } from "./report-utils.js";
import { SAFETENSORS_QUANTIZATION_CONTRACT_SCHEMA } from "./safetensors-quantization-contract.js";

export function safeTensorsQuantizationMarkdown(analysis) {
  if (String(analysis?.format || "").toLowerCase() !== "safetensors") return "";
  const contract = analysis?.safetensors?.quantization_contract || null;
  if (!contract || contract.schema !== SAFETENSORS_QUANTIZATION_CONTRACT_SCHEMA) {
    return "## SafeTensors Packed-weight Quantization Contract\n\nNOT ASSESSED: no packed-weight quantization contract was emitted.";
  }
  if (!["assessed", "fail"].includes(contract.status)) {
    return [
      "## SafeTensors Packed-weight Quantization Contract",
      `**NOT ASSESSED: ${normalized(contract.status)}.** ${contract.reason || "No supported source-bound packed-weight declaration was selected."}`,
      "> Absence of a supported declaration is not evidence that the checkpoint is unquantized.",
    ].join("\n\n");
  }
  const source = contract.source || {};
  const issueRows = [
    ...(contract.declaration_conflicts || []).map((row) => ["declaration", row.field, `${(row.values || []).join(" / ")} at ${(row.paths || []).join(", ")}`]),
    ...(contract.config_issues || []).map((issue) => ["configuration", issue, "declared repository contract"]),
    ...(contract.modules || []).flatMap((module) => (module.issues || []).map((issue) => ["module", issue, code(module.name)])),
  ];
  const modules = Array.isArray(contract.modules) ? contract.modules : [];
  const activations = Array.isArray(contract.activation_quantization_contracts) ? contract.activation_quantization_contracts : [];
  const activationFailures = activations.filter((row) => row.status === "fail").length;
  const shard = contract.shard_ownership || null;
  return [
    "## SafeTensors Packed-weight Quantization Contract (OBSERVED + SOURCE-PINNED DERIVED)",
    markdownTable(["Field", "Value"], [
      ["Status / method", `${contract.status} / ${String(contract.method || "not resolved").toUpperCase()}`],
      ["Code width / storage unit", `${contract.bits ?? (modules.length ? "module-specific" : "not derived")} bits / ${contract.storage_word_bits ?? "not derived"} storage bits${contract.codes_per_storage_word == null ? "" : ` / ${contract.codes_per_storage_word} codes`}`],
      ["Granularity / logical axis", `${contract.granularity || "not derived"} / ${contract.logical_group_axis || "not derived"}`],
      ["Configuration ownership", contract.configuration_scope || "global declaration"],
      ["Declared group size", contract.group_size ?? "not derived"],
      ["Modules", `${formatNumber(contract.valid_module_count || 0)}/${formatNumber(contract.module_count || 0)} valid; ${formatNumber(contract.invalid_module_count || 0)} invalid`],
      ["Logical weight elements / packed code capacity", `${exactInteger(contract.logical_weight_element_count)} / ${exactInteger(contract.packed_weight_code_capacity)}`],
      ["Logical / serialized weight bits / padding", `${exactInteger(contract.logical_weight_bits)} / ${exactInteger(contract.packed_weight_storage_bits)} / ${exactInteger(contract.packing_padding_bits)} (${normalized(contract.packing_conservation_status)})`],
      ["Scale elements / zero-point code capacity", `${exactInteger(contract.scale_element_count)} / ${exactInteger(contract.zero_point_code_capacity)}`],
      ["Packed tensor bytes", `${formatBytes(contract.packed_tensor_bytes || 0)} (${formatNumber(contract.packed_tensor_bytes || 0)} B)`],
      ["Activation / KV contracts", activations.length ? `${formatNumber(activations.length)} assessed; ${formatNumber(activationFailures)} invalid` : "not declared"],
      ["Quantization tensor shard ownership", shard ? `${normalized(shard.status)}; ${formatNumber(shard.shard_bound_tensor_count || 0)}/${formatNumber(shard.tensor_count || 0)} bound` : "not emitted"],
      ["Pinned implementation", source.commit ? `${source.repository}@${source.commit}; ${source.path}; ${source.layout}; SHA-256 ${code(source.sha256 || "not bound")}` : "not bound"],
      ["Evidence boundary", contract.boundary || "not declared"],
    ]),
    issueRows.length ? "### Packed-layout Issues" : "",
    issueRows.length ? markdownTable(["Scope", "Issue", "Binding"], issueRows) : "",
    modules.length ? "### Complete Packed-module Ledger" : "",
    modules.length ? markdownTable(
      ["Module", "Status", "Bits", "Input / output", "Groups / size", "Logical / packed codes", "Scale / zero capacity", "Selection", "Issues"],
      modules.map((row) => [
        code(row.name), row.status, row.bits ?? "not derived", `${formatNumber(row.input_features || 0)} / ${formatNumber(row.output_features || 0)}`,
        `${formatNumber(row.group_count || 0)} / ${formatNumber(row.group_size || 0)}`,
        `${exactInteger(row.logical_weight_element_count)} / ${exactInteger(row.packed_weight_code_capacity)}`,
        `${exactInteger(row.scale_element_count)} / ${exactInteger(row.zero_point_code_capacity)}`,
        row.ownership?.target_precedence_match || row.ownership?.tag || "global",
        (row.issues || []).join(", ") || "none",
      ]),
    ) : "",
    activations.length ? "### Activation and KV Quantization Contracts" : "",
    activations.length ? markdownTable(
      ["Module", "Kind", "Status", "Scale", "Zero point", "Issues"],
      activations.map((row) => [
        code(row.module_name), row.kind, normalized(row.status),
        code(row.scale_tensor?.tensor_name || row.k_scale_tensor?.tensor_name || "runtime/not serialized"),
        code(row.zero_point_tensor?.tensor_name || row.v_scale_tensor?.tensor_name || "omitted/runtime"),
        (row.issues || []).join(", ") || "none",
      ]),
    ) : "",
    "> SafeTensors itself serializes tensor identity, dtype, shape, and byte ranges. The repository-bound contract above validates one supported pinned packed-weight layout; it does not establish an executable graph, activation Q/DQ placement, reconstructed-weight accuracy, calibration quality, runtime placement, or task accuracy.",
  ].filter(Boolean).join("\n\n");
}

function exactInteger(value) {
  return /^(0|[1-9]\d*)$/.test(String(value || "")) ? String(value) : "not derived";
}

function normalized(value) {
  return String(value || "not assessed").replaceAll("_", " ");
}
