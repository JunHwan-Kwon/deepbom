import { formatBytes, formatNumber } from "./format.js";
import { evidenceDisclosure } from "./dom.js";
import { SAFETENSORS_QUANTIZATION_CONTRACT_SCHEMA } from "./safetensors-quantization-contract.js";

export function safeTensorsQuantizationPanel(analysis, { element, panel, panelStatus, emptyState }) {
  if (String(analysis?.format || "").toLowerCase() !== "safetensors") return null;
  const contract = analysis?.safetensors?.quantization_contract || null;
  const node = panel(
    "Packed Weight Quantization",
    "Repository-bound AWQ, GPTQ, HQQ, and compressed-tensors declarations with source-pinned packed-layout conservation.",
    "safetensors-quantization",
    true,
  );
  if (!contract || contract.schema !== SAFETENSORS_QUANTIZATION_CONTRACT_SCHEMA) {
    panelStatus(node, "contract not emitted", "NOT ASSESSED");
    node.append(emptyState("No SafeTensors packed-weight quantization contract was emitted."));
    return node;
  }
  const assessed = contract.status === "assessed" || contract.status === "fail";
  if (!assessed) {
    panelStatus(node, normalizedStatus(contract.status), "NOT ASSESSED");
    node.append(emptyState(contract.reason || "No supported source-bound packed-weight declaration was selected with this repository."));
    return node;
  }

  panelStatus(
    node,
    contract.status === "assessed"
      ? `${formatNumber(contract.valid_module_count || 0)}/${formatNumber(contract.module_count || 0)} modules conserve`
      : `${formatNumber(contract.invalid_module_count || 0)} invalid module(s)`,
    contract.status === "assessed" ? "OBSERVED + SOURCE-PINNED DERIVED" : "REVIEW REQUIRED",
  );
  const issueRows = prioritizedIssues(contract);
  if (issueRows.length) {
    const issues = element("ul", "artifact-quant-issue-list");
    for (const issue of issueRows.slice(0, 8)) issues.append(element("li", "", issue));
    node.append(issues);
  }

  const ledger = element("dl", "artifact-metadata-ledger artifact-quant-ledger");
  const activations = Array.isArray(contract.activation_quantization_contracts) ? contract.activation_quantization_contracts : [];
  const activationFailures = activations.filter((row) => row.status === "fail").length;
  const shard = contract.shard_ownership || null;
  for (const [label, value] of [
    ["Method / packed layout", `${String(contract.method || "not resolved").toUpperCase()} / ${contract.source?.layout || "not bound"}`],
    ["Weight code / grouping", `${contract.bits ?? (contract.module_count ? "module-specific" : "?")} / ${contract.granularity || "not derived"} / group ${contract.group_size ?? "module-specific"} along ${contract.logical_group_axis || "unbound axis"}`],
    ["Configuration ownership", contract.configuration_scope || "global declaration"],
    ["Module conservation", `${formatNumber(contract.valid_module_count || 0)}/${formatNumber(contract.module_count || 0)} valid; ${formatNumber(contract.invalid_module_count || 0)} invalid`],
    ["Logical / packed weight codes", `${exactInteger(contract.logical_weight_element_count)} / ${exactInteger(contract.packed_weight_code_capacity)}`],
    ["Logical / serialized / padding bits", `${exactInteger(contract.logical_weight_bits)} / ${exactInteger(contract.packed_weight_storage_bits)} / ${exactInteger(contract.packing_padding_bits)}`],
    ["Scale / zero-point capacity", `${exactInteger(contract.scale_element_count)} / ${exactInteger(contract.zero_point_code_capacity)}`],
    ["Packed tensors", `${formatBytes(contract.packed_tensor_bytes || 0)} (${formatNumber(contract.packed_tensor_bytes || 0)} B)`],
    ["Activation / KV contracts", activations.length ? `${formatNumber(activations.length)} contract(s); ${formatNumber(activationFailures)} invalid` : "not declared"],
    ["Shard ownership", shard ? `${normalizedStatus(shard.status)}; ${formatNumber(shard.shard_bound_tensor_count || 0)}/${formatNumber(shard.tensor_count || 0)} tensors bound` : "not emitted"],
    ["Pinned implementation", sourceText(contract.source)],
    ["Numerical payload", "Separate full-payload integrity decodes supported scalar storage; this contract does not reconstruct floating weights or calibration error"],
    ["Runtime / quality", "NOT ASSESSED: calibration, reconstructed-weight error, kernel selection, runtime support, and task quality require separate evidence"],
  ]) ledger.append(element("dt", "", label), element("dd", "", String(value)));
  node.append(ledger);

  const modules = Array.isArray(contract.modules) ? contract.modules : [];
  node.append(evidenceDisclosure(
    `Complete module ledger (${formatNumber(modules.length)})`,
    JSON.stringify(modules, null, 2),
    { contentLabel: "Complete source-pinned packed-weight module ledger" },
  ));
  if (activations.length) node.append(evidenceDisclosure(
    `Activation and KV contracts (${formatNumber(activations.length)})`,
    JSON.stringify(activations, null, 2),
    { contentLabel: "Complete activation and KV quantization companion ledger" },
  ));
  if (shard) node.append(evidenceDisclosure(
    `Quantization tensor shard ownership (${formatNumber(shard.tensor_count || 0)})`,
    JSON.stringify(shard, null, 2),
    { contentLabel: "Complete quantization tensor to SafeTensors shard ownership ledger" },
  ));
  return node;
}

function prioritizedIssues(contract) {
  const rows = [];
  for (const conflict of contract.declaration_conflicts || []) {
    rows.push(`Declaration conflict: ${conflict.field} = ${(conflict.values || []).join(" / ")} (${(conflict.paths || []).join(", ")})`);
  }
  for (const issue of contract.config_issues || []) rows.push(`Configuration: ${normalizedStatus(issue)}`);
  for (const module of contract.modules || []) {
    for (const issue of module.issues || []) rows.push(`${module.name}: ${normalizedStatus(issue)}`);
  }
  return rows;
}

function sourceText(source) {
  if (!source?.commit || !source?.sha256) return "not bound";
  return `${source.repository || "repository"}@${source.commit}; ${source.path || "source"}; SHA-256 ${source.sha256}`;
}

function exactInteger(value) {
  return /^(0|[1-9]\d*)$/.test(String(value || "")) ? BigInt(value).toLocaleString("en-US") : "not derived";
}

function normalizedStatus(value) {
  return String(value || "not assessed").replaceAll("_", " ");
}
