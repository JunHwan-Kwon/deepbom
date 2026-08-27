import { SAFETENSORS_QUANTIZATION_CONTRACT_SCHEMA } from "./safetensors-quantization-contract.js";

export function safeTensorsQuantizationPropertyEntries(analysis, { evidencePointer = "/format_extensions/safetensors/quantization_contract" } = {}) {
  if (String(analysis?.format || "").toLowerCase() !== "safetensors") return [];
  const contract = analysis?.safetensors?.quantization_contract;
  if (contract?.schema !== SAFETENSORS_QUANTIZATION_CONTRACT_SCHEMA) return [];
  return [
    ["deepbom:model:packedWeightQuantizationContractSchema", contract.schema],
    ["deepbom:model:packedWeightQuantizationStatus", contract.status],
    ["deepbom:model:packedWeightQuantizationMethod", contract.method],
    ["deepbom:model:packedWeightQuantizationBits", contract.bits],
    ["deepbom:model:packedWeightQuantizationGranularity", contract.granularity],
    ["deepbom:model:packedWeightQuantizationGroupSize", contract.group_size],
    ["deepbom:model:packedWeightQuantizationModuleCount", contract.module_count],
    ["deepbom:model:packedWeightQuantizationInvalidModuleCount", contract.invalid_module_count],
    ["deepbom:model:packedWeightQuantizationConfigurationScope", contract.configuration_scope],
    ["deepbom:model:activationQuantizationContractCount", contract.activation_quantization_contract_count],
    ["deepbom:model:activationQuantizationInvalidContractCount", (contract.activation_quantization_contracts || []).filter((row) => row.status === "fail").length],
    ["deepbom:model:quantizationShardOwnershipStatus", contract.shard_ownership?.status],
    ["deepbom:model:quantizationShardBoundTensorCount", contract.shard_ownership?.shard_bound_tensor_count],
    ["deepbom:model:quantizationTensorCount", contract.shard_ownership?.tensor_count],
    ["deepbom:model:packedWeightQuantizationSourceCommit", contract.source?.commit],
    ["deepbom:model:packedWeightQuantizationSourceSha256", contract.source?.sha256],
    ["deepbom:model:packedWeightQuantizationEvidencePointer", evidencePointer],
  ];
}
