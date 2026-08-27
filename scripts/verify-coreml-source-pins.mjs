import { createHash } from "node:crypto";

import { COREML_BLOB_SOURCE } from "../web/lib/coreml-blob.js";
import { COREML_FORMAT_SOURCE } from "../web/lib/coreml-metadata-adapter.js";
import { COREML_MIL_SOURCE } from "../web/lib/coreml-mil-program.js";
import { COREML_NEURAL_NETWORK_SOURCE } from "../web/lib/coreml-neural-network.js";
import { COREML_CLASSICAL_SOURCE } from "../web/lib/coreml-classical-model.js";

const commit = COREML_FORMAT_SOURCE.source_commit;
const rows = [
  [COREML_FORMAT_SOURCE.model_proto, COREML_FORMAT_SOURCE.model_proto_sha256],
  [COREML_FORMAT_SOURCE.feature_types_proto, COREML_FORMAT_SOURCE.feature_types_proto_sha256],
  [COREML_FORMAT_SOURCE.package_source, COREML_FORMAT_SOURCE.package_source_sha256],
  [COREML_FORMAT_SOURCE.coremltools_init, COREML_FORMAT_SOURCE.coremltools_init_sha256],
  [COREML_FORMAT_SOURCE.deployment_compatibility, COREML_FORMAT_SOURCE.deployment_compatibility_sha256],
  [COREML_FORMAT_SOURCE.compute_plan, COREML_FORMAT_SOURCE.compute_plan_sha256],
  [COREML_NEURAL_NETWORK_SOURCE.neural_network_proto, COREML_NEURAL_NETWORK_SOURCE.neural_network_proto_sha256],
  [COREML_NEURAL_NETWORK_SOURCE.neural_network_validator, COREML_NEURAL_NETWORK_SOURCE.neural_network_validator_sha256],
  [COREML_NEURAL_NETWORK_SOURCE.neural_network_validator_utils, COREML_NEURAL_NETWORK_SOURCE.neural_network_validator_utils_sha256],
  [COREML_NEURAL_NETWORK_SOURCE.quantization_implementation, COREML_NEURAL_NETWORK_SOURCE.quantization_implementation_sha256],
  [COREML_MIL_SOURCE.mil_proto, COREML_MIL_SOURCE.mil_proto_sha256],
  [COREML_MIL_SOURCE.conv_definition, COREML_MIL_SOURCE.conv_definition_sha256],
  [COREML_MIL_SOURCE.linear_definition, COREML_MIL_SOURCE.linear_definition_sha256],
  [COREML_MIL_SOURCE.recurrent_definition, COREML_MIL_SOURCE.recurrent_definition_sha256],
  [COREML_MIL_SOURCE.transformer_ios18_definition, COREML_MIL_SOURCE.transformer_ios18_definition_sha256],
  [COREML_MIL_SOURCE.control_flow_definition, COREML_MIL_SOURCE.control_flow_definition_sha256],
  [COREML_MIL_SOURCE.compression_ios18_definition, COREML_MIL_SOURCE.compression_ios18_definition_sha256],
  [COREML_MIL_SOURCE.constexpr_ios16_definition, COREML_MIL_SOURCE.constexpr_ios16_definition_sha256],
  [COREML_BLOB_SOURCE.storage_format, COREML_BLOB_SOURCE.storage_format_sha256],
  [COREML_BLOB_SOURCE.dtype_source, COREML_BLOB_SOURCE.dtype_source_sha256],
  [COREML_BLOB_SOURCE.reader_source, COREML_BLOB_SOURCE.reader_source_sha256],
  [COREML_BLOB_SOURCE.subbyte_source, COREML_BLOB_SOURCE.subbyte_source_sha256],
  [COREML_BLOB_SOURCE.fp8_source, COREML_BLOB_SOURCE.fp8_source_sha256],
  ...COREML_CLASSICAL_SOURCE.files.map((row) => [row.path, row.sha256]),
];

const semanticRequirements = new Map([
  [COREML_MIL_SOURCE.conv_definition, [
    "class conv_transpose(Operation)",
    "weight: const tensor<[C_in,C_out/groups,*D_in], T>",
    "strides[r] * (D_in[r] - 1)",
  ]],
  [COREML_MIL_SOURCE.linear_definition, [
    "class einsum(Operation)",
    "parse_einsum_equation(self.equation.val)",
    "input1_vec == [0, 1, 2, 3]",
  ]],
  [COREML_MIL_SOURCE.recurrent_definition, [
    "class gru(Operation)",
    "class lstm(Operation)",
    "class rnn(Operation)",
    "dim_factor = 8 if self.direction.val == \"bidirectional\" else 4",
  ]],
  [COREML_MIL_SOURCE.transformer_ios18_definition, [
    "class scaled_dot_product_attention(Operation)",
    "similarity = np.matmul(query, key.swapaxes(-2, -1))",
    "attention = np.matmul(attention_weight, value)",
    "shape = list(self.query.shape[:-1]) + [self.value.shape[-1]]",
  ]],
  [COREML_MIL_SOURCE.control_flow_definition, [
    "class cond(Operation)",
    "Perform a conditional execution. The return types must be identical",
    "class while_loop(Operation)",
    "Perform the body repeatedly while the condition ``cond`` is true.",
    "self.blocks.append(cond_block)",
    "self.blocks.append(body_block)",
  ]],
  [COREML_MIL_SOURCE.compression_ios18_definition, [
    "class constexpr_blockwise_shift_scale(Operation)",
    "block_size[i] = data.shape[i] / scale.shape[i]",
    "class constexpr_lut_to_dense(Operation)",
    "if nbits != indices_dtype.get_bitwidth()",
    "class constexpr_sparse_to_dense(Operation)",
    "np.count_nonzero(mask.val) != nonzero_data.shape[0]",
  ]],
  [COREML_MIL_SOURCE.constexpr_ios16_definition, [
    "class constexpr_affine_dequantize(Operation)",
    "class constexpr_lut_to_dense(Operation)",
    "class constexpr_sparse_to_dense(Operation)",
  ]],
]);

if (!/^[a-f0-9]{40}$/.test(commit)) throw new Error("Core ML source commit is not immutable");
const seen = new Set();
for (const [path, expected] of rows) {
  if (!path || !/^[a-f0-9]{64}$/.test(expected) || seen.has(path)) throw new Error(`Invalid or duplicate Core ML source pin: ${path || "<missing>"}`);
  seen.add(path);
  const url = `https://raw.githubusercontent.com/apple/coremltools/${commit}/${path}`;
  const response = await fetch(url, { redirect: "error" });
  if (!response.ok) throw new Error(`Core ML source fetch failed (${response.status}): ${url}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const actual = createHash("sha256").update(bytes).digest("hex");
  if (actual !== expected) throw new Error(`${path} SHA-256 mismatch: ${actual} !== ${expected}`);
  const text = new TextDecoder().decode(bytes);
  for (const marker of semanticRequirements.get(path) || []) {
    if (!text.includes(marker)) throw new Error(`${path} no longer contains required semantic marker: ${marker}`);
  }
  console.log(`${path}: ${bytes.byteLength} B ${actual}`);
}
console.log(`Core ML source pins verified (${rows.length} files; apple/coremltools@${commit}).`);
