import { compareCanonicalText } from "../../report-utils.js";
import { list } from "./shared.js";

const STORAGE_ROLE_RULES = Object.freeze([
  ["token_embedding", /(?:^|\.)(?:token_embd|embed_tokens|word_embeddings)\.weight$/i],
  ["output_projection", /(?:^|\.)(?:output|lm_head)\.weight$/i],
  ["normalization_weight", /(?:^|\.)(?:attn_norm|ffn_norm|output_norm|input_layernorm|post_attention_layernorm)\.weight$/i],
  ["attention_query_weight", /(?:^|\.)(?:attn_q|q_proj)\.weight$/i],
  ["attention_key_weight", /(?:^|\.)(?:attn_k|k_proj)\.weight$/i],
  ["attention_value_weight", /(?:^|\.)(?:attn_v|v_proj)\.weight$/i],
  ["attention_output_weight", /(?:^|\.)(?:attn_output|o_proj)\.weight$/i],
  ["mlp_gate_weight", /(?:^|\.)(?:ffn_gate|gate_proj)\.weight$/i],
  ["mlp_up_weight", /(?:^|\.)(?:ffn_up|up_proj)\.weight$/i],
  ["mlp_down_weight", /(?:^|\.)(?:ffn_down|down_proj)\.weight$/i],
  ["mixture_of_experts_weight", /(?:^|\.)(?:experts?|expert\.[0-9]+|block_sparse_moe)(?:\.|$)/i],
]);

const OP_ROLE_RULES = Object.freeze([
  ["convolution", /^(?:conv|conv2d|conv3d|depthwiseconv2dnative|depthwise_conv_2d|transposeconv|convtranspose)$/i],
  ["normalization", /(?:layernormalization|layer_norm|batchnormalization|batch_norm|groupnormalization|rmsnorm)/i],
  ["matrix_projection", /^(?:matmul|gemm|fully_connected|linear)$/i],
  ["attention", /(?:^|\.)(?:attention|multiheadattention|scaled_dot_product_attention)$/i],
  ["attention_normalization", /^softmax$/i],
  ["pooling", /pool/i],
  ["activation", /^(?:relu|gelu|silu|swish|sigmoid|tanh|leakyrelu)$/i],
  ["residual_merge_candidate", /^(?:add|addv2)$/i],
]);

export function buildModelProfiles(artifactIr, program, storage) {
  const assignments = [];
  for (const operation of program.operations) {
    const rule = OP_ROLE_RULES.find(([, pattern]) => pattern.test(String(operation.native_op?.name || "")));
    if (rule) assignments.push(assignment(operation.id, rule[0], "native_operator_identity", "OBSERVED_SERIALIZED_ARTIFACT"));
    const namedRole = namedSemanticRole(operation.name);
    if (namedRole) assignments.push(assignment(operation.id, namedRole, "serialized_operation_name_pattern", "DERIVED"));
  }
  for (const object of storage.storage_objects) {
    const rule = STORAGE_ROLE_RULES.find(([, pattern]) => pattern.test(String(object.name || "")));
    if (rule) assignments.push(assignment(object.id, rule[0], "serialized_tensor_namespace_pattern", "DERIVED"));
  }

  const roleSet = new Set(assignments.map((row) => row.role));
  const operationNames = program.operations.map((row) => String(row.native_op?.name || ""));
  const transformerSignals = ["attention_query_weight", "attention_key_weight", "attention_value_weight"].filter((role) => roleSet.has(role)).length;
  const hasAttentionOp = operationNames.some((name) => /attention/i.test(name));
  const hasConvolution = roleSet.has("convolution");
  const hasEncoderNamespace = storage.storage_objects.some((row) => /(?:^|\.)(?:encoder|encoders)(?:\.|$)/i.test(row.name));
  const hasDecoderNamespace = storage.storage_objects.some((row) => /(?:^|\.)(?:decoder|decoders)(?:\.|$)/i.test(row.name));
  const profiles = [];

  if (transformerSignals === 3 || hasAttentionOp) profiles.push(profile("transformer", {
    status: transformerSignals === 3 ? "recognized_from_serialized_weight_roles" : "recognized_from_serialized_operator",
    evidenceClass: transformerSignals === 3 ? "DERIVED" : "OBSERVED_SERIALIZED_ARTIFACT",
    required: ["tensor_shapes", "serialized_storage"],
    assignments: assignments.filter((row) => /attention|mlp|embedding|normalization|output_projection/.test(row.role)),
    limitation: program.status === "not_serialized"
      ? "Weight roles are namespace-derived. Attention flow, block execution, cache behavior, and runtime order are not serialized by this artifact."
      : "Roles preserve serialized operators and values; higher-level attention semantics are not asserted for unmatched operations.",
  }));
  if (hasConvolution) profiles.push(profile("cnn", {
    status: "recognized_from_serialized_convolution_operator",
    evidenceClass: "OBSERVED_SERIALIZED_ARTIFACT",
    required: ["serialized_program_graph", "tensor_shapes"],
    assignments: assignments.filter((row) => ["convolution", "normalization", "pooling", "activation", "residual_merge_candidate"].includes(row.role)),
    limitation: "Residual merges remain candidates unless topology proves a long-range branch. Stage labels are not inferred from names alone.",
  }));
  if (hasEncoderNamespace && hasDecoderNamespace) profiles.push(profile("encoder_decoder", {
    status: "recognized_from_serialized_tensor_namespaces",
    evidenceClass: "DERIVED",
    required: ["serialized_storage"], assignments: assignments.filter((row) => /attention|embedding|normalization/.test(row.role)),
    limitation: "Encoder and decoder namespaces are observed, but cross-attention and execution flow require a serialized graph or separately bound source map.",
  }));
  if (roleSet.has("mixture_of_experts_weight")) profiles.push(profile("mixture_of_experts", {
    status: "candidate_from_serialized_expert_namespace", evidenceClass: "DERIVED", required: ["serialized_storage"],
    assignments: assignments.filter((row) => row.role === "mixture_of_experts_weight"),
    limitation: "Expert storage is identified by namespace. Routing, top-k selection, expert utilization, and runtime scheduling are not established.",
  }));

  return {
    status: profiles.length ? "recognized_profiles" : "no_profile_recognized",
    profiles,
    unassigned_operation_refs: program.operations.map((row) => row.id).filter((id) => !assignments.some((row) => row.subject_ref === id)),
    unassigned_storage_refs: storage.storage_objects.map((row) => row.id).filter((id) => !assignments.some((row) => row.subject_ref === id)),
    rule_contract: {
      id: "deepbom.model_profile_roles.v1",
      version: "1.0.0",
      principle: "Native operator identity has priority. Serialized names may label a subject but never create a program edge, call, or runtime order.",
    },
  };
}

function namedSemanticRole(name) {
  const text = String(name || "");
  return STORAGE_ROLE_RULES.find(([, pattern]) => pattern.test(text))?.[0] || null;
}
function assignment(subjectRef, role, basis, evidenceClass) { return { subject_ref: subjectRef, role, basis, evidence_class: evidenceClass }; }
function profile(id, { status, evidenceClass, required, assignments, limitation }) {
  return {
    id: `deepbom.${id}_profile.v1`, status, evidence_class: evidenceClass,
    required_capabilities: required, assignments: uniqueAssignments(assignments), groups: buildGroups(id, assignments),
    interpretation_boundary: limitation,
  };
}
function uniqueAssignments(rows) {
  const seen = new Set();
  return list(rows).filter((row) => { const key = `${row.subject_ref}\u0000${row.role}`; if (seen.has(key)) return false; seen.add(key); return true; });
}
function buildGroups(profileId, rows) {
  const byRole = new Map();
  for (const row of uniqueAssignments(rows)) {
    if (!byRole.has(row.role)) byRole.set(row.role, []);
    byRole.get(row.role).push(row.subject_ref);
  }
  return [...byRole.entries()].sort(([left], [right]) => compareCanonicalText(left, right)).map(([role, members]) => ({
    id: `profile-group:${profileId}:${role}`,
    role,
    member_refs: [...members].sort(),
    grouping_rule_id: "deepbom.model_profile_roles.v1",
    grouping_rule_version: "1.0.0",
  }));
}
