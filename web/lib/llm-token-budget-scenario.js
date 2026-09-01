import { compareLlmMemoryCapacity, LLM_STATIC_RESIDENCY_ASSUMPTION } from "./llm-memory-feasibility.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const LLM_TOKEN_BUDGET_SCENARIO_SCHEMA = "deepbom.llm_token_budget_scenario.v1";

const SHA256 = /^[a-f0-9]{64}$/;
const STATE_WIDTHS = new Set([8, 16, 32]);

function positiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${label} must be a positive safe integer.`);
  return value;
}

function nonNegativeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${label} must be a non-negative safe integer.`);
  return value;
}

function exactFrom(value) {
  if (value && typeof value === "object" && /^\d+$/.test(String(value.decimal || ""))) return BigInt(value.decimal);
  if (typeof value === "string" && /^\d+$/.test(value)) return BigInt(value);
  if (typeof value === "bigint" && value >= 0n) return value;
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  return null;
}

function exact(value) {
  return { value: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null, decimal: String(value) };
}

function scenarioIdentity(analysis, source) {
  const artifactSha256 = String(analysis?.model_sha256 || analysis?.artifact_bundle?.model_source_sha256 || "").toLowerCase();
  return {
    artifact_sha256: SHA256.test(artifactSha256) ? artifactSha256 : null,
    source,
  };
}

export function buildLlmTokenBudgetScenario(analysis = {}, {
  textTokens,
  imageCount = 0,
  tokensPerImage = null,
  batchSize = 1,
  stateStorageBits = 16,
  memoryCapacityBytes = null,
  source = "declared_scenario",
} = {}) {
  const textTokenCount = positiveInteger(textTokens, "textTokens");
  const images = nonNegativeInteger(imageCount, "imageCount");
  const imageTokenWidth = images > 0 ? positiveInteger(tokensPerImage, "tokensPerImage") : null;
  if (images === 0 && tokensPerImage != null) throw new Error("tokensPerImage requires imageCount greater than zero.");
  const batch = positiveInteger(batchSize, "batchSize");
  if (!STATE_WIDTHS.has(stateStorageBits)) throw new Error("stateStorageBits must be 8, 16, or 32.");
  const capacity = memoryCapacityBytes == null ? null : exactFrom(memoryCapacityBytes);
  if (memoryCapacityBytes != null && (capacity == null || capacity <= 0n)) {
    throw new Error("memoryCapacityBytes must be a positive exact integer.");
  }

  const imageTokens = BigInt(images) * BigInt(imageTokenWidth || 0);
  const totalTokens = BigInt(textTokenCount) + imageTokens;
  if (totalTokens > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("The total token budget exceeds the safe integer range.");

  const contract = analysis?.on_device_llm || {};
  const kvElementsPerToken = exactFrom(contract?.state?.kv_projection?.elements_per_token_per_batch);
  const recurrentElements = exactFrom(contract?.state?.recurrent_projection?.recurrent_state_elements_all_layers_per_batch);
  const serializedWeightBytes = exactFrom(contract?.storage?.serialized_tensor_bytes_decimal);
  const serializedContext = Number.isSafeInteger(contract?.architecture?.context_length)
    && contract.architecture.context_length > 0 ? contract.architecture.context_length : null;
  const contextAssessment = serializedContext == null
    ? "serialized_context_unbound"
    : totalTokens <= BigInt(serializedContext) ? "within_serialized_context" : "exceeds_serialized_context";

  const stateElements = kvElementsPerToken == null
    ? null
    : (kvElementsPerToken * totalTokens + (recurrentElements || 0n)) * BigInt(batch);
  const stateBytes = stateElements == null ? null : stateElements * BigInt(stateStorageBits / 8);
  const staticLowerBound = serializedWeightBytes == null || stateBytes == null
    ? null : serializedWeightBytes + stateBytes;
  const capacityComparison = compareLlmMemoryCapacity(staticLowerBound, capacity);
  const body = {
    schema: LLM_TOKEN_BUDGET_SCENARIO_SCHEMA,
    status: kvElementsPerToken == null
      ? "not_assessable_kv_contract_unbound"
      : contextAssessment === "exceeds_serialized_context"
        ? "derived_scenario_exceeds_serialized_context"
        : "assessed_conditional_token_budget",
    evidence_class: kvElementsPerToken == null
      ? "DECLARED/NOT_ASSESSABLE"
      : "DECLARED/DERIVED_CONDITIONAL_SCENARIO",
    identity: scenarioIdentity(analysis, source),
    token_budget: {
      text_tokens: textTokenCount,
      image_count: images,
      tokens_per_image: imageTokenWidth,
      image_tokens: exact(imageTokens),
      total_context_tokens: exact(totalTokens),
      batch_size: batch,
      state_storage_bits: stateStorageBits,
      evidence_class: "DECLARED",
    },
    serialized_context_contract: {
      context_length: serializedContext,
      assessment: contextAssessment,
      evidence_class: serializedContext == null ? "NOT_ASSESSABLE" : "OBSERVED_OR_SOURCE_BACKED",
    },
    state_projection: {
      state_kind: kvElementsPerToken != null && recurrentElements != null
        ? "hybrid_kv_ssm" : kvElementsPerToken != null ? "transformer_kv" : "not_assessable",
      kv_elements_per_token_per_batch: kvElementsPerToken == null ? null : exact(kvElementsPerToken),
      recurrent_elements_per_batch: recurrentElements == null ? null : exact(recurrentElements),
      logical_state_elements: stateElements == null ? null : exact(stateElements),
      logical_state_bytes: stateBytes == null ? null : exact(stateBytes),
      evidence_class: stateBytes == null ? "NOT_ASSESSABLE" : "DERIVED_CONDITIONAL_SCENARIO",
    },
    memory_feasibility: {
      schema: "deepbom.llm_token_budget_memory_scenario.v1",
      status: capacityComparison.status,
      evidence_class: staticLowerBound == null
        ? "NOT_ASSESSABLE" : "OBSERVED_STORAGE_PLUS_DERIVED_CONDITIONAL_STATE",
      capacity_scope: "single_aggregate_primary_memory_budget",
      residency_assumption: LLM_STATIC_RESIDENCY_ASSUMPTION,
      serialized_weight_floor_bytes: serializedWeightBytes == null ? null : exact(serializedWeightBytes),
      logical_state_bytes: stateBytes == null ? null : exact(stateBytes),
      static_lower_bound_bytes: staticLowerBound == null ? null : exact(staticLowerBound),
      declared_capacity_bytes: capacity == null ? null : exact(capacity),
      deficit_bytes: capacityComparison.deficit_bytes,
      headroom_after_lower_bound_bytes: capacityComparison.headroom_after_lower_bound_bytes,
      fit_claim: capacityComparison.fit_claim,
    },
    boundary: "Text and image-token counts are user-declared scenario inputs. Image preprocessing, projector output cardinality, special-token insertion, runtime allocation, backend lowering, kernel choice, latency, task accuracy, and device fit are not inferred. A lower bound at or below capacity does not prove fit.",
  };
  return { ...body, scenario_sha256: sha256TextHex(canonicalJson(body)) };
}

export function validateLlmTokenBudgetScenario(analysis, scenario) {
  if (!scenario || scenario.schema !== LLM_TOKEN_BUDGET_SCENARIO_SCHEMA) return ["llm_token_budget_schema_invalid"];
  try {
    const expected = buildLlmTokenBudgetScenario(analysis, {
      textTokens: scenario.token_budget?.text_tokens,
      imageCount: scenario.token_budget?.image_count,
      tokensPerImage: scenario.token_budget?.tokens_per_image,
      batchSize: scenario.token_budget?.batch_size,
      stateStorageBits: scenario.token_budget?.state_storage_bits,
      memoryCapacityBytes: scenario.memory_feasibility?.declared_capacity_bytes,
      source: scenario.identity?.source,
    });
    return canonicalJson(expected) === canonicalJson(scenario) ? [] : ["llm_token_budget_recomputation_mismatch"];
  } catch {
    return ["llm_token_budget_recomputation_failed"];
  }
}
