// Conversation transport revisions are independent of the canonical Evidence IR.
export const CHATGPT_RESULT_V1 = "deepbom.chatgpt_analysis_result.v1";
export const CHATGPT_RESULT_V2 = "deepbom.chatgpt_analysis_result.v2";

export function isConversationProducerVersion(value) {
  // A cached widget may have a different producer version from the live server.
  // Preserve that version as declared provenance, not proof of authenticity.
  return typeof value === "string" && value.length <= 64
    && /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(value);
}

export function conversationTransport(result, schema) {
  if (![CHATGPT_RESULT_V1, CHATGPT_RESULT_V2].includes(result?.schema)
    || ![CHATGPT_RESULT_V1, CHATGPT_RESULT_V2].includes(schema)) {
    throw new Error("Unsupported ChatGPT result transport.");
  }
  if (schema === CHATGPT_RESULT_V1 && Object.hasOwn(result, "provenance_summary")) {
    throw new Error("Provenance evidence requires the v2 conversation transport; it cannot be omitted or renamed.");
  }
  return { ...result, schema };
}

function schemaRejection(error) {
  const message = String(error?.message || "");
  return /input.*validat|schema.*(?:invalid|validat|mismatch|expected|constant)|(?:invalid|expected).*schema|must be equal to constant/i.test(message);
}

async function deliver(openai, result) {
  const response = await openai.callTool("deepbom_publish_analysis", { result });
  if (response?.isError || response?.error) {
    throw new Error(response.error?.message || response.content?.filter(row => row.type === "text").map(row => row.text).join(" ") || "ChatGPT did not accept the analysis result.");
  }
  const returned = response?.structuredContent;
  if (!returned || returned.schema !== result.schema
    || returned.analyzer_version !== result.analyzer_version
    || returned.artifact?.sha256 !== result.artifact.sha256
    || returned.model_summary?.model_ir_sha256 !== result.model_summary.model_ir_sha256
    || returned.provenance_summary?.provenance_ir_sha256 !== result.provenance_summary?.provenance_ir_sha256) {
    throw new Error("ChatGPT did not acknowledge the matching analysis result. Retry reporting the result.");
  }
  return returned;
}

export async function publishConversationResult(openai, result) {
  if (Object.hasOwn(result, "provenance_summary")) {
    try {
      return await deliver(openai, conversationTransport(result, CHATGPT_RESULT_V2));
    } catch (error) {
      if (!schemaRejection(error)) throw error;
      throw new Error("This connection has not received the updated metadata reporting tool yet. The model analysis remains available. Save the Provenance IR JSON from this panel, or retry after the connection's tool update is available.");
    }
  }
  // Published hosts may still enforce the historical v1 schema. Hosts that
  // already cached the v2-only definition are supported by a bounded retry.
  // Do not retry network, authentication, or acknowledgement failures.
  try {
    return await deliver(openai, conversationTransport(result, CHATGPT_RESULT_V1));
  } catch (error) {
    if (!schemaRejection(error)) throw error;
    return deliver(openai, conversationTransport(result, CHATGPT_RESULT_V2));
  }
}
