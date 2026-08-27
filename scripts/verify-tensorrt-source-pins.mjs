import { createHash } from "node:crypto";
import { TENSORRT_SOURCE_METADATA } from "../web/lib/tensorrt-source-metadata.js";
import { TENSORRT_LLM_SOURCE_METADATA } from "../web/lib/tensorrt-llm-source-metadata.js";

const SHA256 = /^[a-f0-9]{64}$/;

const sources = [
  ...sourceRows(TENSORRT_SOURCE_METADATA.tensorrt),
  ...sourceRows(TENSORRT_SOURCE_METADATA.onnx_tensorrt_legacy_parser),
  ...sourceRows(TENSORRT_SOURCE_METADATA.onnxruntime_tensorrt_ep),
  ...sourceRows(TENSORRT_LLM_SOURCE_METADATA),
];

for (const source of sources) {
  if (!SHA256.test(source.sha256)) throw new Error(`Invalid pinned SHA-256 for ${source.repository}:${source.path}`);
  if (!source.commit || !source.url.includes(`/${source.commit}/`)) {
    throw new Error(`Source URL is not bound to commit ${source.commit || "<missing>"}: ${source.url}`);
  }
  const response = await fetch(rawGitHubUrl(source.url), {
    headers: { "user-agent": "DEEPBOM-source-pin-verifier/1" },
    redirect: "follow",
  });
  if (!response.ok) throw new Error(`Source retrieval failed (${response.status}) for ${source.url}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const actual = createHash("sha256").update(bytes).digest("hex");
  if (actual !== source.sha256) {
    throw new Error(`Source pin mismatch for ${source.repository}:${source.path}; expected ${source.sha256}, received ${actual}`);
  }
}

console.log(`TensorRT source pins verified against ${sources.length} immutable GitHub files.`);

function sourceRows(metadata) {
  const repository = String(metadata?.repository || "");
  const commit = String(metadata?.source_commit || metadata?.commit || "");
  if (!repository || !commit || !Array.isArray(metadata?.files) || !metadata.files.length) {
    throw new Error("TensorRT source metadata is incomplete.");
  }
  return metadata.files.map((row) => ({
    repository,
    commit,
    path: String(row.path || ""),
    sha256: String(row.sha256 || "").toLowerCase(),
    url: String(row.source_ref || row.url || ""),
  }));
}

function rawGitHubUrl(value) {
  const match = String(value).match(/^https:\/\/github\.com\/([^/]+)\/([^/]+)\/blob\/([^/]+)\/(.+)$/);
  if (!match) throw new Error(`Unsupported immutable GitHub source URL: ${value}`);
  return `https://raw.githubusercontent.com/${match[1]}/${match[2]}/${match[3]}/${match[4]}`;
}
