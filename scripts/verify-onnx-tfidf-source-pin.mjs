import { createHash } from "node:crypto";
import { ONNX_TFIDF_VECTORIZER_SOURCE } from "../web/lib/onnx-tfidf-vectorizer.js";
import { fetchPinnedBytes } from "./fetch-pinned-source.mjs";

const source = ONNX_TFIDF_VECTORIZER_SOURCE;
if (source.onnx_release !== "v1.21.0" || !/^[a-f0-9]{40}$/.test(source.onnx_commit || "")) {
  throw new Error("TfIdfVectorizer ONNX source is not pinned to v1.21.0 and a full commit hash.");
}
if (source.ort_release !== "v1.26.0" || !/^[a-f0-9]{40}$/.test(source.ort_commit || "")) {
  throw new Error("TfIdfVectorizer ORT source is not pinned to v1.26.0 and a full commit hash.");
}
const expectedRoles = new Set([
  "onnx_schema_and_shape", "onnx_reference", "onnx_backend_tests",
  "ort_cpu_kernel_header", "ort_cpu_kernel", "ort_cpu_tests",
]);
const documents = source.documents || [];
if (documents.length !== expectedRoles.size || new Set(documents.map((document) => document.role)).size !== documents.length
  || documents.some((document) => !expectedRoles.has(document.role))) {
  throw new Error("TfIdfVectorizer source ledger must contain six unique schema, reference, kernel, and test roles.");
}

const verified = [];
for (const document of documents) {
  const repository = document.role.startsWith("onnx_") ? "onnx/onnx" : "microsoft/onnxruntime";
  const commit = document.role.startsWith("onnx_") ? source.onnx_commit : source.ort_commit;
  const expectedPrefix = `https://raw.githubusercontent.com/${repository}/${commit}/`;
  if (!String(document.source_ref || "").startsWith(expectedPrefix)) throw new Error(`${document.role} is not bound to its pinned repository commit.`);
  if (!/^[a-f0-9]{64}$/.test(document.sha256 || "")) throw new Error(`${document.role} has an invalid SHA-256.`);
  const bytes = await fetchPinnedBytes(document.source_ref, { label: document.role });
  const digest = createHash("sha256").update(bytes).digest("hex");
  if (digest !== document.sha256) throw new Error(`${document.role} SHA-256 mismatch: ${digest} !== ${document.sha256}`);
  verified.push(`${document.role} ${bytes.byteLength} B ${digest}`);
}

console.log(`Pinned TfIdfVectorizer source verification passed at onnx/onnx@${source.onnx_commit} and microsoft/onnxruntime@${source.ort_commit}:\n${verified.map((item) => `  - ${item}`).join("\n")}`);
