import { createHash } from "node:crypto";
import { ONNX_CONTAINER_VALUE_SOURCE } from "../web/lib/onnx-container-inference.js";
import { fetchPinnedBytes } from "./fetch-pinned-source.mjs";

const commit = ONNX_CONTAINER_VALUE_SOURCE.commit;
if (!/^[a-f0-9]{40}$/.test(commit || "")) throw new Error("ONNX container-value source commit is not pinned to a full commit hash.");
if (ONNX_CONTAINER_VALUE_SOURCE.release !== "v1.21.0") throw new Error("ONNX container-value source release drifted from v1.21.0.");

const documents = ONNX_CONTAINER_VALUE_SOURCE.documents || [];
if (documents.length !== 4 || new Set(documents.map((source) => source.role)).size !== documents.length) {
  throw new Error("ONNX container-value source ledger must contain four uniquely named documents.");
}

const verified = [];
for (const source of documents) {
  const expectedPrefix = `https://raw.githubusercontent.com/onnx/onnx/${commit}/`;
  if (!String(source.source_ref || "").startsWith(expectedPrefix)) {
    throw new Error(`${source.role} is not bound to the pinned ONNX commit.`);
  }
  if (!/^[a-f0-9]{64}$/.test(source.sha256 || "")) throw new Error(`${source.role} has an invalid SHA-256.`);
  const bytes = await fetchPinnedBytes(source.source_ref, { label: "ONNX" });
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (sha256 !== source.sha256) throw new Error(`${source.role} SHA-256 mismatch: ${sha256} !== ${source.sha256}`);
  verified.push(`${source.role} ${bytes.byteLength} B ${sha256}`);
}

console.log(`Pinned ONNX Sequence/Optional source verification passed at onnx/onnx@${commit}:\n${verified.map((item) => `  - ${item}`).join("\n")}`);
