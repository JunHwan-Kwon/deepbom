import { createHash } from "node:crypto";
import { ONNX_EXTENDED_SHAPE_SOURCE } from "../web/lib/onnx-extended-shape-inference.js";
import { fetchPinnedBytes } from "./fetch-pinned-source.mjs";

const commit = ONNX_EXTENDED_SHAPE_SOURCE.commit;
if (!/^[a-f0-9]{40}$/.test(commit || "")) throw new Error("ONNX control-flow source commit is not pinned to a full commit hash.");
if (ONNX_EXTENDED_SHAPE_SOURCE.release !== "v1.21.0") throw new Error("ONNX control-flow source release drifted from v1.21.0.");

const expectedRoles = new Set([
  "function_proto_contract",
  "control_flow_shape_inference",
  "current_control_flow_schema",
  "historical_control_flow_schema",
]);
const documents = ONNX_EXTENDED_SHAPE_SOURCE.documents || [];
if (documents.length !== expectedRoles.size || new Set(documents.map((source) => source.role)).size !== documents.length
  || documents.some((source) => !expectedRoles.has(source.role))) {
  throw new Error("ONNX control-flow source ledger must contain the four required uniquely named documents.");
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

console.log(`Pinned ONNX FunctionProto/control-flow source verification passed at onnx/onnx@${commit}:\n${verified.map((item) => `  - ${item}`).join("\n")}`);
