import { readFileSync } from "node:fs";

import { analyze_tflite_for_target, compute_delegation_repair, initSync } from "../pkg/tflite_wasm_audit.js";
import { buildDeploymentContractDocuments } from "../web/lib/report-export-contracts.js";
import { assertCycloneDx17 } from "./cyclonedx-17-schema.mjs";
import { REFERENCE_CONTRACT_ANALYZER_METADATA, REFERENCE_GENERATED_AT, writeReferenceSet, REFERENCE_DOCUMENT_KEYS } from "./reference-contract-provenance.mjs";

const filename = "mobilenet_v2_1.0_224_quant.tflite";
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const analysis = analyze_tflite_for_target(bytes, filename, "android_mid_a55");
const repair = compute_delegation_repair(bytes, filename, "android_mid_a55");
analysis.model_sha256 = repair.artifact_sha256;
analysis.delegation_repair = repair;
const set = buildDeploymentContractDocuments(analysis, {
  generatedAt: REFERENCE_GENERATED_AT,
  hash: repair.artifact_sha256,
  fileSizeBytes: bytes.byteLength,
  analyzerMetadata: REFERENCE_CONTRACT_ANALYZER_METADATA,
});
for (const [key, file] of Object.entries(set.files)) {
  if (file.endsWith(".cdx.json")) assertCycloneDx17(set.documents[REFERENCE_DOCUMENT_KEYS[key]], file);
}
const {record}=writeReferenceSet(set);
console.log(`Wrote producer-bound reference ${record.directory} (${record.status}); historical references preserved.`);
