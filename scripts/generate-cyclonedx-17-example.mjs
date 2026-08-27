import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

import { analyze_tflite_for_target, compute_delegation_repair, initSync } from "../pkg/tflite_wasm_audit.js";
import { buildDeploymentContractDocuments } from "../web/lib/report-export-contracts.js";
import { assertCycloneDx17 } from "./cyclonedx-17-schema.mjs";
import { REFERENCE_CONTRACT_ANALYZER_METADATA } from "./reference-contract-provenance.mjs";

const filename = "mobilenet_v2_1.0_224_quant.tflite";
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const analysis = analyze_tflite_for_target(bytes, filename, "android_mid_a55");
const repair = compute_delegation_repair(bytes, filename, "android_mid_a55");
analysis.model_sha256 = repair.artifact_sha256;
analysis.delegation_repair = repair;
const set = buildDeploymentContractDocuments(analysis, {
  generatedAt: "2026-07-29T00:00:00.000Z",
  hash: repair.artifact_sha256,
  fileSizeBytes: bytes.byteLength,
  analyzerMetadata: REFERENCE_CONTRACT_ANALYZER_METADATA,
});
const outputDir = "reference/cyclonedx/1.7/examples/mobilenet-v2-quant";
mkdirSync(outputDir, { recursive: true });
for (const [key, filenameOut] of Object.entries(set.files)) {
  const documentKey = {
    cyclonedx: "cyclonedx_evidence",
    cyclonedx20Preview: "cyclonedx_2_0_parameter_contract_preview",
    artifactEnvelope: "artifact_evidence_envelope",
    interfaceContracts: "interface_contract_ledger",
    formulation: "observed_formulation",
    runtime: "runtime_requirement_manifest",
    missingFields: "missing_provenance_field_specification",
  }[key];
  const document = set.documents[documentKey];
  if (filenameOut.endsWith(".cdx.json") && key !== "cyclonedx20Preview") assertCycloneDx17(document, filenameOut);
  writeFileSync(path.join(outputDir, filenameOut), `${JSON.stringify(document, null, 2)}\n`);
}
console.log(`Wrote reproducible public TFLite contract example to ${outputDir}`);
