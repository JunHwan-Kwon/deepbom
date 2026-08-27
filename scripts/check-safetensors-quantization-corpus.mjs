import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { canonicalJson } from "../web/lib/report-utils.js";
import { SAFETENSORS_QUANTIZATION_SOURCES } from "../web/lib/safetensors-quantization-contract.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("SafeTensors quantization corpus check");
const manifest = JSON.parse(await readFile("corpus/safetensors-quantization-contract-corpus.v1.json", "utf8"));
const { ledger_sha256: ledger, ...body } = manifest;
expectEqual(manifest.schema, "deepbom.safetensors_quantization_contract_corpus.v1", "Corpus schema should be stable.");
expectEqual(manifest.artifact_count, 2, "The bounded corpus should retain one AWQ and one GPTQ source.");
expectEqual(ledger, sha256(canonicalJson(body)), "Corpus ledger should reproduce.");
expectEqual(new Set(manifest.artifacts.map((row) => row.method)).size, 2, "AWQ and GPTQ should both be represented.");
for (const artifact of manifest.artifacts) {
  expect(/^[0-9a-f]{40}$/.test(artifact.revision), `${artifact.id}: immutable revision should be complete.`);
  expect(/^[0-9a-f]{64}$/.test(artifact.model_sha256), `${artifact.id}: full model SHA-256 should be present.`);
  expectEqual(artifact.measurement.status, "assessed", `${artifact.id}: structural contract should pass.`);
  expectEqual(artifact.measurement.invalid_module_count, 0, `${artifact.id}: no module should violate the pinned layout.`);
  expectEqual(artifact.measurement.valid_module_count, artifact.measurement.module_count, `${artifact.id}: module denominator should conserve.`);
  expectEqual(artifact.measurement.logical_weight_element_count, artifact.measurement.packed_weight_code_capacity, `${artifact.id}: packed weight capacity should conserve logical weights.`);
  expect(artifact.measurement.module_count > 100, `${artifact.id}: the evidence must not collapse to a toy module.`);
  expectEqual(artifact.measurement.payload_value_scan, "not_performed_header_only_corpus", `${artifact.id}: payload limitation should remain explicit.`);
  expectEqual(artifact.measurement.source.sha256, SAFETENSORS_QUANTIZATION_SOURCES[artifact.method].sha256, `${artifact.id}: implementation source digest should remain pinned.`);
  for (const config of artifact.config_files) {
    const bytes = await readFile(`corpus/safetensors-quantization-contract-corpus/${artifact.id}.${config.path}`);
    expectEqual(bytes.length, config.size_bytes, `${artifact.id}/${config.path}: committed source config size should match.`);
    expectEqual(sha256(bytes), config.sha256, `${artifact.id}/${config.path}: committed source config hash should match.`);
  }
}

done("SafeTensors public AWQ/GPTQ header corpus passed (identity, layout, conservation, and evidence boundary)." );

function sha256(value) { return createHash("sha256").update(value).digest("hex"); }
