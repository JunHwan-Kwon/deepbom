import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";

import Ajv2020 from "ajv/dist/2020.js";

import { validateOnnxContractConflictCapsule } from "../web/lib/onnx-contract-conflict.js";
import { canonicalJson } from "../web/lib/report-utils.js";

const bytes = await readFile("corpus/onnx-contract-conflict-corpus.v1.json.gz");
const corpus = JSON.parse(gunzipSync(bytes));
const schema = JSON.parse(await readFile("docs/schemas/deepbom-onnx-contract-conflict-capsule-v1.schema.json", "utf8"));
const validateSchema = new Ajv2020({ strict: true, allErrors: true }).compile(schema);
assert.equal(corpus.schema, "deepbom.onnx_contract_conflict_corpus.v1");
const { corpus_sha256: corpusSha256, ...body } = corpus;
assert.equal(corpusSha256, sha256(Buffer.from(canonicalJson(body))), "corpus self-hash");
assert.deepEqual(corpus.hash_contract.excluded_pointers, ["/corpus_sha256"]);
assert.equal(corpus.artifact_count, corpus.artifacts.length);
assert.equal(new Set(corpus.artifacts.map((row) => row.artifact_sha256)).size, corpus.artifact_count);
const sourceBytes = await readFile(corpus.source_sweep.path);
assert.equal(corpus.source_sweep.sha256, sha256(sourceBytes), "source sweep SHA-256");
for (const row of corpus.artifacts) {
  assert.equal(row.artifact_sha256, row.capsule.artifact.sha256);
  assert.equal(validateSchema(row.capsule), true, JSON.stringify(validateSchema.errors));
  validateOnnxContractConflictCapsule(row.capsule);
  assert.equal(row.capsule.status, "INVALID_CONTRACT");
  assert.match(row.repository_id, /^[^/]+\/[^/]+$/);
  assert.match(row.revision, /^[a-f0-9]{40}$/);
  assert.match(row.source_analysis_sha256, /^[a-f0-9]{64}$/);
  assert(["pass", "fail", "crash", "timeout"].includes(row.official_onnx_reference?.checker?.status));
  assert(["pass", "fail", "crash", "timeout"].includes(row.official_onnx_reference?.strict_shape_inference?.status));
}
const aggregate = corpus.aggregate;
assert.equal(aggregate.unconditional_root_conflict_count, 6);
assert.equal(aggregate.declaration_root_conflict_count, 4);
assert.equal(aggregate.semantic_root_conflict_count, 2);
assert.equal(aggregate.condition_bound_invalid_variant_count, 543);
assert.equal(aggregate.invalid_node_output_count, 1995);
assert.equal(aggregate.conditionally_invalid_node_output_count, 400);
assert.equal(aggregate.downstream_blocked_node_count, 1901);
assert.equal(aggregate.blocked_mac_row_count, 213);
assert.deepEqual(aggregate.blocked_mac_op_histogram, [
  { name: "Conv", count: 107 },
  { name: "MatMul", count: 88 },
  { name: "ConvTranspose", count: 17 },
  { name: "LSTM", count: 1 },
]);
assert.equal(aggregate.unresolved_root_reference_count, 0, "every affected or blocked row must trace to a root conflict");
console.log(`ONNX conflict corpus passed (${corpus.artifact_count} hash-pinned artifacts, 6 roots, 543 conditional variants, 1,901 downstream nodes, and 213 blocked MAC rows).`);

function sha256(value) { return createHash("sha256").update(value).digest("hex"); }
