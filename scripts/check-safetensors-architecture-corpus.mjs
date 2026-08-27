import assert from "node:assert/strict";

import {
  readSafeTensorsArchitectureCorpus,
  receiptMatchesBaseline,
  validateSafeTensorsArchitectureCorpus,
} from "./safetensors-architecture-corpus-lib.mjs";

const manifest = await readSafeTensorsArchitectureCorpus();
validateSafeTensorsArchitectureCorpus(manifest, { requireBaselines: true });
assert.deepEqual(manifest.artifacts.map((row) => row.architecture_class).sort(), ["dense_decoder", "sparse_moe_decoder", "ssm_recurrent"]);
for (const artifact of manifest.artifacts) {
  const syntheticReceipt = {
    status: artifact.baseline.status,
    architecture_kind: artifact.baseline.architecture_kind,
    tensor_count: artifact.baseline.tensor_count,
    payload_byte_length: artifact.baseline.payload_byte_length,
    canonical_tensor_shape_mismatch_count: artifact.baseline.canonical_tensor_shape_mismatch_count,
    canonical_tensor_missing_count: artifact.baseline.canonical_tensor_missing_count,
    canonical_tensor_unexpected_count: artifact.baseline.canonical_tensor_unexpected_count,
    bundle_sha256: artifact.baseline.bundle_sha256,
    analysis_receipt_sha256: artifact.baseline.analysis_receipt_sha256,
    projection: artifact.baseline.projection,
  };
  assert.equal(receiptMatchesBaseline(syntheticReceipt, artifact), true, `${artifact.id}: baseline comparison is not deterministic`);
}

console.log("Three immutable SafeTensors Dense, Mixtral, and Mamba baselines are structurally valid and hash-bound.");
