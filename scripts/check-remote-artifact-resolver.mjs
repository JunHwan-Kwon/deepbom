import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

import { parseArtifactSource, resolveArtifactSource } from "../bin/remote-artifact-resolver.mjs";
import { resolveHuggingFaceOnnxExternalDataClosure, resolveHuggingFaceSafeTensorsClosure } from "../bin/remote-artifact-closure.mjs";
import { finalizeArtifactSet, validateArtifactSet } from "../web/lib/artifact-set.js";
import { buildOnnxExternalDataStructureBinding } from "../web/lib/onnx-external-data-structure-binding.js";

const bytes = Buffer.from("deterministic remote model fixture\n", "utf8");
const digest = createHash("sha256").update(bytes).digest("hex");
const commit = "1".repeat(40);
const cache = await mkdtemp(path.join(tmpdir(), "deepbom-remote-resolver-"));

try {
  const hf = parseArtifactSource(`hf://owner/repo@${commit}/models/test.onnx`, digest);
  assert.equal(hf.kind, "huggingface");
  assert.equal(hf.immutability.value, commit);
  assert.throws(() => parseArtifactSource("hf://owner/repo@main/model.onnx"), /40-hex-commit/);
  assert.throws(() => parseArtifactSource("https://example.test/model.onnx"), /requires #sha256/);
  assert.throws(() => parseArtifactSource("kaggle://owner/model/framework/variant@1/file.onnx"), /not yet available/);

  let fetchCount = 0;
  const first = await resolveArtifactSource(hf.canonical_locator, {
    cacheDir: cache,
    expectedSha256: digest,
    fetchImpl: async () => { fetchCount += 1; return new Response(bytes, { status: 200, headers: { "content-length": String(bytes.length) } }); },
    progress: null,
  });
  assert.equal(fetchCount, 1);
  assert.equal(first.acquisition.file.sha256, digest);
  assert.equal(first.acquisition.transport.authorization_persisted, false);
  assert.deepEqual(await readFile(first.path), bytes);

  const cached = await resolveArtifactSource(hf.canonical_locator, {
    cacheDir: cache,
    expectedSha256: digest,
    offline: true,
    fetchImpl: async () => { throw new Error("offline cache must not fetch"); },
    progress: null,
  });
  assert.equal(cached.path, first.path);
  assert.equal(cached.acquisition.file.sha256, digest);

  const corrupted = Buffer.from(bytes);
  corrupted[0] ^= 0xff;
  await writeFile(cached.path, corrupted);
  await assert.rejects(resolveArtifactSource(hf.canonical_locator, {
    cacheDir: cache,
    expectedSha256: digest,
    offline: true,
    fetchImpl: async () => { throw new Error("offline cache must not fetch"); },
    progress: null,
  }), /unavailable in offline cache/);
  await writeFile(cached.path, bytes);

  const signed = parseArtifactSource(`https://example.test/model.onnx?secret=redacted#sha256=${digest}`);
  assert.equal(signed.request_url.includes("secret=redacted"), true, "signed query remains available only to the request");
  assert.equal(signed.canonical_locator.includes("secret"), false, "query credentials must not enter evidence");

  let gcsCalls = 0;
  const gcs = await resolveArtifactSource("gs://bucket-name/path/model.onnx#generation=42", {
    cacheDir: path.join(cache, "gcs"),
    expectedSha256: digest,
    fetchImpl: async (url) => {
      gcsCalls += 1;
      if (String(url).startsWith("https://storage.googleapis.com/storage/v1/")) return Response.json({ bucket: "bucket-name", name: "path/model.onnx", generation: "42" });
      return new Response(bytes, { status: 200, headers: { "content-length": String(bytes.length), "x-goog-generation": "42" } });
    },
    progress: null,
  });
  assert.equal(gcsCalls, 2);
  assert.equal(gcs.acquisition.transport.gcs_generation_verified, true);

  const indexBytes = Buffer.from(JSON.stringify({ weight_map: { "layer.0.weight": "model-00001-of-00002.safetensors", "layer.1.weight": "model-00002-of-00002.safetensors" } }));
  const indexDigest = createHash("sha256").update(indexBytes).digest("hex");
  const shardOne = Buffer.from("shard-one");
  const shardTwo = Buffer.from("shard-two");
  const indexSpec = `hf://owner/repo@${commit}/weights/model.safetensors.index.json`;
  const closureFetch = async (url) => {
    const name = new URL(String(url)).pathname.split("/").pop();
    const payload = name === "model.safetensors.index.json" ? indexBytes
      : name === "model-00001-of-00002.safetensors" ? shardOne
        : name === "model-00002-of-00002.safetensors" ? shardTwo : null;
    return payload ? new Response(payload, { status: 200, headers: { "content-length": String(payload.length) } })
      : new Response("missing", { status: 404 });
  };
  const indexPrimary = await resolveArtifactSource(indexSpec, {
    cacheDir: path.join(cache, "closure"), expectedSha256: indexDigest, fetchImpl: closureFetch, progress: null,
  });
  const sharded = await resolveHuggingFaceSafeTensorsClosure(indexSpec, indexPrimary, {
    cacheDir: path.join(cache, "closure"), expectedSha256: indexDigest, fetchImpl: closureFetch, progress: null,
  });
  assert.equal(sharded.closure.kind, "huggingface_safetensors_shards");
  assert.equal(sharded.closure.members.length, 3);
  assert.equal(sharded.closure.materialization, "content_addressed_member_map_no_copy");
  assert.equal(sharded.path, indexPrimary.path, "remote closure must not duplicate the primary artifact");
  const virtualShard = sharded.virtual_bundle_members.find((row) => row.path.endsWith("model-00002-of-00002.safetensors"));
  assert.equal((await stat(virtualShard.resolved_path)).size, shardTwo.length);
  assert.equal(sharded.virtual_bundle_members.every((row) => row.resolved_path.includes(`${path.sep}sha256${path.sep}`)), true,
    "virtual bundle members must reuse content-addressed cache files");
  const structureBinding = buildOnnxExternalDataStructureBinding({ onnx_external_data: { tensors: [{
    tensor_name: "weight", normalized_location: "model.onnx_data", offset: 4, length: 8, range_end: 12,
    expected_payload_bytes: 8, dtype: "FLOAT32", shape: [2],
  }] } }, [{ model_relative_path: "model.onnx_data", sha256: digest, byte_length: { decimal: "12", number: 12 } }]);
  assert.equal(structureBinding.range_conservation_status, "complete");
  assert.equal(structureBinding.numerical_payload_decode, "not_assessed_scan_policy_structure");
  assert.equal(structureBinding.declared_payload_bytes.decimal, "8");
  assert.equal(structureBinding.unique_payload_bytes.decimal, "8");
  assert.equal(structureBinding.overlapping_reference_bytes.decimal, "0");
  assert.equal(structureBinding.declared_element_count.decimal, "2");
  assert.deepEqual(structureBinding.encoding_inventory[0], {
    dtype: "FLOAT32", tensor_count: 1,
    element_count: { decimal: "2", number: 2 },
    declared_payload_bytes: { decimal: "8", number: 8 },
    effective_bits_per_element: 32,
  });
  assert.throws(() => buildOnnxExternalDataStructureBinding({ onnx_external_data: { tensors: [{
    tensor_name: "weight", normalized_location: "model.onnx_data", offset: 4, length: 9, range_end: 13,
    expected_payload_bytes: 9, dtype: "FLOAT32", shape: [2],
  }] } }, [{ model_relative_path: "model.onnx_data", sha256: digest, byte_length: { decimal: "12", number: 12 } }]), /range is invalid/);
  await assert.rejects(
    resolveHuggingFaceOnnxExternalDataClosure(`hf://owner/repo@${commit}/model.onnx`, indexPrimary, ["../escape.bin"], {
      cacheDir: path.join(cache, "closure"), fetchImpl: closureFetch, progress: null,
    }),
    /safe relative path/,
  );

  await assert.rejects(resolveArtifactSource(`https://example.test/model.onnx#sha256=${"0".repeat(64)}`, {
    cacheDir: path.join(cache, "mismatch"),
    fetchImpl: async () => new Response(bytes, { status: 200, headers: { "content-length": String(bytes.length) } }),
    progress: null,
  }), /SHA-256 mismatch/);

  await assert.rejects(resolveArtifactSource(`https://example.test/stalled.onnx#sha256=${digest}`, {
    cacheDir: path.join(cache, "header-timeout"),
    headerTimeoutMs: 15,
    fetchImpl: async (_url, options) => new Promise((_resolve, reject) => {
      options.signal.addEventListener("abort", () => reject(options.signal.reason), { once: true });
    }),
    progress: null,
  }), /response headers before the timeout/);

  await assert.rejects(resolveArtifactSource(`https://example.test/idle.onnx#sha256=${digest}`, {
    cacheDir: path.join(cache, "idle-timeout"),
    idleTimeoutMs: 15,
    downloadTimeoutMs: 1000,
    fetchImpl: async (_url, options) => new Response(new ReadableStream({
      start(controller) {
        options.signal.addEventListener("abort", () => controller.error(options.signal.reason), { once: true });
      },
    }), { status: 200 }),
    progress: null,
  }), /no progress before the idle timeout/);

  const artifactSet = finalizeArtifactSet({
    schema: "deepbom.artifact_set.v1",
    evidence_class: "OBSERVED_ACQUISITION",
    source: first.acquisition.source,
    files: [{ role: "primary", path: "test.onnx", sha256: digest, byte_length: { decimal: String(bytes.length), number: bytes.length } }],
    trust: { remote_code_execution: "forbidden", pickle_execution: "forbidden", model_code_execution: "forbidden" },
  });
  assert.doesNotThrow(() => validateArtifactSet(artifactSet));
  const tampered = structuredClone(artifactSet);
  tampered.files[0].sha256 = "0".repeat(64);
  assert.throws(() => validateArtifactSet(tampered), /SHA-256/);
} finally {
  await rm(cache, { recursive: true, force: true });
}

console.log("Remote artifact resolver checks passed (immutable HF/GCS/HTTPS syntax, bounded streaming identity, no-copy closure, cache/offline replay, credential redaction, mismatch failure, and artifact-set integrity).");
