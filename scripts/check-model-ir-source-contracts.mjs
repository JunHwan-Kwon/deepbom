import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(".");
const manifest = JSON.parse(await readFile(path.join(root, "config/model-ir-upstream-sources.v1.json"), "utf8"));
assert.equal(manifest.schema, "deepbom.model_ir_upstream_sources.v1");
assert(manifest.sources.length >= 8, "Model IR upstream source manifest must cover every current adapter family.");
const ids = new Set();
const repositories = new Set();
for (const source of manifest.sources) {
  assert.match(source.id, /^[a-z0-9_]+$/);
  assert(!ids.has(source.id), `duplicate source id ${source.id}`);
  ids.add(source.id);
  assert.match(source.repository, /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/);
  assert.match(source.commit, /^[a-f0-9]{40}$/);
  assert.match(source.authority, /^upstream_/);
  assert(source.verifier.startsWith("scripts/") && source.verifier.endsWith(".mjs"));
  await access(path.join(root, source.verifier));
  if (source.files) for (const file of source.files) {
    assert.match(file.path, /^(?!\/)(?!.*\.\.\/)[A-Za-z0-9_./-]+$/);
    assert.match(file.sha256, /^[a-f0-9]{64}$/);
    assert(Array.isArray(file.markers) && file.markers.length > 0);
  }
  repositories.add(`${source.repository}@${source.commit}`);
}
for (const required of ["onnx", "tensorflow_tflite", "tensorflow_graphdef_savedmodel", "llama_cpp_gguf", "safetensors", "coremltools", "executorch", "hdf5", "keras", "pytorch"]) assert(ids.has(required), `missing source family ${required}`);
assert(Array.isArray(manifest.non_authorities) && manifest.non_authorities.length >= 3);
console.log(`Model IR source contracts passed (${manifest.sources.length} pinned source roles; ${repositories.size} immutable repository revisions; file digests remain enforced by named verifiers).`);
