import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { externalTensor, model } from "../onnx-proto-fixture.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const fixtureRoot = path.join(root, "corpus", "external-review", "fixtures", "onnx-invalid-external-range");
const modelBytes = model(
  [],
  [externalTensor("weight", 1, [1, 2], [["location", "weights.bin"], ["offset", "20"], ["length", "8"]])],
  [],
  [],
  13,
);
const sidecarBytes = Buffer.alloc(24, 0x3f);
await mkdir(fixtureRoot, { recursive: true });
await writeFile(path.join(fixtureRoot, "model.onnx"), modelBytes);
await writeFile(path.join(fixtureRoot, "weights.bin"), sidecarBytes);
const manifest = {
  schema: "deepbom.external_review_onnx_external_range_fixture.v1",
  files: [
    file("model", "model.onnx", modelBytes),
    file("external_data", "weights.bin", sidecarBytes),
  ],
  expected: {
    external_data_status: "payload_verification_failed",
    range_out_of_bounds_count: 1,
    verified_payload_count: 0,
  },
  mutation_contract: {
    declared_offset: "20",
    declared_length: "8",
    sidecar_byte_length: "24",
    declared_end_exclusive: "28",
  },
  interpretation_boundary: "The model and sidecar are structurally readable, but the serialized external-data range exceeds the selected sidecar. No tensor value may be decoded from that range.",
};
await writeFile(path.join(fixtureRoot, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
console.log(`Wrote ${path.relative(root, fixtureRoot)}.`);

function file(role, name, bytes) {
  return { role, path: name, bytes: bytes.byteLength, sha256: createHash("sha256").update(bytes).digest("hex") };
}
