import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const fixtureRoot = path.join(root, "corpus", "external-review", "fixtures");
const sourcePath = path.join(fixtureRoot, "quant-scale-risk.tflite");
const outputPath = path.join(fixtureRoot, "quant-scale-risk-1000x.tflite");
const manifestPath = path.join(fixtureRoot, "quant-scale-risk-1000x.manifest.json");
const sourceSha256 = "cf04274f7758c08e7e93995a95eeaa48203da6753f94b14002731d4061ffac9b";
const scaleByteOffset = 852;
const factor = 1000;

const bytes = await readFile(sourcePath);
if (sha256(bytes) !== sourceSha256) throw new Error("Quant-scale source fixture SHA-256 changed.");
if (scaleByteOffset + 4 > bytes.byteLength) throw new Error("Quant-scale patch offset exceeds the source fixture.");
const before = bytes.readFloatLE(scaleByteOffset);
if (!Number.isFinite(before) || before <= 0) throw new Error("Quant-scale source value is not a positive finite float32.");
const after = Math.fround(before * factor);
const candidate = Buffer.from(bytes);
candidate.writeFloatLE(after, scaleByteOffset);
const outputSha256 = sha256(candidate);
await writeFile(outputPath, candidate);
await writeFile(manifestPath, `${JSON.stringify({
  schema: "deepbom.external_review_quant_scale_diff_fixture.v1",
  source: { path: path.basename(sourcePath), sha256: sourceSha256, bytes: bytes.byteLength },
  candidate: { path: path.basename(outputPath), sha256: outputSha256, bytes: candidate.byteLength },
  mutation: {
    byte_offset: scaleByteOffset,
    scalar_encoding: "IEEE754_FLOAT32_LITTLE_ENDIAN",
    baseline_value: before,
    candidate_value: after,
    requested_factor: factor,
    observed_factor: after / before,
  },
  interpretation_boundary: "This deterministic fixture changes one serialized per-axis affine scale only. It does not claim that the resulting model has acceptable task accuracy or runtime behaviour.",
}, null, 2)}\n`, "utf8");
console.log(`Wrote ${path.relative(root, outputPath)} (${outputSha256}).`);

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}
