import { corpusModelUrl, readCorpusManifest } from "./public-model-corpus-lib.mjs";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("Public model corpus manifest");
const manifest = await readCorpusManifest();

expectEqual(manifest.artifact_count, 20, "The pinned public corpus should preserve its 20-artifact denominator.");
expectEqual(new Set(manifest.models.map((model) => model.task)).size, 11, "The corpus should cover eleven deployment task families.");
expectEqual(manifest.models.filter((model) => model.published_precision === "int8").length, 3, "The corpus should include three upstream INT8 artifacts.");
expectEqual(manifest.models.filter((model) => model.published_precision.startsWith("float16")).length, 6, "The corpus should include six reduced-precision storage artifacts.");
for (const model of manifest.models) {
  const url = corpusModelUrl(model);
  expect(url.startsWith("https://storage.googleapis.com/mediapipe-models/"), `${model.id}: source should remain in the official MediaPipe model bucket.`);
  expect(url.endsWith(`generation=${model.generation}`), `${model.id}: source URL should bind its immutable GCS generation.`);
}

done("20 generation-pinned artifacts across 11 tasks; 3 INT8 and 6 FP16-storage entries passed");
