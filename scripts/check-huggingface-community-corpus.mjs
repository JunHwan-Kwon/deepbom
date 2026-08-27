import { readFile } from "node:fs/promises";

import {
  classifyFile,
  classifyRepository,
  HF_CLASSIFIER_SCHEMA,
  HF_CORPUS_PATH,
  readHfCorpus,
  sha256Text,
} from "./huggingface-community-corpus-lib.mjs";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectDeepEqual, expectEqual } = createCheck("Hugging Face community corpus");

expectDeepEqual(
  classifyRepository({
    id: "example/tiny-keyword",
    metadata: { pipeline_tag: "audio-classification", tags: ["tinyml"] },
    files: [{ path: "model.tflite", format: "tflite", kind: "model", size_bytes: 2 * 1024 ** 2 }],
  }),
  { name: "micro", confidence: "explicit", reasons: ["explicit_tflite_micro_or_mcu_metadata"] },
  "Explicit TinyML metadata should classify as micro.",
);
expectEqual(
  classifyRepository({
    id: "example/mobile-vision",
    metadata: { pipeline_tag: "image-classification" },
    files: [{ path: "model.tflite", format: "tflite", kind: "model", size_bytes: 8 * 1024 ** 2 }],
  }).confidence,
  "candidate",
  "A bounded small TFLite artifact should remain a micro candidate rather than a proven MCU fit.",
);
expectEqual(
  classifyRepository({
    id: "example/vision-mid",
    metadata: { pipeline_tag: "image-classification" },
    files: [{ path: "model.onnx", format: "onnx", kind: "model", size_bytes: 32 * 1024 ** 2 }],
  }).name,
  "mid",
  "A standalone ONNX vision model should classify as mid.",
);
expectEqual(
  classifyRepository({
    id: "example/generative",
    metadata: { pipeline_tag: "text-generation", library_name: "onnxruntime-genai" },
    files: [{ path: "model.onnx", format: "onnx", kind: "model", size_bytes: 128 * 1024 ** 2 }],
  }).name,
  "large",
  "Generative runtime evidence should classify as large independently of current payload size.",
);
expectEqual(
  classifyRepository({
    id: "example/Qwen3-VL-2B-ONNX",
    metadata: { architectures: ["Qwen3VLForConditionalGeneration"], model_type: "qwen3_vl" },
    files: [{ path: "onnx/embed_tokens.onnx", format: "onnx", kind: "model", size_bytes: 434 }],
  }).name,
  "large",
  "A small external-data graph should retain its VLM tier from architecture lineage rather than graph-file size.",
);
expectDeepEqual(classifyFile("decoder/model.onnx"), { kind: "model", format: "onnx", support: "current" }, "ONNX should be current-analyzer format.");
expectDeepEqual(classifyFile("decoder/model.onnx_data"), { kind: "sidecar", format: "external_data", support: "onnx_companion" }, "ONNX external data should be a companion.");
expectDeepEqual(classifyFile("model.litertlm"), { kind: "model", format: "litertlm", support: "future_large" }, "LiteRT-LM should be future-large.");

const manifest = await readHfCorpus(HF_CORPUS_PATH);
const summary = JSON.parse(await readFile("corpus/huggingface-community-corpus.v1.summary.json", "utf8"));
expectEqual(manifest.classifier_schema, HF_CLASSIFIER_SCHEMA, "Snapshot should use the current classifier.");
expectEqual(summary.snapshot_sha256, sha256Text(await readFile(HF_CORPUS_PATH)), "Readable summary should bind the compressed snapshot SHA-256.");
expectEqual(manifest.summary.repository_count, manifest.repositories.length, "Repository denominator should conserve.");
expectEqual(Object.values(manifest.summary.organization_counts).reduce((sum, value) => sum + value, 0), manifest.repositories.length, "Organization counts should conserve.");
expectEqual(Object.values(manifest.summary.tier_counts).reduce((sum, value) => sum + value, 0), manifest.repositories.length, "Tier counts should conserve.");
expect(manifest.repositories.every((repository) => repository.revision.length === 40), "Every repository should be commit-pinned.");
expect(manifest.repositories.every((repository) => repository.upstream_metadata && typeof repository.upstream_metadata === "object"), "Every repository should preserve the complete public API metadata projection.");
expect(manifest.repositories.some((repository) => repository.tier.name === "micro"), "Snapshot should retain at least one micro candidate or explicit micro repository.");
expect(manifest.repositories.some((repository) => repository.tier.name === "mid"), "Snapshot should retain mid repositories.");
expect(manifest.repositories.some((repository) => repository.tier.name === "large"), "Snapshot should retain large repositories.");

done(`${manifest.repositories.length} commit-pinned repositories; tier and file-identity conservation passed`);
