import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import {
  analyzeCoreMlPerChannelLinearFixture, buildCoreMlPerChannelLinearFixture, coreMlLegacyQuantizationSource,
  COREML_LEGACY_QUANTIZATION_CORPUS_PATH, COREML_LEGACY_QUANTIZATION_MODEL_PATH,
} from "./coreml-legacy-quantization-corpus-lib.mjs";

const payload = buildCoreMlPerChannelLinearFixture();
const manifest = {
  schema: "deepbom.coreml_legacy_quantization_contract_corpus.v1",
  format: "coreml_mlmodel",
  artifact_count: 1,
  population_scope: "One deterministic source-pinned legacy NeuralNetwork fixture establishes per-output-channel linear WeightParams parsing, packed-code conservation, and fail-closed channel cardinality. It is a conformance fixture, not a public-model sample or evidence of ecosystem prevalence, runtime placement, latency, or task quality.",
  generator_source: coreMlLegacyQuantizationSource(),
  artifacts: [{
    id: "per-output-channel-linear-int4", path: COREML_LEGACY_QUANTIZATION_MODEL_PATH,
    provenance_class: "source_pinned_generated_contract_fixture", ecosystem_prevalence_claim: false,
    semantic_granularity: "per_output_channel", baseline: await analyzeCoreMlPerChannelLinearFixture(payload),
  }],
};
await mkdir(dirname(COREML_LEGACY_QUANTIZATION_MODEL_PATH), { recursive: true });
await writeFile(COREML_LEGACY_QUANTIZATION_MODEL_PATH, payload);
await writeFile(COREML_LEGACY_QUANTIZATION_CORPUS_PATH, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`Wrote ${COREML_LEGACY_QUANTIZATION_CORPUS_PATH} and ${COREML_LEGACY_QUANTIZATION_MODEL_PATH}.`);
