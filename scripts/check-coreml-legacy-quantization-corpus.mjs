import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  analyzeCoreMlPerChannelLinearFixture, buildCoreMlPerChannelLinearFixture, coreMlLegacyQuantizationSource,
  readCoreMlLegacyQuantizationCorpus,
} from "./coreml-legacy-quantization-corpus-lib.mjs";

const manifest = await readCoreMlLegacyQuantizationCorpus();
assert.equal(manifest.schema, "deepbom.coreml_legacy_quantization_contract_corpus.v1");
assert.equal(manifest.artifact_count, 1);
assert.deepEqual(manifest.generator_source, coreMlLegacyQuantizationSource());
const artifact = manifest.artifacts[0];
assert.equal(artifact.provenance_class, "source_pinned_generated_contract_fixture");
assert.equal(artifact.ecosystem_prevalence_claim, false);
assert.equal(artifact.semantic_granularity, "per_output_channel");
const payload = await readFile(artifact.path);
assert.deepEqual(payload, buildCoreMlPerChannelLinearFixture());
assert.deepEqual(await analyzeCoreMlPerChannelLinearFixture(payload), artifact.baseline);
assert.deepEqual({
  storage: artifact.baseline.weight_storage, bits: artifact.baseline.number_of_bits,
  scheme: artifact.baseline.scheme, granularity: artifact.baseline.granularity, axis: artifact.baseline.axis,
  channels: artifact.baseline.channel_count, scales: artifact.baseline.scale_count, biases: artifact.baseline.bias_count,
  values: artifact.baseline.value_count, finite: artifact.baseline.finite_count,
  levels: artifact.baseline.quant_code_levels_used, assessed: artifact.baseline.quantization_assessment_status,
  per_axis_parameters: artifact.baseline.per_axis_quantized_weight_parameter_count,
}, {
  storage: "raw_quantized", bits: 4, scheme: "linear", granularity: "per_axis", axis: 0,
  channels: 2, scales: 2, biases: 2, values: 18, finite: 18, levels: 16,
  assessed: "assessed", per_axis_parameters: 1,
});
await assert.rejects(
  () => analyzeCoreMlPerChannelLinearFixture(buildCoreMlPerChannelLinearFixture({ scaleCount: 1 })),
  /scale\/bias cardinality does not match quantization axis 0 \(2\)/,
);
console.log("Core ML per-output-channel linear quantization corpus passed (pinned source, exact bytes, packed codes, and fail-closed cardinality). ");
