import { createHash } from "node:crypto";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { TFLITE_RUNTIME_INFO_SOURCE } from "../web/lib/tflite-runtime-info-adapter.js";
import { TFLITE_PROFILE_INFO_SOURCE } from "../web/lib/tflite-profile-info-adapter.js";
import { fetchPinnedBytes } from "./fetch-pinned-source.mjs";

const commit = TFLITE_RUNTIME_INFO_SOURCE.source_commit.split("@")[1];
if (!/^[a-f0-9]{40}$/.test(commit || "")) throw new Error("TFLite runtime-info source commit is not pinned to a full commit hash.");

const specs = [
  {
    label: "generator",
    path: TFLITE_RUNTIME_INFO_SOURCE.source_file,
    sha256: TFLITE_RUNTIME_INFO_SOURCE.source_sha256,
    expectedBytes: 9797,
    metadataSha: ANALYZER_METADATA.rulepackProvenance.tfliteRuntimeInfoGeneratorSha256,
  },
  {
    label: "proto",
    path: TFLITE_RUNTIME_INFO_SOURCE.proto_file,
    sha256: TFLITE_RUNTIME_INFO_SOURCE.proto_sha256,
    expectedBytes: 4312,
    metadataSha: ANALYZER_METADATA.rulepackProvenance.tfliteRuntimeInfoProtoSha256,
  },
  {
    label: "delegation metadata",
    path: TFLITE_RUNTIME_INFO_SOURCE.delegation_metadata_file,
    sha256: TFLITE_RUNTIME_INFO_SOURCE.delegation_metadata_sha256,
    expectedBytes: 24496,
    metadataSha: ANALYZER_METADATA.rulepackProvenance.tfliteDelegationMetadataSha256,
  },
  {
    label: "benchmark export driver",
    path: TFLITE_RUNTIME_INFO_SOURCE.export_driver_file,
    sha256: TFLITE_RUNTIME_INFO_SOURCE.export_driver_sha256,
    expectedBytes: 57533,
    metadataSha: ANALYZER_METADATA.rulepackProvenance.tfliteRuntimeInfoExportDriverSha256,
  },
  {
    label: "conversion metadata schema",
    path: "tensorflow/compiler/mlir/lite/schema/conversion_metadata.fbs",
    sha256: "2464449e30bfa6032c0218b53a1a83b224c6eda9b5cfd9f12211c4c0017dc20e",
    expectedBytes: 2522,
    metadataSha: ANALYZER_METADATA.rulepackProvenance.tfliteConversionMetadataSchemaSha256,
  },
];

specs.push(
  { label: "profiling proto", path: TFLITE_PROFILE_INFO_SOURCE.proto_file, sha256: TFLITE_PROFILE_INFO_SOURCE.proto_sha256, expectedBytes: 2487, metadataSha: ANALYZER_METADATA.rulepackProvenance.tfliteProfilingInfoProtoSha256 },
  { label: "profiling listener", path: TFLITE_PROFILE_INFO_SOURCE.listener_file, sha256: TFLITE_PROFILE_INFO_SOURCE.listener_sha256, expectedBytes: 3168, metadataSha: ANALYZER_METADATA.rulepackProvenance.tfliteProfilingListenerSha256 },
  { label: "profile summarizer", path: TFLITE_PROFILE_INFO_SOURCE.summarizer_file, sha256: TFLITE_PROFILE_INFO_SOURCE.summarizer_sha256, expectedBytes: 8796, metadataSha: ANALYZER_METADATA.rulepackProvenance.tfliteProfileSummarizerSha256 },
  { label: "profile formatter", path: TFLITE_PROFILE_INFO_SOURCE.formatter_file, sha256: TFLITE_PROFILE_INFO_SOURCE.formatter_sha256, expectedBytes: 12138, metadataSha: ANALYZER_METADATA.rulepackProvenance.tfliteProfileFormatterSha256 },
  { label: "profiler API", path: TFLITE_PROFILE_INFO_SOURCE.profiler_api_file, sha256: TFLITE_PROFILE_INFO_SOURCE.profiler_api_sha256, expectedBytes: 10721, metadataSha: ANALYZER_METADATA.rulepackProvenance.tfliteProfilerApiSha256 },
  { label: "subgraph profiling", path: TFLITE_PROFILE_INFO_SOURCE.subgraph_file, sha256: TFLITE_PROFILE_INFO_SOURCE.subgraph_sha256, expectedBytes: 112159, metadataSha: ANALYZER_METADATA.rulepackProvenance.tfliteSubgraphProfilingSha256 },
  { label: "XNNPACK profiling", path: TFLITE_PROFILE_INFO_SOURCE.xnnpack_delegate_file, sha256: TFLITE_PROFILE_INFO_SOURCE.xnnpack_delegate_sha256, expectedBytes: 328150, metadataSha: ANALYZER_METADATA.rulepackProvenance.tfliteXnnpackProfilingSha256 },
);

const verified = [];
for (const spec of specs) {
  if (spec.metadataSha !== spec.sha256) throw new Error(`Report metadata ${spec.label} digest drifted from the adapter contract.`);
  const url = `https://raw.githubusercontent.com/tensorflow/tensorflow/${commit}/${spec.path}`;
  const bytes = await fetchPinnedBytes(url, { label: "TensorFlow" });
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (sha256 !== spec.sha256) throw new Error(`${spec.path} SHA-256 mismatch: ${sha256} !== ${spec.sha256}`);
  if (bytes.length !== spec.expectedBytes) throw new Error(`${spec.path} byte length mismatch: ${bytes.length} !== ${spec.expectedBytes}`);
  verified.push(`${spec.path} ${bytes.length} B ${sha256}`);
}

console.log(`Pinned TFLite runtime-info source verification passed at ${TFLITE_RUNTIME_INFO_SOURCE.source_commit}:\n${verified.map((item) => `  - ${item}`).join("\n")}`);
