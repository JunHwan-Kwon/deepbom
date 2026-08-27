import { buildTensorStorageSummary, parseMetadataModel } from "../web/lib/metadata-model-adapters.js";
import { createHash } from "node:crypto";
import { Sha256Accumulator, sha256BytesHex } from "../web/lib/sha256-sync.js";
import { buildDeploymentContractDocuments } from "../web/lib/report-export-contracts.js";
import { buildEngineeringEvidenceDocument } from "../web/lib/report-evidence.js";
import { assertCycloneDx17 } from "./cyclonedx-17-schema.mjs";
import { scanSerializedTensorPayloads } from "../web/lib/tensor-numerical-integrity.js";

function expect(value, message) {
  if (!value) throw new Error(message);
}

function expectThrows(callback, pattern, message) {
  try { callback(); } catch (error) {
    if (String(error.message).includes(pattern)) return;
    throw new Error(`${message}: unexpected error ${error.message}`);
  }
  throw new Error(`${message}: expected failure`);
}

for (const length of [0, 1, 55, 56, 63, 64, 65, 1000]) {
  const bytes = Uint8Array.from({ length }, (_, index) => (index * 37 + 11) & 255);
  const expected = createHash("sha256").update(bytes).digest("hex");
  expect(sha256BytesHex(bytes) === expected, `single-buffer SHA-256 ${length}`);
  const streamed = new Sha256Accumulator();
  for (let offset = 0; offset < bytes.length; offset += 17) streamed.update(bytes.subarray(offset, offset + 17));
  expect(streamed.digestHex() === expected, `streaming SHA-256 ${length}`);
}

function safetensors(header, payloadBytes) {
  const encoded = new TextEncoder().encode(header);
  const bytes = new Uint8Array(8 + encoded.length);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(encoded.length), true);
  bytes.set(encoded, 8);
  return { bytes, fileSize: bytes.length + payloadBytes };
}

const safe = safetensors(JSON.stringify({
  __metadata__: { format: "pt" },
  weight: { dtype: "U8", shape: [4], data_offsets: [0, 4] },
}), 4);
const safeAnalysis = parseMetadataModel(safe.bytes, "model.safetensors", safe.fileSize, "safetensors");
expect(safeAnalysis.tensor_count === 1, "SafeTensors tensor inventory");
expect(safeAnalysis.safetensors.payload_coverage_status === "complete_without_gaps_or_overlaps", "SafeTensors range conservation");
expect(safeAnalysis.safetensors.duplicate_key_validation === "complete", "SafeTensors duplicate-key validation");
expect(safeAnalysis.safetensors.reference_implementation.commit === "6eb4dc9a28ebce297606e0f4836bbf28839cacef", "SafeTensors parser source pin");
expect(safeAnalysis.tensor_storage_summary.element_count === 4, "SafeTensors exact stored element count");
expect(safeAnalysis.tensor_storage_summary.byte_length === 4, "SafeTensors exact stored byte count");
expect(safeAnalysis.tensor_storage_summary.effective_bits_per_element === "8", "SafeTensors exact effective bits per element");
safeAnalysis.model_sha256 = "c".repeat(64);
const safeBom = buildDeploymentContractDocuments(safeAnalysis, { hash: safeAnalysis.model_sha256, fileSizeBytes: safe.fileSize, generatedAt: "2026-08-03T00:00:00.000Z" });
assertCycloneDx17(safeBom.documents.cyclonedx_evidence, "SafeTensors evidence BOM");
expect(safeBom.documents.artifact_evidence_envelope.capabilities.conservation.valid, "SafeTensors capability conservation");
expect(safeBom.subject.schema_or_opset === "SafeTensors format (unversioned)", "SafeTensors export format identity");
expect(safeBom.documents.runtime_requirement_manifest.necessary_runtime_floor.runtime === "SafeTensors-compatible loader (unbound)", "SafeTensors runtime identity");

const duplicate = safetensors('{"w":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"w":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', 1);
expectThrows(() => parseMetadataModel(duplicate.bytes, "duplicate.safetensors", duplicate.fileSize, "safetensors"), "duplicate JSON key w", "duplicate SafeTensors names fail closed");
const gap = safetensors('{"w":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}', 2);
expectThrows(() => parseMetadataModel(gap.bytes, "gap.safetensors", gap.fileSize, "safetensors"), "gap or overlap", "SafeTensors gaps fail closed");
const f4 = safetensors('{"w":{"dtype":"F4","shape":[2],"data_offsets":[0,1]}}', 1);
expect(parseMetadataModel(f4.bytes, "f4.safetensors", f4.fileSize, "safetensors").tensors[0].byte_length === 1, "SafeTensors F4 byte cardinality");
const misalignedF4 = safetensors('{"w":{"dtype":"F4","shape":[1],"data_offsets":[0,1]}}', 1);
expectThrows(() => parseMetadataModel(misalignedF4.bytes, "misaligned-f4.safetensors", misalignedF4.fileSize, "safetensors"), "not byte-aligned", "SafeTensors sub-byte tensors fail closed when the reference implementation rejects their alignment");

class Writer {
  constructor(littleEndian = true) { this.bytes = []; this.littleEndian = littleEndian; }
  u32(value) { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, value, this.littleEndian); this.bytes.push(...b); }
  u64(value) { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(value), this.littleEndian); this.bytes.push(...b); }
  text(value) { const b = new TextEncoder().encode(value); this.u64(b.length); this.bytes.push(...b); }
  finish() { return Uint8Array.from(this.bytes); }
}

const bigEndianV3 = new Writer(false);
bigEndianV3.bytes.push(0x47, 0x47, 0x55, 0x46);
bigEndianV3.u32(3); bigEndianV3.u64(0); bigEndianV3.u64(0);
const bigEndianV3Analysis = parseMetadataModel(bigEndianV3.finish(), "big-v3.gguf", 32, "gguf");
expect(bigEndianV3Analysis.gguf.version === 3 && bigEndianV3Analysis.gguf.endianness === "big", "GGUF v3 big-endian header contract");

const invalidBigEndianV2 = new Writer(false);
invalidBigEndianV2.bytes.push(0x47, 0x47, 0x55, 0x46);
invalidBigEndianV2.u32(2); invalidBigEndianV2.u64(0); invalidBigEndianV2.u64(0);
expectThrows(() => parseMetadataModel(invalidBigEndianV2.finish(), "big-v2.gguf", 32, "gguf"), "requires format version 3", "GGUF v2 cannot claim the v3 big-endian contract");

const writer = new Writer();
writer.bytes.push(0x47, 0x47, 0x55, 0x46);
writer.u32(3);
writer.u64(1);
writer.u64(11);
writer.text("general.architecture"); writer.u32(8); writer.text("llama");
writer.text("general.alignment"); writer.u32(4); writer.u32(32);
for (const [key, value] of [
  ["llama.context_length", 16], ["llama.embedding_length", 4], ["llama.block_count", 1], ["llama.feed_forward_length", 8],
  ["llama.attention.head_count", 2], ["llama.attention.head_count_kv", 1], ["llama.attention.key_length", 2], ["llama.attention.value_length", 2],
]) { writer.text(key); writer.u32(4); writer.u32(value); }
writer.text("tokenizer.ggml.tokens"); writer.u32(9); writer.u32(8); writer.u64(32);
for (let index = 0; index < 32; index += 1) writer.text(`t${index}`);
writer.text("token_embd.weight"); writer.u32(2); writer.u64(32); writer.u64(4); writer.u32(2); writer.u64(0);
const ggufHeader = writer.finish();
const ggufFileSize = Math.ceil(ggufHeader.length / 32) * 32 + 72;
const ggufAnalysis = parseMetadataModel(ggufHeader, "model.gguf", ggufFileSize, "gguf");
expect(ggufAnalysis.gguf.version === 3, "GGUF version");
expect(ggufAnalysis.gguf.architecture === "llama", "GGUF architecture metadata");
expect(ggufAnalysis.tensor_count === 1 && ggufAnalysis.tensors[0].shape.join("x") === "32x4", "GGUF tensor inventory");
expect(ggufAnalysis.gguf.invalid_tensor_offset_count === 0, "GGUF offset alignment");
expect(ggufAnalysis.tensors[0].byte_length === 72 && ggufAnalysis.tensor_inventory.total_declared_tensor_bytes === 72, "GGUF Q4_0 byte cardinality");
expect(ggufAnalysis.gguf.payload_byte_length === 72 && ggufAnalysis.gguf.payload_coverage_status === "complete_without_gaps_or_overlaps", "GGUF payload conservation");
expect(ggufAnalysis.gguf.type_traits_source.source_commit === "7bd8282c37fcd9c4d7236106d664761a23318f18", "GGUF type-traits source pin");
expect(ggufAnalysis.tensor_storage_summary.element_count === 128, "GGUF exact stored element count");
expect(ggufAnalysis.tensor_storage_summary.effective_bits_per_element === "4.5", "GGUF Q4_0 effective bits include block scales");
expect(ggufAnalysis.gguf.semantic_contract.architecture === "llama", "GGUF architecture semantic contract");
expect(ggufAnalysis.gguf.semantic_contract.serialized_parameter_count_decimal === "128"
  && ggufAnalysis.gguf.semantic_contract.serialized_tensor_bytes_decimal === "72"
  && ggufAnalysis.gguf.semantic_contract.effective_bits_per_parameter === "4.5", "GGUF semantic contract reuses exact parameter/storage conservation");
expect(ggufAnalysis.gguf.semantic_contract.kv_state_projection.elements_per_token_per_batch.decimal === "4"
  && ggufAnalysis.gguf.semantic_contract.kv_state_projection.elements_at_context_batch_one.decimal === "64", "GGUF exact declared KV-state cardinality");
expect(ggufAnalysis.gguf.semantic_contract.compute_projection.dense_projection_macs_all_layers_per_token.decimal === "144"
  && ggufAnalysis.gguf.semantic_contract.compute_projection.prefill_transformer_core_macs_at_declared_context.decimal === "3392"
  && ggufAnalysis.gguf.semantic_contract.compute_projection.decode_with_one_logit_position_macs.decimal === "400", "GGUF registered canonical decoder compute scenario");
expect(ggufAnalysis.gguf.backend_compatibility.status === "source_candidate", "GGUF pinned llama.cpp backend prerequisite candidate");
expect(ggufAnalysis.gguf.backend_compatibility.architecture_registry_match === true, "GGUF llama.cpp architecture registry match");
expect(ggufAnalysis.gguf.backend_compatibility.profiles.length === 9, "GGUF source-backed backend profile inventory");
expect(ggufAnalysis.gguf.backend_compatibility.compatibility_conclusion === "not_concluded", "GGUF source candidate does not become an execution compatibility claim");
expect(ggufAnalysis.gguf.backend_compatibility.profiles.every((row) => row.execution_evidence_class === "NOT_OBSERVED"), "GGUF backend profiles retain execution evidence boundary");
const qwen3Header = Uint8Array.from(Buffer.from(Buffer.from(ggufHeader).toString("latin1").replaceAll("llama", "qwen3"), "latin1"));
const qwen3Analysis = parseMetadataModel(qwen3Header, "qwen3.gguf", Math.ceil(qwen3Header.length / 32) * 32 + 72, "gguf");
expect(qwen3Analysis.gguf.semantic_contract.architecture === "qwen3"
  && qwen3Analysis.gguf.semantic_contract.compute_projection_status === "assessed_registered_canonical_decoder_scenario"
  && qwen3Analysis.gguf.backend_compatibility.architecture_registry_match,
"Pinned llama.cpp Qwen3 registration and canonical decoder projection agree");

for (const [ggmlType, blockElements, blockBytes] of [
  [16, 256, 66], [17, 256, 74], [18, 256, 98], [19, 256, 50], [20, 32, 18],
  [21, 256, 110], [22, 256, 82], [23, 256, 136], [29, 256, 56],
  [34, 256, 54], [35, 256, 66], [39, 32, 17], [40, 64, 36], [41, 128, 18], [42, 64, 18],
]) {
  const modern = new Writer();
  modern.bytes.push(0x47, 0x47, 0x55, 0x46);
  modern.u32(3); modern.u64(1); modern.u64(1);
  modern.text("general.architecture"); modern.u32(8); modern.text("test");
  modern.text(`tensor_${ggmlType}`); modern.u32(1); modern.u64(blockElements); modern.u32(ggmlType); modern.u64(0);
  const header = modern.finish();
  const dataOffset = Math.ceil(header.length / 32) * 32;
  const fileSize = dataOffset + blockBytes;
  const file = new Uint8Array(fileSize);
  file.set(header);
  const parsed = parseMetadataModel(file, `type_${ggmlType}.gguf`, fileSize, "gguf");
  expect(parsed.tensors[0].byte_length === blockBytes, `GGUF type ${ggmlType} pinned block byte cardinality`);
  expect(parsed.gguf.unsupported_ggml_type_count === 0, `GGUF type ${ggmlType} source-pinned storage support`);
  const numerical = await scanSerializedTensorPayloads(file, parsed, { chunkBytes: 17 });
  expect(numerical.status === "assessed" && numerical.assessed_tensor_bytes === blockBytes, `GGUF type ${ggmlType} parser-to-decoder payload coverage`);
  expect(numerical.decoded_value_count === blockElements && numerical.byte_conservation_status === "complete", `GGUF type ${ggmlType} parser-to-decoder conservation`);
}
const ggufFile = new Uint8Array(ggufFileSize);
ggufFile.set(ggufHeader);
const ggufNumerical = await scanSerializedTensorPayloads(ggufFile, ggufAnalysis, { chunkBytes: 17 });
ggufAnalysis.tensor_numerical_integrity = ggufNumerical;
const ggufNumericalByIndex = new Map(ggufNumerical.tensor_records.map((record) => [record.tensor_index, record]));
for (const tensor of ggufAnalysis.tensors) tensor.numerical_integrity = ggufNumericalByIndex.get(tensor.index) || null;
ggufAnalysis.tensor_inventory.numerical_integrity_status = ggufNumerical.status;
ggufAnalysis.tensor_inventory.assessed_payload_bytes = ggufNumerical.assessed_tensor_bytes;
ggufAnalysis.tensor_inventory.unassessed_payload_bytes = ggufNumerical.unassessed_tensor_bytes;
ggufAnalysis.tensor_inventory.decoded_value_count = ggufNumerical.decoded_value_count;
ggufAnalysis.tensor_storage_summary = buildTensorStorageSummary("gguf", ggufAnalysis.tensors, ggufNumerical);
expect(ggufNumerical.status === "assessed"
  && ggufNumerical.tensor_records.every((record) => record.serialized_endianness === "little"),
"GGUF report fixture carries complete source-endian numerical evidence");

const invalidRow = new Writer();
invalidRow.bytes.push(0x47, 0x47, 0x55, 0x46);
invalidRow.u32(3); invalidRow.u64(1); invalidRow.u64(0);
invalidRow.text("output.weight"); invalidRow.u32(2); invalidRow.u64(16); invalidRow.u64(32); invalidRow.u32(8); invalidRow.u64(0);
const invalidRowHeader = invalidRow.finish();
const invalidRowDataOffset = Math.ceil(invalidRowHeader.length / 32) * 32;
const invalidRowFile = new Uint8Array(invalidRowDataOffset + 544);
invalidRowFile.set(invalidRowHeader);
const invalidRowAnalysis = parseMetadataModel(invalidRowFile, "invalid-row.gguf", invalidRowFile.length, "gguf");
const invalidRowNumerical = await scanSerializedTensorPayloads(invalidRowFile, invalidRowAnalysis);
const invalidRowStorage = buildTensorStorageSummary("gguf", invalidRowAnalysis.tensors, invalidRowNumerical);
expect(invalidRowAnalysis.gguf.invalid_tensor_cardinality_count === 1
  && invalidRowNumerical.status === "not_assessed"
  && invalidRowNumerical.byte_conservation_status === "not_assessed_unknown_tensor_storage_cardinality"
  && invalidRowStorage.status === "partial_unknown_serialized_byte_cardinality"
  && invalidRowStorage.effective_bits_per_element == null,
"GGUF invalid block-row cardinality must remain an explicit partial assessment instead of entering a payload decoder");
ggufAnalysis.model_sha256 = "d".repeat(64);
const ggufEvidence = buildEngineeringEvidenceDocument(ggufAnalysis, {
  reportContext: { identity: { filename: ggufAnalysis.filename, format: "gguf", sha256: ggufAnalysis.model_sha256 } },
  rawEvidenceContext: { identity: { filename: ggufAnalysis.filename, format: "gguf", sha256: ggufAnalysis.model_sha256 } },
});
expect(ggufEvidence.evidence.conformance_report.release_export_allowed
  && ggufEvidence.evidence.conformance_report.checks.some((row) => row.id === "CF-GGUF-SEMANTIC-001" && row.status === "pass")
  && ggufEvidence.evidence.conformance_report.checks.some((row) => row.id === "CF-GGUF-ENDIAN-001" && row.status === "pass"),
"GGUF semantic and endian release conformance");
const tamperedGgufEndian = structuredClone(ggufAnalysis);
tamperedGgufEndian.tensor_numerical_integrity.tensor_records[0].serialized_endianness = "big";
expectThrows(() => buildEngineeringEvidenceDocument(tamperedGgufEndian, {
  reportContext: { identity: { filename: tamperedGgufEndian.filename, format: "gguf", sha256: tamperedGgufEndian.model_sha256 } },
  rawEvidenceContext: { identity: { filename: tamperedGgufEndian.filename, format: "gguf", sha256: tamperedGgufEndian.model_sha256 } },
}), "CF-GGUF-ENDIAN-001", "GGUF tensor-endian tampering fails release conformance");
const tamperedGgufProjection = structuredClone(ggufAnalysis);
tamperedGgufProjection.gguf.semantic_contract.compute_projection.decode_with_one_logit_position_macs.decimal = "401";
expectThrows(() => buildEngineeringEvidenceDocument(tamperedGgufProjection, {
  reportContext: { identity: { filename: tamperedGgufProjection.filename, format: "gguf", sha256: tamperedGgufProjection.model_sha256 } },
  rawEvidenceContext: { identity: { filename: tamperedGgufProjection.filename, format: "gguf", sha256: tamperedGgufProjection.model_sha256 } },
}), "CF-GGUF-SEMANTIC-001", "GGUF compute projection tampering fails release conformance");
const ggufBom = buildDeploymentContractDocuments(ggufAnalysis, { hash: ggufAnalysis.model_sha256, fileSizeBytes: ggufFileSize, generatedAt: "2026-08-03T00:00:00.000Z" });
assertCycloneDx17(ggufBom.documents.cyclonedx_evidence, "GGUF evidence BOM");
expect(ggufBom.documents.artifact_evidence_envelope.format_extensions.gguf.architecture === "llama", "GGUF envelope extension");
expect(ggufBom.subject.schema_or_opset === "GGUF v3", "GGUF export format identity");
expect(ggufBom.documents.runtime_requirement_manifest.necessary_runtime_floor.runtime === "GGUF-compatible runtime (unbound)", "GGUF runtime identity");
const ggufMissing = new Map(ggufBom.documents.missing_provenance_field_specification.fields.map((field) => [field.id, field]));
expect(ggufMissing.get("quantization_calibration_configuration")?.status === "partial", "GGUF block quantization must require a reproducible quantizer recipe");
expect(ggufMissing.get("representative_dataset_id")?.status === "not_applicable", "GGUF block storage must not imply calibration-dataset dependence");

const externalOnnx = {
  format: "onnx", filename: "external.onnx", model_sha256: "e".repeat(64), file_size_bytes: 128,
  inputs: [], outputs: [], ops: [], tensors: [], metadata_presence: { status: "assessed" },
  onnx_external_data: { status: "complete", supplied_files: [{ used: true, path: "weights.bin", byte_length: 4096, sha256: "f".repeat(64), sha1: "1".repeat(40) }] },
};
const externalBom = buildDeploymentContractDocuments(externalOnnx, { generatedAt: "2026-08-03T00:00:00.000Z" }).documents.cyclonedx_evidence;
assertCycloneDx17(externalBom, "external-data ONNX evidence BOM");
expect(externalBom.components.length === 1 && externalBom.components[0].hashes[0].content === "f".repeat(64), "ONNX sidecar component");
expect(externalBom.dependencies[0].dependsOn.includes(externalBom.components[0]["bom-ref"]), "ONNX model depends on external weights");

const hostile = writer.finish().slice();
new DataView(hostile.buffer).setBigUint64(8, 10_000_001n, true);
expectThrows(() => parseMetadataModel(hostile, "hostile.gguf", ggufFileSize, "gguf"), "count exceeds safety limit", "GGUF hostile counts fail closed");
expectThrows(() => parseMetadataModel(new Uint8Array(), "model.bin", 0, "unsupported"), "not implemented", "unknown metadata format fails closed");

const duplicateSummary = buildTensorStorageSummary("safetensors", [
  { index: 0, name: "a", dtype: "F16", shape: [2], byte_length: 4 },
  { index: 1, name: "b", dtype: "F16", shape: [2], byte_length: 4 },
], { tensor_records: [
  { tensor_index: 0, status: "assessed_full_payload", payload_sha256: "a".repeat(64) },
  { tensor_index: 1, status: "assessed_full_payload", payload_sha256: "a".repeat(64) },
] });
expect(duplicateSummary.content_addressed_duplicate_group_count === 1, "content-addressed duplicate group count");
expect(duplicateSummary.content_addressed_duplicate_bytes_after_first === 4, "content-addressed duplicate byte savings candidate");
expect(duplicateSummary.element_count_decimal === "4" && duplicateSummary.byte_length_decimal === "8", "storage summary conservation");

console.log("Metadata model adapters passed (GGUF and SafeTensors bounded parsing, conservation, fail-closed malformed inputs).");
