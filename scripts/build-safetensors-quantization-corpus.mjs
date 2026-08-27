import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { canonicalJson } from "../web/lib/report-utils.js";
import { buildSafeTensorsQuantizationContract } from "../web/lib/safetensors-quantization-contract.js";

const outputDirectory = "corpus/safetensors-quantization-contract-corpus";
const localDirectory = ".local-validation/safetensors-quantization-contract-corpus";
const artifacts = [
  {
    id: "teeny-tiny-llama-460m-awq",
    method: "awq",
    repository: "nicholasKluge/TeenyTinyLlama-460m-awq",
    revision: "1fcff3df8d6c30d380868dddc0ff1f543de68081",
    model_path: "model.safetensors",
    model_size_bytes: 340424488,
    model_sha256: "288b8529802a8f5c03052d010f5e08f2f5e7fb9a988ec486883f5e3dea4122ae",
    license: "Apache-2.0",
    license_status: "declared_in_repository_card_and_license_file",
    config_paths: ["config.json", "quant_config.json"],
  },
  {
    id: "tiny-llama-1.1b-chat-v1-gptq",
    method: "gptq",
    repository: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
    revision: "9d4580af0f21bccafd762dcc50d0c7bac6273584",
    model_path: "model.safetensors",
    model_size_bytes: 768148704,
    model_sha256: "ceef7773c19f5cbda3e5f688f99e045e07a879e70c19ecdb5d069b27c2d45b37",
    license: "Apache-2.0",
    license_status: "declared_in_repository_card",
    config_paths: ["config.json", "quantize_config.json"],
  },
];

await mkdir(outputDirectory, { recursive: true });
await mkdir(localDirectory, { recursive: true });
const rows = [];
for (const source of artifacts) rows.push(await inspectSource(source));
const body = {
  schema: "deepbom.safetensors_quantization_contract_corpus.v1",
  purpose: "Source-bound structural validation of public AWQ and GPTQ SafeTensors repositories without redistributing model payloads.",
  population_boundary: "Two purposeful public architecture anchors. Counts establish parser and contract behavior only; they are not ecosystem prevalence estimates.",
  acquisition: "The complete SafeTensors header is fetched by immutable-revision HTTP range request. Full model size and LFS SHA-256 are independently checked against repository metadata. Weight payload bytes are not downloaded or redistributed.",
  artifact_count: rows.length,
  artifacts: rows,
};
const manifest = { ...body, ledger_sha256: sha256(canonicalJson(body)) };
await writeFile("corpus/safetensors-quantization-contract-corpus.v1.json", `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
console.log(`SafeTensors quantization corpus built (${rows.map((row) => `${row.method} ${row.measurement.module_count}`).join(", ")}).`);

async function inspectSource(source) {
  const api = await json(`https://huggingface.co/api/models/${source.repository}/revision/${source.revision}?blobs=true`);
  if (api.sha !== source.revision) throw new Error(`${source.id}: revision endpoint resolved to unexpected commit ${api.sha}.`);
  const modelRecord = api.siblings?.find((row) => row.rfilename === source.model_path);
  if (modelRecord?.size !== source.model_size_bytes || modelRecord?.lfs?.sha256 !== source.model_sha256) throw new Error(`${source.id}: model LFS identity mismatch.`);
  const base = `https://huggingface.co/${source.repository}/resolve/${source.revision}/`;
  const prefix = await range(`${base}${source.model_path}`, 0, 7);
  if (prefix.bytes.length !== 8) throw new Error(`${source.id}: SafeTensors length prefix range was not exact.`);
  const headerLength = Number(new DataView(prefix.bytes.buffer, prefix.bytes.byteOffset, 8).getBigUint64(0, true));
  if (!Number.isSafeInteger(headerLength) || headerLength < 2 || headerLength > 100_000_000) throw new Error(`${source.id}: invalid SafeTensors header length ${headerLength}.`);
  const headerRange = await range(`${base}${source.model_path}`, 8, 7 + headerLength);
  if (headerRange.bytes.length !== headerLength) throw new Error(`${source.id}: complete header range was not returned.`);
  const header = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(headerRange.bytes).trimEnd());
  const tensors = tensorDirectory(source.id, header, source.model_size_bytes, headerLength);
  const documents = {};
  const configRecords = [];
  for (const configPath of source.config_paths) {
    const response = await fetch(`${base}${configPath}`);
    if (!response.ok) throw new Error(`${source.id}: ${configPath} returned HTTP ${response.status}.`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    const document = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
    documents[configPath] = document;
    configRecords.push({ path: configPath, size_bytes: bytes.length, sha256: sha256(bytes) });
    await writeFile(path.join(outputDirectory, `${source.id}.${configPath}`), bytes);
  }
  const sidecars = Object.fromEntries(source.config_paths.filter((value) => value !== "config.json").map((configPath) => [
    configPath.replace(/\.json$/, ""), { path: configPath, document: documents[configPath] },
  ]));
  const contract = buildSafeTensorsQuantizationContract(documents["config.json"], tensors, { sidecars });
  if (contract.status !== "assessed" || contract.method !== source.method || contract.invalid_module_count !== 0) throw new Error(`${source.id}: quantization contract failed: ${JSON.stringify(contract.config_issues || [])}.`);
  const moduleLedgerSha = sha256(canonicalJson(contract.modules));
  const detailed = { source, header_length_bytes: headerLength, header_sha256: sha256(headerRange.bytes), configs: configRecords, contract };
  await writeFile(path.join(localDirectory, `${source.id}.receipt.json`), `${JSON.stringify(detailed, null, 2)}\n`, "utf8");
  return {
    id: source.id,
    method: source.method,
    repository: source.repository,
    revision: source.revision,
    model_path: source.model_path,
    model_size_bytes: source.model_size_bytes,
    model_sha256: source.model_sha256,
    license: source.license,
    license_status: source.license_status,
    header_length_bytes: headerLength,
    header_sha256: sha256(headerRange.bytes),
    config_files: configRecords,
    measurement: {
      status: contract.status,
      evidence_class: contract.evidence_class,
      bits: contract.bits,
      group_size: contract.group_size,
      module_count: contract.module_count,
      valid_module_count: contract.valid_module_count,
      invalid_module_count: contract.invalid_module_count,
      logical_weight_element_count: contract.logical_weight_element_count,
      packed_weight_code_capacity: contract.packed_weight_code_capacity,
      scale_element_count: contract.scale_element_count,
      zero_point_code_capacity: contract.zero_point_code_capacity,
      packed_tensor_bytes: contract.packed_tensor_bytes,
      module_ledger_sha256: moduleLedgerSha,
      representative_modules: contract.modules.slice(0, 3),
      source: contract.source,
      payload_value_scan: "not_performed_header_only_corpus",
    },
  };
}

function tensorDirectory(id, header, fileSize, headerLength) {
  const tensors = [];
  for (const [name, value] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const offsets = value?.data_offsets;
    if (!Array.isArray(value?.shape) || !Array.isArray(offsets) || offsets.length !== 2) throw new Error(`${id}: malformed descriptor ${name}.`);
    tensors.push({ index: tensors.length, name, dtype: value.dtype, shape: value.shape, data_offset: offsets[0], data_end: offsets[1], byte_length: offsets[1] - offsets[0] });
  }
  const sorted = [...tensors].sort((left, right) => left.data_offset - right.data_offset || left.data_end - right.data_end);
  let cursor = 0;
  for (const tensor of sorted) {
    if (tensor.data_offset !== cursor || tensor.data_end < tensor.data_offset) throw new Error(`${id}: payload gap or overlap before ${tensor.name}.`);
    cursor = tensor.data_end;
  }
  const payloadLength = fileSize - 8 - headerLength;
  if (cursor !== payloadLength) throw new Error(`${id}: descriptor coverage ${cursor}/${payloadLength}.`);
  return tensors;
}

async function range(url, start, end) {
  const response = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (response.status !== 206) throw new Error(`Range request returned HTTP ${response.status}: ${url}`);
  const contentRange = response.headers.get("content-range") || "";
  if (!contentRange.startsWith(`bytes ${start}-${end}/`)) throw new Error(`Unexpected Content-Range ${contentRange}.`);
  return { bytes: new Uint8Array(await response.arrayBuffer()), contentRange };
}

async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}.`);
  return response.json();
}

function sha256(value) { return createHash("sha256").update(value).digest("hex"); }
