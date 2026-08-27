import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { gzipSync, gunzipSync } from "node:zlib";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import Ajv2020 from "ajv/dist/2020.js";

import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(ROOT, "corpus", "cyclonedx-20-quantization-fixtures");
const UPSTREAM = path.join(ROOT, ".local-validation", "upstream", "cyclonedx-specification-990");
const PR = 990;
const HEAD = "49a945618811213e55686a23fa63b287940071c6";
const SCHEMA_REL = "schema/2.0/model/cyclonedx-ai-ml-2.0.schema.json";
const ROOT_SCHEMA_REL = "schema/2.0/cyclonedx-2.0.schema.json";
const BUNDLED_REL = "schema/2.0/cyclonedx-2.0-bundled.schema.json";
const SPDX_REL = "schema/spdx.schema.json";
const CRYPTOGRAPHY_REL = "schema/cryptography-defs.schema.json";
const DEFINITION_FILE = path.join(OUT, "quantization-definition.pr990.json");
const SCHEMA_SET_GZIP_FILE = path.join(OUT, "cyclonedx-2.0-schema-set.pr990.json.gz");
const EVIDENCE_FILE = path.join(OUT, "evidence.v1.json");
const CHECK = process.argv.includes("--check");

const fixtures = fixtureDocuments();
let sourceSchemaRaw;
let bundledRaw;
let rootSchemaRaw;
let schemaSet;
const sourceMode = "PINNED_UPSTREAM_SCHEMA_VENDORED_FOR_OFFLINE_REPLAY";

if (!CHECK) {
  const commit = execFileSync("git", ["-c", `safe.directory=${UPSTREAM.replaceAll("\\", "/")}`, "-C", UPSTREAM, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  assert.equal(commit, HEAD, "The local CycloneDX PR #990 checkout is not at the pinned head.");
  schemaSet = await readUpstreamSchemaSet();
} else {
  schemaSet = decodeSchemaSet(gunzipSync(await readFile(SCHEMA_SET_GZIP_FILE)));
}
sourceSchemaRaw = requireSchema(schemaSet, SCHEMA_REL);
rootSchemaRaw = requireSchema(schemaSet, ROOT_SCHEMA_REL);
bundledRaw = requireSchema(schemaSet, BUNDLED_REL);

const sourceSchema = JSON.parse(sourceSchemaRaw.toString("utf8"));
const rootSchema = JSON.parse(rootSchemaRaw.toString("utf8"));
const quantization = sourceSchema.$defs?.quantization;
assert(quantization && typeof quantization === "object", "Pinned schema must define $defs.quantization.");

const quantizationValidator = createValidator(quantization);
const bomValidator = createValidator(rootSchema, validatorSupportSchemas(schemaSet));
const fixtureResults = fixtures.map(({ filename, expected, document }) => {
  const actual = bomValidator(document);
  const errors = normalizeErrors(bomValidator.errors);
  assert.equal(actual, expected, `${filename} expected validity ${expected}, received ${actual}: ${JSON.stringify(errors)}`);
  return {
    filename,
    expected: expected ? "VALID" : "INVALID",
    actual: actual ? "VALID" : "INVALID",
    sha256: sha256(Buffer.from(pretty(document))),
    errors,
  };
});

const probeResults = probeMatrix().map(({ id, value, expected }) => {
  const actual = quantizationValidator(value);
  const errors = normalizeErrors(quantizationValidator.errors);
  assert.equal(actual, expected, `${id} expected validity ${expected}, received ${actual}`);
  return { id, expected: expected ? "VALID" : "INVALID", actual: actual ? "VALID" : "INVALID", value, errors };
});
assert.equal(probeResults.filter((row) => row.actual === "VALID").length, 8);
assert.equal(probeResults.filter((row) => row.actual === "INVALID").length, 4);

const sourceSchemaSha256 = CHECK
  ? JSON.parse(await readFile(EVIDENCE_FILE, "utf8")).upstream.ai_ml_schema_sha256
  : sha256(sourceSchemaRaw);
const schemaSetArchive = encodeSchemaSet(schemaSet);
const body = {
  schema: "deepbom.cyclonedx20_quantization_fixture_evidence.v1",
  generated_at: "2026-08-18T00:00:00.000Z",
  upstream: {
    repository: "https://github.com/CycloneDX/specification",
    pull_request: PR,
    branch: "2.0-dev-ai-ml",
    head_commit: HEAD,
    source_mode: sourceMode,
    ai_ml_schema_path: SCHEMA_REL,
    ai_ml_schema_sha256: sourceSchemaSha256,
    validation_root_schema_path: ROOT_SCHEMA_REL,
    validation_root_schema_sha256: sha256(rootSchemaRaw),
    validation_schema_set_sha256: sha256(schemaSetArchive),
    validation_schema_files: [...schemaSet.entries()].map(([schemaPath, bytes]) => ({ path: schemaPath, sha256: sha256(bytes) })),
    bundled_schema_observation: {
      path: BUNDLED_REL,
      sha256: sha256(bundledRaw),
      used_for_validation: false,
      contains_model_properties_keyword: bundledRaw.includes(Buffer.from('"modelProperties"')),
      reason: "The PR-head bundled file does not contain the draft modelProperties structure; validation uses the modular PR-head schema graph.",
    },
    quantization_json_pointer: "/$defs/quantization",
    quantization_definition_jcs_sha256: sha256TextHex(canonicalJson(quantization)),
  },
  contribution_scope: {
    ready_valid_fixtures: fixtureResults.filter((row) => row.expected === "VALID").map((row) => row.filename),
    ready_invalid_fixtures: fixtureResults.filter((row) => row.expected === "INVALID").map((row) => row.filename),
    policy_questions_not_encoded_as_invalid_fixtures: [
      "per-tensor with axis",
      "per-tensor with groupSize",
      "per-channel without axis",
      "per-channel with groupSize",
      "per-group without axis or groupSize",
      "negative framework-native axis normalization",
    ],
  },
  fixture_results: fixtureResults,
  twelve_case_probe: {
    valid_count: 8,
    invalid_count: 4,
    rows: probeResults,
  },
  empirical_basis: {
    external_interface_population: "quant-policy-boundary-public-50-2026-08-05",
    artifacts: 50,
    external_parameters: 114,
    complete_affine_parameters: 62,
    explicitly_unquantized_parameters: 52,
    inference_boundary: "The corpus supports parameter identity binding and an explicit not-quantized state. It is not a probability sample and does not establish ecosystem prevalence or per-group behavior.",
  },
  interpretation: "The fixture set is validated against the exact modular JSON Schema graph from the pinned PR head. The 12-case matrix records current schema behavior; accepted cross-field combinations are not relabeled as invalid until the working group decides whether quantization objects are partial metadata or complete normalized contracts.",
};
const evidence = { ...body, ledger_sha256: sha256TextHex(canonicalJson(body)) };

await mkdir(OUT, { recursive: true });
await emit(DEFINITION_FILE, `${pretty(quantization)}\n`);
await emit(SCHEMA_SET_GZIP_FILE, gzipSync(schemaSetArchive, { level: 9, mtime: 0 }));
for (const fixture of fixtures) await emit(path.join(OUT, fixture.filename), `${pretty(fixture.document)}\n`);
await emit(EVIDENCE_FILE, `${pretty(evidence)}\n`);

console.log(`CycloneDX 2.0 quantization fixture evidence ${CHECK ? "check" : "build"} passed (${fixtureResults.length} fixtures; 8 valid / 4 invalid probes; ${HEAD.slice(0, 12)}).`);

function createValidator(schema, supportSchemas = []) {
  const ajv = new Ajv2020({ allErrors: true, strict: false, validateFormats: false });
  for (const [resolutionUri, supportSchema] of supportSchemas) {
    const normalized = structuredClone(supportSchema);
    delete normalized.$schema;
    normalized.$id = resolutionUri;
    ajv.addSchema(normalized, resolutionUri);
  }
  return ajv.compile(schema);
}

async function readUpstreamSchemaSet() {
  const rootFiles = (await readdir(path.join(UPSTREAM, "schema", "2.0"), { withFileTypes: true }))
    .filter((entry) => entry.isFile() && entry.name.endsWith(".schema.json"))
    .map((entry) => `schema/2.0/${entry.name}`);
  const modelFiles = (await readdir(path.join(UPSTREAM, "schema", "2.0", "model"), { withFileTypes: true }))
    .filter((entry) => entry.isFile() && entry.name.endsWith(".schema.json"))
    .map((entry) => `schema/2.0/model/${entry.name}`);
  const files = [...rootFiles, ...modelFiles, SPDX_REL, CRYPTOGRAPHY_REL].sort();
  return new Map(await Promise.all(files.map(async (schemaPath) => [
    schemaPath,
    await readFile(path.join(UPSTREAM, ...schemaPath.split("/"))),
  ])));
}

function encodeSchemaSet(files) {
  return Buffer.from(`${pretty({
    schema: "deepbom.pinned_json_schema_set.v1",
    files: [...files.entries()].map(([schemaPath, bytes]) => ({
      path: schemaPath,
      sha256: sha256(bytes),
      content_base64: bytes.toString("base64"),
    })),
  })}\n`);
}

function decodeSchemaSet(bytes) {
  const archive = JSON.parse(bytes.toString("utf8"));
  assert.equal(archive.schema, "deepbom.pinned_json_schema_set.v1");
  const files = new Map();
  for (const entry of archive.files ?? []) {
    const content = Buffer.from(entry.content_base64, "base64");
    assert.equal(sha256(content), entry.sha256, `Pinned schema hash mismatch: ${entry.path}`);
    files.set(entry.path, content);
  }
  return files;
}

function requireSchema(files, schemaPath) {
  const bytes = files.get(schemaPath);
  assert(bytes, `Pinned schema set is missing ${schemaPath}.`);
  return bytes;
}

function validatorSupportSchemas(files) {
  const excluded = new Set([ROOT_SCHEMA_REL, BUNDLED_REL, "schema/2.0/cyclonedx-2.0-bundled.min.schema.json", "schema/2.0/cyclonedx-api-2.0.schema.json"]);
  return [...files.entries()]
    .filter(([schemaPath]) => !excluded.has(schemaPath))
    .flatMap(([schemaPath, bytes]) => {
      const schema = JSON.parse(bytes.toString("utf8"));
      if (schemaPath === SPDX_REL) return [
        ["https://cyclonedx.org/schema/spdx.schema.json", schema],
        ["https://cyclonedx.org/schema/2.0/spdx.schema.json", schema],
      ];
      if (schemaPath === CRYPTOGRAPHY_REL) return [
        ["https://cyclonedx.org/schema/cryptography-defs.schema.json", schema],
        ["https://cyclonedx.org/schema/2.0/cryptography-defs.schema.json", schema],
      ];
      assert(schema.$id, `Pinned schema has no resolution URI: ${schemaPath}`);
      return [[schema.$id, schema]];
    });
}

function normalizeErrors(errors = []) {
  return (errors ?? []).map((error) => ({
    instancePath: error.instancePath,
    schemaPath: error.schemaPath,
    keyword: error.keyword,
    message: error.message,
  }));
}

async function emit(filename, content) {
  const next = Buffer.isBuffer(content) ? content : Buffer.from(content);
  if (!CHECK) {
    await writeFile(filename, next);
    return;
  }
  const current = await readFile(filename);
  assert(current.equals(next), `${path.relative(ROOT, filename)} is stale; rebuild from the pinned upstream checkout.`);
}

function fixtureDocuments() {
  return [
    fixture("valid-ai-ml-quantization-per-channel-2.0.json", true, {
      inputs: [parameter("image", "uint8", [1, 224, 224, 3], { bits: 8, scheme: "affine", granularity: "per-channel", axis: 3 })],
    }),
    fixture("valid-ai-ml-quantization-custom-scheme-2.0.json", true, {
      quantization: { method: "ternary-rtn", bits: 1.58, scheme: { name: "signed-ternary", description: "Three-level signed representation." }, granularity: { name: "per-block", description: "One mapping per implementation-defined block." } },
    }),
    fixture("invalid-ai-ml-quantization-zero-bits-2.0.json", false, {
      quantization: { bits: 0, scheme: "affine", granularity: "per-tensor" },
    }),
    fixture("invalid-ai-ml-quantization-zero-group-size-2.0.json", false, {
      quantization: { bits: 4, scheme: "affine", granularity: "per-group", groupSize: 0, axis: 0 },
    }),
    fixture("invalid-ai-ml-quantization-unknown-scheme-2.0.json", false, {
      quantization: { bits: 8, scheme: "affine_asymmetric", granularity: "per-tensor" },
    }),
  ];
}

function fixture(filename, expected, modelProperties) {
  const serial = createHash("sha256").update(filename).digest("hex");
  return {
    filename,
    expected,
    document: {
      $schema: "https://cyclonedx.org/schema/2.0/cyclonedx-2.0.schema.json",
      specFormat: "CycloneDX",
      specVersion: "2.0",
      serialNumber: `urn:uuid:${serial.slice(0, 8)}-${serial.slice(8, 12)}-4${serial.slice(13, 16)}-8${serial.slice(17, 20)}-${serial.slice(20, 32)}`,
      version: 1,
      components: [{
        type: "machine-learning-model",
        "bom-ref": `quant-fixture-${filename.replace(/\.json$/, "")}`,
        name: filename.replace(/\.json$/, ""),
        version: "1.0.0",
        modelProperties,
      }],
    },
  };
}

function parameter(name, dataType, shape, quantization) {
  return { name, dataType, shape, quantization };
}

function probeMatrix() {
  return [
    { id: "empty-object", value: {}, expected: true },
    { id: "per-tensor-axis", value: { granularity: "per-tensor", axis: 0 }, expected: true },
    { id: "per-tensor-group-size", value: { granularity: "per-tensor", groupSize: 32 }, expected: true },
    { id: "per-channel-no-axis", value: { granularity: "per-channel" }, expected: true },
    { id: "per-channel-group-size", value: { granularity: "per-channel", groupSize: 32 }, expected: true },
    { id: "per-group-no-axis-or-size", value: { granularity: "per-group" }, expected: true },
    { id: "custom-scheme-object", value: { scheme: { name: "ternary" } }, expected: true },
    { id: "fractional-bits", value: { bits: 1.58 }, expected: true },
    { id: "zero-bits", value: { bits: 0 }, expected: false },
    { id: "negative-axis", value: { axis: -1 }, expected: false },
    { id: "zero-group-size", value: { groupSize: 0 }, expected: false },
    { id: "unknown-scheme-string", value: { scheme: "affine_asymmetric" }, expected: false },
  ];
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
