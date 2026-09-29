// Descriptive crosswalk only. Calculations and capability decisions stay in their owners.
import { EVIDENCE_IR_LAYERS, PROVENANCE_INPUT_SCHEMA } from "../web/lib/evidence-ir.js";
import { PUBLIC_PRODUCT_CONTRACTS } from "../web/lib/public-product-contracts.js";
import { AUDIT_OUTPUT_CONTRACTS } from "../web/lib/audit-output-contracts.js";

export const CATALOG_VERSION = "0.2.0";
const native = {
  tflite: ["TFLite", "Model / SubGraph / Tensor / Buffer", "src/lib.rs", "scripts/check-artifact-ir.mjs"],
  onnx: ["ONNX", "ModelProto.graph / NodeProto / ValueInfoProto / TensorProto", "web/onnx.js", "scripts/check-model-ir-source-contracts.mjs"],
  gguf: ["GGUF", "Header / metadata key-values / tensor directory / GGML blocks", "web/lib/metadata-model-adapters.js", "scripts/check-artifact-ir.mjs"],
  safetensors: ["SafeTensors", "JSON header / dtype / shape / data_offsets / shard index", "web/lib/metadata-model-adapters.js", "scripts/check-metadata-model-adapters.mjs"],
  coreml: ["Core ML", "Model description / NeuralNetwork / ML Program / package manifest", "web/lib/coreml-metadata-adapter.js", "scripts/check-model-ir-source-contracts.mjs"],
  executorch: ["ExecuTorch", "ET12 plans / EValues / instructions / segments / FT01", "web/executorch.js", "scripts/check-model-ir-source-contracts.mjs"],
  graphdef: ["TensorFlow GraphDef", "GraphDef.node / inputs / Placeholder / Const", "web/lib/metadata-model-adapters.js", "scripts/check-metadata-model-adapters.mjs"],
  savedmodel: ["TensorFlow SavedModel", "First MetaGraph / SignatureDef / package member identities", "web/lib/metadata-model-adapters.js", "scripts/check-metadata-model-adapters.mjs"],
  hdf5: ["HDF5", "Superblock / safe container header", "web/lib/metadata-model-adapters.js", "scripts/check-metadata-model-adapters.mjs"],
  keras: ["Keras archive", "config.json / Sequential order / keras_history", "web/lib/metadata-model-adapters.js", "scripts/check-metadata-model-adapters.mjs"],
  pt2: ["PyTorch PT2", "First models/*.json ExportedProgram dependencies", "web/lib/metadata-model-adapters.js", "scripts/check-metadata-model-adapters.mjs"],
  pytorch_checkpoint: ["PyTorch checkpoint", "Archive members / pickle opcode inventory", "web/lib/metadata-model-adapters.js", "scripts/check-metadata-model-adapters.mjs"],
};
const node = (id, kind, name, contract, boundary, json_schema = null) => ({ id, kind, name, contract, boundary, json_schema });
const field = (source, target) => ({ source, target });
const rows = [];
const map = (id, from, to, status, fields, owner, checks, boundary) => rows.push({ id, from, to, status, fields, implementation: owner, checks, boundary });
const nodes = EVIDENCE_IR_LAYERS.map(l => ({ ...node(l.id, "ir", l.name, l.schema, l.optional ? "Optional; source-bound evidence only." : "Availability of individual fields depends on the serialized artifact.", l.json_schema), digest_field: l.digest_field }));

for (const maturity of PUBLIC_PRODUCT_CONTRACTS.format_maturity) {
  const [name, contract, owner, check] = native[maturity.format] || [];
  if (!name) throw Error(`Missing compatibility crosswalk for ${maturity.format}`);
  const id = `input-${maturity.format}`;
  nodes.push({ ...node(id, "input", name, contract, maturity.interpretation_boundary), maturity });
  const fields = [field("complete artifact bytes / verified package members", "/artifact"), field("format facts exposed by the adapter", "/completeness")];
  if (!["hdf5", "pytorch_checkpoint"].includes(maturity.format)) fields.push(field(contract, "/logical_inventory"));
  if (["tflite", "onnx", "gguf", "safetensors", "coreml", "executorch"].includes(maturity.format)) fields.push(field("serialized storage objects / exact byte ranges when available", "/storage_topology"), field("serialized encodings and quantization declarations", "/quantization_contracts"));
  if (maturity.serialized_graph !== "not_applicable") fields.push(field("decoded serialized dependencies within the adapter's scope", "/graph"));
  map(`${maturity.format}-artifact`, id, "artifact", maturity.overall === "preview" ? "preview" : "conditional", fields, [owner, "web/lib/artifact-ir.js"], [check], maturity.interpretation_boundary);
  if (["tflite", "onnx", "gguf", "safetensors", "coreml", "executorch"].includes(maturity.format)) {
    map(`${maturity.format}-weight`, id, "weight", "conditional", [field("supported numeric payloads + Model IR storage bindings", "/tensors"), field("decoder availability / value budget / unlinked payloads", "/coverage")], ["web/lib/numerical-ir/weight-sources.js", "web/lib/weight-ir.js"], ["scripts/check-numerical-ir.mjs"], "Explicit opt-in. Model IR and exact source bytes are required. Unsupported encodings, missing external data and exhausted budgets remain not_assessed; integer codes are not automatically dequantized.");
  }
}

nodes.push(
  node("input-activation", "input", "Activation capture", "deepbom.activation_capture.v1", "Imported run, probe, inputs and captures; no execution or attestation by the static importer.", "https://deepbom.org/schemas/deepbom-numerical-ir-v1.schema.json#/$defs/activation_capture"),
  node("input-provenance", "input", "Metadata & lineage", PROVENANCE_INPUT_SCHEMA, "Declared entities and relationships plus separately supplied files.", "https://deepbom.org/schemas/deepbom-provenance-ir-v1.schema.json#/$defs/generic_input"),
  node("input-omop", "input", "OMOP metadata", "deepbom.omop_metadata_input.v1", "DEEPBOM profile for CDM_SOURCE 5.4 / 5.5 and declared research records. No patient rows or new OMOP vocabulary.", "https://deepbom.org/schemas/deepbom-provenance-ir-v1.schema.json#/$defs/omop_input"),
  node("input-files", "input", "Supporting files & BOM references", "Provided files + declared BOM identifiers", "Local file hashes and selected BOM references can be checked. Declared URLs are not fetched."),
  node("input-runtime", "input", "Runtime placement evidence", "Format-specific imported runtime evidence", "Runtime observations and anticipated Core ML plans retain their original evidence class."),
  node("input-conversion", "input", "Conversion receipt", "deepbom.conversion_receipt.v1", "Source/target artifact identities and declared conversion process; not proof the process occurred."),
  node("analysis", "context", "Analysis & finding ledgers", "Format-specific analysis + normalized findings", "Supplementary analysis context. Some exports consume this alongside IR; not every output is an IR-only projection."),
);
map("artifact-model", "artifact", "model", "implemented", [field("/graph", "/program"), field("/storage_topology", "/tensors_and_storage"), field("/quantization_contracts", "/quantization"), field("/overlays", "/static_runtime"), field("/overlays", "/observed_runtime"), field("/artifact_ir_sha256", "/source_contract"), field("/completeness", "/loss_ledger")], ["web/lib/model-ir.js", "web/lib/model-ir/internal/program.js"], ["scripts/check-model-ir.mjs", "scripts/check-model-ir-source-contracts.mjs"], "Validated projection with a hash-bound native-fact ledger. Native-only facts remain in the source; missing graphs stay not_serialized. Static estimates and observed runtime are separate.");
map("model-weight-binding", "model", "weight", "implemented", [field("/tensors_and_storage", "/tensors"), field("/weight_bindings", "/tensors"), field("/model_ir_sha256", "/source")], ["web/lib/weight-ir.js"], ["scripts/check-numerical-ir.mjs"], "Bindings identify serialized storage usage, not trainable parameters. Statistics also require the original payload decoder.");
map("capture-activation", "input-activation", "activation", "conditional", [field("run", "/run"), field("inputs", "/inputs"), field("captures with Model IR value references", "/tensors"), field("requested_value_refs / missing", "/coverage")], ["web/lib/activation-ir.js"], ["scripts/check-numerical-ir.mjs"], "Source identity, run configuration, tensor contracts and counts are checked; execution provenance is not attested. Captured inputs are retained separately from run metadata.");
map("model-activation-binding", "model", "activation", "implemented", [field("/model_ir_sha256", "/source"), field("/program", "/requested_value_refs")], ["web/lib/activation-ir.js"], ["scripts/check-numerical-ir.mjs"], "Requested value references and entry region must resolve to the same model. A static graph does not produce activation values.");
map("metadata-provenance", "input-provenance", "provenance", "conditional", [field("nodes", "/nodes"), field("relationships", "/relationships"), field("original input", "/input_document"), field("declared fields including unmapped fields", "/field_ledger")], ["web/lib/provenance-ir.js", "web/lib/provenance/contracts.js"], ["scripts/check-provenance.mjs"], "Declarations remain DECLARED_UNVERIFIED. Reference consistency is checked; metadata truth and publisher authenticity are not established.");
map("omop-provenance", "input-omop", "provenance", "conditional", [field("CDM_SOURCE / instance_id / release identity", "/nodes"), field("CDM version and source-field coverage", "/profile"), field("cohort / feature / ETL / vocabulary / quality references", "/relationships"), field("mapped and unsupported declared fields", "/field_ledger")], ["web/lib/provenance/omop.js", "web/lib/provenance-ir.js"], ["scripts/check-provenance.mjs"], "OMOP retains clinical meaning. External records are referenced, not copied into clinical tables. Unrecognized fields remain visible; FAIR or OMOP conformance is not certified.");
map("files-provenance", "input-files", "provenance", "conditional", [field("provided file bytes", "/observations"), field("expected SHA-256 / BOM serialNumber, version, bom-ref or SPDX identifiers", "/checks")], ["web/lib/provenance/files.js", "web/lib/provenance/bom-reference.js"], ["scripts/check-provenance.mjs"], "Only supplied-file consistency and supported reference selectors are checked. External BOM truth, software inventory completeness and URL ownership are not verified.");
map("model-provenance-binding", "model", "provenance", "implemented", [field("/artifact", "/source"), field("/model_ir_sha256", "/source")], ["web/lib/provenance-ir.js"], ["scripts/check-provenance.mjs"], "Primary model identity is fixed. Metadata cannot rewrite the artifact or Model IR digest.");
map("runtime-artifact", "input-runtime", "artifact", "conditional", [field("identity-bound provider/device/operation rows", "/overlays")], ["web/lib/artifact-ir/internal/overlays.js"], ["scripts/check-artifact-ir.mjs"], "Coverage depends on the runtime importer and operation mapping. An anticipated compute plan is not observed execution; source eligibility is not actual placement.");
map("conversion-artifact", "input-conversion", "artifact", "conditional", [field("source / target / process / file identity", "/lineage_evidence")], ["web/lib/conversion-receipt.js"], ["scripts/check-artifact-ir.mjs"], "A matching receipt binds a declaration; it does not authenticate the converter or establish numerical equivalence.");

for (const layer of EVIDENCE_IR_LAYERS) {
  const id = `output-${layer.id}`;
  nodes.push(node(id, "output", `${layer.name} JSON`, layer.schema, "Complete member document when materialized; structural and semantic validation remain required.", layer.json_schema));
  map(`${layer.id}-json`, layer.id, id, "implemented", [field("(whole document)", "(whole document)")], [`web/lib/${layer.id === "artifact" ? "artifact-ir" : `${layer.id}-ir`}.js`], ["scripts/check-evidence-ir-family.mjs"], "Preserves the member schema and digest. The family schema selects one member, not a bundle. Optional members are not generated unless requested and supplied.");
}
const auditNames = { summary: "Audit summary", envelope: "Evidence envelope", json: "Full analysis JSON", "json-compact": "Compact analysis JSON", cyclonedx: "CycloneDX", sarif: "SARIF" };
for (const [id, contract] of Object.entries(AUDIT_OUTPUT_CONTRACTS)) nodes.push(node(`output-${id}`, "output", auditNames[id] || id, contract.schema || contract.derived_from || (id === "cyclonedx" ? `CycloneDX ${contract.spec_version}` : id === "sarif" ? `SARIF ${contract.version}` : contract.stability), "CLI audit output contract; projections differ in completeness."));
nodes.push(
  node("output-spdx", "output", "SPDX artifact inventory", "SPDX-2.3", "Web/widget FILE-purpose package inventory. Not SPDX 3 AI profile; not a CLI audit output-format."),
  node("output-model-summary", "output", "Model summary", "deepbom.model_summary.v1", "Operation, block or storage table / Markdown / JSON."),
  node("output-visualization", "output", "Model visualization", "deepbom.model_ir_visualization_manifest.v1", "Deterministic SVG / derivative PNG / caption and document bundle."),
  node("output-tea", "output", "TEA exchange", "Transparency Exchange API", "Proposed external discovery and delivery adapter; not implemented."),
);
for (const id of ["json", "json-compact", "envelope", "summary", "cyclonedx", "sarif"]) map(`analysis-${id}`, "analysis", `output-${id}`, "implemented", [field("artifact identity / native analysis / findings / coverage", "export-specific identity, evidence and finding fields")], [id === "cyclonedx" ? "web/lib/public-cyclonedx-export.js" : id === "envelope" ? "web/lib/artifact-evidence-envelope.js" : "bin/deepbom-automation.mjs"], ["scripts/check-cli-automation.mjs"], "Consumes analysis context. Projection is not an invertible conversion and is not proof that every Evidence IR field is exported. SARIF findings retain defect/caution/evidence-gap classification.");
map("artifact-envelope", "artifact", "output-envelope", "conditional", [field("/graph", "graph"), field("/artifact", "identity")], ["web/lib/artifact-evidence-envelope.js"], ["scripts/check-artifact-ir-consumers.mjs"], "Canonical context supplies graph selectors; envelope also consumes analysis findings and format capabilities. Envelope schema is independent of IR schemas.");
map("provenance-cyclonedx", "provenance", "output-cyclonedx", "conditional", [field("/nodes", "components / metadata.component.modelCard.modelParameters.datasets"), field("/relationships", "formulation.workflows / deepbom:provenance:relationships"), field("/field_ledger", "deepbom:provenance:fieldLedger"), field("/checks", "deepbom:provenance:checks"), field("/provenance_ir_sha256", "deepbom:provenance:irSha256")], ["web/lib/provenance/cyclonedx.js"], ["scripts/check-provenance.mjs"], "Requires an identity-matching CycloneDX 1.7 base. Only resolved, kind-compatible relations enter standard references; remaining meaning stays in namespaced properties. No new standard fields.");
map("envelope-spdx", "output-envelope", "output-spdx", "implemented", [field("identity / external_files", "packages[].checksums / relationships"), field("graph / findings / envelope_sha256", "annotations[].comment")], ["web/lib/spdx-artifact-export.js"], ["scripts/check-spdx-artifact-export.mjs"], "Serialized file inventory with SHA-256. License conclusions are NOASSERTION. Rich IR and lineage semantics are not mapped into an SPDX AI profile.");
map("model-summary", "model", "output-model-summary", "conditional", [field("/program", "rows / deterministic display order"), field("/tensors_and_storage", "storage totals"), field("/program", "entry-region operation MAC totals"), field("/weight_bindings", "bound storage totals")], ["web/lib/model-summary.js"], ["scripts/check-model-ir.mjs"], "Storage rows are used where no executable graph is serialized. Display order is not an execution schedule; unassessed counts do not become zero.");
map("model-visualization", "model", "output-visualization", "conditional", [field("/program", "architecture and block views"), field("/architecture", "structural groups"), field("/static_runtime", "static-runtime view"), field("/observed_runtime", "observed-runtime view"), field("/model_ir_sha256", "manifest source identity")], ["web/lib/model-ir-visualization.js"], ["scripts/check-model-ir-visualization.mjs"], "Requested views retain explicit unavailable states when evidence is absent. Visual groups are not framework layers or runtime fusion; PNG is a derivative of the canonical SVG.");
map("provenance-tea", "provenance", "output-tea", "proposed", [], ["docs/evidence-ir/TEA_REVIEW.md"], [], "Future delivery of a complete exported document. TEA artifact-file checksum, IR canonical digest and model checksum must remain different identities. No TEA endpoint or client exists in this integration.");

export function compatibilityDefinition() {
  return {
    schema: "deepbom.evidence_compatibility_catalog.v1", catalog_version: CATALOG_VERSION, status: "draft",
    baseline: "Canonical Provenance IR; current Evidence IR members only. Retired identifiers are not aliases.",
    scope: "Core input-to-IR crosswalk and principal evidence exports. Field groups are navigation aids, not an exhaustive native-format field inventory or a claim of round-trip equivalence. Auxiliary CLI commands and runtime-specific profile fields are outside this catalog.",
    validation_boundary: "Implementation and regression-check references are traceability links, not test execution receipts or external certification. Artifact-specific coverage is reported in the resulting IR.",
    version_policy: "Catalog, member schema, calculation method and engine versions are independent. Preserve snapshots; update the catalog version whenever its pinned definitions change.",
    nodes, mappings: rows,
  };
}
