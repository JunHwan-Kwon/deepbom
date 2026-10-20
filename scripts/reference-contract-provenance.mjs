// Reference documents keep the producer that actually generated them. A new
// producer creates a new immutable set; the historical 2.0.0 example is archived.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { canonicalJson } from "../web/lib/report-utils.js";
export const REFERENCE_CONTRACT_ANALYZER_METADATA = ANALYZER_METADATA;
export const REFERENCE_GENERATED_AT = "2026-10-20T00:00:00.000Z";
const root = new URL("../reference/cyclonedx/1.7/examples/", import.meta.url);
const index = new URL("current-reference.json", root);
const hash = value => createHash("sha256").update(value).digest("hex");
export const REFERENCE_DOCUMENT_KEYS = Object.freeze({cyclonedx:"cyclonedx_evidence",artifactEnvelope:"artifact_evidence_envelope",artifactIr:"artifact_ir",modelIr:"model_ir",interfaceContracts:"interface_contract_ledger",formulation:"observed_formulation",runtime:"runtime_requirement_manifest",missingFields:"missing_provenance_field_specification"});
function producer() {
  return {analyzer:ANALYZER_METADATA,wasm_sha256:hash(readFileSync(new URL("../pkg/tflite_wasm_audit_bg.wasm",import.meta.url)))};
}
export function validateReferenceProducer(record, expected=producer(), {allowBuildContext=false}={}) {
  assert.equal(record.schema,"deepbom.export_reference_set.v1");
  // A commit containing generated references necessarily changes Git context.
  // Reuse is permitted only for byte-identical code/rulepack/WASM; no version,
  // source-content hash or other producer field is ignored.
  const identity = value => {
    if (!allowBuildContext) return value;
    const {buildCommit,buildSourceState,...analyzer}=value.analyzer;
    return {...value,analyzer};
  };
  assert.equal(canonicalJson(identity(record.producer)),canonicalJson(identity(expected)),"Reference producer changed: regenerate the documents and producer together with npm run generate:cyclonedx-example; do not relabel archived evidence.");
  assert.equal(record.status,record.producer.analyzer.buildSourceState==="clean"?"clean_source_reference":"development_reference_not_released");
  assert.equal(record.generated_at,REFERENCE_GENERATED_AT);
  assert.match(record.directory,/^mobilenet-v2-quant-[a-f0-9]{64}$/);
}
export function validateArchivedReferences() {
  const archive=JSON.parse(readFileSync(new URL("historical-reference.json",root),"utf8"));
  assert.equal(archive.directory,"mobilenet-v2-quant");
  assert.equal(Object.keys(archive.files).length,8);
  for(const [file,sha] of Object.entries(archive.files)) {
    assert.equal(path.basename(file),file);
    assert.equal(hash(readFileSync(new URL(`${archive.directory}/${file}`,root))),sha,`Archived reference changed: ${file}`);
  }
}
export function readReferenceSet() {
  validateArchivedReferences();
  const record=JSON.parse(readFileSync(index,"utf8"));validateReferenceProducer(record,producer(),{allowBuildContext:true});
  const manifest=readFileSync(new URL(`${record.directory}/producer.json`,root));
  assert.equal(hash(manifest),record.manifest_sha256,"Reference manifest checksum mismatch");
  const {manifest_sha256,...body}=record;assert.equal(canonicalJson(JSON.parse(manifest)),canonicalJson(body));
  const {directory,...identity}=body;
  assert.equal(directory,`mobilenet-v2-quant-${hash(canonicalJson(identity))}`,"Reference directory identity mismatch");
  assert.equal(Object.keys(record.files).length,8);
  const documents={};
  for(const [file,sha] of Object.entries(record.files)) {
    assert.equal(path.basename(file),file);
    const bytes=readFileSync(new URL(`${record.directory}/${file}`,root));assert.equal(hash(bytes),sha,`Reference checksum mismatch: ${file}`);
    documents[file]=JSON.parse(bytes);
  }
  return {record,documents};
}
export function writeReferenceSet(set) {
  validateArchivedReferences();
  const documents={};
  for(const [key,file] of Object.entries(set.files)) {
    assert.equal(path.basename(file),file);assert.ok(REFERENCE_DOCUMENT_KEYS[key]);
    documents[file]=JSON.stringify(set.documents[REFERENCE_DOCUMENT_KEYS[key]],null,2)+"\n";
  }
  assert.equal(Object.keys(documents).length,8);
  const current=producer();
  const body={schema:"deepbom.export_reference_set.v1",status:current.analyzer.buildSourceState==="clean"?"clean_source_reference":"development_reference_not_released",generated_at:REFERENCE_GENERATED_AT,producer:current,files:Object.fromEntries(Object.entries(documents).map(([file,text])=>[file,hash(text)]))};
  // Hash includes the output documents as well as producer. Never overwrite a set.
  body.directory=`mobilenet-v2-quant-${hash(canonicalJson(body))}`;
  const target=new URL(body.directory+"/",root), staged=new URL(`.reference-${process.pid}/`,root);
  const manifest=JSON.stringify(body,null,2)+"\n";
  try {
    mkdirSync(staged);
    for(const [file,text] of Object.entries(documents)) writeFileSync(new URL(file,staged),text,{flag:"wx"});
    writeFileSync(new URL("producer.json",staged),manifest,{flag:"wx"});
    if(existsSync(target)) {
      assert.equal(readFileSync(new URL("producer.json",target),"utf8"),manifest,"Immutable reference collision");
      for(const [file,text] of Object.entries(documents)) assert.equal(readFileSync(new URL(file,target),"utf8"),text);
    } else renameSync(staged,target);
    const temporary=new URL(`.current-reference-${process.pid}.json`,root);
    writeFileSync(temporary,JSON.stringify({...body,manifest_sha256:hash(manifest)},null,2)+"\n",{flag:"wx"});
    renameSync(temporary,index);
  } finally {rmSync(staged,{recursive:true,force:true});}
  return readReferenceSet();
}
