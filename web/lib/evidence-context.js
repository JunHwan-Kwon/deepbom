import {openOptimizationReport} from "./optimization-report-access.js";
import {validateSemanticArtifactDiff} from "./semantic-artifact-diff.js";
import { validateArtifactEvidenceIr } from "./artifact-ir.js";
import { validateArtifactSet } from "./artifact-set.js";
import { validateModelIr, validateModelIrAgainstSource } from "./model-ir.js";
import { validateWeightIr } from "./weight-ir.js";
import { validateActivationIr } from "./activation-ir.js";
import { validateNativeDocument, validateTrainingIr, validateTrainingChunk, validateTrainingBundle } from "./native-evidence.js";
import { validateSnapshotIr } from "./snapshot-ir.js";
import { sha256BytesHex } from "./sha256-sync.js";
import { canonicalEvidenceDigest, requireEvidence as need, referenceKey, subjectReference, validateSubjectReference } from "./evidence-identity.js";

const CONTRACTS = {
  "deepbom.training_chunk.v1": [validateTrainingChunk,"chunk_sha256","evidence_document"],
  "deepbom.optimization_report.v1": [d=>openOptimizationReport(JSON.stringify(d)),"report_sha256","evidence_document"],
  "deepbom.semantic_artifact_diff.v1": [validateSemanticArtifactDiff,"semantic_diff_sha256","evidence_document"],
  "deepbom.artifact_ir.v2": [validateArtifactEvidenceIr, "artifact_ir_sha256", "evidence_document"],
  "deepbom.artifact_set.v1": [validateArtifactSet, "artifact_set_sha256", "artifact_set"],
  "deepbom.model_ir.v1": [validateModelIr, "model_ir_sha256", "evidence_document"],
  "deepbom.model_ir.v2": [validateNativeDocument, "model_ir_sha256", "evidence_document"],
  "deepbom.model_state_snapshot.v1": [validateNativeDocument, "snapshot_sha256", "native_state"],
  "deepbom.weight_ir.v2": [validateNativeDocument, "weight_ir_sha256", "evidence_document"],
  "deepbom.activation_ir.v2": [validateNativeDocument, "activation_ir_sha256", "evidence_document"],
  "deepbom.training_ir.v1": [validateTrainingIr, "training_ir_sha256", "evidence_document"],
  "deepbom.snapshot_ir.v1": [validateSnapshotIr, "snapshot_ir_sha256", "snapshot"],
};

// Validators supplied by the workflow are code-owned, never request arguments.
export function createEvidenceContext(documents = [], { validators = {}, files = [], recordedFiles = [] } = {}) {
  need(Array.isArray(documents) && documents.length <= 512 && Array.isArray(files) && files.length <= 256, "context input limit exceeded");
  need(files.every(f=>f.bytes instanceof Uint8Array && f.bytes.byteLength<=64*1024*1024) && files.reduce((n,f)=>n+f.bytes.byteLength,0)<=256*1024*1024,"context file byte limit exceeded");
  const entries = new Map(), byDigest = new Map();
  const pending = [];
  for (const original of documents) {
    const doc = structuredClone(original);
    const contract = CONTRACTS[doc.schema] || validators[doc.schema];
    if (["deepbom.weight_ir.v1", "deepbom.activation_ir.v1"].includes(doc.schema)) { pending.push(doc); continue; }
    need(contract, `unsupported context contract ${doc.schema}`);
    const [validate, field, kind] = contract;
    validate(doc);
    insert(subjectReference(kind, doc[field], { schema: doc.schema }), doc);
  }
  for (const doc of pending) {
    const model = byDigest.get(doc.source?.model_ir_sha256);
    need(model?.schema === "deepbom.model_ir.v1", "numerical evidence requires its artifact Model IR");
    const weight = doc.schema === "deepbom.weight_ir.v1";
    (weight ? validateWeightIr : validateActivationIr)(doc, model);
    const field = weight ? "weight_ir_sha256" : "activation_ir_sha256";
    insert(subjectReference("evidence_document", doc[field], { schema: doc.schema }), doc);
  }
  for (const file of files) {
    validateSubjectReference(file.ref);
    need(file.ref.kind === "artifact_file" && file.ref.subject_ref === null && file.bytes instanceof Uint8Array, "supplied files require bytes and a file reference");
    need(sha256BytesHex(file.bytes) === file.ref.sha256, "supplied file SHA-256 mismatch");
    insert(file.ref, null);
  }
  // Only the saved-result verifier supplies these observations. They never prove
  // that file bytes were available at the time of a later reading.
  need(Array.isArray(recordedFiles) && recordedFiles.length <= 256, "recorded file limit exceeded");
  for (const ref of recordedFiles) {
    validateSubjectReference(ref);
    need(ref.kind === "artifact_file" && ref.subject_ref === null, "recorded file reference required");
    insert(ref, null);
  }
  for (const d of byDigest.values()) {
    if (d.schema === "deepbom.training_ir.v1") {
      const chunks=d.chunks.map(r=>byDigest.get(r.sha256));
      if(chunks.every(Boolean)){
        const events=chunks.flatMap(c=>c.events);
        const required=events.flatMap(e=>e.kind === "model_snapshot" ? Object.values(e.payload.evidence||{}) : e.kind === "activation_capture" ? [e.payload.evidence] : []);
        if(required.every(r=>r && byDigest.has(r.sha256)))validateTrainingBundle({training:d,events,objects:Object.fromEntries(byDigest)});
      }
    }
    if (d.schema === "deepbom.model_ir.v1") {
      const source=byDigest.get(d.source_contract.sha256);
      if(source)validateModelIrAgainstSource(d,source,{nativeFactLedger:d.source_contract.native_fact_ledger});
    }
    if (d.schema === "deepbom.model_ir.v2") {
      const snapshot = byDigest.get(d.source.snapshot_sha256);
      if (snapshot) need(snapshot.schema === "deepbom.model_state_snapshot.v1" && canonicalEvidenceDigest(d.program) === canonicalEvidenceDigest(snapshot.definition) && canonicalEvidenceDigest(d.storage) === canonicalEvidenceDigest(snapshot.tensors), "native model/snapshot binding mismatch");
    }
    if (["deepbom.weight_ir.v2", "deepbom.activation_ir.v2"].includes(d.schema)) {
      const model = byDigest.get(d.source.model_ir_sha256), snapshot = byDigest.get(d.source.snapshot_sha256);
      if (model) {
        need(model.schema === "deepbom.model_ir.v2" && model.source.snapshot_sha256 === d.source.snapshot_sha256, "native numerical/model binding mismatch");
        if (d.schema === "deepbom.activation_ir.v2") need(d.tensors.every(t => model.program.nodes.some(n => n.id === t.subject_ref)), "activation refers to a missing node");
      }
      if (snapshot && d.schema === "deepbom.weight_ir.v2") need(snapshot.schema === "deepbom.model_state_snapshot.v1" && canonicalEvidenceDigest(d.tensors.map(r=>r.identity)) === canonicalEvidenceDigest(snapshot.tensors), "weight snapshot identities mismatch");
    }
  }
  function insert(ref, doc) {
    const key = referenceKey(ref);
    need(!entries.has(key), "duplicate context subject");
    if (doc) {
      need(!byDigest.has(ref.sha256), "ambiguous document digest");
      byDigest.set(ref.sha256, doc);
    }
    entries.set(key, { ref, document: doc });
  }
  function resolve(ref, active = new Set()) {
    validateSubjectReference(ref);
    const whole = { ...ref, subject_ref: null }, entry = entries.get(referenceKey(whole));
    if (!entry) return { ref, status: "unresolved", reason: "Referenced bytes/document not supplied" };
    if (ref.subject_ref !== null) {
      const d = entry.document;
      const ids = d?.schema === "deepbom.model_ir.v2" ? [...d.program.nodes.map(n => n.id), ...d.storage.map(n => n.id)]
        : d?.schema === "deepbom.model_ir.v1" ? [...d.program.operations,...d.program.values,...d.tensors_and_storage.storage_objects].map(n=>n.id)
        : d?.schema === "deepbom.artifact_ir.v2" ? [...d.graph.operators,...d.graph.values,...d.storage_topology.objects].map(n=>n.id) : null;
      need(ids && ids.includes(ref.subject_ref), "internal subject does not exist or is unsupported by this contract");
    }
    const key = referenceKey(whole), required = documentDependencies(entry.document);
    if (required.length) {
      need(!active.has(key), "cyclic evidence dependency");
      const next = new Set(active); next.add(key);
      if(required.some(r=>resolve(r,next).status!=="matched"))return {ref,status:"unresolved",reason:"Document is present but one or more required source documents or bytes are unresolved"};
    }
    return { ref, status: "matched", verification: ref.kind === "artifact_file" ? "supplied_file_bytes" : "document_semantics_and_digest", authenticity: "not_verified" };
  }
  const dependencies = new Map();
  for (const [key, entry] of entries) {
    const d = entry.document;
    const refs = documentDependencies(d);
    dependencies.set(key, refs.map(referenceKey));
  }
  function visit(key, active, done) {
    need(!active.has(key), "cyclic snapshot dependency");
    if (done.has(key)) return;
    active.add(key);
    for (const dep of dependencies.get(key) || []) visit(dep, active, done);
    active.delete(key); done.add(key);
  }
  const done = new Set();
  for (const key of dependencies.keys()) visit(key, new Set(), done);
  const refs = [...entries.values()].map(e => e.ref).sort((a,b) => referenceKey(a) < referenceKey(b) ? -1 : referenceKey(a) > referenceKey(b) ? 1 : 0);
  return Object.freeze({
    resolve,
    document(ref) { validateSubjectReference(ref); const entry=entries.get(referenceKey({ ...ref, subject_ref: null })); return entry?.document ? structuredClone(entry.document) : null; },
    references: Object.freeze(refs),
    context_sha256: canonicalEvidenceDigest(refs),
  });
}

function documentDependencies(d){
  if(!d)return [];
  const documentRef=(schema,sha256)=>subjectReference("evidence_document",sha256,{schema});
  if(d.schema==="deepbom.snapshot_ir.v1")return d.contents.map(r=>r.ref);
  if(d.schema==="deepbom.artifact_set.v1")return d.files.map(f=>subjectReference("artifact_file",f.sha256));
  if(d.schema==="deepbom.artifact_ir.v2")return [subjectReference("artifact_file",d.artifact.sha256),...(d.artifact.artifact_set_sha256?[subjectReference("artifact_set",d.artifact.artifact_set_sha256,{schema:"deepbom.artifact_set.v1"})]:[])];
  if(d.schema==="deepbom.model_ir.v1")return [documentRef("deepbom.artifact_ir.v2",d.source_contract.sha256)];
  if(d.schema==="deepbom.training_ir.v1")return d.chunks.map(r=>documentRef("deepbom.training_chunk.v1",r.sha256));
  if(d.schema==="deepbom.training_chunk.v1")return d.events.flatMap(e=>e.kind==="model_snapshot"?Object.values(e.payload.evidence||{}):e.kind==="activation_capture"?[e.payload.evidence]:[]).map(r=>subjectReference(r.schema==="deepbom.model_state_snapshot.v1"?"native_state":"evidence_document",r.sha256,{schema:r.schema}));
  if(d.source?.snapshot_sha256)return [subjectReference("native_state",d.source.snapshot_sha256,{schema:"deepbom.model_state_snapshot.v1"}),...(d.source.model_ir_sha256?[documentRef("deepbom.model_ir.v2",d.source.model_ir_sha256)]:[])];
  if(d.source?.model_ir_sha256)return [documentRef("deepbom.model_ir.v1",d.source.model_ir_sha256)];
  return [];
}
