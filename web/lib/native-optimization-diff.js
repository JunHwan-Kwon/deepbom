// Report-output projection, not a new Evidence IR layer. No framework execution.
import { canonicalJson } from "./report-utils.js";
import { compareNativeNodes, validateNativeTensorComparison, nativeDigest } from "./native-evidence.js";

const need = (ok, reason) => { if (!ok) throw Error(`Optimization diff: ${reason}`); };
const equal = (a,b) => canonicalJson(a) === canonicalJson(b);
const object = x => x !== null && typeof x === "object" && !Array.isArray(x);
const pointer = s => String(s).replace(/~/g,"~0").replace(/\//g,"~1");
const own = (x,k) => x !== null && Object.hasOwn(x,k);

// Leaves use genuine JSON Pointers. Arrays remain atomic, avoiding invented
// element correspondence when insertions change their positional indices.
export function optimizationFieldChanges(before, after, beforePresent=true, afterPresent=true) {
  const rows=[];
  function visit(a,b,pa,pb,path) {
    if(pa && pb && equal(a,b)) return;
    const ao=pa&&object(a),bo=pb&&object(b);
    if((ao&&bo || !pa&&bo || ao&&!pb) && Object.keys({...a,...b}).length) {
      const keys=[...new Set([...Object.keys(ao?a:{}),...Object.keys(bo?b:{})])].sort();
      for(const k of keys)visit(ao&&own(a,k)?a[k]:null,bo&&own(b,k)?b[k]:null,ao&&own(a,k),bo&&own(b,k),`${path}/${pointer(k)}`);
    } else rows.push({path,before_present:pa,after_present:pb,before:pa?a:null,after:pb?b:null});
  }
  visit(before,after,beforePresent,afterPresent,"");return rows;
}

export function buildNativeOptimizationDiff(report) {
  need(report?.schema === "deepbom.optimization_report.v1", "unsupported report schema");
  const {report_sha256,...body}=report;
  need(/^[a-f0-9]{64}$/.test(report_sha256) && nativeDigest(body)===report_sha256,"report digest mismatch");
  const {comparison,baseline,candidate}=report;
  need(comparison?.baseline===baseline?.snapshot_sha256 && comparison?.candidate===candidate?.snapshot_sha256,"comparison snapshot binding mismatch");
  const {comparison_sha256,...comparisonBody}=comparison;
  need(nativeDigest(comparisonBody)===comparison_sha256,"comparison digest mismatch");
  const verdicts=compareNativeNodes(baseline.nodes,candidate.nodes);
  need(equal(verdicts,comparison.nodes),"module verdicts disagree with common comparison");
  validateNativeTensorComparison(comparison.tensors, baseline.storage, candidate.storage);
  function side(source) {
    const storage=new Map(source.storage.map(t=>[t.id,t]));
    need(storage.size===source.storage.length,"duplicate storage identity");
    const nodes=new Map(source.nodes.map(n=>[n.id,n])),connections=new Map(),calls=new Map();
    for(const id of nodes.keys()){connections.set(id,[]);calls.set(id,[]);}
    for(const r of source.relationships) {
      need(nodes.has(r.from)&&nodes.has(r.to),"unbound relationship");
      for(const id of new Set([r.from,r.to]))connections.get(id).push(r);
    }
    const callIds=new Set();
    for(const c of source.invocations){
      const key=canonicalJson([c.subject,c.invocation]);
      need(nodes.has(c.subject)&&Number.isSafeInteger(c.invocation)&&c.invocation>0&&!callIds.has(key),"unbound or duplicate call");
      callIds.add(key);calls.get(c.subject).push(c);
    }
    return new Map(source.nodes.map(n=>{
      need(Array.isArray(n.state_refs)&&n.state_refs.every(id=>storage.has(id)),"unbound state reference");
      return [n.id,{module:n,
        recorded_connections:connections.get(n.id).sort((a,b)=>{const x=canonicalJson(a),y=canonicalJson(b);return x<y?-1:x>y?1:0;}),
        registered_state:n.state_refs.map(id=>storage.get(id)),observed_calls:calls.get(n.id)}];
    }));
  }
  const a=side(baseline),b=side(candidate),reasons=new Map(),rules=[];
  for(const [index,change] of report.changes.entries()) {
    need(change.index===index+1,"nonsequential rule index");
    for(const n of change.affected_nodes) {
      need(verdicts.some(v=>v.id===n.id&&v.status===n.status),"unbound rule subject");
      if(!reasons.has(n.id))reasons.set(n.id,[]);reasons.get(n.id).push(change.index);
    }
    rules.push(Object.fromEntries(["index","title","subject","reason","description","tradeoff","state","rule","rule_version"].map(k=>[k,change[k]])));
  }
  const byId=new Map(verdicts.map(n=>[n.id,n])),order=[...b.keys(),...[...a.keys()].filter(id=>!b.has(id))];
  const entries=order.map(id=>{
    const before=a.get(id)??null,after=b.get(id)??null,status=byId.get(id).status,tags=status==="unchanged"?[]:["structure"];
    for(const [field,label] of [["registered_state","state"],["recorded_connections","connections"],["observed_calls","observations"]])
      if(!equal(before?.[field]??[],after?.[field]??[]))tags.push(label);
    return {id,status,tags,before,after,rules:reasons.get(id)??[],fields:optimizationFieldChanges(before,after,before!==null,after!==null)};
  });
  const orphaned=source=>{
    const owned=new Set(source.nodes.flatMap(n=>n.state_refs));
    return source.storage.filter(t=>!owned.has(t.id));
  };
  const unownedBefore=orphaned(baseline),unownedAfter=orphaned(candidate);
  if(!equal(unownedBefore,unownedAfter))entries.push({id:"Unowned registered state",status:"context",tags:["state"],before:unownedBefore,after:unownedAfter,rules:[],fields:optimizationFieldChanges(unownedBefore,unownedAfter)});
  const fields=["framework","scope","inputs","outputs","shape_observation_status","limitations"];
  const before=Object.fromEntries(fields.map(k=>[k,baseline[k]])),after=Object.fromEntries(fields.map(k=>[k,candidate[k]]));
  const sharedBefore=[...a.keys()].filter(k=>b.has(k)),sharedAfter=[...b.keys()].filter(k=>a.has(k));
  if(!equal(sharedBefore,sharedAfter)){before.shared_module_source_order=sharedBefore;after.shared_module_source_order=sharedAfter;}
  const sortedRelations=x=>x.map(canonicalJson).sort();
  if(!equal(baseline.relationships,candidate.relationships)&&equal(sortedRelations(baseline.relationships),sortedRelations(candidate.relationships))){
    before.relationship_source_order=baseline.relationships;after.relationship_source_order=candidate.relationships;
  }
  if(!entries.some(e=>e.tags.length)&&!equal(baseline.model_ir_sha256,candidate.model_ir_sha256)){
    before.unprojected_model_ir_identity=baseline.model_ir_sha256;after.unprojected_model_ir_identity=candidate.model_ir_sha256;
  }
  if(!equal(before,after))entries.unshift({id:"Model contracts / capture",status:"context",tags:["context"],before,after,rules:[],fields:optimizationFieldChanges(before,after)});
  const counts={added:0,removed:0,changed:0,unchanged:0};for(const n of verdicts)counts[n.status]++;
  return {schema:"deepbom.optimization_diff.v1",report_sha256,before_sha256:baseline.snapshot_sha256,after_sha256:candidate.snapshot_sha256,
    counts,changed_entries:entries.filter(e=>e.tags.length).length,entries,rules,
    boundary:"Read-only projection of a hash-checked report. Native-ID alignment, not inferred renames or functional equivalence. A self-consistent report is not independent attestation of model bytes, execution or performance."};
}
