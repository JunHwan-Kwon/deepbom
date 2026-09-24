import { canonicalJson, compareCanonicalText } from "./report-utils.js";
import { exactInteger, validateExactInteger } from "./exact-integer.js";
import { validateArtifactSet } from "./artifact-set.js";

const SHA = /^[a-f0-9]{64}$/;
export function buildArtifactMembers(analysis, identity, storage) {
  const acquisition = analysis.artifact_set ? validateArtifactSet(analysis.artifact_set) : null;
  const bundle = analysis.artifact_bundle;
  const rows = acquisition?.files || bundle?.files || [];
  const members = [{ id: "artifact:primary", kind: bundle ? "bundle" : "artifact", path: identity.filename,
    sha256: identity.sha256, byte_length: identity.byte_length, verification: "identity_bound" }];
  const byPath = new Map();
  for (const row of [...rows].sort((a, b) => compareCanonicalText(a.path, b.path))) {
    const path = String(row.path || ""), id = `member:${encodeURIComponent(path)}`;
    if (!path || byPath.has(path)) throw new Error("IR artifact member path is missing or duplicated.");
    const member = { id, kind: String(row.role || "file"), path, sha256: SHA.test(row.sha256) ? row.sha256 : null,
      byte_length: row.byte_length?.decimal ? row.byte_length : exactInteger(row.byte_length), verification: SHA.test(row.sha256) ? "identity_bound" : "declared_unverified" };
    byPath.set(path, member); members.push(member);
  }
  const bindings = [];
  for (const object of storage.objects) {
    const external = object.native_source?.external_data;
    if (external) {
      const path = external.file_path || external.entries?.find(row => row.key === "location")?.value;
      if (path) {
        let member = byPath.get(path);
        if (!member) { member = { id: `member:${encodeURIComponent(path)}`, kind: "external_data", path, sha256: external.file_sha256,
          byte_length: external.file_byte_length, verification: external.verified ? "identity_bound" : "declared_unverified" }; byPath.set(path, member); members.push(member); }
        if (external.file_sha256 && member.sha256 && member.sha256 !== external.file_sha256) throw new Error("IR external member hash conflicts with acquisition evidence.");
        bindings.push({ storage_ref: object.id, member_ref: member.id, range: object.byte_range });
        continue;
      }
    }
    bindings.push({ storage_ref: object.id, member_ref: bundle ? null : "artifact:primary", range: object.byte_range });
  }
  const memberById = new Map(members.map(row => [row.id, row]));
  for (const row of bindings) row.bounds_assessment = rangeBounds(row, memberById.get(row.member_ref));
  return { schema: "deepbom.artifact_members.v1", members, storage_bindings: bindings, overlap_groups: overlapGroups(bindings),
    completeness: bindings.some(row => row.member_ref === null) ? "partial_member_binding" : "available_members_bound",
    interpretation_boundary: "Member hashes identify files; storage ranges identify tensor payloads only in their declared offset basis. Shared or overlapping storage does not by itself establish a defect or runtime alias." };
}

export function validateArtifactMembers(ledger, identity, storage) {
  if (ledger?.schema !== "deepbom.artifact_members.v1" || !Array.isArray(ledger.members) || !Array.isArray(ledger.storage_bindings)) throw new Error("IR artifact member ledger is invalid.");
  const members = new Map(), objects = new Map(storage.objects.map(row => [row.id, row]));
  for (const member of ledger.members) {
    if (!member.id || members.has(member.id) || (member.sha256 !== null && !SHA.test(member.sha256)) || (member.verification === "identity_bound" && member.sha256 === null)) throw new Error("IR artifact member identity is invalid.");
    validateExactInteger(member.byte_length); members.set(member.id, member);
  }
  if (members.get("artifact:primary")?.sha256 !== identity.sha256 || canonicalJson(members.get("artifact:primary")?.byte_length) !== canonicalJson(identity.byte_length) || members.get("artifact:primary")?.path !== identity.filename) throw new Error("Artifact IR primary member binding is invalid.");
  const seen = new Set();
  for (const row of ledger.storage_bindings) {
    const object = objects.get(row.storage_ref), member = members.get(row.member_ref);
    if (!object || seen.has(row.storage_ref) || (row.member_ref !== null && !member) || canonicalJson(row.range) !== canonicalJson(object.byte_range)) throw new Error("IR storage member reference or range is invalid.");
    seen.add(row.storage_ref);
    if (row.range?.status === "exact") {
      const start = exactInteger(row.range.start), end = exactInteger(row.range.end_exclusive);
      if (!start || !end || BigInt(end.decimal) - BigInt(start.decimal) !== BigInt(object.serialized_byte_length.decimal)) throw new Error("IR storage range length contradicts serialized bytes.");
    }
    if (row.bounds_assessment !== rangeBounds(row, member)) throw new Error("IR storage member bounds assessment is inconsistent.");
  }
  if (seen.size !== objects.size) throw new Error("IR artifact member ledger omitted storage objects.");
  if (canonicalJson(ledger.overlap_groups) !== canonicalJson(overlapGroups(ledger.storage_bindings))) throw new Error("IR storage overlap accounting is inconsistent.");
}

function rangeBounds(row, member) {
  if (!member || row.range?.status !== "exact" || row.range.offset_basis !== "artifact_absolute" || member.kind === "bundle" || !member.byte_length) return "not_assessable";
  return BigInt(row.range.end_exclusive) > BigInt(member.byte_length.decimal) ? "outside_artifact_member" : "within_artifact_member";
}

function overlapGroups(bindings) {
  const byFile = new Map();
  for (const row of bindings) {
    if (!row.member_ref || row.range?.status !== "exact") continue;
    const key = JSON.stringify([row.member_ref, row.range.offset_basis]);
    if (!byFile.has(key)) byFile.set(key, []);
    byFile.get(key).push(row);
  }
  const result = [];
  for (const [key, rows] of [...byFile].sort(([a], [b]) => compareCanonicalText(a, b))) {
    rows.sort((a, b) => BigInt(a.range.start) < BigInt(b.range.start) ? -1 : BigInt(a.range.start) > BigInt(b.range.start) ? 1 : compareCanonicalText(a.storage_ref, b.storage_ref));
    let group = [], end = -1n;
    const flush = () => { if (group.length > 1) { const [member_ref, offset_basis] = JSON.parse(key); result.push({ member_ref, offset_basis, storage_refs: group.map(row => row.storage_ref), classification: "overlapping_serialized_ranges_alias_semantics_not_established" }); } };
    for (const row of rows) {
      const start = BigInt(row.range.start), stop = BigInt(row.range.end_exclusive);
      if (start >= end) { flush(); group = []; end = stop; }
      group.push(row); if (stop > end) end = stop;
    }
    flush();
  }
  return result;
}
