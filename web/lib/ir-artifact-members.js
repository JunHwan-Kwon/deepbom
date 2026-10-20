import { portableContract } from "./portable-contracts.js";
import { canonicalJsonInput } from "./report-utils.js";
const irContract = (operation, input) => portableContract("ir_semantics", canonicalJsonInput({ operation, ...input }));
import { compareCanonicalText } from "./report-utils.js";
import { exactInteger } from "./exact-integer.js";
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
  irContract("members", {ledger,identity,storage});
}
function rangeBounds(row, member) { return irContract("member_bounds", {row,member:member ?? null}); }
function overlapGroups(bindings) { return irContract("member_overlaps", {bindings}); }
