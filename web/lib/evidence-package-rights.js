export const EVIDENCE_PACKAGE_RIGHTS_SCHEMA = "deepbom.evidence_package_rights.v1";

export const DEEPBOM_CITATION_DOI = "https://doi.org/10.5281/zenodo.21834508";

const RIGHTS_HOLDER = Object.freeze({
  name: "Jun-Hwan Kwon",
  orcid: "https://orcid.org/0000-0002-6464-3895",
});

export function buildEvidencePackageRights({ profile = "public" } = {}) {
  return {
    schema: EVIDENCE_PACKAGE_RIGHTS_SCHEMA,
    profile: String(profile || "public"),
    copyright_notice: "Copyright (C) 2026 Jun-Hwan Kwon. All rights reserved.",
    rights_holder: RIGHTS_HOLDER,
    license_grant: "none",
    permitted_purpose: "review, integrity verification, and citation of this specific audit result",
    not_authorized_by_this_package: [
      "redistribution of the package or its protectable report expression",
      "modification presented as an original DEEPBOM result",
      "removal or concealment of attribution, artifact identity, hashes, signatures, or provenance",
      "source-code, executable, model-weight, implementation, or derivative-work reuse",
    ],
    integrity_boundary: "Hashes and detached signatures make covered member changes detectable. They do not prevent copying or editing and do not establish official DEEPBOM authorship without an independently trusted signing-key registry.",
    applicable_law_boundary: "This notice does not restrict rights independently available under applicable law and does not convert factual observations into licensed software or model weights.",
    citation: DEEPBOM_CITATION_DOI,
  };
}

export function validateEvidencePackageRights(rights, { profile = null } = {}) {
  return rights?.schema === EVIDENCE_PACKAGE_RIGHTS_SCHEMA
    && (!profile || rights.profile === profile)
    && rights.license_grant === "none"
    && rights.rights_holder?.name === RIGHTS_HOLDER.name
    && rights.rights_holder?.orcid === RIGHTS_HOLDER.orcid
    && rights.citation === DEEPBOM_CITATION_DOI
    && Array.isArray(rights.not_authorized_by_this_package)
    && rights.not_authorized_by_this_package.length === 4
    && /do not prevent copying or editing/i.test(String(rights.integrity_boundary || ""))
    && /applicable law/i.test(String(rights.applicable_law_boundary || ""));
}

export function evidencePackageRightsText(rights) {
  if (!validateEvidencePackageRights(rights, { profile: rights?.profile })) {
    throw new Error("Evidence Package rights record is invalid.");
  }
  return [
    "DEEPBOM EVIDENCE PACKAGE RIGHTS NOTICE",
    "",
    `Profile: ${rights.profile}`,
    rights.copyright_notice,
    `Rights holder: ${rights.rights_holder.name} (${rights.rights_holder.orcid})`,
    `License grant: ${rights.license_grant}`,
    "",
    "PERMITTED PURPOSE",
    rights.permitted_purpose,
    "",
    "NOT AUTHORIZED BY THIS PACKAGE",
    ...rights.not_authorized_by_this_package.map((item) => `- ${item}`),
    "",
    "INTEGRITY BOUNDARY",
    rights.integrity_boundary,
    "",
    "APPLICABLE-LAW BOUNDARY",
    rights.applicable_law_boundary,
    "",
    `Citation: ${rights.citation}`,
    "",
  ].join("\n");
}
