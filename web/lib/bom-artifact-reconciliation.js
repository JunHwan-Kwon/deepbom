import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import {
  DEEPBOM_RECONCILABLE_COMPONENT_PROPERTIES,
  deepBomPropertyMap,
} from "./deepbom-property-taxonomy.js";

export const BOM_ARTIFACT_RECONCILIATION_SCHEMA = "deepbom.bom_artifact_reconciliation.v1";
const SHA256 = /^[a-f0-9]{64}$/i;

export function reconcileCycloneDx17Artifact({ bom, expectedBom, artifact, componentRef = "", bomSource = null } = {}) {
  const validation = validateRoot(bom);
  const expectedComponent = expectedBom?.metadata?.component || null;
  if (!validation.valid || !expectedComponent) {
    return finalize({
      artifact,
      bomSource,
      subject: null,
      binding: { status: "ambiguous_subject", method: "none", candidates: [], diagnostics: validation.errors },
      comparisons: validation.errors.map((message) => row("$", "ambiguous_subject", "CycloneDX 1.7 document", message, "document_validation")),
    });
  }

  const inventory = components(bom);
  const duplicateRefs = duplicates(inventory.map((entry) => entry.component?.["bom-ref"]).filter(Boolean));
  if (duplicateRefs.length) {
    const diagnostic = `Duplicate bom-ref values: ${duplicateRefs.join(", ")}.`;
    return finalize({
      artifact,
      bomSource,
      subject: null,
      binding: { status: "ambiguous_subject", method: "none", candidates: [], diagnostics: [diagnostic] },
      comparisons: [row("component.bom-ref", "ambiguous_subject", "unique bom-ref values", duplicateRefs, "subject_binding")],
    });
  }
  const selected = selectSubject(inventory, artifact?.sha256, componentRef, duplicateRefs);
  if (!selected.component) {
    return finalize({
      artifact,
      bomSource,
      subject: null,
      binding: selected.binding,
      comparisons: [row("component", "ambiguous_subject", componentRef || artifact?.sha256 || null, null, "subject_binding")],
    });
  }

  const supplied = selected.component;
  const comparisons = [];
  compareScalar(comparisons, "component.type", "machine-learning-model", supplied.type, "standard_field");
  compareSha256(comparisons, expectedComponent, supplied);
  compareModelParameters(comparisons, expectedComponent, supplied);
  compareVendorProperties(comparisons, expectedComponent, supplied);
  for (const field of ["authors", "licenses", "description", "copyright", "supplier", "manufacturer"]) {
    if (Object.hasOwn(supplied, field)) {
      comparisons.push(row(`component.${field}`, "not_assessable_from_artifact", null, supplied[field], "declared_metadata"));
    }
  }
  return finalize({ artifact, bomSource, subject: { pointer: selected.pointer, bom_ref: supplied["bom-ref"] || null }, binding: selected.binding, comparisons });
}

function validateRoot(bom) {
  const errors = [];
  if (!bom || typeof bom !== "object" || Array.isArray(bom)) errors.push("BOM root must be an object.");
  if (bom?.bomFormat !== "CycloneDX") errors.push("bomFormat must be CycloneDX.");
  if (bom?.specVersion !== "1.7") errors.push("Only CycloneDX 1.7 is supported by this reconciliation contract.");
  return { valid: errors.length === 0, errors };
}

function components(bom) {
  const result = [];
  const visit = (component, pointer) => {
    if (!component || typeof component !== "object" || Array.isArray(component)) return;
    result.push({ component, pointer });
    for (const [index, child] of (component.components || []).entries()) visit(child, `${pointer}.components[${index}]`);
  };
  visit(bom?.metadata?.component, "metadata.component");
  for (const [index, component] of (bom?.components || []).entries()) visit(component, `components[${index}]`);
  return result;
}

function selectSubject(inventory, artifactSha256, componentRef, duplicateRefs) {
  if (componentRef) {
    const matches = inventory.filter((entry) => entry.component?.["bom-ref"] === componentRef);
    const valid = matches.length === 1 && !duplicateRefs.includes(componentRef);
    return {
      component: valid ? matches[0].component : null,
      pointer: valid ? matches[0].pointer : null,
      binding: {
        status: valid ? "selected_by_component_ref" : "ambiguous_subject",
        method: "explicit_bom_ref",
        candidates: matches.map((entry) => entry.pointer),
        diagnostics: valid ? [] : [`Expected one unique component with bom-ref ${componentRef}; observed ${matches.length}.`],
      },
    };
  }
  const digest = String(artifactSha256 || "").toLowerCase();
  const matches = inventory.filter((entry) => componentHashes(entry.component).includes(digest));
  const valid = SHA256.test(digest) && matches.length === 1;
  return {
    component: valid ? matches[0].component : null,
    pointer: valid ? matches[0].pointer : null,
    binding: {
      status: valid ? "selected_by_sha256" : "ambiguous_subject",
      method: "artifact_sha256",
      candidates: matches.map((entry) => entry.pointer),
      diagnostics: valid ? [] : [`Expected one component with artifact SHA-256 ${digest || "missing"}; observed ${matches.length}. Use --component-ref only when deliberately comparing a differently bound component.`],
    },
  };
}

function compareSha256(rows, expected, supplied) {
  const expectedHash = componentHashes(expected)[0] || null;
  const suppliedHashes = componentHashes(supplied);
  if (!suppliedHashes.length) rows.push(row("component.hashes[SHA-256]", "absent_in_bom", expectedHash, null, "standard_field"));
  else rows.push(row("component.hashes[SHA-256]", suppliedHashes.includes(expectedHash) ? "match" : "mismatch", expectedHash, suppliedHashes, "standard_field"));
}

function compareModelParameters(rows, expected, supplied) {
  for (const direction of ["inputs", "outputs"]) {
    const wanted = expected?.modelCard?.modelParameters?.[direction] || [];
    const observed = supplied?.modelCard?.modelParameters?.[direction];
    const pointer = `component.modelCard.modelParameters.${direction}`;
    if (observed === undefined) rows.push(row(pointer, "absent_in_bom", wanted, null, "standard_field"));
    else rows.push(row(pointer, canonicalJson(wanted) === canonicalJson(observed) ? "match" : "mismatch", wanted, observed, "standard_field"));
  }
}

function compareVendorProperties(rows, expected, supplied) {
  const wanted = deepBomPropertyMap(expected?.properties);
  const observed = deepBomPropertyMap(supplied?.properties);
  for (const name of DEEPBOM_RECONCILABLE_COMPONENT_PROPERTIES) {
    if (!wanted.has(name)) continue;
    if (!observed.has(name)) rows.push(row(`component.properties[${name}]`, "absent_in_bom", wanted.get(name), null, "documented_deepbom_property"));
    else rows.push(row(`component.properties[${name}]`, wanted.get(name) === observed.get(name) ? "match" : "mismatch", wanted.get(name), observed.get(name), "documented_deepbom_property"));
  }
  for (const name of observed.keys()) {
    if (name.startsWith("deepbom:") && !DEEPBOM_RECONCILABLE_COMPONENT_PROPERTIES.includes(name)) {
      rows.push(row(`component.properties[${name}]`, "unsupported_mapping", null, observed.get(name), "vendor_property_outside_reconciliation_allowlist"));
    }
  }
}

function compareScalar(rows, field, expected, observed, basis) {
  if (observed === undefined) rows.push(row(field, "absent_in_bom", expected, null, basis));
  else rows.push(row(field, canonicalJson(expected) === canonicalJson(observed) ? "match" : "mismatch", expected, observed, basis));
}

function componentHashes(component) {
  return (component?.hashes || [])
    .filter((entry) => String(entry?.alg || "").toUpperCase().replaceAll("-", "") === "SHA256")
    .map((entry) => String(entry?.content || "").toLowerCase())
    .filter((value) => SHA256.test(value));
}

function duplicates(values) {
  const seen = new Set();
  const duplicate = new Set();
  for (const value of values) seen.has(value) ? duplicate.add(value) : seen.add(value);
  return [...duplicate].sort();
}

function row(field, status, artifactValue, bomValue, basis) {
  return { field, status, artifact_observation: artifactValue, bom_declaration: bomValue, comparison_basis: basis };
}

function finalize({ artifact, bomSource, subject, binding, comparisons }) {
  const counts = Object.fromEntries(["match", "mismatch", "absent_in_bom", "not_assessable_from_artifact", "unsupported_mapping", "ambiguous_subject"]
    .map((status) => [status, comparisons.filter((entry) => entry.status === status).length]));
  const status = binding.status === "ambiguous_subject" || counts.ambiguous_subject
    ? "ambiguous_subject"
    : counts.mismatch ? "artifact_contradiction_observed"
      : counts.absent_in_bom ? "reconciled_with_absent_fields" : "reconciled_no_contradiction";
  const gateResult = status === "ambiguous_subject" ? "pending" : counts.mismatch ? "block" : "pass";
  const body = {
    schema: BOM_ARTIFACT_RECONCILIATION_SCHEMA,
    status,
    gate_result: gateResult,
    artifact,
    bom_source: bomSource,
    subject,
    binding,
    counts,
    comparisons,
    interpretation_boundary: "The selected CycloneDX 1.7 component is reconciled only against facts observable from the supplied serialized artifact. Missing declarations, unsupported mappings, and publisher assertions are reported separately. This is not a complete BOM validation, safety or quality claim, regulatory determination, or standards conformance certification.",
  };
  return { ...body, reconciliation_sha256: sha256TextHex(canonicalJson(body)) };
}
