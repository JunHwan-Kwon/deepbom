import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import {
  DEEPBOM_RECONCILABLE_COMPONENT_PROPERTIES,
  componentPropertyReconciliationKind,
} from "./deepbom-property-taxonomy.js";

export const BOM_ARTIFACT_RECONCILIATION_SCHEMA = "deepbom.bom_artifact_reconciliation.v1";
const SHA256 = /^[a-f0-9]{64}$/i;

export function reconcileCycloneDx17Artifact({ bom, expectedBom, additionalExpectedBoms = [], artifact, componentRef = "", bomSource = null } = {}) {
  const validation = validateRoot(bom);
  let expectedComponent = expectedBom?.metadata?.component || null;
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
  // Both the CLI and browser export documented profiles. Select only among
  // locally regenerated profiles; no observations are taken from the supplied BOM.
  const profileNames = new Set((supplied.properties || []).filter((item) => item?.name === "deepbom:contract:schema").map((item) => item.value));
  if (profileNames.size === 1) {
    expectedComponent = [expectedBom, ...additionalExpectedBoms].map((document) => document?.metadata?.component)
      .find((component) => component?.properties?.some((item) => item.name === "deepbom:contract:schema" && profileNames.has(item.value))) || expectedComponent;
  }
  const comparisons = [];
  compareScalar(comparisons, "component.type", "machine-learning-model", supplied.type, "standard_field");
  compareSha256(comparisons, expectedComponent, supplied);
  compareModelParameters(comparisons, expectedComponent, supplied);
  const propertyCoverage = compareVendorProperties(comparisons, expectedComponent, supplied, selected.pointer,
    [expectedBom, ...additionalExpectedBoms].flatMap((document) => document?.metadata?.component?.properties || []));
  for (const field of ["authors", "licenses", "description", "copyright", "supplier", "manufacturer"]) {
    if (Object.hasOwn(supplied, field)) {
      comparisons.push(row(`component.${field}`, "not_assessable_from_artifact", null, supplied[field], "declared_metadata"));
    }
  }
  return finalize({ artifact, bomSource, subject: { pointer: selected.pointer, bom_ref: supplied["bom-ref"] || null }, binding: selected.binding, comparisons, propertyCoverage });
}

function validateRoot(bom) {
  const errors = [];
  if (!bom || typeof bom !== "object" || Array.isArray(bom)) errors.push("BOM root must be an object.");
  if (bom?.bomFormat !== "CycloneDX") errors.push("bomFormat must be CycloneDX.");
  if (bom?.specVersion !== "1.7") errors.push("Only CycloneDX 1.7 is supported by this reconciliation contract.");
  const visit = (component, pointer) => {
    if (!component || typeof component !== "object" || Array.isArray(component)) {
      errors.push(`${pointer} must be a component object.`);
      return;
    }
    for (const key of ["properties", "hashes", "components"]) {
      if (component[key] !== undefined && !Array.isArray(component[key])) errors.push(`${pointer}.${key} must be an array.`);
    }
    if (Array.isArray(component.components)) component.components.forEach((child, index) => visit(child, `${pointer}.components[${index}]`));
  };
  if (bom?.metadata?.component !== undefined) visit(bom.metadata.component, "metadata.component");
  if (bom?.components !== undefined && !Array.isArray(bom.components)) errors.push("components must be an array.");
  if (Array.isArray(bom?.components)) bom.components.forEach((component, index) => visit(component, `components[${index}]`));
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
  const declarations = inventory.flatMap((entry) => rawSha256Hashes(entry.component)
    .map((hash) => ({ pointer: entry.pointer, sha256: hash })));
  const reason = valid ? "unique_sha256_match" : !SHA256.test(digest) ? "invalid_artifact_sha256"
    : matches.length > 1 ? "multiple_sha256_matches"
      : !declarations.length ? "no_sha256_declaration"
        : declarations.some((entry) => !SHA256.test(entry.sha256)) ? "invalid_sha256_declaration" : "sha256_mismatch";
  return {
    component: valid ? matches[0].component : null,
    pointer: valid ? matches[0].pointer : null,
    binding: {
      status: valid ? "selected_by_sha256" : "ambiguous_subject",
      method: "artifact_sha256",
      candidates: matches.map((entry) => entry.pointer),
      reason,
      declared_sha256: declarations,
      diagnostics: valid ? [] : [`Expected one component with artifact SHA-256 ${digest || "missing"}; observed ${matches.length}. Use --component-ref only when deliberately comparing a differently bound component.`],
    },
  };
}

function compareSha256(rows, expected, supplied) {
  const expectedHash = componentHashes(expected)[0] || null;
  const suppliedHashes = rawSha256Hashes(supplied);
  if (!suppliedHashes.length) rows.push(row("component.hashes[SHA-256]", "absent_in_bom", expectedHash, null, "standard_field"));
  else rows.push(row("component.hashes[SHA-256]", suppliedHashes.every((hash) => hash === expectedHash) ? "match" : "mismatch", expectedHash, suppliedHashes, "standard_field_all_sha256_declarations"));
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

function compareVendorProperties(rows, expected, supplied, pointer, additionalObservations) {
  // CycloneDX explicitly permits repeated property names. Never collapse them:
  // every occurrence (including an unknown namespace) needs its own outcome.
  const wanted = new Map();
  for (const item of expected?.properties || []) {
    if (wanted.has(item.name) && wanted.get(item.name) !== item.value) throw new Error(`Conflicting regenerated property: ${item.name}.`);
    wanted.set(item.name, item.value);
  }
  const requiredNames = new Set(wanted.keys());
  for (const item of additionalObservations) if (!wanted.has(item.name)) wanted.set(item.name, item.value);
  const properties = supplied.properties || [];
  const observed = new Set();
  const propertyRows = [];
  for (const [index, item] of properties.entries()) {
    const name = typeof item?.name === "string" ? item.name : "";
    const kind = componentPropertyReconciliationKind(name);
    let status;
    let basis = kind;
    let observation = null;
    if (!name || !item || typeof item !== "object" || Array.isArray(item)
      || (item.value !== undefined && typeof item.value !== "string")) {
      status = "ambiguous_subject";
      basis = "invalid_property_declaration";
    } else if (kind === "unsupported_mapping") {
      status = "unsupported_mapping";
    } else if (kind === "context_or_publisher_declaration" || !wanted.has(name) || item.value === undefined) {
      status = "not_assessable_from_artifact";
      if (kind === "artifact_observation") basis = item.value === undefined ? "property_value_not_declared" : "artifact_observation_unavailable";
    } else {
      // Compare exact emitted strings. Do not coerce large counts to Number or
      // round ratios: a changed declaration must not become equal by rounding.
      observation = wanted.get(name);
      status = observation === item.value ? "match" : "mismatch";
    }
    observed.add(name);
    propertyRows.push({
      ...row(`component.properties[${name || index}]`, status, observation, item?.value ?? null, basis),
      property_name: name || null,
      property_index: index,
      bom_pointer: `${pointer}.properties[${index}]`,
    });
  }
  rows.push(...propertyRows);
  for (const name of DEEPBOM_RECONCILABLE_COMPONENT_PROPERTIES) {
    if (!requiredNames.has(name)) continue;
    if (!observed.has(name)) rows.push(row(`component.properties[${name}]`, "absent_in_bom", wanted.get(name), null, "artifact_observation"));
  }
  const counts = countRows(propertyRows);
  return {
    scope: `${pointer}.properties`,
    supplied: properties.length,
    accounted: propertyRows.length,
    compared: counts.match + counts.mismatch,
    counts,
    duplicate_occurrences: properties.length - observed.size,
    by_namespace: Object.fromEntries([...new Set(propertyRows.map((entry) => entry.property_name?.split(":")[0] || "(invalid)"))]
      .sort().map((namespace) => [namespace, countRows(propertyRows.filter((entry) => (entry.property_name?.split(":")[0] || "(invalid)") === namespace))])),
  };
}

function compareScalar(rows, field, expected, observed, basis) {
  if (observed === undefined) rows.push(row(field, "absent_in_bom", expected, null, basis));
  else rows.push(row(field, canonicalJson(expected) === canonicalJson(observed) ? "match" : "mismatch", expected, observed, basis));
}

function componentHashes(component) {
  return rawSha256Hashes(component).filter((value) => SHA256.test(value));
}

function rawSha256Hashes(component) {
  return (Array.isArray(component?.hashes) ? component.hashes : [])
    .filter((entry) => String(entry?.alg || "").toUpperCase().replaceAll("-", "") === "SHA256")
    .map((entry) => String(entry?.content || "").toLowerCase());
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

function countRows(rows) {
  return Object.fromEntries(["match", "mismatch", "absent_in_bom", "not_assessable_from_artifact", "unsupported_mapping", "ambiguous_subject"]
    .map((status) => [status, rows.filter((entry) => entry.status === status).length]));
}

function finalize({ artifact, bomSource, subject, binding, comparisons, propertyCoverage = null }) {
  const counts = countRows(comparisons);
  const status = binding.status === "ambiguous_subject" || counts.ambiguous_subject
    ? "ambiguous_subject"
    : counts.mismatch ? "artifact_contradiction_observed"
      : counts.absent_in_bom ? "reconciled_with_absent_fields" : "reconciled_no_contradiction";
  const gateResult = status === "ambiguous_subject" ? "pending" : counts.mismatch ? "block" : "pass";
  const body = {
    schema: BOM_ARTIFACT_RECONCILIATION_SCHEMA,
    status,
    gate_result: gateResult,
    result_label: gateResult === "pass" ? "NO_CONTRADICTION" : gateResult.toUpperCase(),
    artifact,
    bom_source: bomSource,
    subject,
    binding,
    counts,
    property_coverage: propertyCoverage,
    coverage_complete: Boolean(propertyCoverage) && !counts.absent_in_bom && !counts.not_assessable_from_artifact
      && !counts.unsupported_mapping && !counts.ambiguous_subject,
    comparisons,
    interpretation_boundary: "NO_CONTRADICTION means no mismatch among the compared fields, not verification of every BOM claim. Property coverage counts each occurrence in the selected component.properties only; other components, document properties, evidence attachments, and declarations are outside this comparison. Missing declarations, unsupported mappings, and publisher assertions are reported separately. This is not a complete BOM validation, safety or quality claim, regulatory determination, or standards conformance certification.",
  };
  return { ...body, reconciliation_sha256: sha256TextHex(canonicalJson(body)) };
}
