const SHA256 = /^[a-f0-9]{64}$/;

export function normalizeArtifactIrRuntimeOverlay(runtimeEvidence, graph, architecture, artifactSha256) {
  if (!runtimeEvidence || typeof runtimeEvidence !== "object") return [];
  const valid = new Set([...graph.operators.map((row) => row.id), ...architecture.nodes.map((row) => row.id)]);
  const primaryOperatorByIndex = new Map(graph.operators
    .filter((row) => row.scope_ref === graph.primary_scope_ref)
    .map((row) => [row.native_index, row.id]));
  const declaredRuntimeNodes = list(runtimeEvidence.runtime_nodes);
  const candidateRows = declaredRuntimeNodes.length ? declaredRuntimeNodes : list(runtimeEvidence.rows || runtimeEvidence.assignments);
  const inputRows = declaredRuntimeNodes.length ? candidateRows : candidateRows.filter((row) => {
    const hasRuntimeIdentity = runtimeNodeIdentity(row) != null;
    const hasCanonicalBinding = runtimeSourceBinding(row, primaryOperatorByIndex).subjectRefs.length > 0;
    return hasRuntimeIdentity && hasCanonicalBinding;
  });
  if (!inputRows.length) return [];
  const runtimeArtifactSha256 = normalizeSha256(runtimeEvidence.artifact_sha256 || runtimeEvidence.artifact?.sha256);
  if (!runtimeArtifactSha256 || runtimeArtifactSha256 !== artifactSha256) throw new Error("Runtime overlay is not bound to the active artifact SHA-256.");
  const groupedRuntimeNodes = new Map();
  inputRows.forEach((row, index) => {
    const runtimeNodeRef = runtimeNodeIdentity(row);
    if (!runtimeNodeRef) throw new Error(`Runtime overlay row ${index} does not declare a runtime node identity.`);
    const binding = runtimeSourceBinding(row, primaryOperatorByIndex);
    if (binding.subjectRefs.some((subjectRef) => !valid.has(subjectRef))) throw new Error("Runtime overlay references an unknown canonical subject.");
    const backend = optionalText(row.backend || row.provider || row.delegate);
    const evidenceClass = String(row.evidence_class || "MEASURED_IMPORTED");
    const existing = groupedRuntimeNodes.get(runtimeNodeRef);
    if (existing && existing.backend !== backend) throw new Error(`Runtime overlay node ${runtimeNodeRef} declares conflicting backends.`);
    if (existing && existing.evidence_class !== evidenceClass) throw new Error(`Runtime overlay node ${runtimeNodeRef} declares conflicting evidence classes.`);
    const current = existing || {
      runtime_node_ref: runtimeNodeRef,
      backend,
      source_subject_refs: new Set(),
      mapping_bases: new Set(),
      runtime_node_kind: optionalText(row.runtime_node_kind || row.kind),
      evidence_class: evidenceClass,
    };
    binding.subjectRefs.forEach((subjectRef) => current.source_subject_refs.add(subjectRef));
    if (binding.mappingBasis) current.mapping_bases.add(binding.mappingBasis);
    groupedRuntimeNodes.set(runtimeNodeRef, current);
  });
  const runtimeNodes = [...groupedRuntimeNodes.values()].map((row) => {
    const sourceSubjectRefs = [...row.source_subject_refs].sort();
    const mappingBasis = sourceSubjectRefs.length === 0
      ? "explicit_unmapped_runtime_node"
      : row.mapping_bases.size === 1 ? [...row.mapping_bases][0] : "explicit_subject_ref_and_native_index_import";
    return {
      runtime_node_ref: row.runtime_node_ref,
      backend: row.backend,
      source_subject_refs: sourceSubjectRefs,
      mapping_cardinality: sourceSubjectRefs.length > 1 ? "fused_one_to_many" : sourceSubjectRefs.length === 1 ? "one_to_one" : "unmapped",
      mapping_basis: mappingBasis,
      runtime_node_kind: row.runtime_node_kind,
      evidence_class: row.evidence_class,
    };
  }).sort((left, right) => left.runtime_node_ref.localeCompare(right.runtime_node_ref));
  if (!runtimeNodes.length) return [];
  const rows = runtimeNodes.flatMap((node) => node.source_subject_refs.map((subjectRef) => ({
    subject_ref: subjectRef,
    runtime_node_ref: node.runtime_node_ref,
    backend: node.backend,
    evidence_class: node.evidence_class,
    mapping_cardinality: node.mapping_cardinality,
  })));
  const mappedSubjects = new Set(rows.map((row) => row.subject_ref));
  return [{
    id: "overlay:runtime:0",
    kind: "runtime_assignment",
    evidence_class: "IMPORTED_IDENTITY_BOUND_RUNTIME_EVIDENCE",
    summary: {
      runtime_node_count: runtimeNodes.length,
      mapped_runtime_node_count: runtimeNodes.filter((row) => row.source_subject_refs.length).length,
      unmapped_runtime_node_count: runtimeNodes.filter((row) => !row.source_subject_refs.length).length,
      one_to_one_runtime_node_count: runtimeNodes.filter((row) => row.mapping_cardinality === "one_to_one").length,
      fused_runtime_node_count: runtimeNodes.filter((row) => row.mapping_cardinality === "fused_one_to_many").length,
      source_subject_reference_count: rows.length,
      distinct_source_subject_count: mappedSubjects.size,
      canonical_subject_count: valid.size,
      canonical_subject_coverage_status: mappedSubjects.size === valid.size ? "complete_explicit_mapping" : mappedSubjects.size ? "partial_explicit_mapping" : "not_mapped",
      name_similarity_mapping_used: false,
    },
    runtime_nodes: runtimeNodes,
    rows,
    interpretation_boundary: "Imported runtime nodes preserve artifact-bound explicit source_subject_refs or primary-scope native op indices, including one-to-one and fused one-to-many mappings. Generated or unresolved runtime nodes remain explicitly unmapped; DEEPBOM never reconciles them by name similarity.",
  }];
}

export function validateArtifactIrRuntimeReconciliation(overlay, validSubjects) {
  const runtimeIds = new Set();
  let flattenedReferenceCount = 0;
  for (const node of list(overlay.runtime_nodes)) {
    if (!boundedText(node.runtime_node_ref, 1000) || runtimeIds.has(node.runtime_node_ref)) throw new Error("Artifact IR runtime node identity is invalid or duplicated.");
    runtimeIds.add(node.runtime_node_ref);
    const references = compactStrings(node.source_subject_refs);
    if (references.some((subjectRef) => !validSubjects.has(subjectRef))) throw new Error("Artifact IR runtime node references an unknown canonical subject.");
    if ((references.length === 0 && node.mapping_cardinality !== "unmapped")
      || (references.length === 1 && node.mapping_cardinality !== "one_to_one")
      || (references.length > 1 && node.mapping_cardinality !== "fused_one_to_many")) throw new Error("Artifact IR runtime mapping cardinality is inconsistent.");
    flattenedReferenceCount += references.length;
  }
  if (overlay.summary?.runtime_node_count !== runtimeIds.size
    || overlay.summary?.source_subject_reference_count !== flattenedReferenceCount
    || overlay.rows.length !== flattenedReferenceCount) throw new Error("Artifact IR runtime reconciliation count conservation failed.");
  const distinctSubjects = new Set(overlay.rows.map((row) => row.subject_ref));
  if (overlay.summary?.distinct_source_subject_count !== distinctSubjects.size
    || overlay.summary?.mapped_runtime_node_count !== list(overlay.runtime_nodes).filter((row) => list(row.source_subject_refs).length > 0).length
    || overlay.summary?.unmapped_runtime_node_count !== list(overlay.runtime_nodes).filter((row) => list(row.source_subject_refs).length === 0).length
    || overlay.summary?.name_similarity_mapping_used !== false) throw new Error("Artifact IR runtime reconciliation summary is inconsistent.");
  for (const row of overlay.rows) if (!runtimeIds.has(row.runtime_node_ref)) throw new Error("Artifact IR runtime assignment references an unknown runtime node.");
}

function runtimeNodeIdentity(row) {
  const explicit = optionalText(row.runtime_node_ref || row.node_ref || row.runtime_node_name || row.node_name || row.id);
  if (explicit) return explicit;
  const nativeIndex = nonNegativeInteger(row.runtime_node_index);
  return nativeIndex == null ? null : `runtime-node:${optionalText(row.backend || row.provider || row.delegate) || "unknown"}:${nativeIndex}`;
}

function runtimeSourceBinding(row, primaryOperatorByIndex) {
  const explicit = compactStrings(row.source_subject_refs?.length ? row.source_subject_refs : row.subject_ref);
  const rawIndices = row.source_op_indices?.length ? row.source_op_indices : row.op_indices?.length ? row.op_indices : row.op_index;
  const indices = list(Array.isArray(rawIndices) ? rawIndices : rawIndices == null ? [] : [rawIndices]).map(Number).filter(Number.isSafeInteger)
    .filter((value, position, all) => value >= 0 && all.indexOf(value) === position);
  const indexed = indices.map((index) => {
    const subjectRef = primaryOperatorByIndex.get(index);
    if (!subjectRef) throw new Error(`Runtime overlay references unknown primary-scope operator index ${index}.`);
    return subjectRef;
  });
  return {
    subjectRefs: [...new Set([...explicit, ...indexed])],
    mappingBasis: explicit.length && indexed.length ? "explicit_subject_ref_and_native_index_import"
      : explicit.length ? "explicit_subject_ref_import" : indexed.length ? "explicit_primary_scope_native_index_import" : null,
  };
}

function normalizeSha256(value) { const normalized = String(value || "").trim().toLowerCase(); return SHA256.test(normalized) ? normalized : null; }
function optionalText(value) { const normalized = String(value ?? "").trim(); return normalized || null; }
function nonNegativeInteger(value) { const number = Number(value); return Number.isSafeInteger(number) && number >= 0 ? number : null; }
function compactStrings(value) { return list(Array.isArray(value) ? value : [value]).map((item) => String(item || "").trim()).filter(Boolean); }
function list(value) { return Array.isArray(value) ? value : []; }
function boundedText(value, maximum) { const normalized = String(value ?? "").trim(); return normalized.length > 0 && normalized.length <= maximum; }
