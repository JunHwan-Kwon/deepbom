import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { validateModelIr } from "./model-ir.js";

export const MODEL_IR_VISUALIZATION_SCHEMA = "deepbom.model_ir_visualization_manifest.v1";
export const MODEL_IR_VISUALIZATION_METHOD_VERSION = "1.0.0";
export const MODEL_IR_VIEW_IDS = Object.freeze([
  "identity-boundary",
  "architecture-overview",
  "block-detail",
  "exhaustive",
  "static-runtime",
  "observed-runtime",
]);

const PAGE = Object.freeze({
  portrait: { widthMm: 210, heightMm: 297, width: 1240, height: 1754, bodyRows: 22 },
  landscape: { widthMm: 297, heightMm: 210, width: 1754, height: 1240, bodyRows: 12 },
});
const MARGIN = 70;
const HEADER_HEIGHT = 132;
const FOOTER_HEIGHT = 82;

export function buildModelIrVisualizationBundle(input, {
  views = MODEL_IR_VIEW_IDS,
  orientation = "portrait",
} = {}) {
  const modelIr = validateModelIr(input);
  const selectedViews = normalizeViews(views);
  if (!PAGE[orientation]) throw new Error("Model IR visualization orientation must be portrait or landscape.");
  const pages = [];
  for (const view of selectedViews) pages.push(...renderViewPages(modelIr, view, orientation));
  const body = {
    schema: MODEL_IR_VISUALIZATION_SCHEMA,
    method_version: MODEL_IR_VISUALIZATION_METHOD_VERSION,
    model_ir_schema: modelIr.schema,
    model_ir_sha256: modelIr.model_ir_sha256,
    artifact: {
      filename: modelIr.artifact.filename,
      sha256: modelIr.artifact.sha256,
      format: modelIr.artifact.format,
    },
    selected_views: selectedViews,
    orientation,
    visual_contract: {
      page_size: "ISO_A4",
      canonical_vector_format: "SVG_1_1_subset",
      color_contract: "black_white_only",
      minimum_text_size_pt: 7,
      evidence_styles: {
        observed: "solid_single_line",
        derived_or_predicted: "dashed_single_line",
        declared: "double_line",
        not_assessable: "dotted_line",
      },
      source_reference_attribute: "data-subject-ref",
      exhaustive_view_omission_policy: "zero_omissions",
    },
    conservation: buildConservation(modelIr, pages),
    pages: pages.map(({ svg, caption, render_model, ...page }) => page),
    word_insertion: {
      body_view: "architecture-overview",
      appendix_views: ["block-detail", "exhaustive", "static-runtime", "observed-runtime"],
      ordered_files: pages.map((page) => page.filename),
      width_mm: orientation === "portrait" ? 160 : 247,
      resize_policy: "do_not_rescale_below_7pt",
      caption_sidecars: pages.map((page) => page.caption_filename),
      placement_boundary: "This manifest provides deterministic insertion order and identity binding; it does not create or approve a regulatory submission.",
    },
    interpretation_boundary: "These diagrams are deterministic projections of deepbom.model_ir.v1. Solid program edges exist only when serialized by the artifact source contract. Dashed inferred groupings do not establish execution order. Static projections are not observed runtime. This is engineering evidence, not a regulatory conclusion or standard-conformance claim.",
  };
  const manifest = Object.freeze({ ...body, visualization_manifest_sha256: sha256TextHex(canonicalJson(body)) });
  return Object.freeze({ manifest, pages: pages.map((page) => Object.freeze(page)) });
}

export function modelIrVisualizationFiles(input, options = {}) {
  const bundle = buildModelIrVisualizationBundle(input, options);
  const files = [];
  for (const page of bundle.pages) {
    files.push({ name: page.filename, data: page.svg, media_type: "image/svg+xml", sha256: page.sha256 });
    files.push({ name: page.caption_filename, data: `${page.caption}\n`, media_type: "text/plain", sha256: sha256TextHex(`${page.caption}\n`) });
  }
  files.push({
    name: "model-views/manifest.json",
    data: `${JSON.stringify(bundle.manifest, null, 2)}\n`,
    media_type: "application/json",
    sha256: sha256TextHex(`${JSON.stringify(bundle.manifest, null, 2)}\n`),
  });
  return { bundle, files };
}

function normalizeViews(views) {
  const values = Array.isArray(views) ? views : String(views || "").split(",");
  const selected = [...new Set(values.map((value) => String(value).trim().toLowerCase()).filter(Boolean))];
  if (!selected.length) throw new Error("At least one Model IR visualization view is required.");
  for (const view of selected) if (!MODEL_IR_VIEW_IDS.includes(view)) throw new Error(`Unsupported Model IR visualization view: ${view}.`);
  return selected;
}

function renderViewPages(modelIr, view, orientation) {
  const page = PAGE[orientation];
  const projection = projectView(modelIr, view);
  const chunks = chunkProjectionRows(projection.rows, page.bodyRows);
  const count = Math.max(1, chunks.length);
  const subjectToPage = new Map();
  chunks.forEach((rows, pageIndex) => rows.forEach((item) => subjectToPage.set(item.subject_ref, pageIndex)));
  const pages = [];
  for (let index = 0; index < count; index += 1) {
    const rows = chunks[index] || [];
    const rowIds = new Set(rows.map((row) => row.subject_ref));
    const edges = projection.edges.filter((edge) => rowIds.has(edge.to_ref));
    const svg = renderPageSvg(modelIr, projection, rows, { index, count, view, orientation, page, edges, subjectToPage });
    const filename = `model-views/${view}/page-${String(index + 1).padStart(3, "0")}.svg`;
    const captionFilename = filename.replace(/\.svg$/, ".caption.txt");
    pages.push({
      id: `${view}:page:${index + 1}`,
      view,
      page_number: index + 1,
      page_count: count,
      filename,
      caption_filename: captionFilename,
      svg,
      sha256: sha256TextHex(svg),
      caption: caption(modelIr, projection, index + 1, count),
      subject_refs: rows.flatMap((row) => row.subject_refs).filter(unique).sort(),
      rendered_member_count: rows.reduce((sum, row) => sum + row.conservation_count, 0),
      omitted_member_count: 0,
      relationship_refs: edges.map((edge) => edge.id),
      rendered_relationship_count: edges.length,
      cross_page_relationship_count: edges.filter((edge) => subjectToPage.get(edge.from_ref) !== index).length,
      continuation: {
        from_previous: index > 0,
        to_next: index + 1 < count,
      },
      render_model: {
        page: { width: page.width, height: page.height, width_mm: page.widthMm, height_mm: page.heightMm },
        projection: { level: projection.level, title: projection.title, boundary: projection.boundary },
        rows,
      },
    });
  }
  return pages;
}

function projectView(modelIr, view) {
  if (view === "identity-boundary") return identityProjection(modelIr);
  if (view === "architecture-overview") return architectureProjection(modelIr);
  if (view === "block-detail") return blockProjection(modelIr);
  if (view === "exhaustive") return exhaustiveProjection(modelIr);
  if (view === "static-runtime") return staticRuntimeProjection(modelIr);
  return observedRuntimeProjection(modelIr);
}

function identityProjection(modelIr) {
  const rows = [
    row("artifact:primary", "Artifact", modelIr.artifact.filename, `${modelIr.artifact.format} | sha256:${modelIr.artifact.sha256}`, "OBSERVED_SERIALIZED_ARTIFACT"),
    row("model-ir:identity", "Common IR", modelIr.schema, `sha256:${modelIr.model_ir_sha256}`, "DERIVED"),
    ...modelIr.profiles.map((profile) => row(`profile:${profile.id}`, profile.requirement, profile.id, `${profile.applicability} | ${profile.status}`, profile.applicability === "not_applicable" ? "NOT_ASSESSABLE" : "DERIVED")),
    ...modelIr.loss_ledger.entries.map((entry, index) => row(`loss:${index}:${entry.subject}`, "Loss / limitation", entry.subject, `${entry.status} | ${entry.explanation}`, entry.evidence_class || "NOT_ASSESSABLE")),
  ];
  return projection("V0", "Identity and evidence boundary", rows, "No model quality, safety, runtime, or regulatory conclusion is implied.");
}

function architectureProjection(modelIr) {
  const groups = modelIr.architecture.hierarchy || [];
  const rows = groups.map((group) => row(group.id, group.kind || "group", group.label || group.id,
    `${group.member_refs.length} members | rule ${group.grouping_rule_id}`, group.evidence_class || "DERIVED", group.member_refs));
  if (!rows.length) rows.push(row("architecture:unavailable", "Architecture", "No architecture groups materialized", modelIr.architecture.interpretation_boundary, "NOT_ASSESSABLE"));
  const ids = new Set(groups.map((row) => row.id));
  const edges = groups.filter((row) => row.parent_ref && ids.has(row.parent_ref)).map((row) => ({
    id: `visual:architecture:${row.parent_ref}:${row.id}`, kind: "group_membership", from_ref: row.parent_ref, to_ref: row.id, evidence_class: row.evidence_class || "DERIVED",
  }));
  return projection("V1", "Architecture overview", rows, modelIr.architecture.interpretation_boundary, { edges });
}

function blockProjection(modelIr) {
  const rows = [];
  const edges = [];
  const operationById = new Map(modelIr.program.operations.map((operation) => [operation.id, operation]));
  if (modelIr.program.blocks.length) {
    for (const block of modelIr.program.blocks) {
      rows.push(row(block.id, "Structural block", block.kind,
        `${block.member_refs.length} operations | signature ${block.signature_sha256.slice(0, 12)} | rule ${block.grouping_rule_id}`,
        block.evidence_class || "DERIVED", block.member_refs, { conservationCount: 0, pageGroup: block.id }));
      for (const operationRef of block.member_refs) {
        const operation = operationById.get(operationRef);
        if (!operation) continue;
        rows.push(row(operation.id, `Operation ${operation.display_order ?? operation.source_order ?? "?"}`,
          `${operation.native_op.domain}:${operation.native_op.name}`, formatBlockOperation(modelIr, operation),
          operation.evidence_class, [], { pageGroup: block.id }));
        edges.push({ id: `visual:block:${block.id}:${operation.id}`, kind: "group_membership", from_ref: block.id, to_ref: operation.id, evidence_class: "DERIVED" });
      }
    }
  } else {
    for (const profile of modelIr.architecture.model_profiles || []) {
      rows.push(row(profile.id, "Model profile", profile.id, `${profile.status} | ${profile.evidence_class}`, profile.evidence_class,
        [], { conservationCount: 0, pageGroup: profile.id }));
      for (const group of profile.groups || []) {
        rows.push(row(group.id, "Role group", group.role, `${group.member_refs.length} source members`, "DERIVED", group.member_refs,
          { pageGroup: profile.id }));
        edges.push({ id: `visual:profile:${profile.id}:${group.id}`, kind: "group_membership", from_ref: profile.id, to_ref: group.id, evidence_class: "DERIVED" });
      }
    }
  }
  if (!rows.length) rows.push(row("profile:none", "Model profile", "No model-specific profile recognized", "Core IR remains valid; no architecture is invented.", "NOT_ASSESSABLE"));
  return projection("V2", modelIr.program.blocks.length ? "Structural block detail" : "Storage-profile detail", rows,
    modelIr.program.blocks.length
      ? "Each operation remains an individually traceable member of one deterministic structural block. Block membership is derived and does not assert a framework layer, runtime fusion, or execution schedule."
      : "Profile groupings are reversible labels over source storage subjects; they do not create execution edges.",
    { edges });
}

function exhaustiveProjection(modelIr) {
  const graphRows = modelIr.program.operations.map((operation) => row(operation.id, `Operation ${operation.display_order ?? operation.source_order ?? "?"}`,
    `${operation.native_op.domain}:${operation.native_op.name}`, formatOperation(operation), operation.evidence_class));
  const storageRows = modelIr.tensors_and_storage.storage_objects.map((storage) => row(storage.id, "Storage", storage.name || storage.id,
    `${storage.dtype || "unknown"} | ${shapeText(storage.shape)} | ${encodingText(storage.encoding)} | ${byteText(storage.serialized_byte_length)}`, storage.evidence_class));
  const rows = graphRows.length ? graphRows : storageRows;
  const rowIds = new Set(rows.map((item) => item.subject_ref));
  const edges = graphRows.length ? modelIr.program.relationships
    .filter((edge) => ["data_dependency", "control_dependency", "state_dependency", "call"].includes(edge.kind) && rowIds.has(edge.from_ref) && rowIds.has(edge.to_ref))
    .map((edge) => ({ id: edge.id, kind: edge.kind, from_ref: edge.from_ref, to_ref: edge.to_ref, evidence_class: edge.evidence_class })) : [];
  if (!rows.length) rows.push(row("exhaustive:none", "Artifact structure", "No graph operation or storage object materialized", "See loss ledger and profile applicability.", "NOT_ASSESSABLE"));
  return projection("V3", graphRows.length ? "Exhaustive serialized program" : "Exhaustive serialized storage", rows,
    graphRows.length ? modelIr.program.interpretation_boundary : modelIr.tensors_and_storage.interpretation_boundary,
    { expectedMembers: graphRows.length ? modelIr.program.operations.length : modelIr.tensors_and_storage.storage_objects.length, edges });
}

function staticRuntimeProjection(modelIr) {
  const segments = (modelIr.static_runtime.projections || []).flatMap((projectionRow) => projectionRow.segments || []);
  const rows = segments.map((segment) => row(segment.id, "Static segment", segment.candidate_backend || segment.state || segment.id,
    `${segment.state || "unknown"} | ${segment.source_subject_refs?.length || 0} source subjects | actual runtime claim: false`, segment.evidence_class || "PREDICTED", segment.source_subject_refs));
  if (!rows.length) rows.push(row("static-runtime:none", "Static runtime", "No static runtime segment materialized", modelIr.static_runtime.interpretation_boundary, "NOT_ASSESSABLE"));
  return projection("V4", "Static runtime projection", rows, modelIr.static_runtime.interpretation_boundary);
}

function observedRuntimeProjection(modelIr) {
  const runtimeNodes = (modelIr.observed_runtime.overlays || []).flatMap((overlay) => overlay.runtime_nodes || []);
  const rows = runtimeNodes.map((runtimeNode) => row(runtimeNode.id, "Observed runtime", runtimeNode.observed_backend || runtimeNode.id,
    `${runtimeNode.observed_sequence ?? "sequence unavailable"} | ${runtimeNode.source_subject_refs?.length || 0} mapped source subjects`, "OBSERVED_RUNTIME", runtimeNode.source_subject_refs));
  if (!rows.length) rows.push(row("observed-runtime:none", "Observed runtime", "No identity-bound runtime evidence imported", modelIr.observed_runtime.interpretation_boundary, "NOT_ASSESSABLE"));
  return projection("V5", "Observed runtime overlay", rows, modelIr.observed_runtime.interpretation_boundary);
}

function projection(level, title, rows, boundary, { expectedMembers = rows.reduce((sum, item) => sum + item.conservation_count, 0), edges = [] } = {}) {
  return { level, title, rows, edges, boundary, expectedMembers };
}

function row(subjectRef, kind, title, detail, evidenceClass, relatedRefs = [], { conservationCount = 1, pageGroup = null } = {}) {
  const refs = [subjectRef, ...relatedRefs].filter(Boolean).filter(unique);
  return { subject_ref: subjectRef, subject_refs: refs, kind, title, detail, evidence_class: evidenceClass, conservation_count: conservationCount, page_group: pageGroup };
}

function renderPageSvg(modelIr, projection, rows, { index, count, view, orientation, page, edges, subjectToPage }) {
  const bodyHeight = page.height - HEADER_HEIGHT - FOOTER_HEIGHT;
  const rowHeight = Math.max(42, Math.floor(bodyHeight / Math.max(1, PAGE[orientation].bodyRows)));
  const availableWidth = page.width - MARGIN * 2;
  const recordOffset = projection.edges.length ? 54 : 0;
  const blocks = rows.map((item, rowIndex) => {
    const y = HEADER_HEIGHT + rowIndex * rowHeight;
    const evidenceClass = evidenceStyle(item.evidence_class);
    const labelWidth = Math.floor(availableWidth * 0.24);
    const titleWidth = Math.floor(availableWidth * 0.30);
    return `<g class="record ${evidenceClass}" data-subject-ref="${xml(item.subject_ref)}" transform="translate(${MARGIN + recordOffset} ${y})"><rect width="${availableWidth - recordOffset}" height="${rowHeight - 5}"/><text class="kind" x="12" y="18">${xml(clip(item.kind, 34))}</text><text class="title" x="${labelWidth}" y="18">${xml(clip(item.title, 48))}</text><text class="detail" x="${labelWidth + titleWidth}" y="18">${xml(clip(item.detail, orientation === "portrait" ? 64 : 106))}</text><text class="ref" x="12" y="${rowHeight - 13}">${xml(clip(item.subject_ref, orientation === "portrait" ? 108 : 170))}</text></g>`;
  }).join("");
  const rowIndexById = new Map(rows.map((row, rowIndex) => [row.subject_ref, rowIndex]));
  const connectorX = MARGIN + 26;
  const connectors = edges.map((edge, edgeIndex) => {
    const targetIndex = rowIndexById.get(edge.to_ref);
    const sourceIndex = rowIndexById.get(edge.from_ref);
    const targetY = HEADER_HEIGHT + targetIndex * rowHeight + Math.floor((rowHeight - 5) / 2);
    const edgeClass = edge.kind === "control_dependency" || String(edge.evidence_class).includes("DERIVED") ? "edge inferred" : "edge observed";
    if (sourceIndex != null) {
      const sourceY = HEADER_HEIGHT + sourceIndex * rowHeight + Math.floor((rowHeight - 5) / 2);
      const laneX = connectorX + (edgeIndex % 4) * 7;
      return `<g class="${edgeClass}" data-relationship-ref="${xml(edge.id)}"><path d="M ${MARGIN + recordOffset} ${sourceY} H ${laneX} V ${targetY} H ${MARGIN + recordOffset}" marker-end="url(#arrow)"/><title>${xml(`${edge.kind}: ${edge.from_ref} -> ${edge.to_ref}`)}</title></g>`;
    }
    const sourcePage = subjectToPage.get(edge.from_ref);
    return `<g class="${edgeClass} cross-page" data-relationship-ref="${xml(edge.id)}"><path d="M ${connectorX} ${targetY} H ${MARGIN + recordOffset}" marker-end="url(#arrow)"/><text x="${connectorX}" y="${targetY - 5}">from p${sourcePage == null ? "?" : sourcePage + 1}</text><title>${xml(`${edge.kind}: ${edge.from_ref} -> ${edge.to_ref}`)}</title></g>`;
  }).join("");
  const continuationTop = index > 0 ? `<text class="continuation" x="${page.width - MARGIN}" y="115" text-anchor="end">continued from page ${index}</text>` : "";
  const continuationBottom = index + 1 < count ? `<text class="continuation" x="${page.width - MARGIN}" y="${page.height - 52}" text-anchor="end">continues on page ${index + 2}</text>` : "";
  const evidenceBadge = evidenceBadgeFor(rows);
  return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${page.widthMm}mm" height="${page.heightMm}mm" viewBox="0 0 ${page.width} ${page.height}" role="img" aria-labelledby="title desc"><title id="title">${xml(projection.title)} - ${xml(modelIr.artifact.filename)}</title><desc id="desc">${xml(projection.boundary)}</desc><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#000"/></marker></defs><style>.page{fill:#fff}.frame,.record rect{fill:#fff;stroke:#000;stroke-width:2}.record.derived rect{stroke-dasharray:12 7}.record.declared rect{stroke-width:5}.record.not-assessable rect{stroke-dasharray:2 7}.edge path{fill:none;stroke:#000;stroke-width:2}.edge.inferred path{stroke-dasharray:8 6}.edge text{font:12px Arial,sans-serif;fill:#000}.heading{font:700 24px Arial,sans-serif;fill:#000}.meta,.kind,.title,.detail,.ref,.footer,.continuation{font-family:Arial,sans-serif;fill:#000;font-size:15px}.kind,.title{font-weight:700}</style><rect class="page" width="100%" height="100%"/><rect class="frame" x="${MARGIN - 18}" y="42" width="${page.width - (MARGIN - 18) * 2}" height="${page.height - 84}"/><text class="heading" x="${MARGIN}" y="76">${xml(`${projection.level} ${projection.title}`)}</text><text class="meta" x="${MARGIN}" y="102">${xml(`${modelIr.artifact.filename} | ${modelIr.artifact.format} | artifact sha256:${modelIr.artifact.sha256.slice(0, 12)}`)}</text><text class="meta" x="${page.width - MARGIN}" y="76" text-anchor="end">Page ${index + 1} of ${count} | ${xml(evidenceBadge)}</text>${continuationTop}<g class="connectors">${connectors}</g><g class="records">${blocks}</g><text class="footer" x="${MARGIN}" y="${page.height - 52}">${xml(`DEEPBOM Model IR ${modelIr.method_version} | model IR sha256:${modelIr.model_ir_sha256.slice(0, 16)} | monochrome theme 1`)}</text>${continuationBottom}</svg>\n`;
}

function buildConservation(modelIr, pages) {
  const exhaustivePages = pages.filter((page) => page.view === "exhaustive");
  const expected = modelIr.program.operations.length || modelIr.tensors_and_storage.storage_objects.length;
  const rendered = exhaustivePages.reduce((sum, page) => sum + page.rendered_member_count, 0);
  const operationIds = new Set(modelIr.program.operations.map((row) => row.id));
  const expectedRelationships = modelIr.program.relationships.filter((row) => ["data_dependency", "control_dependency", "state_dependency", "call"].includes(row.kind)
    && operationIds.has(row.from_ref) && operationIds.has(row.to_ref)).length;
  const renderedRelationships = exhaustivePages.reduce((sum, page) => sum + page.rendered_relationship_count, 0);
  return {
    exhaustive_subject_kind: modelIr.program.operations.length ? "serialized_operation" : "serialized_storage_object",
    expected_count: expected,
    rendered_count: rendered,
    omitted_count: expected - rendered,
    duplicate_count: duplicateCount(exhaustivePages.flatMap((page) => page.subject_refs)),
    expected_relationship_count: expectedRelationships,
    rendered_relationship_count: renderedRelationships,
    omitted_relationship_count: expectedRelationships - renderedRelationships,
    duplicate_relationship_count: duplicateCount(exhaustivePages.flatMap((page) => page.relationship_refs)),
    status: expected === rendered && expectedRelationships === renderedRelationships ? "conserved" : "failed",
  };
}

function caption(modelIr, projection, pageNumber, pageCount) {
  const flow = modelIr.program.status === "serialized"
    ? "Program relationships originate from the serialized artifact graph."
    : "The artifact does not serialize a program graph; no execution flow is synthesized from storage names.";
  return `Figure ${projection.level}-${pageNumber}. ${projection.title} for ${modelIr.artifact.filename} (SHA-256 ${modelIr.artifact.sha256.slice(0, 12)}), page ${pageNumber} of ${pageCount}. ${flow} ${projection.boundary}`;
}

function chunkRows(rows, size) {
  const chunks = [];
  for (let offset = 0; offset < rows.length; offset += size) chunks.push(rows.slice(offset, offset + size));
  return chunks;
}
function chunkProjectionRows(rows, size) {
  if (!rows.some((row) => row.page_group)) return chunkRows(rows, size);
  const groups = [];
  for (const item of rows) {
    const key = item.page_group || item.subject_ref;
    const prior = groups.at(-1);
    if (prior?.key === key) prior.rows.push(item);
    else groups.push({ key, rows: [item] });
  }
  const chunks = [];
  let current = [];
  for (const group of groups) {
    if (group.rows.length <= size) {
      if (current.length && current.length + group.rows.length > size) { chunks.push(current); current = []; }
      current.push(...group.rows);
      continue;
    }
    if (current.length) { chunks.push(current); current = []; }
    for (let offset = 0; offset < group.rows.length; offset += size) chunks.push(group.rows.slice(offset, offset + size));
  }
  if (current.length) chunks.push(current);
  return chunks;
}
function evidenceStyle(value) {
  const text = String(value || "").toUpperCase();
  if (text.includes("NOT_ASSESSABLE")) return "not-assessable";
  if (text.includes("DECLARED")) return "declared";
  if (text.includes("DERIVED") || text.includes("PREDICTED") || text.includes("INFERRED")) return "derived";
  return "observed";
}
function evidenceBadgeFor(rows) {
  const classes = [...new Set(rows.map((item) => evidenceStyle(item.evidence_class)))];
  return classes.length ? classes.join(" + ") : "not assessable";
}
function formatOperation(operation) {
  const macs = operation.metrics?.macs?.decimal;
  const inputCount = operation.input_port_refs?.length || 0;
  const outputCount = operation.output_port_refs?.length || 0;
  return `source ${operation.source_order ?? "?"} | dependency ${operation.dependency_order ?? "partial"} | display ${operation.display_order ?? "?"} | ${inputCount} in / ${outputCount} out${macs == null ? "" : ` | ${macs} MACs`}`;
}
function formatBlockOperation(modelIr, operation) {
  const portIds = new Set(operation.input_port_refs || []);
  const ports = modelIr.program.ports.filter((port) => portIds.has(port.id));
  const valueById = new Map(modelIr.program.values.map((value) => [value.id, value]));
  const storageById = new Map(modelIr.tensors_and_storage.storage_objects.map((storage) => [storage.id, storage]));
  const bound = [];
  for (const port of ports) {
    const value = valueById.get(port.value_ref);
    for (const storageRef of value?.storage_refs || []) {
      const storage = storageById.get(storageRef);
      if (storage) bound.push(`${storage.name || storage.id}:${encodingText(storage.encoding)}:${shapeText(storage.shape)}`);
    }
  }
  const inputContract = ports.slice(0, 3).map((port) => {
    const value = valueById.get(port.value_ref);
    return `${value?.dtype || "?"}${shapeText(value?.shape)}`;
  });
  return `${formatOperation(operation)} | inputs ${inputContract.join(", ") || "not materialized"}${bound.length ? ` | W ${bound.slice(0, 3).join(", ")}` : " | W -"}`;
}
function shapeText(shape) { return Array.isArray(shape) && shape.length ? `[${shape.join(", ")}]` : "shape unknown"; }
function encodingText(encoding) { return encoding?.name || encoding?.id || encoding?.kind || encoding || "encoding unknown"; }
function byteText(value) { return value?.decimal == null ? "bytes unknown" : `${value.decimal} serialized bytes`; }
function duplicateCount(values) { return values.length - new Set(values).size; }
function unique(value, index, array) { return array.indexOf(value) === index; }
function clip(value, maximum) { const text = String(value ?? "").replace(/[\r\n]+/g, " "); return text.length <= maximum ? text : `${text.slice(0, maximum - 1)}…`; }
function xml(value) { return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" })[char]); }
