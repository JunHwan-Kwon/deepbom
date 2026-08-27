// Independent recomputation of every derived number in
// .local-validation/supported-formats/latest/. Nothing here reuses the
// producing code paths: totals are re-derived from the raw fixtures and from
// the leaf records inside each artifact.
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const LATEST = path.join(ROOT, ".local-validation", "supported-formats", "latest");
const problems = [];
const notes = [];

const check = (ok, label, detail) => { (ok ? notes : problems).push(`${ok ? "ok  " : "BAD "} ${label}${detail ? ` — ${detail}` : ""}`); return ok; };
const json = (rel) => JSON.parse(readFileSync(path.join(LATEST, rel), "utf8"));
const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");
const close = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps * Math.max(1, Math.abs(a), Math.abs(b));

function walk(dir, rel = "") {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    const next = rel ? `${rel}/${entry.name}` : entry.name;
    if (entry.isDirectory()) out.push(...walk(full, next));
    else out.push(next);
  }
  return out;
}

// ---------------------------------------------------------------- manifest --
{
  const manifest = json("sha256-manifest.json");
  const matrix = json("validation-matrix.json");
  const declared = new Map(manifest.files.map((f) => [String(f.path).replaceAll("\\", "/"), f]));
  const actual = walk(LATEST).filter((p) => p !== "sha256-manifest.json");
  const actualSet = new Set(actual);

  check(manifest.file_count === manifest.files.length, "manifest.file_count matches manifest.files.length",
    `file_count=${manifest.file_count} files.length=${manifest.files.length}`);
  const absent = [...declared.keys()].filter((p) => !actualSet.has(p));
  const unlisted = actual.filter((p) => !declared.has(p));
  check(absent.length === 0, "every manifest entry exists on disk", `${absent.length} absent, e.g. ${absent.slice(0, 3).join(", ")}`);
  check(unlisted.length === 0, "every file on disk is covered by the manifest", `${unlisted.length} unlisted, e.g. ${unlisted.slice(0, 3).join(", ")}`);
  check(manifest.generated_at === matrix.generated_at, "manifest and matrix share generated_at",
    `manifest=${manifest.generated_at} matrix=${matrix.generated_at}`);

  let sizeBad = 0; let hashBad = 0; let checked = 0;
  for (const rel of actual) {
    const entry = declared.get(rel);
    if (!entry) continue;
    checked += 1;
    const bytes = readFileSync(path.join(LATEST, rel));
    if (bytes.length !== entry.byte_length) sizeBad += 1;
    else if (sha256(bytes) !== entry.sha256) hashBad += 1;
  }
  check(checked > 0 && sizeBad === 0 && hashBad === 0, "manifest hashes/sizes verify for covered files",
    `checked=${checked} sizeMismatch=${sizeBad} hashMismatch=${hashBad}`);
}

// -------------------------------------------------- fixture bytes vs matrix --
const matrix = json("validation-matrix.json");
const currentCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" }).trim();
const repositoryIdentity = repositoryContentIdentity();
check(matrix.environment?.git_commit === currentCommit, "validation matrix is bound to the current commit",
  `matrix=${matrix.environment?.git_commit || "missing"} current=${currentCommit}`);
check(matrix.environment?.repository_content_sha256 === repositoryIdentity.sha256,
  "validation matrix is bound to the exact repository content",
  `matrix=${matrix.environment?.repository_content_sha256 || "missing"} current=${repositoryIdentity.sha256}`);
check(matrix.environment?.repository_content_file_count === repositoryIdentity.fileCount,
  "validation matrix repository file count matches",
  `matrix=${matrix.environment?.repository_content_file_count} current=${repositoryIdentity.fileCount}`);
check(matrix.environment?.repository_content_hash_basis === repositoryIdentity.hashBasis,
  "validation matrix declares the independently reproduced repository hash basis");
notes.push(`info validation source worktree: ${matrix.environment?.git_worktree_dirty ? "dirty development snapshot identified by repository content hash" : "clean commit"}`);
check(matrix.overall_status === "pass", "validation matrix overall_status is pass",
  `overall_status=${matrix.overall_status}`);
check(matrix.formats.every((row) => row.status === "pass"), "every advertised parser/export path passed",
  matrix.formats.filter((row) => row.status !== "pass").map((row) => `${row.id}:${row.status}`).join(", "));
check(matrix.browser_selection?.status === "pass", "browser selection and rendered-surface validation passed",
  `browser_status=${matrix.browser_selection?.status || "missing"}`);

function repositoryContentIdentity() {
  const listing = execFileSync(
    "git",
    ["ls-files", "-z", "--cached", "--others", "--exclude-standard"],
    { cwd: ROOT, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
  );
  const files = listing.split("\0").filter(Boolean).sort();
  const digest = createHash("sha256");
  for (const relative of files) {
    let bytes;
    try {
      bytes = readFileSync(path.join(ROOT, relative));
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
      digest.update(`${relative}\0-1\0missing\n`);
      continue;
    }
    digest.update(`${relative}\0${bytes.byteLength}\0${sha256(bytes)}\n`);
  }
  return {
    sha256: digest.digest("hex"),
    fileCount: files.length,
    hashBasis: "sha256(sorted records: git ls-files --cached --others --exclude-standard path + NUL + byte_length + NUL + file_sha256 + LF; deleted tracked files use -1/missing)",
  };
}
// Fixture filenames come from the matrix itself so this stays correct when the
// harness swaps synthetic fixtures for public sample models.
const fixtureFor = {};
for (const row of matrix.formats) {
  if (!row.filename || row.hash_basis !== "artifact_file_bytes_sha256") continue;
  fixtureFor[row.id] = `fixtures/${row.filename}`;
}
for (const row of matrix.formats) {
  const rel = fixtureFor[row.id];
  if (!rel) continue;
  const bytes = readFileSync(path.join(LATEST, rel));
  check(bytes.length === row.byte_length, `${row.id}: matrix byte_length matches the fixture`, `matrix=${row.byte_length} actual=${bytes.length}`);
  check(sha256(bytes) === row.sha256, `${row.id}: matrix sha256 matches the fixture bytes`, `matrix=${row.sha256.slice(0, 16)}… actual=${sha256(bytes).slice(0, 16)}…`);
}

// ------------------------------------------------------ matrix vs analysis --
for (const row of matrix.formats) {
  let analysis;
  try { analysis = json(`results/${row.id}.analysis.json`); } catch { problems.push(`BAD  ${row.id}: results/${row.id}.analysis.json is missing`); continue; }
  if (row.operator_count != null) check(analysis.operator_count === row.operator_count, `${row.id}: operator_count matrix↔analysis`, `${row.operator_count} vs ${analysis.operator_count}`);
  if (row.tensor_count != null) check(analysis.tensor_count === row.tensor_count, `${row.id}: tensor_count matrix↔analysis`, `${row.tensor_count} vs ${analysis.tensor_count}`);
  if (row.total_macs != null) check(analysis.total_macs === row.total_macs, `${row.id}: total_macs matrix↔analysis`, `${row.total_macs} vs ${analysis.total_macs}`);
  // A null tensor_count is the analyzer declining to decode that inventory; an
  // empty tensors[] alongside it is expected, not a contradiction.
  if (Array.isArray(analysis.tensors) && analysis.tensor_count !== null) {
    check(analysis.tensors.length === analysis.tensor_count, `${row.id}: tensor_count equals tensors[].length`, `${analysis.tensor_count} vs ${analysis.tensors.length}`);
  }
  if (Array.isArray(analysis.ops) && analysis.operator_count != null) check(analysis.ops.length === analysis.operator_count, `${row.id}: operator_count equals ops[].length`, `${analysis.operator_count} vs ${analysis.ops.length}`);
}

// ---------------------------------------------------- TFLite arithmetic ----
{
  const a = json("results/tflite.analysis.json");
  const opMacSum = a.ops.reduce((s, op) => s + Number(op.macs || 0), 0);
  check(opMacSum === a.total_macs, "tflite: sum(ops[].macs) equals total_macs", `sum=${opMacSum} total=${a.total_macs}`);
  const nonMacOps = a.ops.filter((op) => !Number(op.macs)).reduce((s, op) => s + Number(op.ops || 0), 0);
  check(a.total_ops === a.total_macs * 2 + nonMacOps, "tflite: total_ops == 2 × total_macs + non-MAC ops",
    `${a.total_macs}×2+${nonMacOps}=${a.total_macs * 2 + nonMacOps} vs ${a.total_ops}`);
  check(close(a.total_ops / a.total_macs, 2.0, 5e-4), "tflite: displayed MAC:Op ratio 1:2.00 rounds correctly", `exact=${(a.total_ops / a.total_macs).toFixed(5)}`);
  check(Math.round(1000 * (100 * nonMacOps / a.total_ops)) / 1000 >= 0.045 && 100 * nonMacOps / a.total_ops < 0.055,
    "tflite: displayed non-MAC share 0.05% rounds correctly", `exact=${(100 * nonMacOps / a.total_ops).toFixed(4)}%`);
  check(a.delegated_macs + a.fallback_macs === a.total_macs, "tflite: delegated + fallback MACs conserve total", `${a.delegated_macs}+${a.fallback_macs} vs ${a.total_macs}`);
  const pct = a.total_macs ? a.delegated_macs / a.total_macs : 0;
  check(close(pct, a.delegated_mac_percent, 1e-6), "tflite: delegated_mac_percent matches the ratio", `derived=${pct} reported=${a.delegated_mac_percent}`);

  const sb = a.size_breakdown;
  check(sb.file_size === a.file_size, "tflite: size_breakdown.file_size equals analysis.file_size", `${sb.file_size} vs ${a.file_size}`);
  const parts = sb.constant_bytes + sb.metadata_bytes + sb.structure_overhead_bytes;
  check(parts === sb.file_size, "tflite: constants + metadata + overhead equals file size", `${sb.constant_bytes}+${sb.metadata_bytes}+${sb.structure_overhead_bytes}=${parts} vs ${sb.file_size}`);
  check(sb.unique_constant_bytes + sb.duplicate_constant_bytes === sb.constant_bytes, "tflite: unique + duplicate constant bytes conserve", `${sb.unique_constant_bytes}+${sb.duplicate_constant_bytes} vs ${sb.constant_bytes}`);

  const fixture = readFileSync(path.join(LATEST, fixtureFor.tflite));
  check(sha256(fixture) === a.model_sha256, "tflite: analysis.model_sha256 equals SHA-256 of the fixture bytes");
  check(fixture.length === a.file_size, "tflite: analysis.file_size equals real fixture length", `${a.file_size} vs ${fixture.length}`);

  const quantTensors = a.tensors.filter((t) => Number(t.quant_scales) > 0).length;
  check(a.quantized_tensors === quantTensors, "tflite: quantized_tensors matches a recount over tensors[].quant_scales",
    `reported=${a.quantized_tensors} recounted=${quantTensors}`);
  const perChannel = a.tensors.filter((t) => Number(t.quant_scales) > 1).length;
  check(a.per_channel_tensors === perChannel, "tflite: per_channel_tensors matches a recount (quant_scales > 1)",
    `reported=${a.per_channel_tensors} recounted=${perChannel}`);
  const typeHistogram = new Map();
  for (const t of a.tensors) typeHistogram.set(t.dtype, (typeHistogram.get(t.dtype) || 0) + 1);
  const typesOk = a.tensor_types.every((entry) => typeHistogram.get(entry.name) === entry.count)
    && a.tensor_types.reduce((s, e) => s + e.count, 0) === a.tensor_count;
  check(typesOk, "tflite: tensor_types histogram matches a recount over tensors[].dtype",
    `reported=${JSON.stringify(a.tensor_types)} recounted=${JSON.stringify([...typeHistogram])}`);

  const constantBytes = a.tensors.filter((t) => t.constant_buffer).reduce((s, t) => s + Number(t.buffer_data_length || 0), 0);
  notes.push(`info tflite: sum(constant tensor buffer_data_length) = ${constantBytes} vs size_breakdown.constant_bytes = ${sb.constant_bytes}`);
  for (const t of a.tensors) {
    if (!t.constant_buffer) continue;
    if (Number(t.buffer_data_offset) + Number(t.buffer_data_length) > a.file_size) {
      problems.push(`BAD  tflite: constant buffer for ${t.name} extends past EOF (${t.buffer_data_offset}+${t.buffer_data_length} > ${a.file_size})`);
      break;
    }
  }
}

// ------------------------------------------------------ ONNX arithmetic ----
{
  const a = json("results/onnx.analysis.json");
  const opMacSum = a.ops.reduce((s, op) => s + Number(op.macs || 0), 0);
  check(opMacSum === a.total_macs, "onnx: sum(ops[].macs) equals total_macs", `sum=${opMacSum} total=${a.total_macs}`);
  const fixture = readFileSync(path.join(LATEST, fixtureFor.onnx));
  check(fixture.length === a.file_size_bytes || fixture.length === a.file_size, "onnx: analysis file size equals real fixture length", `${a.file_size_bytes ?? a.file_size} vs ${fixture.length}`);
  check(sha256(fixture) === a.model_sha256, "onnx: analysis.model_sha256 equals SHA-256 of the fixture bytes");

  // Recompute Conv/Gemm MACs straight from the declared shapes.
  let derived = 0;
  const detail = [];
  for (const op of a.ops) {
    const macs = Number(op.macs || 0);
    if (!macs) continue;
    const label = op.graph_node_name ? `${op.name}(${op.graph_node_name})` : op.name;
    detail.push(`${label}:${macs}`);
    derived += macs;
  }
  check(derived === a.total_macs, "onnx: MAC-bearing ops re-sum to total_macs", `${derived} vs ${a.total_macs}`);
  notes.push(`info onnx: MAC-bearing ops = ${detail.join(", ")}`);
}

// ------------------------------------ container formats: byte conservation --
{
  const st = json("results/safetensors.analysis.json");
  const bytes = readFileSync(path.join(LATEST, fixtureFor.safetensors));
  const headerLen = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
  check(st.safetensors.header_byte_length === headerLen, "safetensors: header_byte_length matches the 8-byte prefix", `${st.safetensors.header_byte_length} vs ${headerLen}`);
  check(8 + headerLen + st.safetensors.payload_byte_length === bytes.length, "safetensors: 8 + header + payload equals file size",
    `8+${headerLen}+${st.safetensors.payload_byte_length}=${8 + headerLen + st.safetensors.payload_byte_length} vs ${bytes.length}`);
  const declaredTensorBytes = st.tensors.reduce((s, t) => s + Number(t.byte_length || 0), 0);
  check(declaredTensorBytes === st.safetensors.payload_byte_length, "safetensors: sum(tensor byte_length) equals payload length",
    `${declaredTensorBytes} vs ${st.safetensors.payload_byte_length}`);
  check(st.tensor_inventory.total_declared_tensor_bytes === declaredTensorBytes, "safetensors: tensor_inventory total matches recount",
    `${st.tensor_inventory.total_declared_tensor_bytes} vs ${declaredTensorBytes}`);
  const safeElements = st.tensors.reduce((sum, tensor) => sum + (tensor.shape || []).reduce((product, dimension) => product * BigInt(dimension), 1n), 0n);
  const safeStorage = st.tensor_storage_summary || {};
  check(safeStorage.element_count_decimal === String(safeElements), "safetensors: storage summary exact element count matches tensor shapes",
    `${safeStorage.element_count_decimal ?? "missing"} vs ${safeElements}`);
  check(safeStorage.byte_length === declaredTensorBytes, "safetensors: storage summary bytes match declared tensor bytes",
    `${safeStorage.byte_length ?? "missing"} vs ${declaredTensorBytes}`);

  const gg = json("results/gguf.analysis.json");
  const gbytes = readFileSync(path.join(LATEST, fixtureFor.gguf));
  check(gg.file_size === gbytes.length, "gguf: analysis file_size equals real fixture length", `${gg.file_size} vs ${gbytes.length}`);
  check(gg.gguf.tensor_data_offset % gg.gguf.alignment === 0, "gguf: tensor_data_offset is a multiple of alignment",
    `offset=${gg.gguf.tensor_data_offset} align=${gg.gguf.alignment}`);
  check(gg.gguf.tensor_data_offset <= gg.file_size, "gguf: tensor_data_offset lies inside the file", `${gg.gguf.tensor_data_offset} vs ${gg.file_size}`);
  check(gg.gguf.tensor_count === gg.tensors.length, "gguf: header tensor_count equals parsed tensors", `${gg.gguf.tensor_count} vs ${gg.tensors.length}`);
  const ggufElements = gg.tensors.reduce((sum, tensor) => sum + (tensor.shape || []).reduce((product, dimension) => product * BigInt(dimension), 1n), 0n);
  const ggufStorage = gg.tensor_storage_summary || {};
  check(ggufStorage.element_count_decimal === String(ggufElements), "gguf: storage summary exact element count matches tensor shapes",
    `${ggufStorage.element_count_decimal ?? "missing"} vs ${ggufElements}`);
  check(ggufStorage.byte_length === gg.gguf.declared_tensor_byte_length, "gguf: storage summary bytes match declared tensor bytes",
    `${ggufStorage.byte_length ?? "missing"} vs ${gg.gguf.declared_tensor_byte_length}`);
}

// ------------------------------------------------------- bundle arithmetic --
for (const id of ["mlpackage", "sharded_safetensors"]) {
  const a = json(`results/${id}.analysis.json`);
  const bundle = a.artifact_bundle;
  if (!bundle) { problems.push(`BAD  ${id}: analysis has no artifact_bundle`); continue; }
  const row = matrix.formats.find((f) => f.id === id);
  check(bundle.files.length === row.package_file_count, `${id}: package_file_count matches bundle.files length`, `${row.package_file_count} vs ${bundle.files.length}`);
  const total = bundle.files.reduce((s, f) => s + Number(f.byte_length || 0), 0);
  check(total === a.file_size_bytes, `${id}: sum(bundle file byte_length) equals file_size_bytes`, `${total} vs ${a.file_size_bytes}`);
  check(bundle.bundle_sha256 === a.model_sha256, `${id}: bundle_sha256 equals analysis.model_sha256`);
  check(bundle.bundle_sha256 === row.sha256, `${id}: matrix sha256 equals bundle_sha256`);
  if (id === "sharded_safetensors") {
    const elements = a.tensors.reduce((sum, tensor) => sum + (tensor.shape || []).reduce((product, dimension) => product * BigInt(dimension), 1n), 0n);
    const bytes = a.tensors.reduce((sum, tensor) => sum + Number(tensor.byte_length || 0), 0);
    const storage = a.tensor_storage_summary || {};
    check(storage.element_count_decimal === String(elements), `${id}: storage summary covers every shard tensor element`, `${storage.element_count_decimal ?? "missing"} vs ${elements}`);
    check(storage.byte_length === bytes, `${id}: storage summary covers every shard tensor byte`, `${storage.byte_length ?? "missing"} vs ${bytes}`);
  }

  // Re-hash every member file straight off disk against the bundle evidence.
  let bad = 0;
  for (const entry of bundle.files) {
    const rel = path.join("fixtures", entry.path);
    let bytes;
    try { bytes = readFileSync(path.join(LATEST, rel)); } catch { bad += 1; continue; }
    if (bytes.length !== entry.byte_length || sha256(bytes) !== entry.sha256) bad += 1;
  }
  check(bad === 0, `${id}: every bundle file re-hashes to its recorded evidence`, `${bad} mismatched of ${bundle.files.length}`);

  // Recompute the declared bundle digest from its own hash basis.
  const rows = bundle.files
    .map(({ path: p, byte_length, sha256: hash, role, required }) => ({ path: p, byte_length, sha256: hash, role, required }))
    .sort((x, y) => x.path.localeCompare(y.path));
  const recomputed = sha256(Buffer.from(JSON.stringify({ schema: "deepbom.artifact_bundle_digest.v1", files: rows })));
  check(recomputed === bundle.bundle_sha256, `${id}: bundle digest recomputes from its stated hash basis`, `derived=${recomputed.slice(0, 16)}… recorded=${bundle.bundle_sha256.slice(0, 16)}…`);
}

{
  const a = json("results/sharded_safetensors.analysis.json");
  const declared = a.tensors.reduce((s, t) => s + Number(t.byte_length || 0), 0);
  check(a.tensor_inventory.total_declared_tensor_bytes === declared, "sharded: tensor_inventory total matches recount", `${a.tensor_inventory.total_declared_tensor_bytes} vs ${declared}`);
  check(a.tensor_count === a.tensors.length, "sharded: tensor_count equals tensors[].length", `${a.tensor_count} vs ${a.tensors.length}`);
  check(a.safetensors.shard_count === 2 && a.safetensors.index_tensor_count === a.tensor_count, "sharded: shard/index counts agree",
    `shards=${a.safetensors.shard_count} indexTensors=${a.safetensors.index_tensor_count} tensors=${a.tensor_count}`);
}

// --------------------------------------------------- export document math --
for (const id of ["tflite", "onnx", "gguf", "safetensors", "mlmodel", "mlpackage", "sharded_safetensors"]) {
  const analysis = json(`results/${id}.analysis.json`);
  let bom; let envelope;
  try {
    bom = json(`exports/${id}/cyclonedx_evidence.json`);
    envelope = json(`exports/${id}/artifact_evidence_envelope.json`);
  } catch { problems.push(`BAD  ${id}: export documents missing`); continue; }

  const subject = bom.metadata?.component;
  const bomHash = subject?.hashes?.find((h) => h.alg === "SHA-256")?.content;
  if (bomHash) check(bomHash === analysis.model_sha256, `${id}: CycloneDX subject hash equals analysis.model_sha256`, `${bomHash.slice(0, 16)}… vs ${String(analysis.model_sha256).slice(0, 16)}…`);
  const envHash = envelope.artifact?.sha256 || envelope.subject?.sha256 || envelope.model_sha256;
  if (envHash) check(envHash === analysis.model_sha256, `${id}: evidence envelope hash equals analysis.model_sha256`, `${String(envHash).slice(0, 16)}… vs ${String(analysis.model_sha256).slice(0, 16)}…`);

  const envSize = envelope.artifact?.file_size_bytes ?? envelope.subject?.file_size_bytes ?? envelope.file_size_bytes;
  if (envSize != null) check(envSize === (analysis.file_size_bytes ?? analysis.file_size), `${id}: envelope file size equals analysis file size`, `${envSize} vs ${analysis.file_size_bytes ?? analysis.file_size}`);

  const cons = envelope.capabilities?.conservation;
  if (cons) {
    check(cons.valid === true, `${id}: envelope conservation self-check is valid`, JSON.stringify(cons).slice(0, 160));
    if (cons.total_macs != null && analysis.total_macs != null) check(cons.total_macs === analysis.total_macs, `${id}: envelope conservation total_macs matches analysis`, `${cons.total_macs} vs ${analysis.total_macs}`);
  }
}

// ---------------------------------------------- browser-reported numbers ----
{
  // Re-derive each label with the app's own formatter rather than a hand-rolled
  // approximation, so this checks the number and not my rounding convention.
  const { formatBytes } = await import(pathToFileURL(path.join(ROOT, "web", "lib", "format.js")).href);
  const browser = json("results/browser-selection.json");
  for (const item of browser.cases) {
    const row = matrix.formats.find((f) => f.id === item.id);
    const derived = formatBytes(row.byte_length);
    check(derived === item.artifact_binding_bytes, `${item.id}: browser byte label re-derives from matrix byte_length`,
      `derived="${derived}" shown="${item.artifact_binding_bytes}" raw=${row.byte_length}`);
  }
}

// --------------------------------------- v2: artifact catalog arithmetic ----
{
  let catalog;
  try { catalog = json("artifact-catalog.json"); } catch { catalog = null; }
  if (catalog) {
    const totals = { actual: 0, unavailable: 0, capturedUi: 0, skippedUi: 0 };
    for (const entry of catalog.formats) {
      totals.actual += entry.actual_downloads.length;
      totals.unavailable += entry.unavailable_downloads.length;
      totals.capturedUi += entry.captured_ui_surfaces.length;
      totals.skippedUi += (entry.unavailable_ui_surfaces || []).length;
    }
    const cov = matrix.coverage;
    check(cov.actual_provided_downloads === totals.actual, "coverage.actual_provided_downloads recounts from the catalog", `${cov.actual_provided_downloads} vs ${totals.actual}`);
    check(cov.unavailable_download_states === totals.unavailable, "coverage.unavailable_download_states recounts from the catalog", `${cov.unavailable_download_states} vs ${totals.unavailable}`);
    check(cov.captured_ui_surfaces === totals.capturedUi, "coverage.captured_ui_surfaces recounts from the catalog", `${cov.captured_ui_surfaces} vs ${totals.capturedUi}`);
    check(cov.skipped_ui_surface_states === totals.skippedUi, "coverage.skipped_ui_surface_states recounts from the catalog", `${cov.skipped_ui_surface_states} vs ${totals.skippedUi}`);
    check(catalog.formats.length === matrix.formats.length && catalog.formats.length === cov.advertised_paths,
      "catalog format count equals matrix formats and advertised_paths", `${catalog.formats.length} / ${matrix.formats.length} / ${cov.advertised_paths}`);

    // Every catalogued download must exist and re-hash to its recorded evidence.
    let missing = 0; let sizeBad = 0; let hashBad = 0; let zipBad = 0; let total = 0;
    for (const entry of catalog.formats) {
      for (const download of entry.actual_downloads) {
        total += 1;
        const full = path.join(LATEST, download.path);
        let bytes;
        try { bytes = readFileSync(full); } catch { missing += 1; continue; }
        if (bytes.length !== download.byte_length) sizeBad += 1;
        else if (sha256(bytes) !== download.sha256) hashBad += 1;
        if (download.extracted_zip_directory) {
          // `_contents.json` is a harness-written index, not a ZIP member.
          const dir = path.join(LATEST, download.extracted_zip_directory);
          let members = -1;
          try { members = walk(dir).filter((p) => p !== "_contents.json").length; } catch { members = -1; }
          if (members !== download.zip_member_count) {
            zipBad += 1;
            problems.push(`BAD  ${entry.id}: zip_member_count ${download.zip_member_count} but ${members} extracted members under ${download.extracted_zip_directory}`);
          }
          // Cross-check the harness index against the catalog count too.
          try {
            const contents = JSON.parse(readFileSync(path.join(dir, "_contents.json"), "utf8"));
            const listed = Array.isArray(contents) ? contents.length : Array.isArray(contents.files) ? contents.files.length : Array.isArray(contents.entries) ? contents.entries.length : null;
            if (listed != null && listed !== download.zip_member_count) {
              zipBad += 1;
              problems.push(`BAD  ${entry.id}: _contents.json lists ${listed} members but zip_member_count is ${download.zip_member_count} (${download.extracted_zip_directory})`);
            }
          } catch { /* no index to cross-check */ }
        }
      }
    }
    check(missing === 0 && sizeBad === 0 && hashBad === 0, "every catalogued download exists and re-hashes",
      `total=${total} missing=${missing} sizeMismatch=${sizeBad} hashMismatch=${hashBad}`);
    check(zipBad === 0, "every extracted ZIP matches its recorded member count", `${zipBad} mismatched`);

    // Every captured UI surface must have its three recorded artifacts on disk.
    let uiMissing = 0; let uiTotal = 0;
    for (const entry of catalog.formats) {
      for (const surface of entry.captured_ui_surfaces) {
        for (const key of ["screenshot", "text", "dom"]) {
          if (!surface[key]) continue;
          uiTotal += 1;
          try { statSync(path.join(LATEST, surface[key])); } catch { uiMissing += 1; }
        }
      }
    }
    check(uiMissing === 0, "every captured UI surface artifact exists on disk", `${uiMissing} missing of ${uiTotal}`);
  }
}

// --------------------------- "not assessed" must not be reported as zero ----
for (const id of ["mlmodel", "mlpackage"]) {
  const analysis = json(`results/${id}.analysis.json`);
  const row = matrix.formats.find((f) => f.id === id);
  if (analysis.tensor_count === null && row.tensor_count === 0) {
    problems.push(`BAD  ${id}: analysis.tensor_count is null (not decoded) but the matrix/README report tensor_count 0 — "not assessed" is published as "zero"`);
  }
  if (analysis.operator_count === null && row.operator_count === 0) {
    problems.push(`BAD  ${id}: analysis.operator_count is null but the matrix reports 0`);
  }
  notes.push(`info ${id}: analysis tensor_count=${JSON.stringify(analysis.tensor_count)} operator_count=${JSON.stringify(analysis.operator_count)} mac_assessment=${analysis.mac_assessment?.status} | matrix tensor_count=${JSON.stringify(row.tensor_count)}`);
}

// ------------------------------------------------------------------ report --
console.log(notes.join("\n"));
console.log(`\n================ ${problems.length} PROBLEM(S) ================`);
for (const line of problems) console.log(line);
if (!problems.length) console.log("(none)");
else process.exitCode = 1;
