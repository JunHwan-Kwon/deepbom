// Local report history + model-version comparison, entirely in-browser.
//
// Every completed audit stores a compact snapshot (identity, counts, contracts,
// quantization coverage, delegation summary, static hotspots, optional runtime
// results — NOT model bytes or full tensors) in IndexedDB, keyed by
// sha256+target. Report exports append to the entry's artifact log. Snapshots
// are what the version-comparison diff runs on, so two audits of different
// builds of the same model can be compared without re-selecting either file.

const DB_NAME = "deepbom-local-reports";
const DB_VERSION = 1;
const STORE = "snapshots";
const SETTINGS_KEY = "deepbom.local-report-settings.v1";
const DEFAULT_SETTINGS = Object.freeze({ enabled: true, retentionDays: 90, maxSnapshots: 50 });

export function readReportHistorySettings(storage = null) {
  try {
    const parsed = JSON.parse((storage || globalThis.localStorage)?.getItem(SETTINGS_KEY) || "{}");
    const retentionDays = [0, 30, 90, 365].includes(Number(parsed.retentionDays))
      ? Number(parsed.retentionDays)
      : DEFAULT_SETTINGS.retentionDays;
    return { enabled: parsed.enabled !== false, retentionDays, maxSnapshots: DEFAULT_SETTINGS.maxSnapshots };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function writeReportHistorySettings(settings, storage = null) {
  const normalized = {
    enabled: settings?.enabled !== false,
    retentionDays: [0, 30, 90, 365].includes(Number(settings?.retentionDays)) ? Number(settings.retentionDays) : DEFAULT_SETTINGS.retentionDays,
    maxSnapshots: DEFAULT_SETTINGS.maxSnapshots,
  };
  try { (storage || globalThis.localStorage)?.setItem(SETTINGS_KEY, JSON.stringify(normalized)); } catch { /* local storage unavailable */ }
  return normalized;
}

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: "id" });
        store.createIndex("sha256", "sha256", { unique: false });
        store.createIndex("updatedAt", "updatedAt", { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(db, mode, run) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const store = t.objectStore(STORE);
    const out = run(store);
    t.oncomplete = () => resolve(out?.result ?? out);
    t.onerror = () => reject(t.error);
  });
}

// ── Snapshot builder (compact; no weights, no full tensor list) ──────────────

export function buildAuditSnapshot(analysis, { analyzerVersion = "", rulepackVersion = "", runtimeBenchmarkResults = [], sha256 = "", reportMarkdown = "" } = {}) {
  if (!analysis) return null;
  const ops = analysis.ops || [];
  const costsAssessed = ops.length > 0 && ops.every((op) => (
    op.bottleneck_assessment_status !== "not_assessed"
    && [op.bottleneck_total_us, op.bottleneck_packing_us, op.bottleneck_break_us]
      .every((value) => value != null && Number.isFinite(Number(value)))
  ));
  const totalColdStartUs = costsAssessed ? ops.reduce((s, op) => s + Number(op.bottleneck_total_us), 0) : null;
  const totalPackingUs = costsAssessed ? ops.reduce((s, op) => s + Number(op.bottleneck_packing_us), 0) : null;
  const totalBoundarySetupUs = costsAssessed ? ops.reduce((s, op) => s + Number(op.bottleneck_break_us), 0) : null;
  const totalSteadyUs = costsAssessed ? Math.max(0, totalColdStartUs - totalPackingUs - totalBoundarySetupUs) : null;
  const hotspots = [...ops]
    .map((op) => ({
      op,
      steadyUs: op.bottleneck_assessment_status === "not_assessed" || op.bottleneck_total_us == null
        ? null
        : Math.max(0, Number(op.bottleneck_total_us) - Number(op.bottleneck_packing_us) - Number(op.bottleneck_break_us)),
    }))
    .filter((row) => Number.isFinite(row.steadyUs) && row.steadyUs > 0)
    .sort((a, b) => b.steadyUs - a.steadyUs || Number(a.op.index) - Number(b.op.index))
    .slice(0, 10)
    .map(({ op, steadyUs }) => ({
      index: op.index, name: op.name,
      est_us: Number(steadyUs.toFixed(2)),
      cold_start_est_us: Number(Number(op.bottleneck_total_us || 0).toFixed(2)),
      macs: op.macs || 0,
      dominant: op.bottleneck_dominant || "",
    }));
  const contract = (tensors) => (tensors || []).map((t) => `${t.dtype || "?"}[${(t.shape || []).join("x")}]`);
  return {
    // The WASM payload's model_sha256 is empty until the async browser hash
    // completes — callers must pass the resolved hash explicitly.
    sha256: sha256 || analysis.model_sha256 || "",
    filename: analysis.filename || "",
    format: analysis.format || "",
    fileSize: analysis.file_size || 0,
    modelLineageId: analysis.model_lineage_id || "",
    previousArtifactSha256: analysis.previous_artifact_sha256 || "",
    derivationManifestId: analysis.derivation_manifest_id || "",
    target: analysis.target_profile?.id || "",
    targetLabel: analysis.target_profile?.label || analysis.target_profile?.id || "",
    analyzerVersion,
    rulepackVersion,
    operatorCount: analysis.operator_count ?? ops.length,
    tensorCount: analysis.tensor_count || 0,
    totalMacs: analysis.total_macs ?? null,
    modeledCostStatus: costsAssessed ? "assessed" : "not_assessed",
    modeledCostReason: costsAssessed ? null : "required shape, dtype, or target cost inputs are unavailable",
    totalEstUs: costsAssessed ? Number(totalSteadyUs.toFixed(1)) : null,
    totalColdStartEstUs: costsAssessed ? Number(totalColdStartUs.toFixed(1)) : null,
    totalPackingUs: costsAssessed ? Number(totalPackingUs.toFixed(1)) : null,
    totalBoundarySetupUs: costsAssessed ? Number(totalBoundarySetupUs.toFixed(1)) : null,
    opHistogram: (analysis.histogram || []).map((h) => ({ name: h.name, count: h.count })),
    inputContract: contract(analysis.inputs),
    outputContract: contract(analysis.outputs),
    quant: {
      quantizedTensors: analysis.quantized_tensors || 0,
      perChannelTensors: analysis.per_channel_tensors || 0,
      quantComputeMacPercent: analysis.quantization_status?.quantized_compute_mac_percent == null
        ? null
        : Number(analysis.quantization_status.quantized_compute_mac_percent),
      quantHoles: analysis.quant_hole_count || (analysis.quant_holes || []).length || 0,
      classification: analysis.quantization_status?.classification || "",
    },
    delegation: {
      delegatedOps: ops.filter((op) => Number(op.xnnpack_chain_id) >= 0).length,
      chainBreaks: analysis.xnnpack_chain_breaks || 0,
      effectiveChainBreaks: analysis.xnnpack_effective_chain_breaks || 0,
      chains: (analysis.xnnpack_chains || []).length,
      delegatedMacPercent: Number(analysis.delegated_mac_percent || 0),
    },
    hotspots,
    runtime: (runtimeBenchmarkResults || []).map((r) => ({
      backend: r.backend || r.label || "?",
      p50_ms: r.p50_ms ?? r.p50 ?? null,
      mean_ms: r.mean_ms ?? r.mean ?? null,
    })),
    // Full raw static-audit report body (ungated markdown) so "Reopen" shows a
    // real report without the original model. Weights are never included.
    reportMarkdown,
  };
}

export async function updateSnapshotNote(id, userNote) {
  const db = await openDb();
  const entry = await tx(db, "readonly", (s) => s.get(id)).catch(() => null);
  if (entry) {
    entry.userNote = String(userNote || "").slice(0, 500);
    entry.updatedAt = new Date().toISOString();
    await tx(db, "readwrite", (s) => s.put(entry));
  }
  db.close();
  return entry;
}

// ── CRUD ─────────────────────────────────────────────────────────────────────

export async function saveAuditSnapshot(snapshot) {
  if (!snapshot?.sha256) return null;
  const db = await openDb();
  const id = `${snapshot.sha256}:${snapshot.target}`;
  const existing = await tx(db, "readonly", (s) => s.get(id)).catch(() => null);
  const entry = {
    id,
    createdAt: existing?.createdAt || new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    runCount: (existing?.runCount || 0) + 1,
    artifacts: existing?.artifacts || [],
    userNote: existing?.userNote || "",
    ...snapshot,
  };
  await tx(db, "readwrite", (s) => s.put(entry));
  db.close();
  return entry;
}

export async function recordReportArtifact(sha256, target, artifact) {
  const db = await openDb();
  const id = `${sha256}:${target}`;
  const entry = await tx(db, "readonly", (s) => s.get(id)).catch(() => null);
  if (entry) {
    entry.artifacts = [...(entry.artifacts || []), { ...artifact, at: new Date().toISOString() }].slice(-20);
    entry.updatedAt = new Date().toISOString();
    await tx(db, "readwrite", (s) => s.put(entry));
  }
  db.close();
  return entry;
}

export async function listSnapshots() {
  const db = await openDb();
  const all = await tx(db, "readonly", (s) => s.getAll());
  db.close();
  return (all || []).sort((a, b) => (b.updatedAt || "").localeCompare(a.updatedAt || ""));
}

export async function deleteSnapshot(id) {
  const db = await openDb();
  await tx(db, "readwrite", (s) => s.delete(id));
  db.close();
}

export async function pruneAuditSnapshots(settings = readReportHistorySettings()) {
  const db = await openDb();
  const all = await tx(db, "readonly", (s) => s.getAll());
  const cutoff = settings.retentionDays > 0 ? Date.now() - settings.retentionDays * 86400000 : null;
  const ordered = [...(all || [])].sort((a, b) => (b.updatedAt || "").localeCompare(a.updatedAt || ""));
  const remove = ordered.filter((entry, index) => {
    const expired = cutoff != null && (!Date.parse(entry.updatedAt || entry.createdAt || "") || Date.parse(entry.updatedAt || entry.createdAt || "") < cutoff);
    return expired || index >= settings.maxSnapshots;
  });
  if (remove.length) await tx(db, "readwrite", (s) => { for (const entry of remove) s.delete(entry.id); });
  db.close();
  return remove.length;
}

// ── Version-comparison diff engine (pure functions, testable in Node) ────────

function pct(value) {
  if (value == null) return "N/A";
  const scaled = Number(value || 0) * 100;
  return `${Object.is(Math.round(scaled * 10) / 10, -0) ? 0 : Number(scaled.toFixed(1))}%`;
}

export function diffHistograms(histA = [], histB = []) {
  const names = new Set([...histA.map((h) => h.name), ...histB.map((h) => h.name)]);
  const rows = [];
  for (const name of [...names].sort()) {
    const a = histA.find((h) => h.name === name)?.count || 0;
    const b = histB.find((h) => h.name === name)?.count || 0;
    if (a !== b) rows.push({ name, a, b, delta: b - a });
  }
  return rows;
}

export function diffContracts(listA = [], listB = []) {
  const max = Math.max(listA.length, listB.length);
  const rows = [];
  for (let i = 0; i < max; i++) {
    if ((listA[i] || "(none)") !== (listB[i] || "(none)")) {
      rows.push({ slot: i, a: listA[i] || "(none)", b: listB[i] || "(none)" });
    }
  }
  return rows;
}

export function diffHotspots(hotA = [], hotB = []) {
  const byName = (list) => {
    const m = new Map();
    for (const h of list) m.set(`#${h.index} ${h.name}`, h);
    return m;
  };
  const a = byName(hotA), b = byName(hotB);
  const keys = new Set([...a.keys(), ...b.keys()]);
  const rows = [];
  for (const key of keys) {
    const ha = a.get(key), hb = b.get(key);
    if (!ha) rows.push({ op: key, change: `entered top-10 (${hb.est_us} µs)` });
    else if (!hb) rows.push({ op: key, change: `left top-10 (was ${ha.est_us} µs)` });
    else if (Math.abs(ha.est_us - hb.est_us) > Math.max(0.5, ha.est_us * 0.05)) {
      rows.push({ op: key, change: `${ha.est_us} → ${hb.est_us} µs (${hb.est_us > ha.est_us ? "+" : ""}${(hb.est_us - ha.est_us).toFixed(1)})` });
    }
  }
  return rows;
}

export function buildComparisonReport(a, b) {
  const lines = [];
  const same = Boolean(a.sha256) && a.sha256 === b.sha256;
  const fmtNum = (n) => new Intl.NumberFormat("en-US").format(Math.round(n || 0));
  lines.push(`# DEEPBOM Artifact Version Comparison`);
  lines.push("");
  lines.push(`**Evidence level: Static comparison of stored audit snapshots** — values are static estimates from each audit run, not measurements.`);
  lines.push("");
  lines.push(`| Field | A | B |`);
  lines.push(`|---|---|---|`);
  lines.push(`| Filename | ${a.filename} | ${b.filename} |`);
  lines.push(`| SHA-256 | \`${a.sha256.slice(0, 16)}…\` | \`${b.sha256.slice(0, 16)}…\` |`);
  lines.push(`| Artifact identity | ${same ? "**identical bytes**" : "**different artifacts**"} | |`);
  lines.push(`| Audited | ${a.updatedAt || a.createdAt || "-"} | ${b.updatedAt || b.createdAt || "-"} |`);
  lines.push(`| Target profile | ${a.target} | ${b.target} |`);
  lines.push(`| Analyzer / rulepack | ${a.analyzerVersion} / ${a.rulepackVersion} | ${b.analyzerVersion} / ${b.rulepackVersion} |`);
  lines.push(`| File size | ${fmtNum(a.fileSize)} B | ${fmtNum(b.fileSize)} B |`);
  lines.push("");

  lines.push(`## Operator Diff`);
  lines.push(`Ops: ${a.operatorCount} → ${b.operatorCount} (${b.operatorCount - a.operatorCount >= 0 ? "+" : ""}${b.operatorCount - a.operatorCount})`);
  const hist = diffHistograms(a.opHistogram, b.opHistogram);
  if (hist.length) {
    lines.push("");
    lines.push(`| Op family | A | B | Δ |`);
    lines.push(`|---|---|---|---|`);
    for (const row of hist.slice(0, 20)) lines.push(`| ${row.name} | ${row.a} | ${row.b} | ${row.delta > 0 ? "+" : ""}${row.delta} |`);
  } else {
    lines.push(`No op-family count changes.`);
  }
  lines.push("");

  lines.push(`## Tensor Contract Diff`);
  const inDiff = diffContracts(a.inputContract, b.inputContract);
  const outDiff = diffContracts(a.outputContract, b.outputContract);
  if (!inDiff.length && !outDiff.length) lines.push(`Input/output contracts are identical (${a.inputContract.join(", ")} → ${a.outputContract.join(", ")}).`);
  for (const row of inDiff) lines.push(`- input[${row.slot}]: ${row.a} → ${row.b}`);
  for (const row of outDiff) lines.push(`- output[${row.slot}]: ${row.a} → ${row.b}`);
  lines.push("");

  lines.push(`## Quantization Coverage Diff`);
  lines.push(`| Metric | A | B |`);
  lines.push(`|---|---|---|`);
  lines.push(`| Quantized tensors | ${a.quant.quantizedTensors}/${a.tensorCount} | ${b.quant.quantizedTensors}/${b.tensorCount} |`);
  lines.push(`| Per-channel tensors | ${a.quant.perChannelTensors} | ${b.quant.perChannelTensors} |`);
  lines.push(`| Quantized compute MACs | ${pct(a.quant.quantComputeMacPercent)} | ${pct(b.quant.quantComputeMacPercent)} |`);
  lines.push(`| Quant holes | ${a.quant.quantHoles} | ${b.quant.quantHoles} |`);
  lines.push(`| Classification | ${a.quant.classification || "-"} | ${b.quant.classification || "-"} |`);
  lines.push("");

  lines.push(`## Predicted Delegate Segment Diff`);
  lines.push(`| Metric | A | B |`);
  lines.push(`|---|---|---|`);
  lines.push(`| Conditionally delegatable ops | ${a.delegation.delegatedOps}/${a.operatorCount} | ${b.delegation.delegatedOps}/${b.operatorCount} |`);
  lines.push(`| Chains | ${a.delegation.chains} | ${b.delegation.chains} |`);
  lines.push(`| Compute-adjacent predicted boundaries | ${a.delegation.effectiveChainBreaks} | ${b.delegation.effectiveChainBreaks} |`);
  lines.push(`| Conditionally delegatable MACs | ${pct(a.delegation.delegatedMacPercent)} | ${pct(b.delegation.delegatedMacPercent)} |`);
  lines.push("");

  lines.push(`## Static Hotspot Diff`);
  lines.push(`Steady-state est. time: ${a.totalEstUs} µs → ${b.totalEstUs} µs (heuristic static estimate)`);
  lines.push(`Cold-start est. time: ${a.totalColdStartEstUs ?? a.totalEstUs} µs → ${b.totalColdStartEstUs ?? b.totalEstUs} µs (includes one-time packing)`);
  const hot = diffHotspots(a.hotspots, b.hotspots);
  if (hot.length) {
    for (const row of hot.slice(0, 12)) lines.push(`- ${row.op}: ${row.change}`);
  } else {
    lines.push(`Top-10 static hotspots unchanged within tolerance.`);
  }
  lines.push("");

  lines.push(`## Runtime Regression Diff`);
  if (a.runtime.length && b.runtime.length) {
    lines.push(`| Backend | A p50 | B p50 |`);
    lines.push(`|---|---|---|`);
    const backends = new Set([...a.runtime.map((r) => r.backend), ...b.runtime.map((r) => r.backend)]);
    for (const backend of backends) {
      const ra = a.runtime.find((r) => r.backend === backend);
      const rb = b.runtime.find((r) => r.backend === backend);
      lines.push(`| ${backend} | ${ra?.p50_ms != null ? `${ra.p50_ms} ms` : "not run"} | ${rb?.p50_ms != null ? `${rb.p50_ms} ms` : "not run"} |`);
    }
  } else {
    lines.push(`Not comparable: browser runtime benchmark was ${a.runtime.length ? "" : "not run for A"}${!a.runtime.length && !b.runtime.length ? " and " : ""}${b.runtime.length ? "" : "not run for B"}.`);
  }
  lines.push("");
  return lines.join("\n");
}
