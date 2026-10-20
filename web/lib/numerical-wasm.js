// Host-only initialization. Exported numerical functions remain synchronous.
// Browser entrypoints await readiness; Node/SEA uses the same verified bytes
// synchronously. Failure never falls back to a host implementation.
import init from "../../pkg/tflite_wasm_audit.js";
import * as wasm from "../../pkg/tflite_wasm_audit.js";
const ABI = "deepbom.numerical_abi.v1:exact_binary64_v1";
const filename = "tflite_wasm_audit_bg.wasm";
let ready = false, pending = null, digest = null, nodeHost = null;
function handshake() {
  if (wasm.numerical_abi_version() !== ABI) throw new Error("Numerical WASM ABI mismatch; refresh the complete engine package.");
  for (const name of ["numerical_contract", "evidence_sha256", "protobuf_fields", "protobuf_packed_varints", "protobuf_varint", "protobuf_packed_floats"])
    if (typeof wasm[name] !== "function") throw new Error(`Portable WASM feature missing: ${name}; refresh the complete engine package.`);
  for (const name of ["numerical_distribution","numerical_distribution_comparison","numerical_affine","numerical_pruning_prepare","numerical_pruning_simulate","numerical_weight_permute","numerical_shape_count","numerical_count_ratio","numerical_histogram_edges","numerical_quality_metric","numerical_metric_methods","numerical_spectrum","numerical_spectrum_summary","numerical_weight_magnitude","numerical_weight_matrix","numerical_weight_similarity","numerical_weight_tiles","numerical_weight_sparsity","numerical_weight_comparison","numerical_weight_differences","NumericalMoments","NumericalComparison","NumericalStatistics"])
    if (typeof wasm[name] !== "function") throw new Error(`Numerical WASM feature missing: ${name}; refresh the complete engine package.`);
  if (typeof wasm.NumericalComparison.prototype.duplicate !== "function") throw new Error("Numerical WASM comparison ABI is stale.");
  const contracts = JSON.parse(wasm.numerical_contract("features", "{}"));
  for (const name of ["shape", "dimension_expression", "redesign_validate", "evidence_contract", "evidence_identity", "evidence_workflow", "decimal", "native_tensor", "onnx_type", "ir_semantics", "coreml_deployment", "onnx_semantics", "semantic_validation", "conversion_receipt", "provenance"])
    if (!contracts.includes(name)) throw new Error(`Portable WASM contract missing: ${name}; refresh the complete engine package.`);
  ready = true;
  return wasm;
}
export function numericalWasm() {
  if (ready) return wasm;
  if (nodeHost || typeof process !== "undefined" && typeof process.getBuiltinModule === "function") {
    const get = name => nodeHost ? nodeHost[name] : process.getBuiltinModule(name);
    const fs = get("fs"), path = get("path"), url = get("url");
    const roots = [process.env.DEEPBOM_RUNTIME_ASSET_DIR,
      url.fileURLToPath(new URL("../../pkg/", import.meta.url)),
      url.fileURLToPath(new URL("../pkg/", import.meta.url)),
      path.join(path.dirname(process.execPath), "pkg"), path.join(path.dirname(process.execPath), "deepbom-assets")].filter(Boolean);
    const file = roots.map(root => path.join(root, filename)).find(file => fs.existsSync(file));
    if (!file) throw new Error("Required numerical WASM asset is missing.");
    const bytes = fs.readFileSync(file);
    digest = get("crypto").createHash("sha256").update(bytes).digest("hex");
    const expected = typeof __DEEPBOM_TFLITE_WASM_SHA256__ === "string" ? __DEEPBOM_TFLITE_WASM_SHA256__ : null;
    if (expected && digest !== expected) throw new Error("Packaged numerical WASM failed its release SHA-256 check.");
    wasm.initSync({ module: bytes });
    return handshake();
  }
  // The TFLite host may have initialized this shared instance already.
  try { return handshake(); } catch (cause) {
    throw new Error("Numerical WASM is not ready; await initializeNumericalWasm() at the host boundary.", { cause });
  }
}
export async function initializeNumericalWasm() {
  if (ready) return wasm;
  if (typeof process !== "undefined" && typeof process.getBuiltinModule === "function") return numericalWasm();
  if (typeof process !== "undefined" && process.versions?.node) {
    // Node 20 before getBuiltinModule: async host readiness, then sync public APIs.
    // Leave host imports runtime-only so browser bundles do not resolve Node
    // built-ins. This branch is entered only by an actual Node host.
    const importHost = name => import(`node:${name}`);
    const [fs,path,url,crypto] = await Promise.all(["fs","path","url","crypto"].map(importHost));
    nodeHost = {fs,path,url,crypto};
    return numericalWasm();
  }
  pending ||= (async () => {
    const response = await fetch(new URL("../../pkg/tflite_wasm_audit_bg.wasm", import.meta.url));
    if (!response.ok) throw new Error(`Numerical WASM load failed: ${response.status}`);
    const bytes = await response.arrayBuffer();
    digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), x => x.toString(16).padStart(2, "0")).join("");
    await init({ module_or_path: bytes });
    return handshake();
  })().catch(error => { pending = null; throw error; });
  return pending;
}
export function numericalWasmIdentity() {
  numericalWasm();
  if (!digest) throw new Error("Numerical identity requires initialization through the verified common loader.");
  return { abi: ABI, wasm_sha256: digest };
}
