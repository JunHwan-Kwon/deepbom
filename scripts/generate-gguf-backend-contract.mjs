import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const OUTPUT = path.join(ROOT, "web", "lib", "gguf-backend-contract.generated.js");
const TRACE_PATCH = path.join(ROOT, "native", "llama-cpp", "deepbom-ggml-scheduler-trace.patch");
const CHECK = process.argv.includes("--check");
const SOURCE = Object.freeze({
  repository: "ggml-org/llama.cpp",
  source_commit: "7bd8282c37fcd9c4d7236106d664761a23318f18",
  files: Object.freeze({
    architecture_registry: Object.freeze({ path: "src/llama-arch.cpp", sha256: "1f35546f88f825289efb30c779ec9f71d41dfad74c35b898260d3510a86c997c" }),
    build_options: Object.freeze({ path: "ggml/CMakeLists.txt", sha256: "1858523f1703dc31505447a3e162de5f56b6cc5325f9eaad2e0cc8688769737f" }),
    backend_build_rules: Object.freeze({ path: "ggml/src/CMakeLists.txt", sha256: "b4210d3aded4f4d217209624f981ecb5c65b1592eca7e049ff14b13c60743cb5" }),
    hip_build_rules: Object.freeze({ path: "ggml/src/ggml-hip/CMakeLists.txt", sha256: "a9761856c498862828f88d791643d15dea65ec58d189d2d21a1891f3a506bb30" }),
    backend_registration: Object.freeze({ path: "ggml/src/ggml-backend-reg.cpp", sha256: "c0a61f47e7af0359cfff41317ae8e68717960483d141983bba914733b1f68906" }),
    scheduler: Object.freeze({ path: "ggml/src/ggml-backend.cpp", sha256: "507577061d22e673a4f002bb215c65aa9c63b88a4e6d1586f52c74ca0b98fa07" }),
  }),
});
const TRACE_INSTRUMENTATION = Object.freeze({
  patch_id: "deepbom.ggml_scheduler_trace.v1",
  trace_protocol: "DEEPBOM_GGML_TRACE_V1",
  patch_path: "native/llama-cpp/deepbom-ggml-scheduler-trace.patch",
  patch_sha256: "156f8278172c596657c221ff4937585bcf21a7bc3f08fdc29bf43447b955770f",
  scheduler_source_original_sha256: SOURCE.files.scheduler.sha256,
  scheduler_source_patched_sha256: "e027f8876ee9dd04a0b2f0c992b89709ed1a5174011aa5046b488e8aba6af844",
});
const PROFILE_DEFINITIONS = Object.freeze([
  ["cpu", "CPU", "GGML_CPU", "GGML_USE_CPU", "ggml_backend_cpu_reg"],
  ["cuda", "CUDA", "GGML_CUDA", "GGML_USE_CUDA", "ggml_backend_cuda_reg"],
  ["hip", "HIP", "GGML_HIP", "GGML_USE_CUDA", "ggml_backend_cuda_reg"],
  ["metal", "Metal", "GGML_METAL", "GGML_USE_METAL", "ggml_backend_metal_reg"],
  ["vulkan", "Vulkan", "GGML_VULKAN", "GGML_USE_VULKAN", "ggml_backend_vk_reg"],
  ["sycl", "SYCL", "GGML_SYCL", "GGML_USE_SYCL", "ggml_backend_sycl_reg"],
  ["webgpu", "WebGPU", "GGML_WEBGPU", "GGML_USE_WEBGPU", "ggml_backend_webgpu_reg"],
  ["blas", "BLAS", "GGML_BLAS", "GGML_USE_BLAS", "ggml_backend_blas_reg"],
  ["openvino", "OpenVINO", "GGML_OPENVINO", "GGML_USE_OPENVINO", "ggml_backend_openvino_reg"],
]);

const fetched = {};
for (const [id, row] of Object.entries(SOURCE.files)) {
  const url = `https://raw.githubusercontent.com/${SOURCE.repository}/${SOURCE.source_commit}/${row.path}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${id}: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  const actual = createHash("sha256").update(bytes).digest("hex");
  if (actual !== row.sha256) throw new Error(`${id}: expected ${row.sha256}, received ${actual}`);
  fetched[id] = bytes.toString("utf8");
}
const patchBytes = await readFile(TRACE_PATCH);
const patchSha256 = createHash("sha256").update(patchBytes).digest("hex");
if (patchSha256 !== TRACE_INSTRUMENTATION.patch_sha256) throw new Error(`GGML scheduler trace patch digest drifted: ${patchSha256}`);

const architectureBody = fetched.architecture_registry.match(/LLM_ARCH_NAMES\s*=\s*\{([\s\S]*?)\n\};/)?.[1];
if (!architectureBody) throw new Error("Pinned llama.cpp architecture registry was not found");
const architectures = [...architectureBody.matchAll(/\{\s*LLM_ARCH_[^,]+,\s*"([^"]+)"\s*\}/g)]
  .map((match) => match[1])
  .filter((name) => name !== "(unknown)");
if (architectures.length < 100 || new Set(architectures).size !== architectures.length) throw new Error(`Architecture registry extraction is invalid (${architectures.length} rows)`);

const profiles = PROFILE_DEFINITIONS.map(([id, label, option, macro, registration]) => {
  const optionMatch = fetched.build_options.match(new RegExp(`option\\(${option}\\s+"[^"]*"\\s+([^\\)]+)\\)`));
  if (!optionMatch) throw new Error(`${option} build option was not found`);
  const registrationPattern = new RegExp(`#ifdef\\s+${macro}[\\s\\S]{0,400}?${registration}\\(\\)`);
  if (!registrationPattern.test(fetched.backend_registration)) throw new Error(`${macro}/${registration} registration was not found`);
  const backendName = id === "openvino" ? "OPENVINO" : id === "webgpu" ? "WebGPU" : id === "vulkan" ? "Vulkan" : option.slice("GGML_".length);
  if (!fetched.backend_build_rules.includes(`ggml_add_backend(${backendName})`)) throw new Error(`${option} backend build rule was not found`);
  if (id === "hip" && !fetched.hip_build_rules.includes("target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)")) throw new Error("HIP-to-CUDA registration alias was not found");
  return {
    id,
    label,
    cmake_option: option,
    compiled_registration_macro: macro,
    registration_function: registration,
    declared_default: optionMatch[1].trim(),
    runtime_disable_condition: id === "vulkan" ? "GGML_DISABLE_VULKAN environment variable must be absent" : null,
  };
});

const generated = `// @generated by scripts/generate-gguf-backend-contract.mjs. Do not edit manually.\n\n`
  + `export const GGUF_BACKEND_SOURCE = Object.freeze(${JSON.stringify({ ...SOURCE, files: SOURCE.files }, null, 2)});\n\n`
  + `export const GGUF_LLAMA_ARCHITECTURES = Object.freeze(${JSON.stringify(architectures, null, 2)});\n\n`
  + `export const GGUF_BACKEND_PROFILES = Object.freeze(${JSON.stringify(profiles, null, 2)});\n\n`
  + `export const GGUF_RUNTIME_INSTRUMENTATION = Object.freeze(${JSON.stringify(TRACE_INSTRUMENTATION, null, 2)});\n`;

if (CHECK) {
  const existing = await readFile(OUTPUT, "utf8").catch(() => "");
  if (existing !== generated) throw new Error(`${path.relative(ROOT, OUTPUT)} is stale; run npm run generate:gguf-backend-contract`);
  console.log(`GGUF backend contract verified (${architectures.length} architectures, ${profiles.length} backend profiles).`);
} else {
  await writeFile(OUTPUT, generated, "utf8");
  console.log(`GGUF backend contract generated (${architectures.length} architectures, ${profiles.length} backend profiles).`);
}
