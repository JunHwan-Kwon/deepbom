import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { deriveCurrentArtifactCapabilityRow } from "../web/lib/format-capability-view.js";
import { GGUF_BACKEND_PROFILES, GGUF_BACKEND_SOURCE, GGUF_RUNTIME_INSTRUMENTATION } from "../web/lib/gguf-backend-contract.generated.js";
import {
  buildGgufRuntimeEnvironmentTemplate,
  parseGgmlSchedulerTrace,
  parseGgufRuntimeEnvironmentDocument,
} from "../web/lib/gguf-runtime-environment.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";
import { buildRuntimeCapturePlan } from "../web/lib/runtime-evidence-closure.js";
import { buildRuntimeEvidence } from "../web/lib/report-evidence.js";
import { runtimeEnvironmentMarkdown } from "../web/lib/report-sections.js";

const ARTIFACT_SHA = "a".repeat(64);
const BINARY_SHA = "b".repeat(64);
const CACHE_SHA = "c".repeat(64);
const LOG_SHA = "d".repeat(64);
const analysis = {
  format: "gguf",
  filename: "model.gguf",
  model_sha256: ARTIFACT_SHA,
  gguf: { backend_compatibility: { status: "source_candidate" } },
  tensor_inventory: { status: "assessed" },
  tensor_numerical_integrity: { status: "assessed" },
};

const collectorSource = readFileSync(new URL("./capture-gguf-runtime.mjs", import.meta.url), "utf8");
assert.match(collectorSource, /"--single-turn"/);
assert.match(collectorSource, /"--simple-io"/);
assert.match(collectorSource, /computeGraph\?\.successful_dispatch_count > 0/);

const template = buildGgufRuntimeEnvironmentTemplate(analysis);
assert.equal(template.artifact.sha256, ARTIFACT_SHA);
assert.equal(template.runtime.source_commit, GGUF_BACKEND_SOURCE.source_commit);
assert.deepEqual(Object.keys(template.build.options), GGUF_BACKEND_PROFILES.map((profile) => profile.cmake_option));

const source = structuredClone(template);
source.runtime.binary_sha256 = BINARY_SHA;
source.runtime.version_output = "llama.cpp test";
source.build.cmake_cache_sha256 = CACHE_SHA;
for (const profile of GGUF_BACKEND_PROFILES) source.build.options[profile.cmake_option] = ["cpu", "cuda"].includes(profile.id);
source.build.compiled_backend_profile_ids = ["cpu", "cuda"];
const rawBuildAttestation = {
  schema: "deepbom.llama_cpp_instrumented_build_attestation.v1",
  evidence_class: "REPRODUCIBLE_BUILD_ATTESTATION",
  source: {
    repository: GGUF_BACKEND_SOURCE.repository,
    commit: GGUF_BACKEND_SOURCE.source_commit,
    scheduler_path: GGUF_BACKEND_SOURCE.files.scheduler.path,
    scheduler_original_sha256: GGUF_RUNTIME_INSTRUMENTATION.scheduler_source_original_sha256,
    scheduler_patched_sha256: GGUF_RUNTIME_INSTRUMENTATION.scheduler_source_patched_sha256,
  },
  instrumentation: {
    patch_id: GGUF_RUNTIME_INSTRUMENTATION.patch_id,
    patch_path: GGUF_RUNTIME_INSTRUMENTATION.patch_path,
    patch_sha256: GGUF_RUNTIME_INSTRUMENTATION.patch_sha256,
    trace_protocol: GGUF_RUNTIME_INSTRUMENTATION.trace_protocol,
  },
  build: {
    configure_arguments: ["-DGGML_CPU=ON", "-DGGML_CUDA=ON", "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF"],
    build_arguments: ["--build", "build", "--target", "llama-cli", "--parallel", "2"],
    parallel_jobs: 2,
    timeout_ms: 1_800_000,
    configure_stdout_sha256: "2".repeat(64),
    configure_stderr_sha256: "3".repeat(64),
    build_stdout_sha256: "4".repeat(64),
    build_stderr_sha256: "5".repeat(64),
    cmake_cache_sha256: CACHE_SHA,
    cmake_generator: "Ninja",
    c_compiler: "cc",
    cxx_compiler: "c++",
    build_type: "Release",
  },
  binary: { filename: "llama-cli", sha256: BINARY_SHA, version_output: "llama.cpp test" },
  boundary: "Test attestation.",
};
const buildAttestationFileSha = "6".repeat(64);
source.build.attestation = {
  file_sha256: buildAttestationFileSha,
  canonical_sha256: sha256TextHex(canonicalJson(rawBuildAttestation)),
  document: rawBuildAttestation,
};
source.instrumentation.build_attestation_sha256 = buildAttestationFileSha;
source.selection.context_size = 2048;
source.selection.batch_size = 512;
source.selection.ubatch_size = 128;
source.device.platform = "linux test";
source.device.architecture = "x64";
source.device.hostname_sha256 = "e".repeat(64);
source.capture.capture_id = "capture-test";
source.capture.collected_at = "2026-08-11T00:00:00.000Z";
source.observations.backend_inventory_status = "observed_success";
source.observations.model_load_status = "observed_success";
source.observations.inference_status = "observed_success";
source.observations.elapsed_ms = 123.5;
source.observations.process_exit_code = 0;
source.observations.stdout_sha256 = LOG_SHA;
source.observations.stderr_sha256 = "f".repeat(64);
source.observations.selected_backend_observation = "scheduler_graph_named_backend";

const h = (value) => Buffer.from(value, "utf8").toString("hex");
const refs = (first) => [first, ...Array(9).fill("-")].join(";");
const trace = [
  `DEEPBOM_GGML_TRACE_V1\tgraph\t1\t18446744073709551615\t2\t1\t3\t1\t2`,
  `DEEPBOM_GGML_TRACE_V1\tbackend\t1\t0\t${h("CPU")}\t${h("CPU")}`,
  `DEEPBOM_GGML_TRACE_V1\tbackend\t1\t1\t${h("CUDA0")}\t${h("CUDA device")}`,
  `DEEPBOM_GGML_TRACE_V1\tsplit\t1\t0\t0\t1\t0\t${h("CPU")}\t`,
  `DEEPBOM_GGML_TRACE_V1\tsplit\t1\t1\t1\t2\t1\t${h("CUDA0")}\tN0`,
  `DEEPBOM_GGML_TRACE_V1\toriginal_node\t1\t0\t0\t${h("CPU")}\t${h("MUL_MAT")}\t${h("F32")}\t${h("node0")}\t64\t4,4,1,1\t0\t${refs("L0")}\t-`,
  `DEEPBOM_GGML_TRACE_V1\toriginal_node\t1\t1\t1\t${h("CUDA0")}\t${h("ADD")}\t${h("F32")}\t${h("node1")}\t64\t4,4,1,1\t0\t${refs("N0")}\t-`,
  `DEEPBOM_GGML_TRACE_V1\toriginal_leaf\t1\t0\t-1\t${h("CPU")}\t${h("NONE")}\t${h("F32")}\t${h("input")}\t64\t4,4,1,1\t1\t${refs("-")}\t-`,
  `DEEPBOM_GGML_TRACE_V1\tscheduled_node\t1\t0\t-1\t${h("CPU")}\t${h("MUL_MAT")}\t${h("F32")}\t${h("node0")}\t64\t4,4,1,1\t0\t${refs("L0")}\t-`,
  `DEEPBOM_GGML_TRACE_V1\tscheduled_node\t1\t1\t-1\t${h("CUDA0")}\t${h("CPY")}\t${h("F32")}\t${h("copy")}\t64\t4,4,1,1\t0\t${refs("N0")}\t-`,
  `DEEPBOM_GGML_TRACE_V1\tscheduled_node\t1\t2\t-1\t${h("CUDA0")}\t${h("ADD")}\t${h("F32")}\t${h("node1")}\t64\t4,4,1,1\t0\t${refs("N1")}\t-`,
  `DEEPBOM_GGML_TRACE_V1\tscheduled_leaf\t1\t0\t-1\t${h("CPU")}\t${h("NONE")}\t${h("F32")}\t${h("input")}\t64\t4,4,1,1\t1\t${refs("-")}\t-`,
  `DEEPBOM_GGML_TRACE_V1\tgraph_end\t1`,
  `DEEPBOM_GGML_TRACE_V1\tdispatch\t1\t0`,
  "",
].join("\n");
source.compute_graph = parseGgmlSchedulerTrace(trace);
assert.equal(source.compute_graph.graph_count, 1);
assert.equal(source.compute_graph.scheduler_inserted_node_count, 1);
assert.equal(source.compute_graph.split_count, 2);
assert.equal(source.compute_graph.original_backend_transition_edge_count, 1);
assert.equal(source.compute_graph.scheduled_backend_transition_edge_count, 1);
assert.equal(source.compute_graph.dispatched_graph_count, 1);

const parsed = parseGgufRuntimeEnvironmentDocument(source, analysis, { fileSha256: "1".repeat(64) });
assert.equal(parsed.runtime.source_alignment, "exact_pinned_source_match");
assert.equal(parsed.runtime_identity_status, "bound");
assert.equal(parsed.graph_assignment_status, "observed_generated_scheduler_graphs");
assert.equal(parsed.compatibility_conclusion, "runtime_smoke_execution_observed_for_declared_configuration");
assert.equal(parsed.selection.selection_evidence_class, "DECLARED_REQUIREMENT");
assert.equal(parsed.build.attestation.document.build.parallel_jobs, 2);
assert.equal(parsed.build.attestation.document.build.timeout_ms, 1_800_000);
assert.match(parsed.normalized_manifest_sha256, /^[a-f0-9]{64}$/);

const postLoadFailure = structuredClone(source);
postLoadFailure.observations.inference_status = "observed_failure";
postLoadFailure.observations.process_exit_code = 130;
const parsedPostLoadFailure = parseGgufRuntimeEnvironmentDocument(postLoadFailure, analysis);
assert.equal(parsedPostLoadFailure.observations.model_load_status, "observed_success");
assert.equal(parsedPostLoadFailure.compatibility_conclusion, "runtime_model_load_observed_inference_not_established");

const runtime = buildRuntimeEvidence({ analysis, runtimeAssignmentEvidence: parsed });
assert.equal(runtime.runtime_assignment, null);
assert.equal(runtime.runtime_assignment_comparison, null);
assert.equal(runtime.runtime_environment.runtime.binary_sha256, BINARY_SHA);
assert.equal(runtime.assessments.runtime_environment.status, "assessed");
assert.equal(runtime.assessments.runtime_assignment.status, "not_assessed");

const markdown = runtimeEnvironmentMarkdown({ runtimeAssignmentEvidence: parsed });
assert.match(markdown, /GGUF Instrumented Runtime And Scheduler Graph/);
assert.match(markdown, new RegExp(BINARY_SHA));
assert.match(markdown, /scheduler graph/i);
assert.match(markdown, /2 build job\(s\); timeout 1800000 ms/);

const row = deriveCurrentArtifactCapabilityRow("gguf", analysis, parsed);
assert.equal(row.cells[1].id, "generated_observed");
assert.match(row.cells[1].title, /1 generated scheduler graph/);
assert.match(row.cells[1].title, /1\/1 successful dispatch/);
assert.equal(row.cells[4].id, "build_bound");

const plan = buildRuntimeCapturePlan(analysis);
assert.equal(plan.schema, "deepbom.runtime_capture_plan.v1.1");
assert.match(plan.command, /capture:gguf-runtime/);
assert.match(plan.command, new RegExp(GGUF_BACKEND_SOURCE.source_commit));
assert.equal(plan.import_contract.schema, "deepbom.gguf_runtime_environment.v2");

assert.throws(() => parseGgufRuntimeEnvironmentDocument({ ...source, artifact: { ...source.artifact, sha256: "0".repeat(64) } }, analysis), /active artifact/);
assert.throws(() => parseGgufRuntimeEnvironmentDocument({ ...source, build: { ...source.build, compiled_backend_profile_ids: [] } }, analysis), /missing backend profile cpu/);
assert.throws(() => parseGgufRuntimeEnvironmentDocument({ ...source, selection: { ...source.selection, requested_backend_profile_id: "vulkan" } }, analysis), /not present in the compiled backend inventory/);
assert.throws(() => parseGgufRuntimeEnvironmentDocument({ ...source, observations: { ...source.observations, stdout_sha256: null } }, analysis), /stdout and stderr/);
assert.throws(() => parseGgufRuntimeEnvironmentDocument({ ...source, selection: { ...source.selection, ubatch_size: 1024 } }, analysis), /cannot exceed batch/);
assert.throws(() => parseGgufRuntimeEnvironmentDocument({ ...source, compute_graph: { ...source.compute_graph, split_count: 99 } }, analysis), /does not independently reconstruct/);
assert.throws(() => parseGgufRuntimeEnvironmentDocument({ ...source, build: { ...source.build, attestation: { ...source.build.attestation, file_sha256: "7".repeat(64) } } }, analysis), /instrumentation binding/);
const tamperedBuildParallelism = structuredClone(source);
tamperedBuildParallelism.build.attestation.document.build.parallel_jobs = 3;
assert.throws(() => parseGgufRuntimeEnvironmentDocument(tamperedBuildParallelism, analysis), /canonical SHA-256 does not reconstruct/);
assert.throws(() => parseGgmlSchedulerTrace(trace.replace("\tsplit\t1\t0\t0\t1\t0\t", "\tsplit\t1\t0\t0\t3\t0\t")), /split ledger|partition/);

console.log("GGUF runtime environment checks passed (selected build, scheduler graph, split, backend transition, dispatch, and tamper binding).");
