import { createHash } from "node:crypto";
import { cp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const TF_COMMIT = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const XNN_COMMIT = "23a67314f7afdbb76191589ae090d82bf55afbfa";
const EXPECTED_XNN_MICROKERNEL_SYMBOLS = 9643;
const TF_ARCHIVE_SHA256 = "081edc42742db04d154f1d793816d384469dabe1ec95696de7bd866b3f0902c5";
const XNN_ARCHIVE_SHA256 = "b1ac2fcb6ed85623430a4ac05ddb08432e3ca87ccf77596ea2b4bc7d5ebad00a";
const EXPECTED = Object.freeze({
  "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.cc": "d9a1b2b5b28dd67c2c81e73671470da0c720b1774d48d41832ed7550083f93ad",
  "tensorflow/lite/arena_planner.cc": "46a64d943e04f4052a5aaf36feb2a803ff2ade5e9d0edff358d6687c5c712c51",
  "tensorflow/lite/kernels/internal/reference/logistic.h": "dfb750ca298dbf1580147d7d74ce6f77b3fea1c6cf8ffa24a4f513d6f691f5e8",
  "tensorflow/lite/simple_memory_arena.cc": "23cef717a556743db6b550e436ffec1b9f688daa0441e22fefea214a58e9cbc0",
  "third_party/xla/xla/tsl/tsl.bzl": "98f5dc08c4de148b94e202ef3d09066939ce6dfeafa3f1eae9cf4a98e9b6e1d7",
  "include/xnnpack.h": "8ca683fbcd3996bdee68d7e22654e0d6dd0b1229d7124b3dc8d934a752f97b1c",
  "src/xnnpack/subgraph.h": "7e845b54eb40fb9f2da2a89deeff82a096a159ae715b9733e6e8cd454b50a22e",
  "src/runtime.c": "287dc88ee2834698fc5c3889be0b25e5c7767422335c4e17e52c8ebf0f9da9c8",
  "src/operator-run.c": "d272c0b8d8073defea80d4a2b8bd423d66b042b6ec0041bcd028e7c2db75aec7",
  "BUILD.bazel": "48cf91aba9921cd48fe9d51ca54e1f0903bc0a9100ceb562c620b8f9a655ff41",
  "CMakeLists.txt": "27d794fce83e0f82ed1f6543b1937b44800874a88f0b8bd05bddc1440c6972e7",
});

const tensorflowRoot = requiredPath("--tensorflow");
const xnnpackRoot = requiredPath("--xnnpack");
const tensorflowArchive = optionalPath("--tensorflow-archive");
const xnnpackArchive = optionalPath("--xnnpack-archive");
const manifestPath = path.resolve(argument("--manifest") || path.join(xnnpackRoot, "deepbom-instrumentation-manifest.json"));

const sourceVerification = await verifySourceIdentity();
await verifyPinnedFiles();

const catalog = await buildSymbolCatalog(xnnpackRoot);
const originalFiles = await snapshotPatchTargets();
try {
  await installInstrumentationSources(catalog);
  const patchCounts = {
    tensorflow_delegate: await patchTensorflowDelegate(),
    tensorflow_windows_toolchain_compatibility: await patchTensorflowWindowsToolchainCompatibility(),
    tensorflow_reference_logistic_msvc_compatibility: await patchTensorflowReferenceLogisticCompatibility(),
    tensorflow_arena_debug_msvc_compatibility: await patchTensorflowArenaDebugMsvcCompatibility(),
    tensorflow_arena_memory_instrumentation: await patchTensorflowArenaMemoryInstrumentation(),
    xnnpack_public_header: await patchXnnpackPublicHeader(),
    xnnpack_provenance_structs: await patchXnnpackProvenanceStructs(),
    xnnpack_runtime: await patchXnnpackRuntime(),
    xnnpack_dispatch_sites: await patchXnnpackOperatorRun(),
    xnnpack_bazel: await patchXnnpackBazel(),
    xnnpack_cmake: await patchXnnpackCmake(),
  };
  const files = await patchedFileIdentities();
  const manifest = {
    schema: "deepbom.instrumented_runtime_patch.v1.3",
    tensorflow_commit: TF_COMMIT,
    xnnpack_commit: XNN_COMMIT,
    source_verification: sourceVerification,
    attribution_mode: {
      environment: "DEEPBOM_XNN_NO_OPERATOR_FUSION=1",
      reason: "Disables XNN operator fusion only for attribution captures so each surviving runtime operator remains bound to one original TFLite op. The changed runtime option is evidence-bound and is not presented as production-default performance.",
    },
    symbol_resolution: {
      catalog_entries: catalog.length,
      method: "Actual dispatch function pointers are matched to the pinned catalog by linked weak/dynamic symbols on ELF/Mach-O or exact PDB symbol names from DbgHelp on MSVC; unresolved or non-exact pointers are not promoted to microkernel evidence",
      linker_requirement_linux: "--linkopt=-Wl,--export-dynamic provides the dynamic-symbol fallback; linked weak references resolve hidden/static-linked microkernels without relying on it",
      linker_requirement_windows: "The runtime must retain its matching PDB; SymFromAddr results require zero displacement and an exact name in the pinned XNNPACK catalog",
    },
    patch_counts: patchCounts,
    files,
  };
  await writeFile(manifestPath, stableJson(manifest));
  console.log(`Prepared pinned instrumented runtime patch: ${manifestPath}`);
  console.log(`dispatch_call_sites=${patchCounts.xnnpack_dispatch_sites}, symbol_catalog=${catalog.length}`);
} catch (error) {
  await rollbackPatchTargets(originalFiles);
  throw error;
}

async function verifyPinnedFiles() {
  for (const [relative, expected] of Object.entries(EXPECTED)) {
    const root = relative.startsWith("tensorflow/") || relative.startsWith("third_party/xla/") ? tensorflowRoot : xnnpackRoot;
    const actual = sha256(await readFile(path.join(root, relative)));
    if (actual !== expected) throw new Error(`${relative} SHA-256 mismatch: expected ${expected}, received ${actual}`);
  }
}

async function snapshotPatchTargets() {
  const originals = new Map();
  for (const file of patchTargetPaths()) originals.set(file, await readFile(file));
  const previousManifest = await readFile(manifestPath).catch(() => null);
  return { originals, previousManifest };
}

async function rollbackPatchTargets(snapshot) {
  for (const [file, bytes] of snapshot.originals) await writeFile(file, bytes);
  for (const file of generatedInstrumentationPaths()) await rm(file, { force: true });
  if (snapshot.previousManifest) await writeFile(manifestPath, snapshot.previousManifest);
  else await rm(manifestPath, { force: true });
}

function patchTargetPaths() {
  return [
    path.join(tensorflowRoot, "tensorflow", "lite", "delegates", "xnnpack", "xnnpack_delegate.cc"),
    path.join(tensorflowRoot, "tensorflow", "lite", "arena_planner.cc"),
    path.join(tensorflowRoot, "tensorflow", "lite", "simple_memory_arena.cc"),
    path.join(tensorflowRoot, "tensorflow", "lite", "kernels", "internal", "reference", "logistic.h"),
    path.join(tensorflowRoot, "third_party", "xla", "xla", "tsl", "tsl.bzl"),
    ...["include/xnnpack.h", "src/xnnpack/subgraph.h", "src/runtime.c", "src/operator-run.c", "BUILD.bazel", "CMakeLists.txt"]
      .map((file) => path.join(xnnpackRoot, file)),
  ];
}

function generatedInstrumentationPaths() {
  return [
    "src/xnnpack/deepbom.h",
    "src/deepbom.cc",
    "src/deepbom_runtime_trace.h",
    "src/deepbom_runtime_trace.cc",
    "src/deepbom-symbol-catalog.h",
  ].map((file) => path.join(xnnpackRoot, file));
}

async function verifySourceIdentity() {
  if (Boolean(tensorflowArchive) !== Boolean(xnnpackArchive)) {
    throw new Error("--tensorflow-archive and --xnnpack-archive must be provided together");
  }
  if (!tensorflowArchive) {
    await requireCommit(tensorflowRoot, TF_COMMIT, "TensorFlow");
    await requireCommit(xnnpackRoot, XNN_COMMIT, "XNNPACK");
    return { mode: "git_commit", tensorflow: TF_COMMIT, xnnpack: XNN_COMMIT };
  }
  const tensorflowSha256 = sha256(await readFile(tensorflowArchive));
  const xnnpackSha256 = sha256(await readFile(xnnpackArchive));
  if (tensorflowSha256 !== TF_ARCHIVE_SHA256) {
    throw new Error(`TensorFlow archive SHA-256 mismatch: expected ${TF_ARCHIVE_SHA256}, received ${tensorflowSha256}`);
  }
  if (xnnpackSha256 !== XNN_ARCHIVE_SHA256) {
    throw new Error(`XNNPACK archive SHA-256 mismatch: expected ${XNN_ARCHIVE_SHA256}, received ${xnnpackSha256}`);
  }
  return {
    mode: "pinned_archive_and_file_hashes",
    tensorflow_archive_sha256: tensorflowSha256,
    xnnpack_archive_sha256: xnnpackSha256,
  };
}

async function installInstrumentationSources(catalog) {
  const template = path.join(ROOT, "native", "instrumented_runtime", "xnnpack");
  await cp(path.join(template, "deepbom.h"), path.join(xnnpackRoot, "src", "xnnpack", "deepbom.h"));
  await cp(path.join(template, "deepbom.cc"), path.join(xnnpackRoot, "src", "deepbom.cc"));
  await cp(path.join(ROOT, "native", "instrumentation", "deepbom_runtime_trace.h"), path.join(xnnpackRoot, "src", "deepbom_runtime_trace.h"));
  await cp(path.join(ROOT, "native", "instrumentation", "deepbom_runtime_trace.cc"), path.join(xnnpackRoot, "src", "deepbom_runtime_trace.cc"));
  const rows = catalog.map((entry) => `  {${cString(entry.symbol)}, ${cString(entry.source_path)}},`).join("\n");
  const declarations = catalog.map((entry) => `void ${entry.symbol}(void) DEEPBOM_XNN_WEAK;`).join("\n");
  const addressedRows = catalog.map((entry) => `  {${cString(entry.symbol)}, ${cString(entry.source_path)}, reinterpret_cast<uintptr_t>(&${entry.symbol})},`).join("\n");
  const unresolvedRows = rows.replaceAll("},", ", 0},");
  await writeFile(path.join(xnnpackRoot, "src", "deepbom-symbol-catalog.h"), `#ifndef XNNPACK_SRC_DEEPBOM_SYMBOL_CATALOG_H_\n#define XNNPACK_SRC_DEEPBOM_SYMBOL_CATALOG_H_\n#include <stddef.h>\n#include <stdint.h>\n#if defined(__GNUC__) || defined(__clang__)\n#define DEEPBOM_XNN_WEAK __attribute__((weak))\nextern \"C\" {\n${declarations}\n}\n#else\n#define DEEPBOM_XNN_WEAK\n#endif\nstruct DeepBomXnnSymbolCatalogEntry { const char* symbol; const char* source_path; uintptr_t address; };\nstatic const DeepBomXnnSymbolCatalogEntry deepbom_xnn_symbol_catalog[] = {\n#if defined(__GNUC__) || defined(__clang__)\n${addressedRows}\n#else\n${unresolvedRows}\n#endif\n};\nstatic const size_t deepbom_xnn_symbol_catalog_size = sizeof(deepbom_xnn_symbol_catalog) / sizeof(deepbom_xnn_symbol_catalog[0]);\n#endif\n`);
}

async function patchTensorflowDelegate() {
  const file = path.join(tensorflowRoot, "tensorflow", "lite", "delegates", "xnnpack", "xnnpack_delegate.cc");
  let source = await readFile(file, "utf8");
  source = replaceExact(source,
`      if (VisitNode(subgraph.get(), delegate, context, registration, node,
                    node_index, quasi_static_tensors,
                    tflite_tensor_to_xnnpack) != kTfLiteOk) {
        return nullptr;
      }`,
`      const size_t deepbom_first_xnn_node =
          xnn_deepbom_subgraph_node_count(subgraph.get());
      if (VisitNode(subgraph.get(), delegate, context, registration, node,
                    node_index, quasi_static_tensors,
                    tflite_tensor_to_xnnpack) != kTfLiteOk) {
        return nullptr;
      }
      xnn_deepbom_tag_subgraph_nodes(
          subgraph.get(), deepbom_first_xnn_node,
          static_cast<uint32_t>(node_index),
          EnumNameBuiltinOperator(
              static_cast<BuiltinOperator>(registration->builtin_code)));`);
  source = replaceExact(source,
`    flags |= delegate.runtime_flags();`,
`    flags |= delegate.runtime_flags();
    const char* deepbom_no_fusion = std::getenv("DEEPBOM_XNN_NO_OPERATOR_FUSION");
    if (deepbom_no_fusion != nullptr && std::strcmp(deepbom_no_fusion, "1") == 0) {
      flags |= xnn_deepbom_no_operator_fusion_flag();
    }`);
  source = replaceExact(source,
`  return nodes_to_delegate;
}

void* SubgraphInit`,
`  if (xnn_deepbom_trace_open() == xnn_status_success) {
    for (int i = 0; i < execution_plan->size; ++i) {
      const int node_index = execution_plan->data[i];
      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        continue;
      }
      const bool delegated = std::binary_search(
          nodes_to_delegate->data,
          nodes_to_delegate->data + nodes_to_delegate->size, node_index);
      xnn_deepbom_trace_placement(
          static_cast<uint32_t>(node_index),
          EnumNameBuiltinOperator(
              static_cast<BuiltinOperator>(registration->builtin_code)),
          delegated, nullptr);
    }
  }
  return nodes_to_delegate;
}

void* SubgraphInit`);
  await writeFile(file, source);
  return 3;
}

async function patchTensorflowWindowsToolchainCompatibility() {
  const file = path.join(tensorflowRoot, "third_party", "xla", "xla", "tsl", "tsl.bzl");
  let source = await readFile(file, "utf8");
  source = replaceExact(source,
`        if_not_windows([
            "-DEIGEN_AVOID_STL_ARRAY",
            "-Iexternal/gemmlowp",
            "-Wno-sign-compare",
            "-ftemplate-depth=900",
        ]) +`,
`        if_not_windows([
            "-DEIGEN_AVOID_STL_ARRAY",
            "-Iexternal/gemmlowp",
            "-ftemplate-depth=900",
        ]) +`);
  source = replaceExact(source,
`            clean_dep("//xla/tsl:windows"): get_win_copts(is_external, is_msvc = False),`,
`            clean_dep("//xla/tsl:windows"): get_win_copts(is_external, is_msvc = True),`);
  await writeFile(file, source);
  return 2;
}

async function patchTensorflowReferenceLogisticCompatibility() {
  const file = path.join(tensorflowRoot, "tensorflow", "lite", "kernels", "internal", "reference", "logistic.h");
  let source = await readFile(file, "utf8");
  source = replaceExact(source,
`  for (int i = 0; i < flat_size; i++) {
    T val = input_data[i];
    float result;`,
`  for (int i = 0; i < flat_size; i++) {
    // The approximation is evaluated in float below. Convert once so MSVC
    // does not ambiguously compare Eigen::half with the float cutoffs.
    const float val = static_cast<float>(input_data[i]);
    float result;`);
  await writeFile(file, source);
  return 1;
}

async function patchTensorflowArenaDebugMsvcCompatibility() {
  const file = path.join(tensorflowRoot, "tensorflow", "lite", "simple_memory_arena.cc");
  let source = await readFile(file, "utf8");
  source = replaceExact(source,
`// Using weak symbols to create a pluggable debugging module.
TFLITE_ATTRIBUTE_WEAK void DumpArenaInfo(
    const std::string& name, const std::vector<int>& execution_plan,
    size_t arena_size, const std::vector<ArenaAllocWithUsageInterval>& allocs) {
}`,
`// MSVC does not provide the weak override used by this debug hook. The
// benchmark target links the strong simple_memory_arena_debug_dump definition.
#if defined(_WIN32) && defined(DEEPBOM_RUNTIME_INSTRUMENTATION)
void DumpArenaInfo(
    const std::string& name, const std::vector<int>& execution_plan,
    size_t arena_size, const std::vector<ArenaAllocWithUsageInterval>& allocs);
#else
// Using weak symbols to create a pluggable debugging module.
TFLITE_ATTRIBUTE_WEAK void DumpArenaInfo(
    const std::string& name, const std::vector<int>& execution_plan,
    size_t arena_size, const std::vector<ArenaAllocWithUsageInterval>& allocs) {
}
#endif`);
  await writeFile(file, source);
  return 1;
}

async function patchTensorflowArenaMemoryInstrumentation() {
  const file = path.join(tensorflowRoot, "tensorflow", "lite", "arena_planner.cc");
  let source = await readFile(file, "utf8");
  source = replaceExact(source,
`#include "tensorflow/lite/simple_memory_arena.h"

namespace tflite {`,
`#include "tensorflow/lite/simple_memory_arena.h"

#if defined(DEEPBOM_RUNTIME_INSTRUMENTATION)
extern "C" uint64_t deepbom_runtime_trace_begin_memory_snapshot(
    size_t non_persistent_arena_bytes, size_t persistent_arena_bytes,
    size_t tensor_count, size_t execution_node_count, size_t allocation_count,
    size_t alias_count);
extern "C" int deepbom_runtime_trace_emit_memory_allocation(
    uint64_t snapshot_id, size_t tensor_index, const char* arena,
    size_t offset_bytes, size_t size_bytes, int32_t first_node,
    int32_t last_node);
extern "C" int deepbom_runtime_trace_emit_memory_alias(
    uint64_t snapshot_id, size_t tensor_index,
    size_t shared_with_tensor_index);
#endif

namespace tflite {`);
  source = replaceExact(source,
`  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::ReleaseNonPersistentMemory() {`,
`#if defined(DEEPBOM_RUNTIME_INSTRUMENTATION)
  size_t deepbom_allocation_count = 0;
  for (size_t tensor_index = 0; tensor_index < allocs_.size(); ++tensor_index) {
    const TfLiteAllocationType allocation_type =
        tensors[tensor_index].allocation_type;
    if (allocs_[tensor_index].size > 0 &&
        (allocation_type == kTfLiteArenaRw ||
         allocation_type == kTfLiteArenaRwPersistent)) {
      ++deepbom_allocation_count;
    }
  }
  const uint64_t deepbom_snapshot_id =
      deepbom_runtime_trace_begin_memory_snapshot(
          arena_.GetBufferSize(), persistent_arena_.GetBufferSize(),
          num_tensors, num_execution_nodes, deepbom_allocation_count,
          actual_tensor_id_.size());
  TF_LITE_ENSURE(context_,
                 deepbom_snapshot_id != std::numeric_limits<uint64_t>::max());
  for (size_t tensor_index = 0; tensor_index < allocs_.size(); ++tensor_index) {
    const TfLiteAllocationType allocation_type =
        tensors[tensor_index].allocation_type;
    if (allocs_[tensor_index].size == 0 ||
        (allocation_type != kTfLiteArenaRw &&
         allocation_type != kTfLiteArenaRwPersistent)) {
      continue;
    }
    const char* arena_name = allocation_type == kTfLiteArenaRw
        ? "kTfLiteArenaRw"
        : "kTfLiteArenaRwPersistent";
    TF_LITE_ENSURE(
        context_, deepbom_runtime_trace_emit_memory_allocation(
                      deepbom_snapshot_id, tensor_index, arena_name,
                      allocs_[tensor_index].offset, allocs_[tensor_index].size,
                      allocs_[tensor_index].first_node,
                      allocs_[tensor_index].last_node) == 0);
  }
  for (const auto& alias : actual_tensor_id_) {
    TF_LITE_ENSURE(
        context_, deepbom_runtime_trace_emit_memory_alias(
                      deepbom_snapshot_id, alias.first, alias.second) == 0);
  }
#endif

  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::ReleaseNonPersistentMemory() {`);
  await writeFile(file, source);
  return 2;
}

async function patchXnnpackPublicHeader() {
  const file = path.join(xnnpackRoot, "include", "xnnpack.h");
  let source = await readFile(file, "utf8");
  source = replaceExact(source, `/// Create a Runtime object from a subgraph.`,
`// DeepBOM instrumented-build API. These functions are absent from unpatched
// XNNPACK and are bound to the pinned source/build identity by the collector.
enum xnn_status xnn_deepbom_trace_open(void);
uint32_t xnn_deepbom_no_operator_fusion_flag(void);
void xnn_deepbom_trace_placement(uint32_t op_index, const char* op_name,
  bool delegated, const char* partition_id);
size_t xnn_deepbom_subgraph_node_count(xnn_subgraph_t subgraph);
void xnn_deepbom_tag_subgraph_nodes(xnn_subgraph_t subgraph,
  size_t first_node_id, uint32_t op_index, const char* op_name);

/// Create a Runtime object from a subgraph.`);
  await writeFile(file, source);
  return 1;
}

async function patchXnnpackProvenanceStructs() {
  const file = path.join(xnnpackRoot, "src", "xnnpack", "subgraph.h");
  let source = await readFile(file, "utf8");
  source = replaceExact(source, `  uint32_t id;
  /// Static parameters of the operator node.`,
`  uint32_t id;
  uint32_t deepbom_source_op_index;
  bool deepbom_source_valid;
  char deepbom_source_op_name[64];
  /// Static parameters of the operator node.`);
  source = replaceExact(source, `struct xnn_operator_data {
  enum xnn_node_type type;
  uint32_t id;`,
`struct xnn_operator_data {
  enum xnn_node_type type;
  uint32_t id;
  uint32_t deepbom_source_op_index;
  bool deepbom_source_valid;
  char deepbom_source_op_name[64];`);
  await writeFile(file, source);
  return 2;
}

async function patchXnnpackRuntime() {
  const file = path.join(xnnpackRoot, "src", "runtime.c");
  let source = await readFile(file, "utf8");
  source = insertAfterIncludeBlock(source, `#include "src/xnnpack/deepbom.h"`);
  source = replaceExact(source, `    runtime->opdata[i].id = node->id;
    runtime->opdata[i].num_inputs = node->num_inputs;`,
`    runtime->opdata[i].id = node->id;
    runtime->opdata[i].deepbom_source_op_index = node->deepbom_source_op_index;
    runtime->opdata[i].deepbom_source_valid = node->deepbom_source_valid;
    memcpy(runtime->opdata[i].deepbom_source_op_name,
           node->deepbom_source_op_name,
           sizeof(runtime->opdata[i].deepbom_source_op_name));
    runtime->opdata[i].num_inputs = node->num_inputs;`);
  source = replaceExact(source, `      const enum xnn_status status = xnn_run_operator_with_index(runtime->opdata[i].operator_objects[j], i, j, runtime->threadpool);`,
`      xnn_deepbom_bind_operator(&runtime->opdata[i],
                                runtime->opdata[i].operator_objects[j], i);
      const enum xnn_status status = xnn_run_operator_with_index(runtime->opdata[i].operator_objects[j], i, j, runtime->threadpool);`);
  await writeFile(file, source);
  return 3;
}

async function patchXnnpackOperatorRun() {
  const file = path.join(xnnpackRoot, "src", "operator-run.c");
  let source = await readFile(file, "utf8");
  source = insertAfterIncludeBlock(source, `#include "src/xnnpack/deepbom.h"\n\n#define DEEPBOM_XNN_DISPATCH(context, function) \\\n+  (xnn_deepbom_record_dispatch((context), (uintptr_t) (function)), (function))`);
  source = source.replace("\\\n+  (xnn_deepbom_record_dispatch", "\\\n+  (xnn_deepbom_record_dispatch");
  source = replaceExact(source, `#include "src/xnnpack/deepbom.h"\n\n#define DEEPBOM_XNN_DISPATCH(context, function) \\\n+  (xnn_deepbom_record_dispatch((context), (uintptr_t) (function)), (function))`, "");
  source = source.replace("\\\n+  (xnn_deepbom_record_dispatch", "\\\n  (xnn_deepbom_record_dispatch");
  source = replaceExact(source, `#include <pthreadpool.h>`, `#include <pthreadpool.h>\n#include "src/xnnpack/deepbom.h"`);
  const directPattern = /^([ \t]*)(context->[A-Za-z_][A-Za-z0-9_]*(?:(?:\.[A-Za-z_][A-Za-z0-9_]*)|(?:\[[^\]\n]+\]))*)\s*\(/gm;
  let directCount = 0;
  source = source.replace(directPattern, (_match, indent, expression) => {
    directCount += 1;
    return `${indent}xnn_deepbom_record_dispatch(context, (uintptr_t) (${expression}));\n${indent}${expression}(`;
  });
  if (directCount !== 88) throw new Error(`Pinned standalone operator-run dispatch count changed: expected 88, received ${directCount}`);
  source = replaceExact(source,
`  const size_t offset = context->packed_offset_fn(`,
`  xnn_deepbom_record_dispatch(context, (uintptr_t) (context->packed_offset_fn));
  const size_t offset = context->packed_offset_fn(`);
  await writeFile(file, source);
  return directCount + 1;
}

async function patchXnnpackBazel() {
  const file = path.join(xnnpackRoot, "BUILD.bazel");
  let source = await readFile(file, "utf8");
  source = replaceExact(source, `xnnpack_cc_library(
    name = "operators",`,
`xnnpack_cxx_library(
    name = "deepbom_trace",
    srcs = [
        "src/deepbom.cc",
        "src/deepbom_runtime_trace.cc",
    ],
    hdrs = [
        "src/deepbom_runtime_trace.h",
        "src/deepbom-symbol-catalog.h",
        "src/xnnpack/deepbom.h",
    ],
    deps = [
        ":common",
        ":node_type",
        ":operator_h",
        ":subgraph_h",
        ":xnnpack_h",
    ],
)

xnnpack_cc_library(
    name = "operators",`);
  source = replaceExact(source, `        ":datatype",
        ":fingerprint_cache",`, `        ":datatype",
        ":deepbom_trace",
        ":fingerprint_cache",`);
  await writeFile(file, source);
  return 2;
}

async function patchXnnpackCmake() {
  const file = path.join(xnnpackRoot, "CMakeLists.txt");
  let source = await readFile(file, "utf8");
  source = replaceExact(source, `  ADD_LIBRARY(xnnpack-datatype OBJECT src/datatype.c)`,
`  ADD_LIBRARY(xnnpack-datatype OBJECT src/datatype.c)
  ADD_LIBRARY(xnnpack-deepbom OBJECT src/deepbom.cc src/deepbom_runtime_trace.cc)
  TARGET_INCLUDE_DIRECTORIES(xnnpack-deepbom PRIVATE \${CMAKE_CURRENT_SOURCE_DIR})`);
  source = replaceExact(source, `  ADD_LIBRARY(XNNPACK \${XNNPACK_SRCS})`, `  ADD_LIBRARY(XNNPACK \${XNNPACK_SRCS} $<TARGET_OBJECTS:xnnpack-deepbom>)`);
  source = replaceExact(source, `    ADD_LIBRARY(XNNPACK SHARED \${XNNPACK_SRCS})`, `    ADD_LIBRARY(XNNPACK SHARED \${XNNPACK_SRCS} $<TARGET_OBJECTS:xnnpack-deepbom>)`);
  source = replaceExact(source, `    ADD_LIBRARY(XNNPACK STATIC \${XNNPACK_SRCS})`, `    ADD_LIBRARY(XNNPACK STATIC \${XNNPACK_SRCS} $<TARGET_OBJECTS:xnnpack-deepbom>)`);
  await writeFile(file, source);
  return 4;
}

async function buildSymbolCatalog(root) {
  const files = await walk(path.join(root, "src"));
  const candidates = files.filter((file) => /\.(?:c|cc|S|s)$/i.test(file) && !/[\\/]test[\\/]/i.test(file));
  const symbols = new Map();
  const pattern = /\b(xnn_[A-Za-z0-9_]*ukernel[A-Za-z0-9_]*)\s*\(/g;
  for (const file of candidates) {
    const source = await readFile(file, "utf8").catch(() => "");
    for (const match of source.matchAll(pattern)) {
      const relative = path.relative(root, file).replaceAll("\\", "/");
      if (!symbols.has(match[1]) || relative.length < symbols.get(match[1]).length) symbols.set(match[1], relative);
    }
  }
  const rows = [...symbols].map(([symbol, source_path]) => ({ symbol, source_path })).sort((a, b) => a.symbol.localeCompare(b.symbol));
  if (rows.length !== EXPECTED_XNN_MICROKERNEL_SYMBOLS) {
    throw new Error(`Pinned XNNPACK symbol catalog changed: expected ${EXPECTED_XNN_MICROKERNEL_SYMBOLS}, received ${rows.length}`);
  }
  return rows;
}

async function patchedFileIdentities() {
  const paths = [
    path.join(tensorflowRoot, "tensorflow", "lite", "delegates", "xnnpack", "xnnpack_delegate.cc"),
    path.join(tensorflowRoot, "tensorflow", "lite", "arena_planner.cc"),
    path.join(tensorflowRoot, "tensorflow", "lite", "simple_memory_arena.cc"),
    path.join(tensorflowRoot, "tensorflow", "lite", "kernels", "internal", "reference", "logistic.h"),
    path.join(tensorflowRoot, "third_party", "xla", "xla", "tsl", "tsl.bzl"),
    ...["include/xnnpack.h", "src/xnnpack/subgraph.h", "src/runtime.c", "src/operator-run.c", "src/deepbom.cc", "src/deepbom_runtime_trace.cc", "src/deepbom-symbol-catalog.h", "BUILD.bazel", "CMakeLists.txt"].map((file) => path.join(xnnpackRoot, file)),
  ];
  const rows = [];
  for (const file of paths) rows.push({ path: file.replaceAll("\\", "/"), sha256: sha256(await readFile(file)) });
  return rows;
}

async function requireCommit(root, expected, label) {
  const actual = (await run("git", ["-c", `safe.directory=${root.replaceAll("\\", "/")}`, "-C", root, "rev-parse", "HEAD"])).trim();
  if (actual !== expected) throw new Error(`${label} checkout must be ${expected}; received ${actual}`);
}

function replaceExact(source, before, after) {
  const first = source.indexOf(before);
  if (first < 0 || source.indexOf(before, first + before.length) >= 0) throw new Error(`Expected exactly one pinned source fragment: ${before.slice(0, 100)}`);
  return `${source.slice(0, first)}${after}${source.slice(first + before.length)}`;
}

function insertAfterIncludeBlock(source, insertion) {
  const matches = [...source.matchAll(/^#include .+$/gm)];
  if (!matches.length) throw new Error("Source has no include block");
  const last = matches.at(-1);
  const offset = last.index + last[0].length;
  return `${source.slice(0, offset)}\n${insertion}${source.slice(offset)}`;
}

async function walk(root) {
  const output = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const target = path.join(root, entry.name);
    if (entry.isDirectory()) output.push(...await walk(target));
    else if (entry.isFile()) output.push(target);
  }
  return output;
}

function run(command, args) {
  return import("node:child_process").then(({ execFile }) => new Promise((resolve, reject) => execFile(command, args, { maxBuffer: 4 * 1024 * 1024 }, (error, stdout, stderr) => error ? reject(new Error(stderr || error.message)) : resolve(stdout))));
}

function argument(name) {
  const index = process.argv.indexOf(name);
  return index < 0 ? null : process.argv[index + 1] || null;
}

function requiredPath(name) {
  const value = argument(name);
  if (!value) throw new Error(`${name} is required`);
  return path.resolve(value);
}

function optionalPath(name) {
  const value = argument(name);
  return value ? path.resolve(value) : null;
}

function sha256(bytes) { return createHash("sha256").update(bytes).digest("hex"); }
function cString(value) { return JSON.stringify(String(value)); }
function stableJson(value) { return `${JSON.stringify(value, null, 2)}\n`; }
