import { GGUF_BACKEND_PROFILES, GGUF_BACKEND_SOURCE, GGUF_RUNTIME_INSTRUMENTATION } from "./gguf-backend-contract.generated.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const GGUF_RUNTIME_ENVIRONMENT_SCHEMA = "deepbom.gguf_runtime_environment.v2";
export const GGUF_SCHEDULER_GRAPH_SCHEMA = "deepbom.ggml_scheduler_graph_trace.v1";

const PROFILE_BY_ID = new Map(GGUF_BACKEND_PROFILES.map((profile) => [profile.id, profile]));
const STATUS = new Set(["not_run", "observed_success", "observed_failure"]);

export function buildGgufRuntimeEnvironmentTemplate(analysis) {
  requireGgufAnalysis(analysis);
  return {
    schema: GGUF_RUNTIME_ENVIRONMENT_SCHEMA,
    evidence_class: "RUNTIME_ENVIRONMENT_MANIFEST",
    artifact: {
      format: "gguf",
      filename: analysis.filename || null,
      sha256: requireSha(analysis.model_sha256, "artifact SHA-256"),
    },
    runtime: {
      repository: GGUF_BACKEND_SOURCE.repository,
      source_commit: GGUF_BACKEND_SOURCE.source_commit,
      binary_sha256: null,
      version_output: null,
    },
    build: {
      cmake_cache_sha256: null,
      options: Object.fromEntries(GGUF_BACKEND_PROFILES.map((profile) => [profile.cmake_option, null])),
      compiled_backend_profile_ids: [],
      attestation: null,
    },
    selection: {
      requested_backend_profile_id: "cpu",
      context_size: null,
      batch_size: null,
      ubatch_size: null,
      gpu_layers: 0,
    },
    device: {
      platform: null,
      architecture: null,
      hostname_sha256: null,
      cpu_features: [],
      accelerator_inventory: [],
    },
    capture: {
      capture_id: null,
      collected_at: null,
      collector: { name: "deepbom-gguf-runtime-collector", version: "2" },
      import_file_sha256: null,
    },
    observations: {
      backend_inventory_status: "not_run",
      model_load_status: "not_run",
      inference_status: "not_run",
      selected_backend_observation: "not_observed",
      elapsed_ms: null,
      process_exit_code: null,
      stdout_sha256: null,
      stderr_sha256: null,
    },
    instrumentation: { ...GGUF_RUNTIME_INSTRUMENTATION, build_attestation_sha256: null },
    compute_graph: null,
    boundary: "The manifest binds artifact, runtime source, instrumented binary, canonical patch, build attestation, requested execution configuration, and collector observations. When executed, it carries observed generated GGML graph, scheduler split, inserted copy/view-node, per-node backend, and dispatch evidence; it does not identify a selected microkernel or task quality.",
  };
}

export function parseGgufRuntimeEnvironmentDocument(source, analysis, { fileSha256 = null } = {}) {
  requireGgufAnalysis(analysis);
  const document = typeof source === "string" ? parseJsonObject(source) : source;
  if (!document || document.schema !== GGUF_RUNTIME_ENVIRONMENT_SCHEMA) {
    throw new Error(`GGUF runtime evidence must use ${GGUF_RUNTIME_ENVIRONMENT_SCHEMA}.`);
  }
  const artifactSha256 = requireSha(document.artifact?.sha256, "runtime artifact SHA-256");
  if (document.artifact?.format !== "gguf" || artifactSha256 !== requireSha(analysis.model_sha256, "active artifact SHA-256")) {
    throw new Error("GGUF runtime evidence is not bound to the active artifact SHA-256.");
  }
  const binarySha256 = requireSha(document.runtime?.binary_sha256, "runtime binary SHA-256");
  const sourceCommit = requireCommit(document.runtime?.source_commit, "runtime source commit");
  const repository = requireText(document.runtime?.repository, "runtime repository");
  const sourceAlignment = repository === GGUF_BACKEND_SOURCE.repository && sourceCommit === GGUF_BACKEND_SOURCE.source_commit
    ? "exact_pinned_source_match"
    : "different_runtime_source_not_compared";
  const options = validateBuildOptions(document.build?.options);
  const compiledProfiles = uniqueStrings(document.build?.compiled_backend_profile_ids, "compiled backend profile IDs");
  for (const id of compiledProfiles) {
    const profile = PROFILE_BY_ID.get(id);
    if (!profile) throw new Error(`Unknown GGUF backend profile ID: ${id}.`);
    if (options[profile.cmake_option] !== true) {
      throw new Error(`Compiled backend ${id} conflicts with ${profile.cmake_option}=false or unknown.`);
    }
  }
  for (const profile of GGUF_BACKEND_PROFILES) {
    if (options[profile.cmake_option] === true && !compiledProfiles.includes(profile.id)) {
      throw new Error(`${profile.cmake_option}=true is missing backend profile ${profile.id}.`);
    }
  }
  const requestedBackend = requireText(document.selection?.requested_backend_profile_id, "requested backend profile ID").toLowerCase();
  const requestedProfile = PROFILE_BY_ID.get(requestedBackend);
  if (!requestedProfile) throw new Error(`Unknown requested GGUF backend profile ID: ${requestedBackend}.`);
  if (!compiledProfiles.includes(requestedBackend)) {
    throw new Error(`Requested backend ${requestedBackend} is not present in the compiled backend inventory.`);
  }
  const contextSize = requirePositiveInteger(document.selection?.context_size, "context size");
  const batchSize = requirePositiveInteger(document.selection?.batch_size, "batch size");
  const ubatchSize = requirePositiveInteger(document.selection?.ubatch_size, "ubatch size");
  if (ubatchSize > batchSize) throw new Error("GGUF runtime ubatch size cannot exceed batch size.");
  const gpuLayers = requireNonNegativeInteger(document.selection?.gpu_layers, "GPU layer count");
  const collectedAt = requireIsoTimestamp(document.capture?.collected_at, "capture timestamp");
  const captureId = requireText(document.capture?.capture_id, "capture ID");
  const collectorName = requireText(document.capture?.collector?.name, "collector name");
  const collectorVersion = requireText(document.capture?.collector?.version, "collector version");
  const observations = validateObservations(document.observations);
  const cmakeCacheSha256 = requireSha(document.build?.cmake_cache_sha256, "CMake cache SHA-256");
  const instrumentation = validateInstrumentation(document.instrumentation);
  const buildAttestation = validateLlamaCppBuildAttestation(document.build?.attestation);
  if (buildAttestation.file_sha256 !== instrumentation.build_attestation_sha256) {
    throw new Error("GGUF instrumentation binding and embedded build-attestation file SHA-256 differ.");
  }
  if (buildAttestation.document.binary.sha256 !== binarySha256) {
    throw new Error("GGUF runtime binary SHA-256 differs from the selected-build attestation.");
  }
  if (buildAttestation.document.build.cmake_cache_sha256 !== cmakeCacheSha256) {
    throw new Error("GGUF CMake cache SHA-256 differs from the selected-build attestation.");
  }
  const computeGraph = document.compute_graph == null ? null : validateGgmlSchedulerGraphEvidence(document.compute_graph);
  if (observations.inference_status === "observed_success" && (!computeGraph || computeGraph.successful_dispatch_count < 1 || computeGraph.failed_dispatch_count > 0)) {
    throw new Error("Successful GGUF inference requires at least one successful scheduler dispatch and no failed dispatch status.");
  }
  if (observations.selected_backend_observation === "scheduler_graph_named_backend" && !computeGraph) {
    throw new Error("A scheduler-observed backend requires validated scheduler graph evidence.");
  }
  if (computeGraph && computeGraph.trace_protocol !== GGUF_RUNTIME_INSTRUMENTATION.trace_protocol) {
    throw new Error("GGUF compute graph trace protocol does not match the canonical instrumentation patch.");
  }
  const importSha256 = fileSha256 ? requireSha(fileSha256, "import file SHA-256") : null;
  const normalized = {
    schema: GGUF_RUNTIME_ENVIRONMENT_SCHEMA,
    evidence_class: "RUNTIME_ENVIRONMENT_MANIFEST",
    artifact: {
      format: "gguf",
      filename: document.artifact?.filename || analysis.filename || null,
      sha256: artifactSha256,
    },
    runtime: {
      repository,
      source_commit: sourceCommit,
      source_alignment: sourceAlignment,
      binary_sha256: binarySha256,
      version_output: optionalBoundedText(document.runtime?.version_output, 8192),
    },
    build: {
      cmake_cache_sha256: cmakeCacheSha256,
      options,
      compiled_backend_profile_ids: compiledProfiles,
      source_profile_count: GGUF_BACKEND_PROFILES.length,
      attestation: buildAttestation,
    },
    selection: {
      requested_backend_profile_id: requestedBackend,
      requested_backend_label: requestedProfile.label,
      selection_evidence_class: "DECLARED_REQUIREMENT",
      context_size: contextSize,
      batch_size: batchSize,
      ubatch_size: ubatchSize,
      gpu_layers: gpuLayers,
    },
    device: {
      platform: requireText(document.device?.platform, "device platform"),
      architecture: requireText(document.device?.architecture, "device architecture"),
      hostname_sha256: document.device?.hostname_sha256 == null ? null : requireSha(document.device.hostname_sha256, "hostname SHA-256"),
      cpu_features: uniqueStrings(document.device?.cpu_features || [], "CPU features"),
      accelerator_inventory: uniqueStrings(document.device?.accelerator_inventory || [], "accelerator inventory"),
    },
    capture: {
      capture_id: captureId,
      collected_at: collectedAt,
      collector: { name: collectorName, version: collectorVersion },
      import_file_sha256: importSha256,
    },
    observations,
    instrumentation,
    compute_graph: computeGraph,
    static_source_comparison_status: sourceAlignment === "exact_pinned_source_match"
      ? "build_prerequisites_compared"
      : "not_compared_source_revision_differs",
    runtime_identity_status: "bound",
    graph_assignment_status: computeGraph?.dispatched_graph_count > 0
      ? "observed_generated_scheduler_graphs"
      : "not_observed_no_dispatched_scheduler_graph",
    compatibility_conclusion: observations.inference_status === "observed_success"
      ? "runtime_smoke_execution_observed_for_declared_configuration"
      : observations.model_load_status === "observed_success"
        ? "runtime_model_load_observed_inference_not_established"
        : "runtime_identity_bound_execution_not_established",
    boundary: "The imported manifest binds the canonical scheduler patch and its build attestation. Generated GGML nodes, scheduler splits, inserted scheduler nodes, backend assignment, and dispatch status are observed for captured calls only. No selected microkernel, task accuracy, uncaptured path, or production latency is inferred.",
  };
  return { ...normalized, normalized_manifest_sha256: sha256TextHex(canonicalJson(normalized)) };
}

export function isGgufRuntimeEnvironmentDocument(value) {
  return value?.schema === GGUF_RUNTIME_ENVIRONMENT_SCHEMA;
}

export function parseGgmlSchedulerTrace(source, { traceSha256 = null } = {}) {
  const text = String(source || "");
  if (!text || text.length > 128 * 1024 * 1024) throw new Error("GGML scheduler trace is empty or exceeds 128 MiB.");
  const records = text.split(/\r?\n/).filter(Boolean);
  if (records.length > 2_000_000) throw new Error("GGML scheduler trace exceeds the record limit.");
  const graphs = new Map();
  for (const [lineIndex, line] of records.entries()) {
    const fields = line.split("\t");
    if (fields[0] !== GGUF_RUNTIME_INSTRUMENTATION.trace_protocol) throw new Error(`GGML trace line ${lineIndex + 1} has an invalid protocol marker.`);
    const event = fields[1];
    const sequence = traceInteger(fields[2], "graph sequence", { positive: true });
    if (event === "graph") {
      if (fields.length !== 9 || graphs.has(sequence)) throw new Error(`GGML graph ${sequence} header is malformed or duplicated.`);
      if (graphs.size >= 4096) throw new Error("GGML scheduler trace exceeds the graph limit.");
      graphs.set(sequence, {
        sequence,
        graph_uid_decimal: traceDecimal(fields[3], "graph UID"),
        original_node_count: traceInteger(fields[4], "original node count"),
        original_leaf_count: traceInteger(fields[5], "original leaf count"),
        scheduled_node_count: traceInteger(fields[6], "scheduled node count"),
        scheduled_leaf_count: traceInteger(fields[7], "scheduled leaf count"),
        split_count: traceInteger(fields[8], "split count"),
        backends: [], splits: [], original_nodes: [], original_leafs: [], scheduled_nodes: [], scheduled_leafs: [], dispatch_statuses: [], closed: false,
      });
      continue;
    }
    const graph = graphs.get(sequence);
    if (!graph) throw new Error(`GGML trace event ${event} references missing graph ${sequence}.`);
    if (graph.closed && event !== "dispatch") throw new Error(`GGML graph ${sequence} contains ${event} after graph_end.`);
    if (event === "backend") {
      if (fields.length !== 6) throw new Error(`GGML backend row for graph ${sequence} is malformed.`);
      graph.backends.push({ backend_id: traceInteger(fields[3], "backend ID"), name: traceHexText(fields[4]), device: traceHexText(fields[5]) });
    } else if (event === "split") {
      if (fields.length !== 9) throw new Error(`GGML split row for graph ${sequence} is malformed.`);
      graph.splits.push({
        split_id: traceInteger(fields[3], "split ID"), start_node_index: traceInteger(fields[4], "split start"),
        end_node_index: traceInteger(fields[5], "split end"), input_count: traceInteger(fields[6], "split input count"),
        backend: traceHexText(fields[7]), inputs: traceRefs(fields[8]),
      });
    } else if (["original_node", "original_leaf", "scheduled_node", "scheduled_leaf"].includes(event)) {
      if (fields.length !== 14) throw new Error(`GGML tensor row ${event} for graph ${sequence} is malformed.`);
      const row = {
        index: traceInteger(fields[3], `${event} index`), split_id: traceInteger(fields[4], `${event} split ID`, { allowNegativeOne: true }),
        backend: fields[5] === "-" ? null : traceHexText(fields[5]), op: traceHexText(fields[6]), dtype: traceHexText(fields[7]),
        name: traceHexText(fields[8]), byte_length: traceInteger(fields[9], `${event} bytes`), dimensions: traceDimensions(fields[10]),
        flags: traceInteger(fields[11], `${event} flags`, { allowNegative: true }), sources: traceRefs(fields[12]), view_source: traceRef(fields[13]),
      };
      graph[`${event}s`].push(row);
    } else if (event === "graph_end") {
      if (fields.length !== 3 || graph.closed) throw new Error(`GGML graph ${sequence} end marker is malformed or duplicated.`);
      graph.closed = true;
    } else if (event === "dispatch") {
      if (fields.length !== 4 || !graph.closed) throw new Error(`GGML dispatch row for graph ${sequence} is malformed or precedes graph_end.`);
      graph.dispatch_statuses.push(traceInteger(fields[3], "dispatch status", { allowNegative: true }));
    } else throw new Error(`Unsupported GGML trace event: ${event}.`);
  }
  return buildSchedulerGraphEvidence([...graphs.values()], traceSha256 ? requireSha(traceSha256, "GGML trace SHA-256") : sha256TextHex(text));
}

export function validateGgmlSchedulerGraphEvidence(value) {
  if (!value || value.schema !== GGUF_SCHEDULER_GRAPH_SCHEMA || !Array.isArray(value.graphs)) throw new Error(`GGUF compute graph evidence must use ${GGUF_SCHEDULER_GRAPH_SCHEMA}.`);
  const expected = buildSchedulerGraphEvidence(value.graphs.map(({ graph_sha256: _digest, ...graph }) => graph), requireSha(value.trace_sha256, "GGML trace SHA-256"));
  if (canonicalJson(expected) !== canonicalJson(value)) throw new Error("GGUF scheduler graph evidence does not independently reconstruct.");
  return expected;
}

export function validateLlamaCppBuildAttestation(value) {
  if (!value || typeof value !== "object") throw new Error("Instrumented llama.cpp build attestation is required.");
  const document = value.document;
  if (!document || document.schema !== "deepbom.llama_cpp_instrumented_build_attestation.v1"
    || document.evidence_class !== "REPRODUCIBLE_BUILD_ATTESTATION") {
    throw new Error("Instrumented llama.cpp build attestation schema is invalid.");
  }
  const source = {
    repository: requireText(document.source?.repository, "attested llama.cpp repository"),
    commit: requireCommit(document.source?.commit, "attested llama.cpp commit"),
    scheduler_path: requireText(document.source?.scheduler_path, "attested scheduler path"),
    scheduler_original_sha256: requireSha(document.source?.scheduler_original_sha256, "attested original scheduler SHA-256"),
    scheduler_patched_sha256: requireSha(document.source?.scheduler_patched_sha256, "attested patched scheduler SHA-256"),
  };
  if (source.repository !== GGUF_BACKEND_SOURCE.repository || source.commit !== GGUF_BACKEND_SOURCE.source_commit
    || source.scheduler_path !== GGUF_BACKEND_SOURCE.files.scheduler.path
    || source.scheduler_original_sha256 !== GGUF_RUNTIME_INSTRUMENTATION.scheduler_source_original_sha256
    || source.scheduler_patched_sha256 !== GGUF_RUNTIME_INSTRUMENTATION.scheduler_source_patched_sha256) {
    throw new Error("Instrumented llama.cpp build attestation does not bind the pinned scheduler source.");
  }
  const instrumentation = {
    patch_id: requireText(document.instrumentation?.patch_id, "attested instrumentation patch ID"),
    patch_path: requireText(document.instrumentation?.patch_path, "attested instrumentation patch path"),
    patch_sha256: requireSha(document.instrumentation?.patch_sha256, "attested instrumentation patch SHA-256"),
    trace_protocol: requireText(document.instrumentation?.trace_protocol, "attested trace protocol"),
  };
  for (const key of ["patch_id", "patch_path", "patch_sha256", "trace_protocol"]) {
    if (instrumentation[key] !== GGUF_RUNTIME_INSTRUMENTATION[key]) {
      throw new Error(`Instrumented llama.cpp build attestation ${key} differs from the canonical patch contract.`);
    }
  }
  const build = {
    configure_arguments: boundedStringArray(document.build?.configure_arguments, "CMake configure arguments", 64, 4096),
    build_arguments: boundedStringArray(document.build?.build_arguments, "CMake build arguments", 32, 4096),
    parallel_jobs: requirePositiveInteger(document.build?.parallel_jobs, "attested build parallelism"),
    timeout_ms: requirePositiveInteger(document.build?.timeout_ms, "attested build timeout"),
    configure_stdout_sha256: requireSha(document.build?.configure_stdout_sha256, "CMake configure stdout SHA-256"),
    configure_stderr_sha256: requireSha(document.build?.configure_stderr_sha256, "CMake configure stderr SHA-256"),
    build_stdout_sha256: requireSha(document.build?.build_stdout_sha256, "build stdout SHA-256"),
    build_stderr_sha256: requireSha(document.build?.build_stderr_sha256, "build stderr SHA-256"),
    cmake_cache_sha256: requireSha(document.build?.cmake_cache_sha256, "attested CMake cache SHA-256"),
    cmake_generator: optionalBoundedText(document.build?.cmake_generator, 1024),
    c_compiler: optionalBoundedText(document.build?.c_compiler, 4096),
    cxx_compiler: optionalBoundedText(document.build?.cxx_compiler, 4096),
    build_type: requireText(document.build?.build_type, "attested build type"),
  };
  if (build.parallel_jobs > 64) throw new Error("Attested build parallelism exceeds 64 jobs.");
  if (build.timeout_ms > 3_600_000) throw new Error("Attested build timeout exceeds one hour.");
  const binary = {
    filename: requireText(document.binary?.filename, "attested runtime binary filename"),
    sha256: requireSha(document.binary?.sha256, "attested runtime binary SHA-256"),
    version_output: optionalBoundedText(document.binary?.version_output, 8192),
  };
  const normalizedDocument = {
    schema: "deepbom.llama_cpp_instrumented_build_attestation.v1",
    evidence_class: "REPRODUCIBLE_BUILD_ATTESTATION",
    source,
    instrumentation,
    build,
    binary,
    boundary: optionalBoundedText(document.boundary, 8192),
  };
  const canonicalSha256 = sha256TextHex(canonicalJson(normalizedDocument));
  if (requireSha(value.canonical_sha256, "canonical build-attestation SHA-256") !== canonicalSha256) {
    throw new Error("Instrumented llama.cpp build attestation canonical SHA-256 does not reconstruct.");
  }
  return {
    file_sha256: requireSha(value.file_sha256, "build-attestation file SHA-256"),
    canonical_sha256: canonicalSha256,
    document: normalizedDocument,
  };
}

function buildSchedulerGraphEvidence(sourceGraphs, traceSha256) {
  const graphs = sourceGraphs.map(validateSchedulerGraph).sort((left, right) => left.sequence - right.sequence);
  if (!graphs.length || graphs.some((graph, index) => graph.sequence !== index + 1)) throw new Error("GGML graph sequences must be contiguous from one.");
  const graphPayload = graphs.map(({ graph_sha256: _digest, ...graph }) => graph);
  const summary = {
    graph_count: graphs.length,
    dispatched_graph_count: graphs.filter((graph) => graph.dispatch_statuses.length > 0).length,
    dispatch_count: graphs.reduce((sum, graph) => sum + graph.dispatch_statuses.length, 0),
    successful_dispatch_count: graphs.reduce((sum, graph) => sum + graph.dispatch_statuses.filter((status) => status === 0).length, 0),
    failed_dispatch_count: graphs.reduce((sum, graph) => sum + graph.dispatch_statuses.filter((status) => status !== 0).length, 0),
    original_node_count: graphs.reduce((sum, graph) => sum + graph.original_node_count, 0),
    scheduled_node_count: graphs.reduce((sum, graph) => sum + graph.scheduled_node_count, 0),
    scheduler_inserted_node_count: graphs.reduce((sum, graph) => sum + graph.scheduler_inserted_node_count, 0),
    split_count: graphs.reduce((sum, graph) => sum + graph.split_count, 0),
    original_backend_transition_edge_count: graphs.reduce((sum, graph) => sum + graph.original_backend_transition_edge_count, 0),
    scheduled_backend_transition_edge_count: graphs.reduce((sum, graph) => sum + graph.scheduled_backend_transition_edge_count, 0),
    original_backend_unresolved_source_edge_count: graphs.reduce((sum, graph) => sum + graph.original_backend_unresolved_source_edge_count, 0),
    scheduled_backend_unresolved_source_edge_count: graphs.reduce((sum, graph) => sum + graph.scheduled_backend_unresolved_source_edge_count, 0),
  };
  const evidence = {
    schema: GGUF_SCHEDULER_GRAPH_SCHEMA,
    evidence_class: "OBSERVED_INSTRUMENTED_RUNTIME",
    trace_protocol: GGUF_RUNTIME_INSTRUMENTATION.trace_protocol,
    trace_sha256: traceSha256,
    ...summary,
    graphs,
    interpretation_boundary: "Rows are generated inside the pinned ggml backend scheduler after split construction. Original and scheduler-expanded graphs, split ranges, assigned backend names, and dispatch return statuses are observed only for captured calls. Microkernel identity and uncaptured execution paths remain unobserved.",
  };
  return { ...evidence, evidence_sha256: sha256TextHex(canonicalJson({ ...evidence, graphs: graphPayload })) };
}

function validateSchedulerGraph(source) {
  const graph = structuredClone(source);
  delete graph.graph_sha256;
  delete graph.closed;
  const arrays = ["backends", "splits", "original_nodes", "original_leafs", "scheduled_nodes", "scheduled_leafs", "dispatch_statuses"];
  if (arrays.some((key) => !Array.isArray(graph[key]))) throw new Error(`GGML graph ${graph.sequence} is missing a row array.`);
  const contiguous = (rows) => rows.every((row, index) => row.index === index);
  if (graph.original_nodes.length !== graph.original_node_count || graph.original_leafs.length !== graph.original_leaf_count
    || graph.scheduled_nodes.length !== graph.scheduled_node_count || graph.scheduled_leafs.length !== graph.scheduled_leaf_count
    || graph.splits.length !== graph.split_count || !contiguous(graph.original_nodes) || !contiguous(graph.original_leafs)
    || !contiguous(graph.scheduled_nodes) || !contiguous(graph.scheduled_leafs)) throw new Error(`GGML graph ${graph.sequence} row counts do not conserve.`);
  if (source.closed === false) throw new Error(`GGML graph ${graph.sequence} lacks a graph_end marker.`);
  if (graph.backends.some((row, index) => row.backend_id !== index || !row.name) || new Set(graph.backends.map((row) => row.name)).size !== graph.backends.length) {
    throw new Error(`GGML graph ${graph.sequence} backend inventory is invalid.`);
  }
  const backendNames = new Set(graph.backends.map((row) => row.name));
  const assignedRows = [...graph.original_nodes, ...graph.scheduled_nodes];
  const tensorRows = [...assignedRows, ...graph.original_leafs, ...graph.scheduled_leafs];
  if (assignedRows.some((row) => !backendNames.has(row.backend))
    || tensorRows.some((row) => row.backend != null && !backendNames.has(row.backend) || !Array.isArray(row.dimensions) || row.dimensions.length !== 4
    || !Array.isArray(row.sources) || !row.sources.every((ref) => validTraceRef(ref)))) throw new Error(`GGML graph ${graph.sequence} tensor contract is invalid.`);
  if (graph.splits.some((row, index) => row.split_id !== index || row.start_node_index < 0 || row.end_node_index < row.start_node_index
    || row.end_node_index > graph.original_node_count || row.input_count !== row.inputs.length || !backendNames.has(row.backend))) throw new Error(`GGML graph ${graph.sequence} split ledger is invalid.`);
  if (graph.splits.length && (graph.splits[0].start_node_index !== 0 || graph.splits.at(-1).end_node_index !== graph.original_node_count
    || graph.splits.some((row, index) => index > 0 && graph.splits[index - 1].end_node_index !== row.start_node_index))) throw new Error(`GGML graph ${graph.sequence} split ranges do not partition original nodes.`);
  for (const split of graph.splits) for (const ref of split.inputs) validateGraphRef(ref, graph.original_nodes, graph.original_leafs, graph.sequence, "split input");
  validateGraphRefs(graph.original_nodes, graph.original_leafs, graph.sequence, "original");
  validateGraphRefs(graph.scheduled_nodes, graph.scheduled_leafs, graph.sequence, "scheduled");
  if (graph.original_nodes.some((row) => row.split_id < 0 || graph.splits[row.split_id]?.backend !== row.backend)) throw new Error(`GGML graph ${graph.sequence} original node split/backend binding differs.`);
  const originalEdges = backendEdgeAssessment(graph.original_nodes, graph.original_leafs);
  const scheduledEdges = backendEdgeAssessment(graph.scheduled_nodes, graph.scheduled_leafs);
  graph.scheduler_inserted_node_count = graph.scheduled_node_count - graph.original_node_count;
  if (graph.scheduler_inserted_node_count < 0) throw new Error(`GGML graph ${graph.sequence} scheduled graph lost original nodes.`);
  graph.original_backend_transition_edge_count = originalEdges.transitionCount;
  graph.scheduled_backend_transition_edge_count = scheduledEdges.transitionCount;
  graph.original_backend_assessed_source_edge_count = originalEdges.assessedCount;
  graph.scheduled_backend_assessed_source_edge_count = scheduledEdges.assessedCount;
  graph.original_backend_unresolved_source_edge_count = originalEdges.unresolvedCount;
  graph.scheduled_backend_unresolved_source_edge_count = scheduledEdges.unresolvedCount;
  graph.dispatched = graph.dispatch_statuses.length > 0;
  graph.graph_sha256 = sha256TextHex(canonicalJson(graph));
  return graph;
}

function validateGraphRefs(nodes, leafs, sequence, label) {
  for (const row of [...nodes, ...leafs]) for (const ref of [...row.sources, row.view_source]) {
    validateGraphRef(ref, nodes, leafs, sequence, label);
  }
}

function validateGraphRef(ref, nodes, leafs, sequence, label) {
  if (ref === "-" || ref.startsWith("X")) return;
  const index = Number(ref.slice(1));
  if (ref[0] === "N" ? !nodes[index] : ref[0] === "L" ? !leafs[index] : true) throw new Error(`GGML graph ${sequence} ${label} reference ${ref} is out of range.`);
}

function backendEdgeAssessment(nodes, leafs) {
  let assessedCount = 0;
  let transitionCount = 0;
  let unresolvedCount = 0;
  for (const row of nodes) for (const ref of [...row.sources, row.view_source].filter((item) => item !== "-")) {
    const source = ref[0] === "N" ? nodes[Number(ref.slice(1))] : ref[0] === "L" ? leafs[Number(ref.slice(1))] : null;
    if (!source || source.backend == null || row.backend == null) unresolvedCount += 1;
    else {
      assessedCount += 1;
      if (source.backend !== row.backend) transitionCount += 1;
    }
  }
  return { assessedCount, transitionCount, unresolvedCount };
}

function validateInstrumentation(value) {
  if (!value || typeof value !== "object") throw new Error("GGUF runtime instrumentation binding is required.");
  for (const key of ["patch_id", "patch_sha256", "trace_protocol", "scheduler_source_original_sha256", "scheduler_source_patched_sha256"]) {
    if (value[key] !== GGUF_RUNTIME_INSTRUMENTATION[key]) throw new Error(`GGUF runtime instrumentation ${key} differs from the canonical patch contract.`);
  }
  return { ...GGUF_RUNTIME_INSTRUMENTATION, build_attestation_sha256: requireSha(value.build_attestation_sha256, "llama.cpp build attestation SHA-256") };
}

function traceInteger(value, label, { positive = false, allowNegative = false, allowNegativeOne = false } = {}) {
  if (!/^-?(0|[1-9][0-9]*)$/.test(String(value))) throw new Error(`${label} is not a canonical integer.`);
  const number = Number(value);
  if (!Number.isSafeInteger(number) || positive && number <= 0 || !allowNegative && !allowNegativeOne && number < 0 || allowNegativeOne && number < -1) throw new Error(`${label} is out of range.`);
  return number;
}
function traceDecimal(value, label) { if (!/^(0|[1-9][0-9]*)$/.test(String(value))) throw new Error(`${label} is not an unsigned decimal.`); return String(value); }
function traceDimensions(value) { const rows = String(value).split(","); if (rows.length !== 4) throw new Error("GGML tensor dimensions must contain four axes."); return rows.map((item) => traceInteger(item, "tensor dimension")); }
function traceRefs(value) { return String(value).split(";").filter((ref) => ref && ref !== "-").map(traceRef); }
function traceRef(value) { const ref = String(value); if (!validTraceRef(ref)) throw new Error(`Invalid GGML tensor reference: ${ref}.`); return ref; }
function validTraceRef(ref) { return ref === "-" || /^[NL](0|[1-9][0-9]*)$/.test(ref) || /^X(?:[0-9a-f]{2})*$/.test(ref); }
function traceHexText(value) {
  const hex = String(value);
  if (!/^(?:[0-9a-f]{2})*$/.test(hex)) throw new Error("GGML trace text is not canonical hexadecimal UTF-8.");
  const bytes = Uint8Array.from(hex.match(/../g) || [], (pair) => Number.parseInt(pair, 16));
  try { return new TextDecoder("utf-8", { fatal: true }).decode(bytes); } catch { throw new Error("GGML trace text contains invalid UTF-8."); }
}

function requireGgufAnalysis(analysis) {
  if (!analysis || analysis.format !== "gguf") throw new Error("GGUF runtime environment evidence requires an active GGUF audit.");
}

function parseJsonObject(text) {
  let value;
  try { value = JSON.parse(text); } catch { throw new Error("GGUF runtime environment JSON is invalid."); }
  if (!value || Array.isArray(value) || typeof value !== "object") throw new Error("GGUF runtime environment must be a JSON object.");
  return value;
}

function validateBuildOptions(value) {
  if (!value || Array.isArray(value) || typeof value !== "object") throw new Error("GGUF runtime build options are required.");
  const expected = new Set(GGUF_BACKEND_PROFILES.map((profile) => profile.cmake_option));
  for (const key of Object.keys(value)) if (!expected.has(key)) throw new Error(`Unknown GGUF build option: ${key}.`);
  return Object.fromEntries(GGUF_BACKEND_PROFILES.map((profile) => {
    const option = value[profile.cmake_option];
    if (typeof option !== "boolean") throw new Error(`${profile.cmake_option} must be a boolean from CMakeCache.txt.`);
    return [profile.cmake_option, option];
  }));
}

function validateObservations(value) {
  if (!value || Array.isArray(value) || typeof value !== "object") throw new Error("GGUF runtime observations are required.");
  const statuses = ["backend_inventory_status", "model_load_status", "inference_status"];
  const result = {};
  for (const key of statuses) {
    result[key] = String(value[key] || "");
    if (!STATUS.has(result[key])) throw new Error(`${key} has an unsupported status.`);
  }
  const selection = String(value.selected_backend_observation || "not_observed");
  if (!["not_observed", "runtime_output_named_backend", "scheduler_graph_named_backend"].includes(selection)) throw new Error("selected_backend_observation is invalid.");
  result.selected_backend_observation = selection;
  result.elapsed_ms = value.elapsed_ms == null ? null : requireNonNegativeFinite(value.elapsed_ms, "elapsed milliseconds");
  result.process_exit_code = value.process_exit_code == null ? null : requireInteger(value.process_exit_code, "process exit code");
  result.stdout_sha256 = value.stdout_sha256 == null ? null : requireSha(value.stdout_sha256, "stdout SHA-256");
  result.stderr_sha256 = value.stderr_sha256 == null ? null : requireSha(value.stderr_sha256, "stderr SHA-256");
  if (result.inference_status !== "not_run" && (result.elapsed_ms == null || result.process_exit_code == null)) {
    throw new Error("Observed GGUF inference requires elapsed_ms and process_exit_code.");
  }
  if (result.inference_status !== "not_run" && (!result.stdout_sha256 || !result.stderr_sha256)) {
    throw new Error("Observed GGUF inference requires stdout and stderr SHA-256 witnesses.");
  }
  if (result.inference_status === "observed_success" && result.process_exit_code !== 0) throw new Error("Successful inference must have process exit code 0.");
  if (result.inference_status === "observed_failure" && result.process_exit_code === 0) throw new Error("Failed inference cannot have process exit code 0.");
  if (result.inference_status === "observed_success" && result.model_load_status !== "observed_success") throw new Error("Successful inference requires successful model load status.");
  if (result.selected_backend_observation === "runtime_output_named_backend" && result.inference_status === "not_run") throw new Error("Selected backend cannot be observed without a runtime process.");
  return result;
}

function requireSha(value, label) {
  const normalized = String(value || "").toLowerCase();
  if (!/^[a-f0-9]{64}$/.test(normalized)) throw new Error(`${label} is required.`);
  return normalized;
}

function requireCommit(value, label) {
  const normalized = String(value || "").toLowerCase();
  if (!/^[a-f0-9]{40,64}$/.test(normalized)) throw new Error(`${label} is required.`);
  return normalized;
}

function requireText(value, label) {
  const text = String(value || "").trim();
  if (!text) throw new Error(`${label} is required.`);
  if (text.length > 8192) throw new Error(`${label} is too long.`);
  return text;
}

function optionalBoundedText(value, limit) {
  if (value == null || value === "") return null;
  const text = String(value);
  if (text.length > limit) throw new Error("Runtime version output exceeds the bounded length.");
  return text;
}

function boundedStringArray(value, label, maxItems, maxLength) {
  if (!Array.isArray(value) || value.length > maxItems) throw new Error(`${label} must contain at most ${maxItems} entries.`);
  return value.map((item) => {
    const text = String(item);
    if (!text || text.length > maxLength) throw new Error(`${label} contains an empty or oversized entry.`);
    return text;
  });
}

function uniqueStrings(value, label) {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array.`);
  const rows = value.map((item) => requireText(item, label));
  if (new Set(rows).size !== rows.length) throw new Error(`${label} contains duplicates.`);
  return rows;
}

function requirePositiveInteger(value, label) {
  const number = requireInteger(value, label);
  if (number <= 0) throw new Error(`${label} must be greater than zero.`);
  return number;
}

function requireNonNegativeInteger(value, label) {
  const number = requireInteger(value, label);
  if (number < 0) throw new Error(`${label} cannot be negative.`);
  return number;
}

function requireInteger(value, label) {
  const number = Number(value);
  if (!Number.isSafeInteger(number)) throw new Error(`${label} must be a safe integer.`);
  return number;
}

function requireNonNegativeFinite(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) throw new Error(`${label} must be a non-negative finite number.`);
  return number;
}

function requireIsoTimestamp(value, label) {
  const text = requireText(value, label);
  const timestamp = new Date(text);
  if (!Number.isFinite(timestamp.getTime()) || timestamp.toISOString() !== text) throw new Error(`${label} must be an ISO-8601 UTC timestamp.`);
  return text;
}
