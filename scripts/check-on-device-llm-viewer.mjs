import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const productHtml = await readFile("web/index.html", "utf8");
if (!/id="llmEvidencePanel"[^>]+data-format-scope="tflite onnx gguf safetensors"/.test(productHtml)) {
  throw new Error("The integrated LLM evidence panel must remain available to serialized TFLite/ONNX graphs and GGUF/SafeTensors contracts.");
}

const server = createServer(async (request, response) => {
  const path = new URL(request.url, "http://127.0.0.1").pathname;
  if (path === "/fixture.html") {
    response.setHeader("content-type", "text/html; charset=utf-8");
    response.end(`<!doctype html><html data-theme="light"><head><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><link rel="stylesheet" href="/web/styles.css"><link rel="stylesheet" href="/web/research-theme.css"></head><body><main style="max-width:1400px;margin:0 auto;padding:12px"><article id="llmEvidencePanel" class="perf-panel wide llm-evidence-panel"></article></main><script type="module" src="/fixture.js"></script></body></html>`);
    return;
  }
  if (path === "/fixture.js") {
    response.setHeader("content-type", "text/javascript; charset=utf-8");
    response.end(`
      import { renderOnDeviceLlmView } from "/web/lib/on-device-llm-view.js";
      const exact = (decimal) => ({ value: Number(decimal), decimal: String(decimal) });
      const scenarios = [
        { state_kind: "transformer_kv", context_length: 2048, batch_size: 1, storage_bits: 8, logical_bytes: exact(67108864), evidence_class: "DERIVED_CONDITIONAL_SCENARIO" },
        { state_kind: "transformer_kv", context_length: 8192, batch_size: 2, storage_bits: 16, logical_bytes: exact(1073741824), evidence_class: "DERIVED_CONDITIONAL_SCENARIO" },
      ];
      const memoryRows = scenarios.map((row, index) => ({
        ...row,
        serialized_weight_floor_bytes: exact(3579139413),
        logical_state_bytes: row.logical_bytes,
        static_lower_bound_bytes: exact(index ? 4652881237 : 3646248277),
        first_capacity_not_exceeded: index ? "8 GiB" : "4 GiB",
        lower_bound_exceeded_capacity_count: index ? 4 : 3,
      }));
      const contract = {
        schema: "deepbom.on_device_llm_contract.v2", status: "assessed_static_artifact_contract", evidence_class: "OBSERVED/SOURCE_BACKED/DERIVED", format: "gguf", artifact_role: "model",
        architecture: { family: "llama", kind: "dense_transformer_decoder", context_length: 8192, vocabulary_size: 32000, hidden_size: 4096, intermediate_size: 11008, layer_count: 32, attention_head_count: 32, kv_head_count: 8, head_width: 128, gqa_query_heads_per_kv_head: 4 },
        storage: { serialized_parameter_count_decimal: "7000000000", serialized_tensor_bytes_decimal: "3579139413", effective_bits_per_parameter: "4.09", encoding_inventory_sha256: "a".repeat(64), tensor_encoding_assignment_sha256: "b".repeat(64), recipe_interpretation: "Serialized assignment only; recipe authenticity is not inferred.", encoding_inventory: [
          { dtype: "Q4_K", tensor_count: 281, element_count_decimal: "6800000000", byte_length_decimal: "3400000000", effective_bits_per_element: "4" },
          { dtype: "F32", tensor_count: 65, element_count_decimal: "200000000", byte_length_decimal: "179139413", effective_bits_per_element: "32" },
        ] },
        tokenizer: { status: "assessed", vocabulary_count: 32000, chat_template: { status: "declared" }, definition_files: [] }, generation: { status: "not_embedded_as_runtime_policy" },
        state: { kv_projection: { status: "assessed" }, recurrent_projection: null, scenario_matrix: scenarios, scenario_boundary: "Conditional logical KV bytes; not runtime allocation." },
        compute: { projection: { status: "assessed", dense_projection_macs_all_layers_per_token: exact(1), prefill_transformer_core_macs_at_declared_context: exact(2), decode_transformer_core_macs_at_declared_context: exact(3), decode_with_one_logit_position_macs: exact(4) }, boundary: "Static architecture scenario." },
        runtime_contract: { status: "not_artifact_bound", evidence_class: "NOT_ASSESSABLE", required_bindings: ["runtime_engine_version_binary_and_build"] },
        memory_feasibility: { schema: "deepbom.llm_memory_feasibility.v1.1", status: "assessed_static_lower_bound_scenarios", evidence_class: "OBSERVED/DERIVED_CONDITIONAL_SCENARIO", capacity_scope: "single_aggregate_primary_memory_budget", residency_assumption: "All serialized tensor bytes and logical state bytes are simultaneously resident in one aggregate primary-memory budget.", serialized_weight_floor_bytes: exact(3579139413), minimum_static_lower_bound_bytes: exact(3646248277), maximum_static_lower_bound_bytes: exact(4652881237), static_scenarios: memoryRows, runtime_primary_residency: null, fit_claim: "not_emitted", boundary: "A smaller aggregate capacity is insufficient only under the emitted residency assumption; no fit claim is emitted." },
        static_memory_placement: { schema: "deepbom.llm_static_memory_placement.v1", status: "assessed_lower_bound_candidates", evidence_class: "OBSERVED_SERIALIZED_STORAGE/DERIVED_CONDITIONAL_STATIC_LOWER_BOUND", candidate_count: 2, lower_bound_not_exceeding_candidate_count: 1, minimum_accelerator_layer_count_not_disproven: 1, maximum_accelerator_layer_count_not_disproven: 1, logical_state: { kind: "transformer_kv", bytes: exact(67108864) }, fit_claim: "not_emitted", boundary: "Exact serialized bytes and logical state only; runtime packing and fit remain unresolved.", candidates: [
          { accelerator_layer_count: 0, cpu_layer_serialized_bytes: exact(3579139413), accelerator_layer_serialized_bytes: exact(0), cpu_accounted_lower_bound_bytes: exact(3579139413), accelerator_accounted_lower_bound_bytes: exact(67108864), status: "accounted_lower_bound_exceeds_at_least_one_pool" },
          { accelerator_layer_count: 1, cpu_layer_serialized_bytes: exact(179139413), accelerator_layer_serialized_bytes: exact(3400000000), cpu_accounted_lower_bound_bytes: exact(179139413), accelerator_accounted_lower_bound_bytes: exact(3467108864), status: "accounted_lower_bound_not_exceeding_pools_fit_unresolved" },
        ] },
        medical_ai_claim_boundary: { status: "not_established_by_model_artifact", declaration: { coverage: { declared: 0, required: 9 } }, required_external_evidence: ["intended_use"], not_established: ["task_accuracy", "clinical_validity"] },
      };
      window.__llmFixtureContract = contract;
      window.__renderLlmFixture = (value) => renderOnDeviceLlmView(document.querySelector("#llmEvidencePanel"), { on_device_llm: value });
      renderOnDeviceLlmView(document.querySelector("#llmEvidencePanel"), { on_device_llm: contract });
      window.__llmViewerReady = true;
    `);
    return;
  }
  const relative = decodeURIComponent(path).replace(/^\/+/, "");
  try {
    const bytes = await readFile(relative);
    const type = relative.endsWith(".html") ? "text/html; charset=utf-8"
      : relative.endsWith(".css") ? "text/css; charset=utf-8"
        : relative.endsWith(".js") || relative.endsWith(".mjs") ? "text/javascript; charset=utf-8"
          : relative.endsWith(".json") ? "application/json; charset=utf-8"
            : relative.endsWith(".wasm") ? "application/wasm" : "application/octet-stream";
    response.setHeader("content-type", type);
    response.end(bytes);
  } catch {
    response.statusCode = 404;
    response.end("not found");
  }
});

let browser;
try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error" && !message.text().includes("Failed to load resource")) errors.push(message.text());
  });
  page.on("response", (response) => {
    if (response.status() < 400) return;
    const url = new URL(response.url());
    if (!url.pathname.startsWith("/api/")) errors.push(`HTTP ${response.status()} ${url.pathname}`);
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/fixture.html`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__llmViewerReady === true);

  for (const theme of ["light", "dark"]) {
    await page.evaluate((value) => { document.documentElement.dataset.theme = value; }, theme);
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({ width, height: width < 500 ? 844 : 1000 });
      const state = await page.locator("#llmEvidencePanel").evaluate((panel) => {
        const memoryScroll = panel.querySelector(".llm-memory-table")?.parentElement;
        const firstCell = panel.querySelector(".llm-memory-table tbody td:first-child");
        const style = firstCell ? getComputedStyle(firstCell) : null;
        const rgb = (value) => (value.match(/[\d.]+/g) || []).slice(0, 3).map(Number);
        const luminance = (value) => {
          const [r, g, b] = rgb(value).map((item) => {
            const channel = item / 255;
            return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
          });
          return 0.2126 * r + 0.7152 * g + 0.0722 * b;
        };
        const foreground = luminance(style?.color || "rgb(0,0,0)");
        const background = luminance(style?.backgroundColor || "rgb(255,255,255)");
        const contrast = (Math.max(foreground, background) + 0.05) / (Math.min(foreground, background) + 0.05);
        return {
          documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
          panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
          bandCount: panel.querySelectorAll(".llm-contract-band").length,
          tableCount: panel.querySelectorAll(".llm-contract-table").length,
          encodingRows: panel.querySelectorAll(".llm-encoding-table tbody tr").length,
          memoryRows: panel.querySelectorAll(".llm-memory-table tbody tr").length,
          placementRows: panel.querySelectorAll(".llm-memory-placement-table tbody tr").length,
          feasibilityState: panel.querySelector(".llm-device-feasibility")?.dataset.state,
          offloadSummary: panel.querySelector(".llm-device-feasibility-runtime")?.textContent || "",
          memoryScrollOverflow: memoryScroll ? memoryScroll.scrollWidth - memoryScroll.clientWidth : 0,
          stickyPosition: style?.position,
          contrast,
          text: panel.innerText,
        };
      });
      if (state.documentOverflow > 1 || state.panelOverflow > 1 || state.bandCount < 5 || state.tableCount < 3
        || state.encodingRows !== 2 || state.memoryRows !== 2 || state.placementRows !== 2 || state.stickyPosition !== "sticky" || state.contrast < 4.5
        || state.feasibilityState !== "lower_bounds_do_not_exceed_fit_unresolved"
        || !state.offloadSummary.includes("Offload manifest: not artifact-bound")
        || !state.text.toLowerCase().includes("fit claim") || !state.text.includes("not emitted") || !state.text.includes("First tier not exceeded")
        || !state.text.includes("Conditional CPU / accelerator memory placement")
        || (width < 500 && state.memoryScrollOverflow <= 0)) {
        throw new Error(`LLM viewer geometry or evidence failed for ${theme}/${width}: ${JSON.stringify(state)}`);
      }
      if (width < 500) {
        const scroll = page.locator(".llm-memory-table").locator("..");
        await scroll.evaluate((node) => { node.scrollLeft = node.scrollWidth; });
        const positions = await page.evaluate(() => ({
          scrollLeft: document.querySelector(".llm-memory-table")?.parentElement?.scrollLeft || 0,
          containerLeft: document.querySelector(".llm-memory-table")?.parentElement?.getBoundingClientRect().left || 0,
          stickyLeft: document.querySelector(".llm-memory-table tbody td:first-child")?.getBoundingClientRect().left || 0,
        }));
        if (positions.scrollLeft <= 0 || Math.abs(positions.containerLeft - positions.stickyLeft) > 2) {
          throw new Error(`LLM sticky memory context failed for ${theme}/${width}: ${JSON.stringify(positions)}`);
        }
      }
    }
  }
  await page.locator(".llm-device-feasibility select").selectOption({ label: "4 GiB" });
  const constrainedFeasibility = await page.locator(".llm-device-feasibility").evaluate((panel) => ({
    state: panel.dataset.state,
    text: panel.innerText,
  }));
  if (constrainedFeasibility.state !== "insufficient_under_assumption"
    || !constrainedFeasibility.text.includes("1/2 lower-bound scenarios exceed 4 GiB")
    || !constrainedFeasibility.text.includes("Insufficient only under the stated")) {
    throw new Error(`LLM capacity comparison failed closed incorrectly: ${JSON.stringify(constrainedFeasibility)}`);
  }
  await page.evaluate(() => {
    const contract = structuredClone(window.__llmFixtureContract);
    const exact = (decimal) => ({ value: Number(decimal), decimal: String(decimal) });
    contract.runtime_contract = {
      status: "artifact_bound_declared_runtime",
      weight_residency: {
        cpu_bytes: exact(1073741824),
        accelerator_bytes: exact(2147483648),
        unresident_bytes: exact(0),
      },
      layer_placement: { cpu_layer_count: 4, accelerator_layer_count: 28, unresident_layer_count: 0 },
      state_cache: { resident_bytes: exact(67108864), allocated_bytes: exact(134217728) },
    };
    window.__renderLlmFixture(contract);
  });
  const boundOffloadSummary = await page.locator(".llm-device-feasibility-runtime").innerText();
  if (!boundOffloadSummary.includes("artifact_bound_declared_runtime")
    || !boundOffloadSummary.includes("1.00 GiB / 2.00 GiB / 0 B")
    || !boundOffloadSummary.includes("layers 4 / 28 / 0")
    || !boundOffloadSummary.includes("64.00 MiB / 128.00 MiB")) {
    throw new Error(`Artifact-bound offload summary is incomplete: ${boundOffloadSummary}`);
  }
  await page.evaluate(() => {
    const contract = structuredClone(window.__llmFixtureContract);
    contract.format = "onnx";
    contract.serialized_graph = {
      schema: "deepbom.serialized_llm_graph_evidence.v1",
      status: "assessed_serialized_llm_signals",
      evidence_class: "OBSERVED/DERIVED",
      graph_op_count: 4,
      explicit_operator_count: 1,
      explicit_operators: [{ name: "Attention", op_index: 3, domain: "com.microsoft", version: 1 }],
      primitive_counts: { matrix_multiply: 2, softmax: 1, normalization: 1, embedding_gather: 0 },
      transformer_motif_candidate: true,
      external_state_candidate_count: 1,
      external_state_candidates: [{ name: "past_key_values.0.key", dtype: "FLOAT16", shape: [1, 8, 32, 64], logical_bytes_if_static: { value: 32768, decimal: "32768" }, classification: "name_and_static_shape_candidate_only" }],
      graph_signature_sha256: "c".repeat(64),
      interpretation_boundary: "Serialized graph signals do not establish a decoder architecture, KV-cache semantics, generation behavior, or runtime placement.",
    };
    contract.tensorrt_llm = {
      schema: "deepbom.tensorrt_llm_static_deployment_contract.v1",
      status: "artifact_bound_configuration",
      evidence_class: "OBSERVED_CONFIG/SOURCE_PINNED/ARTIFACT_BOUND/DERIVED",
      engine_config: { path: "tensorrt_llm_engine_config.json", sha256: "d".repeat(64) },
      artifact_binding: { status: "artifact_bound", source_artifact_sha256: "e".repeat(64) },
      parallelism: { world_size: 4, tensor_parallel_size: 2, pipeline_parallel_size: 2, context_parallel_size: 1, layer_partition_per_pipeline_rank: [16, 16] },
      build_limits: { max_input_length: 2048, max_sequence_length: 8192, maximum_batch_size: 2, weight_streaming: false },
      quantization: { weight_activation_algorithm: "W4A16_AWQ", kv_cache_algorithm: "FP8" },
      kv_cache_scenario: { logical_bytes: { value: 1073741824, decimal: "1073741824" } },
      interpretation_boundary: "Static configuration and binding only; engine tactics, occupancy, runtime allocation, throughput, and latency are not established.",
    };
    window.__renderLlmFixture(contract);
  });
  for (const theme of ["light", "dark"]) {
    await page.evaluate((value) => { document.documentElement.dataset.theme = value; }, theme);
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({ width, height: width < 500 ? 844 : 1000 });
      const state = await page.locator("#llmEvidencePanel").evaluate((panel) => ({
        documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
        panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
        text: panel.innerText,
        tableCount: panel.querySelectorAll(".llm-contract-table").length,
      }));
      if (state.documentOverflow > 1 || state.panelOverflow > 1 || state.tableCount < 5
        || !state.text.includes("Serialized graph evidence") || !state.text.includes("Attention")
        || !state.text.includes("candidate; not architecture proof")
        || !state.text.includes("TensorRT-LLM static deployment contract")
        || !state.text.includes("Per-rank weight bytes") || !state.text.includes("not inferred")) {
        throw new Error(`ONNX/TensorRT-LLM viewer evidence failed for ${theme}/${width}: ${JSON.stringify(state)}`);
      }
    }
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/index.html`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#sampleEvidenceGlance")?.childElementCount > 0);
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
    await page.locator("#agreementBackdrop").waitFor({ state: "hidden" });
  }
  await page.locator("#sampleModelSelect").selectOption("gguf-tinymqa-q4");
  await page.locator("#trySampleModel").click();
  await page.waitForFunction(() => document.body.dataset.modelFormat === "gguf"
    && /audit run complete|audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
  const liveStatus = await page.locator("#status").innerText();
  if (/audit failed/i.test(liveStatus)) throw new Error(`Live GGUF sample audit failed: ${liveStatus}`);
  const liveTabState = await page.locator('[data-audit-tab="llm"]').evaluate((tab) => ({
    hidden: tab.hidden,
    formatScope: tab.dataset.formatScope,
    bodyFormat: document.body.dataset.modelFormat,
    display: getComputedStyle(tab).display,
    activeAuditTab: document.querySelector(".audit-tab.active")?.dataset.auditTab || null,
  }));
  if (liveTabState.hidden || liveTabState.display === "none" || liveTabState.bodyFormat !== "gguf") {
    throw new Error(`Live GGUF LLM tab visibility failed: ${JSON.stringify(liveTabState)}`);
  }
  await page.locator('[data-audit-tab="llm"]').click();
  await page.locator("#llmEvidencePanel:not([hidden])").waitFor({ timeout: 10_000 });
  const integrated = await page.locator("#llmEvidencePanel").evaluate((panel) => ({
    encodingRows: panel.querySelectorAll(".llm-encoding-table tbody tr").length,
    memoryRows: panel.querySelectorAll(".llm-memory-table tbody tr").length,
    text: panel.innerText,
    documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
  }));
  if (integrated.encodingRows < 1 || integrated.memoryRows < 1 || integrated.documentOverflow > 1
    || !integrated.text.includes("Memory feasibility boundary") || !integrated.text.includes("not emitted")) {
    throw new Error(`Integrated GGUF LLM viewer failed: ${JSON.stringify(integrated)}`);
  }
  await page.locator("#sampleModelSelect").selectOption("onnx-tiny-decoder-llm");
  await page.locator("#trySampleModel").click();
  await page.waitForFunction(() => document.body.dataset.modelFormat === "onnx"
    && /audit run complete|audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
  const onnxStatus = await page.locator("#status").innerText();
  if (/audit failed/i.test(onnxStatus)) throw new Error(`Live ONNX LLM sample audit failed: ${onnxStatus}`);
  await page.locator('[data-audit-tab="llm"]').click();
  await page.locator("#llmEvidencePanel:not([hidden])").waitFor({ timeout: 10_000 });
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: width < 500 ? 844 : 1000 });
    const onnxIntegrated = await page.locator("#llmEvidencePanel").evaluate((panel) => ({
      hidden: panel.hidden,
      documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
      text: panel.innerText,
      explicitRows: panel.querySelectorAll(".llm-contract-table tbody tr").length,
      formatCapability: document.querySelector("#formatCapabilityPanel")?.textContent || "",
    }));
    if (onnxIntegrated.hidden || onnxIntegrated.documentOverflow > 1 || onnxIntegrated.panelOverflow > 1
      || onnxIntegrated.explicitRows < 3 || !onnxIntegrated.text.includes("Serialized graph evidence")
      || !onnxIntegrated.text.includes("8 / 1 / 2 / 1") || !onnxIntegrated.text.includes("present_key")
      || !onnxIntegrated.text.includes("candidate; not architecture proof")
      || !onnxIntegrated.formatCapability.includes("18/18 conditionally eligible")) {
      throw new Error(`Integrated ONNX LLM/TensorRT evidence failed at ${width}px: ${JSON.stringify(onnxIntegrated)}`);
    }
  }
  if (errors.length) throw new Error(`LLM viewer browser errors: ${errors.join(" | ")}`);
  console.log("On-device LLM viewer check passed (real GGUF and ONNX/TensorRT samples, serialized-graph and TensorRT-LLM evidence, light/dark, desktop/mobile, sticky context, contrast, and overflow). ");
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}
