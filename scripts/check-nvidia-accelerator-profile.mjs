import assert from "node:assert/strict";
import { createHash } from "node:crypto";

import { collectNvidiaAcceleratorProfile, parseDeviceRows } from "../bin/nvidia-accelerator-collector.mjs";
import { validateNvidiaAcceleratorProfile } from "../web/lib/nvidia-accelerator-profile.js";
import { buildNvidiaAcceleratorProfileBinding, validateNvidiaAcceleratorProfileBinding } from "../web/lib/accelerator-profile-binding.js";

const fixedTime = "2026-08-30T00:00:00.000Z";
const inventory = [
  '0,"NVIDIA Test, GPU",GPU-0000,00000000:01:00.0,581.86,8188,8.9,3105,8001',
  "1,NVIDIA Headless GPU,GPU-1111,00000000:02:00.0,581.86,24576,9.0,N/A,N/A",
].join("\n");

const rows = parseDeviceRows(inventory);
assert.equal(rows.length, 2);
assert.equal(rows[0].name, "NVIDIA Test, GPU", "quoted NVIDIA CSV fields must remain intact");
assert.equal(rows[0].uuid, null, "raw UUID is private by default");
assert.equal(rows[1].maximum_sm_clock_mhz, null, "unexposed clocks remain null");
assert.equal(rows[1].unexposed_fields.includes("maximum_sm_clock_mhz"), true);

const profile = await collectNvidiaAcceleratorProfile({
  collectorVersion: "1.95.0-test",
  collectedAt: fixedTime,
  commandRunner: fakeRunner,
  executableResolver: fakeResolver,
  fileHasher: fakeHasher,
});
assert.equal(profile.devices.length, 2);
assert.equal(profile.devices[0].memory_total_bytes.decimal, String(8188n * 1024n * 1024n));
assert.equal(profile.software.cuda_driver_api.version, "13.0");
assert.equal(profile.software.cuda_toolkit.version, "12.8");
assert.equal(profile.software.tensorrt.status, "not_exposed", "a failed trtexec invocation must not become observed TensorRT");
assert.equal(profile.collection.tools.find((tool) => tool.role === "tensorrt_exec").status, "execution_failed");
assert.equal(profile.roofline_contract.theoretical_compute_ceiling, null, "collector must not invent a GPU compute roofline");
assert.equal(profile.roofline_contract.theoretical_memory_bandwidth, null, "collector must not invent GPU bandwidth");
assert.doesNotThrow(() => validateNvidiaAcceleratorProfile(profile));

const selected = await collectNvidiaAcceleratorProfile({
  collectorVersion: "1.95.0-test",
  collectedAt: fixedTime,
  deviceIndex: 1,
  includeDeviceIdentifiers: true,
  commandRunner: fakeRunner,
  executableResolver: fakeResolver,
  fileHasher: fakeHasher,
});
assert.deepEqual(selected.devices.map((device) => device.index), [1]);
assert.equal(selected.devices[0].uuid, "GPU-1111");
assert.equal(selected.devices[0].pci_bus_id, "00000000:02:00.0");

const gib = 1024n ** 3n;
const binding = buildNvidiaAcceleratorProfileBinding({
  on_device_llm: {
    memory_feasibility: {
      residency_assumption: "test simultaneous residency",
      static_scenarios: [{
        state_kind: "transformer_kv", context_length: 4096, batch_size: 1, storage_bits: 16,
        logical_state_bytes: { decimal: String(gib), value: Number(gib) },
        static_lower_bound_bytes: { decimal: String(10n * gib), value: Number(10n * gib) },
      }],
    },
    storage: {
      layer_storage: {
        status: "assessed_exact_serialized_layer_storage",
        conservation: { status: "pass" },
        non_layer_bytes: { decimal: String(gib), value: Number(gib) },
        layers: [0, 1, 2].map((layer_index) => ({
          layer_index, serialized_bytes: { decimal: String(2n * gib), value: Number(2n * gib) },
        })),
      },
    },
  },
}, { document: profile, path: "profile.json", sha256: "a".repeat(64) }, { deviceIndex: 0 });
assert.equal(binding.selected_build.status, "not_bound", "host tooling must not imply selected-build inclusion");
assert.equal(binding.llm_accelerator_residency.scenarios[0].status, "simultaneous_residency_lower_bound_exceeds_physical_vram");
assert.equal(binding.llm_accelerator_residency.scenarios[0].fit_claim, "not_emitted");
assert.equal(binding.llm_accelerator_residency.scenarios[0].serialized_layer_offload.maximum_layer_count_not_exceeding_physical_vram_lower_bound, 2);
assert.match(binding.binding_sha256, /^[a-f0-9]{64}$/);
assert.doesNotThrow(() => validateNvidiaAcceleratorProfileBinding(binding));
const tamperedBinding = structuredClone(binding);
tamperedBinding.selected_device.memory_total_bytes.decimal = "1";
assert.throws(() => validateNvidiaAcceleratorProfileBinding(tamperedBinding));
assert.throws(() => buildNvidiaAcceleratorProfileBinding({}, { document: profile, sha256: "a".repeat(64) }), /multiple devices/);

const tampered = structuredClone(profile);
tampered.devices[0].memory_total_bytes.decimal = "1";
assert.throws(() => validateNvidiaAcceleratorProfile(tampered), /number does not match|SHA-256/, "tampered exact values must fail closed");
await assert.rejects(collectNvidiaAcceleratorProfile({
  collectorVersion: "test",
  deviceIndex: 7,
  commandRunner: fakeRunner,
  executableResolver: fakeResolver,
  fileHasher: fakeHasher,
}), /no GPU at index 7/);

console.log("NVIDIA accelerator profile checks passed (CSV, privacy, exact capacity, nullable observations, tool failure, selection, integrity, roofline boundary, and conservative LLM residency binding). ");

function fakeResolver(command) {
  return ({ "nvidia-smi": "C:/tools/nvidia-smi.exe", nvcc: "C:/tools/nvcc.exe", trtexec: "C:/tools/trtexec.exe" })[command] || null;
}

function fakeRunner(executable, args) {
  if (executable.endsWith("nvidia-smi.exe") && args[0]?.startsWith("--query-gpu=")) return { status: 0, stdout: inventory, stderr: "" };
  if (executable.endsWith("nvidia-smi.exe")) return { status: 0, stdout: "NVIDIA-SMI 581.86 CUDA Version: 13.0", stderr: "" };
  if (executable.endsWith("nvcc.exe")) return { status: 0, stdout: "Cuda compilation tools, release 12.8, V12.8.61", stderr: "" };
  if (executable.endsWith("trtexec.exe")) return { status: 1, stdout: "", stderr: "failed to initialize TensorRT" };
  return { status: 1, stdout: "", stderr: "unexpected command" };
}

async function fakeHasher(value) {
  return createHash("sha256").update(String(value), "utf8").digest("hex");
}
