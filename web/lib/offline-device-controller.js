const OFFLINE_TARGETS = [
  { value: "rpi4_a72", label: "RPi4 / Cortex-A72" },
  { value: "android_mid_a55", label: "Android mid-range / Cortex-A55" },
  { value: "zynq_ultrascale_plus_a53", label: "Zynq UltraScale+ MPSoC / Cortex-A53" },
  { value: "x86_avx2", label: "x86 / AVX2" },
  { value: "generic_x86_64", label: "Generic x86-64" },
];

export function createOfflineDeviceController({
  registryList,
  refreshButton,
  modelList,
  status,
  getTargetProfiles,
  getStructure,
  getAnalysis,
  getFilename,
  queueTarget,
  fetchDevices = defaultFetchDevices,
  doc = document,
}) {
  async function loadRegistry() {
    if (!registryList) return;
    registryList.replaceChildren(messageNode(doc, "Loading devices..."));
    try {
      const data = await fetchDevices();
      const devices = Array.isArray(data?.devices) ? data.devices : [];
      if (!devices.length) {
        registryList.replaceChildren(messageNode(doc, "No devices registered yet. Start agent.py on a real device to register it."));
        return;
      }
      registryList.replaceChildren(...devices.map((device) => deviceCard(doc, device, getTargetProfiles?.() || [])));
    } catch (error) {
      registryList.replaceChildren(messageNode(doc, `Failed to load devices: ${error?.message || "unknown error"}`));
    }
  }

  function loadModels() {
    if (!modelList || !status) return;
    status.hidden = true;
    const structure = getStructure?.();
    const analysis = getAnalysis?.();
    const fingerprint = String(structure?.fingerprint || "");
    if (!fingerprint || !analysis) {
      modelList.replaceChildren(emptyModelNode(doc, "Run Static Audit first to enable offline benchmarking."));
      return;
    }

    const wrapper = doc.createElement("div");
    wrapper.className = "offline-model-card";
    wrapper.append(modelInfo(doc, getFilename?.() || "model", structure, fingerprint));

    const form = doc.createElement("div");
    form.className = "offline-target-form";
    const checkboxes = OFFLINE_TARGETS.map((target) => targetCheckbox(doc, target));
    const checkWrap = doc.createElement("div");
    checkWrap.className = "offline-target-checks";
    checkWrap.append(...checkboxes.map((row) => row.element));

    const submit = doc.createElement("button");
    submit.type = "button";
    submit.className = "offline-submit-btn";
    submit.textContent = "Submit Selected";
    submit.addEventListener("click", async () => {
      const selected = checkboxes.filter((row) => row.input.checked).map((row) => row.value);
      if (!selected.length) {
        showStatus(status, "Select at least one target device.");
        return;
      }
      submit.disabled = true;
      submit.textContent = "Queuing...";
      status.hidden = true;
      const results = [];
      try {
        for (const target of selected) {
          try {
            await queueTarget(fingerprint, target);
            results.push(`${target}: queued`);
          } catch (error) {
            results.push(`${target}: ${error?.message || "failed"}`);
          }
        }
      } finally {
        submit.textContent = "Submit Selected";
        submit.disabled = false;
        showStatus(status, results.join(" | "));
        for (const row of checkboxes) row.input.checked = false;
      }
    });

    form.append(checkWrap, submit);
    wrapper.append(form);
    modelList.replaceChildren(wrapper);
  }

  refreshButton?.addEventListener("click", loadRegistry);
  return { loadRegistry, loadModels };
}

export function formatOfflineMacs(value) {
  const count = Number(value);
  if (!Number.isFinite(count) || count < 0) return "?";
  if (count >= 1e9) return `${(count / 1e9).toFixed(2)}G`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`;
  return `${(count / 1e3).toFixed(0)}K`;
}

export function relativeDeviceTime(seconds) {
  const elapsed = Number(seconds);
  if (!Number.isFinite(elapsed) || elapsed < 0) return "unknown";
  if (elapsed < 60) return `${Math.floor(elapsed)}s ago`;
  if (elapsed < 3600) return `${Math.floor(elapsed / 60)}m ago`;
  return `${Math.floor(elapsed / 3600)}h ago`;
}

async function defaultFetchDevices() {
  const response = await fetch("/api/devices", { credentials: "same-origin" });
  if (!response.ok) throw new Error(`Device registry returned HTTP ${response.status}.`);
  return response.json();
}

function deviceCard(doc, device, targetProfiles) {
  const card = doc.createElement("div");
  const state = ["active", "recent", "offline"].includes(device?.status) ? device.status : "offline";
  card.className = `device-card status-${state}`;
  const dot = doc.createElement("span");
  dot.className = "device-status-dot";
  dot.title = state;
  const name = doc.createElement("strong");
  name.className = "device-name";
  name.textContent = String(device?.label || "Unnamed device");
  const target = doc.createElement("span");
  target.className = "device-target";
  target.textContent = targetProfiles.find((profile) => profile.id === device?.target_id)?.label
    || String(device?.target_id || "Unknown target");
  const meta = doc.createElement("span");
  meta.className = "device-meta";
  const capabilities = device?.capabilities || {};
  meta.textContent = [
    state === "active" ? "Active" : state === "recent" ? "Recent" : "Offline",
    device?.last_seen_ms ? relativeDeviceTime(device.last_seen_ago_s) : "never",
    capabilities.os || "",
    capabilities.benchmark_model ? "benchmark model" : "",
  ].filter(Boolean).join(" | ");
  card.append(dot, name, target, meta);
  return card;
}

function modelInfo(doc, filename, structure, fingerprint) {
  const info = doc.createElement("div");
  info.className = "offline-model-info";
  const name = doc.createElement("div");
  name.className = "offline-model-name";
  name.textContent = filename;
  const meta = doc.createElement("div");
  meta.className = "offline-model-meta";
  const values = [
    ["", String(structure?.format || "tflite").toUpperCase()],
    ["", `${structure?.op_count ?? "?"} ops`],
    ["", `${formatOfflineMacs(structure?.total_macs)} MACs`],
    ["mono", `${fingerprint.slice(0, 18)}...`],
  ];
  for (const [className, text] of values) {
    const item = doc.createElement("span");
    if (className) item.className = className;
    item.textContent = text;
    meta.append(item);
  }
  info.append(name, meta);
  return info;
}

function targetCheckbox(doc, { value, label }) {
  const element = doc.createElement("label");
  element.className = "offline-target-row";
  const input = doc.createElement("input");
  input.type = "checkbox";
  input.id = `offline-target-${value}`;
  input.value = value;
  const text = doc.createElement("span");
  text.textContent = label;
  element.append(input, text);
  return { element, input, value };
}

function messageNode(doc, text) {
  const node = doc.createElement("span");
  node.className = "device-empty";
  node.textContent = text;
  return node;
}

function emptyModelNode(doc, text) {
  const node = doc.createElement("p");
  node.className = "offline-test-empty";
  node.textContent = text;
  return node;
}

function showStatus(status, text) {
  status.textContent = text;
  status.hidden = false;
}
