const MCP_APP_PROTOCOL_VERSION = "2026-01-26";
const DEFAULT_TIMEOUT_MS = 30_000;

export class McpAppClient {
  constructor(appInfo, { timeoutMs = DEFAULT_TIMEOUT_MS, autoResize = true } = {}) {
    this.appInfo = Object.freeze({ name: String(appInfo.name), version: String(appInfo.version) });
    this.timeoutMs = timeoutMs;
    this.autoResize = autoResize;
    this.parent = window.parent;
    this.nextId = 0;
    this.pending = new Map();
    this.toolInputHandler = null;
    this.connected = false;
    this.messageHandler = (event) => this.receive(event);
  }

  set ontoolinput(handler) {
    this.toolInputHandler = typeof handler === "function" ? handler : null;
  }

  async connect() {
    if (this.connected) throw new Error("The MCP App client is already connected.");
    if (!this.parent || this.parent === window) throw new Error("The MCP App host frame is unavailable.");
    window.addEventListener("message", this.messageHandler);
    try {
      const initialized = await this.request("ui/initialize", {
        appCapabilities: {},
        appInfo: this.appInfo,
        protocolVersion: MCP_APP_PROTOCOL_VERSION,
      });
      if (!initialized || typeof initialized !== "object") throw new Error("The MCP App host returned an invalid initialize result.");
      this.hostInfo = initialized.hostInfo || null;
      this.hostCapabilities = initialized.hostCapabilities || {};
      this.hostContext = initialized.hostContext || {};
      this.notify("ui/notifications/initialized", {});
      this.connected = true;
      if (this.autoResize) this.startResizeObserver();
      return initialized;
    } catch (error) {
      this.close();
      throw error;
    }
  }

  callServerTool(params) {
    if (!params || typeof params.name !== "string") throw new Error("callServerTool requires a tool name.");
    return this.request("tools/call", params);
  }

  updateModelContext(params) {
    return this.request("ui/update-model-context", params);
  }

  close() {
    window.removeEventListener("message", this.messageHandler);
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    this.connected = false;
    for (const { reject, timer } of this.pending.values()) {
      clearTimeout(timer);
      reject(new Error("The MCP App client closed before the host responded."));
    }
    this.pending.clear();
  }

  request(method, params) {
    const id = `deepbom-${++this.nextId}`;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`The MCP App host did not respond to ${method}.`));
      }, this.timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      this.post({ jsonrpc: "2.0", id, method, params });
    });
  }

  notify(method, params) {
    this.post({ jsonrpc: "2.0", method, params });
  }

  post(message) {
    this.parent.postMessage(message, "*");
  }

  receive(event) {
    if (event.source !== this.parent) return;
    const message = event.data;
    if (!message || message.jsonrpc !== "2.0") return;
    if (Object.hasOwn(message, "id") && !message.method) {
      const pending = this.pending.get(String(message.id));
      if (!pending) return;
      this.pending.delete(String(message.id));
      clearTimeout(pending.timer);
      if (message.error) pending.reject(new Error(String(message.error.message || "The MCP App host returned an error.")));
      else pending.resolve(message.result);
      return;
    }
    if (message.method === "ui/notifications/tool-input") {
      this.toolInputHandler?.(message.params || {});
      return;
    }
    if (Object.hasOwn(message, "id") && message.method === "ping") {
      this.post({ jsonrpc: "2.0", id: message.id, result: {} });
      return;
    }
    if (Object.hasOwn(message, "id") && message.method === "ui/resource-teardown") {
      this.post({ jsonrpc: "2.0", id: message.id, result: {} });
      this.close();
    }
  }

  startResizeObserver() {
    if (typeof ResizeObserver !== "function") return;
    let previousWidth = -1;
    let previousHeight = -1;
    const report = () => {
      const width = Math.ceil(document.documentElement.scrollWidth);
      const height = Math.ceil(document.documentElement.scrollHeight);
      if (width === previousWidth && height === previousHeight) return;
      previousWidth = width;
      previousHeight = height;
      this.notify("ui/notifications/size-changed", { width, height });
    };
    this.resizeObserver = new ResizeObserver(report);
    this.resizeObserver.observe(document.documentElement);
    this.resizeObserver.observe(document.body);
    report();
  }
}

export const MCP_APP_CLIENT_CONTRACT = Object.freeze({
  protocolVersion: MCP_APP_PROTOCOL_VERSION,
  methods: Object.freeze([
    "ui/initialize",
    "ui/notifications/initialized",
    "ui/notifications/tool-input",
    "tools/call",
    "ui/update-model-context",
    "ui/notifications/size-changed",
    "ui/resource-teardown",
  ]),
});
