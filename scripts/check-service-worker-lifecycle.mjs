import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

import { registerServiceWorker } from "../web/lib/service-worker.js";

const workerSource = await readFile("web/sw.js", "utf8");
const currentCacheName = /const CACHE_NAME = "([^"]+)";/.exec(workerSource)?.[1];
assert(currentCacheName, "Service worker must declare CACHE_NAME.");
const currentCacheVersion = Number(/-v(\d+)$/.exec(currentCacheName)?.[1]);
assert(Number.isSafeInteger(currentCacheVersion), "Service-worker cache version must be numeric.");

await existingControllerReplacementReloadsOnce();
await initialControllerClaimDoesNotReload();
await priorAppCacheMigrationNavigatesExistingWindows();
await currentCacheActivationDoesNotNavigateWindows();

console.log("Service-worker lifecycle check passed (atomic controller replacement and first-install stability).");

async function existingControllerReplacementReloadsOnce() {
  const oldController = { id: "old" };
  const nextController = { id: "next" };
  const harness = createHarness(oldController);
  registerServiceWorker(harness.options);
  await settle();

  assert.equal(harness.registerCalls, 1);
  assert.equal(harness.updateCalls, 1);
  assert.deepEqual(harness.registerOptions, { updateViaCache: "none" });
  harness.setController(nextController);
  harness.dispatchServiceWorker("controllerchange");
  harness.dispatchServiceWorker("controllerchange");
  assert.equal(harness.reloadCalls, 1, "A replacement controller must reload exactly once.");
  assert.equal(harness.infoCalls, 1);
}

async function initialControllerClaimDoesNotReload() {
  const harness = createHarness(null);
  registerServiceWorker(harness.options);
  await settle();

  harness.setController({ id: "first" });
  harness.dispatchServiceWorker("controllerchange");
  assert.equal(harness.reloadCalls, 0, "The first controller claim must not reload a fresh page.");
}

async function priorAppCacheMigrationNavigatesExistingWindows() {
  const result = await runWorkerActivation([
    "tflite-wasm-static-audit-v438-old-build",
    `tflite-wasm-static-audit-v${currentCacheVersion - 1}-previous-build`,
    currentCacheName,
  ]);
  assert.deepEqual(result.deleted, [
    "tflite-wasm-static-audit-v438-old-build",
    `tflite-wasm-static-audit-v${currentCacheVersion - 1}-previous-build`,
  ]);
  assert.equal(result.claims, 1);
  assert.equal(result.matchAllCalls, 1);
  assert.deepEqual(result.navigated, ["https://deepbom.org/web/"]);
}

async function currentCacheActivationDoesNotNavigateWindows() {
  const result = await runWorkerActivation([currentCacheName]);
  assert.deepEqual(result.deleted, []);
  assert.equal(result.claims, 1);
  assert.equal(result.matchAllCalls, 0);
  assert.deepEqual(result.navigated, []);
}

async function runWorkerActivation(cacheKeys) {
  const listeners = new Map();
  const deleted = [];
  const navigated = [];
  let claims = 0;
  let matchAllCalls = 0;
  const context = {
    URL,
    Promise,
    console,
    caches: {
      async keys() { return [...cacheKeys]; },
      async delete(key) { deleted.push(key); return true; },
      async open() { throw new Error("install cache should not run in activation test"); },
      async match() { return null; },
    },
    self: {
      addEventListener(type, listener) { listeners.set(type, listener); },
      skipWaiting() {},
      clients: {
        async claim() { claims += 1; },
        async matchAll(options) {
          matchAllCalls += 1;
          assert.equal(options.type, "window");
          assert.equal(options.includeUncontrolled, true);
          return [{
            url: "https://deepbom.org/web/",
            async navigate(url) { navigated.push(url); },
          }];
        },
      },
    },
  };
  vm.runInNewContext(workerSource, context, { filename: "web/sw.js" });
  const activate = listeners.get("activate");
  assert(activate, "Service worker must install an activate handler.");
  let completion;
  activate({ waitUntil(promise) { completion = promise; } });
  await completion;
  return { deleted, navigated, claims, matchAllCalls };
}

function createHarness(initialController) {
  const windowListeners = new Map();
  const serviceWorkerListeners = new Map();
  const state = {
    registerCalls: 0,
    registerOptions: null,
    updateCalls: 0,
    reloadCalls: 0,
    infoCalls: 0,
  };
  const serviceWorker = {
    controller: initialController,
    addEventListener(type, listener) {
      serviceWorkerListeners.set(type, listener);
    },
    async register(url, options) {
      state.registerCalls += 1;
      assert.equal(url, "./sw.js");
      state.registerOptions = options;
      return {
        async update() {
          state.updateCalls += 1;
        },
      };
    },
  };
  const harness = {
    options: {
      windowRef: {
        addEventListener(type, listener) {
          windowListeners.set(type, listener);
        },
        location: {
          reload() {
            state.reloadCalls += 1;
          },
        },
      },
      navigatorRef: { serviceWorker },
      log: {
        info() {
          state.infoCalls += 1;
        },
        warn() {},
      },
    },
    dispatchWindow(type) {
      windowListeners.get(type)?.();
    },
    dispatchServiceWorker(type) {
      serviceWorkerListeners.get(type)?.();
    },
    setController(controller) {
      serviceWorker.controller = controller;
    },
  };
  Object.defineProperties(harness, {
    registerCalls: { get: () => state.registerCalls },
    registerOptions: { get: () => state.registerOptions },
    updateCalls: { get: () => state.updateCalls },
    reloadCalls: { get: () => state.reloadCalls },
    infoCalls: { get: () => state.infoCalls },
  });
  return harness;
}

async function settle() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}
