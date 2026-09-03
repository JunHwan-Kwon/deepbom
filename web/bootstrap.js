let appPromise = null;
let appLoaded = false;
let appLoadAttempt = 0;
let replayingFileChange = false;
let replayingInteraction = false;

const MAX_APP_LOAD_ATTEMPTS = 3;
const APP_LOAD_RETRY_DELAYS_MS = [0, 150, 600];
const APP_RELOAD_SESSION_KEY = "deepbom.bootstrap.module-reload.v1";
const APP_RELOAD_COOLDOWN_MS = 30_000;

function loadApp() {
  appPromise ||= importApplication(appLoadAttempt).then((module) => {
    appLoaded = true;
    clearReloadMarker();
    return module;
  }).catch(async (error) => {
    appPromise = null;
    appLoadAttempt += 1;
    if (!isTransientModuleFetchFailure(error)) throw error;
    if (appLoadAttempt < MAX_APP_LOAD_ATTEMPTS) {
      await delay(APP_LOAD_RETRY_DELAYS_MS[appLoadAttempt] || 0);
      return loadApp();
    }
    if (requestBoundedDocumentReload()) return new Promise(() => {});
    throw error;
  });
  return appPromise;
}

function importApplication(attempt) {
  return attempt === 0
    ? import("./app.js")
    : import(`./app.js?bootstrap-retry=${attempt}`);
}

function isTransientModuleFetchFailure(error) {
  return error instanceof TypeError
    && /failed to fetch dynamically imported module|error loading dynamically imported module|importing a module script failed/i.test(String(error.message || error));
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function requestBoundedDocumentReload() {
  try {
    const previous = Number(window.sessionStorage.getItem(APP_RELOAD_SESSION_KEY));
    if (Number.isFinite(previous) && Date.now() - previous < APP_RELOAD_COOLDOWN_MS) return false;
    window.sessionStorage.setItem(APP_RELOAD_SESSION_KEY, String(Date.now()));
    window.location.reload();
    return true;
  } catch {
    return false;
  }
}

function clearReloadMarker() {
  try {
    window.sessionStorage.removeItem(APP_RELOAD_SESSION_KEY);
  } catch {
    // Storage may be unavailable in restricted browser contexts.
  }
}

function requestsApplication(target) {
  return target instanceof Element && Boolean(target.closest(
    "#fileInput, #artifactBundleInput, #runAudit, #trySampleModel, #dropzone, [data-sample-id], [data-target-id], [data-module]",
  ));
}

function startForInteraction(event) {
  if (requestsApplication(event.target)) loadApp().catch(() => {});
}

async function replayFileChange(event) {
  if (replayingFileChange || appLoaded || !["fileInput", "artifactBundleInput"].includes(event.target?.id)) return;
  event.stopImmediatePropagation();
  await loadApp();
  replayingFileChange = true;
  event.target.dispatchEvent(new Event("change", { bubbles: true }));
  replayingFileChange = false;
}

async function replayCommand(event) {
  if (replayingInteraction || appLoaded || !requestsApplication(event.target)) return;
  if (event.target.closest("#fileInput, #artifactBundleInput, label[for='fileInput'], label[for='artifactBundleInput']")) return;
  const command = event.target.closest("button, [role='button'], [data-sample-id], [data-target-id], [data-module]");
  if (!command) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  await loadApp();
  replayingInteraction = true;
  command.click();
  replayingInteraction = false;
}

document.addEventListener("pointerdown", startForInteraction, { capture: true, passive: true });
document.addEventListener("focusin", startForInteraction, { capture: true, passive: true });
document.addEventListener("change", replayFileChange, { capture: true });
document.addEventListener("click", replayCommand, { capture: true });

const schedule = window.requestIdleCallback
  ? (callback) => window.requestIdleCallback(callback, { timeout: 1_500 })
  : (callback) => window.setTimeout(callback, 250);

schedule(() => loadApp().catch((error) => {
  console.error("[bootstrap]", error);
  const status = document.getElementById("status");
  if (status) status.textContent = "Application failed to initialize";
}));
