let appPromise = null;
let appLoaded = false;
let replayingFileChange = false;
let replayingInteraction = false;

function loadApp() {
  appPromise ||= import("./app.js").then((module) => {
    appLoaded = true;
    return module;
  });
  return appPromise;
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
