export function registerServiceWorker({
  windowRef = window,
  navigatorRef = navigator,
  scriptUrl = "./sw.js",
  log = console,
} = {}) {
  if (!("serviceWorker" in navigatorRef)) {
    return;
  }

  // The worker activates immediately so a deployment can replace its cache.
  // An already-open page still has the previous JavaScript module graph in
  // memory, though, and must not continue after a new controller starts
  // serving WASM or lazy modules. Reload exactly once on an actual controller
  // replacement; the first controller claim on a fresh visit is already
  // version-consistent and does not need a reload.
  let controller = navigatorRef.serviceWorker.controller || null;
  let reloading = false;
  navigatorRef.serviceWorker.addEventListener?.("controllerchange", () => {
    const nextController = navigatorRef.serviceWorker.controller || null;
    if (!controller) {
      controller = nextController;
      return;
    }
    if (!nextController || nextController === controller || reloading) return;
    controller = nextController;
    reloading = true;
    log.info?.("Application assets updated; reloading the version-consistent module graph.");
    windowRef.location.reload();
  });

  // Start the update check as soon as the application module executes. Waiting
  // for window.load can leave an older graph viewer active while large runtime
  // assets are still loading. updateViaCache:none also prevents an HTTP cache
  // entry for sw.js from delaying the version transition.
  navigatorRef.serviceWorker.register(scriptUrl, { updateViaCache: "none" })
    .then((registration) => {
      registration.update().catch((error) => {
        log.warn("Service worker update check failed", error);
      });
    })
    .catch((error) => {
      log.warn("Service worker registration failed", error);
    });
}
