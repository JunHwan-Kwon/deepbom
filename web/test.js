const ACCESS_STORAGE_KEY = "deepbom.external-test.link.v2";
const status = document.getElementById("testAccessStatus");
const title = document.getElementById("testAccessStatusTitle");
const identity = document.getElementById("testAccessIdentity");
const activate = document.getElementById("testActivate");
const openWorkbench = document.getElementById("testOpenWorkbench");
const deactivate = document.getElementById("testDeactivate");

let access = captureAccess();
let currentUser = null;

activate.addEventListener("click", activateTestAccess);
deactivate.addEventListener("click", deactivateTestAccess);

init();

async function init() {
  try {
    if (access) {
      await activateTestAccess();
      return;
    }
    const session = await apiFetch("/api/auth/me");
    currentUser = session.user || null;
    render();
  } catch (error) {
    setStatus(error.message || "External test access is unavailable.", "error");
    activate.hidden = !access;
  }
}

function render() {
  const active = Boolean(currentUser?.test_access?.active);
  activate.hidden = true;
  openWorkbench.hidden = !active;
  deactivate.hidden = !active;

  if (active) {
    title.textContent = "Test access active";
    identity.textContent = `${currentUser.name || "External Tester"} / expires ${formatDate(currentUser.test_access.expires_at)}`;
    setStatus("All product evaluation capabilities are enabled. Admin operations remain unavailable.", "ok");
    return;
  }
  title.textContent = "Access link required";
  identity.textContent = "No active external test session is present in this browser.";
  setStatus("Open the private 24-hour link supplied by the DEEPBOM administrator.", "error");
}

function captureAccess() {
  const token = new URLSearchParams(location.hash.slice(1)).get("access") || "";
  if (token) {
    try { sessionStorage.setItem(ACCESS_STORAGE_KEY, token); } catch { /* tab storage is best-effort */ }
    history.replaceState(null, "", "/test");
    return token;
  }
  try { return sessionStorage.getItem(ACCESS_STORAGE_KEY) || ""; } catch { return ""; }
}

async function activateTestAccess() {
  activate.disabled = true;
  title.textContent = "Activating test access";
  identity.textContent = "Validating the private link in this browser.";
  setStatus("Creating the time-limited evaluation session.");
  try {
    const result = await apiFetch("/api/test/activate", {
      method: "POST",
      body: JSON.stringify({ access }),
    });
    clearAccess();
    location.replace(result.redirect_to || "/web/?test=active");
  } catch (error) {
    setStatus(error.message || "Test access activation failed.", "error");
    activate.hidden = false;
    activate.disabled = false;
  }
}

async function deactivateTestAccess() {
  deactivate.disabled = true;
  try {
    await apiFetch("/api/test/deactivate", { method: "POST", body: "{}" });
    clearAccess();
    currentUser = null;
    render();
  } catch (error) {
    setStatus(error.message || "The test session could not be ended.", "error");
    deactivate.disabled = false;
  }
}

function clearAccess() {
  access = "";
  try { sessionStorage.removeItem(ACCESS_STORAGE_KEY); } catch { /* no-op */ }
}

function setStatus(message, tone = "") {
  status.textContent = message;
  status.className = `status ${tone}`.trim();
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "unknown" : date.toLocaleString();
}

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "content-type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) throw new Error(data.message || data.error || `Request failed: ${response.status}`);
  return data;
}
