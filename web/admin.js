import { statusClass } from "./lib/auth-labels.js";
import { canonicalAccessProfile, hasAccessProfile } from "./lib/access-policy.js";
import { bindAdminElements } from "./lib/admin-elements.js";
import { copyTextToClipboard } from "./lib/clipboard.js";

const {
  adminStatus,
  adminIdentity,
  adminRefreshAll,
  adminMetrics,
  testLinkCreate,
  testLinkResult,
  testLinkUrl,
  testLinkCopy,
  testLinkMeta,
  requestStatusFilter,
  adminRequestBoard,
  userSearch,
  adminUserTable,
  benchStatusFilter,
  benchRefresh,
  adminBenchBoard,
  structuresRefresh,
  adminStructuresBoard,
} = bindAdminElements();

let currentUser = null;
let requests = [];
let users = [];
let benchRuns = [];
let modelStructures = [];

adminRefreshAll.addEventListener("click", loadAdminData);
requestStatusFilter.addEventListener("change", renderRequests);
userSearch.addEventListener("input", renderUsers);
benchStatusFilter.addEventListener("change", loadBenchmarks);
benchRefresh.addEventListener("click", loadBenchmarks);
structuresRefresh.addEventListener("click", loadStructures);
testLinkCreate.addEventListener("click", createTestLink);
testLinkCopy.addEventListener("click", copyTestLink);

initAdmin();

async function initAdmin() {
  try {
    const session = await apiFetch("/api/auth/me");
    currentUser = session.user;
    if (!currentUser) {
      setAdminStatus("Sign in required", "error");
      adminIdentity.textContent = "Open the workbench and sign in with an admin account.";
      renderBlocked("Sign in is required to view the admin console.");
      return;
    }
    if (currentUser.role !== "admin") {
      setAdminStatus("Admin required", "error");
      adminIdentity.textContent = `${currentUser.email} is signed in, but this page requires admin access.`;
      renderBlocked("Admin access is required.");
      return;
    }
    adminIdentity.textContent = `${currentUser.name || currentUser.email} / ${currentUser.email}`;
    await loadAdminData();
  } catch (error) {
    setAdminStatus("Admin load failed", "error");
    adminIdentity.textContent = error.message || "Admin console failed to load.";
    renderBlocked(error.message || "Admin console failed to load.");
  }
}

async function loadAdminData() {
  setAdminStatus("Refreshing");
  adminRefreshAll.disabled = true;
  try {
    const [requestData, userData] = await Promise.all([
      apiFetch("/api/admin/requests"),
      apiFetch("/api/admin/users"),
    ]);
    requests = requestData.requests || [];
    users = userData.users || [];
    renderMetrics();
    renderRequests();
    renderUsers();
    setAdminStatus("Ready", "ok");
  } catch (error) {
    setAdminStatus("Refresh failed", "error");
    renderBlocked(error.message || "Admin refresh failed.");
  } finally {
    adminRefreshAll.disabled = false;
  }
  loadBenchmarks();
  loadStructures();
}

async function createTestLink() {
  testLinkCreate.disabled = true;
  testLinkResult.hidden = true;
  try {
    const result = await apiFetch("/api/admin/test-links", {
      method: "POST",
      body: "{}",
    });
    testLinkUrl.value = result.access_url;
    testLinkMeta.textContent = `Expires ${formatDateTime(result.expires_at)} / automatic access / Medical AI evaluation / no Admin access`;
    testLinkResult.hidden = false;
    setAdminStatus("24-hour test link created", "ok");
  } catch (error) {
    setAdminStatus(error.message || "Test link creation failed", "error");
  } finally {
    testLinkCreate.disabled = false;
  }
}

async function copyTestLink() {
  if (!testLinkUrl.value) return;
  testLinkCopy.disabled = true;
  try {
    await copyTextToClipboard(testLinkUrl.value);
    setAdminStatus("Test link copied", "ok");
  } catch (error) {
    setAdminStatus(error.message || "Copy failed", "error");
  } finally {
    testLinkCopy.disabled = false;
  }
}

async function loadStructures() {
  adminStructuresBoard.replaceChildren(empty("Loading analyzed models…"));
  try {
    const data = await apiFetch("/api/admin/model-structures?limit=50");
    modelStructures = data.structures || [];
    renderStructures();
  } catch (error) {
    adminStructuresBoard.replaceChildren(empty(error.message || "Failed to load model structures."));
  }
}

function renderStructures() {
  if (!modelStructures.length) {
    adminStructuresBoard.replaceChildren(empty("No analyzed models yet. Run an audit with research consent enabled."));
    return;
  }
  const table = document.createElement("table");
  table.className = "admin-bench-table";
  table.innerHTML = `<thead><tr>
    <th>Fingerprint</th><th>Format</th><th>Ops</th><th>MACs</th><th>Target</th><th>Benchmarks</th><th>User</th><th>Created</th><th></th>
  </tr></thead>`;
  const tbody = document.createElement("tbody");
  for (const s of modelStructures) {
    const tr = document.createElement("tr");
    const fp = s.fingerprint || "";
    const shortFp = fp.slice(0, 12) + (fp.length > 12 ? "…" : "");
    const macs = s.total_macs != null ? fmtMacs(s.total_macs) : "—";
    const created = s.created_at ? new Date(s.created_at).toLocaleDateString() : "—";
    const fingerprintCell = td(shortFp, "mono");
    fingerprintCell.title = fp;
    tr.append(
      fingerprintCell,
      td(s.format || "—"),
      td(s.op_count ?? "—"),
      td(macs),
      td(s.target || "—"),
      td(s.bench_count ?? 0),
      td(s.user_email || "—"),
      td(created),
      td(""),
    );
    const queueBtn = document.createElement("button");
    queueBtn.className = "bench-queue-btn";
    queueBtn.textContent = "Queue";
    queueBtn.addEventListener("click", async () => {
      queueBtn.disabled = true;
      queueBtn.textContent = "Queuing…";
      try {
        await apiFetch("/api/admin/benchmarks", {
          method: "POST",
          body: JSON.stringify({ fingerprint: fp, target: s.target || "generic_x86_64" }),
        });
        queueBtn.textContent = "Queued ✓";
        await loadBenchmarks();
      } catch (err) {
        queueBtn.textContent = err.message?.slice(0, 24) || "Failed";
        queueBtn.disabled = false;
      }
    });
    tr.lastElementChild.appendChild(queueBtn);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  adminStructuresBoard.replaceChildren(table);
}

function fmtMacs(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "G";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + "K";
  return String(n);
}

async function loadBenchmarks() {
  adminBenchBoard.replaceChildren(empty("Loading benchmarks…"));
  try {
    const status = benchStatusFilter.value;
    const url = status ? `/api/admin/benchmarks?status=${encodeURIComponent(status)}` : "/api/admin/benchmarks";
    const data = await apiFetch(url);
    benchRuns = data.runs || [];
    renderBenchmarks();
  } catch (error) {
    adminBenchBoard.replaceChildren(empty(error.message || "Failed to load benchmarks."));
  }
}

function renderBenchmarks() {
  if (!benchRuns.length) {
    adminBenchBoard.replaceChildren(empty("No benchmark runs for this filter."));
    return;
  }
  const pending = benchRuns.filter((r) => r.status === "pending").length;
  const running = benchRuns.filter((r) => r.status === "running").length;
  const complete = benchRuns.filter((r) => r.status === "complete").length;
  const failed = benchRuns.filter((r) => r.status === "failed").length;

  const summary = document.createElement("div");
  summary.className = "bench-summary";
  summary.innerHTML = `<span class="bench-pill pending">${pending} pending</span>`
    + `<span class="bench-pill running">${running} running</span>`
    + `<span class="bench-pill complete">${complete} complete</span>`
    + `<span class="bench-pill failed">${failed} failed</span>`;

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const label of ["Status", "Device", "Target", "Fingerprint", "Ops", "p50", "p90", "p99", "Requested", "Completed", ""]) {
    const th = document.createElement("th");
    th.textContent = label;
    headRow.append(th);
  }
  thead.append(headRow);
  const tbody = document.createElement("tbody");
  for (const run of benchRuns) {
    tbody.append(benchRow(run));
  }
  table.append(thead, tbody);
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.append(table);
  adminBenchBoard.replaceChildren(summary, wrap);
}

function benchRow(run) {
  const tr = document.createElement("tr");
  const statusCell = document.createElement("td");
  const status = ["pending", "running", "complete", "failed"].includes(run.status) ? run.status : "unknown";
  const statusPill = document.createElement("span");
  statusPill.className = `bench-pill ${status}`;
  statusPill.textContent = status;
  statusCell.append(statusPill);
  const fp = run.fingerprint ? run.fingerprint.slice(3, 11) + "…" : "-";
  const delBtn = document.createElement("button");
  delBtn.type = "button";
  delBtn.className = "admin-table-button";
  delBtn.textContent = "Delete";
  delBtn.addEventListener("click", async () => {
    delBtn.disabled = true;
    delBtn.textContent = "…";
    try {
      await apiFetch(`/api/admin/benchmarks/${encodeURIComponent(run.id)}`, { method: "DELETE" });
      await loadBenchmarks();
    } catch (e) {
      delBtn.textContent = "Err";
      setTimeout(() => { delBtn.disabled = false; delBtn.textContent = "Delete"; }, 1200);
    }
  });
  tr.append(
    statusCell,
    td(run.device_label || run.backend || "-"),
    td(run.target || "-"),
    td(fp, "mono"),
    td(run.op_count != null ? String(run.op_count) : "-"),
    td(run.p50_ms != null ? `${run.p50_ms} ms` : "-"),
    td(run.p90_ms != null ? `${run.p90_ms} ms` : "-"),
    td(run.p99_ms != null ? `${run.p99_ms} ms` : "-"),
    td(formatDateTime(run.created_at)),
    td(run.completed_at ? formatDateTime(run.completed_at) : "-"),
    td(delBtn),
  );
  if (run.error_message) {
    tr.title = run.error_message;
    tr.classList.add("bench-row-error");
  }
  return tr;
}

function renderMetrics() {
  const newRequests = requests.filter((item) => item.status === "new").length;
  const activeUsers = users.filter((item) => item.access_status === "active").length;
  const verifiedUsers = users.filter((item) => item.email_verified_at || item.role === "admin" || item.provider === "google").length;
  const researchUsers = users.filter((item) => hasAccessProfile(canonicalAccessProfile(item.access_profile, item.role), "research")).length;
  const admins = users.filter((item) => item.role === "admin").length;
  adminMetrics.replaceChildren(
    metricCard("Requests", requests.length, `${newRequests} new`),
    metricCard("Accounts", users.length, `${activeUsers} active / ${verifiedUsers} verified`),
    metricCard("Controlled modules", researchUsers, "advanced / regulatory / admin"),
    metricCard("Admins", admins, "privileged operators"),
  );
}

function renderRequests() {
  const status = requestStatusFilter.value;
  const visible = status === "all" ? requests : requests.filter((item) => item.status === status);
  if (!visible.length) {
    adminRequestBoard.replaceChildren(empty("No requests for this filter."));
    return;
  }
  adminRequestBoard.replaceChildren(...visible.map(requestCard));
}

function renderUsers() {
  const query = userSearch.value.trim().toLowerCase();
  const visible = query
    ? users.filter((user) => [user.email, user.name, user.access_profile, user.role, user.access_status].join(" ").toLowerCase().includes(query))
    : users;
  if (!visible.length) {
    adminUserTable.replaceChildren(empty("No accounts match this search."));
    return;
  }
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const label of ["Account", "Provider", "Verified", "Role", "Access scope", "Authorization", "Requests", "Created", ""]) {
    const th = document.createElement("th");
    th.textContent = label;
    headRow.append(th);
  }
  thead.append(headRow);
  const tbody = document.createElement("tbody");
  for (const user of visible) {
    tbody.append(userRow(user));
  }
  table.append(thead, tbody);
  const wrap = document.createElement("div");
  wrap.className = "table-wrap admin-users-wrap";
  wrap.append(table);
  adminUserTable.replaceChildren(wrap);
}

function requestCard(request) {
  const card = document.createElement("article");
  card.className = "admin-request-card-full";

  const head = document.createElement("div");
  head.className = "admin-request-head";
  const title = document.createElement("strong");
  title.textContent = request.title || request.capability;
  const status = document.createElement("span");
  status.className = `request-status ${statusClass(request.status)}`;
  status.textContent = request.status;
  head.append(title, status);

  const user = document.createElement("p");
  user.className = "request-user";
  user.textContent = `${request.user_name || request.user_email} / ${request.user_email} / ${accessScopeLabel(request.user_access_profile || "verified")}`;

  const meta = document.createElement("p");
  meta.textContent = `${request.type} / ${request.capability} / ${formatDateTime(request.created_at)}`;

  const message = document.createElement("p");
  message.className = "admin-request-message";
  message.textContent = request.message;

  const controls = document.createElement("div");
  controls.className = "admin-edit-grid";
  const select = selectInput(["new", "reviewing", "planned", "granted", "declined", "closed"], request.status || "new");
  const note = document.createElement("textarea");
  note.rows = 3;
  note.value = request.admin_note || "";
  note.placeholder = "Admin note";
  const save = document.createElement("button");
  save.type = "button";
  save.textContent = "Save";
  save.addEventListener("click", async () => {
    await saveRequest(request.id, select.value, note.value, save);
  });
  controls.append(labelWrap("Status", select), labelWrap("Admin note", note), save);

  card.append(head, user, meta, message, controls);
  return card;
}

function userRow(user) {
  const tr = document.createElement("tr");
  const role = selectInput(["user", "admin"], user.role || "user", { user: "account", admin: "admin" });
  const accessProfile = selectInput(["verified", "research", "medical_ai", "admin"], canonicalAccessProfile(user.access_profile, user.role), {
    verified: "verified account",
    research: "advanced modules",
    medical_ai: "regulatory workspace",
    admin: "admin",
  });
  const authorization = selectInput(["active", "paused", "revoked"], user.access_status || "active", {
    active: "authorized",
    paused: "paused",
    revoked: "revoked",
  });
  const save = document.createElement("button");
  save.type = "button";
  save.className = "admin-table-button";
  save.textContent = "Save";
  save.addEventListener("click", async () => {
    await saveUser(user.id, role.value, accessProfile.value, authorization.value, save);
  });
  tr.append(
    td(`${user.name || user.email}\n${user.email}`, "wrap"),
    td(user.provider || "-"),
    td(user.email_verified_at || user.provider === "google" || user.role === "admin" ? "verified" : "pending"),
    td(role),
    td(accessProfile),
    td(authorization),
    td(`${Number(user.request_count || 0)} total / ${Number(user.new_request_count || 0)} new`),
    td(formatDateTime(user.created_at)),
    td(save),
  );
  return tr;
}

async function saveRequest(id, status, adminNote, button) {
  button.disabled = true;
  button.textContent = "Saving";
  try {
    await apiFetch(`/api/admin/requests/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ status, admin_note: adminNote }),
    });
    await loadAdminData();
  } catch (error) {
    setAdminStatus(error.message || "Save failed", "error");
    button.textContent = "Failed";
    setTimeout(() => {
      button.disabled = false;
      button.textContent = "Save";
    }, 1200);
  }
}

async function saveUser(id, role, accessProfile, accessStatus, button) {
  button.disabled = true;
  button.textContent = "Saving";
  try {
    await apiFetch(`/api/admin/users/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ role, access_profile: accessProfile, access_status: accessStatus }),
    });
    await loadAdminData();
  } catch (error) {
    setAdminStatus(error.message || "User save failed", "error");
    button.textContent = "Failed";
    setTimeout(() => {
      button.disabled = false;
      button.textContent = "Save";
    }, 1200);
  }
}

function renderBlocked(message) {
  adminMetrics.replaceChildren(metricCard("Access", "Blocked", message));
  adminRequestBoard.replaceChildren(empty(message));
  adminUserTable.replaceChildren(empty(message));
  adminBenchBoard.replaceChildren(empty(message));
}

function metricCard(label, value, detail) {
  const card = document.createElement("div");
  card.className = "admin-metric-card";
  const span = document.createElement("span");
  span.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value;
  const p = document.createElement("p");
  p.textContent = detail;
  card.append(span, strong, p);
  return card;
}

function labelWrap(label, control) {
  const node = document.createElement("label");
  const span = document.createElement("span");
  span.textContent = label;
  node.append(span, control);
  return node;
}

function selectInput(values, selected, labels = {}) {
  const select = document.createElement("select");
  for (const value of values) select.append(new Option(labels[value] || value, value));
  select.value = selected;
  return select;
}

function accessScopeLabel(value) {
  return {
    verified: "verified account",
    research: "advanced modules",
    medical_ai: "regulatory workspace",
    admin: "admin",
  }[canonicalAccessProfile(value)] || "verified account";
}

function td(value, className = "") {
  const cell = document.createElement("td");
  if (className) cell.className = className;
  if (value instanceof Node) {
    cell.append(value);
  } else {
    cell.textContent = value;
  }
  return cell;
}

function empty(message) {
  const node = document.createElement("p");
  node.className = "empty-state admin-empty";
  node.textContent = message;
  return node;
}

function setAdminStatus(message, tone = "") {
  adminStatus.textContent = message;
  adminStatus.classList.toggle("ok", tone === "ok");
  adminStatus.classList.toggle("error", tone === "error");
}

function formatDateTime(value) {
  const date = value ? new Date(value) : null;
  if (!date || Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: {
      "content-type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(data.message || data.error || `Request failed: ${response.status}`);
  }
  return data;
}
