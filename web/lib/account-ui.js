import { statusClass, ACCESS_REQUEST_PROFILES, profileIdForCapability } from "./auth-labels.js";
import { formatDateTime } from "./format.js";

// Worker-facing access IDs remain stable while the UI presents neutral module scopes.
const STAGE_ACCESS_MAP = {
  "Controlled module": "research",
  "Regulatory workspace": "medical_ai",
};

export function accountCapabilityRow(capability, onRequestProfile) {
  const row = document.createElement("div");
  row.className = "feature-row";
  const top = document.createElement("div");
  const name = document.createElement("strong");
  name.textContent = capability.name;
  const status = document.createElement("span");
  status.className = `feature-status ${statusClass(capability.status)}`;
  status.textContent = capability.status;
  top.append(name, status);

  const profileId = STAGE_ACCESS_MAP[capability.stage];
  const profileDef = profileId ? ACCESS_REQUEST_PROFILES[profileId] : null;
  if (profileDef) {
    const badge = document.createElement("span");
    badge.className = `access-profile-badge access-profile-${profileId}`;
    badge.textContent = profileDef.label;
    top.append(badge);
  }

  const meta = document.createElement("p");
  meta.textContent = capability.description;
  row.append(top, meta);

  // Controlled modules expose a direct request action from the neutral access profile.
  if (capability.status === "request" && profileDef && onRequestProfile) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "access-request-btn";
    btn.textContent = "Request access";
    btn.addEventListener("click", () => onRequestProfile(profileId));
    row.append(btn);
  }

  return row;
}

export function renderAccountCapabilityList(container, countNode, capabilities, onRequestProfile) {
  const items = Array.isArray(capabilities) ? capabilities : [];
  if (countNode) countNode.textContent = `${items.length} items`;
  container.replaceChildren(...items.map((capability) => accountCapabilityRow(capability, onRequestProfile)));
}

export function requestCard(request) {
  const node = document.createElement("article");
  node.className = "request-card";
  const head = document.createElement("div");
  const title = document.createElement("strong");
  title.textContent = request.title || request.capability;
  const status = document.createElement("span");
  status.className = `request-status ${statusClass(request.status)}`;
  status.textContent = request.status;
  head.append(title, status);
  const meta = document.createElement("p");
  const profileDef = ACCESS_REQUEST_PROFILES[request.capability] || ACCESS_REQUEST_PROFILES[profileIdForCapability(request.capability)];
  const capabilityLabel = profileDef ? profileDef.label : request.capability.replace(/_/g, " ");
  meta.textContent = `${formatDateTime(request.created_at)} / ${capabilityLabel}`;
  const message = document.createElement("p");
  message.textContent = request.message;
  node.append(head, meta, message);
  if (request.admin_note) {
    const note = document.createElement("p");
    note.className = "admin-note";
    note.textContent = `Admin note: ${request.admin_note}`;
    node.append(note);
  }
  return node;
}

export function requestLoading(message) {
  const node = document.createElement("p");
  node.className = "empty-state";
  node.textContent = message;
  return node;
}

export function renderRequestList(container, requests, options = {}) {
  const items = Array.isArray(requests) ? requests : [];
  const {
    countNode = null,
    emptyMessage = "No requests yet.",
    cardFactory = requestCard,
  } = options;
  if (countNode) countNode.textContent = `${items.length} items`;
  if (!items.length) {
    container.replaceChildren(requestLoading(emptyMessage));
    return;
  }
  container.replaceChildren(...items.map((request) => cardFactory(request)));
}

export function adminShortcutCard() {
  const card = document.createElement("article");
  card.className = "request-card";
  const head = document.createElement("div");
  const title = document.createElement("strong");
  title.textContent = "Admin operations are separated from account requests";
  const status = document.createElement("span");
  status.className = "request-status ok";
  status.textContent = "console";
  head.append(title, status);
  const body = document.createElement("p");
  body.textContent =
    "Open the admin console to review the request queue, account signups, roles, access scopes, and authorization status in one workspace.";
  card.append(head, body);
  return card;
}
