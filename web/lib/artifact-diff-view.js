import { compareGraphDiffSnapshots } from "./artifact-diff.js";

export function renderArtifactDiffWorkspace(root, leftSnapshot, rightSnapshot, { onSelect = null } = {}) {
  if (!root) return null;
  root.replaceChildren();
  const left = leftSnapshot?.graphDiff;
  const right = rightSnapshot?.graphDiff;
  if (!left || !right) {
    root.append(message("Visual graph diff is not available for one or both legacy snapshots. Re-run those artifacts with the current analyzer to store topology contracts."));
    return null;
  }
  let diff;
  try { diff = compareGraphDiffSnapshots(left, right); }
  catch (error) { root.append(message(String(error?.message || error))); return null; }
  const head = element("div", "artifact-diff-summary");
  head.append(
    datum("Matched", `${diff.matched_count}`),
    datum("Changed", `${diff.changed_match_count}`),
    datum("A only", `${diff.unmatched_left_indices.length}`),
    datum("B only", `${diff.unmatched_right_indices.length}`),
    datum("Ambiguous", `${diff.ambiguous.length}`),
  );
  const note = message(diff.interpretation_boundary);
  const split = element("div", "artifact-diff-split");
  const leftPane = pane("A", leftSnapshot, diff.matches.map((row) => ({ node: row.left, row })), diff.unmatched_left_indices, left.nodes, "left");
  const rightPane = pane("B", rightSnapshot, diff.matches.map((row) => ({ node: row.right, row })), diff.unmatched_right_indices, right.nodes, "right");
  split.append(leftPane, rightPane);
  let syncing = false;
  for (const [source, target] of [[leftPane, rightPane], [rightPane, leftPane]]) {
    source.querySelector(".artifact-diff-list").addEventListener("scroll", (event) => {
      if (syncing) return;
      syncing = true;
      target.querySelector(".artifact-diff-list").scrollTop = event.currentTarget.scrollTop;
      requestAnimationFrame(() => { syncing = false; });
    });
  }
  split.addEventListener("click", (event) => {
    const row = event.target.closest("[data-match-key]");
    if (!row) return;
    for (const item of split.querySelectorAll("[data-match-key]")) item.classList.toggle("selected", item.dataset.matchKey === row.dataset.matchKey);
    const match = diff.matches[Number(row.dataset.matchKey)];
    if (match) onSelect?.(match);
  });
  root.append(head, note, split);
  return diff;
}

function pane(side, snapshot, matched, unmatchedIndices, nodes, direction) {
  const root = element("section", "artifact-diff-pane");
  const title = element("header", "artifact-diff-pane-head");
  title.append(element("strong", "", `${side} | ${snapshot.filename || "artifact"}`), element("span", "", `${String(snapshot.sha256 || "").slice(0, 12)}...`));
  const list = element("div", "artifact-diff-list");
  matched.forEach(({ node, row }, index) => list.append(nodeRow(node, row, index, direction)));
  const byIndex = new Map(nodes.map((node) => [node.index, node]));
  for (const index of unmatchedIndices) list.append(nodeRow(byIndex.get(index), null, `unmatched-${direction}-${index}`, direction));
  root.append(title, list);
  return root;
}

function nodeRow(node, match, key, direction) {
  const root = element("button", `artifact-diff-node${match?.changes?.length ? " changed" : ""}${match ? "" : " unmatched"}`);
  root.type = "button";
  root.dataset.matchKey = String(key);
  root.dataset.opIndex = String(node?.index ?? "");
  const identity = element("span", "artifact-diff-node-identity");
  identity.append(element("strong", "", `#${node?.index ?? "?"} ${node?.name || "unknown"}`), element("small", "", node?.outputs?.[0] || "output contract unavailable"));
  const delta = element("span", "artifact-diff-node-delta", match ? match.changes.length ? match.changes.join(" | ") : "unchanged" : `${direction.toUpperCase()} only`);
  const confidence = match ? `${Math.round(match.confidence * 100)}% ${match.method.replaceAll("_", " ")}` : "unmatched";
  root.append(identity, delta, element("em", "", confidence));
  return root;
}

function datum(label, value) { const root = element("div"); root.append(element("span", "", label), element("strong", "", value)); return root; }
function message(text) { return element("p", "artifact-diff-note", text); }
function element(tag, className = "", text = "") { const node = document.createElement(tag); if (className) node.className = className; if (text !== "") node.textContent = String(text); return node; }
