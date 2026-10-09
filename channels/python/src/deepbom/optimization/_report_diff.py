"""Read-only diff presentation of the verified common report projection."""

import difflib
import html
import json
from pathlib import Path
from .._native.core import engine


def _json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    )


def _patch(before, after, label):
    # JSON quoting keeps identifiers containing newlines on one header line.
    name = json.dumps(label, ensure_ascii=False)
    return "".join(
        difflib.unified_diff(
            [] if before is None else (_json(before) + "\n").splitlines(keepends=True),
            [] if after is None else (_json(after) + "\n").splitlines(keepends=True),
            fromfile="Before/" + name,
            tofile="After/" + name,
            n=3,
        )
    )


def diff_view(data):
    """Fetch common field-level verdicts; unified text is presentation only."""
    view = engine("optimization_diff", data)
    for entry in view["entries"]:
        entry["patch"] = (
            _patch(entry["before"], entry["after"], entry["id"])
            if entry["tags"]
            else ""
        )
    return view


def diff_html(view):
    # A single self-contained file; all untrusted strings are subsequently
    # inserted with textContent. Escape the inline JSON script boundary too.
    payload = (
        json.dumps(view, ensure_ascii=False, allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    root = Path(__file__).parent
    css = (root / "report-diff.css").read_text(encoding="utf-8")
    js = (root / "report-diff.js").read_text(encoding="utf-8")
    counts = view["counts"]
    stats = " · ".join(
        f"{counts.get(k, 0)} {k}" for k in ("changed", "added", "removed", "unchanged")
    )
    return f"""<style>{css}</style><section id="model-diff" aria-label="Interactive model diff">
<h2>Model diff</h2><p class="diff-stats">{html.escape(stats)} modules</p>
<p>Read-only evidence diff, aligned by exact native module ID. Renames are not inferred. Connections are recorded relationships, not inferred execution edges. State and observed-shape changes are shown separately from module changes.</p>
<div class="diff-tools"><label>Find module or rule <input id="diff-search" type="search" placeholder="Module path, operator, rule…"></label>
<label>Show <select id="diff-filter"><option value="changes">Changes only</option><option value="all">All modules</option><option value="added">Added</option><option value="removed">Removed</option><option value="changed">Structure changed</option><option value="state">State changed</option><option value="connections">Connections changed</option></select></label>
<label>View <select id="diff-mode"><option value="unified">Unified diff</option><option value="split">Before / After</option></select></label>
<button id="diff-prev" type="button">Previous change</button><button id="diff-next" type="button">Next change</button>
<button id="diff-download" type="button">Download evidence diff</button></div>
<p id="diff-status" role="status" aria-live="polite"></p>
<div class="diff-workspace"><nav aria-label="Model subjects"><div id="diff-list"></div><div class="diff-pages"><button id="diff-page-prev" type="button">Previous page</button><span id="diff-page-label"></span><button id="diff-page-next" type="button">Next page</button></div></nav>
<article id="diff-detail" aria-label="Selected model change"></article></div>
<noscript>Enable JavaScript to use the interactive diff. The complete evidence tables below remain available.</noscript>
</section><script id="model-diff-data" type="application/json">{payload}</script><script>{js}</script>"""
