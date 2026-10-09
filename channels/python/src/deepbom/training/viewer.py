"""Independent read-only DeepBoard. No model import or execution when viewing."""

from __future__ import annotations
import html
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .integrations import read_run

STYLE = """body{margin:0;background:#f4f6f5;color:#17312d;font:15px system-ui}main{max-width:1250px;margin:auto;padding:32px}h1{font-size:30px}header,p{line-height:1.65}.grid{min-width:0;display:grid;grid-template-columns:1fr 2fr;gap:24px;align-items:start}.card{min-width:0;background:white;border:1px solid #ccd8d3;border-radius:12px;padding:20px;overflow:auto}button,select,input{font:inherit;padding:8px;margin:4px;border:1px solid #a9bdb6;border-radius:5px;background:white}button:hover{background:#dfede7;color:#17312d}button[aria-pressed=true]{background:#267663;color:white}svg{width:100%;min-height:180px}path,rect{transition:all .22s ease}pre{max-height:480px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.muted{color:#536b63;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}.changed{color:#a65120}@media(max-width:750px){.grid{grid-template-columns:1fr}main{padding:14px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}@media(prefers-color-scheme:dark){body{background:#152321;color:#e5efeb}.card,button,select,input{background:#21332f;color:inherit;border-color:#516b62}.muted{color:#adc6bd}}"""
SCRIPT = Path(__file__).with_name("board.js").read_text(encoding="utf-8")


def page(bundle):
    encoded = json.dumps(bundle, ensure_ascii=False, allow_nan=False).replace(
        "<", "\\u003c"
    )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>DEEPBOM DeepBoard</title><style>{STYLE}</style><main><header><span class="muted">DEEPBOM · LOCAL EVIDENCE</span><h1>Training observations</h1><button id="refresh">Refresh local evidence</button><p>Recorded evidence, explicit coverage. Chart position is an observation sequence; it does not infer optimizer updates or task quality.</p></header><div class="grid"><section class="card"><label>Metric <select id="metric"></select></label><label><span id="coordinate"></span><input id="step" type="range" min="0" value="0" aria-label="Observation"></label><pre id="details"></pre></section><section class="card"><p id="plot-label"></p><svg viewBox="0 0 800 200" role="img" aria-label="Recorded metric values"><polyline id="curve" fill="none" stroke="#388c78" stroke-width="3"/></svg><h2>Model state evidence</h2><p id="state-note" class="muted"></p><div id="weights"></div><h3>Captured activations</h3><div id="activations"></div><label>View <select id="view"><option value="histogram">Distribution</option><option value="channels">Channel slices</option><option value="spectrum">Singular spectrum</option><option value="similarity">Similarity</option><option value="projection">Value tiles</option><option value="sparsity">Zero fraction</option></select></label><button id="svg">Download SVG</button><button id="png">Download PNG</button><p id="figure-title" class="muted"></p><svg id="hist" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 220" role="img" aria-label="Stored weight histogram"></svg><table id="summary" aria-label="Selected tensor statistics"></table><details><summary>Inspect full evidence and source context</summary><pre id="tensor"></pre></details></section></div></main><script id="data" type="application/json">{encoded}</script><script>{SCRIPT}</script></html>"""


def write_report(directory, output):
    Path(output).write_text(page(read_run(directory)), encoding="utf-8")
    return Path(output)


def serve(directory, port=0):
    root = Path(directory).resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            try:
                body = page(read_run(root)).encode()
            except (OSError, ValueError, RuntimeError, KeyError, TypeError):
                self.send_error(409, "Evidence is incomplete or invalid")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def write_candidate_report(directory, candidate):
    """Compatibility entry point; the official optimization renderer is shared."""
    from ..optimization.report import export_report
    return export_report(directory, Path(directory) / "report.html")
