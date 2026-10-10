"""Official monochrome report projection. Reading a package never executes it."""

from __future__ import annotations
from pathlib import Path
import html
import base64
import hashlib
import io
import json
import os
import tempfile
from .._native.core import engine, read_json, content_hash
from ._report_diff import diff_view, diff_html


def require_pdf():
    try:
        import reportlab
    except ImportError as error:
        raise RuntimeError(
            'PDF export requires the optional dependency: pip install "deepbom[report]"'
        ) from error


def report_data(directory, *, baseline=None):
    from . import _manifest, _load_evidence, load_evidence

    path, manifest = _manifest(directory)
    candidate = _load_evidence(path, manifest)
    if "report-source.json" in manifest["files"]:
        source = read_json(path / "report-source.json")
        if source.get("schema") != "deepbom.optimization_report_source.v1":
            raise ValueError("Unsupported report source schema")
        original = source["baseline"]
        context = source["context"]
        if (
            baseline is not None
            and load_evidence(baseline)["snapshot"]["snapshot_sha256"]
            != manifest["baseline_sha256"]
        ):
            raise ValueError("Supplied baseline does not match the candidate")
    else:
        if baseline is None:
            raise ValueError(
                "This older package has no saved baseline evidence; supply --baseline <original-package>. The model factory will not be executed."
            )
        original = load_evidence(baseline)
        empty = {
            "status": "not_requested",
            "invocations": [],
            "inputs": [],
            "outputs": [],
        }
        context = {
            "version": 1,
            "objective": "not_recorded",
            "target": "not_recorded",
            "selection": "not_recorded",
            "user_rationale": None,
            "probes": {"baseline": empty, "candidate": empty},
        }
    if original["snapshot"]["snapshot_sha256"] != manifest["baseline_sha256"]:
        raise ValueError("Report baseline identity differs from the package manifest")
    result = engine(
        "optimization_report",
        {
            "baseline": original,
            "candidate": candidate,
            "transformations": read_json(path / "transformation.json")["rules"],
            "validation": read_json(path / "validation.json"),
            "context": context,
        },
    )
    # Recalculate in the common engine rather than trusting stored summary totals.
    recorded = read_json(path / "comparison.json")
    if content_hash(result["comparison"]) != content_hash(recorded):
        raise ValueError(
            "Report comparison disagrees with the complete package comparison"
        )
    if (
        content_hash({k: v for k, v in result.items() if k != "report_sha256"})
        != result["report_sha256"]
    ):
        raise ValueError("Report content digest mismatch")
    return result


def _number(value):
    return "Not assessed" if value is None else f"{int(value):,}"


def _brief(value, limit=180):
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        text
        if len(text) <= limit
        else text[:limit] + " ... [full value in report JSON]"
    )


def _configuration(node):
    cfg = node["config"]
    if "native_configuration" in cfg:
        return (
            node["mode"]
            + "; "
            + (cfg["native_configuration"] or "No scalar configuration")
        )
    keys = [
        "filters",
        "kernel_size",
        "strides",
        "padding",
        "dilation_rate",
        "groups",
        "depth_multiplier",
        "activation",
        "units",
        "axis",
        "epsilon",
        "trainable",
        "dtype",
    ]
    selected = {k: cfg[k] for k in keys if k in cfg}
    return (
        _brief(selected) if selected else "Container / no selected scalar configuration"
    )


def _changed_fields(before, after):
    if not before or not after:
        return ""
    missing = object()
    fields = [k for k in ("kind", "mode", "state_refs") if before[k] != after[k]]
    fields.extend(
        "config." + k
        for k in sorted(set(before["config"]) | set(after["config"]))
        if before["config"].get(k, missing) != after["config"].get(k, missing)
    )
    return "\nFields: " + ", ".join(fields) if fields else ""


def _diagrams(data):
    """Report-local vector layout; edges are containment, not execution."""
    flags = {
        n["id"]: {"added": "+", "removed": "-", "changed": "~", "unchanged": "="}[
            n["status"]
        ]
        for n in data["comparison"]["nodes"]
    }
    chunk = 18
    count = max(len(data["baseline"]["nodes"]), len(data["candidate"]["nodes"]))
    for start in range(0, count, chunk):
        height = 36 + 27 * min(chunk, count - start)
        primitives = []
        subjects = {"baseline": [], "candidate": []}
        drawn_relationships = {"baseline": [], "candidate": []}
        for side, label in enumerate(("baseline", "candidate")):
            s = data[label]
            offset = side * 252
            primitives.append(
                (
                    "text",
                    offset,
                    13,
                    ("BEFORE" if side == 0 else "AFTER")
                    + (
                        f' / subjects {start+1}-{min(start+chunk,len(s["nodes"]))} of {len(s["nodes"])}'
                        if start < len(s["nodes"])
                        else f' / no further subjects (total {len(s["nodes"])})'
                    ),
                    8.5,
                    True,
                )
            )
            parents = {
                r["to"]: r["from"]
                for r in s["relationships"]
                if r["kind"] == "contains"
            }
            relationship_indices = {
                (r["from"], r["to"]): i
                for i, r in enumerate(s["relationships"])
                if r["kind"] == "contains"
            }
            positions = {}
            for index, node in enumerate(s["nodes"][start : start + chunk]):
                current = node["id"]
                seen = set()
                depth = 0
                while current in parents and current not in seen:
                    seen.add(current)
                    current = parents[current]
                    depth += 1
                depth = min(depth, 5)
                x = offset + depth * 9
                y = 24 + index * 27
                w = 235 - depth * 9
                parent = positions.get(parents.get(node["id"]))
                if parent:
                    px, py = parent
                    primitives.append(("line", px + 3, py + 20, px + 3, y + 10))
                    primitives.append(("line", px + 3, y + 10, x, y + 10))
                    drawn_relationships[label].append(
                        relationship_indices[(parents[node["id"]], node["id"])]
                    )
                positions[node["id"]] = (x, y)
                subjects[label].append(node["id"])
                primitives.append(
                    ("rect", x, y, w, 21, flags[node["id"]] != "=", label, node["id"])
                )
                title = f"[{flags[node['id']]}] {node['id']} : {node['kind'].split('.')[-1]}"
                if len(title) > 46:
                    title = title[:43] + "..."
                primitives.append(("text", x + 4, y + 14, title, 7, False))
        yield {
            "width": 487,
            "height": height,
            "primitives": primitives,
            "subjects": subjects,
            "drawn_relationships": drawn_relationships,
        }


def _sections(data, view=None):
    """The same ordered content blocks feed HTML and PDF; no numerical engine here."""
    a, b, diff = data["baseline"], data["candidate"], data["comparison"]
    result = []

    def h(title):
        result.append(("heading", title))

    def p(text):
        result.append(("text", str(text)))

    def table(headers, rows, widths=None):
        result.append(("table", headers, rows, widths))

    h("01  Summary and identity")
    p(
        "A source-bound record of the model before and after transformation, applied changes and objective static differences."
    )
    table(
        ["Field", "Recorded value"],
        [
            ["Framework", f"{a['framework']['name']} {a['framework']['version']}"],
            [
                "Objective / target",
                f"{data['context']['objective']} / {data['context']['target']}",
            ],
            ["Selection", data["context"]["selection"]],
            [
                "User-stated rationale",
                data["context"].get("user_rationale") or "Not supplied",
            ],
            ["Before snapshot SHA-256", a["snapshot_sha256"]],
            ["After snapshot SHA-256", b["snapshot_sha256"]],
            ["Report SHA-256", data["report_sha256"]],
        ],
        [0.25, 0.75],
    )
    p(
        "Snapshot identities bind captured model definition and state, not a standalone model file or an attestation of its author."
    )
    table(
        ["Static metric", "Before", "After", "After - before"],
        [
            [
                label,
                _number(diff["totals"]["baseline"][key]),
                _number(diff["totals"]["candidate"][key]),
                _number(data["static_deltas"][key]),
            ]
            for key, label in [
                ("trainable_elements", "Trainable parameter elements"),
                ("state_elements", "All state elements"),
                ("state_bytes", "State payload bytes"),
            ]
        ]
        + [
            [
                "Mapped invocation MAC subtotal",
                _number(a["cost"]["macs"]),
                _number(b["cost"]["macs"]),
                _number(data["cost_delta"]),
            ]
        ],
        [0.43, 0.19, 0.19, 0.19],
    )
    p(
        "State includes registered parameters and buffers; payload bytes exclude headers, compression, optimizer state and runtime allocation. MACs are derived from observed shapes on a supplied probe path, not measured performance or a complete functional-operation graph."
    )
    h("02  Transformation locations and reasons")
    if not data["changes"]:
        p("No transformation was applied. No optimization benefit is claimed.")
    for change in data["changes"]:
        p(
            f"Change {change['index']}: {change['title']} | subject {change['subject']} | rule {change['rule']}@{change['rule_version']}"
        )
        table(
            ["Item", "Evidence / explanation"],
            [
                ["Requested rule", _brief(change["request"], 1000)],
                ["Why this change is applicable", change["reason"]],
                ["What changes", change["description"] or "Not recorded"],
                ["State treatment", change["state"]],
                ["Scope / trade-off", change["tradeoff"] or change["scope"]],
            ],
            [0.25, 0.75],
        )
        p(change["reason_basis"])
        table(
            ["Affected native subject", "Structural change"],
            [[n["id"], n["status"]] for n in change["affected_nodes"]],
            [0.65, 0.35],
        )
    view = view if view is not None else diff_view(data)
    changed = [entry for entry in view["entries"] if entry["tags"]]
    h("02A  Change index")
    p(
        "Module matching uses exact native IDs; renames are not inferred. Structure, registered state, recorded connections and observed calls are distinguished. The HTML report offers searchable unified and Before / After views, with unchanged evidence hidden by default."
    )
    if changed:
        table(
            ["Subject", "Recorded changes", "Applied rule"],
            [
                [
                    e["id"],
                    ", ".join([e["status"]] + e["tags"]),
                    ", ".join(str(i) for i in e["rules"])
                    or "No direct rule association",
                ]
                for e in changed
            ],
            [0.36, 0.42, 0.22],
        )
    else:
        p("No recorded module, state, connection or observed-call difference.")
    h("02B  Field-level Before / After diff")
    p(
        "Only changed fields are listed. Paths are JSON Pointers into each subject's recorded evidence. Absent and null are different. Arrays are compared as complete values; their elements are not assumed to correspond. HTML and PDF use these same common field changes."
    )
    if not changed:
        p("No changed fields.")
    for entry in changed:
        p(f"{entry['id']} | {entry['status']} | {', '.join(entry['tags'])}")
        if entry["rules"]:
            p(
                "Applied rule references: "
                + ", ".join(str(i) for i in entry["rules"])
                + ". Reasons and boundaries are in section 02."
            )
        table(
            ["Changed field", "Before (−)", "After (+)"],
            [
                [
                    row["path"] or "(whole subject)",
                    (
                        json.dumps(row["before"], ensure_ascii=False, sort_keys=True)
                        if row["before_present"]
                        else "Absent"
                    ),
                    (
                        json.dumps(row["after"], ensure_ascii=False, sort_keys=True)
                        if row["after_present"]
                        else "Absent"
                    ),
                ]
                for row in entry["fields"]
            ],
            [0.26, 0.37, 0.37],
        )
    h("03  Before / After structure diff")
    p(
        "Source: Native Model IR v2 (module_tree). The diagrams cover the recorded module hierarchy, not a complete execution graph. Functional operations, data-flow edges and unobserved branches are not reconstructed. The counts below distinguish modules shown from relationships drawn or listed separately."
    )
    diagrams = list(_diagrams(data))
    coverage = {}
    for side in ("baseline", "candidate"):
        drawn = [subject for d in diagrams for subject in d["subjects"][side]]
        expected = [node["id"] for node in data[side]["nodes"]]
        if drawn != expected or len(set(drawn)) != len(drawn):
            raise ValueError("Report diagram does not conserve native subjects")
        relations = [ref for d in diagrams for ref in d["drawn_relationships"][side]]
        if len(set(relations)) != len(relations):
            raise ValueError("Report diagram repeats a relationship")
        coverage[side] = [
            f"{len(drawn)} / {len(expected)}",
            str(len(data[side]["relationships"])),
            str(len(relations)),
            str(len(data[side]["relationships"]) - len(relations)),
        ]
    table(
        ["Recorded structure coverage", "Before", "After"],
        [
            [label, coverage["baseline"][i], coverage["candidate"][i]]
            for i, label in enumerate(
                [
                    "Modules drawn / captured",
                    "Relationships listed in section 04",
                    "Relationships drawn within diagram pages",
                    "Relationships available only in section 04",
                ]
            )
        ],
        [0.60, 0.20, 0.20],
    )
    p(
        "Before and after module hierarchies. Lines denote containment only. [+] added, [-] removed, [~] changed, [=] unchanged. Gray boxes mark changed subjects. Long labels are abbreviated here; complete identifiers follow below. Relationships spanning diagram pages are listed in section 04."
    )
    for diagram in diagrams:
        result.append(("diagram", diagram))
    p(
        "All native subjects are listed. [+] added, [-] removed, [~] changed, [=] unchanged. The structure is a module/layer hierarchy; adjacency in this table is not a runtime execution edge. Configuration cells are concise projections; full configuration and relationships are retained in the report JSON."
    )
    na, nb = ({n["id"]: n for n in s["nodes"]} for s in (a, b))
    marks = {"added": "[+]", "removed": "[-]", "changed": "[~]", "unchanged": "[=]"}
    table(
        ["Diff / subject", "Before structure", "After structure"],
        [
            [
                marks[n["status"]]
                + " "
                + n["id"]
                + _changed_fields(na.get(n["id"]), nb.get(n["id"])),
                (
                    (
                        na[n["id"]]["kind"].split(".")[-1]
                        + "\n"
                        + _configuration(na[n["id"]])
                    )
                    if n["id"] in na
                    else "Absent"
                ),
                (
                    (
                        nb[n["id"]]["kind"].split(".")[-1]
                        + "\n"
                        + _configuration(nb[n["id"]])
                    )
                    if n["id"] in nb
                    else "Absent"
                ),
            ]
            for n in diff["nodes"]
        ],
        [0.22, 0.39, 0.39],
    )
    h("04  Structure relationships")
    for label, s in [("Before", a), ("After", b)]:
        p(label + " model: source-declared relationships (complete)")
        table(
            ["From", "Relationship", "To"],
            [[r["from"], r["kind"], r["to"]] for r in s["relationships"]],
            [0.38, 0.24, 0.38],
        )
    h("05  Static cost and tensor contracts")
    for label, s in [("Before", a), ("After", b)]:
        p(
            label
            + f" model | shape capture: {s['shape_observation_status']} | mapped {s['cost']['assessed_invocations']} / {s['cost']['observed_invocations']} observed module calls; {s['cost']['unmapped_invocations']} unmapped."
        )
        p(
            "Model input contract: "
            + _brief(s["inputs"], 2000)
            + "; output contract: "
            + _brief(s["outputs"], 2000)
        )
        if not s["invocations"]:
            p(
                "No shape probe was requested or saved. MACs and intermediate tensor sizes are not assessed."
            )
        table(
            [
                "Subject / call",
                "Input shape",
                "Output shape / logical bytes",
                "MACs / status",
            ],
            [
                [
                    f"{c['subject']} #{c['invocation']}",
                    _brief([t["shape"] for t in c["inputs"]]),
                    "\n".join(
                        f"{t['shape']} / {_number(t['logical_bytes'])}"
                        for t in c["outputs"]
                    ),
                    (
                        _number(c["macs"]["value_decimal"])
                        if c["macs"]["status"] == "assessed"
                        else c["macs"]["status"]
                    ),
                ]
                for c in s["invocations"]
            ],
            [0.24, 0.21, 0.31, 0.24],
        )
        for c in s["invocations"]:
            if c["macs"]["status"] == "not_assessed":
                p(
                    f"Unassessed {c['subject']} #{c['invocation']}: {c['macs']['reason']}"
                )
    p(a["cost"]["scope"])
    p(
        "A logical tensor size is not peak live memory, memory traffic, cache occupancy or an allocation measurement. Shared storage, fusion and liveness are not inferred."
    )
    h("06  Weight and buffer correspondence")
    p(
        "Exact payload equality is checked separately from native-ID alignment. Added and removed state are not claimed to be numerically equivalent or transplanted."
    )
    table(
        ["State identifier", "Correspondence", "Payload equal"],
        [
            [
                t["id"],
                t["status"],
                (
                    "Not comparable"
                    if t["payload_equal"] is None
                    else str(t["payload_equal"]).lower()
                ),
            ]
            for t in diff["tensors"]
        ],
        [0.46, 0.34, 0.20],
    )
    h("07  Validation and interpretation boundary")
    table(
        ["Check", "Recorded result"],
        [[k, _brief(v, 500)] for k, v in data["validation"].items()],
        [0.30, 0.70],
    )
    p(data["boundary"])
    for text in dict.fromkeys(a["limitations"] + b["limitations"]):
        p(text)
    p(
        "The report is exported from verified saved evidence. Report generation does not load factories, deserialize executable checkpoints, retrain the model or execute inference. The JSON companion carries complete configurations, rule requests, relationships, exact counts and source identities."
    )
    return result


def _html(data, attachments=None):
    view = diff_view(data)
    blocks = [diff_html(view)]
    esc = lambda t: html.escape(str(t)).replace("\n", "<br>")
    for block in _sections(data, view):
        if block[0] == "heading":
            if block[1].startswith("02B "):
                blocks.append(
                    '<details class="complete-evidence"><summary>Expand complete field diff, structure, connections, tensor contracts and state tables</summary>'
                )
            elif block[1].startswith("07 "):
                blocks.append("</details>")
            blocks.append("<h2>" + esc(block[1]) + "</h2>")
        elif block[0] == "text":
            blocks.append("<p>" + esc(block[1]) + "</p>")
        elif block[0] == "page_break":
            blocks.append('<div class="page-break"></div>')
        elif block[0] == "diagram":
            d = block[1]
            parts = [
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {d["width"]} {d["height"]}" role="img" aria-label="Before and after module hierarchy">'
            ]
            for shape in d["primitives"]:
                if shape[0] == "rect":
                    _, x, y, w, h, changed, side, subject = shape
                    fill = "#eee" if changed else "#fff"
                    parts.append(
                        f'<rect data-model-side="{side}" data-subject-ref="{html.escape(subject, quote=True)}" x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="#111" stroke-width="0.6"><title>{html.escape(subject)}</title></rect>'
                    )
                elif shape[0] == "line":
                    _, x1, y1, x2, y2 = shape
                    parts.append(
                        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#555" stroke-width="0.6"/>'
                    )
                else:
                    _, x, y, text, size, bold = shape
                    weight = "bold" if bold else "normal"
                    parts.append(
                        f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="#111">{esc(text)}</text>'
                    )
            blocks.append("".join(parts) + "</svg>")
        else:
            _, headers, rows, widths = block
            if not rows:
                continue
            blocks.append(
                "<table><thead><tr>"
                + "".join("<th>" + esc(x) + "</th>" for x in headers)
                + "</tr></thead><tbody>"
                + "".join(
                    "<tr>" + "".join("<td>" + esc(x) + "</td>" for x in row) + "</tr>"
                    for row in rows
                )
                + "</tbody></table>"
            )
    # The HTML remains useful for development without changing the printed report.
    blocks.append('<section class="reuse"><h2>Inspect and reuse</h2>')
    for key, label in (("baseline", "Before"), ("candidate", "After")):
        full = {k: data[key][k] for k in ("nodes", "relationships")}
        blocks.append(
            "<details><summary>"
            + label
            + " structure: full recorded configuration</summary><pre>"
            + html.escape(json.dumps(full, ensure_ascii=False, indent=2))
            + "</pre></details>"
        )
    if attachments:
        blocks.append(
            '<p>The generated loader still requires the candidate package and its recorded state files. Copying code does not embed the model weights.</p><button id="copy" type="button">Copy Python code</button> <span id="notice" role="status"></span><p>'
        )
        for name, payload in attachments.items():
            encoded = base64.b64encode(payload).decode("ascii")
            blocks.append(
                f'<a href="{name}" download="{name}" data-bytes="{encoded}">Download {name}</a> '
            )
        blocks.append(
            '</p><details><summary>Generated Python loader</summary><pre id="model-code">'
            + html.escape(attachments["model.py"].decode("utf-8"))
            + "</pre></details>"
        )
        blocks.append("""<script>
document.getElementById('copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(document.getElementById('model-code').textContent);document.getElementById('notice').textContent='Copied';}catch{document.getElementById('notice').textContent='Clipboard unavailable; open the code below to copy it.';}});
document.querySelectorAll('a[data-bytes]').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();const bytes=Uint8Array.from(atob(a.dataset.bytes),c=>c.charCodeAt(0)),url=URL.createObjectURL(new Blob([bytes],{type:'text/plain;charset=utf-8'})),link=document.createElement('a');link.href=url;link.download=a.download;link.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}));
</script>""")
    blocks.append("</section>")
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DEEPBOM Model Optimization Report</title><style>
@page{size:A4;margin:18mm}*{box-sizing:border-box}body{color:#111;background:#fff;font:13px/1.6 Arial,sans-serif;margin:0}main{max-width:1050px;margin:auto;padding:36px}header{border-top:4px solid #111;border-bottom:1px solid #444;padding:22px 0}h1{font-size:30px;letter-spacing:-.6px;margin:6px 0}h2{font-size:18px;border-bottom:1px solid #888;margin-top:30px;padding-bottom:6px;break-after:avoid}p{overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0;table-layout:fixed}th,td{text-align:left;border-bottom:1px solid #ccc;padding:8px;vertical-align:top;overflow-wrap:anywhere}th{background:#eee}tr{break-inside:avoid}thead{display:table-header-group}small{color:#444}svg{width:100%;max-height:700px;break-inside:avoid}@media print{main{padding:0}body{font-size:10pt}table{font-size:8pt}.page-break{break-before:page}}@media(max-width:600px){main{padding:14px}table{font-size:10px}th,td{padding:4px}}</style><main><header><small>DEEPBOM / OFFICIAL REPORT TEMPLATE / MONOCHROME</small><h1>Model Optimization Report</h1><p>Before / After structure. Applied changes. Static evidence.</p></header>"""
        + "<style>.reuse pre{max-height:480px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}.reuse details{border:1px solid #ccc;padding:10px;margin:10px 0}.reuse summary{cursor:pointer}.reuse button{font:inherit;background:white;color:#111;border:1px solid #444;padding:7px 12px;cursor:pointer}.reuse a{color:#111;text-underline-offset:3px;margin-right:16px}@media print{.reuse{display:none}}</style>"
        + "".join(blocks)
        + "</main></html>"
    )


def _pdf(data):
    require_pdf()
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        LongTable,
        TableStyle,
        PageBreak,
    )
    from reportlab.graphics.shapes import Drawing, Rect, Line, String
    from reportlab.pdfgen.canvas import Canvas

    from .._report_fonts import ReportText, UNICODE_NOTE
    report_text = ReportText()
    safe = report_text.safe

    body = ParagraphStyle(
        "body",
        fontName="DeepbomReport",
        fontSize=8,
        leading=12,
        spaceAfter=7,
        textColor=colors.black,
        splitLongWords=True,
    )
    cell = ParagraphStyle("cell", parent=body, fontSize=7, leading=10, spaceAfter=0)
    head = ParagraphStyle(
        "heading",
        parent=body,
        fontName="DeepbomReportBold",
        fontSize=13,
        leading=17,
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True,
    )
    title = ParagraphStyle(
        "title", parent=head, fontSize=24, leading=29, spaceBefore=5, spaceAfter=12
    )
    boldcell = ParagraphStyle("boldcell", parent=cell, fontName="DeepbomReportBold")
    story = [
        Paragraph("DEEPBOM / MODEL EVIDENCE", boldcell),
        Paragraph("Model Optimization Report", title),
        Paragraph(
            "Official monochrome template / Before and after structure, applied changes and static results",
            body,
        ),
    ]
    width = A4[0] - 108
    for block in _sections(data):
        if block[0] == "heading":
            story.append(Paragraph(safe(block[1]), head))
        elif block[0] == "text":
            story.append(Paragraph(safe(block[1]), body))
        elif block[0] == "page_break":
            story.append(PageBreak())
        elif block[0] == "diagram":
            d = block[1]
            drawing = Drawing(d["width"], d["height"])
            for shape in d["primitives"]:
                if shape[0] == "rect":
                    _, x, y, w, h, changed, side, subject = shape
                    drawing.add(
                        Rect(
                            x,
                            d["height"] - y - h,
                            w,
                            h,
                            fillColor=(
                                colors.HexColor("#eeeeee") if changed else colors.white
                            ),
                            strokeColor=colors.black,
                            strokeWidth=0.6,
                        )
                    )
                elif shape[0] == "line":
                    _, x1, y1, x2, y2 = shape
                    drawing.add(
                        Line(
                            x1,
                            d["height"] - y1,
                            x2,
                            d["height"] - y2,
                            strokeColor=colors.HexColor("#555555"),
                            strokeWidth=0.6,
                        )
                    )
                else:
                    _, x, y, text, size, bold = shape
                    font = "DeepbomReportBold" if bold else "DeepbomReport"
                    value = html.unescape(safe(text))
                    available = (235 if x < 252 else 487) - x - 4
                    while pdfmetrics.stringWidth(value, font, size) > available:
                        value = value[:-4] + "..."
                    drawing.add(
                        String(
                            x,
                            d["height"] - y,
                            value,
                            fontName=font,
                            fontSize=size,
                            fillColor=colors.black,
                        )
                    )
            story.extend([drawing, Spacer(1, 12)])
        else:
            _, headers, rows, widths = block
            if not rows:
                continue
            values = [[Paragraph(safe(x), boldcell) for x in headers]] + [
                [Paragraph(safe(x), cell) for x in row] for row in rows
            ]
            table = LongTable(
                values,
                colWidths=[
                    width * x for x in (widths or [1 / len(headers)] * len(headers))
                ],
                repeatRows=1,
                hAlign="LEFT",
                splitByRow=1,
                splitInRow=1,
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                        (
                            "LINEBELOW",
                            (0, 1),
                            (-1, -1),
                            0.3,
                            colors.HexColor("#cccccc"),
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([table, Spacer(1, 9)])
    if report_text.escaped:
        story.append(
            Paragraph(
                UNICODE_NOTE,
                body,
            )
        )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(0.4)
        canvas.line(48, 39, A4[0] - 48, 39)
        canvas.setFont("DeepbomReport", 6.5)
        canvas.drawString(
            48, 28, "DEEPBOM / " + data["report_sha256"][:20] + " / monochrome.v1"
        )
        canvas.drawRightString(A4[0] - 48, 28, str(doc.page))
        canvas.restoreState()

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=48,
        leftMargin=48,
        topMargin=44,
        bottomMargin=53,
        title="DEEPBOM Model Optimization Report",
        author="DEEPBOM",
        pageCompression=1,
    )
    doc.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer,
        canvasmaker=lambda *a, **kw: Canvas(*a, **{**kw, "invariant": 1}),
    )
    return output.getvalue()


def export_report(directory, output, *, format=None, baseline=None):
    """Export a deterministic report; never overwrite a different existing file."""
    output = Path(output).resolve()
    format = format or output.suffix.lstrip(".").lower()
    if format not in ("pdf", "html", "json"):
        raise ValueError("Report format must be pdf, html or json")
    if format == "pdf":
        require_pdf()
    data = report_data(directory, baseline=baseline)
    attachments = None
    if format == "html":
        from . import _manifest

        path, manifest = _manifest(directory)
        if manifest["snapshot_sha256"] != data["candidate"]["snapshot_sha256"]:
            raise ValueError("Candidate changed during report export")
        attachments = {}
        for name in ("model.py", "model-code.md"):
            payload = (path / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != manifest["files"][name]:
                raise ValueError(f"Package file changed during report export: {name}")
            attachments[name] = payload
    payload = (
        _pdf(data)
        if format == "pdf"
        else (
            _html(data, attachments)
            if format == "html"
            else json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode("utf-8")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".deepbom-report-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        try:
            os.link(name, output)
        except FileExistsError:
            if output.is_symlink() or output.read_bytes() != payload:
                raise FileExistsError(
                    "Report export will not overwrite a different existing file"
                )
    finally:
        os.unlink(name)
    return output
