#!/usr/bin/env python3
"""Real framework/CLI regression checks for the official report projection."""

import os, sys, json, copy, tempfile, shutil, subprocess, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(
    0, os.environ.get("DEEPBOM_TEST_PACKAGE_ROOT", str(ROOT / "channels/python/src"))
)
import torch
from deepbom.optimization import optimize, export_report, load_model
from deepbom.optimization.report import report_data, _pdf, _html, _diagrams, _sections
from deepbom.optimization._report_diff import diff_view
from deepbom._native.core import content_hash

out = Path(
    os.environ.get("REPORT_TEST_ROOT") or tempfile.mkdtemp(prefix="deepbom-report-")
)
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


def reject(fn):
    try:
        fn()
    except (ValueError, RuntimeError, FileExistsError):
        return
    raise AssertionError("Expected rejection")


def seal_fixture(data):
    """Seal synthetic projection fixtures; this is not a framework observation."""
    data["comparison"]["comparison_sha256"] = content_hash(
        {k: v for k, v in data["comparison"].items() if k != "comparison_sha256"}
    )
    data["report_sha256"] = content_hash(
        {k: v for k, v in data.items() if k != "report_sha256"}
    )
    return data


def check_structure_rendering(data):
    from html.parser import HTMLParser

    class Subjects(HTMLParser):
        def __init__(self):
            super().__init__()
            self.values = {"baseline": [], "candidate": []}

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "rect":
                self.values[attrs["data-model-side"]].append(attrs["data-subject-ref"])

    diagrams = list(_diagrams(data))
    html_text = _html(data)
    subjects = Subjects()
    subjects.feed(html_text)
    sections = _sections(data)
    coverage = next(
        b
        for b in sections
        if b[0] == "table" and b[1][0] == "Recorded structure coverage"
    )
    for column, side in enumerate(("baseline", "candidate"), 1):
        expected = [n["id"] for n in data[side]["nodes"]]
        drawn = [s for d in diagrams for s in d["subjects"][side]]
        assert subjects.values[side] == drawn == expected
        assert len(drawn) == len(set(drawn))
        refs = [r for d in diagrams for r in d["drawn_relationships"][side]]
        assert len(refs) == len(set(refs))
        assert all(0 <= r < len(data[side]["relationships"]) for r in refs)
        assert coverage[2][0][column] == f"{len(expected)} / {len(expected)}"
        assert int(coverage[2][1][column]) == len(data[side]["relationships"])
        assert int(coverage[2][2][column]) == len(refs)
        assert int(coverage[2][3][column]) == len(data[side]["relationships"]) - len(
            refs
        )
    assert "Before / After structure diff" in html_text
    assert "<th>Before</th>" in html_text and "<th>After</th>" in html_text
    assert "<th>Input shape</th>" in html_text or not data["baseline"]["invocations"]
    assert "Input structure" not in html_text and "Output structure" not in html_text
    return diagrams


def source():
    torch.manual_seed(41)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.BatchNorm2d(8),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 2),
    ).eval()


model = source()
state = copy.deepcopy(model.state_dict())
x = torch.ones(2, 3, 16, 16)
base = optimize(model, rules=[], output_dir=out / "baseline")
candidate = optimize(
    model,
    objective="structure",
    rules=[{"kind": "inverted", "subject": "3", "expansion": 2}],
    example_args=(x,),
    output_dir=out / "selected",
    rationale="Preserve the first convolution; inspect only the second block.",
)
d = report_data(out / "selected")
check_structure_rendering(d)
view = diff_view(d)
assert {e["id"]: e["status"] for e in view["entries"] if e["status"] != "context"} == {
    n["id"]: n["status"] for n in d["comparison"]["nodes"]
}
entry = next(e for e in view["entries"] if e["id"] == "3")
assert entry["rules"] == [1] and entry["status"] == "changed"
assert '--- Before/"3"' in entry["patch"] and '+++ After/"3"' in entry["patch"]
assert '"out_channels": 8' in entry["patch"]
assert d["comparison"]["totals"]["baseline"]["trainable_elements"] == "842"
assert d["comparison"]["totals"]["candidate"]["trainable_elements"] == "666"
assert d["baseline"]["cost"]["macs"] == "405536"
assert d["candidate"]["cost"]["macs"] == "315424"
assert d["static_deltas"]["state_bytes"] == "-704"
assert d["cost_delta"] == "-90112"
assert any(
    c["outputs"][0]["logical_bytes"] == "32768"
    for c in d["candidate"]["invocations"]
    if c["subject"] == "3.expand"
)
assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
assert d["context"]["selection"] == "explicit_rules"
assert d["changes"][0]["affected_nodes"] == [
    n for n in d["comparison"]["nodes"] if n["id"] == "3" or n["id"].startswith("3.")
]
pdf = candidate.export_report(out / "selected.pdf")
assert pdf.read_bytes().startswith(b"%PDF-")
assert _pdf(d) == pdf.read_bytes(), "PDF must be repeatable"
pdf_text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
assert "Field-level Before / After diff" in pdf_text
compact_text = "".join(pdf_text.split())
for entry in view["entries"]:
    for field in entry["fields"]:
        assert "".join((field["path"] or "(whole subject)").split()) in compact_text
assert export_report(out / "selected", pdf) == pdf
wrong = out / "occupied.pdf"
wrong.write_bytes(b"keep")
reject(lambda: export_report(out / "selected", wrong))
assert wrong.read_bytes() == b"keep"
print(
    "PASS Torch replacement, common costs, byte counts, unchanged source and deterministic exports",
    flush=True,
)

fold = optimize(model, example_args=(x,), output_dir=out / "fold")
f = report_data(out / "fold")
assert f["baseline"]["cost"]["macs"] == f["candidate"]["cost"]["macs"]
assert f["changes"][0]["rule"] == "fold_conv_bn"
assert f["comparison"]["totals"]["candidate"]["state_bytes"] == "3304"
torch.testing.assert_close(model(x), load_model(out / "fold")(x))
fold.export_report(out / "fold.pdf")
no = report_data(out / "baseline")
assert no["changes"] == [] and no["baseline"]["cost"]["macs"] is None
assert diff_view(no)["changed_entries"] == 0
# Presentation fixtures exercise differences that do not change module verdicts.
state_only = copy.deepcopy(no)
tensor = state_only["candidate"]["storage"][0]
tensor["payload_sha256"] = "0" * 64
reject(lambda: diff_view(seal_fixture(state_only)))
next(row for row in state_only["comparison"]["tensors"] if row["id"] == tensor["id"])["payload_equal"] = False
state_rows = diff_view(seal_fixture(state_only))["entries"]
assert any("state" in e["tags"] and e["status"] == "unchanged" for e in state_rows)
assert not any("structure" in e["tags"] for e in state_rows)
edge_only = copy.deepcopy(no)
edge_only["candidate"]["relationships"][0]["from"] = "1"
edge_rows = diff_view(seal_fixture(edge_only))["entries"]
assert any("connections" in e["tags"] for e in edge_rows)
assert all(e["status"] == "unchanged" for e in edge_rows)
reordered = copy.deepcopy(no)
reordered["candidate"]["nodes"] = list(reversed(reordered["candidate"]["nodes"]))
assert diff_view(seal_fixture(reordered))["entries"][0]["status"] == "context"
assert "shared_module_source_order" in diff_view(reordered)["entries"][0]["patch"]
renamed = copy.deepcopy(no)
for n in renamed["candidate"]["nodes"]:
    if n["id"] == "3":
        n["id"] = "renamed_spatial"
for r in renamed["candidate"]["relationships"]:
    for endpoint in ("from", "to"):
        if r[endpoint] == "3":
            r[endpoint] = "renamed_spatial"
for n in renamed["comparison"]["nodes"]:
    if n["id"] == "3":
        n["status"] = "removed"
renamed["comparison"]["nodes"].append({"id": "renamed_spatial", "status": "added"})
rename_rows = diff_view(seal_fixture(renamed))["entries"]
assert next(e for e in rename_rows if e["id"] == "3")["after"] is None
assert next(e for e in rename_rows if e["id"] == "renamed_spatial")["before"] is None
print("PASS Folding scope, exact state and no-input / no-change reports", flush=True)

# A saved report must not need its original factory module at all.
factory = out / "temporary_factory.py"
factory.write_text(
    "import torch\ndef build():\n return torch.nn.Sequential(torch.nn.Linear(2,2))\n"
)
sys.path.insert(0, str(out))
import temporary_factory

c = optimize(
    temporary_factory.build(),
    rules=[],
    factory="temporary_factory:build",
    output_dir=out / "factory",
)
factory.unlink()
sys.path.remove(str(out))
sys.modules.pop("temporary_factory", None)
env = os.environ.copy()
env["PYTHONPATH"] = os.environ.get(
    "DEEPBOM_TEST_PACKAGE_ROOT", str(ROOT / "channels/python/src")
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "deepbom.optimization",
        "report",
        str(out / "factory"),
        "--output",
        str(out / "factory.pdf"),
    ],
    check=True,
    env=env,
    stdout=subprocess.PIPE,
)
print("PASS CLI saved-package PDF with original factory unavailable", flush=True)

tampered = out / "tampered"
shutil.copytree(out / "selected", tampered)
with (tampered / "report-source.json").open("a") as stream:
    stream.write(" ")
reject(lambda: export_report(tampered, out / "bad.pdf"))
assert not (out / "bad.pdf").exists()
wrong_comparison = out / "comparison-mismatch"
shutil.copytree(out / "selected", wrong_comparison)
comparison_path = wrong_comparison / "comparison.json"
comparison = json.loads(comparison_path.read_text())
comparison["tensors"][0]["payload_equal"] = not comparison["tensors"][0][
    "payload_equal"
]
comparison["comparison_sha256"] = content_hash(
    {k: v for k, v in comparison.items() if k != "comparison_sha256"}
)
comparison_path.write_text(json.dumps(comparison))
manifest = json.loads((wrong_comparison / "manifest.json").read_text())
manifest["files"]["comparison.json"] = hashlib.sha256(
    comparison_path.read_bytes()
).hexdigest()
manifest["manifest_sha256"] = content_hash(
    {k: v for k, v in manifest.items() if k != "manifest_sha256"}
)
(wrong_comparison / "manifest.json").write_text(json.dumps(manifest))
reject(lambda: report_data(wrong_comparison))
legacy = out / "legacy"
shutil.copytree(out / "selected", legacy)
(legacy / "report-source.json").unlink()
manifest = json.loads((legacy / "manifest.json").read_text())
manifest["files"].pop("report-source.json")
manifest["manifest_sha256"] = content_hash(
    {k: v for k, v in manifest.items() if k != "manifest_sha256"}
)
(legacy / "manifest.json").write_text(json.dumps(manifest))
reject(lambda: report_data(legacy))
reject(lambda: report_data(legacy, baseline=out / "fold"))
old = report_data(legacy, baseline=out / "baseline")
assert old["candidate"]["cost"]["macs"] is None
assert old["context"]["selection"] == "not_recorded"
print(
    "PASS Integrity rejection and explicit older-package baseline fallback", flush=True
)

import tensorflow as tf, keras

keras.utils.set_random_seed(51)
inputs = keras.Input((8, 8, 3), name="image")
y = keras.layers.Conv2D(8, 3, padding="same", name="conv")(inputs)
y = keras.layers.BatchNormalization(name="bn")(y)
y = keras.layers.ReLU(name="relu")(y)
y = keras.layers.GlobalAveragePooling2D(name="pool")(y)
model = keras.Model(inputs, keras.layers.Dense(2, name="classifier")(y))
candidate = optimize(
    model,
    objective="structure",
    example_args=(tf.ones((2, 8, 8, 3)),),
    output_dir=out / "keras",
)
k = report_data(out / "keras")
check_structure_rendering(k)
assert k["baseline"]["cost"]["macs"] == "27680", k["baseline"]["invocations"]
assert k["candidate"]["cost"]["macs"] == "7712", k["candidate"]["invocations"]
assert k["context"]["selection"] == "automatic_rules"
candidate.export_report(out / "keras.pdf")
print(
    "PASS Keras nested inverted shapes, shared Conv/Gemm cost and official PDF",
    flush=True,
)
print("REPORTS", out, flush=True)

# Unicode and adversarial markup remain data; long hierarchies paginate in full.
from collections import OrderedDict

long_model = torch.nn.Sequential(
    OrderedDict(
        [("한글<script>", torch.nn.Linear(2, 2))]
        + [(f"stage_{i:02d}", torch.nn.Identity()) for i in range(40)]
    )
)
long_candidate = optimize(long_model, rules=[], output_dir=out / "long")
long_candidate.export_report(out / "long.pdf")
html_text = (out / "long" / "report.html").read_text()
assert "한글&lt;script&gt;" in html_text and "한글<script>" not in html_text
assert html_text.count("<svg ") == 3
assert "stage_39" in html_text
assert "model-diff-data" in html_text
assert "\\u003cscript\\u003e" in html_text
long_data = report_data(out / "long")
diagrams = check_structure_rendering(long_data)
assert len(long_data["baseline"]["nodes"]) == 42
assert len(long_data["baseline"]["relationships"]) == 41
assert sum(len(d["drawn_relationships"]["baseline"]) for d in diagrams) == 17
duplicate_subject = copy.deepcopy(long_data)
duplicate_subject["baseline"]["nodes"].append(duplicate_subject["baseline"]["nodes"][0])
reject(lambda: _sections(duplicate_subject))
print(
    "PASS Complete paginated hierarchy, Unicode preservation and escaped markup",
    flush=True,
)

# A large presentation-only fixture verifies bounded UI navigation without
# pretending that synthetic rows are a newly measured framework model.
large = copy.deepcopy(no)
large["changes"] = []
for side in ("baseline", "candidate"):
    large[side]["nodes"] = [
        {
            "id": f"block_{i:04d}",
            "kind": "test.Identity",
            "config": {},
            "state_refs": [],
            "mode": "eval",
        }
        for i in range(1500)
    ]
    large[side]["relationships"] = []
    large[side]["storage"] = []
    large[side]["invocations"] = []
large["comparison"]["nodes"] = [
    {"id": n["id"], "status": "unchanged"} for n in large["baseline"]["nodes"]
]
for i in (5, 700, 1499):
    large["candidate"]["nodes"][i]["config"]["scale"] = 2
    large["comparison"]["nodes"][i]["status"] = "changed"
large["comparison"]["tensors"] = []
assert diff_view(seal_fixture(large))["changed_entries"] == 3
(out / "large-diff.html").write_text(_html(large))
(out / "state-only-diff.html").write_text(_html(state_only))
(out / "edge-only-diff.html").write_text(_html(edge_only))
(out / "rename-diff.html").write_text(_html(renamed))
print(
    "PASS Common verdict reuse, state-only / connection-only changes, source-order context and 1,500-module diff fixture",
    flush=True,
)
