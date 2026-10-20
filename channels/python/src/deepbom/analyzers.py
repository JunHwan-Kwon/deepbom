"""Opt-in analysis extensions. Discovery reads package metadata, never imports plugins."""
from importlib import metadata
import json
import re
import sys
import math
import os
import tempfile
from pathlib import Path

GROUP = "deepbom.analyzers"
CONTRACT = "deepbom.analyzer_plugin.v1"
CONTRACT_V2 = "deepbom.analyzer_plugin.v2"
_OPERATIONS = {
    "snapshot": "deepbom.snapshot_ir.v1",
    "protocol": "deepbom.evaluation_protocol.v1",
    "evaluation": "deepbom.evaluation_record.v1",
    "provenance": "deepbom.provenance_ir.v2",
    "deployment": "deepbom.deployment_context.v1",
}


def _entries():
    found = metadata.entry_points()
    return list(found.select(group=GROUP) if hasattr(found, "select") else found.get(GROUP, ()))


def _manifest(entry):
    if entry.dist is None:
        raise ValueError(f"Analyzer {entry.name} has no distribution metadata")
    text = entry.dist.read_text("deepbom-analyzers.json")
    if text is None or len(text.encode("utf-8")) > 65536:
        raise ValueError(f"Analyzer {entry.name} needs a bounded deepbom-analyzers.json metadata file")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate analyzer metadata key")
            result[key] = value
        return result
    doc = json.loads(text, object_pairs_hook=unique)
    if set(doc) != {"schema", "analyzers"} or doc["schema"] not in (CONTRACT,CONTRACT_V2) or not isinstance(doc["analyzers"], list) or len(doc["analyzers"]) > 128:
        raise ValueError("Unsupported analyzer metadata contract")
    selected = [row for row in doc["analyzers"] if row.get("name") == entry.name]
    if len(selected) != 1:
        raise ValueError("Analyzer metadata name is missing or ambiguous")
    row = selected[0]
    fields={"name", "version", "operation", "result_schema", "required_inputs", "executes_model", "cost", "dependencies"}
    if doc["schema"]==CONTRACT_V2: fields.add("execution")
    if set(row) != fields:
        raise ValueError("Unknown or missing analyzer metadata fields")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", row["name"]) or not re.fullmatch(r"\d+\.\d+\.\d+", row["version"]):
        raise ValueError("Invalid analyzer name or version")
    if _OPERATIONS.get(row["operation"]) != row["result_schema"]:
        raise ValueError("Analyzer result schema has no supported common workflow")
    if not isinstance(row["executes_model"], bool) or not isinstance(row["cost"], str) or not row["cost"]:
        raise ValueError("Analyzer must declare execution and cost boundaries")
    for key in ("required_inputs", "dependencies"):
        values = row[key]
        if not isinstance(values, list) or len(values) > 128 or len(set(values)) != len(values) or any(not isinstance(v, str) or not v or len(v) > 200 for v in values):
            raise ValueError(f"Invalid analyzer {key}")
    missing = []
    for dependency in row["dependencies"]:
        try:
            if doc["schema"]==CONTRACT_V2:
                from packaging.requirements import Requirement
                requirement=Requirement(dependency)
                if requirement.url or requirement.extras or requirement.marker:
                    raise ValueError("Analyzer requirements must be distribution names with version specifiers")
                if metadata.version(requirement.name) not in requirement.specifier: missing.append(dependency)
            else:
                metadata.version(dependency)
        except metadata.PackageNotFoundError:
            missing.append(dependency)
    if doc["schema"]==CONTRACT_V2:
        _execution_contract(row["execution"])
    return {**row, "distribution": entry.dist.metadata["Name"], "distribution_version": entry.dist.version,
            "missing_dependencies": missing, "available": not missing,
            "descriptor_schema":doc["schema"],"runner":"legacy_trusted_in_process" if doc["schema"]==CONTRACT else "trusted_process"}


def _execution_contract(value):
    if not isinstance(value,dict) or set(value)!={"python","platforms","network","gpu","limits"}:
        raise ValueError("Invalid analyzer execution contract")
    from packaging.specifiers import SpecifierSet
    SpecifierSet(value["python"])
    if not isinstance(value["platforms"],list) or not value["platforms"] or any(x not in ("linux",) for x in value["platforms"]):
        raise ValueError("Only qualified Linux trusted-process execution is supported")
    if value["network"] not in ("inherit","none") or value["gpu"] not in ("not_requested","required","forbidden"):
        raise ValueError("Invalid network/GPU requirements")
    limits=value["limits"]
    if not isinstance(limits,dict) or set(limits)!={"wall_seconds","cpu_seconds","address_space_bytes","file_bytes"}:
        raise ValueError("Invalid analyzer resource limits")
    if type(limits["wall_seconds"]) not in (int,float) or not math.isfinite(limits["wall_seconds"]) or not 0<limits["wall_seconds"]<=300:
        raise ValueError("Invalid analyzer deadline")
    for key in ("cpu_seconds","address_space_bytes","file_bytes"):
        if type(limits[key]) is not int or limits[key]<=0: raise ValueError("Invalid analyzer resource limit")
    if limits["cpu_seconds"]>300 or limits["file_bytes"]>128*1024*1024 or limits["address_space_bytes"]>64*1024**3:
        raise ValueError("Analyzer limits exceed runner budget")


def runner_capabilities():
    return {"runner":"trusted_process","platform":sys.platform,"supported":sys.platform=="linux",
            "enforced":["wall_deadline_process_group_kill","RLIMIT_CPU","RLIMIT_AS","RLIMIT_FSIZE"],
            "not_enforced":["network_isolation","GPU_VRAM","filesystem_allowlist","aggregate_directory_quota"],
            "boundary":"Trusted installed code only; this is not a hostile-code sandbox. RLIMIT_AS is virtual address space, not RSS or GPU memory."}


def discover():
    """List installed descriptors without importing any analyzer or framework."""
    names, result = set(), []
    for entry in _entries():
        if entry.name in names:
            raise ValueError(f"Analyzer name collision: {entry.name}; select an unambiguous installed distribution")
        names.add(entry.name)
        result.append(_manifest(entry))
    return sorted(result, key=lambda row: row["name"])


def run(name, context, *, allow_model_execution=False, requirements=None, return_receipt=False):
    """Execute one explicitly selected trusted plugin, then use the shared engine.

    A plugin returns an Evidence Workflow request; it cannot introduce unchecked
    result schemas. No plugin is executed merely because it is installed.
    """
    descriptors = {row["name"]: row for row in discover()}
    if name not in descriptors:
        raise ValueError(f"Unknown analyzer: {name}")
    descriptor = descriptors[name]
    if not descriptor["available"]:
        raise ImportError("Missing analyzer dependencies: " + ", ".join(descriptor["missing_dependencies"]))
    if descriptor["executes_model"] and not allow_model_execution:
        raise ValueError("This analyzer declares model execution; enable allow_model_execution explicitly")
    if not isinstance(context, dict) or any(key not in context for key in descriptor["required_inputs"]):
        raise ValueError("Analyzer context is missing declared inputs")
    from .jobs import _json_primitives
    _json_primitives(context,budget=[1_000_000,16*1024*1024])
    payload = json.dumps(context, allow_nan=False)
    if len(payload.encode("utf-8")) > 16 * 1024 * 1024:
        raise ValueError("Analyzer context exceeds 16 MiB; use immutable artifact references")
    entries = [entry for entry in _entries() if entry.name == name]
    if len(entries) != 1 or _manifest(entries[0]) != descriptor:
        raise ValueError("Analyzer installation changed during selection")
    if descriptor["descriptor_schema"]==CONTRACT:
        if requirements: raise ValueError("Legacy v1 plugins do not declare or enforce requested limits")
        plugin = entries[0].load()
        analyze = getattr(plugin, "analyze", plugin)
        if not callable(analyze): raise TypeError("Analyzer entry point must be callable or expose analyze(context)")
        request = analyze(json.loads(payload))
        receipt={"runner":"legacy_trusted_in_process","applied":[],"verified":False}
    else:
        request,receipt=_run_process_plugin(name,descriptor,json.loads(payload),requirements or {})
    if not isinstance(request, dict) or request.get("operation") != descriptor["operation"]:
        raise ValueError("Analyzer returned an undeclared workflow operation")
    from .api import evidence_workflow
    result = evidence_workflow(request)
    if result["document"]["schema"] != descriptor["result_schema"]:
        raise ValueError("Analyzer returned an undeclared evidence schema")
    return {"result":result,"execution":receipt} if return_receipt else result


def _run_process_plugin(name, descriptor, context, requirements):
    from packaging.specifiers import SpecifierSet
    from ._process import run_process
    from ._native.core import atomic_json, read_json
    execution=descriptor["execution"]
    if sys.platform not in execution["platforms"] or sys.version.split()[0] not in SpecifierSet(execution["python"]):
        raise ValueError("Analyzer environment does not meet its declared requirements")
    if execution["network"]!="inherit" or execution["gpu"]!="not_requested" or requirements:
        raise ValueError("Requested mandatory isolation is unsupported by the trusted CPU process runner")
    limits=execution["limits"]
    with tempfile.TemporaryDirectory(prefix="deepbom-plugin-") as directory:
        source=Path(directory)/"request.json";output=Path(directory)/"result.json"
        atomic_json(source,{"name":name,"descriptor":descriptor,"context":context},compact=True)
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
        result=run_process([sys.executable,"-m","deepbom._plugin_runner",str(source),str(output)],environment=environment,
                           timeout=limits["wall_seconds"],file_limit=limits["file_bytes"],memory_bytes=limits["address_space_bytes"],cpu_seconds=limits["cpu_seconds"])
        if result["returncode"]!=0: raise RuntimeError(f"Analyzer process failed ({result['returncode']}): {result['diagnostic']}")
        if output.stat().st_size>limits["file_bytes"]: raise ValueError("Plugin result exceeds its output budget")
        request=read_json(output)
    return request,{"runner":"trusted_process","declared":execution,"requested":requirements,
                    "applied":runner_capabilities()["enforced"],"process_stopped":result["process_stopped"],
                    "elapsed_seconds":result["elapsed_seconds"],"boundary":runner_capabilities()["boundary"]}
