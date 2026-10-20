"""Private entrypoint; called only after the parent has installed OS limits."""
import sys
from pathlib import Path
from .analyzers import _entries, _manifest
from ._native.core import read_json, atomic_json
from .jobs import _json_primitives

if __name__ == "__main__":
    source,output=map(Path,sys.argv[1:])
    if source.stat().st_size>17*1024*1024: raise ValueError("Plugin request exceeds budget")
    request=read_json(source)
    entries=[e for e in _entries() if e.name==request["name"]]
    if len(entries)!=1 or _manifest(entries[0])!=request["descriptor"]:
        raise ValueError("Plugin descriptor changed before isolated import")
    plugin=entries[0].load()
    function=getattr(plugin,"analyze",plugin)
    if not callable(function): raise TypeError("Analyzer must expose analyze(context)")
    result=function(request["context"])
    _json_primitives(result,budget=[1_000_000,request["descriptor"]["execution"]["limits"]["file_bytes"]])
    atomic_json(output,result,compact=True)
