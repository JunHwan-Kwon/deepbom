"""Python-only autolog launcher. exec preserves the target process exit/signal."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["keras"], required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--preset", choices=["basic", "weights"], default="basic")
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 2 or command[1].startswith("-") or not Path(command[1]).is_file():
        parser.error("Expected -- <python interpreter> <local training script.py> [arguments]; -m/-c are not supported")
    if args.every < 1:
        parser.error("--every must be positive")
    config = {"framework":args.framework, "logdir":str(Path(args.logdir).resolve()), "preset":args.preset, "every":args.every}
    os.execvp(command[0], [command[0], "-m", "deepbom.instrument", "--run", json.dumps(config), *command[1:]])


def _run(config, script, argv):
    from .autolog import autolog
    sys.argv = [script, *argv]
    sys.path[0] = str(Path(script).resolve().parent)
    with autolog(**json.loads(config)):
        runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--run":
        _run(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        main()
