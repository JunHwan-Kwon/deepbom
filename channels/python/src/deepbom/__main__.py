"""Execute the packaged DEEPBOM engine without reimplementing analysis."""

from __future__ import annotations

import os
import subprocess
import sys

from ._engine import _verified_engine


def main() -> int:
    try:
        engine, asset_root = _verified_engine()
        environment = os.environ.copy()
        if asset_root is not None:
            environment["DEEPBOM_RUNTIME_ASSET_DIR"] = str(asset_root)
        completed = subprocess.run([str(engine), *sys.argv[1:]], check=False, env=environment)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"deepbom: {error}", file=sys.stderr)
        return 2
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
