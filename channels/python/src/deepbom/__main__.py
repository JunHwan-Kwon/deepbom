"""Execute the packaged DEEPBOM engine without reimplementing analysis."""

from __future__ import annotations

import os
import subprocess
import sys

from ._engine import _verified_engine


def main() -> int:
    try:
        if len(sys.argv) > 1 and sys.argv[1] in (
            "optimize",
            "training",
            "optimization-report",
        ):
            if sys.argv[1] == "optimization-report":
                from .optimization.__main__ import main as native_main

                return native_main(["report", *sys.argv[2:]])
            if sys.argv[1] == "optimize":
                from .optimization.__main__ import main as native_main

                return native_main(["optimize", *sys.argv[2:]])
            from .training.__main__ import main as training_main

            return training_main(sys.argv[2:])
        engine, asset_root = _verified_engine()
        environment = os.environ.copy()
        if asset_root is not None:
            environment["DEEPBOM_RUNTIME_ASSET_DIR"] = str(asset_root)
        completed = subprocess.run(
            [str(engine), *sys.argv[1:]], check=False, env=environment
        )
        if (
            len(sys.argv) == 2
            and sys.argv[1] in ("--help", "-h")
            and completed.returncode == 0
        ):
            print(
                "\nOptional Python native workflows (install framework extras):\n  deepbom optimize --help\n  deepbom optimization-report --help\n  deepbom training --help\n  python -m deepbom.optimization --help  # suggest / optimize / compare / report\nLocal factories execute trusted Python; saved-package reports do not execute models. Remote static MCP is unchanged."
            )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"deepbom: {error}", file=sys.stderr)
        return 2
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
