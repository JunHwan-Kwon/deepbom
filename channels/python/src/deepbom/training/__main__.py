"""Read, import and project local Training evidence."""

import argparse
import json
from .integrations import import_logs, export, read_run
from .viewer import write_report, serve


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("view")
    x.add_argument("directory")
    x.add_argument("--output")
    x.add_argument("--port", type=int, default=0)
    x = sub.add_parser("inspect")
    x.add_argument("directory")
    x = sub.add_parser("import")
    x.add_argument("source", choices=["csv", "tensorboard", "mlflow", "wandb"])
    x.add_argument("--output", required=True)
    x.add_argument("--logdir")
    x.add_argument("--tracking-uri")
    x.add_argument("--run-id")
    x.add_argument("--project")
    x = sub.add_parser("export")
    x.add_argument("directory")
    x.add_argument("--destination", choices=["mlflow", "tensorboard"], required=True)
    x.add_argument("--tracking-uri")
    x.add_argument("--run-id")
    x.add_argument("--logdir")
    a = p.parse_args(argv)
    if a.command == "view":
        if a.output:
            print(write_report(a.directory, a.output))
            return 0
        server = serve(a.directory, a.port)
        print(f"http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    elif a.command == "inspect":
        print(json.dumps(read_run(a.directory)["training"], indent=2))
    elif a.command == "import":
        print(
            import_logs(
                a.source,
                output=a.output,
                logdir=a.logdir,
                tracking_uri=a.tracking_uri,
                run_id=a.run_id,
                project=a.project,
            )
        )
    else:
        print(
            json.dumps(
                export(
                    a.directory,
                    destination=a.destination,
                    tracking_uri=a.tracking_uri,
                    run_id=a.run_id,
                    logdir=a.logdir,
                ),
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        raise SystemExit(str(error))
