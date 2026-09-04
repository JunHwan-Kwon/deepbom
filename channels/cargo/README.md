# DEEPBOM Cargo launcher

`cargo install deepbom` installs a small native launcher for the same generated
DEEPBOM analysis engine used by the npm and Python channels.

On first execution, the launcher downloads the engine, TFLite WASM, and bounded
self-test probe matching its exact package version and operating-system/
architecture tuple from the corresponding immutable GitHub Release. It
validates the strict release matrix, declared byte lengths, and every SHA-256
digest before committing files beneath `~/.deepbom/engines/<version>/`. Later
runs revalidate the cached files before execution and work offline.

```console
cargo install deepbom
deepbom audit model.tflite --compact
deepbom audit model.onnx --format cyclonedx
deepbom audit model.onnx --format sarif --output deepbom.sarif --fail-on high
deepbom capabilities --compact
deepbom self-test
deepbom engine verify
```

Supported release targets are Windows, Linux, and macOS on x86-64 and ARM64.
Set `DEEPBOM_HOME` to relocate the verified cache. A deliberate external engine
override requires both `DEEPBOM_ENGINE` and its exact
`DEEPBOM_ENGINE_SHA256`; no unbound executable is accepted.

The launcher contains no parser or numerical-analysis implementation. Static
provider/delegate placement remains predicted or conditionally eligible unless
identity-bound runtime evidence is imported.

For the installed version, `deepbom --help` and `deepbom capabilities --compact`
are authoritative. The current source inventory is maintained in the
[generated CLI reference](https://github.com/JunHwan-Kwon/deepbom/blob/main/docs/CLI_REFERENCE.md).
