# DEEPBOM Cargo launcher

`cargo install deepbom` installs a small native launcher for the same generated
DEEPBOM analysis engine used by the npm and Python channels.

On first execution, the launcher downloads the engine and TFLite WASM matching
its exact package version and operating-system/architecture tuple from the
corresponding immutable GitHub Release. It validates the strict release matrix,
declared byte lengths, and both SHA-256 digests before committing files beneath
`~/.deepbom/engines/<version>/`. Later runs revalidate the cached files before
execution and work offline.

```console
cargo install deepbom
deepbom audit model.tflite --compact
deepbom audit model.onnx --format cyclonedx
deepbom audit model.onnx --format sarif --output deepbom.sarif --fail-on high
deepbom capabilities --compact
deepbom engine verify
```

Supported release targets are Windows, Linux, and macOS on x86-64 and ARM64.
Set `DEEPBOM_HOME` to relocate the verified cache. A deliberate external engine
override requires both `DEEPBOM_ENGINE` and its exact
`DEEPBOM_ENGINE_SHA256`; no unbound executable is accepted.

The launcher contains no parser or numerical-analysis implementation. Static
provider/delegate placement remains predicted or conditionally eligible unless
identity-bound runtime evidence is imported.
