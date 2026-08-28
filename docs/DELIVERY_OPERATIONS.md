# Delivery Operations

DEEPBOM separates public installation-channel delivery from private hosted-web
delivery. Both surfaces consume the same reviewed analyzer sources and release
schemas, but they do not share credentials or automatic triggers.

## Measured baseline

The machine-readable record is `config/delivery-operations.v1.json`.

| Observation | Result | Wall time |
| --- | --- | ---: |
| Public channels 1.94.4 | All platform and registry jobs completed; later superseded after a fresh Cargo install exposed Windows stack pressure | 443 s critical path |
| Public channels 1.94.5, attempt 1 | Every product gate, npm, and PyPI passed; Cargo alone failed because the token lacked `publish-update` | 529 s critical path |
| Public channels 1.94.5, failed-job retry | Cargo publication only | 21 s |
| Fresh crates.io 1.94.5 smoke | install, local version, engine SHA-256, and real ONNX audit passed | pass |
| Private web 1.94.5 preflight | 58 bounded parser, export, cache, privacy, and deployment-contract checks; no browser or corpus matrix | 140.6 s |
| Private web 1.94.6 preflight | same bounded 58-check production contract after release-identity alignment | 147.4 s |
| Local Windows channel build | npm, single executable, Python wheel, and Cargo launcher source after generated-directory filtering | 145.757 s |
| Local Windows platform smoke | installed Python and standalone TFLite/WASM plus ONNX parity | 198.7 s |
| Local Windows 1.94.6 channel rebuild | version-aligned npm, executable, Python wheel, and Cargo launcher source | 225.2 s |
| Local Windows 1.94.6 platform smoke | installed Python and standalone TFLite/WASM plus ONNX parity | 194.1 s |
| Local Windows full channel equivalence | installed npm and Python, five formats, two package forms, standalone/Cargo parity, and WASM tamper rejection | 327 s |
| Local Windows 1.94.6 release-contract equivalence | npm full-format and command parity, native/Python execution, Cargo binding, and npm/Python WASM tamper rejection | 446.3 s |
| Local Windows 1.94.6 public-PR channel equivalence | npm full-format and command parity plus npm WASM tamper rejection; native/Python/Cargo execution reserved for release | 51.7 s |
| Packaged public CLI corpus sweep | 113 hash-bound artifacts; bounded human and compact JSON output in fresh processes | 403.9 s |

The 1.94.4 run is retained as timing and incident evidence. It is not a known-good
quality baseline. A registry success is never treated as sufficient without a
fresh installation and real artifact execution.

The first local channel rebuild exceeded 604 seconds because a recursive source
copy included 3,989 generated Cargo files (1,380 MiB) from
`channels/cargo/target`. Channel assembly now excludes Cargo `target` and Python
build, dist, cache, and egg-info directories before packaging.

## Trigger policy

- Public quality runs only for pull requests that touch executable, corpus,
  package, workflow, or configuration paths, or by manual dispatch. It never
  runs for every push.
- Public channel publication is manual-only and requires an exact
  `channels-v<version>` tag when `publish=true`.
- Private full quality is manual-only and requires `DEEPBOM_ENABLE_CI=true`.
- Private Cloudflare delivery is manual-only and requires
  `DEEPBOM_ENABLE_DEPLOY=true`.
- Documentation-only commits do not consume the public browser/package matrix.
- The downloaded 113-artifact packaged CLI sweep runs locally for release
  qualification when shared parser, CLI, or corpus contracts change. It is not
  part of ordinary pull-request or web-deployment workflows.

## Gate ownership

| Change | Required local gate | Hosted gate |
| --- | --- | --- |
| Web layout, copy, or browser orchestration | `npm run check:deploy`, `npm run build:release`, dist asset and budget checks | Manual private web deploy |
| Parser or shared analysis calculation | `npm run check:cli`, `npm run check:formats`, affected exact check, then `npm run check:release` before publication | Public PR quality; private full quality when hosted UI is affected |
| Public package adapter | package boundary, channel build, npm full-format/command equivalence | Public PR quality; `--release-contract` only in the manual channel release |
| Private WASM or rulepack generator | private WASM/rulepack checks plus public boundary checks | Manual private full quality and web deploy |
| Corpus manifest or aggregate evidence | corpus-specific deterministic verifier | Public PR quality only when public evidence changes |

The web deployment gate is the bounded `scripts/check-deploy.mjs` set. The
exhaustive release tier remains available as `npm run check:release`; it must not
be wired back into every Cloudflare deployment. The measured preflight budget is
240 seconds; the current successful baseline is 147.4 seconds.

## Cross-platform release rule

Every release still builds and executes the native engine and Python wheel on
Windows, Linux, and macOS for x86-64 and ARM64. Platform jobs run TFLite/WASM
and ONNX parity. The platform-independent npm package, Cargo adapter, five file
formats, Core ML package, sharded SafeTensors repository, and WASM tamper checks
run once on Linux x64. This removes duplicate work without reducing platform
execution coverage. Public pull requests run the 51.7-second npm package gate;
they do not rebuild and launch the same 92 MiB native engine through standalone,
Python, and Cargo adapters. The manual release invokes `--release-contract` to
close those adapter execution contracts exactly once.

## Retry rule

Do not dispatch a new release when one terminal registry job fails after all
immutable artifacts and upstream gates have passed. Correct the external state
and use GitHub's `rerun failed jobs` operation. A full rerun is required only if
source, tag, release assets, a dependency job, or a verification result changed.

Crate yanking is isolated in `crates-maintenance.yml`. It is manual-only, fixes
the crate name to `deepbom`, requires an exact confirmation phrase, and uses the
`crates-io` environment.

## Public and private alignment

The private monorepo is the source of truth. The public repository is rebuilt
from `config/public-source-files.v1.txt`, receives a clean-source manifest, and
must pass source-boundary, privacy, and channel-equivalence checks. Shared
calculations are changed once in the private source and exported; they are not
maintained as independent web and CLI implementations.

Public corpus records contain licensed small fixtures, pinned source identities,
hashes, reproducible scripts, and reviewed aggregate/per-artifact evidence.
Large downloaded model bytes, restricted artifacts, user models, clinical data,
raw private traces, unreleased research modules, and rulepack-generation inputs
remain outside the public export. Private GitHub may retain manifests and
non-sensitive receipts; large or restricted bytes belong in access-controlled
object storage or local verified caches, not ordinary Git history.
