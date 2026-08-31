# DEEPBOM release channels

These adapters expose one analysis implementation through several installation
surfaces. They do not contain format parsers or numerical analysis logic.
The executable command inventory is maintained in the
[generated CLI reference](../docs/CLI_REFERENCE.md).

| Channel | Runtime | Analysis implementation |
| --- | --- | --- |
| npm / npx | Node.js | Bundled `bin/deepbom.mjs` plus the canonical TFLite WASM |
| Python / pip | OS/architecture wheel | The same Node single-executable engine plus the same WASM asset, both manifest-hash verified before launch |
| Cargo | Native launcher | Version-locked, SHA-256-verified engine assets from the immutable GitHub Release |
| Hugging Face Space | Browser | The same generated `dist/` browser bundle used by deepbom.org |

`scripts/check-channel-equivalence.mjs` installs the generated npm tarball and
Python wheel into isolated environments, then compares complete parsed JSON
from all runnable adapters for real single-file and multi-file artifacts. A
channel release is invalid if any value diverges.

The Cargo launcher accepts a deliberately supplied engine only when
`DEEPBOM_ENGINE_SHA256` binds it. Otherwise, first execution downloads the exact
version/platform engine matrix, executable, and TFLite WASM from the matching
immutable GitHub Release. Declared lengths and both SHA-256 digests are checked
before an atomic cache commit; cached files are revalidated before execution.

`.github/workflows/release-channels.yml` is manual-only. It builds and installs
six platform wheels and standalone engines, verifies their manifests and wheel
`RECORD` ledgers, compares channel output, and can publish npm, PyPI, Cargo, and
immutable GitHub Release artifacts only from the
exact `channels-v<version>` tag. Publication uses registry-bound OpenID Connect
Trusted Publishers and the protected GitHub environments `npm` and `pypi`.
No npm or PyPI publishing credential is stored in GitHub Secrets, the workflow,
the package, or the repository. The first crates.io release uses a scoped
`publish-new` bootstrap secret; subsequent Cargo releases use crates.io Trusted
Publishing and remove that secret.

## License boundary

Published npm, PyPI, and launcher-channel artifacts are licensed under
Apache-2.0 using `channels/LICENSE`. The private monorepo, protected analyzer,
rulepack generators, private roadmap, and unreleased research modules are not
relicensed by channel publication. The channel build uses an esbuild input
graph check and an exact package-member check to prevent those private sources
from entering a public distribution.
