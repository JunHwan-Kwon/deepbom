# DEEPBOM release channels

These adapters expose one analysis implementation through several installation
surfaces. They do not contain format parsers or numerical analysis logic.

| Channel | Runtime | Analysis implementation |
| --- | --- | --- |
| npm / npx | Node.js | Bundled `bin/deepbom.mjs` plus the canonical TFLite WASM |
| Python / pip | OS/architecture wheel | The same Node single-executable engine plus the same WASM asset, both manifest-hash verified before launch |
| Cargo | Native launcher | A verified local DEEPBOM engine selected through `DEEPBOM_ENGINE` |
| Hugging Face Space | Browser | The same generated `dist/` browser bundle used by deepbom.org |

`scripts/check-channel-equivalence.mjs` installs the generated npm tarball and
Python wheel into isolated environments, then compares complete parsed JSON
from all runnable adapters for real single-file and multi-file artifacts. A
channel release is invalid if any value diverges.

The Cargo launcher deliberately does not download an engine implicitly. A
cross-platform engine manifest and signed release location must be established
before crates.io publication. This keeps installation behavior and supply-chain
claims explicit.

`.github/workflows/release-channels.yml` is manual-only. It builds and installs
six platform wheels, verifies their manifests and wheel `RECORD` ledgers,
compares channel output, and can publish npm and PyPI artifacts only from the
exact `channels-v<version>` tag. Publication uses registry-bound OpenID Connect
Trusted Publishers and the protected GitHub environments `npm` and `pypi`.
No npm or PyPI publishing credential is stored in GitHub Secrets, the workflow,
the package, or the repository. Private GitHub repositories do not receive npm
provenance attestations; publishing identity is still bound through OIDC.

## License boundary

Published npm, PyPI, and launcher-channel artifacts are licensed under
Apache-2.0 using `channels/LICENSE`. The private monorepo, protected analyzer,
rulepack generators, private roadmap, and unreleased research modules are not
relicensed by channel publication. The channel build uses an esbuild input
graph check and an exact package-member check to prevent those private sources
from entering a public distribution.
