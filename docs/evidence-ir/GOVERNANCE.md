# Maintenance and public review

Status: draft; DEEPBOM project governance. Maintainer: JunHwan Kwon. No independent committee or external approval is implied.

## Publication location

Maintain this specification, its generated catalog, schemas, examples and tests in the existing public `JunHwan-Kwon/deepbom` repository. The current source of these files is the reviewed public export from the private development repository. Do not publish the private repository or bypass its exact public-file allowlist.

A separate `evidence-ir` repository can become useful if independent implementations and maintainers emerge. It is not required now and must not become a second manually maintained schema source. Until then, one reviewed source, one generator and one release lineage are easier to audit.

GitHub makes proposals, decisions and releases inspectable. It does not grant standards status. Credibility must be supported by precise semantics, reproducible tests, transparent limitations, independent review and actual interoperable implementations.

## Change process

Input/IR/output mapping changes follow the [compatibility catalog policy](COMPATIBILITY.md).
Maintain the definition in one place, preserve versioned snapshots and source digests,
and generate the GitHub table and website data together. Catalog versions do not
replace IR schema identities, calculation methods or engine release versions.

1. Open an issue with the use case, source standard, proposed semantics, compatibility impact and a minimal valid/invalid example. Do not include confidential artifacts or clinical data.
2. Implement a proposal in a pull request. Identify affected schemas, method identities, mappings, validators, examples and consumers. Separate editorial renaming from semantic changes.
3. Run applicable local checks and attach their results. GitHub Actions is not required. Keep review and unresolved questions in the pull request; document who reviewed and merged the change.
4. Update the change log and publish a tagged, versioned release with the schemas/catalog and a digest manifest. Pin published examples to that tag or commit. Do not silently replace a release's schema contents; corrections require an explicit new release.
5. Record external implementations and reviews only when independently verifiable. Publish supported member/method matrices and known failures instead of claiming blanket compliance.

The maintainer currently makes merge and release decisions. This is project stewardship, not community consensus. If governance broadens, name the additional maintainers and decision procedure in an explicit reviewed change.

## Stable identifiers and hosting

The canonical schema identifiers use `https://deepbom.org/schemas/`. The web build copies the authoritative schema files to those paths and publishes the generated catalog at `/schemas/evidence-ir-catalog.json`. These URLs become reachable only after deployment; `$id` is an identity, not proof that a URL is already live. Use a locally registered copy or a pinned GitHub release for reproducible offline validation.

Maintain versioned schema identities and explicit method compatibility. Preserve previous published releases as archives; the current runtime need not retain retired contracts or aliases. Incompatible changes require an explicit migration record and coordinated consumer update. Do not label a DEEPBOM rule as an OMOP, CycloneDX, SPDX, TEA or W3C requirement unless the cited standard actually imposes it.

References: [GitHub versioned releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases), [JSON Schema identifiers and references](https://json-schema.org/understanding-json-schema/structuring), [W3C PROV-DM](https://www.w3.org/TR/prov-dm/).
