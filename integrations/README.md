# DeepBOM integration kit

These adapters expose DeepBOM's artifact-evidence contract without claiming
endorsement, adoption, or conformance by another project or organization.

- `bom-authoring/`: neutral evidence input for any authoring pipeline
- `model-store/`: explicit-manifest local batch audit
- `pipeline/`: callback-based evidence attachment for Python pipelines
- `pre-commit/`: local defect gate for staged artifact files
- `oci/`: unsigned, digest-bound evidence predicate

Named third-party connectors, marketplace listings, and namespace reservations
remain outside this repository until their owners accept the integration. The
generic adapters are the technical seam for such future work; they do not make
those relationships current facts.
