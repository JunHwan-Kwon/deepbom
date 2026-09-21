# DeepBOM property taxonomy

Status: vendor contract. This document describes DeepBOM-owned CycloneDX 1.7 properties, including the `mlbom:` compatibility projection. It does not claim registration, standardization, endorsement, or conformance beyond schema-valid CycloneDX output.

The complete exact-name inventory is generated at
`docs/DEEPBOM_PROPERTY_TAXONOMY.v1.json`. Names used by artifact reconciliation
are `stable`; historical JSON-string names are `deprecated`; every remaining
emitted vendor property is `experimental`. The `reconciliation` column separately states whether a value is an artifact observation, a context/publisher declaration, or an unsupported mapping.

DeepBOM prefers standard fields when their defined meaning fits without loss. Scalar `deepbom:` properties provide searchable artifact observations or hashes of structured evidence. Structured arrays and objects are carried in a hash-bound evidence document, not JSON encoded inside a property value.

## Stable reconciliation properties

| Property | Meaning | Evidence boundary |
| --- | --- | --- |
| `deepbom:contract:schema` | DeepBOM export-profile identifier | Identifies the vendor contract only |
| `deepbom:model:artifactIrSchema` | Artifact IR schema identifier | Static artifact projection |
| `deepbom:model:artifactIrSha256` | Canonical Artifact IR digest | Digest does not prove provenance |
| `deepbom:model:format` | Detected serialized format | Artifact observation |
| `deepbom:model:fileSizeBytes` | Serialized artifact byte length | Artifact observation |
| `deepbom:model:hashBasis` | Identity hashing basis | Artifact or package-set basis |
| `deepbom:model:storageEncodingClassification` | Storage-only encoding classification | Does not assert runtime compute precision |
| `deepbom:model:quantizationClassification` | Static graph quantization classification | Does not assert runtime fusion or assignment |
| `deepbom:model:llmEncodingInventorySha256` | Canonical encoding-inventory digest | Static serialized tensor storage |
| `deepbom:model:llmTensorEncodingAssignmentSha256` | Canonical tensor-to-encoding assignment digest | Does not establish a quantization recipe |
| `deepbom:model:llmChatTemplateSha256` | Observed or bound chat-template digest | Does not establish runtime selection or behavior |
| `deepbom:model:interfaceContractSchema` | External-interface ledger schema | Serialized interface only |
| `deepbom:model:interfaceContractLedgerSha256` | External-interface ledger digest | Serialized interface only |
| `deepbom:model:completeAffineInterfaceCount` | Complete affine interface count | Static serialized contract |
| `deepbom:model:unquantizedInterfaceCount` | Interface count without affine mapping | Static serialized contract |
| `deepbom:model:invalidOrIncompleteInterfaceCount` | Invalid or incomplete interface count | Static serialized contract |

## Structured evidence rule

The following historical JSON-string properties are not emitted by the stabilized exporter: `deepbom:model:activationPath`, `deepbom:model:tensorDtypeInventory`, `deepbom:finding:highSeverityFindings`, `deepbom:conformance:violationCodes`, `deepbom:preprocessing:exactContractIds`, `deepbom:model:graphTotals`, `deepbom:model:artifactByteIntegritySummary`, and `deepbom:model:interfaceContractLedger`.

Their structured content belongs in the inline `declarations.evidence` attachment for standalone output or in a SHA-256-bound sidecar for a deployment-contract bundle. Scalar counts and digests may remain as properties.

## Reconciliation policy

`deepbom verify <artifact> --bom <bom>` regenerates observations from the supplied
artifact. It supports both the CLI evidence profile and the browser standalone
profile. The table above is a short introduction; the generated exact-name
inventory is authoritative. Quantization counts (INT8, UINT8, float, per-axis,
and Q/DQ), full-integer classification, graph counts, MAC coverage, and every
`mlbom:model:` compatibility field are mapped. A digest is not inverted to
recover counts: the counts and digest are independently projected from the
same newly analyzed artifact.

Every occurrence in the selected component's `properties` array produces one
comparison row, including repeated names and unknown namespaces. No first- or
last-value-wins map is used for supplied declarations. A correct SHA-256 does
not hide another contradictory or malformed SHA-256 declaration. Binding
reasons distinguish missing hashes, malformed hashes, no matching hash, and
multiple matching components. A hash mismatch alone cannot establish whether
the BOM was forged or belongs to a different artifact.

`property_coverage.supplied` equals `property_coverage.accounted`. Its `counts`
sum to the supplied property occurrences; `compared` counts only match and
mismatch. The top-level `counts` also include supported standard fields and
missing expected declarations, so they must not be read as a BOM property total.
Each supplied row includes its property index and BOM pointer. Other components,
document properties, inline evidence attachments, and declarations are outside
this reconciliation scope, which is stated in every result.

Observable property values are compared as exact exported strings. Counts are
not rounded through floating-point conversion; numeric-looking strings with
different spellings are not silently normalized. Unknown or unavailable
observations remain `not_assessable_from_artifact`, never zero or false.
Unknown names are `unsupported_mapping`. Analyzer versions, provenance,
selected build/target assumptions, runtime evidence, and publisher assertions
cannot be authenticated solely from model bytes and are reported separately.
`mlbom:` is a compatibility namespace, not a claim of standards registration.

The human headline is `NO_CONTRADICTION`, with visible property coverage.
For compatibility, JSON `gate_result: "pass"` and exit 0 mean only that no
compared declaration contradicts the artifact. They do not mean every claim
was verified. Exit 2 means an observed contradiction; exit 3 means an invalid
comparison declaration or unresolved subject. Consumers requiring complete
coverage must inspect missing, unassessable, and unsupported counts as well as
the gate; a name/hash-only BOM does not establish quantization coverage.

SARIF preserves original severity and finding kind in `deepbomSeverity` and
`deepbomFindingKind`. High artifact defects map to `error`; cautions map to at
most `warning`; evidence gaps map to `note`. This presentation does not change
explicit DEEPBOM severity or policy gates.

Namespace reservation or third-party interpretation is outside this document.
