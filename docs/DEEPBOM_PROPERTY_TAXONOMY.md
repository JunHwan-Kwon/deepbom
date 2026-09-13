# DeepBOM property taxonomy

Status: vendor contract. This document describes DeepBOM-owned CycloneDX 1.7 properties. It does not claim registration, standardization, endorsement, or conformance beyond schema-valid CycloneDX output.

The complete exact-name inventory is generated at
`docs/DEEPBOM_PROPERTY_TAXONOMY.v1.json`. Names used by artifact reconciliation
are `stable`; historical JSON-string names are `deprecated`; every remaining
emitted vendor property is `experimental` and is not compared by default.

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

`deepbom verify <artifact> --bom <bom>` compares only the stable reconciliation properties above plus supported standard fields. Unknown vendor properties are reported as `unsupported_mapping`; they are never silently treated as verified. Publisher declarations such as authorship, licensing, intended use, training data, safety, quality, and regulatory status are reported as not assessable from artifact bytes.

Namespace reservation or third-party interpretation is intentionally outside
this document. A separate external process may be considered only after these
names remain stable through a public release and consumer testing.
