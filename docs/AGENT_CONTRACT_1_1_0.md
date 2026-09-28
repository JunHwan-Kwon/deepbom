# Agent contract 1.1.0 review candidate

Engine release: 1.109.0. Reviewed locally on 2026-09-28. This records an intentional additive agent-contract change; it does not assert marketplace submission, approval or an actual ChatGPT/Claude host review.

- Local `deepbom_audit` accepts optional `metadata_template`, `metadata` and `evidence_files` inputs. Existing inputs, tools, allowed-root controls and output defaults are retained. An explicitly requested incomplete metadata result preserves its JSON and exit-code explanation.
- Remote widgets add an optional metadata workspace. Existing static results remain valid. The ChatGPT result schema adds one optional, closed-schema `evidence_links` summary. Both remote bridges validate its exact hashes, safe counts, count partitions and source identity before returning it.
- Host summaries contain only counts, status and digests. They exclude imported OMOP rows, institution/release identifiers, report text and supporting file bytes. The user explicitly requests sharing. No new server storage, authentication scope, tool or remote metadata fetch is added.
- Exporting through a host file API remains a separate user action that shares the selected generated file with that host. The workspace explains this distinction.
- Artifact Evidence Envelope remains 1.0.0. Model/Artifact IR meanings and weight/activation contracts are unchanged. Metadata truth, execution and publisher authenticity are never inferred from hashes.

Validation includes local stdio transport parity, both remote RPC bridges, hostile summary counts/extra fields, and browser widget/file export checks. The snapshot is intentionally marked `candidate_unsubmitted`. Existing directory entries do not become approved for new metadata automatically; use the generated candidate manifest and refreshed host testing when updating a listing. An ongoing external review is not withdrawn or replaced by this deployment.

The release gate baseline was refreshed explicitly for this feature review, not as part of routine engine version synchronization.
