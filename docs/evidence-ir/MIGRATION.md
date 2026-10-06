# Transition to the canonical Provenance IR contract

Status: released with DEEPBOM 2.0.0 on September 29, 2026. This is a breaking provenance transition; the earlier presentation-only alias approach is superseded. It does not change the Artifact, Model, Weight or Activation IR contracts. The Evidence IR specification remains a project-maintained draft.

## One current contract

| Surface | Current contract |
| --- | --- |
| Generic input | `deepbom.provenance_input.v1` |
| OMOP input | `deepbom.omop_metadata_input.v1` (unchanged) |
| IR identity | `deepbom.provenance_ir.v1` |
| IR digest and excluded hash pointer | `provenance_ir_sha256`, `/provenance_ir_sha256` |
| Full analysis and CLI/MCP selection | `provenance_ir` |
| Bounded summary | `deepbom.provenance_summary.v1` |
| Remote result member | `provenance_summary` |
| ChatGPT / browser result envelope | ChatGPT v1 base transport and v2 provenance transport; `deepbom.browser_analysis_result.v2` |
| JSON export | `deepbom_provenance_ir.json` |
| CycloneDX extension properties / references | `deepbom:provenance:*` / `deepbom-provenance:*` |
| Implementation | `web/lib/provenance-ir.js` with helpers in `web/lib/provenance/` |
| JSON Schema | `docs/schemas/deepbom-provenance-ir-v1.schema.json` |

The schema root validates an IR document. Its `$defs` also expose `generic_input`, `omop_input`, and `summary` for local validation. There is no second compatibility schema or re-export facade.

## Retired documents

The retired `evidence_link_ir` selector, `deepbom.evidence_link_ir.v1` identity, `evidence_link_ir_sha256` digest field, generic input and host summary contracts are not accepted by current consumers. Their identifiers occur only in negative tests, this migration record and historical review records. Old files are not deleted from users' disks or rewritten in Git history.

To produce current evidence:

1. Keep the original model and supporting files, their recorded digests, and the archived old report.
2. Use DEEPBOM 2.0.0 or a later compatible CLI to create a fresh `--metadata-template generic` or `--metadata-template omop` input. Transfer the intended declarations into that input and review them. Do not claim observed checks merely by copying old report rows.
3. Analyze the exact original model with `--metadata`, explicitly supplied `--evidence-files`, and `--section provenance_ir --json`.
4. Validate the new schema, source bindings and semantic checks. Store the new document independently.

New documents have new digests because their schemas, digest pointer, input identities and possibly source bindings differ. An old digest or signature does not authenticate a renamed/recomputed document. We deliberately provide no silent converter or legacy runtime reader.

## Delivery boundary

The 2.0.0 package release and Agent contract v2 carry this explicitly breaking transition. Do not republish existing package versions or silently overwrite published schema releases. Installed 1.x packages do not understand the new provenance contract.

Deploy web assets and remote MCP validators together. The September 29 ChatGPT compatibility repair preserves the original basic `deepbom.chatgpt_analysis_result.v1` transport alongside v2 while published tool metadata is reviewed. This is only a conversation-envelope adapter: it does not accept retired provenance schemas, rename old evidence, or convert old IR digests. v1 cannot carry `provenance_summary`; v2 is required for that explicit optional action. Cached results retain their declared producer version instead of being relabeled as the current engine.

The base widget tries v1 first and retries v2 only on an explicit schema rejection, accommodating both historical v1 and cached v2-only host definitions. A successful matching acknowledgement is required. Unsupported provenance reporting offers the canonical JSON export without discarding its evidence. Browser/Claude result envelopes remain v2. Preserve an old release archive for historical IR verification. See [ChatGPT publication compatibility](../chatgpt-app/PUBLICATION_COMPATIBILITY.md).

No TEA endpoints, remote lookup, upload, attestation validation or registry publication are introduced by this transition.
