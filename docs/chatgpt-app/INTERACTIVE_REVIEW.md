# Conversational Evidence IR queries — directory update 2.0.0

The directory update adds follow-up evidence inspection. It does not change the
canonical Evidence IR schemas, execute an optimization, or assert a measured
improvement. The directory version and engine version are separate identities.

## Data and control flow

1. `deepbom_analyze_file` retains its existing attachment/depth contract and
   v1/v2 basic result compatibility.
2. After that result identifies the artifact, `deepbom_query_file` takes the
   same host-authorized file, its expected SHA-256, and a bounded query.
3. The browser verifies the complete file hash before querying. It reuses the
   existing parsers, Artifact IR → Model IR → Model Summary, finding envelope,
   execution-placement projections and static fusion annotations.
4. `deepbom_publish_query` validates the bounded projection, pagination and
   consistency digests and returns it to the conversation. This does not
   authenticate the browser's claims as independently reproduced server facts.
5. The component requests one follow-up reply for the explicit query. If the
   host does not render a reply, **Report query in chat** remains available.

There is no server model cache, uploaded weight database or persistent analysis
session. Each tool-driven follow-up re-reads the authorized attachment in the
browser; widget controls reuse that mounted analysis. An expired attachment URL
needs renewed authorization. `browser_query_started` is never a finished result.
Host-rendered components remain necessary for this browser path; local MCP is
the alternative for clients without a component host or for repeated automation.

## Query contract

`web/lib/evidence-query-contract.js` defines the MCP transport schema and
validation. `web/lib/evidence-query.js` projects existing evidence; it does not
implement another parser, MAC formula, quantization rulepack or delegate matcher.

| Section | Result |
| --- | --- |
| `operators` | Search/paginate all common operation or serialized-storage rows |
| `operator` | One exact `subject_ref`, or an unambiguous native `source_index` |
| `findings` | Paged findings with the existing classes, severity and recommendations |
| `improvements` | Existing findings by severity, followed by exact workload/storage ranks |
| `profiles` | Available static backend profiles; CPU planning profiles for TFLite |
| `placement` | Independent per-profile static states, reasons, source identities and coverage |
| `fusion` | TFLite encoded fused activations and existing static review hints |

Pagination applies after filtering; `next_offset` advances over matching rows.
The result caps rows at 40 and total encoded JSON at 64 KiB. Nested detail
truncation is reported explicitly. Native indices are scoped and distinct from
display order. Placement rows bind to the primary-scope Model IR subject IDs;
nested scopes are not silently assigned the primary graph's backend state.
Unknown values remain null and exact decimal metric contracts are preserved.

`query.target` selects an existing TFLite CPU cost profile and is labeled
`explicit_id`; it is not detected hardware or a GPU/delegate selector. Backend
IDs are discovered from the actual analysis, never assumed from a chip name.

The current public TFLite path exposes the XNNPACK CPU baseline and the bundled
Core ML and Qualcomm QNN source prechecks. Additional TFLite GPU/NNAPI and ORT
rule ledgers belong to the capability-authorized module and are not loaded by
the unauthenticated ChatGPT widget. They are not copied into a second rule
implementation or exposed by bypassing access controls. Missing profiles fail
with the available IDs; no substitute profile is silently selected.

Serialized activation fusion, adjacent operators, static eligibility, selected
build acceptance and executed fusion/placement are separate evidence levels.
Investigation rankings do not establish speedup, quality loss or pruning safety.

## Visualization and file sharing

The scrollable query diagram displays the returned page. Clicking a row shows
its structured details and reference. Solid connectors come from serialized
relationships whose endpoints appear on that page; no absent edge is inferred.
Placement cards distinguish static eligibility from observed execution. Full
model views remain below the query panel. Query SVG/PNG uses the existing
explicit export/file-sharing controls. A preview is not a chat attachment.

## Submission update

- Keep the published directory version operating while preparing this draft.
- Import the updated root `chatgpt-app-submission.json`: six tool justifications,
  exactly five positive cases (including multi-turn follow-ups), and three
  negative cases.
- Rescan tools to discover the public `deepbom_query_file` and component-only
  `deepbom_publish_query`, in addition to the existing four tools.
- Replace the old `deepbom-artifact-evidence` ZIP with the newly built ZIP; its
  earlier Passed status covered the previous content. Await the new scan.
- Record the follow-up flow in Developer Mode before declaring it demonstrated
  by the recording. The previous video establishes only the original flow.
- Submit the 2.0.0 draft after checking the imported fields. Server deployment,
  local tests and this document do not establish OpenAI approval or completion
  of real-host tests.

The release baseline records this explicitly requested additive MCP/Skill review,
not an approval. Its OpenAI package fingerprint now includes the ChatGPT Skill,
so changing that instruction file cannot evade the submission-change check.

Suggested additional recording sequence: attach the test TFLite file, request
its basic report, ask for available profiles, compare `xnnpack_cpu` and
`tflite_coreml_delegate`, inspect a returned subject reference, inspect serialized
activation fusion, advance one evidence page and save its SVG/PNG through the
existing host handoff. Keep static predicates and runtime boundaries visible.

## References

- [OpenAI component flow and bridge](https://developers.openai.com/plugins/build/chatgpt-ui)
- [OpenAI tool results and file parameters](https://developers.openai.com/plugins/reference)
- [Tool review and directory updates](https://developers.openai.com/plugins/deploy/app-review)
- [Publication compatibility](PUBLICATION_COMPATIBILITY.md)
