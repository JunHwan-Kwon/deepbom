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
| `weights` | Optional Weight IR statistics and selected feature, with decoder/budget coverage |
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

## Compatible query navigation improvements

The existing eight query sections and public input/output schemas remain
unchanged. Within the extensible result context, queries now include the shared
review summary's finding counts and analysis coverage, and the existing Model
Hierarchy identity and coverage when available. These describe the source
analysis before filtering or pagination; filtered/page counts retain their
separate meaning. The graph summary keeps its source scope, identified by
`native_detail_scope`, rather than recomputing a total across nested graphs. No
new metric formulas, format parsers or eligibility rules are introduced.

Selecting a diagram row opens its evidence without changing the query filter.
**Inspect selected layer** resolves the exact subject in the same Model IR.
Its incoming/outgoing navigation is projected from the full Model Summary's
serialized relationships, including neighbors outside the original page.
Relationship kinds and truncation counts remain explicit; navigation does not
infer execution, gradients or causal influence. **Back to results** restores
the original query and page. **Clear filters** recovers from an empty search.
The JSON disclosure retains the original detailed evidence for inspection.
Narrow panels scroll the graph internally instead of shrinking its text.

Finding references retain their envelope index when improvement priorities are
sorted, so a reference from `findings` identifies the same finding in
`improvements`. They are scoped to the artifact, scan policy and returned
`evidence_envelope_sha256`, not stable identity across changed analyses or
releases. Literal search runs before display
truncation. Source names exceeding the display budget can still match, with
omissions marked in the returned row.

Failed page requests preserve the displayed result and navigation history and
do not publish that result as the failed request's answer. Replacing a panel
prevents late results from the previous panel from initiating a new report or
replacing the current view. Export content and filenames use the same captured
result. The unchanged contract fingerprint is checked separately from browser
interaction tests; neither establishes actual host rendering or portal status.

## 2.0.0 submission history

- Keep the published directory version operating while preparing this draft.
- Import the updated root `chatgpt-app-submission.json`: seven tool justifications,
  exactly five positive cases (including multi-turn follow-ups), and three
  negative cases.
- Rescan tools to discover the public `deepbom_query_file` and component-only
  `deepbom_publish_query`, plus `deepbom_workflow_guide`, in addition to the existing four tools.
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

## Conversational workflow coverage

The shared `web/lib/workflow-catalog.js` routes 18 workflows across conversation,
web and local CLI. `deepbom_workflow_guide` returns this catalog without file
access or command execution. The same catalog powers the widget's **Explore
next · Chat, web and CLI** controls and website continuation. It is navigation
metadata, not another IR or duplicate analysis implementation. Format-specific
coverage and access gates still apply. CLI capability discovery remains the
complete installed-machine contract; the catalog does not claim to expose every
local function as a remote executable tool.

`weights` queries explicitly opt into the existing optional Weight IR and
advanced numerical analysis worker. They return identity-bound tensor statistics
and a selected feature: distribution, channels, similarity, spectrum, sparsity or
quantization. Decoder and computation budgets, coverage and omitted detail are
preserved. The 128 MiB widget limit is explicit. The widget reuses computed
weights within its mounted session and the website's visual encoders; plots use
complete in-browser feature data rather than truncated conversation arrays.
Weight SVG/PNG can use the same explicit host file-sharing controls as other
exports. Similarity and sparsity are not proofs of safe pruning or acceleration.

Web URLs have the form `https://deepbom.org/#workflow=weights`. Only a public
workflow ID is carried. No model bytes, filename, hash, prompt or result is
transferred. The landing banner asks the user to select the model and compare
its SHA-256 with the conversation. After the audit, the normal navigation
controller opens the requested view once. Unsupported formats and access gates
are respected. No model is automatically fetched or executed.

Additional recording steps: request ONNX weight distribution and spectrum,
select a tensor and export the selected plot; request a pruning workflow and
open its website link, select the same file and show the Weight workspace; ask
for BOM verification and batch commands and show that these are local templates,
not results. The old recording does not establish these new interactions.

## Compatible conversational guidance update

The workflow guide now returns Korean/English question examples and valid query
examples for the existing workflows. Capabilities explains literal search,
ambiguous references and asynchronous completion. Query results provide
`context.follow_up`: optional next queries tied to the returned artifact hash,
page cursor and exact subjects. They preserve filters for continuation and
reset offsets when switching views. These suggestions do not execute themselves
and do not create a natural-language parser, new tool or new query section.

Typical follow-ups include “다음 페이지”, “이 레이어로 들어오는 연결”, “이 텐서의
채널별 통계”, and “어디부터 개선을 검토할까?”. Weight payload analysis still
requires an explicit request; measured speed, clinical conclusions and actual
runtime placement are outside static evidence. The shared IR remains the only
source of graph and numerical facts. Query tests exercise the actual next-query
arguments; host interpretation of the example questions is evaluated separately.

This deployment updates compatible Web/MCP implementation and guidance data on
the existing engine 2.5.0 and plugin 2.0.0 contracts. It does not replace immutable
2.5.0 npm/release archives or change the submitted Skill, tools or CSP.

## Result-first interaction and selection context

The initial component shows the format/graph summary, existing finding counts,
one first finding and the static-evidence boundary. Two actions open the grouped
structure or inspect improvement investigations. Identity details, full findings,
the visualization/export surface and the long summary table remain available in
explicit disclosures. Opening structure exposes the existing export controls.

Query controls are under Advanced query settings. Contextual follow-up buttons
reuse `queryFollowUps`, the same builder used in the bounded query result: two
are shown initially, with additional actions under More questions. Page actions
retain filters/cursors; selected-subject actions use the actual selected row,
including a tensor other than the first returned tensor. Korean/English question
labels follow the host locale. This is navigation, not another analysis engine.

Selection updates optional model-visible widget state with the artifact hash,
exact subject reference, query/result identity and finding envelope identity.
Selecting a row does not send a message or change filters. Ask about this
selection explicitly requests an explanation of that row with the validated
query result. Clear selection and new results clear the old subject. The shared
state writer preserves selection during visualization/export updates and drops
prior-model state, report digests and image IDs when the artifact changes. Hosts
without state support retain explicit selected-subject follow-up messages.

The full summary table and architecture overview continue to use
`resolveModelHierarchyPresentation`. Source annotations produce declared source
module groups; absent or invalid annotations use derived/unclassified groups.
Grouped views retain member and edge provenance; query diagrams represent only
the returned page. This update adds no semantic hierarchy inference, MAC formula,
runtime observation or new MCP schema. The compact chat summary keeps its
existing operation/storage ledger contract; it is not the grouped table.

Verification includes `check-conversation-ux.mjs`, the widget browser scenarios,
and model hierarchy/visualization conservation checks. Browser tests use a
controlled host bridge and do not establish natural-language routing accuracy
inside an independently running ChatGPT session.

Deployment regression checks also cover a failed replacement query clearing the
previous selection, and failed weight-chart rendering preserving the selected
tensor's source evidence and explicit conversation action. Rendering failure is
not reported as missing or invalid underlying evidence.
