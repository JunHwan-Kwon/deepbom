# Host-observed agent evaluation

This package separates deterministic DEEPBOM execution checks from a real
assistant host's tool-selection behavior. `scripts/check-agent-evaluation.mjs`
continues to verify fixed CLI invocations. It is not evidence that ChatGPT,
Codex, or Claude selected DEEPBOM from a natural-language request.

Use `host-evaluation-cases.v1.json` in a new conversation on each target
surface. Start without naming DEEPBOM for every `indirect` case. Record the
selected tool, supplied arguments, execution result, evidence boundary, and any
unnecessary selection. Copy `host-evaluation-run.template.json` to an untracked
location under `.local-validation/agent-host-evaluation/` before recording a
run. Do not replace the template with invented or retrospective observations.

Count these separately:

- discovery: the product or installed plugin is offered as a candidate;
- selection: a relevant installed tool is chosen;
- execution: the tool completes against the fixture;
- evidence preservation: artifact identity, analyzer identity, and limits remain;
- false selection: a negative or unsupported request invokes an analyzer.

A request for a missing file is not a selection failure when the assistant asks
for the file. A policy block caused by a real finding is not an invocation
failure. Public-directory discovery must not be recorded until the listing is
actually approved and visible.

The run record contains potentially sensitive host, prompt, and fixture data.
Review it before publication; this repository publishes the cases and schema,
not fabricated host results.

The catalog also includes Korean and English follow-up scenarios for pagination,
ambiguous references, layer connections, weight views, backend profiles and web
continuation. Establish the stated conversation context before each follow-up.
The deterministic `check-evidence-query.mjs` executes real source-bound next
queries and the structured query examples against the shared IR. It does not
interpret the natural-language prompts. Report these checks separately from a
host-observed tool-selection run; neither the prompt count nor a passing query
check is a language-model accuracy score.
