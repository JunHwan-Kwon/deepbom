# DEEPBOM Claude remote-connector qualification

The candidate endpoint is `https://deepbom.org/mcp/claude`. It is separate
from the ChatGPT attachment endpoint and does not claim automatic access to
Claude attachments. The user selects one file inside an MCP App. Analysis runs
in the host browser sandbox; only a bounded evidence result is returned through
the MCP server.

This directory contains submission preparation and blank host-test material.
It is not proof of Anthropic review, approval, directory listing, or operation
on a named Claude host until a real run has been recorded.

## Connect with a personal Claude account

1. Open Claude, then **Customize > Connectors > Add custom connector**.
2. Name it **DEEPBOM Artifact Evidence** and enter
   `https://deepbom.org/mcp/claude`. No DEEPBOM account, API key, or OAuth
   credentials are required.
3. Enable this connector in a new conversation and send:
   **Use DEEPBOM to open the local model analyzer. I will choose an ONNX file
   in the app. Do not use another parser or execute model code.**
4. Allow the app to display if Claude asks. Select the file **inside the
   DEEPBOM app**, then click **Analyze**. A conversation attachment alone does
   not select a file for this connector.
5. Wait for **Static evidence ready**, then send:
   **Summarize the DEEPBOM result. Include the full SHA-256, serialized model
   structure, quantization evidence, defects, cautions, and evidence gaps.
   State what static analysis cannot establish.**

The current Claude widget provides analysis and a text model-summary table.
It does not yet provide the ChatGPT widget's diagram/export controls or
consented usage statistics. Use the website or local CLI for diagram,
CycloneDX, and SPDX exports; do not advertise those as Claude widget buttons.

Run `npm run check:claude-widget` for the simulated cross-origin MCP App test.
It exercises all four tools, ONNX and TFLite, identity checks, context delivery,
missing selection, and unsafe-format rejection. This is a browser regression
test, **not** evidence that a real Claude account or every host passed.

## Public directory submission

Personal custom-connector testing and public directory submission are separate.
As checked on 2026-09-17, remote directory submissions require a Team or
Enterprise organization and an owner or delegated submission role. A personal
Free/Pro/Max account can still test a custom connector. Desktop MCPB submissions
have a separate path; see `../claude-app/README.md`.

`PORTAL_VALUES.md` contains draft listing copy and a real-host test checklist.
It is not a portal-import file. Do not mark host tests complete or submit
screenshots until those observations have been captured in Claude itself.

Sources: [custom-connector testing](https://claude.com/docs/connectors/building/testing)
and [directory submission](https://claude.com/docs/connectors/building/submission).

## Required host validation

1. Add the endpoint as a private custom connector in Claude.ai.
2. Confirm the four runtime tools have titles and accurate read-only,
   non-destructive annotations.
3. Open the analyzer, choose a public non-confidential fixture, and compare the
   artifact SHA-256 and bounded summary with DEEPBOM CLI of the same version.
4. Repeat on Claude Desktop and mobile where MCP Apps are available. Record
   unsupported host capabilities as failures or limitations; do not infer
   cross-surface support from one passing host.
5. Run the positive and negative cases in
   `../agent-evaluation/host-evaluation-cases.v1.json`.

## Privacy boundary

The remote MCP App does not send selected model bytes to the DEEPBOM service.
It does return a bounded analysis result to the Claude conversation. That result
can contain model names, tensor names, hashes, architecture facts, findings, and
evidence gaps. It also contains a row-bounded format-neutral Model IR summary;
the app shows the complete local table and keeps serialized storage distinct
from framework trainability and runtime order. Use the local MCPB or CLI for confidential files when those
derived facts must not leave the computer.

See https://deepbom.org/privacy for the public policy. No DEEPBOM account or
test account is required for this no-auth endpoint.
