# DEEPBOM Claude remote-connector qualification

The candidate endpoint is `https://deepbom.org/mcp/claude`. It is separate
from the ChatGPT attachment endpoint and does not claim automatic access to
Claude attachments. The user selects one file inside an MCP App. Analysis runs
in the host browser sandbox; only a bounded evidence result is returned through
the MCP server.

This directory contains submission preparation and blank host-test material.
It is not proof of Anthropic review, approval, directory listing, or operation
on a named Claude host until a real run has been recorded.

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
