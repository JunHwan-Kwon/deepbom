# DEEPBOM Claude Desktop review package

The supported Claude path in this package is the version-matched local MCPB
desktop extension. It runs the packaged analyzer on the user's machine and
restricts reads to one directory selected during installation. The current
ChatGPT Streamable HTTP endpoint uses ChatGPT browser-component semantics and
is not claimed here as a tested Claude remote connector.

## Local installation review

1. Download `deepbom-1.98.0.mcpb` from the `channels-v1.98.0` GitHub Release.
2. In Claude Desktop, open Settings, Extensions, Advanced settings, and Install
   Extension. Select a dedicated model directory as `allowed_root`.
3. Start a new chat. Confirm that `deepbom_capabilities`, `deepbom_audit`,
   `deepbom_diff`, and `deepbom_explain_rule` are available.
4. Run the Claude cases in
   `../agent-evaluation/host-evaluation-cases.v1.json`, including an allowed
   file, an outside-root path, and a request that should not select DEEPBOM.
5. Compare one successful MCP result with the same release's CLI result for the
   artifact SHA-256, analyzer version, format, and bounded summary.

## Directory submission boundary

`submission-profile.json` is preparation material. It does not represent an
Anthropic review, approval, directory listing, trademark clearance, or user
adoption. Submission requires the publisher to review the current Anthropic
requirements, attest the data flow, provide the requested policy URLs and
assets, and complete the account-owned submission form manually.

Do not publish a host-evaluation result until it contains real observations
from the named Claude version and no confidential artifact identifiers.
