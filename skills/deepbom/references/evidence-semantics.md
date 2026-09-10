# Evidence semantics

Preserve DEEPBOM's evidence class and applicability state in the answer.

- `OBSERVED` describes serialized artifact facts.
- `SOURCE_BACKED` identifies a source-pinned rule or compatibility statement.
- `DERIVED` is deterministically calculated from declared inputs.
- `PREDICTED` and `ESTIMATED` are model-dependent, not runtime observations.
- `MEASURED` is valid only when an imported measurement is identity-bound.
- `NOT_APPLICABLE` means the domain does not apply to this artifact.
- `NOT_ASSESSABLE` means required evidence is absent or insufficient.
- `NOT_ASSESSED_YET` means analysis has not been requested or completed.

Finding kinds are independent from evidence classes:

- `artifact_defect`: a contradiction or defect found in the artifact.
- `caution`: a bounded condition that merits review but is not a defect.
- `evidence_gap`: a claim the artifact cannot settle alone.

When a target is present, report whether its binding source is a default assumption, explicit ID, or profile file. `host_observed: false` means the host was not measured.
