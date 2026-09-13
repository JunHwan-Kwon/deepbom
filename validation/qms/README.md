# Engineering evidence review package

This package provides a small, reviewable record around a validated DeepBOM
artifact-evidence envelope. It records the exact artifact and evidence digests,
the human reviewer, time, disposition, and immutable limitations.

It supports engineering traceability and technical-document preparation. It is
not a regulatory determination and does not establish safety, effectiveness,
clinical validity, legal compliance, or conformity by itself.

```bash
node validation/qms/create-record.mjs \
  --evidence evidence.json \
  --output review-record.json \
  --reviewer "reviewer-id" \
  --reviewed-at 2026-09-13T00:00:00Z \
  --disposition follow_up_required
```
