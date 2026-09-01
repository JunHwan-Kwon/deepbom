# Security Policy

## Supported releases

Security fixes target the latest published DEEPBOM release. Older releases may
be documented as affected without receiving a backport.

## Reporting a vulnerability

Use GitHub private vulnerability reporting for
`JunHwan-Kwon/deepbom` when available. Otherwise contact `kjh0442@yuhs.ac`
with the subject `DEEPBOM security report`.

Include the DEEPBOM version, platform, command or browser path, artifact format,
failure class, and a minimal reproduction. Do not send proprietary model bytes,
patient data, secrets, or production runtime traces by email. Begin with hashes,
sizes, relevant JSON pointers, and a synthetic reproducer; a protected transfer
method can be agreed separately if the original artifact is essential.

Relevant classes include parser crashes or resource exhaustion, path traversal,
unsafe package extraction, artifact identity bypass, signature or digest
verification bypass, unintended network transfer of model data, and disclosure
across the documented public/private source boundary.

## Security boundary

DEEPBOM statically parses untrusted deployment artifacts and therefore treats
malformed input as hostile. A clean rejection, timeout, or configured resource
limit is not itself a vulnerability. Executable pickle formats are not
deserialized. Static compatibility predictions do not establish runtime safety,
model quality, clinical validity, or release readiness.
