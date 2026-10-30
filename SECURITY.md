# Security Policy

## Supported Versions

Before the first stable release, security fixes apply to the latest commit on
the default branch. A version-support table will be published with the first
stable release.

## Reporting a Vulnerability

Do not open a public issue for vulnerabilities involving:

- exposure or deletion of private audio;
- arbitrary file access;
- unsafe model or prompt deserialization;
- authentication or deployment-boundary bypasses; or
- denial of service with realistic untrusted inputs.

Use the repository host's private security advisory feature. If that feature
is unavailable, open a non-sensitive issue asking maintainers to establish a
private channel. Do not include vulnerability details in that public issue.

Include affected versions, reproduction steps, impact, and any suggested
mitigation. Do not attach real user voice data or credentials.

Maintainers will acknowledge a complete report within five business days,
coordinate remediation and disclosure, and credit reporters who want public
recognition.

## Deployment Guidance

The default server binds only to `127.0.0.1`. Treat profile samples,
transcripts, saved generations, and model-management operations as private.

Before exposing the API beyond a trusted host:

- place it behind authentication and an access-controlled reverse proxy;
- enable and verify TLS;
- restrict allowed origins;
- limit request body size at the proxy and application;
- isolate the data and model-cache directories;
- run as a dedicated unprivileged account; and
- monitor disk, memory, and GPU utilization.

The service does not provide multi-user authorization in v1.

## Dependency Resolution

The repository delays newly published registry artifacts for seven days using
uv's dependency cooldown and npm's minimum release age. npm installs require
version 11.10 or newer so an older client cannot silently ignore the policy.

Lockfile hashes and the cooldown complement one another: hashes make an
existing resolution reproducible, while the cooldown reduces exposure during
future dependency updates. Neither replaces dependency review or vulnerability
response.
