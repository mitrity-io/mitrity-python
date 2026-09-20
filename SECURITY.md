# Security policy

`mitrity` sits between an AI agent's framework and the MITRITY edge that
governs it. A bug here can turn a deny into a run, so we treat every report
as high priority.

## Reporting a vulnerability

Email **soc@mitrity.com**. Do not open a public issue for anything that could
let a tool call bypass admission, leak the admission token, or make the
adapter report coverage it does not have.

Please include the package version (`python -c "import mitrity; print(mitrity.__version__)"`),
the framework and its version, a minimal reproduction, and what the edge
answered (or did not).

We acknowledge reports within two business days and keep you informed until
the fix ships. Coordinated disclosure is welcome; we ask for 90 days.

## Supported versions

Only the latest minor release receives security fixes. The admission protocol
version an adapter release speaks is recorded in
[the adapter contract](https://mitrity.com/docs/integrations/adapters).

## What is and is not a vulnerability here

- A tool call that runs after the edge answered `deny` or `held`, or after the
  edge could not be reached: **vulnerability**.
- The admission token appearing in a log line, an exception message, an
  environment variable the adapter sets, or a command line: **vulnerability**.
- An attestation that claims a tool is hooked when it is not: **vulnerability**.
- A policy that allows something you did not expect: a policy question for
  your MITRITY console, not an adapter defect.
