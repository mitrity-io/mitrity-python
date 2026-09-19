"""Failures to obtain a decision.

Every exception here means the same thing at the call site: the tool call is
denied. There is no failure the adapter turns into an allow. The subclasses
exist so a caller can *say why* — an outage reads differently from a policy
decision in the model's context — never so it can pick one to ignore.
"""

from __future__ import annotations


class AdmissionError(Exception):
    """Base class: the edge did not produce a decision the adapter can act on."""


class AdmissionConfigError(AdmissionError):
    """The adapter's own configuration is unusable.

    Raised before any I/O — a routable admission address, no token file path.
    """


class AdmissionUnreachable(AdmissionError):
    """The edge could not be reached: no socket, connection refused, token file absent."""


class AdmissionTimeout(AdmissionError):
    """The edge did not answer inside the adapter's deadline."""


class AdmissionUnauthorized(AdmissionError):
    """The edge rejected the token after the one permitted re-read and retry."""


class AdmissionNotReady(AdmissionError):
    """The edge answered ``503``: it has no mission profile to judge against."""


class AdmissionProtocolError(AdmissionError):
    """The exchange did not follow the contract.

    A ``400``, a version header the adapter does not speak, a body that is
    not JSON, a decision value it does not recognize.
    """


class AdmissionPayloadTooLarge(AdmissionProtocolError):
    """The request body exceeds the edge's 64 KiB cap; denied without sending."""
