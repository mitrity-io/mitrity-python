"""RFC 8785 (JSON Canonicalization Scheme) for the values an attestation hashes.

Two independent implementations hash the same configuration — this adapter
and whatever later checks a runtime against it — so the byte sequence has to
be pinned rather than left to a JSON library's defaults. The subset here is
what a governed configuration contains: strings, booleans, integers, ``None``,
lists and objects. Floats are refused on purpose: nothing in the hashed
document is a float, and JCS float formatting is the one part worth not
getting subtly wrong.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

_SHORT_ESCAPES = {
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
    '"': '\\"',
    "\\": "\\\\",
}


def _escape(text: str) -> str:
    out: list[str] = ['"']
    for char in text:
        short = _SHORT_ESCAPES.get(char)
        if short is not None:
            out.append(short)
        elif ord(char) < 0x20:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def canonical_json(value: Any) -> str:  # noqa: PLR0911 — one return per JSON type
    """Serialize ``value`` per RFC 8785: sorted members, no whitespace, minimal escapes."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _escape(value)
    if isinstance(value, bytes | bytearray | memoryview):
        raise TypeError("canonical_json does not serialize bytes")
    if isinstance(value, Mapping):
        items = sorted(
            ((str(key), inner) for key, inner in value.items()),
            key=lambda item: item[0].encode("utf-16-be"),
        )
        return (
            "{" + ",".join(f"{_escape(key)}:{canonical_json(inner)}" for key, inner in items) + "}"
        )
    if isinstance(value, Sequence):
        return "[" + ",".join(canonical_json(inner) for inner in value) + "]"
    raise TypeError(f"canonical_json does not serialize {type(value).__name__}")


def config_hash(value: Any) -> str:
    """SHA-256, hex, over the canonical serialization of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
