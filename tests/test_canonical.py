from __future__ import annotations

import re

import pytest

from mitrity.admission import canonical_json, config_hash


def test_canonical_json_sorts_members_and_drops_whitespace() -> None:
    value = {"b": 1, "a": [True, None, "x\n", {"z": False, "y": 0}]}
    assert canonical_json(value) == '{"a":[true,null,"x\\n",{"y":0,"z":false}],"b":1}'


def test_canonical_json_keeps_unicode_literal_and_escapes_control_characters() -> None:
    assert canonical_json({"k": 'é"\\'}) == '{"k":"é\\u001f\\"\\\\"}'


def test_canonical_json_orders_keys_by_utf16_code_units() -> None:
    # U+1D11E (surrogate pair in UTF-16) sorts before U+FF5E under JCS,
    # though it sorts after by code point.
    assert canonical_json({"～": 1, "\U0001d11e": 2}) == '{"\U0001d11e":2,"～":1}'


def test_canonical_json_refuses_floats() -> None:
    with pytest.raises(TypeError):
        canonical_json({"risk": 0.5})


def test_config_hash_is_sha256_hex() -> None:
    digest = config_hash({"hooked_tools": ["Bash"], "adapter": "mitrity-python"})
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert digest == config_hash({"adapter": "mitrity-python", "hooked_tools": ["Bash"]})
    assert digest != config_hash({"adapter": "mitrity-python", "hooked_tools": ["Write"]})
