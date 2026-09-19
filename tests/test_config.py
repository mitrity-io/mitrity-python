from __future__ import annotations

import pytest

from mitrity.admission import (
    DEFAULT_HOLD_TIMEOUT,
    DEFAULT_TIMEOUT,
    ENV_ADDR,
    ENV_HOLD_TIMEOUT,
    ENV_TIMEOUT,
    ENV_TOKEN_FILE,
    MAX_HOLD_TIMEOUT,
    MAX_TIMEOUT,
    AdmissionConfigError,
    Config,
    parse_duration,
    parse_loopback_addr,
    split_addr,
    validate_addr,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("500ms", 0.5),
        ("9m", 540.0),
        ("1h30m", 5400.0),
        ("1.5s", 1.5),
        ("540", 540.0),
        ("0", 0.0),
        ("  2s ", 2.0),
        ("-1s", -1.0),
    ],
)
def test_parse_duration_accepts_go_durations_and_bare_seconds(value: str, expected: float) -> None:
    assert parse_duration(value, 99.0) == expected


@pytest.mark.parametrize("value", ["garbage", "", "   ", "1x", "ms", "5m3", None])
def test_parse_duration_falls_back_on_malformed_input(value: str | None) -> None:
    # A misconfigured timeout must not crash a hook and leave a session ungoverned.
    assert parse_duration(value, 7.0) == 7.0


@pytest.mark.parametrize(
    ("addr", "expected"),
    [
        ("unix:/run/mitrity/admission.sock", ("unix", "/run/mitrity/admission.sock")),
        ("unix:///tmp/a.sock", ("unix", "/tmp/a.sock")),
        ("/tmp/a.sock", ("unix", "/tmp/a.sock")),
        ("127.0.0.1:8777", ("tcp", "127.0.0.1:8777")),
    ],
)
def test_split_addr(addr: str, expected: tuple[str, str]) -> None:
    assert split_addr(addr) == expected


@pytest.mark.parametrize(
    "addr",
    [
        "unix:/run/mitrity/admission.sock",
        "/tmp/x.sock",
        "127.0.0.1:8777",
        "[::1]:8777",
        "localhost:8777",
    ],
)
def test_validate_addr_accepts_loopback_and_sockets(addr: str) -> None:
    validate_addr(addr)


@pytest.mark.parametrize(
    "addr",
    [
        "0.0.0.0:8777",
        ":8777",
        "10.0.0.5:8777",
        "example.com:80",
        "[2001:db8::1]:8777",
        "unix:",
        # URL-authority confusion: a loopback-looking prefix in front of another host.
        "localhost:8777@attacker.example",
        "127.0.0.1:8777@attacker.example:80",
        "127.0.0.1:8777/../x",
        "127.0.0.1:8777?x=1",
        "127.0.0.1:8777#f",
        "127.0.0.1",
        "127.0.0.1:abc",
        "127.0.0.1:0",
        "127.0.0.1:70000",
        "127.0.0.1:8777 ",
        " 127.0.0.1:8777",
        "[::1]:8777@attacker.example",
    ],
)
def test_validate_addr_refuses_routable_addresses(addr: str) -> None:
    with pytest.raises(AdmissionConfigError):
        validate_addr(addr)


def test_config_from_env_reads_the_hook_variables() -> None:
    cfg = Config.from_env(
        {
            ENV_ADDR: "unix:/tmp/edge.sock",
            ENV_TOKEN_FILE: "/tmp/edge.token",
            ENV_TIMEOUT: "250ms",
            ENV_HOLD_TIMEOUT: "60",
        }
    )
    assert cfg.addr == "unix:/tmp/edge.sock"
    assert cfg.token_file == "/tmp/edge.token"
    assert cfg.timeout == 0.25
    assert cfg.hold_timeout == 60.0
    assert cfg.network == "unix"
    assert cfg.address == "/tmp/edge.sock"


def test_config_from_env_uses_platform_defaults() -> None:
    cfg = Config.from_env({})
    assert cfg.addr
    assert cfg.token_file
    assert cfg.timeout == DEFAULT_TIMEOUT
    assert cfg.hold_timeout == DEFAULT_HOLD_TIMEOUT


def test_config_clamps_timeouts_to_the_hook_ceilings() -> None:
    cfg = Config(addr="unix:/x", token_file="/t", timeout=10_000, hold_timeout=10_000)
    assert cfg.timeout == MAX_TIMEOUT
    assert cfg.hold_timeout == MAX_HOLD_TIMEOUT
    zero = Config(addr="unix:/x", token_file="/t", timeout=0, hold_timeout=-5)
    assert zero.timeout == DEFAULT_TIMEOUT
    assert zero.hold_timeout == 0.0


def test_localhost_is_a_spelling_of_the_ipv4_loopback_literal() -> None:
    # Never resolved: a hosts-file entry cannot point it off-box.
    assert parse_loopback_addr("localhost:8777") == ("127.0.0.1", 8777)
    assert parse_loopback_addr("127.0.0.2:1") == ("127.0.0.2", 1)
    assert parse_loopback_addr("[::1]:8777") == ("::1", 8777)
