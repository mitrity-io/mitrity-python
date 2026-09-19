"""Conformance C1–C3, C6, C14, C15, C17, C19: the wire client."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from mitrity.admission import (
    HEADER_TOKEN,
    HEADER_VERSION,
    PROTOCOL_VERSION,
    AdmissionConfigError,
    AdmissionNotReady,
    AdmissionPayloadTooLarge,
    AdmissionProtocolError,
    AdmissionUnauthorized,
    AdmissionUnreachable,
    AdmitRequest,
    Attestation,
    Client,
    Decision,
)
from tests.fake_edge import FakeEdge, Scripted, allow, deny

REQUEST = AdmitRequest(surface="custom", tool_name="Bash", tool_input={"command": "ls -la"})


def test_c1_every_request_carries_token_version_and_content_type(
    edge: FakeEdge, client: Client
) -> None:
    decision = client.admit(REQUEST)
    assert decision.allowed
    request = edge.admits()[0]
    assert request.headers[HEADER_TOKEN] == edge.token
    assert request.headers[HEADER_VERSION] == PROTOCOL_VERSION
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Host"] == "mitrity-admission"
    assert request.body == {
        "surface": "custom",
        "tool_name": "Bash",
        "tool_input": {"command": "ls -la"},
    }


def test_c19_protocol_version_constant_is_what_is_sent(edge: FakeEdge, client: Client) -> None:
    client.attest(
        Attestation(framework="custom", adapter="mitrity-python", adapter_version="0.0.0")
    )
    assert edge.attests()[0].headers[HEADER_VERSION] == PROTOCOL_VERSION == "1"


def test_c2_token_is_reread_and_retried_exactly_once_on_401(edge: FakeEdge, client: Client) -> None:
    stale = edge.token
    edge.rotate_on_next_request = True
    assert client.admit(REQUEST).allowed
    presented = [r.headers[HEADER_TOKEN] for r in edge.admits()]
    assert presented == [stale, edge.token]


def test_c2_second_401_is_a_deny(edge: FakeEdge, client: Client) -> None:
    edge.reject_all_tokens = True
    with pytest.raises(AdmissionUnauthorized):
        client.admit(REQUEST)
    assert len(edge.admits()) == 2


def test_c3_routable_address_is_refused_before_any_io(edge: FakeEdge) -> None:
    client = Client(addr="10.0.0.5:8777", token_file=str(edge.token_file))
    with pytest.raises(AdmissionConfigError):
        client.admit(REQUEST)
    verdict = client.decide(REQUEST)
    assert not verdict.allowed
    assert isinstance(verdict.error, AdmissionConfigError)
    assert edge.requests == []


def test_missing_token_file_is_unreachable(edge: FakeEdge, tmp_path: Path) -> None:
    client = Client(addr=edge.addr, token_file=str(tmp_path / "absent.token"))
    with pytest.raises(AdmissionUnreachable):
        client.admit(REQUEST)
    assert edge.requests == []


def test_empty_token_file_is_unreachable(edge: FakeEdge, tmp_path: Path) -> None:
    empty = tmp_path / "empty.token"
    empty.write_text("\n")
    client = Client(addr=edge.addr, token_file=str(empty))
    with pytest.raises(AdmissionUnreachable):
        client.admit(REQUEST)


@pytest.mark.parametrize(
    ("scripted", "error"),
    [
        (Scripted(status=400, body={"error": "unknown surface"}), AdmissionProtocolError),
        (
            Scripted(status=503, body={"decision": "deny", "reason": "no_profile"}),
            AdmissionNotReady,
        ),
        (Scripted(status=200, body=b""), AdmissionProtocolError),
        (Scripted(status=200, body=b"not json"), AdmissionProtocolError),
        (Scripted(status=200, body={"decision": "maybe", "reason": "x"}), AdmissionProtocolError),
        (Scripted(status=200, body=allow().body, version="2"), AdmissionProtocolError),
        (Scripted(status=200, body=allow().body, version=None), AdmissionProtocolError),
        (Scripted(status=200, body=[1, 2]), AdmissionProtocolError),
        (Scripted(status=500, body=b"boom"), AdmissionProtocolError),
    ],
)
def test_c6_every_non_decision_is_the_matching_error(
    edge: FakeEdge, client: Client, scripted: Scripted, error: type[Exception]
) -> None:
    edge.script(scripted)
    with pytest.raises(error):
        client.admit(REQUEST)


def test_c6_not_ready_carries_the_edge_reason(edge: FakeEdge, client: Client) -> None:
    edge.script(Scripted(status=503, body={"decision": "deny", "reason": "no_profile: not primed"}))
    with pytest.raises(AdmissionNotReady, match="no_profile: not primed"):
        client.admit(REQUEST)


def test_c14_oversized_input_is_denied_without_sending(edge: FakeEdge, client: Client) -> None:
    huge = AdmitRequest(
        surface="custom", tool_name="Write", tool_input={"content": "x" * (70 * 1024)}
    )
    with pytest.raises(AdmissionPayloadTooLarge):
        client.admit(huge)
    verdict = client.decide(huge)
    assert not verdict.allowed
    assert "larger than MITRITY will judge" in verdict.reason
    assert edge.requests == []


def test_c15_fail_mode_open_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, edge: FakeEdge
) -> None:
    monkeypatch.setenv("MITRITY_HOOK_FAIL_MODE", "open")
    client = Client(addr=f"unix:{tmp_path}/nobody-listens.sock", token_file=str(edge.token_file))
    verdict = client.decide(REQUEST)
    assert not verdict.allowed
    assert isinstance(verdict.error, AdmissionUnreachable)


def test_c17_token_and_tool_input_never_reach_the_log(
    edge: FakeEdge, client: Client, caplog: pytest.LogCaptureFixture
) -> None:
    marker = "SECRET-INPUT-7f3a"
    with caplog.at_level(logging.DEBUG, logger="mitrity"):
        client.admit(
            AdmitRequest(surface="custom", tool_name="Bash", tool_input={"command": marker})
        )
        edge.script(deny(f"rule denied {marker}"))
        client.decide(
            AdmitRequest(surface="custom", tool_name="Bash", tool_input={"command": marker})
        )
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert caplog.records, "the client logs its decisions"
    assert marker not in text
    assert edge.token not in text


def test_decision_parsing_keeps_every_field(edge: FakeEdge, client: Client) -> None:
    edge.script(
        allow(
            updated_input={"command": "mitrity-hook exec met_1"},
            routed_to="governed_shell",
            approval_id=None,
            risk_score=0.25,
            admission_id="adm-7",
            reason="allowed; routed to the governed shell",
        )
    )
    decision = client.admit(REQUEST)
    assert decision == Decision(
        decision="allow",
        reason="allowed; routed to the governed shell",
        admission_id="adm-7",
        risk_score=0.25,
        approval_id=None,
        updated_input={"command": "mitrity-hook exec met_1"},
        routed_to="governed_shell",
    )


def test_request_validation_happens_before_io(edge: FakeEdge, client: Client) -> None:
    with pytest.raises(AdmissionProtocolError):
        client.admit(AdmitRequest(surface="custom", tool_name="   ", tool_input={}))
    with pytest.raises(AdmissionProtocolError):
        client.admit(
            AdmitRequest(surface="custom", tool_name="Bash", tool_input={}, hold_timeout_seconds=-1)
        )
    with pytest.raises(AdmissionProtocolError):
        client.admit(AdmitRequest(surface="nope", tool_name="Bash", tool_input={}))  # type: ignore[arg-type]
    assert edge.requests == []


def test_attest_returns_none_on_204(edge: FakeEdge, client: Client) -> None:
    attestation = Attestation(
        framework="custom",
        adapter="mitrity-python",
        adapter_version="0.0.0",
        hooked_tools=("Bash",),
        config_hash="ab" * 32,
    )
    client.attest(attestation)
    assert edge.attests()[0].body == {
        "framework": "custom",
        "adapter": "mitrity-python",
        "adapter_version": "0.0.0",
        "hooked_tools": ["Bash"],
        "config_hash": "ab" * 32,
    }


def test_health_is_unauthenticated(edge: FakeEdge, client: Client) -> None:
    assert client.health()["status"] == "ok"
    assert HEADER_TOKEN not in edge.requests[0].headers


def test_tcp_loopback_address_works(edge: FakeEdge) -> None:
    # No TCP listener in the fake; a connection refused on loopback is the
    # unreachable path, which is the point: the address itself is accepted.
    client = Client(addr="127.0.0.1:1", token_file=str(edge.token_file), timeout=0.5)
    with pytest.raises(AdmissionUnreachable):
        client.admit(REQUEST)


@pytest.mark.anyio
async def test_async_client_speaks_the_same_wire(edge: FakeEdge, client: Client) -> None:
    decision = await client.admit_async(REQUEST)
    assert decision.allowed
    await client.attest_async(
        Attestation(framework="custom", adapter="mitrity-python", adapter_version="0")
    )
    assert (await client.health_async())["status"] == "ok"
    edge.script(Scripted(status=503, body={"reason": "no_profile"}))
    with pytest.raises(AdmissionNotReady):
        await client.admit_async(REQUEST)
    edge.reject_all_tokens = True
    with pytest.raises(AdmissionUnauthorized):
        await client.admit_async(REQUEST)


def test_attest_accepts_a_204_without_the_version_header(edge: FakeEdge, client: Client) -> None:
    # The edge answers /v1/attest with a bare 204; only a decision needs the header.
    edge.default_attest = Scripted(status=204, body=None, version=None, content_type=None)
    client.attest(Attestation(framework="custom", adapter="mitrity-python", adapter_version="0"))
    assert len(edge.attests()) == 1


def test_tcp_url_is_built_from_the_parsed_literal(edge: FakeEdge) -> None:
    client = Client(addr="localhost:1", token_file=str(edge.token_file), timeout=0.2)
    assert client._url("/v1/admit") == "http://127.0.0.1:1/v1/admit"
    v6 = Client(addr="[::1]:1", token_file=str(edge.token_file), timeout=0.2)
    assert v6._url("/healthz") == "http://[::1]:1/healthz"
