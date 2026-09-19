"""Conformance C4, C5, C9: the two-phase decision never fabricates an allow."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mitrity.admission import (
    AdmissionNotReady,
    AdmissionTimeout,
    AdmissionUnreachable,
    AdmitRequest,
    Client,
)
from tests.fake_edge import FakeEdge, HoldScript, Scripted, allow, deny, held

REQUEST = AdmitRequest(surface="custom", tool_name="Bash", tool_input={"command": "rm -rf build"})


def test_c4_unreachable_edge_is_a_deny_within_the_deadline(edge: FakeEdge, tmp_path: Path) -> None:
    client = Client(
        addr=f"unix:{tmp_path}/missing.sock", token_file=str(edge.token_file), timeout=0.5
    )
    started = time.monotonic()
    verdict = client.decide(REQUEST)
    elapsed = time.monotonic() - started
    assert not verdict.allowed
    assert isinstance(verdict.error, AdmissionUnreachable)
    assert verdict.unreachable and not verdict.policy_denied
    assert "not a policy decision" in verdict.reason
    assert elapsed < 0.6


def test_c5_late_answer_is_discarded(edge: FakeEdge) -> None:
    client = Client(addr=edge.addr, token_file=str(edge.token_file), timeout=0.3)
    edge.script(Scripted(body=allow().body, delay=1.0))
    started = time.monotonic()
    verdict = client.decide(REQUEST)
    elapsed = time.monotonic() - started
    assert not verdict.allowed
    assert isinstance(verdict.error, AdmissionTimeout)
    assert elapsed < 0.3 + 0.2
    assert len(edge.admits()) == 1


def test_policy_deny_carries_the_reason(edge: FakeEdge, client: Client) -> None:
    edge.script(deny())
    verdict = client.decide(REQUEST)
    assert verdict.policy_denied
    assert verdict.error is None
    assert verdict.reason == (
        'MITRITY denied this action: policy rule "no destructive commands" '
        "denied resolved command: rm"
    )
    assert edge.admits()[0].body["hold_timeout_seconds"] == 0


def test_allow_carries_updated_input_and_routing(edge: FakeEdge, client: Client) -> None:
    edge.script(
        allow(updated_input={"command": "mitrity-hook exec met_1"}, routed_to="governed_shell")
    )
    verdict = client.decide(REQUEST)
    assert verdict.allowed
    assert verdict.updated_input == {"command": "mitrity-hook exec met_1"}
    assert verdict.routed_to == "governed_shell"


def test_c9_hold_resubmits_with_the_budget_and_runs_on_allow(
    edge: FakeEdge, client: Client
) -> None:
    edge.default_admit = HoldScript(outcome=allow())
    verdict = client.decide(REQUEST)
    assert verdict.allowed
    budgets = [r.body["hold_timeout_seconds"] for r in edge.admits()]
    assert budgets == [0, 2]


def test_c9_hold_denied_after_wait_names_the_approval(edge: FakeEdge, client: Client) -> None:
    edge.default_admit = HoldScript(outcome=deny("approval apr-hold timed out"))
    verdict = client.decide(REQUEST)
    assert verdict.policy_denied
    assert "apr-hold" in verdict.reason


def test_c9_still_held_after_budget_is_a_deny(edge: FakeEdge, client: Client) -> None:
    edge.default_admit = HoldScript(outcome=held("apr-hold"))
    verdict = client.decide(REQUEST)
    assert not verdict.allowed
    assert verdict.held
    assert "apr-hold" in verdict.reason
    assert "has not been approved" in verdict.reason


def test_c9_zero_hold_budget_sends_no_second_request(edge: FakeEdge) -> None:
    client = Client(addr=edge.addr, token_file=str(edge.token_file), timeout=0.5, hold_timeout=0)
    edge.default_admit = HoldScript(outcome=allow())
    verdict = client.decide(REQUEST)
    assert verdict.held and not verdict.allowed
    assert len(edge.admits()) == 1


def test_c9_hold_budget_is_clamped() -> None:
    assert Client(addr="unix:/x", token_file="/t", hold_timeout=99_999).config.hold_timeout == 570.0


def test_c9_failure_while_waiting_blocks(edge: FakeEdge, client: Client) -> None:
    edge.default_admit = HoldScript(outcome=Scripted(status=503, body={"reason": "no_profile"}))
    verdict = client.decide(REQUEST)
    assert not verdict.allowed
    assert verdict.held
    assert isinstance(verdict.error, AdmissionNotReady)
    assert "apr-hold" in verdict.reason


@pytest.mark.anyio
async def test_c9_async_two_phase(edge: FakeEdge, client: Client) -> None:
    edge.default_admit = HoldScript(outcome=allow())
    verdict = await client.decide_async(REQUEST)
    assert verdict.allowed
    assert [r.body["hold_timeout_seconds"] for r in edge.admits()] == [0, 2]


@pytest.mark.anyio
async def test_c4_async_unreachable(edge: FakeEdge, tmp_path: Path) -> None:
    client = Client(
        addr=f"unix:{tmp_path}/missing.sock", token_file=str(edge.token_file), timeout=0.5
    )
    verdict = await client.decide_async(REQUEST)
    assert not verdict.allowed
    assert isinstance(verdict.error, AdmissionUnreachable)
