from __future__ import annotations

import asyncio

import pytest

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.test_utils.receiver_fault import ReceiverFaultOutcome, ReceiverIdentity
from miles.utils.workers.cell_operations import base as cell_operations_base
from miles.utils.workers.cell_operations.base import BaseCellOperations, FaultInjectionOutcome, ReceiverFaultRequest

_RECEIVER = ReceiverIdentity(
    receiver_boot_uuid="boot-1", session_id="session-1", rank=2, control_url="http://10.0.0.9:41111"
)
_REQUEST = ReceiverFaultRequest(
    cell_id="engine-0",
    expected_workers_hash="hash-1",
    receiver=_RECEIVER,
    mode=FailureMode.SIGKILL,
    request_id="req-1",
)


class _ControlPlane(BaseCellOperations):
    def __init__(self, *, answer: asyncio.Future | None = None, is_current: bool = True) -> None:
        self._answer = answer
        self._is_current = is_current
        self.checks = 0

    async def cell_infos(self, *, pool_ids):
        raise AssertionError("not used")

    async def suspend(self, *, cell_id: str) -> None:
        raise AssertionError("not used")

    async def resume(self, *, cell_id: str) -> None:
        raise AssertionError("not used")

    async def terminate_incarnation(self, *, cell_id: str, expected_workers_hash: str, timeout: float = 0.0):
        raise AssertionError("not used")

    async def inject_fault(self, **kwargs):
        raise AssertionError("not used")

    async def incarnation_is_current(self, *, cell_id: str, expected_workers_hash: str) -> bool:
        self.checks += 1
        if self._answer is not None:
            return await self._answer
        return self._is_current


def _recording_receiver(asked: list[dict], outcome: ReceiverFaultOutcome):
    async def request(*, receiver, mode, request_id):
        asked.append({"receiver": receiver, "mode": mode, "request_id": request_id})
        return outcome

    return request


class TestTheIncarnationCheckIsBounded:
    async def test_a_control_plane_that_never_answers_returns_within_the_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The after-submit hook runs on the trainer's own thread, so a wedged control plane must not hold it."""
        monkeypatch.setattr(cell_operations_base, "INCARNATION_CHECK_TIMEOUT_SECONDS", 0.05)
        asked: list[dict] = []
        monkeypatch.setattr(
            cell_operations_base, "request_receiver_fault", _recording_receiver(asked, ReceiverFaultOutcome.ACCEPTED)
        )
        never_answers: asyncio.Future = asyncio.get_running_loop().create_future()
        operations = _ControlPlane(answer=never_answers)

        outcome = await asyncio.wait_for(operations._inject_receiver_fault(_REQUEST), timeout=5.0)

        assert outcome is FaultInjectionOutcome.UNKNOWN
        assert asked == []
        never_answers.cancel()

    async def test_an_unverified_incarnation_is_never_reported_as_stale(self, monkeypatch: pytest.MonkeyPatch):
        """Calling it stale would let a run claim a recovery of a target this request never reached."""
        monkeypatch.setattr(cell_operations_base, "INCARNATION_CHECK_TIMEOUT_SECONDS", 0.05)
        monkeypatch.setattr(
            cell_operations_base, "request_receiver_fault", _recording_receiver([], ReceiverFaultOutcome.ACCEPTED)
        )
        never_answers: asyncio.Future = asyncio.get_running_loop().create_future()

        outcome = await _ControlPlane(answer=never_answers)._inject_receiver_fault(_REQUEST)

        assert outcome not in (FaultInjectionOutcome.STALE, FaultInjectionOutcome.INJECTED)
        never_answers.cancel()

    async def test_a_failing_check_sends_no_request_either(self, monkeypatch: pytest.MonkeyPatch):
        """An unverified identity must not be handed to the receiver, whatever the reason the check failed."""
        asked: list[dict] = []
        monkeypatch.setattr(
            cell_operations_base, "request_receiver_fault", _recording_receiver(asked, ReceiverFaultOutcome.ACCEPTED)
        )
        failed: asyncio.Future = asyncio.get_running_loop().create_future()
        failed.set_exception(ConnectionError("the worker manager is gone"))

        outcome = await _ControlPlane(answer=failed)._inject_receiver_fault(_REQUEST)

        assert outcome is FaultInjectionOutcome.UNKNOWN
        assert asked == []

    async def test_a_verified_incarnation_still_reaches_the_receiver(self, monkeypatch: pytest.MonkeyPatch):
        """The deadline is a bound on the check, not a change to what a healthy control plane delivers."""
        asked: list[dict] = []
        monkeypatch.setattr(
            cell_operations_base, "request_receiver_fault", _recording_receiver(asked, ReceiverFaultOutcome.ACCEPTED)
        )
        operations = _ControlPlane(is_current=True)

        outcome = await operations._inject_receiver_fault(_REQUEST)

        assert outcome is FaultInjectionOutcome.ACCEPTED
        assert asked == [{"receiver": _RECEIVER, "mode": FailureMode.SIGKILL, "request_id": "req-1"}]

    async def test_a_replaced_incarnation_is_still_stale(self, monkeypatch: pytest.MonkeyPatch):
        """A check that answered is conclusive, and its answer keeps its meaning."""
        asked: list[dict] = []
        monkeypatch.setattr(
            cell_operations_base, "request_receiver_fault", _recording_receiver(asked, ReceiverFaultOutcome.ACCEPTED)
        )

        outcome = await _ControlPlane(is_current=False)._inject_receiver_fault(_REQUEST)

        assert outcome is FaultInjectionOutcome.STALE
        assert asked == []
