from __future__ import annotations

import logging
from typing import Any

import pytest

from miles.utils.test_utils.fault_hooks import FaultHookOutcome, RemoteInferenceTarget
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.test_utils.receiver_fault import ReceiverFaultRefusedError, ReceiverIdentity
from miles.utils.test_utils.remote_fault_executor import CellOperationsRemoteFaultExecutor
from miles.utils.workers.cell_operations.base import FaultInjectionOutcome

_RECEIVER = ReceiverIdentity(
    receiver_boot_uuid="boot-1", session_id="session-1", rank=2, control_url="http://10.0.0.9:41111"
)
_TARGET = RemoteInferenceTarget(cell_id="engine-0", workers_hash="hash-1", receiver=_RECEIVER)


class _RecordingCellOperations:
    def __init__(self, *, outcome: FaultInjectionOutcome | Exception) -> None:
        self.calls: list[dict[str, Any]] = []
        self._outcome = outcome

    async def inject_fault(self, **kwargs: Any) -> FaultInjectionOutcome:
        self.calls.append(kwargs)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class _Capability:
    def __init__(self, operations: _RecordingCellOperations) -> None:
        self._operations = operations

    def cell_operations(self) -> _RecordingCellOperations:
        return self._operations


def _execute(
    operations: _RecordingCellOperations,
    *,
    request_id: str = "req-1",
    mode: FailureMode = FailureMode.SIGKILL,
) -> FaultHookOutcome:
    return CellOperationsRemoteFaultExecutor(capability=_Capability(operations))(
        target=_TARGET, mode=mode, request_id=request_id
    )


class TestDeliveringARemoteFault:
    def test_the_receiver_identity_and_request_id_travel_with_the_request(self):
        """The receiver only acts on a request naming its own incarnation, so both have to reach it unchanged."""
        operations = _RecordingCellOperations(outcome=FaultInjectionOutcome.ACCEPTED)

        outcome = _execute(operations, request_id="req-7")

        assert outcome is FaultHookOutcome.ACCEPTED
        assert operations.calls == [
            dict(
                cell_id="engine-0",
                mode=FailureMode.SIGKILL,
                expected_workers_hash="hash-1",
                receiver=_RECEIVER,
                request_id="req-7",
            )
        ]

    def test_a_sigstop_is_aimed_at_the_same_frozen_receiver_as_a_kill(self):
        """A hang has to reach the engine rank holding the session, never a supervisor that only knows its name."""
        operations = _RecordingCellOperations(outcome=FaultInjectionOutcome.ACCEPTED)

        outcome = _execute(operations, request_id="req-7", mode=FailureMode.SIGSTOP)

        assert outcome is FaultHookOutcome.ACCEPTED
        assert operations.calls == [
            dict(
                cell_id="engine-0",
                mode=FailureMode.SIGSTOP,
                expected_workers_hash="hash-1",
                receiver=_RECEIVER,
                request_id="req-7",
            )
        ]

    def test_the_executor_does_not_mint_its_own_request_id(self):
        """A second id would leave the arm, the fire and the receiver's answer impossible to pair."""
        operations = _RecordingCellOperations(outcome=FaultInjectionOutcome.ACCEPTED)

        _execute(operations, request_id="req-9")

        assert operations.calls[0]["request_id"] == "req-9"

    def test_an_accepted_answer_is_not_reported_as_fired(self):
        """Accepting only says the receiver took the request; the signal is its own step."""
        operations = _RecordingCellOperations(outcome=FaultInjectionOutcome.ACCEPTED)

        assert _execute(operations) is FaultHookOutcome.ACCEPTED

    def test_a_refused_fault_is_reported_as_a_stale_target(self):
        """The replacement must not be harmed, and the run must not count the refusal as harm done."""
        operations = _RecordingCellOperations(outcome=FaultInjectionOutcome.STALE)

        assert _execute(operations) is FaultHookOutcome.STALE_TARGET

    def test_an_unanswered_request_is_reported_as_unknown(self):
        """A lost answer says nothing about whether the receiver signalled itself, and must not read as success."""
        operations = _RecordingCellOperations(outcome=FaultInjectionOutcome.UNKNOWN)

        assert _execute(operations) is FaultHookOutcome.UNKNOWN

    def test_a_refusal_reaches_the_caller(self):
        """A receiver that rejects for any other reason leaves the request unaccounted for, which must fail loudly."""
        operations = _RecordingCellOperations(outcome=ReceiverFaultRefusedError("pending_action_exists"))

        with pytest.raises(ReceiverFaultRefusedError):
            _execute(operations)


class TestOutcomeLogging:
    def test_an_unknown_answer_is_not_logged_as_harmless(self, caplog: pytest.LogCaptureFixture):
        """A request whose answer was lost may still have signalled the receiver, so it did not prove innocence."""
        caplog.set_level(logging.WARNING)
        operations = _RecordingCellOperations(outcome=FaultInjectionOutcome.UNKNOWN)

        _execute(operations)

        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "nothing was harmed" not in messages
        assert "unknown" in messages and "outstanding" in messages

    def test_a_stale_answer_is_logged_as_harmless(self, caplog: pytest.LogCaptureFixture):
        """The receiver refused because its incarnation is gone, which is the one conclusive no-harm answer."""
        caplog.set_level(logging.WARNING)
        operations = _RecordingCellOperations(outcome=FaultInjectionOutcome.STALE)

        _execute(operations)

        assert "nothing was harmed" in " ".join(record.getMessage() for record in caplog.records)

    def test_an_accepted_answer_logs_no_warning(self, caplog: pytest.LogCaptureFixture):
        """Nothing went wrong, and a warning here would train a reader to ignore the two that matter."""
        caplog.set_level(logging.WARNING)
        operations = _RecordingCellOperations(outcome=FaultInjectionOutcome.ACCEPTED)

        _execute(operations)

        assert [record for record in caplog.records if record.name.endswith("remote_fault_executor")] == []
