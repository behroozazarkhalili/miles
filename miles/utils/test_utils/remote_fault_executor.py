from __future__ import annotations

import logging

from miles.utils import async_utils
from miles.utils.test_utils.fault_hooks import FaultHookOutcome, RemoteInferenceTarget
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.cell_operations.base import FaultInjectionOutcome

logger = logging.getLogger(__name__)

_OUTCOME_OF_INJECTION: dict[FaultInjectionOutcome, FaultHookOutcome] = {
    FaultInjectionOutcome.INJECTED: FaultHookOutcome.FIRED,
    FaultInjectionOutcome.ACCEPTED: FaultHookOutcome.ACCEPTED,
    FaultInjectionOutcome.STALE: FaultHookOutcome.STALE_TARGET,
    FaultInjectionOutcome.UNKNOWN: FaultHookOutcome.UNKNOWN,
}


class CellOperationsRemoteFaultExecutor:
    def __init__(self, *, capability: BackendCapability) -> None:
        self._capability = capability

    def __call__(self, *, target: RemoteInferenceTarget, mode: FailureMode, request_id: str) -> FaultHookOutcome:
        outcome = async_utils.run(
            self._capability.cell_operations().inject_fault(
                cell_id=target.cell_id,
                mode=mode,
                expected_workers_hash=target.workers_hash,
                receiver=target.receiver,
                request_id=request_id,
            )
        )
        hook_outcome = _OUTCOME_OF_INJECTION[outcome]
        _log_outcome(hook_outcome, target=target, request_id=request_id)
        return hook_outcome


def _log_outcome(outcome: FaultHookOutcome, *, target: RemoteInferenceTarget, request_id: str) -> None:
    match outcome:
        case FaultHookOutcome.STALE_TARGET:
            logger.warning(
                "The receiver %s of %s refused request %s because that incarnation is gone, so nothing was harmed",
                target.receiver.receiver_boot_uuid,
                target.cell_id,
                request_id,
            )
        case FaultHookOutcome.UNKNOWN:
            logger.warning(
                "Request %s for the receiver %s of %s has no answer, so whether that incarnation was signalled is "
                "unknown and the request stays outstanding",
                request_id,
                target.receiver.receiver_boot_uuid,
                target.cell_id,
            )
        case _:
            return None
