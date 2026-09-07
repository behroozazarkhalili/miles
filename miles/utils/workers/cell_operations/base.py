from __future__ import annotations

import abc
import asyncio
import enum
import logging

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.test_utils.receiver_fault import ReceiverFaultOutcome, ReceiverIdentity, request_receiver_fault
from miles.utils.workers.worker_provider.base import CellInfo

logger = logging.getLogger(__name__)

TERMINATE_INCARNATION_TIMEOUT_SECONDS = 120.0
CELL_TERMINATION_NOT_CONFIRMED = "not_confirmed"
INCARNATION_CHECK_TIMEOUT_SECONDS = 30.0


class FaultInjectionOutcome(enum.Enum):
    INJECTED = "injected"
    ACCEPTED = "accepted"
    STALE = "stale"
    UNKNOWN = "unknown"


_OUTCOME_OF_RECEIVER_ANSWER: dict[ReceiverFaultOutcome, FaultInjectionOutcome] = {
    ReceiverFaultOutcome.ACCEPTED: FaultInjectionOutcome.ACCEPTED,
    ReceiverFaultOutcome.STALE_TARGET: FaultInjectionOutcome.STALE,
    ReceiverFaultOutcome.UNKNOWN: FaultInjectionOutcome.UNKNOWN,
}


class IncarnationBoundInjectionUnsupportedError(Exception):
    pass


class ReceiverFaultRequest(FrozenStrictBaseModel):
    cell_id: str
    expected_workers_hash: str
    receiver: ReceiverIdentity
    mode: FailureMode
    request_id: str


class CellTerminationOutcome(enum.Enum):
    TERMINATED = "terminated"
    ALREADY_GONE = "already_gone"
    STALE = "stale"


class CellTerminationNotConfirmedError(Exception):
    pass


class BaseCellOperations(abc.ABC):
    @abc.abstractmethod
    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]: ...

    @abc.abstractmethod
    async def suspend(self, *, cell_id: str) -> None: ...

    @abc.abstractmethod
    async def resume(self, *, cell_id: str) -> None: ...

    @abc.abstractmethod
    async def terminate_incarnation(
        self,
        *,
        cell_id: str,
        expected_workers_hash: str,
        timeout: float = TERMINATE_INCARNATION_TIMEOUT_SECONDS,
    ) -> CellTerminationOutcome: ...

    @abc.abstractmethod
    async def inject_fault(
        self,
        *,
        cell_id: str,
        mode: FailureMode,
        sub_index: int | None = None,
        expected_workers_hash: str | None = None,
        receiver: ReceiverIdentity | None = None,
        request_id: str | None = None,
    ) -> FaultInjectionOutcome: ...

    async def _inject_receiver_fault(self, request: ReceiverFaultRequest) -> FaultInjectionOutcome:
        try:
            is_current = await asyncio.wait_for(
                self.incarnation_is_current(
                    cell_id=request.cell_id, expected_workers_hash=request.expected_workers_hash
                ),
                timeout=INCARNATION_CHECK_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.error(
                "Reading whether %s still runs %s did not finish within %ss, so request %s was never sent and "
                "nothing can be concluded about the incarnation it named",
                request.cell_id,
                request.expected_workers_hash,
                INCARNATION_CHECK_TIMEOUT_SECONDS,
                request.request_id,
                exc_info=True,
            )
            return FaultInjectionOutcome.UNKNOWN

        if not is_current:
            logger.warning(
                "Not asking the receiver of %s for %s: the cell no longer runs %s",
                request.cell_id,
                request.request_id,
                request.expected_workers_hash,
            )
            return FaultInjectionOutcome.STALE

        answer = await request_receiver_fault(
            receiver=request.receiver, mode=request.mode, request_id=request.request_id
        )
        return _OUTCOME_OF_RECEIVER_ANSWER[answer]

    @abc.abstractmethod
    async def incarnation_is_current(self, *, cell_id: str, expected_workers_hash: str) -> bool: ...
