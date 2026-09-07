from __future__ import annotations

import asyncio

import ray.actor

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.test_utils.receiver_fault import ReceiverIdentity
from miles.utils.workers.cell_operations.base import (
    CELL_TERMINATION_NOT_CONFIRMED,
    TERMINATE_INCARNATION_TIMEOUT_SECONDS,
    BaseCellOperations,
    CellTerminationNotConfirmedError,
    CellTerminationOutcome,
    FaultInjectionOutcome,
    ReceiverFaultRequest,
)
from miles.utils.workers.worker_provider.base import CellInfo


class RayCellOperations(BaseCellOperations):
    def __init__(self, *, worker_manager_handle: ray.actor.ActorHandle) -> None:
        self._worker_manager_handle = worker_manager_handle

    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        return await self._worker_manager_handle.get_cell_infos.remote(pool_ids=pool_ids)

    async def suspend(self, *, cell_id: str) -> None:
        await self._worker_manager_handle.stop_cells.remote([cell_id])

    async def resume(self, *, cell_id: str) -> None:
        await self._worker_manager_handle.start_cells.remote([cell_id])

    async def terminate_incarnation(
        self,
        *,
        cell_id: str,
        expected_workers_hash: str,
        timeout: float = TERMINATE_INCARNATION_TIMEOUT_SECONDS,
    ) -> CellTerminationOutcome:
        try:
            outcome = await asyncio.wait_for(
                _stop_cell_incarnation(
                    self._worker_manager_handle, cell_id=cell_id, expected_workers_hash=expected_workers_hash
                ),
                timeout=timeout,
            )
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise CellTerminationNotConfirmedError(
                f"the worker manager did not answer within {timeout}s whether it stopped {cell_id} "
                f"({expected_workers_hash}), so its workers may still be running"
            ) from e
        if outcome == CELL_TERMINATION_NOT_CONFIRMED:
            raise CellTerminationNotConfirmedError(
                f"the worker manager killed {cell_id} ({expected_workers_hash}) but its actors kept answering, "
                f"so their workers may still be running"
            )
        return CellTerminationOutcome(outcome)

    async def inject_fault(
        self,
        *,
        cell_id: str,
        mode: FailureMode,
        sub_index: int | None = None,
        expected_workers_hash: str | None = None,
        receiver: ReceiverIdentity | None = None,
        request_id: str | None = None,
    ) -> FaultInjectionOutcome:
        if receiver is not None:
            assert expected_workers_hash is not None and request_id is not None, (
                f"a fault aimed at the receiver of {cell_id} needs both the incarnation it was observed at and the "
                f"request id the receiver checks it against"
            )
            return await self._inject_receiver_fault(
                ReceiverFaultRequest(
                    cell_id=cell_id,
                    expected_workers_hash=expected_workers_hash,
                    receiver=receiver,
                    mode=mode,
                    request_id=request_id,
                )
            )

        assert sub_index is not None, f"a fault aimed at a worker of {cell_id} needs the index of that worker"
        outcome = await self._worker_manager_handle.inject_fault.remote(
            cell_id,
            mode=mode.value,
            worker_in_cell_index=sub_index,
            expected_workers_hash=expected_workers_hash,
        )
        return FaultInjectionOutcome(outcome)

    async def incarnation_is_current(self, *, cell_id: str, expected_workers_hash: str) -> bool:
        return await self._worker_manager_handle.incarnation_is_current.remote(
            cell_id, expected_workers_hash=expected_workers_hash
        )


async def _stop_cell_incarnation(
    worker_manager_handle: ray.actor.ActorHandle, *, cell_id: str, expected_workers_hash: str
) -> str:
    return await worker_manager_handle.stop_cell_incarnation.remote(
        cell_id, expected_workers_hash=expected_workers_hash
    )
