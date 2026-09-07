from __future__ import annotations

import asyncio
from typing import Protocol

from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.models import Cell, CellStatus, FaultHookArmingReport
from miles.utils.test_utils.fault_hooks import FaultHookName, FaultHookTarget
from miles.utils.test_utils.fault_injector import FailureMode


class FaultHookSourceNotFoundError(Exception):
    pass


class _FaultHookSourceController(Protocol):
    async def get_cell_statuses(self) -> dict[str, CellStatus]: ...

    async def arm_fault_hook(
        self,
        cell_id: str,
        *,
        expected_workers_hash: str,
        hook: str,
        mode: str,
        sub_index: int,
        request_id: str,
        target: str,
    ) -> FaultHookArmingReport: ...


class _FaultHookSourceRegistry:
    def __init__(self, *, handler: _CellHandler | None, controllers: list[_FaultHookSourceController]) -> None:
        self._handler = handler
        self._controllers = controllers

    async def list_cells(self) -> list[Cell]:
        return [] if self._handler is None else await self._handler.list_cells()

    async def arm_fault_hook(
        self,
        cell_id: str,
        *,
        expected_workers_hash: str,
        hook: FaultHookName,
        mode: FailureMode,
        sub_index: int,
        request_id: str,
        target: FaultHookTarget,
    ) -> FaultHookArmingReport:
        controller = await self._resolve(cell_id)
        return await controller.arm_fault_hook(
            cell_id,
            expected_workers_hash=expected_workers_hash,
            hook=hook.value,
            mode=mode.value,
            sub_index=sub_index,
            request_id=request_id,
            target=target.value,
        )

    async def _resolve(self, cell_id: str) -> _FaultHookSourceController:
        per_controller = await asyncio.gather(*(c.get_cell_statuses() for c in self._controllers))
        owners = [
            controller
            for controller, statuses in zip(self._controllers, per_controller, strict=True)
            if cell_id in statuses
        ]
        if not owners:
            raise FaultHookSourceNotFoundError(f"no trainer of this deployment answers for {cell_id}")
        assert (
            len(owners) == 1
        ), f"{len(owners)} trainers answer for {cell_id}, so arming it would pick one of them by accident"
        return owners[0]
