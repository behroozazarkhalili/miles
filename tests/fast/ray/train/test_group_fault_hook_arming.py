from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI

import miles.ray.train.group as group_module
from miles.ray.train.group import TrainerController
from miles.utils.test_utils.fault_hooks import FaultHookName, FaultHookTarget
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.rpc.common.metadata import rpc
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.types import DeploymentIdentity
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

pytestmark = pytest.mark.asyncio

_CELL_ID = "trainer-engine-actor-0"
_HASH_A = "workers-hash-a"
_HASH_B = "workers-hash-b"


class _RecordingHandle:
    def __init__(self, *, tag: str = "a", before_arm=None) -> None:
        self.tag = tag
        self.armed: list[dict[str, object]] = []
        self._before_arm = before_arm
        self.hangs = False

    async def arm_fault_hook(self, *, hook: str, mode: str, request_id: str, target: str) -> None:
        if self._before_arm is not None:
            await self._before_arm()
        if self.hangs:
            await asyncio.Event().wait()
        self.armed.append(dict(hook=hook, mode=mode, request_id=request_id, target=target))


class _FakeCell:
    def __init__(self, *, workers_hash: str, handles: list[object], alive: bool = True) -> None:
        self.cell_id = _CELL_ID
        self.workers_hash = workers_hash
        self.worker_handles = handles
        self.is_alive = alive
        self.state_name = "StateAllocatedAlive" if alive else "StateAllocatedUninitialized"


def _controller(cells: dict[str, _FakeCell]) -> TrainerController:
    controller = TrainerController(
        deployment_identity=DeploymentIdentity(run_uuid="0123456789abcdef", deploy_component="trainer"),
        cell_provider=MagicMock(spec=BaseWorkerProvider),
        cell_operations=MagicMock(spec=BaseCellOperations),
        trainer_id="actor",
        role="actor",
        with_ref=False,
    )
    controller._cells_by_id = cells
    return controller


async def _arm(controller: TrainerController, *, expected_workers_hash: str = _HASH_A, sub_index: int = 0):
    return await controller.arm_fault_hook(
        _CELL_ID,
        expected_workers_hash=expected_workers_hash,
        hook=FaultHookName.WEIGHT_UPDATE_BEFORE_P2P_WRITE.value,
        mode=FailureMode.SIGKILL.value,
        sub_index=sub_index,
        request_id="req-1",
        target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
    )


class TestArmingTheGenerationTheCallerChose:
    async def test_the_request_reaches_the_worker_it_named(self):
        """The hook has to be armed in the rank the request picked, carrying every field it was armed with."""
        handles = [_RecordingHandle(tag="rank-0"), _RecordingHandle(tag="rank-1")]
        controller = _controller({_CELL_ID: _FakeCell(workers_hash=_HASH_A, handles=handles)})

        report = await _arm(controller, sub_index=1)

        assert report.refused_because is None
        assert handles[0].armed == []
        assert handles[1].armed == [
            dict(
                hook="weight_update.before_p2p_write",
                mode="sigkill",
                request_id="req-1",
                target="remote_inference_cell",
            )
        ]

    async def test_a_snapshot_of_a_replaced_generation_arms_nothing(self):
        """The caller chose the generation it saw, and the replacement is a cell nobody asked to harm."""
        handle = _RecordingHandle()
        controller = _controller({_CELL_ID: _FakeCell(workers_hash=_HASH_B, handles=[handle])})

        report = await _arm(controller, expected_workers_hash=_HASH_A)

        assert handle.armed == []
        assert _HASH_A in report.refused_because and _HASH_B in report.refused_because

    async def test_a_replacement_created_after_the_hash_matched_is_never_reached(self):
        """The handle is frozen before the first await, so a cell swapped in mid-request cannot take the fault."""
        replacement = _RecordingHandle(tag="b")
        cells: dict[str, _FakeCell] = {}

        async def _replace_the_cell() -> None:
            cells[_CELL_ID] = _FakeCell(workers_hash=_HASH_B, handles=[replacement])

        original = _RecordingHandle(tag="a", before_arm=_replace_the_cell)
        cells[_CELL_ID] = _FakeCell(workers_hash=_HASH_A, handles=[original])
        controller = _controller(cells)

        report = await _arm(controller)

        assert report.refused_because is None
        assert len(original.armed) == 1
        assert replacement.armed == []
        assert cells[_CELL_ID].workers_hash == _HASH_B


class TestArmingWhatCannotCarryTheHook:
    async def test_a_cell_of_another_trainer_is_refused(self):
        """Arming some cell this controller does happen to hold would harm a process nobody named."""
        controller = _controller({})

        report = await _arm(controller)

        assert "trainer-engine-actor" in report.refused_because

    async def test_a_cell_that_has_not_finished_init_is_refused(self):
        """Its workers have taken no rank in the run yet, so nothing there will ever reach a weight-update hook."""
        handle = _RecordingHandle()
        controller = _controller({_CELL_ID: _FakeCell(workers_hash=_HASH_A, handles=[handle], alive=False)})

        report = await _arm(controller)

        assert handle.armed == []
        assert "StateAllocatedUninitialized" in report.refused_because

    async def test_a_sub_index_past_the_last_worker_is_refused(self):
        """Arming a neighbouring rank instead would leave the named one clean and the run unexplained."""
        handle = _RecordingHandle()
        controller = _controller({_CELL_ID: _FakeCell(workers_hash=_HASH_A, handles=[handle])})

        report = await _arm(controller, sub_index=3)

        assert handle.armed == []
        assert "sub_index 3" in report.refused_because

    async def test_a_worker_that_never_answers_does_not_wedge_the_caller(self, monkeypatch: pytest.MonkeyPatch):
        """The api request that armed this hook would otherwise never be answered either way."""
        handle = _RecordingHandle()
        handle.hangs = True
        controller = _controller({_CELL_ID: _FakeCell(workers_hash=_HASH_A, handles=[handle])})
        monkeypatch.setattr(group_module, "_ARM_FAULT_HOOK_TIMEOUT_SECONDS", 0.05)

        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await _arm(controller)


class _HookableWorker:
    def __init__(self) -> None:
        self.armed: list[dict[str, object]] = []

    @rpc(concurrency_group="fault_injector")
    def arm_fault_hook(self, *, hook: str, mode: str, request_id: str, target: str) -> None:
        self.armed.append(dict(hook=hook, mode=mode, request_id=request_id, target=target))


class _EndpointTransport(httpx.AsyncBaseTransport):
    def __init__(self, app: FastAPI) -> None:
        self._transport = httpx.ASGITransport(app=app)

    def serve(self, app: FastAPI) -> None:
        self._transport = httpx.ASGITransport(app=app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._transport.handle_async_request(request)


@asynccontextmanager
async def _served(*apps: FastAPI) -> AsyncIterator[None]:
    if not apps:
        yield
        return
    async with apps[0].router.lifespan_context(apps[0]):
        async with _served(*apps[1:]):
            yield


class TestTheRpcBootGuardCloses:
    async def test_a_replacement_that_reuses_the_endpoint_is_refused_before_it_is_armed(self):
        """Under kubernetes the address survives the process, so only the pinned boot uuid tells the two apart."""
        first, second = _HookableWorker(), _HookableWorker()
        first_app, second_app = create_rpc_app(first), create_rpc_app(second)
        transport = _EndpointTransport(first_app)

        async with _served(first_app, second_app):
            async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
                handle = RpcWorkerHandle(
                    _HookableWorker,
                    server_url="http://worker",
                    require_stable_boot_uuid=True,
                    http_client=client,
                )
                controller = _controller({_CELL_ID: _FakeCell(workers_hash=_HASH_A, handles=[handle])})

                report = await _arm(controller)
                transport.serve(second_app)

                with pytest.raises(ServerRestartedError):
                    await _arm(controller)

        assert report.refused_because is None
        assert len(first.armed) == 1
        assert second.armed == []
