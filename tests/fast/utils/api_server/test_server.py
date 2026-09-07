from __future__ import annotations

import socket
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import httpx
import pytest
from fastapi import FastAPI

from tests.fast.ray.rollout.conftest import make_args as make_rollout_args

from miles.ray.rollout.server_cell import compute_pending_rollout_cell_status
from miles.utils.ft_utils.api_server import server
from miles.utils.ft_utils.api_server.fault_hook_sources import _FaultHookSourceRegistry
from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.registry import _CellRegistry
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.fault_hooks import MAX_FAULT_HOOK_DELAY_MS
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.ray import RayCellOperations

from .conftest import (
    SOURCE_CELL_ID,
    SOURCE_WORKERS_HASH,
    MockHandler,
    MockInferenceController,
    MockSourceController,
    MockTrainerCell,
    MockWorkerManager,
    make_cell_summaries,
    make_mock_controller,
)

_EMPTY_SOURCES = _FaultHookSourceRegistry(handler=None, controllers=[])


class TestGetHealth:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, async_client: httpx.AsyncClient) -> None:
        resp = await async_client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestGetCells:
    @pytest.mark.asyncio
    async def test_empty_registry_returns_empty_cell_list(self, async_client: httpx.AsyncClient) -> None:
        resp = await async_client.get("/api/v1/cells")
        assert resp.status_code == 200
        assert resp.json() == {
            "apiVersion": "miles.io/v1",
            "kind": "CellList",
            "items": [],
        }

    @pytest.mark.asyncio
    async def test_returns_all_cells_golden(
        self, actor_handler: MockHandler, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """Golden test: full JSON response for GET /api/v1/cells with two cells."""
        actor_handler.add("actor-0", phase="Running")
        rollout_handler.add(
            "rollout-0",
            phase="Suspended",
            is_suspended=True,
            conditions=[
                {"type": "Allocated", "status": "False"},
                {"type": "Healthy", "status": "False"},
            ],
        )

        resp = await async_client.get("/api/v1/cells")
        assert resp.status_code == 200
        assert resp.json() == {
            "apiVersion": "miles.io/v1",
            "kind": "CellList",
            "items": [
                {
                    "apiVersion": "miles.io/v1",
                    "kind": "Cell",
                    "metadata": {
                        "name": "actor-0",
                        "labels": {
                            "miles.io/cell-type": "actor",
                            "miles.io/cell-id": "actor-0",
                        },
                    },
                    "spec": {"suspend": False},
                    "status": {
                        "phase": "Running",
                        "conditions": [
                            {
                                "type": "Allocated",
                                "status": "True",
                                "reason": None,
                                "message": None,
                                "lastTransitionTime": None,
                            },
                            {
                                "type": "Healthy",
                                "status": "True",
                                "reason": None,
                                "message": None,
                                "lastTransitionTime": None,
                            },
                        ],
                        "workers_hash": "pseudo-hash-0",
                    },
                },
                {
                    "apiVersion": "miles.io/v1",
                    "kind": "Cell",
                    "metadata": {
                        "name": "rollout-0",
                        "labels": {
                            "miles.io/cell-type": "rollout",
                            "miles.io/cell-id": "rollout-0",
                        },
                    },
                    "spec": {"suspend": True},
                    "status": {
                        "phase": "Suspended",
                        "conditions": [
                            {
                                "type": "Allocated",
                                "status": "False",
                                "reason": None,
                                "message": None,
                                "lastTransitionTime": None,
                            },
                            {
                                "type": "Healthy",
                                "status": "False",
                                "reason": None,
                                "message": None,
                                "lastTransitionTime": None,
                            },
                        ],
                        "workers_hash": "pseudo-hash-0",
                    },
                },
            ],
        }


class TestGetCell:
    @pytest.mark.asyncio
    async def test_returns_single_cell_golden(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """Golden test: full JSON response for GET /api/v1/cells/{name}."""
        actor_handler.add("actor-0", phase="Running")

        resp = await async_client.get("/api/v1/cells/actor-0")
        assert resp.status_code == 200
        assert resp.json() == {
            "apiVersion": "miles.io/v1",
            "kind": "Cell",
            "metadata": {
                "name": "actor-0",
                "labels": {
                    "miles.io/cell-type": "actor",
                    "miles.io/cell-id": "actor-0",
                },
            },
            "spec": {"suspend": False},
            "status": {
                "phase": "Running",
                "conditions": [
                    {
                        "type": "Allocated",
                        "status": "True",
                        "reason": None,
                        "message": None,
                        "lastTransitionTime": None,
                    },
                    {"type": "Healthy", "status": "True", "reason": None, "message": None, "lastTransitionTime": None},
                ],
                "workers_hash": "pseudo-hash-0",
            },
        }

    @pytest.mark.asyncio
    async def test_not_found_returns_k8s_status_golden(self, async_client: httpx.AsyncClient) -> None:
        """Golden test: K8s Status error response for 404."""
        resp = await async_client.get("/api/v1/cells/nonexistent")
        assert resp.status_code == 404
        assert resp.json() == {
            "apiVersion": "v1",
            "kind": "Status",
            "status": "Failure",
            "message": "Cell 'nonexistent' not found",
            "reason": "NotFound",
            "code": 404,
        }


class TestPatchCell:
    @pytest.mark.asyncio
    async def test_suspend_cell_via_patch(self, actor_handler: MockHandler, async_client: httpx.AsyncClient) -> None:
        cell = actor_handler.add("actor-0", phase="Running")

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": True}})
        assert resp.status_code == 200
        assert cell.suspend_calls == 1
        assert resp.json()["status"]["phase"] == "Suspended"
        assert resp.json()["spec"]["suspend"] is True

    @pytest.mark.asyncio
    async def test_resume_cell_via_patch(self, actor_handler: MockHandler, async_client: httpx.AsyncClient) -> None:
        cell = actor_handler.add("actor-0", phase="Suspended", is_suspended=True)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": False}})
        assert resp.status_code == 200
        assert cell.resume_calls == 1
        assert resp.json()["status"]["phase"] == "Running"

    @pytest.mark.asyncio
    async def test_patch_with_no_spec_is_noop(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        cell = actor_handler.add("actor-0", phase="Running")

        resp = await async_client.patch("/api/v1/cells/actor-0", json={})
        assert resp.status_code == 200
        assert cell.suspend_calls == 0
        assert cell.resume_calls == 0

    @pytest.mark.asyncio
    async def test_patch_with_empty_spec_does_not_suspend_or_resume(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """A spec that omits suspend carries no instruction, so the cell must be left exactly as it was."""
        cell = actor_handler.add("actor-0", phase="Running")

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {}})

        assert resp.status_code == 200
        assert (cell.suspend_calls, cell.resume_calls) == (0, 0)
        assert resp.json()["spec"]["suspend"] is False
        assert resp.json()["status"]["phase"] == "Running"

    @pytest.mark.asyncio
    async def test_patch_not_found_returns_k8s_status(self, async_client: httpx.AsyncClient) -> None:
        resp = await async_client.patch("/api/v1/cells/nonexistent", json={"spec": {"suspend": True}})
        assert resp.status_code == 404
        assert resp.json()["kind"] == "Status"
        assert resp.json()["reason"] == "NotFound"

    @pytest.mark.asyncio
    async def test_patch_suspend_idempotent(self, actor_handler: MockHandler, async_client: httpx.AsyncClient) -> None:
        cell = actor_handler.add("actor-0", phase="Suspended", is_suspended=True)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": True}})
        assert resp.status_code == 200
        assert cell.suspend_calls == 1

    @pytest.mark.asyncio
    async def test_patch_error_returns_500_k8s_status(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        actor_handler.add("actor-0", suspend_error=RuntimeError("engine crashed"))

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": True}})
        assert resp.status_code == 500
        assert resp.json()["kind"] == "Status"
        assert resp.json()["reason"] == "InternalError"

    @pytest.mark.asyncio
    async def test_resume_error_returns_500_k8s_status(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """A resume that fails must be reported as loudly as a suspend that fails, or the cell looks resumed."""
        actor_handler.add("actor-0", is_suspended=True, resume_error=RuntimeError("relaunch rejected"))

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": False}})

        assert resp.status_code == 500
        assert resp.json() == {
            "apiVersion": "v1",
            "kind": "Status",
            "status": "Failure",
            "message": "Failed to patch cell 'actor-0'",
            "reason": "InternalError",
            "code": 500,
        }


class _StartedApiServer(NamedTuple):
    registry: _CellRegistry
    sources: _FaultHookSourceRegistry


class TestStartApiServerRegistration:
    def _start(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        ft_components: list[str],
        cell_ids: list[str],
        actor_cells: list[MockTrainerCell] | None = None,
    ) -> _CellRegistry:
        return self._start_server(
            monkeypatch, ft_components=ft_components, cell_ids=cell_ids, actor_cells=actor_cells
        ).registry

    def _start_server(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        ft_components: list[str],
        cell_ids: list[str],
        actor_cells: list[MockTrainerCell] | None = None,
    ) -> _StartedApiServer:
        manager = MockWorkerManager(make_cell_summaries(*cell_ids))
        started: list[_StartedApiServer] = []

        monkeypatch.setattr(server, "compute_engine_pool_ids", lambda args: ["inference-engine-0-0"])
        monkeypatch.setattr(
            server,
            "_start_api_server_raw",
            lambda *, registry, sources, port, host: started.append(_StartedApiServer(registry, sources)),
        )

        server.start_api_server(
            args=make_rollout_args(),
            trainer_models={"actor": make_mock_controller(actor_cells if actor_cells is not None else [])},
            inference_controller=MockInferenceController(
                {cell_id: compute_pending_rollout_cell_status(workers_hash="pseudo-hash-0") for cell_id in cell_ids}
            ),
            port=18080,
            ft_components=ft_components,
            cell_operations=RayCellOperations(
                worker_manager_handle=manager,
            ),
        )

        (one,) = started
        return one

    @pytest.mark.asyncio
    async def test_the_rollout_handler_enumerates_every_engine_cell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cell the handler never enumerates is invisible to the heal loop forever."""
        registry = self._start(
            monkeypatch,
            ft_components=["rollout"],
            cell_ids=["inference-engine-0-0-1", "inference-engine-0-0-0"],
        )

        assert [cell.metadata.name for cell in await registry.list_cells()] == [
            "inference-engine-0-0-0",
            "inference-engine-0-0-1",
        ]

    @pytest.mark.asyncio
    async def test_no_rollout_handler_exists_when_rollout_ft_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exposing suspend on engines nobody heals would let a request strand the pool."""
        registry = self._start(monkeypatch, ft_components=["train"], cell_ids=["inference-engine-0-0-0"])

        assert await registry.list_cells() == []

    @pytest.mark.asyncio
    async def test_the_actor_handler_enumerates_the_real_trainer_spec_cells(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The unpatched trainer spec name must match the real trainer cells, or the heal loop sees no trainer."""
        registry = self._start(
            monkeypatch,
            ft_components=["train"],
            cell_ids=["trainer-engine-actor-0"],
            actor_cells=[MockTrainerCell(phase="Running")],
        )

        cells = await registry.list_cells()
        assert [cell.metadata.name for cell in cells] == ["trainer-engine-actor-0"]
        assert cells[0].status.phase == "Running"

    @pytest.mark.asyncio
    async def test_both_handlers_coexist_under_mixed_ft(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mixed ft heals trainer and rollout cells through the one endpoint."""
        registry = self._start(monkeypatch, ft_components=["train", "rollout"], cell_ids=["inference-engine-0-0-0"])

        assert [handler.cell_type for handler in registry._handlers] == ["actor", "rollout"]

    @pytest.mark.asyncio
    async def test_mixed_ft_serves_the_trainer_and_the_engine_cells(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mixed ft is the only mode where one endpoint must answer for two pools, so both must be enumerable."""
        registry = self._start(
            monkeypatch,
            ft_components=["train", "rollout"],
            cell_ids=["trainer-engine-actor-0", "inference-engine-0-0-0"],
            actor_cells=[MockTrainerCell(phase="Running")],
        )

        cells = await registry.list_cells()
        assert [(cell.metadata.name, cell.metadata.labels["miles.io/cell-type"]) for cell in cells] == [
            ("trainer-engine-actor-0", "actor"),
            ("inference-engine-0-0-0", "rollout"),
        ]

    @pytest.mark.asyncio
    async def test_no_actor_handler_exists_when_train_ft_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exposing suspend on trainer cells nobody heals would let one request strand the whole run."""
        registry = self._start(
            monkeypatch,
            ft_components=["rollout"],
            cell_ids=["trainer-actor-0"],
            actor_cells=[MockTrainerCell(phase="Running")],
        )

        assert await registry.list_cells() == []

    @pytest.mark.asyncio
    async def test_a_rollout_only_run_still_offers_its_trainer_as_a_hook_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hook is armed in a trainer, so a rollout-only run that cannot see one can never arm anything."""
        started = self._start_server(
            monkeypatch,
            ft_components=["rollout"],
            cell_ids=["trainer-engine-actor-0", "inference-engine-0-0-0"],
            actor_cells=[MockTrainerCell(phase="Running")],
        )

        assert [cell.metadata.name for cell in await started.registry.list_cells()] == ["inference-engine-0-0-0"]
        assert [cell.metadata.name for cell in await started.sources.list_cells()] == ["trainer-engine-actor-0"]

    @pytest.mark.asyncio
    async def test_a_rollout_only_run_arms_the_trainer_that_holds_the_hook(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every layer between the request and the worker has to be wired, not just the source list."""
        cell = MockTrainerCell(phase="Running")
        started = self._start_server(
            monkeypatch,
            ft_components=["rollout"],
            cell_ids=["trainer-engine-actor-0", "inference-engine-0-0-0"],
            actor_cells=[cell],
        )
        app = server._create_api_app(started.registry, started.sources)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            listed = await client.get("/api/v1/cells")
            armed = await client.post(
                "/api/v1/cells/trainer-engine-actor-0/arm-fault-hook",
                json={
                    "expected_workers_hash": cell.workers_hash,
                    "hook": "weight_update.before_p2p_write",
                    "mode": "sigkill",
                    "request_id": "req-1",
                    "target": "remote_inference_cell",
                    "delay_ms": 400,
                },
            )
            suspended = await client.patch("/api/v1/cells/trainer-engine-actor-0", json={"spec": {"suspend": True}})

        assert [item["metadata"]["name"] for item in listed.json()["items"]] == ["inference-engine-0-0-0"]
        assert armed.status_code == 200
        assert cell.armed == [
            dict(
                hook="weight_update.before_p2p_write",
                mode="sigkill",
                request_id="req-1",
                target="remote_inference_cell",
                delay_ms=400,
            )
        ]
        assert suspended.status_code == 404

    @pytest.mark.asyncio
    async def test_the_requested_port_reaches_the_server_that_binds_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The FT controller is told this port out of band, so binding any other one makes the api unreachable."""
        ports: list[int] = []
        manager = MockWorkerManager(make_cell_summaries("trainer-engine-actor-0"))
        monkeypatch.setattr(
            server, "_start_api_server_raw", lambda *, registry, sources, port, host: ports.append(port)
        )

        server.start_api_server(
            args=make_rollout_args(),
            trainer_models={"actor": make_mock_controller([MockTrainerCell()])},
            inference_controller=MockInferenceController(),
            port=19137,
            ft_components=["train"],
            cell_operations=RayCellOperations(
                worker_manager_handle=manager,
            ),
        )

        assert ports == [19137]


class TestStartApiServerRaw:
    def test_uvicorn_serves_the_registry_app_on_a_daemon_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A foreground server blocks the training driver, and a non-loopback bind exposes unauthenticated cell control."""
        serving_threads: list[threading.Thread] = []
        start_and_wait_thread = server._start_and_wait_thread

        def _record_thread(**kwargs) -> threading.Thread:
            thread = start_and_wait_thread(**kwargs)
            serving_threads.append(thread)
            return thread

        monkeypatch.setattr(server, "_start_and_wait_thread", _record_thread)
        port = find_available_port(21200)

        running = server._start_api_server_raw(
            registry=_CellRegistry([]), sources=_EMPTY_SOURCES, port=port, host="127.0.0.1"
        )

        try:
            [thread] = serving_threads
            assert thread.daemon is True
            assert isinstance(running.config.app, FastAPI)
            assert (running.config.host, running.config.port) == ("127.0.0.1", port)
        finally:
            running.should_exit = True

    def test_a_caller_supplied_host_overrides_the_loopback_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Remote FT controllers are opted into by passing a wider bind host explicitly."""
        serving_threads: list[threading.Thread] = []
        start_and_wait_thread = server._start_and_wait_thread

        def _record_thread(**kwargs) -> threading.Thread:
            thread = start_and_wait_thread(**kwargs)
            serving_threads.append(thread)
            return thread

        monkeypatch.setattr(server, "_start_and_wait_thread", _record_thread)
        port = find_available_port(21300)

        running = server._start_api_server_raw(
            registry=_CellRegistry([]), sources=_EMPTY_SOURCES, port=port, host="0.0.0.0"
        )

        try:
            assert (running.config.host, running.config.port) == ("0.0.0.0", port)
        finally:
            running.should_exit = True

    def test_a_bound_port_serves_and_can_be_reached(self) -> None:
        """The happy path must still return once uvicorn is actually accepting connections."""
        port = find_available_port(21000)

        running = server._start_api_server_raw(
            registry=_CellRegistry([]), sources=_EMPTY_SOURCES, port=port, host="127.0.0.1"
        )
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/api/v1/health", timeout=10.0)
            assert resp.status_code == 200
        finally:
            running.should_exit = True

    def test_a_port_already_taken_fails_the_caller(self) -> None:
        """A second job silently losing the port would then poll the first job's cell registry."""
        port = find_available_port(21100)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            occupied.bind(("0.0.0.0", port))
            occupied.listen()

            with pytest.raises(RuntimeError, match=f"port {port} failed during startup"):
                server._start_api_server_raw(
                    registry=_CellRegistry([]), sources=_EMPTY_SOURCES, port=port, host="127.0.0.1"
                )


class TestDynamicCells:
    @pytest.mark.asyncio
    async def test_a_cell_that_appears_after_startup_is_served(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """Engine cells are reconciled in, so the server cannot snapshot them once at startup."""
        assert (await async_client.get("/api/v1/cells/rollout-engine-0")).status_code == 404

        rollout_handler.add("rollout-engine-0")

        assert (await async_client.get("/api/v1/cells/rollout-engine-0")).status_code == 200

    @pytest.mark.asyncio
    async def test_a_cell_that_disappears_stops_being_served(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """A removed cell must 404 instead of reporting stale status."""
        rollout_handler.add("rollout-engine-0")
        assert (await async_client.get("/api/v1/cells/rollout-engine-0")).status_code == 200

        del rollout_handler.cells["rollout-engine-0"]

        assert (await async_client.get("/api/v1/cells/rollout-engine-0")).status_code == 404


class TestInjectFault:
    @pytest.mark.asyncio
    async def test_injection_reaches_the_handler_of_that_cell(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """CI fault injection targets one cell by name."""
        rollout_handler.supports_inject_fault = True
        rollout_handler.add("rollout-engine-0")

        resp = await async_client.post(
            "/api/v1/cells/rollout-engine-0/inject-fault", json={"mode": "sigkill", "sub_index": 1}
        )

        assert resp.status_code == 200
        assert rollout_handler.injected == [("rollout-engine-0", FailureMode.SIGKILL, 1)]

    @pytest.mark.asyncio
    async def test_a_handler_without_injection_support_answers_bad_request(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """Not every kind of cell can be crashed on demand."""
        actor_handler.add("actor-0")

        resp = await async_client.post("/api/v1/cells/actor-0/inject-fault", json={"mode": "sigkill"})

        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_inject_fault_uses_zero_sub_index_by_default(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """The documented default targets worker zero, and a client omitting sub_index relies on it."""
        rollout_handler.supports_inject_fault = True
        rollout_handler.add("rollout-engine-0")

        resp = await async_client.post("/api/v1/cells/rollout-engine-0/inject-fault", json={"mode": "exit"})

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert rollout_handler.injected == [("rollout-engine-0", FailureMode.EXIT, 0)]

    @pytest.mark.asyncio
    async def test_inject_fault_rejects_missing_or_unknown_mode(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """An unrecognised failure mode must be refused by the schema rather than forwarded to the cell."""
        rollout_handler.supports_inject_fault = True
        rollout_handler.add("rollout-engine-0")

        missing = await async_client.post("/api/v1/cells/rollout-engine-0/inject-fault", json={})
        unknown = await async_client.post("/api/v1/cells/rollout-engine-0/inject-fault", json={"mode": "nuke"})

        assert (missing.status_code, unknown.status_code) == (422, 422)
        assert rollout_handler.injected == []

    @pytest.mark.asyncio
    async def test_an_injection_that_blows_up_returns_500_k8s_status(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """A crashed injection is not a bad request, and the CI harness needs the difference to fail the run."""
        rollout_handler.supports_inject_fault = True
        rollout_handler.inject_fault_error = RuntimeError("worker manager unreachable")
        rollout_handler.add("rollout-engine-0")

        resp = await async_client.post(
            "/api/v1/cells/rollout-engine-0/inject-fault", json={"mode": "sigkill", "sub_index": 1}
        )

        assert resp.status_code == 500
        assert resp.json() == {
            "apiVersion": "v1",
            "kind": "Status",
            "status": "Failure",
            "message": "Failed to inject fault into cell 'rollout-engine-0'",
            "reason": "InternalError",
            "code": 500,
        }


class TestGetFaultHookSources:
    @pytest.mark.asyncio
    async def test_the_trainer_cells_are_listed(self, async_client: httpx.AsyncClient) -> None:
        """A harness that cannot see a live trainer generation has nothing to arm a hook against."""
        resp = await async_client.get("/api/v1/fault-hook-sources")

        assert resp.status_code == 200
        assert [item["metadata"]["name"] for item in resp.json()["items"]] == [SOURCE_CELL_ID]

    @pytest.mark.asyncio
    async def test_a_deployment_without_a_trainer_answers_an_empty_list(
        self, registry: _CellRegistry, actor_handler: MockHandler
    ) -> None:
        """A run that drives no trainer has no source, and the poll must read that rather than fail."""
        actor_handler.add("actor-0")
        app = server._create_api_app(registry, _FaultHookSourceRegistry(handler=None, controllers=[]))

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/fault-hook-sources")

        assert resp.status_code == 200
        assert resp.json()["items"] == []


class TestArmFaultHook:
    @pytest.mark.asyncio
    async def test_the_request_reaches_the_trainer_that_owns_the_cell(
        self, source_controller: MockSourceController, async_client: httpx.AsyncClient
    ) -> None:
        """Arming names one incarnation, one hook, one worker and one request, and every part must survive."""
        resp = await async_client.post(
            f"/api/v1/cells/{SOURCE_CELL_ID}/arm-fault-hook",
            json={
                "expected_workers_hash": SOURCE_WORKERS_HASH,
                "hook": "weight_update.before_p2p_write",
                "mode": "sigkill",
                "sub_index": 2,
                "request_id": "req-1",
            },
        )

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert source_controller.armed == [
            dict(
                cell_id=SOURCE_CELL_ID,
                expected_workers_hash=SOURCE_WORKERS_HASH,
                hook="weight_update.before_p2p_write",
                mode="sigkill",
                sub_index=2,
                request_id="req-1",
                target="local",
                delay_ms=0,
            )
        ]

    @pytest.mark.asyncio
    async def test_a_remote_target_survives_the_route(
        self, source_controller: MockSourceController, async_client: httpx.AsyncClient
    ) -> None:
        """Whether the fault harms the trainer or the engine it is writing to is the request's own choice."""
        resp = await async_client.post(
            f"/api/v1/cells/{SOURCE_CELL_ID}/arm-fault-hook",
            json={
                "expected_workers_hash": SOURCE_WORKERS_HASH,
                "hook": "weight_update.after_p2p_submit",
                "mode": "sigkill",
                "request_id": "req-1",
                "target": "remote_inference_cell",
            },
        )

        assert resp.status_code == 200
        assert source_controller.armed[0]["target"] == "remote_inference_cell"

    @pytest.mark.asyncio
    async def test_arming_targets_the_armed_process_by_default(
        self, source_controller: MockSourceController, async_client: httpx.AsyncClient
    ) -> None:
        """A request that names no target must not be read as a licence to crash somebody else's cell."""
        resp = await async_client.post(
            f"/api/v1/cells/{SOURCE_CELL_ID}/arm-fault-hook",
            json={
                "expected_workers_hash": SOURCE_WORKERS_HASH,
                "hook": "weight_update.before_all_gather",
                "mode": "sigkill",
                "request_id": "req-1",
            },
        )

        assert resp.status_code == 200
        assert source_controller.armed[0]["target"] == "local"

    @pytest.mark.asyncio
    async def test_arming_uses_zero_sub_index_by_default(
        self, source_controller: MockSourceController, async_client: httpx.AsyncClient
    ) -> None:
        """A single-worker cell is named without a sub_index, and the documented default targets worker zero."""
        resp = await async_client.post(
            f"/api/v1/cells/{SOURCE_CELL_ID}/arm-fault-hook",
            json={
                "expected_workers_hash": SOURCE_WORKERS_HASH,
                "hook": "weight_update.before_all_gather",
                "mode": "exit",
                "request_id": "req-1",
            },
        )

        assert resp.status_code == 200
        assert source_controller.armed[0]["sub_index"] == 0

    @pytest.mark.asyncio
    async def test_an_unknown_hook_mode_or_field_is_refused_by_the_schema(
        self, source_controller: MockSourceController, async_client: httpx.AsyncClient
    ) -> None:
        """A request the trainer cannot honour must be refused here rather than reported as armed."""
        url = f"/api/v1/cells/{SOURCE_CELL_ID}/arm-fault-hook"
        armed = {
            "expected_workers_hash": SOURCE_WORKERS_HASH,
            "hook": "weight_update.before_all_gather",
            "mode": "sigkill",
            "request_id": "req-1",
        }

        unknown_hook = await async_client.post(url, json={**armed, "hook": "weight_update.before_typo"})
        unknown_mode = await async_client.post(url, json={**armed, "mode": "nuke"})
        unknown_target = await async_client.post(url, json={**armed, "target": "somebody_else"})
        no_request_id = await async_client.post(url, json={k: v for k, v in armed.items() if k != "request_id"})
        extra_field = await async_client.post(url, json={**armed, "hold_ms": 100})

        assert [
            resp.status_code for resp in (unknown_hook, unknown_mode, unknown_target, no_request_id, extra_field)
        ] == [422] * 5
        assert source_controller.armed == []

    @pytest.mark.asyncio
    async def test_an_arm_that_names_no_incarnation_is_refused_by_the_schema(
        self, source_controller: MockSourceController, async_client: httpx.AsyncClient
    ) -> None:
        """Without the hash the caller chose, the server would arm whatever generation happens to be current."""
        url = f"/api/v1/cells/{SOURCE_CELL_ID}/arm-fault-hook"
        armed = {"hook": "weight_update.before_all_gather", "mode": "sigkill", "request_id": "req-1"}

        missing = await async_client.post(url, json=armed)
        empty = await async_client.post(url, json={**armed, "expected_workers_hash": ""})
        null = await async_client.post(url, json={**armed, "expected_workers_hash": None})

        assert [resp.status_code for resp in (missing, empty, null)] == [422] * 3
        assert source_controller.armed == []

    @pytest.mark.asyncio
    async def test_a_delay_survives_the_route(
        self, source_controller: MockSourceController, async_client: httpx.AsyncClient
    ) -> None:
        """A delayed fault is a different scenario from an immediate one, so the wait cannot be dropped in transit."""
        resp = await async_client.post(
            f"/api/v1/cells/{SOURCE_CELL_ID}/arm-fault-hook",
            json={
                "expected_workers_hash": SOURCE_WORKERS_HASH,
                "hook": "weight_update.before_all_gather",
                "mode": "sigkill",
                "request_id": "req-1",
                "delay_ms": 250,
            },
        )

        assert resp.status_code == 200
        assert source_controller.armed[0]["delay_ms"] == 250

    @pytest.mark.asyncio
    async def test_arming_waits_for_nothing_by_default(
        self, source_controller: MockSourceController, async_client: httpx.AsyncClient
    ) -> None:
        """Every scenario written before delays existed asks for a fault that runs where the hook is reached."""
        resp = await async_client.post(
            f"/api/v1/cells/{SOURCE_CELL_ID}/arm-fault-hook",
            json={
                "expected_workers_hash": SOURCE_WORKERS_HASH,
                "hook": "weight_update.before_all_gather",
                "mode": "sigkill",
                "request_id": "req-1",
            },
        )

        assert resp.status_code == 200
        assert source_controller.armed[0]["delay_ms"] == 0

    @pytest.mark.asyncio
    async def test_a_delay_that_is_not_a_bounded_whole_number_of_milliseconds_is_refused(
        self, source_controller: MockSourceController, async_client: httpx.AsyncClient
    ) -> None:
        """A delay nobody can honour must fail the arm rather than be rounded into some other scenario."""
        url = f"/api/v1/cells/{SOURCE_CELL_ID}/arm-fault-hook"
        armed = {
            "expected_workers_hash": SOURCE_WORKERS_HASH,
            "hook": "weight_update.before_all_gather",
            "mode": "sigkill",
            "request_id": "req-1",
        }

        negative = await async_client.post(url, json={**armed, "delay_ms": -1})
        too_long = await async_client.post(url, json={**armed, "delay_ms": MAX_FAULT_HOOK_DELAY_MS + 1})
        fractional = await async_client.post(url, json={**armed, "delay_ms": 1.5})
        boolean = await async_client.post(url, json={**armed, "delay_ms": True})
        text = await async_client.post(url, json={**armed, "delay_ms": "100"})

        assert [resp.status_code for resp in (negative, too_long, fractional, boolean, text)] == [422] * 5
        assert source_controller.armed == []

    @pytest.mark.asyncio
    async def test_the_longest_delay_the_worker_accepts_survives_the_route(
        self, source_controller: MockSourceController, async_client: httpx.AsyncClient
    ) -> None:
        """The bound is inclusive, and a request at it must reach the worker rather than be refused here."""
        resp = await async_client.post(
            f"/api/v1/cells/{SOURCE_CELL_ID}/arm-fault-hook",
            json={
                "expected_workers_hash": SOURCE_WORKERS_HASH,
                "hook": "weight_update.before_all_gather",
                "mode": "sigkill",
                "request_id": "req-1",
                "delay_ms": MAX_FAULT_HOOK_DELAY_MS,
            },
        )

        assert resp.status_code == 200
        assert source_controller.armed[0]["delay_ms"] == MAX_FAULT_HOOK_DELAY_MS

    @pytest.mark.asyncio
    async def test_an_engine_cell_is_no_fault_hook_source(
        self, rollout_handler: MockHandler, source_controller: MockSourceController, async_client: httpx.AsyncClient
    ) -> None:
        """Hooks live in trainer processes, and a managed engine cell must not be answered as if it held one."""
        rollout_handler.add("rollout-engine-0")

        resp = await async_client.post(
            "/api/v1/cells/rollout-engine-0/arm-fault-hook",
            json={
                "expected_workers_hash": SOURCE_WORKERS_HASH,
                "hook": "weight_update.before_all_gather",
                "mode": "sigkill",
                "request_id": "req-1",
            },
        )

        assert resp.status_code == 404
        assert source_controller.armed == []

    @pytest.mark.asyncio
    async def test_a_trainer_that_refuses_answers_bad_request_with_its_reason(
        self, source_controller: MockSourceController, async_client: httpx.AsyncClient
    ) -> None:
        """A snapshot that went stale must be told so, not answered as if the fault had been armed."""
        source_controller.refused_because = "trainer-engine-actor-0 now runs pseudo-hash-9"

        resp = await async_client.post(
            f"/api/v1/cells/{SOURCE_CELL_ID}/arm-fault-hook",
            json={
                "expected_workers_hash": SOURCE_WORKERS_HASH,
                "hook": "weight_update.before_all_gather",
                "mode": "sigkill",
                "request_id": "req-1",
            },
        )

        assert resp.status_code == 400
        assert resp.json() == {
            "apiVersion": "v1",
            "kind": "Status",
            "status": "Failure",
            "message": "trainer-engine-actor-0 now runs pseudo-hash-9",
            "reason": "BadRequest",
            "code": 400,
        }

    @pytest.mark.asyncio
    async def test_an_arm_that_blows_up_returns_500_k8s_status(
        self, source_controller: MockSourceController, async_client: httpx.AsyncClient
    ) -> None:
        """A failed arm leaves nothing armed, and the harness needs a failure to stop waiting for the fault."""
        source_controller.arm_fault_hook_error = RuntimeError("the trainer controller is unreachable")

        resp = await async_client.post(
            f"/api/v1/cells/{SOURCE_CELL_ID}/arm-fault-hook",
            json={
                "expected_workers_hash": SOURCE_WORKERS_HASH,
                "hook": "weight_update.before_all_gather",
                "mode": "sigkill",
                "request_id": "req-1",
            },
        )

        assert resp.status_code == 500
        assert resp.json() == {
            "apiVersion": "v1",
            "kind": "Status",
            "status": "Failure",
            "message": "Failed to arm a fault hook in cell 'trainer-engine-actor-0'",
            "reason": "InternalError",
            "code": 500,
        }

    @pytest.mark.asyncio
    async def test_an_unknown_cell_is_not_found(self, async_client: httpx.AsyncClient) -> None:
        """A typo in the cell name must not be answered by arming some other cell."""
        resp = await async_client.post(
            "/api/v1/cells/trainer-engine-actor-9/arm-fault-hook",
            json={
                "expected_workers_hash": SOURCE_WORKERS_HASH,
                "hook": "weight_update.before_all_gather",
                "mode": "sigkill",
                "request_id": "req-1",
            },
        )

        assert resp.status_code == 404


class TestStartAndWaitThread:
    def test_it_returns_once_the_thread_reports_ready(self):
        """The caller may only proceed after the thing it started is actually usable."""
        ready = threading.Event()

        thread = server._start_and_wait_thread(
            target=ready.set, is_ready=ready.is_set, description="probe", timeout_seconds=5.0
        )

        assert ready.is_set()
        assert isinstance(thread, threading.Thread)

    def test_a_failure_on_the_thread_reaches_the_caller(self):
        """A daemon thread that dies alone is invisible, which is how a lost port went unnoticed."""

        def _boom() -> None:
            raise ValueError("could not start")

        with pytest.raises(RuntimeError, match="probe failed during startup") as excinfo:
            server._start_and_wait_thread(
                target=_boom, is_ready=lambda: False, description="probe", timeout_seconds=5.0
            )

        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_a_thread_that_exits_without_becoming_ready_fails_the_caller(self):
        """Returning quietly is its own failure: nothing is serving afterwards."""
        with pytest.raises(RuntimeError, match="probe exited during startup"):
            server._start_and_wait_thread(
                target=lambda: None, is_ready=lambda: False, description="probe", timeout_seconds=5.0
            )

    def test_a_thread_that_never_becomes_ready_times_out(self):
        """A wedged startup must not block the caller forever."""
        stop = threading.Event()

        try:
            with pytest.raises(TimeoutError, match="probe did not finish startup"):
                server._start_and_wait_thread(
                    target=stop.wait, is_ready=lambda: False, description="probe", timeout_seconds=0.2
                )
        finally:
            stop.set()

    def test_a_ready_thread_is_not_judged_by_a_later_failure(self):
        """Readiness wins the race: a server that starts and later dies is the caller's problem, not startup's."""
        ready = threading.Event()

        def _ready_then_raise() -> None:
            ready.set()
            raise ValueError("died after serving")

        server._start_and_wait_thread(
            target=_ready_then_raise, is_ready=ready.is_set, description="probe", timeout_seconds=5.0
        )


class TestRequestValidation:
    @pytest.mark.asyncio
    async def test_invalid_write_bodies_return_422_without_side_effects(
        self, actor_handler: MockHandler, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """Unknown fields must be refused outright, so a typo cannot silently half-apply a write."""
        cell = actor_handler.add("actor-0", phase="Running")
        rollout_handler.supports_inject_fault = True
        rollout_handler.add("rollout-engine-0")

        patch_resp = await async_client.patch(
            "/api/v1/cells/actor-0", json={"spec": {"suspend": True, "gracePeriod": 5}}
        )
        inject_resp = await async_client.post(
            "/api/v1/cells/rollout-engine-0/inject-fault", json={"mode": "sigkill", "subIndex": 1}
        )

        assert (patch_resp.status_code, inject_resp.status_code) == (422, 422)
        assert (cell.suspend_calls, cell.resume_calls) == (0, 0)
        assert rollout_handler.injected == []


class TestSeveralTrainers:
    def test_every_trainer_pool_is_listed_under_the_actor_cell_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A run of several trainer models heals none of them if the api server only knows the legacy actor pool."""
        registries: list[_CellRegistry] = []
        monkeypatch.setattr(server, "compute_trainer_pool_id", lambda trainer_id: f"trainer-engine-{trainer_id}")
        monkeypatch.setattr(server, "compute_engine_pool_ids", lambda args: ["engine"])
        monkeypatch.setattr(
            server, "_start_api_server_raw", lambda *, registry, sources, port, host: registries.append(registry)
        )

        server.start_api_server(
            args=SimpleNamespace(),
            trainer_models={
                "a-actor": make_mock_controller([], pool_id="trainer-engine-a-actor"),
                "b-actor": make_mock_controller([], pool_id="trainer-engine-b-actor"),
            },
            inference_controller=MockInferenceController(),
            port=1234,
            ft_components=["train"],
            cell_operations=object(),
        )

        [handler] = registries[0]._handlers
        assert handler._pool_ids == ["trainer-engine-a-actor", "trainer-engine-b-actor"]

    @pytest.mark.asyncio
    async def test_the_cell_statuses_of_every_trainer_are_merged(self) -> None:
        """Each trainer answers only about its own cells, and a cell missing from the merge reads as pending."""
        first = make_mock_controller([MockTrainerCell()], pool_id="trainer-engine-a-actor")
        second = make_mock_controller([MockTrainerCell()], pool_id="trainer-engine-b-actor")

        statuses = await _CellHandler(
            cell_type="actor", operations=object(), controllers=[first, second], pool_ids=[]
        )._get_cell_statuses()

        assert sorted(statuses) == ["trainer-engine-a-actor-0", "trainer-engine-b-actor-0"]


class TestOperationsSelection:
    def test_every_handler_gets_the_operations_of_the_process_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under Kubernetes the api server must act on pods, not on a Ray worker manager."""
        operations = object()
        registries: list[_CellRegistry] = []
        monkeypatch.setattr(server, "compute_trainer_pool_id", lambda role: f"trainer-{role}")
        monkeypatch.setattr(server, "compute_engine_pool_ids", lambda args: ["engine"])
        monkeypatch.setattr(
            server, "_start_api_server_raw", lambda *, registry, sources, port, host: registries.append(registry)
        )

        server.start_api_server(
            args=SimpleNamespace(),
            trainer_models={"actor": make_mock_controller([])},
            inference_controller=MockInferenceController(),
            port=1234,
            ft_components=["train", "rollout"],
            cell_operations=operations,
        )

        handlers = registries[0]._handlers
        assert len(handlers) == 2
        assert [handler._operations for handler in handlers] == [operations, operations]

    def test_the_api_server_names_no_ray_worker_manager(self) -> None:
        """Naming it would make the Kubernetes assembly pointless for the half that heals cells."""
        source = Path(server.__file__).read_text()

        assert "RayWorkerManager" not in source
        assert "cell_operations: BaseCellOperations" in source
