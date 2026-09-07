from __future__ import annotations

import pytest

from miles.utils.ft_utils.api_server.fault_hook_sources import FaultHookSourceNotFoundError, _FaultHookSourceRegistry
from miles.utils.test_utils.fault_hooks import FaultHookName, FaultHookTarget
from miles.utils.test_utils.fault_injector import FailureMode

from .conftest import SOURCE_CELL_ID, SOURCE_WORKERS_HASH, MockHandler, MockSourceController, make_source_status

_OTHER_CELL_ID = "trainer-engine-critic-0"


async def _arm(registry: _FaultHookSourceRegistry, *, cell_id: str = SOURCE_CELL_ID) -> None:
    await registry.arm_fault_hook(
        cell_id,
        expected_workers_hash=SOURCE_WORKERS_HASH,
        hook=FaultHookName.WEIGHT_UPDATE_BEFORE_P2P_WRITE,
        mode=FailureMode.SIGKILL,
        sub_index=2,
        request_id="req-1",
        target=FaultHookTarget.REMOTE_INFERENCE_CELL,
    )


class TestListingTheSources:
    @pytest.mark.asyncio
    async def test_the_trainer_cells_are_the_sources(
        self, source_handler: MockHandler, source_controller: MockSourceController
    ) -> None:
        """The soak picks its trigger from this list, so it must carry the trainer cells themselves."""
        registry = _FaultHookSourceRegistry(handler=source_handler, controllers=[source_controller])

        assert [cell.metadata.name for cell in await registry.list_cells()] == [SOURCE_CELL_ID]

    @pytest.mark.asyncio
    async def test_a_deployment_with_no_trainer_lists_nothing(self) -> None:
        """A run that drives no trainer must answer an empty list rather than fail the poll."""
        registry = _FaultHookSourceRegistry(handler=None, controllers=[])

        assert await registry.list_cells() == []


class TestRoutingAnArm:
    @pytest.mark.asyncio
    async def test_the_request_reaches_the_trainer_that_owns_the_cell(
        self, source_handler: MockHandler, source_controller: MockSourceController
    ) -> None:
        """Every field of the request has to survive the route, or the fault armed is not the one asked for."""
        registry = _FaultHookSourceRegistry(handler=source_handler, controllers=[source_controller])

        await _arm(registry)

        assert source_controller.armed == [
            dict(
                cell_id=SOURCE_CELL_ID,
                expected_workers_hash=SOURCE_WORKERS_HASH,
                hook="weight_update.before_p2p_write",
                mode="sigkill",
                sub_index=2,
                request_id="req-1",
                target="remote_inference_cell",
            )
        ]

    @pytest.mark.asyncio
    async def test_only_the_trainer_that_answers_for_the_cell_is_asked(self, source_handler: MockHandler) -> None:
        """A run of several trainer models must not have its hook armed in whichever one was listed first."""
        stranger = MockSourceController({_OTHER_CELL_ID: make_source_status()})
        owner = MockSourceController({SOURCE_CELL_ID: make_source_status()})
        registry = _FaultHookSourceRegistry(handler=source_handler, controllers=[stranger, owner])

        await _arm(registry)

        assert stranger.armed == []
        assert [entry["cell_id"] for entry in owner.armed] == [SOURCE_CELL_ID]

    @pytest.mark.asyncio
    async def test_a_cell_no_trainer_answers_for_is_not_found(self, source_handler: MockHandler) -> None:
        """Arming some other trainer's cell instead would harm a process the caller never named."""
        registry = _FaultHookSourceRegistry(
            handler=source_handler, controllers=[MockSourceController({_OTHER_CELL_ID: make_source_status()})]
        )

        with pytest.raises(FaultHookSourceNotFoundError):
            await _arm(registry)

    @pytest.mark.asyncio
    async def test_the_refusal_of_the_trainer_is_handed_back(
        self, source_handler: MockHandler, source_controller: MockSourceController
    ) -> None:
        """The reason the trainer refused is the only thing that tells the caller its snapshot went stale."""
        source_controller.refused_because = "it now runs another incarnation"
        registry = _FaultHookSourceRegistry(handler=source_handler, controllers=[source_controller])

        report = await registry.arm_fault_hook(
            SOURCE_CELL_ID,
            expected_workers_hash=SOURCE_WORKERS_HASH,
            hook=FaultHookName.WEIGHT_UPDATE_BEFORE_P2P_WRITE,
            mode=FailureMode.SIGKILL,
            sub_index=0,
            request_id="req-1",
            target=FaultHookTarget.LOCAL,
        )

        assert report.refused_because == "it now runs another incarnation"
