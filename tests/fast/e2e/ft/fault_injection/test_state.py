import pytest
from tests.e2e.ft.conftest_ft.fault_injection import state
from tests.fast.e2e.ft.fault_injection.utils import (
    RUNNING_NOT_SERVING,
    SERVING,
    cell,
    initialized_trainer_cell,
    paused_cell,
    staged,
    uninitialized_trainer_cell,
)

from miles.ray.rollout.inference_controller import HEALTH_PAUSED_FOR_OFFLOAD, HEALTH_PAUSED_FOR_WEIGHT_UPDATE
from miles.utils.ft_utils.api_server.models import TriState


def test_cell_is_alive_true_only_when_healthy_condition_is_true() -> None:
    """cell_is_alive reflects the Healthy condition status."""
    assert state.cell_is_alive(cell("c", healthy=True))
    assert not state.cell_is_alive(cell("c", healthy=False))


def test_cell_is_alive_false_when_no_healthy_condition_present() -> None:
    """A cell with no Healthy condition is not considered alive."""
    assert not state.cell_is_alive({"metadata": {"name": "c"}, "status": {"conditions": []}})


def test_a_running_cell_that_is_not_in_the_router_is_not_serving() -> None:
    """The api server renders PendingWeights and Serving alike, so the Serving condition must split them."""
    assert state.compute_observed_cell_state(staged("c", RUNNING_NOT_SERVING)) is RUNNING_NOT_SERVING
    assert state.compute_observed_cell_state(staged("c", SERVING)) is SERVING


def test_cell_is_allocated_reflects_the_allocated_condition() -> None:
    """A suspended cell holds no GPU, and injecting into it exercises nothing."""
    assert state.cell_is_allocated(cell("c", healthy=True))
    assert not state.cell_is_allocated(cell("c", healthy=True, allocated=False, phase="Suspended"))


def test_a_cell_without_a_worker_generation_is_rejected() -> None:
    """A cell with no identity cannot be told apart from its own replacement."""
    with pytest.raises(AssertionError, match="no worker generation"):
        state.cell_workers_hash({"metadata": {"name": "c"}, "status": {"workers_hash": ""}})


class TestCellInfoIsInService:
    """The one predicate that says a cell is answering right now, per cell kind."""

    def test_a_serving_engine_is_in_service(self) -> None:
        """The ordinary case a rollout soak targets."""
        assert state.cell_info_is_in_service(state.compute_cell_info(staged("rollout-engine-0", SERVING)))

    def test_an_engine_outside_the_router_is_not(self) -> None:
        """It reads Healthy long before it can answer a request."""
        assert not state.cell_info_is_in_service(
            state.compute_cell_info(staged("rollout-engine-0", RUNNING_NOT_SERVING))
        )

    def test_a_trainer_cell_needs_no_serving_condition(self) -> None:
        """Trainer cells never carry one, so requiring it would stop every trainer soak."""
        assert state.cell_info_is_in_service(state.compute_cell_info(cell("actor-0", healthy=True)))

    def test_a_de_allocated_cell_is_not_in_service(self) -> None:
        """Its workers are gone, whatever its last Healthy reading said."""
        assert not state.cell_info_is_in_service(
            state.compute_cell_info(cell("actor-0", healthy=True, allocated=False, phase="Suspended"))
        )


class TestCellInfoIsPausedForWeightUpdate:
    """A paused probe reads Unknown, and only an explicit weight-update reason may carry a qualification."""

    def test_a_weight_update_pause_of_a_serving_engine_qualifies(self) -> None:
        """This is the window the weight-update fault tolerance has to survive a crash in."""
        info = state.compute_cell_info(paused_cell("rollout-engine-0", reason=HEALTH_PAUSED_FOR_WEIGHT_UPDATE))

        assert state.cell_info_is_paused_for_weight_update(info)

    def test_an_offload_pause_does_not(self) -> None:
        """Colocate offload hands the GPUs to the trainer, and the engine there is not in service."""
        info = state.compute_cell_info(paused_cell("rollout-engine-0", reason=HEALTH_PAUSED_FOR_OFFLOAD))

        assert not state.cell_info_is_paused_for_weight_update(info)

    def test_an_unexplained_unknown_does_not(self) -> None:
        """A cell nobody has probed yet is not evidence of a live engine."""
        info = state.compute_cell_info(cell("rollout-engine-0", healthy=False, cell_type="rollout"))

        assert not state.cell_info_is_paused_for_weight_update(info)

    def test_a_paused_engine_that_left_the_router_does_not(self) -> None:
        """A pause is only an excuse for probing, not for having stopped answering requests."""
        info = state.compute_cell_info(paused_cell("rollout-engine-0", reason=HEALTH_PAUSED_FOR_WEIGHT_UPDATE))
        unrouted = info.model_copy(update={"state": RUNNING_NOT_SERVING})

        assert not state.cell_info_is_paused_for_weight_update(unrouted)


class TestATrainerCellThatHasNotInitialized:
    def test_it_is_not_in_service(self) -> None:
        """It reports Healthy=True the moment it is allocated, long before it can take a rank in the run."""
        info = state.compute_cell_info(uninitialized_trainer_cell("actor-0"))

        assert not state.cell_info_is_in_service(info)

    def test_the_same_generation_is_in_service_once_it_has_initialized(self) -> None:
        """The qualification must arrive when the cell really joins, not be withheld from the process forever."""
        info = state.compute_cell_info(initialized_trainer_cell("actor-0"))

        assert state.cell_info_is_in_service(info)

    def test_an_initialized_cell_that_failed_its_probe_is_not_in_service(self) -> None:
        """Initialization is necessary, not sufficient: a cell that fails its health check is still no spare."""
        info = state.compute_cell_info(initialized_trainer_cell("actor-0", health_status=TriState.FALSE))

        assert not state.cell_info_is_in_service(info)

    def test_an_initialized_cell_that_has_not_been_probed_yet_is_not_in_service(self) -> None:
        """A freshly started checker reads Unknown, which is the absence of evidence, not evidence of health."""
        info = state.compute_cell_info(initialized_trainer_cell("actor-0", health_status=TriState.UNKNOWN))

        assert not state.cell_info_is_in_service(info)
