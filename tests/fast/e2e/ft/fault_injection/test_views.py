from tests.e2e.ft.conftest_ft.fault_injection import state, views
from tests.fast.e2e.ft.fault_injection.utils import (
    PENDING,
    RUNNING_NOT_SERVING,
    SERVING,
    SUSPENDED,
    initialized_trainer_cell,
    log_of,
    note_injected,
    staged,
    uninitialized_trainer_cell,
)

from miles.ray.rollout.inference_controller import HEALTH_PAUSED_FOR_OFFLOAD, HEALTH_PAUSED_FOR_WEIGHT_UPDATE


def test_observed_states_record_only_transitions() -> None:
    """Polling runs for the life of the training run, so repeats must not accumulate."""
    log = log_of([SERVING, SERVING, SUSPENDED, SUSPENDED, SERVING])

    assert views.compute_states_of_cell_name(log.events) == {"rollout-engine-0": [SERVING, SUSPENDED, SERVING]}


def test_a_serve_after_the_last_injection_clears_the_cell() -> None:
    """This is the recovery the soak asserts: the injected engine ends up serving again."""
    log = log_of([SERVING, PENDING, SERVING], inject_before={1: 1})

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout") == {}


def test_a_completely_missed_down_window_is_not_an_offence() -> None:
    """A replacement can finish between two polls, and the witness must not demand the down it never saw."""
    log = log_of([SERVING, SERVING, SERVING], inject_before={1: 1})

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout") == {}


def test_a_replacement_that_never_reaches_the_router_is_an_offence() -> None:
    """Regression: a relaunched engine stuck at PendingWeights also reads Running, and must not pass."""
    log = log_of([SERVING, PENDING, RUNNING_NOT_SERVING], inject_before={1: 1})

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout") == {
        "rollout-engine-0": [SERVING.value, PENDING.value, RUNNING_NOT_SERVING.value]
    }


def test_a_cell_that_was_never_injected_owes_no_serve() -> None:
    """Otherwise a run that injected nothing would still fail the witness on ordinary churn."""
    log = log_of([PENDING, SERVING])

    assert views.compute_num_injections(log.events, cell_type="rollout") == 0
    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout") == {}


def test_a_serve_that_predates_the_last_injection_does_not_discharge_it() -> None:
    """Otherwise the last crash of a soak is paid for by the recovery of the crash before it."""
    log = log_of([SERVING, PENDING, SERVING, PENDING], inject_before={1: 1, 3: 1})

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout") == {
        "rollout-engine-0": [SERVING.value, PENDING.value, SERVING.value, PENDING.value]
    }


def test_only_the_last_injection_of_a_cell_needs_a_serve_after_it() -> None:
    """Only one harm is outstanding run-wide, so one final serve of the newest generation settles the cell."""
    log = log_of([SERVING, PENDING, SERVING, SERVING], inject_before={1: 1, 3: 1})

    assert views.compute_num_injections(log.events, cell_type="rollout") == 2
    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout") == {}


def test_offences_of_another_cell_kind_do_not_count() -> None:
    """A mixed soak injects both kinds, and the rollout view must only see rollout cells."""
    log = state.EventLog()
    log.observe([staged("actor-0", SERVING, cell_type="actor")])
    note_injected(log, "actor-0", workers_hash="generation-0")
    log.observe([staged("actor-0", PENDING, cell_type="actor")])

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout") == {}
    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="actor") == {
        "actor-0": [SERVING.value, PENDING.value]
    }


def test_a_siblings_serve_cannot_clear_the_injected_cells_debt() -> None:
    """Only the victim's own fresh Serving reading proves its recovery."""
    log = state.EventLog()
    log.observe([staged("rollout-engine-0", SERVING), staged("rollout-engine-1", SERVING)])
    note_injected(log, "rollout-engine-0", workers_hash="generation-0")
    log.observe([staged("rollout-engine-0", PENDING), staged("rollout-engine-1", SERVING)])

    offenders = views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout")
    assert set(offenders) == {"rollout-engine-0"}


class TestIdentityBasedRecovery:
    """The api server keeps reporting a just-killed cell Serving, so only a new generation proves recovery."""

    def test_a_serve_by_the_generation_that_was_killed_does_not_clear_the_cell(self) -> None:
        """This is the stale reading the old time-based grace period was guessing around."""
        events = [
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0"),
            _injection("rollout-engine-0", workers_hash="generation-0"),
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0"),
        ]

        assert views.compute_cells_not_serving_after_injection(events, cell_type="rollout") == {
            "rollout-engine-0": [SERVING.value]
        }

    def test_a_serve_by_the_replacement_generation_clears_the_cell(self) -> None:
        """A different generation answering requests is the only direct evidence that the heal completed."""
        events = [
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0"),
            _injection("rollout-engine-0", workers_hash="generation-0"),
            _observation("rollout-engine-0", SERVING, workers_hash="generation-1"),
        ]

        assert views.compute_cells_not_serving_after_injection(events, cell_type="rollout") == {}

    def test_a_replacement_that_is_not_serving_does_not_clear_the_cell(self) -> None:
        """A relaunched engine reads Running long before it can answer, and a heal is only half done there."""
        events = [
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0"),
            _injection("rollout-engine-0", workers_hash="generation-0"),
            _observation("rollout-engine-0", RUNNING_NOT_SERVING, workers_hash="generation-1"),
        ]

        assert views.compute_cells_not_serving_after_injection(events, cell_type="rollout") == {
            "rollout-engine-0": [SERVING.value, RUNNING_NOT_SERVING.value]
        }

    def test_a_serve_reported_by_a_cell_that_is_not_healthy_does_not_clear_it(self) -> None:
        """A cell that died without being deregistered keeps reporting Serving, and heals nothing."""
        events = [
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0"),
            _injection("rollout-engine-0", workers_hash="generation-0"),
            _observation("rollout-engine-0", SERVING, workers_hash="generation-1", alive=False),
        ]

        assert views.compute_cells_not_serving_after_injection(events, cell_type="rollout") == {
            "rollout-engine-0": [SERVING.value]
        }

    def test_a_second_kill_of_the_replacement_reopens_the_debt(self) -> None:
        """Otherwise the recovery of the previous crash would pay for the one that is still down."""
        events = [
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0"),
            _injection("rollout-engine-0", workers_hash="generation-0"),
            _observation("rollout-engine-0", SERVING, workers_hash="generation-1"),
            _injection("rollout-engine-0", workers_hash="generation-1"),
            _observation("rollout-engine-0", SERVING, workers_hash="generation-1"),
        ]

        assert views.compute_cells_not_serving_after_injection(events, cell_type="rollout") == {
            "rollout-engine-0": [SERVING.value]
        }


def _observation(
    name: str,
    cell_state: state.ObservedCellState,
    *,
    workers_hash: str = "generation-0",
    alive: bool = True,
    allocated: bool = True,
    health_reason: str | None = None,
    cell_type: str = "rollout",
) -> state.ObservationsEvent:
    return state.ObservationsEvent(
        cell_infos={
            name: state.CellInfo(
                cell_type=cell_type,
                state=cell_state,
                alive=alive,
                allocated=allocated,
                health_reason=health_reason,
                workers_hash=workers_hash,
            )
        }
    )


def _injection(name: str, *, workers_hash: str = "generation-0") -> state.InjectionEvent:
    return state.InjectionEvent(
        cell_name=name,
        form_name="inject_fault:sigkill",
        succeeded=True,
        harmed=True,
        workers_hash=workers_hash,
    )


class TestWhichInjectionsCount:
    def test_a_form_that_left_its_cell_running_is_not_a_crash_anything_has_to_heal(self) -> None:
        """A hot restart replaces the orchestration script and harms no cell, so no cell owes a recovery."""
        log = _log_of_one_injection(form_name="hot_restart", succeeded=True, harmed=False)

        assert views.compute_num_injections(log.events, cell_type="rollout") == 0
        assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout") == {}

    def test_a_form_that_left_its_cell_running_is_still_a_draw_that_fired(self) -> None:
        """A soak counting what it actually did to the run has to see it, and asks for it by name."""
        log = _log_of_one_injection(form_name="hot_restart", succeeded=True, harmed=False)

        assert views.compute_num_injections(log.events, cell_type="rollout", harmed_only=False) == 1

    def test_a_form_that_crashed_its_cell_counts_as_both(self) -> None:
        """The crash forms every floor assertion was written for must go on counting as they did."""
        log = _log_of_one_injection(form_name="crash_pod", succeeded=True, harmed=True)

        assert views.compute_num_injections(log.events, cell_type="rollout") == 1
        assert views.compute_num_injections(log.events, cell_type="rollout", harmed_only=False) == 1

    def test_a_draw_that_never_landed_counts_as_neither(self) -> None:
        """An attempt the cluster refused did nothing to the run, whatever the form would have done."""
        log = _log_of_one_injection(form_name="hot_restart", succeeded=False, harmed=False)

        assert views.compute_num_injections(log.events, cell_type="rollout", harmed_only=False) == 0

    def test_the_successes_of_one_form_are_counted_apart_from_another(self) -> None:
        """A mixed soak draws several forms, and each one's own assertions count only its own draws."""
        log = state.EventLog()
        log.observe([staged("rollout-engine-0", SERVING)])
        log.note_injection_attempt(
            cell_name="rollout-engine-0",
            form_name="hot_restart",
            succeeded=True,
            harmed=False,
            workers_hash="generation-0",
        )
        log.note_injection_attempt(
            cell_name="rollout-engine-0",
            form_name="crash_pod",
            succeeded=True,
            harmed=True,
            workers_hash="generation-0",
        )
        log.note_injection_attempt(
            cell_name="rollout-engine-0",
            form_name="hot_restart",
            succeeded=False,
            harmed=False,
            workers_hash="generation-0",
        )

        assert views.compute_num_successful_injections_of_form(log.events, form_name="hot_restart") == 1
        assert views.compute_num_successful_injections_of_form(log.events, form_name="crash_pod") == 1


def _log_of_one_injection(*, form_name: str, succeeded: bool, harmed: bool) -> state.EventLog:
    log = state.EventLog()
    log.observe([staged("rollout-engine-0", SERVING)])
    log.note_injection_attempt(
        cell_name="rollout-engine-0",
        form_name=form_name,
        succeeded=succeeded,
        harmed=harmed,
        workers_hash="generation-0",
    )
    log.observe([staged("rollout-engine-0", SERVING)])
    return log


class TestComputeInjectionEligibility:
    """Only a generation a real Healthy observation vouched for may be harmed."""

    def test_a_real_healthy_observation_qualifies_its_generation_at_once(self):
        """There is no waiting window: the first honest reading is enough."""
        events = [_observation("rollout-engine-0", SERVING, workers_hash="generation-0")]

        assert views.compute_injection_eligibility(events) == {("rollout-engine-0", "generation-0")}

    def test_a_generation_first_seen_paused_is_not_a_spare(self):
        """A replacement that appears mid weight update was never probed, so it proves no live engine."""
        events = [
            _observation(
                "rollout-engine-0",
                SERVING,
                workers_hash="generation-1",
                alive=False,
                health_reason=HEALTH_PAUSED_FOR_WEIGHT_UPDATE,
            )
        ]

        assert views.compute_injection_eligibility(events) == set()

    def test_a_pause_carries_the_qualification_of_the_same_generation(self):
        """A verified engine stays a legal target while the controller pauses its probing to push weights."""
        events = [
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0"),
            _observation(
                "rollout-engine-0",
                SERVING,
                workers_hash="generation-0",
                alive=False,
                health_reason=HEALTH_PAUSED_FOR_WEIGHT_UPDATE,
            ),
        ]

        assert views.compute_injection_eligibility(events) == {("rollout-engine-0", "generation-0")}

    def test_a_new_generation_does_not_inherit_the_old_one_qualification(self):
        """The replacement is a different process, and the killed one's health says nothing about it."""
        events = [
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0"),
            _observation(
                "rollout-engine-0",
                SERVING,
                workers_hash="generation-1",
                alive=False,
                health_reason=HEALTH_PAUSED_FOR_WEIGHT_UPDATE,
            ),
        ]

        assert views.compute_injection_eligibility(events) == set()

    def test_an_explicitly_unhealthy_reading_drops_the_qualification(self):
        """A cell that failed its probe is not a spare, whatever it read a poll earlier."""
        events = [
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0"),
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0", alive=False),
        ]

        assert views.compute_injection_eligibility(events) == set()

    def test_a_pause_cannot_re_qualify_a_generation_that_read_unhealthy(self):
        """Otherwise the next weight update would launder a cell the health checker had already failed."""
        events = [
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0"),
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0", alive=False),
            _observation(
                "rollout-engine-0",
                SERVING,
                workers_hash="generation-0",
                alive=False,
                health_reason=HEALTH_PAUSED_FOR_WEIGHT_UPDATE,
            ),
        ]

        assert views.compute_injection_eligibility(events) == set()

    def test_a_real_healthy_reading_re_qualifies_a_generation_that_had_lapsed(self):
        """A cell that flapped and genuinely came back is a legal target again."""
        events = [
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0"),
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0", alive=False),
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0"),
        ]

        assert views.compute_injection_eligibility(events) == {("rollout-engine-0", "generation-0")}

    def test_a_cell_that_vanished_loses_its_qualification(self):
        """A deleted pod drops out of the listing entirely instead of reading unhealthy."""
        events = [
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0"),
            state.ObservationsEvent(cell_infos={}),
        ]

        assert views.compute_injection_eligibility(events) == set()

    def test_a_de_allocated_cell_is_not_qualified(self):
        """A suspended cell holds no GPU, so crashing it exercises nothing."""
        events = [_observation("rollout-engine-0", SUSPENDED, workers_hash="generation-0", allocated=False)]

        assert views.compute_injection_eligibility(events) == set()

    def test_an_engine_that_is_not_in_the_router_is_not_qualified(self):
        """A relaunched engine reads Healthy long before it can answer a request."""
        events = [_observation("rollout-engine-0", RUNNING_NOT_SERVING, workers_hash="generation-0")]

        assert views.compute_injection_eligibility(events) == set()

    def test_a_trainer_cell_is_qualified_without_a_serving_condition(self):
        """Trainer cells never carry one, so requiring it would stop every trainer soak."""
        events = [_observation("actor-0", RUNNING_NOT_SERVING, workers_hash="generation-0", cell_type="actor")]

        assert views.compute_injection_eligibility(events) == {("actor-0", "generation-0")}

    def test_a_trainer_cell_paused_for_a_weight_update_keeps_its_qualification(self):
        """Its probing is paused by the same window, and it carries no Serving condition to check."""
        events = [
            _observation("actor-0", RUNNING_NOT_SERVING, workers_hash="generation-0", cell_type="actor"),
            _observation(
                "actor-0",
                RUNNING_NOT_SERVING,
                workers_hash="generation-0",
                cell_type="actor",
                alive=False,
                health_reason=HEALTH_PAUSED_FOR_WEIGHT_UPDATE,
            ),
        ]

        assert views.compute_injection_eligibility(events) == {("actor-0", "generation-0")}

    def test_an_offload_pause_drops_the_qualification(self):
        """Colocate offload hands the GPUs to the trainer, and the engine there is not in service."""
        events = [
            _observation("rollout-engine-0", SERVING, workers_hash="generation-0"),
            _observation(
                "rollout-engine-0",
                SERVING,
                workers_hash="generation-0",
                alive=False,
                health_reason=HEALTH_PAUSED_FOR_OFFLOAD,
            ),
        ]

        assert views.compute_injection_eligibility(events) == set()


class TestPendingInjectionsAcrossCellKinds:
    """A trainer cell carries no Serving condition, so its debt must not be judged by one."""

    def test_a_trainer_replacement_clears_its_debt_without_ever_serving(self):
        """Requiring Serving would leave every trainer injection outstanding for the rest of the run."""
        events = [
            _observation("actor-0", RUNNING_NOT_SERVING, workers_hash="generation-0", cell_type="actor"),
            state.InjectionEvent(
                cell_name="actor-0",
                form_name="inject_fault:sigkill",
                succeeded=True,
                harmed=True,
                workers_hash="generation-0",
            ),
            _observation("actor-0", RUNNING_NOT_SERVING, workers_hash="generation-1", cell_type="actor"),
        ]

        assert views.compute_pending_injections(events) == []

    def test_a_trainer_replacement_that_is_not_healthy_still_owes_a_recovery(self):
        """A relaunched trainer that fails its probe has not recovered, whatever its generation says."""
        events = [
            _observation("actor-0", RUNNING_NOT_SERVING, workers_hash="generation-0", cell_type="actor"),
            state.InjectionEvent(
                cell_name="actor-0",
                form_name="inject_fault:sigkill",
                succeeded=True,
                harmed=True,
                workers_hash="generation-0",
            ),
            _observation("actor-0", RUNNING_NOT_SERVING, workers_hash="generation-1", cell_type="actor", alive=False),
        ]

        assert [entry.cell_name for entry in views.compute_pending_injections(events)] == ["actor-0"]


class TestATrainerReplacementThatHasNotInitialized:
    def test_it_does_not_qualify_as_a_spare(self) -> None:
        """The api server reports it Healthy on allocation, and treating that as a live replica is the whole bug."""
        log = state.EventLog()
        log.observe([uninitialized_trainer_cell("actor-0", workers_hash="generation-1")])

        assert views.compute_injection_eligibility(log.events) == set()

    def test_it_qualifies_once_that_same_generation_has_initialized(self) -> None:
        """A cell that really joined the run is a legal target again, under the generation it joined as."""
        log = state.EventLog()
        log.observe([uninitialized_trainer_cell("actor-0", workers_hash="generation-1")])
        log.observe([initialized_trainer_cell("actor-0", workers_hash="generation-1")])

        assert views.compute_injection_eligibility(log.events) == {("actor-0", "generation-1")}

    def test_it_does_not_clear_the_debt_of_the_cell_it_replaces(self) -> None:
        """Clearing it would let the next draw crash the one trainer that has actually initialized."""
        log = state.EventLog()
        log.observe([initialized_trainer_cell("actor-0", workers_hash="generation-0")])
        note_injected(log, "actor-0", workers_hash="generation-0")
        log.observe([uninitialized_trainer_cell("actor-0", workers_hash="generation-1")])

        assert [entry.cell_name for entry in views.compute_pending_injections(log.events)] == ["actor-0"]

    def test_the_debt_clears_once_the_replacement_has_initialized(self) -> None:
        """The recovery really did complete, and the soak has to keep injecting after it."""
        log = state.EventLog()
        log.observe([initialized_trainer_cell("actor-0", workers_hash="generation-0")])
        note_injected(log, "actor-0", workers_hash="generation-0")
        log.observe([uninitialized_trainer_cell("actor-0", workers_hash="generation-1")])
        log.observe([initialized_trainer_cell("actor-0", workers_hash="generation-1")])

        assert views.compute_pending_injections(log.events) == []
