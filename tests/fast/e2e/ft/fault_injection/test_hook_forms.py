import contextlib
import random
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tests.e2e.ft.conftest_ft.fault_injection import hook_forms
from tests.e2e.ft.conftest_ft.fault_injection.core import (
    UnresolvedInjectionError,
    _raise_if_an_unknown_injection_is_stuck,
    run_fault_injection_loop,
)
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import InjectFaultForm
from tests.e2e.ft.conftest_ft.fault_injection.hook_forms import (
    HookFaultContext,
    HookFireCollector,
    LocalHookFaultForm,
    RemoteHookFaultForm,
    compute_hook_sources,
    compute_local_hook_candidates,
    merge_hook_fault_forms,
)
from tests.e2e.ft.conftest_ft.fault_injection.recovery_source import CHECKPOINT_TRACKER_FILENAME
from tests.e2e.ft.conftest_ft.fault_injection.state import (
    ACTOR_CELL_TYPE,
    ROLLOUT_CELL_TYPE,
    EventLog,
    HookFireEvent,
    InjectionEvent,
    ObservationsEvent,
)
from tests.e2e.ft.conftest_ft.fault_injection.views import (
    compute_hook_harms,
    compute_injected_cell_names,
    compute_pending_injections,
    compute_successful_form_names,
    compute_unresolved_hook_harms,
)
from tests.fast.e2e.ft.fault_injection.utils import (
    RUNNING_NOT_SERVING,
    SERVING,
    mock_response,
    patched_requests,
    staged,
    typed_cell,
)

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import (
    EventBase,
    FaultHookFireEvent,
    InferenceEngineWeightChecksumEvent,
    TrainGroupStepEndEvent,
    WeightUpdateAssignmentEvent,
)
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity, TrainProcessIdentity
from miles.utils.ft_utils.api_server.models import K8sStatus
from miles.utils.test_utils.fault_hooks import FaultHookName, FaultHookOutcome, FaultHookTarget
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.test_utils.receiver_fault import RECEIVER_SUPPORTED_MODES

_TRAINER = "actor-cell-00001"
_TRAINER_HASH = "trainer-generation-0"
_TRAINER_CELL_INDEX = 1
_ENGINE = "rollout-engine-00000"
_ENGINE_HASH = "engine-generation-0"
_OTHER_ENGINE = "rollout-engine-00001"
_SOURCES_PATH = "/api/v1/fault-hook-sources"
_WEIGHT_VERSION = 7
_REQUEST_ID = "soak-req-1"

_T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
_CONTROLLER = TrainerControllerProcessIdentity(trainer_id="actor")


def _source_identity(*, cell_index: int = _TRAINER_CELL_INDEX, rank: int = 0) -> TrainProcessIdentity:
    return TrainProcessIdentity(component="actor", model_id=None, cell_index=cell_index, rank_within_cell=rank)


def _assignment(
    *,
    trainer_cell_id: str = _TRAINER,
    trainer_workers_hash: str = _TRAINER_HASH,
    assigned: dict[str, str] | None = None,
    weight_version: int = _WEIGHT_VERSION,
) -> WeightUpdateAssignmentEvent:
    return WeightUpdateAssignmentEvent(
        timestamp=_T0,
        source=_CONTROLLER,
        weight_version=weight_version,
        trainer_cell_id=trainer_cell_id,
        trainer_cell_index=_TRAINER_CELL_INDEX,
        trainer_workers_hash=trainer_workers_hash,
        assigned_workers_hash_of_cell_id=dict(assigned if assigned is not None else {_ENGINE: _ENGINE_HASH}),
    )


def _fire(
    *,
    request_id: str = _REQUEST_ID,
    hook: str = FaultHookName.WEIGHT_UPDATE_AFTER_P2P_SUBMIT.value,
    mode: str = FailureMode.SIGKILL.value,
    target: FaultHookTarget = FaultHookTarget.LOCAL,
    outcome: FaultHookOutcome | None = None,
    weight_version: int | None = _WEIGHT_VERSION,
    source: TrainProcessIdentity | None = None,
    victim_cell_id: str | None = None,
    victim_workers_hash: str | None = None,
    receiver_identity: bool = True,
    delay_ms: int = 0,
    at: datetime | None = None,
) -> FaultHookFireEvent:
    if outcome is None:
        outcome = hook_forms.EXPECTED_OUTCOME_OF_TARGET[target]
    remote = target is FaultHookTarget.REMOTE_INFERENCE_CELL
    return FaultHookFireEvent(
        timestamp=at if at is not None else _T0 + timedelta(seconds=10),
        source=source if source is not None else _source_identity(),
        hook=hook,
        mode=mode,
        request_id=request_id,
        weight_version=weight_version,
        target=target.value,
        outcome=outcome.value,
        delay_ms=delay_ms,
        victim_cell_id=(victim_cell_id if victim_cell_id is not None else _ENGINE) if remote else None,
        victim_workers_hash=(
            (victim_workers_hash if victim_workers_hash is not None else _ENGINE_HASH) if remote else None
        ),
        victim_worker_in_cell_index=0 if remote else None,
        victim_receiver_rank=0 if remote and receiver_identity else None,
        victim_receiver_boot_uuid="boot-uuid" if remote and receiver_identity else None,
        victim_session_id="session-1" if remote and receiver_identity else None,
    )


def _publication(
    *,
    at: datetime,
    weight_version: int = _WEIGHT_VERSION + 1,
    cell_ids: tuple[str, ...] = (_ENGINE,),
    trainer_model_id: str | None = None,
) -> InferenceEngineWeightChecksumEvent:
    return InferenceEngineWeightChecksumEvent(
        timestamp=at,
        source=_CONTROLLER,
        rollout_id=1,
        weight_version=weight_version,
        trainer_model_id=trainer_model_id,
        engine_checksums={cell_id: {"w": f"hash-{cell_id}"} for cell_id in cell_ids},
    )


def _write_events(directory: Path, *, file_name: str, events: list[EventBase]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / file_name).open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(event.model_dump_json() + "\n")


def _context(tmp_path: Path, *, log: EventLog, tensor_parallel_size: int = 1, ft_components=("train", "rollout")):
    return HookFaultContext(
        base_url="http://control",
        event_log=log,
        event_dir=tmp_path / "events",
        checkpoint_dir=tmp_path / "ckpt",
        tensor_parallel_size=tensor_parallel_size,
        ft_components=ft_components,
    )


def _observe_cluster(
    log: EventLog,
    *,
    trainer_hash: str = _TRAINER_HASH,
    engine_hash: str = _ENGINE_HASH,
    engine_state=SERVING,
) -> None:
    log.observe(
        [
            typed_cell(_TRAINER, ACTOR_CELL_TYPE, workers_hash=trainer_hash),
            staged(_ENGINE, engine_state, workers_hash=engine_hash),
            staged(_OTHER_ENGINE, SERVING, workers_hash=engine_hash),
        ]
    )


def _arm(
    log: EventLog,
    *,
    target: FaultHookTarget = FaultHookTarget.LOCAL,
    acknowledged: bool = True,
    delay_ms: int = 0,
    mode: FailureMode = FailureMode.SIGKILL,
) -> None:
    for state in [False, True] if acknowledged else [False]:
        log.note_hook_arm(
            request_id=_REQUEST_ID,
            form_name=(
                hook_forms.LOCAL_HOOK_FORM_NAME
                if target is FaultHookTarget.LOCAL
                else hook_forms.REMOTE_HOOK_FORM_NAME
            ),
            cell_type=ACTOR_CELL_TYPE if target is FaultHookTarget.LOCAL else ROLLOUT_CELL_TYPE,
            source_cell_name=_TRAINER,
            source_workers_hash=_TRAINER_HASH,
            source_cell_index=_TRAINER_CELL_INDEX,
            source_rank_within_cell=0,
            hook=FaultHookName.WEIGHT_UPDATE_AFTER_P2P_SUBMIT.value,
            mode=mode.value,
            target=target.value,
            delay_ms=delay_ms,
            acknowledged=state,
        )


def _api_answer(*, status_code: int, body: dict | None, raises: bool = False) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=body) if body is not None else MagicMock(side_effect=ValueError("no json"))
    response.raise_for_status = MagicMock(
        side_effect=RuntimeError(f"the api server answered {status_code}") if raises else None
    )
    return response


def _refusal_body(*, status_code: int, reason: str, message: str) -> dict:
    return K8sStatus(message=message, reason=reason, code=status_code).model_dump()


def _inject_against(tmp_path: Path, log: EventLog, response: MagicMock) -> None:
    _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
    form = LocalHookFaultForm(_context(tmp_path, log=log))
    with _api_server_listing() as mock_requests:
        mock_requests.post.side_effect = lambda url, json, timeout: response
        form.inject(_drawn_trainer_cell(), random.Random(20260908))


def _drawn_trainer_cell(*, workers_hash: str = _TRAINER_HASH) -> dict:
    return typed_cell(_TRAINER, ACTOR_CELL_TYPE, workers_hash=workers_hash)


def _collect(log: EventLog, tmp_path: Path) -> None:
    HookFireCollector(event_log=log, event_dir=tmp_path / "events").collect()


@contextlib.contextmanager
def _api_server_listing(*, trainer_hash: str = _TRAINER_HASH):
    with patched_requests() as mock_requests:

        def _get(url: str, timeout: float):
            trainers = [typed_cell(_TRAINER, ACTOR_CELL_TYPE, workers_hash=trainer_hash)]
            return mock_response({"items": trainers if url.endswith(_SOURCES_PATH) else []})

        mock_requests.get.side_effect = _get
        yield mock_requests


# ========================= source and point selection =========================


class TestHookSourceSelection:
    def test_the_sources_come_from_the_endpoint_that_answers_whether_trainer_ft_is_on(self, tmp_path: Path) -> None:
        """The managed cell list omits the trainers of a rollout-only run, which would leave no source at all."""
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])

        with _api_server_listing() as mock_requests:
            sources = compute_hook_sources(base_url="http://control", event_dir=tmp_path / "events")

        assert [call.args[0] for call in mock_requests.get.call_args_list] == [f"http://control{_SOURCES_PATH}"]
        assert [source.cell_name for source in sources] == [_TRAINER]

    def test_a_trainer_whose_current_generation_owns_a_non_empty_assignment_is_a_source(self, tmp_path: Path) -> None:
        """A trainer is offered as a hook source only through its own live assignment."""
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])

        with _api_server_listing():
            sources = compute_hook_sources(base_url="http://control", event_dir=tmp_path / "events")

        assert [source.cell_name for source in sources] == [_TRAINER]
        assert sources[0].assigned_workers_hash_of_cell_id == {_ENGINE: _ENGINE_HASH}

    def test_an_assignment_of_a_previous_generation_does_not_make_the_replacement_a_source(
        self, tmp_path: Path
    ) -> None:
        """A replacement trainer may not be armed against its predecessor's assignment."""
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])

        with _api_server_listing(trainer_hash="trainer-generation-1"):
            assert compute_hook_sources(base_url="http://control", event_dir=tmp_path / "events") == []

    def test_a_trainer_assigned_no_targets_is_not_a_source(self, tmp_path: Path) -> None:
        """A trainer with an empty assignment reaches no write hook, so it is never armed."""
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment(assigned={})])

        with _api_server_listing():
            assert compute_hook_sources(base_url="http://control", event_dir=tmp_path / "events") == []

    def test_the_all_gather_hook_is_only_drawn_above_tensor_parallel_size_one(self) -> None:
        """Below TP 2 the branch holding the all-gather hook is skipped, so it is not a candidate."""
        assert FaultHookName.WEIGHT_UPDATE_BEFORE_ALL_GATHER not in compute_local_hook_candidates(
            tensor_parallel_size=1
        )
        assert FaultHookName.WEIGHT_UPDATE_BEFORE_ALL_GATHER in compute_local_hook_candidates(tensor_parallel_size=2)

    def test_a_remote_request_is_only_drawn_at_the_hooks_that_name_one_peer(self) -> None:
        """A remote fault may only be armed where the site knows which engine it is writing to."""
        assert hook_forms.compute_remote_hook_candidates() == sorted(
            {FaultHookName.WEIGHT_UPDATE_BEFORE_P2P_WRITE, FaultHookName.WEIGHT_UPDATE_AFTER_P2P_SUBMIT}
        )

    def test_a_form_is_unavailable_until_a_recovery_source_exists(self, tmp_path: Path) -> None:
        """Arming before a checkpoint and a real publication exist would remove the only heal source."""
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        form = LocalHookFaultForm(_context(tmp_path, log=EventLog()))

        with _api_server_listing():
            with patch.object(hook_forms, "recovery_source_exists", return_value=False):
                assert not form.is_available(_drawn_trainer_cell())
            with patch.object(hook_forms, "recovery_source_exists", return_value=True):
                assert form.is_available(_drawn_trainer_cell())

    def test_a_trainer_replaced_since_the_listing_is_no_longer_its_own_source(self, tmp_path: Path) -> None:
        """A cell drawn at one generation must not be armed against another generation's assignment."""
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        form = LocalHookFaultForm(_context(tmp_path, log=EventLog()))

        with _api_server_listing():
            with patch.object(hook_forms, "recovery_source_exists", return_value=True):
                assert not form.is_available(_drawn_trainer_cell(workers_hash="trainer-generation-1"))

    def test_a_remote_form_ignores_the_drawn_engine_and_picks_a_real_trainer(self, tmp_path: Path) -> None:
        """The drawn rollout cell only names the kind; the trigger is a trainer with an assignment."""
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        form = RemoteHookFaultForm(_context(tmp_path, log=EventLog()))

        with _api_server_listing():
            sources = form.eligible_sources(staged(_ENGINE, SERVING))

        assert [source.cell_name for source in sources] == [_TRAINER]


# =================================== arming ===================================


class TestArming:
    def test_the_request_carries_the_source_hash_the_form_drew(self, tmp_path: Path) -> None:
        """The draw and the arm are two moments, and only this hash stops the second one hitting a replacement."""
        log = EventLog()
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        form = LocalHookFaultForm(_context(tmp_path, log=log))

        with _api_server_listing() as mock_requests:
            mock_requests.post.side_effect = lambda url, json, timeout: mock_response({})
            form.inject(_drawn_trainer_cell(), random.Random(20260929))

        (harm,) = compute_hook_harms(log.events)
        (post,) = mock_requests.post.call_args_list
        assert post.kwargs["json"]["expected_workers_hash"] == _TRAINER_HASH
        assert harm.source_workers_hash == _TRAINER_HASH

    def test_the_attempt_is_recorded_before_the_request_leaves(self, tmp_path: Path) -> None:
        """A request that never gets an answer still holds the responsibility it created."""
        log = EventLog()
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        form = LocalHookFaultForm(_context(tmp_path, log=log))

        with _api_server_listing() as mock_requests:
            mock_requests.post.side_effect = RuntimeError("the api server never answered")
            with pytest.raises(RuntimeError):
                form.inject(_drawn_trainer_cell(), random.Random(0))

        (harm,) = compute_hook_harms(log.events)
        assert not harm.acknowledged and not harm.delivered
        assert [entry.request_id for entry in compute_pending_injections(log.events)] == [harm.request_id]

    def test_an_acknowledged_arm_is_not_yet_a_successful_form(self, tmp_path: Path) -> None:
        """A 200 from the api server proves the request was accepted, not that a fault landed."""
        log = EventLog()
        _arm(log)

        assert compute_successful_form_names(log.events, cell_type=ACTOR_CELL_TYPE) == set()
        assert compute_unresolved_hook_harms(log.events)

    def test_the_two_arm_records_of_one_request_are_one_outstanding_harm(self, tmp_path: Path) -> None:
        """The attempt and its acknowledgement are the same fault, joined by request id."""
        log = EventLog()
        _arm(log)

        assert len(compute_hook_harms(log.events)) == 1
        assert len(compute_pending_injections(log.events)) == 1

    def test_a_waiting_hook_blocks_the_next_fault_of_any_kind(self, tmp_path: Path) -> None:
        """One armed hook holds the run-wide harm slot even before its victim is known."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log, target=FaultHookTarget.REMOTE_INFERENCE_CELL)
        _observe_cluster(log)

        assert compute_pending_injections(log.events)

    def test_an_unfired_hook_eventually_fails_the_run_rather_than_being_rearmed(self, tmp_path: Path) -> None:
        """An arm whose outcome never became known is a hard failure, never a blind retry."""
        log = EventLog()
        _arm(log)

        with pytest.raises(UnresolvedInjectionError):
            _raise_if_an_unknown_injection_is_stuck(compute_pending_injections(log.events), timeout_seconds=0.0)


class TestArmingRefusedBeforeAnythingWasArmed:
    """A refusal this api server raises before it reaches a worker proves the request cost the run nothing."""

    def test_a_refused_request_releases_the_slot_without_counting_a_success(self, tmp_path: Path) -> None:
        """The trainer changed generation between the listing and the arm, so no worker was ever asked."""
        log = EventLog()
        _observe_cluster(log)
        body = _refusal_body(status_code=400, reason="BadRequest", message=f"{_TRAINER} now runs another generation")

        _inject_against(tmp_path, log, _api_answer(status_code=400, body=body))

        (harm,) = compute_hook_harms(log.events)
        assert harm.resolved_without_harm and not harm.completed
        assert compute_pending_injections(log.events) == []
        assert compute_unresolved_hook_harms(log.events) == []
        assert compute_successful_form_names(log.events, cell_type=ACTOR_CELL_TYPE) == set()
        assert compute_injected_cell_names(log.events) == []

    def test_a_cell_this_deployment_never_served_releases_the_slot(self, tmp_path: Path) -> None:
        """A 404 is raised by the source registry, which answers before any controller is asked to arm."""
        log = EventLog()
        _observe_cluster(log)
        body = _refusal_body(status_code=404, reason="NotFound", message=f"'{_TRAINER}' is not a fault hook source")

        _inject_against(tmp_path, log, _api_answer(status_code=404, body=body))

        assert compute_unresolved_hook_harms(log.events) == []

    def test_a_request_the_api_server_read_as_malformed_releases_the_slot(self, tmp_path: Path) -> None:
        """A 422 is raised while validating the body, so the route function never ran."""
        log = EventLog()
        _observe_cluster(log)

        detail = [
            {
                "type": "less_than_equal",
                "loc": ["body", "delay_ms"],
                "msg": "Input should be less than or equal to 10000",
                "input": 10001,
            }
        ]
        _inject_against(tmp_path, log, _api_answer(status_code=422, body={"detail": detail}))

        assert compute_unresolved_hook_harms(log.events) == []

    @pytest.mark.parametrize(
        "detail",
        [
            "the request was bad",
            [],
            [{"loc": ["body", "delay_ms"]}],
            [{"type": "missing", "loc": ["query", "delay_ms"], "msg": "Field required"}],
        ],
    )
    def test_an_unknown_422_detail_is_no_proof(self, tmp_path: Path, detail: object) -> None:
        """Only a complete body-validation error proves FastAPI rejected the request before routing it."""
        log = EventLog()
        _observe_cluster(log)

        with pytest.raises(RuntimeError):
            _inject_against(tmp_path, log, _api_answer(status_code=422, body={"detail": detail}, raises=True))

        assert compute_unresolved_hook_harms(log.events)

    @pytest.mark.parametrize(
        ("status_code", "reason"),
        [(400, "NotFound"), (404, "BadRequest"), (400, "Unknown")],
    )
    def test_a_status_whose_reason_disagrees_with_its_code_is_no_proof(
        self, tmp_path: Path, status_code: int, reason: str
    ) -> None:
        """The service emits one fixed Kubernetes reason for each pre-arm refusal code."""
        log = EventLog()
        _observe_cluster(log)
        body = _refusal_body(status_code=status_code, reason=reason, message="not routed")

        with pytest.raises(RuntimeError):
            _inject_against(tmp_path, log, _api_answer(status_code=status_code, body=body, raises=True))

        assert compute_unresolved_hook_harms(log.events)

    def test_an_internal_error_keeps_the_request_outstanding(self, tmp_path: Path) -> None:
        """A 500 wraps everything the arming call itself can raise, including a reply lost after it armed."""
        log = EventLog()
        _observe_cluster(log)

        with pytest.raises(RuntimeError):
            _inject_against(tmp_path, log, _api_answer(status_code=500, body=None, raises=True))

        (harm,) = compute_hook_harms(log.events)
        assert not harm.resolved_without_harm
        assert [entry.request_id for entry in compute_pending_injections(log.events)] == [harm.request_id]

    def test_a_refusal_status_carrying_an_unknown_body_is_no_proof(self, tmp_path: Path) -> None:
        """Only this service's own refusal shape says the request stopped before a worker was asked."""
        log = EventLog()
        _observe_cluster(log)

        with pytest.raises(RuntimeError):
            _inject_against(
                tmp_path, log, _api_answer(status_code=400, body={"error": "rejected by something else"}, raises=True)
            )

        assert compute_unresolved_hook_harms(log.events)

    def test_a_refusal_status_with_no_body_at_all_is_no_proof(self, tmp_path: Path) -> None:
        """A proxy between the injector and the api server can answer 400 without ever reaching it."""
        log = EventLog()
        _observe_cluster(log)

        with pytest.raises(RuntimeError):
            _inject_against(tmp_path, log, _api_answer(status_code=400, body=None, raises=True))

        assert compute_unresolved_hook_harms(log.events)

    def test_an_acknowledgement_after_a_refusal_fails_the_run(self) -> None:
        """A request cannot both stop before routing and later be acknowledged as armed."""
        log = EventLog()
        _arm(log, acknowledged=False)
        log.note_hook_arm_refusal(request_id=_REQUEST_ID, refused_because="not routed")
        _arm(log)

        with pytest.raises(AssertionError, match="later acknowledged"):
            compute_hook_harms(log.events)

    def test_a_refusal_after_an_acknowledgement_fails_the_run(self) -> None:
        """The reverse event order cannot make acknowledgement and pre-arm refusal consistent."""
        log = EventLog()
        _arm(log)
        log.note_hook_arm_refusal(request_id=_REQUEST_ID, refused_because="not routed")

        with pytest.raises(AssertionError, match="acknowledged as armed"):
            compute_hook_harms(log.events)


# =================================== delays ===================================


class TestDrawingASoakDelay:
    def test_the_same_seed_draws_the_same_delays(self) -> None:
        """A soak that failed on a delayed fault has to be replayable from its seed alone."""
        first = [hook_forms.draw_fault_hook_delay_ms(random.Random(20260929)) for _ in range(5)]
        second = [hook_forms.draw_fault_hook_delay_ms(random.Random(20260929)) for _ in range(5)]

        assert first == second

    def test_successive_draws_from_one_generator_differ(self) -> None:
        """A helper returning one constant would give the soak a single delay dressed up as a random one."""
        rng = random.Random(20260929)

        drawn = [hook_forms.draw_fault_hook_delay_ms(rng) for _ in range(20)]

        assert len(set(drawn)) > 1

    def test_every_draw_is_a_delay_the_worker_accepts(self) -> None:
        """An arm above the worker's bound is refused, so a draw outside it would break the soak, not delay it."""
        rng = random.Random(20260929)

        drawn = [hook_forms.draw_fault_hook_delay_ms(rng) for _ in range(200)]

        assert all(isinstance(delay, int) and not isinstance(delay, bool) for delay in drawn)
        assert min(drawn) >= hook_forms.SOAK_DELAY_MS_MIN
        assert max(drawn) <= hook_forms.SOAK_DELAY_MS_MAX

    def test_the_whole_documented_range_is_asked_of_the_generator(self) -> None:
        """An exclusive upper bound would drop the longest delay the soak is written to cover."""
        asked: list[tuple[int, int]] = []

        class _RecordingRandom(random.Random):
            def randint(self, a: int, b: int) -> int:
                asked.append((a, b))
                return b

        drawn = hook_forms.draw_fault_hook_delay_ms(_RecordingRandom(0))

        assert asked == [(hook_forms.SOAK_DELAY_MS_MIN, hook_forms.SOAK_DELAY_MS_MAX)]
        assert drawn == hook_forms.SOAK_DELAY_MS_MAX


class TestTheSoakArmsADrawnDelay:
    def test_the_drawn_delay_is_both_sent_and_recorded(self, tmp_path: Path) -> None:
        """The delay only becomes evidence if the same value reaches the worker and the run's own books."""
        log = EventLog()
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        form = LocalHookFaultForm(_context(tmp_path, log=log))

        with _api_server_listing() as mock_requests:
            mock_requests.post.side_effect = lambda url, json, timeout: mock_response({})
            form.inject(_drawn_trainer_cell(), random.Random(20260929))

        (harm,) = compute_hook_harms(log.events)
        (post,) = mock_requests.post.call_args_list
        assert post.kwargs["json"]["delay_ms"] == harm.delay_ms
        assert hook_forms.SOAK_DELAY_MS_MIN <= harm.delay_ms <= hook_forms.SOAK_DELAY_MS_MAX

    def test_the_delay_comes_from_the_generator_the_form_was_drawn_with(self, tmp_path: Path) -> None:
        """The soak draws its source, its point and its delay from one seeded stream, or a replay diverges."""
        log = EventLog()
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        form = LocalHookFaultForm(_context(tmp_path, log=log))
        replay = random.Random(20260929)
        replay.choice([_TRAINER])
        replay.choice(form.candidate_hooks())
        replay.choice(form.candidate_modes())

        with _api_server_listing() as mock_requests:
            mock_requests.post.side_effect = lambda url, json, timeout: mock_response({})
            form.inject(_drawn_trainer_cell(), random.Random(20260929))

        (harm,) = compute_hook_harms(log.events)
        assert harm.delay_ms == hook_forms.draw_fault_hook_delay_ms(replay)

    def test_a_fire_that_waited_the_armed_delay_is_delivered(self, tmp_path: Path) -> None:
        """A delayed fault that really ran is the evidence the soak armed it for."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log, delay_ms=500)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(tmp_path / "events", file_name="actor.jsonl", events=[_fire(delay_ms=500)])

        _collect(log, tmp_path)

        (harm,) = compute_hook_harms(log.events)
        assert harm.delivered and harm.fire.delay_ms == 500

    def test_the_record_keeps_the_moment_production_reached_the_hook(self, tmp_path: Path) -> None:
        """The collector reads the log later, and its own clock would date the fault to whenever it looked."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log)
        fire = _fire()
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(tmp_path / "events", file_name="actor.jsonl", events=[fire])

        _collect(log, tmp_path)

        (harm,) = compute_hook_harms(log.events)
        assert harm.fire.fired_at == fire.timestamp
        assert harm.fire.timestamp > fire.timestamp

    def test_a_fire_that_waited_another_time_leaves_the_request_outstanding(self, tmp_path: Path) -> None:
        """A fault that landed inside the update it was armed to outlive tested the moment nobody asked about."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log, delay_ms=500)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(tmp_path / "events", file_name="actor.jsonl", events=[_fire(delay_ms=0)])

        _collect(log, tmp_path)

        (harm,) = compute_hook_harms(log.events)
        assert not harm.delivered and "not the 500ms it was armed for" in harm.fire.rejected_because
        assert compute_unresolved_hook_harms(log.events)


# ================================ failure modes ===============================


class TestDrawingASoakFailureMode:
    def test_a_local_hook_may_crash_or_hang_the_trainer_that_reached_it(self, tmp_path: Path) -> None:
        """The three are what a trainer process can inflict on itself, and each fails the run differently."""
        form = LocalHookFaultForm(_context(tmp_path, log=EventLog()))

        assert form.candidate_modes() == [FailureMode.SIGKILL, FailureMode.DEADLOCK, FailureMode.SIGSTOP]

    def test_a_remote_hook_draws_only_what_the_receiver_can_raise_on_itself(self, tmp_path: Path) -> None:
        """Drawing anything else would arm a request the worker refuses, breaking the run instead of an engine."""
        form = RemoteHookFaultForm(_context(tmp_path, log=EventLog()))

        assert set(form.candidate_modes()) == RECEIVER_SUPPORTED_MODES

    def test_the_drawn_mode_is_both_sent_and_recorded(self, tmp_path: Path) -> None:
        """A run that armed a hang and recorded a kill could not tell which fault its witnesses judged."""
        log = EventLog()
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        form = LocalHookFaultForm(_context(tmp_path, log=log))

        with _api_server_listing() as mock_requests:
            mock_requests.post.side_effect = lambda url, json, timeout: mock_response({})
            form.inject(_drawn_trainer_cell(), random.Random(20260929))

        (harm,) = compute_hook_harms(log.events)
        (post,) = mock_requests.post.call_args_list
        assert post.kwargs["json"]["mode"] == harm.mode
        assert harm.mode in {mode.value for mode in form.candidate_modes()}

    def test_the_mode_comes_from_the_generator_the_form_was_drawn_with(self, tmp_path: Path) -> None:
        """A soak that hung on one draw has to replay that same draw from its seed alone."""
        log = EventLog()
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        form = LocalHookFaultForm(_context(tmp_path, log=log))
        replay = random.Random(20260929)
        replay.choice([_TRAINER])
        replay.choice(form.candidate_hooks())

        with _api_server_listing() as mock_requests:
            mock_requests.post.side_effect = lambda url, json, timeout: mock_response({})
            form.inject(_drawn_trainer_cell(), random.Random(20260929))

        (harm,) = compute_hook_harms(log.events)
        assert harm.mode == replay.choice(form.candidate_modes()).value

    def test_a_hang_that_fired_as_the_mode_it_was_armed_for_is_delivered(self, tmp_path: Path) -> None:
        """The fire record is what proves which fault production ran, so it has to carry the armed mode."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log, mode=FailureMode.SIGSTOP)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(tmp_path / "events", file_name="actor.jsonl", events=[_fire(mode=FailureMode.SIGSTOP.value)])

        _collect(log, tmp_path)

        (harm,) = compute_hook_harms(log.events)
        assert harm.delivered and harm.fire.mode == FailureMode.SIGSTOP.value

    def test_a_fire_in_another_mode_leaves_the_request_outstanding(self, tmp_path: Path) -> None:
        """A kill recorded for a request that armed a hang tested a boundary nobody asked about."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log, mode=FailureMode.SIGSTOP)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(tmp_path / "events", file_name="actor.jsonl", events=[_fire()])

        _collect(log, tmp_path)

        (harm,) = compute_hook_harms(log.events)
        assert not harm.delivered and "not the armed" in harm.fire.rejected_because
        assert compute_unresolved_hook_harms(log.events)


# ============================== fire collection ===============================


class TestFireCollection:
    def test_a_delivered_local_fire_names_the_armed_trainer_as_the_victim(self, tmp_path: Path) -> None:
        """A local fault is paid for by the trainer incarnation that was armed."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(tmp_path / "events", file_name="actor.jsonl", events=[_fire()])

        _collect(log, tmp_path)

        (harm,) = compute_hook_harms(log.events)
        assert harm.delivered and harm.victim_cell_name == _TRAINER and harm.victim_workers_hash == _TRAINER_HASH
        assert compute_injected_cell_names(log.events, cell_type=ACTOR_CELL_TYPE) == [_TRAINER]

    def test_a_delivered_remote_fire_is_counted_against_the_engine_it_names(self, tmp_path: Path) -> None:
        """A remote fault is a rollout injection, even though a trainer triggered it."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log, target=FaultHookTarget.REMOTE_INFERENCE_CELL)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(
            tmp_path / "events",
            file_name="actor.jsonl",
            events=[_fire(target=FaultHookTarget.REMOTE_INFERENCE_CELL)],
        )

        _collect(log, tmp_path)

        (harm,) = compute_hook_harms(log.events)
        assert harm.delivered and harm.victim_cell_name == _ENGINE
        assert compute_injected_cell_names(log.events, cell_type=ROLLOUT_CELL_TYPE) == [_ENGINE]
        assert compute_injected_cell_names(log.events, cell_type=ACTOR_CELL_TYPE) == []

    def test_the_same_fire_read_twice_stays_one_harm(self, tmp_path: Path) -> None:
        """The event dir is re-read every poll, so a fire must not accumulate."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(tmp_path / "events", file_name="actor.jsonl", events=[_fire(), _fire()])

        collector = HookFireCollector(event_log=log, event_dir=tmp_path / "events")
        collector.collect()
        collector.collect()

        assert len(compute_hook_harms(log.events)) == 1
        assert len(compute_pending_injections(log.events)) == 1
        assert len([event for event in log.events if isinstance(event, HookFireEvent)]) == 1

    @pytest.mark.parametrize(
        "fires",
        [
            [
                _fire(target=FaultHookTarget.REMOTE_INFERENCE_CELL, outcome=FaultHookOutcome.STALE_TARGET),
                _fire(target=FaultHookTarget.REMOTE_INFERENCE_CELL, outcome=FaultHookOutcome.ACCEPTED),
            ],
            [
                _fire(target=FaultHookTarget.REMOTE_INFERENCE_CELL, outcome=FaultHookOutcome.ACCEPTED),
                _fire(target=FaultHookTarget.REMOTE_INFERENCE_CELL, outcome=FaultHookOutcome.STALE_TARGET),
            ],
        ],
    )
    def test_harmless_and_delivered_fires_of_one_request_fail_in_either_order(
        self, tmp_path: Path, fires: list[FaultHookFireEvent]
    ) -> None:
        """One request cannot both be refused by its stale receiver and deliver the armed fault."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log, target=FaultHookTarget.REMOTE_INFERENCE_CELL)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(tmp_path / "events", file_name="actor.jsonl", events=fires[:1])
        collector = HookFireCollector(event_log=log, event_dir=tmp_path / "events")
        collector.collect()
        _write_events(tmp_path / "events", file_name="actor.jsonl", events=fires[1:])

        with pytest.raises(AssertionError, match="different raw fault hook fires"):
            collector.collect()

    def test_a_second_fire_reaches_the_reducer_after_the_first_resolved_the_request(self, tmp_path: Path) -> None:
        """The reducer must reject a late delivery after a harmless stale answer released the slot."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log, target=FaultHookTarget.REMOTE_INFERENCE_CELL)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(
            tmp_path / "events",
            file_name="actor.jsonl",
            events=[_fire(target=FaultHookTarget.REMOTE_INFERENCE_CELL, outcome=FaultHookOutcome.STALE_TARGET)],
        )
        _collect(log, tmp_path)
        (first,) = [event for event in log.events if isinstance(event, HookFireEvent)]
        log.note_hook_fire(
            first.model_copy(
                update={
                    "outcome": FaultHookOutcome.ACCEPTED.value,
                    "delivered": True,
                    "harmless_because": None,
                    "victim_cell_name": _ENGINE,
                    "victim_workers_hash": _ENGINE_HASH,
                }
            )
        )

        with pytest.raises(AssertionError, match="conflicting fault hook fires"):
            compute_hook_harms(log.events)

    def test_a_fire_already_written_before_the_arm_was_acknowledged_is_still_paired(self, tmp_path: Path) -> None:
        """The armed worker dies at the hook, so its fire can be recorded before its reply arrives."""
        log = EventLog()
        _observe_cluster(log)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(tmp_path / "events", file_name="actor.jsonl", events=[_fire()])
        _arm(log, acknowledged=False)

        _collect(log, tmp_path)

        (harm,) = compute_hook_harms(log.events)
        assert harm.delivered and not harm.acknowledged

    def test_a_fire_of_another_request_is_ignored(self, tmp_path: Path) -> None:
        """Only the request this soak minted can resolve this soak's responsibility."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(tmp_path / "events", file_name="actor.jsonl", events=[_fire(request_id="someone-else")])

        _collect(log, tmp_path)

        (harm,) = compute_hook_harms(log.events)
        assert harm.fire is None


@pytest.mark.parametrize(
    "fire, expected_in_reason",
    [
        (_fire(hook=FaultHookName.WEIGHT_UPDATE_BEFORE_ALL_GATHER.value), "not the armed"),
        (_fire(mode=FailureMode.EXIT.value), "not the armed"),
        (_fire(source=_source_identity(cell_index=0)), "not in the armed"),
        (_fire(source=_source_identity(rank=1)), "not in the armed"),
        (_fire(weight_version=None), "outside any weight update"),
        (_fire(weight_version=99), "not given exactly one assignment"),
        (_fire(outcome=FaultHookOutcome.STALE_TARGET), "not the fired"),
        (_fire(outcome=FaultHookOutcome.UNKNOWN), "not the fired"),
        (_fire(outcome=FaultHookOutcome.REFUSED), "not the fired"),
        (_fire(target=FaultHookTarget.REMOTE_INFERENCE_CELL), "not the armed local"),
        (_fire(delay_ms=500), "not the 0ms it was armed for"),
    ],
)
def test_a_fire_that_does_not_match_the_arm_is_rejected(
    tmp_path: Path, fire: FaultHookFireEvent, expected_in_reason: str
) -> None:
    """A fault at another point, in another process or with another answer harmed nobody here."""
    log = EventLog()
    _observe_cluster(log)
    _arm(log)
    _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
    _write_events(tmp_path / "events", file_name="actor.jsonl", events=[fire])

    _collect(log, tmp_path)

    (harm,) = compute_hook_harms(log.events)
    assert not harm.delivered and expected_in_reason in harm.fire.rejected_because
    assert compute_unresolved_hook_harms(log.events)


@pytest.mark.parametrize(
    "fire",
    [
        _fire(target=FaultHookTarget.REMOTE_INFERENCE_CELL, victim_cell_id=_OTHER_ENGINE),
        _fire(target=FaultHookTarget.REMOTE_INFERENCE_CELL, victim_workers_hash="engine-generation-9"),
        _fire(target=FaultHookTarget.REMOTE_INFERENCE_CELL, receiver_identity=False),
    ],
)
def test_a_remote_fire_naming_an_engine_the_write_never_reached_is_rejected(
    tmp_path: Path, fire: FaultHookFireEvent
) -> None:
    """A remote victim has to be the assigned engine at the incarnation the receiver identified."""
    log = EventLog()
    _observe_cluster(log)
    _arm(log, target=FaultHookTarget.REMOTE_INFERENCE_CELL)
    _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
    _write_events(tmp_path / "events", file_name="actor.jsonl", events=[fire])

    _collect(log, tmp_path)

    (harm,) = compute_hook_harms(log.events)
    assert not harm.delivered and harm.victim_cell_name is None


class TestStaleTargetResolution:
    """A receiver that refused a request as stale proves it harmed nothing, once the fire is shown to be this one."""

    def _stale(self, tmp_path: Path, log: EventLog, **fire_kwargs) -> None:
        _observe_cluster(log)
        _arm(log, target=FaultHookTarget.REMOTE_INFERENCE_CELL)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(
            tmp_path / "events",
            file_name="actor.jsonl",
            events=[
                _fire(
                    target=FaultHookTarget.REMOTE_INFERENCE_CELL,
                    outcome=FaultHookOutcome.STALE_TARGET,
                    **fire_kwargs,
                )
            ],
        )
        _collect(log, tmp_path)

    def test_a_stale_answer_from_the_assigned_engine_releases_the_request(self, tmp_path: Path) -> None:
        """The receiver of the engine this update assigned refused the request, so nothing was harmed."""
        log = EventLog()

        self._stale(tmp_path, log)

        (harm,) = compute_hook_harms(log.events)
        assert harm.resolved_without_harm and not harm.delivered
        assert compute_unresolved_hook_harms(log.events) == []
        assert compute_pending_injections(log.events) == []

    def test_a_released_stale_request_is_no_injection_and_no_successful_form(self, tmp_path: Path) -> None:
        """Releasing the slot pays no floor: the form still owes a fault that actually landed."""
        log = EventLog()

        self._stale(tmp_path, log)

        assert compute_injected_cell_names(log.events, cell_type=ROLLOUT_CELL_TYPE) == []
        assert compute_successful_form_names(log.events, cell_type=ROLLOUT_CELL_TYPE) == set()

    def test_a_stale_answer_naming_an_engine_the_assignment_never_gave_stays_outstanding(self, tmp_path: Path) -> None:
        """A fire about another engine says nothing about the request this soak is still holding."""
        log = EventLog()

        self._stale(tmp_path, log, victim_cell_id=_OTHER_ENGINE)

        (harm,) = compute_hook_harms(log.events)
        assert not harm.resolved_without_harm
        assert compute_unresolved_hook_harms(log.events)

    def test_a_stale_answer_from_another_incarnation_of_the_assigned_engine_stays_outstanding(
        self, tmp_path: Path
    ) -> None:
        """The assignment names one generation, and a later one was never written to by this update."""
        log = EventLog()

        self._stale(tmp_path, log, victim_workers_hash="engine-generation-9")

        assert compute_unresolved_hook_harms(log.events)

    def test_a_stale_answer_without_a_receiver_identity_stays_outstanding(self, tmp_path: Path) -> None:
        """Without the receiver's own boot uuid and session nothing says which process refused."""
        log = EventLog()

        self._stale(tmp_path, log, receiver_identity=False)

        assert compute_unresolved_hook_harms(log.events)

    def test_a_stale_answer_from_another_process_identity_stays_outstanding(self, tmp_path: Path) -> None:
        """The fire has to have happened in the worker this request was armed in."""
        log = EventLog()

        self._stale(tmp_path, log, source=_source_identity(cell_index=0))

        assert compute_unresolved_hook_harms(log.events)

    def test_a_stale_answer_outside_any_weight_update_stays_outstanding(self, tmp_path: Path) -> None:
        """Without a weight version there is no assignment to check the victim against."""
        log = EventLog()

        self._stale(tmp_path, log, weight_version=None)

        assert compute_unresolved_hook_harms(log.events)

    def test_a_stale_answer_in_an_update_the_armed_generation_never_sent_stays_outstanding(
        self, tmp_path: Path
    ) -> None:
        """A version the armed incarnation owns no single assignment for names targets nobody can check."""
        log = EventLog()

        self._stale(tmp_path, log, weight_version=99)

        assert compute_unresolved_hook_harms(log.events)

    def test_a_local_request_is_never_released_by_a_stale_answer(self, tmp_path: Path) -> None:
        """A local fault has no receiver to be stale against, so the answer proves nothing about it."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(
            tmp_path / "events",
            file_name="actor.jsonl",
            events=[_fire(outcome=FaultHookOutcome.STALE_TARGET)],
        )

        _collect(log, tmp_path)

        assert compute_unresolved_hook_harms(log.events)


@pytest.mark.parametrize(
    "outcome", [FaultHookOutcome.REFUSED, FaultHookOutcome.NOT_SCHEDULED, FaultHookOutcome.ERRORED]
)
def test_a_remote_answer_other_than_stale_keeps_the_request_outstanding(
    tmp_path: Path, outcome: FaultHookOutcome
) -> None:
    """These are exceptions inside the sender, which can still reach the write and retire a cell."""
    log = EventLog()
    _observe_cluster(log)
    _arm(log, target=FaultHookTarget.REMOTE_INFERENCE_CELL)
    _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
    _write_events(
        tmp_path / "events",
        file_name="actor.jsonl",
        events=[_fire(target=FaultHookTarget.REMOTE_INFERENCE_CELL, outcome=outcome)],
    )

    _collect(log, tmp_path)

    (harm,) = compute_hook_harms(log.events)
    assert not harm.resolved_without_harm and not harm.delivered
    assert compute_unresolved_hook_harms(log.events)


def test_a_delivered_fire_naming_a_refused_request_fails_the_run(tmp_path: Path) -> None:
    """One of the two records is wrong about what happened, and quietly keeping the refusal would hide a live fault."""
    log = EventLog()
    _observe_cluster(log)
    body = _refusal_body(status_code=400, reason="BadRequest", message="another generation")
    _inject_against(tmp_path, log, _api_answer(status_code=400, body=body))
    (harm,) = compute_hook_harms(log.events)
    _write_events(
        tmp_path / "events",
        file_name="actor.jsonl",
        events=[_fire(request_id=harm.request_id, hook=harm.hook, mode=harm.mode, delay_ms=harm.delay_ms)],
    )

    _collect(log, tmp_path)

    with pytest.raises(AssertionError, match="refused before it armed anything"):
        compute_hook_harms(log.events)


def test_a_refusal_recorded_after_a_delivered_fire_fails_the_run(tmp_path: Path) -> None:
    """A late refusal cannot release a slot after the request already delivered a fault."""
    log = EventLog()
    _observe_cluster(log)
    _arm(log, acknowledged=False)
    _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
    _write_events(tmp_path / "events", file_name="actor.jsonl", events=[_fire()])
    _collect(log, tmp_path)
    log.note_hook_arm_refusal(request_id=_REQUEST_ID, refused_because="not routed")

    with pytest.raises(AssertionError, match="delivered a fault"):
        compute_hook_harms(log.events)


def test_a_fire_in_an_update_another_trainer_incarnation_sent_is_rejected(tmp_path: Path) -> None:
    """The identity alone repeats every generation, so the update's own sender has to be the armed one."""
    log = EventLog()
    _observe_cluster(log)
    _arm(log)
    _write_events(
        tmp_path / "events",
        file_name="controller.jsonl",
        events=[_assignment(trainer_workers_hash="trainer-generation-1")],
    )
    _write_events(tmp_path / "events", file_name="actor.jsonl", events=[_fire()])

    _collect(log, tmp_path)

    (harm,) = compute_hook_harms(log.events)
    assert not harm.delivered and _TRAINER_HASH in harm.fire.rejected_because
    assert harm.victim_cell_name is None


def test_a_replacement_already_serving_does_not_pay_off_an_arm_whose_fire_was_rejected(tmp_path: Path) -> None:
    """A generation that appeared for its own reasons cannot settle a fault that never landed."""
    log = EventLog()
    _observe_cluster(log)
    _arm(log)
    _write_events(
        tmp_path / "events",
        file_name="controller.jsonl",
        events=[_assignment(trainer_workers_hash="trainer-generation-1")],
    )
    _write_events(tmp_path / "events", file_name="actor.jsonl", events=[_fire()])
    _collect(log, tmp_path)
    _observe_cluster(log, trainer_hash="trainer-generation-1")
    _observe_cluster(log, trainer_hash="trainer-generation-1")

    assert compute_unresolved_hook_harms(log.events)
    assert compute_successful_form_names(log.events, cell_type=ACTOR_CELL_TYPE) == set()
    assert compute_injected_cell_names(log.events, cell_type=ACTOR_CELL_TYPE) == []
    assert [entry.request_id for entry in compute_pending_injections(log.events)] == [_REQUEST_ID]


def test_an_assignment_of_another_cell_index_under_the_armed_name_is_rejected(tmp_path: Path) -> None:
    """A cell id reused at another index is a different member of the run, not the armed one."""
    log = EventLog()
    _observe_cluster(log)
    _arm(log)
    assignment = _assignment().model_copy(update={"trainer_cell_index": _TRAINER_CELL_INDEX + 1})
    _write_events(tmp_path / "events", file_name="controller.jsonl", events=[assignment])
    _write_events(tmp_path / "events", file_name="actor.jsonl", events=[_fire()])

    _collect(log, tmp_path)

    (harm,) = compute_hook_harms(log.events)
    assert not harm.delivered and harm.victim_cell_name is None


def test_a_second_assignment_of_the_same_update_and_cell_is_rejected(tmp_path: Path) -> None:
    """Two attempts at one update leave it unknown which one the fault fired inside."""
    log = EventLog()
    _observe_cluster(log)
    _arm(log)
    _write_events(
        tmp_path / "events",
        file_name="controller.jsonl",
        events=[_assignment(), _assignment(trainer_workers_hash="trainer-generation-1")],
    )
    _write_events(tmp_path / "events", file_name="actor.jsonl", events=[_fire()])

    _collect(log, tmp_path)

    (harm,) = compute_hook_harms(log.events)
    assert not harm.delivered and harm.victim_cell_name is None


# ================================== recovery ==================================


class TestRecovery:
    @staticmethod
    def _harm_a_remote_victim(log: EventLog, tmp_path: Path) -> None:
        _observe_cluster(log)
        _arm(log, target=FaultHookTarget.REMOTE_INFERENCE_CELL)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(
            tmp_path / "events",
            file_name="actor.jsonl",
            events=[_fire(target=FaultHookTarget.REMOTE_INFERENCE_CELL)],
        )
        _collect(log, tmp_path)

    @staticmethod
    def _publish(tmp_path: Path, **kwargs) -> None:
        _write_events(
            tmp_path / "events",
            file_name="main.jsonl",
            events=[_publication(at=datetime.now(timezone.utc), **kwargs)],
        )

    def test_a_harm_is_resolved_by_a_replacement_that_took_new_weights(self, tmp_path: Path) -> None:
        """A form counts as successful once the replacement of the harmed incarnation is publishing again."""
        log = EventLog()
        self._harm_a_remote_victim(log, tmp_path)
        _observe_cluster(log, engine_hash="engine-generation-1")
        self._publish(tmp_path)
        _collect(log, tmp_path)

        assert compute_successful_form_names(log.events, cell_type=ROLLOUT_CELL_TYPE) == {
            hook_forms.REMOTE_HOOK_FORM_NAME
        }
        assert compute_unresolved_hook_harms(log.events) == []
        assert compute_pending_injections(log.events) == []

    def test_a_replacement_that_never_took_weights_again_stays_pending(self, tmp_path: Path) -> None:
        """Serving is not proof it rejoined the fan-out, and the next fault must wait for that proof."""
        log = EventLog()
        self._harm_a_remote_victim(log, tmp_path)
        _observe_cluster(log, engine_hash="engine-generation-1")
        _collect(log, tmp_path)

        assert compute_unresolved_hook_harms(log.events)
        assert compute_pending_injections(log.events)

    def test_a_publication_from_before_the_recovery_does_not_resolve_it(self, tmp_path: Path) -> None:
        """The harmed incarnation can still answer a push, and reading the log later does not make it later."""
        log = EventLog()
        self._harm_a_remote_victim(log, tmp_path)
        self._publish(tmp_path)
        _observe_cluster(log, engine_hash="engine-generation-1")
        _collect(log, tmp_path)

        assert compute_unresolved_hook_harms(log.events)

    def test_another_cells_publication_does_not_resolve_it(self, tmp_path: Path) -> None:
        """Weights reaching an engine the fault never touched say nothing about the one it killed."""
        log = EventLog()
        self._harm_a_remote_victim(log, tmp_path)
        _observe_cluster(log, engine_hash="engine-generation-1")
        self._publish(tmp_path, cell_ids=(_OTHER_ENGINE,))
        _collect(log, tmp_path)

        assert compute_unresolved_hook_harms(log.events)

    def test_another_policys_publication_does_not_resolve_it(self, tmp_path: Path) -> None:
        """Another policy's weights never travel the path this fault broke."""
        log = EventLog()
        self._harm_a_remote_victim(log, tmp_path)
        _observe_cluster(log, engine_hash="engine-generation-1")
        self._publish(tmp_path, trainer_model_id="other")
        _collect(log, tmp_path)

        assert compute_unresolved_hook_harms(log.events)

    def test_republishing_the_version_the_fault_fired_in_does_not_resolve_it(self, tmp_path: Path) -> None:
        """That update is the one the fault landed in, so finishing it is not going on past it."""
        log = EventLog()
        self._harm_a_remote_victim(log, tmp_path)
        _observe_cluster(log, engine_hash="engine-generation-1")
        self._publish(tmp_path, weight_version=_WEIGHT_VERSION)
        _collect(log, tmp_path)

        assert compute_unresolved_hook_harms(log.events)

    def test_a_replacement_observed_before_the_fault_fired_does_not_resolve_it(self, tmp_path: Path) -> None:
        """The collector reads the log late, and an observation from before the fire answers an earlier harm."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log, target=FaultHookTarget.REMOTE_INFERENCE_CELL)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(
            tmp_path / "events",
            file_name="actor.jsonl",
            events=[
                _fire(
                    target=FaultHookTarget.REMOTE_INFERENCE_CELL,
                    at=datetime.now(timezone.utc) + timedelta(hours=1),
                )
            ],
        )
        _collect(log, tmp_path)
        _observe_cluster(log, engine_hash="engine-generation-1")
        self._publish(tmp_path)
        _collect(log, tmp_path)

        assert compute_unresolved_hook_harms(log.events)

    def test_a_local_harm_is_resolved_by_its_assigned_targets_publishing_again(self, tmp_path: Path) -> None:
        """A trainer fault costs the engines it was writing to, so those are the cells that owe the evidence."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(tmp_path / "events", file_name="actor.jsonl", events=[_fire()])
        _collect(log, tmp_path)
        _observe_cluster(log, trainer_hash="trainer-generation-1")
        self._publish(tmp_path)
        _collect(log, tmp_path)

        assert compute_unresolved_hook_harms(log.events) == []

    def test_the_harmed_generation_reading_healthy_again_does_not_resolve_it(self, tmp_path: Path) -> None:
        """The killed incarnation reads healthy for a long time; only a new one is evidence."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log, target=FaultHookTarget.REMOTE_INFERENCE_CELL)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(
            tmp_path / "events",
            file_name="actor.jsonl",
            events=[_fire(target=FaultHookTarget.REMOTE_INFERENCE_CELL)],
        )
        _collect(log, tmp_path)
        _observe_cluster(log)

        assert compute_unresolved_hook_harms(log.events)
        assert compute_successful_form_names(log.events, cell_type=ROLLOUT_CELL_TYPE) == set()

    def test_a_replacement_that_is_not_serving_yet_does_not_resolve_it(self, tmp_path: Path) -> None:
        """An engine that took weights but cannot answer requests is not back in service."""
        log = EventLog()
        _observe_cluster(log)
        _arm(log, target=FaultHookTarget.REMOTE_INFERENCE_CELL)
        _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])
        _write_events(
            tmp_path / "events",
            file_name="actor.jsonl",
            events=[_fire(target=FaultHookTarget.REMOTE_INFERENCE_CELL)],
        )
        _collect(log, tmp_path)
        _observe_cluster(
            log,
            engine_hash="engine-generation-1",
            engine_state=RUNNING_NOT_SERVING,
        )

        assert compute_unresolved_hook_harms(log.events)


# =================================== wiring ===================================


class TestWiring:
    def test_each_target_joins_the_kind_that_owes_its_recovery(self, tmp_path: Path) -> None:
        """A local hook is an actor form and a remote one is a rollout form."""
        forms = {ACTOR_CELL_TYPE: [], ROLLOUT_CELL_TYPE: []}
        merged = merge_hook_fault_forms(forms, _context(tmp_path, log=EventLog()))

        assert [form.name for form in merged[ACTOR_CELL_TYPE]] == [hook_forms.LOCAL_HOOK_FORM_NAME]
        assert [form.name for form in merged[ROLLOUT_CELL_TYPE]] == [hook_forms.REMOTE_HOOK_FORM_NAME]

    def test_a_kind_without_fault_tolerance_gets_no_hook_form(self, tmp_path: Path) -> None:
        """A hook may only harm a kind the run can heal."""
        forms = {ACTOR_CELL_TYPE: [], ROLLOUT_CELL_TYPE: []}
        merged = merge_hook_fault_forms(forms, _context(tmp_path, log=EventLog(), ft_components=("rollout",)))

        assert merged[ACTOR_CELL_TYPE] == []
        assert [form.name for form in merged[ROLLOUT_CELL_TYPE]] == [hook_forms.REMOTE_HOOK_FORM_NAME]

    def test_the_wall_clock_forms_are_kept_beside_the_hook_forms(self, tmp_path: Path) -> None:
        """A hook form is added to the draw, never in place of the forms already there."""
        wall_clock = InjectFaultForm(base_url="http://control", failure_mode=FailureMode.SIGKILL)
        merged = merge_hook_fault_forms(
            {ACTOR_CELL_TYPE: [wall_clock], ROLLOUT_CELL_TYPE: [wall_clock]}, _context(tmp_path, log=EventLog())
        )

        assert [form.name for form in merged[ACTOR_CELL_TYPE]] == [
            wall_clock.name,
            hook_forms.LOCAL_HOOK_FORM_NAME,
        ]

    def test_a_hook_form_keeps_its_own_books(self, tmp_path: Path) -> None:
        """The loop must not record an InjectionEvent claiming an arm harmed the cell it was sent to."""
        assert LocalHookFaultForm(_context(tmp_path, log=EventLog())).records_own_attempt
        assert RemoteHookFaultForm(_context(tmp_path, log=EventLog())).records_own_attempt
        assert not InjectFaultForm(base_url="http://control", failure_mode=FailureMode.SIGKILL).records_own_attempt


# ============================ rollout-only wiring =============================


def _write_recovery_source(tmp_path: Path) -> None:
    (tmp_path / "ckpt").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ckpt" / CHECKPOINT_TRACKER_FILENAME).write_text("3", encoding="utf-8")
    _write_events(
        tmp_path / "events",
        file_name="train.jsonl",
        events=[
            TrainGroupStepEndEvent(
                timestamp=_T0, source=_CONTROLLER, rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL]}
            ),
            InferenceEngineWeightChecksumEvent(
                timestamp=_T0,
                source=_CONTROLLER,
                rollout_id=1,
                weight_version=1,
                engine_checksums={"rollout-engine-00000": {"embedding": "abc"}},
            ),
        ],
    )


def test_a_rollout_only_soak_arms_a_trainer_it_never_crashes(tmp_path: Path) -> None:
    """A remote hook needs a trainer trigger even where trainer ft is off and trainers are never listed."""
    _write_recovery_source(tmp_path)
    _write_events(tmp_path / "events", file_name="controller.jsonl", events=[_assignment()])

    log = EventLog()
    context = _context(tmp_path, log=log, ft_components=("rollout",))
    forms = merge_hook_fault_forms({ACTOR_CELL_TYPE: [], ROLLOUT_CELL_TYPE: []}, context)
    assert [form.name for form in forms[ROLLOUT_CELL_TYPE]] == [hook_forms.REMOTE_HOOK_FORM_NAME]

    stop_event = threading.Event()
    posts: list[dict] = []
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> object:
        polls["n"] += 1
        if posts or polls["n"] >= 50:
            stop_event.set()
        if url.endswith(_SOURCES_PATH):
            return mock_response({"items": [typed_cell(_TRAINER, ACTOR_CELL_TYPE, workers_hash=_TRAINER_HASH)]})
        return mock_response(
            {
                "items": [
                    staged(_ENGINE, SERVING, workers_hash=_ENGINE_HASH),
                    staged(_OTHER_ENGINE, SERVING, workers_hash=_ENGINE_HASH),
                ]
            }
        )

    def fake_post(url: str, json: dict, timeout: float) -> object:
        posts.append(dict(url=url, json=json))
        return mock_response({})

    with patched_requests() as mock_requests:
        mock_requests.get.side_effect = fake_get
        mock_requests.post.side_effect = fake_post
        run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type={ROLLOUT_CELL_TYPE: 1e-9},
            stop_event=stop_event,
            event_log=log,
            cell_fault_forms=forms,
            poll_interval_seconds=1e-6,
        )

    assert posts and posts[0]["url"] == f"http://control/api/v1/cells/{_TRAINER}/arm-fault-hook"
    assert posts[0]["json"]["expected_workers_hash"] == _TRAINER_HASH

    (harm,) = compute_hook_harms(log.events)
    assert harm.source_cell_name == _TRAINER and harm.cell_type == ROLLOUT_CELL_TYPE
    assert harm.source_workers_hash == _TRAINER_HASH

    observed = {name for event in log.events if isinstance(event, ObservationsEvent) for name in event.cell_infos}
    assert observed == {_ENGINE, _OTHER_ENGINE}
    assert not [event for event in log.events if isinstance(event, InjectionEvent)]


def test_the_loop_reads_the_production_log_once_more_after_it_is_told_to_stop() -> None:
    """Weights published in the last moments of the run are the ones the final harm is still owed."""
    stop_event = threading.Event()
    stop_event.set()
    collected: list[int] = []

    run_fault_injection_loop(
        base_url="http://control",
        seed=0,
        mean_interval_seconds_of_cell_type={ROLLOUT_CELL_TYPE: 1e-9},
        stop_event=stop_event,
        event_log=EventLog(),
        cell_fault_forms={ROLLOUT_CELL_TYPE: []},
        collect_hook_fires=lambda: collected.append(1),
        poll_interval_seconds=1e-6,
    )

    assert collected == [1]
