from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.e2e.ft.conftest_ft import hook_injection
from tests.e2e.ft.conftest_ft.fault_injection.state import EventLog
from tests.fast.e2e.ft.fault_injection.utils import RUNNING_NOT_SERVING, SERVING, cell, staged

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import (
    CellReconfigureEvent,
    EventBase,
    FaultHookFireEvent,
    InferenceEngineWeightChecksumEvent,
    TrainGroupStepEndEvent,
    WeightUpdateAssignmentEvent,
)
from miles.utils.audit_utils.process_identity import (
    SimpleProcessIdentity,
    TrainerControllerProcessIdentity,
    TrainProcessIdentity,
)
from miles.utils.test_utils.fault_hooks import FaultHookName, FaultHookOutcome, FaultHookTarget
from miles.utils.test_utils.fault_injector import FailureMode

_HOOK = FaultHookName.WEIGHT_UPDATE_BEFORE_ALL_GATHER
_REQUEST_ID = "req-1"
_WEIGHT_VERSION = 7
_ARMED_CELL = "trainer-engine-actor-00001"
_ARMED_HASH = "trainer-generation-0"
_ASSIGNED = {"rollout-engine-00000": "engine-generation-0", "rollout-engine-00001": "engine-generation-0"}
_UNRELATED = {"rollout-engine-00002": "engine-generation-0", "rollout-engine-00003": "engine-generation-0"}

_T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
_CONTROLLER = TrainerControllerProcessIdentity(trainer_id="actor")
_DRIVER = SimpleProcessIdentity(component="main")


class _RecordingRequests:
    def __init__(self, posts: list[dict]) -> None:
        self._posts = posts

    def post(self, url: str, *, json: dict, timeout: float):
        self._posts.append(dict(url=url, json=json, timeout=timeout))
        return SimpleNamespace(raise_for_status=lambda: None)


def _at(seconds: int) -> datetime:
    return _T0 + timedelta(seconds=seconds)


_FIRED_AT = _at(10)
_ASSIGNED_AT = _at(5)


def _source(*, component: str = "actor", model_id: str | None = None, cell_index: int = 1, rank: int = 0):
    return TrainProcessIdentity(component=component, model_id=model_id, cell_index=cell_index, rank_within_cell=rank)


def _armed(
    *,
    sub_index: int = 0,
    hook: FaultHookName = _HOOK,
    trainer_hash: str = _ARMED_HASH,
    inference: dict[str, str] | None = None,
    snapshot_at: datetime = _T0,
    expected_source=None,
    target: FaultHookTarget = FaultHookTarget.LOCAL,
    delay_ms: int = 0,
) -> hook_injection.ArmedFaultHook:
    return hook_injection.ArmedFaultHook(
        cell_name=_ARMED_CELL,
        sub_index=sub_index,
        hook=hook,
        mode=FailureMode.SIGKILL,
        target=target,
        request_id=_REQUEST_ID,
        delay_ms=delay_ms,
        expected_source=expected_source or _source(rank=sub_index),
        trainer_workers_hash=trainer_hash,
        inference_workers_hash_of_cell_id=dict(inference if inference is not None else {**_ASSIGNED, **_UNRELATED}),
        snapshot_at=snapshot_at,
        acknowledged_at=snapshot_at + timedelta(seconds=1),
    )


def _write_events(directory: Path, *, file_name: str, events: list[EventBase]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / file_name).open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(event.model_dump_json() + "\n")


def _fire(
    *,
    at: datetime = _FIRED_AT,
    weight_version: int | None = _WEIGHT_VERSION,
    source=None,
    request_id: str = _REQUEST_ID,
    hook: str = _HOOK.value,
    target: FaultHookTarget = FaultHookTarget.LOCAL,
    outcome: FaultHookOutcome = FaultHookOutcome.FIRED,
    victim: tuple[str, str] | None = None,
    receiver_rank: int | None = 1,
    delay_ms: int = 0,
) -> FaultHookFireEvent:
    return FaultHookFireEvent(
        timestamp=at,
        source=source or _source(),
        hook=hook,
        mode=FailureMode.SIGKILL.value,
        request_id=request_id,
        weight_version=weight_version,
        target=target.value,
        outcome=outcome.value,
        delay_ms=delay_ms,
        victim_cell_id=None if victim is None else victim[0],
        victim_workers_hash=None if victim is None else victim[1],
        victim_worker_in_cell_index=None,
        victim_receiver_rank=None if victim is None else receiver_rank,
        victim_receiver_boot_uuid=None if victim is None else "boot-1",
        victim_session_id=None if victim is None else "session-1",
    )


def _assignment(
    *,
    at: datetime = _ASSIGNED_AT,
    weight_version: int = _WEIGHT_VERSION,
    trainer_hash: str = _ARMED_HASH,
    assigned: dict[str, str] | None = None,
) -> WeightUpdateAssignmentEvent:
    return WeightUpdateAssignmentEvent(
        timestamp=at,
        source=_CONTROLLER,
        weight_version=weight_version,
        trainer_cell_id=_ARMED_CELL,
        trainer_cell_index=1,
        trainer_workers_hash=trainer_hash,
        assigned_workers_hash_of_cell_id=dict(assigned if assigned is not None else _ASSIGNED),
    )


def _reconfigure(*, at: datetime, alive: list[int], healed: list[int]) -> CellReconfigureEvent:
    return CellReconfigureEvent(
        timestamp=at,
        source=_CONTROLLER,
        rollout_id=1,
        quorum_id=1,
        src_cell_index=0 if healed else None,
        healed_cell_indices=healed,
        alive_cell_indices_after=alive,
    )


def _publication(*, at: datetime, checksums: list[dict[str, str]], rollout_id: int = 1):
    return InferenceEngineWeightChecksumEvent(
        timestamp=at, source=_DRIVER, rollout_id=rollout_id, engine_checksums=checksums
    )


def _step_end(*, at: datetime = _T0, outcome=TrainStepOutcome.NORMAL) -> TrainGroupStepEndEvent:
    return TrainGroupStepEndEvent(timestamp=at, source=_CONTROLLER, rollout_id=0, cell_outcomes={0: [outcome]})


def _observe(log: EventLog, *, trainer_hash: str, engines: dict[str, str], trainer_alive: bool = True) -> None:
    cells = [
        cell(_ARMED_CELL, healthy=trainer_alive, cell_type="actor", workers_hash=trainer_hash),
        *[staged(name, SERVING, workers_hash=workers_hash) for name, workers_hash in engines.items()],
    ]
    log.observe(cells)


def _healthy_run_log(
    *,
    trainer_hashes: list[str],
    assigned_after: dict[str, str] | None = None,
    unrelated_after: dict[str, str] | None = None,
) -> EventLog:
    log = EventLog()
    _observe(log, trainer_hash=trainer_hashes[0], engines={**_ASSIGNED, **_UNRELATED})
    for trainer_hash in trainer_hashes[1:]:
        _observe(
            log,
            trainer_hash=trainer_hash,
            engines={
                **(assigned_after if assigned_after is not None else {k: "engine-generation-1" for k in _ASSIGNED}),
                **(unrelated_after if unrelated_after is not None else _UNRELATED),
            },
        )
    return log


class TestTheArmRequestNamesTheGenerationItChose:
    def test_the_snapshot_hash_is_what_the_request_carries(self, monkeypatch: pytest.MonkeyPatch):
        """Without it the server would arm whatever generation is current, which may be a replacement."""
        posts: list[dict] = []
        monkeypatch.setattr(hook_injection, "requests", _RecordingRequests(posts))
        snapshot = hook_injection.CellSnapshot(
            taken_at=_T0,
            workers_hash_of_cell_id={_ARMED_CELL: _ARMED_HASH},
            alive_cell_ids=frozenset({_ARMED_CELL}),
        )

        armed = hook_injection.arm_fault_hook_over_api(
            base_url="http://control",
            cell_name=_ARMED_CELL,
            sub_index=2,
            hook=_HOOK,
            mode=FailureMode.SIGKILL,
            target=FaultHookTarget.LOCAL,
            request_id=_REQUEST_ID,
            delay_ms=300,
            trainer_snapshot=snapshot,
            inference_snapshot=hook_injection.CellSnapshot(
                taken_at=_T0, workers_hash_of_cell_id={}, alive_cell_ids=frozenset()
            ),
        )

        assert [post["url"] for post in posts] == [f"http://control/api/v1/cells/{_ARMED_CELL}/arm-fault-hook"]
        assert posts[0]["json"] == {
            "expected_workers_hash": _ARMED_HASH,
            "hook": _HOOK.value,
            "mode": FailureMode.SIGKILL.value,
            "target": FaultHookTarget.LOCAL.value,
            "sub_index": 2,
            "request_id": _REQUEST_ID,
            "delay_ms": 300,
        }
        assert armed.trainer_workers_hash == _ARMED_HASH


class TestRecoverySourceGate:
    def test_a_run_without_a_checkpoint_is_not_ready_for_a_fault(self, tmp_path: Path):
        """Killing the only source of the state a replacement needs would fail the run for the wrong reason."""
        _write_events(
            tmp_path / "events",
            file_name="main.jsonl",
            events=[_step_end(), _publication(at=_T0, checksums=[{"w": "h"}])],
        )

        assert not hook_injection.recovery_source_exists(
            event_dir=tmp_path / "events", checkpoint_dir=tmp_path / "ckpt"
        )

    def test_a_checkpoint_tracker_that_names_no_iteration_is_not_a_recovery_source(self, tmp_path: Path):
        """An empty or non-numeric tracker is written before the checkpoint is usable, and a fault then lands early."""
        (tmp_path / "ckpt").mkdir()
        (tmp_path / "ckpt" / hook_injection.CHECKPOINT_TRACKER_FILENAME).write_text("release\n")
        _write_events(
            tmp_path / "events",
            file_name="main.jsonl",
            events=[_step_end(), _publication(at=_T0, checksums=[{"w": "h"}])],
        )

        assert not hook_injection.recovery_source_exists(
            event_dir=tmp_path / "events", checkpoint_dir=tmp_path / "ckpt"
        )

    def test_an_iteration_zero_tracker_is_not_a_recovery_source(self, tmp_path: Path):
        """Iteration zero is the tracker a run writes before any step of its own has been saved."""
        (tmp_path / "ckpt").mkdir()
        (tmp_path / "ckpt" / hook_injection.CHECKPOINT_TRACKER_FILENAME).write_text("0")

        assert hook_injection.read_checkpoint_iteration(tmp_path / "ckpt") is None

    def test_an_initial_publication_alone_is_not_a_recovery_source(self, tmp_path: Path):
        """The out-of-loop startup sync stamps rollout -1, so arming on it injects into the very first update."""
        (tmp_path / "ckpt").mkdir()
        (tmp_path / "ckpt" / hook_injection.CHECKPOINT_TRACKER_FILENAME).write_text("1")
        _write_events(
            tmp_path / "events",
            file_name="main.jsonl",
            events=[_step_end(), _publication(at=_T0, checksums=[{"w": "h"}], rollout_id=-1)],
        )

        assert not hook_injection.recovery_source_exists(
            event_dir=tmp_path / "events", checkpoint_dir=tmp_path / "ckpt"
        )

    def test_an_empty_publication_is_not_a_recovery_source(self, tmp_path: Path):
        """A checksum event carrying no engine, or only empty dicts, proves no weights ever reached an engine."""
        (tmp_path / "ckpt").mkdir()
        (tmp_path / "ckpt" / hook_injection.CHECKPOINT_TRACKER_FILENAME).write_text("1")
        _write_events(
            tmp_path / "events",
            file_name="main.jsonl",
            events=[_step_end(), _publication(at=_T0, checksums=[{}])],
        )

        assert not hook_injection.recovery_source_exists(
            event_dir=tmp_path / "events", checkpoint_dir=tmp_path / "ckpt"
        )

    def test_a_run_that_completed_no_training_step_is_not_a_recovery_source(self, tmp_path: Path):
        """A discarded or errored step leaves the checkpoint describing nothing this run actually trained."""
        (tmp_path / "ckpt").mkdir()
        (tmp_path / "ckpt" / hook_injection.CHECKPOINT_TRACKER_FILENAME).write_text("1")
        _write_events(
            tmp_path / "events",
            file_name="main.jsonl",
            events=[
                _step_end(outcome=TrainStepOutcome.DISCARDED_SHOULD_RETRY),
                _publication(at=_T0, checksums=[{"w": "h"}]),
            ],
        )

        assert not hook_injection.recovery_source_exists(
            event_dir=tmp_path / "events", checkpoint_dir=tmp_path / "ckpt"
        )

    def test_a_completed_step_a_checkpoint_and_a_real_publication_open_the_gate(self, tmp_path: Path):
        """All three exist, so a crashed cell has state to be healed from and the fault may land."""
        (tmp_path / "ckpt").mkdir()
        (tmp_path / "ckpt" / hook_injection.CHECKPOINT_TRACKER_FILENAME).write_text("1")
        _write_events(
            tmp_path / "events",
            file_name="main.jsonl",
            events=[_step_end(), _publication(at=_T0, checksums=[{"w": "h"}])],
        )

        assert hook_injection.recovery_source_exists(event_dir=tmp_path / "events", checkpoint_dir=tmp_path / "ckpt")


class TestFireWitness:
    def test_a_hook_that_never_fired_fails_the_run(self, tmp_path: Path):
        """A successful arm is a request, not a fault; counting it as one would pass a test that injected nothing."""
        with pytest.raises(AssertionError, match="reached it 0 time"):
            hook_injection.assert_hook_fired(_armed(), event_dir=tmp_path)

    def test_another_requests_fire_does_not_satisfy_this_one(self, tmp_path: Path):
        """Arms and fires interleave in a soak log, so they are paired by request id and nothing else."""
        _write_events(tmp_path, file_name="actor.jsonl", events=[_fire(request_id="req-2")])

        with pytest.raises(AssertionError, match="reached it 0 time"):
            hook_injection.assert_hook_fired(_armed(), event_dir=tmp_path)

    def test_a_fire_in_another_rank_fails_the_run(self, tmp_path: Path):
        """A fire attributed to the wrong rank would let a neighbouring worker's fault pay for this one."""
        _write_events(tmp_path, file_name="actor.jsonl", events=[_fire(source=_source(rank=1))])

        with pytest.raises(AssertionError, match="not in"):
            hook_injection.assert_hook_fired(_armed(sub_index=0), event_dir=tmp_path)

    def test_a_fire_in_another_cell_fails_the_run(self, tmp_path: Path):
        """Two cells run the same rank index, and the surviving cell's fault must not pay for the armed one's."""
        _write_events(tmp_path, file_name="actor.jsonl", events=[_fire(source=_source(cell_index=0))])

        with pytest.raises(AssertionError, match="not in"):
            hook_injection.assert_hook_fired(_armed(), event_dir=tmp_path)

    def test_a_fire_in_another_role_fails_the_run(self, tmp_path: Path):
        """A critic runs the same cell and rank indices, and its fault is a different scenario entirely."""
        _write_events(tmp_path, file_name="critic.jsonl", events=[_fire(source=_source(component="critic"))])

        with pytest.raises(AssertionError, match="not in"):
            hook_injection.assert_hook_fired(_armed(), event_dir=tmp_path)

    def test_a_fire_for_another_trained_model_fails_the_run(self, tmp_path: Path):
        """Multi-model runs repeat every cell and rank index once per model."""
        _write_events(tmp_path, file_name="other.jsonl", events=[_fire(source=_source(model_id="other"))])

        with pytest.raises(AssertionError, match="not in"):
            hook_injection.assert_hook_fired(_armed(), event_dir=tmp_path)

    def test_a_fire_outside_any_weight_update_fails_the_run(self, tmp_path: Path):
        """Without the update it fired in there is nothing to hold the assigned engines against."""
        _write_events(tmp_path, file_name="actor.jsonl", events=[_fire(weight_version=None)])

        with pytest.raises(AssertionError, match="outside any weight update"):
            hook_injection.assert_hook_fired(_armed(), event_dir=tmp_path)

    def test_a_fire_that_waited_another_time_than_it_was_armed_for_fails_the_run(self, tmp_path: Path):
        """A delayed request that fired inside the update tested the moment the scenario was not written about."""
        _write_events(tmp_path, file_name="actor.jsonl", events=[_fire(delay_ms=0)])

        with pytest.raises(AssertionError, match="delay"):
            hook_injection.assert_hook_fired(_armed(delay_ms=500), event_dir=tmp_path)

    def test_the_delay_the_request_asked_for_is_carried_by_the_fire(self, tmp_path: Path):
        """The worker records what it waited, which is how a run proves the fault landed after the update moved on."""
        _write_events(tmp_path, file_name="actor.jsonl", events=[_fire(delay_ms=500)])

        fire = hook_injection.assert_hook_fired(_armed(delay_ms=500), event_dir=tmp_path)

        assert fire.delay_ms == 500

    def test_a_run_that_never_armed_fails_before_any_other_witness(self):
        """Without an arm there is no request id to look for, and a silent skip would report green."""

        class _Armer:
            armed = None

        with pytest.raises(AssertionError, match="No fault hook was ever armed"):
            hook_injection.require_armed(_Armer())

    def test_a_fire_recorded_before_the_arm_was_acknowledged_still_counts(self, tmp_path: Path):
        """The worker dies at the hook before answering the rpc, so the ack is never the fault's timestamp."""
        armed = _armed()
        _write_events(
            tmp_path, file_name="actor.jsonl", events=[_fire(at=armed.acknowledged_at - timedelta(seconds=5))]
        )

        fire = hook_injection.assert_hook_fired(armed, event_dir=tmp_path)

        assert fire.timestamp < armed.acknowledged_at


class TestAssignmentWitness:
    def test_an_update_nobody_recorded_an_assignment_for_fails_the_run(self, tmp_path: Path):
        """Without the controller's own record, which engines the harmed sender owned is a guess from cell names."""
        _write_events(tmp_path, file_name="actor.jsonl", events=[_fire()])

        with pytest.raises(AssertionError, match="never assigned targets"):
            hook_injection.resolve_fire_assignment(_fire(), _armed(), event_dir=tmp_path)

    def test_an_assignment_sent_by_a_replacement_fails_the_run(self, tmp_path: Path):
        """A cell replaced before the fault fired proves the fault landed in a process nobody armed."""
        _write_events(
            tmp_path,
            file_name="controller.jsonl",
            events=[_assignment(trainer_hash="trainer-generation-1")],
        )

        with pytest.raises(AssertionError, match="not by the"):
            hook_injection.resolve_fire_assignment(_fire(), _armed(), event_dir=tmp_path)

    def test_two_attempts_of_the_same_version_fail_the_run(self, tmp_path: Path):
        """Which attempt the fault landed in decides which engines it owned, and a guess would check the wrong set."""
        _write_events(
            tmp_path,
            file_name="controller.jsonl",
            events=[_assignment(), _assignment(at=_at(6), trainer_hash="trainer-generation-1")],
        )

        with pytest.raises(AssertionError, match="times, so which attempt"):
            hook_injection.resolve_fire_assignment(_fire(), _armed(), event_dir=tmp_path)

    def test_the_assignment_of_the_armed_incarnation_is_returned(self, tmp_path: Path):
        """This is the join the rest of the witnesses read: this update, this sender, these engines."""
        _write_events(
            tmp_path,
            file_name="controller.jsonl",
            events=[_assignment(weight_version=_WEIGHT_VERSION - 1, assigned={}), _assignment()],
        )

        assignment = hook_injection.resolve_fire_assignment(_fire(), _armed(), event_dir=tmp_path)

        assert assignment.assigned_workers_hash_of_cell_id == _ASSIGNED


class TestTrainerReplacementWitness:
    def test_a_cell_that_kept_the_armed_incarnation_fails_the_run(self):
        """A fault that left its target running is a fault that did not land."""
        log = _healthy_run_log(trainer_hashes=[_ARMED_HASH, _ARMED_HASH])

        with pytest.raises(AssertionError, match="never observed under an incarnation other than"):
            hook_injection.assert_armed_trainer_was_replaced(log.events, armed=_armed())

    def test_a_replacement_that_never_became_healthy_fails_the_run(self):
        """A cell that lost its incarnation and never came back leaves the run a replica short."""
        log = EventLog()
        _observe(log, trainer_hash=_ARMED_HASH, engines={**_ASSIGNED, **_UNRELATED})
        _observe(log, trainer_hash="trainer-generation-1", engines={**_ASSIGNED, **_UNRELATED}, trainer_alive=False)

        with pytest.raises(AssertionError, match="never observed healthy under a replacement"):
            hook_injection.assert_armed_trainer_was_replaced(log.events, armed=_armed())

    def test_an_armed_incarnation_still_alive_at_the_end_fails_the_run(self):
        """A hash that flickered and came back is the same process, so nothing was evicted."""
        log = EventLog()
        _observe(log, trainer_hash=_ARMED_HASH, engines={**_ASSIGNED, **_UNRELATED})
        _observe(log, trainer_hash="trainer-generation-1", engines={**_ASSIGNED, **_UNRELATED})
        _observe(log, trainer_hash=_ARMED_HASH, engines={**_ASSIGNED, **_UNRELATED})

        with pytest.raises(AssertionError, match="still running the armed incarnation"):
            hook_injection.assert_armed_trainer_was_replaced(log.events, armed=_armed())

    def test_losing_the_armed_incarnation_and_coming_back_healthy_passes(self):
        """The armed process is gone and its cell serves again, which is what recovery means for a trainer."""
        log = _healthy_run_log(trainer_hashes=[_ARMED_HASH, "trainer-generation-1"])

        harm_observed_at = hook_injection.assert_armed_trainer_was_replaced(log.events, armed=_armed())

        assert harm_observed_at is not None


class TestTrainerHealingWitness:
    def test_an_eviction_from_before_the_fault_does_not_count(self, tmp_path: Path):
        """Picking any historical reconfigure would let an unrelated earlier crash pay for this fault."""
        _write_events(
            tmp_path,
            file_name="controller.jsonl",
            events=[
                _reconfigure(at=_at(1), alive=[0], healed=[]),
                _reconfigure(at=_at(2), alive=[0, 1], healed=[1]),
                _assignment(at=_at(5)),
            ],
        )

        with pytest.raises(AssertionError, match="no reconfigure after the update"):
            hook_injection.assert_trainer_cell_healed(tmp_path, assignment=_assignment(at=_at(5)))

    def test_a_healing_that_precedes_the_eviction_does_not_count(self, tmp_path: Path):
        """A heal recorded before the cell was dropped belongs to a different loss of that cell."""
        _write_events(
            tmp_path,
            file_name="controller.jsonl",
            events=[
                _assignment(at=_at(5)),
                _reconfigure(at=_at(6), alive=[0, 1], healed=[1]),
                _reconfigure(at=_at(7), alive=[0], healed=[]),
            ],
        )

        with pytest.raises(AssertionError, match="never healed back after the eviction"):
            hook_injection.assert_trainer_cell_healed(tmp_path, assignment=_assignment(at=_at(5)))

    def test_an_eviction_then_a_healing_after_the_fault_passes(self, tmp_path: Path):
        """The cell the fault killed lost its membership and got it back, in that order."""
        _write_events(
            tmp_path,
            file_name="controller.jsonl",
            events=[
                _assignment(at=_at(5)),
                _reconfigure(at=_at(6), alive=[0], healed=[]),
                _reconfigure(at=_at(7), alive=[0, 1], healed=[1]),
            ],
        )

        hook_injection.assert_trainer_cell_healed(tmp_path, assignment=_assignment(at=_at(5)))


class TestIsolationWitness:
    def test_a_target_the_sender_owned_that_kept_serving_fails_the_run(self):
        """A sender that died mid-update leaves its targets half-written, so each must leave service."""
        log = _healthy_run_log(
            trainer_hashes=[_ARMED_HASH, "trainer-generation-1"],
            assigned_after={
                "rollout-engine-00000": "engine-generation-1",
                "rollout-engine-00001": "engine-generation-0",
            },
        )

        with pytest.raises(AssertionError, match="still runs"):
            hook_injection.assert_assigned_targets_isolated(log.events, assignment=_assignment(), since=_T0)

    def test_a_target_that_never_ran_the_assigned_incarnation_fails_the_run(self):
        """An engine that never held the hash the sender wrote to says nothing about that write."""
        log = _healthy_run_log(
            trainer_hashes=[_ARMED_HASH, "trainer-generation-1"],
            assigned_after={name: "engine-generation-1" for name in _ASSIGNED},
        )

        with pytest.raises(AssertionError, match="never observed running"):
            hook_injection.assert_assigned_targets_isolated(
                log.events,
                assignment=_assignment(assigned={name: "engine-generation-9" for name in _ASSIGNED}),
                since=_T0,
            )

    def test_a_target_that_never_served_again_fails_the_run(self):
        """Every engine the harmed sender owned has to come back, not just the first one to be replaced."""
        log = EventLog()
        _observe(log, trainer_hash=_ARMED_HASH, engines={**_ASSIGNED, **_UNRELATED})
        log.observe(
            [
                cell(_ARMED_CELL, healthy=True, cell_type="actor", workers_hash="trainer-generation-1"),
                staged("rollout-engine-00000", SERVING, workers_hash="engine-generation-1"),
                staged("rollout-engine-00001", RUNNING_NOT_SERVING, workers_hash="engine-generation-1"),
                *[staged(name, SERVING, workers_hash=h) for name, h in _UNRELATED.items()],
            ]
        )

        with pytest.raises(AssertionError, match="never observed healthy and Serving under a replacement"):
            hook_injection.assert_assigned_targets_isolated(log.events, assignment=_assignment(), since=_T0)

    def test_every_assigned_target_replaced_and_serving_again_passes(self):
        """This is the claim: the sender's own engines were dropped and each came back."""
        log = _healthy_run_log(trainer_hashes=[_ARMED_HASH, "trainer-generation-1"])

        hook_injection.assert_assigned_targets_isolated(log.events, assignment=_assignment(), since=_T0)


class TestBlastRadiusWitness:
    def test_an_unrelated_engine_that_was_replaced_too_fails_the_run(self):
        """A fault aimed at one sender's targets that costs every engine its incarnation was not contained."""
        log = _healthy_run_log(
            trainer_hashes=[_ARMED_HASH, "trainer-generation-1"],
            unrelated_after={name: "engine-generation-1" for name in _UNRELATED},
        )

        with pytest.raises(AssertionError, match="Blast-radius witness failed"):
            hook_injection.assert_unrelated_target_kept_serving(
                log.events, armed=_armed(), assignment=_assignment(), since=_T0
            )

    def test_an_unrelated_engine_that_stopped_serving_fails_the_run(self):
        """Keeping a hash while stuck is not evidence that an unrelated target went on working."""
        log = EventLog()
        _observe(log, trainer_hash=_ARMED_HASH, engines={**_ASSIGNED, **_UNRELATED})
        log.observe(
            [
                cell(_ARMED_CELL, healthy=True, cell_type="actor", workers_hash="trainer-generation-1"),
                *[staged(name, SERVING, workers_hash="engine-generation-1") for name in _ASSIGNED],
                *[staged(name, RUNNING_NOT_SERVING, workers_hash=h) for name, h in _UNRELATED.items()],
            ]
        )

        with pytest.raises(AssertionError, match="Blast-radius witness failed"):
            hook_injection.assert_unrelated_target_kept_serving(
                log.events, armed=_armed(), assignment=_assignment(), since=_T0
            )

    def test_an_assignment_covering_every_engine_fails_the_run(self):
        """With nothing outside the assignment the witness could never distinguish contained from total damage."""
        log = _healthy_run_log(trainer_hashes=[_ARMED_HASH, "trainer-generation-1"])

        with pytest.raises(AssertionError, match="was assigned to the harmed sender"):
            hook_injection.assert_unrelated_target_kept_serving(
                log.events,
                armed=_armed(inference=dict(_ASSIGNED)),
                assignment=_assignment(),
                since=_T0,
            )

    def test_an_engine_outside_the_assignment_that_kept_serving_passes(self):
        """The engines the harmed sender did not own carried on under the incarnation they already had."""
        log = _healthy_run_log(trainer_hashes=[_ARMED_HASH, "trainer-generation-1"])

        survivor = hook_injection.assert_unrelated_target_kept_serving(
            log.events, armed=_armed(), assignment=_assignment(), since=_T0
        )

        assert survivor in _UNRELATED


class TestProgressWitness:
    def test_no_publication_after_the_fault_fails_the_run(self, tmp_path: Path):
        """A run that stops publishing after the fault has not survived it, however cleanly it exits."""
        _write_events(tmp_path, file_name="main.jsonl", events=[_publication(at=_at(1), checksums=[{"w": "h"}])])

        with pytest.raises(AssertionError, match="no non-empty weight publication"):
            hook_injection.assert_weights_published_after(tmp_path, after=_at(5))

    def test_an_empty_publication_after_the_fault_is_not_progress(self, tmp_path: Path):
        """An engine list that is empty, or full of empty dicts, records that nothing was pushed anywhere."""
        _write_events(
            tmp_path,
            file_name="main.jsonl",
            events=[_publication(at=_at(9), checksums=[]), _publication(at=_at(10), checksums=[{}, {}])],
        )

        with pytest.raises(AssertionError, match="no non-empty weight publication"):
            hook_injection.assert_weights_published_after(tmp_path, after=_at(5))

    def test_a_non_empty_publication_after_the_fault_passes(self, tmp_path: Path):
        """Weights reaching an engine after the fault is the evidence that the update path recovered."""
        _write_events(tmp_path, file_name="main.jsonl", events=[_publication(at=_at(9), checksums=[{"w": "h"}])])

        hook_injection.assert_weights_published_after(tmp_path, after=_at(5))


_VICTIM = "rollout-engine-00000"
_VICTIM_HASH = "engine-generation-0"


def _remote_armed(**kwargs) -> hook_injection.ArmedFaultHook:
    return _armed(
        hook=FaultHookName.WEIGHT_UPDATE_AFTER_P2P_SUBMIT,
        target=FaultHookTarget.REMOTE_INFERENCE_CELL,
        **kwargs,
    )


def _remote_fire(*, victim=(_VICTIM, _VICTIM_HASH), outcome=FaultHookOutcome.ACCEPTED) -> FaultHookFireEvent:
    return _fire(
        hook=FaultHookName.WEIGHT_UPDATE_AFTER_P2P_SUBMIT.value,
        target=FaultHookTarget.REMOTE_INFERENCE_CELL,
        outcome=outcome,
        victim=victim,
    )


class TestRemoteFireWitness:
    @pytest.mark.parametrize(
        "outcome", [FaultHookOutcome.STALE_TARGET, FaultHookOutcome.UNKNOWN, FaultHookOutcome.REFUSED]
    )
    def test_an_answer_that_is_not_an_acceptance_fails_the_run(self, tmp_path: Path, outcome: FaultHookOutcome):
        """A stale, unknown or refused answer means the receiver of this transfer was never asked to die."""
        _write_events(tmp_path, file_name="actor.jsonl", events=[_remote_fire(outcome=outcome)])

        with pytest.raises(AssertionError, match="not the accepted"):
            hook_injection.assert_hook_fired(_remote_armed(expected_source=_source()), event_dir=tmp_path)

    def test_an_accepted_remote_answer_passes_the_fire_witness(self, tmp_path: Path):
        """Accepting is what the receiver records; the harm itself is asserted by the witnesses that follow."""
        _write_events(tmp_path, file_name="actor.jsonl", events=[_remote_fire()])

        fire = hook_injection.assert_hook_fired(_remote_armed(expected_source=_source()), event_dir=tmp_path)

        assert fire.outcome == FaultHookOutcome.ACCEPTED.value

    def test_a_local_fire_outcome_does_not_satisfy_a_remote_request(self, tmp_path: Path):
        """A local fault kills the trainer; recording it as the remote one would credit the wrong process."""
        _write_events(tmp_path, file_name="actor.jsonl", events=[_remote_fire(outcome=FaultHookOutcome.FIRED)])

        with pytest.raises(AssertionError, match="not the accepted"):
            hook_injection.assert_hook_fired(_remote_armed(expected_source=_source()), event_dir=tmp_path)

    def test_a_local_fire_does_not_satisfy_a_remote_request(self, tmp_path: Path):
        """The two actions harm different processes, so one must never be read as the other."""
        _write_events(
            tmp_path, file_name="actor.jsonl", events=[_fire(hook=FaultHookName.WEIGHT_UPDATE_AFTER_P2P_SUBMIT.value)]
        )

        with pytest.raises(AssertionError, match="fired against"):
            hook_injection.assert_hook_fired(_remote_armed(expected_source=_source()), event_dir=tmp_path)


class TestRemoteVictimWitness:
    def test_a_fire_without_receiver_identity_fails_the_run(self):
        """A victim named only by cell id could have been chosen from a listing rather than from the write."""
        log = _healthy_run_log(trainer_hashes=[_ARMED_HASH, _ARMED_HASH])

        with pytest.raises(AssertionError, match="carries no receiver incarnation"):
            hook_injection.assert_remote_victim_was_harmed(
                _remote_fire(receiver_rank=None),
                log.events,
                armed=_remote_armed(),
                assignment=_assignment(),
            )

    def test_a_fire_naming_no_victim_fails_the_run(self):
        """Without a victim there is nothing to hold harmed, and a green run would prove only that a hook ran."""
        log = _healthy_run_log(trainer_hashes=[_ARMED_HASH])

        with pytest.raises(AssertionError, match="names no target"):
            hook_injection.assert_remote_victim_was_harmed(
                _remote_fire(victim=None), log.events, armed=_remote_armed(), assignment=_assignment()
            )

    def test_a_victim_outside_this_updates_assignment_fails_the_run(self):
        """A cell this sender was not writing to in this update is not a target of this write."""
        log = _healthy_run_log(trainer_hashes=[_ARMED_HASH])

        with pytest.raises(AssertionError, match="not what the update assigned"):
            hook_injection.assert_remote_victim_was_harmed(
                _remote_fire(victim=("rollout-engine-00003", _VICTIM_HASH)),
                log.events,
                armed=_remote_armed(),
                assignment=_assignment(),
            )

    def test_a_victim_incarnation_that_was_not_in_service_fails_the_run(self):
        """An incarnation nobody observed serving cannot be shown to have been taken out of service by this fault."""
        log = _healthy_run_log(trainer_hashes=[_ARMED_HASH])
        armed = _remote_armed(inference={**_ASSIGNED, **_UNRELATED, _VICTIM: "engine-generation-9"})

        with pytest.raises(AssertionError, match="was not running"):
            hook_injection.assert_remote_victim_was_harmed(
                _remote_fire(), log.events, armed=armed, assignment=_assignment()
            )

    def test_a_victim_still_running_that_incarnation_fails_the_run(self):
        """The engine the write reached has to lose the incarnation it was reached at."""
        log = _healthy_run_log(
            trainer_hashes=[_ARMED_HASH, _ARMED_HASH],
            assigned_after=dict(_ASSIGNED),
        )

        with pytest.raises(AssertionError, match="still runs"):
            hook_injection.assert_remote_victim_was_harmed(
                _remote_fire(), log.events, armed=_remote_armed(), assignment=_assignment()
            )

    def test_the_named_incarnation_being_replaced_passes(self):
        """This is the claim: the target of this very write lost the incarnation the write reached it at."""
        log = _healthy_run_log(trainer_hashes=[_ARMED_HASH, _ARMED_HASH])

        victim = hook_injection.assert_remote_victim_was_harmed(
            _remote_fire(), log.events, armed=_remote_armed(), assignment=_assignment()
        )

        assert victim == _VICTIM

    def test_a_victim_that_never_served_again_fails_the_run(self):
        """A harmed engine that never comes back leaves the fleet a replica short."""
        log = EventLog()
        _observe(log, trainer_hash=_ARMED_HASH, engines={**_ASSIGNED, **_UNRELATED})
        log.observe(
            [
                cell(_ARMED_CELL, healthy=True, cell_type="actor", workers_hash=_ARMED_HASH),
                staged(_VICTIM, RUNNING_NOT_SERVING, workers_hash="engine-generation-1"),
                staged("rollout-engine-00001", SERVING, workers_hash=_VICTIM_HASH),
                *[staged(name, SERVING, workers_hash=h) for name, h in _UNRELATED.items()],
            ]
        )

        with pytest.raises(AssertionError, match="never observed healthy and Serving under a replacement"):
            hook_injection.assert_remote_victim_recovered(log.events, fire=_remote_fire(), since=_T0)

    def test_a_victim_serving_again_under_a_replacement_passes(self):
        """Recovery of an engine is a Serving reading under a generation the killed one cannot produce."""
        log = _healthy_run_log(trainer_hashes=[_ARMED_HASH, _ARMED_HASH])

        hook_injection.assert_remote_victim_recovered(log.events, fire=_remote_fire(), since=_T0)

    def test_a_victim_that_never_changed_incarnation_has_no_harm_moment(self):
        """Without an observation of the replacement there is no point in time after the harm to read anything at."""
        log = _healthy_run_log(trainer_hashes=[_ARMED_HASH], assigned_after=dict(_ASSIGNED))

        with pytest.raises(AssertionError, match="never observed under an incarnation other than"):
            hook_injection.compute_victim_harm_observed_at(log.events, fire=_remote_fire(), armed=_remote_armed())


class TestScenarioRecipe:
    def test_the_run_enables_the_receiver_fault_control_and_the_transfer_engine_seed(self):
        """Without both flags the engines publish no receiver identity and no remote fault can name one."""
        assert "--sglang-enable-p2p-fault-injection " in hook_injection.P2P_WEIGHT_TRANSFER_ARGS
        assert (
            "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine "
            in hook_injection.P2P_WEIGHT_TRANSFER_ARGS
        )
        assert "--update-weight-transfer-mode p2p " in hook_injection.P2P_WEIGHT_TRANSFER_ARGS

    def test_no_cluster_backend_is_ruled_out(self):
        """Both backends reach the receiver through its own endpoint, so neither is refused up front any more."""
        assert not hasattr(hook_injection, "assert_backend_binds_faults_to_incarnations")

    def test_a_remote_request_expects_an_acceptance_and_a_local_one_a_fire(self):
        """The two actions are delivered by different processes and record different evidence of delivery."""
        assert hook_injection.EXPECTED_OUTCOME_OF_TARGET == {
            FaultHookTarget.LOCAL: FaultHookOutcome.FIRED,
            FaultHookTarget.REMOTE_INFERENCE_CELL: FaultHookOutcome.ACCEPTED,
        }
