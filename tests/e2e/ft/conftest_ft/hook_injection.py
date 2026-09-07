# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

from __future__ import annotations

import dataclasses
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import requests
from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.execution import P2P_FAULT_INJECTION_ARGS
from tests.e2e.ft.conftest_ft.execution import P2P_WEIGHT_TRANSFER_ARGS as BASE_P2P_WEIGHT_TRANSFER_ARGS
from tests.e2e.ft.conftest_ft.execution import (
    get_api_server_args,
    get_common_train_args,
    get_ft_args,
    get_train_script,
    prepare,
    run_training,
)
from tests.e2e.ft.conftest_ft.fault_injection.core import list_cells
from tests.e2e.ft.conftest_ft.fault_injection.entrypoint import API_SERVER_PORT
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE
from tests.e2e.ft.conftest_ft.fault_injection.hook_forms import (
    ARM_REQUEST_TIMEOUT_SECONDS,
    ARMED_TRAINER_MODEL_ID,
    EXPECTED_OUTCOME_OF_TARGET,
)
from tests.e2e.ft.conftest_ft.fault_injection.recovery_source import (
    CHECKPOINT_DIRNAME,
    carries_checksums,
    recovery_source_exists,
)
from tests.e2e.ft.conftest_ft.fault_injection.state import (
    CellInfo,
    Event,
    EventLog,
    ObservationsEvent,
    ObservedCellState,
    cell_is_alive,
    cell_type_of,
    cell_workers_hash,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode

from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.ray.specs.train import compute_trainer_pool_id
from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.audit_utils.event_logger.models import (
    CellReconfigureEvent,
    FaultHookFireEvent,
    InferenceEngineWeightChecksumEvent,
    WeightUpdateAssignmentEvent,
)
from miles.utils.audit_utils.process_identity import TrainProcessIdentity
from miles.utils.external_utils import command_utils
from miles.utils.test_utils.fault_hooks import FaultHookName, FaultHookTarget
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.test_utils.polling_worker import PollingWorker
from miles.utils.workers.naming import compute_cell_id, parse_cell_id

logger = logging.getLogger(__name__)

P2P_WEIGHT_TRANSFER_ARGS: str = f"{BASE_P2P_WEIGHT_TRANSFER_ARGS}{P2P_FAULT_INJECTION_ARGS}"

POLL_INTERVAL_SECONDS: float = 2.0
STOP_AND_JOIN_TIMEOUT_SECONDS: float = 120.0


# ============================== scenario runner ==============================


@dataclasses.dataclass(frozen=True)
class TargetedHookRun:
    armer: HookArmer
    ft_mode: FTTestMode
    dump_dir: str
    armed_cell_id: str

    @property
    def event_dir(self) -> Path:
        return Path(self.dump_dir) / EVENTS_DIRNAME

    @property
    def events(self) -> list[Event]:
        return self.armer.event_log.events


def run_targeted_hook_scenario(
    *,
    test_name: str,
    mode: str,
    num_steps: int,
    hook: FaultHookName,
    failure_mode: FailureMode,
    target: FaultHookTarget,
    request_id: str,
    sub_index: int,
    delay_ms: int = 0,
) -> TargetedHookRun:
    ft_mode = resolve_mode(mode)
    assert_mode_reaches_the_hooks(ft_mode, mode=mode)

    config = command_utils.default_config()
    dump_dir: str = resolve_dump_dir(f"{test_name}_{mode}", run_id=config.run_id)
    print(f"Dump directory: {dump_dir}")
    print(f"Steps: {num_steps}, cluster backend: {config.cluster_backend.value}")
    print(f"Arming {hook.value} ({failure_mode.value}, {target.value}, delay {delay_ms}ms) as {request_id!r}")

    prepare(ft_mode, config=config)

    armed_cell_id: str = compute_cell_id(pool_id=compute_trainer_pool_id(ACTOR_ROLE), cell_index=ft_mode.num_cells - 1)
    train_args = (
        get_common_train_args(ft_mode, dump_dir=dump_dir, num_steps=num_steps)
        + get_ft_args(ft_mode)
        + get_api_server_args(config)
        + P2P_WEIGHT_TRANSFER_ARGS
        + f"--save {dump_dir}/{CHECKPOINT_DIRNAME} --save-interval 1 "
        + "--mini-ft-controller-enable "
    )

    armer = HookArmer(
        base_url=f"http://{config.create_backend().api_server_host(config)}:{API_SERVER_PORT}",
        cell_types={ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE},
        event_dir=Path(dump_dir) / EVENTS_DIRNAME,
        checkpoint_dir=Path(dump_dir) / CHECKPOINT_DIRNAME,
        cell_name=armed_cell_id,
        sub_index=sub_index,
        hook=hook,
        mode=failure_mode,
        target=target,
        request_id=request_id,
        delay_ms=delay_ms,
    )
    armer.start()

    try:
        run_training(
            train_args=train_args,
            mode=ft_mode,
            dump_dir=dump_dir,
            extra_env_vars={},
            config=config,
            train_script=get_train_script(fully_async=False),
        )
    finally:
        armer.stop_and_join()

    return TargetedHookRun(armer=armer, ft_mode=ft_mode, dump_dir=dump_dir, armed_cell_id=armed_cell_id)


def assert_mode_reaches_the_hooks(ft_mode: FTTestMode, *, mode: str) -> None:
    assert ft_mode.has_real_rollout and not ft_mode.colocate, (
        f"Mode {mode!r} has no disaggregated rollout engines, so the p2p weight update these scenarios crash inside "
        f"would never run"
    )
    assert "--tensor-model-parallel-size 2" in ft_mode.parallel_args, (
        f"Mode {mode!r} does not shard the trainer over tensor parallelism, so the weight update performs no "
        f"cross-rank all-gather and the hook that sits before it is never reached"
    )
    assert ft_mode.ft_components == ("train", "rollout"), (
        f"Mode {mode!r} enables ft on {ft_mode.ft_components}: these faults cost a trainer cell or an engine its "
        f"incarnation, so both kinds have to be recoverable for the run to prove recovery"
    )
    assert (
        ft_mode.num_cells >= 2
    ), f"Mode {mode!r} runs {ft_mode.num_cells} trainer cell(s), leaving no survivor to heal the crashed one from"
    assert ft_mode.rollout_num_engines >= 2, (
        f"Mode {mode!r} runs {ft_mode.rollout_num_engines} engine(s), so no unrelated target could show that the "
        f"damage stayed inside the one the fault named"
    )


# ============================== arming ==============================


@dataclasses.dataclass(frozen=True)
class CellSnapshot:
    taken_at: datetime
    workers_hash_of_cell_id: dict[str, str]
    alive_cell_ids: frozenset[str]


@dataclasses.dataclass(frozen=True)
class ArmedFaultHook:
    cell_name: str
    sub_index: int
    hook: FaultHookName
    mode: FailureMode
    target: FaultHookTarget
    request_id: str
    delay_ms: int
    expected_source: TrainProcessIdentity
    trainer_workers_hash: str
    inference_workers_hash_of_cell_id: dict[str, str]
    snapshot_at: datetime
    acknowledged_at: datetime


def compute_cell_snapshot(cells: list[dict], *, cell_type: str) -> CellSnapshot:
    matching = [cell for cell in cells if cell_type_of(cell) == cell_type]
    return CellSnapshot(
        taken_at=datetime.now(timezone.utc),
        workers_hash_of_cell_id={cell["metadata"]["name"]: cell_workers_hash(cell) for cell in matching},
        alive_cell_ids=frozenset(cell["metadata"]["name"] for cell in matching if cell_is_alive(cell)),
    )


def arm_fault_hook_over_api(
    *,
    base_url: str,
    cell_name: str,
    sub_index: int,
    hook: FaultHookName,
    mode: FailureMode,
    target: FaultHookTarget,
    request_id: str,
    delay_ms: int,
    trainer_snapshot: CellSnapshot,
    inference_snapshot: CellSnapshot,
) -> ArmedFaultHook:
    trainer_workers_hash = trainer_snapshot.workers_hash_of_cell_id[cell_name]
    response = requests.post(
        f"{base_url}/api/v1/cells/{cell_name}/arm-fault-hook",
        json={
            "expected_workers_hash": trainer_workers_hash,
            "hook": hook.value,
            "mode": mode.value,
            "target": target.value,
            "sub_index": sub_index,
            "request_id": request_id,
            "delay_ms": delay_ms,
        },
        timeout=ARM_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return ArmedFaultHook(
        cell_name=cell_name,
        sub_index=sub_index,
        hook=hook,
        mode=mode,
        target=target,
        request_id=request_id,
        delay_ms=delay_ms,
        expected_source=TrainProcessIdentity(
            component=ACTOR_ROLE,
            model_id=ARMED_TRAINER_MODEL_ID,
            cell_index=parse_cell_id(cell_name).cell_index,
            rank_within_cell=sub_index,
        ),
        trainer_workers_hash=trainer_workers_hash,
        inference_workers_hash_of_cell_id=dict(inference_snapshot.workers_hash_of_cell_id),
        snapshot_at=trainer_snapshot.taken_at,
        acknowledged_at=datetime.now(timezone.utc),
    )


class HookArmer:
    def __init__(
        self,
        *,
        base_url: str,
        cell_types: set[str],
        event_dir: Path,
        checkpoint_dir: Path,
        cell_name: str,
        sub_index: int,
        hook: FaultHookName,
        mode: FailureMode,
        target: FaultHookTarget,
        request_id: str,
        delay_ms: int = 0,
        poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    ) -> None:
        self.event_log = EventLog()
        self.armed: ArmedFaultHook | None = None
        self._base_url = base_url
        self._cell_types = cell_types
        self._event_dir = event_dir
        self._checkpoint_dir = checkpoint_dir
        self._cell_name = cell_name
        self._sub_index = sub_index
        self._hook = hook
        self._mode = mode
        self._target = target
        self._request_id = request_id
        self._delay_ms = delay_ms

        def observe_and_arm(stop_event: threading.Event) -> None:
            while not stop_event.wait(timeout=poll_interval_seconds):
                cells = self._observe_once()
                if self.armed is None and cells is not None:
                    self.armed = self._try_arm(cells)

        self._worker = PollingWorker(name="ft-fault-hook-armer", run=observe_and_arm)

    def start(self) -> None:
        self._worker.start()

    def stop_and_join(self) -> None:
        self._worker.stop_and_join(timeout_seconds=STOP_AND_JOIN_TIMEOUT_SECONDS)
        self._worker.assert_not_running(
            message=(
                f"The fault hook armer was still mid-request {STOP_AND_JOIN_TIMEOUT_SECONDS}s after being asked to "
                f"stop, so a fault could still be armed after the run it was meant to hit has ended"
            )
        )
        self._observe_once()

    def _try_arm(self, cells: list[dict]) -> ArmedFaultHook | None:
        if not recovery_source_exists(event_dir=self._event_dir, checkpoint_dir=self._checkpoint_dir):
            return None

        trainer_snapshot = compute_cell_snapshot(cells, cell_type=ACTOR_CELL_TYPE)
        if self._cell_name not in trainer_snapshot.alive_cell_ids:
            logger.info("Not arming yet: %s is not observed healthy", self._cell_name)
            return None

        armed = arm_fault_hook_over_api(
            base_url=self._base_url,
            cell_name=self._cell_name,
            sub_index=self._sub_index,
            hook=self._hook,
            mode=self._mode,
            target=self._target,
            request_id=self._request_id,
            delay_ms=self._delay_ms,
            trainer_snapshot=trainer_snapshot,
            inference_snapshot=compute_cell_snapshot(cells, cell_type=ROLLOUT_CELL_TYPE),
        )
        logger.info("Armed fault hook %s", armed)
        return armed

    def _observe_once(self) -> list[dict] | None:
        cells = list_cells(base_url=self._base_url, cell_types=self._cell_types)
        if cells is not None:
            self.event_log.observe(cells)
        return cells


# ============================== fire witnesses ==============================


def load_fire_events(event_dir: Path, *, request_id: str) -> list[FaultHookFireEvent]:
    return [
        event
        for event in read_events(event_dir)
        if isinstance(event, FaultHookFireEvent) and event.request_id == request_id
    ]


def require_armed(armer: HookArmer) -> ArmedFaultHook:
    armed = armer.armed
    assert armed is not None, (
        "No fault hook was ever armed: the run never reached a state where a completed training step, a checkpoint "
        "and a non-empty publication existed to recover from, so nothing about the hook under test was exercised"
    )
    return armed


def assert_hook_fired(armed: ArmedFaultHook, *, event_dir: Path) -> FaultHookFireEvent:
    fires = load_fire_events(event_dir, request_id=armed.request_id)
    assert len(fires) == 1, (
        f"Fault hook witness failed: request {armed.request_id!r} armed {armed.hook.value} in {armed.cell_name} "
        f"worker {armed.sub_index} but production reached it {len(fires)} time(s); an arm that never fires proves "
        f"nothing, and arming is not injecting ({fires})"
    )

    (fire,) = fires
    assert fire.hook == armed.hook.value and fire.mode == armed.mode.value, (
        f"Fault hook witness failed: request {armed.request_id!r} fired as {fire.hook}/{fire.mode}, "
        f"not the {armed.hook.value}/{armed.mode.value} it was armed for"
    )
    assert fire.source == armed.expected_source, (
        f"Fault hook witness failed: request {armed.request_id!r} fired in {fire.source}, not in "
        f"{armed.expected_source}; a fault in another role, another trained model or another cell must not pay for "
        f"the one that was armed"
    )
    assert fire.weight_version is not None, (
        f"Fault hook witness failed: request {armed.request_id!r} fired outside any weight update, so nothing names "
        f"the update whose targets the run then holds responsible"
    )
    assert fire.target == armed.target.value, (
        f"Fault hook witness failed: request {armed.request_id!r} fired against {fire.target}, not the "
        f"{armed.target.value} it was armed for"
    )
    assert fire.delay_ms == armed.delay_ms, (
        f"Fault hook witness failed: request {armed.request_id!r} fired after a {fire.delay_ms}ms delay, not the "
        f"{armed.delay_ms}ms it was armed for, so the fault landed at a moment the scenario did not ask about"
    )
    expected_outcome = EXPECTED_OUTCOME_OF_TARGET[armed.target]
    assert fire.outcome == expected_outcome.value, (
        f"Fault hook witness failed: request {armed.request_id!r} reached its point and answered {fire.outcome}, not "
        f"the {expected_outcome.value} a delivered {armed.target.value} fault records; a refusal, an unknown answer "
        f"or an incarnation that was already gone harmed nobody"
    )
    print(f"Fault hook fire witness passed: {fire.hook} fired once in {fire.source} for request {armed.request_id!r}")
    return fire


def resolve_fire_assignment(
    fire: FaultHookFireEvent, armed: ArmedFaultHook, *, event_dir: Path
) -> WeightUpdateAssignmentEvent:
    assignments = [
        event
        for event in read_events(event_dir)
        if isinstance(event, WeightUpdateAssignmentEvent)
        and event.weight_version == fire.weight_version
        and event.trainer_cell_id == armed.cell_name
    ]
    assert assignments, (
        f"Assignment witness failed: no weight update of version {fire.weight_version} ever assigned targets to "
        f"{armed.cell_name}, so the fault cannot be held against any set of engines"
    )
    assert len(assignments) == 1, (
        f"Assignment witness failed: version {fire.weight_version} was assigned to {armed.cell_name} "
        f"{len(assignments)} times, so which attempt the fault landed in is unknown "
        f"({[event.trainer_workers_hash for event in assignments]})"
    )

    (assignment,) = assignments
    assert assignment.trainer_workers_hash == armed.trainer_workers_hash, (
        f"Assignment witness failed: version {fire.weight_version} was sent by {assignment.trainer_workers_hash} of "
        f"{armed.cell_name}, not by the {armed.trainer_workers_hash} that was armed, so the fault fired in a "
        f"replacement rather than in the incarnation the request named"
    )
    print(
        f"Assignment witness passed: version {fire.weight_version} was sent by the armed incarnation "
        f"{assignment.trainer_workers_hash} to {sorted(assignment.assigned_workers_hash_of_cell_id)}"
    )
    return assignment


# ============================== recovery witnesses ==============================


def compute_incarnations_of_cell(events: list[Event], *, cell_type: str) -> dict[str, list[str]]:
    incarnations: dict[str, list[str]] = {}
    for event in events:
        if not isinstance(event, ObservationsEvent):
            continue
        for name, info in event.cell_infos.items():
            if info.cell_type != cell_type:
                continue
            seen = incarnations.setdefault(name, [])
            if not seen or seen[-1] != info.workers_hash:
                seen.append(info.workers_hash)
    return incarnations


def compute_harm_observed_at(events: list[Event], *, armed: ArmedFaultHook) -> datetime:
    for event in events:
        if not isinstance(event, ObservationsEvent) or event.timestamp < armed.snapshot_at:
            continue
        info = event.cell_infos.get(armed.cell_name)
        if info is not None and info.workers_hash != armed.trainer_workers_hash:
            return event.timestamp

    raise AssertionError(
        f"Eviction witness failed: {armed.cell_name} was never observed under an incarnation other than the armed "
        f"{armed.trainer_workers_hash}, so the fault the run reports as fired cost it nothing "
        f"(observed: {compute_incarnations_of_cell(events, cell_type=ACTOR_CELL_TYPE)})"
    )


def assert_armed_trainer_was_replaced(events: list[Event], *, armed: ArmedFaultHook) -> datetime:
    harm_observed_at = compute_harm_observed_at(events, armed=armed)

    assert _was_alive_after(
        events, cell_name=armed.cell_name, other_than=armed.trainer_workers_hash, since=harm_observed_at
    ), (
        f"Recovery witness failed: {armed.cell_name} was never observed healthy under a replacement of the armed "
        f"{armed.trainer_workers_hash}, so the run ended with the crashed cell unreplaced"
    )

    last = _last_observation_of(events, cell_name=armed.cell_name)
    assert last is not None and not (last.workers_hash == armed.trainer_workers_hash and last.alive), (
        f"Eviction witness failed: {armed.cell_name} is still running the armed incarnation "
        f"{armed.trainer_workers_hash} at the end of the run, so the fault left its target alive ({last})"
    )
    print(
        f"Trainer eviction witness passed: {armed.cell_name} left {armed.trainer_workers_hash} and came back healthy"
    )
    return harm_observed_at


def assert_trainer_cell_healed(event_dir: Path, *, assignment: WeightUpdateAssignmentEvent) -> None:
    reconfigures = [
        event
        for event in read_events(event_dir)
        if isinstance(event, CellReconfigureEvent) and event.timestamp > assignment.timestamp
    ]
    cell_index = assignment.trainer_cell_index

    evictions = [event for event in reconfigures if cell_index not in event.alive_cell_indices_after]
    assert evictions, (
        f"Eviction witness failed: no reconfigure after the update the fault fired in dropped cell index "
        f"{cell_index} from the alive set, so an earlier unrelated eviction is all this run can show "
        f"({reconfigures})"
    )

    healings = [
        event
        for event in reconfigures
        if event.timestamp > evictions[0].timestamp and cell_index in event.healed_cell_indices
    ]
    assert healings, (
        f"Healing witness failed: cell index {cell_index} was never healed back after the eviction that followed the "
        f"fault, so the run ended a replica short ({reconfigures})"
    )
    print(f"Trainer healing witness passed: cell index {cell_index} was evicted and then healed after the fault")


def assert_assigned_targets_isolated(
    events: list[Event], *, assignment: WeightUpdateAssignmentEvent, since: datetime
) -> None:
    incarnations = compute_incarnations_of_cell(events, cell_type=ROLLOUT_CELL_TYPE)

    for cell_id, workers_hash in sorted(assignment.assigned_workers_hash_of_cell_id.items()):
        observed = incarnations.get(cell_id, [])
        assert workers_hash in observed, (
            f"Isolation witness failed: {cell_id} was never observed running {workers_hash}, the incarnation the "
            f"harmed sender was writing to (observed: {incarnations})"
        )
        last = _last_observation_of(events, cell_name=cell_id)
        assert last is not None and not (last.workers_hash == workers_hash and last.alive), (
            f"Isolation witness failed: {cell_id} still runs {workers_hash} at the end of the run, so a target of "
            f"the sender that died mid-update was never taken out of service ({last})"
        )
        assert _was_serving_after(events, cell_name=cell_id, other_than=workers_hash, since=since), (
            f"Isolation witness failed: {cell_id} was never observed healthy and Serving under a replacement of "
            f"{workers_hash}, so the run ended with one of the harmed sender's targets missing "
            f"(observed: {incarnations})"
        )

    print(
        f"Isolation witness passed: every assigned target {sorted(assignment.assigned_workers_hash_of_cell_id)} lost "
        f"the incarnation the harmed sender wrote to and served again under a replacement"
    )


def assert_unrelated_target_kept_serving(
    events: list[Event], *, armed: ArmedFaultHook, assignment: WeightUpdateAssignmentEvent, since: datetime
) -> str:
    unrelated = {
        cell_id: workers_hash
        for cell_id, workers_hash in armed.inference_workers_hash_of_cell_id.items()
        if cell_id not in assignment.assigned_workers_hash_of_cell_id
    }
    assert unrelated, (
        f"Blast-radius witness failed: every engine observed when the hook was armed "
        f"({sorted(armed.inference_workers_hash_of_cell_id)}) was assigned to the harmed sender, so no unrelated "
        f"target could show that the damage stayed inside the assignment"
    )

    survivors = sorted(
        cell_id
        for cell_id, workers_hash in unrelated.items()
        if _was_serving_after(events, cell_name=cell_id, exactly=workers_hash, since=since)
        and _kept_incarnation(events, cell_name=cell_id, workers_hash=workers_hash)
    )
    assert survivors, (
        f"Blast-radius witness failed: none of the engines {sorted(unrelated)} outside the harmed sender's "
        f"assignment kept the incarnation it had when the hook was armed and went on serving, so the fault was not "
        f"confined to the targets that sender was writing to"
    )
    print(f"Blast-radius witness passed: {survivors} kept serving the incarnation they had when the hook was armed")
    return survivors[0]


def assert_weights_published_after(event_dir: Path, *, after: datetime) -> None:
    published = [
        event
        for event in read_events(event_dir)
        if isinstance(event, InferenceEngineWeightChecksumEvent)
        and event.timestamp > after
        and carries_checksums(event)
    ]
    assert published, (
        f"Progress witness failed: no non-empty weight publication reached an engine after {after.isoformat()}, so "
        f"the run did not go on updating weights past the injected fault (events in {event_dir})"
    )
    print(f"Progress witness passed: {len(published)} non-empty weight publication(s) after {after.isoformat()}")


# ============================== observation utils ==============================


def _last_observation_of(events: list[Event], *, cell_name: str) -> CellInfo | None:
    seen: CellInfo | None = None
    for event in events:
        if isinstance(event, ObservationsEvent) and (info := event.cell_infos.get(cell_name)) is not None:
            seen = info
    return seen


def _kept_incarnation(events: list[Event], *, cell_name: str, workers_hash: str) -> bool:
    last = _last_observation_of(events, cell_name=cell_name)
    return last is not None and last.workers_hash == workers_hash


def _observations_of_since(events: list[Event], *, cell_name: str, since: datetime) -> list[CellInfo]:
    return [
        info
        for event in events
        if isinstance(event, ObservationsEvent) and event.timestamp >= since
        for name, info in event.cell_infos.items()
        if name == cell_name
    ]


def _was_alive_after(events: list[Event], *, cell_name: str, other_than: str, since: datetime) -> bool:
    return any(
        info.alive and info.workers_hash != other_than
        for info in _observations_of_since(events, cell_name=cell_name, since=since)
    )


def _was_serving_after(
    events: list[Event],
    *,
    cell_name: str,
    since: datetime,
    exactly: str | None = None,
    other_than: str | None = None,
) -> bool:
    for info in _observations_of_since(events, cell_name=cell_name, since=since):
        if not info.alive or info.state is not ObservedCellState.SERVING:
            continue
        if exactly is not None and info.workers_hash != exactly:
            continue
        if other_than is not None and info.workers_hash == other_than:
            continue
        return True
    return False


# ============================== remote victim witnesses ==============================


def assert_remote_victim_was_harmed(
    fire: FaultHookFireEvent, events: list[Event], *, armed: ArmedFaultHook, assignment: WeightUpdateAssignmentEvent
) -> str:
    victim = fire.victim_cell_id
    assert victim is not None and fire.victim_workers_hash is not None, (
        f"Remote fault witness failed: the fire of request {fire.request_id!r} names no target, so no cell can be "
        f"held to have been harmed"
    )
    assert assignment.assigned_workers_hash_of_cell_id.get(victim) == fire.victim_workers_hash, (
        f"Remote fault witness failed: the write named {victim} at {fire.victim_workers_hash}, which is not what the "
        f"update assigned this sender ({assignment.assigned_workers_hash_of_cell_id})"
    )
    assert fire.victim_receiver_boot_uuid and fire.victim_session_id and fire.victim_receiver_rank is not None, (
        f"Remote fault witness failed: the fire of request {fire.request_id!r} carries no receiver incarnation "
        f"(uuid={fire.victim_receiver_boot_uuid}, session={fire.victim_session_id}, "
        f"rank={fire.victim_receiver_rank}), so the fault was aimed at a cell name rather than at the process this "
        f"transfer reached"
    )
    assert armed.inference_workers_hash_of_cell_id.get(victim) == fire.victim_workers_hash, (
        f"Remote fault witness failed: {victim} was not running {fire.victim_workers_hash} when the hook was armed, "
        f"so the fault was aimed at an incarnation this run never observed in service "
        f"({armed.inference_workers_hash_of_cell_id})"
    )

    incarnations = compute_incarnations_of_cell(events, cell_type=ROLLOUT_CELL_TYPE)
    last = _last_observation_of(events, cell_name=victim)
    assert last is not None and not (last.workers_hash == fire.victim_workers_hash and last.alive), (
        f"Remote fault witness failed: {victim} still runs {fire.victim_workers_hash} at the end of the run, so the "
        f"write's own target survived the fault aimed at it (observed: {incarnations})"
    )
    print(f"Remote fault witness passed: {victim} lost the incarnation {fire.victim_workers_hash} the write reached")
    return victim


def assert_remote_victim_recovered(events: list[Event], *, fire: FaultHookFireEvent, since: datetime) -> None:
    victim = fire.victim_cell_id
    assert _was_serving_after(events, cell_name=victim, other_than=fire.victim_workers_hash, since=since), (
        f"Remote fault witness failed: {victim} was never observed healthy and Serving under a replacement of "
        f"{fire.victim_workers_hash}, so the run ended with the harmed engine missing"
    )
    print(f"Remote recovery witness passed: {victim} served again under a replacement of {fire.victim_workers_hash}")


def compute_victim_harm_observed_at(
    events: list[Event], *, fire: FaultHookFireEvent, armed: ArmedFaultHook
) -> datetime:
    for event in events:
        if not isinstance(event, ObservationsEvent) or event.timestamp < armed.snapshot_at:
            continue
        info = event.cell_infos.get(fire.victim_cell_id)
        if info is not None and info.workers_hash != fire.victim_workers_hash:
            return event.timestamp

    raise AssertionError(
        f"Remote fault witness failed: {fire.victim_cell_id} was never observed under an incarnation other than the "
        f"{fire.victim_workers_hash} the write reached, so the fault the run reports as delivered cost it nothing"
    )
