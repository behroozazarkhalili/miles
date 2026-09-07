# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import abc
import dataclasses
import logging
import random
from pathlib import Path

import requests
from tests.e2e.ft.conftest_ft.fault_injection.core import list_fault_hook_sources
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import BaseFaultForm, CellFaultForms
from tests.e2e.ft.conftest_ft.fault_injection.recovery_source import recovery_source_exists
from tests.e2e.ft.conftest_ft.fault_injection.state import (
    ACTOR_CELL_TYPE,
    ROLLOUT_CELL_TYPE,
    EventLog,
    HookFireEvent,
    WeightPublicationEvent,
    cell_is_alive,
    cell_is_allocated,
    cell_workers_hash,
)
from tests.e2e.ft.conftest_ft.fault_injection.views import compute_hook_harms

from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import (
    FaultHookFireEvent,
    InferenceEngineWeightChecksumEvent,
    WeightUpdateAssignmentEvent,
)
from miles.utils.audit_utils.process_identity import TrainProcessIdentity
from miles.utils.test_utils.fault_hooks import (
    MAX_FAULT_HOOK_DELAY_MS,
    REMOTE_CAPABLE_HOOKS,
    FaultHookName,
    FaultHookOutcome,
    FaultHookTarget,
)
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.test_utils.receiver_fault import RECEIVER_SUPPORTED_MODES

logger = logging.getLogger(__name__)

LOCAL_HOOK_FORM_NAME: str = "fault_hook:local"
REMOTE_HOOK_FORM_NAME: str = "fault_hook:remote_inference_cell"

HOOK_SOURCE_SUB_INDEX: int = 0
LOCAL_HOOK_FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.DEADLOCK, FailureMode.SIGSTOP]
REMOTE_HOOK_FAILURE_MODES: list[FailureMode] = sorted(RECEIVER_SUPPORTED_MODES, key=lambda mode: mode.value)
ARM_REQUEST_TIMEOUT_SECONDS: float = 60.0
ARM_REFUSAL_STATUS_CODES: frozenset[int] = frozenset({400, 404, 422})
_ARM_REFUSAL_REASON_OF_STATUS_CODE: dict[int, str] = {400: "BadRequest", 404: "NotFound"}
ARMED_TRAINER_MODEL_ID: str | None = None

SOAK_DELAY_MS_MIN: int = 0
SOAK_DELAY_MS_MAX: int = 1000

EXPECTED_OUTCOME_OF_TARGET: dict[FaultHookTarget, FaultHookOutcome] = {
    FaultHookTarget.LOCAL: FaultHookOutcome.FIRED,
    FaultHookTarget.REMOTE_INFERENCE_CELL: FaultHookOutcome.ACCEPTED,
}


# =================================== delays ===================================


def draw_fault_hook_delay_ms(rng: random.Random) -> int:
    assert SOAK_DELAY_MS_MAX <= MAX_FAULT_HOOK_DELAY_MS, (
        f"a delay of up to {SOAK_DELAY_MS_MAX}ms cannot be drawn for a hook that refuses anything over "
        f"{MAX_FAULT_HOOK_DELAY_MS}ms, so every draw above it would be an arm the worker rejects"
    )
    return rng.randint(SOAK_DELAY_MS_MIN, SOAK_DELAY_MS_MAX)


# ============================== source selection ==============================


@dataclasses.dataclass(frozen=True)
class HookSource:
    cell_name: str
    cell_index: int
    workers_hash: str
    assigned_workers_hash_of_cell_id: dict[str, str]


def compute_hook_sources(*, base_url: str, event_dir: Path) -> list[HookSource]:
    workers_hash_of_cell_id = _compute_live_trainer_hashes(base_url=base_url)
    if not workers_hash_of_cell_id:
        return []

    assignment_of_cell_id: dict[str, WeightUpdateAssignmentEvent] = {}
    for event in read_events(event_dir):
        if not isinstance(event, WeightUpdateAssignmentEvent):
            continue
        if workers_hash_of_cell_id.get(event.trainer_cell_id) != event.trainer_workers_hash:
            continue
        if not event.assigned_workers_hash_of_cell_id:
            continue
        assignment_of_cell_id[event.trainer_cell_id] = event

    return [
        HookSource(
            cell_name=cell_id,
            cell_index=assignment.trainer_cell_index,
            workers_hash=assignment.trainer_workers_hash,
            assigned_workers_hash_of_cell_id=dict(assignment.assigned_workers_hash_of_cell_id),
        )
        for cell_id, assignment in sorted(assignment_of_cell_id.items())
    ]


def _compute_live_trainer_hashes(*, base_url: str) -> dict[str, str]:
    cells = list_fault_hook_sources(base_url=base_url)
    if cells is None:
        return {}
    return {
        cell["metadata"]["name"]: cell_workers_hash(cell)
        for cell in cells
        if cell_is_alive(cell) and cell_is_allocated(cell)
    }


def compute_local_hook_candidates(*, tensor_parallel_size: int) -> list[FaultHookName]:
    candidates = [
        FaultHookName.WEIGHT_UPDATE_BEFORE_P2P_WRITE,
        FaultHookName.WEIGHT_UPDATE_AFTER_P2P_SUBMIT,
        FaultHookName.WEIGHT_UPDATE_AFTER_BASE_WEIGHTS,
    ]
    if tensor_parallel_size > 1:
        candidates.append(FaultHookName.WEIGHT_UPDATE_BEFORE_ALL_GATHER)
    return sorted(candidates)


def compute_remote_hook_candidates() -> list[FaultHookName]:
    return sorted(REMOTE_CAPABLE_HOOKS)


# =================================== forms ====================================


@dataclasses.dataclass(frozen=True)
class HookFaultContext:
    base_url: str
    event_log: EventLog
    event_dir: Path
    checkpoint_dir: Path
    tensor_parallel_size: int
    ft_components: tuple[str, ...]


class _BaseHookFaultForm(BaseFaultForm):
    def __init__(self, context: HookFaultContext) -> None:
        self._context = context

    @property
    def records_own_attempt(self) -> bool:
        return True

    @property
    @abc.abstractmethod
    def target(self) -> FaultHookTarget: ...

    @property
    @abc.abstractmethod
    def victim_cell_type(self) -> str: ...

    @abc.abstractmethod
    def candidate_hooks(self) -> list[FaultHookName]: ...

    @abc.abstractmethod
    def candidate_modes(self) -> list[FailureMode]: ...

    @abc.abstractmethod
    def eligible_sources(self, cell: dict) -> list[HookSource]: ...

    def is_available(self, cell: dict) -> bool:
        if not self.candidate_hooks() or not self.eligible_sources(cell):
            return False
        return recovery_source_exists(event_dir=self._context.event_dir, checkpoint_dir=self._context.checkpoint_dir)

    def inject(self, cell: dict, rng: random.Random) -> None:
        sources = self.eligible_sources(cell)
        assert sources, (
            f"{self.name} was drawn with no trainer whose current generation owns a non-empty weight update "
            f"assignment, so there is no source that can be proven to reach the hook"
        )
        source = rng.choice(sources)
        hook = rng.choice(self.candidate_hooks())
        mode = rng.choice(self.candidate_modes())
        delay_ms = draw_fault_hook_delay_ms(rng)
        request_id = f"soak-{self.name}-{rng.getrandbits(64):016x}"

        self._note_arm(
            request_id=request_id, source=source, hook=hook, mode=mode, delay_ms=delay_ms, acknowledged=False
        )
        response = requests.post(
            f"{self._context.base_url}/api/v1/cells/{source.cell_name}/arm-fault-hook",
            json={
                "expected_workers_hash": source.workers_hash,
                "hook": hook.value,
                "mode": mode.value,
                "target": self.target.value,
                "sub_index": HOOK_SOURCE_SUB_INDEX,
                "request_id": request_id,
                "delay_ms": delay_ms,
            },
            timeout=ARM_REQUEST_TIMEOUT_SECONDS,
        )
        if (refused_because := compute_arm_refusal(response)) is not None:
            logger.info("Fault hook request %s never armed anything: %s", request_id, refused_because)
            self._context.event_log.note_hook_arm_refusal(request_id=request_id, refused_because=refused_because)
            return

        response.raise_for_status()
        self._note_arm(
            request_id=request_id, source=source, hook=hook, mode=mode, delay_ms=delay_ms, acknowledged=True
        )

    def _note_arm(
        self,
        *,
        request_id: str,
        source: HookSource,
        hook: FaultHookName,
        mode: FailureMode,
        delay_ms: int,
        acknowledged: bool,
    ) -> None:
        self._context.event_log.note_hook_arm(
            request_id=request_id,
            form_name=self.name,
            cell_type=self.victim_cell_type,
            source_cell_name=source.cell_name,
            source_workers_hash=source.workers_hash,
            source_cell_index=source.cell_index,
            source_rank_within_cell=HOOK_SOURCE_SUB_INDEX,
            hook=hook.value,
            mode=mode.value,
            target=self.target.value,
            delay_ms=delay_ms,
            acknowledged=acknowledged,
        )

    def _sources(self) -> list[HookSource]:
        return compute_hook_sources(base_url=self._context.base_url, event_dir=self._context.event_dir)


class LocalHookFaultForm(_BaseHookFaultForm):
    @property
    def name(self) -> str:
        return LOCAL_HOOK_FORM_NAME

    @property
    def target(self) -> FaultHookTarget:
        return FaultHookTarget.LOCAL

    @property
    def victim_cell_type(self) -> str:
        return ACTOR_CELL_TYPE

    def candidate_hooks(self) -> list[FaultHookName]:
        return compute_local_hook_candidates(tensor_parallel_size=self._context.tensor_parallel_size)

    def candidate_modes(self) -> list[FailureMode]:
        return list(LOCAL_HOOK_FAILURE_MODES)

    def eligible_sources(self, cell: dict) -> list[HookSource]:
        cell_name = cell["metadata"]["name"]
        workers_hash = cell_workers_hash(cell)
        return [
            source
            for source in self._sources()
            if source.cell_name == cell_name and source.workers_hash == workers_hash
        ]


class RemoteHookFaultForm(_BaseHookFaultForm):
    @property
    def name(self) -> str:
        return REMOTE_HOOK_FORM_NAME

    @property
    def target(self) -> FaultHookTarget:
        return FaultHookTarget.REMOTE_INFERENCE_CELL

    @property
    def victim_cell_type(self) -> str:
        return ROLLOUT_CELL_TYPE

    def candidate_hooks(self) -> list[FaultHookName]:
        return compute_remote_hook_candidates()

    def candidate_modes(self) -> list[FailureMode]:
        return list(REMOTE_HOOK_FAILURE_MODES)

    def eligible_sources(self, cell: dict) -> list[HookSource]:
        return self._sources()


def merge_hook_fault_forms(forms: CellFaultForms, context: HookFaultContext) -> CellFaultForms:
    merged: CellFaultForms = {cell_type: list(cell_forms) for cell_type, cell_forms in forms.items()}
    if "train" in context.ft_components and ACTOR_CELL_TYPE in merged:
        merged[ACTOR_CELL_TYPE].append(LocalHookFaultForm(context))
    if "rollout" in context.ft_components and ROLLOUT_CELL_TYPE in merged:
        merged[ROLLOUT_CELL_TYPE].append(RemoteHookFaultForm(context))
    return merged


# =============================== arm refusals =================================


def compute_arm_refusal(response: requests.Response) -> str | None:
    if response.status_code not in ARM_REFUSAL_STATUS_CODES:
        return None

    try:
        body = response.json()
    except ValueError:
        logger.info("The %s answer to an arming request carried no json body", response.status_code, exc_info=True)
        return None

    if not isinstance(body, dict):
        return None
    if response.status_code == 422:
        detail = body.get("detail")
        if not _is_request_validation_detail(detail):
            return None
        return f"the api server read the arming request as malformed and never routed it: {detail}"
    if not _is_k8s_arm_refusal(body, status_code=response.status_code):
        return None
    return (
        f"the api server refused the arming request with {body.get('reason')!r} before any worker was asked to "
        f"arm anything: {body.get('message')}"
    )


def _is_request_validation_detail(detail: object) -> bool:
    if not isinstance(detail, list) or not detail:
        return False
    for error in detail:
        if not isinstance(error, dict):
            return False
        location = error.get("loc")
        if not isinstance(location, list) or not location or location[0] != "body":
            return False
        if not all(isinstance(part, (str, int)) for part in location):
            return False
        if not isinstance(error.get("type"), str) or not error["type"]:
            return False
        if not isinstance(error.get("msg"), str) or not error["msg"]:
            return False
    return True


def _is_k8s_arm_refusal(body: dict[str, object], *, status_code: int) -> bool:
    return (
        body.get("apiVersion") == "v1"
        and body.get("kind") == "Status"
        and body.get("status") == "Failure"
        and body.get("code") == status_code
        and body.get("reason") == _ARM_REFUSAL_REASON_OF_STATUS_CODE[status_code]
        and isinstance(body.get("message"), str)
        and bool(body["message"])
    )


# ============================== fire collection ===============================


class HookFireCollector:
    def __init__(self, *, event_log: EventLog, event_dir: Path) -> None:
        self._event_log = event_log
        self._event_dir = event_dir
        self._raw_fire_of_request_id: dict[str, FaultHookFireEvent] = {}

    def collect(self) -> None:
        if not self._event_dir.is_dir():
            return

        harms = {harm.request_id: harm for harm in compute_hook_harms(self._event_log.events)}
        awaiting_publication = [harm for harm in harms.values() if harm.delivered and not harm.published]
        if not harms:
            return

        events = read_events(self._event_dir)
        if awaiting_publication:
            self._collect_weight_publications(events)

        assignments = [event for event in events if isinstance(event, WeightUpdateAssignmentEvent)]
        for event in events:
            if not isinstance(event, FaultHookFireEvent):
                continue
            harm = harms.get(event.request_id)
            if harm is None:
                continue
            if (seen := self._raw_fire_of_request_id.get(event.request_id)) is not None:
                assert (
                    seen == event
                ), f"request {event.request_id} has different raw fault hook fires: {seen} and {event}"
                continue
            record = compute_hook_fire_record(
                event,
                expected_hook=harm.hook,
                expected_mode=harm.mode,
                expected_target=harm.target,
                expected_delay_ms=harm.delay_ms,
                expected_source=TrainProcessIdentity(
                    component=ACTOR_ROLE,
                    model_id=ARMED_TRAINER_MODEL_ID,
                    cell_index=harm.source_cell_index,
                    rank_within_cell=harm.source_rank_within_cell,
                ),
                expected_trainer_workers_hash=harm.source_workers_hash,
                assignment=compute_fire_assignment(
                    assignments,
                    weight_version=event.weight_version,
                    trainer_cell_id=harm.source_cell_name,
                    trainer_cell_index=harm.source_cell_index,
                    trainer_workers_hash=harm.source_workers_hash,
                ),
            )
            compute_hook_harms([*self._event_log.events, record])
            self._raw_fire_of_request_id[event.request_id] = event
            if harm.fire is None:
                self._event_log.note_hook_fire(record)
            logger.info("Fault hook request %s reached its point: %s", event.request_id, record)

    def _collect_weight_publications(self, events: list) -> None:
        seen = {
            (one.trainer_model_id, one.weight_version, one.published_at)
            for one in self._event_log.events
            if isinstance(one, WeightPublicationEvent)
        }
        for event in events:
            if not isinstance(event, InferenceEngineWeightChecksumEvent):
                continue
            if (event.trainer_model_id, event.weight_version, event.timestamp) in seen:
                continue
            self._event_log.note_weight_publication(
                WeightPublicationEvent(
                    published_at=event.timestamp,
                    trainer_model_id=event.trainer_model_id,
                    weight_version=event.weight_version,
                    cell_ids=sorted(cell_id for cell_id, checksums in event.engine_checksums.items() if checksums),
                )
            )


def compute_fire_assignment(
    assignments: list[WeightUpdateAssignmentEvent],
    *,
    weight_version: int | None,
    trainer_cell_id: str,
    trainer_cell_index: int,
    trainer_workers_hash: str,
) -> WeightUpdateAssignmentEvent | None:
    matching = [
        event
        for event in assignments
        if event.weight_version == weight_version and event.trainer_cell_id == trainer_cell_id
    ]
    if len(matching) != 1:
        return None

    (assignment,) = matching
    if assignment.trainer_workers_hash != trainer_workers_hash:
        return None
    if assignment.trainer_cell_index != trainer_cell_index:
        return None
    return assignment


def compute_hook_fire_record(
    event: FaultHookFireEvent,
    *,
    expected_hook: str,
    expected_mode: str,
    expected_target: str,
    expected_delay_ms: int,
    expected_source: TrainProcessIdentity,
    expected_trainer_workers_hash: str,
    assignment: WeightUpdateAssignmentEvent | None,
) -> HookFireEvent:
    rejected_because = compute_hook_fire_rejection(
        event,
        expected_hook=expected_hook,
        expected_mode=expected_mode,
        expected_target=expected_target,
        expected_delay_ms=expected_delay_ms,
        expected_source=expected_source,
        expected_trainer_workers_hash=expected_trainer_workers_hash,
        assignment=assignment,
    )
    harmless_because = compute_hook_fire_harmless_resolution(
        event,
        expected_hook=expected_hook,
        expected_mode=expected_mode,
        expected_target=expected_target,
        expected_delay_ms=expected_delay_ms,
        expected_source=expected_source,
        expected_trainer_workers_hash=expected_trainer_workers_hash,
        assignment=assignment,
    )
    source = event.source
    return HookFireEvent(
        fired_at=event.timestamp,
        request_id=event.request_id,
        trainer_model_id=source.model_id if isinstance(source, TrainProcessIdentity) else None,
        assigned_cell_ids=sorted(assignment.assigned_workers_hash_of_cell_id) if assignment is not None else [],
        hook=event.hook,
        mode=event.mode,
        target=event.target,
        outcome=event.outcome,
        delay_ms=event.delay_ms,
        weight_version=event.weight_version,
        source_cell_index=source.cell_index if isinstance(source, TrainProcessIdentity) else None,
        source_rank_within_cell=source.rank_within_cell if isinstance(source, TrainProcessIdentity) else None,
        victim_cell_name=event.victim_cell_id,
        victim_workers_hash=event.victim_workers_hash,
        victim_receiver_boot_uuid=event.victim_receiver_boot_uuid,
        victim_session_id=event.victim_session_id,
        victim_receiver_rank=event.victim_receiver_rank,
        delivered=rejected_because is None,
        rejected_because=rejected_because,
        harmless_because=harmless_because,
    )


def compute_hook_fire_rejection(
    event: FaultHookFireEvent,
    *,
    expected_hook: str,
    expected_mode: str,
    expected_target: str,
    expected_delay_ms: int,
    expected_source: TrainProcessIdentity,
    expected_trainer_workers_hash: str,
    assignment: WeightUpdateAssignmentEvent | None,
) -> str | None:
    mismatch = compute_hook_fire_identity_mismatch(
        event,
        expected_hook=expected_hook,
        expected_mode=expected_mode,
        expected_target=expected_target,
        expected_delay_ms=expected_delay_ms,
        expected_source=expected_source,
        expected_trainer_workers_hash=expected_trainer_workers_hash,
        assignment=assignment,
    )
    if mismatch is not None:
        return mismatch

    expected_outcome = EXPECTED_OUTCOME_OF_TARGET[FaultHookTarget(expected_target)]
    if event.outcome != expected_outcome.value:
        return (
            f"answered {event.outcome}, not the {expected_outcome.value} a delivered {expected_target} fault records"
        )
    return None


def compute_hook_fire_harmless_resolution(
    event: FaultHookFireEvent,
    *,
    expected_hook: str,
    expected_mode: str,
    expected_target: str,
    expected_delay_ms: int,
    expected_source: TrainProcessIdentity,
    expected_trainer_workers_hash: str,
    assignment: WeightUpdateAssignmentEvent | None,
) -> str | None:
    if expected_target != FaultHookTarget.REMOTE_INFERENCE_CELL.value:
        return None
    if event.outcome != FaultHookOutcome.STALE_TARGET.value:
        return None

    mismatch = compute_hook_fire_identity_mismatch(
        event,
        expected_hook=expected_hook,
        expected_mode=expected_mode,
        expected_target=expected_target,
        expected_delay_ms=expected_delay_ms,
        expected_source=expected_source,
        expected_trainer_workers_hash=expected_trainer_workers_hash,
        assignment=assignment,
    )
    if mismatch is not None:
        return None

    return (
        f"reached {event.victim_cell_id} at {event.victim_workers_hash}, which update {event.weight_version} did "
        f"assign this sender, and the receiver of session {event.victim_session_id} refused it as a stale target, "
        f"so this request harmed nothing"
    )


def compute_hook_fire_identity_mismatch(
    event: FaultHookFireEvent,
    *,
    expected_hook: str,
    expected_mode: str,
    expected_target: str,
    expected_delay_ms: int,
    expected_source: TrainProcessIdentity,
    expected_trainer_workers_hash: str,
    assignment: WeightUpdateAssignmentEvent | None,
) -> str | None:
    if event.hook != expected_hook or event.mode != expected_mode:
        return f"fired as {event.hook}/{event.mode}, not the armed {expected_hook}/{expected_mode}"
    if event.target != expected_target:
        return f"fired against {event.target}, not the armed {expected_target}"
    if event.delay_ms != expected_delay_ms:
        return f"fired after a {event.delay_ms}ms delay, not the {expected_delay_ms}ms it was armed for"
    if event.source != expected_source:
        return f"fired in {event.source}, not in the armed {expected_source}"
    if event.weight_version is None:
        return "fired outside any weight update, so it names no update whose targets could be held responsible"
    if assignment is None:
        return (
            f"fired in weight update {event.weight_version}, which the armed incarnation "
            f"{expected_trainer_workers_hash} of cell index {expected_source.cell_index} was not given exactly one "
            f"assignment for, so the fault fired in a replacement or in an update whose targets are unknown"
        )
    if expected_target == FaultHookTarget.LOCAL.value:
        return None

    if not (event.victim_cell_id and event.victim_workers_hash):
        return f"named no inference incarnation ({event.victim_cell_id}, {event.victim_workers_hash})"
    if not (event.victim_receiver_boot_uuid and event.victim_session_id and event.victim_receiver_rank is not None):
        return (
            f"carries no receiver incarnation (uuid={event.victim_receiver_boot_uuid}, "
            f"session={event.victim_session_id}, rank={event.victim_receiver_rank})"
        )
    if assignment.assigned_workers_hash_of_cell_id.get(event.victim_cell_id) != event.victim_workers_hash:
        return (
            f"named {event.victim_cell_id} at {event.victim_workers_hash}, which is not what update "
            f"{event.weight_version} assigned this sender ({assignment.assigned_workers_hash_of_cell_id})"
        )
    return None
