# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import enum
import threading
from datetime import datetime, timezone

from pydantic import Field

from miles.ray.rollout.inference_controller import HEALTH_PAUSED_FOR_WEIGHT_UPDATE
from miles.ray.train.cell_monitor import HEALTH_TRAINER_UNINITIALIZED
from miles.utils.pydantic_utils import FrozenStrictBaseModel

ACTOR_CELL_TYPE: str = "actor"
ROLLOUT_CELL_TYPE: str = "rollout"


def cell_type_of(cell: dict) -> str:
    return cell["metadata"]["labels"]["miles.io/cell-type"]


def cell_is_alive(cell: dict) -> bool:
    return _condition_status(cell, "Healthy") == "True"


def cell_is_allocated(cell: dict) -> bool:
    return _condition_status(cell, "Allocated") == "True"


def cell_health_reason(cell: dict) -> str | None:
    for cond in cell["status"]["conditions"]:
        if cond["type"] == "Healthy":
            return cond.get("reason")
    return None


def cell_workers_hash(cell: dict) -> str:
    workers_hash = cell["status"]["workers_hash"]
    assert (
        workers_hash
    ), f"cell {cell['metadata']['name']} reports no worker generation, so it has no identity to track"
    return workers_hash


def _condition_status(cell: dict, condition_type: str) -> str | None:
    for cond in cell["status"]["conditions"]:
        if cond["type"] == condition_type:
            return cond["status"]
    return None


class ObservedCellState(enum.Enum):
    SUSPENDED = "Suspended"  # torn down, holding no gpu
    PENDING = "Pending"  # allocated but gated: no engine serving yet
    RUNNING_NOT_SERVING = "RunningNotServing"  # engine is up but not registered in the router
    SERVING = "Serving"  # registered in the router, i.e. actually able to answer requests


def compute_observed_cell_state(cell: dict) -> ObservedCellState:
    phase = cell["status"]["phase"]
    if phase == "Suspended":
        return ObservedCellState.SUSPENDED
    if phase == "Pending":
        return ObservedCellState.PENDING
    serving = any(cond["type"] == "Serving" and cond["status"] == "True" for cond in cell["status"]["conditions"])
    return ObservedCellState.SERVING if serving else ObservedCellState.RUNNING_NOT_SERVING


class BaseEvent(FrozenStrictBaseModel):
    # Wall clock, so an event can be lined up against the timestamps the metric events carry.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InjectionEvent(BaseEvent):
    cell_name: str
    form_name: str
    succeeded: bool
    workers_hash: str
    harmed: bool = True


class CellInfo(FrozenStrictBaseModel):
    cell_type: str
    state: ObservedCellState
    alive: bool
    allocated: bool
    health_reason: str | None
    workers_hash: str


def cell_info_is_in_service(info: CellInfo) -> bool:
    if not (info.alive and info.allocated):
        return False
    if info.health_reason == HEALTH_TRAINER_UNINITIALIZED:
        return False
    if info.cell_type != ROLLOUT_CELL_TYPE:
        return True
    return info.state is ObservedCellState.SERVING


def cell_info_is_paused_for_weight_update(info: CellInfo) -> bool:
    if info.alive or not info.allocated:
        return False
    if info.health_reason != HEALTH_PAUSED_FOR_WEIGHT_UPDATE:
        return False
    if info.cell_type != ROLLOUT_CELL_TYPE:
        return True
    return info.state is ObservedCellState.SERVING


class ObservationsEvent(BaseEvent):
    # One whole poll, so a cell that has vanished is as recorded as one that answered.
    cell_infos: dict[str, CellInfo]


class HookArmEvent(BaseEvent):
    request_id: str
    form_name: str
    cell_type: str
    source_cell_name: str
    source_workers_hash: str
    source_cell_index: int
    source_rank_within_cell: int
    hook: str
    mode: str
    target: str
    acknowledged: bool


class HookArmRefusedEvent(BaseEvent):
    request_id: str
    refused_because: str


class HookFireEvent(BaseEvent):
    request_id: str
    hook: str
    mode: str
    target: str
    outcome: str
    weight_version: int | None
    source_cell_index: int | None
    source_rank_within_cell: int | None
    victim_cell_name: str | None
    victim_workers_hash: str | None
    victim_receiver_boot_uuid: str | None
    victim_session_id: str | None
    victim_receiver_rank: int | None
    delivered: bool
    rejected_because: str | None
    harmless_because: str | None


Event = InjectionEvent | ObservationsEvent | HookArmEvent | HookArmRefusedEvent | HookFireEvent


class EventLog:
    """The fault injector's only mutable state: what happened, in order. Every question is a view of it."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._lock = threading.Lock()

    @property
    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def note_injection_attempt(
        self, *, cell_name: str, form_name: str, succeeded: bool, workers_hash: str, harmed: bool = True
    ) -> None:
        self._append(
            InjectionEvent(
                cell_name=cell_name,
                form_name=form_name,
                succeeded=succeeded,
                harmed=harmed,
                workers_hash=workers_hash,
            )
        )

    def note_hook_arm(
        self,
        *,
        request_id: str,
        form_name: str,
        cell_type: str,
        source_cell_name: str,
        source_workers_hash: str,
        source_cell_index: int,
        source_rank_within_cell: int,
        hook: str,
        mode: str,
        target: str,
        acknowledged: bool,
    ) -> None:
        self._append(
            HookArmEvent(
                request_id=request_id,
                form_name=form_name,
                cell_type=cell_type,
                source_cell_name=source_cell_name,
                source_workers_hash=source_workers_hash,
                source_cell_index=source_cell_index,
                source_rank_within_cell=source_rank_within_cell,
                hook=hook,
                mode=mode,
                target=target,
                acknowledged=acknowledged,
            )
        )

    def note_hook_arm_refusal(self, *, request_id: str, refused_because: str) -> None:
        self._append(HookArmRefusedEvent(request_id=request_id, refused_because=refused_because))

    def note_hook_fire(self, event: HookFireEvent) -> None:
        self._append(event)

    def observe(self, cells: list[dict]) -> None:
        self._append(ObservationsEvent(cell_infos={c["metadata"]["name"]: compute_cell_info(c) for c in cells}))

    def _append(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)


def compute_cell_info(cell: dict) -> CellInfo:
    return CellInfo(
        cell_type=cell_type_of(cell),
        state=compute_observed_cell_state(cell),
        alive=cell_is_alive(cell),
        allocated=cell_is_allocated(cell),
        health_reason=cell_health_reason(cell),
        workers_hash=cell_workers_hash(cell),
    )
