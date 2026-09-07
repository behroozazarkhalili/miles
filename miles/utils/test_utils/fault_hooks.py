from __future__ import annotations

import dataclasses
import functools
import logging
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Annotated, Protocol

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

from pydantic import Field, StrictInt

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import FaultHookFireEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector import FailureMode, inject_fault
from miles.utils.test_utils.receiver_fault import ReceiverFaultRefusedError, ReceiverIdentity
from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)

LOG_TAG = "fault_hook"

MAX_FAULT_HOOK_DELAY_MS: int = 60_000

FaultHookDelayMs = Annotated[StrictInt, Field(ge=0, le=MAX_FAULT_HOOK_DELAY_MS)]


class FaultHookName(StrEnum):
    WEIGHT_UPDATE_BEFORE_ALL_GATHER = "weight_update.before_all_gather"
    WEIGHT_UPDATE_BEFORE_P2P_WRITE = "weight_update.before_p2p_write"
    WEIGHT_UPDATE_AFTER_P2P_SUBMIT = "weight_update.after_p2p_submit"
    WEIGHT_UPDATE_AFTER_BASE_WEIGHTS = "weight_update.after_base_weights"


class FaultHookTarget(StrEnum):
    LOCAL = "local"
    REMOTE_INFERENCE_CELL = "remote_inference_cell"


class FaultHookOutcome(StrEnum):
    FIRED = "fired"
    ACCEPTED = "accepted"
    STALE_TARGET = "stale_target"
    UNKNOWN = "unknown"
    REFUSED = "refused"
    NOT_SCHEDULED = "not_scheduled"
    ERRORED = "errored"


REMOTE_CAPABLE_HOOKS: frozenset[FaultHookName] = frozenset(
    {FaultHookName.WEIGHT_UPDATE_BEFORE_P2P_WRITE, FaultHookName.WEIGHT_UPDATE_AFTER_P2P_SUBMIT}
)


class RemoteInferenceTarget(FrozenStrictBaseModel):
    cell_id: str
    workers_hash: str
    receiver: ReceiverIdentity
    worker_in_cell_index: int | None = None


class FaultHookAction(FrozenStrictBaseModel):
    hook: FaultHookName
    mode: FailureMode
    request_id: str
    target: FaultHookTarget = FaultHookTarget.LOCAL
    delay_ms: FaultHookDelayMs = 0


class RemoteFaultExecutor(Protocol):
    def __call__(self, *, target: RemoteInferenceTarget, mode: FailureMode, request_id: str) -> FaultHookOutcome: ...


class DelayTimer(Protocol):
    def start(self) -> None: ...


class WeightUpdateSpan(FrozenStrictBaseModel):
    weight_version: int


class FaultHookAlreadyArmedError(Exception):
    pass


class FaultHookTargetUnsupportedError(Exception):
    pass


# ============================== public api ==============================


def arm_fault_hook(
    *, hook: str, mode: str, request_id: str, target: str = FaultHookTarget.LOCAL.value, delay_ms: int = 0
) -> FaultHookAction:
    action = FaultHookAction(
        hook=FaultHookName(hook),
        mode=FailureMode(mode),
        request_id=request_id,
        target=FaultHookTarget(target),
        delay_ms=delay_ms,
    )
    _assert_target_is_reachable(action)
    _REGISTRY.arm(action)
    return action


def reach_fault_hook(
    hook: FaultHookName,
    *,
    remote_target: RemoteInferenceTarget | None = None,
    span: WeightUpdateSpan | None = None,
) -> None:
    if (action := _REGISTRY.consume(hook)) is None:
        return

    frozen_span = span if span is not None else current_weight_update_span()

    if action.delay_ms == 0:
        try:
            _execute_fire(_freeze_fire(action, remote_target=remote_target, span=frozen_span))
        finally:
            _REGISTRY.finish(hook)
        return

    _schedule_fire(action, remote_target=remote_target, span=frozen_span)


def install_remote_fault_executor(executor: RemoteFaultExecutor | None) -> None:
    global _REMOTE_EXECUTOR
    _REMOTE_EXECUTOR = executor


@contextmanager
def weight_update_span(*, weight_version: int) -> Iterator[None]:
    global _WEIGHT_UPDATE_SPAN
    assert _WEIGHT_UPDATE_SPAN is None, (
        f"a weight update of version {_WEIGHT_UPDATE_SPAN} is already open in this process, and nesting a second one "
        f"would leave a fault reached inside it attributed to whichever span happened to be read"
    )
    _WEIGHT_UPDATE_SPAN = WeightUpdateSpan(weight_version=weight_version)
    try:
        yield
    finally:
        _WEIGHT_UPDATE_SPAN = None


def current_weight_update_span() -> WeightUpdateSpan | None:
    return _WEIGHT_UPDATE_SPAN


# =================================== firing ===================================


@dataclasses.dataclass(frozen=True)
class _FrozenFire:
    action: FaultHookAction
    remote_target: RemoteInferenceTarget | None
    remote_executor: RemoteFaultExecutor | None
    span: WeightUpdateSpan | None


def _freeze_fire(
    action: FaultHookAction, *, remote_target: RemoteInferenceTarget | None, span: WeightUpdateSpan | None
) -> _FrozenFire:
    if action.target is FaultHookTarget.LOCAL:
        return _FrozenFire(action=action, remote_target=None, remote_executor=None, span=span)

    assert remote_target is not None, (
        f"{action.hook.value} consumed a remote request but the site named no inference target it could prove it was "
        f"writing to; without the receiver identity of this transfer a fault could only be aimed at a name"
    )
    return _FrozenFire(action=action, remote_target=remote_target, remote_executor=_remote_executor(action), span=span)


def _execute_fire(fire: _FrozenFire) -> None:
    action = fire.action

    if action.target is FaultHookTarget.LOCAL:
        _record_fire(action, outcome=FaultHookOutcome.FIRED, target=None, span=fire.span)
        inject_fault(mode=action.mode.value)
        return

    executor, target = fire.remote_executor, fire.remote_target
    assert executor is not None and target is not None, (
        f"the fire of request {action.request_id!r} at {action.hook.value} froze no receiver and no executor, so the "
        f"fault could only be aimed by whatever this process holds now"
    )
    try:
        outcome = executor(target=target, mode=action.mode, request_id=action.request_id)
    except ReceiverFaultRefusedError:
        _record_fire(action, outcome=FaultHookOutcome.REFUSED, target=target, span=fire.span)
        raise
    except BaseException:
        _record_fire(action, outcome=FaultHookOutcome.ERRORED, target=target, span=fire.span)
        raise
    _record_fire(action, outcome=outcome, target=target, span=fire.span)


def _schedule_fire(
    action: FaultHookAction, *, remote_target: RemoteInferenceTarget | None, span: WeightUpdateSpan | None
) -> None:
    try:
        fire = _freeze_fire(action, remote_target=remote_target, span=span)
        _create_delay_timer(
            delay_seconds=action.delay_ms / 1000.0,
            run=functools.partial(_run_delayed_fire, fire),
            name=f"fault-hook-delay-{action.request_id}",
        ).start()
    except BaseException:
        _REGISTRY.finish(action.hook)
        _record_fire(action, outcome=FaultHookOutcome.NOT_SCHEDULED, target=remote_target, span=span)
        log_structured(
            logger.error,
            tag=LOG_TAG,
            op="schedule_failed",
            hook=action.hook.value,
            request_id=action.request_id,
            delay_ms=action.delay_ms,
            pid=os.getpid(),
            exc_info=True,
        )
        raise

    log_structured(
        logger.warning,
        tag=LOG_TAG,
        op="schedule",
        hook=action.hook.value,
        mode=action.mode.value,
        request_id=action.request_id,
        target=action.target.value,
        delay_ms=action.delay_ms,
        weight_version=None if span is None else span.weight_version,
        pid=os.getpid(),
    )


def _run_delayed_fire(fire: _FrozenFire) -> None:
    action = fire.action
    try:
        _execute_fire(fire)
    except BaseException:
        log_structured(
            logger.error,
            tag=LOG_TAG,
            op="delayed_fire_failed",
            hook=action.hook.value,
            request_id=action.request_id,
            delay_ms=action.delay_ms,
            pid=os.getpid(),
            exc_info=True,
        )
    finally:
        _REGISTRY.finish(action.hook)


def _create_delay_timer(*, delay_seconds: float, run: Callable[[], None], name: str) -> DelayTimer:
    timer = threading.Timer(delay_seconds, run)
    timer.name = name
    timer.daemon = True
    return timer


# ============================== fire bookkeeping ==============================


def _record_fire(
    action: FaultHookAction,
    *,
    outcome: FaultHookOutcome,
    target: RemoteInferenceTarget | None,
    span: WeightUpdateSpan | None,
) -> None:
    fields = dict(
        hook=action.hook.value,
        mode=action.mode.value,
        request_id=action.request_id,
        weight_version=None if span is None else span.weight_version,
        target=action.target.value,
        outcome=outcome.value,
        delay_ms=action.delay_ms,
        victim_cell_id=None if target is None else target.cell_id,
        victim_workers_hash=None if target is None else target.workers_hash,
        victim_worker_in_cell_index=None if target is None else target.worker_in_cell_index,
        victim_receiver_rank=None if target is None else target.receiver.rank,
        victim_receiver_boot_uuid=None if target is None else target.receiver.receiver_boot_uuid,
        victim_session_id=None if target is None else target.receiver.session_id,
    )

    log_structured(logger.warning, tag=LOG_TAG, op="fire", pid=os.getpid(), **fields)
    if is_event_logger_initialized():
        get_event_logger().log(FaultHookFireEvent, fields)


def _assert_target_is_reachable(action: FaultHookAction) -> None:
    if action.target is FaultHookTarget.LOCAL:
        return
    if action.hook not in REMOTE_CAPABLE_HOOKS:
        raise FaultHookTargetUnsupportedError(
            f"{action.hook.value} runs where no single inference target is being written to, so a remote request "
            f"there could only guess which cell to harm; the remote-capable hooks are "
            f"{sorted(hook.value for hook in REMOTE_CAPABLE_HOOKS)}"
        )
    if _REMOTE_EXECUTOR is None:
        raise FaultHookTargetUnsupportedError(
            f"this process cannot reach an inference cell's incarnation, so the remote request "
            f"{action.request_id!r} would be accepted and then dropped when {action.hook.value} is reached"
        )


def _remote_executor(action: FaultHookAction) -> RemoteFaultExecutor:
    executor = _REMOTE_EXECUTOR
    assert executor is not None, (
        f"the remote fault executor was removed between arming {action.request_id!r} and reaching "
        f"{action.hook.value}, leaving its requester waiting for a fault nothing can deliver"
    )
    return executor


# ============================== registry ================================


class _FaultHookRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._armed: dict[FaultHookName, FaultHookAction] = {}
        self._pending: dict[FaultHookName, FaultHookAction] = {}

    def arm(self, action: FaultHookAction) -> None:
        with self._lock:
            if (armed := self._armed.get(action.hook)) is not None:
                raise FaultHookAlreadyArmedError(
                    f"{action.hook.value} already carries request {armed.request_id!r}, and taking request "
                    f"{action.request_id!r} in its place would drop a fault its requester is still waiting for"
                )
            if (pending := self._pending.get(action.hook)) is not None:
                raise FaultHookAlreadyArmedError(
                    f"{action.hook.value} was already reached by request {pending.request_id!r}, whose fault has not "
                    f"finished running yet, so taking request {action.request_id!r} now would leave two faults "
                    f"outstanding at one point"
                )
            self._armed[action.hook] = action

        log_structured(
            logger.warning,
            tag=LOG_TAG,
            op="arm",
            hook=action.hook.value,
            mode=action.mode.value,
            request_id=action.request_id,
            target=action.target.value,
            delay_ms=action.delay_ms,
            pid=os.getpid(),
        )

    def consume(self, hook: FaultHookName) -> FaultHookAction | None:
        if hook not in self._armed:
            return None
        with self._lock:
            if (action := self._armed.pop(hook, None)) is None:
                return None
            self._pending[hook] = action
            return action

    def finish(self, hook: FaultHookName) -> None:
        with self._lock:
            self._pending.pop(hook, None)

    def armed_hooks(self) -> dict[FaultHookName, FaultHookAction]:
        with self._lock:
            return dict(self._armed)

    def pending_hooks(self) -> dict[FaultHookName, FaultHookAction]:
        with self._lock:
            return dict(self._pending)


_REGISTRY = _FaultHookRegistry()
_WEIGHT_UPDATE_SPAN: WeightUpdateSpan | None = None
_REMOTE_EXECUTOR: RemoteFaultExecutor | None = None
