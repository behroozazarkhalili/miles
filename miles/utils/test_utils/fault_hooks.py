from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import FaultHookFireEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector import FailureMode, inject_fault
from miles.utils.test_utils.receiver_fault import ReceiverFaultRefusedError, ReceiverIdentity
from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)

LOG_TAG = "fault_hook"


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


class RemoteFaultExecutor(Protocol):
    def __call__(self, *, target: RemoteInferenceTarget, mode: FailureMode, request_id: str) -> FaultHookOutcome: ...


class WeightUpdateSpan(FrozenStrictBaseModel):
    weight_version: int


class FaultHookAlreadyArmedError(Exception):
    pass


class FaultHookTargetUnsupportedError(Exception):
    pass


# ============================== public api ==============================


def arm_fault_hook(
    *, hook: str, mode: str, request_id: str, target: str = FaultHookTarget.LOCAL.value
) -> FaultHookAction:
    action = FaultHookAction(
        hook=FaultHookName(hook),
        mode=FailureMode(mode),
        request_id=request_id,
        target=FaultHookTarget(target),
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

    if action.target is FaultHookTarget.LOCAL:
        _record_fire(action, outcome=FaultHookOutcome.FIRED, target=None, span=frozen_span)
        inject_fault(mode=action.mode.value)
        return

    assert remote_target is not None, (
        f"{action.hook.value} consumed a remote request but the site named no inference target it could prove it was "
        f"writing to; without the receiver identity of this transfer a fault could only be aimed at a name"
    )
    try:
        outcome = _remote_executor(action)(target=remote_target, mode=action.mode, request_id=action.request_id)
    except ReceiverFaultRefusedError:
        _record_fire(action, outcome=FaultHookOutcome.REFUSED, target=remote_target, span=frozen_span)
        raise
    _record_fire(action, outcome=outcome, target=remote_target, span=frozen_span)


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

    def arm(self, action: FaultHookAction) -> None:
        with self._lock:
            if (armed := self._armed.get(action.hook)) is not None:
                raise FaultHookAlreadyArmedError(
                    f"{action.hook.value} already carries request {armed.request_id!r}, and taking request "
                    f"{action.request_id!r} in its place would drop a fault its requester is still waiting for"
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
            pid=os.getpid(),
        )

    def consume(self, hook: FaultHookName) -> FaultHookAction | None:
        if hook not in self._armed:
            return None
        with self._lock:
            return self._armed.pop(hook, None)

    def armed_hooks(self) -> dict[FaultHookName, FaultHookAction]:
        with self._lock:
            return dict(self._armed)


_REGISTRY = _FaultHookRegistry()
_WEIGHT_UPDATE_SPAN: WeightUpdateSpan | None = None
_REMOTE_EXECUTOR: RemoteFaultExecutor | None = None
