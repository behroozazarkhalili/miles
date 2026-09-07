from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import FaultHookFireEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector import FailureMode, inject_fault
from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)

LOG_TAG = "fault_hook"


class FaultHookName(StrEnum):
    WEIGHT_UPDATE_BEFORE_ALL_GATHER = "weight_update.before_all_gather"
    WEIGHT_UPDATE_BEFORE_P2P_WRITE = "weight_update.before_p2p_write"
    WEIGHT_UPDATE_AFTER_P2P_SUBMIT = "weight_update.after_p2p_submit"
    WEIGHT_UPDATE_AFTER_BASE_WEIGHTS = "weight_update.after_base_weights"


class FaultHookAction(FrozenStrictBaseModel):
    hook: FaultHookName
    mode: FailureMode
    request_id: str


class WeightUpdateSpan(FrozenStrictBaseModel):
    weight_version: int


class FaultHookAlreadyArmedError(Exception):
    pass


# ============================== public api ==============================


def arm_fault_hook(*, hook: str, mode: str, request_id: str) -> FaultHookAction:
    action = FaultHookAction(hook=FaultHookName(hook), mode=FailureMode(mode), request_id=request_id)
    _REGISTRY.arm(action)
    return action


def reach_fault_hook(hook: FaultHookName) -> None:
    if (action := _REGISTRY.consume(hook)) is None:
        return

    _record_fire(action)
    inject_fault(mode=action.mode.value)


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


def _record_fire(action: FaultHookAction) -> None:
    span = current_weight_update_span()
    fields = dict(
        hook=action.hook.value,
        mode=action.mode.value,
        request_id=action.request_id,
        weight_version=None if span is None else span.weight_version,
    )

    log_structured(logger.warning, tag=LOG_TAG, op="fire", pid=os.getpid(), **fields)
    if is_event_logger_initialized():
        get_event_logger().log(FaultHookFireEvent, fields)


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
