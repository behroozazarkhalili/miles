from __future__ import annotations

import logging
import os
import threading

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

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

    log_structured(
        logger.warning,
        tag=LOG_TAG,
        op="fire",
        hook=action.hook.value,
        mode=action.mode.value,
        request_id=action.request_id,
        pid=os.getpid(),
    )
    inject_fault(mode=action.mode.value)


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
        with self._lock:
            return self._armed.pop(hook, None)

    def armed_hooks(self) -> dict[FaultHookName, FaultHookAction]:
        with self._lock:
            return dict(self._armed)


_REGISTRY = _FaultHookRegistry()
