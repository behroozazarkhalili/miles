from __future__ import annotations

import asyncio
import logging

import httpx

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)

LOG_TAG = "receiver_fault"

INJECT_FAULT_PATH = "/inject_fault"
REQUEST_TIMEOUT_SECONDS = 10.0

STALE_STATUS_CODE = 409
ACCEPTED_STATUS = "accepted"

RECEIVER_SUPPORTED_MODES: frozenset[FailureMode] = frozenset({FailureMode.SIGKILL, FailureMode.SIGSTOP})

STALE_REFUSAL_REASONS: frozenset[str] = frozenset(
    {"receiver_inactive", "receiver_boot_uuid_mismatch", "session_id_mismatch", "rank_mismatch"}
)


class ReceiverIdentity(FrozenStrictBaseModel):
    receiver_boot_uuid: str
    session_id: str
    rank: int
    control_url: str


class ReceiverFaultOutcome(StrEnum):
    ACCEPTED = "accepted"
    STALE_TARGET = "stale_target"
    UNKNOWN = "unknown"


class ReceiverFaultRefusedError(Exception):
    pass


async def request_receiver_fault(
    *,
    receiver: ReceiverIdentity,
    mode: FailureMode,
    request_id: str,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> ReceiverFaultOutcome:
    if mode not in RECEIVER_SUPPORTED_MODES:
        raise ReceiverFaultRefusedError(
            f"a p2p weight receiver can only be asked for {sorted(m.value for m in RECEIVER_SUPPORTED_MODES)}, and "
            f"{mode.value} would have to be faked from outside the process that holds the session"
        )

    payload = {
        "request_id": request_id,
        "expected_receiver_boot_uuid": receiver.receiver_boot_uuid,
        "expected_session_id": receiver.session_id,
        "expected_rank": receiver.rank,
        "mode": mode.value,
    }
    log_structured(logger.warning, tag=LOG_TAG, op="request", url=receiver.control_url, **payload)

    try:
        response = await asyncio.wait_for(
            GeneralHttpClientProvider.client().post(
                f"{receiver.control_url.rstrip('/')}{INJECT_FAULT_PATH}", json=payload, timeout=timeout
            ),
            timeout=timeout + 1.0,
        )
    except (httpx.TransportError, TimeoutError, asyncio.TimeoutError):
        logger.warning(
            "Receiver %s did not answer the fault request %s, so whether it landed is unknown",
            receiver.control_url,
            request_id,
            exc_info=True,
        )
        return ReceiverFaultOutcome.UNKNOWN

    return _read_outcome(response, receiver=receiver, request_id=request_id)


def _read_outcome(response: httpx.Response, *, receiver: ReceiverIdentity, request_id: str) -> ReceiverFaultOutcome:
    body = _read_body(response, request_id=request_id)
    _assert_answers_this_request(body, request_id=request_id, response=response)

    if response.status_code == 200:
        _assert_accepted_by_this_incarnation(body, receiver=receiver, response=response)
        log_structured(
            logger.warning,
            tag=LOG_TAG,
            op="accepted",
            request_id=request_id,
            receiver_boot_uuid=receiver.receiver_boot_uuid,
            session_id=receiver.session_id,
            rank=receiver.rank,
        )
        return ReceiverFaultOutcome.ACCEPTED

    if response.status_code == STALE_STATUS_CODE and body.get("reason") in STALE_REFUSAL_REASONS:
        log_structured(
            logger.warning,
            tag=LOG_TAG,
            op="stale",
            request_id=request_id,
            reason=body.get("reason"),
            expected_receiver_boot_uuid=receiver.receiver_boot_uuid,
            answered_receiver_boot_uuid=body.get("receiver_boot_uuid"),
        )
        return ReceiverFaultOutcome.STALE_TARGET

    raise ReceiverFaultRefusedError(
        f"receiver {receiver.control_url} refused the fault request {request_id!r} with "
        f"{response.status_code} {body!r}; only an identity mismatch means the incarnation this write reached is "
        f"gone, and every other refusal leaves the request neither delivered nor accounted for"
    )


def _read_body(response: httpx.Response, *, request_id: str) -> dict:
    try:
        body = response.json()
    except ValueError as error:
        raise ReceiverFaultRefusedError(
            f"receiver answered the fault request {request_id!r} with {response.status_code} and a body that is not "
            f"json, so nothing in it can be matched against what was asked"
        ) from error
    if not isinstance(body, dict):
        raise ReceiverFaultRefusedError(
            f"receiver answered the fault request {request_id!r} with {body!r}, which names no request, receiver or "
            f"session"
        )
    return body


def _assert_answers_this_request(body: dict, *, request_id: str, response: httpx.Response) -> None:
    if body.get("request_id") != request_id:
        raise ReceiverFaultRefusedError(
            f"receiver answered {response.status_code} for request {body.get('request_id')!r}, not for the "
            f"{request_id!r} that was sent, so this answer describes some other fault"
        )


def _assert_accepted_by_this_incarnation(body: dict, *, receiver: ReceiverIdentity, response: httpx.Response) -> None:
    answered = (body.get("status"), body.get("receiver_boot_uuid"), body.get("session_id"), body.get("rank"))
    expected = (ACCEPTED_STATUS, receiver.receiver_boot_uuid, receiver.session_id, receiver.rank)
    if answered != expected:
        raise ReceiverFaultRefusedError(
            f"receiver answered {response.status_code} with {answered}, not {expected}; an ok that names another "
            f"incarnation, session or rank is not an acceptance by the process this write reached"
        )
