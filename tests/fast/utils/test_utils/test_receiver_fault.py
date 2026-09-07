from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from miles.utils import http_utils
from miles.utils.test_utils import receiver_fault
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.test_utils.receiver_fault import (
    ReceiverFaultOutcome,
    ReceiverFaultRefusedError,
    ReceiverIdentity,
    request_receiver_fault,
)

_RECEIVER = ReceiverIdentity(
    receiver_boot_uuid="boot-1", session_id="session-1", rank=2, control_url="http://10.0.0.9:41111"
)
_REQUEST_ID = "req-1"


def _accepted_body(**overrides) -> dict:
    body = {
        "status": "accepted",
        "request_id": _REQUEST_ID,
        "receiver_boot_uuid": _RECEIVER.receiver_boot_uuid,
        "session_id": _RECEIVER.session_id,
        "rank": _RECEIVER.rank,
    }
    return {**body, **overrides}


def _rejected_body(*, reason: str, **overrides) -> dict:
    return _accepted_body(status="rejected", reason=reason, **overrides)


class _RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, status_code: int = 200, body: dict | str | None = None, error: Exception | None = None):
        self.requests: list[httpx.Request] = []
        self._status_code = status_code
        self._body = body if body is not None else _accepted_body()
        self._error = error

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        if isinstance(self._body, str):
            return httpx.Response(self._status_code, text=self._body, request=request)
        return httpx.Response(self._status_code, json=self._body, request=request)


def _run(transport: _RecordingTransport, *, mode: FailureMode = FailureMode.SIGKILL) -> ReceiverFaultOutcome:
    async def attempt() -> ReceiverFaultOutcome:
        client = httpx.AsyncClient(transport=transport)
        http_utils.GeneralHttpClientProvider._clients[asyncio.get_running_loop()] = client
        try:
            return await request_receiver_fault(receiver=_RECEIVER, mode=mode, request_id=_REQUEST_ID, timeout=0.5)
        finally:
            http_utils.GeneralHttpClientProvider._clients.pop(asyncio.get_running_loop(), None)
            await client.aclose()

    return asyncio.run(attempt())


class TestTheRequest:
    def test_the_expected_incarnation_is_what_the_receiver_is_asked_to_check(self):
        """The receiver is the only process that can rule out a replacement, and only if it is told what to expect."""
        transport = _RecordingTransport()

        _run(transport)

        (request,) = transport.requests
        assert str(request.url) == "http://10.0.0.9:41111/inject_fault"
        assert json.loads(request.content) == {
            "request_id": _REQUEST_ID,
            "expected_receiver_boot_uuid": "boot-1",
            "expected_session_id": "session-1",
            "expected_rank": 2,
            "mode": "sigkill",
        }

    @pytest.mark.parametrize("mode", [FailureMode.SEGFAULT, FailureMode.EXIT, FailureMode.DEADLOCK])
    def test_a_mode_the_receiver_cannot_perform_is_refused_before_it_is_sent(self, mode: FailureMode):
        """Exiting, segfaulting and deadlocking are what a process does to itself; no signal reproduces them."""
        transport = _RecordingTransport()

        with pytest.raises(ReceiverFaultRefusedError):
            _run(transport, mode=mode)

        assert transport.requests == []

    def test_a_sigstop_is_sent_as_the_mode_the_receiver_implements(self):
        """A frozen receiver is a fault only the process holding the session can inflict on itself."""
        transport = _RecordingTransport()

        assert _run(transport, mode=FailureMode.SIGSTOP) is ReceiverFaultOutcome.ACCEPTED

        (request,) = transport.requests
        assert json.loads(request.content) == {
            "request_id": _REQUEST_ID,
            "expected_receiver_boot_uuid": "boot-1",
            "expected_session_id": "session-1",
            "expected_rank": 2,
            "mode": "sigstop",
        }


class TestAccepting:
    def test_an_acceptance_by_this_incarnation_is_accepted(self):
        """This is the whole point: the process holding the session took the request naming its own identity."""
        assert _run(_RecordingTransport()) is ReceiverFaultOutcome.ACCEPTED

    def test_an_ok_naming_another_boot_uuid_is_refused(self):
        """A replacement answering ok would otherwise be recorded as harm done to the incarnation that was written."""
        transport = _RecordingTransport(body=_accepted_body(receiver_boot_uuid="boot-2"))

        with pytest.raises(ReceiverFaultRefusedError, match="not"):
            _run(transport)

    def test_an_ok_naming_another_session_is_refused(self):
        """The session is what the write addressed, so an ok for a different one describes a different transfer."""
        transport = _RecordingTransport(body=_accepted_body(session_id="session-2"))

        with pytest.raises(ReceiverFaultRefusedError):
            _run(transport)

    def test_an_ok_naming_another_rank_is_refused(self):
        """A multi-rank engine answers for every rank, and only the one this transfer reached may be harmed."""
        transport = _RecordingTransport(body=_accepted_body(rank=3))

        with pytest.raises(ReceiverFaultRefusedError):
            _run(transport)

    def test_an_ok_for_another_request_is_refused(self):
        """Two faults can be in flight in a soak, and an answer only speaks for the request it names."""
        transport = _RecordingTransport(body=_accepted_body(request_id="req-2"))

        with pytest.raises(ReceiverFaultRefusedError, match="some other fault"):
            _run(transport)

    def test_an_ok_that_is_not_an_acceptance_is_refused(self):
        """A 200 whose status is not `accepted` is not the receiver taking the request."""
        transport = _RecordingTransport(body=_accepted_body(status="ignored"))

        with pytest.raises(ReceiverFaultRefusedError):
            _run(transport)


class TestRefusing:
    @pytest.mark.parametrize(
        "reason", ["receiver_inactive", "receiver_boot_uuid_mismatch", "session_id_mismatch", "rank_mismatch"]
    )
    def test_an_identity_refusal_means_the_written_incarnation_is_gone(self, reason: str):
        """The receiver that answered is not the one the transfer reached, so nothing of it survives to harm."""
        transport = _RecordingTransport(status_code=409, body=_rejected_body(reason=reason))

        assert _run(transport) is ReceiverFaultOutcome.STALE_TARGET

    def test_a_stale_answer_may_name_the_replacement_identity(self):
        """The body reports the receiver's current identity, which is exactly the one that differs."""
        transport = _RecordingTransport(
            status_code=409,
            body=_rejected_body(reason="receiver_boot_uuid_mismatch", receiver_boot_uuid="boot-2", session_id="s-2"),
        )

        assert _run(transport) is ReceiverFaultOutcome.STALE_TARGET

    def test_a_stale_answer_for_another_request_is_refused(self):
        """Pairing by request id is the only thing that makes a refusal this request's business."""
        transport = _RecordingTransport(
            status_code=409, body=_rejected_body(reason="receiver_inactive", request_id="req-2")
        )

        with pytest.raises(ReceiverFaultRefusedError):
            _run(transport)

    def test_a_pending_action_conflict_is_not_a_stale_target(self):
        """The incarnation is alive and refused for its own reason; counting it as gone would invent a recovery."""
        transport = _RecordingTransport(status_code=409, body=_rejected_body(reason="pending_action_exists"))

        with pytest.raises(ReceiverFaultRefusedError, match="only an identity mismatch"):
            _run(transport)

    def test_a_request_id_payload_conflict_is_not_a_stale_target(self):
        """The same id carrying a different payload is a bug in the caller, not a replaced receiver."""
        transport = _RecordingTransport(status_code=409, body=_rejected_body(reason="request_id_payload_conflict"))

        with pytest.raises(ReceiverFaultRefusedError):
            _run(transport)

    def test_a_bad_request_is_not_a_stale_target(self):
        """A mode the receiver does not implement is a contract error, and silently passing it would hide it."""
        transport = _RecordingTransport(status_code=400, body=_rejected_body(reason="unsupported_mode"))

        with pytest.raises(ReceiverFaultRefusedError):
            _run(transport)

    def test_a_body_that_is_not_json_is_refused(self):
        """Nothing in it can be matched against what was asked, so it cannot prove an acceptance either way."""
        transport = _RecordingTransport(body="not json at all")

        with pytest.raises(ReceiverFaultRefusedError, match="not"):
            _run(transport)


class TestUnknownAnswers:
    def test_a_timeout_is_unknown_rather_than_delivered(self):
        """The request may still have landed, so neither success nor a blind retry is allowed."""
        transport = _RecordingTransport(error=httpx.ReadTimeout("no answer"))

        assert _run(transport) is ReceiverFaultOutcome.UNKNOWN

    def test_a_disconnect_is_unknown_rather_than_delivered(self):
        """A receiver that dies while answering looks the same as one that never received the request."""
        transport = _RecordingTransport(error=httpx.ConnectError("connection refused"))

        assert _run(transport) is ReceiverFaultOutcome.UNKNOWN

    def test_an_unknown_answer_sends_nothing_a_second_time(self):
        """Retrying blindly could take out the replacement that came up in the meantime."""
        transport = _RecordingTransport(error=httpx.ReadTimeout("no answer"))

        _run(transport)

        assert len(transport.requests) == 1


class TestSupportedModes:
    def test_only_the_signals_the_receiver_implements_are_offered(self):
        """Exit, segfault and deadlock cannot be produced from outside, and offering them would fake a fault form."""
        assert receiver_fault.RECEIVER_SUPPORTED_MODES == frozenset({FailureMode.SIGKILL, FailureMode.SIGSTOP})
