import ctypes
import os
import signal

import pytest

from miles.utils.test_utils import fault_injector
from miles.utils.test_utils.fault_injector import DEADLOCK_SLEEP_SECONDS, FailureMode, inject_fault


class _RecordingKill:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, pid: int, sig: int) -> None:
        self.calls.append((pid, sig))


class _RecordingLibc:
    def __init__(self) -> None:
        self.sleep = _RecordingSleep()


class _RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.argtypes: tuple | None = None
        self.restype: type | None = None

    def __call__(self, seconds: int) -> int:
        self.calls.append(seconds)
        return 0


@pytest.fixture(name="kills")
def _kills(monkeypatch: pytest.MonkeyPatch) -> _RecordingKill:
    recorder = _RecordingKill()
    monkeypatch.setattr(os, "kill", recorder)
    return recorder


@pytest.fixture(name="libc")
def _libc(monkeypatch: pytest.MonkeyPatch) -> _RecordingLibc:
    recorded = _RecordingLibc()
    monkeypatch.setattr(ctypes, "PyDLL", lambda handle: recorded)
    return recorded


class TestSigstop:
    def test_sigstop_freezes_the_process_that_reached_the_fault(self, kills: _RecordingKill) -> None:
        """A hang has to be inflicted on the process holding the work, not on any pid the caller happens to know."""
        inject_fault(mode=FailureMode.SIGSTOP.value)

        assert kills.calls == [(os.getpid(), signal.SIGSTOP)]

    def test_sigstop_is_a_mode_the_enum_offers(self) -> None:
        """The soak, the api server body and the receiver protocol all name this mode by its string."""
        assert FailureMode("sigstop") is FailureMode.SIGSTOP


class TestDeadlock:
    def test_deadlock_still_sleeps_inside_the_interpreter_lock(
        self, libc: _RecordingLibc, kills: _RecordingKill
    ) -> None:
        """A sleep that released the GIL would leave the worker answering heartbeats, which is not the hang."""
        inject_fault(mode=FailureMode.DEADLOCK.value)

        assert libc.sleep.calls == [DEADLOCK_SLEEP_SECONDS]
        assert libc.sleep.argtypes == (ctypes.c_uint,)
        assert not kills.calls

    def test_the_deadlock_is_loaded_through_pydll_rather_than_cdll(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CDLL releases the GIL for the call; only PyDLL holds it for the whole sleep."""
        recorded = _RecordingLibc()
        loaders: list[str] = []
        monkeypatch.setattr(ctypes, "PyDLL", lambda handle: loaders.append("PyDLL") or recorded)
        monkeypatch.setattr(ctypes, "CDLL", lambda handle: loaders.append("CDLL") or recorded)

        inject_fault(mode=FailureMode.DEADLOCK.value)

        assert loaders == ["PyDLL"]

    def test_the_sleep_is_the_witness_deadline_the_scenarios_read(self) -> None:
        """The hang witnesses fail a run whose eviction takes longer than the process would take to return alone."""
        assert fault_injector.DEADLOCK_SLEEP_SECONDS == 600


class TestSigkill:
    def test_sigkill_still_signals_this_process(self, kills: _RecordingKill) -> None:
        """Adding a second signal mode must not have changed what the default mode does."""
        inject_fault(mode=FailureMode.SIGKILL.value)

        assert kills.calls == [(os.getpid(), signal.SIGKILL)]
