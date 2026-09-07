from __future__ import annotations

import threading
from pathlib import Path

import pytest

from miles.utils.audit_utils.event_logger import logger as event_logger_module
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events
from miles.utils.audit_utils.event_logger.models import Event, FaultHookFireEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils import fault_hooks
from miles.utils.test_utils.fault_hooks import FaultHookAlreadyArmedError, FaultHookName, arm_fault_hook

_HOOK = FaultHookName.WEIGHT_UPDATE_BEFORE_ALL_GATHER
_OTHER_HOOK = FaultHookName.WEIGHT_UPDATE_BEFORE_P2P_WRITE

_THREAD_JOIN_TIMEOUT_SECONDS = 5.0
_CONCURRENT_ARRIVAL_COUNT = 8


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> fault_hooks._FaultHookRegistry:
    fresh = fault_hooks._FaultHookRegistry()
    monkeypatch.setattr(fault_hooks, "_REGISTRY", fresh)
    return fresh


@pytest.fixture
def injected(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    recorded: list[str] = []
    monkeypatch.setattr(fault_hooks, "inject_fault", lambda mode: recorded.append(mode))
    return recorded


class TestReachingAHookThatIsNotArmed:
    def test_an_unarmed_hook_injects_nothing(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str]
    ) -> None:
        """Production code runs through its hooks on every step, and an unarmed one must cost it nothing."""
        fault_hooks.reach_fault_hook(_HOOK)

        assert injected == []

    def test_arming_one_hook_leaves_the_others_untouched(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str]
    ) -> None:
        """A fault armed at the all-gather point must not fire at the p2p write the run reaches first."""
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1")

        fault_hooks.reach_fault_hook(_OTHER_HOOK)

        assert injected == []
        assert sorted(registry.armed_hooks()) == [_HOOK]


class TestReachingAnArmedHook:
    def test_arming_does_not_inject_before_the_hook_is_reached(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str]
    ) -> None:
        """Arming is a request about a future moment, so the process must survive the call itself."""
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1")

        assert injected == []
        assert registry.armed_hooks()[_HOOK].request_id == "req-1"

    def test_the_armed_mode_is_injected_once_and_the_hook_disarms(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str]
    ) -> None:
        """One request buys one fault: a second pass through the same hook must run clean."""
        arm_fault_hook(hook=_HOOK.value, mode="exit", request_id="req-1")

        fault_hooks.reach_fault_hook(_HOOK)
        fault_hooks.reach_fault_hook(_HOOK)

        assert injected == ["exit"]
        assert registry.armed_hooks() == {}

    def test_two_hooks_armed_together_each_fire_at_their_own_point(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str]
    ) -> None:
        """Two outstanding requests must not collapse into one shared slot."""
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1")
        arm_fault_hook(hook=_OTHER_HOOK.value, mode="exit", request_id="req-2")

        fault_hooks.reach_fault_hook(_OTHER_HOOK)
        fault_hooks.reach_fault_hook(_HOOK)

        assert injected == ["exit", "sigkill"]
        assert registry.armed_hooks() == {}

    def test_a_hook_reached_again_while_the_fault_runs_neither_fires_twice_nor_deadlocks(
        self, registry: fault_hooks._FaultHookRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The action is consumed before it runs and runs outside the lock, so a hang mode cannot wedge the hook."""
        recorded: list[str] = []
        reentered: list[bool] = []

        def _inject_and_reenter(mode: str) -> None:
            recorded.append(mode)
            thread = threading.Thread(target=lambda: fault_hooks.reach_fault_hook(_HOOK))
            thread.start()
            thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
            reentered.append(not thread.is_alive())

        monkeypatch.setattr(fault_hooks, "inject_fault", _inject_and_reenter)
        arm_fault_hook(hook=_HOOK.value, mode="deadlock", request_id="req-1")

        fault_hooks.reach_fault_hook(_HOOK)

        assert recorded == ["deadlock"]
        assert reentered == [True]

    def test_many_threads_reaching_the_hook_together_inject_exactly_once(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str]
    ) -> None:
        """Every trainer rank thread walks the same hook, and one armed request must kill the process once."""
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1")
        start = threading.Barrier(_CONCURRENT_ARRIVAL_COUNT)

        def _arrive() -> None:
            start.wait(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
            fault_hooks.reach_fault_hook(_HOOK)

        threads = [threading.Thread(target=_arrive) for _ in range(_CONCURRENT_ARRIVAL_COUNT)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)

        assert [thread.is_alive() for thread in threads] == [False] * _CONCURRENT_ARRIVAL_COUNT
        assert injected == ["sigkill"]


class TestArmingIsValidated:
    def test_a_second_request_at_an_armed_hook_is_rejected(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str]
    ) -> None:
        """Silently replacing the first request would strand whoever is waiting for its fault."""
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1")

        with pytest.raises(FaultHookAlreadyArmedError, match="req-1"):
            arm_fault_hook(hook=_HOOK.value, mode="exit", request_id="req-2")

        assert registry.armed_hooks()[_HOOK].request_id == "req-1"
        fault_hooks.reach_fault_hook(_HOOK)
        assert injected == ["sigkill"]

    def test_an_unknown_hook_name_is_rejected_before_anything_is_armed(
        self, registry: fault_hooks._FaultHookRegistry
    ) -> None:
        """A misspelled hook would otherwise sit armed at a point production never reaches."""
        with pytest.raises(ValueError):
            arm_fault_hook(hook="weight_update.before_typo", mode="sigkill", request_id="req-1")

        assert registry.armed_hooks() == {}

    def test_an_unknown_failure_mode_is_rejected_before_anything_is_armed(
        self, registry: fault_hooks._FaultHookRegistry
    ) -> None:
        """The mode is only interpreted when the hook fires, so an unknown one must be refused at arming time."""
        with pytest.raises(ValueError):
            arm_fault_hook(hook=_HOOK.value, mode="nuke", request_id="req-1")

        assert registry.armed_hooks() == {}

    def test_a_disarmed_hook_can_be_armed_again(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str]
    ) -> None:
        """A soak run arms the same point every round, and the slot must free itself when the fault fires."""
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1")
        fault_hooks.reach_fault_hook(_HOOK)

        arm_fault_hook(hook=_HOOK.value, mode="exit", request_id="req-2")
        fault_hooks.reach_fault_hook(_HOOK)

        assert injected == ["sigkill", "exit"]


class TestFireEvidence:
    def test_a_fire_is_recorded_as_an_event_before_the_fault_runs(
        self, registry: fault_hooks._FaultHookRegistry, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A sigkill leaves no chance to write afterwards, so the evidence of the fire must already be on disk."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )
        events_when_injected: list[list[Event]] = []
        monkeypatch.setattr(
            fault_hooks, "inject_fault", lambda mode: events_when_injected.append(read_events(tmp_path))
        )

        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1")
        with fault_hooks.weight_update_span(weight_version=7):
            fault_hooks.reach_fault_hook(_HOOK)

        [recorded] = events_when_injected
        [fire] = [event for event in recorded if isinstance(event, FaultHookFireEvent)]
        assert (fire.hook, fire.mode, fire.request_id, fire.weight_version) == (_HOOK.value, "sigkill", "req-1", 7)

    def test_a_process_without_an_event_logger_still_fires(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A process with no event dir must still fire, so unit-test paths keep working."""
        monkeypatch.setattr(event_logger_module, "_event_logger", None)

        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1")
        fault_hooks.reach_fault_hook(_HOOK)

        assert injected == ["sigkill"]


class TestWeightUpdateSpan:
    def test_a_fire_outside_an_update_names_no_version(
        self, registry: fault_hooks._FaultHookRegistry, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Every hook site sits inside an update today, and a fire without one must not claim to belong to any."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )
        monkeypatch.setattr(fault_hooks, "inject_fault", lambda mode: None)

        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1")
        fault_hooks.reach_fault_hook(_HOOK)

        [fire] = [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)]
        assert fire.weight_version is None

    def test_the_span_is_visible_from_a_thread_the_update_did_not_create(self) -> None:
        """The p2p write hooks run on per-cell writer threads, which no contextvar would reach."""
        seen: list[object] = []

        with fault_hooks.weight_update_span(weight_version=3):
            reader = threading.Thread(target=lambda: seen.append(fault_hooks.current_weight_update_span()))
            reader.start()
            reader.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)

        assert [span.weight_version for span in seen] == [3]

    def test_the_span_is_cleared_when_the_update_raises(self) -> None:
        """A failed update must not leave its version attached to the fires of the next one."""
        with pytest.raises(RuntimeError):
            with fault_hooks.weight_update_span(weight_version=3):
                raise RuntimeError("update failed")

        assert fault_hooks.current_weight_update_span() is None

    def test_a_second_span_in_one_process_is_refused(self) -> None:
        """Updates are serial in a trainer worker, and a nested span would misattribute whatever fires inside it."""
        with fault_hooks.weight_update_span(weight_version=3):
            with pytest.raises(AssertionError):
                with fault_hooks.weight_update_span(weight_version=4):
                    pass

        assert fault_hooks.current_weight_update_span() is None
