from __future__ import annotations

import threading

import pytest

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
