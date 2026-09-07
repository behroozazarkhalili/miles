from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from miles.utils.audit_utils.event_logger import logger as event_logger_module
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events
from miles.utils.audit_utils.event_logger.models import Event, FaultHookFireEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils import fault_hooks
from miles.utils.test_utils.fault_hooks import (
    MAX_FAULT_HOOK_DELAY_MS,
    FaultHookAlreadyArmedError,
    FaultHookName,
    FaultHookOutcome,
    FaultHookTarget,
    RemoteInferenceTarget,
    arm_fault_hook,
)
from miles.utils.test_utils.receiver_fault import ReceiverFaultRefusedError, ReceiverIdentity

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


_RECEIVER = ReceiverIdentity(
    receiver_boot_uuid="boot-1", session_id="session-1", rank=3, control_url="http://10.0.0.9:41111"
)
_REMOTE_TARGET = RemoteInferenceTarget(cell_id="engine-0", workers_hash="hash-1", receiver=_RECEIVER)

_LATER_RECEIVER = ReceiverIdentity(
    receiver_boot_uuid="boot-2", session_id="session-2", rank=5, control_url="http://10.0.0.11:41111"
)
_LATER_REMOTE_TARGET = RemoteInferenceTarget(cell_id="engine-1", workers_hash="hash-2", receiver=_LATER_RECEIVER)


@pytest.fixture
def remote_executor(monkeypatch: pytest.MonkeyPatch) -> list[tuple[RemoteInferenceTarget, str, str]]:
    delivered: list[tuple[RemoteInferenceTarget, str, str]] = []

    def execute(*, target: RemoteInferenceTarget, mode, request_id: str) -> FaultHookOutcome:
        delivered.append((target, mode.value, request_id))
        return FaultHookOutcome.ACCEPTED

    monkeypatch.setattr(fault_hooks, "_REMOTE_EXECUTOR", execute)
    return delivered


class TestArmingARemoteFault:
    def test_a_remote_request_at_a_hook_with_no_target_is_refused(
        self, registry: fault_hooks._FaultHookRegistry, remote_executor: list
    ) -> None:
        """A point that writes to nobody could only aim at a guessed cell, and guessing is not targeting."""
        with pytest.raises(fault_hooks.FaultHookTargetUnsupportedError):
            arm_fault_hook(
                hook=FaultHookName.WEIGHT_UPDATE_AFTER_BASE_WEIGHTS.value,
                mode="sigkill",
                request_id="req-1",
                target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
            )

        assert registry.armed_hooks() == {}

    def test_a_remote_request_in_a_process_that_cannot_reach_a_cell_is_refused(
        self, registry: fault_hooks._FaultHookRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Accepting it would answer ok and then drop the fault when the hook is reached."""
        monkeypatch.setattr(fault_hooks, "_REMOTE_EXECUTOR", None)

        with pytest.raises(fault_hooks.FaultHookTargetUnsupportedError):
            arm_fault_hook(
                hook=_OTHER_HOOK.value,
                mode="sigkill",
                request_id="req-1",
                target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
            )

        assert registry.armed_hooks() == {}

    def test_an_unknown_target_is_refused_before_anything_is_armed(
        self, registry: fault_hooks._FaultHookRegistry, remote_executor: list
    ) -> None:
        """The target decides who dies, so a value nobody implements must not be read as the default."""
        with pytest.raises(ValueError):
            arm_fault_hook(hook=_OTHER_HOOK.value, mode="sigkill", request_id="req-1", target="somebody_else")

        assert registry.armed_hooks() == {}


class TestReachingARemoteFault:
    def test_the_fault_is_delivered_to_the_target_the_site_names(
        self,
        registry: fault_hooks._FaultHookRegistry,
        injected: list[str],
        remote_executor: list,
    ) -> None:
        """The victim comes from the write's own context, never from the request, which names no cell."""
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )

        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        assert remote_executor == [(_REMOTE_TARGET, "sigkill", "req-1")]
        assert injected == []

    def test_a_site_that_names_no_target_refuses_to_guess(
        self, registry: fault_hooks._FaultHookRegistry, remote_executor: list
    ) -> None:
        """Falling back to any cell id would let a fault land on a target this rank never wrote to."""
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )

        with pytest.raises(AssertionError, match="named no inference target"):
            fault_hooks.reach_fault_hook(_OTHER_HOOK)

        assert remote_executor == []

    def test_a_local_request_ignores_the_target_the_site_offers(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str], remote_executor: list
    ) -> None:
        """The same site serves both kinds of request, and a local one must harm the rank that reached it."""
        arm_fault_hook(hook=_OTHER_HOOK.value, mode="sigkill", request_id="req-1")

        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        assert injected == ["sigkill"]
        assert remote_executor == []

    def test_a_refused_remote_fault_is_recorded_as_a_stale_target(
        self,
        registry: fault_hooks._FaultHookRegistry,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A fault the backend refused harmed nobody, and a witness reading the fire must be able to tell."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )
        monkeypatch.setattr(
            fault_hooks, "_REMOTE_EXECUTOR", lambda *, target, mode, request_id: FaultHookOutcome.STALE_TARGET
        )

        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )
        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        [fire] = [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)]
        assert fire.outcome == FaultHookOutcome.STALE_TARGET.value
        assert (fire.victim_cell_id, fire.victim_workers_hash) == ("engine-0", "hash-1")

    def test_a_delivered_remote_fault_records_its_victim(
        self,
        registry: fault_hooks._FaultHookRegistry,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        remote_executor: list,
    ) -> None:
        """The trigger and the victim are different cells, and a run has to be able to tell them apart."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )

        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )
        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        [fire] = [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)]
        assert (fire.target, fire.outcome) == (FaultHookTarget.REMOTE_INFERENCE_CELL.value, "accepted")
        assert (fire.victim_cell_id, fire.victim_worker_in_cell_index) == ("engine-0", None)
        assert (fire.victim_receiver_rank, fire.victim_receiver_boot_uuid, fire.victim_session_id) == (
            3,
            "boot-1",
            "session-1",
        )


class TestRemoteOutcomesAreNotFires:
    def test_an_accepted_request_is_not_recorded_as_fired(
        self,
        registry: fault_hooks._FaultHookRegistry,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        remote_executor: list,
    ) -> None:
        """The receiver taking the request is not the receiver signalling itself, and a witness must tell them apart."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )

        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )
        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        [fire] = [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)]
        assert fire.outcome == FaultHookOutcome.ACCEPTED.value

    def test_an_unknown_answer_is_recorded_as_unknown(
        self,
        registry: fault_hooks._FaultHookRegistry,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A lost answer may still have landed, so it is neither a fire nor a stale target."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )
        monkeypatch.setattr(
            fault_hooks, "_REMOTE_EXECUTOR", lambda *, target, mode, request_id: FaultHookOutcome.UNKNOWN
        )

        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )
        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        [fire] = [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)]
        assert fire.outcome == FaultHookOutcome.UNKNOWN.value

    def test_a_refusal_is_recorded_and_then_raised(
        self,
        registry: fault_hooks._FaultHookRegistry,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A refusal leaves the request unaccounted for, and a run that swallowed it would report green."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )

        def refuse(*, target, mode, request_id):
            raise ReceiverFaultRefusedError("pending_action_exists")

        monkeypatch.setattr(fault_hooks, "_REMOTE_EXECUTOR", refuse)

        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )
        with pytest.raises(ReceiverFaultRefusedError):
            fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        [fire] = [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)]
        assert fire.outcome == FaultHookOutcome.REFUSED.value


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

    def test_a_frozen_span_wins_over_the_one_open_now(
        self, registry: fault_hooks._FaultHookRegistry, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A writer that reaches its hook after the next update started must not name that next update."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )
        monkeypatch.setattr(fault_hooks, "inject_fault", lambda mode: None)
        frozen = fault_hooks.WeightUpdateSpan(weight_version=3)

        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1")
        with fault_hooks.weight_update_span(weight_version=4):
            fault_hooks.reach_fault_hook(_HOOK, span=frozen)

        [fire] = [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)]
        assert fire.weight_version == 3

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


# =================================== delays ===================================


class _FakeTimer:
    def __init__(self, *, delay_seconds: float, run: Callable[[], None], name: str) -> None:
        self.delay_seconds = delay_seconds
        self.name = name
        self.started = False
        self._run = run

    def start(self) -> None:
        self.started = True

    def elapse(self) -> None:
        self._run()


@pytest.fixture
def timers(monkeypatch: pytest.MonkeyPatch) -> list[_FakeTimer]:
    created: list[_FakeTimer] = []

    def create(*, delay_seconds: float, run: Callable[[], None], name: str) -> _FakeTimer:
        created.append(_FakeTimer(delay_seconds=delay_seconds, run=run, name=name))
        return created[-1]

    monkeypatch.setattr(fault_hooks, "_create_delay_timer", create)
    return created


class TestADelayedFaultLeavesTheCallerAlone:
    def test_reaching_the_hook_schedules_the_fault_and_returns(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str], timers: list[_FakeTimer]
    ) -> None:
        """The weight update runs on the thread that reached the point, so it must not wait out the delay there."""
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1", delay_ms=250)

        fault_hooks.reach_fault_hook(_HOOK)

        assert [(timer.delay_seconds, timer.started) for timer in timers] == [(0.25, True)]
        assert injected == []

    def test_the_fault_runs_when_the_delay_runs_out(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str], timers: list[_FakeTimer]
    ) -> None:
        """A delayed request is still a request for a real fault, not for a timer nobody ever fires."""
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1", delay_ms=250)
        fault_hooks.reach_fault_hook(_HOOK)

        timers[0].elapse()

        assert injected == ["sigkill"]

    def test_the_fault_runs_once_however_often_the_hook_is_reached(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str], timers: list[_FakeTimer]
    ) -> None:
        """Every rank walks the same point on every update, and one request buys exactly one fault."""
        arm_fault_hook(hook=_HOOK.value, mode="exit", request_id="req-1", delay_ms=250)

        fault_hooks.reach_fault_hook(_HOOK)
        fault_hooks.reach_fault_hook(_HOOK)
        timers[0].elapse()
        fault_hooks.reach_fault_hook(_HOOK)

        assert len(timers) == 1
        assert injected == ["exit"]

    def test_an_immediate_request_runs_the_fault_before_the_caller_moves_on(
        self, registry: fault_hooks._FaultHookRegistry, monkeypatch: pytest.MonkeyPatch, timers: list[_FakeTimer]
    ) -> None:
        """Everything armed before delays existed crashes the thread that reached the hook, and still must."""
        order: list[str] = []
        monkeypatch.setattr(fault_hooks, "inject_fault", lambda mode: order.append(f"injected {mode}"))
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1")

        fault_hooks.reach_fault_hook(_HOOK)
        order.append("the hook returned")

        assert timers == []
        assert order == ["injected sigkill", "the hook returned"]

    def test_the_delay_is_carried_into_the_fire_record(
        self,
        registry: fault_hooks._FaultHookRegistry,
        injected: list[str],
        timers: list[_FakeTimer],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A witness reading the fire has to tell a fault that waited from one that landed inside the update."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )

        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1", delay_ms=250)
        fault_hooks.reach_fault_hook(_HOOK)
        assert [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)] == []

        timers[0].elapse()

        [fire] = [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)]
        assert (fire.delay_ms, fire.outcome) == (250, FaultHookOutcome.FIRED.value)


class TestADelayedFaultKeepsHoldingItsHook:
    def test_the_hook_cannot_be_armed_again_while_its_fault_is_still_waiting(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str], timers: list[_FakeTimer]
    ) -> None:
        """Two outstanding faults at one point would leave whoever armed the second waiting for the first's."""
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1", delay_ms=250)
        fault_hooks.reach_fault_hook(_HOOK)

        with pytest.raises(FaultHookAlreadyArmedError, match="req-1"):
            arm_fault_hook(hook=_HOOK.value, mode="exit", request_id="req-2", delay_ms=250)

        assert registry.armed_hooks() == {}
        assert [action.request_id for action in registry.pending_hooks().values()] == ["req-1"]

    def test_the_hook_frees_itself_once_the_delayed_fault_has_run(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str], timers: list[_FakeTimer]
    ) -> None:
        """A soak arms the same point every round, so a slot that never frees itself would stop the run."""
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1", delay_ms=250)
        fault_hooks.reach_fault_hook(_HOOK)
        timers[0].elapse()

        arm_fault_hook(hook=_HOOK.value, mode="exit", request_id="req-2", delay_ms=250)
        fault_hooks.reach_fault_hook(_HOOK)
        timers[1].elapse()

        assert registry.pending_hooks() == {}
        assert injected == ["sigkill", "exit"]

    def test_the_hook_stays_held_while_its_fault_is_running(
        self, registry: fault_hooks._FaultHookRegistry, monkeypatch: pytest.MonkeyPatch, timers: list[_FakeTimer]
    ) -> None:
        """A local kill never returns, so the slot has to be held for the execution and not only for the wait."""
        pending_during_injection: list[list[str]] = []
        monkeypatch.setattr(
            fault_hooks,
            "inject_fault",
            lambda mode: pending_during_injection.append(
                [action.request_id for action in registry.pending_hooks().values()]
            ),
        )

        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1", delay_ms=250)
        fault_hooks.reach_fault_hook(_HOOK)
        timers[0].elapse()

        assert pending_during_injection == [["req-1"]]

    def test_an_immediate_request_holds_no_slot_after_it_has_fired(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str], timers: list[_FakeTimer]
    ) -> None:
        """Nothing outlives the call, so a pending slot left behind would refuse the next arm forever."""
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1")
        fault_hooks.reach_fault_hook(_HOOK)

        assert registry.pending_hooks() == {}
        arm_fault_hook(hook=_HOOK.value, mode="exit", request_id="req-2")

    def test_no_more_faults_can_ever_be_outstanding_than_there_are_hooks(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str], timers: list[_FakeTimer]
    ) -> None:
        """A soak arming in a loop must not be able to pile up timers in a worker that is trying to train."""
        for index, hook in enumerate(FaultHookName):
            arm_fault_hook(hook=hook.value, mode="sigkill", request_id=f"req-{index}", delay_ms=250)
        for hook in FaultHookName:
            for _ in range(3):
                fault_hooks.reach_fault_hook(hook)

        assert len(timers) == len(FaultHookName)
        assert len(registry.pending_hooks()) == len(FaultHookName)


class TestAnImmediateFaultAlsoHoldsItsHook:
    def test_the_hook_cannot_be_armed_again_while_an_immediate_fault_is_still_being_delivered(
        self, registry: fault_hooks._FaultHookRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A remote fault is delivered over the network, and the point stays taken for as long as that call runs."""
        refusals: list[str] = []

        def _deliver(*, target: RemoteInferenceTarget, mode, request_id: str) -> FaultHookOutcome:
            with pytest.raises(FaultHookAlreadyArmedError, match="req-1") as refused:
                arm_fault_hook(
                    hook=_OTHER_HOOK.value,
                    mode="exit",
                    request_id="req-2",
                    target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
                )
            refusals.append(str(refused.value))
            return FaultHookOutcome.ACCEPTED

        monkeypatch.setattr(fault_hooks, "_REMOTE_EXECUTOR", _deliver)
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )

        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        assert len(refusals) == 1

    def test_another_writer_cannot_consume_a_fault_armed_while_the_first_is_still_delivering(
        self, registry: fault_hooks._FaultHookRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each cell has its own writer thread, and a second one must not carry a fault into a second engine."""
        delivered: list[str] = []

        def _deliver(*, target: RemoteInferenceTarget, mode, request_id: str) -> FaultHookOutcome:
            delivered.append(target.cell_id)
            if len(delivered) == 1:
                with pytest.raises(FaultHookAlreadyArmedError):
                    arm_fault_hook(
                        hook=_OTHER_HOOK.value,
                        mode="exit",
                        request_id="req-2",
                        target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
                    )
                fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_LATER_REMOTE_TARGET)
            return FaultHookOutcome.ACCEPTED

        monkeypatch.setattr(fault_hooks, "_REMOTE_EXECUTOR", _deliver)
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )

        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        assert delivered == [_REMOTE_TARGET.cell_id]

    def test_the_hook_is_armable_again_once_the_immediate_fault_has_been_delivered(
        self, registry: fault_hooks._FaultHookRegistry, remote_executor: list
    ) -> None:
        """The point is only taken while the fault is being delivered, or a soak could arm it exactly once."""
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )
        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="exit",
            request_id="req-2",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )
        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_LATER_REMOTE_TARGET)

        assert registry.pending_hooks() == {}
        assert remote_executor == [
            (_REMOTE_TARGET, "sigkill", "req-1"),
            (_LATER_REMOTE_TARGET, "exit", "req-2"),
        ]

    def test_an_immediate_request_whose_site_names_no_target_frees_the_hook(
        self, registry: fault_hooks._FaultHookRegistry, remote_executor: list
    ) -> None:
        """The request never reached anybody, so holding its point would refuse every later arm for nothing."""
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )

        with pytest.raises(AssertionError, match="named no inference target"):
            fault_hooks.reach_fault_hook(_OTHER_HOOK)

        assert registry.pending_hooks() == {}
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="exit",
            request_id="req-2",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )

    def test_an_immediate_remote_fault_that_raises_frees_the_hook(
        self, registry: fault_hooks._FaultHookRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A delivery that blew up left the point unheld, and the run has to be able to arm it again."""

        def _explode(*, target: RemoteInferenceTarget, mode, request_id: str) -> FaultHookOutcome:
            raise ConnectionError("the worker manager is gone")

        monkeypatch.setattr(fault_hooks, "_REMOTE_EXECUTOR", _explode)
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )

        with pytest.raises(ConnectionError):
            fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        assert registry.pending_hooks() == {}
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="exit",
            request_id="req-2",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )

    def test_a_refused_immediate_remote_fault_frees_the_hook(
        self, registry: fault_hooks._FaultHookRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A receiver that refused the request harmed nobody, and the point it took must not stay taken."""

        def _refuse(*, target: RemoteInferenceTarget, mode, request_id: str) -> FaultHookOutcome:
            raise ReceiverFaultRefusedError("pending_action_exists")

        monkeypatch.setattr(fault_hooks, "_REMOTE_EXECUTOR", _refuse)
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )

        with pytest.raises(ReceiverFaultRefusedError):
            fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        assert registry.pending_hooks() == {}

    def test_immediate_and_delayed_faults_share_the_one_slot_per_hook_bound(
        self, registry: fault_hooks._FaultHookRegistry, monkeypatch: pytest.MonkeyPatch, timers: list[_FakeTimer]
    ) -> None:
        """Both kinds hold a point the same way, so a worker can never carry more of them than it has hooks."""
        outstanding: list[int] = []
        monkeypatch.setattr(
            fault_hooks, "inject_fault", lambda mode: outstanding.append(len(registry.pending_hooks()))
        )
        delayed, immediate = list(FaultHookName)[:2], list(FaultHookName)[2:]
        for index, hook in enumerate(FaultHookName):
            arm_fault_hook(
                hook=hook.value,
                mode="sigkill",
                request_id=f"req-{index}",
                delay_ms=250 if hook in delayed else 0,
            )

        for hook in delayed:
            fault_hooks.reach_fault_hook(hook)
        for hook in immediate:
            fault_hooks.reach_fault_hook(hook)

        assert outstanding == [len(delayed) + 1, len(delayed) + 1]
        assert max(outstanding) <= len(FaultHookName)
        assert sorted(registry.pending_hooks()) == sorted(delayed)


class TestADelayedFaultKeepsTheContextItWasReachedWith:
    def test_the_victim_is_the_target_the_site_named_when_the_hook_was_reached(
        self, registry: fault_hooks._FaultHookRegistry, remote_executor: list, timers: list[_FakeTimer]
    ) -> None:
        """By the time the delay runs out this rank writes to somebody else, who never had the fault aimed at them."""
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
            delay_ms=250,
        )
        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_LATER_REMOTE_TARGET)
        timers[0].elapse()

        assert remote_executor == [(_REMOTE_TARGET, "sigkill", "req-1")]

    def test_the_request_id_of_the_frozen_action_reaches_the_receiver(
        self, registry: fault_hooks._FaultHookRegistry, remote_executor: list, timers: list[_FakeTimer]
    ) -> None:
        """The receiver answers for a request id, and a delivery naming another one proves nothing about this arm."""
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
            delay_ms=250,
        )
        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        timers[0].elapse()

        assert [request_id for _, _, request_id in remote_executor] == ["req-1"]

    def test_the_executor_frozen_at_the_hook_delivers_the_fault(
        self, registry: fault_hooks._FaultHookRegistry, remote_executor: list, timers: list[_FakeTimer]
    ) -> None:
        """An executor installed later belongs to a process state nobody checked this request against."""
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
            delay_ms=250,
        )
        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        replaced: list[RemoteInferenceTarget] = []
        fault_hooks.install_remote_fault_executor(
            lambda *, target, mode, request_id: replaced.append(target) or FaultHookOutcome.ACCEPTED
        )
        timers[0].elapse()
        fault_hooks.install_remote_fault_executor(None)

        assert remote_executor == [(_REMOTE_TARGET, "sigkill", "req-1")]
        assert replaced == []

    def test_the_span_the_hook_was_reached_in_is_the_one_the_fire_names(
        self,
        registry: fault_hooks._FaultHookRegistry,
        injected: list[str],
        timers: list[_FakeTimer],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The fault runs after the next update opened, and blaming that update would move the harm one version on."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )

        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1", delay_ms=250)
        with fault_hooks.weight_update_span(weight_version=3):
            fault_hooks.reach_fault_hook(_HOOK)
        with fault_hooks.weight_update_span(weight_version=4):
            timers[0].elapse()

        [fire] = [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)]
        assert fire.weight_version == 3

    def test_the_span_the_site_handed_over_survives_the_delay(
        self,
        registry: fault_hooks._FaultHookRegistry,
        remote_executor: list,
        timers: list[_FakeTimer],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A p2p writer carries the span of the update that queued it, and the timer must not re-read the global one."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )

        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
            delay_ms=250,
        )
        with fault_hooks.weight_update_span(weight_version=9):
            fault_hooks.reach_fault_hook(
                _OTHER_HOOK, remote_target=_REMOTE_TARGET, span=fault_hooks.WeightUpdateSpan(weight_version=3)
            )
        timers[0].elapse()

        [fire] = [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)]
        assert fire.weight_version == 3

    def test_a_site_that_names_no_target_is_refused_where_it_is_reached(
        self, registry: fault_hooks._FaultHookRegistry, remote_executor: list, timers: list[_FakeTimer]
    ) -> None:
        """The caller has the context to report the mistake, and a timer would only lose it."""
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
            delay_ms=250,
        )

        with pytest.raises(AssertionError, match="named no inference target"):
            fault_hooks.reach_fault_hook(_OTHER_HOOK)

        assert timers == []
        assert registry.pending_hooks() == {}


class TestADelayThatCannotBeHonoured:
    def test_a_timer_that_refuses_to_start_frees_the_hook_and_tells_the_caller(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worker out of threads must not answer a scheduled fault that nothing will ever run."""

        def _refuse(*, delay_seconds: float, run: Callable[[], None], name: str) -> fault_hooks.DelayTimer:
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(fault_hooks, "_create_delay_timer", _refuse)
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1", delay_ms=250)

        with pytest.raises(RuntimeError, match="can't start new thread"):
            fault_hooks.reach_fault_hook(_HOOK)

        assert injected == []
        assert registry.pending_hooks() == {}
        arm_fault_hook(hook=_HOOK.value, mode="exit", request_id="req-2")

    def test_a_timer_that_refuses_to_start_is_recorded_as_a_fault_nobody_scheduled(
        self, registry: fault_hooks._FaultHookRegistry, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A request dropped in silence would leave the run waiting for a fault and then passing without one."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )

        def _refuse(*, delay_seconds: float, run: Callable[[], None], name: str) -> fault_hooks.DelayTimer:
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(fault_hooks, "_create_delay_timer", _refuse)
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1", delay_ms=250)

        with pytest.raises(RuntimeError):
            fault_hooks.reach_fault_hook(_HOOK)

        [fire] = [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)]
        assert (fire.outcome, fire.delay_ms) == (FaultHookOutcome.NOT_SCHEDULED.value, 250)

    def test_a_remote_executor_that_raises_after_the_delay_is_recorded_and_frees_the_hook(
        self,
        registry: fault_hooks._FaultHookRegistry,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        timers: list[_FakeTimer],
    ) -> None:
        """Nobody is left to catch the failure of a fault that runs on a timer, so the record has to carry it."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )

        def _explode(*, target: RemoteInferenceTarget, mode, request_id: str) -> FaultHookOutcome:
            raise ConnectionError("the worker manager is gone")

        monkeypatch.setattr(fault_hooks, "_REMOTE_EXECUTOR", _explode)
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
            delay_ms=250,
        )
        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        timers[0].elapse()

        [fire] = [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)]
        assert fire.outcome == FaultHookOutcome.ERRORED.value
        assert (fire.victim_cell_id, fire.request_id) == ("engine-0", "req-1")
        assert registry.pending_hooks() == {}

    def test_a_receiver_that_refuses_after_the_delay_is_still_recorded_as_a_refusal(
        self,
        registry: fault_hooks._FaultHookRegistry,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        timers: list[_FakeTimer],
    ) -> None:
        """A refusal is a different answer from a delivery that blew up, and the delay must not blur the two."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )

        def _refuse(*, target: RemoteInferenceTarget, mode, request_id: str) -> FaultHookOutcome:
            raise ReceiverFaultRefusedError("pending_action_exists")

        monkeypatch.setattr(fault_hooks, "_REMOTE_EXECUTOR", _refuse)
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
            delay_ms=250,
        )
        fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        timers[0].elapse()

        [fire] = [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)]
        assert fire.outcome == FaultHookOutcome.REFUSED.value
        assert registry.pending_hooks() == {}

    def test_a_remote_executor_that_raises_without_a_delay_still_reaches_its_caller(
        self, registry: fault_hooks._FaultHookRegistry, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The thread that reached the hook can report the failure, and losing it would hide a fault nobody sent."""
        monkeypatch.setattr(
            event_logger_module,
            "_event_logger",
            EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"), file_name="worker.jsonl"),
        )

        def _explode(*, target: RemoteInferenceTarget, mode, request_id: str) -> FaultHookOutcome:
            raise ConnectionError("the worker manager is gone")

        monkeypatch.setattr(fault_hooks, "_REMOTE_EXECUTOR", _explode)
        arm_fault_hook(
            hook=_OTHER_HOOK.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.REMOTE_INFERENCE_CELL.value,
        )

        with pytest.raises(ConnectionError):
            fault_hooks.reach_fault_hook(_OTHER_HOOK, remote_target=_REMOTE_TARGET)

        [fire] = [event for event in read_events(tmp_path) if isinstance(event, FaultHookFireEvent)]
        assert fire.outcome == FaultHookOutcome.ERRORED.value


class TestADelayIsValidatedWhereItIsArmed:
    @pytest.mark.parametrize("delay_ms", [-1, MAX_FAULT_HOOK_DELAY_MS + 1, 1.5, True, "250", None])
    def test_a_delay_that_is_not_a_bounded_whole_number_of_milliseconds_is_refused(
        self, registry: fault_hooks._FaultHookRegistry, delay_ms: object
    ) -> None:
        """A delay the worker cannot honour must fail the arm rather than be read as some other wait."""
        with pytest.raises(ValidationError):
            arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1", delay_ms=delay_ms)

        assert registry.armed_hooks() == {}

    @pytest.mark.parametrize("delay_ms", [0, 1, MAX_FAULT_HOOK_DELAY_MS])
    def test_the_ends_of_the_accepted_range_are_armed(
        self, registry: fault_hooks._FaultHookRegistry, delay_ms: int
    ) -> None:
        """Both bounds are inclusive, and a scenario arming at one of them must not be refused."""
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1", delay_ms=delay_ms)

        assert registry.armed_hooks()[_HOOK].delay_ms == delay_ms

    def test_a_request_that_names_no_delay_fires_where_it_is_reached(
        self, registry: fault_hooks._FaultHookRegistry, injected: list[str], timers: list[_FakeTimer]
    ) -> None:
        """Every caller written before delays existed keeps asking for a fault in the thread that reached the hook."""
        arm_fault_hook(hook=_HOOK.value, mode="sigkill", request_id="req-1")

        assert registry.armed_hooks()[_HOOK].delay_ms == 0
        fault_hooks.reach_fault_hook(_HOOK)
        assert (timers, injected) == ([], ["sigkill"])
