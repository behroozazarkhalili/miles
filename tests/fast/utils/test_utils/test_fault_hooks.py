from __future__ import annotations

import threading
from pathlib import Path

import pytest

from miles.utils.audit_utils.event_logger import logger as event_logger_module
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events
from miles.utils.audit_utils.event_logger.models import Event, FaultHookFireEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils import fault_hooks
from miles.utils.test_utils.fault_hooks import (
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
