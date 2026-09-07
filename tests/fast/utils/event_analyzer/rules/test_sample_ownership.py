from datetime import datetime, timedelta, timezone
from typing import Literal

import pytest

from miles.utils.audit_utils.event_analyzer.analyzer import _partition_by_model_id
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership import (
    SampleConsumedTwiceIssue,
    SampleLostIssue,
    check,
)
from miles.utils.audit_utils.event_logger.models import (
    RolloutHoldingsSnapshotEvent,
    RolloutStateRestoreEvent,
    SampleOwner,
    SampleOwnerTransitionEvent,
    TrainerTrainedSamplesEvent,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
SOURCE = SimpleProcessIdentity(component="rollout_executor")


def at(seconds: int) -> datetime:
    return START + timedelta(seconds=seconds)


def transition(
    seconds: int, *, indices: list[int], to_owner: SampleOwner, lineage_id: str | None = None
) -> SampleOwnerTransitionEvent:
    return SampleOwnerTransitionEvent(
        timestamp=at(seconds),
        source=SOURCE,
        sample_indices=indices,
        from_owner=SampleOwner.DATA_SOURCE,
        to_owner=to_owner,
        lineage_id=lineage_id,
    )


def snapshot(
    seconds: int,
    *,
    rollout_id: int,
    holdings: dict[SampleOwner, list[int]],
    reason: Literal["step", "save", "final"] = "step",
    lineage_id: str | None = None,
    replays_samples: bool = False,
    trainer_model_id: str | None = None,
) -> RolloutHoldingsSnapshotEvent:
    return RolloutHoldingsSnapshotEvent(
        timestamp=at(seconds),
        source=SOURCE,
        rollout_id=rollout_id,
        holdings=holdings,
        trainer_model_id=trainer_model_id,
        replays_samples=replays_samples,
        reason=reason,
        lineage_id=lineage_id,
    )


def trained(
    seconds: int,
    *,
    rollout_id: int,
    indices: list[int],
    lineage_id: str | None = None,
    trainer_model_id: str | None = None,
) -> TrainerTrainedSamplesEvent:
    return TrainerTrainedSamplesEvent(
        timestamp=at(seconds),
        source=SOURCE,
        rollout_id=rollout_id,
        sample_indices=indices,
        lineage_id=lineage_id,
        trainer_model_id=trainer_model_id,
    )


def generated(seconds: int, *indices: int) -> SampleOwnerTransitionEvent:
    return transition(seconds, indices=list(indices), to_owner=SampleOwner.IN_FLIGHT)


class TestNothingIsLost:
    def test_restored_holdings_that_disappear_are_reported(self) -> None:
        """Restored output belongs to the new lineage and must not silently disappear."""
        events = [
            restore(1, rollout_id=0, lineage_id="b", parent_lineage_id="a"),
            transition(2, indices=[1], to_owner=SampleOwner.OUTPUT_BUFFER, lineage_id="b"),
            snapshot(3, rollout_id=1, holdings={}, reason="save", lineage_id="b"),
        ]

        [issue] = check(events)
        assert isinstance(issue, SampleLostIssue)
        assert issue.sample_indices == [1]

    def test_the_last_step_loss_is_reported_without_a_final_checkpoint(self) -> None:
        """Final snapshots detect a permanent loss even before three runtime snapshots."""
        events = [
            generated(1, 1),
            snapshot(2, rollout_id=0, holdings={}),
            snapshot(3, rollout_id=0, holdings={}, reason="final"),
        ]

        [issue] = check(events)
        assert isinstance(issue, SampleLostIssue)
        assert issue.sample_indices == [1]

    def test_a_sample_held_by_the_buffer_is_accounted_for(self):
        """The buffer reporting what it holds is the whole point of DataBuffer.snapshot."""
        events = [
            generated(1, 1),
            snapshot(2, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: [1]}, reason="save"),
        ]

        assert check(events) == []

    def test_a_sample_that_reached_the_weights_is_accounted_for(self):
        """A trained sample is owned by the weights and by nothing the rollout side still holds."""
        events = [
            generated(1, 1),
            trained(2, rollout_id=0, indices=[1]),
            snapshot(3, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: []}, reason="save"),
        ]

        assert check(events) == []

    def test_a_sample_with_an_explicit_drop_is_accounted_for(self):
        """A dynamic-filter reject is a decision, and the checker must not confuse it with a leak."""
        events = [
            generated(1, 1),
            transition(2, indices=[1], to_owner=SampleOwner.DROPPED),
            snapshot(3, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: []}, reason="save"),
        ]

        assert check(events) == []

    def test_a_sample_nobody_owns_at_a_checkpoint_is_reported(self):
        """This is the failure a checkpoint makes permanent: the resumed run never sees that prompt again."""
        events = [
            generated(1, 1, 2),
            snapshot(2, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: [1]}, reason="save"),
        ]

        [issue] = check(events)

        assert isinstance(issue, SampleLostIssue)
        assert issue.sample_indices == [2]

    def test_one_unaccounted_instant_is_tolerated(self):
        """The snapshots come from several processes, so a sample can be between two of them for a moment."""
        events = [
            generated(1, 1),
            snapshot(2, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: []}),
            snapshot(3, rollout_id=1, holdings={SampleOwner.OUTPUT_BUFFER: [1]}),
            snapshot(4, rollout_id=2, holdings={SampleOwner.OUTPUT_BUFFER: [1]}),
        ]

        assert check(events) == []

    def test_a_sample_unaccounted_at_three_snapshots_in_a_row_is_reported(self):
        """A leak that outlives the lag between processes is a leak."""
        events = [
            generated(1, 1),
            snapshot(2, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: []}),
            snapshot(3, rollout_id=1, holdings={SampleOwner.OUTPUT_BUFFER: []}),
            snapshot(4, rollout_id=2, holdings={SampleOwner.OUTPUT_BUFFER: []}),
        ]

        [issue] = check(events)

        assert issue.sample_indices == [1]

    def test_an_old_snapshot_cannot_hide_loss_in_the_latest_policy_snapshot(self) -> None:
        """Stale holdings cannot account for a sample missing from the policy's latest snapshot."""
        events = [
            generated(1, 1, 2),
            snapshot(2, rollout_id=0, trainer_model_id="solver", holdings={SampleOwner.OUTPUT_BUFFER: [1, 2]}),
            snapshot(
                3, rollout_id=0, trainer_model_id="solver", holdings={SampleOwner.OUTPUT_BUFFER: [2]}, reason="save"
            ),
        ]

        [issue] = check(events)
        assert isinstance(issue, SampleLostIssue)
        assert issue.sample_indices == [1]

    def test_a_run_that_never_snapshotted_is_not_judged(self):
        """Without a snapshot there is no instant to evaluate an owner at, so silence is the only honest answer."""
        assert check([generated(1, 1)]) == []


class TestNothingIsConsumedTwice:
    def test_sibling_indices_trained_within_one_event_are_counted_once(self) -> None:
        """Sibling indices within one event are consumed once."""
        events = [trained(seconds=i, rollout_id=i, indices=[1, 1]) for i in range(1)]
        events.append(snapshot(3, rollout_id=2, holdings={SampleOwner.HANDED_TO_TRAINER: [1]}))

        assert check(events) == []

    def test_sibling_indices_trained_in_separate_events_are_reported(self) -> None:
        """Sibling indices consumed in separate events reveal replay."""
        events = [trained(seconds=i, rollout_id=i, indices=[1, 1]) for i in range(2)]
        events.append(snapshot(3, rollout_id=2, holdings={SampleOwner.HANDED_TO_TRAINER: [1]}))

        [issue] = check(events)
        assert isinstance(issue, SampleConsumedTwiceIssue)
        assert issue.sample_indices == [1]

    def test_sibling_indices_handed_within_one_event_are_counted_once(self) -> None:
        """Sibling indices within one event are consumed once."""
        events = [transition(seconds=i, indices=[1, 1], to_owner=SampleOwner.HANDED_TO_TRAINER) for i in range(1)]
        events.append(snapshot(3, rollout_id=2, holdings={SampleOwner.HANDED_TO_TRAINER: [1]}))

        assert check(events) == []

    def test_sibling_indices_handed_in_separate_events_are_reported(self) -> None:
        """Sibling indices consumed in separate events reveal replay."""
        events = [transition(seconds=i, indices=[1, 1], to_owner=SampleOwner.HANDED_TO_TRAINER) for i in range(2)]
        events.append(snapshot(3, rollout_id=2, holdings={SampleOwner.HANDED_TO_TRAINER: [1]}))

        [issue] = check(events)
        assert isinstance(issue, SampleConsumedTwiceIssue)
        assert issue.sample_indices == [1]

    def test_a_sample_trained_twice_is_reported(self):
        """The same gradient applied twice moves training off the curve the run reports."""
        events = [
            generated(1, 1),
            trained(2, rollout_id=0, indices=[1]),
            trained(3, rollout_id=1, indices=[1]),
            snapshot(4, rollout_id=1, holdings={SampleOwner.OUTPUT_BUFFER: []}),
        ]

        [issue] = check(events)

        assert isinstance(issue, SampleConsumedTwiceIssue)
        assert issue.sample_indices == [1]

    def test_a_sample_handed_to_training_twice_is_reported(self):
        """An executor replay that fires twice would train the same batch twice."""
        events = [
            generated(1, 1),
            transition(2, indices=[1], to_owner=SampleOwner.HANDED_TO_TRAINER),
            transition(3, indices=[1], to_owner=SampleOwner.HANDED_TO_TRAINER),
            trained(4, rollout_id=0, indices=[1]),
            snapshot(5, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: []}),
        ]

        [issue] = check(events)

        assert issue.sample_indices == [1]

    def test_generating_a_sample_twice_is_not_a_violation(self):
        """A retry regenerates a prompt under its own index; only reaching the weights twice is wrong."""
        events = [
            generated(1, 1),
            transition(2, indices=[1], to_owner=SampleOwner.RETRY_BUFFER),
            generated(3, 1),
            trained(4, rollout_id=0, indices=[1]),
            snapshot(5, rollout_id=0, holdings={SampleOwner.OUTPUT_BUFFER: []}, reason="save"),
        ]

        assert check(events) == []

    def test_a_buffer_that_declares_replays_is_only_held_to_not_losing_samples(self):
        """A replay buffer hands the same sample out on purpose, and that is not the invariant to enforce on it."""
        events = [
            generated(1, 1),
            trained(2, rollout_id=0, indices=[1]),
            trained(3, rollout_id=1, indices=[1]),
            snapshot(4, rollout_id=1, holdings={SampleOwner.OUTPUT_BUFFER: []}, replays_samples=True),
        ]

        assert check(events) == []


class TestLineage:
    def test_a_restored_batch_can_be_handed_once_without_counting_initialization(self) -> None:
        """Registering restored holdings is not another handoff to the trainer."""
        initial = transition(2, indices=[1], to_owner=SampleOwner.HANDED_TO_TRAINER, lineage_id="b")
        events = [
            restore(1, rollout_id=0, lineage_id="b", parent_lineage_id="a"),
            initial.model_copy(update={"reason": "restored"}),
            transition(3, indices=[1], to_owner=SampleOwner.HANDED_TO_TRAINER, lineage_id="b"),
            snapshot(4, rollout_id=1, holdings={SampleOwner.HANDED_TO_TRAINER: [1]}, lineage_id="b"),
        ]

        assert check(events) == []

    def test_single_restore_counts_only_training_retained_in_the_weights(self) -> None:
        """A restore retains ancestor training only through its checkpoint step."""
        events = [
            trained(2, rollout_id=0, indices=[1], lineage_id="a"),
            trained(3, rollout_id=1, indices=[2], lineage_id="a"),
            restore(5, rollout_id=0, lineage_id="b", parent_lineage_id="a"),
            trained(6, rollout_id=1, indices=[2], lineage_id="b"),
            snapshot(7, rollout_id=1, holdings={}, reason="save", lineage_id="b"),
        ]
        assert check(events) == []
        events.append(trained(8, rollout_id=2, indices=[1], lineage_id="b"))

        [issue] = check(events)
        assert isinstance(issue, SampleConsumedTwiceIssue)
        assert issue.sample_indices == [1]

    def test_a_follower_retains_training_up_to_its_own_restore_point(self) -> None:
        """A follower ahead of the leader retains its own checkpointed training."""
        events = [
            trained(2, rollout_id=2, indices=[1], lineage_id="a", trainer_model_id="verifier"),
            restore(5, rollout_id=0, lineage_id="b", parent_lineage_id="a", rollout_ids={"solver": 0, "verifier": 2}),
            trained(6, rollout_id=3, indices=[1], lineage_id="b", trainer_model_id="verifier"),
            snapshot(7, rollout_id=3, holdings={}, reason="save", lineage_id="b", trainer_model_id="verifier"),
        ]

        [issue] = check(events)
        assert isinstance(issue, SampleConsumedTwiceIssue)
        assert issue.sample_indices == [1]

    def test_two_restores_exclude_training_on_the_abandoned_branch(self) -> None:
        """Rolling back and restoring again must exclude the abandoned suffix."""
        events = [
            trained(2, rollout_id=0, indices=[1], lineage_id="a"),
            trained(3, rollout_id=1, indices=[2], lineage_id="a"),
            restore(5, rollout_id=0, lineage_id="b", parent_lineage_id="a"),
            trained(6, rollout_id=1, indices=[2], lineage_id="b"),
            restore(8, rollout_id=1, lineage_id="c", parent_lineage_id="b"),
            trained(9, rollout_id=2, indices=[3], lineage_id="c"),
            snapshot(10, rollout_id=2, holdings={}, reason="save", lineage_id="c"),
        ]

        assert check(events) == []
        events.append(trained(11, rollout_id=3, indices=[2], lineage_id="c"))
        [issue] = check(events)
        assert issue.sample_indices == [2]

    def test_clock_skew_does_not_change_current_or_ancestor_training(self) -> None:
        """Trainer timestamps on either side of restore do not determine lineage."""
        events = [
            trained(100, rollout_id=0, indices=[1], lineage_id="a"),
            restore(50, rollout_id=0, lineage_id="b", parent_lineage_id="a"),
            transition(51, indices=[1, 2], to_owner=SampleOwner.RETRY_BUFFER, lineage_id="b"),
            trained(1, rollout_id=1, indices=[2], lineage_id="b"),
            snapshot(52, rollout_id=1, holdings={}, reason="save", lineage_id="b"),
        ]

        assert check(events) == []
        events.append(trained(2, rollout_id=2, indices=[1], lineage_id="b"))
        [issue] = check(events)
        assert issue.sample_indices == [1]

    def test_fresh_run_uses_the_latest_snapshot_lineage(self) -> None:
        """Events from an unrelated fresh run cannot count as current training."""
        events = [
            trained(1, rollout_id=0, indices=[1], lineage_id="old"),
            snapshot(2, rollout_id=0, holdings={}, lineage_id="old"),
            trained(3, rollout_id=0, indices=[1], lineage_id="new"),
            snapshot(4, rollout_id=0, holdings={}, lineage_id="new"),
        ]

        assert check(events) == []


class TestPolicyIsolation:
    @pytest.mark.parametrize("peer_fate", ["held", "dropped"])
    def test_a_peer_with_the_same_index_cannot_hide_policy_loss(self, peer_fate: str) -> None:
        """Another policy holding or dropping an index cannot account for a lost sample."""
        events = [
            generated(1, 1),
            snapshot(3, rollout_id=0, holdings={}, reason="save", trainer_model_id="solver"),
            snapshot(
                3,
                rollout_id=0,
                holdings={SampleOwner.OUTPUT_BUFFER: [1]} if peer_fate == "held" else {},
                reason="save",
                trainer_model_id="verifier",
            ),
        ]
        if peer_fate == "dropped":
            events.append(
                transition(2, indices=[1], to_owner=SampleOwner.DROPPED).model_copy(
                    update={"trainer_model_id": "verifier"}
                )
            )

        [issue] = [issue for partition in _partition_by_model_id(events) for issue in check(partition)]
        assert isinstance(issue, SampleLostIssue)
        assert issue.sample_indices == [1]

    def test_a_replay_policy_does_not_disable_fifo_duplicate_detection(self) -> None:
        """Replay declarations apply only to the policy whose buffer makes them."""
        events = [
            trained(1, rollout_id=0, indices=[1], trainer_model_id="solver"),
            trained(2, rollout_id=1, indices=[1], trainer_model_id="solver"),
            trained(1, rollout_id=0, indices=[1], trainer_model_id="verifier"),
            trained(2, rollout_id=1, indices=[1], trainer_model_id="verifier"),
            snapshot(3, rollout_id=1, holdings={}, trainer_model_id="solver"),
            snapshot(3, rollout_id=1, holdings={}, trainer_model_id="verifier", replays_samples=True),
        ]

        [issue] = [issue for partition in _partition_by_model_id(events) for issue in check(partition)]
        assert isinstance(issue, SampleConsumedTwiceIssue)
        assert issue.sample_indices == [1]


def restore(
    seconds: int,
    *,
    rollout_id: int,
    lineage_id: str,
    parent_lineage_id: str,
    rollout_ids: dict[str, int] | None = None,
) -> RolloutStateRestoreEvent:
    return RolloutStateRestoreEvent(
        timestamp=at(seconds),
        source=SOURCE,
        rollout_id=rollout_id,
        lineage_id=lineage_id,
        parent_lineage_id=parent_lineage_id,
        rollout_ids=rollout_ids,
    )
