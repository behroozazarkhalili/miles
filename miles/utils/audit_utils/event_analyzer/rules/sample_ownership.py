"""Always-on check that every rollout sample has exactly one owner.

Two invariants, from the events ``miles/utils/audit_utils/sample_ownership.py`` emits:

* no sample is lost: a sample that ever entered ``in_flight`` is, at a snapshot instant,
  either still held by some owner, or explicitly dropped, or trained. Always on.
* no sample is trained twice: within one lineage a sample is handed to training at most
  once and trained at most once. Only for buffers that do not set ``replays_samples``.

The run-time snapshots are taken across processes, so a sample can look unaccounted for
at one instant; only an object that fails at three snapshots in a row is reported. The
snapshot a checkpoint takes is exact (it is collected in one stretch of the rollout event
loop) and is held to zero tolerance.
"""

from collections.abc import Iterable, Sequence

from miles.utils.audit_utils.event_analyzer.utils import filter_by_type
from miles.utils.audit_utils.event_logger.models import (
    Event,
    RolloutHoldingsSnapshotEvent,
    RolloutStateRestoreEvent,
    SampleOwner,
    SampleOwnerTransitionEvent,
    TrainerTrainedSamplesEvent,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel

RUNTIME_TOLERANCE_SNAPSHOTS = 3


class SampleLostIssue(FrozenStrictBaseModel):
    rollout_id: int
    description: str
    sample_indices: list[int]


class SampleConsumedTwiceIssue(FrozenStrictBaseModel):
    description: str
    sample_indices: list[int]


SampleOwnershipIssue = SampleLostIssue | SampleConsumedTwiceIssue


def check(events: list[Event]) -> list[SampleOwnershipIssue]:
    lineage_id = current_lineage_id(events)
    restores = {event.lineage_id: event for event in filter_by_type(events, RolloutStateRestoreEvent)}
    transitions = [e for e in filter_by_type(events, SampleOwnerTransitionEvent) if e.lineage_id == lineage_id]
    snapshots = [e for e in filter_by_type(events, RolloutHoldingsSnapshotEvent) if e.lineage_id == lineage_id]
    trained = [
        event
        for event in filter_by_type(events, TrainerTrainedSamplesEvent)
        if event.lineage_id == lineage_id or _ancestor_trained(event=event, lineage_id=lineage_id, restores=restores)
    ]
    if not snapshots:
        return []

    return [
        *_check_nothing_lost(transitions=transitions, snapshots=snapshots, trained=trained),
        *_check_nothing_consumed_twice(transitions=transitions, snapshots=snapshots, trained=trained),
    ]


def current_lineage_id(events: Sequence[Event]) -> str | None:
    restores = [event for event in filter_by_type(events, RolloutStateRestoreEvent) if event.lineage_id is not None]
    snapshots = filter_by_type(events, RolloutHoldingsSnapshotEvent)
    if not restores:
        return max(snapshots, key=lambda event: event.timestamp).lineage_id if snapshots else None
    parents = {event.parent_lineage_id for event in restores}
    leaves = [event for event in restores if event.lineage_id not in parents]
    by_id = {event.lineage_id: event for event in restores}

    def _depth(event: RolloutStateRestoreEvent) -> int:
        seen: set[str | None] = set()
        while event.lineage_id not in seen:
            seen.add(event.lineage_id)
            if event.parent_lineage_id not in by_id:
                break
            event = by_id[event.parent_lineage_id]
        return len(seen)

    return max(leaves, key=lambda event: (_depth(event), event.timestamp)).lineage_id


def _ancestor_trained(
    *, event: TrainerTrainedSamplesEvent, lineage_id: str | None, restores: dict[str | None, RolloutStateRestoreEvent]
) -> bool:
    seen: set[str | None] = set()
    cutoff: int | None = None
    while lineage_id in restores and lineage_id not in seen:
        seen.add(lineage_id)
        restore = restores[lineage_id]
        point = (restore.rollout_ids or {}).get(event.trainer_model_id, restore.rollout_id)
        if point is None:
            return False
        cutoff = point if cutoff is None else min(cutoff, point)
        lineage_id = restore.parent_lineage_id
        if event.lineage_id == lineage_id:
            return event.rollout_id <= cutoff
    return False


# ================================ nothing lost ================================


def _check_nothing_lost(
    *,
    transitions: list[SampleOwnerTransitionEvent],
    snapshots: list[RolloutHoldingsSnapshotEvent],
    trained: list[TrainerTrainedSamplesEvent],
) -> list[SampleOwnershipIssue]:
    ordered_transitions = sorted(transitions, key=lambda event: event.timestamp)
    ordered_snapshots = sorted(snapshots, key=lambda event: event.timestamp)
    trained_at: dict[int, int] = {}
    for event in trained:
        for index in event.sample_indices:
            trained_at[index] = min(trained_at.get(index, event.rollout_id), event.rollout_id)

    entered: set[int] = set()
    dropped: set[int] = set()
    transition_pos = 0

    issues: list[SampleOwnershipIssue] = []
    recent: list[set[int]] = []
    for snapshot in ordered_snapshots:
        at = snapshot.timestamp
        while transition_pos < len(ordered_transitions) and ordered_transitions[transition_pos].timestamp <= at:
            event = ordered_transitions[transition_pos]
            if event.to_owner in {
                SampleOwner.IN_FLIGHT,
                SampleOwner.RETRY_BUFFER,
                SampleOwner.OUTPUT_BUFFER,
                SampleOwner.HANDED_TO_TRAINER,
            }:
                entered |= set(event.sample_indices)
            elif event.to_owner == SampleOwner.DROPPED:
                dropped |= set(event.sample_indices)
            transition_pos += 1
        held = {index for indices in snapshot.holdings.values() for index in indices}
        orphans = {
            index
            for index in entered - dropped - held
            if index not in trained_at or trained_at[index] > snapshot.rollout_id
        }

        if snapshot.reason in {"save", "final"} and orphans:
            issues.append(
                SampleLostIssue(
                    rollout_id=snapshot.rollout_id,
                    description=(
                        f"the {snapshot.reason} snapshot of rollout {snapshot.rollout_id} accounts for no owner of "
                        f"{len(orphans)} samples that were generated; resuming from it would train on fewer "
                        f"prompts than the run consumed"
                    ),
                    sample_indices=sorted(orphans),
                )
            )

        recent = [*recent, orphans][-RUNTIME_TOLERANCE_SNAPSHOTS:]
        if len(recent) == RUNTIME_TOLERANCE_SNAPSHOTS and (persistent := set.intersection(*recent)):
            issues.append(
                SampleLostIssue(
                    rollout_id=snapshot.rollout_id,
                    description=(
                        f"{len(persistent)} samples had no owner at {RUNTIME_TOLERANCE_SNAPSHOTS} consecutive "
                        f"snapshots up to rollout {snapshot.rollout_id}, so this is not the lag between processes"
                    ),
                    sample_indices=sorted(persistent),
                )
            )
    return issues


# =========================== nothing consumed twice ===========================


def _check_nothing_consumed_twice(
    *,
    transitions: list[SampleOwnerTransitionEvent],
    snapshots: list[RolloutHoldingsSnapshotEvent],
    trained: list[TrainerTrainedSamplesEvent],
) -> list[SampleOwnershipIssue]:
    if any(snapshot.replays_samples for snapshot in snapshots):
        return []

    return [
        *_repeated_issue((event.sample_indices for event in trained), what="trained"),
        *_repeated_issue(
            (
                event.sample_indices
                for event in transitions
                if event.to_owner == SampleOwner.HANDED_TO_TRAINER and event.reason != "restored"
            ),
            what="handed",
        ),
    ]


def _repeated_issue(indices_lists: Iterable[Iterable[int]], *, what: str) -> list[SampleOwnershipIssue]:
    if not (repeated := _repeated(indices_lists)):
        return []
    description = (
        f"{len(repeated)} samples reached the weights of one lineage more than once; the same "
        f"gradient was applied twice and the effective batch is not what the run reports"
        if what == "trained"
        else f"{len(repeated)} samples were handed to training twice within one lineage; a replay that "
        f"the buffer does not declare would train them twice"
    )
    return [SampleConsumedTwiceIssue(description=description, sample_indices=sorted(repeated))]


# ==================================== misc ====================================


def _repeated(event_indices: Iterable[Iterable[int]]) -> set[int]:
    seen: set[int] = set()
    ans: set[int] = set()
    for indices in event_indices:
        unique = set(indices)
        ans |= seen & unique
        seen |= unique
    return ans
