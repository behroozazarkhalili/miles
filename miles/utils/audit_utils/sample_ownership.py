"""Emit the sample ownership trail the always-on checker reads.

Every sample the rollout produces has exactly one owner at any instant (see
``SampleOwner``). The rollout side reports each move and, once per step and once per
checkpoint, the full set of what it holds; the trainer side reports which groups
actually reached the weights. ``event_analyzer/rules/sample_ownership.py`` turns the
two into the "no sample is lost, no sample is trained twice" invariants.
"""

import logging
from collections.abc import Iterable
from typing import Literal

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import (
    RolloutHoldingsSnapshotEvent,
    RolloutStateRestoreEvent,
    SampleOwner,
    SampleOwnerTransitionEvent,
    TrainerTrainedSamplesEvent,
)
from miles.utils.types import Sample

logger = logging.getLogger(__name__)
_lineage_id: str | None = None


def set_lineage_id(lineage_id: str) -> None:
    global _lineage_id
    _lineage_id = lineage_id


def log_owner_transition(
    prompt_group: Iterable[Sample],
    *,
    from_owner: SampleOwner,
    to_owner: SampleOwner,
    trainer_model_id: str | None = None,
    rollout_id: int | None = None,
    reason: str | None = None,
) -> None:
    if not is_event_logger_initialized():
        return

    samples = list(prompt_group)
    if not samples:
        return

    get_event_logger().log(
        SampleOwnerTransitionEvent,
        dict(
            lineage_id=_lineage_id,
            sample_indices=[sample.index for sample in samples if sample.index is not None],
            trainer_model_id=trainer_model_id,
            from_owner=from_owner,
            to_owner=to_owner,
            rollout_id=rollout_id,
            reason=reason,
        ),
        print_log=False,
    )


def log_holdings_snapshot(
    *,
    rollout_id: int,
    trainer_model_id: str | None,
    holdings: dict[SampleOwner, list[int]],
    replays_samples: bool,
    reason: Literal["step", "save", "final"],
) -> None:
    if not is_event_logger_initialized():
        return

    get_event_logger().log(
        RolloutHoldingsSnapshotEvent,
        dict(
            lineage_id=_lineage_id,
            rollout_id=rollout_id,
            trainer_model_id=trainer_model_id,
            holdings=holdings,
            replays_samples=replays_samples,
            reason=reason,
        ),
        print_log=False,
    )


def log_state_restore(
    rollout_id: int | None, *, rollout_ids: dict[str, int] | None, parent_lineage_id: str | None
) -> None:
    if not is_event_logger_initialized():
        return

    get_event_logger().log(
        RolloutStateRestoreEvent,
        dict(
            rollout_id=rollout_id,
            rollout_ids=rollout_ids,
            lineage_id=_lineage_id,
            parent_lineage_id=parent_lineage_id,
        ),
    )


def log_trained_samples(
    *, rollout_id: int, trainer_model_id: str | None, sample_indices: list[int], lineage_id: str | None
) -> None:
    if not is_event_logger_initialized():
        return

    get_event_logger().log(
        TrainerTrainedSamplesEvent,
        dict(
            lineage_id=lineage_id,
            rollout_id=rollout_id,
            trainer_model_id=trainer_model_id,
            sample_indices=sample_indices,
        ),
        print_log=False,
    )
