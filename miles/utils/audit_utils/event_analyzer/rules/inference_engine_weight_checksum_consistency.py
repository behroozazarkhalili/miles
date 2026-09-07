from collections.abc import Iterable

from miles.utils.audit_utils.event_analyzer.rules.checksum_compare import ChecksumMismatchIssue, compare_flat_dicts
from miles.utils.audit_utils.event_logger.models import Event
from miles.utils.audit_utils.inference_engine_checksum_grouping import (
    CellObservation,
    PublicationKey,
    format_publication,
    group_observations_by_publication,
)

__all__ = ["check"]


def check(events: list[Event]) -> list[ChecksumMismatchIssue]:
    """Check: every cell that took one published weight version must hold exactly the same weights."""
    issues: list[ChecksumMismatchIssue] = []
    for key, observations in group_observations_by_publication(events).items():
        issues += list(_check_one_publication(key=key, observations=observations))
    return issues


def _check_one_publication(
    *, key: PublicationKey, observations: list[CellObservation]
) -> Iterable[ChecksumMismatchIssue]:
    if len(observations) < 2:
        return
    baseline = observations[0]
    for other in observations[1:]:
        yield from compare_flat_dicts(
            a=baseline.checksums,
            b=other.checksums,
            label_a=_label(key=key, cell_id=baseline.cell_id),
            label_b=_label(key=key, cell_id=other.cell_id),
        )


def _label(*, key: PublicationKey, cell_id: str) -> str:
    return f"{format_publication(key)}/cell_{cell_id}"
