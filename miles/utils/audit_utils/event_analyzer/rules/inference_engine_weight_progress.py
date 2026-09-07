from miles.utils.audit_utils.checksum_utils import InferenceEngineChecksums
from miles.utils.audit_utils.event_logger.models import Event
from miles.utils.audit_utils.inference_engine_checksum_grouping import (
    PublicationKey,
    canonical_checksums_by_publication,
    format_publication,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel

__all__ = ["WeightProgressIssue", "check"]


class WeightProgressIssue(FrozenStrictBaseModel):
    label_previous: str
    label_current: str
    num_tensors: int


def check(events: list[Event]) -> list[WeightProgressIssue]:
    canonical = canonical_checksums_by_publication(events)
    return [
        issue
        for key, checksums in canonical.items()
        if (issue := _check_one_publication(key=key, checksums=checksums, canonical=canonical)) is not None
    ]


def _check_one_publication(
    *,
    key: PublicationKey,
    checksums: InferenceEngineChecksums,
    canonical: dict[PublicationKey, InferenceEngineChecksums],
) -> WeightProgressIssue | None:
    previous_key = PublicationKey(trainer_model_id=key.trainer_model_id, weight_version=key.weight_version - 1)
    previous = canonical.get(previous_key)
    if previous is None or previous != checksums:
        return None

    return WeightProgressIssue(
        label_previous=format_publication(previous_key),
        label_current=format_publication(key),
        num_tensors=len(checksums),
    )
