from collections.abc import Iterable
from typing import NamedTuple

from miles.utils.audit_utils.checksum_utils import InferenceEngineChecksums
from miles.utils.audit_utils.event_logger.models import Event, InferenceEngineWeightChecksumEvent


class PublicationKey(NamedTuple):
    trainer_model_id: str | None
    weight_version: int


class CellObservation(NamedTuple):
    cell_id: str
    checksums: InferenceEngineChecksums


def group_observations_by_publication(events: Iterable[Event]) -> dict[PublicationKey, list[CellObservation]]:
    grouped: dict[PublicationKey, dict[tuple[str, tuple[tuple[str, str], ...]], CellObservation]] = {}

    for event in events:
        if not isinstance(event, InferenceEngineWeightChecksumEvent):
            continue
        key = PublicationKey(trainer_model_id=event.trainer_model_id, weight_version=event.weight_version)
        assert event.engine_checksums, (
            f"the checksum event of {format_publication(key)} names no inference cell, so this publication would "
            f"leave the audit with no evidence while still looking covered"
        )
        bucket = grouped.setdefault(key, {})
        for cell_id, checksums in event.engine_checksums.items():
            assert checksums, (
                f"cell {cell_id!r} reported no tensor checksum for {format_publication(key)}, so it would compare "
                f"equal to every other cell and to the version before it"
            )
            assert all(name and digest for name, digest in checksums.items()), (
                f"cell {cell_id!r} reported an unnamed tensor or an empty digest for {format_publication(key)}, so "
                f"that entry compares equal wherever it is copied: {checksums}"
            )
            fingerprint = (cell_id, tuple(sorted(checksums.items())))
            bucket[fingerprint] = CellObservation(cell_id=cell_id, checksums=checksums)

    return {
        key: [bucket[fingerprint] for fingerprint in sorted(bucket)]
        for key, bucket in sorted(grouped.items(), key=lambda item: publication_sort_key(item[0]))
    }


def canonical_checksums_by_publication(events: Iterable[Event]) -> dict[PublicationKey, InferenceEngineChecksums]:
    return {key: observations[0].checksums for key, observations in group_observations_by_publication(events).items()}


def publication_sort_key(key: PublicationKey) -> tuple[str, int]:
    return ("" if key.trainer_model_id is None else key.trainer_model_id, key.weight_version)


def format_publication(key: PublicationKey) -> str:
    model_id = "default" if key.trainer_model_id is None else key.trainer_model_id
    return f"{model_id}/weight_v{key.weight_version}"
