from pathlib import Path

from miles.utils.audit_utils.checksum_utils import InferenceEngineChecksums
from miles.utils.audit_utils.event_analyzer.rules import inference_engine_weight_checksum_consistency
from miles.utils.audit_utils.event_analyzer.rules.checksum_compare import ChecksumMismatchIssue, compare_flat_dicts
from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.inference_engine_checksum_grouping import (
    PublicationKey,
    canonical_checksums_by_publication,
    format_publication,
    publication_sort_key,
)


def compare_inference_engine_checksums(baseline_dir: str, target_dir: str) -> None:
    baseline = _read_inference_engine_checksum_events(Path(baseline_dir))
    target = _read_inference_engine_checksum_events(Path(target_dir))
    assert baseline, f"No InferenceEngineWeightChecksumEvents found in baseline dir: {baseline_dir}"
    assert target, f"No InferenceEngineWeightChecksumEvents found in target dir: {target_dir}"

    # Each side's engines must already agree internally (same invariant as the production rule), so
    # one representative engine per rollout then proves baseline == target regardless of engine count.
    assert not inference_engine_weight_checksum_consistency.check(
        baseline
    ), "Baseline engines disagree with each other"
    assert not inference_engine_weight_checksum_consistency.check(target), "Target engines disagree with each other"

    baseline_by_publication = canonical_checksums_by_publication(baseline)
    target_by_publication = canonical_checksums_by_publication(target)
    assert baseline_by_publication.keys() == target_by_publication.keys(), (
        f"Engine checksum (model_id, weight_version) sets differ: "
        f"baseline={_describe_publications(baseline_by_publication)} "
        f"vs target={_describe_publications(target_by_publication)}"
    )

    mismatches: list[ChecksumMismatchIssue] = []
    for key in sorted(baseline_by_publication, key=publication_sort_key):
        mismatches += list(
            compare_flat_dicts(
                a=baseline_by_publication[key],
                b=target_by_publication[key],
                label_a=f"baseline/{format_publication(key)}",
                label_b=f"target/{format_publication(key)}",
            )
        )
    assert not mismatches, "Engine weight checksum baseline-vs-target mismatch:\n" + "\n".join(
        f"  - {m.label_a} vs {m.label_b} key {m.key}: {m.value_a} != {m.value_b}" for m in mismatches
    )
    print(f"Engine weight checksum comparison passed: {len(baseline_by_publication)} publication(s) compared")


def assert_engine_count(*, side: str, dump_dir: str, expected: int) -> None:
    events = _read_inference_engine_checksum_events(Path(dump_dir))
    assert events, f"{side}: no InferenceEngineWeightChecksumEvents in {dump_dir}, so no engine ever took weights"

    counted = sorted({len(event.engine_checksums) for event in events})
    assert counted == [expected], (
        f"{side}: weights were pushed to {counted} engine(s), not {expected}; a run served by fewer engines still "
        f"trains, so nothing else would notice engines that never joined"
    )

    print(f"{side}: every weight update covered {expected} engine(s)")


def assert_engines_rejoined_weight_updates(*, side: str, dump_dir: str, expected: int) -> None:
    events = _read_inference_engine_checksum_events(Path(dump_dir))
    assert events, f"{side}: no InferenceEngineWeightChecksumEvents in {dump_dir}, so no engine ever took weights"

    oversized: dict[tuple[str | None, int], int] = {
        (event.trainer_model_id, event.rollout_id): len(event.engine_checksums)
        for event in events
        if len(event.engine_checksums) > expected
    }
    assert not oversized, (
        f"{side}: weight updates reached more than the {expected} engine(s) this run owns: {oversized}; a replaced "
        f"engine whose predecessor stayed in the fan-out would keep serving weights nobody audits"
    )

    last_of_model: dict[str | None, InferenceEngineWeightChecksumEvent] = {}
    for event in sorted(events, key=lambda one: one.rollout_id):
        last_of_model[event.trainer_model_id] = event

    missing: dict[tuple[str | None, int], int] = {
        (model_id, event.rollout_id): len(event.engine_checksums)
        for model_id, event in last_of_model.items()
        if len(event.engine_checksums) != expected
    }
    assert not missing, (
        f"{side}: the run's last weight update covered {missing} instead of all {expected} engine(s), so an engine "
        f"that crashed was never brought back into the fan-out; partial-target weight update is how a run survives "
        f"a crash mid-update, not how it is allowed to end"
    )

    print(f"{side}: the last weight update of every policy covered all {expected} engine(s)")


def assert_engine_weights_moved(*, side: str, dump_dir: str) -> None:
    by_publication = canonical_checksums_by_publication(_read_inference_engine_checksum_events(Path(dump_dir)))
    assert len(by_publication) > 1, (
        f"{side}: engine weight checksums cover {_describe_publications(by_publication)}, so there is no pair to "
        f"compare and nothing proves the run pushed an update at all"
    )

    distinct: set[tuple[tuple[str, str], ...]] = {tuple(sorted(one.items())) for one in by_publication.values()}
    assert len(distinct) > 1, (
        f"{side}: every one of {len(by_publication)} publications pushed byte-identical engine weights, so the "
        f"optimizer moved nothing and a bitwise comparison against another such run would prove nothing"
    )

    print(
        f"{side}: engine weights moved across {len(by_publication)} publication(s), "
        f"{len(distinct)} distinct checksum(s)"
    )


def _describe_publications(by_publication: dict[PublicationKey, InferenceEngineChecksums]) -> list[str]:
    return [format_publication(key) for key in sorted(by_publication, key=publication_sort_key)]


def _read_inference_engine_checksum_events(dump_dir: Path) -> list[InferenceEngineWeightChecksumEvent]:
    """Read all InferenceEngineWeightChecksumEvents from the events directory."""
    events_dir: Path = dump_dir / EVENTS_DIRNAME
    if not events_dir.exists():
        return []
    all_events = read_events(events_dir)
    return [e for e in all_events if isinstance(e, InferenceEngineWeightChecksumEvent)]
