"""Tests for event_analyzer rules/inference_engine_weight_checksum_consistency."""

from datetime import datetime, timezone

import pytest

from miles.utils.audit_utils.event_analyzer.rules.inference_engine_weight_checksum_consistency import check
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_event(
    *,
    rollout_id: int = 0,
    weight_version: int,
    engine_checksums: dict[str, dict[str, str]],
    trainer_model_id: str | None = None,
    adjacent_weight_change_expected: bool = True,
) -> InferenceEngineWeightChecksumEvent:
    return InferenceEngineWeightChecksumEvent(
        timestamp=_FIXED_TS,
        source=SimpleProcessIdentity(component="main"),
        rollout_id=rollout_id,
        weight_version=weight_version,
        trainer_model_id=trainer_model_id,
        adjacent_weight_change_expected=adjacent_weight_change_expected,
        engine_checksums=engine_checksums,
    )


class TestCheck:
    def test_empty_events_no_mismatches(self) -> None:
        """No engine checksum events means nothing to check."""
        assert check([]) == []

    def test_single_cell_no_comparison(self) -> None:
        """A single cell has no peer to compare against."""
        events = [_make_event(weight_version=1, engine_checksums={"cell-a": {"rank0/w": "aaa"}})]
        assert check(events) == []

    def test_matching_cells_no_mismatches(self) -> None:
        """All cells holding identical checksums produce no issues."""
        events = [
            _make_event(
                weight_version=1,
                engine_checksums={
                    "cell-a": {"rank0/w": "aaa"},
                    "cell-b": {"rank0/w": "aaa"},
                    "cell-c": {"rank0/w": "aaa"},
                },
            )
        ]
        assert check(events) == []

    def test_tensor_mismatch_names_the_real_cell_ids(self) -> None:
        """A differing tensor is reported against the cell ids that reported it, not against list positions."""
        events = [
            _make_event(
                weight_version=5, engine_checksums={"cell-a": {"rank0/w": "aaa"}, "cell-b": {"rank0/w": "zzz"}}
            )
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].key == "rank0/w"
        assert mismatches[0].label_a == "default/weight_v5/cell_cell-a"
        assert mismatches[0].label_b == "default/weight_v5/cell_cell-b"

    def test_cell_order_within_the_event_does_not_change_the_verdict(self) -> None:
        """Cells are compared by identity, so a shuffled per-cell mapping yields the very same issue."""
        shuffled = [
            _make_event(
                weight_version=5, engine_checksums={"cell-b": {"rank0/w": "zzz"}, "cell-a": {"rank0/w": "aaa"}}
            )
        ]
        in_order = [
            _make_event(
                weight_version=5, engine_checksums={"cell-a": {"rank0/w": "aaa"}, "cell-b": {"rank0/w": "zzz"}}
            )
        ]
        assert check(shuffled) == check(in_order)

    def test_missing_tensor_on_one_cell_detected(self) -> None:
        """A tensor present on one cell but absent on another is a mismatch."""
        events = [
            _make_event(
                weight_version=1,
                engine_checksums={"cell-a": {"rank0/w": "aaa", "rank0/b": "bbb"}, "cell-b": {"rank0/w": "aaa"}},
            )
        ]
        mismatches = check(events)
        assert any(m.key == "rank0/b" and "<missing>" in m.value_b for m in mismatches)

    def test_cells_of_one_version_reported_by_two_events_are_compared(self) -> None:
        """Two cells that took the same version in different updates must still agree with each other."""
        events = [
            _make_event(rollout_id=0, weight_version=4, engine_checksums={"cell-a": {"rank0/w": "aaa"}}),
            _make_event(rollout_id=1, weight_version=4, engine_checksums={"cell-b": {"rank0/w": "zzz"}}),
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].label_a == "default/weight_v4/cell_cell-a"
        assert mismatches[0].label_b == "default/weight_v4/cell_cell-b"

    def test_one_cell_answering_twice_for_one_version_must_answer_the_same(self) -> None:
        """A cell whose weights changed under a version it already reported has silently drifted."""
        events = [
            _make_event(rollout_id=0, weight_version=4, engine_checksums={"cell-a": {"rank0/w": "aaa"}}),
            _make_event(rollout_id=1, weight_version=4, engine_checksums={"cell-a": {"rank0/w": "zzz"}}),
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].key == "rank0/w"

    def test_a_cell_repeating_the_very_same_answer_is_not_an_issue(self) -> None:
        """Re-observing one cell at one version is expected and must not read as a disagreement."""
        events = [
            _make_event(rollout_id=0, weight_version=4, engine_checksums={"cell-a": {"rank0/w": "aaa"}}),
            _make_event(rollout_id=1, weight_version=4, engine_checksums={"cell-a": {"rank0/w": "aaa"}}),
        ]
        assert check(events) == []

    def test_event_arrival_order_does_not_decide_which_cell_is_the_baseline(self) -> None:
        """File order is not semantic, so the same disagreement reads identically however the events were written."""
        forwards = [
            _make_event(rollout_id=0, weight_version=4, engine_checksums={"cell-a": {"rank0/w": "aaa"}}),
            _make_event(rollout_id=1, weight_version=4, engine_checksums={"cell-b": {"rank0/w": "zzz"}}),
        ]
        backwards = list(reversed(forwards))
        assert check(backwards) == check(forwards)
        assert check(backwards)[0].label_a == "default/weight_v4/cell_cell-a"

    def test_different_versions_are_not_compared(self) -> None:
        """Weights must change between versions, so comparing two versions would report every healthy run."""
        events = [
            _make_event(rollout_id=0, weight_version=1, engine_checksums={"cell-a": {"rank0/w": "aaa"}}),
            _make_event(rollout_id=1, weight_version=2, engine_checksums={"cell-a": {"rank0/w": "zzz"}}),
        ]
        assert check(events) == []

    def test_different_policies_are_not_compared(self) -> None:
        """Two policies hold different weights by construction, so they are never each other's baseline."""
        events = [
            _make_event(weight_version=1, trainer_model_id="a", engine_checksums={"cell-a": {"rank0/w": "aaa"}}),
            _make_event(weight_version=1, trainer_model_id="b", engine_checksums={"cell-b": {"rank0/w": "zzz"}}),
        ]
        assert check(events) == []

    def test_the_policy_name_is_part_of_the_label(self) -> None:
        """A multi policy run must say which policy's weights disagreed."""
        events = [
            _make_event(
                weight_version=2,
                trainer_model_id="solver",
                engine_checksums={"cell-a": {"rank0/w": "aaa"}, "cell-b": {"rank0/w": "zzz"}},
            )
        ]
        mismatches = check(events)
        assert mismatches[0].label_a == "solver/weight_v2/cell_cell-a"

    def test_tp_rank_prefixes_are_compared_rank_by_rank(self) -> None:
        """Two cells that hold the same tensors on swapped ranks hold different shards, not equal weights."""
        events = [
            _make_event(
                weight_version=1,
                engine_checksums={
                    "cell-a": {"rank0/w": "aaa", "rank1/w": "bbb"},
                    "cell-b": {"rank0/w": "bbb", "rank1/w": "aaa"},
                },
            )
        ]
        assert len(check(events)) == 2

    def test_only_the_inconsistent_version_is_reported(self) -> None:
        """A healthy publication next to a broken one must not be dragged into the report."""
        events = [
            _make_event(
                rollout_id=0,
                weight_version=1,
                engine_checksums={"cell-a": {"rank0/w": "aaa"}, "cell-b": {"rank0/w": "aaa"}},
            ),
            _make_event(
                rollout_id=1,
                weight_version=2,
                engine_checksums={"cell-a": {"rank0/w": "bbb"}, "cell-b": {"rank0/w": "zzz"}},
            ),
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert "weight_v2/" in mismatches[0].label_a

    def test_a_publication_with_no_cell_at_all_is_rejected(self) -> None:
        """An empty publication has no cell to compare, so it would silently pass the consistency check."""
        events = [
            _make_event(rollout_id=0, weight_version=1, engine_checksums={"cell-a": {"rank0/w": "aaa"}}),
            _make_event(rollout_id=1, weight_version=2, engine_checksums={}),
        ]
        with pytest.raises(AssertionError, match="names no inference cell"):
            check(events)

    def test_a_cell_with_no_tensor_at_all_is_rejected(self) -> None:
        """Two cells that both reported nothing would agree perfectly and prove nothing."""
        events = [
            _make_event(weight_version=1, engine_checksums={"cell-a": {}, "cell-b": {}}),
        ]
        with pytest.raises(AssertionError, match="reported no tensor checksum"):
            check(events)

    def test_a_mode_that_disables_the_adjacency_check_still_compares_cells(self) -> None:
        """Only weight movement is unverifiable in those modes; two cells of one version must still agree."""
        events = [
            _make_event(
                weight_version=1,
                adjacent_weight_change_expected=False,
                engine_checksums={"cell-a": {"rank0/w": "aaa"}, "cell-b": {"rank0/w": "zzz"}},
            )
        ]
        assert len(check(events)) == 1
