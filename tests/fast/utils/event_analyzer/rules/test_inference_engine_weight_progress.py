"""Tests for event_analyzer rules/inference_engine_weight_progress."""

from datetime import datetime, timezone

from miles.utils.audit_utils.event_analyzer.rules.inference_engine_weight_progress import check
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_event(
    *,
    weight_version: int,
    engine_checksums: dict[str, dict[str, str]],
    rollout_id: int | None = None,
    trainer_model_id: str | None = None,
) -> InferenceEngineWeightChecksumEvent:
    return InferenceEngineWeightChecksumEvent(
        timestamp=_FIXED_TS,
        source=SimpleProcessIdentity(component="main"),
        rollout_id=weight_version if rollout_id is None else rollout_id,
        weight_version=weight_version,
        trainer_model_id=trainer_model_id,
        engine_checksums=engine_checksums,
    )


class TestCheck:
    def test_empty_events_no_issues(self) -> None:
        """No engine checksum events means nothing to check."""
        assert check([]) == []

    def test_a_single_publication_has_no_predecessor(self) -> None:
        """The first published version has nothing before it to have moved away from."""
        events = [_make_event(weight_version=1, engine_checksums={"cell-a": {"w": "aaa"}})]
        assert check(events) == []

    def test_an_unchanged_adjacent_publication_is_reported(self) -> None:
        """A published version identical to the one before it means the update transported no new weights at all."""
        events = [
            _make_event(weight_version=1, engine_checksums={"cell-a": {"w": "aaa", "b": "bbb"}}),
            _make_event(weight_version=2, engine_checksums={"cell-a": {"w": "aaa", "b": "bbb"}}),
        ]

        issues = check(events)

        assert len(issues) == 1
        assert issues[0].label_previous == "default/weight_v1"
        assert issues[0].label_current == "default/weight_v2"
        assert issues[0].num_tensors == 2

    def test_one_changed_tensor_is_enough(self) -> None:
        """Some tensors legitimately do not move in a step, so demanding that every tensor change fails healthy runs."""
        events = [
            _make_event(weight_version=1, engine_checksums={"cell-a": {"w": "aaa", "b": "bbb", "c": "ccc"}}),
            _make_event(weight_version=2, engine_checksums={"cell-a": {"w": "aaa", "b": "bbb", "c": "zzz"}}),
        ]

        assert check(events) == []

    def test_the_check_ignores_the_order_the_events_were_written_in(self) -> None:
        """Event file order is not semantic, so a late-written earlier version is still the predecessor."""
        backwards = [
            _make_event(weight_version=2, engine_checksums={"cell-a": {"w": "aaa"}}),
            _make_event(weight_version=1, engine_checksums={"cell-a": {"w": "aaa"}}),
        ]

        assert len(check(backwards)) == 1

    def test_non_adjacent_versions_are_not_compared(self) -> None:
        """With a version's observations missing, two surviving versions are not one step apart and cannot be judged."""
        events = [
            _make_event(weight_version=1, engine_checksums={"cell-a": {"w": "aaa"}}),
            _make_event(weight_version=3, engine_checksums={"cell-a": {"w": "aaa"}}),
        ]

        assert check(events) == []

    def test_a_replayed_version_is_not_a_new_publication(self) -> None:
        """Observing one version twice is a repeat, not a step, so it must not read as a stalled update."""
        events = [
            _make_event(rollout_id=0, weight_version=1, engine_checksums={"cell-a": {"w": "aaa"}}),
            _make_event(rollout_id=1, weight_version=1, engine_checksums={"cell-a": {"w": "aaa"}}),
        ]

        assert check(events) == []

    def test_a_replaced_cell_does_not_reset_the_progression(self) -> None:
        """After healing, the same weights arrive from another cell, and that must not excuse a stalled update."""
        events = [
            _make_event(weight_version=1, engine_checksums={"cell-old": {"w": "aaa"}}),
            _make_event(weight_version=2, engine_checksums={"cell-new": {"w": "aaa"}}),
        ]

        assert len(check(events)) == 1

    def test_the_cells_that_reported_do_not_change_the_verdict(self) -> None:
        """The representative map is canonical, so a differently sized cell set still compares like for like."""
        events = [
            _make_event(weight_version=1, engine_checksums={"cell-a": {"w": "aaa"}}),
            _make_event(weight_version=2, engine_checksums={"cell-a": {"w": "aaa"}, "cell-b": {"w": "aaa"}}),
        ]

        assert len(check(events)) == 1

    def test_policies_are_not_compared_with_each_other(self) -> None:
        """Two policies hold unrelated weights, so one policy's version is never the other's predecessor."""
        events = [
            _make_event(weight_version=1, trainer_model_id="a", engine_checksums={"cell-a": {"w": "aaa"}}),
            _make_event(weight_version=2, trainer_model_id="b", engine_checksums={"cell-b": {"w": "aaa"}}),
        ]

        assert check(events) == []

    def test_the_reported_policy_is_named(self) -> None:
        """A multi policy run must say whose weights stopped moving."""
        events = [
            _make_event(weight_version=1, trainer_model_id="solver", engine_checksums={"cell-a": {"w": "aaa"}}),
            _make_event(weight_version=2, trainer_model_id="solver", engine_checksums={"cell-a": {"w": "aaa"}}),
        ]

        assert check(events)[0].label_current == "solver/weight_v2"

    def test_a_tensor_that_appeared_between_two_versions_is_not_a_stall(self) -> None:
        """Different tensor name sets are a different failure, caught by the consistency rule, not by this one."""
        events = [
            _make_event(weight_version=1, engine_checksums={"cell-a": {"w": "aaa"}}),
            _make_event(weight_version=2, engine_checksums={"cell-a": {"w": "aaa", "b": "bbb"}}),
        ]

        assert check(events) == []

    def test_every_stalled_step_of_a_run_is_reported(self) -> None:
        """A run that stalls for several steps must not report only its first stalled step."""
        events = [
            _make_event(weight_version=1, engine_checksums={"cell-a": {"w": "aaa"}}),
            _make_event(weight_version=2, engine_checksums={"cell-a": {"w": "aaa"}}),
            _make_event(weight_version=3, engine_checksums={"cell-a": {"w": "aaa"}}),
        ]

        assert [issue.label_current for issue in check(events)] == ["default/weight_v2", "default/weight_v3"]
