from datetime import datetime, timezone

import pytest

from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent, MetricEvent
from miles.utils.audit_utils.inference_engine_checksum_grouping import (
    PublicationKey,
    adjacent_change_expected_by_publication,
    canonical_checksums_by_publication,
    format_publication,
    group_observations_by_publication,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
_SOURCE = SimpleProcessIdentity(component="main")


def _event(
    *,
    weight_version: int,
    engine_checksums: dict[str, dict[str, str]],
    rollout_id: int = 0,
    trainer_model_id: str | None = None,
    adjacent_weight_change_expected: bool = True,
) -> InferenceEngineWeightChecksumEvent:
    return InferenceEngineWeightChecksumEvent(
        timestamp=_FIXED_TS,
        source=_SOURCE,
        rollout_id=rollout_id,
        weight_version=weight_version,
        trainer_model_id=trainer_model_id,
        adjacent_weight_change_expected=adjacent_weight_change_expected,
        engine_checksums=engine_checksums,
    )


class TestGroupObservationsByPublication:
    def test_unrelated_events_are_ignored(self) -> None:
        """The analyzer hands every event to every rule, so a foreign event must not enter this grouping."""
        events = [MetricEvent(timestamp=_FIXED_TS, source=_SOURCE, metrics={"loss": 1.0})]
        assert group_observations_by_publication(events) == {}

    def test_cells_of_one_version_reported_by_two_events_land_in_one_bucket(self) -> None:
        """A version observed twice is one publication, so its cells must be compared with each other."""
        events = [
            _event(rollout_id=0, weight_version=3, engine_checksums={"cell-a": {"w": "aaa"}}),
            _event(rollout_id=1, weight_version=3, engine_checksums={"cell-b": {"w": "aaa"}}),
        ]

        grouped = group_observations_by_publication(events)

        assert list(grouped) == [PublicationKey(trainer_model_id=None, weight_version=3)]
        assert [observation.cell_id for observation in grouped[PublicationKey(None, 3)]] == ["cell-a", "cell-b"]

    def test_an_identical_repeat_observation_is_deduplicated(self) -> None:
        """Re-observing one cell at one version is expected and must not read as a second, disagreeing cell."""
        events = [
            _event(rollout_id=0, weight_version=3, engine_checksums={"cell-a": {"w": "aaa"}}),
            _event(rollout_id=1, weight_version=3, engine_checksums={"cell-a": {"w": "aaa"}}),
        ]

        assert len(group_observations_by_publication(events)[PublicationKey(None, 3)]) == 1

    def test_one_cell_reporting_two_different_maps_keeps_both(self) -> None:
        """Dropping the second answer would hide a cell whose weights drifted under a version it already reported."""
        events = [
            _event(rollout_id=0, weight_version=3, engine_checksums={"cell-a": {"w": "aaa"}}),
            _event(rollout_id=1, weight_version=3, engine_checksums={"cell-a": {"w": "zzz"}}),
        ]

        assert len(group_observations_by_publication(events)[PublicationKey(None, 3)]) == 2

    def test_policies_are_kept_apart(self) -> None:
        """Two policies publish their own version numbers, so one policy's version 3 is not the other's."""
        events = [
            _event(weight_version=3, trainer_model_id="a", engine_checksums={"cell-a": {"w": "aaa"}}),
            _event(weight_version=3, trainer_model_id="b", engine_checksums={"cell-b": {"w": "bbb"}}),
        ]

        assert sorted(group_observations_by_publication(events)) == [
            PublicationKey(trainer_model_id="a", weight_version=3),
            PublicationKey(trainer_model_id="b", weight_version=3),
        ]

    def test_the_unnamed_policy_sorts_next_to_named_ones(self) -> None:
        """A run that mixes an unnamed policy with a named one must still order deterministically, not crash."""
        events = [
            _event(weight_version=3, trainer_model_id="a", engine_checksums={"cell-a": {"w": "aaa"}}),
            _event(weight_version=3, trainer_model_id=None, engine_checksums={"cell-b": {"w": "bbb"}}),
        ]

        assert list(group_observations_by_publication(events)) == [
            PublicationKey(trainer_model_id=None, weight_version=3),
            PublicationKey(trainer_model_id="a", weight_version=3),
        ]

    def test_versions_are_ordered_numerically_whatever_order_they_were_written_in(self) -> None:
        """Event file order is not semantic, so a late-written earlier version still sorts first."""
        events = [
            _event(rollout_id=1, weight_version=11, engine_checksums={"cell-a": {"w": "bbb"}}),
            _event(rollout_id=0, weight_version=2, engine_checksums={"cell-a": {"w": "aaa"}}),
        ]

        assert [key.weight_version for key in group_observations_by_publication(events)] == [2, 11]


class TestGroupObservationsRejectsEmptyEvidence:
    def test_an_event_naming_no_cell_is_rejected(self) -> None:
        """Such a publication would drop out of every grouped audit while still looking like it was covered."""
        events = [_event(weight_version=3, engine_checksums={})]

        with pytest.raises(AssertionError, match="names no inference cell"):
            group_observations_by_publication(events)

    def test_a_cell_reporting_no_tensor_is_rejected(self) -> None:
        """An empty tensor map compares equal to every other empty one, so it would pass every comparison."""
        events = [_event(weight_version=3, engine_checksums={"cell-a": {}})]

        with pytest.raises(AssertionError, match="cell-a"):
            group_observations_by_publication(events)

    def test_one_empty_publication_among_valid_ones_is_rejected(self) -> None:
        """The valid publications around it would otherwise make the run look fully audited."""
        events = [
            _event(rollout_id=0, weight_version=1, engine_checksums={"cell-a": {"w": "aaa"}}),
            _event(rollout_id=1, weight_version=2, engine_checksums={}),
            _event(rollout_id=2, weight_version=3, engine_checksums={"cell-a": {"w": "bbb"}}),
        ]

        with pytest.raises(AssertionError, match="weight_v2"):
            group_observations_by_publication(events)

    def test_a_cell_reporting_an_empty_digest_is_rejected(self) -> None:
        """A hash of nothing compares equal wherever it is copied, so it would pass every comparison it entered."""
        events = [_event(weight_version=3, engine_checksums={"cell-a": {"w": ""}})]

        with pytest.raises(AssertionError, match="unnamed tensor or an empty digest"):
            group_observations_by_publication(events)

    def test_a_cell_reporting_an_unnamed_tensor_is_rejected(self) -> None:
        """A digest filed under no name lines up with nothing another cell of this version reported."""
        events = [_event(weight_version=3, engine_checksums={"cell-a": {"": "aaa"}})]

        with pytest.raises(AssertionError, match="unnamed tensor or an empty digest"):
            group_observations_by_publication(events)

    def test_an_empty_publication_does_not_silently_vanish_from_the_canonical_view(self) -> None:
        """Dropping it here is exactly how two sides with equally missing evidence used to compare equal."""
        events = [_event(weight_version=3, engine_checksums={})]

        with pytest.raises(AssertionError, match="names no inference cell"):
            canonical_checksums_by_publication(events)


class TestCanonicalChecksumsByPublication:
    def test_the_representative_does_not_depend_on_which_cell_answered_first(self) -> None:
        """A run whose cells were replaced serves the same weights, so the representative must not follow cell order."""
        answered_late_first = [
            _event(weight_version=3, engine_checksums={"cell-z": {"w": "aaa"}, "cell-a": {"w": "aaa"}})
        ]
        answered_early_first = [
            _event(weight_version=3, engine_checksums={"cell-a": {"w": "aaa"}, "cell-z": {"w": "aaa"}})
        ]

        assert canonical_checksums_by_publication(answered_late_first) == canonical_checksums_by_publication(
            answered_early_first
        )

    def test_one_representative_per_publication(self) -> None:
        """Consumers compare whole per-version maps, so each publication contributes exactly one map."""
        events = [
            _event(rollout_id=0, weight_version=1, engine_checksums={"cell-a": {"w": "aaa"}}),
            _event(rollout_id=1, weight_version=2, engine_checksums={"cell-a": {"w": "bbb"}}),
        ]

        assert canonical_checksums_by_publication(events) == {
            PublicationKey(None, 1): {"w": "aaa"},
            PublicationKey(None, 2): {"w": "bbb"},
        }


class TestFormatPublication:
    def test_a_named_policy_is_shown(self) -> None:
        """A multi policy report has to say which policy a publication belongs to."""
        assert format_publication(PublicationKey(trainer_model_id="solver", weight_version=4)) == "solver/weight_v4"

    def test_the_unnamed_policy_gets_a_stable_label(self) -> None:
        """Single policy runs still need a readable, stable label in every issue they raise."""
        assert format_publication(PublicationKey(trainer_model_id=None, weight_version=4)) == "default/weight_v4"


class TestAdjacentChangeExpectedByPublication:
    def test_the_producers_answer_is_carried_through(self) -> None:
        """The rule is disabled by the run that produced the events, not by anything the analyzer infers."""
        events = [_event(weight_version=3, adjacent_weight_change_expected=False, engine_checksums={"a": {"w": "x"}})]

        assert adjacent_change_expected_by_publication(events) == {PublicationKey(None, 3): False}

    def test_a_publication_observed_twice_under_one_policy_is_one_answer(self) -> None:
        """A version reported by several updates of the same run repeats its claim rather than restating it."""
        events = [
            _event(rollout_id=0, weight_version=3, engine_checksums={"a": {"w": "x"}}),
            _event(rollout_id=1, weight_version=3, engine_checksums={"a": {"w": "y"}}),
        ]

        assert adjacent_change_expected_by_publication(events) == {PublicationKey(None, 3): True}

    def test_two_observations_of_one_publication_that_disagree_are_rejected(self) -> None:
        """Letting either value win would judge one published version under a rule the other half never claimed."""
        events = [
            _event(rollout_id=0, weight_version=3, engine_checksums={"a": {"w": "x"}}),
            _event(
                rollout_id=1,
                weight_version=3,
                adjacent_weight_change_expected=False,
                engine_checksums={"a": {"w": "x"}},
            ),
        ]

        with pytest.raises(AssertionError, match="both with and without"):
            adjacent_change_expected_by_publication(events)

    def test_each_publication_keeps_its_own_answer(self) -> None:
        """A run that changes mode mid-flight disables only the versions it published in that mode."""
        events = [
            _event(weight_version=3, engine_checksums={"a": {"w": "x"}}),
            _event(weight_version=4, adjacent_weight_change_expected=False, engine_checksums={"a": {"w": "y"}}),
        ]

        assert adjacent_change_expected_by_publication(events) == {
            PublicationKey(None, 3): True,
            PublicationKey(None, 4): False,
        }
