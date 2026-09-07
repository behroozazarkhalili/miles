"""Tests for test_utils.comparisons.inference_engine_checksums.compare_inference_engine_checksums."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, EventLogger
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity, TrainerControllerProcessIdentity
from miles.utils.test_utils.comparisons.inference_engine_checksums import (
    assert_engine_count,
    assert_engine_weights_moved,
    compare_inference_engine_checksums,
)


def _write_inference_engine_events(
    side_dir: Path, partials: list[dict[str, Any]], *, model_id: str | None = None
) -> None:
    events_dir = side_dir / EVENTS_DIRNAME
    source = _source_of(model_id)
    event_logger = EventLogger(log_dir=events_dir, source=source, file_name=f"{source.to_name()}.jsonl")
    for partial in partials:
        event_logger.log(InferenceEngineWeightChecksumEvent, partial, print_log=False)
    event_logger.close()


def _write_events_without_trainer_model_id(side_dir: Path, partials: list[dict[str, Any]]) -> None:
    events_dir = side_dir / EVENTS_DIRNAME
    events_dir.mkdir(parents=True, exist_ok=True)
    source = _source_of(None)
    lines: list[str] = []
    for partial in partials:
        dumped = InferenceEngineWeightChecksumEvent(
            **partial, timestamp=datetime.now(timezone.utc), source=source
        ).model_dump(mode="json")
        del dumped["trainer_model_id"]
        lines.append(json.dumps(dumped))
    (events_dir / f"{source.to_name()}.jsonl").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _source_of(model_id: str | None) -> SimpleProcessIdentity | TrainerControllerProcessIdentity:
    if model_id is None:
        return SimpleProcessIdentity(component="main")
    return TrainerControllerProcessIdentity(trainer_id=f"{model_id}-actor", model_id=model_id)


def _partial(
    *,
    weight_version: int,
    engine_checksums: dict[str, dict[str, str]],
    rollout_id: int | None = None,
    trainer_model_id: str | None = None,
) -> dict[str, Any]:
    return dict(
        rollout_id=weight_version if rollout_id is None else rollout_id,
        weight_version=weight_version,
        trainer_model_id=trainer_model_id,
        adjacent_weight_change_expected=True,
        engine_checksums=engine_checksums,
    )


class TestCompareInferenceEngineChecksums:
    def test_identical_passes(self, tmp_path: Path) -> None:
        """Internally-consistent sides with equal representative checksums pass."""
        partials = [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}, "b": {"rank0/w": "aaa"}})]
        _write_inference_engine_events(tmp_path / "baseline", partials)
        _write_inference_engine_events(tmp_path / "target", partials)

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_differing_engine_counts_still_pass(self, tmp_path: Path) -> None:
        """Engine count may differ between sides; only internal agreement + representative equality matter."""
        _write_inference_engine_events(
            tmp_path / "baseline", [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}})]
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(
                    weight_version=1,
                    engine_checksums={
                        "a": {"rank0/w": "aaa"},
                        "b": {"rank0/w": "aaa"},
                        "c": {"rank0/w": "aaa"},
                    },
                )
            ],
        )

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_cells_that_were_replaced_between_the_two_runs_still_compare(self, tmp_path: Path) -> None:
        """A healed run serves the same weights from other cells, and comparing by cell id would call that drift."""
        _write_inference_engine_events(
            tmp_path / "baseline", [_partial(weight_version=1, engine_checksums={"cell-old": {"rank0/w": "aaa"}})]
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(weight_version=1, engine_checksums={"cell-new": {"rank0/w": "aaa"}})]
        )

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_the_first_published_version_is_compared_like_any_other(self, tmp_path: Path) -> None:
        """The startup push is recorded rather than skipped, so two runs that boot from different weights fail."""
        _write_inference_engine_events(
            tmp_path / "baseline",
            [
                _partial(weight_version=1, engine_checksums={"a": {"rank0/w": "init_baseline"}}),
                _partial(weight_version=2, engine_checksums={"a": {"rank0/w": "aaa"}}),
            ],
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(weight_version=1, engine_checksums={"a": {"rank0/w": "init_target"}}),
                _partial(weight_version=2, engine_checksums={"a": {"rank0/w": "aaa"}}),
            ],
        )

        with pytest.raises(AssertionError):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_baseline_engines_disagree_fails(self, tmp_path: Path) -> None:
        """If baseline's own engines disagree, the comparison fails (caught by the consistency rule)."""
        _write_inference_engine_events(
            tmp_path / "baseline",
            [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}, "b": {"rank0/w": "zzz"}})],
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}})]
        )

        with pytest.raises(AssertionError, match="Baseline engines disagree"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_target_engines_disagree_fails(self, tmp_path: Path) -> None:
        """If target's own engines disagree, the comparison fails."""
        _write_inference_engine_events(
            tmp_path / "baseline", [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}})]
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}, "b": {"rank0/w": "zzz"}})],
        )

        with pytest.raises(AssertionError, match="Target engines disagree"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_representative_mismatch_fails(self, tmp_path: Path) -> None:
        """Internally-consistent sides whose representatives differ fail and name the tensor."""
        _write_inference_engine_events(
            tmp_path / "baseline", [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}})]
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "zzz"}})]
        )

        with pytest.raises(AssertionError, match=r"key rank0/w"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_missing_version_fails(self, tmp_path: Path) -> None:
        """A published version present only on one side fails closed."""
        _write_inference_engine_events(
            tmp_path / "baseline",
            [
                _partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}}),
                _partial(weight_version=2, engine_checksums={"a": {"rank0/w": "ccc"}}),
            ],
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}})]
        )

        with pytest.raises(AssertionError, match=r"\(model_id, weight_version\) sets differ"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_empty_baseline_fails(self, tmp_path: Path) -> None:
        """No baseline events fails closed rather than vacuously passing."""
        _write_inference_engine_events(tmp_path / "baseline", [])
        _write_inference_engine_events(
            tmp_path / "target", [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}})]
        )

        with pytest.raises(AssertionError, match="No InferenceEngineWeightChecksumEvents found in baseline"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_two_sides_whose_evidence_is_equally_missing_do_not_pass(self, tmp_path: Path) -> None:
        """Both sides recording an empty publication is the shape under which a real drift would go unnoticed."""
        partials = [
            _partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}}),
            _partial(weight_version=2, engine_checksums={}),
        ]
        _write_inference_engine_events(tmp_path / "baseline", partials)
        _write_inference_engine_events(tmp_path / "target", partials)

        with pytest.raises(AssertionError, match="names no inference cell"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_a_cell_that_recorded_no_tensor_does_not_pass(self, tmp_path: Path) -> None:
        """An empty tensor map compares equal to anything, so it would hide whatever that cell actually holds."""
        partials = [_partial(weight_version=1, engine_checksums={"a": {}})]
        _write_inference_engine_events(tmp_path / "baseline", partials)
        _write_inference_engine_events(tmp_path / "target", partials)

        with pytest.raises(AssertionError, match="reported no tensor checksum"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))


class TestSeveralPolicies:
    def test_the_same_weight_version_of_two_policies_is_not_a_duplicate(self, tmp_path: Path) -> None:
        """Every policy counts its own versions, so keying by version alone rejects a legal multi policy run."""
        for side in ("baseline", "target"):
            _write_inference_engine_events(
                tmp_path / side,
                [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}}, trainer_model_id="a")],
                model_id="a",
            )
            _write_inference_engine_events(
                tmp_path / side,
                [_partial(weight_version=1, engine_checksums={"b": {"rank0/w": "bbb"}}, trainer_model_id="b")],
                model_id="b",
            )

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_a_policy_whose_weights_differ_is_reported(self, tmp_path: Path) -> None:
        """Comparing only one of the two policies would hide exactly the drift this comparison exists to catch."""
        _write_inference_engine_events(
            tmp_path / "baseline",
            [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}}, trainer_model_id="a")],
            model_id="a",
        )
        _write_inference_engine_events(
            tmp_path / "baseline",
            [_partial(weight_version=1, engine_checksums={"b": {"rank0/w": "bbb"}}, trainer_model_id="b")],
            model_id="b",
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}}, trainer_model_id="a")],
            model_id="a",
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [_partial(weight_version=1, engine_checksums={"b": {"rank0/w": "ccc"}}, trainer_model_id="b")],
            model_id="b",
        )

        with pytest.raises(AssertionError, match=r"baseline/b/weight_v1 vs target/b/weight_v1"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_the_writers_identity_no_longer_decides_which_policy_an_event_belongs_to(self, tmp_path: Path) -> None:
        """The orchestration script writes every policy's event, so only the payload can name the policy."""
        for side in ("baseline", "target"):
            _write_inference_engine_events(
                tmp_path / side,
                [
                    _partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}}, trainer_model_id="a"),
                    _partial(weight_version=1, engine_checksums={"b": {"rank0/w": "bbb"}}, trainer_model_id="b"),
                ],
            )

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))


class TestLogsPredatingTheTrainerModelIdField:
    def test_events_without_the_field_read_back_as_one_unnamed_policy(self, tmp_path: Path) -> None:
        """A log written before the event named its policy is a single policy run, so it compares under None."""
        partials = [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}})]
        _write_events_without_trainer_model_id(tmp_path / "baseline", partials)
        _write_inference_engine_events(tmp_path / "target", partials)

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_a_field_less_event_still_compares_its_checksums(self, tmp_path: Path) -> None:
        """Reading the missing field as None must not turn the old side into an unchecked one."""
        _write_events_without_trainer_model_id(
            tmp_path / "baseline", [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}})]
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "zzz"}})]
        )

        with pytest.raises(AssertionError, match=r"key rank0/w"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))


class TestAssertEngineWeightsMoved:
    def test_fewer_than_two_updates_do_not_prove_that_engine_weights_moved(self, tmp_path: Path) -> None:
        """A lone weight update provides no pair from which movement can be established."""
        _write_inference_engine_events(
            tmp_path / "target", [_partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}})]
        )

        with pytest.raises(AssertionError, match="there is no pair to compare"):
            assert_engine_weights_moved(side="target", dump_dir=str(tmp_path / "target"))

    def test_identical_updates_do_not_prove_that_engine_weights_moved(self, tmp_path: Path) -> None:
        """Reordered checksum items remain the same weights and cannot establish movement."""
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(weight_version=1, engine_checksums={"a": {"rank0/a": "aaa", "rank0/b": "bbb"}}),
                _partial(weight_version=2, engine_checksums={"a": {"rank0/b": "bbb", "rank0/a": "aaa"}}),
            ],
        )

        with pytest.raises(AssertionError, match="byte-identical engine weights"):
            assert_engine_weights_moved(side="target", dump_dir=str(tmp_path / "target"))

    def test_distinct_updates_prove_that_engine_weights_moved(self, tmp_path: Path) -> None:
        """Two distinct checksum dictionaries establish that engine weights changed."""
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}}),
                _partial(weight_version=2, engine_checksums={"a": {"rank0/w": "bbb"}}),
            ],
        )

        assert_engine_weights_moved(side="target", dump_dir=str(tmp_path / "target"))


class TestAssertEngineCount:
    def test_a_run_every_update_of_which_covered_both_engines_passes(self, tmp_path: Path) -> None:
        """The happy path has to stay reachable, or the refusals below prove nothing."""
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}, "b": {"rank0/w": "aaa"}}),
                _partial(weight_version=2, engine_checksums={"a": {"rank0/w": "bbb"}, "b": {"rank0/w": "bbb"}}),
            ],
        )

        assert_engine_count(side="target", dump_dir=str(tmp_path / "target"), expected=2)

    def test_an_update_that_reached_one_engine_of_two_is_caught(self, tmp_path: Path) -> None:
        """An engine that dropped out mid-run still leaves a run that trains, so nothing else notices."""
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(weight_version=1, engine_checksums={"a": {"rank0/w": "aaa"}, "b": {"rank0/w": "aaa"}}),
                _partial(weight_version=2, engine_checksums={"a": {"rank0/w": "bbb"}}),
            ],
        )

        with pytest.raises(AssertionError, match=r"pushed to \[1, 2\] engine"):
            assert_engine_count(side="target", dump_dir=str(tmp_path / "target"), expected=2)

    def test_a_side_that_pushed_weights_to_no_engine_at_all_is_caught(self, tmp_path: Path) -> None:
        """A side whose engines never registered would otherwise report every count it was asked for."""
        (tmp_path / "target" / EVENTS_DIRNAME).mkdir(parents=True)

        with pytest.raises(AssertionError, match="no engine ever took weights"):
            assert_engine_count(side="target", dump_dir=str(tmp_path / "target"), expected=2)
