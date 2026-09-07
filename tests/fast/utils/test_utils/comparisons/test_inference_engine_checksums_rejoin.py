from pathlib import Path

import pytest

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, EventLogger
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.comparisons.inference_engine_checksums import assert_engines_rejoined_weight_updates


def _write_updates(side_dir: Path, engines_per_rollout: list[int], *, trainer_model_id: str | None = None) -> None:
    source = SimpleProcessIdentity(component="main")
    event_logger = EventLogger(
        log_dir=side_dir / EVENTS_DIRNAME,
        source=source,
        file_name=f"{source.to_name()}.jsonl",
    )
    for rollout_id, num_engines in enumerate(engines_per_rollout):
        event_logger.log(
            InferenceEngineWeightChecksumEvent,
            dict(
                rollout_id=rollout_id,
                weight_version=rollout_id + 1,
                adjacent_weight_change_expected=True,
                trainer_model_id=trainer_model_id,
                engine_checksums={f"cell-{index}": {"rank0/w": f"hash-{rollout_id}"} for index in range(num_engines)},
            ),
            print_log=False,
        )
    event_logger.close()


class TestAssertEnginesRejoinedWeightUpdates:
    def test_an_update_that_lost_an_engine_mid_run_is_accepted(self, tmp_path: Path) -> None:
        """Publishing to the engines still alive is what partial-target weight update is for."""
        _write_updates(tmp_path, [4, 3, 4])

        assert_engines_rejoined_weight_updates(side="target", dump_dir=str(tmp_path), expected=4)

    def test_a_run_ending_with_an_engine_still_out_of_the_fan_out_fails(self, tmp_path: Path) -> None:
        """A replacement that never took weights leaves the run serving from an unaudited engine."""
        _write_updates(tmp_path, [4, 3, 3])

        with pytest.raises(AssertionError, match="last weight update covered"):
            assert_engines_rejoined_weight_updates(side="target", dump_dir=str(tmp_path), expected=4)

    def test_an_update_reaching_more_engines_than_the_run_owns_fails(self, tmp_path: Path) -> None:
        """A dead engine left in the fan-out beside its replacement is a leak this count is the only witness to."""
        _write_updates(tmp_path, [4, 5, 4])

        with pytest.raises(AssertionError, match="more than the 4 engine"):
            assert_engines_rejoined_weight_updates(side="target", dump_dir=str(tmp_path), expected=4)

    def test_a_run_that_pushed_no_weights_at_all_fails(self, tmp_path: Path) -> None:
        """No events means no engine ever took weights, which must not read as every engine rejoining."""
        with pytest.raises(AssertionError, match="no InferenceEngineWeightChecksumEvents"):
            assert_engines_rejoined_weight_updates(side="target", dump_dir=str(tmp_path), expected=4)

    def test_each_policy_is_judged_by_its_own_last_update(self, tmp_path: Path) -> None:
        """One policy's complete final update must not cover for another policy's missing engine."""
        _write_updates(tmp_path, [4, 4])
        _write_updates(tmp_path, [4, 3], trainer_model_id="second")

        with pytest.raises(AssertionError, match="last weight update covered"):
            assert_engines_rejoined_weight_updates(side="target", dump_dir=str(tmp_path), expected=4)
