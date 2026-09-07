import dataclasses
from pathlib import Path

import pytest

from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_ft_args, get_weight_transfer_args
from tests.e2e.ft.conftest_ft.modes import MODES


class TestGetCommonTrainArgs:
    def test_fault_tolerance_runs_request_inference_engine_weight_checksums(self, tmp_path: Path) -> None:
        """The shared FT launch arguments must request inference-engine weight checksums."""
        args = get_common_train_args(MODES["kill_rollout__dp4"], dump_dir=str(tmp_path))

        assert "--save-inference-engine-weight-checksum " in args

    def test_a_colocated_real_rollout_mode_emits_the_colocate_flag(self, tmp_path: Path) -> None:
        """A colocated mode must tell the trainer to share its gpus with the rollout engines."""
        mode = dataclasses.replace(MODES["kill_rollout__dp4"], colocate=True)

        args = get_common_train_args(mode, dump_dir=str(tmp_path))

        assert "--colocate " in args

    def test_a_disaggregated_real_rollout_mode_omits_the_colocate_flag(self, tmp_path: Path) -> None:
        """Rollout engines on their own gpus must not be colocated with the trainer."""
        args = get_common_train_args(MODES["kill_train__dp2_cp2__moe_5layer"], dump_dir=str(tmp_path))

        assert "--rollout-num-gpus" in args
        assert "--colocate" not in args

    def test_a_debug_rollout_mode_omits_the_colocate_flag_even_when_the_mode_is_colocated(
        self, tmp_path: Path
    ) -> None:
        """Without real rollout engines there is nothing to colocate, whatever the mode declares."""
        mode = dataclasses.replace(
            MODES["kill_rollout__dp4"], rollout_num_engines=0, ft_components=("train",), colocate=True
        )

        args = get_common_train_args(mode, dump_dir=str(tmp_path))

        assert mode.colocate is True
        assert "--debug-train-only" in args
        assert "--colocate" not in args


class TestGetFtArgs:
    def test_a_rollout_only_ft_mode_propagates_the_rollout_component_and_api_server_port(self) -> None:
        """Rollout-only fault tolerance must not silently enable trainer fault tolerance."""
        args = get_ft_args(MODES["kill_rollout__dp4"])

        assert args == "--use-fault-tolerance --ft-components rollout --api-server-port 0 "

    def test_a_trainer_mode_propagates_the_train_component(self) -> None:
        """The trainer-fault-tolerance modes keep sending the train component."""
        args = get_ft_args(MODES["kill_train__dp2_cp2__moe_5layer"])

        assert args == "--use-fault-tolerance --ft-components train --api-server-port 0 "


class TestGetWeightTransferArgs:
    def test_a_real_rollout_mode_asks_for_the_p2p_protocol_the_fault_scenarios_exercise(self) -> None:
        """The injection scenarios crash engines mid-update, which is a claim about the p2p protocol, not the default."""
        args = get_weight_transfer_args(MODES["kill_rollout__dp4"])

        assert "--update-weight-transfer-mode p2p " in args

    def test_a_real_rollout_mode_starts_the_engine_side_transfer_engine(self) -> None:
        """P2P reads /get_remote_instance_transfer_engine_info, which an engine without that seed never serves."""
        args = get_weight_transfer_args(MODES["kill_train_rollout__dp2_cp2"])

        assert "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine " in args

    def test_a_debug_rollout_mode_claims_no_transfer_protocol_at_all(self) -> None:
        """A trainer-only mode pushes weights to nothing, so naming a protocol would fake the coverage."""
        assert get_weight_transfer_args(MODES["kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer"]) == ""

    def test_a_colocated_real_rollout_mode_is_refused(self) -> None:
        """validate_args rejects p2p under --colocate, so a scenario must fail before it burns a cluster."""
        mode = dataclasses.replace(MODES["kill_rollout__dp4"], colocate=True)

        with pytest.raises(AssertionError, match="colocated mode"):
            get_weight_transfer_args(mode)
