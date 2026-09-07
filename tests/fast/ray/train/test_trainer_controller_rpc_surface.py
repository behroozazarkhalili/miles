from __future__ import annotations

import pytest
from pydantic_core import PydanticSerializationError

from miles.ray.rollout.inference_controller import UpdatableEngines
from miles.ray.train.group import TrainerController
from miles.utils.ft_utils.api_server.models import FaultHookArmingReport
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.types import DeploymentIdentity

DRIVEN_METHODS = (
    "init",
    "train",
    "save_model",
    "export_hf",
    "update_weights",
    "get_deployment_identity",
    "get_train_parallel_config",
    "get_cell_statuses",
    "arm_fault_hook",
    "onload",
    "offload",
    "clear_memory",
    "reconcile_adapters",
    "dispose",
)


class TestTheTrainerControllerSurfaceIsCallableOverRpc:
    def test_the_whole_surface_is_accepted(self):
        """A split run reaches this pool only over rpc, and one unannotated parameter makes it unstartable."""
        specs = collect_rpc_method_specs(TrainerController)

        assert set(DRIVEN_METHODS) <= set(specs)

    def test_the_engines_a_weight_update_names_cross_the_wire(self):
        """This parameter was unannotated once, and it took the whole pool down at import rather than at call."""
        spec = collect_rpc_method_specs(TrainerController)["update_weights"]
        info = UpdatableEngines(
            rollout_engines=[],
            engine_gpu_counts=[],
            engine_gpu_offsets=[],
            engine_cell_ids=[],
            snapshot_cell_id_to_hashes={},
        )

        decoded = spec.serializer.decode_query(spec.serializer.encode_query(dict(info=info, rollout_id=3)))

        assert decoded["rollout_id"] == 3

    def test_the_incarnation_an_arm_names_crosses_the_wire(self):
        """The api server picks the trainer, but only the trainer can refuse a hash that has moved on."""
        spec = collect_rpc_method_specs(TrainerController)["arm_fault_hook"]

        decoded = spec.serializer.decode_query(
            spec.serializer.encode_query(
                dict(
                    cell_id="trainer-engine-actor-0",
                    expected_workers_hash="workers-hash-a",
                    hook="weight_update.before_p2p_write",
                    mode="sigkill",
                    sub_index=1,
                    request_id="req-1",
                    target="remote_inference_cell",
                    delay_ms=250,
                )
            )
        )

        assert decoded["expected_workers_hash"] == "workers-hash-a" and decoded["sub_index"] == 1

    def test_a_refusal_crosses_back_as_the_model_the_api_server_reads(self):
        """The reason is what turns a stale snapshot into a 400 instead of an arm nobody can account for."""
        spec = collect_rpc_method_specs(TrainerController)["arm_fault_hook"]
        refusal = FaultHookArmingReport(refused_because="it now runs another incarnation")

        restored = spec.serializer.decode_result(spec.serializer.encode_result(refusal))

        assert restored == refusal
        assert spec.serializer.decode_result(spec.serializer.encode_result(FaultHookArmingReport())) == (
            FaultHookArmingReport()
        )

    def test_no_internal_method_is_exposed(self):
        """A method that never crosses the wire is one whose types nobody has to keep honest."""
        specs = collect_rpc_method_specs(TrainerController)

        assert not {"pool_id", "expected_num_cells", "cell_ids", "num_cells"} & set(specs)


class TestTheHandshakeThatJoinsTwoDeployments:
    def test_the_identity_crosses_as_the_model_the_caller_compares(self):
        """The driving launch asserts on its fields, so a mapping arriving instead fails at attribute access."""
        spec = collect_rpc_method_specs(TrainerController)["get_deployment_identity"]
        identity = DeploymentIdentity(run_uuid="0123456789abcdef", deploy_component="trainer")

        restored = spec.serializer.decode_result(spec.serializer.encode_result(identity))

        assert restored == identity

    def test_an_identity_missing_a_field_is_refused_before_it_crosses(self):
        """A half-built identity would compare unequal for a reason the message would not name."""
        spec = collect_rpc_method_specs(TrainerController)["get_deployment_identity"]

        with pytest.raises(PydanticSerializationError, match="DeploymentIdentity"):
            spec.serializer.encode_result({"run_uuid": "0123456789abcdef"})
