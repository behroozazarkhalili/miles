from dataclasses import dataclass

import pytest
import torch


@dataclass
class _FakeMapping:
    sglang_name: str
    num_shards: int
    num_local_experts: int | None = None


class _FakeParameterMapper:
    def __init__(self, mappings: dict[str, _FakeMapping]):
        self.mappings = mappings

    def map(self, name: str) -> _FakeMapping:
        return self.mappings[name]


def _stager_with(tensor_staging, mappings: dict[str, _FakeMapping], sglang_names: list[str]):
    return (
        tensor_staging.NamedTensorStager(),
        _FakeParameterMapper(mappings),
        {name: torch.zeros(1) for name in sglang_names},
    )


class TestGetTransferReadyParams:
    """Staging of HF-named shards until their mapped sglang parameter is complete."""

    def test_a_single_shard_parameter_is_ready_as_soon_as_it_arrives(self, tensor_staging) -> None:
        """A parameter that maps one-to-one needs no accumulation and transfers immediately."""
        stager, mapper, params_dict = _stager_with(
            tensor_staging, {"hf.embed": _FakeMapping("embed", num_shards=1)}, ["embed"]
        )
        tensor = torch.zeros(2)

        ready_params, ready_tensors = stager.get_transfer_ready_params(
            [("hf.embed", tensor)], param_mapper=mapper, params_dict=params_dict
        )

        assert ready_params == ["embed"]
        assert ready_tensors == [("hf.embed", tensor)]
        stager.assert_all_transferred()

    def test_a_fused_parameter_waits_until_every_shard_has_been_staged(self, tensor_staging) -> None:
        """q/k/v arrive separately, so nothing may transfer before the third shard lands."""
        mappings = {
            "hf.q": _FakeMapping("qkv_proj", num_shards=3),
            "hf.k": _FakeMapping("qkv_proj", num_shards=3),
            "hf.v": _FakeMapping("qkv_proj", num_shards=3),
        }
        stager, mapper, params_dict = _stager_with(tensor_staging, mappings, ["qkv_proj"])
        tensors = {name: torch.zeros(2) for name in mappings}

        first = stager.get_transfer_ready_params(
            [("hf.q", tensors["hf.q"])], param_mapper=mapper, params_dict=params_dict
        )
        second = stager.get_transfer_ready_params(
            [("hf.k", tensors["hf.k"])], param_mapper=mapper, params_dict=params_dict
        )
        third = stager.get_transfer_ready_params(
            [("hf.v", tensors["hf.v"])], param_mapper=mapper, params_dict=params_dict
        )

        assert first == ([], [])
        assert second == ([], [])
        assert third == (
            ["qkv_proj"],
            [("hf.q", tensors["hf.q"]), ("hf.k", tensors["hf.k"]), ("hf.v", tensors["hf.v"])],
        )
        stager.assert_all_transferred()

    def test_all_shards_of_a_fused_parameter_inside_one_bucket_are_returned_together(self, tensor_staging) -> None:
        """A bucket that already carries every shard completes the parameter in a single call."""
        mappings = {
            "hf.q": _FakeMapping("qkv_proj", num_shards=2),
            "hf.k": _FakeMapping("qkv_proj", num_shards=2),
        }
        stager, mapper, params_dict = _stager_with(tensor_staging, mappings, ["qkv_proj"])
        tensors = {name: torch.zeros(2) for name in mappings}

        ready_params, ready_tensors = stager.get_transfer_ready_params(
            [("hf.q", tensors["hf.q"]), ("hf.k", tensors["hf.k"])],
            param_mapper=mapper,
            params_dict=params_dict,
        )

        assert ready_params == ["qkv_proj"]
        assert ready_tensors == [("hf.q", tensors["hf.q"]), ("hf.k", tensors["hf.k"])]

    def test_an_expert_parameter_expects_one_shard_per_local_expert(self, tensor_staging) -> None:
        """MoE weights are fused over experts too, so the expected count multiplies by the local expert count."""
        mappings = {
            f"hf.expert{i}.{part}": _FakeMapping("w13", num_shards=2, num_local_experts=2)
            for i in range(2)
            for part in ("gate", "up")
        }
        stager, mapper, params_dict = _stager_with(tensor_staging, mappings, ["w13"])
        names = list(mappings)

        partial = stager.get_transfer_ready_params(
            [(name, torch.zeros(1)) for name in names[:3]], param_mapper=mapper, params_dict=params_dict
        )
        ready_params, ready_tensors = stager.get_transfer_ready_params(
            [(names[3], torch.zeros(1))], param_mapper=mapper, params_dict=params_dict
        )

        assert partial == ([], [])
        assert ready_params == ["w13"]
        assert [name for name, _tensor in ready_tensors] == names

    def test_a_parameter_missing_from_the_replica_is_skipped_without_being_staged(self, tensor_staging) -> None:
        """The shared replica holds only the target's shard, so unknown mapped names must not accumulate."""
        mappings = {
            "hf.unknown": _FakeMapping("not_in_replica", num_shards=2),
            "hf.embed": _FakeMapping("embed", num_shards=1),
        }
        stager, mapper, params_dict = _stager_with(tensor_staging, mappings, ["embed"])

        ready_params, ready_tensors = stager.get_transfer_ready_params(
            [("hf.unknown", torch.zeros(1)), ("hf.embed", torch.zeros(1))],
            param_mapper=mapper,
            params_dict=params_dict,
        )

        assert ready_params == ["embed"]
        assert [name for name, _tensor in ready_tensors] == ["hf.embed"]
        stager.assert_all_transferred()

    def test_two_fused_parameters_are_accumulated_independently(self, tensor_staging) -> None:
        """Interleaved shards of different parameters must not be mixed into one transfer."""
        mappings = {
            "hf.q": _FakeMapping("qkv_proj", num_shards=2),
            "hf.k": _FakeMapping("qkv_proj", num_shards=2),
            "hf.gate": _FakeMapping("gate_up_proj", num_shards=2),
            "hf.up": _FakeMapping("gate_up_proj", num_shards=2),
        }
        stager, mapper, params_dict = _stager_with(tensor_staging, mappings, ["qkv_proj", "gate_up_proj"])

        first = stager.get_transfer_ready_params(
            [("hf.q", torch.zeros(1)), ("hf.gate", torch.zeros(1))], param_mapper=mapper, params_dict=params_dict
        )
        ready_params, ready_tensors = stager.get_transfer_ready_params(
            [("hf.up", torch.zeros(1)), ("hf.k", torch.zeros(1))], param_mapper=mapper, params_dict=params_dict
        )

        assert first == ([], [])
        assert ready_params == ["gate_up_proj", "qkv_proj"]
        assert [name for name, _tensor in ready_tensors] == ["hf.gate", "hf.up", "hf.q", "hf.k"]
        stager.assert_all_transferred()


class TestAssertAllTransferred:
    """The end-of-stream guard against silently dropped shards."""

    def test_a_freshly_created_stager_holds_nothing(self, tensor_staging) -> None:
        """Nothing was staged yet, so the guard must stay quiet."""
        tensor_staging.NamedTensorStager().assert_all_transferred()

    def test_an_incomplete_fused_parameter_is_reported(self, tensor_staging) -> None:
        """A missing shard would silently ship a half-written buffer, so the leftover must raise."""
        mappings = {"hf.q": _FakeMapping("qkv_proj", num_shards=3)}
        stager, mapper, params_dict = _stager_with(tensor_staging, mappings, ["qkv_proj"])

        stager.get_transfer_ready_params([("hf.q", torch.zeros(1))], param_mapper=mapper, params_dict=params_dict)

        with pytest.raises(AssertionError, match="qkv_proj"):
            stager.assert_all_transferred()
