import hashlib
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.training_utils.weight_update.base_weight_checksums import (
    BaseWeightChecksumRecorder,
    CellChecksumEvidence,
    digest_named_params,
    gather_base_weight_checksums,
)

_MODULE = "miles.backends.training_utils.weight_update.base_weight_checksums"


def _sha256_of(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()).hexdigest()


def _patched_gloo_group(other_rank_evidence: list[dict[str, CellChecksumEvidence]]):
    def all_gather_object(gathered, obj, group=None):
        gathered[0] = obj
        for index, evidence in enumerate(other_rank_evidence):
            gathered[index + 1] = evidence

    dist_mock = MagicMock()
    dist_mock.get_world_size.return_value = 1 + len(other_rank_evidence)
    dist_mock.all_gather_object.side_effect = all_gather_object
    return patch(f"{_MODULE}.dist", dist_mock)


def _evidence(cell_id: str, engine_rank: int, digests: dict[str, str]) -> dict[str, CellChecksumEvidence]:
    return {cell_id: CellChecksumEvidence(engine_ranks=(engine_rank,), manifest={str(engine_rank): digests})}


class TestDigestRepresentation:
    def test_a_digest_is_the_sha256_of_the_raw_tensor_bytes(self) -> None:
        """The receiver hashes the raw uint8 view, so any other representation never matches."""
        params_dict = {"model.layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}

        digests = digest_named_params(names=["model.layer.weight"], params_dict=params_dict)

        assert digests == {"model.layer.weight": _sha256_of(params_dict["model.layer.weight"])}

    def test_a_digest_covers_every_byte_of_a_non_float32_tensor(self) -> None:
        """Quantized and bf16 shards reach the engine as raw bytes, not as numeric values."""
        params_dict = {"w": torch.arange(4, dtype=torch.bfloat16), "scale": torch.arange(2, dtype=torch.uint8)}

        digests = digest_named_params(names=["w", "scale"], params_dict=params_dict)

        assert digests == {"w": _sha256_of(params_dict["w"]), "scale": _sha256_of(params_dict["scale"])}

    def test_a_digest_describes_the_bytes_of_a_non_contiguous_view(self) -> None:
        """A transposed or sliced shared buffer must be hashed in the layout the receiver reads back."""
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        params_dict = {"w": source.t()}

        digests = digest_named_params(names=["w"], params_dict=params_dict)

        assert digests == {"w": _sha256_of(params_dict["w"].contiguous())}
        assert digests["w"] != _sha256_of(source)

    def test_a_changed_byte_changes_the_digest(self) -> None:
        """A truncated or partially written transfer differs from the source in single bytes only."""
        params_dict = {"w": torch.zeros(8, dtype=torch.float32)}
        before = digest_named_params(names=["w"], params_dict=params_dict)

        params_dict["w"][3] = 1.0

        assert digest_named_params(names=["w"], params_dict=params_dict) != before

    def test_only_the_named_params_are_digested(self) -> None:
        """A bucket describes the params it just loaded; the rest of the buffer still holds older weights."""
        params_dict = {"w": torch.zeros(2), "other": torch.zeros(2)}

        assert list(digest_named_params(names=["w"], params_dict=params_dict)) == ["w"]


class TestRecorder:
    def test_a_manifest_is_keyed_by_the_engine_tp_rank_as_a_string(self) -> None:
        """The wire manifest the engine looks itself up in is keyed by str(tp_rank)."""
        recorder = BaseWeightChecksumRecorder()

        recorder.record(cell_id="cell-a", engine_rank=3, digests={"w": "aa"})

        assert recorder.manifest_of("cell-a") == {"3": {"w": "aa"}}

    def test_each_cell_keeps_its_own_evidence(self) -> None:
        """Publishing one cell's manifest to another cell would verify bytes that cell never received."""
        recorder = BaseWeightChecksumRecorder()

        recorder.record(cell_id="cell-a", engine_rank=0, digests={"w": "aa"})
        recorder.record(cell_id="cell-b", engine_rank=0, digests={"w": "bb"})

        assert recorder.manifest_of("cell-a") == {"0": {"w": "aa"}}
        assert recorder.manifest_of("cell-b") == {"0": {"w": "bb"}}

    def test_later_buckets_extend_the_manifest_of_an_engine_rank(self) -> None:
        """The engine checks the full parameter set, which only one whole round of buckets describes."""
        recorder = BaseWeightChecksumRecorder()

        recorder.record(cell_id="cell-a", engine_rank=0, digests={"w1": "aa"})
        recorder.record(cell_id="cell-a", engine_rank=0, digests={"w2": "bb"})

        assert recorder.manifest_of("cell-a") == {"0": {"w1": "aa", "w2": "bb"}}

    def test_two_different_digests_for_one_tensor_are_rejected(self) -> None:
        """Picking one of them would publish a manifest that does not describe what was sent."""
        recorder = BaseWeightChecksumRecorder()
        recorder.record(cell_id="cell-a", engine_rank=0, digests={"w": "aa"})

        with pytest.raises(AssertionError, match="two different checksums"):
            recorder.record(cell_id="cell-a", engine_rank=0, digests={"w": "bb"})

    def test_the_same_digest_may_be_recorded_again(self) -> None:
        """A retried or re-planned write ships identical bytes and is not a contradiction."""
        recorder = BaseWeightChecksumRecorder()

        recorder.record(cell_id="cell-a", engine_rank=0, digests={"w": "aa"})
        recorder.record(cell_id="cell-a", engine_rank=0, digests={"w": "aa"})

        assert recorder.manifest_of("cell-a") == {"0": {"w": "aa"}}

    def test_clearing_drops_the_evidence_of_the_previous_round(self) -> None:
        """Stale evidence would let a reconnected cell be verified against a weight version it never got."""
        recorder = BaseWeightChecksumRecorder()
        recorder.record(cell_id="cell-a", engine_rank=0, digests={"w": "aa"})

        recorder.clear()

        assert recorder.manifest_of("cell-a") == {}

    def test_a_cell_without_evidence_reports_an_empty_manifest(self) -> None:
        """An empty manifest still has to travel as a manifest, so the engine rejects the cell."""
        assert BaseWeightChecksumRecorder().manifest_of("cell-a") == {}

    def test_the_returned_manifest_does_not_alias_the_recorder(self) -> None:
        """The caller hands the manifest to the wire; mutating it must not rewrite the recorded evidence."""
        recorder = BaseWeightChecksumRecorder()
        recorder.record(cell_id="cell-a", engine_rank=0, digests={"w": "aa"})

        recorder.manifest_of("cell-a")["0"]["w"] = "tampered"

        assert recorder.manifest_of("cell-a") == {"0": {"w": "aa"}}


class TestGatherAcrossTrainerRanks:
    def test_pipeline_shards_of_one_engine_rank_are_merged(self) -> None:
        """Each pp source rank sends its own parameter subset, and the engine checks the union."""
        local = _evidence("cell-a", 0, {"layer0.w": "aa"})
        remote = _evidence("cell-a", 0, {"layer1.w": "bb"})

        with _patched_gloo_group([remote]):
            manifests = gather_base_weight_checksums(local_evidence=local, group=MagicMock())

        assert manifests == {"cell-a": {"0": {"layer0.w": "aa", "layer1.w": "bb"}}}

    def test_engine_ranks_written_by_different_trainer_ranks_are_merged(self) -> None:
        """The transfer plan spreads the tp ranks of one cell over several trainer ranks."""
        local = _evidence("cell-a", 0, {"w": "aa"})
        remote = _evidence("cell-a", 1, {"w": "bb"})

        with _patched_gloo_group([remote]):
            manifests = gather_base_weight_checksums(local_evidence=local, group=MagicMock())

        assert manifests == {"cell-a": {"0": {"w": "aa"}, "1": {"w": "bb"}}}

    def test_every_cell_of_the_update_gets_its_own_manifest(self) -> None:
        """A cell written only by another trainer rank still has to be verified by the driver."""
        local = _evidence("cell-a", 0, {"w": "aa"})
        remote = _evidence("cell-b", 0, {"w": "bb"})

        with _patched_gloo_group([remote]):
            manifests = gather_base_weight_checksums(local_evidence=local, group=MagicMock())

        assert manifests == {"cell-a": {"0": {"w": "aa"}}, "cell-b": {"0": {"w": "bb"}}}

    def test_a_rank_that_wrote_nothing_still_joins_the_gather(self) -> None:
        """A collective behind an is_sender branch hangs every rank that skipped it."""
        remote = _evidence("cell-a", 0, {"w": "aa"})

        with _patched_gloo_group([remote]) as dist_mock:
            manifests = gather_base_weight_checksums(local_evidence={}, group=MagicMock())

        assert dist_mock.all_gather_object.call_count == 1
        assert manifests == {"cell-a": {"0": {"w": "aa"}}}

    def test_an_engine_rank_nobody_described_fails_the_gather(self) -> None:
        """Publishing the other tp ranks would let the engine accept a shard that was never checked."""
        local = {"cell-a": CellChecksumEvidence(engine_ranks=(0, 1), manifest={"0": {"w": "aa"}})}

        with _patched_gloo_group([]), pytest.raises(AssertionError, match="engine ranks"):
            gather_base_weight_checksums(local_evidence=local, group=MagicMock())

    def test_an_engine_rank_described_by_another_trainer_rank_completes_the_cover(self) -> None:
        """The rank that planned a peer and the rank that wrote it are the same rank only by accident."""
        local = {"cell-a": CellChecksumEvidence(engine_ranks=(0, 1), manifest={"0": {"w": "aa"}})}
        remote = {"cell-a": CellChecksumEvidence(engine_ranks=(1,), manifest={"1": {"w": "bb"}})}

        with _patched_gloo_group([remote]):
            manifests = gather_base_weight_checksums(local_evidence=local, group=MagicMock())

        assert manifests == {"cell-a": {"0": {"w": "aa"}, "1": {"w": "bb"}}}

    def test_two_trainer_ranks_contradicting_each_other_fail_the_gather(self) -> None:
        """One of them shipped bytes the manifest does not describe, and neither is known to be right."""
        local = _evidence("cell-a", 0, {"w": "aa"})
        remote = _evidence("cell-a", 0, {"w": "bb"})

        with _patched_gloo_group([remote]), pytest.raises(AssertionError, match="two different checksums"):
            gather_base_weight_checksums(local_evidence=local, group=MagicMock())

    def test_a_cell_nobody_wrote_to_keeps_an_empty_manifest(self) -> None:
        """Turning it into no manifest at all would let the engine publish unverified weights."""
        local = {"cell-a": CellChecksumEvidence(engine_ranks=(), manifest={})}

        with _patched_gloo_group([{}]):
            manifests = gather_base_weight_checksums(local_evidence=local, group=MagicMock())

        assert manifests == {"cell-a": {}}
