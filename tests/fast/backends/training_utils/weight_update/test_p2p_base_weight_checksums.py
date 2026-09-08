import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.training_utils.weight_update.base_weight_checksums import BaseWeightChecksumRecorder
from miles.backends.training_utils.weight_update.inference_cell_health import InferenceCellHealth

_CHECKSUM_MODULE = "miles.backends.training_utils.weight_update.base_weight_checksums"
_P2P_MODULE = "miles.backends.training_utils.weight_update.protocols.p2p"
_REGISTRY = {"w": (0x1000, 4, 4), "w2": (0x2000, 4, 4)}


def _sha256_of(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()).hexdigest()


def _digest_of(value: float) -> str:
    return _sha256_of(torch.full((4,), value, dtype=torch.float32))


class _ShardLoadingReplica:
    def __init__(self, shared_params_dict: dict[str, torch.Tensor], value: float):
        self._shared_params_dict = shared_params_dict
        self._value = value

    def load_weights(self, named_tensors) -> None:
        for name, _tensor in named_tensors:
            self._shared_params_dict[name].fill_(self._value)


class _BucketStager:
    def __init__(self, buckets: list[list[str]]):
        self._buckets = list(buckets)

    def get_transfer_ready_params(self, converted_named_tensors, param_mapper, params_dict):
        names = self._buckets.pop(0)
        return names, [(name, torch.zeros(4, dtype=torch.float32)) for name in names]


class _RecordingCellUpdater:
    def __init__(self, cell_id: str, *, accepts_writes: bool = True):
        self.cell_id = cell_id
        self.accepts_writes = accepts_writes
        self.written_engine_ranks: list[int] = []

    def submit_write(self, engine_rank: int, names: list[str], weight_memory_registry):
        if not self.accepts_writes:
            return None
        self.written_engine_ranks.append(engine_rank)
        return MagicMock()

    def wait_for_write(self, future) -> None:
        pass


def _make_sender(
    p2p,
    *,
    replica_values: dict[int, float],
    cell_updaters: list[_RecordingCellUpdater],
    buckets: list[list[str]],
):
    shared_params_dict = {name: torch.zeros(4, dtype=torch.float32) for name in _REGISTRY}
    recorder = BaseWeightChecksumRecorder()
    protocol = SimpleNamespace(
        is_sender=True,
        _shared_param_mapper=object(),
        _shared_params_dict=shared_params_dict,
        _weight_memory_registry=_REGISTRY,
        _checksum_recorder=recorder,
        _tensor_stager=_BucketStager(buckets),
        _transfer_engine_meta_list=[
            p2p.TransferEngineMeta(
                engine_rank=engine_rank,
                model_replica=_ShardLoadingReplica(shared_params_dict, value),
                cell_updaters=cell_updaters,
            )
            for engine_rank, value in replica_values.items()
        ],
    )
    return protocol, recorder


def _send_one_bucket(p2p, *, replica_values: dict[int, float], cell_updaters: list[_RecordingCellUpdater]):
    protocol, recorder = _make_sender(p2p, replica_values=replica_values, cell_updaters=cell_updaters, buckets=[["w"]])
    p2p.UpdateWeightP2P.send_bucket(protocol, [("hf.w", torch.zeros(4))])
    return recorder


class TestSendBucketRecordsWhatWasSent:
    def test_the_digest_is_taken_after_the_replica_loaded_the_bucket(self, p2p_protocol) -> None:
        """Hashing before load_weights would describe the previous weight version instead of this one."""
        cell = _RecordingCellUpdater("cell-a")

        recorder = _send_one_bucket(p2p_protocol, replica_values={0: 7.0}, cell_updaters=[cell])

        assert recorder.manifest_of("cell-a") == {"0": {"w": _digest_of(7.0)}}

    def test_the_digest_describes_the_shared_buffer_and_not_the_incoming_tensor(self, p2p_protocol) -> None:
        """The engine receives the converted shard, which the HF bucket the trainer streamed never equals."""
        cell = _RecordingCellUpdater("cell-a")

        recorder = _send_one_bucket(p2p_protocol, replica_values={0: 7.0}, cell_updaters=[cell])

        assert recorder.manifest_of("cell-a")["0"]["w"] != _digest_of(0.0)

    def test_each_engine_rank_is_described_by_its_own_load(self, p2p_protocol) -> None:
        """Every rank overwrites the same shared buffer, so a digest taken later describes the wrong shard."""
        cell = _RecordingCellUpdater("cell-a")

        recorder = _send_one_bucket(p2p_protocol, replica_values={0: 1.0, 1: 2.0}, cell_updaters=[cell])

        assert recorder.manifest_of("cell-a") == {"0": {"w": _digest_of(1.0)}, "1": {"w": _digest_of(2.0)}}

    def test_the_buckets_of_one_round_add_up_to_the_whole_parameter_set(self, p2p_protocol) -> None:
        """The engine compares the manifest against every registered parameter, not against one bucket."""
        cell = _RecordingCellUpdater("cell-a")
        protocol, recorder = _make_sender(
            p2p_protocol, replica_values={0: 4.0}, cell_updaters=[cell], buckets=[["w"], ["w2"]]
        )

        p2p_protocol.UpdateWeightP2P.send_bucket(protocol, [("hf.w", torch.zeros(4))])
        p2p_protocol.UpdateWeightP2P.send_bucket(protocol, [("hf.w2", torch.zeros(4))])

        assert recorder.manifest_of("cell-a") == {"0": {"w": _digest_of(4.0), "w2": _digest_of(4.0)}}

    def test_cells_sharing_an_engine_rank_get_the_same_digest_under_their_own_identity(self, p2p_protocol) -> None:
        """The bytes are identical, but each cell is verified and dropped on its own."""
        cells = [_RecordingCellUpdater("cell-a"), _RecordingCellUpdater("cell-b")]

        recorder = _send_one_bucket(p2p_protocol, replica_values={0: 3.0}, cell_updaters=cells)

        assert recorder.manifest_of("cell-a") == {"0": {"w": _digest_of(3.0)}}
        assert recorder.manifest_of("cell-b") == {"0": {"w": _digest_of(3.0)}}

    def test_a_cell_that_was_not_written_to_is_not_described(self, p2p_protocol) -> None:
        """Claiming bytes an errored cell never received would make a later retry look verified."""
        cells = [_RecordingCellUpdater("cell-dead", accepts_writes=False), _RecordingCellUpdater("cell-live")]

        recorder = _send_one_bucket(p2p_protocol, replica_values={0: 5.0}, cell_updaters=cells)

        assert recorder.manifest_of("cell-dead") == {}
        assert recorder.manifest_of("cell-live") == {"0": {"w": _digest_of(5.0)}}

    def test_nothing_is_hashed_when_no_cell_accepts_the_write(self, p2p_protocol) -> None:
        """Hashing the whole model for a rank with no live target is pure overhead on the bucket stream."""
        cells = [_RecordingCellUpdater("cell-dead", accepts_writes=False)]

        with patch(f"{_P2P_MODULE}.digest_named_params") as digest_mock:
            _send_one_bucket(p2p_protocol, replica_values={0: 5.0}, cell_updaters=cells)

        assert digest_mock.call_count == 0


class TestRoundLifetime:
    def test_beginning_a_sync_drops_the_evidence_of_the_previous_round(self, p2p_protocol) -> None:
        """A stale digest would verify a cell against weights of a version it never received."""
        protocol = SimpleNamespace(
            is_sender=False,
            _checksum_recorder=BaseWeightChecksumRecorder(),
            _model_registered=True,
            _shared_params_dict={},
        )
        protocol._checksum_recorder.record(cell_id="cell-a", engine_rank=0, digests={"w": "aa"})

        p2p_protocol.UpdateWeightP2P.begin_sync(protocol, 4, lambda **_kwargs: iter([]))

        assert protocol._checksum_recorder.manifest_of("cell-a") == {}

    def test_disconnecting_drops_the_evidence_of_the_old_peers(self, p2p_protocol) -> None:
        """The cell ids of the previous topology may be reused by engines that were never written to."""
        protocol = SimpleNamespace(
            _checksum_recorder=BaseWeightChecksumRecorder(),
            _cell_updaters_by_cell_id={},
            _unfinished_writes=[],
            _stalled_executors=[],
        )
        protocol._checksum_recorder.record(cell_id="cell-a", engine_rank=0, digests={"w": "aa"})

        p2p_protocol.UpdateWeightP2P.disconnect(protocol)

        assert protocol._checksum_recorder.manifest_of("cell-a") == {}


def _collect(p2p, *, cell_ids: list[str], errored: list[str], engine_ranks: dict[str, tuple[int, ...]], recorder):
    health = InferenceCellHealth(cell_ids)
    for cell_id in errored:
        health.mark_errored(cell_id, RuntimeError("boom"))
    protocol = SimpleNamespace(
        inference_cell_health=health,
        _checksum_recorder=recorder,
        _cell_updaters_by_cell_id={
            cell_id: SimpleNamespace(engine_ranks=engine_ranks.get(cell_id, ())) for cell_id in cell_ids
        },
    )

    gathered_evidence: list[dict] = []

    def all_gather_object(gathered, obj, group=None):
        gathered_evidence.append(obj)
        gathered[0] = obj

    dist_mock = MagicMock()
    dist_mock.get_world_size.return_value = 1
    dist_mock.all_gather_object.side_effect = all_gather_object
    with (
        patch(f"{_CHECKSUM_MODULE}.dist", dist_mock),
        patch(f"{_P2P_MODULE}.get_gloo_group", return_value=MagicMock()),
    ):
        manifests = p2p.UpdateWeightP2P.collect_base_weight_checksums(protocol)
    return manifests, gathered_evidence


class TestCollectBaseWeightChecksums:
    def test_a_healthy_cell_contributes_its_planned_engine_ranks_and_digests(self, p2p_protocol) -> None:
        """Both halves are needed: the digests to verify, the planned ranks to prove the cover is complete."""
        recorder = BaseWeightChecksumRecorder()
        recorder.record(cell_id="cell-a", engine_rank=2, digests={"w": "aa"})

        manifests, gathered = _collect(
            p2p_protocol, cell_ids=["cell-a"], errored=[], engine_ranks={"cell-a": (2,)}, recorder=recorder
        )

        assert gathered[0]["cell-a"].engine_ranks == (2,)
        assert manifests == {"cell-a": {"2": {"w": "aa"}}}

    def test_an_errored_cell_is_left_out_of_the_manifest(self, p2p_protocol) -> None:
        """It is not resumed either way, and its partial cover would fail the completeness check."""
        recorder = BaseWeightChecksumRecorder()
        recorder.record(cell_id="cell-dead", engine_rank=0, digests={"w": "aa"})
        recorder.record(cell_id="cell-live", engine_rank=0, digests={"w": "bb"})

        manifests, _gathered = _collect(
            p2p_protocol,
            cell_ids=["cell-dead", "cell-live"],
            errored=["cell-dead"],
            engine_ranks={"cell-dead": (0,), "cell-live": (0,)},
            recorder=recorder,
        )

        assert manifests == {"cell-live": {"0": {"w": "bb"}}}

    def test_a_rank_without_any_healthy_cell_still_joins_the_gather(self, p2p_protocol) -> None:
        """The gather is a collective over the whole trainer cell, not over its senders."""
        manifests, gathered = _collect(
            p2p_protocol,
            cell_ids=["cell-dead"],
            errored=["cell-dead"],
            engine_ranks={},
            recorder=BaseWeightChecksumRecorder(),
        )

        assert gathered == [{}]
        assert manifests == {}

    def test_a_planned_engine_rank_without_evidence_fails_the_update(self, p2p_protocol) -> None:
        """A tp rank whose bytes nobody described must not be published as verified."""
        recorder = BaseWeightChecksumRecorder()
        recorder.record(cell_id="cell-a", engine_rank=0, digests={"w": "aa"})

        with pytest.raises(AssertionError, match="engine ranks"):
            _collect(p2p_protocol, cell_ids=["cell-a"], errored=[], engine_ranks={"cell-a": (0, 1)}, recorder=recorder)
