from collections.abc import Mapping, Sequence
from typing import NamedTuple

import torch
import torch.distributed as dist

from miles.backends.training_utils.weight_update.utils import sha256_tensor

CellChecksumManifest = dict[str, dict[str, str]]
BaseWeightChecksums = dict[str, CellChecksumManifest]


class CellChecksumEvidence(NamedTuple):
    engine_ranks: tuple[int, ...]
    manifest: CellChecksumManifest


class BaseWeightChecksumRecorder:
    def __init__(self) -> None:
        self._manifest_by_cell_id: BaseWeightChecksums = {}

    def clear(self) -> None:
        self._manifest_by_cell_id = {}

    def record(self, *, cell_id: str, engine_rank: int, digests: Mapping[str, str]) -> None:
        recorded = self._manifest_by_cell_id.setdefault(cell_id, {}).setdefault(str(engine_rank), {})
        _merge_digests(
            recorded,
            digests,
            source=f"cell {cell_id} engine rank {engine_rank} written by this trainer rank",
        )

    def manifest_of(self, cell_id: str) -> CellChecksumManifest:
        manifest = self._manifest_by_cell_id.get(cell_id, {})
        return {engine_rank: dict(digests) for engine_rank, digests in manifest.items()}


def digest_named_params(names: Sequence[str], params_dict: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {name: sha256_tensor(params_dict[name]) for name in names}


def gather_base_weight_checksums(
    *, local_evidence: Mapping[str, CellChecksumEvidence], group: dist.ProcessGroup
) -> BaseWeightChecksums:
    gathered: list[Mapping[str, CellChecksumEvidence] | None] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, dict(local_evidence), group=group)

    engine_ranks_by_cell_id: dict[str, set[int]] = {}
    manifest_by_cell_id: BaseWeightChecksums = {}
    for trainer_rank, evidence_by_cell_id in enumerate(gathered):
        for cell_id, evidence in evidence_by_cell_id.items():
            engine_ranks_by_cell_id.setdefault(cell_id, set()).update(evidence.engine_ranks)
            manifest = manifest_by_cell_id.setdefault(cell_id, {})
            for engine_rank, digests in evidence.manifest.items():
                _merge_digests(
                    manifest.setdefault(engine_rank, {}),
                    digests,
                    source=f"cell {cell_id} engine rank {engine_rank} written by trainer rank {trainer_rank}",
                )

    for cell_id, engine_ranks in engine_ranks_by_cell_id.items():
        _assert_every_engine_rank_is_covered(
            cell_id=cell_id, engine_ranks=engine_ranks, manifest=manifest_by_cell_id[cell_id]
        )
    return manifest_by_cell_id


def _merge_digests(recorded: dict[str, str], digests: Mapping[str, str], source: str) -> None:
    for name, digest in digests.items():
        previous = recorded.setdefault(name, digest)
        assert previous == digest, (
            f"[BASE-WEIGHT-CHECK] {source} carries two different checksums for {name!r} "
            f"({previous} and {digest}), so the bytes this trainer sent cannot be described"
        )


def _assert_every_engine_rank_is_covered(cell_id: str, engine_ranks: set[int], manifest: CellChecksumManifest) -> None:
    expected = {str(engine_rank) for engine_rank in engine_ranks}
    covered = {engine_rank for engine_rank, digests in manifest.items() if digests}
    assert covered == expected, (
        f"[BASE-WEIGHT-CHECK] the base weights this trainer sent to cell {cell_id} are described for engine ranks "
        f"{sorted(covered)}, but the transfer plan writes engine ranks {sorted(expected)}"
    )
