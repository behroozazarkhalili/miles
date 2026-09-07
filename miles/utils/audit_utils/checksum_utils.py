from collections.abc import Mapping
from typing import Any, NamedTuple

InferenceEngineChecksums = dict[str, str]
CellIdToChecksums = dict[str, InferenceEngineChecksums]


class EngineRankIdentity(NamedTuple):
    rank: int
    size: int


def flatten_inference_engine_checksums(cell_id_to_body: Mapping[str, Any]) -> CellIdToChecksums:
    assert cell_id_to_body, (
        f"check_weights('checksum') covered no inference cell at all, so this weight version is unaudited: "
        f"{cell_id_to_body!r}"
    )
    return {
        cell_id: _merge_inference_engine_ranks(cell_id=cell_id, engine_body=engine_body)
        for cell_id, engine_body in sorted(cell_id_to_body.items())
    }


def _merge_inference_engine_ranks(*, cell_id: str, engine_body: Any) -> InferenceEngineChecksums:
    # Ranks arrive in non-deterministic (zmq) order under TP>1; sort and prefix each tensor
    # name with rank{r}/ so distinct shards' identically-named tensors never clobber.
    assert engine_body is not None, (
        f"check_weights engine {cell_id!r} answered an empty body, so its weights are unaudited: "
        f"a cell that cannot report a checksum must fail the audit, not vanish from it"
    )
    assert engine_body.get("success", False), f"check_weights engine {cell_id!r} reported failure: {engine_body!r}"
    ranks: list[dict[str, Any]] = engine_body.get("ranks", []) or []
    assert ranks, f"check_weights engine {cell_id!r} body has no ranks: {engine_body!r}"

    identities = [_compute_rank_identity(cell_id=cell_id, rank_info=rank_info) for rank_info in ranks]
    _assert_covers_every_rank(cell_id=cell_id, identities=identities)

    merged: InferenceEngineChecksums = {}
    for identity, rank_info in sorted(zip(identities, ranks, strict=True), key=lambda pair: pair[0].rank):
        for name, digest in _read_rank_checksums(cell_id=cell_id, rank=identity.rank, rank_info=rank_info).items():
            merged[f"rank{identity.rank}/{name}"] = digest
    return merged


def _read_rank_checksums(*, cell_id: str, rank: int, rank_info: dict[str, Any]) -> InferenceEngineChecksums:
    rank_checksums = rank_info["checksums"]
    assert isinstance(rank_checksums, dict) and rank_checksums, (
        f"check_weights engine {cell_id!r} rank {rank} hashed no tensor at all, so recording it would put a cell "
        f"with no evidence into the audit: {rank_info!r}"
    )
    invalid = sorted(
        f"{name!r}: {digest!r}"
        for name, digest in rank_checksums.items()
        if not (isinstance(name, str) and name and isinstance(digest, str) and digest)
    )
    assert not invalid, (
        f"check_weights engine {cell_id!r} rank {rank} answered {invalid} instead of a tensor name and its hex "
        f"digest, and an entry that names nothing or hashes to nothing compares equal wherever it is copied"
    )
    return rank_checksums


def _compute_rank_identity(*, cell_id: str, rank_info: dict[str, Any]) -> EngineRankIdentity:
    parallelism_info = rank_info["parallelism_info"]
    assert parallelism_info, (
        f"check_weights engine {cell_id!r} answered a rank that names no parallelism group, so which shard of the "
        f"engine it hashed is unknown: {rank_info!r}"
    )
    reported = {(role_info["rank"], role_info["size"]) for role_info in parallelism_info}
    assert len(reported) == 1, (
        f"check_weights engine {cell_id!r} roles disagree on the global rank and world size of one response, got "
        f"{sorted(reported)}: {rank_info!r}"
    )
    [(rank, size)] = reported
    assert _is_ordinal(size) and size > 0, f"check_weights engine {cell_id!r} reports world size {size!r}"
    assert (
        _is_ordinal(rank) and 0 <= rank < size
    ), f"check_weights engine {cell_id!r} reports rank {rank!r} of a world of {size}, which names no shard of it"
    return EngineRankIdentity(rank=rank, size=size)


def _assert_covers_every_rank(*, cell_id: str, identities: list[EngineRankIdentity]) -> None:
    sizes = sorted({identity.size for identity in identities})
    assert len(sizes) == 1, (
        f"check_weights engine {cell_id!r} answered ranks of {sizes} different world sizes in one response, so what "
        f"this audit covers is unknown"
    )
    [size] = sizes

    ranks = [identity.rank for identity in identities]
    duplicated = sorted({rank for rank in ranks if ranks.count(rank) > 1})
    assert not duplicated, (
        f"check_weights engine {cell_id!r} answered GPU rank(s) {duplicated} more than once, so one shard's report "
        f"would overwrite another's and {len(ranks)} answers would pass as a world of {size}"
    )
    missing = sorted(set(range(size)) - set(ranks))
    assert not missing, (
        f"check_weights engine {cell_id!r} answered {len(ranks)} of the {size} ranks it spans and is missing "
        f"{missing}, so the tensors those ranks hold would go unaudited"
    )


def _is_ordinal(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
