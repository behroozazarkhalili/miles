"""Tests for engine_weight_checksum.flatten_inference_engine_checksums."""

from typing import Any

import pytest

from miles.utils.audit_utils.checksum_utils import flatten_inference_engine_checksums


def _engine_body(*, success: bool, ranks: list[dict[str, Any]] | None) -> dict[str, Any]:
    body: dict[str, Any] = {"success": success, "message": "ok"}
    if ranks is not None:
        body["ranks"] = ranks
    return body


def _rank(rank: int, checksums: Any, *, size: int = 1) -> dict[str, Any]:
    return {"checksums": checksums, "parallelism_info": [{"role": "target", "rank": rank, "size": size}]}


def _tp2(*checksums: dict[str, str]) -> dict[str, Any]:
    return _engine_body(success=True, ranks=[_rank(rank, one, size=2) for rank, one in enumerate(checksums)])


class TestFlattenInferenceEngineChecksums:
    def test_single_engine_single_rank(self) -> None:
        """One cell with one rank yields one prefixed checksum dict under that cell's id."""
        result = {"cell-a": _engine_body(success=True, ranks=[_rank(0, {"w": "aaa"})])}
        assert flatten_inference_engine_checksums(result) == {"cell-a": {"rank0/w": "aaa"}}

    def test_multiple_cells_keyed_by_their_own_ids(self) -> None:
        """Each cell's checksums stay attached to the cell that reported them, not to a list position."""
        result = {
            "cell-a": _engine_body(success=True, ranks=[_rank(0, {"w": "e0"})]),
            "cell-b": _engine_body(success=True, ranks=[_rank(0, {"w": "e1"})]),
        }
        assert flatten_inference_engine_checksums(result) == {
            "cell-a": {"rank0/w": "e0"},
            "cell-b": {"rank0/w": "e1"},
        }

    def test_cells_are_ordered_by_cell_id_whatever_order_they_answered_in(self) -> None:
        """Reply order is not semantic, so the flattened map is ordered by cell id."""
        answered_late_first = {
            "cell-b": _engine_body(success=True, ranks=[_rank(0, {"w": "e1"})]),
            "cell-a": _engine_body(success=True, ranks=[_rank(0, {"w": "e0"})]),
        }
        assert list(flatten_inference_engine_checksums(answered_late_first)) == ["cell-a", "cell-b"]

    def test_a_none_body_fails_loud(self) -> None:
        """A cell that answered nothing must fail the audit instead of vanishing from it."""
        result = {"cell-a": _engine_body(success=True, ranks=[_rank(0, {"w": "e0"})]), "cell-b": None}
        with pytest.raises(AssertionError, match="cell-b"):
            flatten_inference_engine_checksums(result)

    def test_no_cell_at_all_fails_loud(self) -> None:
        """An empty snapshot means the checksum action covered nothing, so fail loud."""
        with pytest.raises(AssertionError, match="covered no inference cell"):
            flatten_inference_engine_checksums({})

    def test_multi_rank_merged_with_rank_prefix(self) -> None:
        """Multiple ranks of one engine merge into one dict, prefixed by rank to avoid clobber."""
        result = {"cell-a": _tp2({"w": "r0"}, {"w": "r1"})}
        assert flatten_inference_engine_checksums(result) == {"cell-a": {"rank0/w": "r0", "rank1/w": "r1"}}

    def test_ranks_out_of_order_sorted_by_parallelism_rank(self) -> None:
        """Ranks arriving out of order (zmq) are sorted by parallelism rank deterministically."""
        out_of_order = {
            "cell-a": _engine_body(success=True, ranks=[_rank(1, {"w": "r1"}, size=2), _rank(0, {"w": "r0"}, size=2)])
        }
        in_order = {"cell-a": _tp2({"w": "r0"}, {"w": "r1"})}
        assert flatten_inference_engine_checksums(out_of_order) == flatten_inference_engine_checksums(in_order)

    def test_engine_failure_fails_loud(self) -> None:
        """An engine reporting success=False fails loud rather than silently dropping it."""
        result = {"cell-a": _engine_body(success=False, ranks=[_rank(0, {"w": "aaa"})])}
        with pytest.raises(AssertionError, match="reported failure"):
            flatten_inference_engine_checksums(result)

    def test_engine_without_ranks_fails_loud(self) -> None:
        """A success body with no ranks fails loud (nothing to compare)."""
        result = {"cell-a": _engine_body(success=True, ranks=None)}
        with pytest.raises(AssertionError, match="no ranks"):
            flatten_inference_engine_checksums(result)

    def test_multi_role_parallelism_info_collapses_to_one_gpu_rank(self) -> None:
        """Per-role parallelism_info (target + draft) sharing a GPU rank uses that single rank."""
        rank_info = {
            "checksums": {"w": "aaa"},
            "parallelism_info": [{"role": "target", "rank": 0, "size": 1}, {"role": "draft", "rank": 0, "size": 1}],
        }
        result = {"cell-a": _engine_body(success=True, ranks=[rank_info])}
        assert flatten_inference_engine_checksums(result) == {"cell-a": {"rank0/w": "aaa"}}

    def test_parallelism_info_rank_disagreement_fails_loud(self) -> None:
        """Roles reporting different GPU ranks for one engine rank is a contract violation."""
        rank_info = {
            "checksums": {"w": "aaa"},
            "parallelism_info": [{"role": "target", "rank": 0, "size": 2}, {"role": "draft", "rank": 1, "size": 2}],
        }
        result = {"cell-a": _engine_body(success=True, ranks=[rank_info])}
        with pytest.raises(AssertionError, match="disagree on the global rank"):
            flatten_inference_engine_checksums(result)

    def test_parallelism_info_world_size_disagreement_fails_loud(self) -> None:
        """Roles that disagree on the world size leave the rank set this response should cover undefined."""
        rank_info = {
            "checksums": {"w": "aaa"},
            "parallelism_info": [{"role": "target", "rank": 0, "size": 2}, {"role": "draft", "rank": 0, "size": 4}],
        }
        result = {"cell-a": _engine_body(success=True, ranks=[rank_info])}
        with pytest.raises(AssertionError, match="disagree on the global rank"):
            flatten_inference_engine_checksums(result)

    def test_a_rank_naming_no_parallelism_group_fails_loud(self) -> None:
        """Without a parallelism group nothing says which shard of the engine was hashed."""
        result = {"cell-a": _engine_body(success=True, ranks=[{"checksums": {"w": "aaa"}, "parallelism_info": []}])}
        with pytest.raises(AssertionError, match="names no parallelism group"):
            flatten_inference_engine_checksums(result)

    def test_a_rank_outside_its_world_fails_loud(self) -> None:
        """A rank that is not a member of the world it reports names no shard of this engine."""
        result = {"cell-a": _engine_body(success=True, ranks=[_rank(2, {"w": "aaa"}, size=2)])}
        with pytest.raises(AssertionError, match="names no shard"):
            flatten_inference_engine_checksums(result)

    def test_a_rank_that_hashed_no_tensor_fails_loud(self) -> None:
        """Emitting an empty tensor map would put a cell with no evidence into the audit."""
        result = {"cell-a": _engine_body(success=True, ranks=[_rank(0, {})])}
        with pytest.raises(AssertionError, match="hashed no tensor"):
            flatten_inference_engine_checksums(result)

    def test_one_empty_rank_of_a_multi_rank_engine_fails_loud(self) -> None:
        """A TP shard that reported nothing leaves a partial map that still looks like a complete one."""
        result = {"cell-a": _tp2({"w": "r0"}, {})}
        with pytest.raises(AssertionError, match="hashed no tensor"):
            flatten_inference_engine_checksums(result)


class TestFlattenRejectsInvalidDigests:
    def test_an_empty_digest_fails_loud(self) -> None:
        """An empty hash compares equal to every other empty one, so it would pass every audit it entered."""
        result = {"cell-a": _engine_body(success=True, ranks=[_rank(0, {"w": ""})])}
        with pytest.raises(AssertionError, match="instead of a tensor name and its hex digest"):
            flatten_inference_engine_checksums(result)

    def test_an_unnamed_tensor_fails_loud(self) -> None:
        """A hash filed under no name says nothing about which weight it belongs to."""
        result = {"cell-a": _engine_body(success=True, ranks=[_rank(0, {"": "aaa"})])}
        with pytest.raises(AssertionError, match="instead of a tensor name and its hex digest"):
            flatten_inference_engine_checksums(result)

    def test_a_non_string_digest_fails_loud(self) -> None:
        """A number or a null would be serialised into the event and read back as a value that looks legitimate."""
        result = {"cell-a": _engine_body(success=True, ranks=[_rank(0, {"w": None})])}
        with pytest.raises(AssertionError, match="instead of a tensor name and its hex digest"):
            flatten_inference_engine_checksums(result)

    def test_a_non_string_tensor_name_fails_loud(self) -> None:
        """A non-string key would compare against nothing the other cells of this version reported."""
        result = {"cell-a": _engine_body(success=True, ranks=[_rank(0, {0: "aaa"})])}
        with pytest.raises(AssertionError, match="instead of a tensor name and its hex digest"):
            flatten_inference_engine_checksums(result)

    def test_one_bad_entry_among_valid_ones_fails_loud(self) -> None:
        """The valid tensors beside it would otherwise make the cell look completely audited."""
        result = {"cell-a": _engine_body(success=True, ranks=[_rank(0, {"w": "aaa", "b": ""})])}
        with pytest.raises(AssertionError, match="instead of a tensor name and its hex digest"):
            flatten_inference_engine_checksums(result)

    def test_a_checksum_payload_that_is_not_a_map_fails_loud(self) -> None:
        """A list of hashes names no tensor, so nothing could line it up with another cell's answer."""
        result = {"cell-a": _engine_body(success=True, ranks=[_rank(0, ["aaa"])])}
        with pytest.raises(AssertionError, match="hashed no tensor"):
            flatten_inference_engine_checksums(result)


class TestFlattenRejectsIncompleteRankSets:
    def test_a_repeated_rank_with_identical_payloads_fails_loud(self) -> None:
        """Two answers from rank 0 are not a TP2 engine, however alike they look."""
        result = {"cell-a": _engine_body(success=True, ranks=[_rank(0, {"w": "r0"}, size=2)] * 2)}
        with pytest.raises(AssertionError, match=r"rank\(s\) \[0\] more than once"):
            flatten_inference_engine_checksums(result)

    def test_a_repeated_rank_with_conflicting_payloads_fails_loud(self) -> None:
        """Last-wins here would silently discard one shard's evidence in favour of another's."""
        result = {
            "cell-a": _engine_body(
                success=True, ranks=[_rank(0, {"w": "first"}, size=2), _rank(0, {"w": "second"}, size=2)]
            )
        }
        with pytest.raises(AssertionError, match=r"rank\(s\) \[0\] more than once"):
            flatten_inference_engine_checksums(result)

    def test_a_missing_rank_fails_loud(self) -> None:
        """Ranks 0 and 2 of a world of 3 must not pass as the whole engine."""
        result = {
            "cell-a": _engine_body(success=True, ranks=[_rank(0, {"w": "r0"}, size=3), _rank(2, {"w": "r2"}, size=3)])
        }
        with pytest.raises(AssertionError, match=r"missing \[1\]"):
            flatten_inference_engine_checksums(result)

    def test_ranks_of_different_world_sizes_fail_loud(self) -> None:
        """One response cannot describe two worlds, so what this audit covers would be unknown."""
        result = {
            "cell-a": _engine_body(success=True, ranks=[_rank(0, {"w": "r0"}, size=1), _rank(1, {"w": "r1"}, size=2)])
        }
        with pytest.raises(AssertionError, match="different world sizes"):
            flatten_inference_engine_checksums(result)
