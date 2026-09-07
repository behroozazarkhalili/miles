from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.workers.cell_operations.base import CellTerminationNotConfirmedError, CellTerminationOutcome

_PUBLISHED_VERSION = 7


class _FakeCell:
    def __init__(
        self,
        cell_id: str,
        *,
        workers_hash: str | None = None,
        reported_weight_version: Any = _PUBLISHED_VERSION,
        checksum_body: Any = None,
        serving: bool = True,
    ) -> None:
        self.meta = SimpleNamespace(cell_id=cell_id, workers_hash=workers_hash or f"{cell_id}-gen0")
        self.is_serving = serving
        self.is_errored = False
        self._reported_weight_version = reported_weight_version
        self._checksum_body = checksum_body if checksum_body is not None else {"success": True, "cell": cell_id}
        self.check_weights_calls: list[dict[str, Any]] = []

    async def get_weight_version(self) -> Any:
        return self._reported_weight_version

    async def check_weights(self, action: str, allow_quant_error: bool, selector: str, skip_list: Any) -> Any:
        self.check_weights_calls.append(
            dict(action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list)
        )
        return self._checksum_body

    async def mark_errored(self) -> None:
        self.is_serving = False
        self.is_errored = True


class _HangingCell(_FakeCell):
    async def check_weights(self, action: str, allow_quant_error: bool, selector: str, skip_list: Any) -> Any:
        await asyncio.sleep(3600)
        raise AssertionError("a hung engine must not be waited for forever")


class _UnreachableCell(_FakeCell):
    async def check_weights(self, action: str, allow_quant_error: bool, selector: str, skip_list: Any) -> Any:
        raise httpx.ConnectError("connection refused")


class _UnreachableVersionCell(_FakeCell):
    async def get_weight_version(self) -> Any:
        raise httpx.ReadError("the engine went away mid-read")


class _FakeServer:
    def __init__(self, cells: list[_FakeCell], *, model_name: str = "default", update_weights: bool = True) -> None:
        self.server_cells = {cell.meta.cell_id: cell for cell in cells}
        self.model_name = model_name
        self.update_weights = update_weights


class _RecordingCellOperations:
    def __init__(self, outcome_of_cell_id: dict[str, Any] | None = None) -> None:
        self.outcome_of_cell_id = dict(outcome_of_cell_id or {})
        self.calls: list[tuple[str, str]] = []

    async def terminate_incarnation(self, *, cell_id: str, expected_workers_hash: str) -> CellTerminationOutcome:
        self.calls.append((cell_id, expected_workers_hash))
        outcome = self.outcome_of_cell_id.get(cell_id, CellTerminationOutcome.TERMINATED)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _make_controller(
    servers: dict[str, _FakeServer],
    *,
    cell_operations: _RecordingCellOperations | None = None,
    **arg_overrides: Any,
) -> InferenceController:
    controller = InferenceController(
        make_args(update_weight_engine_request_timeout=0.05, **arg_overrides),
        engine_provider=None,
        router_providers=[],
        cell_operations=cell_operations if cell_operations is not None else _RecordingCellOperations(),
    )
    controller.servers = servers
    return controller


def _published(*cells: _FakeCell) -> dict[str, str]:
    return {cell.meta.cell_id: cell.meta.workers_hash for cell in cells}


async def _snapshot(controller: InferenceController, published: dict[str, str], **kwargs: Any) -> dict[str, Any]:
    return await controller.snapshot_weight_checksums(
        expected_weight_version=_PUBLISHED_VERSION, published_cell_id_to_hashes=published, **kwargs
    )


class TestSnapshotWeightChecksums:
    async def test_each_cells_checksums_are_keyed_by_its_own_cell_id(self) -> None:
        """Positional results cannot survive a healed cell set, so identity and checksums are captured together."""
        cells = [_FakeCell("cell-b"), _FakeCell("cell-a")]
        controller = _make_controller({"default": _FakeServer(cells)})

        snapshot = await _snapshot(controller, _published(*cells))

        assert snapshot == {
            "cell-a": {"success": True, "cell": "cell-a"},
            "cell-b": {"success": True, "cell": "cell-b"},
        }

    async def test_a_cell_this_update_never_published_to_is_not_audited(self) -> None:
        """A cell that took no weights this time would answer for a version nobody gave it."""
        audited, untouched = _FakeCell("cell-a"), _FakeCell("cell-b")
        controller = _make_controller({"default": _FakeServer([audited, untouched])})

        snapshot = await _snapshot(controller, _published(audited))

        assert list(snapshot) == ["cell-a"]
        assert untouched.check_weights_calls == []

    async def test_a_cell_claiming_another_version_is_rejected(self) -> None:
        """A cell that missed the push still answers a checksum, which would be filed under a version it never took."""
        cells = [_FakeCell("cell-a"), _FakeCell("cell-b", reported_weight_version=6)]
        controller = _make_controller({"default": _FakeServer(cells)})

        with pytest.raises(AssertionError, match="cell-b serves weight version"):
            await _snapshot(controller, _published(*cells))

    async def test_a_version_reported_as_a_string_matches_the_published_integer(self) -> None:
        """sglang answers the version as a string, and refusing it would break every real run."""
        cell = _FakeCell("cell-a", reported_weight_version="7")
        controller = _make_controller({"default": _FakeServer([cell])})

        assert list(await _snapshot(controller, _published(cell))) == ["cell-a"]

    async def test_an_engine_that_never_took_a_version_is_rejected(self) -> None:
        """sglang's placeholder version means this engine was never updated, so its checksum proves nothing."""
        cell = _FakeCell("cell-a", reported_weight_version="default")
        controller = _make_controller({"default": _FakeServer([cell])})

        with pytest.raises(AssertionError, match="cell-a serves weight version"):
            await _snapshot(controller, _published(cell))

    async def test_the_checksum_request_covers_every_tensor(self) -> None:
        """A skip list or a partial selector would let a missing tensor pass the audit unnoticed."""
        cell = _FakeCell("cell-a")
        controller = _make_controller({"default": _FakeServer([cell])})

        await _snapshot(controller, _published(cell))

        assert cell.check_weights_calls == [
            dict(action="checksum", allow_quant_error=False, selector="all", skip_list=None)
        ]

    async def test_a_model_this_run_never_updates_is_refused(self) -> None:
        """A publication that names no updatable model belongs to weights this controller does not serve."""
        controller = _make_controller({"frozen": _FakeServer([_FakeCell("cell-a")], update_weights=False)})

        with pytest.raises(AssertionError, match="drives no updatable model"):
            await controller.snapshot_weight_checksums(
                expected_weight_version=_PUBLISHED_VERSION, published_cell_id_to_hashes={}
            )

    async def test_only_the_named_policys_cells_are_snapshotted(self) -> None:
        """One policy's update must not audit another policy's engines against its own version."""
        alpha = _FakeCell("cell-alpha")
        controller = _make_controller(
            {
                "alpha": _FakeServer([alpha], model_name="alpha"),
                "beta": _FakeServer([_FakeCell("cell-beta")], model_name="beta"),
            }
        )

        snapshot = await _snapshot(controller, _published(alpha), model_id="alpha")

        assert list(snapshot) == ["cell-alpha"]

    async def test_a_publication_naming_another_models_cell_is_rejected(self) -> None:
        """Another policy's engine holds unrelated weights, so it is no evidence about this publication."""
        alpha, beta = _FakeCell("cell-alpha"), _FakeCell("cell-beta")
        controller = _make_controller(
            {
                "alpha": _FakeServer([alpha], model_name="alpha"),
                "beta": _FakeServer([beta], model_name="beta"),
            }
        )

        with pytest.raises(AssertionError, match="serve another model of this run"):
            await _snapshot(controller, _published(alpha, beta), model_id="alpha")

        assert beta.check_weights_calls == []


class TestSnapshotWeightChecksumsIsBounded:
    async def test_a_hung_engine_is_given_up_on(self) -> None:
        """Neither engine request has a read timeout of its own, so a hung engine would stall the whole run."""
        hung, healthy = _HangingCell("cell-b"), _FakeCell("cell-a")
        controller = _make_controller({"default": _FakeServer([healthy, hung])}, ft_components=["rollout"])

        snapshot = await _snapshot(controller, _published(healthy, hung))

        assert list(snapshot) == ["cell-a"]
        assert hung.is_errored

    async def test_the_deadline_is_the_configured_engine_request_timeout(self) -> None:
        """A deadline of its own would ignore the budget the run was launched with."""
        cell = _HangingCell("cell-a")
        controller = _make_controller({"default": _FakeServer([cell])}, ft_components=["rollout"])
        controller.args.update_weight_engine_request_timeout = 3600.0

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(_snapshot(controller, _published(cell)), timeout=0.2)

    async def test_the_next_weight_update_cannot_open_while_the_snapshot_is_reading(self) -> None:
        """Letting an update start mid-snapshot would tear the version and the tensors it records apart."""
        cell = _HangingCell("cell-a")
        controller = _make_controller({"default": _FakeServer([cell])}, ft_components=["rollout"])
        controller.args.update_weight_engine_request_timeout = 3600.0

        snapshot = asyncio.ensure_future(_snapshot(controller, _published(cell)))
        await asyncio.sleep(0)
        update = asyncio.ensure_future(controller.start_update_weights())
        await asyncio.sleep(0.05)

        assert not update.done()
        snapshot.cancel()
        update.cancel()
        await asyncio.gather(snapshot, update, return_exceptions=True)


class TestSnapshotWeightChecksumsRetiresCellsItLost:
    async def test_a_cell_that_stopped_answering_is_taken_out_of_service(self) -> None:
        """A delayed fault can land between the push and the audit, and that cell must not keep serving."""
        lost, healthy = _UnreachableCell("cell-b"), _FakeCell("cell-a")
        operations = _RecordingCellOperations()
        controller = _make_controller(
            {"default": _FakeServer([healthy, lost])}, cell_operations=operations, ft_components=["rollout"]
        )

        snapshot = await _snapshot(controller, _published(healthy, lost))

        assert list(snapshot) == ["cell-a"]
        assert lost.is_errored and not healthy.is_errored
        assert operations.calls == [("cell-b", "cell-b-gen0")]

    async def test_a_cell_lost_while_answering_the_version_probe_is_retired_too(self) -> None:
        """The version probe is the first request the audit makes, so it is where a dead engine shows up."""
        lost, healthy = _UnreachableVersionCell("cell-b"), _FakeCell("cell-a")
        controller = _make_controller({"default": _FakeServer([healthy, lost])}, ft_components=["rollout"])

        snapshot = await _snapshot(controller, _published(healthy, lost))

        assert list(snapshot) == ["cell-a"]
        assert lost.is_errored

    async def test_an_already_gone_incarnation_counts_as_confirmed(self) -> None:
        """A process the manager can no longer find is exactly the positive answer this audit needs."""
        lost, healthy = _UnreachableCell("cell-b"), _FakeCell("cell-a")
        controller = _make_controller(
            {"default": _FakeServer([healthy, lost])},
            cell_operations=_RecordingCellOperations({"cell-b": CellTerminationOutcome.ALREADY_GONE}),
            ft_components=["rollout"],
        )

        assert list(await _snapshot(controller, _published(healthy, lost))) == ["cell-a"]

    async def test_a_run_without_rollout_fault_tolerance_propagates_the_failure(self) -> None:
        """Nothing would replace the retired cell, so the run must fail rather than quietly shrink."""
        lost = _UnreachableCell("cell-b")
        controller = _make_controller({"default": _FakeServer([_FakeCell("cell-a"), lost])}, ft_components=["train"])

        with pytest.raises(Exception, match="did not answer the weight checksum audit"):
            await _snapshot(controller, _published(lost))

        assert not lost.is_errored


class TestSnapshotWeightChecksumsDemandsTerminationEvidence:
    @staticmethod
    def _controller(outcome: Any) -> tuple[InferenceController, dict[str, str]]:
        lost, healthy = _UnreachableCell("cell-b"), _FakeCell("cell-a")
        controller = _make_controller(
            {"default": _FakeServer([healthy, lost])},
            cell_operations=_RecordingCellOperations({"cell-b": outcome}),
            ft_components=["rollout"],
        )
        return controller, _published(healthy, lost)

    async def test_a_stale_answer_does_not_prove_the_old_incarnation_died(self) -> None:
        """The manager left that process alone, so a healthy neighbour must not turn this into a passing audit."""
        controller, published = self._controller(CellTerminationOutcome.STALE)

        with pytest.raises(AssertionError, match="never confirmed gone"):
            await _snapshot(controller, published)

    async def test_a_termination_that_could_not_be_confirmed_fails_the_audit(self) -> None:
        """Being unable to prove death is the case this evidence exists for, not one to log and move past."""
        controller, published = self._controller(CellTerminationNotConfirmedError("the probe never answered"))

        with pytest.raises(AssertionError, match="never confirmed gone"):
            await _snapshot(controller, published)

    async def test_a_termination_that_raised_fails_the_audit(self) -> None:
        """The old helper only logged this, which let a run report a clean audit beside a live rogue engine."""
        controller, published = self._controller(RuntimeError("the worker manager refused"))

        with pytest.raises(AssertionError, match="never confirmed gone"):
            await _snapshot(controller, published)

    async def test_a_confirmed_retirement_lets_the_healthy_cells_be_audited(self) -> None:
        """Retirement is not meant to fail the run, only to be proven before the audit is believed."""
        controller, published = self._controller(CellTerminationOutcome.TERMINATED)

        assert list(await _snapshot(controller, published)) == ["cell-a"]


class TestSnapshotWeightChecksumsAccountsForEveryPublishedTarget:
    async def test_a_target_that_stopped_serving_must_be_confirmed_retired(self) -> None:
        """It still holds whatever half of this publication reached it, so it cannot just drop out of the audit."""
        gone, healthy = _FakeCell("cell-b", serving=False), _FakeCell("cell-a")
        operations = _RecordingCellOperations()
        controller = _make_controller(
            {"default": _FakeServer([healthy, gone])}, cell_operations=operations, ft_components=["rollout"]
        )

        snapshot = await _snapshot(controller, _published(healthy, gone))

        assert list(snapshot) == ["cell-a"]
        assert gone.check_weights_calls == []
        assert operations.calls == [("cell-b", "cell-b-gen0")]

    async def test_a_target_that_stopped_serving_and_was_not_confirmed_fails(self) -> None:
        """Silently skipping it is how a half-updated engine used to vanish behind a healthy neighbour."""
        gone, healthy = _FakeCell("cell-b", serving=False), _FakeCell("cell-a")
        controller = _make_controller(
            {"default": _FakeServer([healthy, gone])},
            cell_operations=_RecordingCellOperations({"cell-b": CellTerminationOutcome.STALE}),
            ft_components=["rollout"],
        )

        with pytest.raises(AssertionError, match="never confirmed gone"):
            await _snapshot(controller, _published(healthy, gone))

    async def test_a_target_that_came_back_under_a_new_generation_is_retired_by_its_old_hash(self) -> None:
        """The replacement never took this publication, and the conditional stop must not take it out."""
        replacement, healthy = _FakeCell("cell-b", workers_hash="cell-b-gen1"), _FakeCell("cell-a")
        operations = _RecordingCellOperations({"cell-b": CellTerminationOutcome.ALREADY_GONE})
        controller = _make_controller(
            {"default": _FakeServer([healthy, replacement])}, cell_operations=operations, ft_components=["rollout"]
        )

        snapshot = await _snapshot(controller, {**_published(healthy), "cell-b": "cell-b-gen0"})

        assert list(snapshot) == ["cell-a"]
        assert operations.calls == [("cell-b", "cell-b-gen0")]
        assert replacement.check_weights_calls == [] and not replacement.is_errored

    async def test_a_replacement_that_answers_stale_leaves_the_old_generation_unproven(self) -> None:
        """A new generation running is not evidence that the one this update wrote to has exited."""
        replacement, healthy = _FakeCell("cell-b", workers_hash="cell-b-gen1"), _FakeCell("cell-a")
        controller = _make_controller(
            {"default": _FakeServer([healthy, replacement])},
            cell_operations=_RecordingCellOperations({"cell-b": CellTerminationOutcome.STALE}),
            ft_components=["rollout"],
        )

        with pytest.raises(AssertionError, match="never confirmed gone"):
            await _snapshot(controller, {**_published(healthy), "cell-b": "cell-b-gen0"})

    async def test_a_target_the_controller_no_longer_knows_is_still_accounted_for(self) -> None:
        """A cell reconciled away between the push and the audit is not an audited cell."""
        healthy = _FakeCell("cell-a")
        operations = _RecordingCellOperations({"cell-gone": CellTerminationOutcome.ALREADY_GONE})
        controller = _make_controller(
            {"default": _FakeServer([healthy])}, cell_operations=operations, ft_components=["rollout"]
        )

        snapshot = await _snapshot(controller, {**_published(healthy), "cell-gone": "cell-gone-gen0"})

        assert list(snapshot) == ["cell-a"]
        assert operations.calls == [("cell-gone", "cell-gone-gen0")]

    async def test_a_publication_whose_targets_are_all_confirmed_gone_reads_nothing(self) -> None:
        """Every target this version reached has left, so there is no engine left to be wrong about it."""
        cell = _FakeCell("cell-a", serving=False)
        operations = _RecordingCellOperations()
        controller = _make_controller(
            {"default": _FakeServer([cell])}, cell_operations=operations, ft_components=["rollout"]
        )

        snapshot = await _snapshot(controller, _published(cell))

        assert snapshot == {}
        assert operations.calls == [("cell-a", "cell-a-gen0")]

    async def test_every_target_falling_silent_is_the_same_once_each_is_confirmed_gone(self) -> None:
        """A fleet that stopped answering mid-audit is retired, and retiring it all leaves nothing to record."""
        cell = _UnreachableCell("cell-a")
        operations = _RecordingCellOperations({"cell-a": CellTerminationOutcome.ALREADY_GONE})
        controller = _make_controller(
            {"default": _FakeServer([cell])}, cell_operations=operations, ft_components=["rollout"]
        )

        assert await _snapshot(controller, _published(cell)) == {}
        assert operations.calls == [("cell-a", "cell-a-gen0")]

    async def test_a_publication_with_no_evidence_and_no_confirmation_still_fails(self) -> None:
        """A process that may still hold half of this publication is not evidence that nothing serves it."""
        cell = _FakeCell("cell-a", serving=False)
        controller = _make_controller(
            {"default": _FakeServer([cell])},
            cell_operations=_RecordingCellOperations({"cell-a": CellTerminationOutcome.STALE}),
            ft_components=["rollout"],
        )

        with pytest.raises(AssertionError, match="never confirmed gone"):
            await _snapshot(controller, _published(cell))

    async def test_a_publication_that_names_no_target_at_all_fails_loud(self) -> None:
        """Only a version some cell actually serves is published, so naming none of them is a bug upstream."""
        controller = _make_controller({"default": _FakeServer([_FakeCell("cell-a")])}, ft_components=["rollout"])

        with pytest.raises(AssertionError, match="published to no cell at all"):
            await _snapshot(controller, {})

    async def test_a_run_without_rollout_fault_tolerance_refuses_to_retire_a_lost_target(self) -> None:
        """Nothing would replace it, so shrinking the fleet silently is worse than failing the update."""
        gone, healthy = _FakeCell("cell-b", serving=False), _FakeCell("cell-a")
        controller = _make_controller({"default": _FakeServer([healthy, gone])}, ft_components=["train"])

        with pytest.raises(AssertionError, match="tolerates no rollout fault"):
            await _snapshot(controller, _published(healthy, gone))


class TestSnapshotWeightChecksumsNeverSwallowsASemanticFailure:
    async def test_a_wrong_version_survives_a_transport_failure_beside_it(self) -> None:
        """Retiring the silent cell must not turn its neighbour's stale weights into a passing audit."""
        lost = _UnreachableCell("cell-b")
        stale = _FakeCell("cell-c", reported_weight_version=6)
        controller = _make_controller(
            {"default": _FakeServer([_FakeCell("cell-a"), lost, stale])}, ft_components=["rollout"]
        )

        with pytest.raises(AssertionError, match="cell-c serves weight version"):
            await _snapshot(controller, _published(lost, stale))

        assert lost.is_errored
