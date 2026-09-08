import logging
from concurrent.futures import CancelledError, Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, NamedTuple

import torch

from miles.backends.training_utils.weight_update.inference_cell_health import InferenceCellHealth
from miles.backends.training_utils.weight_update.protocols.p2p_cell_executor import _CellWriteExecutor
from miles.backends.training_utils.weight_update.protocols.p2p_transfer_utils import RemoteWeightInfo
from miles.utils.test_utils.fault_hooks import (
    FaultHookName,
    RemoteInferenceTarget,
    WeightUpdateSpan,
    current_weight_update_span,
    reach_fault_hook,
)

logger = logging.getLogger(__name__)


class _P2PInferenceCellUpdater:
    def __init__(
        self,
        cell_id: str,
        transfer_engine: Any,
        health: InferenceCellHealth,
        transfer_timeout: float,
    ) -> None:
        self.cell_id = cell_id
        self._transfer_engine = transfer_engine
        self._health = health
        self._transfer_timeout = transfer_timeout
        self._executor = _CellWriteExecutor(cell_id)
        self._disposed = False
        self._peer_by_engine_rank: dict[int, RemoteWeightInfo] = {}
        self._pending_writes: list[Future] = []
        self._workers_hash: str | None = None

    @property
    def is_errored(self) -> bool:
        return self._health.is_errored(self.cell_id)

    @property
    def is_disposed(self) -> bool:
        return self._disposed

    @property
    def accepts_writes(self) -> bool:
        return not self._disposed and not self.is_errored

    @property
    def engine_ranks(self) -> tuple[int, ...]:
        return tuple(sorted(self._peer_by_engine_rank))

    def mark_errored(self, error: BaseException) -> None:
        self._health.mark_errored(self.cell_id, error)

    def bind_incarnation(self, workers_hash: str) -> None:
        self._workers_hash = workers_hash

    def remote_target_of(self, peer: RemoteWeightInfo) -> RemoteInferenceTarget | None:
        if self._workers_hash is None or peer.receiver_identity is None:
            return None
        assert peer.receiver_identity.session_id == peer.session_id, (
            f"[P2P-Shared] cell {self.cell_id} holds a receiver identity of session "
            f"{peer.receiver_identity.session_id} for the peer of session {peer.session_id}"
        )
        return RemoteInferenceTarget(
            cell_id=self.cell_id,
            workers_hash=self._workers_hash,
            receiver=peer.receiver_identity,
        )

    def add_peer(self, engine_rank: int, remote_weight_info: RemoteWeightInfo) -> None:
        assert (
            engine_rank not in self._peer_by_engine_rank
        ), f"[P2P-Shared] Engine rank {engine_rank} already registered for cell {self.cell_id}"
        self._peer_by_engine_rank[engine_rank] = remote_weight_info

    def submit_write(
        self, engine_rank: int, names: list[str], weight_memory_registry: dict[str, tuple[int, int, int]]
    ) -> Future | None:
        if not self.accepts_writes:
            return None
        peer = self._peer_by_engine_rank[engine_rank]
        span = current_weight_update_span()
        future = self._executor.submit(
            self._write_one_peer,
            peer,
            names,
            weight_memory_registry,
            span,
        )
        self._pending_writes.append(future)
        reach_fault_hook(
            FaultHookName.WEIGHT_UPDATE_AFTER_P2P_SUBMIT, remote_target=self.remote_target_of(peer), span=span
        )
        return future

    def wait_for_write(self, future: Future | None) -> None:
        if future is None:
            return
        self._collect_write(future)

    def wait_for_pending_writes(self) -> None:
        for future in list(self._pending_writes):
            self._collect_write(future)

    def take_unfinished_writes(self) -> list[Future]:
        unfinished, self._pending_writes = self._pending_writes, []
        return unfinished

    def dispose(self) -> _CellWriteExecutor | None:
        self._disposed = True
        return None if self._executor.close() else self._executor

    def _collect_write(self, future: Future) -> None:
        try:
            future.result(timeout=0.0 if self.is_errored else self._transfer_timeout)
        except FutureTimeoutError as error:
            self._abandon_write(future, error)
            return
        except CancelledError:
            self._forget_write(future)
            return
        except Exception as error:
            logger.exception(f"[P2P-Shared] a write to cell {self.cell_id} failed")
            self.mark_errored(error)
        self._forget_write(future)

    def _abandon_write(self, future: Future, error: BaseException) -> None:
        self.mark_errored(error)
        if future.cancel():
            self._forget_write(future)
            return
        logger.error(f"[P2P-Shared] a write to cell {self.cell_id} is still running after the transfer timeout")

    def _forget_write(self, future: Future) -> None:
        if future in self._pending_writes:
            self._pending_writes.remove(future)

    def _write_one_peer(
        self,
        remote_session: RemoteWeightInfo,
        names: list[str],
        weight_memory_registry: dict[str, tuple[int, int, int]],
        span: WeightUpdateSpan | None = None,
    ) -> None:
        """P2P write from shared CPU pinned buffers to a single remote session.

        Used by the parallelized submission path where each session within an
        engine rank is submitted as a separate task to this cell's write executor.
        """
        if not self.accepts_writes:
            logger.warning(f"[P2P-Shared] skipping a queued write to cell {self.cell_id}")
            return

        source_ptrs, source_lens = [], []
        valid_names = []

        for name in names:
            cpu_reg = weight_memory_registry.get(name)
            assert cpu_reg, f"the _weight_memory_registry of {name} failed"

            data_ptr, numel, ele_size = cpu_reg
            source_ptrs.append(data_ptr)
            source_lens.append(numel * ele_size)
            valid_names.append(name)

        if not source_ptrs:
            return

        session_id = remote_session.session_id
        target_ptrs = []
        for name, source_len in zip(valid_names, source_lens, strict=True):
            if name in remote_session.weights_info:
                location = remote_session.weights_info[name]
                target_len = location.numel * location.element_size
                assert target_len == source_len, (
                    f"[P2P-Shared] {name} spans {source_len} bytes here and {target_len} bytes on session "
                    f"{session_id}, so writing it would run past the target buffer"
                )
                target_ptrs.append(location.address)

        assert len(target_ptrs) == len(source_ptrs), (
            f"[P2P-Shared] Pointer count mismatch for session {session_id}, "
            f"source: {len(source_ptrs)}, target: {len(target_ptrs)}"
        )

        reach_fault_hook(
            FaultHookName.WEIGHT_UPDATE_BEFORE_P2P_WRITE,
            remote_target=self.remote_target_of(remote_session),
            span=span,
        )
        ret = self._transfer_engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
        if ret < 0:
            raise RuntimeError(f"[P2P-Shared] Transfer failed for session {session_id}, error: {ret}")


class TransferEngineMeta(NamedTuple):
    engine_rank: int
    model_replica: torch.nn.Module
    cell_updaters: list[_P2PInferenceCellUpdater]
