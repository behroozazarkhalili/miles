import logging
from concurrent.futures import Future
from typing import Any, NamedTuple

import torch

from miles.backends.training_utils.weight_update.protocols.p2p_transfer_utils import (
    P2PTransferManager,
    RemoteWeightInfo,
)

logger = logging.getLogger(__name__)


class _P2PInferenceCellUpdater:
    def __init__(self, engine_ind: int, transfer_engine: Any, transfer_manager: P2PTransferManager) -> None:
        self.engine_ind = engine_ind
        self._transfer_engine = transfer_engine
        self._transfer_manager = transfer_manager
        self._peer_by_engine_rank: dict[int, RemoteWeightInfo] = {}

    def add_peer(self, engine_rank: int, remote_weight_info: RemoteWeightInfo) -> None:
        assert (
            engine_rank not in self._peer_by_engine_rank
        ), f"[P2P-Shared] Engine rank {engine_rank} already registered for engine {self.engine_ind}"
        self._peer_by_engine_rank[engine_rank] = remote_weight_info

    def submit_write(
        self, engine_rank: int, names: list[str], weight_memory_registry: dict[str, tuple[int, int, int]]
    ) -> Future:
        return self._transfer_manager.submit_returning_future(
            self._write_one_peer,
            self._peer_by_engine_rank[engine_rank],
            names,
            weight_memory_registry,
        )

    def _write_one_peer(
        self,
        remote_session: RemoteWeightInfo,
        names: list[str],
        weight_memory_registry: dict[str, tuple[int, int, int]],
    ) -> None:
        """P2P write from shared CPU pinned buffers to a single remote session.

        Used by the parallelized submission path where each session within an
        engine rank is submitted as a separate task to P2PTransferManager.
        """
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
        for name in valid_names:
            if name in remote_session.weights_info:
                target_ptrs.append(remote_session.weights_info[name].address)

        assert len(target_ptrs) == len(source_ptrs), (
            f"[P2P-Shared] Pointer count mismatch for session {session_id}, "
            f"source: {len(source_ptrs)}, target: {len(target_ptrs)}"
        )

        ret = self._transfer_engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
        if ret < 0:
            raise RuntimeError(f"[P2P-Shared] Transfer failed for session {session_id}, error: {ret}")


class TransferEngineMeta(NamedTuple):
    engine_rank: int
    model_replica: torch.nn.Module
    cell_updaters: list[_P2PInferenceCellUpdater]
