import contextlib
import random
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock, patch

from tests.e2e.ft.conftest_ft.fault_injection import core, fault_forms, state

from miles.ray.train.cell_monitor import compute_cell_status
from miles.ray.train.cell_state import CellState, StateAllocatedAlive, StateAllocatedUninitialized
from miles.utils.external_utils import command_utils
from miles.utils.ft_utils.api_server.models import Cell, CellMetadata, CellSpec, TriState
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.workers.types import ClusterBackend


def note_injected(log: state.EventLog, cell_name: str, *, workers_hash: str = "generation-0") -> None:
    log.note_injection_attempt(
        cell_name=cell_name,
        form_name="sigkill",
        succeeded=True,
        workers_hash=workers_hash,
    )


@contextlib.contextmanager
def patched_requests() -> Iterator[MagicMock]:
    # the loop lists cells through core and injects through fault_forms, so a mock on core alone
    # leaves every injection reaching the real network and timing out against a host nobody serves
    mock_requests = MagicMock()
    with patch.object(core, "requests", mock_requests), patch.object(fault_forms, "requests", mock_requests):
        yield mock_requests


NAMESPACE = "miles-e2e"
RUN_ID = "abc123"


def cell(
    name: str,
    *,
    healthy: bool,
    cell_type: str = "actor",
    phase: str = "Running",
    serving: bool = True,
    allocated: bool = True,
    workers_hash: str = "generation-0",
    health_reason: str | None = None,
) -> dict:
    conditions: list[dict] = [{"type": "Allocated", "status": "True" if allocated else "False"}]
    conditions.append({"type": "Healthy", "status": "True" if healthy else "False", "reason": health_reason})
    if cell_type == "rollout":
        conditions.append({"type": "Serving", "status": "True" if serving else "False"})
    return {
        "metadata": {
            "name": name,
            "labels": {"miles.io/cell-type": cell_type, "miles.io/workers-hash": workers_hash},
        },
        "status": {"phase": phase, "conditions": conditions, "workers_hash": workers_hash},
    }


def paused_cell(name: str, *, cell_type: str = "rollout", reason: str, workers_hash: str = "generation-0") -> dict:
    built = cell(name, healthy=False, cell_type=cell_type, workers_hash=workers_hash, health_reason=reason)
    for condition in built["status"]["conditions"]:
        if condition["type"] == "Healthy":
            condition["status"] = "Unknown"
    return built


def trainer_cell_of_state(
    name: str,
    *,
    cell_state: CellState,
    health_status: TriState,
    workers_hash: str = "generation-0",
) -> dict:
    return Cell(
        metadata=CellMetadata(
            name=name, labels={"miles.io/cell-type": state.ACTOR_CELL_TYPE, "miles.io/cell-id": name}
        ),
        spec=CellSpec(),
        status=compute_cell_status(cell_state, health_status, workers_hash=workers_hash),
    ).model_dump(mode="json")


def uninitialized_trainer_cell(name: str, *, workers_hash: str = "generation-0") -> dict:
    return trainer_cell_of_state(
        name,
        cell_state=StateAllocatedUninitialized(worker_handles=[]),
        health_status=TriState.UNKNOWN,
        workers_hash=workers_hash,
    )


def initialized_trainer_cell(
    name: str, *, workers_hash: str = "generation-0", health_status: TriState = TriState.TRUE
) -> dict:
    return trainer_cell_of_state(
        name,
        cell_state=StateAllocatedAlive(
            worker_handles=[],
            indep_dp_info=IndepDPInfo(
                cell_index=0, num_cells=2, alive_rank=0, alive_size=2, quorum_id=1, alive_cell_indices=[0, 1]
            ),
        ),
        health_status=health_status,
        workers_hash=workers_hash,
    )


def names(cells: list[dict]) -> set[str]:
    return {c["metadata"]["name"] for c in cells}


def mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


SERVING = state.ObservedCellState.SERVING
RUNNING_NOT_SERVING = state.ObservedCellState.RUNNING_NOT_SERVING
PENDING = state.ObservedCellState.PENDING
SUSPENDED = state.ObservedCellState.SUSPENDED


def staged(
    name: str,
    cell_state: state.ObservedCellState,
    *,
    cell_type: str = "rollout",
    workers_hash: str = "generation-0",
) -> dict:
    phase = {
        SUSPENDED: "Suspended",
        PENDING: "Pending",
        RUNNING_NOT_SERVING: "Running",
        SERVING: "Running",
    }[cell_state]
    return cell(
        name,
        healthy=phase == "Running",
        cell_type=cell_type,
        phase=phase,
        serving=cell_state is SERVING,
        allocated=cell_state is not SUSPENDED,
        workers_hash=workers_hash,
    )


def log_of(
    cell_states: list[state.ObservedCellState], *, inject_before: dict[int, int] | None = None
) -> state.EventLog:
    log = state.EventLog()
    generation = 0
    for index, cell_state in enumerate(cell_states):
        for _ in range((inject_before or {}).get(index, 0)):
            note_injected(log, "rollout-engine-0", workers_hash=f"generation-{generation}")
            generation += 1
        log.observe([staged("rollout-engine-0", cell_state, workers_hash=f"generation-{generation}")])
    return log


def typed_cell(
    name: str,
    cell_type: str,
    *,
    healthy: bool = True,
    serving: bool = True,
    workers_hash: str = "generation-0",
    health_reason: str | None = None,
) -> dict:
    return cell(
        name,
        healthy=healthy,
        cell_type=cell_type,
        serving=serving,
        workers_hash=workers_hash,
        health_reason=health_reason,
    )


def config_of(backend: ClusterBackend, *, namespace: str = NAMESPACE) -> command_utils.ExecuteTrainConfig:
    return command_utils.ExecuteTrainConfig(cluster_backend=backend, namespace=namespace, run_id=RUN_ID)


def api_server_fault_forms() -> fault_forms.CellFaultForms:
    return fault_forms.create_cell_fault_forms(base_url="http://control", config=config_of(ClusterBackend.RAY))


class StubFaultForm(fault_forms.BaseFaultForm):
    def __init__(self, form_name: str, on_inject: Callable[[dict, random.Random], None]) -> None:
        self._name = form_name
        self._on_inject = on_inject

    @property
    def name(self) -> str:
        return self._name

    def inject(self, cell: dict, rng: random.Random) -> None:
        self._on_inject(cell, rng)


def fixed_fault_forms(forms: list[fault_forms.BaseFaultForm]) -> fault_forms.CellFaultForms:
    return {fault_forms.ACTOR_CELL_TYPE: forms, fault_forms.ROLLOUT_CELL_TYPE: forms}


def intervals(cell_types: tuple[str, ...], mean_interval_seconds: float) -> dict[str, float]:
    return {cell_type: mean_interval_seconds for cell_type in cell_types}
