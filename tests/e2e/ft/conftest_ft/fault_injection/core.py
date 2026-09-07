# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import logging
import random
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

import requests

from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import BaseFaultForm, CellFaultForms
from tests.e2e.ft.conftest_ft.fault_injection.state import Event, EventLog, cell_type_of, cell_workers_hash
from tests.e2e.ft.conftest_ft.fault_injection.views import (
    PendingInjection,
    compute_injection_eligibility,
    compute_pending_injections,
    compute_successful_form_names,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS: float = 2.0
UNKNOWN_INJECTION_RESOLUTION_TIMEOUT_SECONDS: float = 900.0


class UnresolvedInjectionError(RuntimeError):
    pass


def _compute_next_injection_time(rng: random.Random, mean_interval_seconds: float) -> float:
    return time.monotonic() + rng.expovariate(1.0 / mean_interval_seconds)


def run_fault_injection_loop(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds_of_cell_type: dict[str, float],
    stop_event: threading.Event,
    event_log: EventLog,
    cell_fault_forms: CellFaultForms,
    get_virtual_cells: Callable[[], list[dict]] | None = None,
    injection_enabled: Callable[[], bool] | None = None,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    unknown_injection_timeout_seconds: float = UNKNOWN_INJECTION_RESOLUTION_TIMEOUT_SECONDS,
) -> None:
    rng = random.Random(seed)
    next_injection_time_of_cell_type: dict[str, float] = {
        cell_type: _compute_next_injection_time(rng, mean_interval_seconds)
        for cell_type, mean_interval_seconds in sorted(mean_interval_seconds_of_cell_type.items())
    }
    max_num_cells_of_cell_type: dict[str, int] = dict.fromkeys(next_injection_time_of_cell_type, 0)

    while not stop_event.is_set():
        if stop_event.wait(timeout=poll_interval_seconds):
            break

        cells = list_cells(base_url=base_url, cell_types=set(mean_interval_seconds_of_cell_type))
        if cells is None:
            continue
        if get_virtual_cells is not None:
            cells.extend(get_virtual_cells())

        # Record every poll so the post-run witnesses see the same stream the injector saw.
        event_log.observe(cells)

        if stop_event.is_set():
            break

        cells_of_type: dict[str, list[dict]] = {cell_type: [] for cell_type in next_injection_time_of_cell_type}
        for cell in cells:
            cells_of_type[cell_type_of(cell)].append(cell)
        for cell_type, kind_cells in cells_of_type.items():
            max_num_cells_of_cell_type[cell_type] = max(max_num_cells_of_cell_type[cell_type], len(kind_cells))

        now: float = time.monotonic()
        due_types = sorted(kind for kind, due_at in next_injection_time_of_cell_type.items() if now >= due_at)
        if not due_types:
            continue

        events = event_log.events
        pending = compute_pending_injections(events)
        _raise_if_an_unknown_injection_is_stuck(pending, timeout_seconds=unknown_injection_timeout_seconds)
        if pending:
            logger.info("Deferring injection: %s has not been observed serving a new generation yet", pending[0])
            continue

        eligible = compute_injection_eligibility(events)
        unready_types = [
            kind
            for kind in sorted(next_injection_time_of_cell_type)
            if not _kind_is_ready(
                cells_of_type[kind], expected_num_cells=max_num_cells_of_cell_type[kind], eligible=eligible
            )
        ]
        if unready_types:
            logger.info(
                "Deferring injection: cell kinds %s have not fully recovered, so a fault on any kind could "
                "compound a cascade (due %s, replicas %s, eligible %s)",
                unready_types,
                due_types,
                {kind: len(cells) for kind, cells in cells_of_type.items()},
                {
                    kind: len([c for c in cells if _cell_is_eligible(c, eligible=eligible)])
                    for kind, cells in cells_of_type.items()
                },
            )
            continue

        cell_type = rng.choice(due_types)
        target = rng.choice(cells_of_type[cell_type])
        cell_name = target["metadata"]["name"]
        workers_hash = cell_workers_hash(target)
        form = _draw_form(cell_fault_forms[cell_type], events=events, cell_type=cell_type, rng=rng)
        if injection_enabled is not None and not injection_enabled():
            continue
        try:
            form.inject(target, rng)
        except Exception:
            event_log.note_injection_attempt(
                cell_name=cell_name,
                form_name=form.name,
                succeeded=False,
                harmed=form.harms_cell,
                workers_hash=workers_hash,
            )
            logger.info("Failed to inject fault %s into %s", form.name, cell_name, exc_info=True)
            continue

        event_log.note_injection_attempt(
            cell_name=cell_name,
            form_name=form.name,
            succeeded=True,
            harmed=form.harms_cell,
            workers_hash=workers_hash,
        )
        next_injection_time_of_cell_type[cell_type] = _compute_next_injection_time(
            rng, mean_interval_seconds_of_cell_type[cell_type]
        )
        logger.info("Injected fault %s into %s (%s)", form.name, cell_name, workers_hash)


def _raise_if_an_unknown_injection_is_stuck(pending: list[PendingInjection], *, timeout_seconds: float) -> None:
    now = datetime.now(timezone.utc)
    for entry in pending:
        if entry.outcome_known:
            continue
        if (now - entry.injected_at).total_seconds() >= timeout_seconds:
            raise UnresolvedInjectionError(
                f"The injection into {entry.cell_name} ({entry.workers_hash}) failed to report an outcome and no "
                f"replacement generation has served in the {timeout_seconds:.0f}s since, so it is unknown whether "
                f"the fault landed; retrying blindly could take out a second replica"
            )


def _kind_is_ready(kind_cells: list[dict], *, expected_num_cells: int, eligible: set[tuple[str, str]]) -> bool:
    if not kind_cells or len(kind_cells) < expected_num_cells:
        return False
    if len(kind_cells) < 2:
        return False
    return all(_cell_is_eligible(cell, eligible=eligible) for cell in kind_cells)


def _cell_is_eligible(cell: dict, *, eligible: set[tuple[str, str]]) -> bool:
    return (cell["metadata"]["name"], cell_workers_hash(cell)) in eligible


def _draw_form(
    forms: list[BaseFaultForm], *, events: list[Event], cell_type: str, rng: random.Random
) -> BaseFaultForm:
    worked = compute_successful_form_names(events, cell_type=cell_type)
    unproven = [form for form in forms if form.name not in worked]
    return rng.choice(unproven or forms)


def list_cells(*, base_url: str, cell_types: set[str]) -> list[dict] | None:
    try:
        resp = requests.get(f"{base_url}/api/v1/cells", timeout=5)
        resp.raise_for_status()
        return [c for c in resp.json()["items"] if cell_type_of(c) in cell_types]
    except Exception:
        logger.info("Failed to list cells from api server", exc_info=True)
        return None
