# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import dataclasses
from datetime import datetime
from typing import Literal

from tests.e2e.ft.conftest_ft.fault_injection.state import (
    Event,
    HookArmEvent,
    HookArmRefusedEvent,
    HookFireEvent,
    InjectionEvent,
    ObservationsEvent,
    ObservedCellState,
    cell_info_is_in_service,
    cell_info_is_paused_for_weight_update,
)

from miles.utils.test_utils.fault_hooks import FaultHookTarget


def compute_num_injections(events: list[Event], *, cell_type: str | None = None, harmed_only: bool = True) -> int:
    return len(compute_injected_cell_names(events, cell_type=cell_type, harmed_only=harmed_only))


def compute_injected_cell_names(
    events: list[Event], *, cell_type: str | None = None, harmed_only: bool = True
) -> list[str]:
    names = [
        name
        for name, cell_events in _compute_matching_cell_events(
            events, cell_type=cell_type, harmed_only=harmed_only
        ).items()
        for one in cell_events
        if one.kind == "injected"
    ]
    return names + [
        harm.victim_cell_name
        for harm in compute_hook_harms(events)
        if harm.delivered and (cell_type is None or harm.cell_type == cell_type)
    ]


def compute_num_successful_injections_of_form(events: list[Event], *, form_name: str) -> int:
    return len(
        [
            event
            for event in events
            if isinstance(event, InjectionEvent) and event.succeeded and event.form_name == form_name
        ]
    )


def compute_cells_not_serving_after_injection(events: list[Event], *, cell_type: str) -> dict[str, list[str]]:
    cell_type_of_name = _compute_cell_type_of_name(events)
    observed_states = compute_states_of_cell_name(events)
    return {
        entry.cell_name: [one.value for one in observed_states.get(entry.cell_name, [])]
        for entry in compute_pending_injections(events)
        if cell_type_of_name.get(entry.cell_name) == cell_type
    }


@dataclasses.dataclass(frozen=True)
class PendingInjection:
    cell_name: str
    workers_hash: str
    injected_at: datetime
    outcome_known: bool
    request_id: str | None = None


def compute_pending_injections(events: list[Event]) -> list[PendingInjection]:
    pending: dict[str, PendingInjection] = {}
    for event in events:
        if isinstance(event, InjectionEvent):
            if event.harmed:
                pending[event.cell_name] = PendingInjection(
                    cell_name=event.cell_name,
                    workers_hash=event.workers_hash,
                    injected_at=event.timestamp,
                    outcome_known=event.succeeded,
                )
            continue
        if not isinstance(event, ObservationsEvent):
            continue
        for name, entry in list(pending.items()):
            info = event.cell_infos.get(name)
            if info is not None and info.workers_hash != entry.workers_hash and cell_info_is_in_service(info):
                del pending[name]

    entries = list(pending.values()) + _compute_pending_hook_injections(events)
    return sorted(entries, key=lambda entry: entry.injected_at)


def _compute_pending_hook_injections(events: list[Event]) -> list[PendingInjection]:
    pending: list[PendingInjection] = []
    for harm in compute_hook_harms(events):
        if harm.completed or harm.resolved_without_harm:
            continue
        if harm.delivered:
            pending.append(
                PendingInjection(
                    cell_name=harm.victim_cell_name,
                    workers_hash=harm.victim_workers_hash,
                    injected_at=harm.armed_at,
                    outcome_known=True,
                    request_id=harm.request_id,
                )
            )
            continue
        pending.append(
            PendingInjection(
                cell_name=harm.source_cell_name,
                workers_hash=harm.source_workers_hash,
                injected_at=harm.armed_at,
                outcome_known=False,
                request_id=harm.request_id,
            )
        )
    return pending


def compute_injection_eligibility(events: list[Event]) -> set[tuple[str, str]]:
    eligible_hash_of_name: dict[str, str] = {}
    for event in events:
        if not isinstance(event, ObservationsEvent):
            continue
        for name in [name for name in eligible_hash_of_name if name not in event.cell_infos]:
            del eligible_hash_of_name[name]
        for name, info in event.cell_infos.items():
            if cell_info_is_in_service(info):
                eligible_hash_of_name[name] = info.workers_hash
            elif not (
                cell_info_is_paused_for_weight_update(info) and eligible_hash_of_name.get(name) == info.workers_hash
            ):
                eligible_hash_of_name.pop(name, None)
    return {(name, workers_hash) for name, workers_hash in eligible_hash_of_name.items()}


def compute_successful_form_names(events: list[Event], *, cell_type: str) -> set[str]:
    cell_type_of_name = _compute_cell_type_of_name(events)
    worked = {
        event.form_name
        for event in events
        if isinstance(event, InjectionEvent)
        and event.succeeded
        and cell_type_of_name.get(event.cell_name) == cell_type
    }
    return worked | {
        harm.form_name for harm in compute_hook_harms(events) if harm.completed and harm.cell_type == cell_type
    }


def compute_forms_drawn_without_success(events: list[Event]) -> list[tuple[str, str]]:
    cell_type_of_name = _compute_cell_type_of_name(events)
    drawn: set[tuple[str, str]] = set()
    worked: set[tuple[str, str]] = set()
    for event in events:
        if not isinstance(event, InjectionEvent):
            continue
        key = (cell_type_of_name.get(event.cell_name, ""), event.form_name)
        drawn.add(key)
        if event.succeeded:
            worked.add(key)
    for harm in compute_hook_harms(events):
        drawn.add((harm.cell_type, harm.form_name))
        if harm.completed:
            worked.add((harm.cell_type, harm.form_name))
    return sorted(drawn - worked)


def compute_injection_times(events: list[Event], *, cell_type: str | None = None) -> list[datetime]:
    cell_type_of_name = _compute_cell_type_of_name(events)
    return [
        event.timestamp
        for event in events
        if isinstance(event, InjectionEvent)
        and event.succeeded
        and (cell_type is None or cell_type_of_name.get(event.cell_name) == cell_type)
    ]


@dataclasses.dataclass(frozen=True)
class HookHarm:
    request_id: str
    form_name: str
    cell_type: str
    hook: str
    mode: str
    target: str
    delay_ms: int
    source_cell_name: str
    source_workers_hash: str
    source_cell_index: int
    source_rank_within_cell: int
    armed_at: datetime
    acknowledged: bool
    fire: HookFireEvent | None
    victim_cell_name: str | None
    victim_workers_hash: str | None
    recovered: bool
    refused_because: str | None
    harmless_because: str | None

    @property
    def delivered(self) -> bool:
        return self.fire is not None and self.fire.delivered

    @property
    def completed(self) -> bool:
        return self.delivered and self.recovered

    @property
    def resolved_without_harm(self) -> bool:
        return self.refused_because is not None or self.harmless_because is not None

    @property
    def resolved(self) -> bool:
        return self.completed or self.resolved_without_harm


def compute_hook_harms(events: list[Event]) -> list[HookHarm]:
    harms: dict[str, HookHarm] = {}
    for event in events:
        if isinstance(event, HookArmEvent):
            _absorb_hook_arm(harms, event)
        elif isinstance(event, HookArmRefusedEvent):
            _absorb_hook_arm_refusal(harms, event)
        elif isinstance(event, HookFireEvent):
            _absorb_hook_fire(harms, event)
        elif isinstance(event, ObservationsEvent):
            _absorb_hook_recovery(harms, event)
    return list(harms.values())


def compute_unresolved_hook_harms(events: list[Event]) -> list[HookHarm]:
    return [harm for harm in compute_hook_harms(events) if not harm.resolved]


def _absorb_hook_arm(harms: dict[str, HookHarm], event: HookArmEvent) -> None:
    if (harm := harms.get(event.request_id)) is not None:
        if event.acknowledged:
            assert harm.refused_because is None, (
                f"request {event.request_id} was refused before anything was armed ({harm.refused_because}), and "
                "yet the api server later acknowledged it, so one of the two records is wrong about what this run did"
            )
            harms[event.request_id] = dataclasses.replace(harm, acknowledged=True)
        return
    harms[event.request_id] = HookHarm(
        request_id=event.request_id,
        form_name=event.form_name,
        cell_type=event.cell_type,
        hook=event.hook,
        mode=event.mode,
        target=event.target,
        delay_ms=event.delay_ms,
        source_cell_name=event.source_cell_name,
        source_workers_hash=event.source_workers_hash,
        source_cell_index=event.source_cell_index,
        source_rank_within_cell=event.source_rank_within_cell,
        armed_at=event.timestamp,
        acknowledged=event.acknowledged,
        fire=None,
        victim_cell_name=None,
        victim_workers_hash=None,
        recovered=False,
        refused_because=None,
        harmless_because=None,
    )


def _absorb_hook_arm_refusal(harms: dict[str, HookHarm], event: HookArmRefusedEvent) -> None:
    if (harm := harms.get(event.request_id)) is None:
        return
    assert not harm.acknowledged, (
        f"request {event.request_id} was acknowledged as armed and then recorded as refused before anything was "
        "armed, so one of the two records is wrong about what this run did"
    )
    assert not harm.delivered, (
        f"request {event.request_id} delivered a fault and then was recorded as refused before anything was armed, "
        "so one of the two records is wrong about what this run did"
    )
    harms[event.request_id] = dataclasses.replace(harm, refused_because=event.refused_because)


def _absorb_hook_fire(harms: dict[str, HookHarm], event: HookFireEvent) -> None:
    harm = harms.get(event.request_id)
    if harm is None:
        return
    if harm.fire is not None:
        assert (
            harm.fire == event
        ), f"request {event.request_id} has conflicting fault hook fires: {harm.fire} and {event}"
        return
    assert not (event.delivered and harm.refused_because is not None), (
        f"request {event.request_id} was refused before it armed anything ({harm.refused_because}), and yet a "
        f"delivered fault hook fire names it, so one of the two records is wrong about what this run did"
    )
    assert not (event.delivered and event.harmless_because is not None), (
        f"request {event.request_id} was recorded as both delivered and harmless, so the fire does not have one "
        "consistent outcome"
    )
    victim_cell_name, victim_workers_hash = _compute_hook_victim(harm, event)
    harms[event.request_id] = dataclasses.replace(
        harm,
        fire=event,
        victim_cell_name=victim_cell_name,
        victim_workers_hash=victim_workers_hash,
        harmless_because=event.harmless_because,
    )


def _absorb_hook_recovery(harms: dict[str, HookHarm], event: ObservationsEvent) -> None:
    for request_id, harm in list(harms.items()):
        if harm.recovered or harm.victim_cell_name is None:
            continue
        info = event.cell_infos.get(harm.victim_cell_name)
        if info is not None and info.workers_hash != harm.victim_workers_hash and cell_info_is_in_service(info):
            harms[request_id] = dataclasses.replace(harm, recovered=True)


def _compute_hook_victim(harm: HookHarm, event: HookFireEvent) -> tuple[str | None, str | None]:
    if not event.delivered:
        return None, None
    if event.target == FaultHookTarget.LOCAL.value:
        return harm.source_cell_name, harm.source_workers_hash
    return event.victim_cell_name, event.victim_workers_hash


def compute_states_of_cell_name(events: list[Event]) -> dict[str, list[ObservedCellState]]:
    return {
        name: states
        for name, cell_events in _compute_cell_events(events).items()
        if (states := _compute_distinct_states(cell_events))
    }


@dataclasses.dataclass(frozen=True)
class _CellEvent:
    kind: Literal["injected", "observed"]
    state: ObservedCellState | None = None


def _compute_cell_events(events: list[Event], *, harmed_only: bool = True) -> dict[str, list[_CellEvent]]:
    cell_events_of_name: dict[str, list[_CellEvent]] = {}
    for event in events:
        if isinstance(event, InjectionEvent):
            if event.succeeded and (event.harmed or not harmed_only):
                cell_events_of_name.setdefault(event.cell_name, []).append(_CellEvent(kind="injected"))
            continue
        if not isinstance(event, ObservationsEvent):
            continue
        for name, info in event.cell_infos.items():
            cell_events_of_name.setdefault(name, []).append(_CellEvent(kind="observed", state=info.state))
    return cell_events_of_name


def _compute_matching_cell_events(
    events: list[Event], *, cell_type: str | None, harmed_only: bool
) -> dict[str, list[_CellEvent]]:
    cell_events_of_name = _compute_cell_events(events, harmed_only=harmed_only)
    if cell_type is None:
        return cell_events_of_name
    cell_type_of_name = _compute_cell_type_of_name(events)
    return {
        name: cell_events
        for name, cell_events in cell_events_of_name.items()
        if cell_type_of_name.get(name) == cell_type
    }


def _compute_cell_type_of_name(events: list[Event]) -> dict[str, str]:
    cell_type_of_name: dict[str, str] = {}
    for event in events:
        if isinstance(event, ObservationsEvent):
            cell_type_of_name.update({name: info.cell_type for name, info in event.cell_infos.items()})
    return cell_type_of_name


def _compute_distinct_states(events: list[_CellEvent]) -> list[ObservedCellState]:
    states: list[ObservedCellState] = []
    for event in events:
        if event.kind == "observed" and event.state is not None and (not states or states[-1] != event.state):
            states.append(event.state)
    return states
