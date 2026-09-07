import random
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from tests.e2e.ft.conftest_ft.fault_injection import core, fault_forms, state, views
from tests.fast.e2e.ft.fault_injection.utils import (
    StubFaultForm,
    api_server_fault_forms,
    cell,
    fixed_fault_forms,
    initialized_trainer_cell,
    intervals,
    mock_response,
    patched_requests,
    paused_cell,
    typed_cell,
    uninitialized_trainer_cell,
)

from miles.ray.rollout.inference_controller import HEALTH_PAUSED_FOR_OFFLOAD, HEALTH_PAUSED_FOR_WEIGHT_UPDATE


def _run_injection_loop(
    *,
    fake_get,
    fake_post=None,
    cell_types: tuple[str, ...] = ("actor", "rollout"),
    mean_interval_seconds_of_cell_type: dict[str, float] | None = None,
    event_log: state.EventLog | None = None,
    cell_fault_forms: fault_forms.CellFaultForms | None = None,
    get_virtual_cells: Callable[[], list[dict]] | None = None,
    injection_enabled: Callable[[], bool] | None = None,
    unknown_injection_timeout_seconds: float = 1e9,
    stop_event: threading.Event,
) -> None:
    with patched_requests() as mock_requests:
        mock_requests.get.side_effect = fake_get
        if fake_post is not None:
            mock_requests.post.side_effect = fake_post
        core.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=(
                mean_interval_seconds_of_cell_type
                if mean_interval_seconds_of_cell_type is not None
                else intervals(cell_types, 1e-9)
            ),
            stop_event=stop_event,
            event_log=event_log or state.EventLog(),
            cell_fault_forms=cell_fault_forms or api_server_fault_forms(),
            get_virtual_cells=get_virtual_cells,
            injection_enabled=injection_enabled,
            poll_interval_seconds=1e-6,
            unknown_injection_timeout_seconds=unknown_injection_timeout_seconds,
        )


def _name_from_inject_url(url: str) -> str:
    return url.rsplit("/cells/", 1)[1].split("/")[0]


def _run_typed_injection_loop(cells: list[dict], *, cell_types: tuple[str, ...], num_polls: int = 8) -> list[str]:
    injected: list[str] = []
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= num_polls:
            stop_event.set()
        return mock_response({"items": cells})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injected.append(_name_from_inject_url(url))
        return mock_response({})

    _run_injection_loop(fake_get=fake_get, fake_post=fake_post, cell_types=cell_types, stop_event=stop_event)

    return injected


class TestOneOutstandingHarmAtATime:
    """A second kill before the first victim is proven back would leave nobody able to attribute the damage."""

    def test_no_second_injection_before_a_new_generation_serves(self) -> None:
        """The api server reports a just-killed cell healthy for a while, and its hash is the only honest signal."""
        injected: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 20:
                stop_event.set()
            return mock_response({"items": [cell(name, healthy=True) for name in ("actor-0", "actor-1")]})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            injected.append(_name_from_inject_url(url))
            return mock_response({})

        _run_injection_loop(fake_get=fake_get, fake_post=fake_post, cell_types=("actor",), stop_event=stop_event)

        assert injected == injected[:1], injected

    def test_a_replacement_generation_that_serves_unlocks_the_next_injection(self) -> None:
        """Once the victim really is back the soak has to keep crashing, or it stops testing anything."""
        injected: list[str] = []
        stop_event = threading.Event()
        generation = {"actor-0": 0, "actor-1": 0}
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if len(injected) >= 2 or polls["n"] >= 60:
                stop_event.set()
            return mock_response(
                {
                    "items": [
                        cell(name, healthy=True, workers_hash=f"generation-{gen}") for name, gen in generation.items()
                    ]
                }
            )

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            name = _name_from_inject_url(url)
            injected.append(name)
            generation[name] += 1
            return mock_response({})

        _run_injection_loop(fake_get=fake_get, fake_post=fake_post, cell_types=("actor",), stop_event=stop_event)

        assert len(injected) >= 2, injected

    def test_the_same_generation_reading_healthy_again_does_not_unlock(self) -> None:
        """A stale Healthy poll of the old incarnation is exactly what the old time-based gate mistook for recovery."""
        stop_event = threading.Event()
        polls = {"n": 0}
        log = state.EventLog()

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 20:
                stop_event.set()
            return mock_response({"items": [cell(name, healthy=True) for name in ("actor-0", "actor-1")]})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            return mock_response({})

        _run_injection_loop(
            fake_get=fake_get, fake_post=fake_post, cell_types=("actor",), event_log=log, stop_event=stop_event
        )

        assert views.compute_num_injections(log.events, cell_type="actor") == 1

    def test_one_kind_recovering_holds_back_the_other_kind_too(self) -> None:
        """A trainer loss cascades into its assigned engines, and harming those as well hides the cascade."""
        injected: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}
        cells = [
            typed_cell("actor-0", "actor"),
            typed_cell("actor-1", "actor"),
            typed_cell("rollout-engine-0", "rollout"),
            typed_cell("rollout-engine-1", "rollout"),
        ]

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 25:
                stop_event.set()
            return mock_response({"items": cells})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            injected.append(_name_from_inject_url(url))
            return mock_response({})

        _run_injection_loop(
            fake_get=fake_get, fake_post=fake_post, cell_types=("actor", "rollout"), stop_event=stop_event
        )

        assert len(injected) == 1, injected


class TestKindReadiness:
    """A kind is ready only when every replica it ever showed is a qualified, in-service spare."""

    def test_a_kind_with_a_cell_that_is_not_injectable_is_not_ready(self) -> None:
        """Injecting while one replica is mid-relaunch is how a kind loses its last live member."""
        injected = _run_typed_injection_loop(
            [
                typed_cell("rollout-engine-0", "rollout"),
                typed_cell("rollout-engine-1", "rollout", serving=False),
            ],
            cell_types=("rollout",),
        )

        assert injected == []

    def test_two_injectable_engines_still_leave_one_of_them_injectable(self) -> None:
        """The readiness rule must not block the case it was never meant to block."""
        injected = _run_typed_injection_loop(
            [typed_cell("rollout-engine-0", "rollout"), typed_cell("rollout-engine-1", "rollout")],
            cell_types=("rollout",),
        )

        assert injected

    def test_a_generation_first_seen_paused_is_not_a_spare(self) -> None:
        """A replacement appearing mid weight update was never probed, so the kind is still recovering."""
        injected = _run_typed_injection_loop(
            [
                typed_cell("rollout-engine-0", "rollout"),
                paused_cell("rollout-engine-1", reason=HEALTH_PAUSED_FOR_WEIGHT_UPDATE, workers_hash="generation-9"),
            ],
            cell_types=("rollout",),
        )

        assert injected == []

    def test_a_kind_verified_before_the_pause_is_still_injectable(self) -> None:
        """A crash landing mid weight update is the case this suite exists to cover."""
        injected: list[tuple[int, str]] = []
        stop_event = threading.Event()
        polls = {"n": 0}
        healthy = [typed_cell(f"rollout-engine-{i}", "rollout") for i in range(2)]
        paused = [
            paused_cell(f"rollout-engine-{i}", reason=HEALTH_PAUSED_FOR_WEIGHT_UPDATE, workers_hash="generation-0")
            for i in range(2)
        ]

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 8:
                stop_event.set()
            return mock_response({"items": healthy if polls["n"] == 1 else paused})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            injected.append((polls["n"], _name_from_inject_url(url)))
            return mock_response({})

        _run_injection_loop(
            fake_get=fake_get,
            fake_post=fake_post,
            cell_types=("rollout",),
            injection_enabled=lambda: polls["n"] > 1,
            stop_event=stop_event,
        )

        assert injected, injected
        assert all(poll > 1 for poll, _name in injected), injected

    def test_an_offloaded_kind_is_never_injectable(self) -> None:
        """Colocate offload hands the GPUs to the trainer, and killing the engine there kills the wrong process."""
        injected = _run_typed_injection_loop(
            [
                paused_cell("rollout-engine-0", reason=HEALTH_PAUSED_FOR_OFFLOAD),
                paused_cell("rollout-engine-1", reason=HEALTH_PAUSED_FOR_OFFLOAD),
            ],
            cell_types=("rollout",),
        )

        assert injected == []

    def test_a_vanished_replica_blocks_its_kind_even_when_the_survivors_serve(self) -> None:
        """A killed pod can disappear from the listing entirely, which must read as still recovering."""
        injected: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}
        all_names = ("actor-0", "actor-1", "actor-2")

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 20:
                stop_event.set()
            names = [n for n in all_names if n not in injected] if injected else list(all_names)
            return mock_response({"items": [cell(n, healthy=True) for n in names]})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            injected.append(_name_from_inject_url(url))
            return mock_response({})

        _run_injection_loop(fake_get=fake_get, fake_post=fake_post, cell_types=("actor",), stop_event=stop_event)

        assert len(injected) == 1, injected


class TestEveryKindMustHaveRecovered:
    """A trainer loss cascades into its engines, so one kind reading healthy is not enough to harm again."""

    def test_a_kind_short_of_a_replica_blocks_the_other_kind_too(self) -> None:
        """The engines a lost trainer was writing to are still being rebuilt, and a second fault compounds it."""
        injected = _run_typed_injection_loop(
            [
                typed_cell("actor-0", "actor"),
                typed_cell("actor-1", "actor"),
                typed_cell("rollout-engine-0", "rollout"),
            ],
            cell_types=("actor", "rollout"),
        )

        assert injected == []

    def test_a_kind_still_relaunching_blocks_the_recovered_kind(self) -> None:
        """The trainer comes back first; harming it again while its targets rebuild is the cascade to avoid."""
        injected = _run_typed_injection_loop(
            [
                typed_cell("actor-0", "actor"),
                typed_cell("actor-1", "actor"),
                typed_cell("rollout-engine-0", "rollout"),
                typed_cell("rollout-engine-1", "rollout", serving=False),
            ],
            cell_types=("actor", "rollout"),
        )

        assert injected == []

    def test_a_trainer_that_recovered_first_is_not_harmed_again_while_its_engines_rebuild(self) -> None:
        """The trainer's own debt clears as soon as it is back, but the cells it was writing to are still down."""
        injected: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}
        actor_generation = {"actor-0": 0, "actor-1": 0}

        def _cells() -> list[dict]:
            actors = [
                typed_cell(name, "actor", workers_hash=f"generation-{gen}") for name, gen in actor_generation.items()
            ]
            if not injected:
                return actors + [typed_cell(f"rollout-engine-{i}", "rollout") for i in range(2)]
            return actors + [
                typed_cell("rollout-engine-0", "rollout"),
                typed_cell("rollout-engine-1", "rollout", serving=False, workers_hash="generation-1"),
            ]

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 30:
                stop_event.set()
            return mock_response({"items": _cells()})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            name = _name_from_inject_url(url)
            injected.append(name)
            if name in actor_generation:
                actor_generation[name] += 1
            return mock_response({})

        _run_injection_loop(
            fake_get=fake_get,
            fake_post=fake_post,
            mean_interval_seconds_of_cell_type={"actor": 1e-9, "rollout": 1e9},
            stop_event=stop_event,
        )

        assert len(injected) == 1, injected
        assert injected[0] in actor_generation, injected

    def test_a_replacement_trainer_that_has_not_initialized_buys_no_second_harm(self) -> None:
        """It reports Healthy on allocation, and taking that as recovery would crash the one trainer that works."""
        injected: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def _cells() -> list[dict]:
            if not injected:
                return [initialized_trainer_cell(f"actor-{i}", workers_hash="generation-0") for i in range(2)]
            replaced = injected[0]
            return [
                (
                    uninitialized_trainer_cell(name, workers_hash="generation-1")
                    if name == replaced
                    else initialized_trainer_cell(name, workers_hash="generation-0")
                )
                for name in ("actor-0", "actor-1")
            ]

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 25:
                stop_event.set()
            return mock_response({"items": _cells()})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            injected.append(_name_from_inject_url(url))
            return mock_response({})

        _run_injection_loop(fake_get=fake_get, fake_post=fake_post, cell_types=("actor",), stop_event=stop_event)

        assert len(injected) == 1, injected

    def test_the_run_resumes_once_the_replacement_trainer_has_initialized(self) -> None:
        """The heal really finished, so the soak must go on crashing rather than stall on the new generation."""
        injected: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def _cells() -> list[dict]:
            if not injected:
                return [initialized_trainer_cell(f"actor-{i}", workers_hash="generation-0") for i in range(2)]
            replaced = injected[0]
            builder = uninitialized_trainer_cell if polls["n"] < 6 else initialized_trainer_cell
            return [
                (
                    builder(name, workers_hash="generation-1")
                    if name == replaced
                    else initialized_trainer_cell(name, workers_hash="generation-0")
                )
                for name in ("actor-0", "actor-1")
            ]

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if len(injected) >= 2 or polls["n"] >= 40:
                stop_event.set()
            return mock_response({"items": _cells()})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            injected.append(_name_from_inject_url(url))
            return mock_response({})

        _run_injection_loop(fake_get=fake_get, fake_post=fake_post, cell_types=("actor",), stop_event=stop_event)

        assert len(injected) >= 2, injected

    def test_the_run_resumes_once_every_kind_is_back(self) -> None:
        """The gate must let go when the cascade really has settled, or the soak stops testing anything."""
        injected: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}
        recovering = [
            typed_cell("actor-0", "actor"),
            typed_cell("actor-1", "actor"),
            typed_cell("rollout-engine-0", "rollout"),
            typed_cell("rollout-engine-1", "rollout", serving=False),
        ]
        recovered = [
            typed_cell("actor-0", "actor"),
            typed_cell("actor-1", "actor"),
            typed_cell("rollout-engine-0", "rollout"),
            typed_cell("rollout-engine-1", "rollout"),
        ]

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 12:
                stop_event.set()
            return mock_response({"items": recovering if polls["n"] < 4 else recovered})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            injected.append(_name_from_inject_url(url))
            return mock_response({})

        _run_injection_loop(
            fake_get=fake_get, fake_post=fake_post, cell_types=("actor", "rollout"), stop_event=stop_event
        )

        assert injected, injected


class TestUnknownInjectionOutcome:
    """A lost response does not prove a lost kill, so the cell still owes a recovery."""

    def test_a_failed_request_is_not_retried_straight_away(self) -> None:
        """The fault may well have landed, and a blind retry would take out a second replica."""
        attempts: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 20:
                stop_event.set()
            return mock_response({"items": [cell(name, healthy=True) for name in ("actor-0", "actor-1")]})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            attempts.append(_name_from_inject_url(url))
            raise RuntimeError("response lost after the kill may have landed")

        _run_injection_loop(fake_get=fake_get, fake_post=fake_post, cell_types=("actor",), stop_event=stop_event)

        assert len(attempts) == 1, attempts

    def test_a_failed_request_is_not_counted_as_an_injection(self) -> None:
        """The soak's floors have to count faults that really landed, not requests that were sent."""
        log = state.EventLog()
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 8:
                stop_event.set()
            return mock_response({"items": [typed_cell(f"rollout-engine-{i}", "rollout") for i in range(2)]})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            raise RuntimeError("inject-fault refused")

        _run_injection_loop(
            fake_get=fake_get, fake_post=fake_post, cell_types=("rollout",), event_log=log, stop_event=stop_event
        )

        assert views.compute_num_injections(log.events, cell_type="rollout") == 0

    def test_a_replacement_generation_clears_an_unknown_outcome(self) -> None:
        """The fault did land after all, and the run may continue once the victim is demonstrably back."""
        attempts: list[str] = []
        stop_event = threading.Event()
        generation = {"actor-0": 0, "actor-1": 0}
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if len(attempts) >= 2 or polls["n"] >= 60:
                stop_event.set()
            return mock_response(
                {
                    "items": [
                        cell(name, healthy=True, workers_hash=f"generation-{gen}") for name, gen in generation.items()
                    ]
                }
            )

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            name = _name_from_inject_url(url)
            attempts.append(name)
            generation[name] += 1
            raise RuntimeError("response lost after the kill landed")

        _run_injection_loop(fake_get=fake_get, fake_post=fake_post, cell_types=("actor",), stop_event=stop_event)

        assert len(attempts) >= 2, attempts

    def test_an_unknown_outcome_that_never_resolves_fails_the_run(self) -> None:
        """Neither outcome can be assumed, and a silent stall would read as a soak that simply injected less."""
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 40:
                stop_event.set()
            return mock_response({"items": [cell(name, healthy=True) for name in ("actor-0", "actor-1")]})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            raise RuntimeError("response lost after the kill may have landed")

        with pytest.raises(core.UnresolvedInjectionError, match="unknown whether"):
            _run_injection_loop(
                fake_get=fake_get,
                fake_post=fake_post,
                cell_types=("actor",),
                unknown_injection_timeout_seconds=0.0,
                stop_event=stop_event,
            )

    def test_a_known_successful_injection_does_not_trip_the_unknown_deadline(self) -> None:
        """A slow but healthy recovery is judged by the post-run witnesses, not aborted mid-soak."""
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 20:
                stop_event.set()
            return mock_response({"items": [cell(name, healthy=True) for name in ("actor-0", "actor-1")]})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            return mock_response({})

        _run_injection_loop(
            fake_get=fake_get,
            fake_post=fake_post,
            cell_types=("actor",),
            unknown_injection_timeout_seconds=0.0,
            stop_event=stop_event,
        )


class TestTargetSelection:
    """Which cell kind a run crashes is the mode's decision, and each kind keeps its own spare."""

    def test_virtual_cells_use_the_regular_targeted_injection_path(self) -> None:
        """Synthetic replicas satisfy the ordinary scheduler without a real FT cell."""
        injected: list[str] = []
        stop_event = threading.Event()
        virtual_cells = [typed_cell("virtual-0", "virtual"), typed_cell("virtual-1", "virtual")]

        def inject(target: dict, _rng: random.Random) -> None:
            injected.append(target["metadata"]["name"])
            stop_event.set()

        def fake_get(url: str, timeout: float) -> MagicMock:
            return mock_response({"items": []})

        _run_injection_loop(
            fake_get=fake_get,
            cell_types=("virtual",),
            cell_fault_forms={"virtual": [StubFaultForm("virtual-fault", inject)]},
            get_virtual_cells=lambda: virtual_cells,
            stop_event=stop_event,
        )

        assert len(injected) == 1
        assert injected[0] in {"virtual-0", "virtual-1"}

    def test_injection_can_be_restricted_to_one_kind_of_cell(self) -> None:
        """Rollout and trainer cells share one api server, so a run targets one kind at a time."""
        injected = _run_typed_injection_loop(
            [
                typed_cell("actor-0", "actor"),
                typed_cell("actor-1", "actor"),
                typed_cell("rollout-engine-0", "rollout"),
                typed_cell("rollout-engine-1", "rollout"),
            ],
            cell_types=("rollout",),
        )

        assert injected
        assert all(name.startswith("rollout-") for name in injected), injected

    def test_the_live_replica_count_only_considers_the_targeted_kind(self) -> None:
        """A single rollout cell must not be killed just because trainer cells are also alive."""
        injected = _run_typed_injection_loop(
            [
                typed_cell("actor-0", "actor"),
                typed_cell("actor-1", "actor"),
                typed_cell("rollout-engine-0", "rollout"),
            ],
            cell_types=("rollout",),
        )

        assert injected == []

    def test_a_mixed_run_sees_every_targeted_kind(self) -> None:
        """A mixed-ft soak schedules both kinds, and must be able to crash either one."""
        injected = _run_typed_injection_loop(
            [
                typed_cell("actor-0", "actor"),
                typed_cell("actor-1", "actor"),
                typed_cell("rollout-engine-0", "rollout"),
                typed_cell("rollout-engine-1", "rollout"),
            ],
            cell_types=("actor", "rollout"),
        )

        assert injected


class TestListingTheApiServer:
    def test_the_managed_cells_and_the_hook_sources_are_two_lists(self) -> None:
        """The soak heals only what /cells names, and arms only what the source endpoint names."""
        asked: list[str] = []

        def fake_get(url: str, timeout: float) -> MagicMock:
            asked.append(url)
            if url.endswith("/api/v1/fault-hook-sources"):
                return mock_response({"items": [typed_cell("actor-0", "actor")]})
            return mock_response({"items": [typed_cell("rollout-engine-0", "rollout")]})

        with patched_requests() as mock_requests:
            mock_requests.get.side_effect = fake_get
            managed = core.list_cells(base_url="http://control", cell_types={"actor", "rollout"})
            sources = core.list_fault_hook_sources(base_url="http://control")

        assert asked == ["http://control/api/v1/cells", "http://control/api/v1/fault-hook-sources"]
        assert [item["metadata"]["name"] for item in managed] == ["rollout-engine-0"]
        assert [item["metadata"]["name"] for item in sources] == ["actor-0"]

    def test_an_api_server_that_cannot_be_reached_yields_no_sources(self) -> None:
        """A poll that failed is not the same as a run with no trainer, and neither may raise out of the loop."""
        with patched_requests() as mock_requests:
            mock_requests.get.side_effect = RuntimeError("api server unreachable")

            assert core.list_fault_hook_sources(base_url="http://control") is None


class TestFaultInjectionLoopErrorHandling:
    def test_list_cells_failure_is_retried_and_does_not_stop_the_loop(self) -> None:
        """A transient api-server outage must cost one poll, not the rest of the soak."""
        cells = [typed_cell("actor-0", "actor"), typed_cell("actor-1", "actor")]
        log = state.EventLog()
        injected: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] == 1:
                raise RuntimeError("api server unreachable")
            if polls["n"] >= 6:
                stop_event.set()
            return mock_response({"items": cells})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            injected.append(_name_from_inject_url(url))
            return mock_response({})

        _run_injection_loop(
            fake_get=fake_get,
            fake_post=fake_post,
            cell_types=("actor",),
            event_log=log,
            stop_event=stop_event,
        )

        assert injected, injected
        assert views.compute_num_injections(log.events, cell_type="actor") == len(injected)

    def test_a_stop_that_arrives_while_listing_buys_no_further_injection(self) -> None:
        """A fault injected on the way out is one nothing is left polling to see recover."""
        injected: list[str] = []
        stop_event = threading.Event()

        def fake_get(url: str, timeout: float) -> MagicMock:
            stop_event.set()
            return mock_response({"items": [typed_cell("actor-0", "actor"), typed_cell("actor-1", "actor")]})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            injected.append(url)
            return mock_response({})

        _run_injection_loop(fake_get=fake_get, fake_post=fake_post, stop_event=stop_event)

        assert injected == []

    def test_disabled_injection_still_observes_cells_without_injecting(self) -> None:
        """A closing scenario keeps recovery evidence while admitting no new fault."""
        injected: list[str] = []
        event_log = state.EventLog()
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 3:
                stop_event.set()
            return mock_response({"items": [cell("actor-0", healthy=True), cell("actor-1", healthy=True)]})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            injected.append(url)
            return mock_response({})

        _run_injection_loop(
            fake_get=fake_get,
            fake_post=fake_post,
            cell_types=("actor",),
            event_log=event_log,
            injection_enabled=lambda: False,
            stop_event=stop_event,
        )

        assert injected == []
        assert event_log.events


class TestFormDrawing:
    def test_the_loop_injects_through_the_forms_of_the_cell_it_picked(self) -> None:
        """A pod deletion drawn by the loop must reach kubectl, not the api server's inject-fault route."""
        drawn: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 6:
                stop_event.set()
            return mock_response({"items": [typed_cell(f"actor-{i}", "actor") for i in range(3)]})

        with patched_requests() as mock_requests:
            mock_requests.get.side_effect = fake_get
            core.run_fault_injection_loop(
                base_url="http://control",
                seed=0,
                mean_interval_seconds_of_cell_type=intervals(("actor", "rollout"), 1e-12),
                stop_event=stop_event,
                event_log=state.EventLog(),
                cell_fault_forms=fixed_fault_forms(
                    [
                        StubFaultForm(
                            fault_forms.DELETE_POD_FORM_NAME,
                            lambda cell, rng: drawn.append(fault_forms.DELETE_POD_FORM_NAME),
                        )
                    ]
                ),
                poll_interval_seconds=1e-6,
            )

            assert drawn, drawn
            assert set(drawn) == {fault_forms.DELETE_POD_FORM_NAME}, drawn
            mock_requests.post.assert_not_called()

    def test_the_loop_draws_a_form_that_has_never_worked_before_repeating_a_proven_one(self) -> None:
        """Uniform sampling can leave the rarest fault untried for a whole soak, which is the one worth trying."""
        drawn: list[str] = []
        log = state.EventLog()
        stop_event = threading.Event()
        generation = {f"actor-{i}": 0 for i in range(4)}
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if len(drawn) >= 3 or polls["n"] >= 200:
                stop_event.set()
            return mock_response(
                {
                    "items": [
                        typed_cell(name, "actor", workers_hash=f"generation-{gen}") for name, gen in generation.items()
                    ]
                }
            )

        def _draw(name: str, target: dict) -> None:
            drawn.append(name)
            generation[target["metadata"]["name"]] += 1

        with patched_requests() as mock_requests:
            mock_requests.get.side_effect = fake_get
            core.run_fault_injection_loop(
                base_url="http://control",
                seed=0,
                mean_interval_seconds_of_cell_type=intervals(("actor", "rollout"), 1e-12),
                stop_event=stop_event,
                event_log=log,
                cell_fault_forms=fixed_fault_forms(
                    [StubFaultForm(name, lambda cell, rng, n=name: _draw(n, cell)) for name in ("a", "b", "c")]
                ),
                poll_interval_seconds=1e-6,
            )

        assert set(drawn[:3]) == {"a", "b", "c"}, drawn

    def test_a_form_that_always_refuses_keeps_being_drawn_so_the_soak_can_see_it(self) -> None:
        """A form that rides on the ones that did work would end the run green while never having fired."""
        log = state.EventLog()
        stop_event = threading.Event()
        generation = {f"actor-{i}": 0 for i in range(3)}
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 200:
                stop_event.set()
            return mock_response(
                {
                    "items": [
                        typed_cell(name, "actor", workers_hash=f"generation-{gen}") for name, gen in generation.items()
                    ]
                }
            )

        def _works(target: dict, rng: random.Random) -> None:
            generation[target["metadata"]["name"]] += 1

        with patched_requests() as mock_requests:
            mock_requests.get.side_effect = fake_get
            core.run_fault_injection_loop(
                base_url="http://control",
                seed=0,
                mean_interval_seconds_of_cell_type=intervals(("actor", "rollout"), 1e-12),
                stop_event=stop_event,
                event_log=log,
                cell_fault_forms=fixed_fault_forms(
                    [StubFaultForm("works", _works), StubFaultForm("broken", _always_refuse)]
                ),
                poll_interval_seconds=1e-6,
                unknown_injection_timeout_seconds=1e9,
            )

        assert ("actor", "broken") in views.compute_forms_drawn_without_success(log.events)


def _always_refuse(cell: dict, rng: random.Random) -> None:
    raise RuntimeError("this form never works")


class TestComputePendingInjections:
    """The run-wide gate reads this, so it must never call a victim recovered on the old generation."""

    def test_a_harmful_injection_is_pending_until_a_new_generation_serves(self) -> None:
        log = state.EventLog()
        log.observe([typed_cell("rollout-engine-0", "rollout")])
        log.note_injection_attempt(
            cell_name="rollout-engine-0", form_name="sigkill", succeeded=True, workers_hash="generation-0"
        )
        log.observe([typed_cell("rollout-engine-0", "rollout", workers_hash="generation-0")])

        assert [entry.cell_name for entry in views.compute_pending_injections(log.events)] == ["rollout-engine-0"]

    def test_a_new_generation_that_serves_clears_it(self) -> None:
        log = state.EventLog()
        log.note_injection_attempt(
            cell_name="rollout-engine-0", form_name="sigkill", succeeded=True, workers_hash="generation-0"
        )
        log.observe([typed_cell("rollout-engine-0", "rollout", workers_hash="generation-1")])

        assert views.compute_pending_injections(log.events) == []

    def test_a_new_generation_that_is_not_serving_does_not_clear_it(self) -> None:
        log = state.EventLog()
        log.note_injection_attempt(
            cell_name="rollout-engine-0", form_name="sigkill", succeeded=True, workers_hash="generation-0"
        )
        log.observe([typed_cell("rollout-engine-0", "rollout", workers_hash="generation-1", serving=False)])

        assert [entry.cell_name for entry in views.compute_pending_injections(log.events)] == ["rollout-engine-0"]

    def test_a_form_that_harms_nothing_owes_no_recovery(self) -> None:
        log = state.EventLog()
        log.note_injection_attempt(
            cell_name="rollout-engine-0",
            form_name="observe_only",
            succeeded=True,
            harmed=False,
            workers_hash="generation-0",
        )

        assert views.compute_pending_injections(log.events) == []

    def test_a_request_whose_outcome_is_unknown_is_marked_as_such(self) -> None:
        log = state.EventLog()
        log.note_injection_attempt(
            cell_name="rollout-engine-0", form_name="sigkill", succeeded=False, workers_hash="generation-0"
        )

        [entry] = views.compute_pending_injections(log.events)
        assert not entry.outcome_known


class TestRaiseIfAnUnknownInjectionIsStuck:
    def test_a_fresh_unknown_outcome_is_given_time(self) -> None:
        entry = views.PendingInjection(
            cell_name="actor-0",
            workers_hash="generation-0",
            injected_at=datetime.now(timezone.utc),
            outcome_known=False,
        )

        core._raise_if_an_unknown_injection_is_stuck([entry], timeout_seconds=600.0)

    def test_a_stale_unknown_outcome_stops_the_run(self) -> None:
        entry = views.PendingInjection(
            cell_name="actor-0",
            workers_hash="generation-0",
            injected_at=datetime.now(timezone.utc) - timedelta(seconds=1200),
            outcome_known=False,
        )

        with pytest.raises(core.UnresolvedInjectionError):
            core._raise_if_an_unknown_injection_is_stuck([entry], timeout_seconds=600.0)

    def test_a_stale_but_known_outcome_is_left_to_the_post_run_witnesses(self) -> None:
        entry = views.PendingInjection(
            cell_name="actor-0",
            workers_hash="generation-0",
            injected_at=datetime.now(timezone.utc) - timedelta(seconds=1200),
            outcome_known=True,
        )

        core._raise_if_an_unknown_injection_is_stuck([entry], timeout_seconds=600.0)


class TestFormAvailabilityAndSelfRecording:
    def test_a_kind_whose_every_form_is_unavailable_injects_nothing(self) -> None:
        """A hook form that has no source to arm yet defers the injection instead of forcing one."""
        drawn: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 6:
                stop_event.set()
            return mock_response({"items": [typed_cell(f"actor-{i}", "actor") for i in range(3)]})

        log = state.EventLog()
        _run_injection_loop(
            fake_get=fake_get,
            event_log=log,
            cell_fault_forms=fixed_fault_forms(
                [StubFaultForm("never_ready", lambda cell, rng: drawn.append("never_ready"), available=False)]
            ),
            stop_event=stop_event,
        )

        assert drawn == []
        assert views.compute_num_injections(log.events) == 0

    def test_the_loop_keeps_no_books_for_a_form_that_keeps_its_own(self) -> None:
        """Arming a hook is not harming the cell the request was sent to, so the loop records nothing."""
        drawn: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if len(drawn) >= 1 or polls["n"] >= 50:
                stop_event.set()
            return mock_response({"items": [typed_cell(f"actor-{i}", "actor") for i in range(3)]})

        log = state.EventLog()
        _run_injection_loop(
            fake_get=fake_get,
            event_log=log,
            cell_fault_forms=fixed_fault_forms(
                [
                    StubFaultForm(
                        "fault_hook:local",
                        lambda cell, rng: drawn.append("fault_hook:local"),
                        records_own_attempt=True,
                    )
                ]
            ),
            stop_event=stop_event,
        )

        assert drawn
        assert not [event for event in log.events if isinstance(event, state.InjectionEvent)]

    def test_a_form_that_keeps_its_own_books_records_nothing_when_it_raises(self) -> None:
        """A failed arm owes its own record; the loop must not invent a harmed cell for it."""
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 4:
                stop_event.set()
            return mock_response({"items": [typed_cell(f"actor-{i}", "actor") for i in range(3)]})

        def _raise(cell: dict, rng: random.Random) -> None:
            raise RuntimeError("the api server never answered")

        log = state.EventLog()
        _run_injection_loop(
            fake_get=fake_get,
            event_log=log,
            cell_fault_forms=fixed_fault_forms([StubFaultForm("fault_hook:local", _raise, records_own_attempt=True)]),
            stop_event=stop_event,
        )

        assert not [event for event in log.events if isinstance(event, state.InjectionEvent)]
