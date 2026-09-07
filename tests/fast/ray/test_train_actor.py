import os
import socket
import threading
from types import SimpleNamespace

import pytest

from miles.ray import placement_group, train_actor
from miles.ray.train_actor import TrainRayActor
from miles.utils.init_once import InitOnce
from miles.utils.test_utils import fault_hooks
from miles.utils.test_utils.fault_hooks import FaultHookAlreadyArmedError, FaultHookName, FaultHookTarget
from miles.utils.workers.env_vars import CELL_INDEX_ENV_VAR, SUBPROCESS_INDEX_ENV_VAR


def _unimplemented_actor_operation(*args: object, **kwargs: object) -> None:
    raise NotImplementedError


class _ActorWithoutReloadSupport(TrainRayActor):
    init = _unimplemented_actor_operation
    sleep = _unimplemented_actor_operation
    wake_up = _unimplemented_actor_operation
    train = _unimplemented_actor_operation
    save_model = _unimplemented_actor_operation
    update_weights = _unimplemented_actor_operation
    _get_parallel_config = _unimplemented_actor_operation


def _inited_guard() -> InitOnce:
    guard = InitOnce("TrainRayActor")
    with guard.guarding():
        pass
    return guard


class TestConstructorSignature:
    def test_positional_constructor_arguments_are_rejected(self):
        """Workers are built from a spec's kwargs, so silently shifted positional args must not construct one."""
        with pytest.raises(TypeError):
            TrainRayActor(SimpleNamespace(), 2, 1, "10.0.0.1:1234", "actor", 0)


class TestProposeMasterAddrAndPort:
    def test_the_proposal_steps_past_a_port_that_is_already_taken(self, monkeypatch: pytest.MonkeyPatch):
        """A cell rendezvouses on the proposing worker's own node, on a port no other process already holds."""
        monkeypatch.setattr(train_actor, "get_current_node_ip", lambda: "10.0.0.3")

        with socket.socket() as occupied:
            occupied.bind(("", train_actor.get_free_port(start_port=20500)))
            occupied.listen(1)
            taken_port = occupied.getsockname()[1]
            monkeypatch.setattr(train_actor.random, "randint", lambda _low, _high: taken_port)

            addr, port = TrainRayActor.__new__(TrainRayActor).propose_master_addr_and_port()

        assert addr == "10.0.0.3"
        assert port > taken_port
        with socket.socket() as probe:
            probe.bind(("", port))


class TestKillSelf:
    def test_kill_self_exits_with_a_failure_status(self, monkeypatch: pytest.MonkeyPatch):
        """A worker asked to die must leave no survivor and must not look like a clean shutdown."""
        exit_statuses: list[int] = []
        monkeypatch.setattr(train_actor.os, "_exit", exit_statuses.append)

        TrainRayActor.__new__(TrainRayActor).kill_self()

        assert exit_statuses == [1]


class TestLoadState:
    def test_an_actor_without_reload_support_refuses_in_place_state_loading(self):
        """An unsupported backend must require a restart instead of pretending to reload in place."""
        actor = _ActorWithoutReloadSupport.__new__(_ActorWithoutReloadSupport)

        with pytest.raises(
            NotImplementedError,
            match="_ActorWithoutReloadSupport cannot reload its state without restarting",
        ):
            actor.load_state()


class TestConfigureMasterAddrAndPort:
    def _make_actor(self) -> TrainRayActor:
        return TrainRayActor.__new__(TrainRayActor)

    def test_the_master_addr_and_port_land_in_the_environment(self, monkeypatch: pytest.MonkeyPatch):
        """The driver-assigned addr/port must reach the env vars that torch's env:// init reads."""
        monkeypatch.delenv("MASTER_ADDR", raising=False)
        monkeypatch.delenv("MASTER_PORT", raising=False)

        self._make_actor().configure_master_addr_and_port(master_addr="10.0.0.1", master_port=20001)

        assert os.environ["MASTER_ADDR"] == "10.0.0.1"
        assert os.environ["MASTER_PORT"] == "20001"

    def test_a_stale_master_addr_and_port_are_overwritten(self, monkeypatch: pytest.MonkeyPatch):
        """A worker inheriting another run's env must end up on the addr/port the driver assigned."""
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "1")

        self._make_actor().configure_master_addr_and_port(master_addr="10.0.0.2", master_port=20002)

        assert os.environ["MASTER_ADDR"] == "10.0.0.2"
        assert os.environ["MASTER_PORT"] == "20002"


class TestTrainParallelConfigWiring:
    async def test_the_driver_passes_the_resolved_actor_config_to_the_rollout_executor(self, monkeypatch):
        """The driver resolves the actor config before handing it to the rollout executor."""
        train_parallel_config = {"dp_size": 4, "topology": {"tp_size": 2}}
        trainer_config = SimpleNamespace(role="actor", trainer_id="actor")

        class FakeActorHandle:
            async def get_train_parallel_config(self):
                return train_parallel_config

        class FakeRolloutExecutor:
            def __init__(self):
                self.received_config = None

            async def set_train_parallel_config(self, config):
                assert not isinstance(config, FakeActorHandle)
                self.received_config = config

            async def load(self, rollout_id):
                pass

        actor_handle = FakeActorHandle()
        rollout_executor = FakeRolloutExecutor()
        monkeypatch.setattr(placement_group, "compute_trainer_configs", lambda args: [trainer_config])
        monkeypatch.setattr(
            placement_group,
            "create_trainer_handles",
            lambda args, *, trainer_configs: {trainer_config.trainer_id: actor_handle},
        )
        monkeypatch.setattr(placement_group, "take_over_trainers", lambda args, *, handles: _return(False))
        monkeypatch.setattr(placement_group, "compute_trainer_args", lambda args, config: args)
        monkeypatch.setattr(
            placement_group,
            "create_training_model",
            lambda args, *, handle, trainer_id, resumed: _return(
                placement_group.TrainerInfo(handle=actor_handle, restored_rollout_id=0, start_rollout_id=0)
            ),
        )

        await placement_group.create_training_models(
            SimpleNamespace(use_critic=False, start_rollout_id=None), rollout_executor=rollout_executor
        )

        assert rollout_executor.received_config is train_parallel_config


async def _return(value):
    return value


def _actor_with(guard: InitOnce) -> TrainRayActor:
    actor = TrainRayActor.__new__(TrainRayActor)
    actor._init_once = guard
    return actor


class TestInitRunsExactlyOnce:
    def test_init_rebinds_reporting_after_distributed_rank_is_final(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reporter must receive the rank and world size discovered by process-group initialization."""
        args = SimpleNamespace(
            debug_deterministic_collective=False,
            distributed_backend="nccl",
            distributed_timeout_minutes=1,
            fsdp_cpu_offload=False,
        )
        actor = _ActorWithoutReloadSupport.__new__(_ActorWithoutReloadSupport)
        actor._init_once = InitOnce("TrainRayActor")
        actor._heartbeat = SimpleNamespace(bump=lambda: None)
        rebound: list[object] = []
        monkeypatch.setattr(train_actor.torch.cuda, "set_device", lambda _device: None)
        monkeypatch.setattr(train_actor.dist, "init_process_group", lambda **_kwargs: None)
        monkeypatch.setattr(train_actor.dist, "get_rank", lambda: 3)
        monkeypatch.setattr(train_actor.dist, "get_world_size", lambda: 8)
        monkeypatch.setattr(train_actor, "init_gloo_group", lambda: None)
        monkeypatch.setattr(train_actor, "rebind_env_reporting", rebound.append)
        monkeypatch.setattr(train_actor.torch.version, "hip", "test")

        actor._init_common(args=args, role="actor")

        assert (args.rank, args.world_size) == (3, 8)
        assert rebound == [args]

    def test_a_second_init_is_refused(self):
        """A worker that already initialized is a stale process; reusing it must fail loudly, not train on."""
        actor = _actor_with(_inited_guard())

        with pytest.raises(AssertionError, match="stale worker"):
            actor._init_common(None, "actor")

    def test_a_worker_that_never_ran_init_reports_itself_uninitialized(self):
        """A restarted script asks a worker it found running whether to initialize it or to resume it."""
        assert _actor_with(InitOnce("TrainRayActor")).is_initialized() is False

    def test_a_worker_that_ran_init_reports_itself_initialized(self):
        """The take-over path has to see the worker the previous script built as built."""
        assert _actor_with(_inited_guard()).is_initialized() is True


class TestTheLocalGpuIsFoundWithoutRay:
    def test_a_supervised_rank_reads_its_index_rather_than_asking_ray(self, monkeypatch):
        """A served worker is a plain process, not an actor, so ray owns no assignment and answers []."""
        monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
        monkeypatch.setenv(SUBPROCESS_INDEX_ENV_VAR, "3")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [])

        assert train_actor.get_local_gpu_id() == 3

    def test_a_pod_running_one_worker_needs_no_supervisor_to_know_its_card(self, monkeypatch):
        """A cell one worker wide is launched without the supervisor, so the index it would set is absent."""
        monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
        monkeypatch.delenv(SUBPROCESS_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [])

        assert train_actor.get_local_gpu_id() == 0

    def test_a_ray_placed_actor_still_reads_its_assignment_from_ray(self, monkeypatch):
        """Every existing run takes this path, where ray does own the assignment."""
        monkeypatch.delenv(CELL_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv(SUBPROCESS_INDEX_ENV_VAR, raising=False)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [6])

        assert train_actor.get_local_gpu_id() == 2

    def test_a_ray_actor_without_a_visible_device_list_still_reads_its_assignment(self, monkeypatch):
        """The other ray shape: no mask, so the assignment is the id itself."""
        monkeypatch.delenv(CELL_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv(SUBPROCESS_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [5])

        assert train_actor.get_local_gpu_id() == 5


class TestArmFaultHook:
    def test_an_acknowledged_arm_fires_on_the_thread_that_reaches_the_hook(self, monkeypatch: pytest.MonkeyPatch):
        """The rpc is answered by the trainer process itself, so the hook a worker thread walks is the armed one."""
        monkeypatch.setattr(fault_hooks, "_REGISTRY", fault_hooks._FaultHookRegistry())
        injected: list[str] = []
        monkeypatch.setattr(fault_hooks, "inject_fault", lambda mode: injected.append(mode))
        actor = _actor_with(InitOnce("TrainRayActor"))

        actor.arm_fault_hook(
            hook=FaultHookName.WEIGHT_UPDATE_BEFORE_P2P_WRITE.value,
            mode="sigkill",
            request_id="req-1",
            target=FaultHookTarget.LOCAL.value,
        )
        assert injected == []

        reached = threading.Thread(
            target=lambda: fault_hooks.reach_fault_hook(FaultHookName.WEIGHT_UPDATE_BEFORE_P2P_WRITE)
        )
        reached.start()
        reached.join(timeout=5.0)

        assert not reached.is_alive()
        assert injected == ["sigkill"]

    def test_a_second_arm_at_the_same_hook_is_reported_to_the_caller(self, monkeypatch: pytest.MonkeyPatch):
        """The caller must learn its request was refused instead of waiting for a fault that replaced nothing."""
        monkeypatch.setattr(fault_hooks, "_REGISTRY", fault_hooks._FaultHookRegistry())
        actor = _actor_with(InitOnce("TrainRayActor"))
        hook = FaultHookName.WEIGHT_UPDATE_AFTER_P2P_SUBMIT.value

        actor.arm_fault_hook(hook=hook, mode="sigkill", request_id="req-1", target=FaultHookTarget.LOCAL.value)

        with pytest.raises(FaultHookAlreadyArmedError):
            actor.arm_fault_hook(hook=hook, mode="exit", request_id="req-2", target=FaultHookTarget.LOCAL.value)

    def test_an_unknown_hook_name_is_refused_by_the_trainer(self, monkeypatch: pytest.MonkeyPatch):
        """A name no production site carries would sit armed forever and silently pass the test it was meant for."""
        monkeypatch.setattr(fault_hooks, "_REGISTRY", fault_hooks._FaultHookRegistry())
        actor = _actor_with(InitOnce("TrainRayActor"))

        with pytest.raises(ValueError):
            actor.arm_fault_hook(
                hook="weight_update.before_typo",
                mode="sigkill",
                request_id="req-1",
                target=FaultHookTarget.LOCAL.value,
            )
