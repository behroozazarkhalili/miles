# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import typer
from tests.e2e.ft.conftest_ft.cli_options import FailureModeOption, ModeOption, NumStepsOption
from tests.e2e.ft.conftest_ft.hook_injection import (
    DEFAULT_TARGETED_FAILURE_MODE,
    TargetedHookRun,
    assert_hook_fired,
    assert_remote_victim_recovered,
    assert_remote_victim_was_harmed,
    assert_unrelated_target_kept_serving,
    assert_weights_published_after,
    compute_victim_harm_observed_at,
    require_armed,
    resolve_fire_assignment,
    run_targeted_hook_scenario,
)

from miles.utils.test_utils.fault_hooks import FaultHookName, FaultHookTarget
from miles.utils.test_utils.fault_injector import FailureMode

app: typer.Typer = typer.Typer()

TEST_NAME: str = "weight_update_p2p_remote"

DEFAULT_NUM_STEPS: int = 8
REQUEST_ID: str = "weight-update-p2p-remote"
FAILURE_MODE: FailureMode = DEFAULT_TARGETED_FAILURE_MODE
HOOK: FaultHookName = FaultHookName.WEIGHT_UPDATE_AFTER_P2P_SUBMIT
ARMED_WORKER_IN_CELL_INDEX: int = 0


@app.command(name="run")
def run_ci(
    mode: ModeOption,
    num_steps: NumStepsOption = DEFAULT_NUM_STEPS,
    failure_mode: FailureModeOption = FAILURE_MODE,
) -> None:
    run = run_targeted_hook_scenario(
        test_name=TEST_NAME,
        mode=mode,
        num_steps=num_steps,
        hook=HOOK,
        failure_mode=failure_mode,
        target=FaultHookTarget.REMOTE_INFERENCE_CELL,
        request_id=REQUEST_ID,
        sub_index=ARMED_WORKER_IN_CELL_INDEX,
    )

    assert_hook_outcome(run)

    print(
        f"Targeted p2p target fault test PASSED ({TEST_NAME}, mode={mode}, steps={num_steps}, "
        f"failure_mode={failure_mode.value})"
    )


def assert_hook_outcome(run: TargetedHookRun) -> None:
    armed = require_armed(run.armer)
    fire = assert_hook_fired(armed, event_dir=run.event_dir)
    assignment = resolve_fire_assignment(fire, armed, event_dir=run.event_dir)

    victim = assert_remote_victim_was_harmed(fire, run.events, armed=armed, assignment=assignment)
    harm_observed_at = compute_victim_harm_observed_at(run.events, fire=fire)
    victim_recovered_at = assert_remote_victim_recovered(run.events, fire=fire, since=harm_observed_at)
    assert_unrelated_target_kept_serving(run.events, armed=armed, assignment=assignment, since=harm_observed_at)
    assert_weights_published_after(
        run.event_dir,
        fire=fire,
        trainer_model_id=armed.expected_source.model_id,
        required_cell_ids={victim},
        recovered_at=victim_recovered_at,
    )

    print(f"Remote fault confined to {victim}")


if __name__ == "__main__":
    app()
