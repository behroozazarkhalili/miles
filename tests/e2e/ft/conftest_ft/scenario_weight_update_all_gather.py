# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import typer
from tests.e2e.ft.conftest_ft.cli_options import FailureModeOption, ModeOption, NumStepsOption
from tests.e2e.ft.conftest_ft.hook_injection import (
    DEFAULT_TARGETED_FAILURE_MODE,
    TargetedHookRun,
    assert_armed_trainer_was_replaced,
    assert_assigned_targets_isolated,
    assert_hook_fired,
    assert_trainer_cell_healed,
    assert_unrelated_target_kept_serving,
    assert_weights_published_after,
    require_armed,
    resolve_fire_assignment,
    run_targeted_hook_scenario,
)

from miles.utils.test_utils.fault_hooks import FaultHookName, FaultHookTarget
from miles.utils.test_utils.fault_injector import FailureMode

app: typer.Typer = typer.Typer()

TEST_NAME: str = "weight_update_all_gather"

DEFAULT_NUM_STEPS: int = 8
REQUEST_ID: str = "weight-update-all-gather"
FAILURE_MODE: FailureMode = DEFAULT_TARGETED_FAILURE_MODE
HOOK: FaultHookName = FaultHookName.WEIGHT_UPDATE_BEFORE_ALL_GATHER
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
        target=FaultHookTarget.LOCAL,
        request_id=REQUEST_ID,
        sub_index=ARMED_WORKER_IN_CELL_INDEX,
    )

    assert_hook_outcome(run)

    print(
        f"Targeted all-gather fault test PASSED ({TEST_NAME}, mode={mode}, steps={num_steps}, "
        f"failure_mode={failure_mode.value})"
    )


def assert_hook_outcome(run: TargetedHookRun) -> None:
    armed = require_armed(run.armer)
    fire = assert_hook_fired(armed, event_dir=run.event_dir)
    assignment = resolve_fire_assignment(fire, armed, event_dir=run.event_dir)

    harm_observed_at = assert_armed_trainer_was_replaced(run.events, armed=armed, fire=fire)
    assert_trainer_cell_healed(run.event_dir, assignment=assignment)
    assert_assigned_targets_isolated(run.events, assignment=assignment, since=harm_observed_at)
    assert_unrelated_target_kept_serving(run.events, armed=armed, assignment=assignment, since=harm_observed_at)
    assert_weights_published_after(run.event_dir, after=assignment.timestamp)


if __name__ == "__main__":
    app()
