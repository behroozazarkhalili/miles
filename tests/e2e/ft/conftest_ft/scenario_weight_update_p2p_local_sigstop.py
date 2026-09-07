# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import typer
from tests.e2e.ft.conftest_ft import scenario_weight_update_p2p_local
from tests.e2e.ft.conftest_ft.cli_options import ModeOption, NumStepsOption

from miles.utils.test_utils.fault_injector import FailureMode

app: typer.Typer = typer.Typer()

TEST_NAME: str = "weight_update_p2p_local_sigstop"

FAILURE_MODE: FailureMode = FailureMode.SIGSTOP


@app.command(name="run")
def run_ci(mode: ModeOption, num_steps: NumStepsOption = scenario_weight_update_p2p_local.DEFAULT_NUM_STEPS) -> None:
    scenario_weight_update_p2p_local.run_ci(mode, num_steps=num_steps, failure_mode=FAILURE_MODE)


if __name__ == "__main__":
    app()
