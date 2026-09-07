# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

from pathlib import Path

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent, TrainGroupStepEndEvent

CHECKPOINT_TRACKER_FILENAME: str = "latest_checkpointed_iteration.txt"
CHECKPOINT_DIRNAME: str = "ckpt"


def recovery_source_exists(*, event_dir: Path, checkpoint_dir: Path) -> bool:
    if read_checkpoint_iteration(checkpoint_dir) is None:
        return False
    if not event_dir.is_dir():
        return False

    events = read_events(event_dir)
    if not any(isinstance(event, TrainGroupStepEndEvent) and completed_a_step(event) for event in events):
        return False
    return any(
        isinstance(event, InferenceEngineWeightChecksumEvent) and event.rollout_id >= 0 and carries_checksums(event)
        for event in events
    )


def read_checkpoint_iteration(checkpoint_dir: Path) -> int | None:
    tracker = checkpoint_dir / CHECKPOINT_TRACKER_FILENAME
    if not tracker.is_file():
        return None
    content = tracker.read_text().strip()
    if not content.isdigit() or int(content) < 1:
        return None
    return int(content)


def completed_a_step(event: TrainGroupStepEndEvent) -> bool:
    return any(outcome != "error" and TrainStepOutcome.NORMAL in outcome for outcome in event.cell_outcomes.values())


def carries_checksums(event: InferenceEngineWeightChecksumEvent) -> bool:
    return any(checksums for checksums in event.engine_checksums.values())
