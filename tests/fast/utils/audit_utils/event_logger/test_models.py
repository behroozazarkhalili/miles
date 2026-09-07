import pytest
from pydantic import ValidationError

from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent


class TestInferenceEngineWeightChecksumEvent:
    def test_none_rollout_id_is_rejected(self) -> None:
        """A checksum event rejects the former null startup rollout identifier."""
        data = {
            "timestamp": "2026-01-01T00:00:00Z",
            "source": {"component": "main"},
            "rollout_id": None,
            "weight_version": 3,
            "engine_checksums": {"cell-a": {"rank0/embed.weight": "aaa"}},
        }

        with pytest.raises(ValidationError, match="rollout_id"):
            InferenceEngineWeightChecksumEvent.model_validate(data)

    def test_a_missing_weight_version_is_rejected(self) -> None:
        """Without the published version, every consumer would have to guess which weights it is looking at."""
        data = {
            "timestamp": "2026-01-01T00:00:00Z",
            "source": {"component": "main"},
            "rollout_id": 3,
            "engine_checksums": {"cell-a": {"rank0/embed.weight": "aaa"}},
        }

        with pytest.raises(ValidationError, match="weight_version"):
            InferenceEngineWeightChecksumEvent.model_validate(data)

    def test_a_positional_engine_list_is_rejected(self) -> None:
        """Checksums keyed by list position cannot name the cell they came from, so the old shape must fail."""
        data = {
            "timestamp": "2026-01-01T00:00:00Z",
            "source": {"component": "main"},
            "rollout_id": 3,
            "weight_version": 3,
            "engine_checksums": [{"rank0/embed.weight": "aaa"}],
        }

        with pytest.raises(ValidationError, match="engine_checksums"):
            InferenceEngineWeightChecksumEvent.model_validate(data)
