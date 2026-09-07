from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as file:
            file.write(text)
            # mkstemp opens 0600, which leaves a file one uid wrote unreadable to the next one
            os.fchmod(file.fileno(), 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_torch_save(path: str | Path, obj: Any) -> None:
    path = Path(path)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as file:
            torch.save(obj, file)
            os.fchmod(file.fileno(), 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
