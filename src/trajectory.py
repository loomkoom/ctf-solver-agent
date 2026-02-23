from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrajectoryLogger:
    path: Path
    challenge_id: str
    run_id: str | None = None
    enabled: bool = True

    def log(self, event: str, data: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        payload: dict[str, Any] = {
            "ts": time.time(),
            "event": event,
            "challenge_id": self.challenge_id,
        }
        if self.run_id:
            payload["run_id"] = self.run_id
        if data:
            payload.update(data)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
