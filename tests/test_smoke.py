from __future__ import annotations

from pathlib import Path

from src.challenges import load_challenge_dir


def test_load_local_challenge(tmp_path: Path) -> None:
    (tmp_path / "description.txt").write_text("demo challenge", encoding="utf-8")
    attachments = tmp_path / "attachments"
    attachments.mkdir()
    (attachments / "artifact.bin").write_bytes(b"demo")

    challenge = load_challenge_dir(tmp_path)
    assert challenge is not None
    assert challenge.description.strip() == "demo challenge"
    assert challenge.files
