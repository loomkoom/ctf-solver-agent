from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.agent import build_graph
from src.config import settings
from src.state import DCipherState
from src.tools.rag import ingest_knowledge_base
from src.tools.toolbox import Toolbox


@dataclass
class BenchChallenge:
    challenge_id: str
    name: str
    category: str
    description: str
    path: Path
    files: list[Path]


class BenchmarkRunner:
    def __init__(
        self,
        bench_root: Path,
        toolbox: Toolbox,
        max_iterations: int = 15,
        flag_format: str = "flag{",
        run_id: str | None = None,
    ):
        self.bench_root = bench_root
        self.toolbox = toolbox
        self.max_iterations = max_iterations
        self.flag_format = flag_format
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.graph = build_graph(toolbox, connector=None)

    def load_challenges(self) -> list[BenchChallenge]:
        challenges: list[BenchChallenge] = []
        seen_dirs: set[Path] = set()
        for dir_path in self._iter_candidate_dirs():
            if dir_path in seen_dirs:
                continue
            desc_path = self._find_description_file(dir_path)
            if not desc_path:
                continue
            seen_dirs.add(dir_path)

            description = self._read_text(desc_path)
            files = self._collect_artifacts(dir_path, desc_path)
            challenge_id, name, category = self._infer_metadata(dir_path)

            if not files and self._has_child_descriptions(dir_path):
                continue

            challenges.append(
                BenchChallenge(
                    challenge_id=challenge_id,
                    name=name,
                    category=category,
                    description=description,
                    path=dir_path,
                    files=files,
                )
            )
        return challenges

    def run(self, limit: int = 0) -> list[dict]:
        settings.max_iterations = self.max_iterations
        settings.flag_format = self.flag_format

        results = []
        challenges = self.load_challenges()
        if limit:
            challenges = challenges[:limit]

        for challenge in challenges:
            result = self._solve_challenge(challenge)
            results.append(result)
        return results

    def _solve_challenge(self, challenge: BenchChallenge) -> dict:
        container_dir = self._stage_challenge(challenge)
        file_list = [f.relative_to(challenge.path).as_posix() for f in challenge.files]
        context = self._build_context(challenge, container_dir, file_list)

        state: DCipherState = {
            "challenge_id": challenge.challenge_id,
            "challenge_name": challenge.name,
            "category": challenge.category,
            "description": challenge.description,
            "current_files": file_list,
            "attempt_history": [],
            "reasoning_log": [],
            "messages": [],
            "challenge_context": context,
            "flag_format": self.flag_format,
            "current_objective": "Initial reconnaissance",
            "plan": "",
            "last_command": "",
            "last_output": "",
            "last_error": "",
            "flag_candidates": [],
            "iteration": 0,
            "submitted_flags": [],
            "done": False,
        }

        final_state = self.graph.invoke(state)
        expected_flag = self._extract_expected_flag(challenge.path)
        found_flags = final_state.get("flag_candidates") or []
        success = False
        if expected_flag:
            success = expected_flag in found_flags
        else:
            success = bool(found_flags)

        return {
            "challenge_id": challenge.challenge_id,
            "name": challenge.name,
            "category": challenge.category,
            "expected_flag": expected_flag,
            "flag_candidates": found_flags,
            "success": success,
            "iterations": final_state.get("iteration"),
            "reasoning_log": final_state.get("reasoning_log"),
        }

    def _iter_candidate_dirs(self) -> list[Path]:
        skip_dirs = {"solve", "solution", "writeup", "writeups", "images", "img", "assets"}
        dirs = []
        for path in self.bench_root.rglob("*"):
            if not path.is_dir():
                continue
            if any(part.lower() in skip_dirs for part in path.parts):
                continue
            dirs.append(path)
        return dirs

    def _find_description_file(self, dir_path: Path) -> Path | None:
        candidates = {
            "readme.md",
            "readme",
            "challenge.md",
            "challenge.txt",
            "challenge_description.txt",
            "description.txt",
        }
        for file in dir_path.iterdir():
            if file.is_file() and file.name.lower() in candidates:
                return file
        return None

    def _collect_artifacts(self, dir_path: Path, desc_path: Path) -> list[Path]:
        skip_dirs = {"solve", "solution", "writeup", "writeups", "images", "img", "assets"}
        files = []
        for file in dir_path.rglob("*"):
            if not file.is_file() or file == desc_path:
                continue
            if any(part.lower() in skip_dirs for part in file.parts):
                continue
            files.append(file)
        return files

    def _has_child_descriptions(self, dir_path: Path) -> bool:
        for child in dir_path.iterdir():
            if not child.is_dir():
                continue
            if self._find_description_file(child):
                return True
        return False

    def _infer_metadata(self, dir_path: Path) -> tuple[str, str, str]:
        rel_parts = dir_path.relative_to(self.bench_root).parts
        name = rel_parts[-1] if rel_parts else dir_path.name
        category = rel_parts[-2] if len(rel_parts) >= 2 else "unknown"
        challenge_id = "/".join(rel_parts) if rel_parts else name
        return challenge_id, name, category

    def _stage_challenge(self, challenge: BenchChallenge) -> str:
        container_base = f"{settings.sandbox_workdir}/bench/{self.run_id}"
        self.toolbox.run(f"mkdir -p {container_base}")
        slug = challenge.challenge_id.replace("/", "_")
        container_dir = f"{container_base}/{slug}"
        self.toolbox.copy_to_container(str(challenge.path), container_dir)
        return container_dir

    def _build_context(
        self,
        challenge: BenchChallenge,
        container_dir: str,
        files: list[str],
    ) -> str:
        file_lines = "\n".join(f"- {fname}" for fname in files) if files else "None"
        return (
            f"CHALLENGE: {challenge.name}\n"
            f"CATEGORY: {challenge.category}\n"
            f"DESCRIPTION:\n{challenge.description}\n\n"
            f"FILES (container path {container_dir}):\n{file_lines}\n"
        )

    def _extract_expected_flag(self, dir_path: Path) -> str | None:
        for file in dir_path.rglob("*"):
            if not file.is_file():
                continue
            if "flag" not in file.name.lower():
                continue
            if file.stat().st_size > 10240:
                continue
            text = self._read_text(file)
            flags = _extract_flags(text, self.flag_format)
            if flags:
                return flags[0]
        return None

    def _read_text(self, file: Path) -> str:
        try:
            return file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file.read_text(encoding="latin-1", errors="ignore")


def _extract_flags(text: str, flag_format: str) -> list[str]:
    patterns = [
        re.escape(flag_format.rstrip("{")) + r"\{[^}]{3,}\}",
        r"[A-Za-z0-9]{2,}\{[^}]{3,}\}",
    ]
    matches = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            if match not in matches:
                matches.append(match)
    return matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NYU CTF Bench runner")
    parser.add_argument("--bench", required=True, help="Path to local benchmark dataset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--knowledge-base", default=settings.knowledge_base_path)
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=settings.max_iterations)
    parser.add_argument("--flag-format", default=settings.flag_format)
    parser.add_argument("--output", default="logs/nyu_bench_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bench_root = Path(args.bench)
    if not bench_root.exists():
        raise SystemExit(f"Benchmark path not found: {bench_root}")
    if not args.skip_rag:
        ingest_knowledge_base(args.knowledge_base, force=args.reindex)

    toolbox = Toolbox()
    runner = BenchmarkRunner(
        bench_root=bench_root,
        toolbox=toolbox,
        max_iterations=args.max_iterations,
        flag_format=args.flag_format,
    )
    results = runner.run(limit=args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
