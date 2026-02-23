from pathlib import Path
import re

def list_challenge_artifacts(source_path: str):
    """
    Lists files in the artifacts directory matching the source_path of a writeup.
    Example input: 'angstrom/2021/pwn/pawn'
    """
    if not source_path:
        return "No source path provided."

    clean_path = source_path.replace("\\", "/")
    if clean_path.startswith("writeups/"):
        clean_path = clean_path[len("writeups/"):]
    if clean_path.endswith(".md"):
        clean_path = str(Path(clean_path).parent)

    artifact_base = Path("data/artifacts") / clean_path

    if not artifact_base.exists():
        return f"No artifact folder found at {artifact_base}"

    files = [f.name for f in artifact_base.iterdir() if f.is_file()]
    return f"Artifacts available for this challenge: {', '.join(files)}"


def extract_path_from_text(text: str):
    """
    Extracts a potential file path from the LLM's objective string.
    Looks for common CTF path patterns like 'Competition/Year/Category'.
    """
    # Regex to find strings like 'angstrom/2021/pwn/pawn'
    # (Matches alphanumeric, hyphens, and slashes)
    match = re.search(r'([a-zA-Z0-9\-_]+(?:/[a-zA-Z0-9\-_]+)+)', text)
    return match.group(1) if match else None

