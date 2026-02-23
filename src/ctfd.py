import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class ChallengeFile:
    name: str
    url: str


class CTFdConnector:
    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        username: str | None = None,
        password: str | None = None,
        verify_tls: bool = True,
        timeout_s: int = 15,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.verify = verify_tls

        if token:
            self.session.headers.update({"Authorization": f"Token {token}"})
        elif username and password:
            self.login(username, password)

        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "ig-ctf-solver/1.0",
            }
        )

    def login(self, username: str, password: str) -> None:
        login_api = f"{self.base_url}/api/v1/users/login"
        resp = self.session.post(
            login_api,
            json={"name": username, "password": password},
            timeout=self.timeout_s,
        )
        if resp.ok:
            data = resp.json().get("data", {})
            token = data.get("token")
            if token:
                self.session.headers.update({"Authorization": f"Token {token}"})
                return

        login_page = self.session.get(f"{self.base_url}/login", timeout=self.timeout_s)
        nonce = self._extract_nonce(login_page.text)
        payload = {"name": username, "password": password}
        if nonce:
            payload["nonce"] = nonce
        self.session.post(f"{self.base_url}/login", data=payload, timeout=self.timeout_s)

    def list_challenges(self) -> list[dict[str, Any]]:
        data = self._get_json("/api/v1/challenges")
        return data.get("data", [])

    def get_challenge(self, challenge_id: int) -> dict[str, Any]:
        data = self._get_json(f"/api/v1/challenges/{challenge_id}")
        return data.get("data", {})

    def get_challenge_files(self, challenge_id: int) -> list[ChallengeFile]:
        data = self._get_json(f"/api/v1/challenges/{challenge_id}/files")
        files = []
        for item in data.get("data", []):
            location = item.get("location") or item.get("url") or ""
            name = item.get("name") or Path(location).name or f"file_{item.get('id', '')}"
            if location.startswith("/"):
                location = f"{self.base_url}{location}"
            files.append(ChallengeFile(name=name, url=location))
        return files

    def download_challenge_files(self, challenge_id: int, dest_dir: Path) -> list[Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        for ctf_file in self.get_challenge_files(challenge_id):
            file_url = f"{self.base_url}/files/{ctf_file.url}"
            data = self.session.get(file_url , timeout=self.timeout_s)
            if not data.ok:
                continue
            out_path = dest_dir / ctf_file.name
            out_path.write_bytes(data.content)
            saved.append(out_path)
        return saved

    def submit_flag(self, challenge_id: int, flag: str) -> dict[str, Any]:
        payload = {"challenge_id": challenge_id, "submission": flag}
        resp = self.session.post(
            f"{self.base_url}/api/v1/challenges/attempt",
            json=payload,
            timeout=self.timeout_s,
        )
        try:
            return resp.json()
        except json.JSONDecodeError:
            return {"success": False, "message": resp.text, "status_code": resp.status_code}

    def _get_json(self, path: str) -> dict[str, Any]:
        resp = self.session.get(f"{self.base_url}{path}", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def _extract_nonce(self, html: str) -> str | None:
        match = re.search(r'name="nonce" value="([^"]+)"', html)
        return match.group(1) if match else None
