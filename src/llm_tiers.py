from __future__ import annotations

from dataclasses import dataclass
import json

from src.config import settings


@dataclass(frozen=True)
class ModelTier:
    provider: str
    model: str
    max_tokens: int | None = None


def load_tiers(role: str) -> list[ModelTier]:
    if role not in {"planner", "executor"}:
        raise ValueError(f"Unknown role: {role}")
    raw = settings.planner_tiers if role == "planner" else settings.executor_tiers
    default_provider = settings.planner_provider if role == "planner" else settings.executor_provider
    default_model = settings.planner_model if role == "planner" else settings.executor_model
    default_max = settings.planner_max_tokens if role == "planner" else settings.executor_max_tokens
    tiers = _parse_tiers(raw, default_provider, default_model, default_max)
    return tiers


def select_tier_index(state: dict, tiers: list[ModelTier], role: str) -> int:
    if not tiers:
        return 0
    if len(tiers) == 1:
        return 0

    phase_cycles = int(state.get("phase_cycles", 0) or 0)
    tool_calls = int(state.get("tool_calls", 0) or 0)
    pivots = int(state.get("category_pivots", 0) or 0)
    failures = _recent_failures(state, count=settings.tier_failure_window)

    tier_max = len(tiers) - 1
    phase_score = _score_threshold(phase_cycles, _thresholds(settings.tier_phase_cycles, tier_max))
    tool_score = _score_threshold(tool_calls, _thresholds(settings.tier_tool_calls, tier_max))
    pivot_score = _score_threshold(pivots, _thresholds(settings.tier_category_pivots, tier_max))
    fail_score = _score_threshold(failures, _thresholds(settings.tier_recent_failures, tier_max))
    return min(max(phase_score, tool_score, pivot_score, fail_score), tier_max)


def _recent_failures(state: dict, count: int = 3) -> int:
    attempts = state.get("attempt_history") or []
    recent = attempts[-count:] if count else attempts
    return sum(1 for a in recent if int(a.get("exit_code", 0) or 0) != 0)


def _score_threshold(value: int, thresholds: list[int]) -> int:
    score = 0
    for idx, threshold in enumerate(thresholds, start=1):
        if value >= threshold:
            score = idx
    return score


def _thresholds(raw: str, count: int) -> list[int]:
    if count <= 0:
        return []
    values = []
    if raw:
        for part in raw.replace(" ", "").split(","):
            if not part:
                continue
            try:
                values.append(int(part))
            except ValueError:
                continue
    if not values:
        values = [999999] * count
    if len(values) < count:
        values.extend([values[-1]] * (count - len(values)))
    return values[:count]


def _parse_tiers(
    raw: str,
    default_provider: str,
    default_model: str,
    default_max_tokens: int | None,
) -> list[ModelTier]:
    if not raw:
        return [ModelTier(default_provider, default_model, default_max_tokens)]

    raw = raw.strip()
    tiers: list[ModelTier] = []
    data = None

    if raw.startswith("[") or raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tier JSON: {exc}") from exc

    if data is not None:
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    tiers.append(_parse_tier_token(item, default_provider, default_max_tokens))
                elif isinstance(item, dict):
                    provider = item.get("provider", default_provider)
                    model = item.get("model", default_model)
                    max_tokens = item.get("max_tokens", default_max_tokens)
                    tiers.append(ModelTier(provider, model, max_tokens))
        if tiers:
            return tiers

    # Comma-separated fallback: provider:model@max_tokens or model@max_tokens
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        tiers.append(_parse_tier_token(token, default_provider, default_max_tokens))

    if not tiers:
        tiers = [ModelTier(default_provider, default_model, default_max_tokens)]
    return tiers


def _parse_tier_token(token: str, default_provider: str, default_max_tokens: int | None) -> ModelTier:
    max_tokens = default_max_tokens
    if "@" in token:
        token, max_tok = token.rsplit("@", 1)
        try:
            max_tokens = int(max_tok)
        except ValueError:
            max_tokens = default_max_tokens

    if ":" in token:
        provider, model = token.split(":", 1)
    else:
        provider, model = default_provider, token

    return ModelTier(provider.strip(), model.strip(), max_tokens)
