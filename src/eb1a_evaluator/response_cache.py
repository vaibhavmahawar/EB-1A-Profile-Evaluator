"""Persistent cache and diagnostics for structured provider responses."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class CacheEntry:
    """A successful parsed provider response stored for deterministic replay."""

    value: dict[str, Any]
    created_at: str
    latency_seconds: float | None
    token_usage: dict[str, Any] | None


class SQLiteResponseCache:
    """Small SQLite cache that stores successful parsed responses only."""

    def __init__(self, path: str | Path, *, refresh: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                cache_key TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                request_type TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                latency_seconds REAL,
                token_usage_json TEXT
            )
            """
        )
        self._connection.commit()
        if refresh:
            self.clear()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def get(self, cache_key: str) -> CacheEntry | None:
        row = self._connection.execute(
            "SELECT response_json, created_at, latency_seconds, token_usage_json "
            "FROM responses WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        try:
            value = json.loads(row[0])
            token_usage = json.loads(row[3]) if row[3] else None
            if not isinstance(value, dict):
                raise ValueError("cached response must be a JSON object")
        except (TypeError, ValueError, json.JSONDecodeError):
            self.delete(cache_key)
            self.misses += 1
            return None
        self.hits += 1
        return CacheEntry(value, row[1], row[2], token_usage)

    def set(
        self,
        cache_key: str,
        *,
        provider: str,
        model: str,
        prompt_version: str,
        request_type: str,
        value: dict[str, Any],
        latency_seconds: float | None = None,
        token_usage: dict[str, Any] | None = None,
    ) -> None:
        self._connection.execute(
            """
            INSERT OR REPLACE INTO responses
            (cache_key, provider, model, prompt_version, request_type,
             response_json, created_at, latency_seconds, token_usage_json)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?, ?)
            """,
            (
                cache_key,
                provider,
                model,
                prompt_version,
                request_type,
                json.dumps(value, sort_keys=True),
                latency_seconds,
                json.dumps(token_usage, sort_keys=True) if token_usage else None,
            ),
        )
        self._connection.commit()
        self.writes += 1

    def delete(self, cache_key: str) -> None:
        self._connection.execute("DELETE FROM responses WHERE cache_key = ?", (cache_key,))
        self._connection.commit()

    def clear(self) -> None:
        self._connection.execute("DELETE FROM responses")
        self._connection.commit()

    def diagnostics(self) -> dict[str, Any]:
        count = self._connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        return {
            "enabled": True,
            "path": str(self.path),
            "entries": count,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLiteResponseCache":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def make_cache_key(
    *,
    provider: str,
    model: str,
    prompt_version: str,
    request_type: str,
    system: str,
    user: str,
) -> str:
    """Create a stable key without storing credentials or raw prompt text."""

    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "request_type": request_type,
            "system": system,
            "user": user,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def timed_call(call: Callable[[], Any]) -> tuple[Any, float]:
    """Run a call and return its result with elapsed seconds."""

    started = time.perf_counter()
    result = call()
    return result, time.perf_counter() - started


def failure_category(error: Exception) -> str:
    """Normalize provider/application exceptions for benchmark reporting."""

    name = type(error).__name__
    message = str(error).lower()
    if name == "ResumeInputError":
        return "input_validation"
    if name in {"AuthenticationError", "PermissionDeniedError"}:
        return "authentication"
    if name in {"RateLimitError", "RateLimitAbortError"}:
        if "tokens per day" in message or "quota" in message:
            return "quota_exhausted"
        return "rate_limit"
    if name in {"APITimeoutError", "TimeoutError"}:
        return "timeout"
    if name == "APIConnectionError":
        return "connection"
    if name in {"InternalServerError", "APIStatusError"}:
        return "provider_server"
    if name in {"ValidationError", "BadRequestError"}:
        return "structured_output"
    if name in {"JSONDecodeError", "ValueError"}:
        return "json_parse"
    if "citation" in message:
        return "citation_validation"
    if "resume evidence" in message:
        return "evidence_validation"
    return "unknown"
