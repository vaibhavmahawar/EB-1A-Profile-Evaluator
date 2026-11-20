import sqlite3

from eb1a_evaluator.benchmark_config import BenchmarkConfig
from eb1a_evaluator.response_cache import (
    SQLiteResponseCache,
    failure_category,
    make_cache_key,
)


def test_cache_round_trips_structured_response_without_prompt_text(tmp_path) -> None:
    path = tmp_path / "responses.sqlite3"
    prompt = "candidate secret text that must not be persisted"
    key = make_cache_key(
        provider="groq",
        model="test-model",
        prompt_version="test-v1",
        request_type="criterion",
        system="system",
        user=prompt,
    )
    with SQLiteResponseCache(path) as cache:
        cache.set(
            key,
            provider="groq",
            model="test-model",
            prompt_version="test-v1",
            request_type="criterion",
            value={"rating": "Missing"},
        )
        assert cache.get(key).value == {"rating": "Missing"}
        assert cache.diagnostics()["hits"] == 1

    raw = sqlite3.connect(path).execute("SELECT * FROM responses").fetchone()
    assert prompt not in repr(raw)


def test_cache_malformed_entry_is_ignored(tmp_path) -> None:
    path = tmp_path / "responses.sqlite3"
    key = "bad-entry"
    with SQLiteResponseCache(path) as cache:
        cache.set(
            key,
            provider="groq",
            model="test-model",
            prompt_version="test-v1",
            request_type="criterion",
            value={"valid": True},
        )
        cache._connection.execute(
            "UPDATE responses SET response_json = ? WHERE cache_key = ?",
            ("not-json", key),
        )
        cache._connection.commit()
        assert cache.get(key) is None
        assert cache.diagnostics()["entries"] == 0


def test_failure_category_distinguishes_quota_from_regular_rate_limit() -> None:
    class RateLimitError(Exception):
        pass

    assert failure_category(RateLimitError("tokens per day limit reached")) == "quota_exhausted"
    assert failure_category(RateLimitError("too many requests")) == "rate_limit"


def test_benchmark_config_hash_changes_when_settings_change() -> None:
    first = BenchmarkConfig()
    second = BenchmarkConfig(batch_size=3)
    assert first.config_hash() != second.config_hash()
