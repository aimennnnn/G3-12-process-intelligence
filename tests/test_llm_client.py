"""Unit tests for upgraded LLMClient: 3-key rotation, tracking, and roles."""
from unittest.mock import MagicMock, patch
import pytest

from src.llm.client import ClientStats, LLMClient, MockLLM, QuotaExhausted, make_llm


def test_client_stats_as_dict():
    stats = ClientStats()
    stats.total_calls = 5
    stats.cached_calls = 2
    stats.api_calls = 3
    stats.prompt_tokens = 300
    stats.completion_tokens = 50
    stats.total_tokens = 350
    stats.total_latency_s = 1.234
    stats.key_rotations = 1

    d = stats.as_dict()
    assert d["total_calls"] == 5
    assert d["cached_calls"] == 2
    assert d["total_tokens"] == 350
    assert d["key_rotations"] == 1
    assert d["total_latency_s"] == 1.234

    # Test differential subtraction
    before = d
    stats.total_calls += 2
    stats.total_tokens += 100
    after = stats.as_dict()
    diff = {k: after[k] - before.get(k, 0) for k in after}
    assert diff["total_calls"] == 2
    assert diff["total_tokens"] == 100


def test_mock_llm_stats():
    mock = MockLLM()
    resp = mock.chat([{"role": "user", "content": "Sample prompt with 20 chars"}])
    assert resp.prompt_tokens > 0
    assert mock.stats.total_calls == 1
    assert mock.stats.total_tokens > 0
    assert "total_calls" in mock.stats.as_dict()


def test_key_rotation_ring():
    keys = ["key_alpha", "key_beta", "key_gamma"]
    client = LLMClient(
        provider="groq",
        model="openai/gpt-oss-20b",
        api_keys=keys,
        requests_per_minute=1000,
    )
    assert client.current_key == "key_alpha"
    client.rotate_key("test rotation 1")
    assert client.current_key == "key_beta"
    assert client.stats.key_rotations == 1

    client.rotate_key("test rotation 2")
    assert client.current_key == "key_gamma"
    assert client.stats.key_rotations == 2

    # Ring wraps around to key_alpha
    client.rotate_key("test rotation 3")
    assert client.current_key == "key_alpha"
    assert client.stats.key_rotations == 3


def test_automatic_failover_on_429():
    keys = ["key_1", "key_2"]
    client = LLMClient(
        provider="groq",
        model="openai/gpt-oss-20b",
        api_keys=keys,
        requests_per_minute=10000,
    )

    # First key returns 429, second key returns 200
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {"retry-after": "0"}
    mock_resp_429.text = "Rate limit hit"

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = {
        "choices": [{"message": {"content": "2"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
    }

    with patch("requests.post", side_effect=[mock_resp_429, mock_resp_200]) as mock_post:
        resp = client.chat([{"role": "user", "content": "test question"}])
        assert resp.text == "2"
        assert client.stats.key_rotations == 1
        assert client.stats.rate_limit_hits == 1
        assert client.current_key == "key_2"
        assert mock_post.call_count == 2


def test_all_keys_exhausted_raises_quota_exhausted():
    keys = ["key_1", "key_2"]
    client = LLMClient(
        provider="groq",
        model="openai/gpt-oss-20b",
        api_keys=keys,
        requests_per_minute=10000,
    )

    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {"retry-after": "0"}
    mock_resp_429.text = "Rate limit hit"

    with patch("requests.post", return_value=mock_resp_429):
        with patch("time.sleep", return_value=None):
            with pytest.raises(QuotaExhausted):
                client.chat([{"role": "user", "content": "exhaustion test"}])


def test_make_llm_roles():
    cfg = {
        "llm": {
            "provider": "mock",
            "model": "base-model",
            "temperature": 0.0,
            "max_tokens": 16,
            "requests_per_minute": 28,
            "timeout_s": 60,
        },
        "optimizer_llm": {
            "model": "opt-model",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
    }
    # Mock LLM provider returns MockLLM regardless of role
    executor = make_llm(cfg, role="executor")
    assert isinstance(executor, MockLLM)

    optimizer = make_llm(cfg, role="optimizer")
    assert isinstance(optimizer, MockLLM)


def test_gemini_client_initialization():
    client = LLMClient(
        provider="gemini",
        model="gemini-3.8-flash",
        api_keys=["gemini_test_key_1", "gemini_test_key_2"],
        requests_per_minute=1000,
    )
    assert client.provider == "gemini"
    assert client.model == "gemini-3.8-flash"
    assert len(client.api_keys) == 2
    assert client.current_key == "gemini_test_key_1"

