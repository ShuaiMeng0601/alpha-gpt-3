"""Tests for the shared LLM plumbing (alpha_gpt.llm)."""

from types import SimpleNamespace

from alpha_gpt.llm import call_json, extract_json_payload


class _FakeClient:
    """Returns each queued content string on successive create() calls."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        content = self._contents.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_extract_json_payload_plain_object_and_array():
    assert extract_json_payload('{"a": 1}') == {"a": 1}
    assert extract_json_payload("[1, 2, 3]") == [1, 2, 3]


def test_extract_json_payload_handles_code_fences():
    assert extract_json_payload('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_payload('```\n[1, 2]\n```') == [1, 2]


def test_extract_json_payload_recovers_from_surrounding_prose():
    assert extract_json_payload('Sure, here you go: {"x": 1} — done.') == {"x": 1}
    assert extract_json_payload('The reviews are: [{"score": 4}] thanks!') == [{"score": 4}]


def test_extract_json_payload_returns_none_on_garbage():
    assert extract_json_payload("") is None
    assert extract_json_payload(None) is None
    assert extract_json_payload("not json at all") is None


def test_call_json_retries_then_succeeds():
    client = _FakeClient(["garbage, not json", '{"ok": true}'])
    out = call_json(client, "m", "sys", "user", empty_factory=dict, retries=2)
    assert out == {"ok": True}
    assert client.calls == 2  # one bad attempt + one repair attempt


def test_call_json_returns_empty_factory_when_all_attempts_fail():
    client = _FakeClient(["nope", "still nope"])
    assert call_json(client, "m", "sys", "user", empty_factory=list, retries=2) == []
    assert client.calls == 2
