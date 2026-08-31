from types import SimpleNamespace
import unittest

from alpha_gpt.debate.agents import DebateAgent


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class _FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.contents.pop(0))


class _FakeClient:
    def __init__(self, contents):
        self.chat = SimpleNamespace(completions=_FakeCompletions(contents))


class DebateAgentJsonTest(unittest.TestCase):
    def test_call_json_repairs_invalid_json_once(self):
        client = _FakeClient([
            "this is not json",
            '{"title": "Recovered", "mechanism": "Works"}',
        ])
        agent = DebateAgent(
            name="Momentum",
            system_prompt="system",
            client=client,
            model="fake-model",
            json_retries=2,
        )

        payload = agent._call_json("original prompt", empty_factory=dict)

        self.assertEqual(payload["title"], "Recovered")
        self.assertEqual(len(client.chat.completions.calls), 2)
        self.assertIn("Rewrite the answer as valid JSON only", client.chat.completions.calls[1]["messages"][1]["content"])

