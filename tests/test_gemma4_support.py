"""
GPU-free tests for Gemma-4 support pieces: native tool-call parsing and
HF config architecture detection (used for reasoning-tag/tool-format defaults).
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from common.transformers_utils import HuggingFaceConfig
from endpoints.OAI.types.chat_completion import ChatCompletionMessage, ChatCompletionRequest
from endpoints.OAI.utils.tools import get_toolcall_tags, parse_toolcalls


class Gemma4ToolCallTests(unittest.TestCase):
    def test_tags(self):
        start, end = get_toolcall_tags("gemma4")
        self.assertEqual(start, "<|tool_call>")
        self.assertEqual(end, "<tool_call|>")

    def test_simple_call(self):
        text = (
            "<|tool_call>call:get_weather{"
            'location:<|"|>Paris<|"|>,units:<|"|>metric<|"|>'
            "}<tool_call|>"
        )
        calls = parse_toolcalls(text, "gemma4")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "get_weather")
        self.assertEqual(
            json.loads(calls[0].function.arguments),
            {"location": "Paris", "units": "metric"},
        )

    def test_typed_and_nested_args(self):
        text = (
            "<|tool_call>call:search{"
            'query:<|"|>exllamav2<|"|>,limit:5,verbose:true,'
            'filters:{lang:<|"|>en<|"|>}'
            "}<tool_call|>"
        )
        calls = parse_toolcalls(text, "gemma4")
        self.assertEqual(len(calls), 1)
        args = json.loads(calls[0].function.arguments)
        self.assertEqual(args["query"], "exllamav2")
        self.assertEqual(args["limit"], 5)
        self.assertIs(args["verbose"], True)
        self.assertEqual(args["filters"], {"lang": "en"})

    def test_multiple_calls(self):
        text = (
            "<|tool_call>call:a{x:1}<tool_call|>"
            "<|tool_call>call:b{y:2}<tool_call|>"
        )
        calls = parse_toolcalls(text, "gemma4")
        self.assertEqual([c.function.name for c in calls], ["a", "b"])

    def test_no_calls(self):
        self.assertEqual(parse_toolcalls("plain answer", "gemma4"), [])

    def test_call_inside_thought_channel(self):
        # Tool calls may appear after a thought channel; parsing is tag-based
        text = (
            "<|channel>thought\nI should call the tool.\n<channel|>"
            "<|tool_call>call:ping{}<tool_call|>"
        )
        calls = parse_toolcalls(text, "gemma4")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "ping")
        self.assertEqual(json.loads(calls[0].function.arguments), {})


class Gemma4DetectionTests(unittest.TestCase):
    def test_architectures_parsed(self):
        cfg = HuggingFaceConfig.model_validate(
            {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "text_config": {"max_position_embeddings": 262144},
            }
        )
        self.assertEqual(cfg.architectures, ["Gemma4ForConditionalGeneration"])
        self.assertEqual(cfg.model_type, "gemma4")
        self.assertEqual(cfg.get_max_position_embeddings(), 262144)

    def test_detection_predicate(self):
        # Mirrors the container's Gemma-4 detection
        def is_gemma4(cfg):
            return (cfg.model_type or "").startswith("gemma4") or any(
                "gemma4" in a.lower() for a in (cfg.architectures or [])
            )

        g4 = HuggingFaceConfig.model_validate(
            {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4"}
        )
        other = HuggingFaceConfig.model_validate(
            {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
        )
        self.assertTrue(is_gemma4(g4))
        self.assertFalse(is_gemma4(other))


class ReasoningEffortTests(unittest.IsolatedAsyncioTestCase):
    async def _render(self, request, force_enable_thinking=False):
        from endpoints.OAI.utils import chat_completion as cc

        captured = {}

        async def fake_format(messages, template_vars):
            captured.update(template_vars)
            return "prompt", None, template_vars

        container = SimpleNamespace(force_enable_thinking=force_enable_thinking)
        with (
            patch.object(cc.model, "container", container),
            patch.object(cc, "format_messages_with_template", fake_format),
        ):
            await cc.apply_chat_template(request)
        return captured

    async def test_effort_maps_to_enable_thinking(self):
        for effort, thinking in [
            ("none", False),
            ("minimal", False),
            ("low", True),
            ("medium", True),
            ("high", True),
        ]:
            req = ChatCompletionRequest(
                messages=[ChatCompletionMessage(content="hi")],
                reasoning_effort=effort,
            )
            vars_ = await self._render(req)
            self.assertEqual(vars_["enable_thinking"], thinking, effort)
            self.assertEqual(vars_["reasoning_effort"], effort)

    async def test_effort_overrides_force_enable_thinking(self):
        req = ChatCompletionRequest(
            messages=[ChatCompletionMessage(content="hi")],
            reasoning_effort="none",
        )
        vars_ = await self._render(req, force_enable_thinking=True)
        self.assertIs(vars_["enable_thinking"], False)

    async def test_force_enable_thinking_without_effort(self):
        req = ChatCompletionRequest(messages=[ChatCompletionMessage(content="hi")])
        vars_ = await self._render(req, force_enable_thinking=True)
        self.assertIs(vars_["enable_thinking"], True)
        self.assertNotIn("reasoning_effort", vars_)


if __name__ == "__main__":
    unittest.main()
