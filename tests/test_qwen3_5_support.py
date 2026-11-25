"""
GPU-free tests for Qwen3.5/3.8 support pieces: the HF generation-block Jinja
extension, native pseudo-XML tool-call parsing, HF config architecture
detection, and the reasoning_effort high -> xhigh alias.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from common.templating import PromptTemplate
from common.transformers_utils import HuggingFaceConfig
from endpoints.OAI.types.chat_completion import ChatCompletionMessage, ChatCompletionRequest
from endpoints.OAI.utils.tools import get_toolcall_tags, parse_toolcalls

TC_OPEN = "<tool" + "_call>"
TC_CLOSE = "</tool" + "_call>"
PC = "</para" + "meter>"
FC = "</func" + "tion>"


def _call(name: str, **params) -> str:
    """Build one native Qwen3.5/3.8 pseudo-XML tool call block."""
    parts = [TC_OPEN, f"<function={name}>"]
    for key, value in params.items():
        parts += ["<parameter=" + key + ">", value, PC]
    parts += [FC, TC_CLOSE]
    return "\n".join(parts)


class GenerationBlockTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_block_passthrough(self):
        tpl = PromptTemplate(
            name="gen",
            raw_template="A{% generation %}B{{ x }}C{% endgeneration %}D",
        )
        out = await tpl.render({"x": "!"})
        self.assertEqual(out, "AB!CD")

    async def test_generation_block_whitespace_control(self):
        tpl = PromptTemplate(
            name="gen",
            raw_template="{%- generation %}\nhello\n{%- endgeneration %}\n",
        )
        out = await tpl.render({})
        # trim_blocks eats the newline after the endgeneration tag
        self.assertEqual(out, "hello")

    async def test_nested_in_if_and_loop(self):
        raw = (
            "{%- for m in messages %}"
            "{%- if m.role == 'assistant' %}"
            "{%- generation %}{{ m.content }}{% endgeneration %}"
            "{%- endif %}"
            "{%- endfor %}"
        )
        tpl = PromptTemplate(name="gen", raw_template=raw)
        out = await tpl.render(
            {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}
        )
        self.assertEqual(out, "a")


class Qwen3_5ToolCallTests(unittest.TestCase):
    def test_tags(self):
        start, end = get_toolcall_tags("qwen3_5")
        self.assertEqual(start, TC_OPEN)
        self.assertEqual(end, TC_CLOSE)

    def test_simple_call(self):
        # Exact shape the Qwen3.8 chat template instructs the model to emit
        calls = parse_toolcalls(_call("get_weather", city="Paris"), "qwen3_5")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "get_weather")
        self.assertEqual(json.loads(calls[0].function.arguments), {"city": "Paris"})

    def test_typed_args(self):
        text = _call("search", query="exllamav2", limit="5", verbose="true")
        calls = parse_toolcalls(text, "qwen3_5")
        args = json.loads(calls[0].function.arguments)
        self.assertEqual(args, {"query": "exllamav2", "limit": 5, "verbose": True})

    def test_multiple_calls(self):
        text = _call("a", x="1") + "\n" + _call("b", y="2")
        calls = parse_toolcalls(text, "qwen3_5")
        self.assertEqual([c.function.name for c in calls], ["a", "b"])

    def test_call_after_think_block(self):
        text = "...\nI should call the tool.\n\n" + _call("ping")
        calls = parse_toolcalls(text, "qwen3_5")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "ping")
        self.assertEqual(json.loads(calls[0].function.arguments), {})

    def test_no_calls(self):
        self.assertEqual(parse_toolcalls("plain answer", "qwen3_5"), [])


class Qwen3_5DetectionTests(unittest.TestCase):
    def test_detection_predicate(self):
        # Mirrors the container's Qwen3.5/3.8 detection
        def is_qwen3_5(cfg):
            return (cfg.model_type or "").startswith("qwen3_5") or any(
                "qwen3_5" in a.lower() for a in (cfg.architectures or [])
            )

        q = HuggingFaceConfig.model_validate(
            {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"}
        )
        other = HuggingFaceConfig.model_validate(
            {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
        )
        self.assertTrue(is_qwen3_5(q))
        self.assertFalse(is_qwen3_5(other))


class ReasoningEffortAliasTests(unittest.IsolatedAsyncioTestCase):
    async def _render(self, request, aliases=None):
        from endpoints.OAI.utils import chat_completion as cc

        captured = {}

        async def fake_format(messages, template_vars):
            captured.update(template_vars)
            return "prompt", None, template_vars

        container = SimpleNamespace(force_enable_thinking=False)
        if aliases is not None:
            container.reasoning_effort_aliases = aliases
        with (
            patch.object(cc.model, "container", container),
            patch.object(cc, "format_messages_with_template", fake_format),
        ):
            await cc.apply_chat_template(request)
        return captured

    async def test_high_aliases_to_xhigh(self):
        req = ChatCompletionRequest(
            messages=[ChatCompletionMessage(content="hi")], reasoning_effort="high"
        )
        vars_ = await self._render(req, aliases={"high": "xhigh"})
        self.assertEqual(vars_["reasoning_effort"], "xhigh")
        self.assertIs(vars_["enable_thinking"], True)

    async def test_xhigh_passthrough(self):
        req = ChatCompletionRequest(
            messages=[ChatCompletionMessage(content="hi")], reasoning_effort="xhigh"
        )
        vars_ = await self._render(req, aliases={"high": "xhigh"})
        self.assertEqual(vars_["reasoning_effort"], "xhigh")
        self.assertIs(vars_["enable_thinking"], True)

    async def test_none_disables_thinking(self):
        req = ChatCompletionRequest(
            messages=[ChatCompletionMessage(content="hi")], reasoning_effort="none"
        )
        vars_ = await self._render(req, aliases={"high": "xhigh"})
        self.assertIs(vars_["enable_thinking"], False)

    async def test_no_aliases_keeps_high(self):
        req = ChatCompletionRequest(
            messages=[ChatCompletionMessage(content="hi")], reasoning_effort="high"
        )
        vars_ = await self._render(req)
        self.assertEqual(vars_["reasoning_effort"], "high")
        self.assertIs(vars_["enable_thinking"], True)

    async def test_medium_low_pass_through(self):
        for effort in ("medium", "low"):
            req = ChatCompletionRequest(
                messages=[ChatCompletionMessage(content="hi")], reasoning_effort=effort
            )
            vars_ = await self._render(req, aliases={"high": "xhigh"})
            self.assertEqual(vars_["reasoning_effort"], effort)
            self.assertIs(vars_["enable_thinking"], True)


if __name__ == "__main__":
    unittest.main()