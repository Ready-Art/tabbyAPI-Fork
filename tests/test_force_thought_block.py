"""GPU-free tests for force_thought_block: prefill the reasoning opener so finetunes
that don't spontaneously open the thought channel are forced into it (Gemma-4)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from endpoints.OAI.types.chat_completion import ChatCompletionMessage, ChatCompletionRequest


def _container(force_thought_block=True, reasoning=True):
    return SimpleNamespace(
        force_thought_block=force_thought_block,
        force_enable_thinking=False,
        reasoning=reasoning,
        reasoning_start_token="<|channel>",
        reasoning_end_token="<channel|>",
        reasoning_suppress_header="thought\n",
        reasoning_effort_aliases={},
    )


class ForceThoughtBlockTests(unittest.IsolatedAsyncioTestCase):
    async def _render(self, prompt, container, template_vars=None):
        from endpoints.OAI.utils import chat_completion as cc

        async def fake_format(messages, tv):
            return prompt, None, dict(tv)

        req = ChatCompletionRequest(
            messages=[ChatCompletionMessage(content="hi")], add_generation_prompt=True
        )
        if template_vars:
            req.template_vars.update(template_vars)
        with (
            patch.object(cc.model, "container", container),
            patch.object(cc, "format_messages_with_template", fake_format),
        ):
            out, _ = await cc.apply_chat_template(req)
        return out

    async def test_appends_opener_on_normal_turn(self):
        from endpoints.OAI.utils import chat_completion as cc

        prompt = "<|turn>user\nhi<|turn>\n<|turn>model\n"
        out = await self._render(prompt, _container())
        self.assertTrue(out.endswith("<|channel>thought\n"))
        with patch.object(cc.model, "container", _container()):
            self.assertTrue(cc._start_in_reasoning_mode(out))

    async def test_no_append_when_thinking_disabled(self):
        prompt = "<|turn>model\n<|channel>thought\n<channel|>"
        out = await self._render(prompt, _container(), {"enable_thinking": False})
        self.assertEqual(out, prompt)  # unchanged

    async def test_no_duplicate_when_already_open(self):
        prompt = "<|turn>model\n<|channel>thought\n"  # e.g. after a tool response
        out = await self._render(prompt, _container())
        self.assertEqual(out, prompt)  # not doubled

    async def test_no_append_when_disabled(self):
        prompt = "<|turn>model\n"
        out = await self._render(prompt, _container(force_thought_block=False))
        self.assertEqual(out, prompt)

    async def test_no_append_when_reasoning_off(self):
        prompt = "<|turn>model\n"
        out = await self._render(prompt, _container(reasoning=False))
        self.assertEqual(out, prompt)


if __name__ == "__main__":
    unittest.main()