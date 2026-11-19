"""
GPU-free unit tests for the ReadyArt fork generation wrapper
(backends/exllamav2/fork_generator.py).

The streaming generator is stubbed out so these run anywhere exllamav2 is
importable; the real TP/MTP paths are exercised by the on-GPU integration
checks.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from backends.exllamav2 import fork_generator as fg
from exllamav2.generator import ExLlamaV2Sampler


def _chunk(text, ids, eos=False, **extra):
    r = {
        "chunk": text,
        "eos": eos,
        "chunk_token_ids": torch.tensor([ids], dtype=torch.long),
    }
    r.update(extra)
    return r


class _StubCache:
    def __init__(self):
        self.current_seq_len = 0
        self.window_beg = 0
        self.max_seq_len = 4096


class _StubTokenizer:
    def __init__(self):
        self.pieces = {10: "Hello", 106: "<turn|>", 11: " world"}

    def get_id_to_piece_list(self, special=True):
        return [self.pieces.get(i, f"<{i}>") for i in range(256)]

    def decode(self, ids, decode_special_tokens=True):
        return ["".join(self.pieces.get(int(t), f"<{t}>") for t in ids.flatten())]


class _StubStreaming:
    """Scripted stand-in for ExLlamaV2StreamingGenerator."""

    last_instance = None

    def __init__(self, model, cache, tokenizer, mtp_drafter=None, num_speculative_tokens=5):
        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        self.mtp_drafter = mtp_drafter
        self.sequence_ids = torch.tensor([[2, 3]])  # prompt prefix
        self.speculative_ngram = False
        self.script = []
        self.ban_log = []
        self.begin_kwargs = None
        _StubStreaming.last_instance = self

    def begin_stream_ex(self, input_ids, settings, **kwargs):
        self.begin_kwargs = kwargs
        self.settings = settings
        self.cache.current_seq_len = input_ids.shape[-1]
        self.sequence_ids = input_ids.clone()
        self._i = 0

    def stream_ex(self, ban_tokens=None):
        self.ban_log.append(ban_tokens)
        r = self.script[self._i]
        self._i += 1
        ids = r["chunk_token_ids"]
        if ids.numel():
            self.sequence_ids = torch.cat([self.sequence_ids, ids], dim=1)
        return r


async def _collect(job):
    return [r async for r in job]


class ForkJobTests(unittest.IsolatedAsyncioTestCase):
    async def _make_gen(self, script):
        with patch.object(fg, "ExLlamaV2StreamingGenerator", _StubStreaming):
            gen = fg.ExLlamaV2ForkGeneratorAsync(
                model=Mock(), cache=_StubCache(), tokenizer=_StubTokenizer()
            )
        _StubStreaming.last_instance.script = script
        return gen

    async def test_stop_token_eos(self):
        gen = await self._make_gen(
            [_chunk("Hello", [10]), _chunk("", [], eos=True)]
        )
        # last generated token is the stop token
        _StubStreaming.last_instance.script[-1]["chunk_token_ids"] = torch.tensor(
            [[106]], dtype=torch.long
        )
        job = fg.ExLlamaV2ForkJobAsync(
            gen,
            input_ids=[torch.tensor([[2, 3]])],
            max_new_tokens=16,
            stop_conditions=[106, "<turn|>"],
            identifier="req1",
        )
        results = await _collect(job)
        await gen.close()

        self.assertEqual(results[0]["text"], "Hello")
        final = results[-1]
        self.assertTrue(final["eos"])
        self.assertEqual(final["eos_reason"], "stop_token")
        self.assertEqual(final["eos_triggering_token_id"], 106)
        self.assertEqual(final["eos_triggering_token_str"], "<turn|>")
        self.assertEqual(final["new_tokens"], 2)
        self.assertEqual(final["prompt_tokens"], 2)
        self.assertEqual(final["cached_tokens"], 0)
        for key in ("time_enqueued", "time_prefill", "time_generate"):
            self.assertIn(key, final)

    async def test_max_new_tokens(self):
        gen = await self._make_gen([_chunk("Hello", [10]), _chunk(" world", [11])])
        job = fg.ExLlamaV2ForkJobAsync(
            gen,
            input_ids=torch.tensor([[2, 3]]),
            max_new_tokens=2,
            stop_conditions=[106],
        )
        results = await _collect(job)
        await gen.close()

        final = results[-1]
        self.assertTrue(final["eos"])
        self.assertEqual(final["eos_reason"], "max_new_tokens")
        self.assertEqual(final["new_tokens"], 2)
        self.assertEqual(final["full_completion"], "Hello world")

    async def test_stop_string_eos(self):
        # eos with a non-stop last token; decoded tail contains the stop string
        gen = await self._make_gen([_chunk("HelloSTOP", [10], eos=True)])
        gen.tokenizer.pieces[10] = "HelloSTOP"
        job = fg.ExLlamaV2ForkJobAsync(
            gen,
            input_ids=torch.tensor([[2, 3]]),
            max_new_tokens=16,
            stop_conditions=[106, "STOP"],
        )
        results = await _collect(job)
        await gen.close()

        final = results[-1]
        self.assertEqual(final["eos_reason"], "stop_string")
        self.assertEqual(final["eos_triggering_string"], "STOP")

    async def test_min_new_tokens_bans_eos(self):
        gen = await self._make_gen(
            [_chunk("Hello", [10]), _chunk(" world", [11]), _chunk("", [], eos=True)]
        )
        job = fg.ExLlamaV2ForkJobAsync(
            gen,
            input_ids=torch.tensor([[2, 3]]),
            max_new_tokens=16,
            min_new_tokens=2,
            stop_conditions=[106],
        )
        results = await _collect(job)
        await gen.close()

        bans = _StubStreaming.last_instance.ban_log
        self.assertEqual(bans[0], [106])
        self.assertEqual(bans[1], [106])
        self.assertIsNone(bans[2])

    async def test_cfg_rejected(self):
        gen = await self._make_gen([])
        with self.assertRaises(ValueError):
            fg.ExLlamaV2ForkJobAsync(
                gen,
                input_ids=[torch.tensor([[2, 3]]), torch.tensor([[2]])],
            )
        await gen.close()

    async def test_embeddings_rejected(self):
        gen = await self._make_gen([])
        with self.assertRaises(ValueError):
            fg.ExLlamaV2ForkJobAsync(
                gen, input_ids=torch.tensor([[2, 3]]), embeddings=[1]
            )
        await gen.close()

    async def test_cancel(self):
        gen = await self._make_gen([_chunk("Hello", [10])])
        job = fg.ExLlamaV2ForkJobAsync(
            gen, input_ids=torch.tensor([[2, 3]]), max_new_tokens=16
        )
        await job.cancel()
        results = await _collect(job)
        await gen.close()
        self.assertTrue(job.cancelled)
        # Cancelled before iteration: the iterator exits immediately
        self.assertEqual(results, [])


class GreedyDetectionTests(unittest.TestCase):
    def test_greedy_top_k1(self):
        s = ExLlamaV2Sampler.Settings()
        s.top_k = 1
        s.top_p = 0
        s.token_repetition_penalty = 1.0
        self.assertTrue(fg._is_greedy(s))

    def test_greedy_with_default_filters(self):
        # Defaults like tfs=1.0 / top_p=0.8 are no-ops at top_k=1 in the C++
        # sampler (single candidate), so this is still greedy.
        s = ExLlamaV2Sampler.Settings()
        s.top_k = 1
        s.token_repetition_penalty = 1.0
        self.assertTrue(fg._is_greedy(s))

    def test_sampling_not_greedy(self):
        s = ExLlamaV2Sampler.Settings()
        s.top_k = 40
        s.top_p = 0.9
        s.token_repetition_penalty = 1.0
        self.assertFalse(fg._is_greedy(s))

    def test_penalties_not_greedy(self):
        s = ExLlamaV2Sampler.Settings()
        s.top_k = 1
        s.token_repetition_penalty = 1.1
        self.assertFalse(fg._is_greedy(s))

    def test_token_bias_not_greedy(self):
        s = ExLlamaV2Sampler.Settings()
        s.top_k = 1
        s.token_repetition_penalty = 1.0
        s.token_bias = torch.zeros(32)
        self.assertFalse(fg._is_greedy(s))


class StopStringMatcherTests(unittest.TestCase):
    def test_full_match(self):
        m = fg._StopStringMatcher(["STOP", "END"])
        emit, matched, hold = m.feed("hello STOP tail")
        self.assertEqual(emit, "hello ")
        self.assertEqual(matched, "STOP")
        self.assertFalse(hold)

    def test_partial_hold(self):
        m = fg._StopStringMatcher(["STOP"])
        emit, matched, hold = m.feed("hello ST")
        self.assertEqual(emit, "")
        self.assertIsNone(matched)
        self.assertTrue(hold)

    def test_no_match(self):
        m = fg._StopStringMatcher(["STOP"])
        emit, matched, hold = m.feed("hello world")
        self.assertEqual(emit, "hello world")
        self.assertIsNone(matched)
        self.assertFalse(hold)

    def test_empty(self):
        m = fg._StopStringMatcher([])
        emit, matched, hold = m.feed("anything")
        self.assertEqual(emit, "anything")
        self.assertIsNone(matched)
        self.assertFalse(hold)


class MPAdapterTests(unittest.TestCase):
    def _pg(self):
        pg = Mock()
        pg.cache = SimpleNamespace(current_seq_len=7, max_seq_len=4096)
        pg.config = SimpleNamespace(max_seq_len=4096)
        pg.forward = Mock(return_value=torch.zeros(1, 1, 8))
        return pg

    def test_forward_translates(self):
        pg = self._pg()
        adapter = fg._MPModelAdapter(pg)
        ids = torch.tensor([[1, 2, 3]])
        out = adapter.forward(ids, cache=None)
        pg.forward.assert_called_once()
        args = pg.forward.call_args[0]
        self.assertEqual(args[1], 7)  # past_len from master cache
        self.assertEqual(args[2], 3)  # q_len
        self.assertEqual(args[3], 1)  # batch
        self.assertIsNotNone(out)

    def test_preprocess_only(self):
        pg = self._pg()
        adapter = fg._MPModelAdapter(pg)
        out = adapter.forward(torch.tensor([[1, 2]]), cache=None, preprocess_only=True)
        self.assertIsNone(out)

    def test_empty_forward(self):
        pg = self._pg()
        adapter = fg._MPModelAdapter(pg)
        out = adapter.forward(torch.empty((1, 0), dtype=torch.long), cache=None)
        self.assertIsNone(out)
        pg.forward.assert_not_called()

    def test_loras_rejected(self):
        adapter = fg._MPModelAdapter(self._pg())
        with self.assertRaises(ValueError):
            adapter.forward(torch.tensor([[1]]), cache=None, loras=[object()])

    def test_cache_proxy_fans_out(self):
        pg = self._pg()
        # Simulate the distributed setter updating the master cache
        pg.mtp_set_seq_len.side_effect = lambda n: setattr(pg.cache, "current_seq_len", n)
        proxy = fg._MPCacheProxy(pg)
        self.assertEqual(proxy.current_seq_len, 7)
        self.assertEqual(proxy.max_seq_len, 4096)
        proxy.current_seq_len = 0
        pg.mtp_set_seq_len.assert_called_once_with(0)
        proxy.current_seq_len -= 1
        pg.mtp_set_seq_len.assert_called_with(-1)


if __name__ == "__main__":
    unittest.main()
