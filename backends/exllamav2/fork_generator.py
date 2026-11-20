"""
Async generation wrapper for the ReadyArt exllamav2 fork's TP/MTP paths.

The fork's dynamic (paged) generator has no sliding-window support in its
tensor-parallel attention kernel, and the multiprocess backend (TPProcessGroup)
is not a drop-in ExLlamaV2 model. Those paths are driven instead by:

  * SP: the fork's ExLlamaV2StreamingGenerator on a load_tp/load_tp_gen model,
    optionally with a Gemma4AssistantDrafter (mtp_drafter=...).
  * MP: a TPProcessGroup (one worker process per GPU). Plain decoding reuses the
    streaming generator through a model/cache adapter; greedy MTP uses the
    fork's MTPMPDriver (graphed verify window).

This module exposes the same job/result interface the container consumes from
ExLlamaV2DynamicGeneratorAsync / ExLlamaV2DynamicJobAsync: jobs are async
iterables yielding dicts with stage="streaming", and the final dict carries
eos + metrics (prompt_tokens, cached_tokens, time_prefill, ...).

Only one job runs at a time (batch size 1): these paths own a single cache.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import List, Optional

import numpy as np
import torch

from common.logger import xlogger
from exllamav2 import ExLlamaV2Tokenizer
from exllamav2.ext import exllamav2_ext as ext_c
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator
from exllamav2.generator.filters import ExLlamaV2Filter


def _is_greedy(settings: ExLlamaV2Sampler.Settings) -> bool:
    """
    True when sampling reduces to argmax of the raw logits, i.e. when the
    greedy-only MP MTP driver produces exactly what the sampler would pick.

    Mirrors the fork's C++ sampler: top_k == 1 leaves a single candidate, so
    top_p/top_a/min_p/tfs/typical/mirostat/temperature are all no-ops (each is
    guarded by num_candidates > 1, and softmax/temperature are monotonic).
    Only logit modifiers applied before top_k (penalties, bias, DRY, XTC,
    filters, CFG) or post-sampling hooks can change the result.
    """

    return (
        settings.top_k == 1
        and settings.token_bias is None
        and settings.token_repetition_penalty == 1.0
        and settings.token_frequency_penalty == 0.0
        and settings.token_presence_penalty == 0.0
        and settings.dry_multiplier == 0.0
        and settings.xtc_probability == 0.0
        and settings.cfg_scale is None
        and not settings.post_sampling_hooks
    )


def _strings_to_utf32(strings: List[str]):
    """UTF-32 buffer + offsets for ext_c.partial_strings_match (see streaming.py)."""

    if not strings:
        return bytearray(), None

    encoded = [s.encode("utf-32-le") for s in strings]
    lengths = [len(s) for s in encoded]
    offsets = [0] + lengths
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i - 1]
    buf = bytearray(offsets[-1])
    for s, off in zip(encoded, offsets[:-1]):
        buf[off : off + len(s)] = s
    return (
        np.frombuffer(buf, dtype=np.uint8),
        np.frombuffer(np.array(offsets, dtype=np.int32), dtype=np.uint8),
    )


class _StopStringMatcher:
    """
    Incremental stop-string matching with partial-match hold, mirroring the
    streaming generator's behavior for the MP MTP driver path.
    """

    def __init__(self, stop_strings: List[str]):
        self.stop_strings = stop_strings
        self.buf, self.offsets = _strings_to_utf32(stop_strings)

    def feed(self, text: str):
        """
        Returns (emit_text, matched_string_or_None, hold: bool).
        hold=True means the text ends in a partial stop-string match and must
        be retained until more output resolves it.
        """

        if self.offsets is None or not text:
            return text, None, False

        # partial_strings_match returns the TEXT POSITION of a full match
        # (or -2 for a partial match at the tail), not a string index.
        match = ext_c.partial_strings_match(
            np.frombuffer(text.encode("utf-32-le"), dtype=np.uint8),
            self.offsets,
            self.buf,
        )
        if match >= 0:
            matched = next(
                (s for s in self.stop_strings if text[match:].startswith(s)), None
            )
            return text[:match], matched, False
        if match == -2:
            return "", None, True
        return text, None, False


class _MPModelAdapter:
    """
    Presents a TPProcessGroup as the model object expected by the streaming
    generator: every forward becomes a distributed collective. The group's
    master rank cache advances in lockstep with the workers, so cache
    bookkeeping stays consistent through _MPCacheProxy.
    """

    def __init__(self, tp_group):
        self.pg = tp_group
        self.config = tp_group.config

    def forward(
        self,
        input_ids: torch.Tensor,
        cache,
        preprocess_only: bool = False,
        last_id_only: bool = False,
        loras=None,
        input_mask=None,
        position_offsets=None,
        abort_event=None,
        indexed_embeddings=None,
        **kwargs,
    ):
        if loras:
            raise ValueError("LoRAs are not supported with the MP tensor-parallel backend")
        if input_mask is not None or position_offsets is not None:
            raise ValueError("input_mask/position_offsets are not supported with the MP backend")
        if indexed_embeddings is not None:
            raise ValueError("Multimodal embeddings are not supported with the MP backend")
        if kwargs.get("return_last_state"):
            raise ValueError("return_last_state is not supported through the MP adapter")

        q_len = input_ids.shape[-1]
        if q_len == 0:
            return None

        logits = self.pg.forward(
            input_ids.cpu(),
            self.pg.cache.current_seq_len,
            q_len,
            1,
            last_id_only,
        )
        return None if preprocess_only else logits


class _MPCacheProxy:
    """
    Cache facade for the streaming generator in MP mode. Reads delegate to the
    master rank cache; writes to current_seq_len fan out to every worker so
    rewinds (token healing, banned strings) stay distributed-consistent. Other
    attribute writes (e.g. window_beg) go to the master rank cache.
    """

    def __init__(self, tp_group):
        object.__setattr__(self, "_pg", tp_group)

    @property
    def current_seq_len(self):
        return self._pg.cache.current_seq_len

    def __setattr__(self, name, value):
        if name == "current_seq_len":
            self._pg.mtp_set_seq_len(int(value))
        else:
            setattr(self._pg.cache, name, value)

    def __getattr__(self, name):
        return getattr(self._pg.cache, name)


class ExLlamaV2ForkGeneratorAsync:
    """
    Async wrapper running fork TP/MTP jobs sequentially on a worker thread.
    See module docstring for the supported paths.
    """

    def __init__(
        self,
        model,
        cache,
        tokenizer: ExLlamaV2Tokenizer,
        tp_group=None,
        mtp_drafter=None,
        mtp_window: int = 6,
    ):
        self.tokenizer = tokenizer
        # Append-only prompt cache (MP MTP): token ids whose K/V is live in the cache.
        # The SWA ring is not rewind-safe, so reuse is restricted to prompts that
        # strictly EXTEND this sequence (chat continuations); anything else resets.
        self._cached_ids = None
        self._cached_pos = 0
        self._cache_valid = False
        self.tp_group = tp_group
        self.mtp_drafter = mtp_drafter
        self.mtp_window = mtp_window
        self.mp = tp_group is not None

        if self.mp:
            self.streaming = ExLlamaV2StreamingGenerator(
                _MPModelAdapter(tp_group), _MPCacheProxy(tp_group), tokenizer
            )
        else:
            self.streaming = ExLlamaV2StreamingGenerator(
                model,
                cache,
                tokenizer,
                mtp_drafter=mtp_drafter,
                num_speculative_tokens=mtp_window,
            )

        # Shim for container access patterns (generator.generator.*)
        self.generator = self
        self.max_batch_size = 1
        self.current_loras = []
        self.speculative_ngram = False

        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._run())
        self._closed = False

    def set_loras(self, loras: list):
        if self.mp and loras:
            raise ValueError("LoRAs are not supported with the MP tensor-parallel backend")
        self.current_loras = loras

    def enqueue(self, job: "ExLlamaV2ForkJobAsync"):
        self._queue.put_nowait(job)

    async def _run(self):
        while True:
            job = await self._queue.get()
            if job is None:
                break
            if not job.cancelled:
                try:
                    await asyncio.to_thread(self._run_job, job)
                except Exception as e:
                    await job.put_result(e)
            self._queue.task_done()

    async def close(self):
        if self._closed:
            return
        self._closed = True

        # Abort the running job, cancel anything still queued
        job = getattr(self, "_current_job", None)
        if job is not None:
            self._signal_abort(job)
        while not self._queue.empty():
            queued = self._queue.get_nowait()
            if queued is not None and not queued.cancelled:
                await queued.cancel()
            self._queue.task_done()

        await self._queue.put(None)
        try:
            await self._worker
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------ job execution

    def _put(self, job, result):
        if not job.cancelled:
            self._loop.call_soon_threadsafe(job.queue.put_nowait, result)

    def _signal_abort(self, job):
        """Set BOTH abort signals: the job-level event (checked per emitted token) and
        the MP prefill flag (checked per prefill chunk on every rank). Without the
        latter, a disconnected 80k prompt keeps prefilling for minutes after the
        request is gone -- the token loop never runs until prefill finishes."""
        job.abort_event.set()
        pg = getattr(self, "tp_group", None)
        af = getattr(pg, "abort_flag", None)
        if af is not None:
            af.set()

    def _run_job(self, job):
        self._current_job = job
        try:
            if self.mp and self.mtp_drafter is not None and self._mp_mtp_eligible(job):
                self._run_job_mp_mtp(job)
            else:
                self._run_job_streaming(job)
        finally:
            self._current_job = None

    def _mp_mtp_eligible(self, job) -> bool:
        # BUG 104: MP MTP now supports sampling (exact-resync from the target's verified
        # logits, mirroring SP), so non-greedy requests no longer fall back to plain MP.
        if job.filters or job.banned_strings:
            xlogger.warning(
                "Filters/banned strings are not supported by MP MTP; "
                "using plain MP decoding for this request"
            )
            return False
        return True

    def _fresh_cache(self, cache):
        """
        Force a full prefill per job. Cross-job K/V reuse would rewind the SWA
        ring's window_beg bookkeeping, which is not rewind-safe on these paths.
        """

        cache.current_seq_len = 0
        if hasattr(cache, "window_beg"):
            cache.window_beg = 0

    def _run_job_streaming(self, job):
        g = self.streaming
        settings = job.gen_settings

        g.speculative_ngram = self.speculative_ngram and self.mtp_drafter is None
        g.sequence_ids = None
        if self.mp:
            # Full distributed reset (current_seq_len + window_beg on every rank,
            # plus stale MTP gather buffers) -- invalidates the MP MTP prompt cache
            self.tp_group.mtp_reset_cache()
            self._cache_valid = False
        else:
            self._fresh_cache(g.cache)

        stop_tokens = {t for t in job.stop_conditions if isinstance(t, int)}
        stop_strings = [t for t in job.stop_conditions if isinstance(t, str)]

        time_start = time.perf_counter()
        g.begin_stream_ex(
            job.input_ids,
            settings,
            token_healing=job.token_healing,
            loras=self.current_loras or None,
            return_probabilities=job.return_probs,
            return_top_tokens=job.return_top_tokens,
            return_logits=job.return_logits,
            abort_event=job.abort_event,
            decode_special_tokens=job.decode_special_tokens,
            banned_strings=job.banned_strings,
            filters=job.filters,
            filter_prefer_eos=job.filter_prefer_eos,
        )
        time_prefill = time.perf_counter() - time_start

        new_tokens = 0
        full_text = ""
        stop_string = None
        time_gen_start = time.perf_counter()

        while new_tokens < job.max_new_tokens:
            if job.abort_event.is_set():
                break

            ban = list(stop_tokens) if new_tokens < job.min_new_tokens else None
            r = g.stream_ex(ban_tokens=ban)

            chunk = r["chunk"]
            chunk_ids = r["chunk_token_ids"]
            n = chunk_ids.shape[-1] if chunk_ids is not None else 0
            new_tokens += n
            full_text += chunk

            result = {
                "stage": "streaming",
                "identifier": job.identifier,
                "text": chunk,
                "token_ids": chunk_ids,
                "eos": False,
            }
            if job.return_probs and "probs" in r:
                result["token_probs"] = r["probs"]
            if job.return_top_tokens:
                result["top_k_tokens"] = r.get("top_tokens")
                result["top_k_probs"] = r.get("top_probs")
            if job.return_logits and "logits" in r:
                result["logits"] = r["logits"]

            if r["eos"]:
                last_tok = int(g.sequence_ids[0, -1].item())
                if last_tok in stop_tokens:
                    reason = "stop_token"
                else:
                    # The stop string itself is excluded from the emitted text;
                    # look for it in the decoded tail of the full sequence.
                    tail = self.tokenizer.decode(
                        g.sequence_ids[:, -64:], decode_special_tokens=True
                    )[0]
                    matched = next((s for s in stop_strings if s in tail), None)
                    if matched is not None:
                        reason = "stop_string"
                        stop_string = matched
                    elif job.filters:
                        reason = "end_filter"
                    else:
                        reason = "stop_token"
                self._finish(
                    job,
                    reason,
                    new_tokens,
                    full_text,
                    time_start,
                    time_prefill,
                    time_gen_start,
                    stop_token=last_tok if reason == "stop_token" else None,
                    stop_string=stop_string if reason == "stop_string" else None,
                    extra_text=chunk,
                    extra_ids=chunk_ids,
                )
                return

            self._put(job, result)

        self._finish(
            job,
            "max_new_tokens",
            new_tokens,
            full_text,
            time_start,
            time_prefill,
            time_gen_start,
        )

    def _run_job_mp_mtp(self, job):
        try:
            self._run_job_mp_mtp_impl(job)
        finally:
            self._record_cache_state(job)

    def _record_cache_state(self, job):
        """Snapshot (prompt + every emitted token) and the live cache position.

        The cache position can differ from the snapshot length by a SMALL amount:
        at rest it is len-1 (the last round's bonus is emitted but K/V-committed
        by the next verify); a mid-round stop leaves up to W extra committed draft
        tokens past the emitted prefix. Both are reusable -- see the geometric
        check in _run_job_mp_mtp_impl. Large deviations (aborted prefill) fail the
        prefix/overshoot checks there and fall back to a full re-prefill.
        """
        pg = self.tp_group
        ids = getattr(self, "_job_prompt_ids", None)
        all_ids = getattr(self, "_job_all_ids", None)
        self._job_prompt_ids = None
        self._job_all_ids = None
        if ids is None or all_ids is None or pg is None:
            self._cache_valid = False
            return
        if all_ids:
            full = torch.cat([ids, torch.tensor([all_ids], dtype=torch.long)], dim=-1)
        else:
            full = ids
        self._cached_ids = full.contiguous()
        self._cached_pos = pg.cache.current_seq_len
        self._cache_valid = True

    def _run_job_mp_mtp_impl(self, job):
        from exllamav2.mtp_mp import MTPMPDriver

        pg = self.tp_group
        settings = job.gen_settings
        stop_tokens = {t for t in job.stop_conditions if isinstance(t, int)}
        matcher = _StopStringMatcher([t for t in job.stop_conditions if isinstance(t, str)])

        # Append-only prompt cache: a chat continuation shares the entire previous
        # context as a prefix, so keep the K/V and prefill only the new suffix instead
        # of reprocessing tens of thousands of tokens. The SWA ring's window_beg
        # bookkeeping is not rewind-safe, so reuse is allowed ONLY for a strict
        # extension of the cached sequence; any divergence falls back to full reset.
        ids_cpu = job.input_ids.cpu()
        self._job_prompt_ids = ids_cpu
        n_prompt = ids_cpu.shape[-1]
        cached = self._cached_ids if self._cache_valid else None
        reuse = False
        prefill_ids = ids_cpu
        if cached is not None:
            clen = cached.shape[-1]
            m = min(clen, n_prompt)
            prefix_ok = torch.equal(ids_cpu[:, :m], cached[:, :m])
            cpos = self._cached_pos
            # Reuse only when the new prompt strictly EXTENDS the snapshot and the
            # cache overshoot is small: -1 (at rest -- the last bonus is emitted but
            # K/V-committed by the next verify) up to a few extra committed drafts
            # (mid-round stop). Trimming those few tokens is SWA-ring-safe: the
            # 1280-deep ring still holds every position below the trim point, and
            # swa_len truncation ignores the tail slots. Large rewinds (edited or
            # re-rendered history) are NOT ring-safe -> full reset + re-prefill.
            overshoot = cpos - clen
            start = min(cpos, clen)
            if (prefix_ok and m == clen and n_prompt > start
                    and -1 <= overshoot <= 8 and clen >= pg.cache.window_beg):
                reuse = True
                if cpos > clen:
                    pg.mtp_set_seq_len(clen)   # distributed trim of unemitted drafts
                pg.abort_flag.clear()
                prefill_ids = ids_cpu[:, start:]
                xlogger.info(
                    f"MP MTP prompt cache hit: reusing {start} tokens, "
                    f"prefilling {n_prompt - start} new"
                )
            else:
                if prefix_ok:
                    why = (f"cache overshoot {overshoot} outside [-1, 8]"
                           if not (-1 <= overshoot <= 8)
                           else "prompt does not extend the cached sequence")
                else:
                    diff = (ids_cpu[0, :m] != cached[0, :m]).nonzero()
                    at = int(diff[0]) if diff.numel() else m
                    why = f"history diverges at token {at} (re-rendered/edited prompt?)"
                xlogger.info(f"MP MTP prompt cache miss: {why}; full re-prefill")
        if not reuse:
            pg.mtp_reset_cache()
        driver = MTPMPDriver(pg, self.mtp_drafter, window=self.mtp_window)

        time_start = time.perf_counter()
        logits, state = pg.forward_state(prefill_ids, last_id_only=True, cpu_logits=True)
        time_prefill = time.perf_counter() - time_start
        if logits is None or job.abort_event.is_set():
            # Client disconnected during prefill: every rank aborted at the same chunk
            # boundary (collective flag in model.forward). Stop instead of generating.
            self._cache_valid = False
            self._finish(job, "abort", 0, "", time_start, time_prefill, time_start)
            return

        # begin() takes the FULL prompt: its token history and heal base use absolute
        # cache positions. logits/state are the last row of the suffix forward, which
        # is the last prompt token on both the reuse and the fresh-prefill paths.
        driver.begin(job.input_ids, logits, state)

        gen_ids: List[int] = []
        all_ids: List[int] = []   # every consumed token, incl. suppressed stop tokens
        self._job_all_ids = all_ids
        seq_ids = job.input_ids.clone()
        emitted = ""
        new_tokens = 0
        time_gen_start = time.perf_counter()

        def decode_full() -> str:
            if not gen_ids:
                return ""
            return self.tokenizer.decode(
                torch.tensor([gen_ids], dtype=torch.long),
                decode_special_tokens=job.decode_special_tokens,
            )[0]

        def pending_text() -> str:
            full = decode_full()
            return full[len(emitted) :] if full.startswith(emitted) else ""

        while new_tokens < job.max_new_tokens:
            if job.abort_event.is_set():
                break

            ban = list(stop_tokens) if new_tokens < job.min_new_tokens else None
            tok = int(driver.step(settings, self.tokenizer, seq_ids, ban_tokens=ban))
            all_ids.append(tok)
            new_tokens += 1
            seq_ids = torch.cat([seq_ids, torch.tensor([[tok]], dtype=torch.long)], dim=-1)

            if tok in stop_tokens:
                if new_tokens > job.min_new_tokens:
                    flush = pending_text()
                    self._finish(
                        job,
                        "stop_token",
                        new_tokens,
                        emitted + flush,
                        time_start,
                        time_prefill,
                        time_gen_start,
                        stop_token=tok,
                        extra_text=flush,
                    )
                    return
                # Below min_new_tokens: keep generating, don't emit the stop token
                continue

            gen_ids.append(tok)
            full = decode_full()
            if not full.startswith(emitted):
                # Unstable prefix (multi-byte character in progress); wait
                continue
            emit_text, matched, hold = matcher.feed(full[len(emitted) :])
            if matched is not None:
                self._finish(
                    job,
                    "stop_string",
                    new_tokens,
                    emitted + emit_text,
                    time_start,
                    time_prefill,
                    time_gen_start,
                    stop_string=matched,
                    extra_text=emit_text,
                )
                return
            if hold:
                continue
            if emit_text:
                emitted += emit_text
                self._put(
                    job,
                    {
                        "stage": "streaming",
                        "identifier": job.identifier,
                        "text": emit_text,
                        "token_ids": torch.tensor([[tok]], dtype=torch.long),
                        "eos": False,
                    },
                )

        flush = pending_text()
        self._finish(
            job,
            "max_new_tokens",
            new_tokens,
            emitted + flush,
            time_start,
            time_prefill,
            time_gen_start,
            extra_text=flush,
        )

    def _finish(
        self,
        job,
        reason: str,
        new_tokens: int,
        full_text: str,
        time_start: float,
        time_prefill: float,
        time_gen_start: float,
        stop_token: Optional[int] = None,
        stop_string: Optional[str] = None,
        extra_text: str = "",
        extra_ids: Optional[torch.Tensor] = None,
    ):
        result = {
            "stage": "streaming",
            "identifier": job.identifier,
            "text": extra_text,
            "token_ids": extra_ids if extra_ids is not None else torch.empty((1, 0), dtype=torch.long),
            "eos": True,
            "eos_reason": reason,
            "full_completion": full_text,
            "new_tokens": new_tokens,
            "prompt_tokens": job.input_ids.shape[-1],
            "cached_tokens": 0,
            "time_enqueued": time_start - job.time_enqueue,
            "time_prefill": time_prefill,
            "time_generate": time.perf_counter() - time_gen_start,
        }
        if reason == "stop_token" and stop_token is not None:
            result["eos_triggering_token_id"] = stop_token
            result["eos_triggering_token_str"] = self.tokenizer.get_id_to_piece_list(True)[stop_token]
        if reason == "stop_string" and stop_string is not None:
            result["eos_triggering_string"] = stop_string
        self._put(job, result)


class ExLlamaV2ForkJobAsync:
    """
    Async job for ExLlamaV2ForkGeneratorAsync. Accepts the same kwargs as
    ExLlamaV2DynamicJobAsync so the container can use either interchangeably.
    """

    def __init__(
        self,
        generator: ExLlamaV2ForkGeneratorAsync,
        input_ids,
        max_new_tokens: int = 16,
        min_new_tokens: int = 0,
        gen_settings: Optional[ExLlamaV2Sampler.Settings] = None,
        stop_conditions: Optional[list] = None,
        decode_special_tokens: bool = True,
        filters: Optional[List[ExLlamaV2Filter]] = None,
        filter_prefer_eos: bool = False,
        return_probs: bool = False,
        return_top_tokens: int = 0,
        return_logits: bool = False,
        banned_strings: Optional[List[str]] = None,
        token_healing: bool = False,
        identifier: Optional[str] = None,
        embeddings=None,
        **kwargs,
    ):
        if isinstance(input_ids, (list, tuple)):
            if len(input_ids) > 1:
                raise ValueError("CFG is not supported by the fork TP/MTP paths")
            input_ids = input_ids[0]
        if embeddings:
            raise ValueError("Multimodal embeddings are not supported by the fork TP/MTP paths")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] != 1:
            raise ValueError("Fork TP/MTP paths support batch size 1 only")

        self.generator = generator
        self.input_ids = input_ids
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.gen_settings = gen_settings or ExLlamaV2Sampler.Settings()
        self.stop_conditions = stop_conditions or []
        self.decode_special_tokens = decode_special_tokens
        self.filters = filters
        self.filter_prefer_eos = filter_prefer_eos
        self.return_probs = return_probs
        self.return_top_tokens = return_top_tokens
        self.return_logits = return_logits
        self.banned_strings = banned_strings
        self.token_healing = token_healing
        self.identifier = identifier

        self.queue: asyncio.Queue = asyncio.Queue()
        self.cancelled = False
        self.abort_event = threading.Event()
        self.time_enqueue = time.perf_counter()
        generator.enqueue(self)

    async def put_result(self, result):
        await self.queue.put(result)

    async def __aiter__(self):
        while True:
            if self.cancelled:
                break
            result = await self.queue.get()
            if isinstance(result, Exception):
                raise result
            yield result
            if result["eos"]:
                break

    async def cancel(self):
        self.cancelled = True
        self.generator._signal_abort(self)
        self.queue.put_nowait(
            {
                "stage": "streaming",
                "identifier": self.identifier,
                "text": "",
                "token_ids": torch.empty((1, 0), dtype=torch.long),
                "eos": True,
                "eos_reason": "cancelled",
            }
        )
