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

With max_batch == 1 (the default) one job runs at a time: these paths own a single
cache. With max_batch > 1 (MP + MTP only, opt-in via the container's max_batch_size)
a slot scheduler runs up to max_batch concurrent jobs: one cache row per job,
serialized chunked prefills interleaved with shared batched graphed MTP verify
rounds, and a per-slot append-only prompt cache (reuse only on strict prefix
extension of the slot's own lineage -- slots never share K/V).
"""

from __future__ import annotations

import asyncio
import os
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

        # preprocess_only discards logits, so only the last row is needed -- computing
        # full-vocab logits for every chunk and concatenating them OOMs on long prompts
        # (the non-MTP streaming prefill path; stuff91.txt: 1.5 GiB torch.cat).
        logits = self.pg.forward(
            input_ids.cpu(),
            self.pg.cache.current_seq_len,
            q_len,
            1,
            last_id_only or preprocess_only,
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


class _MPModelAdapterRow(_MPModelAdapter):
    """
    Row-scoped MP adapter for plain (non-MTP) solo runs in multi-batch mode: every
    forward targets ONE cache row. Prefill chunks go through the eager row path;
    single-token decode uses the batched graph machinery at B=1 (row_map=[r]) and
    advances the row's position explicitly (forward_batch, like the MTP driver,
    leaves bookkeeping to its caller).
    """

    def __init__(self, tp_group, row):
        super().__init__(tp_group)
        self.row = row

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
        r = self.row
        if q_len == 1 and not preprocess_only:
            logits = self.pg.forward_batch(
                input_ids.cpu(), [self.pg.cache.seq_lens[r]], [r],
                last_id_only=last_id_only,
            )
            self.pg.mtp_set_seq_len_row(r, self.pg.cache.seq_lens[r] + 1)
            return logits
        return self.pg.forward_row(
            r, input_ids.cpu(), last_id_only or preprocess_only, preprocess_only
        )


class _MPCacheProxyRow:
    """
    Row-scoped cache facade for the streaming generator: current_seq_len reads/writes
    map to seq_lens[row] (writes fan out to every worker), window_beg to
    window_begs[row]; everything else delegates to the persistent row view.
    """

    def __init__(self, tp_group, row):
        object.__setattr__(self, "_pg", tp_group)
        object.__setattr__(self, "_row", row)

    @property
    def current_seq_len(self):
        return self._pg.cache.seq_lens[self._row]

    def __setattr__(self, name, value):
        if name == "current_seq_len":
            self._pg.mtp_set_seq_len_row(self._row, int(value))
        elif name == "window_beg":
            self._pg.cache.window_begs[self._row] = int(value)
        else:
            setattr(self._pg.cache.row_view(self._row), name, value)

    def __getattr__(self, name):
        return getattr(self._pg.cache.row_view(self._row), name)


def _prefix_match_len(ids_cpu, cached, m):
    """Length of the shared token prefix of two (1, n) id tensors, capped at m."""
    if m <= 0:
        return 0
    if torch.equal(ids_cpu[:, :m], cached[:, :m]):
        return m
    diff = (ids_cpu[0, :m] != cached[0, :m]).nonzero()
    return int(diff[0]) if diff.numel() else m


class _ColdPromptStore:
    """Host-RAM tier of the MP+MTP prompt cache (opt-in, EXPERIMENTAL).

    The VRAM tier is the scheduler's preallocated cache rows (max_batch lineages).
    This store extends lineage capacity beyond the rows: when an idle slot's row is
    needed by a new job, its lineage is offloaded here (per-rank K/V slices stay
    rank-local; the MTP draft shadow + prompt-boundary ring snapshot travel with it)
    instead of being discarded. A later request matching a cold lineage restores it
    into a free row in milliseconds -- no re-prefill of the shared context.

    Budget: a hard byte cap (prompt_cache_ram_limit_mb). Before every offload the
    coldest entries (oldest last_used, then fewest hits) are dropped until the new
    one fits; an entry larger than the whole cap is skipped. Nothing here allocates
    VRAM, and every failure degrades to a full re-prefill -- never a failed job."""

    def __init__(self, pg, cap_mb):
        self.pg = pg
        self.cap = int(cap_mb) * 2**20
        self.entries = {}
        self.total = 0
        self.n_offload = 0
        self.n_restore = 0
        self.n_evict = 0

    def match(self, ids_cpu, min_len):
        """Best restorable prefix among cold lineages: (record, match_len, p) or
        None. p = min(match, committed pos) must reach into the lineage's live SWA
        ring (> its window origin) -- the same validity rule as cross-slot clones."""
        n = ids_cpu.shape[-1]
        best = None
        for rec in self.entries.values():
            m = min(rec["ids"].shape[-1], n, rec["pos"])
            if m < min_len:
                continue
            match = _prefix_match_len(ids_cpu, rec["ids"], m)
            if match < min_len:
                continue
            p = min(match, rec["pos"])
            # Restorable if the shared prefix reaches into the FINAL live ring
            # (p > wb), OR -- when generation shifted the ring past the divergence
            # (long conversations) -- the prompt-boundary snapshot can rewind the ring
            # to an origin at/below the shared prefix (match >= snap_wb). The snapshot
            # is what makes a cold thought-stripped continuation of a LONG chat restore
            # its document instead of re-prefilling it.
            snap_ok = (rec["snap_wb"] is not None and rec["prompt_boundary"] > 0
                       and match >= rec["snap_wb"])
            if p <= rec["wb"] and not snap_ok:
                continue
            if best is None or match > best[1]:
                best = (rec, match, p)
        return best

    def put(self, bundle, ids, pos, wb, prompt_boundary):
        """Store a successful offload bundle, evicting coldest entries to fit the
        cap. Returns the record, or None when skipped (entry alone exceeds cap)."""
        size = bundle["bytes"]
        if size > self.cap:
            self.pg.drop_offload_bundle(bundle)
            xlogger.info(
                f"[prompt-offload] skip: lineage {size / 2**20:.0f} MB exceeds the "
                f"RAM cap ({self.cap / 2**20:.0f} MB)"
            )
            return None
        while self.total + size > self.cap and self.entries:
            self._evict_coldest("RAM cap")
        eid = bundle["entry_id"]
        # The boundary snapshot's ring origin (None when the lineage had no snapshot):
        # match() uses it to accept long-conversation lineages whose final ring shifted
        # past the shared prefix but whose snapshot can still rewind to it.
        local_snap = bundle.get("local_snap")
        snap_wb = local_snap[1] if local_snap is not None else None
        rec = {
            "id": eid, "bundle": bundle, "ids": ids, "pos": pos, "wb": wb,
            "prompt_boundary": prompt_boundary, "snap_wb": snap_wb,
            "last_used": time.monotonic(), "hits": 0, "bytes": size,
        }
        self.entries[eid] = rec
        self.total += size
        self.n_offload += 1
        return rec

    def _evict_coldest(self, reason):
        eid = min(self.entries,
                  key=lambda k: (self.entries[k]["last_used"], self.entries[k]["hits"]))
        self.drop(eid, reason)

    def drop(self, eid, reason=""):
        rec = self.entries.pop(eid, None)
        if rec is None:
            return
        self.total -= rec["bytes"]
        try:
            self.pg.drop_offload_bundle(rec["bundle"])
        except Exception:
            pass
        self.n_evict += 1
        xlogger.info(
            f"[prompt-offload] evict cold lineage {str(eid)[:8]} "
            f"({rec['pos']} tok, {rec['bytes'] / 2**20:.0f} MB)"
            f"{': ' + reason if reason else ''}; store "
            f"{self.total / 2**20:.0f}/{self.cap / 2**20:.0f} MB, "
            f"{len(self.entries)} entries"
        )

    def consume(self, eid):
        """Remove a record whose lineage was FULLY restored into a VRAM row: the
        row owns the K/V again and the ring snapshot was re-attached, so the cold
        copies (worker storage + parent tensors) are freed."""
        rec = self.entries.pop(eid, None)
        if rec is None:
            return
        self.total -= rec["bytes"]
        try:
            self.pg.drop_offload_bundle(rec["bundle"])
        except Exception:
            pass

    def touch(self, rec):
        rec["last_used"] = time.monotonic()
        rec["hits"] += 1

    def clear(self):
        for eid in list(self.entries):
            self.drop(eid)


class _Slot:
    """One cache row of the multi-batch scheduler: the live job (if any) plus the
    slot's append-only prompt-cache lineage. The lineage survives job completion so
    a chat continuation admitted to the same slot reuses its K/V; a job whose prompt
    does not extend the lineage resets the row (never cross-contaminates)."""

    def __init__(self):
        self.job = None
        self.state = None
        self.active = False
        self.cached_ids = None
        self.cached_pos = 0
        self.cache_valid = False
        self.prompt_boundary = 0
        self.cached_tokens = 0
        # RAM-offload tiering stats: which idle lineage to evict first when a new
        # job needs the row (oldest / least-used). Unused when offload is disabled.
        self.last_used = 0.0
        self.hits = 0


class _BatchJobState:
    """Per-job generation bookkeeping for the batched token loop (mirrors the local
    variables of the sequential _run_job_mp_mtp_impl)."""

    def __init__(self, job, time_start, time_prefill):
        self.job = job
        self.stop_tokens = {t for t in job.stop_conditions if isinstance(t, int)}
        self.matcher = _StopStringMatcher(
            [t for t in job.stop_conditions if isinstance(t, str)]
        )
        self.gen_ids: List[int] = []
        self.all_ids: List[int] = []
        self.seq_ids = job.input_ids.clone()
        self.emitted = ""
        self.new_tokens = 0
        self.time_start = time_start
        self.time_prefill = time_prefill
        self.time_gen_start = time.perf_counter()

    def decode_full(self, tokenizer) -> str:
        if not self.gen_ids:
            return ""
        return tokenizer.decode(
            torch.tensor([self.gen_ids], dtype=torch.long),
            decode_special_tokens=self.job.decode_special_tokens,
        )[0]

    def pending_text(self, tokenizer) -> str:
        full = self.decode_full(tokenizer)
        return full[len(self.emitted):] if full.startswith(self.emitted) else ""


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
        max_batch: int = 1,
        prompt_cache_offload: bool = False,
        prompt_cache_ram_limit_mb: int = 8192,
        priority_queue: bool = False,
        batch_window_ms: int = 50,
        batch_window_max_tokens: int = 4096,
        priority_max_wait_ms: int = 30000,
        reprocess_min_tokens: int = 8192,
        reprocess_strikes: int = 2,
        reprocess_window_s: int = 300,
        reprocess_hold_ms: int = 5000,
        reprocess_max_wait_ms: int = 120000,
        priority_preempt: bool = False,
    ):
        self.tokenizer = tokenizer
        # Append-only prompt cache (MP MTP): token ids whose K/V is live in the cache.
        # The SWA ring is not rewind-safe, so reuse is restricted to prompts that
        # strictly EXTEND this sequence (chat continuations); anything else resets.
        self._cached_ids = None
        self._cached_pos = 0
        self._cache_valid = False
        # Prompt-boundary ring snapshot position = the PREVIOUS job's prompt length.
        # Clients that re-render generated history (e.g. strip thought channels)
        # diverge exactly there (the latest assistant turn); the saved SWA ring lets
        # us rewind to it exactly and prefill only the last rendered turn + the new
        # message, keeping the whole earlier conversation cached.
        self._prompt_boundary = 0
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

        # Multi-batch (MP + MTP only): one cache row (slot) per concurrent job, batched
        # graphed verify rounds, per-slot prompt cache. max_batch == 1 keeps the verified
        # sequential path below untouched.
        self.max_batch = int(max_batch) if (self.mp and mtp_drafter is not None) else 1

        # Priority prompt scheduler (multi-batch only): shortest-prompt-first admission
        # plus a small micro-batch window that collects arriving short prompts so they
        # admit together. Long prompts rank last; priority_max_wait_ms ages them forward
        # to bound starvation. Off => FIFO admission, unchanged.
        self.priority_queue = bool(priority_queue) and self.max_batch > 1
        self.batch_window_ms = max(0, int(batch_window_ms))
        self.batch_window_max_tokens = max(1, int(batch_window_max_tokens))
        self.priority_max_wait_ms = max(0, int(priority_max_wait_ms))
        # Reprocess-abuse penalty: long prompts that repeatedly FULLY re-prefill (cache
        # miss every reply — e.g. a 20k document re-sent each turn) are deprioritized
        # behind well-behaved traffic and held briefly before admission. Tracked by
        # prompt fingerprint (prefix hash + length bucket), so edited variants of the
        # same document count together; genuinely new documents are not penalized.
        self.reprocess_min_tokens = max(1024, int(reprocess_min_tokens))
        self.reprocess_strikes = max(1, int(reprocess_strikes))
        self.reprocess_window_s = max(1, int(reprocess_window_s))
        self.reprocess_hold_ms = max(0, int(reprocess_hold_ms))
        self.reprocess_max_wait_ms = max(0, int(reprocess_max_wait_ms))
        self.priority_preempt = bool(priority_preempt)
        self._reprocess_hits = {}
        if self.priority_queue:
            xlogger.info(
                f"Priority prompt scheduler: shortest-first, batch window "
                f"{self.batch_window_ms} ms (<= {self.batch_window_max_tokens} tok), "
                f"max wait {self.priority_max_wait_ms} ms; reprocess penalty >= "
                f"{self.reprocess_min_tokens} tok x{self.reprocess_strikes} strikes/"
                f"{self.reprocess_window_s}s, hold {self.reprocess_hold_ms} ms"
            )

        # Prompt-cache RAM offload (opt-in, EXPERIMENTAL): a capped host-RAM cold tier
        # for idle lineages, on both the sequential and multi-batch MP+MTP paths.
        # EXL2_PROMPT_OFFLOAD=0 is a debug kill switch. DFlash keeps its own draft KV
        # (no gather shadow) and is not supported. Off => every path below is unchanged.
        self.offload_enabled = (
            bool(prompt_cache_offload)
            and self.mp
            and mtp_drafter is not None
            and not getattr(mtp_drafter, "is_dflash", False)
            and os.environ.get("EXL2_PROMPT_OFFLOAD", "1") != "0"
        )
        self._cold_store = (
            _ColdPromptStore(tp_group, prompt_cache_ram_limit_mb)
            if self.offload_enabled else None
        )
        if self.offload_enabled:
            xlogger.info(
                f"MP MTP prompt-cache RAM offload: enabled (cap "
                f"{prompt_cache_ram_limit_mb} MB, max_batch {self.max_batch})"
            )

        # Shim for container access patterns (generator.generator.*)
        self.generator = self
        self.max_batch_size = self.max_batch
        self.current_loras = []
        self.speculative_ngram = False

        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._prefilling_job = None
        if self.max_batch > 1:
            import queue as _thread_queue
            self._bq = _thread_queue.Queue()
            self._batch_slots = [_Slot() for _ in range(self.max_batch)]
            self._worker = asyncio.create_task(self._run_batch())
        else:
            self._worker = asyncio.create_task(self._run())
        self._closed = False

    def set_loras(self, loras: list):
        if self.mp and loras:
            raise ValueError("LoRAs are not supported with the MP tensor-parallel backend")
        self.current_loras = loras

    def enqueue(self, job: "ExLlamaV2ForkJobAsync"):
        if self.max_batch > 1:
            self._bq.put_nowait(job)
        else:
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

    async def _run_batch(self):
        try:
            await asyncio.to_thread(self._batch_scheduler)
        except Exception as e:
            xlogger.error(f"Fork batch scheduler crashed: {e}")
            # Surface the failure to every live job instead of hanging their streams.
            for slot in self._batch_slots:
                if slot.job is not None and not slot.job.cancelled:
                    slot.job.queue.put_nowait(e)
            raise

    async def close(self):
        if self._closed:
            return
        self._closed = True

        # Free the cold store's worker-local KV storage + detached ring snapshots
        # (best-effort; worker exit releases everything anyway).
        if self._cold_store is not None:
            try:
                self._cold_store.clear()
            except Exception:
                pass

        # Abort the running job, cancel anything still queued
        job = getattr(self, "_current_job", None)
        if job is not None:
            self._signal_abort(job)
        if self.max_batch > 1:
            for slot in self._batch_slots:
                if slot.job is not None:
                    self._signal_abort(slot.job)
            while not self._bq.empty():
                try:
                    queued = self._bq.get_nowait()
                except Exception:
                    break
                if queued is not None and not queued.cancelled:
                    await queued.cancel()
            self._bq.put(None)
        else:
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
        request is gone -- the token loop never runs until prefill finishes.

        Multi-batch: the prefill flag is GLOBAL (one collective abort per chunk), so it
        is set only for the job currently prefilling. Other jobs' aborts are handled at
        the next round boundary via their job-level event -- setting the global flag
        for them would kill an unrelated job's in-flight prefill."""
        job.abort_event.set()
        pg = getattr(self, "tp_group", None)
        af = getattr(pg, "abort_flag", None)
        if af is not None:
            if self.max_batch <= 1 or getattr(self, "_prefilling_job", None) is job:
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

    def _run_job_streaming(self, job, streaming=None, row=None):
        g = streaming if streaming is not None else self.streaming
        settings = job.gen_settings

        g.speculative_ngram = self.speculative_ngram and self.mtp_drafter is None
        g.sequence_ids = None
        if row is not None:
            # Batch-mode plain solo run: reset ONLY this slot's row (other slots keep
            # their live K/V). The plain path does not maintain the append-only
            # snapshot, so the slot's lineage is invalidated (tiered to RAM first
            # when offload is enabled, so an MTP continuation can still restore it).
            self._offload_slot(self._batch_slots[row], row)
            self.tp_group.mtp_reset_row(row)
            self._batch_slots[row].cache_valid = False
            depth = self.tp_group.cache.max_seq_len
            cap = max(0, depth - job.input_ids.shape[-1] - 1)
            if job.max_new_tokens > cap:
                job.max_new_tokens = cap
        elif self.mp:
            # Full distributed reset (current_seq_len + window_beg on every rank,
            # plus stale MTP gather buffers) -- invalidates the MP MTP prompt cache
            self._offload_scalar_lineage()
            self.tp_group.mtp_reset_cache()
            self._cache_valid = False
        else:
            self._fresh_cache(g.cache)

        stop_tokens = {t for t in job.stop_conditions if isinstance(t, int)}
        stop_strings = [t for t in job.stop_conditions if isinstance(t, str)]

        time_start = time.perf_counter()
        self._prefilling_job = job
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
        self._prefilling_job = None
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
        except torch.OutOfMemoryError:
            # Self-diagnosing OOM: dump the component VRAM + context state at the
            # failure point so one run localizes the culprit (aux staging, draft
            # KV, graph pools, prefill transients) without a guess-and-check cycle.
            try:
                pg = self.tp_group
                d = self.mtp_drafter
                per = " ".join(
                    f"dev{i}={torch.cuda.memory_allocated(i) / 2**20:.0f}/"
                    f"{torch.cuda.memory_reserved(i) / 2**20:.0f}"
                    for i in range(torch.cuda.device_count()))
                stag = getattr(pg.model, "_dflash_aux_staging", None)
                xlogger.error(
                    f"[dflash-oom] alloc/reserved MiB: {per} | "
                    f"prompt={job.input_ids.shape[-1]} "
                    f"cache_pos={pg.cache.current_seq_len} "
                    f"draft_ctx={getattr(d, 'ctx_len', -1)} "
                    f"draft_gpu={getattr(d, 'footprint', lambda: 0)() / 2**20:.0f} MiB "
                    f"aux_staging={tuple(stag.shape) if stag is not None else None}")
            except Exception:
                pass
            raise
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

    def _offload_scalar_lineage(self):
        """Tier the sequential path's idle lineage (row 0) to RAM before it is
        discarded (opt-in). Failure-soft: any error skips the offload; the reset /
        re-prefill proceeds exactly as without the feature. Never allocates VRAM."""
        if not self.offload_enabled or not self._cache_valid or self._cached_ids is None:
            return
        pg = self.tp_group
        pos = pg.cache.current_seq_len
        min_len = int(os.environ.get("EXL2_BATCH_CLONE_MIN", "512"))
        if pos < min_len or pos != self._cached_pos:
            return
        try:
            bundle = pg.offload_row(0, pos=pos, wb=pg.cache.window_beg)
            if bundle is None:
                return
            rec = self._cold_store.put(bundle, self._cached_ids, pos,
                                       pg.cache.window_beg, self._prompt_boundary)
            if rec is not None:
                xlogger.info(
                    f"[prompt-offload] lineage -> RAM ({pos} tok, "
                    f"{bundle['bytes'] / 2**20:.0f} MB); store "
                    f"{self._cold_store.total / 2**20:.0f}/"
                    f"{self._cold_store.cap / 2**20:.0f} MB, "
                    f"{len(self._cold_store.entries)} entries"
                )
        except Exception as e:
            xlogger.warning(
                f"[prompt-offload] offload failed ({type(e).__name__}: {e}); "
                f"lineage discarded"
            )

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
        restored = False
        restore_pos = 0   # the OLD prompt boundary restored to (DFlash aux_p0);
                          # captured here because _prompt_boundary is overwritten
                          # with the new prompt length before begin() below
        prefill_ids = ids_cpu
        cold_partial = False
        start = 0
        if self.offload_enabled:
            # Cold-tier lookup: when the hot lineage will neither reuse nor restore,
            # a RAM-resident lineage may match this prompt better. Restore it into
            # row 0 and adopt its metadata so the decision block below runs against
            # it (reuse / boundary restore); a divergent prompt prefills the shared
            # prefix's suffix directly (cold_partial).
            own_ok = False
            if cached is not None:
                clen0 = cached.shape[-1]
                m0 = min(clen0, n_prompt)
                cpos0 = self._cached_pos
                if (torch.equal(ids_cpu[:, :m0], cached[:, :m0]) and m0 == clen0
                        and n_prompt > min(cpos0, clen0)
                        and -(self.mtp_window + 1) <= cpos0 - clen0 <= 8
                        and clen0 >= pg.cache.window_beg):
                    own_ok = True   # reuse will hit
                elif (self._prompt_boundary > 0 and n_prompt > self._prompt_boundary
                        and self._prompt_boundary <= clen0
                        and torch.equal(ids_cpu[:, :self._prompt_boundary],
                                        cached[:, :self._prompt_boundary])):
                    own_ok = True   # boundary restore may hit
            if not own_ok:
                min_len = int(os.environ.get("EXL2_BATCH_CLONE_MIN", "512"))
                cold = self._cold_store.match(ids_cpu, min_len)
                p_own = 0
                if cached is not None:
                    m1 = min(cached.shape[-1], n_prompt)
                    match1 = _prefix_match_len(ids_cpu, cached, m1)
                    p1 = min(match1, self._cached_pos)
                    if match1 >= min_len and p1 > 0 and p1 >= pg.cache.window_beg:
                        p_own = p1
                if cold is not None and cold[2] > p_own:
                    rec, match_len, p = cold
                    self._cold_store.touch(rec)
                    self._offload_scalar_lineage()
                    clen = rec["ids"].shape[-1]
                    cpos = rec["pos"]
                    boundary = rec["prompt_boundary"]
                    overshoot = cpos - clen
                    has_snap = rec["snap_wb"] is not None
                    try:
                        t0 = time.perf_counter()
                        self._cold_store.n_restore += 1
                        # Full restore + re-attach the boundary snapshot + adopt, then
                        # decide reuse / boundary-rewind / partial inline (mirrors the
                        # batch path). The snapshot rewinds the ring to the boundary even
                        # after a long generation shifted it past, so a thought-stripped
                        # continuation of a LONG cold chat restores its document.
                        pg.restore_row(0, rec["bundle"], cpos, attach_snapshot=True)
                        self._cached_ids = rec["ids"]
                        self._cached_pos = cpos
                        self._prompt_boundary = boundary
                        self._cache_valid = True
                        self._cold_store.consume(rec["id"])
                        cached = self._cached_ids
                        wb = pg.cache.window_beg
                        if (match_len >= clen and n_prompt >= clen
                                and torch.equal(ids_cpu[:, :clen], rec["ids"][:, :clen])
                                and -(self.mtp_window + 1) <= overshoot <= 8
                                and clen >= wb):
                            start = min(cpos, clen)
                            if cpos > clen:
                                pg.mtp_set_seq_len(clen)
                            reuse = True
                            kind = "cold restore"
                        elif (has_snap and boundary > 0
                                and pg.restore_prompt_ring(boundary)):
                            wb_now = pg.cache.window_beg
                            target = min(match_len, boundary)
                            if target >= wb_now:
                                if target < boundary:
                                    pg.truncate_cache(target)
                                start = target
                                restored = True
                                restore_pos = boundary
                                kind = "cold restore + boundary rewind"
                            else:
                                pg.mtp_reset_cache()
                                start = 0
                                kind = "cold restore (reset)"
                        elif match_len >= wb:
                            pg.truncate_cache(match_len)
                            start = match_len
                            kind = "cold restore (partial prefix)"
                        else:
                            pg.mtp_reset_cache()
                            start = 0
                            kind = "cold restore (reset)"
                        if n_prompt <= start:
                            start -= 1   # exact-match prompt: rewrite the last token
                        prefill_ids = ids_cpu[:, start:]
                        cold_partial = True   # cold path decided; skip the hot block
                        dt = (time.perf_counter() - t0) * 1000
                        xlogger.info(
                            f"MP MTP prompt cache {kind}: restored {cpos} tok from RAM "
                            f"in {dt:.0f} ms, prefilling {n_prompt - start} new"
                        )
                    except Exception as e:
                        xlogger.warning(
                            f"[prompt-offload] cold restore failed "
                            f"({type(e).__name__}: {e}); dropping entry, full re-prefill"
                        )
                        try:
                            self._cold_store.drop(rec["id"], "restore failed")
                        except Exception:
                            pass
                        cached = self._cached_ids if self._cache_valid else None
                        cold_partial = False
                        prefill_ids = ids_cpu
        if cached is not None and not cold_partial:
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
            # Negative overshoot down to -(W+1): a mid-queue stop leaves up to
            # W+1 decided-but-uncommitted tokens (common with DFlash's larger
            # windows). Reuse stays safe at ANY pending length: no rewind
            # happens -- prefill continues FORWARD from cpos, rewriting those
            # tokens' K/V (the ring only ever moves forward here).
            if (prefix_ok and m == clen and n_prompt > start
                    and -(self.mtp_window + 1) <= overshoot <= 8
                    and clen >= pg.cache.window_beg):
                reuse = True
                if cpos > clen:
                    pg.mtp_set_seq_len(clen)   # distributed trim of unemitted drafts
                # DFlash: the draft KV context for the reused prefix is still
                # valid (same lineage); trim its length to the prefill start.
                if getattr(self.mtp_drafter, "is_dflash", False):
                    self.mtp_drafter.trim(start)
                pg.abort_flag.clear()
                prefill_ids = ids_cpu[:, start:]
                xlogger.info(
                    f"MP MTP prompt cache hit: reusing {start} tokens, "
                    f"prefilling {n_prompt - start} new"
                )
            elif (self._prompt_boundary > 0
                    and n_prompt > self._prompt_boundary
                    and self._prompt_boundary <= clen
                    and torch.equal(ids_cpu[:, : self._prompt_boundary],
                                    cached[:, : self._prompt_boundary])
                    and pg.restore_prompt_ring(self._prompt_boundary)):
                # History diverges at/after the original prompt boundary (clients
                # commonly strip thought channels when re-rendering assistant turns).
                # The saved SWA ring makes the rewind EXACT -- no recompute; only the
                # re-rendered tail is prefilled, the big prompt stays cached.
                restored = True
                restore_pos = self._prompt_boundary
                pg.abort_flag.clear()
                prefill_ids = ids_cpu[:, self._prompt_boundary:]
                xlogger.info(
                    f"MP MTP prompt cache restore: rewound to prompt boundary "
                    f"{self._prompt_boundary}, prefilling "
                    f"{n_prompt - self._prompt_boundary} re-rendered/new tokens"
                )
            else:
                if prefix_ok:
                    why = (f"cache overshoot {overshoot} outside "
                           f"[{-(self.mtp_window + 1)}, 8]"
                           if not (-(self.mtp_window + 1) <= overshoot <= 8)
                           else "prompt does not extend the cached sequence")
                else:
                    diff = (ids_cpu[0, :m] != cached[0, :m]).nonzero()
                    at = int(diff[0]) if diff.numel() else m
                    why = f"history diverges at token {at} (re-rendered/edited prompt?)"
                xlogger.info(f"MP MTP prompt cache miss: {why}; full re-prefill")
        partial_scalar = False
        if (not reuse and not restored and not cold_partial
                and self.offload_enabled and cached is not None):
            # Own-lineage partial prefix (offload tier): the prompt diverges from the
            # cached history but shares a long prefix inside the live ring. Rewind to
            # the divergence point and prefill only the suffix.
            min_len = int(os.environ.get("EXL2_BATCH_CLONE_MIN", "512"))
            m = min(cached.shape[-1], n_prompt)
            match = _prefix_match_len(ids_cpu, cached, m)
            p = min(match, self._cached_pos)
            if match >= min_len and p > 0 and p >= pg.cache.window_beg:
                self._offload_scalar_lineage()
                pg.truncate_cache(p)
                start = p if n_prompt > p else p - 1
                prefill_ids = ids_cpu[:, start:]
                partial_scalar = True
                xlogger.info(
                    f"MP MTP prompt cache partial reuse: rewound to shared prefix "
                    f"{p}, prefilling {n_prompt - start} new"
                )
        if not reuse and not restored and not cold_partial and not partial_scalar:
            # Tier the discarded lineage to RAM (opt-in) before the full reset.
            self._offload_scalar_lineage()
            pg.mtp_reset_cache()
        driver = MTPMPDriver(pg, self.mtp_drafter, window=self.mtp_window)
        dflash = getattr(self.mtp_drafter, "is_dflash", False)

        time_start = time.perf_counter()
        if dflash:
            # DFlash: the prefill also returns the per-token aux states (the
            # target-layer residual taps) that become the draft KV context.
            r = pg.forward_state(prefill_ids, last_id_only=True, cpu_logits=True,
                                 return_row_aux=True)
            logits, state, aux = r if r is not None else (None, None, None)
        else:
            logits, state = pg.forward_state(prefill_ids, last_id_only=True, cpu_logits=True)
            aux = None
        time_prefill = time.perf_counter() - time_start
        if logits is None or job.abort_event.is_set():
            # Client disconnected during prefill: every rank aborted at the same chunk
            # boundary (collective flag in model.forward). Stop instead of generating.
            self._cache_valid = False
            self._finish(job, "abort", 0, "", time_start, time_prefill, time_start)
            return

        # Save every rank's SWA ring at THIS prompt's end -- on every path (fresh,
        # extension, restore). A client that re-renders generated history (e.g.
        # strips thought channels) always diverges at the PREVIOUS prompt's end
        # (the latest assistant turn), so the latest snapshot is exactly the one
        # the next reply needs: each reply restores to the previous prompt boundary
        # and prefills only the last rendered turn + the new message -- a constant
        # tail, instead of re-prefilling the whole rendered history every turn.
        # One backup buffer (~262 MB/rank), replaced per job; no growth.
        pg.snapshot_prompt_ring()
        self._prompt_boundary = n_prompt

        # begin() takes the FULL prompt: its token history and heal base use absolute
        # cache positions. logits/state are the last row of the suffix forward, which
        # is the last prompt token on both the reuse and the fresh-prefill paths.
        # DFlash: the aux rows cover only the tokens prefilled THIS job, at aux_p0
        # (0 fresh, the reuse start, or the OLD boundary captured at restore time --
        # self._prompt_boundary already holds the NEW prompt length here).
        if dflash:
            aux_p0 = start if reuse else (restore_pos if restored else 0)
            driver.begin(job.input_ids, logits, state, aux=aux, aux_p0=aux_p0)
        else:
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

    # ------------------------------------------------------------------ multi-batch scheduler

    def _pq_tag(self, j):
        """Arrival time + prompt length + reprocess-abuse flag (set once).

        Fingerprint = hash of the prompt's first 256 tokens + length bucket: identifies
        the same long document re-sent reply after reply (edited variants share the
        prefix). The strike HISTORY is updated at admission when a full re-prefill is
        detected; here we only read it, so ordering reflects past behavior.
        """
        if not hasattr(j, "_pq_arrival"):
            j._pq_arrival = time.monotonic()
            try:
                ids = j.input_ids
                j._pq_len = int(ids.shape[-1])
                if j._pq_len >= self.reprocess_min_tokens:
                    head = ids[0, :min(256, j._pq_len)].to(torch.long).cpu().numpy().tobytes()
                    j._pq_fp = (hash(head), j._pq_len // 1024)
                    now = time.monotonic()
                    hist = self._reprocess_hits.get(j._pq_fp)
                    if hist:
                        hist[:] = [t for t in hist if now - t <= self.reprocess_window_s]
                        j._pq_abuser = len(hist) >= self.reprocess_strikes
                    else:
                        j._pq_abuser = False
                else:
                    j._pq_fp = None
                    j._pq_abuser = False
            except Exception:
                j._pq_len = 1 << 30
                j._pq_fp = None
                j._pq_abuser = False

    def _pq_key(self, j):
        """Shortest-first; reprocess abusers rank behind well-behaved traffic; aged jobs
        promote ahead of everything (FIFO among themselves) to bound starvation. Abusers
        age on the longer reprocess_max_wait_ms — delays for them are intentional."""
        now = time.monotonic()
        abuser = getattr(j, "_pq_abuser", False)
        max_wait = (self.reprocess_max_wait_ms if abuser else self.priority_max_wait_ms) / 1000.0
        aged = max_wait > 0 and now - j._pq_arrival >= max_wait
        cls = 0 if aged else (2 if abuser else 1)
        return (cls, 0 if aged else getattr(j, "_pq_len", 1 << 30), j._pq_arrival)

    def _pq_held(self, j):
        """Abuser jobs are held briefly before admission so well-behaved arrivals can
        jump ahead; aging (reprocess_max_wait_ms) overrides the hold."""
        if not getattr(j, "_pq_abuser", False) or self.reprocess_hold_ms <= 0:
            return False
        now = time.monotonic()
        if self.reprocess_max_wait_ms > 0 and \
                now - j._pq_arrival >= self.reprocess_max_wait_ms / 1000.0:
            return False
        return now - j._pq_arrival < self.reprocess_hold_ms / 1000.0

    def _batch_scheduler(self):
        """
        Slot-based concurrency for MP + MTP (max_batch > 1).

        Phases: admit pending jobs into free cache rows (serialized chunked prefill,
        with decode rounds interleaved between chunks so active jobs stall at most one
        chunk), then shared batched MTP rounds until every active job finishes or new
        work arrives. Plain (filters/banned-strings) jobs run solo between batches:
        admission stops, the active batch drains, the plain job runs on its own row
        through the verified streaming path, then batching resumes.

        Prompt-cache isolation: each slot keeps its OWN lineage (ids + position + ring
        snapshot). A job reuses a slot's K/V only if its prompt strictly extends that
        slot's lineage (or restores its prompt boundary); anything else resets the row.
        Cache rows are physically separate batch rows, so slots never share K/V.
        """
        import queue as _thread_queue
        from exllamav2.mtp_mp import MTPMPBatchDriver

        pg = self.tp_group
        driver = MTPMPBatchDriver(pg, self.mtp_drafter, window=self.mtp_window)
        slots = self._batch_slots
        pending = []
        shutdown = False

        def _tag(j):
            self._pq_tag(j)

        while not shutdown:
            if not pending and not any(s.active for s in slots):
                item = self._bq.get()
                if item is None:
                    break
                _tag(item)
                pending.append(item)
                # Micro-batch window: when the batch is idle, wait briefly for more
                # arrivals before admitting, then admit shortest-first. This lets short
                # prompts arriving alongside (or just after) a long prompt go first —
                # a lone long prompt would otherwise admit immediately and its
                # multi-second prefill stalls everyone. Bounded by batch_window_ms.
                if (self.priority_queue and self.batch_window_ms > 0
                        and not shutdown):
                    deadline = time.monotonic() + self.batch_window_ms / 1000.0
                    while len(pending) < self.max_batch:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            more = self._bq.get(timeout=remaining)
                        except _thread_queue.Empty:
                            break
                        if more is None:
                            shutdown = True
                            break
                        _tag(more)
                        pending.append(more)
            while True:
                try:
                    item = self._bq.get_nowait()
                except _thread_queue.Empty:
                    break
                if item is None:
                    shutdown = True
                    break
                _tag(item)
                pending.append(item)
            pending = [j for j in pending if not j.cancelled]
            if shutdown:
                break
            # Shortest-prompt-first admission; aged jobs promote ahead (see _pq_key).
            if self.priority_queue and len(pending) > 1:
                pending.sort(key=self._pq_key)
            for j in pending:
                if not hasattr(j, "_fork_eligible"):
                    j._fork_eligible = self._mp_mtp_eligible(j)

            plain = [j for j in pending if not j._fork_eligible]
            if plain:
                while any(s.active for s in slots):
                    self._batch_decode_round_safe(slots, driver)
                for j in plain:
                    pending.remove(j)
                    if j.cancelled:
                        continue
                    r = next(i for i, s in enumerate(slots) if not s.active)
                    g = ExLlamaV2StreamingGenerator(
                        _MPModelAdapterRow(pg, r), _MPCacheProxyRow(pg, r),
                        self.tokenizer,
                    )
                    try:
                        self._run_job_streaming(j, streaming=g, row=r)
                    except Exception as e:
                        self._put(j, e)
                continue

            for j in [j for j in pending if j._fork_eligible and not self._pq_held(j)]:
                free = [i for i, s in enumerate(slots) if not s.active]
                if not free:
                    break
                r = self._choose_slot(slots, free, j)
                pending.remove(j)
                if j.cancelled:
                    continue
                try:
                    self._admit_job(j, slots[r], r, driver, slots, pending)
                except Exception as e:
                    slots[r].active = False
                    slots[r].job = None
                    slots[r].cache_valid = False
                    self._put(j, e)

            # Held abusers (or transient no-free-slot): avoid a busy spin while waiting.
            if pending and not any(s.active for s in slots):
                time.sleep(0.02)

            while any(s.active for s in slots):
                self._batch_decode_round_safe(slots, driver)
                if pending or shutdown:
                    break
                try:
                    item = self._bq.get_nowait()
                except _thread_queue.Empty:
                    continue
                if item is None:
                    shutdown = True
                    break
                _tag(item)
                pending.append(item)
                break

        for i, s in enumerate(slots):
            if s.active:
                self._finish_slot(slots, i, driver, "abort")
        for j in pending:
            if not j.cancelled:
                j.cancelled = True
                self._loop.call_soon_threadsafe(j.queue.put_nowait, {
                    "stage": "streaming",
                    "identifier": j.identifier,
                    "text": "",
                    "token_ids": torch.empty((1, 0), dtype=torch.long),
                    "eos": True,
                    "eos_reason": "cancelled",
                })

    def _batch_decode_round_safe(self, slots, driver):
        try:
            self._batch_decode_round(slots, driver)
        except Exception as e:
            # A failed round (worker crash, CUDA error) must surface to every live job
            # instead of hanging their streams; invalidate the affected slots.
            for i, s in enumerate(slots):
                if s.active:
                    s.active = False
                    s.cache_valid = False
                    job, s.job = s.job, None
                    s.state = None
                    if job is not None and not job.cancelled:
                        self._put(job, e)
            raise

    def _choose_slot(self, slots, free, job):
        """Prefix affinity: prefer the free slot whose cached lineage shares the longest
        prefix with the job's prompt (chat continuations land on their own slot). Any
        free slot is correct -- _admit_job reuses or resets based on the exact check.

        With RAM offload enabled, a COLD lineage may match better than any free slot's
        hot lineage; then (or when nothing hot matches) pick the COLDEST free slot so
        the least-valued idle lineage is the one evicted to RAM."""
        best, best_len = free[0], -1
        ids = job.input_ids
        for i in free:
            s = slots[i]
            if not s.cache_valid or s.cached_ids is None:
                continue
            m = min(s.cached_ids.shape[-1], ids.shape[-1])
            if m > 0 and torch.equal(s.cached_ids[:, :m], ids[:, :m]):
                if m > best_len:
                    best, best_len = i, m
        if self.offload_enabled:
            min_len = int(os.environ.get("EXL2_BATCH_CLONE_MIN", "512"))
            cold = self._cold_store.match(ids.cpu(), min_len)
            cold_len = cold[1] if cold is not None else -1
            if cold_len > best_len or best_len < 0:
                return min(free, key=lambda i: (slots[i].last_used, slots[i].hits))
        return best

    def _clone_source(self, slots, r, ids_cpu):
        """Best shared-prefix K/V source among the OTHER slots for a job whose own
        lineage did not match: (slot index, clone position) or None.

        Active slots offer their prompt + emitted-so-far (conservative: the committed
        cache can lag the emitted ids by at most the pending queue, and the clone
        position is clamped to the committed seq_lens anyway); free slots offer their
        recorded lineage. The position must sit inside the source's live SWA ring
        (>= its window origin) -- a long-lived slot whose ring has shifted past the
        shared prefix cannot serve it (SWA K/V is gone; full layers alone are not a
        valid state). Runs between rounds, so an active source's prefix is stable."""
        if os.environ.get("EXL2_BATCH_KV_CLONE", "1") == "0":
            return None
        min_len = int(os.environ.get("EXL2_BATCH_CLONE_MIN", "512"))
        n = ids_cpu.shape[-1]
        cache = self.tp_group.cache
        best = None
        for i, s in enumerate(slots):
            if i == r:
                continue
            if s.active:
                if s.state is None:
                    continue
                src_ids = s.state.seq_ids
                pos = cache.seq_lens[i]
            elif s.cache_valid and s.cached_ids is not None:
                src_ids = s.cached_ids
                pos = s.cached_pos
            else:
                continue
            wb = cache.window_begs[i]
            m = min(src_ids.shape[-1], n, pos)
            if m < min_len or m <= wb:
                continue
            a, b = ids_cpu[0, :m], src_ids[0, :m]
            if torch.equal(a, b):
                match = m
            else:
                diff = (a != b).nonzero()
                match = int(diff[0]) if diff.numel() else m
            if match < min_len or match <= wb:
                continue
            if best is None or match > best[1]:
                best = (i, match)
        return best

    def _offload_slot(self, slot, r):
        """Tier an idle slot's lineage to RAM before its row is reused (opt-in).
        Failure-soft: any error skips the offload and the row resets/re-prefills
        exactly as it would without the feature. Never allocates VRAM."""
        if not self.offload_enabled or not slot.cache_valid or slot.cached_ids is None:
            return
        pg = self.tp_group
        pos = pg.cache.seq_lens[r]
        min_len = int(os.environ.get("EXL2_BATCH_CLONE_MIN", "512"))
        if pos < min_len or pos != slot.cached_pos:
            return
        try:
            bundle = pg.offload_row(r, pos=pos, wb=pg.cache.window_begs[r])
            if bundle is None:
                return
            rec = self._cold_store.put(bundle, slot.cached_ids, pos,
                                       pg.cache.window_begs[r], slot.prompt_boundary)
            if rec is not None:
                xlogger.info(
                    f"[prompt-offload] slot {r} lineage -> RAM ({pos} tok, "
                    f"{bundle['bytes'] / 2**20:.0f} MB); store "
                    f"{self._cold_store.total / 2**20:.0f}/"
                    f"{self._cold_store.cap / 2**20:.0f} MB, "
                    f"{len(self._cold_store.entries)} entries"
                )
        except Exception as e:
            xlogger.warning(
                f"[prompt-offload] offload slot {r} failed "
                f"({type(e).__name__}: {e}); lineage discarded"
            )

    def _admit_job(self, job, slot, r, driver, slots, pending=None):
        """Prompt-cache decision + chunked prefill into row r, then arm the MTP row.
        Decode rounds for already-active slots interleave between prefill chunks."""
        pg = self.tp_group
        ids_cpu = job.input_ids.cpu()
        n_prompt = ids_cpu.shape[-1]
        cached = slot.cached_ids if slot.cache_valid else None
        reuse = False
        restored = False
        prefill_ids = ids_cpu
        start = 0
        if cached is not None:
            clen = cached.shape[-1]
            m = min(clen, n_prompt)
            prefix_ok = torch.equal(ids_cpu[:, :m], cached[:, :m])
            cpos = slot.cached_pos
            overshoot = cpos - clen
            start = min(cpos, clen)
            if (prefix_ok and m == clen and n_prompt > start
                    and -(self.mtp_window + 1) <= overshoot <= 8
                    and clen >= pg.cache.window_begs[r]):
                reuse = True
                if cpos > clen:
                    pg.mtp_set_seq_len_row(r, clen)
                prefill_ids = ids_cpu[:, start:]
                xlogger.info(
                    f"MP MTP prompt cache hit (slot {r}): reusing {start} tokens, "
                    f"prefilling {n_prompt - start} new"
                )
            elif (slot.prompt_boundary > 0
                    and n_prompt > slot.prompt_boundary
                    and slot.prompt_boundary <= clen
                    and torch.equal(ids_cpu[:, :slot.prompt_boundary],
                                    cached[:, :slot.prompt_boundary])
                    and pg.restore_prompt_ring_row(r, slot.prompt_boundary)):
                restored = True
                start = slot.prompt_boundary
                prefill_ids = ids_cpu[:, slot.prompt_boundary:]
                xlogger.info(
                    f"MP MTP prompt cache restore (slot {r}): rewound to prompt "
                    f"boundary {slot.prompt_boundary}, prefilling "
                    f"{n_prompt - slot.prompt_boundary} re-rendered/new tokens"
                )
            else:
                if prefix_ok:
                    why = (f"cache overshoot {overshoot} outside "
                           f"[{-(self.mtp_window + 1)}, 8]"
                           if not (-(self.mtp_window + 1) <= overshoot <= 8)
                           else "prompt does not extend the cached sequence")
                else:
                    diff = (ids_cpu[0, :m] != cached[0, :m]).nonzero()
                    at = int(diff[0]) if diff.numel() else m
                    why = f"history diverges at token {at} (re-rendered/edited prompt?)"
                xlogger.info(f"MP MTP prompt cache miss (slot {r}): {why}")
        partial = False
        cloned = False
        cold_restored = False
        if not reuse and not restored:
            min_len = int(os.environ.get("EXL2_BATCH_CLONE_MIN", "512"))
            clone_on = os.environ.get("EXL2_BATCH_KV_CLONE", "1") != "0"
            # Candidate 1 -- own-lineage partial prefix: the prompt diverges from this
            # slot's history (edited/regenerated turn, different question on the same
            # document) but shares a long prefix with it. Rewind the slot to the
            # divergence point and prefill only the suffix -- a position truncation,
            # not copies or a re-prefill.
            p_own = 0
            if slot.cache_valid and slot.cached_ids is not None and clone_on:
                clen = slot.cached_ids.shape[-1]
                m = min(clen, n_prompt)
                match = _prefix_match_len(ids_cpu, slot.cached_ids, m)
                p = min(match, slot.cached_pos)
                if match >= min_len and 0 < p >= pg.cache.window_begs[r]:
                    p_own = p
            # Candidate 2 -- shared-prefix clone from another HOT slot: concurrent
            # requests commonly share a long system / few-shot / document prefix, and
            # a continuation whose own slot is busy would otherwise re-prefill the
            # whole context on a free slot. A physical D2D copy (slots never share
            # live K/V, so the isolation invariant holds).
            src = self._clone_source(slots, r, ids_cpu) if clone_on else None
            p_clone = src[1] if src is not None else 0
            # Candidate 3 -- cold lineage restore from RAM (opt-in offload tier): an
            # idle lineage evicted earlier matches this prompt; H2D its K/V + MTP
            # shadow back instead of re-prefilling.
            cold = (self._cold_store.match(ids_cpu, min_len)
                    if self.offload_enabled else None)
            p_cold = cold[2] if cold is not None else 0
            if self.offload_enabled:
                # Longest restorable prefix wins (prefill cost dominates copy cost);
                # ties prefer the cheaper source: own truncate < hot clone < cold H2D.
                best_p = max(p_own, p_clone, p_cold)
            else:
                # Verified precedence: own partial first, then cross-slot clone.
                best_p = p_own if p_own > 0 else p_clone
            if best_p > 0 and best_p == p_own:
                self._offload_slot(slot, r)
                pg.truncate_row(r, p_own)
                # Exact-match prompt (n_prompt == p): rewrite the last token instead
                # of an empty prefill (idempotent KV rewrite; an empty suffix would
                # leave logits None and abort the job).
                start = p_own if n_prompt > p_own else p_own - 1
                prefill_ids = ids_cpu[:, start:]
                partial = True
                slot.hits += 1
                xlogger.info(
                    f"MP MTP prompt cache partial reuse (slot {r}): rewound to shared "
                    f"prefix {p_own}, prefilling {n_prompt - start} new"
                )
            elif best_p > 0 and best_p == p_clone:
                self._offload_slot(slot, r)
                i, p = src
                pg.clone_row(i, r, p)
                start = p if n_prompt > p else p - 1
                prefill_ids = ids_cpu[:, start:]
                cloned = True
                xlogger.info(
                    f"MP MTP prompt cache clone (slot {r} <- slot {i}): reusing {p} "
                    f"shared-prefix tokens, prefilling {n_prompt - start} new"
                )
            elif best_p > 0 and cold is not None and best_p == p_cold:
                rec, match_len, p = cold
                # Keep the entry hot in the store while the row's old lineage is
                # offloaded (the cap eviction must not drop the entry we're restoring).
                self._cold_store.touch(rec)
                self._offload_slot(slot, r)
                clen = rec["ids"].shape[-1]
                cpos = rec["pos"]
                boundary = rec["prompt_boundary"]
                overshoot = cpos - clen
                has_snap = rec["snap_wb"] is not None
                if os.environ.get("EXL2_OFFLOAD_DEBUG", "0") == "1":
                    xlogger.info(
                        f"[offload-dbg] slot {r} cold match={match_len} boundary={boundary} "
                        f"p={p} pos={cpos} clen={clen} wb={rec['wb']} "
                        f"snap_wb={rec['snap_wb']} n_prompt={n_prompt}"
                    )
                try:
                    t0 = time.perf_counter()
                    # Restore the FULL lineage state and re-attach its prompt-boundary
                    # ring snapshot, then adopt the lineage. The snapshot is what makes a
                    # COLD thought-stripped continuation behave like a hot one even for a
                    # LONG conversation: it rewinds the SWA ring to the boundary after
                    # generation shifted the ring past it, so only the re-rendered tail
                    # re-prefills -- never the document.
                    pg.restore_row(r, rec["bundle"], cpos, attach_snapshot=True)
                    slot.cached_ids = rec["ids"]
                    slot.cached_pos = cpos
                    slot.prompt_boundary = boundary
                    slot.cache_valid = True
                    self._cold_store.consume(rec["id"])
                    wb = pg.cache.window_begs[r]
                    if (match_len >= clen and n_prompt >= clen
                            and torch.equal(ids_cpu[:, :clen], rec["ids"][:, :clen])
                            and -(self.mtp_window + 1) <= overshoot <= 8
                            and clen >= wb):
                        # strict extension -> reuse (trim uncommitted drafts)
                        start = min(cpos, clen)
                        if cpos > clen:
                            pg.mtp_set_seq_len_row(r, clen)
                        reuse = True
                        kind = "cold restore"
                    elif (has_snap and boundary > 0
                            and pg.restore_prompt_ring_row(r, boundary)):
                        # Thought-stripped / re-rendered assistant turn: rewind the ring
                        # to the boundary snapshot, then truncate to the TRUE shared
                        # prefix (the prompt may diverge a few tokens before the boundary
                        # at the turn cue). Works even when the final ring shifted past
                        # the boundary (long chats) -- the snapshot restores that K/V.
                        wb_now = pg.cache.window_begs[r]
                        target = min(match_len, boundary)
                        if target >= wb_now:
                            if target < boundary:
                                pg.truncate_row(r, target)
                            start = target
                            restored = True
                            kind = "cold restore + boundary rewind"
                        else:
                            pg.mtp_reset_row(r)
                            start = 0
                            kind = "cold restore (reset)"
                    elif match_len >= wb:
                        # No snapshot rewind: truncate to the shared prefix inside the
                        # final live ring.
                        pg.truncate_row(r, match_len)
                        start = match_len
                        kind = "cold restore (partial prefix)"
                    else:
                        pg.mtp_reset_row(r)
                        start = 0
                        kind = "cold restore (reset)"
                    slot.hits += 1
                    restored_pos = cpos
                    dt = (time.perf_counter() - t0) * 1000
                    if n_prompt <= start:
                        start -= 1   # exact-match prompt: rewrite the last token
                    prefill_ids = ids_cpu[:, start:]
                    cold_restored = True
                    self._cold_store.n_restore += 1
                    xlogger.info(
                        f"MP MTP prompt cache {kind} (slot {r}): restored {restored_pos} "
                        f"tok from RAM in {dt:.0f} ms, prefilling {n_prompt - start} new"
                    )
                except Exception as e:
                    # Failure-soft: drop the entry and re-prefill from scratch. The
                    # job never fails because of the offload system.
                    xlogger.warning(
                        f"[prompt-offload] cold restore slot {r} failed "
                        f"({type(e).__name__}: {e}); dropping entry, full re-prefill"
                    )
                    try:
                        self._cold_store.drop(rec["id"], "restore failed")
                    except Exception:
                        pass
                    pg.mtp_reset_row(r)
                    start = 0
                    prefill_ids = ids_cpu
            else:
                self._offload_slot(slot, r)
                pg.mtp_reset_row(r)
                start = 0
        slot.last_used = time.monotonic()
        slot.cached_tokens = start
        slot.job = job

        # Reprocess-abuse strike: a LONG prompt fully re-prefilled from scratch (no
        # reuse/restore/clone/cold-restore). Re-rendered last replies take the partial
        # paths above and never strike. Repeated strikes deprioritize future repeats of
        # the same fingerprint (see _pq_tag/_pq_key).
        if (self.priority_queue and getattr(job, "_pq_fp", None) is not None
                and start == 0 and not reuse and not restored and not cloned
                and not cold_restored and n_prompt >= self.reprocess_min_tokens):
            _now = time.monotonic()
            _hist = self._reprocess_hits.setdefault(job._pq_fp, [])
            _hist.append(_now)
            _hist[:] = [t for t in _hist if _now - t <= self.reprocess_window_s]
            if len(_hist) >= self.reprocess_strikes:
                xlogger.info(
                    f"Priority scheduler: reprocess strike {len(_hist)} on long prompt "
                    f"({n_prompt} tok, full re-prefill) — repeats deprioritized + held"
                )

        time_start = time.perf_counter()
        pg.abort_flag.clear()
        self._prefilling_job = job
        chunk_size = max(256, pg.config.max_input_len)
        n = prefill_ids.shape[-1]
        pos = 0
        logits = state = None
        aborted = False
        if n > chunk_size:
            xlogger.info(
                f"MP batch prefill slot {r}: {n} tokens in {chunk_size}-token chunks "
                f"(the per-chunk [prefill] trace resets its denominator each call; "
                f"this counter is the cumulative progress)"
            )
        try:
            while pos < n:
                if job.abort_event.is_set():
                    aborted = True
                    break
                end = min(pos + chunk_size, n)
                if end == n:
                    logits, state = pg.forward_state_row(
                        r, prefill_ids[:, pos:end], last_id_only=True)
                else:
                    pg.forward_row(r, prefill_ids[:, pos:end],
                                   last_id_only=True, preprocess_only=True)
                pos = end
                if n > chunk_size:
                    xlogger.info(f"MP batch prefill slot {r}: {pos}/{n} ({100 * pos // n}%)")
                if job.abort_event.is_set():
                    aborted = True
                    break
                # Keep active jobs decoding during long prefills: stall <= one chunk.
                if any(s.active for s in slots):
                    self._batch_decode_round_safe(slots, driver)
                # Priority preemption (opt-in priority_preempt): admit newly arrived
                # SHORT jobs between chunks of a long prefill. EXPERIMENTAL: can crash
                # the batch KV gather (negative shape) when rows admit mid-prefill;
                # default off until root-caused. Single-chunk jobs only, nested
                # admission without further preemption; _prefilling_job saved/restored.
                if self.priority_queue and self.priority_preempt and pending is not None:
                    import queue as _tq
                    while True:
                        try:
                            item = self._bq.get_nowait()
                        except _tq.Empty:
                            break
                        if item is None:
                            self._bq.put(None)
                            break
                        self._pq_tag(item)
                        pending.append(item)
                    if pending:
                        pending.sort(key=self._pq_key)
                        for j2 in pending:
                            if not hasattr(j2, "_fork_eligible"):
                                j2._fork_eligible = self._mp_mtp_eligible(j2)
                        for j2 in [j for j in pending
                                   if getattr(j, "_fork_eligible", False)
                                   and j.input_ids.shape[-1] <= chunk_size
                                   and not self._pq_held(j)]:
                            free2 = [i for i, s in enumerate(slots) if not s.active]
                            if not free2:
                                break
                            r2 = self._choose_slot(slots, free2, j2)
                            pending.remove(j2)
                            if j2.cancelled:
                                continue
                            prev_pref = self._prefilling_job
                            try:
                                self._admit_job(j2, slots[r2], r2, driver, slots, None)
                            except Exception as e:
                                slots[r2].active = False
                                slots[r2].job = None
                                slots[r2].cache_valid = False
                                self._put(j2, e)
                            finally:
                                self._prefilling_job = prev_pref
        finally:
            self._prefilling_job = None
            pg.abort_flag.clear()
        time_prefill = time.perf_counter() - time_start

        if aborted or logits is None:
            slot.cache_valid = False
            slot.job = None
            self._finish(job, "abort", 0, "", time_start, time_prefill, time_start)
            return

        # Bound generation by cache depth. A verify round writes W extra rows past the
        # current position and kv_write silently drops rows beyond the buffer; an explicit
        # client max_tokens larger than the remaining depth would push seq_lens past
        # max_seq_len, desyncing the incremental gather (worker clamps, parent doesn't).
        # The container bounds prompt length but passes max_tokens through uncapped.
        depth = pg.cache.max_seq_len
        cap = max(0, depth - n_prompt - self.mtp_window)
        if job.max_new_tokens > cap:
            xlogger.info(
                f"MP batch slot {r}: capping max_new_tokens {job.max_new_tokens} -> "
                f"{cap} (cache depth {depth}, prompt {n_prompt}, verify headroom "
                f"{self.mtp_window})"
            )
            job.max_new_tokens = cap

        pg.snapshot_prompt_ring_row(r)
        slot.prompt_boundary = n_prompt

        st = _BatchJobState(job, time_start, time_prefill)
        slot.state = st
        ban = list(st.stop_tokens) if job.min_new_tokens > 0 else None
        driver.begin_row(r, job.input_ids, logits, state,
                         settings=job.gen_settings, tokenizer=self.tokenizer,
                         seq_ids=st.seq_ids, ban_tokens=ban)
        slot.active = True

    def _batch_decode_round(self, slots, driver):
        for i, s in enumerate(slots):
            if s.active and s.job.abort_event.is_set():
                flush = s.state.pending_text(self.tokenizer)
                self._finish_slot(slots, i, driver, "abort", extra_text=flush)
        # Depth guard (belt-and-suspenders over the admission cap): a round at position
        # pos writes through pos-1+W, so pos+W must stay inside the cache. Covers any
        # path that moves a row's position outside the driver (restore/recommit).
        depth = self.tp_group.cache.max_seq_len
        for i, s in enumerate(slots):
            if s.active and self.tp_group.cache.seq_lens[i] + self.mtp_window > depth:
                flush = s.state.pending_text(self.tokenizer)
                self._finish_slot(slots, i, driver, "max_new_tokens", extra_text=flush)
        active = [i for i, s in enumerate(slots) if s.active]
        if not active:
            return
        settings_map = {i: slots[i].job.gen_settings for i in active}
        seq_ids_map = {i: slots[i].state.seq_ids for i in active}
        ban_map = {
            i: (list(slots[i].state.stop_tokens)
                if slots[i].state.new_tokens < slots[i].job.min_new_tokens else None)
            for i in active
        }
        toks = driver.step_batch(active, settings_map, self.tokenizer,
                                 seq_ids_map, ban_map)
        # Bursts (m_acc+1 tokens per refreshed row): consume in order; a stop/max
        # finish mid-burst deactivates the slot and discards the tail (the same
        # truncation the old one-token loop applied at round boundaries).
        for i in active:
            for tok in toks.get(i, ()):
                if not slots[i].active:
                    break
                self._consume_token(slots, i, driver, tok)

    def _consume_token(self, slots, r, driver, tok):
        """One emitted token for slot r -- the sequential MTP token loop body, scoped
        to the job's own state (stop conditions, text matching, streaming chunks)."""
        slot = slots[r]
        job = slot.job
        st = slot.state
        st.all_ids.append(tok)
        st.new_tokens += 1
        st.seq_ids = torch.cat(
            [st.seq_ids, torch.tensor([[tok]], dtype=torch.long)], dim=-1)

        if tok in st.stop_tokens:
            if st.new_tokens > job.min_new_tokens:
                flush = st.pending_text(self.tokenizer)
                self._finish_slot(slots, r, driver, "stop_token",
                                  stop_token=tok, extra_text=flush)
                return
            # Below min_new_tokens: consume but don't emit the stop token.
        else:
            st.gen_ids.append(tok)
            full = st.decode_full(self.tokenizer)
            if full.startswith(st.emitted):
                emit_text, matched, hold = st.matcher.feed(full[len(st.emitted):])
                if matched is not None:
                    self._finish_slot(slots, r, driver, "stop_string",
                                      stop_string=matched, extra_text=emit_text)
                    return
                if not hold and emit_text:
                    st.emitted += emit_text
                    self._put(job, {
                        "stage": "streaming",
                        "identifier": job.identifier,
                        "text": emit_text,
                        "token_ids": torch.tensor([[tok]], dtype=torch.long),
                        "eos": False,
                    })

        if st.new_tokens >= job.max_new_tokens:
            flush = st.pending_text(self.tokenizer)
            self._finish_slot(slots, r, driver, "max_new_tokens", extra_text=flush)

    def _finish_slot(self, slots, r, driver, reason, stop_token=None,
                     stop_string=None, extra_text=""):
        slot = slots[r]
        job = slot.job
        st = slot.state
        driver.finish_row(r)
        # Record the slot's lineage: prompt + every consumed token + live cache
        # position. The next job admitted to this slot reuses it only on a strict
        # prefix extension (or prompt-boundary restore); otherwise the row resets.
        ids_cpu = job.input_ids.cpu()
        if st.all_ids:
            full = torch.cat(
                [ids_cpu, torch.tensor([st.all_ids], dtype=torch.long)], dim=-1)
        else:
            full = ids_cpu
        slot.cached_ids = full.contiguous()
        slot.cached_pos = self.tp_group.cache.seq_lens[r]
        slot.cache_valid = True
        slot.last_used = time.monotonic()
        slot.active = False
        slot.job = None
        slot.state = None
        self._finish(
            job, reason, st.new_tokens, st.emitted + extra_text,
            st.time_start, st.time_prefill, st.time_gen_start,
            stop_token=stop_token, stop_string=stop_string,
            extra_text=extra_text, cached_tokens=slot.cached_tokens,
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
        cached_tokens: int = 0,
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
            "cached_tokens": cached_tokens,
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
