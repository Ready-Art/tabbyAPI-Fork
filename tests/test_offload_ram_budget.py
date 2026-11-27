"""
GPU-free unit tests for the prompt-cache RAM-offload host-RAM budget
(backends/exllamav2/fork_generator.py).

2026-11-26 server.log: a near-full cold store (75 GB of a 90 GB cap on a
125 GB no-swap box) plus one post-prefill pinned ring snapshot OOM-killed a
TP worker -> EOFError -> BrokenPipe FATAL cascade. The fix evicts coldest
entries BEFORE the offload allocates, against both the configured cap and
physical MemAvailable headroom (the offload transiently holds ~2x the bundle:
pinned staging + the stored copy). These tests exercise that budget logic
with a scripted MemAvailable; the real D2H/H2D paths are on-GPU integration.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Allow direct invocation (python tests/test_offload_ram_budget.py) from any cwd:
# put the tabbyAPI-Fork root on sys.path so `backends` / `common` resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from backends.exllamav2 import fork_generator as fg


MB = 2**20


def _fake_pg(world=4, full_layers=2, swa_layers=3, i8_layers=1,
             ring=128, kh=4, hd=16, batch=2):
    """Master-rank cache geometry stand-in for _estimate_offload_bytes.

    fp16 full layers: (batch, max_seq, kh, hd); SWA ring layers:
    (batch, ring, kh, hd); INT8 primary: int8 K/V + fp16 scales.
    """
    max_seq = 1024
    ks, vs, i8 = [], [], {}
    for l in range(full_layers + swa_layers + i8_layers):
        if l < i8_layers:
            ks.append(None)
            vs.append(None)
            i8[l] = (
                torch.zeros(batch, max_seq, kh, hd, dtype=torch.int8),
                torch.zeros(batch, max_seq, kh, hd, dtype=torch.int8),
                torch.zeros(batch, max_seq, kh, dtype=torch.half),
                torch.zeros(batch, max_seq, kh, dtype=torch.half),
            )
        elif l < i8_layers + swa_layers:
            ks.append(torch.zeros(batch, ring, kh, hd, dtype=torch.half))
            vs.append(torch.zeros(batch, ring, kh, hd, dtype=torch.half))
        else:
            ks.append(torch.zeros(batch, max_seq, kh, hd, dtype=torch.half))
            vs.append(torch.zeros(batch, max_seq, kh, hd, dtype=torch.half))
    cache = SimpleNamespace(key_states=ks, value_states=vs, kv_i8=i8,
                            kv_state_size=ring, batch_size=batch)
    workers = {d: Mock() for d in range(1, world)}
    return SimpleNamespace(cache=cache, workers=workers)


def _bundle(eid, size):
    return {"entry_id": eid, "bytes": size, "local_snap": None,
            "master_flat": None, "shadow": None}


class TestEstimateOffloadBytes(unittest.TestCase):
    def test_matches_pack_walk(self):
        pg = _fake_pg(world=4, full_layers=2, swa_layers=3, i8_layers=1,
                      ring=128, kh=4, hd=16)
        pos, wb = 300, 100
        n = min(pos - wb, 128)
        # per rank: i8 (2*pos*kh*hd + 4*pos*kh) + swa (4*n*kh*hd)
        #           + full (4*pos*kh*hd); snapshot: 4*ring*kh*hd per swa layer
        per_rank = (2 * pos * 4 * 16 + 4 * pos * 4
                    + 3 * 4 * n * 4 * 16
                    + 2 * 4 * pos * 4 * 16
                    + 3 * 4 * 128 * 4 * 16)
        self.assertEqual(fg._estimate_offload_bytes(pg, 0, pos, wb),
                         per_rank * 4)

    def test_zero_window_fill(self):
        pg = _fake_pg(world=2, full_layers=1, swa_layers=1, i8_layers=0,
                      ring=64, kh=2, hd=8)
        # pos == wb: SWA pack rows 0, snapshot still travels with the bundle
        est = fg._estimate_offload_bytes(pg, 0, 50, 50)
        per_rank = 4 * 50 * 2 * 8 + 4 * 64 * 2 * 8
        self.assertEqual(est, per_rank * 2)


class TestMakeRoom(unittest.TestCase):
    def setUp(self):
        # The budget paths log evictions/skips through loguru's default console
        # sink in tests (setup_logger is not called). Silence them so expected
        # WARNING lines (e.g. the deliberate low-RAM skip test) are not mistaken
        # for test failures.
        for level in ("info", "warning"):
            p = patch.object(fg.xlogger, level, Mock())
            p.start()
            self.addCleanup(p.stop)

    def _store(self, cap_mb, sizes):
        pg = Mock()
        pg.drop_offload_bundle = Mock()
        store = fg._ColdPromptStore(pg, cap_mb)
        for i, s in enumerate(sizes):
            rec = store.put(_bundle(f"e{i}", s), torch.zeros(1, dtype=torch.long),
                            100, 0, 100)
            self.assertIsNotNone(rec)
            # age entries deterministically: e0 coldest
            rec["last_used"] = float(i)
        return store

    def test_no_eviction_when_room(self):
        store = self._store(1000, [100 * MB, 100 * MB])
        with patch.object(fg, "_mem_available_bytes", return_value=64 * 2**30):
            self.assertTrue(store.make_room(50 * MB))
        self.assertEqual(len(store.entries), 2)

    def test_cap_eviction_before_offload(self):
        store = self._store(300, [100 * MB, 100 * MB, 100 * MB])
        with patch.object(fg, "_mem_available_bytes", return_value=64 * 2**30):
            self.assertTrue(store.make_room(150 * MB))
        # 300 + 150 > 300 -> evict coldest (e0) -> 200 + 150 > 300 -> evict e1
        self.assertEqual(list(store.entries), ["e2"])

    def test_entry_larger_than_cap_skipped(self):
        store = self._store(300, [100 * MB])
        with patch.object(fg, "_mem_available_bytes", return_value=64 * 2**30):
            self.assertFalse(store.make_room(400 * MB))
        self.assertEqual(len(store.entries), 1)   # nothing evicted pointlessly

    def test_physical_headroom_eviction(self):
        store = self._store(10000, [1000 * MB] * 5)
        # MemAvailable models pageable eviction returning RAM to the OS:
        # 2 GiB base + everything evicted so far.
        initial = store.total

        def avail():
            return 2 * 2**30 + (initial - store.total)

        with patch.object(fg, "_mem_available_bytes", side_effect=avail), \
             patch.dict(os.environ, {"EXL2_OFFLOAD_RAM_MARGIN_MB": "2048"}):
            # need = 2*1000MB + 2048MB = 4048MB; avail starts 2048MB -> evict
            # until freed >= 2048MB -> 2 entries evicted (2000MB) -> 4048MB ok
            self.assertTrue(store.make_room(1000 * MB))
        self.assertEqual(len(store.entries), 3)

    def test_headroom_exhausted_skips(self):
        store = self._store(10000, [1000 * MB, 1000 * MB])
        with patch.object(fg, "_mem_available_bytes", return_value=1 * 2**30), \
             patch.dict(os.environ, {"EXL2_OFFLOAD_RAM_MARGIN_MB": "8192"}):
            # need = 2*500MB + 8192MB = 9192MB, avail fixed 1GiB, evict all ->
            # still short -> skip (degrade to re-prefill), never OOM the box
            self.assertFalse(store.make_room(500 * MB))
        self.assertEqual(len(store.entries), 0)

    def test_no_meminfo_falls_back_to_cap_only(self):
        store = self._store(300, [100 * MB, 100 * MB])
        with patch.object(fg, "_mem_available_bytes", return_value=None):
            self.assertTrue(store.make_room(50 * MB))
        self.assertEqual(len(store.entries), 2)


if __name__ == "__main__":
    unittest.main()
