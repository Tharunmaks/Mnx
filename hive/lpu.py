"""A virtual LPU: a cycle-accurate simulator of a Groq-style chip, and the deterministic compiler for it.

Virtual silicon (`Chip`): the only memory is SRAM, a banks × words 2-D array of 32-bit words. There are no
caches and no runtime decisions: every read or write takes a fixed number of cycles. Functional units (matrix
units = systolic arrays, vector units, one move port per unit, the host link, chip-to-chip links) take a fixed
number of cycles per instruction. A master tick loop advances one clock cycle per iteration; in each cycle the
instructions that end write their results into SRAM, then the instructions that start check that their inputs
arrived and that their unit is free.

The brain (`compile_model`): takes a transformer (Qwen2 / Llama shapes) and plans it exactly: which SRAM address
holds which tensor from which cycle, and when every move, matmul, vector op, load, store, send and receive starts
and ends, on which unit. The simulator never guesses: if data is not where the compiler said it would be at the
cycle it said, it stops with a ScheduleError (a compiler bug, never a stall).

Virtual network (`Network`): chips joined by fixed-latency queues (no switches), stepping in lockstep on one
global clock. A model is laid out as pipeline stages of tensor-parallel groups: a layer too big for one chip is
spread over a group (whole heads and MLP rows per chip, partial sums combined with a ring all-reduce over the
links), and groups hand the hidden state down the pipeline.

Only numpy is needed, so the same file runs on the Hive and on a phone (`stream.py run --engine lpu`).
"""

from __future__ import annotations

import functools
import heapq
import json
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ====================================================================== the chip
@dataclass(frozen=True)
class ChipSpec:
    name: str = "LPU v1 (Groq-like)"
    clock_hz: float = 900e6
    sram_bytes: int = 230_000_000       # all the memory there is: no caches, no DRAM
    word_bytes: int = 2                  # SRAM holds 16-bit values (bf16, like the model's weights); the units compute in 32-bit
    banks: int = 88                      # SRAM is a banks × words 2-D array
    sram_words_per_cycle: int = 320      # one access stream moves this many words per cycle
    sram_latency: int = 2                # fixed cycles before the first word of an access arrives
    mxm_units: int = 2                   # matrix units
    mxm_rows: int = 320                  # systolic array, weights stationary: rows × cols cells
    mxm_cols: int = 320
    vxm_units: int = 4                   # vector units
    vector_lanes: int = 320
    vxm_latency: int = 4                 # pipeline depth of a vector instruction
    bus_words_per_cycle: int = 320       # a unit's move port: SRAM → the unit's input stream
    bus_latency: int = 3
    host_bytes_per_cycle: int = 36       # host link, ~32 GB/s at 900 MHz: non-resident weights stream in here
    link_bytes_per_cycle: int = 110      # chip-to-chip link, ~100 GB/s each way
    link_latency: int = 450              # fixed cycles the last word takes to reach the next chip (~0.5 µs)

    @property
    def bank_words(self) -> int:
        return self.sram_bytes // self.word_bytes // self.banks

    @property
    def words(self) -> int:
        return self.bank_words * self.banks


class ScheduleError(RuntimeError):
    """The compiler's plan and the chip disagree: data was not ready, or a unit was double-booked."""


@dataclass(eq=False)
class Buf:
    """A tensor in SRAM: a rows × cols window on the flat word array starting at `addr`, `stride` words per row."""
    name: str
    addr: int
    rows: int
    cols: int
    stride: int
    avail: int = 0              # the cycle from which this region may be written (its previous user is done)
    base: "Buf | None" = None   # a slice shares readiness with the buffer it is cut from

    @property
    def words(self) -> int:
        return self.rows * self.cols

    @property
    def root(self) -> "Buf":
        return self.base.root if self.base is not None else self

    def cols_slice(self, name: str, c0: int, c1: int) -> "Buf":
        return Buf(name, self.addr + c0, self.rows, c1 - c0, self.stride, self.avail, base=self)


@dataclass(eq=False)
class Instr:
    op: str                 # load, move, matmul, vec, store, send, recv
    unit: str               # exact unit instance: "mxm1", "vxm0", "port.mxm1" (its move port), "host", "link.out", "link.in"
    start: int
    end: int
    reads: list = field(default_factory=list)
    writes: list = field(default_factory=list)
    args: dict = field(default_factory=dict)
    fn: str = ""            # vector op name
    key: str = ""           # host binding key / link message key
    note: str = ""
    phase: str = "body"     # "prologue" runs once (resident weights), "body" runs every token


@dataclass
class Program:
    chip: int
    instrs: list            # sorted by start cycle
    cycles: int             # last end cycle
    sram_peak_words: int
    resident: bool = True

    def phase(self, phase: str) -> "Program":
        ins = [i for i in self.instrs if i.phase == phase]
        return Program(self.chip, ins, max((i.end for i in ins), default=0), self.sram_peak_words, self.resident)

    def unit_busy(self, phase: str = "body") -> dict:
        busy: dict[str, int] = {}
        for i in self.instrs:
            if i.phase == phase:
                busy[i.unit] = busy.get(i.unit, 0) + i.end - i.start
        return busy

    def summary(self) -> dict:
        ops: dict[str, int] = {}
        for i in self.instrs:
            ops[i.op] = ops.get(i.op, 0) + 1
        return {"instructions": len(self.instrs), "ops": ops, "cycles": self.cycles, "sram_peak_words": self.sram_peak_words}


# ---------- vector ops the vector units know (numpy semantics; the compiler prices them in cycles) ----------
def _rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _rope(x, cos, sin):
    h = x.shape[-1] // 2
    rot = np.concatenate([-x[..., h:], x[..., :h]], -1)
    return x * cos + rot * sin


def _softmax_causal(s, scale):
    s = s * scale
    s = np.where(np.triu(np.ones(s.shape, dtype=bool), 1), -1e30, s)
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


VEC_OPS = {
    "rmsnorm": lambda a, args: _rmsnorm(a[0], a[1], args["eps"]),
    "rope": lambda a, args: _rope(a[0], a[1], a[2]),
    "softmax_causal": lambda a, args: _softmax_causal(a[0], args["scale"]),
    "silu_mul": lambda a, args: a[0] / (1.0 + np.exp(-a[0])) * a[1],
    "add": lambda a, args: a[0] + a[1],
    "transpose": lambda a, args: a[0].T,
}


def round_bf16(a: np.ndarray) -> np.ndarray:
    """What a 16-bit SRAM word keeps of a 32-bit result (round to nearest even, like PyTorch)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


class Chip:
    """Virtual silicon: the SRAM array plus the bookkeeping the tick loop needs."""

    def __init__(self, spec: ChipSpec, index: int = 0):
        self.spec, self.index = spec, index
        self.sram = np.zeros((spec.banks, 0), dtype=np.float32)  # grown to a program's footprint when it runs
        self._flat = self.sram.reshape(-1)
        self.cycle = 0
        self.busy: dict[str, int] = {}
        self.host_bytes = 0
        self.link_bytes = 0

    def reserve(self, words: int) -> None:
        """Back the SRAM words a program touches (the chip has spec.words; untouched banks cost no host memory)."""
        words = min(self.spec.words, words)
        rows = _cdiv(words, self.spec.bank_words)
        if rows > self.sram.shape[0] or self.sram.shape[1] == 0:
            new = np.zeros((rows, self.spec.bank_words), dtype=np.float32)
            new.reshape(-1)[:self._flat.size] = self._flat
            self.sram, self._flat = new, new.reshape(-1)

    def view(self, b: Buf) -> np.ndarray:
        return np.lib.stride_tricks.as_strided(self._flat[b.addr:], shape=(b.rows, b.cols), strides=(b.stride * 4, 4), writeable=True)

    def write(self, b: Buf, value: np.ndarray) -> None:
        self.view(b)[...] = round_bf16(value) if self.spec.word_bytes == 2 else value

    def _start(self, ins: Instr, ready: dict, unit_free: dict, inbox: dict) -> None:
        for b in ins.reads:
            if ready.get(id(b.root), 0) > ins.start:
                raise ScheduleError(f"chip {self.index}: {ins.op} {ins.note} at cycle {ins.start} reads {b.name}, "
                                    f"which is ready only at cycle {ready[id(b.root)]}")
        if unit_free.get(ins.unit, 0) > ins.start:
            raise ScheduleError(f"chip {self.index}: unit {ins.unit} is busy until cycle {unit_free[ins.unit]}, "
                                f"so {ins.op} {ins.note} cannot start at {ins.start}")
        if ins.op == "recv":
            arrived = inbox.get(ins.key)
            if arrived is None or arrived[0] > ins.start:
                raise ScheduleError(f"chip {self.index}: recv {ins.key} at cycle {ins.start}, but the link delivers it at "
                                    f"{arrived[0] if arrived else 'no time'}")
        unit_free[ins.unit] = ins.end
        self.busy[ins.unit] = self.busy.get(ins.unit, 0) + ins.end - ins.start
        for b in ins.writes:
            ready[id(b.root)] = max(ready.get(id(b.root), 0), ins.end)

    def _end(self, ins: Instr, bindings: dict, outbox: dict, inbox: dict, links: dict) -> None:
        wb = self.spec.word_bytes
        if ins.op == "load":
            dst = ins.writes[0]
            self.write(dst, np.asarray(bindings[ins.key], dtype=np.float32).reshape(dst.rows, dst.cols))
            self.host_bytes += dst.words * wb
        elif ins.op == "move":
            pass  # the words are now in the unit's input registers; the unit reads them from SRAM when it runs
        elif ins.op == "matmul":
            out = self.view(ins.reads[0]) @ self.view(ins.reads[1]).T
            if len(ins.reads) > 2:
                out = out + self.view(ins.reads[2])
            self.write(ins.writes[0], out)
        elif ins.op == "vec":
            self.write(ins.writes[0], VEC_OPS[ins.fn]([self.view(b) for b in ins.reads], ins.args))
        elif ins.op == "store":
            outbox[ins.key] = self.view(ins.reads[0]).copy()
            self.host_bytes += ins.reads[0].words * wb
        elif ins.op == "send":
            data = self.view(ins.reads[0]).copy()
            links.setdefault(ins.args["to"], {})[ins.key] = (ins.end + self.spec.link_latency, data)
            self.link_bytes += data.size * wb
        elif ins.op == "recv":
            self.write(ins.writes[0], inbox.pop(ins.key)[1])


# ====================================================================== the network: chips in lockstep
class Network:
    """Chips on one global clock, joined by fixed-latency queues (no switches)."""

    def __init__(self, spec: ChipSpec, n: int):
        self.spec = spec
        self.chips = [Chip(spec, i) for i in range(n)]

    @staticmethod
    def run(chips: list, programs: list, bindings: list, links: dict | None = None, skip_idle: bool = True,
            on_progress: Callable | None = None) -> dict:
        """The master tick loop: one iteration is one clock cycle on every chip. Ends are applied before starts.

        With skip_idle the loop jumps over cycles in which nothing starts, ends or arrives; the state after each
        cycle is identical either way, only the number of Python iterations differs."""
        links = {} if links is None else links
        inboxes = [links.setdefault(i, {}) for i in range(len(chips))]
        outboxes: list[dict] = [{} for _ in chips]
        ready: list[dict] = [{} for _ in chips]
        unit_free: list[dict] = [{} for _ in chips]
        starts = [sorted(p.instrs, key=lambda i: (i.start, i.end)) for p in programs]
        for chip, p in zip(chips, programs):
            chip.reserve(p.sram_peak_words)
        ends: list[list] = [[] for _ in chips]
        si = [0] * len(chips)
        last = max((i.end for p in programs for i in p.instrs), default=0)
        cycle = ticks = 0
        while cycle <= last:
            for k, chip in enumerate(chips):
                chip.cycle = cycle
                while ends[k] and ends[k][0][0] == cycle:
                    _, _, ins = heapq.heappop(ends[k])
                    chip._end(ins, bindings[k], outboxes[k], inboxes[k], links)
                while si[k] < len(starts[k]) and starts[k][si[k]].start == cycle:
                    ins = starts[k][si[k]]
                    chip._start(ins, ready[k], unit_free[k], inboxes[k])
                    heapq.heappush(ends[k], (ins.end, si[k], ins))
                    si[k] += 1
            ticks += 1
            if on_progress and ticks % 2048 == 0:
                on_progress(cycle, last)
            if skip_idle:
                nxt = last + 1
                for k in range(len(chips)):
                    if ends[k]:
                        nxt = min(nxt, ends[k][0][0])
                    if si[k] < len(starts[k]):
                        nxt = min(nxt, starts[k][si[k]].start)
                    for arrival, _ in inboxes[k].values():
                        nxt = min(nxt, arrival)
                cycle = max(cycle + 1, nxt)
            else:
                cycle += 1
        for k in range(len(chips)):
            if si[k] < len(starts[k]):
                raise ScheduleError(f"chip {k}: {len(starts[k]) - si[k]} instructions never started")
        return {"cycles": last, "outboxes": outboxes, "ticks": ticks}


# ====================================================================== the compiler
def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class _Alloc:
    """Static SRAM allocation with time: a region released at cycle c may be written by anything starting at or after c."""

    def __init__(self, words: int):
        self.free: list[tuple[int, int, int]] = [(0, words, 0)]  # (addr, words, free_from_cycle)
        self.peak = 0
        self.words = words

    def alloc(self, name: str, rows: int, cols: int, at: int) -> Buf:
        n = rows * cols
        best = None
        for i, (addr, size, free_from) in enumerate(self.free):
            if size >= n:
                cand = (max(free_from, at), addr, i)
                if best is None or cand < best:
                    best = cand
        if best is None:
            raise MemoryError(f"SRAM is full: {name} ({n:,} words) does not fit in the chip's {self.words:,} words")
        when, addr, i = best
        _, size, free_from = self.free[i]
        if size == n:
            self.free.pop(i)
        else:
            self.free[i] = (addr + n, size - n, free_from)
        self.peak = max(self.peak, addr + n)
        return Buf(name, addr, rows, cols, cols, avail=when)

    def release(self, b: Buf, at: int) -> None:
        self.free.append((b.addr, b.words, at))
        self.free.sort()
        merged: list[tuple[int, int, int]] = []
        for r in self.free:
            if merged and merged[-1][0] + merged[-1][1] == r[0]:
                a, s, f = merged[-1]
                merged[-1] = (a, s + r[1], max(f, r[2]))
            else:
                merged.append(r)
        self.free = merged


class Compiler:
    """List-schedules a straight-line tensor program onto one chip with exact cycles. No runtime decisions remain."""

    def __init__(self, spec: ChipSpec, chip: int = 0):
        self.spec, self.chip = spec, chip
        self.alloc = _Alloc(spec.words)
        self.instrs: list[Instr] = []
        self.ready: dict[int, int] = {}      # buf root id → cycle its data is complete
        self.last_read: dict[int, int] = {}  # buf root id → last cycle anything reads it
        self.unit_free: dict[str, int] = {}
        self.mxm = [f"mxm{i}" for i in range(spec.mxm_units)]
        self.vxm = [f"vxm{i}" for i in range(spec.vxm_units)]
        self.phase = "body"

    # ----- costs, all exact integers -----
    def c_sram(self, words: int) -> int:
        return self.spec.sram_latency + _cdiv(words, self.spec.sram_words_per_cycle)

    def c_move(self, words: int) -> int:
        return self.spec.bus_latency + _cdiv(words, self.spec.bus_words_per_cycle)

    def c_matmul(self, m: int, k: int, n: int) -> int:
        s = self.spec
        tiles = _cdiv(k, s.mxm_rows) * _cdiv(n, s.mxm_cols)
        return tiles * (s.mxm_rows + m + s.mxm_rows + s.mxm_cols) + self.c_sram(m * n)  # fill weights, stream m rows, drain

    def c_vec(self, fn: str, rows: int, cols: int) -> int:
        s = self.spec
        lanes = s.vector_lanes
        tree = math.ceil(math.log2(lanes))
        per_row = _cdiv(cols, lanes)
        passes = {"rmsnorm": rows * (2 * per_row + tree), "softmax_causal": rows * (3 * per_row + 2 * tree),
                  "rope": 2 * _cdiv(rows * cols, lanes), "silu_mul": 2 * _cdiv(rows * cols, lanes),
                  "add": _cdiv(rows * cols, lanes), "transpose": _cdiv(rows * cols, lanes)}[fn]
        return s.vxm_latency + passes + self.c_sram(rows * cols)

    # ----- scheduling -----
    def _when(self, unit: str, reads: list, writes: list, at: int = 0) -> int:
        t = max(at, self.unit_free.get(unit, 0))
        for b in reads:
            t = max(t, self.ready.get(id(b.root), 0))
        for b in writes:
            t = max(t, b.root.avail)
        return t

    def _emit(self, ins: Instr) -> Instr:
        ins.phase = self.phase
        self.unit_free[ins.unit] = ins.end
        for b in ins.reads:
            self.last_read[id(b.root)] = max(self.last_read.get(id(b.root), 0), ins.end)
        for b in ins.writes:
            self.ready[id(b.root)] = max(self.ready.get(id(b.root), 0), ins.end)
        self.instrs.append(ins)
        return ins

    def new(self, name: str, rows: int, cols: int, at: int = 0) -> Buf:
        return self.alloc.alloc(name, rows, cols, at)

    def free(self, *bufs: Buf) -> None:
        for b in bufs:
            self.alloc.release(b, max(self.last_read.get(id(b), 0), self.ready.get(id(b), 0)))

    def _pick(self, units: list, reads: list, at: int) -> str:
        return min(units, key=lambda u: (self._when(u, reads, [], at), u))

    def load(self, key: str, rows: int, cols: int, at: int = 0) -> Buf:
        """Host → SRAM over the host link (one link, so loads are serial in program order)."""
        b = self.new(key, rows, cols, at)
        start = self._when("host", [], [b], at)
        end = start + _cdiv(rows * cols * self.spec.word_bytes, self.spec.host_bytes_per_cycle)
        self._emit(Instr("load", "host", start, end, writes=[b], key=key, note=key))
        return b

    def move(self, src: Buf, unit: str, at: int = 0) -> int:
        """Route an operand to a unit: SRAM → the unit's input registers over that unit's own move port. Returns the
        cycle the last word is there; the unit may start that very cycle."""
        port = f"port.{unit}"
        start = self._when(port, [src], [], at)
        end = start + self.c_move(src.words)
        self._emit(Instr("move", port, start, end, reads=[src], args={"to": unit}, note=f"{src.name}→{unit}"))
        return end

    def matmul(self, a: Buf, w: Buf, name: str, bias: Buf | None = None, into: Buf | None = None, at: int = 0) -> Buf:
        """out[m,n] = a[m,k] · w[n,k]ᵀ (+ bias[1,n]). Operands are routed to the matrix unit first; it then runs for exact cycles."""
        unit = self._pick(self.mxm, [a, w], at)
        srcs = [a, w] + ([bias] if bias is not None else [])
        t = max(self.move(b, unit, at) for b in srcs)
        out = into if into is not None else self.new(name, a.rows, w.rows, at)
        start = self._when(unit, srcs, [out], t)
        end = start + self.c_matmul(a.rows, a.cols, w.rows)
        self._emit(Instr("matmul", unit, start, end, reads=srcs, writes=[out], note=name))
        return out

    def vec(self, fn: str, srcs: list, name: str, rows: int | None = None, cols: int | None = None,
            into: Buf | None = None, at: int = 0, **args) -> Buf:
        unit = self._pick(self.vxm, srcs, at)
        t = max(self.move(b, unit, at) for b in srcs)
        rows = srcs[0].rows if rows is None else rows
        cols = srcs[0].cols if cols is None else cols
        out = into if into is not None else self.new(name, rows, cols, at)
        start = self._when(unit, srcs, [out], t)
        end = start + self.c_vec(fn, out.rows, out.cols)
        self._emit(Instr("vec", unit, start, end, reads=srcs, writes=[out], fn=fn, args=args, note=name))
        return out

    def store(self, src: Buf, key: str) -> int:
        start = self._when("host", [src], [])
        end = start + _cdiv(src.words * self.spec.word_bytes, self.spec.host_bytes_per_cycle)
        self._emit(Instr("store", "host", start, end, reads=[src], key=key, note=key))
        return end

    def send(self, src: Buf, key: str, to: int) -> int:
        """Put a tensor on the outgoing link to chip `to`; returns the cycle its last word has arrived there."""
        start = self._when("link.out", [src], [])
        end = start + _cdiv(src.words * self.spec.word_bytes, self.spec.link_bytes_per_cycle)
        self._emit(Instr("send", "link.out", start, end, reads=[src], key=key, args={"to": to}, note=key))
        return end + self.spec.link_latency

    def recv(self, key: str, rows: int, cols: int, arrives: int, into: Buf | None = None) -> Buf:
        b = into if into is not None else self.new(key, rows, cols, arrives)
        start = self._when("link.in", [], [b], arrives)
        end = start + self.c_sram(rows * cols)
        self._emit(Instr("recv", "link.in", start, end, writes=[b], key=key, note=key))
        return b

    def now(self) -> int:
        return max((i.end for i in self.instrs), default=0)

    def program(self, resident: bool = True) -> Program:
        return Program(self.chip, sorted(self.instrs, key=lambda i: (i.start, i.end)), self.now(), self.alloc.peak, resident)


# ---------- the transformer layer, as the compiler sees it ----------
@dataclass(frozen=True)
class LayerShape:
    hidden: int
    heads: int
    kv_heads: int
    intermediate: int
    seq: int
    eps: float = 1e-6
    rope_theta: float = 10000.0
    qkv_bias: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @classmethod
    def from_config(cls, cfg: dict, seq: int) -> "LayerShape":
        rp = cfg.get("rope_parameters") or {}
        return cls(int(cfg["hidden_size"]), int(cfg["num_attention_heads"]),
                   int(cfg.get("num_key_value_heads") or cfg["num_attention_heads"]), int(cfg["intermediate_size"]), seq,
                   float(cfg.get("rms_norm_eps", 1e-6)), float(rp.get("rope_theta", cfg.get("rope_theta", 10000.0))),
                   bool(cfg.get("attention_bias", cfg.get("model_type") == "qwen2")))

    @property
    def weight_words(self) -> int:
        return sum(r * c for r, c in shard_shapes(self, Shard(self, 1, 0)).values())

    def valid_tp(self, chips: int) -> list:
        """Group sizes a layer can be spread over: whole heads per chip, kv heads shared or split evenly."""
        return [t for t in range(1, min(chips, self.heads) + 1)
                if self.heads % t == 0 and (self.kv_heads % t == 0 or t % self.kv_heads == 0) and self.intermediate >= t]

    def activation_words(self, tp: int = 1) -> int:
        """Rough peak of live activations on one chip of a group of `tp` (the exact figure comes from compiling)."""
        sh = Shard(self, tp, 0)
        d = self.head_dim
        return self.seq * (6 * self.hidden + 3 * (sh.hq + 2 * sh.hkv) * d + 3 * (sh.i1 - sh.i0) + 4 * self.seq + 2 * d)


@dataclass(frozen=True)
class Shard:
    """The slice of a layer chip `j` of a group of `tp` holds: whole q heads, their kv heads, and rows of the MLP."""
    shape: LayerShape
    tp: int
    j: int

    @property
    def hq(self) -> int:
        return self.shape.heads // self.tp

    @property
    def q0(self) -> int:
        return self.j * self.hq

    @property
    def hkv(self) -> int:
        return max(1, self.shape.kv_heads // self.tp)

    @property
    def g0(self) -> int:
        return self.j * self.hkv if self.shape.kv_heads >= self.tp else self.j // (self.tp // self.shape.kv_heads)

    def kv_local(self, h_local: int) -> int:
        return (self.q0 + h_local) // (self.shape.heads // self.shape.kv_heads) - self.g0

    @property
    def i0(self) -> int:
        return self.shape.intermediate * self.j // self.tp

    @property
    def i1(self) -> int:
        return self.shape.intermediate * (self.j + 1) // self.tp


def shard_shapes(shape: LayerShape, sh: Shard) -> dict:
    h, d, mi = shape.hidden, shape.head_dim, sh.i1 - sh.i0
    shapes = {"input_layernorm.weight": (1, h), "post_attention_layernorm.weight": (1, h),
              "self_attn.q_proj.weight": (sh.hq * d, h), "self_attn.k_proj.weight": (sh.hkv * d, h), "self_attn.v_proj.weight": (sh.hkv * d, h),
              "self_attn.o_proj.weight": (h, sh.hq * d), "mlp.gate_proj.weight": (mi, h), "mlp.up_proj.weight": (mi, h),
              "mlp.down_proj.weight": (h, mi)}
    if shape.qkv_bias:
        shapes.update({"self_attn.q_proj.bias": (1, sh.hq * d), "self_attn.k_proj.bias": (1, sh.hkv * d), "self_attn.v_proj.bias": (1, sh.hkv * d)})
    return shapes


def shard_weights(w: dict, shape: LayerShape, sh: Shard) -> dict:
    """Cut a layer's full numpy weights down to what one chip of the group holds."""
    d = shape.head_dim
    q, kv, mi = slice(sh.q0 * d, (sh.q0 + sh.hq) * d), slice(sh.g0 * d, (sh.g0 + sh.hkv) * d), slice(sh.i0, sh.i1)
    cut = {"input_layernorm.weight": w["input_layernorm.weight"], "post_attention_layernorm.weight": w["post_attention_layernorm.weight"],
           "self_attn.q_proj.weight": w["self_attn.q_proj.weight"][q], "self_attn.k_proj.weight": w["self_attn.k_proj.weight"][kv],
           "self_attn.v_proj.weight": w["self_attn.v_proj.weight"][kv], "self_attn.o_proj.weight": w["self_attn.o_proj.weight"][:, q],
           "mlp.gate_proj.weight": w["mlp.gate_proj.weight"][mi], "mlp.up_proj.weight": w["mlp.up_proj.weight"][mi],
           "mlp.down_proj.weight": w["mlp.down_proj.weight"][:, mi]}
    if shape.qkv_bias:
        cut.update({"self_attn.q_proj.bias": w["self_attn.q_proj.bias"].reshape(-1)[q], "self_attn.k_proj.bias": w["self_attn.k_proj.bias"].reshape(-1)[kv],
                    "self_attn.v_proj.bias": w["self_attn.v_proj.bias"].reshape(-1)[kv]})
    return {k: np.ascontiguousarray(v, dtype=np.float32).reshape(shard_shapes(shape, sh)[k]) for k, v in cut.items()}


@functools.lru_cache(maxsize=4096)
def shard_words(shape: LayerShape, tp: int) -> int:
    return max(sum(r * c for r, c in shard_shapes(shape, Shard(shape, tp, j)).values()) for j in range(tp))


def rope_tables(shape: LayerShape) -> tuple:
    d = shape.head_dim
    inv = 1.0 / (shape.rope_theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
    t = np.arange(shape.seq, dtype=np.float64)[:, None] * inv[None, :]
    emb = np.concatenate([t, t], -1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def load_layer_weights(c: Compiler, shape: LayerShape, sh: Shard, prefix: str, at: int = 0) -> dict:
    return {k: c.load(f"{prefix}{k}", r, cols, at) for k, (r, cols) in shard_shapes(shape, sh).items()}


def allreduce(cs: list, bufs: list, name: str) -> list:
    """Ring all-reduce over the group's links: reduce-scatter then all-gather, one column chunk per chip, exact cycles."""
    tp = len(cs)
    if tp == 1:
        return bufs
    S, H = bufs[0].rows, bufs[0].cols
    bounds = [H * i // tp for i in range(tp + 1)]
    chunk = lambda j, ci: bufs[j].cols_slice(f"{name}.c{ci}", bounds[ci], bounds[ci + 1])
    for s in range(tp - 1):
        arrive = [cs[j].send(chunk(j, (j - s) % tp), f"{name}.rs{s}", cs[(j + 1) % tp].chip) for j in range(tp)]
        for j in range(tp):
            ci = (j - 1 - s) % tp
            tmp = cs[j].recv(f"{name}.rs{s}", S, bounds[ci + 1] - bounds[ci], arrive[(j - 1) % tp])
            cs[j].vec("add", [chunk(j, ci), tmp], f"{name}.sum{ci}", into=chunk(j, ci))
            cs[j].free(tmp)
    for s in range(tp - 1):
        arrive = [cs[j].send(chunk(j, (j + 1 - s) % tp), f"{name}.ag{s}", cs[(j + 1) % tp].chip) for j in range(tp)]
        for j in range(tp):
            ci = (j - s) % tp
            cs[j].recv(f"{name}.ag{s}", S, bounds[ci + 1] - bounds[ci], arrive[(j - 1) % tp], into=chunk(j, ci))
    return bufs


def emit_layer_group(cs: list, shape: LayerShape, xs: list, ws: list, cos: list, sin: list, name: str) -> list:
    """Plan one decoder layer on a group of chips (each holds its shard, x is replicated): xs [seq, hidden] → outs."""
    tp = len(cs)
    S, D = shape.seq, shape.head_dim
    partial, h1s = [], []
    for j, c in enumerate(cs):
        sh, x, w = Shard(shape, tp, j), xs[j], ws[j]
        xn = c.vec("rmsnorm", [x, w["input_layernorm.weight"]], f"{name}.xn", eps=shape.eps)
        q = c.matmul(xn, w["self_attn.q_proj.weight"], f"{name}.q", w.get("self_attn.q_proj.bias"))
        k = c.matmul(xn, w["self_attn.k_proj.weight"], f"{name}.k", w.get("self_attn.k_proj.bias"))
        v = c.matmul(xn, w["self_attn.v_proj.weight"], f"{name}.v", w.get("self_attn.v_proj.bias"))
        c.free(xn)
        qr, kr = c.new(f"{name}.qr", S, sh.hq * D), c.new(f"{name}.kr", S, sh.hkv * D)
        for h in range(sh.hq):
            c.vec("rope", [q.cols_slice(f"{name}.q{h}", h * D, (h + 1) * D), cos[j], sin[j]], f"{name}.qr{h}",
                  into=qr.cols_slice(f"{name}.qr{h}", h * D, (h + 1) * D))
        for g in range(sh.hkv):
            c.vec("rope", [k.cols_slice(f"{name}.k{g}", g * D, (g + 1) * D), cos[j], sin[j]], f"{name}.kr{g}",
                  into=kr.cols_slice(f"{name}.kr{g}", g * D, (g + 1) * D))
        c.free(q, k)
        vt = [c.vec("transpose", [v.cols_slice(f"{name}.v{g}", g * D, (g + 1) * D)], f"{name}.vT{g}", rows=D, cols=S) for g in range(sh.hkv)]
        c.free(v)
        attn = c.new(f"{name}.attn", S, sh.hq * D)
        for h in range(sh.hq):
            g = sh.kv_local(h)
            scores = c.matmul(qr.cols_slice(f"{name}.qr{h}", h * D, (h + 1) * D), kr.cols_slice(f"{name}.kr{g}", g * D, (g + 1) * D), f"{name}.s{h}")
            probs = c.vec("softmax_causal", [scores], f"{name}.p{h}", scale=1.0 / math.sqrt(D))
            c.free(scores)
            c.matmul(probs, vt[g], f"{name}.attn{h}", into=attn.cols_slice(f"{name}.attn{h}", h * D, (h + 1) * D))
            c.free(probs)
        c.free(qr, kr, *vt)
        partial.append(c.matmul(attn, w["self_attn.o_proj.weight"], f"{name}.o"))
        c.free(attn)
    o = allreduce(cs, partial, f"{name}.o")
    partial = []
    for j, c in enumerate(cs):
        w = ws[j]
        h1 = c.vec("add", [xs[j], o[j]], f"{name}.h1")
        c.free(o[j])
        hn = c.vec("rmsnorm", [h1, w["post_attention_layernorm.weight"]], f"{name}.hn", eps=shape.eps)
        gate = c.matmul(hn, w["mlp.gate_proj.weight"], f"{name}.gate")
        up = c.matmul(hn, w["mlp.up_proj.weight"], f"{name}.up")
        c.free(hn)
        act = c.vec("silu_mul", [gate, up], f"{name}.act")
        c.free(gate, up)
        partial.append(c.matmul(act, w["mlp.down_proj.weight"], f"{name}.down"))
        c.free(act)
        h1s.append(h1)
    down = allreduce(cs, partial, f"{name}.d")
    outs = []
    for j, c in enumerate(cs):
        outs.append(c.vec("add", [h1s[j], down[j]], f"{name}.out"))
        c.free(h1s[j], down[j])
    return outs


def compile_layer(spec: ChipSpec, shape: LayerShape, tp: int = 1) -> list:
    """One layer on a group of `tp` chips: x and the shards stream in from the host, out goes back. Returns the programs."""
    cs = [Compiler(spec, chip=j) for j in range(tp)]
    xs = [c.load("x", shape.seq, shape.hidden) for c in cs]
    cos = [c.load("cos", shape.seq, shape.head_dim) for c in cs]
    sin = [c.load("sin", shape.seq, shape.head_dim) for c in cs]
    ws = [load_layer_weights(c, shape, Shard(shape, tp, j), "") for j, c in enumerate(cs)]
    for c, out in zip(cs, emit_layer_group(cs, shape, xs, ws, cos, sin, "L")):
        c.store(out, "out")
    return [c.program() for c in cs]


@dataclass
class ModelPlan:
    spec: ChipSpec
    shape: LayerShape
    n_layers: int
    vocab: int
    tp: int                      # chips per group (a layer is spread over a group)
    stages: int                  # pipeline stages (groups)
    layers_of: list              # stage → its layer indices
    resident: bool               # weights loaded into SRAM once, or streamed from the host every token
    programs: list               # chip index = stage * tp + j
    cycles_per_token: int
    sram_peak_bytes: int
    weight_bytes: int
    host_bytes_per_token: int
    link_bytes_per_token: int

    @property
    def chips(self) -> int:
        return self.tp * self.stages

    @property
    def instructions_per_token(self) -> int:
        return sum(1 for p in self.programs for i in p.instrs if i.phase == "body")


def _split_layers(n_layers: int, stages: int) -> list:
    per = _cdiv(n_layers, max(1, stages))
    return [list(range(i, min(n_layers, i + per))) for i in range(0, n_layers, per)]


def chip_fits(spec: ChipSpec, shape: LayerShape, layers_on_chip: int, vocab: int, with_head: bool, tp: int) -> bool:
    words = layers_on_chip * shard_words(shape, tp) + (_cdiv(vocab, tp) * shape.hidden + shape.hidden if with_head else 0) + shape.activation_words(tp)
    return words <= spec.words * 0.95


def stages_resident(spec: ChipSpec, shape: LayerShape, n_layers: int, vocab: int, tp: int, max_stages: int) -> int | None:
    """The fewest pipeline stages (groups of `tp`) that hold every layer and the head in SRAM, or None within max_stages."""
    for stages in range(1, min(max_stages, n_layers) + 1):
        layers_of = _split_layers(n_layers, stages)
        if all(chip_fits(spec, shape, len(ls), vocab, s == len(layers_of) - 1, tp) for s, ls in enumerate(layers_of)):
            return len(layers_of)
    return None


def chips_for_resident(spec: ChipSpec, shape: LayerShape, n_layers: int, vocab: int) -> int | None:
    best = None
    for t in shape.valid_tp(10 ** 9):
        st = stages_resident(spec, shape, n_layers, vocab, t, n_layers)
        if st and (best is None or t * st < best):
            best = t * st
    return best


def layout(spec: ChipSpec, shape: LayerShape, n_layers: int, vocab: int, chips: int, tp: int | None = None) -> tuple:
    """Group size and layers per stage for a chip budget, using the whole budget (every chip gets layers): the smallest
    group that keeps the weights resident; else the smallest group a layer fits on at all (streaming); else the largest."""
    chips = max(1, int(chips))
    valid = shape.valid_tp(chips)
    if tp is not None:
        tp = max(t for t in valid if t <= tp)
        return tp, _split_layers(n_layers, chips // tp)
    fitting = [t for t in valid if chip_fits(spec, shape, 1, vocab, True, t)]
    for t in fitting:
        layers_of = _split_layers(n_layers, chips // t)
        if all(chip_fits(spec, shape, len(ls), vocab, s == len(layers_of) - 1, t) for s, ls in enumerate(layers_of)):
            return t, layers_of
    tp = fitting[0] if fitting else valid[-1]
    return tp, _split_layers(n_layers, chips // tp)


def compile_model(spec: ChipSpec, shape: LayerShape, n_layers: int, vocab: int, chips: int = 1, resident: bool | None = None,
                  tp: int | None = None) -> ModelPlan:
    """The whole model as pipeline stages of tensor-parallel groups. Everything is scheduled against one global clock:
    a stage's first instruction starts the exact cycle the previous stage's output arrives over the links."""
    tp, layers_of = layout(spec, shape, n_layers, vocab, chips, tp)
    stages = len(layers_of)
    if resident is None:
        resident = all(chip_fits(spec, shape, len(ls), vocab, s == stages - 1, tp) for s, ls in enumerate(layers_of))
    programs: list = []
    arrive = [0] * tp
    S, H = shape.seq, shape.hidden
    vb = [vocab * j // tp for j in range(tp + 1)]
    for s, ls in enumerate(layers_of):
        cs = [Compiler(spec, chip=s * tp + j) for j in range(tp)]
        last = s == stages - 1
        for c in cs:
            c.phase = "prologue" if resident else "body"
        cos = [c.load("cos", S, shape.head_dim) for c in cs]
        sin = [c.load("sin", S, shape.head_dim) for c in cs]
        weights: dict[int, list] = {}
        norm_w = head_w = None
        if resident:
            for i in ls:
                weights[i] = [load_layer_weights(c, shape, Shard(shape, tp, j), f"layer{i}.") for j, c in enumerate(cs)]
            if last:
                norm_w = [c.load("norm", 1, H) for c in cs]
                head_w = [c.load("head", vb[j + 1] - vb[j], H) for j, c in enumerate(cs)]
        t0 = [c.now() for c in cs]
        for c in cs:
            c.phase = "body"
        xs = [c.load("x", S, H, at=t0[j]) if s == 0 else c.recv("x", S, H, arrive[j]) for j, c in enumerate(cs)]
        for i in ls:
            ws = weights.get(i) or [load_layer_weights(c, shape, Shard(shape, tp, j), f"layer{i}.", at=t0[j]) for j, c in enumerate(cs)]
            outs = emit_layer_group(cs, shape, xs, ws, cos, sin, f"layer{i}")
            for j, c in enumerate(cs):
                if not resident:
                    c.free(*ws[j].values())
                c.free(xs[j])
            xs = outs
        if last:
            if not resident:
                norm_w = [c.load("norm", 1, H, at=t0[j]) for j, c in enumerate(cs)]
                head_w = [c.load("head", vb[j + 1] - vb[j], H, at=t0[j]) for j, c in enumerate(cs)]
            for j, c in enumerate(cs):
                xn = c.vec("rmsnorm", [xs[j], norm_w[j]], "final.norm", eps=shape.eps)
                c.store(c.matmul(xn, head_w[j], "logits"), "logits")
        else:
            arrive = [c.send(xs[j], "x", (s + 1) * tp + j) for j, c in enumerate(cs)]
        programs += [c.program(resident) for c in cs]
    body = [i for p in programs for i in p.instrs if i.phase == "body"]
    wb = spec.word_bytes
    cycles = max(i.end for i in body) - min(i.start for i in body)
    host = sum(i.writes[0].words * wb for i in body if i.op == "load") + sum(i.reads[0].words * wb for i in body if i.op == "store")
    link = sum(i.reads[0].words * wb for i in body if i.op == "send")
    return ModelPlan(spec, shape, n_layers, vocab, tp, stages, layers_of, resident, programs, cycles,
                     max(p.sram_peak_words for p in programs) * wb, (n_layers * shape.weight_words + vocab * H + H) * wb, host, link)


# ====================================================================== running a real model
def bf16_to_f32(raw: bytes, shape: list) -> np.ndarray:
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)


def read_safetensors(path: str) -> dict:
    """Every tensor of a safetensors file as float32 numpy, no torch."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            s, e = info["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            if info["dtype"] == "BF16":
                out[name] = bf16_to_f32(raw, info["shape"])
            else:
                dt = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "I64": np.int64, "I32": np.int32}[info["dtype"]]
                out[name] = np.frombuffer(raw, dtype=dt).reshape(info["shape"]).astype(np.float32)
    return out


def strip_layer_prefix(tensors: dict, i: int) -> dict:
    key = f".layers.{i}."
    return {k.split(key, 1)[1]: v for k, v in tensors.items() if key in k}


class Model:
    """A model on the virtual LPU(s). `load_layer(i)` and `shared()` return numpy weights (from files, or a phone's streaming Store)."""

    def __init__(self, cfg: dict, load_layer: Callable, shared: Callable, seq: int, spec: ChipSpec | None = None,
                 chips: int = 1, resident: bool | None = None, tp: int | None = None):
        self.spec = spec or ChipSpec()
        self.cfg = cfg
        self.n_layers, self.vocab = int(cfg["num_hidden_layers"]), int(cfg["vocab_size"])
        self.shape = LayerShape.from_config(cfg, seq)
        self.plan = compile_model(self.spec, self.shape, self.n_layers, self.vocab, chips, resident, tp)
        self.net = Network(self.spec, self.plan.chips)
        self.load_layer, self.shared_fn = load_layer, shared
        self._shared = None
        self.cos, self.sin = rope_tables(self.shape)
        self.prologue_done = False
        self.stats = {"tokens": 0, "ticks": 0, "sim_seconds": 0.0, "host_bytes": 0, "link_bytes": 0, "weight_bytes_read": 0}

    def shared(self) -> dict:
        if self._shared is None:
            self._shared = self.shared_fn()
        return self._shared

    def _find(self, suffix: str) -> np.ndarray:
        for k, v in self.shared().items():
            if k.endswith(suffix) and ".layers." not in k:
                return v
        raise KeyError(suffix)

    def embed(self, ids: list) -> np.ndarray:
        return self._find("embed_tokens.weight")[np.asarray(ids)]

    def _bindings(self, x: np.ndarray | None, with_weights: bool) -> list:
        P = self.plan
        out = [{"cos": self.cos, "sin": self.sin} for _ in range(P.chips)]
        if x is not None:
            for j in range(P.tp):
                out[j]["x"] = x
        if with_weights:
            for s, ls in enumerate(P.layers_of):
                for i in ls:
                    w = self.load_layer(i)
                    self.stats["weight_bytes_read"] += sum(v.size * self.spec.word_bytes for v in w.values())
                    for j in range(P.tp):
                        for name, arr in shard_weights(w, self.shape, Shard(self.shape, P.tp, j)).items():
                            out[s * P.tp + j][f"layer{i}.{name}"] = arr
            try:
                head = self._find("lm_head.weight")
            except KeyError:
                head = self._find("embed_tokens.weight")
            norm = self._find("norm.weight").reshape(1, -1)
            for j in range(P.tp):
                b = out[(P.stages - 1) * P.tp + j]
                b["norm"], b["head"] = norm, head[self.vocab * j // P.tp:self.vocab * (j + 1) // P.tp]
        return out

    def forward(self, ids: list, on_progress: Callable | None = None) -> np.ndarray:
        """Logits [seq, vocab] for ids padded to the compiled length (attention is causal, so padding never reaches real positions)."""
        S, P = self.shape.seq, self.plan
        ids = list(ids)[:S] + [0] * (S - len(ids))
        if P.resident and not self.prologue_done:
            Network.run(self.net.chips, [p.phase("prologue") for p in P.programs], self._bindings(None, True))
            self.prologue_done = True
        x = self.embed(ids).astype(np.float32)
        t0 = time.time()
        host0 = sum(c.host_bytes for c in self.net.chips)
        link0 = sum(c.link_bytes for c in self.net.chips)
        res = Network.run(self.net.chips, [p.phase("body") for p in P.programs], self._bindings(x, not P.resident), on_progress=on_progress)
        self.stats["tokens"] += 1
        self.stats["ticks"] += res["ticks"]
        self.stats["sim_seconds"] += time.time() - t0
        self.stats["host_bytes"] += sum(c.host_bytes for c in self.net.chips) - host0
        self.stats["link_bytes"] += sum(c.link_bytes for c in self.net.chips) - link0
        last = (P.stages - 1) * P.tp
        return np.concatenate([res["outboxes"][last + j]["logits"] for j in range(P.tp)], axis=1)

    def generate(self, ids: list, max_new: int, temperature: float = 0.7, eos: int | None = None,
                 on_token: Callable | None = None, seed: int = 0) -> list:
        rng = np.random.default_rng(seed)
        ids, out = list(ids), []
        for _ in range(max_new):
            if len(ids) >= self.shape.seq:
                break
            logits = self.forward(ids)[len(ids) - 1]
            if temperature > 0:
                p = np.exp((logits - logits.max()) / temperature)
                nxt = int(rng.choice(len(p), p=p / p.sum()))
            else:
                nxt = int(np.argmax(logits))
            out.append(nxt)
            ids.append(nxt)
            if on_token:
                on_token(len(out), nxt, self.token_stats())
            if eos is not None and nxt == eos:
                break
        return out

    def token_stats(self) -> dict:
        s, n, P = self.stats, max(1, self.stats["tokens"]), self.plan
        busy: dict[str, int] = {}
        for p in P.programs:
            for u, cyc in p.unit_busy("body").items():
                busy[u] = busy.get(u, 0) + cyc
        util = {u: round(100 * cyc / (P.cycles_per_token * P.chips), 1) for u, cyc in sorted(busy.items()) if not u.startswith("port.")}
        return {"tokens": s["tokens"], "cycles_per_token": P.cycles_per_token, "instructions_per_token": P.instructions_per_token,
                "chip_seconds_per_token": P.cycles_per_token / self.spec.clock_hz, "chip_tokens_per_second": self.spec.clock_hz / P.cycles_per_token,
                "sim_seconds_per_token": s["sim_seconds"] / n, "ticks_per_token": s["ticks"] / n,
                "host_mb_per_token": s["host_bytes"] / n / 1e6, "link_mb_per_token": s["link_bytes"] / n / 1e6,
                "chips": P.chips, "tp": P.tp, "stages": P.stages, "resident": P.resident, "utilisation": util}


# ====================================================================== honest planning (no weights needed)
def _fmt_bytes(n: float) -> str:
    for unit, div in (("PB", 1e15), ("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n:.0f} B"


def _fmt_time(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1e3:.1f} ms"
    if s < 90:
        return f"{s:.1f} seconds"
    if s < 5400:
        return f"{s / 60:.0f} minutes"
    if s < 172800:
        return f"{s / 3600:.1f} hours"
    return f"{s / 86400:.0f} days"


def plan(cfg: dict, params: float, seq: int = 64, chips: int = 1, spec: ChipSpec | None = None,
         disk_bytes_per_s: float = 1.5e9, host_flops: float = 3e10, instr_seconds: float = 3e-5) -> dict:
    """What running this model on `chips` virtual LPUs costs: exact compiled cycles, and what simulating it here costs."""
    spec = spec or ChipSpec()
    shape = LayerShape.from_config(cfg, seq)
    n_layers, vocab = int(cfg["num_hidden_layers"]), int(cfg["vocab_size"])
    tp, layers_of = layout(spec, shape, n_layers, vocab, chips)
    stages = len(layers_of)
    wb = spec.word_bytes
    layer_bytes, head_bytes = shape.weight_words * wb, (vocab * shape.hidden + shape.hidden) * wb
    weight_bytes = n_layers * layer_bytes + head_bytes
    fits = chip_fits(spec, shape, 1, vocab, True, tp)
    min_tp = next((t for t in shape.valid_tp(10 ** 9) if chip_fits(spec, shape, 1, vocab, True, t)), None)
    resident = fits and all(chip_fits(spec, shape, len(ls), vocab, s == stages - 1, tp) for s, ls in enumerate(layers_of))
    chips_resident = chips_for_resident(spec, shape, n_layers, vocab)
    out = {"chip": spec.name, "chips": tp * stages, "tp": tp, "stages": stages, "layers_per_chip": len(layers_of[0]), "resident": resident,
           "fits": fits, "min_chips": min_tp, "chips_for_resident": chips_resident, "sram_per_chip": _fmt_bytes(spec.sram_bytes),
           "word_bits": wb * 8, "weights": _fmt_bytes(weight_bytes), "weight_bytes": weight_bytes, "layer": _fmt_bytes(layer_bytes),
           "shard": _fmt_bytes(shard_words(shape, tp) * wb), "seq": seq, "clock_mhz": spec.clock_hz / 1e6, "max_tp": shape.valid_tp(10 ** 9)[-1],
           "source": "SRAM, loaded once" if resident else f"the host link ({spec.host_bytes_per_cycle * spec.clock_hz / 1e9:.0f} GB/s per chip), every token"}
    if not fits:
        out.update(cycles_per_token=None, exact=False, per_token="never: it doesn't fit", sim_per_token="n/a", instructions_per_token=0,
                   chip_seconds_per_token=None, chip_tokens_per_second=0, sim_seconds_per_token=None)
        return out
    per_chip_instrs = (shape.heads // tp) * 8 + 40 + 6 * (tp - 1)
    exact = n_layers * tp * per_chip_instrs <= 60000
    if exact:
        try:
            mp = compile_model(spec, shape, n_layers, vocab, chips, resident)
        except MemoryError:
            resident = False
            try:
                mp = compile_model(spec, shape, n_layers, vocab, chips, False)
            except MemoryError:
                out.update(fits=False, resident=False, cycles_per_token=None, exact=True, per_token="never: it doesn't fit", sim_per_token="n/a",
                           instructions_per_token=0, chip_seconds_per_token=None, chip_tokens_per_second=0, sim_seconds_per_token=None)
                return out
        out["resident"] = resident
        cycles, n_instr = mp.cycles_per_token, mp.instructions_per_token
    else:  # one exact layer on its group, multiplied out
        progs = compile_layer(spec, shape, tp)
        body = [i for p in progs for i in p.instrs if i.op not in ("load", "store")]
        layer_cycles = max(i.end for i in body) - min(i.start for i in body)
        stream = 0 if resident else _cdiv(shard_words(shape, tp) * wb, spec.host_bytes_per_cycle)
        hop = spec.link_latency + _cdiv(seq * shape.hidden * wb, spec.link_bytes_per_cycle)
        c = Compiler(spec)
        cycles = n_layers * max(layer_cycles, stream) + (stages - 1) * hop + c.c_matmul(seq, shape.hidden, _cdiv(vocab, tp))
        n_instr = n_layers * tp * per_chip_instrs
    chip_sec = cycles / spec.clock_hz
    sim_sec = 2 * params * seq / host_flops + (0 if resident else weight_bytes / disk_bytes_per_s) + n_instr * instr_seconds
    out.update(cycles_per_token=cycles, exact=exact, instructions_per_token=n_instr, chip_seconds_per_token=chip_sec,
               chip_tokens_per_second=1 / chip_sec, per_token=_fmt_time(chip_sec), sim_seconds_per_token=sim_sec,
               sim_per_token="under a second" if sim_sec < 1 else _fmt_time(sim_sec))
    return out


def plan_text(p: dict, name: str) -> str:
    chips = f"{p['chips']} virtual {p['chip']} chip{'s' if p['chips'] != 1 else ''} ({p['sram_per_chip']} of SRAM each, {p['word_bits']}-bit words)"
    if p["tp"] > 1:
        chips += f", every layer spread over {p['tp']} chips" + (f" in {p['stages']} pipeline stages" if p["stages"] > 1 else "")
    elif p["stages"] > 1:
        chips += f" as a {p['stages']}-stage pipeline"
    if not p["fits"]:
        if p["min_chips"]:
            return (f"{name} can't run on {chips}: one chip's share of a layer ({p['shard']}) plus its activations is bigger than its SRAM, "
                    f"and there is no other memory on an LPU. Spread over at least {p['min_chips']} chips a layer fits.")
        return (f"{name} can't run on {chips}: a layer can be spread over at most {p['max_tp']} chips (one head each), and even then one "
                f"chip's share plus its activations is bigger than its SRAM. There is no other memory on an LPU.")
    s = (f"On {chips}, {name} {'keeps its ' + p['weights'] + ' of weights in SRAM' if p['resident'] else 'streams its ' + p['weights'] + ' of weights over the host links every token'}: "
         f"{p['cycles_per_token']:,} cycles per token, {p['per_token']} at {p['clock_mhz']:.0f} MHz, so {p['chip_tokens_per_second']:,.0f} tokens/s on the chip"
         f"{'' if p['exact'] else ' (one exact layer, multiplied out)'}. ")
    if not p["resident"]:
        s += (f"It would take {p['chips_for_resident']:,} chips to keep the weights resident. " if p["chips_for_resident"]
              else "No number of chips keeps the weights resident. ")
    s += f"Simulating one token here takes {p['sim_per_token']}: the chip is software, every multiply really runs."
    return s
