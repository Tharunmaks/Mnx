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


def _rms_r(x, eps):
    return 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def _rmsnorm_bwd_x(dy, x, w, eps):
    r = _rms_r(x, eps)
    g = dy * w
    return r * (g - x * (r * r) * (g * x).mean(-1, keepdims=True))


def _rope_bwd(dy, cos, sin):
    h = dy.shape[-1] // 2
    t = dy * sin
    return dy * cos + np.concatenate([t[..., h:], -t[..., :h]], -1)


def _softmax_rows(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _ce_loss(logits, onehot):
    z = logits - logits.max(-1, keepdims=True)
    lse = np.log(np.exp(z).sum(-1, keepdims=True))
    return np.array([[float((lse - (z * onehot).sum(-1, keepdims=True)).mean())]], dtype=np.float32)


def _sigmoid(g):
    return 1.0 / (1.0 + np.exp(-g))


VEC_OPS = {
    "rmsnorm": lambda a, args: _rmsnorm(a[0], a[1], args["eps"]),
    "rope": lambda a, args: _rope(a[0], a[1], a[2]),
    "softmax_causal": lambda a, args: _softmax_causal(a[0], args["scale"]),
    "silu_mul": lambda a, args: a[0] * _sigmoid(a[0]) * a[1],
    "add": lambda a, args: a[0] + a[1],
    "transpose": lambda a, args: a[0].T,
    "copy": lambda a, args: a[0],
    # backward
    "rmsnorm_bwd_x": lambda a, args: _rmsnorm_bwd_x(a[0], a[1], a[2], args["eps"]),
    "rmsnorm_bwd_w": lambda a, args: (a[0] * a[1] * _rms_r(a[1], args["eps"])).sum(0, keepdims=True),
    "rope_bwd": lambda a, args: _rope_bwd(a[0], a[1], a[2]),
    "softmax_bwd": lambda a, args: a[1] * (a[0] - (a[0] * a[1]).sum(-1, keepdims=True)) * args["scale"],
    "silu_bwd_gate": lambda a, args: a[0] * a[2] * (_sigmoid(a[1]) * (1.0 + a[1] * (1.0 - _sigmoid(a[1])))),
    "silu_bwd_up": lambda a, args: a[0] * a[1] * _sigmoid(a[1]),
    "ce_grad": lambda a, args: (_softmax_rows(a[0]) - a[1]) / a[0].shape[0],
    "ce_loss": lambda a, args: _ce_loss(a[0], a[1]),
    "colsum": lambda a, args: a[0].sum(0, keepdims=True),
    "sgd": lambda a, args: a[0] - args["lr"] * a[1],
}
# vector-unit passes per op: (per row, with a lane tree) or (per element)
VEC_COST = {"rmsnorm": (2, 1), "softmax_causal": (3, 2), "rmsnorm_bwd_x": (3, 1), "rmsnorm_bwd_w": (1, 1), "softmax_bwd": (2, 1),
            "ce_grad": (3, 2), "ce_loss": (3, 2), "colsum": (1, 1),
            "rope": (2, 0), "rope_bwd": (2, 0), "silu_mul": (2, 0), "silu_bwd_gate": (3, 0), "silu_bwd_up": (2, 0),
            "add": (1, 0), "transpose": (1, 0), "copy": (1, 0), "sgd": (1, 0)}


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
        self.reading: dict[int, int] = {}  # buffer → the last cycle an instruction that has started reads it
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
        for b in ins.writes:
            if self.reading.get(id(b.root), 0) > ins.end:
                raise ScheduleError(f"chip {self.index}: {ins.op} {ins.note} overwrites {b.name} at cycle {ins.end} while an earlier "
                                    f"instruction still reads it until cycle {self.reading[id(b.root)]}")
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
        for b in ins.reads:
            self.reading[id(b.root)] = max(self.reading.get(id(b.root), 0), ins.end)
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
            a, w = self.view(ins.reads[0]), self.view(ins.reads[1])
            out = (a.T if ins.args.get("ta") else a) @ (w if ins.args.get("tb") else w.T)
            if ins.args.get("bias"):
                out = out + self.view(ins.reads[2])
            if ins.args.get("acc"):
                out = out + self.view(ins.writes[0])
            self.write(ins.writes[0], out)
        elif ins.op == "vec":
            self.write(ins.writes[0], VEC_OPS[ins.fn]([self.view(b) for b in ins.reads], ins.args))
        elif ins.op == "store":
            outbox[ins.key] = self.view(ins.reads[0]).copy()
            self.host_bytes += ins.reads[0].words * wb
        elif ins.op == "send":
            data = self.view(ins.reads[0]).copy()
            inbox_to = links.setdefault(ins.args["to"], {})
            if ins.key in inbox_to:
                raise ScheduleError(f"chip {self.index}: message {ins.key} to chip {ins.args['to']} would overwrite one not yet received")
            inbox_to[ins.key] = (ins.end + self.spec.link_latency, data)
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
        for chip in chips:
            chip.reading = {}
        outboxes: list[dict] = [{} for _ in chips]
        ready: list[dict] = [{} for _ in chips]
        unit_free: list[dict] = [{} for _ in chips]
        starts = [sorted(p.instrs, key=lambda i: (i.start, i.end)) for p in programs]
        for chip, p in zip(chips, programs):
            chip.reserve(max((b.addr + b.words for i in p.instrs for b in i.reads + i.writes), default=0))
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
        self.peak = 0           # highest address ever touched
        self.used = self.peak_used = 0  # words allocated now / at most
        self.words = words

    def alloc(self, name: str, rows: int, cols: int, at: int, side: str = "high") -> Buf:
        """Long-lived tensors (weights, inputs) grow from the low addresses, short-lived ones from the high end, so the
        free space in between stays in one piece."""
        n = rows * cols
        best = None
        for i, (addr, size, free_from) in enumerate(self.free):
            if size >= n:
                cand = (max(free_from, at), addr if side == "low" else -(addr + size), i)
                if best is None or cand < best:
                    best = cand
        if best is None:
            raise MemoryError(f"SRAM is full: {name} ({n:,} words) does not fit in the chip's {self.words:,} words")
        when, _, i = best
        addr, size, free_from = self.free[i]
        if size == n:
            self.free.pop(i)
        elif side == "low":
            self.free[i] = (addr + n, size - n, free_from)
        else:
            self.free[i] = (addr, size - n, free_from)
            addr = addr + size - n
        self.peak = max(self.peak, addr + n)
        self.used += n
        self.peak_used = max(self.peak_used, self.used)
        return Buf(name, addr, rows, cols, cols, avail=when)

    def release(self, b: Buf, at: int) -> None:
        self.used -= b.words
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
        """Vector-unit cycles over a rows × cols input: whole passes over the lanes, plus a lane tree per row reduction."""
        s = self.spec
        lanes = s.vector_lanes
        passes, trees = VEC_COST[fn]
        if trees:
            return s.vxm_latency + rows * (passes * _cdiv(cols, lanes) + trees * math.ceil(math.log2(lanes)))
        return s.vxm_latency + passes * _cdiv(rows * cols, lanes)

    # ----- scheduling -----
    def _when(self, unit: str, reads: list, writes: list, at: int = 0) -> int:
        """Earliest start: the unit is free, every input is complete (RAW), and nothing emitted earlier still reads what
        this instruction overwrites (WAR), since an instruction reads its operands when it finishes."""
        t = max(at, self.unit_free.get(unit, 0))
        for b in reads:
            t = max(t, self.ready.get(id(b.root), 0))
        for b in writes:
            t = max(t, b.root.avail, self.last_read.get(id(b.root), 0))
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

    def new(self, name: str, rows: int, cols: int, at: int = 0, side: str = "high") -> Buf:
        return self.alloc.alloc(name, rows, cols, at, side)

    def free(self, *bufs: Buf) -> None:
        for b in bufs:
            self.alloc.release(b, max(self.last_read.get(id(b), 0), self.ready.get(id(b), 0)))

    def _pick(self, units: list, reads: list, at: int) -> str:
        return min(units, key=lambda u: (self._when(u, reads, [], at), u))

    def load(self, key: str, rows: int, cols: int, at: int = 0) -> Buf:
        """Host → SRAM over the host link (one link, so loads are serial in program order)."""
        b = self.new(key, rows, cols, at, side="low")
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

    def matmul(self, a: Buf, w: Buf, name: str, bias: Buf | None = None, into: Buf | None = None, at: int = 0,
               ta: bool = False, tb: bool = False, acc: bool = False) -> Buf:
        """out[m,n] = A · B (+ bias[1,n]), A = aᵀ if ta else a (so [m,k]), B = w if tb else wᵀ (so [k,n]); acc adds into `into`.
        Operands are routed to the matrix unit first; the systolic array then runs for exact cycles."""
        m, k = (a.cols, a.rows) if ta else (a.rows, a.cols)
        k2, n = (w.rows, w.cols) if tb else (w.cols, w.rows)
        if k != k2:
            raise ValueError(f"{name}: inner sizes differ ({k} vs {k2})")
        unit = self._pick(self.mxm, [a, w], at)
        srcs = [a, w] + ([bias] if bias is not None else [])
        t = max(self.move(b, unit, at) for b in srcs)
        out = into if into is not None else self.new(name, m, n, at)
        reads = srcs + ([out] if acc else [])
        start = self._when(unit, reads, [out], t)
        end = start + self.c_matmul(m, k, n) + (self.c_sram(m * n) if acc else 0)
        self._emit(Instr("matmul", unit, start, end, reads=reads, writes=[out], args={"ta": ta, "tb": tb, "acc": acc, "bias": bias is not None}, note=name))
        return out

    def vec(self, fn: str, srcs: list, name: str, rows: int | None = None, cols: int | None = None,
            into: Buf | None = None, at: int = 0, **args) -> Buf:
        unit = self._pick(self.vxm, srcs, at)
        t = max(self.move(b, unit, at) for b in srcs)
        rows = srcs[0].rows if rows is None else rows
        cols = srcs[0].cols if cols is None else cols
        out = into if into is not None else self.new(name, rows, cols, at)
        start = self._when(unit, srcs, [out], t)
        end = start + self.c_vec(fn, max(b.rows for b in srcs), max(b.cols for b in srcs)) + self.c_sram(out.words)
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
        return Program(self.chip, sorted(self.instrs, key=lambda i: (i.start, i.end)), self.now(), self.alloc.peak_used, resident)


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


def emit_layer_group(cs: list, shape: LayerShape, xs: list, ws: list, cos: list, sin: list, name: str, keep: bool = False):
    """Plan one decoder layer on a group of chips (each holds its shard, x is replicated): xs [seq, hidden] → outs.
    With keep, the intermediates the backward pass needs stay in SRAM and come back as a dict per chip."""
    tp = len(cs)
    S, D = shape.seq, shape.head_dim
    partial, h1s, saved = [], [], [dict() for _ in cs]
    for j, c in enumerate(cs):
        sh, x, w = Shard(shape, tp, j), xs[j], ws[j]
        xn = c.vec("rmsnorm", [x, w["input_layernorm.weight"]], f"{name}.xn", eps=shape.eps)
        q = c.matmul(xn, w["self_attn.q_proj.weight"], f"{name}.q", w.get("self_attn.q_proj.bias"))
        k = c.matmul(xn, w["self_attn.k_proj.weight"], f"{name}.k", w.get("self_attn.k_proj.bias"))
        v = c.matmul(xn, w["self_attn.v_proj.weight"], f"{name}.v", w.get("self_attn.v_proj.bias"))
        qr, kr = c.new(f"{name}.qr", S, sh.hq * D), c.new(f"{name}.kr", S, sh.hkv * D)
        for h in range(sh.hq):
            c.vec("rope", [q.cols_slice(f"{name}.q{h}", h * D, (h + 1) * D), cos[j], sin[j]], f"{name}.qr{h}",
                  into=qr.cols_slice(f"{name}.qr{h}", h * D, (h + 1) * D))
        for g in range(sh.hkv):
            c.vec("rope", [k.cols_slice(f"{name}.k{g}", g * D, (g + 1) * D), cos[j], sin[j]], f"{name}.kr{g}",
                  into=kr.cols_slice(f"{name}.kr{g}", g * D, (g + 1) * D))
        c.free(q, k)
        vt = [c.vec("transpose", [v.cols_slice(f"{name}.v{g}", g * D, (g + 1) * D)], f"{name}.vT{g}", rows=D, cols=S) for g in range(sh.hkv)]
        attn = c.new(f"{name}.attn", S, sh.hq * D)
        probs = []
        for h in range(sh.hq):
            g = sh.kv_local(h)
            scores = c.matmul(qr.cols_slice(f"{name}.qr{h}", h * D, (h + 1) * D), kr.cols_slice(f"{name}.kr{g}", g * D, (g + 1) * D), f"{name}.s{h}")
            p = c.vec("softmax_causal", [scores], f"{name}.p{h}", scale=1.0 / math.sqrt(D))
            c.free(scores)
            c.matmul(p, vt[g], f"{name}.attn{h}", into=attn.cols_slice(f"{name}.attn{h}", h * D, (h + 1) * D))
            probs.append(p)
        c.free(*vt)
        partial.append(c.matmul(attn, w["self_attn.o_proj.weight"], f"{name}.o"))
        if keep:
            saved[j].update(xn=xn, v=v, qr=qr, kr=kr, probs=probs, attn=attn)
        else:
            c.free(xn, v, qr, kr, attn, *probs)
    o = allreduce(cs, partial, f"{name}.o")
    partial = []
    for j, c in enumerate(cs):
        w = ws[j]
        h1 = c.vec("add", [xs[j], o[j]], f"{name}.h1")
        c.free(o[j])
        hn = c.vec("rmsnorm", [h1, w["post_attention_layernorm.weight"]], f"{name}.hn", eps=shape.eps)
        gate = c.matmul(hn, w["mlp.gate_proj.weight"], f"{name}.gate")
        up = c.matmul(hn, w["mlp.up_proj.weight"], f"{name}.up")
        act = c.vec("silu_mul", [gate, up], f"{name}.act")
        partial.append(c.matmul(act, w["mlp.down_proj.weight"], f"{name}.down"))
        if keep:
            saved[j].update(h1=h1, hn=hn, gate=gate, up=up, act=act)
        else:
            c.free(hn, gate, up, act)
        h1s.append(h1)
    down = allreduce(cs, partial, f"{name}.d")
    outs = []
    for j, c in enumerate(cs):
        outs.append(c.vec("add", [h1s[j], down[j]], f"{name}.out"))
        c.free(down[j])
        if not keep:
            c.free(h1s[j])
    return (outs, saved) if keep else outs


def emit_layer_backward(cs: list, shape: LayerShape, xs: list, ws: list, cos: list, sin: list, douts: list, name: str, lr: float) -> list:
    """The backward pass of one layer on its group: recompute the forward from the saved inputs xs, then propagate douts
    back to dxs, and update every weight shard in place (SGD). Gradients of replicated inputs are all-reduced over the group,
    like the forward's partial sums; shared kv heads are reduced over the chips that share them."""
    tp = len(cs)
    S, D, H = shape.seq, shape.head_dim, shape.hidden
    scale = 1.0 / math.sqrt(D)
    _, saved = emit_layer_group(cs, shape, xs, ws, cos, sin, f"{name}.r", keep=True)
    # ----- MLP -----
    d_hn, d_h1 = [], []
    for j, c in enumerate(cs):
        w, sv, dout = ws[j], saved[j], douts[j]
        d_act = c.matmul(dout, w["mlp.down_proj.weight"], f"{name}.d_act", tb=True)
        dW = c.matmul(dout, sv["act"], f"{name}.dWd", ta=True, tb=True)
        c.vec("sgd", [w["mlp.down_proj.weight"], dW], f"{name}.upd", into=w["mlp.down_proj.weight"], lr=lr)
        c.free(dW, sv["act"])
        d_gate = c.vec("silu_bwd_gate", [d_act, sv["gate"], sv["up"]], f"{name}.d_gate")
        d_up = c.vec("silu_bwd_up", [d_act, sv["gate"]], f"{name}.d_up")
        c.free(d_act, sv["gate"], sv["up"])
        dh = c.matmul(d_gate, w["mlp.gate_proj.weight"], f"{name}.d_hn", tb=True)
        c.matmul(d_up, w["mlp.up_proj.weight"], f"{name}.d_hn", tb=True, into=dh, acc=True)
        for d_, key in ((d_gate, "mlp.gate_proj.weight"), (d_up, "mlp.up_proj.weight")):
            dW = c.matmul(d_, sv["hn"], f"{name}.dW", ta=True, tb=True)
            c.vec("sgd", [w[key], dW], f"{name}.upd", into=w[key], lr=lr)
            c.free(dW)
        c.free(d_gate, d_up, sv["hn"])
        d_hn.append(dh)
    d_hn = allreduce(cs, d_hn, f"{name}.b_hn")
    for j, c in enumerate(cs):
        w, sv = ws[j], saved[j]
        dx = c.vec("rmsnorm_bwd_x", [d_hn[j], sv["h1"], w["post_attention_layernorm.weight"]], f"{name}.d_h1n", eps=shape.eps)
        dw = c.vec("rmsnorm_bwd_w", [d_hn[j], sv["h1"]], f"{name}.d_ln2", rows=1, cols=H, eps=shape.eps)
        c.vec("sgd", [w["post_attention_layernorm.weight"], dw], f"{name}.upd", into=w["post_attention_layernorm.weight"], lr=lr)
        c.free(dw, d_hn[j], sv["h1"])
        d_h1.append(c.vec("add", [douts[j], dx], f"{name}.d_h1"))
        c.free(dx)
    # ----- attention -----
    d_xn, d_ks, d_vs = [], [], []
    for j, c in enumerate(cs):
        sh, w, sv = Shard(shape, tp, j), ws[j], saved[j]
        d_attn = c.matmul(d_h1[j], w["self_attn.o_proj.weight"], f"{name}.d_attn", tb=True)
        dW = c.matmul(d_h1[j], sv["attn"], f"{name}.dWo", ta=True, tb=True)
        c.vec("sgd", [w["self_attn.o_proj.weight"], dW], f"{name}.upd", into=w["self_attn.o_proj.weight"], lr=lr)
        c.free(dW, sv["attn"])
        d_q, d_k, d_v = c.new(f"{name}.d_q", S, sh.hq * D), c.new(f"{name}.d_k", S, sh.hkv * D), c.new(f"{name}.d_v", S, sh.hkv * D)
        seen: set = set()
        for h in range(sh.hq):
            g = sh.kv_local(h)
            da = d_attn.cols_slice(f"{name}.d_attn{h}", h * D, (h + 1) * D)
            vg = sv["v"].cols_slice(f"{name}.v{g}", g * D, (g + 1) * D)
            d_p = c.matmul(da, vg, f"{name}.d_p{h}")
            c.matmul(sv["probs"][h], da, f"{name}.d_v{g}", ta=True, tb=True, into=d_v.cols_slice(f"{name}.d_v{g}", g * D, (g + 1) * D), acc=g in seen)
            d_s = c.vec("softmax_bwd", [d_p, sv["probs"][h]], f"{name}.d_s{h}", scale=scale)
            c.free(d_p, sv["probs"][h])
            qh = sv["qr"].cols_slice(f"{name}.qr{h}", h * D, (h + 1) * D)
            kg = sv["kr"].cols_slice(f"{name}.kr{g}", g * D, (g + 1) * D)
            d_qr = c.matmul(d_s, kg, f"{name}.d_qr{h}", tb=True)
            c.matmul(d_s, qh, f"{name}.d_kr{g}", ta=True, tb=True, into=d_k.cols_slice(f"{name}.d_kr{g}", g * D, (g + 1) * D), acc=g in seen)
            seen.add(g)
            c.free(d_s)
            c.vec("rope_bwd", [d_qr, cos[j], sin[j]], f"{name}.d_q{h}", into=d_q.cols_slice(f"{name}.d_q{h}", h * D, (h + 1) * D))
            c.free(d_qr)
        c.free(d_attn, sv["v"], sv["qr"])
        for g in range(sh.hkv):
            kg = d_k.cols_slice(f"{name}.d_kr{g}", g * D, (g + 1) * D)
            c.vec("rope_bwd", [kg, cos[j], sin[j]], f"{name}.d_k{g}", into=kg)
        c.free(sv["kr"])
        d_ks.append(d_k)
        d_vs.append(d_v)
        saved[j]["d_q"] = d_q
    if shape.kv_heads < tp:  # a kv head shared by several chips: its gradient is the sum over them
        share = tp // shape.kv_heads
        for g0 in range(0, tp, share):
            allreduce(cs[g0:g0 + share], d_ks[g0:g0 + share], f"{name}.b_k{g0}")
            allreduce(cs[g0:g0 + share], d_vs[g0:g0 + share], f"{name}.b_v{g0}")
    for j, c in enumerate(cs):
        w, sv = ws[j], saved[j]
        d_q, d_k, d_v = sv["d_q"], d_ks[j], d_vs[j]
        dxn = c.matmul(d_q, w["self_attn.q_proj.weight"], f"{name}.d_xn", tb=True)
        if shape.kv_heads >= tp or j % (tp // shape.kv_heads) == 0:  # a shared kv head counts once in the group's sum
            c.matmul(d_k, w["self_attn.k_proj.weight"], f"{name}.d_xn", tb=True, into=dxn, acc=True)
            c.matmul(d_v, w["self_attn.v_proj.weight"], f"{name}.d_xn", tb=True, into=dxn, acc=True)
        for d_, key in ((d_q, "self_attn.q_proj"), (d_k, "self_attn.k_proj"), (d_v, "self_attn.v_proj")):
            dW = c.matmul(d_, sv["xn"], f"{name}.dW", ta=True, tb=True)
            c.vec("sgd", [w[key + ".weight"], dW], f"{name}.upd", into=w[key + ".weight"], lr=lr)
            c.free(dW)
            if key + ".bias" in w:
                db = c.vec("colsum", [d_], f"{name}.db", rows=1, cols=d_.cols)
                c.vec("sgd", [w[key + ".bias"], db], f"{name}.upd", into=w[key + ".bias"], lr=lr)
                c.free(db)
        c.free(d_q, d_k, d_v, sv["xn"])
        d_xn.append(dxn)
    d_xn = allreduce(cs, d_xn, f"{name}.b_xn")
    dxs = []
    for j, c in enumerate(cs):
        w = ws[j]
        dx = c.vec("rmsnorm_bwd_x", [d_xn[j], xs[j], w["input_layernorm.weight"]], f"{name}.d_xln", eps=shape.eps)
        dw = c.vec("rmsnorm_bwd_w", [d_xn[j], xs[j]], f"{name}.d_ln1", rows=1, cols=H, eps=shape.eps)
        c.vec("sgd", [w["input_layernorm.weight"], dw], f"{name}.upd", into=w["input_layernorm.weight"], lr=lr)
        c.free(dw, d_xn[j])
        dxs.append(c.vec("add", [d_h1[j], dx], f"{name}.dx"))
        c.free(dx, d_h1[j])
    return dxs


def allgather(cs: list, shards: list, bounds: list, name: str) -> list:
    """Every chip ends with the full [rows, bounds[-1]] tensor of which it held one column block; ring, exact cycles."""
    tp = len(cs)
    S = shards[0].rows
    full = [c.new(f"{name}.full", S, bounds[-1]) for c in cs]
    chunk = lambda j, ci: full[j].cols_slice(f"{name}.g{ci}", bounds[ci], bounds[ci + 1])
    for j, c in enumerate(cs):
        c.vec("copy", [shards[j]], f"{name}.own", into=chunk(j, j))
    for s in range(tp - 1):
        arrive = [cs[j].send(chunk(j, (j - s) % tp), f"{name}.ag{s}", cs[(j + 1) % tp].chip) for j in range(tp)]
        for j in range(tp):
            ci = (j - 1 - s) % tp
            cs[j].recv(f"{name}.ag{s}", S, bounds[ci + 1] - bounds[ci], arrive[(j - 1) % tp], into=chunk(j, ci))
    return full


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
    kind: str = "run"            # "run": one token per body; "train": one SGD step per body

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


def chip_fits_train(spec: ChipSpec, shape: LayerShape, layers_on_chip: int, vocab: int, with_head: bool, tp: int) -> bool:
    """Training keeps every layer's input, recomputes one layer at a time and holds one shard of gradients."""
    sw = shard_words(shape, tp)
    vj = _cdiv(vocab, tp)
    words = (layers_on_chip * sw + sw + layers_on_chip * shape.seq * shape.hidden + 3 * shape.activation_words(tp)
             + (3 * vj * shape.hidden + 2 * shape.seq * vocab + shape.seq * vj if with_head else 0))
    return words <= spec.words * 0.95


def compile_train(spec: ChipSpec, shape: LayerShape, n_layers: int, vocab: int, chips: int = 1, resident: bool | None = None,
                  tp: int | None = None, lr: float = 1e-3) -> ModelPlan:
    """One SGD step of the whole model, scheduled exactly: forward through the stages keeping each layer's input, cross-entropy
    at the head, backward from the last stage down with one layer recomputed at a time, every weight shard updated in place.
    The embedding table lives on the host: the chips send back dx0 (its gradient rows) and the head's gradient."""
    tp, layers_of = layout(spec, shape, n_layers, vocab, chips, tp)
    stages = len(layers_of)
    if resident is None:
        resident = all(chip_fits_train(spec, shape, len(ls), vocab, s == stages - 1, tp) for s, ls in enumerate(layers_of))
    S, H = shape.seq, shape.hidden
    vb = [vocab * j // tp for j in range(tp + 1)]
    groups: list = []
    for s, ls in enumerate(layers_of):
        cs = [Compiler(spec, chip=s * tp + j) for j in range(tp)]
        for c in cs:
            c.phase = "prologue" if resident else "body"
        cos = [c.load("cos", S, shape.head_dim) for c in cs]
        sin = [c.load("sin", S, shape.head_dim) for c in cs]
        weights: dict = {}
        norm_w = None
        if resident:
            for i in ls:
                weights[i] = [load_layer_weights(c, shape, Shard(shape, tp, j), f"layer{i}.") for j, c in enumerate(cs)]
            if s == stages - 1:
                norm_w = [c.load("norm", 1, H) for c in cs]
        for c in cs:
            c.phase = "body"
        groups.append({"cs": cs, "cos": cos, "sin": sin, "weights": weights, "norm": norm_w, "t0": [c.now() for c in cs], "xs": {}})
    # ----- forward -----
    arrive = [0] * tp
    for s, ls in enumerate(layers_of):
        g = groups[s]
        cs, t0 = g["cs"], g["t0"]
        xs = [c.load("x", S, H, at=t0[j]) if s == 0 else c.recv("x", S, H, arrive[j]) for j, c in enumerate(cs)]
        for i in ls:
            g["xs"][i] = xs
            ws = g["weights"].get(i) or [load_layer_weights(c, shape, Shard(shape, tp, j), f"layer{i}.", at=t0[j]) for j, c in enumerate(cs)]
            if not resident:
                g["weights"][i] = ws
            xs = emit_layer_group(cs, shape, xs, ws, g["cos"], g["sin"], f"layer{i}")
            if not resident:
                for j, c in enumerate(cs):
                    c.free(*ws[j].values())
        if s < stages - 1:
            arrive = [c.send(xs[j], "x", (s + 1) * tp + j) for j, c in enumerate(cs)]
        g["top"] = xs
    # ----- head, loss, and its backward (last stage) -----
    g = groups[-1]
    cs, t0 = g["cs"], g["t0"]
    norm_w = g["norm"] or [c.load("norm", 1, H, at=t0[j]) for j, c in enumerate(cs)]
    head_w = [c.load("head", vb[j + 1] - vb[j], H, at=t0[j]) for j, c in enumerate(cs)]
    onehot = [c.load("onehot", S, vocab, at=t0[j]) for j, c in enumerate(cs)]
    xn = [c.vec("rmsnorm", [g["top"][j], norm_w[j]], "final.norm", eps=shape.eps) for j, c in enumerate(cs)]
    shards = [c.matmul(xn[j], head_w[j], "logits") for j, c in enumerate(cs)]
    logits = allgather(cs, shards, vb, "logits") if tp > 1 else shards
    if tp > 1:
        for j, c in enumerate(cs):
            c.free(shards[j])
    loss = cs[0].vec("ce_loss", [logits[0], onehot[0]], "loss", rows=1, cols=1)
    cs[0].store(loss, "loss")
    cs[0].free(loss)
    d_xn = []
    for j, c in enumerate(cs):
        d_logits = c.vec("ce_grad", [logits[j], onehot[j]], "d_logits")
        c.free(logits[j], onehot[j])
        dl = d_logits.cols_slice(f"d_logits{j}", vb[j], vb[j + 1])
        d_head = c.matmul(dl, xn[j], "d_head", ta=True, tb=True)
        c.store(d_head, "dhead")
        c.free(d_head)
        d_xn.append(c.matmul(dl, head_w[j], "d_xn", tb=True))
        c.free(d_logits, head_w[j])
    d_xn = allreduce(cs, d_xn, "b_final")
    douts = []
    for j, c in enumerate(cs):
        dx = c.vec("rmsnorm_bwd_x", [d_xn[j], g["top"][j], norm_w[j]], "d_top", eps=shape.eps)
        dw = c.vec("rmsnorm_bwd_w", [d_xn[j], g["top"][j]], "d_norm", rows=1, cols=H, eps=shape.eps)
        c.vec("sgd", [norm_w[j], dw], "upd", into=norm_w[j], lr=lr)
        c.free(dw, d_xn[j], xn[j])
        if not resident:
            c.store(norm_w[j], "norm")
            c.free(norm_w[j])
        douts.append(dx)
    # ----- backward through the stages -----
    for s in range(stages - 1, -1, -1):
        g = groups[s]
        cs, t0 = g["cs"], g["t0"]
        if s < stages - 1:
            douts = [c.recv("dx", S, H, arrive[j]) for j, c in enumerate(cs)]
        for i in reversed(layers_of[s]):
            xs = g["xs"][i]
            ws = g["weights"][i] if resident else [load_layer_weights(c, shape, Shard(shape, tp, j), f"layer{i}.", at=t0[j]) for j, c in enumerate(cs)]
            dxs = emit_layer_backward(cs, shape, xs, ws, g["cos"], g["sin"], douts, f"layer{i}", lr)
            for j, c in enumerate(cs):
                c.free(douts[j], *([] if i == layers_of[s][0] and s == 0 else [xs[j]]))
                if not resident:
                    for key, b in ws[j].items():
                        c.store(b, f"layer{i}.{key}")
                        c.free(b)
            douts = dxs
        if s > 0:
            arrive = [c.send(douts[j], "dx", (s - 1) * tp + j) for j, c in enumerate(cs)]
        else:
            cs[0].store(douts[0], "dx0")
            for j, c in enumerate(cs):
                c.free(douts[j], g["xs"][layers_of[0][0]][j])
    # ----- epilogue: resident weights come back to the host when training ends -----
    if resident:
        for s, g in enumerate(groups):
            for c in g["cs"]:
                c.phase = "epilogue"
            for i, ws in g["weights"].items():
                for j, c in enumerate(g["cs"]):
                    for key, b in ws[j].items():
                        c.store(b, f"layer{i}.{key}")
            if g["norm"]:
                for j, c in enumerate(g["cs"]):
                    c.store(g["norm"][j], "norm")
    programs = [c.program(resident) for g in groups for c in g["cs"]]
    body = [i for p in programs for i in p.instrs if i.phase == "body"]
    wb = spec.word_bytes
    cycles = max(i.end for i in body) - min(i.start for i in body)
    host = sum(i.writes[0].words * wb for i in body if i.op == "load") + sum(i.reads[0].words * wb for i in body if i.op == "store")
    link = sum(i.reads[0].words * wb for i in body if i.op == "send")
    return ModelPlan(spec, shape, n_layers, vocab, tp, stages, layers_of, resident, programs, cycles,
                     max(p.sram_peak_words for p in programs) * wb, (n_layers * shape.weight_words + vocab * H + H) * wb, host, link, kind="train")


def unshard_into(full: dict, shard: dict, shape: LayerShape, sh: Shard) -> None:
    """Write one chip's updated shard back into the full numpy weights (inverse of shard_weights)."""
    d = shape.head_dim
    q, kv, mi = slice(sh.q0 * d, (sh.q0 + sh.hq) * d), slice(sh.g0 * d, (sh.g0 + sh.hkv) * d), slice(sh.i0, sh.i1)
    for key, v in shard.items():
        if key.endswith("layernorm.weight"):
            full[key] = v.reshape(full[key].shape)
        elif key in ("self_attn.q_proj.weight",):
            full[key][q] = v
        elif key in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):
            full[key][kv] = v
        elif key == "self_attn.o_proj.weight":
            full[key][:, q] = v
        elif key in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            full[key][mi] = v
        elif key == "mlp.down_proj.weight":
            full[key][:, mi] = v
        elif key == "self_attn.q_proj.bias":
            full[key].reshape(-1)[q] = v.reshape(-1)
        elif key in ("self_attn.k_proj.bias", "self_attn.v_proj.bias"):
            full[key].reshape(-1)[kv] = v.reshape(-1)


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


def write_safetensors(path: str, tensors: dict, dtype: str = "BF16") -> int:
    """Write numpy tensors as one safetensors file in the model's stored dtype (BF16 rounds to nearest even). Returns the size."""
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        if dtype == "BF16":
            raw = (round_bf16(a).view(np.uint32) >> 16).astype(np.uint16).tobytes()
        elif dtype == "F16":
            raw = a.astype(np.float16).tobytes()
        else:
            dtype, raw = "F32", a.tobytes()
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    header["__metadata__"] = {"format": "pt", "mnx": "layer"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        for raw in blobs:
            f.write(raw)
    return 8 + len(hb) + offset


def strip_layer_prefix(tensors: dict, i: int) -> dict:
    key = f".layers.{i}."
    return {k.split(key, 1)[1]: v for k, v in tensors.items() if key in k}


class Model:
    """A model on the virtual LPU(s). `load_layer(i)` and `shared()` return numpy weights (from files, or a phone's streaming Store).
    With train=True the compiled body is one SGD step instead of one token; the host keeps the embedding table and the
    authoritative copy of every weight (updated shards come back from the chips each step, or at the end when resident)."""

    def __init__(self, cfg: dict, load_layer: Callable, shared: Callable, seq: int, spec: ChipSpec | None = None,
                 chips: int = 1, resident: bool | None = None, tp: int | None = None, train: bool = False, lr: float = 1e-3):
        self.spec = spec or ChipSpec()
        self.cfg = cfg
        self.n_layers, self.vocab = int(cfg["num_hidden_layers"]), int(cfg["vocab_size"])
        self.shape = LayerShape.from_config(cfg, seq)
        self.train, self.lr = train, lr
        self.plan = (compile_train(self.spec, self.shape, self.n_layers, self.vocab, chips, resident, tp, lr) if train
                     else compile_model(self.spec, self.shape, self.n_layers, self.vocab, chips, resident, tp))
        self.net = Network(self.spec, self.plan.chips)
        self.load_layer, self.shared_fn = load_layer, shared
        self._shared = None
        self.weights: dict = {}
        self.cos, self.sin = rope_tables(self.shape)
        self.prologue_done = False
        self.stats = {"tokens": 0, "steps": 0, "ticks": 0, "sim_seconds": 0.0, "host_bytes": 0, "link_bytes": 0, "weight_bytes_read": 0, "losses": []}

    def shared(self) -> dict:
        if self._shared is None:
            self._shared = {k: np.array(v, dtype=np.float32) for k, v in self.shared_fn().items()}
        return self._shared

    def _key(self, suffix: str) -> str:
        for k in self.shared():
            if k.endswith(suffix) and ".layers." not in k:
                return k
        raise KeyError(suffix)

    def _find(self, suffix: str) -> np.ndarray:
        return self.shared()[self._key(suffix)]

    @property
    def tied(self) -> bool:
        return not any(k.endswith("lm_head.weight") for k in self.shared())

    def head_matrix(self) -> np.ndarray:
        return self._find("embed_tokens.weight") if self.tied else self._find("lm_head.weight")

    def embed(self, ids: list) -> np.ndarray:
        return self._find("embed_tokens.weight")[np.asarray(ids)]

    def layer_weights(self, i: int) -> dict:
        """The host's copy of layer i (read once; updated by training)."""
        if i not in self.weights:
            w = self.load_layer(i)
            self.stats["weight_bytes_read"] += sum(v.size * self.spec.word_bytes for v in w.values())
            self.weights[i] = {k: np.array(v, dtype=np.float32) for k, v in w.items()}
        return self.weights[i]

    def _bindings(self, x: np.ndarray | None, with_weights: bool, onehot: np.ndarray | None = None) -> list:
        P = self.plan
        out = [{"cos": self.cos, "sin": self.sin} for _ in range(P.chips)]
        if x is not None:
            for j in range(P.tp):
                out[j]["x"] = x
        if with_weights:
            for s, ls in enumerate(P.layers_of):
                for i in ls:
                    w = self.layer_weights(i) if self.train else self.load_layer(i)
                    for j in range(P.tp):
                        for name, arr in shard_weights(w, self.shape, Shard(self.shape, P.tp, j)).items():
                            out[s * P.tp + j][f"layer{i}.{name}"] = arr
        last = (P.stages - 1) * P.tp
        if with_weights or self.train:
            head, norm = self.head_matrix(), self._find("norm.weight").reshape(1, -1)
            for j in range(P.tp):
                b = out[last + j]
                b["head"] = head[self.vocab * j // P.tp:self.vocab * (j + 1) // P.tp]
                if with_weights:
                    b["norm"] = norm
                if onehot is not None:
                    b["onehot"] = onehot
        return out

    # ----- training -----
    def train_step(self, ids: list, targets: list) -> float:
        """One SGD step on the chips for a full window (len(ids) == len(targets) == seq). Returns the loss."""
        S, P = self.shape.seq, self.plan
        if not self.train or len(ids) != S or len(targets) != S:
            raise ValueError(f"train_step wants a Model(train=True) and windows of exactly {S} tokens")
        if P.resident and not self.prologue_done:
            Network.run(self.net.chips, [p.phase("prologue") for p in P.programs], self._bindings(None, True))
            self.prologue_done = True
        x = self.embed(ids).astype(np.float32)
        onehot = np.zeros((S, self.vocab), dtype=np.float32)
        onehot[np.arange(S), np.asarray(targets)] = 1.0
        t0 = time.time()
        host0 = sum(c.host_bytes for c in self.net.chips)
        link0 = sum(c.link_bytes for c in self.net.chips)
        res = Network.run(self.net.chips, [p.phase("body") for p in P.programs], self._bindings(x, not P.resident, onehot))
        boxes = res["outboxes"]
        last = (P.stages - 1) * P.tp
        loss = float(boxes[last]["loss"][0, 0])
        # the host owns the embedding table (and the head when it is separate): apply their gradients here
        E = self._find("embed_tokens.weight")
        np.add.at(E, np.asarray(ids), -self.lr * boxes[0]["dx0"])
        d_head = np.concatenate([boxes[last + j]["dhead"] for j in range(P.tp)], axis=0)
        self.head_matrix()[...] -= self.lr * d_head
        if not P.resident:
            self._absorb(boxes)
        self.stats["steps"] += 1
        self.stats["losses"].append(loss)
        self.stats["ticks"] += res["ticks"]
        self.stats["sim_seconds"] += time.time() - t0
        self.stats["host_bytes"] += sum(c.host_bytes for c in self.net.chips) - host0
        self.stats["link_bytes"] += sum(c.link_bytes for c in self.net.chips) - link0
        return loss

    def _absorb(self, boxes: list) -> None:
        """Updated shards and norm that came back over the host link go into the host's copies."""
        P = self.plan
        for s, ls in enumerate(P.layers_of):
            for i in ls:
                full = self.layer_weights(i)
                for j in range(P.tp):
                    box = boxes[s * P.tp + j]
                    shard = {k[len(f"layer{i}."):]: v for k, v in box.items() if k.startswith(f"layer{i}.")}
                    if shard:
                        unshard_into(full, shard, self.shape, Shard(self.shape, P.tp, j))
        last = (P.stages - 1) * P.tp
        if "norm" in boxes[last]:
            self.shared()[self._key("norm.weight")][...] = boxes[last]["norm"].reshape(-1)

    def pull_weights(self) -> None:
        """After resident training: bring every chip's weights back into the host's copies (the compiled epilogue)."""
        P = self.plan
        if P.resident and self.prologue_done:
            res = Network.run(self.net.chips, [p.phase("epilogue") for p in P.programs], [{} for _ in range(P.chips)])
            self._absorb(res["outboxes"])

    def step_stats(self) -> dict:
        s, n, P = self.stats, max(1, self.stats["steps"]), self.plan
        busy: dict[str, int] = {}
        for p in P.programs:
            for u, cyc in p.unit_busy("body").items():
                busy[u] = busy.get(u, 0) + cyc
        util = {u: round(100 * cyc / (P.cycles_per_token * P.chips), 1) for u, cyc in sorted(busy.items()) if not u.startswith("port.")}
        return {"steps": s["steps"], "cycles_per_step": P.cycles_per_token, "instructions_per_step": P.instructions_per_token,
                "chip_seconds_per_step": P.cycles_per_token / self.spec.clock_hz, "tokens_per_second": self.shape.seq * self.spec.clock_hz / P.cycles_per_token,
                "sim_seconds_per_step": s["sim_seconds"] / n, "host_mb_per_step": s["host_bytes"] / n / 1e6, "link_mb_per_step": s["link_bytes"] / n / 1e6,
                "chips": P.chips, "tp": P.tp, "stages": P.stages, "resident": P.resident, "utilisation": util, "losses": list(s["losses"])}

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
         disk_bytes_per_s: float = 1.5e9, host_flops: float = 3e10, instr_seconds: float = 3e-5, train: bool = False, lr: float = 1e-3) -> dict:
    """What running (or, with train, one SGD step of) this model on `chips` virtual LPUs costs: exact compiled cycles when the
    schedule is small enough to build, else one exact layer multiplied out; and what simulating it here costs."""
    spec = spec or ChipSpec()
    shape = LayerShape.from_config(cfg, seq)
    n_layers, vocab = int(cfg["num_hidden_layers"]), int(cfg["vocab_size"])
    fits_fn = chip_fits_train if train else chip_fits
    tp, layers_of = layout(spec, shape, n_layers, vocab, chips)
    if train:  # training needs more SRAM per chip: pick the group size with that in mind
        valid = shape.valid_tp(max(1, int(chips)))
        fitting = [t for t in valid if fits_fn(spec, shape, 1, vocab, True, t)]
        tp = fitting[0] if fitting else valid[-1]
        for t in fitting:
            lo = _split_layers(n_layers, max(1, int(chips)) // t)
            if all(fits_fn(spec, shape, len(ls), vocab, s == len(lo) - 1, t) for s, ls in enumerate(lo)):
                tp = t
                break
        layers_of = _split_layers(n_layers, max(1, int(chips)) // tp)
    stages = len(layers_of)
    wb = spec.word_bytes
    layer_bytes, head_bytes = shape.weight_words * wb, (vocab * shape.hidden + shape.hidden) * wb
    weight_bytes = n_layers * layer_bytes + head_bytes
    fits = fits_fn(spec, shape, 1, vocab, True, tp)
    min_tp = next((t for t in shape.valid_tp(10 ** 9) if fits_fn(spec, shape, 1, vocab, True, t)), None)
    resident = fits and all(fits_fn(spec, shape, len(ls), vocab, s == stages - 1, tp) for s, ls in enumerate(layers_of))
    chips_resident = chips_for_resident(spec, shape, n_layers, vocab)
    unit = "step" if train else "token"
    out = {"chip": spec.name, "kind": "train" if train else "run", "chips": tp * stages, "tp": tp, "stages": stages, "layers_per_chip": len(layers_of[0]),
           "resident": resident, "fits": fits, "min_chips": min_tp, "chips_for_resident": chips_resident, "sram_per_chip": _fmt_bytes(spec.sram_bytes),
           "word_bits": wb * 8, "weights": _fmt_bytes(weight_bytes), "weight_bytes": weight_bytes, "layer": _fmt_bytes(layer_bytes),
           "shard": _fmt_bytes(shard_words(shape, tp) * wb), "seq": seq, "clock_mhz": spec.clock_hz / 1e6, "max_tp": shape.valid_tp(10 ** 9)[-1], "lr": lr,
           "source": "SRAM, loaded once" if resident else f"the host link ({spec.host_bytes_per_cycle * spec.clock_hz / 1e9:.0f} GB/s per chip), every {unit}"}
    nope = dict(cycles_per_token=None, exact=False, per_token="never: it doesn't fit", sim_per_token="n/a", instructions_per_token=0,
                chip_seconds_per_token=None, chip_tokens_per_second=0, sim_seconds_per_token=None)
    if not fits:
        out.update(nope)
        return out
    per_chip_instrs = ((shape.heads // tp) * 8 + 40 + 6 * (tp - 1)) * (6 if train else 1)
    exact = n_layers * tp * per_chip_instrs <= 60000
    compile_fn = (lambda res: compile_train(spec, shape, n_layers, vocab, chips, res, tp, lr)) if train else (lambda res: compile_model(spec, shape, n_layers, vocab, chips, res, tp))
    if exact:
        try:
            mp = compile_fn(resident)
        except MemoryError:
            resident = False
            try:
                mp = compile_fn(False)
            except MemoryError:
                out.update(nope, fits=False, resident=False, exact=True)
                return out
        out["resident"] = resident
        cycles, n_instr = mp.cycles_per_token, mp.instructions_per_token
    else:  # one exact layer on its group, multiplied out (a training step is about three forward passes plus the updates)
        progs = compile_layer(spec, shape, tp)
        body = [i for p in progs for i in p.instrs if i.op not in ("load", "store")]
        layer_cycles = (max(i.end for i in body) - min(i.start for i in body)) * (3 if train else 1)
        stream = 0 if resident else _cdiv(shard_words(shape, tp) * wb, spec.host_bytes_per_cycle) * (3 if train else 1)
        hop = spec.link_latency + _cdiv(seq * shape.hidden * wb, spec.link_bytes_per_cycle)
        c = Compiler(spec)
        cycles = n_layers * max(layer_cycles, stream) + (stages - 1) * hop * (2 if train else 1) + c.c_matmul(seq, shape.hidden, _cdiv(vocab, tp)) * (3 if train else 1)
        n_instr = n_layers * tp * per_chip_instrs
    chip_sec = cycles / spec.clock_hz
    sim_sec = (6 if train else 2) * params * seq / host_flops + (0 if resident else weight_bytes / disk_bytes_per_s * (3 if train else 1)) + n_instr * instr_seconds
    out.update(cycles_per_token=cycles, exact=exact, instructions_per_token=n_instr, chip_seconds_per_token=chip_sec,
               chip_tokens_per_second=(seq if train else 1) / chip_sec, per_token=_fmt_time(chip_sec), sim_seconds_per_token=sim_sec,
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
    train = p.get("kind") == "train"
    unit = "step" if train else "token"
    what = f"training step of {p['seq']} tokens" if train else "token"
    s = (f"On {chips}, {name} {'keeps its ' + p['weights'] + ' of weights in SRAM' if p['resident'] else 'streams its ' + p['weights'] + ' of weights over the host links every ' + unit}: "
         f"{p['cycles_per_token']:,} cycles per {what}, {p['per_token']} at {p['clock_mhz']:.0f} MHz, so {p['chip_tokens_per_second']:,.0f} tokens/s on the chip"
         f"{'' if p['exact'] else ' (one exact layer, multiplied out)'}. ")
    if not p["resident"]:
        s += (f"It would take {p['chips_for_resident']:,} chips to keep the weights resident. " if p["chips_for_resident"]
              else "No number of chips keeps the weights resident. ")
    s += f"Simulating one {unit} here takes {p['sim_per_token']}: the chip is software, every multiply really runs."
    return s
