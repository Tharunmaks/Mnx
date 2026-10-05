"""The virtual LPU against an independent numpy model: forward, training, meshes, schedules, files and plans."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reference as ref  # noqa: E402
from hive import layers, lpu  # noqa: E402

GQA = {"hidden_size": 64, "num_attention_heads": 8, "num_key_value_heads": 2, "intermediate_size": 96, "num_hidden_layers": 3,
       "vocab_size": 50, "model_type": "qwen2", "rms_norm_eps": 1e-6, "tie_word_embeddings": True}
LLAMA = {"hidden_size": 48, "num_attention_heads": 4, "num_key_value_heads": 4, "intermediate_size": 80, "num_hidden_layers": 2,
         "vocab_size": 40, "model_type": "llama", "attention_bias": False, "rms_norm_eps": 1e-5, "tie_word_embeddings": False}
EXACT = lpu.ChipSpec(word_bytes=4)  # 32-bit SRAM words: the chip must match the reference to float32 precision
S = 10


def build(cfg, seed=0, seq=S, spec=EXACT, **kw):
    """A lpu.Model over the reference's random weights (copied, so training never touches the reference's arrays)."""
    layers_, E, norm, head = ref.random_model(cfg, seed)
    shared = {"model.embed_tokens.weight": E.copy(), "model.norm.weight": norm.copy()}
    if head is not None:
        shared["lm_head.weight"] = head.copy()
    copies = [{k: v.copy() for k, v in w.items()} for w in layers_]
    m = lpu.Model(cfg, lambda i: copies[i], lambda: shared, seq=seq, spec=spec, **kw)
    return m, (layers_, E, norm, head)


def tables(cfg, seq=S):
    return ref.rope_tables(seq, cfg["hidden_size"] // cfg["num_attention_heads"])


def rel(a, b):
    return float(np.abs(np.asarray(a, np.float64) - b).max() / (np.abs(b).max() + 1e-12))


# ---------------------------------------------------------------- the reference itself
@pytest.mark.parametrize("cfg", [GQA, LLAMA], ids=["qwen2-gqa", "llama-untied"])
def test_reference_gradient_matches_finite_differences(cfg):
    layers_, E, norm, head = ref.random_model(cfg, 1)
    cos, sin = tables(cfg)
    rng = np.random.default_rng(2)
    ids, targets = rng.integers(0, cfg["vocab_size"], S), rng.integers(0, cfg["vocab_size"], S)
    lr, eps = 0.05, 1e-5
    _, new_layers, _, _, _ = ref.sgd_step(layers_, E, norm, ids, targets, cfg, cos, sin, lr, head)
    head_ = E if head is None else head
    for li, key in [(0, "self_attn.q_proj.weight"), (len(layers_) - 1, "mlp.down_proj.weight"), (0, "input_layernorm.weight"),
                    (0, "self_attn.v_proj.weight")]:
        idx = tuple(rng.integers(0, n) for n in layers_[li][key].shape)
        grad = (layers_[li][key][idx] - new_layers[li][key][idx]) / lr
        plus = [{k: v.copy() for k, v in w.items()} for w in layers_]
        minus = [{k: v.copy() for k, v in w.items()} for w in layers_]
        plus[li][key][idx] += eps
        minus[li][key][idx] -= eps
        fd = (ref.loss(plus, E, norm, head_, ids, targets, cfg, cos, sin) - ref.loss(minus, E, norm, head_, ids, targets, cfg, cos, sin)) / (2 * eps)
        assert abs(grad - fd) <= 1e-5 * max(1.0, abs(fd)), (key, grad, fd)


# ---------------------------------------------------------------- forward
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_one_layer_on_a_group_matches_reference(tp):
    cfg = dict(GQA, num_hidden_layers=1)
    shape = lpu.LayerShape.from_config(cfg, S)
    layers_, *_ = ref.random_model(cfg, 3)
    w = layers_[0]
    x = np.random.default_rng(4).standard_normal((S, cfg["hidden_size"]))
    cos, sin = tables(cfg)
    want = ref.layer_forward(x, w, cfg, cos, sin)["out"]
    progs = lpu.compile_layer(EXACT, shape, tp)
    binds = [{"x": x, "cos": cos, "sin": sin, **lpu.shard_weights(w, shape, lpu.Shard(shape, tp, j))} for j in range(tp)]
    res = lpu.Network.run([lpu.Chip(EXACT, j) for j in range(tp)], progs, binds)
    for j in range(tp):
        assert rel(res["outboxes"][j]["out"], want) < 1e-5


@pytest.mark.parametrize("chips,tp,resident", [(1, None, None), (1, None, False), (2, 2, None), (3, 1, None), (4, 2, False), (8, 4, None)])
def test_model_logits_match_reference_on_any_layout(chips, tp, resident):
    m, (layers_, E, norm, head) = build(GQA, 5, chips=chips, tp=tp, resident=resident)
    ids = list(np.random.default_rng(6).integers(0, GQA["vocab_size"], S))
    cos, sin = tables(GQA)
    want = ref.logits(layers_, E, norm, E if head is None else head, ids, GQA, cos, sin)
    assert rel(m.forward(ids), want) < 1e-5


def test_untied_head_and_no_bias_model():
    m, (layers_, E, norm, head) = build(LLAMA, 7, chips=2, tp=2)
    ids = [1, 5, 9, 3, 0, 2, 8, 7, 6, 4]
    cos, sin = tables(LLAMA)
    assert rel(m.forward(ids), ref.logits(layers_, E, norm, head, ids, LLAMA, cos, sin)) < 1e-5


def test_bf16_words_stay_close_to_reference():
    m, (layers_, E, norm, head) = build(GQA, 8, spec=lpu.ChipSpec(), chips=2, tp=2)
    ids = list(range(S))
    cos, sin = tables(GQA)
    want = ref.logits(layers_, E, norm, E, ids, GQA, cos, sin)
    got = m.forward(ids)
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 2e-2


def test_runs_are_deterministic_and_padding_never_leaks_backwards():
    m, _ = build(GQA, 9, chips=2, tp=2)
    a = m.forward([3, 1, 4, 1, 5])
    b = m.forward([3, 1, 4, 1, 5])
    c = m.forward([3, 1, 4, 1, 5, 9, 2, 6])
    assert np.array_equal(a, b)
    assert rel(c[:5], a[:5]) < 1e-6  # causal: later tokens change nothing before them


def test_greedy_generation_is_identical_on_one_chip_and_a_mesh():
    one, _ = build(GQA, 10, seq=16, chips=1)
    mesh, _ = build(GQA, 10, seq=16, chips=4, tp=2)
    assert one.generate([1, 2, 3], 10, temperature=0) == mesh.generate([1, 2, 3], 10, temperature=0)


def test_cycle_by_cycle_loop_equals_event_skipping_loop():
    shape = lpu.LayerShape.from_config(GQA, S)
    layers_, *_ = ref.random_model(GQA, 11)
    cos, sin = tables(GQA)
    x = np.ones((S, GQA["hidden_size"]))
    prog = lpu.compile_layer(EXACT, shape, 1)
    b = [{"x": x, "cos": cos, "sin": sin, **lpu.shard_weights(layers_[0], shape, lpu.Shard(shape, 1, 0))}]
    fast = lpu.Network.run([lpu.Chip(EXACT)], prog, b)
    slow = lpu.Network.run([lpu.Chip(EXACT)], prog, b, skip_idle=False)
    assert np.array_equal(fast["outboxes"][0]["out"], slow["outboxes"][0]["out"])
    assert slow["ticks"] == prog[0].cycles + 1 and fast["ticks"] < slow["ticks"]


def test_the_chip_refuses_a_schedule_that_reads_too_early():
    shape = lpu.LayerShape.from_config(GQA, S)
    layers_, *_ = ref.random_model(GQA, 12)
    cos, sin = tables(GQA)
    prog = lpu.compile_layer(EXACT, shape, 1)[0]
    mm = next(i for i in prog.instrs if i.op == "matmul")
    mm.end -= mm.start
    mm.start = 0  # before its input was computed
    b = [{"x": np.ones((S, 64)), "cos": cos, "sin": sin, **lpu.shard_weights(layers_[0], shape, lpu.Shard(shape, 1, 0))}]
    with pytest.raises(lpu.ScheduleError):
        lpu.Network.run([lpu.Chip(EXACT)], [prog], b)


def test_compiled_programs_never_double_book_a_unit():
    mp = lpu.compile_train(EXACT, lpu.LayerShape.from_config(GQA, S), 3, 50, chips=4, tp=2)
    for p in mp.programs:
        last: dict = {}
        for i in sorted(p.instrs, key=lambda i: (i.start, i.end)):
            assert i.start >= last.get(i.unit, 0), (p.chip, i.unit, i.note)
            last[i.unit] = i.end


# ---------------------------------------------------------------- training
@pytest.mark.parametrize("cfg", [GQA, LLAMA], ids=["qwen2-gqa-tied", "llama-untied"])
@pytest.mark.parametrize("chips,tp,resident", [(1, None, None), (1, None, False), (2, 2, None), (4, 2, None), (8, 4, False), (3, 1, None)])
def test_training_step_matches_reference(cfg, chips, tp, resident):
    if tp and cfg["num_attention_heads"] % tp:
        pytest.skip("heads don't split that way")
    lr = 0.05
    m, (layers_, E, norm, head) = build(cfg, 13, chips=chips, tp=tp, resident=resident, train=True, lr=lr)
    rng = np.random.default_rng(14)
    ids, targets = list(rng.integers(0, cfg["vocab_size"], S)), list(rng.integers(0, cfg["vocab_size"], S))
    cos, sin = tables(cfg)
    want_loss, want_layers, want_E, want_norm, want_head = ref.sgd_step(layers_, E, norm, ids, targets, cfg, cos, sin, lr, head)
    loss = m.train_step(ids, targets)
    m.pull_weights()
    assert abs(loss - want_loss) < 1e-5
    for i, w in enumerate(want_layers):
        for k, v in w.items():
            assert rel(m.weights[i][k].reshape(v.shape), v) < 1e-5, (i, k)
    assert rel(m.shared()["model.embed_tokens.weight"], want_E) < 1e-5
    assert rel(m.shared()["model.norm.weight"], want_norm) < 1e-5
    if head is not None:
        assert rel(m.shared()["lm_head.weight"], want_head) < 1e-5


def test_two_training_steps_in_a_row_match_reference():
    lr = 0.05
    m, (layers_, E, norm, head) = build(GQA, 15, chips=2, tp=2, train=True, lr=lr)
    cos, sin = tables(GQA)
    rng = np.random.default_rng(16)
    for _ in range(2):
        ids, targets = list(rng.integers(0, 50, S)), list(rng.integers(0, 50, S))
        want, layers_, E, norm, _ = ref.sgd_step(layers_, E, norm, ids, targets, GQA, cos, sin, lr)
        assert abs(m.train_step(ids, targets) - want) < 1e-5
    m.pull_weights()
    assert rel(m.weights[1]["self_attn.k_proj.weight"], layers_[1]["self_attn.k_proj.weight"]) < 1e-5


def test_loss_goes_down_with_16_bit_words():
    m, _ = build(GQA, 17, spec=lpu.ChipSpec(), chips=2, tp=2, train=True, lr=0.05)
    ids, targets = list(range(S)), list(range(1, S + 1))
    losses = [m.train_step(ids, targets) for _ in range(12)]
    assert losses[-1] < 0.5 * losses[0]


def test_train_step_wants_full_windows():
    m, _ = build(GQA, 18, train=True)
    with pytest.raises(ValueError):
        m.train_step([1, 2, 3], [2, 3, 4])


# ---------------------------------------------------------------- files: safetensors and layer split / merge without torch
@pytest.mark.parametrize("dtype", ["BF16", "F16", "F32"])
def test_safetensors_round_trip(tmp_path, dtype):
    a = np.random.default_rng(19).standard_normal((7, 5)).astype(np.float32)
    lpu.write_safetensors(str(tmp_path / "t.safetensors"), {"w": a}, dtype)
    back = lpu.read_safetensors(str(tmp_path / "t.safetensors"))["w"]
    tol = {"BF16": 1e-2, "F16": 1e-3, "F32": 0.0}[dtype]
    assert back.shape == a.shape and np.abs(back - a).max() <= tol * np.abs(a).max() + 1e-12


def test_bf16_rounding_is_round_to_nearest_even():
    x = np.array([1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -2.5, 3.0e38], dtype=np.float32)
    got = lpu.round_bf16(x)
    assert got[0] == 1.0 and got[1] == 1.0 + 2 ** -6 and got[2] == -2.5 and np.isfinite(got[3])


def test_layer_split_and_merge_round_trip(tmp_path):
    layers_, E, norm, _ = ref.random_model(GQA, 20)
    tensors = {"model.embed_tokens.weight": E, "model.norm.weight": norm}
    for i, w in enumerate(layers_):
        tensors.update({f"model.layers.{i}.{k}": v for k, v in w.items()})
    model_dir, layers_dir = tmp_path / "model", tmp_path / "layers"
    model_dir.mkdir()
    lpu.write_safetensors(str(model_dir / "model.safetensors"), tensors, "BF16")
    (model_dir / "config.json").write_text(json.dumps(GQA))
    before = lpu.read_safetensors(str(model_dir / "model.safetensors"))
    manifest = layers.split(model_dir, layers_dir)
    assert manifest["layers"] == GQA["num_hidden_layers"] and manifest["dtype"] == "BF16"
    one = lpu.read_safetensors(str(layers_dir / "layer_0001.safetensors"))
    assert set(one) == {f"model.layers.1.{k}" for k in layers_[1]}
    layers.merge(layers_dir, model_dir)
    after = lpu.read_safetensors(str(model_dir / "model.safetensors"))
    assert set(after) == set(before) and all(np.array_equal(after[k], before[k]) for k in before)


# ---------------------------------------------------------------- plans
BIG = {"hidden_size": 16384, "num_attention_heads": 128, "num_key_value_heads": 8, "intermediate_size": 53248, "num_hidden_layers": 126,
       "vocab_size": 152064, "model_type": "qwen2", "tie_word_embeddings": False}


def test_plan_for_a_tiny_model_is_exact_and_resident():
    p = lpu.plan(GQA, 1e5, seq=16, chips=1)
    assert p["fits"] and p["resident"] and p["exact"] and p["cycles_per_token"] > 0
    assert "keeps its" in lpu.plan_text(p, "tiny")


def test_500b_needs_64_chips_per_layer_and_never_fits_one():
    one = lpu.plan(BIG, 5e11, seq=64, chips=1)
    assert not one["fits"] and one["min_chips"] == 64
    many = lpu.plan(BIG, 5e11, seq=64, chips=128)
    assert many["fits"] and not many["resident"] and many["tp"] == 64 and not many["exact"]


def test_training_plan_counts_a_whole_step():
    run = lpu.plan(GQA, 1e5, seq=16, chips=1, spec=EXACT)
    train = lpu.plan(GQA, 1e5, seq=16, chips=1, spec=EXACT, train=True)
    assert train["kind"] == "train" and train["cycles_per_token"] > 2 * run["cycles_per_token"]
    assert "training step" in lpu.plan_text(train, "tiny")
