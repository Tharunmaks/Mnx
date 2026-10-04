"""Write a tiny random-weight Qwen2-architecture GGUF for runtime tests.

The model produces gibberish, but it loads and runs in real llama.cpp, so it
exercises Mnx's model loading, chat templating, streaming and aborts.
Usage: python3 test/fixtures/make-tiny-gguf.py out.gguf   (needs: pip install gguf numpy)
"""
import sys

import numpy as np
import gguf


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for _, c in sorted(zip(bs, cs))]


QWEN_TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def main(path):
    rng = np.random.default_rng(0)
    byte_tokens = bytes_to_unicode()
    merges = ["Ġ t", "Ġ a", "h e", "i n", "Ġt he"]
    merged = ["Ġt", "Ġa", "he", "in", "Ġthe"]
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tokens = byte_tokens + merged + specials
    types = [gguf.TokenType.NORMAL] * (len(byte_tokens) + len(merged)) + [gguf.TokenType.CONTROL] * len(specials)
    n_vocab = len(tokens)
    n_embd, n_ff, n_head, n_head_kv, n_layer = 64, 128, 4, 2, 2
    head_dim = n_embd // n_head

    w = gguf.GGUFWriter(path, "qwen2")
    w.add_name("mnx-tiny-test")
    w.add_context_length(32768)
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_head_kv)
    w.add_rope_freq_base(1000000.0)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_bos_token_id(tokens.index("<|endoftext|>"))
    w.add_eos_token_id(tokens.index("<|im_end|>"))
    w.add_pad_token_id(tokens.index("<|endoftext|>"))
    w.add_add_bos_token(False)
    w.add_chat_template(QWEN_TEMPLATE)

    def t(name, *shape, scale=0.02):
        w.add_tensor(name, (rng.standard_normal(shape) * scale).astype(np.float32))

    t("token_embd.weight", n_vocab, n_embd)
    w.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
    t("output.weight", n_vocab, n_embd)
    for i in range(n_layer):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", np.ones(n_embd, dtype=np.float32))
        w.add_tensor(p + "ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
        t(p + "attn_q.weight", n_embd, n_embd)
        t(p + "attn_q.bias", n_embd)
        t(p + "attn_k.weight", n_head_kv * head_dim, n_embd)
        t(p + "attn_k.bias", n_head_kv * head_dim)
        t(p + "attn_v.weight", n_head_kv * head_dim, n_embd)
        t(p + "attn_v.bias", n_head_kv * head_dim)
        t(p + "attn_output.weight", n_embd, n_embd)
        t(p + "ffn_gate.weight", n_ff, n_embd)
        t(p + "ffn_up.weight", n_ff, n_embd)
        t(p + "ffn_down.weight", n_embd, n_ff)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main(sys.argv[1])
