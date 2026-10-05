"""Fine-tune your Mnx model: LoRA on Qwen2.5-Instruct, then export GGUF files.

The Colab notebook (mnx_train.ipynb) and a rented GPU both run this script:

    python3 training/train.py --preset smoke     # ~15-25 min check that every step works
    python3 training/train.py --preset full      # the real run (resumes after a disconnect)

Run `npm run train:data` first. The smoke preset trains the same base model
for a few steps, saves and resumes a checkpoint, writes a sample answer,
exports both GGUF files, and prints how long the full run would take on this
GPU. Its files go to <out>/smoke, so they never mix with the full run's.

Uses Unsloth when it's installed (faster, 4-bit), otherwise plain
transformers + peft (also how the tests run it on a CPU).
"""
import argparse
import glob
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

PRESETS = {
    # Same model and batch shape as the full run, so it proves the GPU has
    # enough memory and the timing is a fair estimate for the full run.
    "smoke": dict(use=400, max_steps=12, save_steps=6, logging_steps=2, warmup_steps=2),
    "full": dict(use=115000, max_steps=-1, save_steps=250, logging_steps=25, warmup_steps=50),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, default="smoke")
    p.add_argument("--base", default="unsloth/Qwen2.5-3B-Instruct",
                   help='or "unsloth/Qwen2.5-1.5B-Instruct" (2x faster on a phone) / "unsloth/Qwen2.5-0.5B-Instruct" (5x, weaker)')
    p.add_argument("--out", default="mnx-training", help="checkpoints, LoRA and GGUF files (put this on Google Drive in Colab)")
    p.add_argument("--work", default="mnx-work", help="big temporary files (the merged 16-bit model); local disk is best")
    p.add_argument("--data", default=os.path.join(HERE, "data/train.jsonl"))
    p.add_argument("--test", default=os.path.join(HERE, "data/test.jsonl"))
    p.add_argument("--extra", default="extra.jsonl", help="your learned examples (npm run learn), each seen 3 times")
    p.add_argument("--use", type=int, help="how many conversations to train on (e.g. 25000 for a ~9 h T4 run)")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-len", type=int, default=6144, help="the longest generated conversation is ~4,000 tokens")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--save-steps", type=int)
    p.add_argument("--no-unsloth", action="store_true", help="use transformers + peft even if Unsloth is installed")
    p.add_argument("--no-export", action="store_true", help="skip the GGUF export")
    p.add_argument("--quants", default="q4_k_m,q4_0")
    a = p.parse_args(argv)
    for k, v in PRESETS[a.preset].items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    if a.preset == "smoke":
        a.out = os.path.join(a.out, "smoke")
        a.work = os.path.join(a.work, "smoke")
    return a


def load_model(a):
    import torch
    cuda = torch.cuda.is_available()
    if not a.no_unsloth:
        try:
            from unsloth import FastLanguageModel
        except Exception as e:  # not installed, or no GPU
            print(f"Unsloth unavailable ({str(e)[:120]}); using transformers + peft.")
        else:
            model, tok = FastLanguageModel.from_pretrained(a.base, max_seq_length=a.max_len, load_in_4bit=True)
            model = FastLanguageModel.get_peft_model(
                model, r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0, bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                use_gradient_checkpointing="unsloth", random_state=3407)
            return model, tok, True
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {}
    if cuda:
        kw["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kw["device_map"] = {"": 0}
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, **kw)
    if cuda:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    return model, tok, False


def load_data(a, tok):
    from datasets import concatenate_datasets, load_dataset
    if not os.path.exists(a.data):
        sys.exit(f"{a.data} not found. Run `npm run train:data` first.")
    data = load_dataset("json", data_files=a.data, split="train").shuffle(seed=3407)
    data = data.select(range(min(a.use, len(data))))
    if a.extra and os.path.exists(a.extra):
        # Learned examples are the most valuable data you have, so each is seen 3 times.
        def last_user(ex):
            return max(i for i, m in enumerate(ex["messages"]) if m["role"] == "user" and not m["content"].startswith("<tool_response>"))
        extra = load_dataset("json", data_files=a.extra, split="train")
        extra = extra.map(lambda ex: {"category": "extra", "train_from": ex["train_from"] if ex.get("train_from") is not None else last_user(ex)})
        extra = extra.select_columns(data.column_names)
        print(f"Adding {len(extra)} learned examples (x3)")
        data = concatenate_datasets([data, extra, extra, extra]).shuffle(seed=3407)

    head = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    nl = tok("\n", add_special_tokens=False)["input_ids"]

    def to_features(ex):
        # Render exactly like Qwen's chat template, and learn ONLY the assistant
        # messages after the user's latest message (thinking, tool calls, answer).
        # Earlier turns, the system prompt and tool results are context only.
        ids, labels = [], []
        for i, m in enumerate(ex["messages"]):
            seg = tok(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n", add_special_tokens=False)["input_ids"]
            if m["role"] == "assistant" and i > ex["train_from"]:
                assert seg[:len(head)] == head
                body = seg[len(head):]
                tail = len(nl) if body[-len(nl):] == nl else 0   # don't learn the newline after <|im_end|>
                labels += [-100] * len(head) + body[:len(body) - tail] + [-100] * tail
            else:
                labels += [-100] * len(seg)
            ids += seg
        return {"input_ids": ids[:a.max_len], "attention_mask": [1] * min(len(ids), a.max_len), "labels": labels[:a.max_len]}

    n = len(data)
    data = data.map(to_features, remove_columns=data.column_names, num_proc=min(4, os.cpu_count() or 1))
    # A conversation cut so short that nothing is left to learn would make the loss NaN.
    data = data.filter(lambda ex: any(x != -100 for x in ex["labels"]))
    if len(data) < n:
        print(f"Skipped {n - len(data)} conversations longer than --max-len {a.max_len}")
    sample = data["input_ids"][:1000]
    print(f"{len(data):,} conversations, {sum(map(len, sample)) // len(sample):,} tokens each on average")
    return data


def train(a, model, tok, data, unsloth):
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    try:
        import bitsandbytes  # noqa: F401
        optim = "adamw_8bit" if cuda else "adamw_torch"
    except Exception:
        optim = "adamw_torch"
    kw = dict(
        output_dir=a.out, per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr, warmup_steps=a.warmup_steps,
        lr_scheduler_type="cosine", optim=optim, weight_decay=0.01, fp16=cuda and not bf16, bf16=bf16,
        logging_steps=a.logging_steps, save_steps=a.save_steps, save_total_limit=2, seed=3407, report_to="none")
    if "group_by_length" in inspect.signature(TrainingArguments).parameters:
        kw["group_by_length"] = True   # removed in transformers 5
    trainer = Trainer(model=model, args=TrainingArguments(**kw), train_dataset=data,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    ckpts = glob.glob(os.path.join(a.out, "checkpoint-*"))
    if ckpts:
        print(f"Resuming from the last checkpoint in {a.out}")
    t0 = time.time()
    if a.preset == "smoke" and not ckpts:
        # Stop half way and resume, to prove a Colab disconnect won't lose the run.
        first = a.save_steps
        trainer.args.max_steps = first
        trainer.train()
        trainer.args.max_steps = a.max_steps
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train(resume_from_checkpoint=bool(ckpts))
    secs = time.time() - t0
    final = os.path.join(a.out, "lora-final")
    model.save_pretrained(final)
    tok.save_pretrained(final)
    steps = trainer.state.global_step
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(f"Trained {steps} steps in {secs / 60:.1f} min; loss {losses[0]:.3f} -> {losses[-1]:.3f}" if losses else f"Trained {steps} steps")
    print(f"LoRA saved to {final}")
    return steps, secs


def estimate(a, steps, secs):
    """How long the full preset would take on this GPU, from the smoke run's speed."""
    if a.preset != "smoke" or not steps:
        return
    with open(a.data) as f:
        total = sum(1 for _ in f)
    per_step = a.batch * a.grad_accum
    for use in (total, 25000):
        full_steps = math.ceil(min(use, total) / per_step)
        print(f"Full run on this machine, {min(use, total):,} conversations: ~{full_steps * secs / steps / 3600:.1f} h ({full_steps:,} steps)")


def sample_answer(a, model, tok, unsloth):
    import torch
    if unsloth:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    model.eval()
    with open(a.test) as f:
        case = json.loads(f.readline())
    enc = tok.apply_chat_template(case["messages"], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
    print("\nUSER:", case["messages"][-1]["content"])
    print("MNX: ", tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False), "\n")


def export(a, model, tok, unsloth):
    merged = os.path.join(a.work, "merged")
    shutil.rmtree(merged, ignore_errors=True)
    print(f"Merging the LoRA into a 16-bit model in {merged}")
    if unsloth:
        model.save_pretrained_merged(merged, tok, save_method="merged_16bit")
    else:
        model.merge_and_unload().save_pretrained(merged)
        tok.save_pretrained(merged)
    subprocess.run(["bash", os.path.join(HERE, "export_gguf.sh"), merged, a.out, a.quants], check=True)


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    print(f"Preset {a.preset}: {a.base}, up to {a.use:,} conversations, output in {a.out}")
    model, tok, unsloth = load_model(a)
    data = load_data(a, tok)
    steps, secs = train(a, model, tok, data, unsloth)
    sample_answer(a, model, tok, unsloth)
    if not a.no_export:
        export(a, model, tok, unsloth)
    estimate(a, steps, secs)
    print("Done." + (" Smoke test passed: every step works. Now run --preset full." if a.preset == "smoke" else ""))


if __name__ == "__main__":
    main()
