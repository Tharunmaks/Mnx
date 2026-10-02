"""OpenAI-compatible chat server for a fine-tuned model.

POST /v1/chat/completions {"messages": [...], "max_tokens": 256, "temperature": 0.7}
GET  /v1/models, /health
"""
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D = os.environ.get("MODEL_DIR", "output")
meta = json.load(open(os.path.join(D, "mnx_model.json")))
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
if meta.get("merged", True):
    path = os.path.join(D, "model")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device)
else:
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(os.path.join(D, "adapter"))
    model = AutoModelForCausalLM.from_pretrained(meta["base_model"], torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(model, os.path.join(D, "adapter"))
model.eval()
NAME = os.environ.get("MODEL_NAME", "mnx")
lock = threading.Lock()  # one generation at a time keeps memory predictable


def chat(messages, max_tokens=256, temperature=0.7):
    if tok.chat_template:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant: "
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with lock, torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max(1, min(int(max_tokens), 4096)),
                             do_sample=temperature > 0, temperature=max(float(temperature), 1e-5),
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    new = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True), inputs["input_ids"].shape[1], len(new)


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        if self.path in ("/v1/models", "/info"):
            return self._send(200, {"object": "list", "kind": "llm", "data": [{"id": NAME, "object": "model"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            text, n_in, n_out = chat(req["messages"], req.get("max_tokens", 256), req.get("temperature", 0.7))
            self._send(200, {
                "id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion", "created": int(time.time()),
                "model": NAME, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
            })
        except Exception as exc:
            self._send(400, {"error": str(exc)})

    def log_message(self, *a):
        pass


port = int(os.environ.get("PORT", "8080"))
print(f"Serving {NAME} on :{port} ({device})", flush=True)
ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
