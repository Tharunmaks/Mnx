"""Serve a spreadsheet predictor. POST /predict {"row": {...}} or {"rows": [{...}, ...]}."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import joblib
import pandas as pd

b = joblib.load(os.path.join(os.environ.get("MODEL_DIR", "output"), "model.joblib"))
pipe, task, inputs = b["pipeline"], b["task"], b["inputs"]


def predict(rows):
    df = pd.DataFrame(rows).reindex(columns=inputs)
    out = [{"prediction": p.item() if hasattr(p, "item") else p} for p in pipe.predict(df)]
    if task == "classification":
        for o, probs in zip(out, pipe.predict_proba(df)):
            o["scores"] = {str(c): round(float(p), 4) for c, p in zip(pipe.classes_, probs)}
    return out


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        if self.path == "/info":
            return self._send(200, {"kind": "tabular", "task": task, "inputs": inputs, "target": b["target"], "example": {"row": b["example"]}})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            rows = req.get("rows") or [req.get("row", {})]
            res = predict(rows)
            self._send(200, res[0] if "row" in req else {"results": res})
        except Exception as exc:
            self._send(400, {"error": str(exc)})

    def log_message(self, *a):
        pass


port = int(os.environ.get("PORT", "8080"))
print(f"Serving {task} model on :{port}", flush=True)
ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
