"""Serve a trained text classifier. POST /predict {"text": "..."} or {"texts": [...]}."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import joblib

MODEL_DIR = os.environ.get("MODEL_DIR", "output")
bundle = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
vec, clf, labels = bundle["vectorizer"], bundle["model"], bundle["labels"]


def predict(texts):
    probs = clf.predict_proba(vec.transform(texts))
    out = []
    for row in probs:
        scores = {str(c): round(float(p), 4) for c, p in zip(clf.classes_, row)}
        best = max(scores, key=scores.get)
        out.append({"label": best, "scores": scores})
    return out


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
        if self.path == "/info":
            return self._send(200, {"kind": "classifier", "labels": labels})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            texts = req.get("texts") or [req.get("text", "")]
            res = predict([str(t) for t in texts])
            self._send(200, res[0] if "text" in req else {"results": res})
        except Exception as exc:
            self._send(400, {"error": str(exc)})

    def log_message(self, *a):
        pass


port = int(os.environ.get("PORT", "8080"))
print(f"Serving classifier on :{port}", flush=True)
ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
