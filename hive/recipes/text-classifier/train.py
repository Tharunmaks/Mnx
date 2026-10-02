"""Text classifier: TF-IDF + logistic regression trained epoch by epoch (SGD).

Mnx Lab protocol: reads params.json and data/, writes output/, prints
"MNX_METRIC {json}" per epoch and "MNX_RESULT {json}" at the end.
"""
import csv
import glob
import json
import os
import random
import sys

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

P = json.load(open("params.json"))
os.makedirs("output", exist_ok=True)


def metric(**kw):
    print("MNX_METRIC " + json.dumps(kw), flush=True)


def load_rows():
    files = sorted(glob.glob("data/*.csv") + glob.glob("data/*.jsonl"))
    if not files:
        sys.exit("No .csv or .jsonl file in the dataset")
    tc, lc = P.get("text_column", "text"), P.get("label_column", "label")
    rows = []
    for f in files:
        if f.endswith(".jsonl"):
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    o = json.loads(line)
                    rows.append((str(o[tc]), str(o[lc])))
        else:
            with open(f, newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                missing = {tc, lc} - set(reader.fieldnames or [])
                if missing:
                    sys.exit(f"{f}: missing column(s) {', '.join(missing)}; found {reader.fieldnames}")
                rows += [(r[tc], r[lc]) for r in reader if r[tc] and r[lc]]
    return rows


rows = load_rows()
labels = sorted({l for _, l in rows})
if len(labels) < 2:
    sys.exit("Need at least two different labels")
print(f"{len(rows)} examples, {len(labels)} labels: {', '.join(labels[:20])}", flush=True)

random.seed(42)
random.shuffle(rows)
n_test = max(1, int(len(rows) * float(P.get("test_split", 0.2))))
test, train = rows[:n_test], rows[n_test:]

vec = TfidfVectorizer(max_features=int(P.get("max_features", 20000)),
                      ngram_range=(1, int(P.get("ngram_max", 2))), sublinear_tf=True)
Xtr = vec.fit_transform([t for t, _ in train])
Xte = vec.transform([t for t, _ in test])
ytr = np.array([l for _, l in train])
yte = np.array([l for _, l in test])

clf = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=42)
epochs = int(P.get("epochs", 20))
idx = np.arange(Xtr.shape[0])
for epoch in range(1, epochs + 1):
    np.random.shuffle(idx)
    clf.partial_fit(Xtr[idx], ytr[idx], classes=np.array(labels))
    loss = log_loss(ytr, clf.predict_proba(Xtr), labels=labels)
    acc = accuracy_score(yte, clf.predict(Xte))
    metric(step=epoch, epoch=epoch, loss=round(float(loss), 5), val_accuracy=round(float(acc), 4))

pred = clf.predict(Xte)
result = {
    "accuracy": round(float(accuracy_score(yte, pred)), 4),
    "f1_macro": round(float(f1_score(yte, pred, average="macro")), 4),
    "examples": len(rows), "labels": labels,
}
joblib.dump({"vectorizer": vec, "model": clf, "labels": labels}, "output/model.joblib")
json.dump(result, open("output/metrics.json", "w"), indent=2)
print("MNX_RESULT " + json.dumps(result), flush=True)
