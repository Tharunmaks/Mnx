"""Writes session7.ipynb (Colab, free T4). Run: python mk_nb_s7.py > session7.ipynb"""
import json

MD = """# Mnx 1 (306M) - session 7 on free Colab T4: fresh questions + unseen-question score before and after
**Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**. If Colab disconnects, just Run all again: every step resumes from Drive `Mnx1-100m-colab/runs/s7`.

What it does, in order:
1. Starts from the session 6 best (`runs/s6/export_ckpt_best`, read only).
2. Generates ~600k **fresh** verified Q&A pairs (new seed), dropping any question already in your training data, and adds them to the mix at 25%.
3. Scores the session 6 model on 300 questions it has **never trained on** (and never will: they are excluded from the fresh data too).
4. Trains, with early stop after 4 evals without a better held-out loss (same trainer as s6).
5. Scores the new best on the same 300 questions. The result is kept (`runs/s7/export_ckpt_best`) **only if it answers more of them right**. `runs/s7/VERDICT.txt` says which.

No paid GPU, no compute units. Writes only to `runs/s7` and `data_s7`. Notebook source: github.com/Tharunmaks/Mnx/tree/claude/project-thread-7qmqag/colab/s7"""

C1 = r"""S7_COMMIT = @@COMMIT@@
S7_FILES = @@FILES@@
import os, time, json, glob, shutil, subprocess, sys, threading
T_NB0 = time.time()
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
D = '/content/drive/MyDrive/Mnx1-100m-colab'
C6 = D + '/code_s6'
START_DIR = D + '/runs/s6/export_ckpt_best'      # s6 best, model-only, ckpt_best.pt.partNN + SHA256SUMS (read only)
RUN_DRIVE = D + '/runs/s7'; assert RUN_DRIVE.endswith('/runs/s7'); os.makedirs(RUN_DRIVE, exist_ok=True)
DATA7 = D + '/data_s7'; os.makedirs(DATA7, exist_ok=True)
DATA, CODE, RD = '/content/data', '/content/code', '/content/run'
for p in (DATA, CODE, RD, '/content/ckpt', '/content/data7'): os.makedirs(p, exist_ok=True)
for f in ('train_s6.py', 'mnx1_custom_s6.py', 'mnx1_small.py', 'colab_t4_s6.json', 'mk_s6_start.py'): assert os.path.exists(C6 + '/' + f), 'Drive code_s6/ missing ' + f
# session-7 helpers come from GitHub (public repo Tharunmaks/Mnx), pinned to one commit and sha256-checked
import urllib.request, hashlib
C7 = '/content/code_s7'; os.makedirs(C7, exist_ok=True)
for f, h in S7_FILES.items():
    urllib.request.urlretrieve('https://raw.githubusercontent.com/Tharunmaks/Mnx/%s/colab/s7/%s' % (S7_COMMIT, f), C7 + '/' + f)
    assert hashlib.sha256(open(C7 + '/' + f, 'rb').read()).hexdigest() == h, 'download corrupted: ' + f
assert os.path.exists(START_DIR + '/SHA256SUMS'), 'session 6 export missing: ' + START_DIR
print(sorted(os.listdir(D)))"""

C2 = r"""import torch
assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > T4 GPU (free tier). Do NOT buy compute units.'
cap = torch.cuda.get_device_capability(0); AMP = 'bf16' if cap[0] >= 8 else 'fp16'
print('GPU', torch.cuda.get_device_name(0), cap, 'amp', AMP, 'torch', torch.__version__)"""

C3 = r"""# code + data from Drive (same as s6), files are split into parts -> join + sha256 check
subprocess.run(['tar', 'xzf', D + '/code/mnx1_code.tar.gz', '-C', CODE], check=True)
os.makedirs(CODE + '/trainer/adapters', exist_ok=True)
for f in ('mnx1_custom_s6.py', 'mnx1_small.py'): shutil.copyfile(C6 + '/' + f, CODE + '/trainer/adapters/' + f)
shutil.copyfile(C6 + '/train_s6.py', CODE + '/trainer/train_s6.py')
for f in ('s7_tools.py', 'qa_gen.py'): shutil.copyfile(C7 + '/' + f, CODE + '/' + f)
os.environ['MNX_REPO'] = CODE + '/repo'
def join_parts(src_dir, dst_dir, sums_file='SHA256SUMS'):
    sums = {k: v for k, v in (l.split()[::-1] for l in open(src_dir + '/' + sums_file) if l.strip()) if '.part' not in k}
    for name, h in sums.items():
        dst = dst_dir + '/' + name
        if os.path.exists(dst + '.ok') and open(dst + '.ok').read().strip() == h: continue
        parts = sorted(glob.glob(src_dir + '/' + name + '.part*')); assert parts, 'missing parts for ' + name
        for attempt in range(3):
            with open(dst, 'wb') as o:
                for p in parts:
                    with open(p, 'rb') as i: shutil.copyfileobj(i, o, 16 << 20)
            got = subprocess.run(['sha256sum', dst], capture_output=True, text=True).stdout.split()[0]
            if got == h: break
            print('sha256 mismatch, retry', name, attempt)
        assert got == h, 'sha256 mismatch ' + name
        open(dst + '.ok', 'w').write(h); print('ok', name, os.path.getsize(dst))
t = time.time(); join_parts(D + '/data', DATA); join_parts(D + '/data_s3', DATA, 'SHA256SUMS_v4'); join_parts(D + '/data_s6', DATA)
print('data ready in %.0fs' % (time.time() - t))"""

C4 = r"""# fresh questions + the unseen eval set. Built once (~15-25 min on Colab CPU), then cached in Drive data_s7/
NEED = ('fresh_qa.train.bin', 'fresh_qa.val.bin', 'eval_unseen.jsonl', 'format.json')
if all(os.path.exists(DATA7 + '/' + f) for f in NEED):
    for f in NEED: shutil.copyfile(DATA7 + '/' + f, '/content/data7/' + f)
    print('fresh data from Drive cache')
else:
    subprocess.run([sys.executable, '-u', CODE + '/s7_tools.py', 'fresh', '--code', CODE, '--data', DATA, '--out', '/content/data7', '--n', '600000', '--eval-n', '300'], check=True)
    for f in NEED: shutil.copyfile('/content/data7/' + f, DATA7 + '/' + f + '.tmp'); os.replace(DATA7 + '/' + f + '.tmp', DATA7 + '/' + f)
for f in ('fresh_qa.train.bin', 'fresh_qa.val.bin'): shutil.copyfile('/content/data7/' + f, DATA + '/' + f)
print({f: os.path.getsize('/content/data7/' + f) for f in NEED})"""

C5 = r"""# session 7 config = s6 config + fresh_qa at 25% of the mix (other sources scaled to 75%)
cfg = json.load(open(C6 + '/colab_t4_s6.json'))
for s in cfg['data']['sources']: s['weight'] = round(s['weight'] * 0.75, 4)
cfg['data']['sources'].insert(0, {'name': 'fresh_qa', 'train': 'fresh_qa.train.bin', 'weight': 0.25})
cfg['extra_val'] = {**cfg.get('extra_val', {}), 'fresh_qa': 'fresh_qa.val.bin'}
LIMIT = 11 * 3600 + 15 * 60 - 25 * 60   # Colab free ~12h per VM; leave ~25 min for the after-score
cfg.update(run_dir=RD, device='cuda', amp=AMP, adapter=CODE + '/trainer/adapters/mnx1_custom_s6.py',
           stop_after_sec=int(LIMIT - (time.time() - T_NB0)), ckpt_every_sec=600, eval_at_start=True, early_stop_patience=4,
           _note='s7: start from s6 best; + fresh_qa 0.25 (new seed 7007, no question shared with training data or the unseen eval set)')
cfg['data']['root'] = DATA
json.dump(cfg, open('/content/cfg.json', 'w'), indent=1)
print('lr', cfg['lr'], 'dropout', cfg['model_cfg'].get('dropout'), 'max_steps', cfg['max_steps'], 'stop_after_sec', cfg['stop_after_sec'])
print([(s['name'], s['weight']) for s in cfg['data']['sources']])"""

C6_ = r"""# start or resume. First start: score the s6 best on the unseen questions, then make s7 step 0 (fresh AdamW).
import torch as _t
def cp(a, b):
    if os.path.exists(a): shutil.copyfile(a, b + '.tmp'); os.replace(b + '.tmp', b); return True
    return False
def ck_state(path):
    try: ck = _t.load(path, map_location='cpu', weights_only=False, mmap=True); st = dict(ck['state']); del ck; return st
    except Exception as e: print('   cannot read', path, repr(e)[:200]); return None
SCORE = [sys.executable, '-u', CODE + '/s7_tools.py', 'score', '--code', CODE, '--items', '/content/data7/eval_unseen.jsonl', '--fmt', '/content/data7/format.json']
if not os.path.exists(RUN_DRIVE + '/score_before.json') or not os.path.exists(RUN_DRIVE + '/ckpt_latest.pt'):
    join_parts(START_DIR, '/content/ckpt')
    print('[START] s6 best', {k: v for k, v in (ck_state('/content/ckpt/ckpt_best.pt') or {}).items() if k != 'from_state'})
if not os.path.exists(RUN_DRIVE + '/score_before.json'):
    print('=== BEFORE: s6 best on 300 unseen questions ===')
    subprocess.run(SCORE + ['--ckpt', '/content/ckpt/ckpt_best.pt', '--out', '/content/score_before.json'], check=True)
    cp('/content/score_before.json', RUN_DRIVE + '/score_before.json')
BEFORE = json.load(open(RUN_DRIVE + '/score_before.json')); print('BEFORE', BEFORE['total'], '/', BEFORE['n'], BEFORE['by_kind'])
st = ck_state(RUN_DRIVE + '/ckpt_latest.pt') if os.path.exists(RUN_DRIVE + '/ckpt_latest.pt') else None
if st:
    cp(RUN_DRIVE + '/ckpt_latest.pt', RD + '/ckpt_latest.pt')
    for f in ('ckpt_best.pt', 'log.jsonl', 'meta.json'): cp(RUN_DRIVE + '/' + f, RD + '/' + f)
    print('[RESUME] s7 step', st['step'], 'best', st.get('best'))
else:
    subprocess.run([sys.executable, C6 + '/mk_s6_start.py', '/content/ckpt/ckpt_best.pt', '/content/cfg.json', RD + '/ckpt_latest.pt', '--any-step'], check=True)
    st = ck_state(RD + '/ckpt_latest.pt'); assert st['step'] == 0 and st['best'] == float('inf'), st
    print('[FIRST START] s7 step 0 from s6 best')
S7_DONE = st.get('noimp', 0) >= cfg['early_stop_patience'] or st.get('step', 0) >= cfg['max_steps']
if os.path.exists('/content/ckpt/ckpt_best.pt'): os.remove('/content/ckpt/ckpt_best.pt')"""

C7_ = r"""# background sync: local run dir -> Drive runs/s7 every 60 s (ckpts copied atomically: never a half ckpt on Drive)
SYNC = ['ckpt_latest.pt', 'ckpt_best.pt', 'log.jsonl', 'meta.json']; _seen = {}; _stop_sync = False
if os.path.exists(RD + '/ckpt_latest.pt'): _seen['ckpt_latest.pt'] = os.path.getmtime(RD + '/ckpt_latest.pt')
def sync_once():
    for f in SYNC:
        a = RD + '/' + f
        if not os.path.exists(a) or _seen.get(f) == os.path.getmtime(a): continue
        m = os.path.getmtime(a)
        try:
            tmp = '/content/_sync_' + f; shutil.copyfile(a, tmp)
            shutil.copyfile(tmp, RUN_DRIVE + '/' + f + '.uploading'); os.replace(RUN_DRIVE + '/' + f + '.uploading', RUN_DRIVE + '/' + f)
            os.remove(tmp); _seen[f] = m
            if f.endswith('.pt'): print('[sync] uploaded', f, time.strftime('%T'))
        except Exception as e: print('sync error', f, e)
def sync_loop():
    while not _stop_sync: sync_once(); time.sleep(60)
threading.Thread(target=sync_loop, daemon=True).start(); print('sync thread on')"""

C8 = r"""# train (single T4). Same trainer as s6: answer-only loss, eval every 50 steps, early stop after 4 evals without a new best
os.environ.setdefault('OMP_NUM_THREADS', '2')
if S7_DONE: print('[S7 TRAINING DONE] already finished at step', st.get('step'), '-> going straight to the after-score')
else:
    p = subprocess.Popen([sys.executable, '-u', CODE + '/trainer/train_s6.py', '--config', '/content/cfg.json'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in p.stdout: print(line, end='')
    except KeyboardInterrupt:
        p.terminate(); print('interrupted: trainer saves ckpt_latest on SIGTERM')
    print('train exit', p.wait())
_stop_sync = True; sync_once()"""

C9 = r"""# AFTER: score the s7 best on the SAME 300 unseen questions. Keep it only if it gets more right.
best = RD + '/ckpt_best.pt'
if not os.path.exists(best): cp(RUN_DRIVE + '/ckpt_best.pt', best)
assert os.path.exists(best), 'no s7 ckpt_best.pt yet'
bst = ck_state(best); print('s7 best: step', bst.get('best_step'), 'held-out', bst.get('best'))
subprocess.run(SCORE + ['--ckpt', best, '--out', '/content/score_after.json'], check=True)
cp('/content/score_after.json', RUN_DRIVE + '/score_after.json')
A, B = json.load(open('/content/score_after.json')), BEFORE
lines = ['session 7 verdict ' + time.strftime('%F %T'),
         'unseen questions right: before (s6 best) %d/%d  ->  after (s7 best, step %s) %d/%d' % (B['total'], B['n'], bst.get('best_step'), A['total'], A['n'])]
for k in sorted(set(A['by_kind']) | set(B['by_kind'])):
    b, a = B['by_kind'].get(k, [0, 0]), A['by_kind'].get(k, [0, 0]); lines.append('  %-11s %3d/%-3d -> %3d/%-3d' % (k, b[1], b[0], a[1], a[0]))
if A['total'] > B['total']:
    lines.append('KEEP: s7 is better. Model-only copy in runs/s7/export_ckpt_best (use this one from now on).')
    ck = _t.load(best, map_location='cpu', weights_only=False, mmap=True); out = '/content/_export.pt'
    _t.save({'model': ck['model'], 'state': ck['state'], 'config': ck['config']}, out); del ck
    ed = RUN_DRIVE + '/export_ckpt_best'; os.makedirs(ed, exist_ok=True); n = 95 << 20; k = -1
    with open(out, 'rb') as fi:
        while True:
            buf = fi.read(n)
            if not buf: break
            k += 1; open(ed + '/ckpt_best.pt.part%02d' % k, 'wb').write(buf)
    h = subprocess.run(['sha256sum', out], capture_output=True, text=True).stdout.split()[0]
    open(ed + '/SHA256SUMS', 'w').write('%s  ckpt_best.pt\n' % h); open(ed + '/NPARTS', 'w').write(str(k + 1)); os.remove(out)
else:
    lines.append('DO NOT USE s7: it is not better on unseen questions. Keep using runs/s6/export_ckpt_best.')
open(RUN_DRIVE + '/VERDICT.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))"""

import hashlib, sys
C1 = C1.replace('@@COMMIT@@', repr(sys.argv[1])).replace('@@FILES@@', repr({f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in ('s7_tools.py', 'qa_gen.py')}))
cells = [("markdown", MD)] + [("code", c) for c in (C1, C2, C3, C4, C5, C6_, C7_, C8, C9)]
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True), **({"execution_count": None, "outputs": []} if t == "code" else {})} for t, s in cells],
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
print(json.dumps(nb, indent=1))
