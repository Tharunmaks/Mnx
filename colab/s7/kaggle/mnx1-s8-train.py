# Mnx 1 (306M) session 8 on a free Kaggle T4: reasoning / logical thinking / research.
# Starts from the session 7 step-1500 model (kernel output of tharun77mnx/mnx1-s7-train, sha256-checked).
#  1. Data, all with answers computed by code (no LLM output, no personal data):
#     Ai-room logic, discrete maths, probability, research, statistics, number theory, optimisation, data science,
#     cs and algorithms families (new seed 838383, 40% of the mix) + rehearsal of everything session 7 trained on (60%).
#  2. BEFORE / AFTER on three unseen sets: 300 usual-style, 300 Ai-room (same as session 7) and 400 new reasoning
#     questions (seed 737373; no question shared with any training data). Every 250-step snapshot is scored.
#  3. KEEP only if the reasoning score rises while the usual-style score stays within 9 and Ai-room within 15.
import os, glob, json, shutil, subprocess, sys, time, hashlib, urllib.request
T0 = time.time()
KAGGLE_LIMIT = 12 * 3600
S7_COMMIT = 'S7COMMIT'
S7_FILES = {'s7_tools.py': '39f837ba5f8909c543586594e682112e55c7e9ebbf13732028561ef3aefa76d6', 'qa_gen.py': '36baa86989c93f37a55c23b18d1a3b64d880607db99d0ba603f76524546d7f80', 's8_tools.py': 'S8HASH', 'grow_mnx.py': 'GROWHASH'}
AIROOM_COMMIT = 'e5c7ca349af7dee9cb6e3d53a8e82aa90796d48b'
START_FROM = 's7'   # 's6' = s6 best from Drive (preferred), 's5' = s5 best step 100 already in the mnx1-s6-data Kaggle dataset
S6_EXPORT_FOLDER = '1xQDkAmsyhkCcugJtgx8YMjWXO51lxmf8'   # Drive Mnx1-100m-colab/runs/s6/export_ckpt_best (link-shared, view only)
W = '/kaggle/working'; TMP = '/tmp/s8'
CODE, DATA, RD, CK, D7, AIROOM = TMP + '/code', TMP + '/data', W + '/runs/s8', TMP + '/ckpt', W + '/data_s8', TMP + '/ai-room'
for p in (CODE, DATA, RD, CK, D7): os.makedirs(p, exist_ok=True)
def sh(*a, **k): print('$', ' '.join(a), flush=True); return subprocess.run(list(a), check=True, **k)
def left(): return KAGGLE_LIMIT - (time.time() - T0)
def stamp(msg): print('[%5.1f h] %s' % ((time.time() - T0) / 3600, msg), flush=True)

import torch
assert torch.cuda.is_available(), 'no GPU: set Accelerator to GPU T4'
cap = torch.cuda.get_device_capability(0); AMP = 'bf16' if cap[0] >= 8 else 'fp16'
stamp('GPU %s %s amp %s torch %s' % (torch.cuda.get_device_name(0), cap, AMP, torch.__version__))

# ---- 1. code: Kaggle mnx1-s5-code (trainer, repo, tokenizer) + s6 trainer/adapters from mnx1-s6-data + s7 helpers from GitHub
src_code = os.path.dirname(os.path.dirname(glob.glob('/kaggle/input/**/trainer/train.py', recursive=True)[0]))
s6d = os.path.dirname(glob.glob('/kaggle/input/**/train_s6.py', recursive=True)[0])
shutil.copytree(src_code, CODE, dirs_exist_ok=True)
for f in ('mnx1_custom_s6.py', 'mnx1_small.py'): shutil.copyfile(s6d + '/' + f, CODE + '/trainer/adapters/' + f)
shutil.copyfile(s6d + '/train_s6.py', CODE + '/trainer/train_s6.py')
sums6 = dict(l.split()[::-1] for l in open(s6d + '/SHA256SUMS') if l.strip())
for f in ('train_s6.py', 'mnx1_custom_s6.py', 'mnx1_small.py', 'kaggle_t4_s6.json'):
    assert hashlib.sha256(open(s6d + '/' + f, 'rb').read()).hexdigest() == sums6[f], 'mnx1-s6-data corrupted: ' + f
for f, h in S7_FILES.items():
    urllib.request.urlretrieve('https://raw.githubusercontent.com/Tharunmaks/Mnx/%s/colab/s7/%s' % (S7_COMMIT, f), CODE + '/' + f)
    assert hashlib.sha256(open(CODE + '/' + f, 'rb').read()).hexdigest() == h, 'download corrupted: ' + f
if not os.path.isdir(AIROOM + '/ai_room'):
    sh('git', 'clone', '-q', 'https://github.com/Tharunmaks/ai-room', AIROOM); sh('git', '-C', AIROOM, 'checkout', '-q', AIROOM_COMMIT)
os.environ['MNX_REPO'] = CODE + '/repo'
stamp('code ready: ' + src_code + ' + ' + s6d)

# ---- 2. start weights: the session 7 step-1500 export (kernel output of mnx1-s7-train)
S7_SHA = 'b723ce80'
if not os.path.exists(CK + '/ckpt_best.pt'):
    cands = [p for p in glob.glob('/kaggle/input/**/export_ckpt_best.pt', recursive=True) if 's7' in p]
    assert cands, 'session 7 export not found: add tharun77mnx/mnx1-s7-train as a kernel source'
    h = hashlib.sha256(open(cands[0], 'rb').read()).hexdigest(); assert h.startswith(S7_SHA), 'wrong start model ' + h
    shutil.copyfile(cands[0], CK + '/ckpt_best.pt'); stamp('start = s7 step 1500 export ' + h[:12])

# ---- 3. data: every *.bin from the Kaggle datasets, linked into one folder (the fresh step dedupes against all *.train.bin)
for p in glob.glob('/kaggle/input/**/*.bin', recursive=True):
    d = DATA + '/' + os.path.basename(p)
    if not os.path.exists(d): os.symlink(p, d)
cfg = json.load(open(s6d + '/kaggle_t4_s6.json'))
for s in cfg['data']['sources'] + [{'train': cfg['data']['val']}] + [{'train': v} for v in cfg.get('extra_val', {}).values()]:
    assert os.path.exists(DATA + '/' + s['train']), 'missing data file ' + s['train']
NEED = ('fresh_qa.train.bin', 'fresh_qa.val.bin', 'eval_unseen.jsonl', 'format.json', 'airoom.train.bin', 'airoom.val.bin', 'eval_airoom.jsonl')
if not all(os.path.exists(D7 + '/' + f) for f in NEED):
    sh(sys.executable, '-u', CODE + '/s7_tools.py', 'fresh', '--code', CODE, '--data', DATA, '--out', D7, '--n', '300000', '--eval-n', '300', '--airoom', AIROOM, '--airoom-n', '200000')
    if os.path.exists(D7 + '/train_q_hashes.npy'): os.remove(D7 + '/train_q_hashes.npy')
if not all(os.path.exists(D7 + '/' + f) for f in ('reason.train.bin', 'reason.val.bin', 'eval_reason.jsonl')):
    sh(sys.executable, '-u', CODE + '/s8_tools.py', 'reason', '--code', CODE, '--data', DATA, '--out', D7, '--airoom', AIROOM, '--n', '500000', '--eval-n', '400')
for f in ('fresh_qa.train.bin', 'fresh_qa.val.bin', 'airoom.train.bin', 'airoom.val.bin', 'reason.train.bin', 'reason.val.bin'):
    if not os.path.exists(DATA + '/' + f): os.symlink(D7 + '/' + f, DATA + '/' + f)
stamp('fresh data: ' + str({f: os.path.getsize(D7 + '/' + f) for f in NEED}))

# ---- 4. BEFORE scores on the unseen sets
SCORE = [sys.executable, '-u', CODE + '/s7_tools.py', 'score', '--code', CODE, '--fmt', D7 + '/format.json', '--airoom', AIROOM]
ITEMS = {'qa': D7 + '/eval_unseen.jsonl', 'ar': D7 + '/eval_airoom.jsonl', 'rs': D7 + '/eval_reason.jsonl'}
TAGS = ('qa', 'ar', 'rs')
for tag in TAGS:
    if not os.path.exists(RD + '/score_before_%s.json' % tag):
        stamp('BEFORE (%s): s7 step 1500 on unseen questions' % tag)
        sh(*SCORE, '--items', ITEMS[tag], '--ckpt', CK + '/ckpt_best.pt', '--out', RD + '/score_before_%s.json' % tag)
BEFORE = {t: json.load(open(RD + '/score_before_%s.json' % t)) for t in TAGS}
for t, b in BEFORE.items(): stamp('BEFORE %s %d/%d %s' % (t, b['total'], b['n'], dict(b['by_kind'])))

# ---- 5. config (same as Colab s7) + step 0 from the s6 best with fresh AdamW
for s in cfg['data']['sources']: s['weight'] = round(s['weight'] * 0.40, 4)
cfg['data']['sources'].insert(0, {'name': 'fresh_qa', 'train': 'fresh_qa.train.bin', 'weight': 0.10})
cfg['data']['sources'].insert(0, {'name': 'airoom', 'train': 'airoom.train.bin', 'weight': 0.10})
cfg['data']['sources'].insert(0, {'name': 'reason', 'train': 'reason.train.bin', 'weight': 0.40})
cfg['extra_val'] = {**cfg.get('extra_val', {}), 'fresh_qa': 'fresh_qa.val.bin', 'airoom': 'airoom.val.bin', 'reason': 'reason.val.bin'}
cfg['lr'] = 4e-5
SNAP_EVERY = 250
AFTER_RESERVE = 150 * 60   # score every snapshot on 3 sets (~13 min each, up to 7) + export + Kaggle saving the output
cfg.update(run_dir=RD, device='cuda', amp=AMP, adapter=CODE + '/trainer/adapters/mnx1_custom_s6.py',
           stop_after_sec=int(left() - AFTER_RESERVE), ckpt_every_sec=600, eval_at_start=True, early_stop_patience=0,   # session 7 rerun: never stop on old-data held-out loss; judge by the unseen-question score instead
           
           _note='s8 (Kaggle T4): start from s7 step 1500; reason 0.40 + rehearsal (old data x0.40, fresh_qa 0.10, airoom 0.10); lr 4e-5')
cfg['data']['root'] = DATA
json.dump(cfg, open(RD + '/cfg.json', 'w'), indent=1)
stamp('lr %s max_steps %s stop_after_sec %s mix %s' % (cfg['lr'], cfg['max_steps'], cfg['stop_after_sec'], [(s['name'], s['weight']) for s in cfg['data']['sources']]))
import importlib.util
sys.path.insert(0, CODE)
sp = importlib.util.spec_from_file_location('ad', cfg['adapter']); ad = importlib.util.module_from_spec(sp); sp.loader.exec_module(ad)
ck = torch.load(CK + '/ckpt_best.pt', map_location='cpu', weights_only=False); fs = ck.get('state') or {}
import grow_mnx   # depth growth 24 -> 36 layers: new blocks have zero output matrices, so the model computes exactly what step 1500 did
NEW_LAYERS = 36
ck = grow_mnx.grow(ck, NEW_LAYERS); cfg['model_cfg']['n_layers'] = NEW_LAYERS; ck['config'] = {**ck['config'], 'model_cfg': cfg['model_cfg']}
cfg['max_steps'] = 1200   # 36 layers is ~1.5x slower per step; keeps the cosine schedule inside the Kaggle limit
stamp('grown to %d layers, %.0fM params' % (NEW_LAYERS, sum(v.numel() for v in ck['model'].values()) / 1e6))
stamp('[START] ' + START_FROM + ' best ' + str({k: fs.get(k) for k in ('step', 'tokens', 'best', 'best_step')}))
m = ad.build_model({**cfg['model_cfg'], 'grad_ckpt': True}); m.load_state_dict(ck['model'])
osd = ad.configure_optimizer(m, cfg).state_dict()
for pg in osd['param_groups']: pg['fused'] = True
st = {'step': 0, 'tokens': 0, 'best': float('inf'), 'noimp': 0, 'from': START_FROM + ' best, fresh AdamW', 'from_state': fs}
torch.save({'model': m.state_dict(), 'opt': osd, 'scaler': None, 'state': st, 'config': cfg}, RD + '/ckpt_latest.pt'); del ck, m, osd
os.remove(CK + '/ckpt_best.pt')

# ---- 6. train, with a watcher that keeps a model-only snapshot of every 250-step checkpoint.
# Nothing is deleted before it has been scored: snapshots and ckpt_latest stay in the kernel output.
import threading
SNAPS = RD + '/snaps'; os.makedirs(SNAPS, exist_ok=True)
_stop_w = False
def snap_once():
    p = RD + '/ckpt_latest.pt'
    if not os.path.exists(p): return
    try:
        c = torch.load(p, map_location='cpu', weights_only=False, mmap=True); step = int(c['state']['step'])
        out = SNAPS + '/step_%04d.pt' % step
        if step > 0 and not os.path.exists(out) and (step % SNAP_EVERY == 0 or _stop_w):
            torch.save({'model': c['model'], 'state': dict(c['state']), 'config': c['config']}, out + '.tmp'); os.replace(out + '.tmp', out)
            stamp('snapshot step %d' % step)
        del c
    except Exception as e: print('snap error', repr(e)[:200], flush=True)
def watch():
    last = None
    while not _stop_w:
        p = RD + '/ckpt_latest.pt'
        m = os.path.getmtime(p) if os.path.exists(p) else None
        if m and m != last: last = m; snap_once()
        time.sleep(20)
threading.Thread(target=watch, daemon=True).start()
os.environ.setdefault('OMP_NUM_THREADS', '2')
r = subprocess.run([sys.executable, '-u', CODE + '/trainer/train_s6.py', '--config', RD + '/cfg.json'])
stamp('train exit %d' % r.returncode)
_stop_w = True; time.sleep(1); snap_once()   # the final checkpoint too

# ---- 7. score EVERY snapshot on the same 600 unseen questions, keep the best one
RES = {}
for f in sorted(glob.glob(SNAPS + '/step_*.pt')):
    step = int(os.path.basename(f)[5:9]); R = {}
    for tag in TAGS:
        o = RD + '/score_step%04d_%s.json' % (step, tag)
        if not os.path.exists(o): sh(*SCORE, '--items', ITEMS[tag], '--ckpt', f, '--out', o)
        R[tag] = json.load(open(o))
    RES[step] = R; stamp('step %d: usual %d/%d  ai-room %d/%d  reasoning %d/%d' % (step, R['qa']['total'], R['qa']['n'], R['ar']['total'], R['ar']['n'], R['rs']['total'], R['rs']['n']))
ok = {s_: R for s_, R in RES.items() if R['qa']['total'] >= BEFORE['qa']['total'] - 9 and R['ar']['total'] >= BEFORE['ar']['total'] - 15}
bestst = max(ok, key=lambda s_: (ok[s_]['rs']['total'], sum(ok[s_][t]['total'] for t in ok[s_]))) if ok else None
lines = ['session 8 verdict (Kaggle T4, started from the s7 step 1500 model) ' + time.strftime('%F %T')]
def row(lbl, R): return '%s: usual %3d/%d  ai-room %3d/%d  reasoning %3d/%d' % (lbl, R['qa']['total'], R['qa']['n'], R['ar']['total'], R['ar']['n'], R['rs']['total'], R['rs']['n'])
lines.append(row('step    0 (= s7 step 1500)', BEFORE))
for s_, R in sorted(RES.items()): lines.append(row('step %4d' % s_, R))
KEEP = bestst is not None and RES[bestst]['rs']['total'] > BEFORE['rs']['total']
if KEEP:
    A, B = RES[bestst], BEFORE
    lines.append('KEEP step %d. Model-only copy: runs/s8/export_ckpt_best.pt.' % bestst)
    for tag, name in (('qa', 'usual-style'), ('ar', 'Ai-room'), ('rs', 'reasoning/logic/research')):
        lines.append('%s: %d/%d -> %d/%d' % (name, B[tag]['total'], B[tag]['n'], A[tag]['total'], A[tag]['n']))
        for k in sorted(set(A[tag]['by_kind']) | set(B[tag]['by_kind'])):
            b_, a_ = B[tag]['by_kind'].get(k, [0, 0]), A[tag]['by_kind'].get(k, [0, 0]); lines.append('  %-16s %3d/%-3d -> %3d/%-3d' % (k, b_[1], b_[0], a_[1], a_[0]))
    shutil.copyfile(SNAPS + '/step_%04d.pt' % bestst, RD + '/export_ckpt_best.pt')
    lines.append('export sha256 ' + subprocess.run(['sha256sum', RD + '/export_ckpt_best.pt'], capture_output=True, text=True).stdout.split()[0])
else:
    lines.append('DO NOT USE: no snapshot raised the reasoning score while staying within the usual-style/Ai-room limits. Keep using the s7 step 1500 model.')
open(RD + '/VERDICT.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines), flush=True)
# the checkpoints stay (snapshots are model-only, ckpt_latest allows resuming); only the regenerable training bins go
for f in ('fresh_qa.train.bin', 'airoom.train.bin', 'reason.train.bin'):
    if os.path.exists(D7 + '/' + f): os.remove(D7 + '/' + f)
subprocess.run('ls -la %s %s %s; tail -3 %s/log.jsonl; du -sh %s' % (RD, SNAPS, D7, RD, W), shell=True)
stamp('done')
