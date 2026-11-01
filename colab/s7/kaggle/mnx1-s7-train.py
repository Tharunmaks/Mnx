# Mnx 1 (306M) session 7 on a free Kaggle T4: the Kaggle port of colab/s7/session7.ipynb.
# Same steps, same seeds, same keep rule. Differences from Colab: code and old data come from Tharun's
# Kaggle datasets (mnx1-s5-code, mnx1-s5-data, mnx1-s6-data) instead of Drive, the s6 best is
# downloaded from its link-shared Drive folder, and results land in /kaggle/working/runs/s7 (kernel output).
#  1. s6 best (model-only export, 13 parts + SHA256SUMS) from Drive, sha256-checked.
#  2. ~600k fresh verified Q&A (seed 7007) + ~400k Ai-room problems, deduped against every training question.
#  3. BEFORE: s6 best on 300 unseen usual-style + 300 unseen Ai-room questions (never trained on).
#  4. Train with the s6 trainer: answer-only loss, eval every 50, early stop after 4 evals without a new best.
#  5. AFTER: s7 best on the same 600. KEEP only if more of the 600 right and no more than 9 usual-style lost.
import os, glob, json, shutil, subprocess, sys, time, hashlib, urllib.request
T0 = time.time()
KAGGLE_LIMIT = 12 * 3600
S7_COMMIT = '0470553d526466025c29adec87776c1706a7090e'
S7_FILES = {'s7_tools.py': '39f837ba5f8909c543586594e682112e55c7e9ebbf13732028561ef3aefa76d6', 'qa_gen.py': '36baa86989c93f37a55c23b18d1a3b64d880607db99d0ba603f76524546d7f80'}
AIROOM_COMMIT = 'e5c7ca349af7dee9cb6e3d53a8e82aa90796d48b'
START_FROM = 's6'   # 's6' = s6 best from Drive (preferred), 's5' = s5 best step 100 already in the mnx1-s6-data Kaggle dataset
S6_EXPORT_FOLDER = '1xQDkAmsyhkCcugJtgx8YMjWXO51lxmf8'   # Drive Mnx1-100m-colab/runs/s6/export_ckpt_best (link-shared, view only)
W = '/kaggle/working'; TMP = '/tmp/s7'
CODE, DATA, RD, CK, D7, AIROOM = TMP + '/code', TMP + '/data', W + '/runs/s7', TMP + '/ckpt', W + '/data_s7', TMP + '/ai-room'
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

# ---- 2. start weights. s6 best from Drive (fail fast if the folder is not link-shared), or the s5 best from Kaggle
if START_FROM == 's5' and not os.path.exists(CK + '/ckpt_best.pt'):
    assert hashlib.sha256(open(s6d + '/s5_best_step100.pt', 'rb').read()).hexdigest() == sums6['s5_best_step100.pt'], 's5 best corrupted'
    shutil.copyfile(s6d + '/s5_best_step100.pt', CK + '/ckpt_best.pt'); stamp('start = s5 best step 100 (Kaggle mnx1-s6-data)')
if not os.path.exists(CK + '/ckpt_best.pt'):
    try: import gdown
    except ImportError: sh(sys.executable, '-m', 'pip', 'install', '-q', 'gdown'); import gdown
    dl = TMP + '/s6_export'
    files = gdown.download_folder(id=S6_EXPORT_FOLDER, output=dl, quiet=True)
    assert files and os.path.exists(dl + '/SHA256SUMS'), ('Could not download the s6 export from Drive. Share folder %s as "Anyone with the link: Viewer".' % S6_EXPORT_FOLDER)
    want = dict(l.split()[::-1] for l in open(dl + '/SHA256SUMS') if l.strip())['ckpt_best.pt']
    parts = sorted(glob.glob(dl + '/ckpt_best.pt.part*')); stamp('s6 export: %d parts' % len(parts))
    h = hashlib.sha256()
    with open(CK + '/ckpt_best.pt', 'wb') as o:
        for p in parts:
            b = open(p, 'rb').read(); h.update(b); o.write(b); os.remove(p)
    assert h.hexdigest() == want, 'sha256 mismatch on the s6 export (%d parts)' % len(parts)
    stamp('s6 best joined, sha256 ok')

# ---- 3. data: every *.bin from the Kaggle datasets, linked into one folder (the fresh step dedupes against all *.train.bin)
for p in glob.glob('/kaggle/input/**/*.bin', recursive=True):
    d = DATA + '/' + os.path.basename(p)
    if not os.path.exists(d): os.symlink(p, d)
cfg = json.load(open(s6d + '/kaggle_t4_s6.json'))
for s in cfg['data']['sources'] + [{'train': cfg['data']['val']}] + [{'train': v} for v in cfg.get('extra_val', {}).values()]:
    assert os.path.exists(DATA + '/' + s['train']), 'missing data file ' + s['train']
NEED = ('fresh_qa.train.bin', 'fresh_qa.val.bin', 'eval_unseen.jsonl', 'format.json', 'airoom.train.bin', 'airoom.val.bin', 'eval_airoom.jsonl')
if not all(os.path.exists(D7 + '/' + f) for f in NEED):
    sh(sys.executable, '-u', CODE + '/s7_tools.py', 'fresh', '--code', CODE, '--data', DATA, '--out', D7, '--n', '600000', '--eval-n', '300', '--airoom', AIROOM, '--airoom-n', '400000')
    if os.path.exists(D7 + '/train_q_hashes.npy'): os.remove(D7 + '/train_q_hashes.npy')
for f in ('fresh_qa.train.bin', 'fresh_qa.val.bin', 'airoom.train.bin', 'airoom.val.bin'):
    if not os.path.exists(DATA + '/' + f): os.symlink(D7 + '/' + f, DATA + '/' + f)
stamp('fresh data: ' + str({f: os.path.getsize(D7 + '/' + f) for f in NEED}))

# ---- 4. BEFORE scores on the unseen sets
SCORE = [sys.executable, '-u', CODE + '/s7_tools.py', 'score', '--code', CODE, '--fmt', D7 + '/format.json', '--airoom', AIROOM]
ITEMS = {'qa': D7 + '/eval_unseen.jsonl', 'ar': D7 + '/eval_airoom.jsonl'}
for tag in ('qa', 'ar'):
    if not os.path.exists(RD + '/score_before_%s.json' % tag):
        stamp('BEFORE (%s): s6 best on unseen questions' % tag)
        sh(*SCORE, '--items', ITEMS[tag], '--ckpt', CK + '/ckpt_best.pt', '--out', RD + '/score_before_%s.json' % tag)
BEFORE = {t: json.load(open(RD + '/score_before_%s.json' % t)) for t in ('qa', 'ar')}
for t, b in BEFORE.items(): stamp('BEFORE %s %d/%d %s' % (t, b['total'], b['n'], dict(b['by_kind'])))

# ---- 5. config (same as Colab s7) + step 0 from the s6 best with fresh AdamW
for s in cfg['data']['sources']: s['weight'] = round(s['weight'] * 0.65, 4)
cfg['data']['sources'].insert(0, {'name': 'fresh_qa', 'train': 'fresh_qa.train.bin', 'weight': 0.15})
cfg['data']['sources'].insert(0, {'name': 'airoom', 'train': 'airoom.train.bin', 'weight': 0.20})
cfg['extra_val'] = {**cfg.get('extra_val', {}), 'fresh_qa': 'fresh_qa.val.bin', 'airoom': 'airoom.val.bin'}
AFTER_RESERVE = 75 * 60   # after-scores (600 items) + export + Kaggle saving the output
cfg.update(run_dir=RD, device='cuda', amp=AMP, adapter=CODE + '/trainer/adapters/mnx1_custom_s6.py',
           stop_after_sec=int(left() - AFTER_RESERVE), ckpt_every_sec=600, eval_at_start=True, early_stop_patience=4,
           _note='s7 (Kaggle T4): start from s6 best; + airoom 0.20 + fresh_qa 0.15 (new seeds, no question shared with training data or the unseen eval sets)')
cfg['data']['root'] = DATA
json.dump(cfg, open(RD + '/cfg.json', 'w'), indent=1)
stamp('lr %s max_steps %s stop_after_sec %s mix %s' % (cfg['lr'], cfg['max_steps'], cfg['stop_after_sec'], [(s['name'], s['weight']) for s in cfg['data']['sources']]))
import importlib.util
sp = importlib.util.spec_from_file_location('ad', cfg['adapter']); ad = importlib.util.module_from_spec(sp); sp.loader.exec_module(ad)
ck = torch.load(CK + '/ckpt_best.pt', map_location='cpu', weights_only=False); fs = ck.get('state') or {}
stamp('[START] ' + START_FROM + ' best ' + str({k: fs.get(k) for k in ('step', 'tokens', 'best', 'best_step')}))
m = ad.build_model({**cfg['model_cfg'], 'grad_ckpt': True}); m.load_state_dict(ck['model'])
osd = ad.configure_optimizer(m, cfg).state_dict()
for pg in osd['param_groups']: pg['fused'] = True
st = {'step': 0, 'tokens': 0, 'best': float('inf'), 'noimp': 0, 'from': START_FROM + ' best, fresh AdamW', 'from_state': fs}
torch.save({'model': m.state_dict(), 'opt': osd, 'scaler': None, 'state': st, 'config': cfg}, RD + '/ckpt_latest.pt'); del ck, m, osd
os.remove(CK + '/ckpt_best.pt')

# ---- 6. train
os.environ.setdefault('OMP_NUM_THREADS', '2')
r = subprocess.run([sys.executable, '-u', CODE + '/trainer/train_s6.py', '--config', RD + '/cfg.json'])
stamp('train exit %d' % r.returncode)

# ---- 7. AFTER + verdict
best = RD + '/ckpt_best.pt'; assert os.path.exists(best), 'no s7 ckpt_best.pt'
bst = dict(torch.load(best, map_location='cpu', weights_only=False, mmap=True)['state'])
AFTER = {}
for tag in ('qa', 'ar'):
    sh(*SCORE, '--items', ITEMS[tag], '--ckpt', best, '--out', RD + '/score_after_%s.json' % tag)
    AFTER[tag] = json.load(open(RD + '/score_after_%s.json' % tag))
lines = ['session 7 verdict (Kaggle T4, started from the %s best) ' % START_FROM + time.strftime('%F %T'), 's7 best = step %s, held-out loss %s' % (bst.get('best_step'), bst.get('best'))]
for tag, name in (('qa', 'usual-style questions'), ('ar', 'Ai-room questions')):
    B, A = BEFORE[tag], AFTER[tag]
    lines.append('%s: start (%s best) %d/%d  ->  s7 best %d/%d' % (name, START_FROM, B['total'], B['n'], A['total'], A['n']))
    for k in sorted(set(A['by_kind']) | set(B['by_kind'])):
        b_, a_ = B['by_kind'].get(k, [0, 0]), A['by_kind'].get(k, [0, 0]); lines.append('  %-12s %3d/%-3d -> %3d/%-3d' % (k, b_[1], b_[0], a_[1], a_[0]))
tb, ta = sum(BEFORE[t]['total'] for t in BEFORE), sum(AFTER[t]['total'] for t in AFTER)
lines.append('combined: %d -> %d' % (tb, ta))
KEEP = ta > tb and AFTER['qa']['total'] >= BEFORE['qa']['total'] - 9
if KEEP:
    lines.append('KEEP: s7 is better. Model-only copy in runs/s7/export_ckpt_best.pt (use this one from now on).')
    ck = torch.load(best, map_location='cpu', weights_only=False, mmap=True)
    torch.save({'model': ck['model'], 'state': ck['state'], 'config': ck['config']}, RD + '/export_ckpt_best.pt'); del ck
    lines.append('export sha256 ' + subprocess.run(['sha256sum', RD + '/export_ckpt_best.pt'], capture_output=True, text=True).stdout.split()[0])
else:
    lines.append('DO NOT USE s7: not better overall, or it lost more than 9 usual-style answers. Keep using the ' + START_FROM + ' best.')
open(RD + '/VERDICT.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines), flush=True)
# keep the kernel output small enough to save: the full ckpts (with AdamW) are only needed to resume
if os.path.exists(RD + '/ckpt_latest.pt'): os.remove(RD + '/ckpt_latest.pt')
if KEEP: os.remove(best)
for f in ('fresh_qa.train.bin', 'airoom.train.bin'):
    if os.path.exists(D7 + '/' + f): os.remove(D7 + '/' + f)
subprocess.run('ls -la %s %s; tail -3 %s/log.jsonl' % (RD, D7, RD), shell=True)
stamp('done')
