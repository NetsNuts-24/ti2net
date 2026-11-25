"""Check the extraction output and merge it into the four files train.py reads.
Same checks as kaggle/merge_and_check.ipynb, for running on any machine (needs only numpy).

  python src/check_merge.py --input /data/ti2net/halves --out /data/ti2net

--input: a folder that contains (at any depth) the unzipped extraction outputs, i.e. two or more
         .../id_vectors/FFpp_c23 folders, and optionally authors_released/real_lens.npy.
--out:   the merged files are written to <out>/FFpp_c23/.
"""
import argparse, os
_p = argparse.ArgumentParser()
_p.add_argument("--input", required=True)
_p.add_argument("--out", required=True)
_a = _p.parse_args()
os.environ["TI2_INPUT"] = os.path.abspath(_a.input)
os.environ["TI2_WORK"] = os.path.abspath(_a.out)

import csv, glob, os, subprocess, sys
import numpy as np

ROOT = os.environ.get('TI2_INPUT', '/kaggle/input')
WORK = os.environ.get('TI2_WORK', '/kaggle/working')
HALVES = sorted(glob.glob(ROOT + '/**/id_vectors/FFpp_c23', recursive=True))
print('extraction outputs attached:')
for h in HALVES:
    print('  ', h)
assert len(HALVES) >= 2, 'attach BOTH extraction outputs (half 0/2 and half 1/2) as inputs'
auth = glob.glob(ROOT + '/**/authors_released/real_lens.npy', recursive=True)
AUTH = os.path.dirname(auth[0]) if auth else None
print('authors length files:', AUTH)
OUT = os.path.join(WORK, 'FFpp_c23')
SUBSETS = ['real', 'df', 'ff', 'fs', 'nt']
MIN_FRAMES = 131   # the official dataset code needs more than 130 usable frames per video
BAD = ['281', '344', '356', '370', '492', '509', '738', '908', '950']   # flagged by the authors
os.makedirs(OUT, exist_ok=True)

report = []
def say(*a):
    line = ' '.join(str(x) for x in a)
    print(line); report.append(line)

ok = True
ours = {}
for s in SUBSETS:
    files, logs = {}, {}
    for h in HALVES:
        for f in glob.glob(os.path.join(h, s, '*.npy')):
            if f.endswith('.tmp.npy'):
                continue
            name = os.path.splitext(os.path.basename(f))[0]
            assert name not in files, f'{s}/{name} is in two halves'
            files[name] = f
        lp = os.path.join(h, s, '_log.csv')
        if os.path.exists(lp):
            for r in csv.DictReader(open(lp)):
                logs[r['video']] = {k: int(r[k]) for k in ['frames', 'faces', 'no_face', 'multi_face', 'fallback']}
    names = sorted(files)
    lens = np.array([np.load(files[n], mmap_mode='r').shape[0] for n in names])
    dims = {int(np.load(files[n], mmap_mode='r').shape[1]) for n in names}
    ours[s] = (names, lens)
    targets = sorted({n.split('_')[0] for n in names})
    tot = {k: sum(logs[n][k] for n in names if n in logs) for k in ['frames', 'faces', 'no_face', 'multi_face', 'fallback']}
    mism = [n for n, l in zip(names, lens) if n in logs and logs[n]['faces'] != l]
    short = [(n, int(l)) for n, l in zip(names, lens) if l < MIN_FRAMES]
    say(f'[{s}] videos {len(names)} | vectors {int(lens.sum())} | dim {dims} | {tot}')
    say(f'[{s}] shortest {int(lens.min())}, longest {int(lens.max())}, videos under {MIN_FRAMES} frames: {len(short)} {short[:30]}')
    worst = sorted(((logs[n]['no_face'], n) for n in names if n in logs), reverse=True)[:8]
    say(f'[{s}] most frames without a face (count, video): {worst}')
    bad = [(n, int(l), logs.get(n, {}).get('no_face')) for n, l in zip(names, lens) if n.split('_')[0] in BAD]
    say(f'[{s}] videos the authors flag as bad (name, vectors, no_face): {bad}')
    checks = {'1000 videos': len(names) == 1000, 'targets are 000..999': targets == [f'{i:03d}' for i in range(1000)],
              'vector size 1024': dims == {1024}, 'log has every video': all(n in logs for n in names),
              'file length = log faces': not mism}
    for k, v in checks.items():
        if not v:
            ok = False
            say(f'[{s}] CHECK FAILED: {k} {mism[:10] if k.startswith("file") else ""}')
say('ALL STRUCTURE CHECKS PASSED' if ok else 'SOME CHECKS FAILED - do not train on this yet')

# Comparison with the authors' released length files (they used raw FF++ and kept every detected face).
if AUTH:
    ar = np.load(os.path.join(AUTH, 'real_lens.npy')); af = np.load(os.path.join(AUTH, 'fake_lens.npy'))
    auth = {'real': ar, 'df': af[0:1000], 'ff': af[1000:2000], 'fs': af[2000:3000], 'nt': af[3000:4000]}
    for s in SUBSETS:
        names, lens = ours[s]
        a = auth[s]
        say(f'[{s}] total: ours {int(lens.sum())}, authors {int(a.sum())}, ours/authors {lens.sum() / a.sum():.4f} | '
            f'median: ours {int(np.median(lens))}, authors {int(np.median(a))} | min: ours {int(lens.min())}, authors {int(a.min())}')
        if len(lens) == len(a):
            d = lens - a   # only meaningful if the authors' order is the sorted order (true for real and df at the start)
            say(f'[{s}] same position: equal {int((d == 0).sum())}, within 5 frames {int((np.abs(d) <= 5).sum())}, '
                f'within 5% {int((np.abs(d) <= 0.05 * a).sum())} of {len(a)}')
            big = np.argsort(-np.abs(d))[:10]
            say(f'[{s}] largest gaps (video, ours, authors): {[(names[i], int(lens[i]), int(a[i])) for i in big]}')
            say(f'[{s}] same lengths ignoring order: within 5 frames {int((np.abs(np.sort(lens) - np.sort(a)) <= 5).sum())} of {len(a)}')
else:
    say('authors length files not attached: comparison skipped')

subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'merge_ids.py'), '--in_dirs', *HALVES, '--out_dir', OUT,
                '--real', 'real', '--fake', 'df', 'ff', 'fs', 'nt', '--min_len', '0'], check=True)

# Spot check: 40 random videos read back from the merged files must equal the per-video files.
rng = np.random.default_rng(0)
for tag, subs in [('real', ['real']), ('fake', ['df', 'ff', 'fs', 'nt'])]:
    ids = np.load(os.path.join(OUT, f'{tag}_ids.npy'), mmap_mode='r')
    lens = np.load(os.path.join(OUT, f'{tag}_lens.npy'))
    names = open(os.path.join(OUT, f'{tag}_names.txt')).read().split('\n')
    assert len(names) == len(lens) and lens.sum() == len(ids)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    for i in rng.choice(len(lens), 40, replace=False):
        s, n = names[i].split('/')
        f = [p for h in HALVES for p in [os.path.join(h, s, n + '.npy')] if os.path.exists(p)][0]
        assert np.array_equal(ids[starts[i]:starts[i] + lens[i]], np.load(f)), names[i]
    say(f'{tag}: merged {len(lens)} videos, {len(ids)} vectors, finite {bool(np.isfinite(ids[::997]).all())}, spot check passed')
    del ids
open(os.path.join(OUT, 'check_report.txt'), 'w').write('\n'.join(report))
for f in sorted(os.listdir(OUT)):
    print(f, f'{os.path.getsize(os.path.join(OUT, f)) / 1e9:.2f} GB')
