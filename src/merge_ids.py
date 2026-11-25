"""Merge per-video .npy files from extract_ids.py into the files TI2Net's train.py reads.

  python src/merge_ids.py --in_dirs half0/FFpp_c23 half1/FFpp_c23 --out_dir merged \
      --real real --fake df ff fs nt --min_len 0

--in_dirs: one or more extraction output folders (e.g. the two halves of a sharded run).
Writes real_ids.npy, real_lens.npy, fake_ids.npy, fake_lens.npy (+ *_names.txt) into --out_dir.
Videos are in sorted name order inside each subset; fake subsets are concatenated in the order given.
--min_len drops videos with fewer usable frames than this (0 keeps everything except empty videos).
The big array is written straight to disk, so memory use stays small.
"""
import argparse
import os
from glob import glob

import numpy as np


def collect(in_dirs, subset):
    """Sorted [(video_name, path)] for one subset across all input folders. A video found twice is an error."""
    found = {}
    for d in in_dirs:
        for f in glob(os.path.join(d, subset, "*.npy")):
            if f.endswith(".tmp.npy"):
                continue
            name = os.path.splitext(os.path.basename(f))[0]
            if name in found:
                raise SystemExit(f"{subset}/{name} is in two input folders: {found[name]} and {f}")
            found[name] = f
    return sorted(found.items())


def merge(in_dirs, out_dir, subsets, tag, min_len):
    keep, dropped = [], []
    for s in subsets:
        for name, f in collect(in_dirs, s):
            shape = np.load(f, mmap_mode="r").shape
            if shape[0] == 0 or shape[0] < min_len:
                dropped.append(f"{s}/{name}")
                continue
            keep.append((f"{s}/{name}", f, shape[0], shape[1]))
    assert keep, f"no videos found for {subsets} in {in_dirs}"
    dim = keep[0][3]
    assert all(k[3] == dim for k in keep), "vector sizes differ between videos"
    lens = np.array([k[2] for k in keep])
    os.makedirs(out_dir, exist_ok=True)
    out = np.lib.format.open_memmap(os.path.join(out_dir, f"{tag}_ids.npy"), mode="w+",
                                    dtype=np.float32, shape=(int(lens.sum()), dim))
    pos = 0
    for _, f, n, _ in keep:
        out[pos:pos + n] = np.load(f)
        pos += n
    out.flush()
    del out
    np.save(os.path.join(out_dir, f"{tag}_lens.npy"), lens)
    with open(os.path.join(out_dir, f"{tag}_names.txt"), "w") as fh:
        fh.write("\n".join(k[0] for k in keep))
    print(f"{tag}: {len(lens)} videos, {int(lens.sum())} frames, dim {dim}, dropped {len(dropped)} {dropped[:20]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dirs", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--real", nargs="+", default=["real"])
    p.add_argument("--fake", nargs="+", default=["df", "ff", "fs", "nt"])
    p.add_argument("--min_len", type=int, default=0)
    a = p.parse_args()
    merge(a.in_dirs, a.out_dir, a.real, "real", a.min_len)
    merge(a.in_dirs, a.out_dir, a.fake, "fake", a.min_len)
