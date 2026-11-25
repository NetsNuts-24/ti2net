"""R0 launcher: runs the official TI2Net train.py / evaluate.py WITHOUT editing them.

The official files stay byte-for-byte as cloned. This launcher only does what is needed to run and
to keep records (each item is logged in DECISIONS.md, D025):
  1. TensorFlow is not installed: train.py imports it only to see if a GPU exists. A tiny stand-in
     module answers that question from PyTorch instead.
  2. Random seeds are fixed (python, numpy, torch).
  3. The random 75/25 split made by train.py is recorded to disk (same split, just written down).
  4. Everything printed is also written to a log file.
  5. ./weights/torch/ is created (train.py assumes it exists).
  6. Current PyTorch (needed for the RTX 5090) refuses to build nn.TripletMarginLoss with margin 0, which
     the official code asks for. The same loss is computed through PyTorch's underlying function, which
     has no such check: max(d(anchor, positive) - d(anchor, negative) + 0, 0), identical maths.
  7. Videos with fewer than 131 usable frames crash the official sampler (GitHub issue #2). The launcher
     stops and lists them, unless --drop_short is given, which leaves them out.

Train:
  python src/r0/run_r0.py train --name r0_seed0 --seed 0 --data D:/ti2net/FFpp_c23
Evaluate a saved model on a set of vectors:
  python src/r0/run_r0.py eval --name r0_seed0 --data D:/ti2net/FFpp_c23
"""
import argparse
import json
import os
import random
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "TI2Net"))
MIN_FRAMES = 131  # sample_seq needs len - (64 + 1) * 2 > 0


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def fake_tensorflow(has_gpu):
    """Stand-in for the three TensorFlow calls train.py makes at import time."""
    import importlib.machinery
    tf = types.ModuleType("tensorflow")
    tf.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)   # torch probes installed modules by spec
    tf.config = types.SimpleNamespace(
        list_physical_devices=lambda device_type=None: ["GPU:0"] if has_gpu else [],
        experimental=types.SimpleNamespace(set_memory_growth=lambda device=None, enable=True: None),
    )
    return tf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("--name", required=True, help="run name; weights go to TI2Net/weights/torch/<name>.pth")
    p.add_argument("--data", required=True, help="folder with real_ids.npy, real_lens.npy, fake_ids.npy, fake_lens.npy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100, help="official value is 100")
    p.add_argument("--gpu", default="0", help="which GPU to use (0 or 1)")
    p.add_argument("--drop_short", action="store_true", help=f"leave out videos with fewer than {MIN_FRAMES} frames")
    p.add_argument("--runs_dir", default=os.path.join(HERE, "..", "..", "runs"))
    a = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import numpy as np
    import torch

    data = os.path.abspath(a.data)
    files = {k: os.path.join(data, f"{k}.npy") for k in ["real_ids", "real_lens", "fake_ids", "fake_lens"]}
    for f in files.values():
        assert os.path.exists(f), f"missing {f}"
    run_dir = os.path.abspath(os.path.join(a.runs_dir, a.name))
    os.makedirs(run_dir, exist_ok=True)
    log = open(os.path.join(run_dir, f"{a.mode}_log.txt"), "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    sys.stderr = Tee(sys.__stderr__, log)

    has_gpu = torch.cuda.is_available()
    print(f"=== {a.mode} {a.name} | {time.strftime('%Y-%m-%d %H:%M:%S')} | torch {torch.__version__} | "
          f"gpu {torch.cuda.get_device_name(0) if has_gpu else 'none'} | seed {a.seed} | epochs {a.epochs}")

    # Short videos (GitHub issue #2).
    short = {}
    for tag in ["real", "fake"]:
        lens = np.load(files[f"{tag}_lens"])
        short[tag] = [int(i) for i in np.where(lens < MIN_FRAMES)[0]]
        print(f"{tag}: {len(lens)} videos, {int(lens.sum())} vectors, shorter than {MIN_FRAMES}: {len(short[tag])}")
    if (short["real"] or short["fake"]) and not a.drop_short:
        sys.exit(f"There are videos shorter than {MIN_FRAMES} frames (positions: {short}). The official sampler "
                 "crashes on them. Re-run with --drop_short to leave them out (decision O17).")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    os.chdir(REPO)                      # the official code uses paths relative to the repo
    sys.path.insert(0, REPO)
    os.makedirs("./weights/torch", exist_ok=True)
    sys.modules["tensorflow"] = fake_tensorflow(has_gpu)
    import train                         # official file, unmodified

    class TripletLossMargin0(torch.nn.Module):
        """What models/triplet_loss.py computes for a numeric margin, without the constructor check."""
        def __init__(self, margin=None):
            super().__init__()
            assert margin is not None, "official train.py always passes a number"
            self.margin = float(margin)
            print(f"Using TripletMarginLoss (margin {self.margin}, via torch.triplet_margin_loss)")

        def forward(self, anchor, pos, neg):
            # arguments: margin, p=2, eps=1e-6, swap=False, reduction=1 (mean): nn.TripletMarginLoss defaults
            return torch.triplet_margin_loss(anchor, pos, neg, self.margin, 2.0, 1e-6, False, 1)

    try:
        torch.nn.TripletMarginLoss(margin=0, p=2)
    except ValueError:
        print("this PyTorch rejects margin 0 in nn.TripletMarginLoss: using the equivalent direct call")
        train.TripletLoss = TripletLossMargin0

    orig_load = train.id_len_to_set
    kept = []

    def load_set(id_path, len_path):
        videos = orig_load(id_path, len_path)
        keep = [i for i, v in enumerate(videos) if len(v) >= MIN_FRAMES or not a.drop_short]
        kept.append(keep)                # positions in the merged files (real first, then fake)
        return [videos[i] for i in keep]

    train.id_len_to_set = load_set
    meta = dict(vars(a), data=data, torch=torch.__version__, short_videos=short,
                repo_commit="4702844", started=time.strftime("%Y-%m-%d %H:%M:%S"))

    if a.mode == "train":
        train.device = "cuda:0" if has_gpu else "cpu"   # official value on CPU is "CPU", which torch rejects
        train.EPOCHS = a.epochs
        orig_split = train.train_test_split
        split = {}

        def recorded_split(videos):
            # Same call as the official one, with the positions carried along so the split can be saved.
            tag = "real" if "real" not in split else "fake"
            pos = kept[0 if tag == "real" else 1]
            assert len(pos) == len(videos)
            tr, te, tr_idx, te_idx = orig_split(videos, pos)
            # positions are rows of <tag>_lens.npy / lines of <tag>_names.txt in the data folder
            split[tag] = {"train": [int(i) for i in tr_idx], "test": [int(i) for i in te_idx]}
            return tr, te

        train.train_test_split = recorded_split
        ap = argparse.ArgumentParser()
        for name in ["fake_selection", "branch", "model", "real_path", "real_len_path", "fake_path", "fake_len_path"]:
            ap.add_argument("--" + name, type=str)
        ap.add_argument("--train_contrastive", type=int, default=0)
        train.parse = ap                 # official main() reads this global
        sys.argv = ["train.py", "--fake_selection=all", f"--model={a.name}", "--train_contrastive=1",
                    f"--real_path={files['real_ids']}", f"--real_len_path={files['real_lens']}",
                    f"--fake_path={files['fake_ids']}", f"--fake_len_path={files['fake_lens']}"]

        orig_loop = train.train_loop

        def loop(*args, **kw):
            # the split exists by the time training starts: write it down first
            with open(os.path.join(run_dir, "split.json"), "w") as fh:
                json.dump(split, fh)
            with open(os.path.join(run_dir, "meta.json"), "w") as fh:
                json.dump(meta, fh, indent=1)
            out = orig_loop(*args, **kw)
            with open(os.path.join(run_dir, "curves.json"), "w") as fh:
                json.dump(out, fh)
            return out

        train.train_loop = loop
        train.main()
    else:
        import evaluate                  # official file, unmodified
        evaluate.main(a.name + ".pth", files["real_ids"], files["real_lens"], files["fake_ids"], files["fake_lens"])
    print(f"=== finished {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
