"""R1: TI2Net re-implemented with the protocol problems of the official code corrected.

Same model, optimiser, learning rate, loss weights and sampling as the official code. What differs
(each item is in DECISIONS.md, D028):
  - split: official FF++ train / val / test lists (720 / 140 / 140 videos), no identity in two splits
  - anchor: a random real training video each step (official: always the same single video)
  - evaluation: every real and every fake video of the split, fixed windows, one score per video
    (official: one real video only, and "video-level" was in fact sequence-level)
  - checkpoint: chosen on the validation split; the test split is scored once, with that checkpoint
  - loss: CrossEntropy on logits (official: softmax applied twice)
Switches bring back single official behaviours, or turn on variants, so each change can be measured.

  python src/r1/train_r1.py --name r1_seed0 --seed 0 --gpu 0 --data ~/ti2net_data/FFpp_c23
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ_LEN, STEP = 64, 2
MIN_FRAMES = 131          # same limit as the official sampler (and as R0's --drop_short)
TYPES = ["df", "ff", "fs", "nt"]


# ----------------------------------------------------------------------------- model
class LandmarkDropout(nn.Module):
    """Same as the official one (from LRNet): one mask over 512 positions, shared by all frames and by
    both halves (image / flipped image) of the 1024-D vector, scaled by 1/(1-p)."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        _, frames, dim = x.size()
        mask = torch.bernoulli(torch.full((dim // 2,), 1 - self.p)).repeat(1, frames, 2)
        return x * mask.to(x.device) / (1 - self.p)


class TI2Net(nn.Module):
    """Official architecture and initialisation; forward returns (embedding, logits)."""

    def __init__(self, feature_dim=1024, rnn_unit=512, dropout_rate=0.5, double_softmax=False):
        super().__init__()
        self.dropout_landmark = LandmarkDropout(dropout_rate)
        self.rnn = nn.GRU(feature_dim, rnn_unit, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout_feature = nn.Dropout(0.7)
        self.linear_1 = nn.Linear(2 * rnn_unit, 128)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(128, 2)
        self.double_softmax = double_softmax
        self.rnn.weight_ih_l0.data.normal_(0.0, 0.1)      # official: forward direction only
        self.rnn.weight_hh_l0.data.normal_(0.0, 0.1)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, x):
        x = self.dropout_landmark(x)
        out, _ = self.rnn(x)
        emb = out[:, -1, :]
        logits = self.linear_2(self.relu(self.linear_1(self.dropout_feature(emb))))
        if self.double_softmax:                            # official behaviour: softmax, then CrossEntropy
            logits = torch.softmax(logits, dim=1)
        return emb, logits

    def fake_prob(self, logits):
        return logits[:, 1] if self.double_softmax else torch.softmax(logits, dim=1)[:, 1]


# ----------------------------------------------------------------------------- data
def load_videos(data, tag):
    ids = np.load(os.path.join(data, f"{tag}_ids.npy"))
    lens = np.load(os.path.join(data, f"{tag}_lens.npy"))
    names = open(os.path.join(data, f"{tag}_names.txt")).read().split("\n")
    assert len(names) == len(lens) and int(lens.sum()) == len(ids), f"{tag}: files do not fit together"
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return [dict(name=n, pos=i, x=ids[s:s + l]) for i, (n, s, l) in enumerate(zip(names, starts, lens))]


def official_split(videos, split_dir):
    """FF++ lists hold pairs [a, b]. Real video a belongs to the split of its pair; so do fakes a_b and b_a."""
    where = {}
    for s in ["train", "val", "test"]:
        for a, b in json.load(open(os.path.join(split_dir, f"ffpp_{s}.json"))):
            assert a not in where and b not in where
            where[a] = where[b] = s
    out = {"train": [], "val": [], "test": []}
    for v in videos:
        ids = v["name"].split("/")[1].split("_")
        splits = {where[i] for i in ids}
        assert len(splits) == 1, f"{v['name']} mixes identities from different splits"
        out[splits.pop()].append(v)
    return out


def sample_train(x, mode, rng):
    """One training sequence of 64 frames -> 63 differences."""
    n = len(x)
    if mode == "stride2":                                  # official: fixed stride 2 from a random start
        start = rng.integers(0, n - (SEQ_LEN + 1) * STEP)
        idx = start + STEP * np.arange(SEQ_LEN)
    else:                                                  # "random": 64 random frames kept in time order
        idx = np.sort(rng.choice(n, SEQ_LEN, replace=False))
    return np.diff(x[idx], axis=0)


def eval_windows(x):
    """Fixed windows (stride 2, 64 frames), a new window every 64 frames, for repeatable evaluation."""
    span = (SEQ_LEN - 1) * STEP + 1
    starts = range(0, len(x) - span + 1, SEQ_LEN)
    return np.stack([np.diff(x[s + STEP * np.arange(SEQ_LEN)], axis=0) for s in starts])


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, videos, device, batch=512):
    """videos: list of dicts with x, label, type. Returns metrics and per-video scores."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    wins, owner = [], []
    for i, v in enumerate(videos):
        w = eval_windows(v["x"])
        wins.append(w)
        owner += [i] * len(w)
    wins, owner = np.concatenate(wins), np.array(owner)
    probs = np.concatenate([model.fake_prob(model(torch.from_numpy(wins[i:i + batch]).to(device))[1]).cpu().numpy()
                            for i in range(0, len(wins), batch)])
    model.train()
    labels = np.array([v["label"] for v in videos])
    types = np.array([v["type"] for v in videos])
    vid = np.array([probs[owner == i].mean() for i in range(len(videos))])
    m = {"seq_auc": roc_auc_score(labels[owner], probs), "seq_acc": float(((probs >= 0.5) == labels[owner]).mean()),
         "video_auc": roc_auc_score(labels, vid), "video_acc": float(((vid >= 0.5) == labels).mean()),
         "n_real": int((labels == 0).sum()), "n_fake": int((labels == 1).sum()), "n_seq": int(len(probs))}
    for t in TYPES:                                        # all real videos against one fake type
        keep = (labels == 0) | (types == t)
        if (types == t).any():
            m[f"video_auc_{t}"] = roc_auc_score(labels[keep], vid[keep])
    return m, vid


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--data", required=True, help="folder with real/fake _ids.npy, _lens.npy, _names.txt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="0")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=1, help="triplets per step (official: 1)")
    p.add_argument("--train_types", nargs="+", default=TYPES, choices=TYPES,
                   help="fake types used for training; the others are then unseen at test time")
    p.add_argument("--sampling", default="stride2", choices=["stride2", "random"])
    p.add_argument("--triplet_weight", type=float, default=0.1)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--lap_weight", type=float, default=0.0, help="weight of the anchor-positive distance (paper's L_ap)")
    p.add_argument("--fixed_anchor", action="store_true", help="official behaviour: always the first real video")
    p.add_argument("--double_softmax", action="store_true", help="official behaviour")
    p.add_argument("--split_dir", default=os.path.join(HERE, "splits"))
    p.add_argument("--runs_dir", default=os.path.join(HERE, "..", "..", "runs"))
    a = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    rng = np.random.default_rng(a.seed)

    run_dir = os.path.abspath(os.path.join(a.runs_dir, a.name))
    os.makedirs(run_dir, exist_ok=True)
    log = open(os.path.join(run_dir, "train_log.txt"), "a", encoding="utf-8")

    def say(*x):
        line = " ".join(str(i) for i in x)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"=== R1 {a.name} | {time.strftime('%Y-%m-%d %H:%M:%S')} | torch {torch.__version__} | "
        f"{torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}")
    say(json.dumps(vars(a)))

    data = os.path.abspath(a.data)
    split, dropped = {}, []
    for tag, label in [("real", 0), ("fake", 1)]:
        vids = load_videos(data, tag)
        for v in vids:
            v["label"], v["type"] = label, v["name"].split("/")[0]
        dropped += [v["name"] for v in vids if len(v["x"]) < MIN_FRAMES]
        vids = [v for v in vids if len(v["x"]) >= MIN_FRAMES]
        for s, lst in official_split(vids, a.split_dir).items():
            split[(s, tag)] = lst
            say(f"{s:5s} {tag}: {len(lst)} videos")
    say(f"left out (fewer than {MIN_FRAMES} frames): {dropped}")
    train_real = split[("train", "real")]
    train_fake = [v for v in split[("train", "fake")] if v["type"] in a.train_types]
    say(f"training fakes used: {len(train_fake)} ({a.train_types})")
    val = split[("val", "real")] + split[("val", "fake")]
    test = split[("test", "real")] + split[("test", "fake")]
    with open(os.path.join(run_dir, "split.json"), "w") as fh:
        json.dump({f"{s}_{tag}": [v["name"] for v in lst] for (s, tag), lst in split.items()}, fh)
    with open(os.path.join(run_dir, "meta.json"), "w") as fh:
        json.dump(dict(vars(a), data=data, torch=torch.__version__, dropped=dropped), fh, indent=1)

    model = TI2Net(double_softmax=a.double_softmax).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ce = nn.CrossEntropyLoss()
    steps = len(train_fake) // a.batch_size                # official: one step per training fake video
    best, best_epoch = -1.0, 0
    ckpt = os.path.join(run_dir, "best.pth")
    B = a.batch_size
    y = torch.tensor([0] * B + [0] * B + [1] * B, device=device)
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        tot = {"loss": 0.0, "ce": 0.0, "tri": 0.0, "acc": 0.0}
        for _ in range(steps):
            anc, pos, neg = [], [], []
            for _ in range(B):
                ai = 0 if a.fixed_anchor else int(rng.integers(len(train_real)))
                pi = int(rng.integers(len(train_real)))
                while pi == ai:                            # anchor and positive never the same video
                    pi = int(rng.integers(len(train_real)))
                ni = int(rng.integers(len(train_fake)))    # official: a random fake each step
                anc.append(sample_train(train_real[ai]["x"], a.sampling, rng))
                pos.append(sample_train(train_real[pi]["x"], a.sampling, rng))
                neg.append(sample_train(train_fake[ni]["x"], a.sampling, rng))
            x = torch.from_numpy(np.stack(anc + pos + neg)).to(device)
            emb, logits = model(x)
            ea, ep, en = emb[:B], emb[B:2 * B], emb[2 * B:]
            # arguments: margin, p=2, eps=1e-6, swap=False, reduction=mean (nn.TripletMarginLoss defaults)
            tri = torch.triplet_margin_loss(ea, ep, en, a.margin, 2.0, 1e-6, False, 1)
            l_ce = ce(logits, y)
            loss = l_ce + a.triplet_weight * tri
            if a.lap_weight:
                loss = loss + a.lap_weight * torch.pairwise_distance(ea, ep).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot["loss"] += loss.item()
            tot["ce"] += l_ce.item()
            tot["tri"] += tri.item()
            tot["acc"] += (logits.argmax(1) == y).float().mean().item()
        m, _ = evaluate(model, val, device)
        note = ""
        if m["video_auc"] > best:
            best, best_epoch = m["video_auc"], epoch
            torch.save(model.state_dict(), ckpt)
            note = " <- best on validation, saved"
        rec = dict(epoch=epoch, seconds=round(time.time() - t0, 1), **{k: v / steps for k, v in tot.items()},
                   **{"val_" + k: v for k, v in m.items()})
        with open(os.path.join(run_dir, "epochs.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        say(f"epoch {epoch:3d} | {rec['seconds']:.0f}s | loss {rec['loss']:.4f} ce {rec['ce']:.4f} tri {rec['tri']:.4f} "
            f"train_acc {rec['acc']:.4f} | val seq_auc {m['seq_auc']:.4f} video_auc {m['video_auc']:.4f} "
            f"video_acc {m['video_acc']:.4f}{note}")
    torch.save(model.state_dict(), os.path.join(run_dir, "last.pth"))

    model.load_state_dict(torch.load(ckpt, map_location=device))
    m, scores = evaluate(model, test, device)
    result = dict(name=a.name, best_val_video_auc=best, best_epoch=best_epoch, test=m)
    with open(os.path.join(run_dir, "result.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    with open(os.path.join(run_dir, "test_scores.csv"), "w") as fh:
        fh.write("video,label,score\n" + "\n".join(f"{v['name']},{v['label']},{s:.6f}" for v, s in zip(test, scores)))
    say(f"=== TEST (checkpoint of epoch {best_epoch}, chosen on validation): " + json.dumps({k: round(v, 4) for k, v in m.items()}))
    say(f"=== finished {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
