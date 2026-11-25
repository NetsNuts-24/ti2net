# TI2Net reproduction — temporal branch of a multimodal deepfake detector

## What this project is

We are building a multimodal deepfake detection model. This folder covers **only the temporal part**.
The reference paper is TI2Net (Liu et al., WACV 2023): `Liu_TI2Net_..._WACV_2023_paper.pdf` in this folder.
Official code: https://github.com/BaopingLiu/TI2Net

## Working order (agreed, do not skip ahead)

1. **Stage 1 — Reproduce TI2Net.** ← current stage
2. Stage 2 — Diagnose where it fails (by manipulation type, pose, motion, quality).
3. Stage 3 — Beat it with simple upgrades.
4. Stage 4 — Only then design the new paper.

Do not propose or build new-paper ideas while Stage 1 is open. Earlier novelty-check results are parked
in `docs/novelty_check_2026-11-23.json` for Stage 4.

## Rules for working here

- **Every decision goes in `DECISIONS.md`** at the moment it is made: what, why, alternatives rejected.
  Never change a decided item silently; add a new entry that supersedes the old one.
- Every deviation from the paper or from the official code is a decision and must be logged.
- Report numbers as "paper", "official code as-is" or "our run"; never mix them.
- Fix random seeds and save the train/test split to disk for every run.
- Ask before downloading datasets or model weights (state source and size).

## Paper targets (video-level AUC %, trained on FF++)

| FF++ | DFD | DeeperForensics | CDF1 | CDF2 |
|---|---|---|---|---|
| 99.95 | 72.03 | 76.08 | 66.65 | 68.22 |

## What the official code actually does (read 2026-11-23)

Pipeline: dlib face crop → ArcFace ResNet-18 (no SE, grayscale 128×128, flipped image concatenated → 1024-D)
→ ID vectors saved offline as `.npy` (+ a lengths file per dataset) → `np.diff` → model.

- Model (`models/ti2net.py`): input dropout 0.5 (LRNet's LandmarkDropout) → **bidirectional** GRU,
  1 layer, hidden 512 → last time step → dropout 0.7 → Linear(1024→128) → ReLU → Linear(128→2) → Softmax.
- Training (`train.py`): Adam, lr 1e-5, batch size 1 (one triplet), 100 epochs,
  loss = CrossEntropy + 0.1 × triplet (margin 0).
- Sampling (`utils/dataset.py`): 64 frames at a fixed stride of 2 from a random start → 63 differences.
- Authors released FF++ ID vectors (Google Drive) and ArcFace weights. No trained TI2Net weights.
- FF++ videos the authors flag as bad: 281, 344, 356, 370, 492, 509, 738, 908, 950.

### Mismatches between code and paper (each needs a decision before we train)

1. Paper says random frame sampling; code uses fixed stride 2.
2. Paper has the anchor-positive term `L_ap`; in code it is commented out. Triplet margin is 0.
3. Softmax is applied before `CrossEntropyLoss`, so softmax is applied twice.
4. `Pair_Dataset.__getitem__` uses `len(fake) % len(real)` as the anchor / test-real index, which is a
   constant: the same single real video is used every time.
5. Split is a random 75/25 `train_test_split` with no seed, done separately for real and fake, not the
   official FF++ split. The same source identity can land in train and test.
6. The saved checkpoint is chosen by best accuracy on the test set.
7. Triplet loss receives `embeddings[0], [1], [2]`, which only works with batch size 1.
8. `train.py` imports TensorFlow only to detect a GPU; requirements pin torch 1.12.1 / Python 3.7 era.

## Current state of Stage 1

- First run is **R0 = official code as-is** (D007). No fixes to the mismatches above in R0.
- R0 uses **our own** FF++ **c23** vectors (D012, D015, D022): extraction finished 2026-11-25 on Kaggle,
  5,000 videos, 2,337,870 vectors (D023). Merge + checks: `kaggle/merge_and_check.ipynb`.
- Training happens on the faculty machine (2 × RTX 5090), needs a current PyTorch (D024).
- R0 is started with `src/r0/run_r0.py`, which runs the official files unmodified (D025); see
  `src/r0/README.md`. Merge checked (D027); `df/638_640` (91 frames) is left out with `--drop_short`.
  R0 runs the official 100 epochs.
- R1 = corrected version, `src/r1/train_r1.py` (D028), runs in the same trip as R0. R0 and R1 numbers are
  not comparable (different test sets); variants are compared against R1.
- No cross-dataset data yet; cross-dataset tests come after R0.
- R0 command sequence (from `demo_offline_id.ipynb`): merge the four fake sets (first 250 videos of
  each type) into `fake_ids.npy` / `fake_lens.npy` → `python train.py --model=... --real_path=...
  --real_len_path=... --fake_path=... --fake_len_path=...` → `python evaluate.py ...`.

## Layout

- `TI2Net/` — clone of the official repo (commit 4702844), kept unmodified as reference.
- `src/` — our reproduction code.
- `data/`, `id_vectors/`, `weights/` — not committed.
- `docs/` — notes and parked research.
- `DECISIONS.md` — decision log.
