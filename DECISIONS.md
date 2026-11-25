# Decision log

Newest entries at the bottom. Never edit a past decision; add a new entry that supersedes it.

Format: **ID — date — title** / Decision / Why / Alternatives rejected / Status.

---

## Decided

### D001 — 2026-11-23 — Reference paper
- **Decision:** TI2Net (Liu et al., WACV 2023) is the reference for the temporal part.
- **Why:** chosen by the team as the starting point.
- **Status:** active.

### D002 — 2026-11-23 — Scope is the temporal part only
- **Decision:** work here covers only the temporal branch of the multimodal detector. Audio, fusion and
  other branches are out of scope for now.
- **Why:** one branch at a time.
- **Status:** active.

### D003 — 2026-11-23 — Order of work: reproduce → diagnose → beat → new paper
- **Decision:** first reproduce TI2Net, then analyse its failures, then beat it, and only then plan a new paper.
- **Why:** we need our own baseline numbers, and the failures we measure should choose the upgrade,
  not ideas from reading alone.
- **Alternatives rejected:** picking the new-paper idea first (was started too early on 2026-11-23).
- **Status:** active. Stage 1 in progress.

### D004 — 2026-11-23 — Park the novelty check until Stage 4
- **Decision:** the literature/novelty results (ideas A, B, D, F, I) are saved in
  `docs/novelty_check_2026-11-23.json` and not acted on now.
- **Why:** follows D003. Summary for later: only idea A (identity drift not explained by pose,
  expression and quality) looked new; B, F, D, I overlap heavily with published work.
- **Status:** parked.

### D005 — 2026-11-23 — Use the official repo as the implementation reference
- **Decision:** reproduce from https://github.com/BaopingLiu/TI2Net, not from the paper text alone.
- **Why:** official code exists and fixes most hyperparameters the paper leaves out.
- **Status:** active.

### D006 — 2026-11-23 — Keep CLAUDE.md and this decision log
- **Decision:** `CLAUDE.md` holds project context and rules; `DECISIONS.md` records every decision
  and every deviation from paper or code.
- **Status:** active.

### D007 — 2026-11-23 — First run is R0: official code as-is (closes O1)
- **Decision:** run the official code unchanged first ("R0"). Corrected versions come later and each
  correction will be its own logged decision.
- **Why:** establishes what the authors' code gives before we change anything.
- **Consequence:** R0 keeps the code's behaviour for O4–O9 (fixed stride 2, random unseeded 75/25 split,
  constant anchor index, CE + 0.1·triplet with margin 0, double softmax, checkpoint picked on test accuracy).
  Only changes allowed in R0 are ones needed to make it run (environment, paths); each gets logged.
- **Status:** active.

### D008 — 2026-11-23 — Hardware: faculty GPU machine over remote desktop (closes O11)
- **Decision:** training runs on a GPU computer provided by faculty, accessed by remote desktop.
  Code is prepared on this Windows machine and moved over.
- **Open detail:** GPU model, OS, and whether Python/CUDA are already installed there are not known yet.
- **Status:** active.

### D009 — 2026-11-23 — Use the authors' released FF++ ID vectors for R0 (closes O2 for R0)
- **Decision:** R0 trains on the ID vectors the authors published on Google Drive; the user downloads
  them into `TI2Net/id_vectors/FFpp/`.
- **Why:** no datasets are downloaded yet (O12 answer: none), and this removes face cropping and
  ArcFace extraction as sources of difference.
- **Note:** the authors' extraction notebook reads FF++ from `.../raw/aligned_faces`, so the released
  vectors are from **raw** (uncompressed) FF++. Cross-dataset tests (CDF, DFD, DeeperForensics) still
  need those datasets and our own extraction later.
- **Status:** active.

### D010 — 2026-11-23 — Official repo cloned, kept unmodified
- **Decision:** cloned to `F:\Ti2net\TI2Net` at commit `4702844`. This copy stays as the reference;
  changes needed to run go in a separate copy or branch so the diff is visible.
- **Status:** active.

### D011 — 2026-11-23 — Finding: released ID vectors are incomplete (D009 blocked)
- **Finding:** the authors' Google Drive folder `id_vectors` contains only `real_lens.npy` (1,000 videos,
  481,034 frames) and `fake_lens.npy` (4,000 videos, 1,874,871 frames). The vector files
  `real_ids.npy` / `fake_ids.npy` (about 2 GB and 7.7 GB if float32) are not there. Checked in the
  browser on 2026-11-23.
- **Effect:** D009 cannot be carried out as written. R0 needs the vectors from somewhere: either the
  authors send them, or we extract them ourselves from FF++ raw (needs FF++ access, dlib cropping and
  the authors' ArcFace weights).
- **Also found:** GitHub issue #2 on the official repo reports `ValueError: low >= high` in training.
  Cause: `sample_seq` needs more than (64+1)×2 = 130 frames, and the shortest videos have 100. R0 will
  need a minimal fix for short videos to run at all; to be logged when chosen.
- **Status:** open; waiting on the user's choice of route (see O13).

### D012 — 2026-11-23 — Get ID vectors by our own extraction; ask authors in parallel (closes O13, supersedes D009)
- **Decision:** we extract FF++ ID vectors ourselves and do not wait for the authors. The user also
  sends an email and posts a GitHub issue asking for the missing files (drafts in `docs/`).
- **Why:** time is limited and the authors have not answered earlier GitHub issues.
- **Consequence:** R0 is no longer "authors' vectors + authors' code"; it is "our extraction with the
  authors' pipeline + authors' training code". Extraction choices must be logged.
- **Status:** active.

### D013 — 2026-11-23 — ArcFace weights verified and files placed
- **Finding:** `resnet18_110.pth` (102.6 MB, from the authors' Drive) loads into the repo's
  `resnet_face18` with all keys matched; 1×128×128 input gives a 512-D output.
- **Placement:** weights in `TI2Net/pretrained/`; authors' length files in
  `TI2Net/id_vectors/authors_released/` (kept for comparing frame counts with our extraction).
- **Status:** done.

### D014 — 2026-11-23 — R0 target on FF++ taken from the authors' own log
- **Finding:** the training log saved in `demo_offline_id.ipynb` shows 3,000 train / 1,000 test
  iterations per epoch and a best video AUC of 0.9945 (sequence accuracy 0.941).
- **Decision:** R0 counts as reproduced on FF++ if it lands near this, alongside the paper's 99.95.
- **Status:** active.

### D015 — 2026-11-23 — FF++ c23 for our extraction (closes O14)
- **Decision:** extract from FF++ **c23**, not raw.
- **Why:** time is limited; c23 is a small fraction of the download and is what later papers use.
- **Deviation from the authors:** they used raw. R0 numbers are therefore "official training code on
  c23 vectors" and may come out lower than the authors' raw numbers. Must be stated wherever R0 is reported.
- **Status:** active.

### D016 — 2026-11-23 — Extraction script and its choices (closes O15, O16 provisionally)
- **Decision:** `src/extract_ids.py` (video → 1024-D vectors, one `.npy` per video, resumable, parallel
  face detection) and `src/merge_ids.py` (builds the four files `train.py` reads).
- **Same as the authors:** dlib frontal detector with upsample 1, ×1.3 square crop, grayscale,
  128×128, flip concatenation, ArcFace ResNet-18 weights `resnet18_110.pth`.
- **Our choices** (authors' behaviour unknown; adopted from the recommendations, can be overridden):
  frames in temporal order; largest face when several are detected; frames with no face skipped;
  crops not written to disk.
- **Verified on this PC (CPU):** for the same crop, our vectors equal the authors' `load_image` +
  ArcFace path to 1e-7. This needed an in-memory PNG round trip, because `cv2.imread(path, 0)`
  converts to grayscale ±1 level differently from `cvtColor`. Merge output passes `train.py`'s
  length check.
- **Not verified:** the dlib detection and video-reading part (dlib is not installed on this PC and no
  videos are available yet). First test must happen on a few real videos.
- **Status:** active.

### D017 — 2026-11-23 — Finding: the official evaluation does not measure what the paper reports
Read from `utils/dataset.py`, `utils/metric.py`, `train.py` and the authors' saved log.
- **One real video only.** The real index is `len(fake) % len(real)`: 3000 % 750 = 0 in training and
  1000 % 250 = 0 in testing. Training always uses real video 0 as the anchor; testing compares windows
  of **one** real video against 1,000 fake videos.
- **"Video-level" = sequence-level.** `evaluate` groups scores in a dict keyed by a torch tensor;
  tensors hash by object identity, so no two samples are ever grouped. The authors' own log shows
  video accuracy identical to sequence accuracy every epoch (0.562/0.562, 0.829/0.829, …).
- **Other items already listed in CLAUDE.md:** unseeded random split done separately for real and fake,
  checkpoint chosen on the test set, double softmax.
- **Effect:** an R0 number near the authors' 0.9945 confirms the code runs as theirs did, but it is not
  a trustworthy FF++ result. The corrected run (R1) is the baseline we will actually compare against.
- **Unknown:** whether the paper's tables were produced with exactly this code.
- **Status:** finding; R1 corrections to be decided one by one.

### D018 — 2026-11-23 — Finding: Kaggle copy of FF++ c23 is usable as the video source
- **Source checked in the browser:** https://www.kaggle.com/datasets/xdxd003/ff-c23 (17.92 GB, one version,
  about 124K downloads, 195 public notebooks use it).
- **Contents:** says it was downloaded with the original FF++ scripts at c23. Folders `original`,
  `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `FaceShifter`, `DeepFakeDetection`
  (1,000 mp4 each) and `csv` (10 metadata files). The first five are what R0 needs.
- **Limits:** unofficial copy; the page links the official FF++ licence and terms of use, which still
  apply (the user has submitted the official access form). `DeepFakeDetection` holds 1,000 fake videos
  only, so it is not the full DFD test set. Integrity to be checked after download: 1,000 videos per
  folder and frame counts against the authors' length files.
- **Status:** recommended; waiting for the user to choose where to download/run (O19).

### D019 — 2026-11-23 — Extraction runs on Kaggle, from the Kaggle FF++ c23 copy (closes O19)
- **Decision:** run `extract_ids.py` in a Kaggle notebook with `xdxd003/ff-c23` attached; download
  only the resulting vector files. Training still goes to the faculty GPU machine.
- **Why:** starts today, no 18 GB download, does not depend on the faculty machine.
- **Prepared:** `kaggle/ti2net-assets.zip` (scripts, repo `models/`, ArcFace weights, authors' length
  files; 95 MB) to upload as a private Kaggle dataset, and `kaggle/extract_ffpp.ipynb`.
- **Script additions:** `--limit` (test on N videos), `--shard K/N` (split across parallel sessions),
  `--skip_dirs` (resume from an earlier session's output). Logic tested locally with a stub detector.
- **Order:** Step 1 is a 4-video timing test; the full run is started only after its output is checked
  (speed, frame counts vs the authors' first four real videos, vector shape).
- **Risk:** dlib on every frame is slow on Kaggle's few cores and sessions stop at 12 h; the full run
  will likely need several resumed or parallel sessions. No speed-up that changes the crops will be
  made without a logged decision.
- **Status:** active; waiting for the Step 1 output.

### D020 — 2026-11-23 — Finding: Kaggle Step 1 result; exact detection is too slow
- **Works:** dlib 20.0.1 installs; extraction runs on GPU T4 with 4 CPU cores; vectors are
  (frames, 1024) float32, all finite.
- **Matches the authors:** our face counts for real videos 000–003 are 396, 460, 693, 303, identical
  to the first four entries of the authors' `real_lens.npy`; no frame without a face, none with two.
  So the Kaggle copy and our frame reading agree with theirs on these videos.
- **Too slow:** 3.9 frames/s (3 workers, includes start-up) → about 169 hours for 2.36 M frames,
  roughly 14 full 12-hour Kaggle sessions.
- **Response:** added an optional faster detection mode to `extract_ids.py`
  (`--detect_height H --upsample U`: run dlib on a copy of the frame shrunk to height H, scale the box
  back; crop, grayscale, resize and ArcFace unchanged). Default stays the authors' setting.
  `kaggle/extract_ffpp_v2.ipynb` Step 1b measures speed, missed faces and vector agreement with the
  exact setting on real videos 000, 250, 500, 750.
- **Status:** waiting for Step 1b numbers; the detection setting for the full run is O20.

### D021 — 2026-11-23 — Finding: faster detection keeps every face but changes the temporal signal
Step 1b on real videos 000 (480p), 250 (720p), 500 and 750 (1080p), 1,693 frames:

| setting | frames/s | faces missed | vector cosine vs exact (mean / 1st pct / min) | cosine of frame-to-frame differences |
|---|---|---|---|---|
| exact (authors) | 2.3 | 0 | 1 / 1 / 1 | 1 (0.9976 printed; identical frames give 0) |
| height 480, upsample 1 | 7.8 | 0 | 0.959 / 0.884 / 0.754 | 0.63 |
| height 480, upsample 0 | 23.8 | 0 | 0.950 / 0.863 / 0.821 | 0.51 |
| height 360, upsample 0 | 34.9 | 0 | 0.945 / 0.886 / 0.828 | 0.50 |

- **Reading:** the per-frame vectors stay similar (≈0.95), but the differences between frames — the
  only thing TI2Net's GRU sees — agree only about half. A few pixels of change in the face box moves
  the ArcFace vector by about as much as the real frame-to-frame change does. So TI2Net's input is
  strongly shaped by face-box jitter. (Worth keeping for Stage 2.)
- **Exact speed on this mix is 2.3 frames/s** (1080p videos are the slow ones): the full run would be
  roughly 180–280 hours on one Kaggle session at a time.
- **Not yet known:** whether the faster settings add *more* jitter than exact or just *different*
  jitter. A one-cell check of difference sizes on the saved comparison files answers this.
- **Status:** O20 still open, pending that check.

### D022 — 2026-11-23 — Detection setting for the full run: height 360, no upsample, with exact fallback (closes O20)
- **Evidence:** size of frame-to-frame change relative to vector size, per video (000, 250, 500, 750):
  exact 0.292 / 0.358 / 0.341 / 0.377; h480_up1 0.292 / 0.363 / 0.326 / 0.362;
  h480_up0 0.316 / 0.351 / 0.357 / 0.366; h360_up0 0.289 / 0.370 / 0.336 / 0.375.
  All settings are within a few percent of exact: the faster settings give a different realisation of
  the face-box jitter, not more of it. None missed a face on these videos.
- **Proposal:** `--detect_height 360 --upsample 0` (34.9 frames/s measured, about 19 h for all of FF++,
  vs 180–280 h exact). If the shrunk frame yields no face, that frame is retried with the authors'
  exact setting; such frames are counted per video in the `fallback` column of `_log.csv`.
- **Deviation from the authors (must be reported with R0):** face boxes differ by a few pixels from
  theirs, so our vectors are not theirs frame for frame (difference-vector cosine about 0.5), although
  statistically equivalent on the four test videos. Evidence base is small: 4 real videos, no fakes.
- **Run mechanics:** `kaggle/extract_ffpp_full.ipynb`; `SHARD` splits the videos across parallel
  sessions; each session stops cleanly after 10.5 h so Kaggle saves its output; later sessions skip
  finished videos. Stop/resume logic tested locally with a stub detector.
- **Status:** active. Approved by the user on 2026-11-23 (height 360, upsample 0, exact fallback).

### D023 — 2026-11-25 — FF++ c23 extraction finished (our run); join-and-check notebook written
- **Result (from the two Kaggle logs, setting D022):** both halves completed all 5 × 500 videos,
  no time-outs. 8.48 h + 10.31 h = 18.8 GPU-hours.

| subset | videos | frames | vectors (faces) | no face | several faces | exact fallback |
|---|---|---|---|---|---|---|
| real | 1000 | 509,128 | 509,035 | 93 | 11,789 | 14,739 |
| df | 1000 | 509,128 | 507,897 | 1,231 | 12,165 | 17,112 |
| ff | 1000 | 509,128 | 508,966 | 162 | 11,824 | 15,646 |
| fs | 1000 | 406,140 | 406,024 | 116 | 8,970 | 12,168 |
| nt | 1000 | 406,137 | 405,948 | 189 | 8,895 | 12,809 |

  Total 2,337,870 vectors (about 9.6 GB as float32). Frames without a face: 0.08 %. Fallback to the
  authors' exact detector: about 3 % of frames. Several faces in about 2.3 % of frames (largest kept, D016).
- **Against the authors' length files:** df 508,451 / ff 508,006 / fs 405,560 — ours within 0.2 %.
  Their real (481,034; 75 videos under 131 frames, shortest 100) and nt (452,854; shortest 100) totals
  do not match ours, and their nt total is larger than the number of frames FF++ has for that
  subset. Not explained yet; the per-video comparison in the check notebook will show where.
- **To verify on Kaggle (not visible in the logs):** the Face2Face half 0 has a different frame total
  (254,018) from real/df half 0 (251,615) although the full totals are equal, so its file order differs
  from the others. The check notebook confirms every subset has exactly targets 000–999.
- **Tooling:** `src/merge_ids.py` now takes several input folders (`--in_dirs`), refuses duplicates and
  writes the big array straight to disk. `kaggle/merge_and_check.ipynb` (CPU only) runs the structure
  checks, the comparison with the authors' lengths, lists videos under 131 frames (input for O17),
  merges all 4,000 fakes in the order df, ff, fs, nt, and spot-checks the merged files. Tested locally
  on dummy data only.
- **Status:** extraction done; merge pending on Kaggle.

### D024 — 2026-11-25 — R0 trains on the faculty machine (2 × RTX 5090 Founders Edition)
- **Decision (user):** training runs on the faculty machine, not on Kaggle. Confirms D008 with the
  hardware now known.
- **Consequence:** the RTX 5090 needs a recent PyTorch (CUDA 12.8 build), so the pinned
  torch 1.12.1 / Python 3.7 environment cannot be used. This closes O10 (current versions).
- **Rejected:** Kaggle (vectors are already there, but weekly GPU quota; 18.8 h already used).
- **Still unknown:** operating system, Python, internet access and free disk on that machine.
- **Data transfer:** the merged files (about 9.6 GB) are downloaded on that machine from the Kaggle
  merge notebook's output.

### D025 — 2026-11-25 — R0 is run through a launcher; official files stay unmodified
- **What:** `src/r0/run_r0.py` imports the official `train.py` / `evaluate.py` as they are (clone still
  clean at `4702844`) and adds only the following. Setup notes: `src/r0/README.md`.
  1. **TensorFlow stand-in.** `train.py` imports TensorFlow only to ask whether a GPU exists; the
     launcher answers from PyTorch. Device is `cuda:0` (official) or `cpu` (official string "CPU" is
     invalid in PyTorch).
  2. **Seeds fixed** (python, numpy, torch); `--seed`, default 0. The official code has none.
  3. **Split recorded**: the same random 75/25 `train_test_split`, with video positions saved to
     `runs/<name>/split.json`. Positions refer to the merged files (`*_names.txt` gives the names).
  4. **Log file**, `meta.json`, `curves.json`; `weights/torch/` is created.
  5. **Triplet loss with margin 0.** Current PyTorch raises an error when `nn.TripletMarginLoss` is
     built with margin 0 (the authors' PyTorch 1.12.1 allowed it). The launcher computes the same loss
     through `torch.triplet_margin_loss(..., margin=0, p=2, eps=1e-6, mean)`, which is the function
     `nn.TripletMarginLoss` itself calls. Checked: equals max(d(a,p) − d(a,n) + m, 0) and equals
     `nn.TripletMarginLoss` for a positive margin.
  6. **Short videos (O17, still open):** the launcher counts videos under 131 frames and stops if there
     are any; `--drop_short` leaves them out. Whether any exist is known after the merge check.
- **Not changed:** model, optimiser, learning rate, batch size 1, 100 epochs, sampling, loss weights,
  the constant real index, double softmax, checkpoint choice on the test set, evaluation (D017).
- **Tested:** end to end on this PC (CPU, torch 2.9.1) with dummy vectors: train 2 epochs, save, eval.
  Not yet run on a GPU or on the real vectors.
- **Deviation to report with R0:** PyTorch version (current vs 1.12.1); results are labelled "our run".

### D026 — 2026-11-25 — Faculty machine facts; environment for R0
- **Machine (from `nvidia-smi`, user screenshot):** Linux with conda (`base`), 2 × RTX 5090 (32 GB each),
  driver 595.99.02, both GPUs idle. System Python is 3.8.5.
- **Decision:** a separate conda environment `ti2net` with Python 3.11 and the current PyTorch CUDA 12.8
  build. Python 3.8.5 is too old for any PyTorch that supports the RTX 5090. The `base` environment
  is not touched.
- **Tooling:** `src/check_merge.py` = the same checks and merge as `kaggle/merge_and_check.ipynb`, runnable
  on any machine, so the two halves can be merged on the faculty machine directly instead of on Kaggle.
  Tested on dummy data only.
- **Not verified:** that the install works on that machine; internet access and free disk there.

### D027 — 2026-11-25 — Merge done and checked (our run); one short video found
- **Result** (`kaggle/merge_and_check.ipynb`, log saved in `docs/logs/merge_2026-11-25.log`): all structure
  checks passed for the five subsets (1000 videos each, targets 000–999, 1024-D, file lengths equal the
  extraction log). Merged files: real 1000 videos / 509,035 vectors (2.09 GB); fake 4000 videos /
  1,828,835 vectors (7.49 GB), order df, ff, fs, nt. Spot check of 40 + 40 videos passed.
- **Against the authors' length files:**
  - real: 840 of 1000 videos have exactly the authors' frame count at the same position. Their file
    ends with many entries of exactly 100 (e.g. 950, 951, 978, 982, 986, 989, 992), which looks like
    truncation on their side; that explains their lower total (481,034 vs our 509,035).
  - df / ff / fs: the authors' order is not the sorted order, but as sets of lengths 992 / 978 / 997 of
    1000 agree within 5 frames.
  - nt: does not match (98 of 1000). Their nt entries have original-video lengths (up to 1814) and 100
    entries of exactly 100, while FF++ c23 NeuralTextures videos are shorter. Not explained; it is a
    property of their file, not checkable further without their data.
- **Short videos (O17):** exactly one video has fewer than 131 vectors: `df/638_640` with 91 (607 of its
  698 frames have no detected face). Every other video has at least 193.
- **Proposal for O17:** leave this one video out (`--drop_short`), i.e. R0 uses 3,999 fakes. Rejected:
  shortening the sequence length (changes the method for all videos).
  **Status:** approved by the user on 2026-11-25. Closes O17. R0 runs with `--drop_short`
  (3,999 fakes); to be reported as a deviation with R0.
- **Launcher change:** `split.json` now stores positions in the merged files (rows of `*_lens.npy`,
  lines of `*_names.txt`) even when a video is dropped. Re-tested on dummy data.

### D028 — 2026-11-25 — Corrected version (R1) is prepared before the first training trip; runs next to R0
- **Decision (user):** make the code changes for the flaws we found here first, then run on the faculty
  machine, so one trip shows whether our changes bring anything. Changes the plan in D007 (R0 alone
  first); R0 itself is unchanged and still runs as-is (D025).
- **What was built:** `src/r1/train_r1.py` (our code; official repo still untouched), `src/r1/README.md`.
  Same model, initialisation, Adam 1e-5, batch of one triplet, 100 epochs, CE + 0.1 × triplet (margin 0),
  stride-2 sampling, steps per epoch = number of training fakes. Differences from the official code:

| Open item | Official | R1 |
|---|---|---|
| O5 split | random 75/25, unseeded, identities shared | official FF++ lists, 720 / 140 / 140 videos (files in `src/r1/splits/`, from github.com/ondyari/FaceForensics) |
| O6 anchor / test real | always the same one real video | random real training video; all real videos of the split are tested |
| D017 evaluation | "video-level" = sequence-level; random windows | fixed windows (64 frames, stride 2, every 64 frames), mean score per video; AUC per fake type as well |
| O9 checkpoint | best accuracy on the test set | best video AUC on validation; test scored once |
| O8 softmax | applied twice | CrossEntropy on logits |
| O4 sampling | stride 2 | stride 2 by default; `--sampling random` as a variant |
| O7 L_ap | off | off by default; `--lap_weight` as a variant |

- **Other choices:** videos under 131 frames left out as in R0 (D027). `--train_types` allows training
  on some fake types only (unseen-manipulation test without new data). `--fixed_anchor` and
  `--double_softmax` restore single official behaviours so each fix can be measured.
- **Closes:** O4–O9 for R1 (variants cover the "run both" items).
- **Caution for reading results:** R0 and R1 are not comparable (different test sets; R0's has one real
  video). FF++ in-dataset results are near 99 % in the paper, so improvements will mostly be visible in
  the unseen-manipulation runs and later on other datasets, not in the plain FF++ number.
- **Tested:** on this PC (CPU) with dummy data using real FF++ names: split, training, validation
  selection, test, all switches. Not yet run on the real vectors or on a GPU.
- **O19** (novelty changes in the first trip): closed by D029, none.

### D029 — 2026-11-25 — First training trip: fixes and paper variants only (closes O19)
- **Decision (user):** no new-paper ideas in the first trip. D003's order stands.
- **Runs** (`src/run_trip1.sh`, seed 0, 100 epochs each, four in sequence per GPU):
  GPU 0: `r0_seed0` (official as-is), `r1_random` (random sampling), `r1_no_df`, `r1_no_ff`;
  GPU 1: `r1_seed0` (corrected baseline), `r1_lap` (anchor-positive term, weight 0.5 as in the
  commented-out official line), `r1_no_fs`, `r1_no_nt` (each leaves one fake type out of training).
- **If one run fails,** the later runs on that GPU still start (each is its own command).
- **Not verified:** the script has only been syntax-checked; it has not run on the machine.

### D030 — 2026-11-25 — What goes into our git repo
- **Decision:** commit code, notebooks, docs, `DECISIONS.md`, `CLAUDE.md` and the saved splits
  (`src/r1/splits/*.json`). Keep out: ID vectors (`*.npy`, `merged_id_vectors.zip`, 9.5 GB), weights
  (`*.pth`, incl. the 98 MB ArcFace file, over GitHub's 100 MB-ish limit with history), the Kaggle bundle
  `kaggle/ti2net-assets*` (a copy of the official repo + weights), run outputs, and the paper PDF
  (publisher copy).
- **Official repo:** `TI2Net/` is added as a **git submodule** pinned at commit 4702844, so the exact
  reference code is recorded without copying it. Its local extras (`TI2Net/pretrained/`,
  `TI2Net/id_vectors/`) stay untracked inside the submodule; on a fresh clone the ArcFace weights must be
  fetched again from the authors' link.
- **Alternatives rejected:** vendoring the official files into our repo (duplicates code, loses the
  upstream commit link); Git LFS for vectors/weights (9.5 GB, LFS quota; data lives on Kaggle/the
  faculty machine instead).

---

## Open (need a decision before training)


| ID | Question | Options | Recommendation |
|---|---|---|---|
| O18 | How much time to spend on R0 | full 100 epochs / one short run for the record, then move to R1 | One short run, then R1 (see D017). |

Closed so far: O1 (→D007), O2 for R0 (→D009), O11 (→D008), O12 (answer: no datasets yet).

| ID | Question | Options | Recommendation |
|---|---|---|---|
| O1 | Which reproduction do we run? | (a) official code as-is; (b) corrected version; (c) both | (c): run as-is first as "R0", then a corrected "R1". The as-is code has protocol problems (see O4–O6) that can inflate FF++ numbers. |
| O2 | Where do ID vectors come from? | (a) authors' released FF++ vectors; (b) extract ourselves from video | (a) for the first FF++ run (no dataset or GPU preprocessing needed); (b) is required anyway for CDF/DFD/DeeperForensics. |
| O3 | Which FF++ compression for our own extraction? | raw / c23 / c40 | c23 (standard in later papers); paper does not say which it used. |
| O4 | Frame sampling | fixed stride 2 (code) / random ordered (paper) | Run both; the paper's Table 6 says it matters a lot. |
| O5 | Train/test split | random 75/25 unseeded (code) / official FF++ 720-140-140 split | Official split for R1, seeded and saved. |
| O6 | Anchor / test-real index bug (constant index) | keep / fix | Keep in R0, fix in R1. |
| O7 | Loss | CE + 0.1·triplet, margin 0, no L_ap (code) / add L_ap as in paper | Code version in R0; test L_ap in R1. |
| O8 | Double softmax before CrossEntropy | keep / remove | Keep in R0, remove in R1. |
| O9 | Checkpoint selection | best test accuracy (code) / validation set | Validation set in R1. |
| O10 | Environment | Python 3.7 + torch 1.12.1 (pinned) / current Python + torch | Current versions; model code is simple. TensorFlow import to be dropped. |
| O11 | Hardware | this Windows machine / a server | Needs your input. |
| O12 | Datasets available | FF++, Celeb-DF v1/v2, DFD, DeeperForensics | Needs your input. |
