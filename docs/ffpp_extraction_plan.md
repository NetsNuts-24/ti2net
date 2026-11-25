# FF++ → ID vectors: our own extraction

Goal: produce the files `train.py` needs (`real_ids.npy`, `real_lens.npy`, `fake_ids.npy`, `fake_lens.npy`)
from FaceForensics++ ourselves.

## Step 1 — Get FF++ access (start today; approval can take days)

1. Open https://github.com/ondyari/FaceForensics and follow the "Access" section: fill in the Google
   form with your institutional email.
2. You receive a download script by email (`faceforensics_download_v4.py`).
3. Download videos per subset (run on the machine that will do the extraction):

   ```
   python faceforensics_download_v4.py <out_dir> -d original       -c <raw|c23> -t videos
   python faceforensics_download_v4.py <out_dir> -d Deepfakes      -c <raw|c23> -t videos
   python faceforensics_download_v4.py <out_dir> -d Face2Face      -c <raw|c23> -t videos
   python faceforensics_download_v4.py <out_dir> -d FaceSwap       -c <raw|c23> -t videos
   python faceforensics_download_v4.py <out_dir> -d NeuralTextures -c <raw|c23> -t videos
   ```

   Approximate size for these five subsets (from the FF++ README, to be confirmed at download time):
   raw is in the hundreds of GB; c23 is on the order of 10–20 GB.

## Step 2 — What the authors' pipeline does (from their code)

1. `utils/video2image_cropping.py`: for **every frame**, dlib frontal face detector (upsample 1),
   square box enlarged ×1.3 around the face centre, crop saved as PNG. Every detected face is saved.
2. `utils/libs.py: load_image`: read crop as grayscale → resize to 128×128 → stack with its
   left-right flip → scale to [-1, 1].
3. ArcFace ResNet-18 (`pretrained/resnet18_110.pth`, checked: loads with all keys matched, 512-D output)
   → 512-D for the image + 512-D for the flip → 1024-D per frame.
4. All frames of all videos stacked into one array, plus a lengths array.

Authors' released length files: 1,000 real videos / 481,034 frames; 4,000 fake / 1,874,871 frames.
About 2.36 million face crops in total.

## Step 3 — Costs and risks

- **Speed:** dlib's detector runs on CPU, on every frame, about 2.36 M frames. Expect days on one CPU
  core; it must be run in parallel across cores.
- **Disk:** saving every crop as PNG adds a large amount on top of the videos.
- **Unknowns in the authors' pipeline** (asked in the email): frame order (their code does not sort),
  a possible unpublished alignment step, and which face is kept when several are detected.

## Step 4 — Proposed extraction script (to be written once Step-1 choices are made)

One script, video → ID vectors directly, with these choices (each to be logged in DECISIONS.md):

- Same detector, same ×1.3 crop, same grayscale/128×128/flip/ArcFace steps as the authors.
- Frames kept in temporal order.
- Largest face per frame when several are detected; frames with no face are skipped and counted.
- No PNGs written by default (optional flag to save them); one `.npy` per video so a crash loses
  nothing, then merged into the four files `train.py` reads.
- Per-video log: frames read, faces found, frames skipped. Videos the authors flag as bad
  (281, 344, 356, 370, 492, 509, 738, 908, 950) reported separately.
- Parallel over CPU cores for detection; GPU only for ArcFace.

## Reference: the authors' own training log (in `demo_offline_id.ipynb`)

3,000 training and 1,000 test iterations per epoch (all 4,000 fakes, 75/25 split).
Best in the visible log: sequence accuracy 0.941, video AUC 0.9945 (around epoch 25–40).
This is the number R0 should land near on FF++.
