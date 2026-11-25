# R0 on the faculty machine (2 × RTX 5090)

R0 = the official TI2Net training code, unmodified, on our FF++ c23 vectors. `run_r0.py` starts it and
keeps records; what it adds is listed at the top of that file and in `DECISIONS.md` (D025).

## 1. Copy to the machine

- The whole `Ti2net` folder (it is small: code, the `TI2Net` clone, notes).
- The merged vectors from Kaggle (output of `merge_and_check`): `real_ids.npy`, `real_lens.npy`,
  `fake_ids.npy`, `fake_lens.npy`, about 9.6 GB, into one folder, e.g. `D:\ti2net_data\FFpp_c23`.
  Download them on the faculty machine directly from the Kaggle notebook's Output tab.

## 2. Install (once)

    conda create -n ti2net python=3.11 -y      (the machine's own Python 3.8.5 is too old)
    conda activate ti2net
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install -r src/r0/requirements_r0.txt
    python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

The last line must print `True` and the RTX 5090.

## 3. Train (run from the `Ti2net` folder)

    python src/r0/run_r0.py train --name r0_seed0 --seed 0 --gpu 0 --drop_short --data D:\ti2net_data\FFpp_c23

- Official settings: 100 epochs, 3,000 training + 1,000 test steps per epoch.
- `--drop_short` leaves out the one video with fewer than 131 frames, `df/638_640` (D027).
- The second GPU can run another seed at the same time: `--name r0_seed1 --seed 1 --gpu 1 --drop_short`.

## 4. What comes out

- `runs/<name>/train_log.txt` — everything printed (send this back).
- `runs/<name>/split.json` — which videos were train / test. `meta.json` — settings. `curves.json`.
- `TI2Net/weights/torch/<name>.pth` — the model the official code keeps (best test accuracy).

The number to compare with the authors' own log is `best_video_auc` (theirs: 0.9945).
