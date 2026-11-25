# R1: corrected TI2Net (our code)

`train_r1.py` is our re-implementation: same model and training settings as the official code, with
the protocol problems fixed (D028). It reads the same merged files as R0, plus `real_names.txt` and
`fake_names.txt` from the same folder. Same environment as R0 (`src/r0/README.md`).

## Runs (from the `Ti2net` folder)

| Run | What it answers | Command (add `--data <folder>`) |
|---|---|---|
| R1 | honest baseline on FF++ | `python src/r1/train_r1.py --name r1_seed0 --seed 0 --gpu 1` |
| R1-random | paper's random frame sampling instead of stride 2 | `... --name r1_random --sampling random` |
| R1-lap | paper's anchor-positive loss term | `... --name r1_lap --lap_weight 0.5` |
| R1-no-X | unseen manipulation: train on three types, read `video_auc_X` | `... --name r1_no_nt --train_types df ff fs` |

Switches `--fixed_anchor` and `--double_softmax` bring back single official behaviours.

## Output (`runs/<name>/`)

`train_log.txt`, `epochs.jsonl` (one line per epoch), `split.json` (video names per split),
`best.pth` (best validation video AUC), `last.pth`, `result.json` and `test_scores.csv`
(test split, scored once with `best.pth`).

R0 and R1 numbers are **not comparable**: R0's test set is the official code's (random split, one real
video). Compare variants only against R1.
