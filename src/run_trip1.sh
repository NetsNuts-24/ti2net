#!/bin/bash
# First training trip (D029): 8 runs, 4 one after another on each GPU. Run from the Ti2net folder:
#   bash src/run_trip1.sh ~/ti2net_data/FFpp_c23
# It keeps running if the terminal or remote desktop is closed. Progress: tail -f runs/gpu0.out runs/gpu1.out
set -e
DATA="$1"
[ -f "$DATA/fake_ids.npy" ] || { echo "usage: bash src/run_trip1.sh <folder with the merged files>"; exit 1; }
mkdir -p runs
R1="python src/r1/train_r1.py --seed 0 --data $DATA"
nohup bash -c "
python src/r0/run_r0.py train --name r0_seed0 --seed 0 --gpu 0 --drop_short --data $DATA
$R1 --gpu 0 --name r1_random --sampling random
$R1 --gpu 0 --name r1_no_df --train_types ff fs nt
$R1 --gpu 0 --name r1_no_ff --train_types df fs nt
" > runs/gpu0.out 2>&1 &
nohup bash -c "
$R1 --gpu 1 --name r1_seed0
$R1 --gpu 1 --name r1_lap --lap_weight 0.5
$R1 --gpu 1 --name r1_no_fs --train_types df ff nt
$R1 --gpu 1 --name r1_no_nt --train_types df ff fs
" > runs/gpu1.out 2>&1 &
echo "started. watch with: tail -f runs/gpu0.out runs/gpu1.out"
