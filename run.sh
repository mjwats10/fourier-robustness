#!/usr/bin/env bash

proj_root="$(dirname "$(readlink -f "$0")")"
device='cuda:0'

cd $proj_root

for seed in 0 1 2 3 4
do
    python -u -W ignore::UserWarning -m code.experiments.mnist_baseline_cnn $device $seed | tee logs/mnist_baseline_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.mnist_aug_cnn $device $seed | tee logs/mnist_aug_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.mnist_mispredict_cnn $device $seed | tee logs/mnist_mispredict_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.mnist_fourier_cnn $device $seed | tee logs/mnist_fourier_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.mnist_fourier_mlp $device $seed 5 | tee logs/mnist_fourier_mlp_s$seed.txt

    python -u -W ignore::UserWarning -m code.experiments.qd3_baseline_cnn $device $seed | tee logs/qd3_baseline_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd3_aug_cnn $device $seed | tee logs/qd3_aug_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd3_mispredict_cnn $device $seed | tee logs/qd3_mispredict_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd3_fourier_cnn $device $seed | tee logs/qd3_fourier_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd3_fourier_mlp $device $seed 5 | tee logs/qd3_fourier_mlp_s$seed.txt

    python -u -W ignore::UserWarning -m code.experiments.qd345_baseline_cnn $device $seed | tee logs/qd345_baseline_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd345_aug_cnn $device $seed | tee logs/qd345_aug_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd345_mispredict_cnn $device $seed | tee logs/qd345_mispredict_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd345_fourier_cnn_largest $device $seed | tee logs/qd345_fourier_cnn_largest_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd345_fourier_gnn $device $seed 20 0.25 --deep --skip_conn | tee logs/qd345_fourier_gnn_s$seed.txt
done