#!/usr/bin/env bash

device='cuda:0'
terms_mlp=5
terms_gnn=20
gnn_width=0.25

proj_root="$(dirname "$(readlink -f "$0")")"
cd $proj_root

for seed in 0 1 2 3 4
do
    python -u -W ignore::UserWarning -m code.experiments.mnist_baseline_cnn $device $seed | tee logs/mnist_baseline_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.mnist_aug_cnn $device $seed | tee logs/mnist_aug_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.mnist_mispredict_cnn $device $seed | tee logs/mnist_mispredict_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.mnist_fourier_cnn $device $seed | tee logs/mnist_fourier_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.mnist_fourier_mlp $device $seed $terms_mlp | tee logs/mnist_fourier_mlp_N${terms_mlp}_s$seed.txt

    python -u -W ignore::UserWarning -m code.experiments.qd3_baseline_cnn $device $seed | tee logs/qd3_baseline_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd3_aug_cnn $device $seed | tee logs/qd3_aug_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd3_mispredict_cnn $device $seed | tee logs/qd3_mispredict_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd3_fourier_cnn $device $seed | tee logs/qd3_fourier_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd3_fourier_mlp $device $seed $terms_mlp | tee logs/qd3_fourier_mlp_N${terms_mlp}_s$seed.txt

    python -u -W ignore::UserWarning -m code.experiments.qd345_baseline_cnn $device $seed | tee logs/qd345_baseline_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd345_aug_cnn $device $seed | tee logs/qd345_aug_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd345_mispredict_cnn $device $seed | tee logs/qd345_mispredict_cnn_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd345_fourier_cnn_largest $device $seed | tee logs/qd345_fourier_cnn_largest_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd345_fourier_gnn $device $seed $terms_gnn $gnn_width --deep --skip_conn | tee logs/qd345_fourier_gnn_N${terms_gnn}_w${gnn_width}_deep_s$seed.txt
done