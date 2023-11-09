#!/usr/bin/env bash

device='cuda:0'
seed=0
terms_gnn=20

proj_root="$(dirname "$(readlink -f "$0")")"
cd $proj_root

python -u -W ignore::UserWarning -m code.experiments.qd345_fourier_cnn_largest $device $seed --skip_test | tee logs/ablations/qd345_fourier_cnn_largest_s$seed.txt
python -u -W ignore::UserWarning -m code.experiments.qd345_fourier_cnn_avg $device $seed --skip_test | tee logs/ablations/qd345_fourier_cnn_avg_s$seed.txt

for gnn_width in 0.125 0.25 0.5
do
    python -u -W ignore::UserWarning -m code.experiments.qd345_fourier_gnn $device $seed $terms_gnn $gnn_width --skip_conn --skip_test | tee logs/ablations/qd345_fourier_gnn_N${terms_gnn}_w${gnn_width}_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd345_fourier_gnn $device $seed $terms_gnn $gnn_width --deep --skip_conn --skip_test | tee logs/ablations/qd345_fourier_gnn_N${terms_gnn}_w${gnn_width}_deep_s$seed.txt
done

for terms_mlp in 5 10 20
do
    python -u -W ignore::UserWarning -m code.experiments.mnist_fourier_mlp $device $seed $terms_mlp --skip_test | tee logs/ablations/mnist_fourier_mlp_N${terms_mlp}_s$seed.txt
    python -u -W ignore::UserWarning -m code.experiments.qd3_fourier_mlp $device $seed $terms_mlp --skip_test | tee logs/ablations/qd3_fourier_mlp_N${terms_mlp}_s$seed.txt
done