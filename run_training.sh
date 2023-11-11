#!/usr/bin/env bash

path='/home/matt/fourier'
device='cuda:0'

cd $path
source env/bin/activate

for seed in 0 1 2 3 4
do
    python -u -W ignore::UserWarning code/models/mnist_baseline_cnn.py $path $device $seed | tee logs/mnist_baseline_cnn_s$seed.txt
    python -u -W ignore::UserWarning code/models/mnist_aug_cnn.py $path $device $seed | tee logs/mnist_aug_cnn_s$seed.txt
    python -u -W ignore::UserWarning code/models/mnist_aug_mispredict_cnn.py $path $device $seed | tee logs/mnist_aug_mispredict_cnn_s$seed.txt
    python -u -W ignore::UserWarning code/models/mnist_fourier_cnn.py $path $device $seed | tee logs/mnist_fourier_cnn_s$seed.txt
    python -u -W ignore::UserWarning code/models/mnist_fourier_mlp.py $path $device $seed 5 | tee logs/mnist_fourier_mlp_s$seed.txt

    python -u -W ignore::UserWarning code/models/qd-3_baseline_cnn.py $path $device $seed | tee logs/qd-3_baseline_cnn_s$seed.txt
    python -u -W ignore::UserWarning code/models/qd-3_aug_cnn.py $path $device $seed | tee logs/qd-3_aug_cnn_s$seed.txt
    python -u -W ignore::UserWarning code/models/qd-3_aug_mispredict_cnn.py $path $device $seed | tee logs/qd-3_aug_mispredict_cnn_s$seed.txt
    python -u -W ignore::UserWarning code/models/qd-3_fourier_cnn.py $path $device $seed | tee logs/qd-3_fourier_cnn_s$seed.txt
    python -u -W ignore::UserWarning code/models/qd-3_fourier_mlp.py $path $device $seed 5 | tee logs/qd-3_fourier_mlp_s$seed.txt

    python -u -W ignore::UserWarning code/models/qd-345_baseline_cnn.py $path $device $seed | tee logs/qd-345_baseline_cnn_s$seed.txt
    python -u -W ignore::UserWarning code/models/qd-345_aug_cnn.py $path $device $seed | tee logs/qd-345_aug_cnn_s$seed.txt
    python -u -W ignore::UserWarning code/models/qd-345_aug_mispredict_cnn.py $path $device $seed | tee logs/qd-345_aug_mispredict_cnn_s$seed.txt
    python -u -W ignore::UserWarning code/models/qd-345_fourier_cnn_largest.py $path $device $seed | tee logs/qd-345_fourier_cnn_largest_s$seed.txt
    python -u -W ignore::UserWarning code/models/qd-345_fourier_gnn.py $path $device $seed 20 0.25 --deep --skip_conn | tee logs/qd-345_fourier_gnn_s$seed.txt
done