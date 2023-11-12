#!/usr/bin/env bash

path='/home/matt/fourier'
device='cuda:0'
seed=0
terms_mlp=5
terms_gnn=20
gnn_width=0.25

cd $path
source env/bin/activate

python -u -W ignore::UserWarning code/models/mnist_aug_cnn.py $path $device $seed --skip_train
python -u -W ignore::UserWarning code/models/mnist_aug_mispredict_cnn.py $path $device $seed --skip_train
python -u -W ignore::UserWarning code/models/mnist_baseline_cnn.py $path $device $seed --skip_train
python -u -W ignore::UserWarning code/models/mnist_fourier_cnn.py $path $device $seed --skip_train
python -u -W ignore::UserWarning code/models/mnist_fourier_mlp.py $path $device $seed $terms_mlp --skip_train

python -u -W ignore::UserWarning code/models/mnist_noise_aug_cnn.py $path $device $seed --skip_train
python -u -W ignore::UserWarning code/models/mnist_noise_baseline_cnn.py $path $device $seed --skip_train
python -u -W ignore::UserWarning code/models/mnist_noise_fourier_mlp.py $path $device $seed $terms_mlp --skip_train

python -u -W ignore::UserWarning code/models/qd-3_aug_cnn.py $path $device $seed --skip_train
python -u -W ignore::UserWarning code/models/qd-3_aug_mispredict_cnn.py $path $device $seed --skip_train
python -u -W ignore::UserWarning code/models/qd-3_baseline_cnn.py $path $device $seed --skip_train
python -u -W ignore::UserWarning code/models/qd-3_fourier_cnn.py $path $device $seed --skip_train
python -u -W ignore::UserWarning code/models/qd-3_fourier_mlp.py $path $device $seed $terms_mlp --skip_train

python -u -W ignore::UserWarning code/models/qd-345_aug_cnn.py $path $device $seed --skip_train
python -u -W ignore::UserWarning code/models/qd-345_aug_mispredict_cnn.py $path $device $seed --skip_train
python -u -W ignore::UserWarning code/models/qd-345_baseline_cnn.py $path $device $seed --skip_train
python -u -W ignore::UserWarning code/models/qd-345_fourier_cnn_largest.py $path $device $seed --skip_train
python -u -W ignore::UserWarning code/models/qd-345_fourier_gnn.py $path $device $seed $terms_gnn $gnn_width --deep --skip_conn --skip_train