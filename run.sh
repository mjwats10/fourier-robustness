#!/bin/bash

python -u -W ignore::UserWarning code/models/qd-345_fourier_cnn_avg.py /home/matt/fourier cuda:0 1 --resume | tee -a logs/qd-345_fourier_cnn_avg_s1.txt
python -u -W ignore::UserWarning code/models/qd-345_fourier_gnn.py /home/matt/fourier cuda:0 1 20 1.5 | tee logs/qd-345_fourier_gnn_s1.txt

python -u -W ignore::UserWarning code/models/qd-345_baseline_cnn.py /home/matt/fourier cuda:0 2 | tee logs/qd-345_baseline_cnn_s2.txt
python -u -W ignore::UserWarning code/models/qd-345_aug_cnn.py /home/matt/fourier cuda:0 2 | tee logs/qd-345_aug_cnn_s2.txt
python -u -W ignore::UserWarning code/models/qd-345_aug_mispredict_cnn.py /home/matt/fourier cuda:0 2 | tee logs/qd-345_aug_mispredict_cnn_s2.txt
python -u -W ignore::UserWarning code/models/qd-345_fourier_cnn_largest.py /home/matt/fourier cuda:0 2 | tee logs/qd-345_fourier_cnn_largest_s2.txt
python -u -W ignore::UserWarning code/models/qd-345_fourier_cnn_avg.py /home/matt/fourier cuda:0 2 | tee logs/qd-345_fourier_cnn_avg_s2.txt
python -u -W ignore::UserWarning code/models/qd-345_fourier_gnn.py /home/matt/fourier cuda:0 2 20 1.5 | tee logs/qd-345_fourier_gnn_s2.txt

python -u -W ignore::UserWarning code/models/qd-345_baseline_cnn.py /home/matt/fourier cuda:0 3 | tee logs/qd-345_baseline_cnn_s3.txt
python -u -W ignore::UserWarning code/models/qd-345_aug_cnn.py /home/matt/fourier cuda:0 3 | tee logs/qd-345_aug_cnn_s3.txt
python -u -W ignore::UserWarning code/models/qd-345_aug_mispredict_cnn.py /home/matt/fourier cuda:0 3 | tee logs/qd-345_aug_mispredict_cnn_s3.txt
python -u -W ignore::UserWarning code/models/qd-345_fourier_cnn_largest.py /home/matt/fourier cuda:0 3 | tee logs/qd-345_fourier_cnn_largest_s3.txt
python -u -W ignore::UserWarning code/models/qd-345_fourier_cnn_avg.py /home/matt/fourier cuda:0 3 | tee logs/qd-345_fourier_cnn_avg_s3.txt
python -u -W ignore::UserWarning code/models/qd-345_fourier_gnn.py /home/matt/fourier cuda:0 3 20 1.5 | tee logs/qd-345_fourier_gnn_s3.txt

python -u -W ignore::UserWarning code/models/qd-345_baseline_cnn.py /home/matt/fourier cuda:0 4 | tee logs/qd-345_baseline_cnn_s4.txt
python -u -W ignore::UserWarning code/models/qd-345_aug_cnn.py /home/matt/fourier cuda:0 4 | tee logs/qd-345_aug_cnn_s4.txt
python -u -W ignore::UserWarning code/models/qd-345_aug_mispredict_cnn.py /home/matt/fourier cuda:0 4 | tee logs/qd-345_aug_mispredict_cnn_s4.txt
python -u -W ignore::UserWarning code/models/qd-345_fourier_cnn_largest.py /home/matt/fourier cuda:0 4 | tee logs/qd-345_fourier_cnn_largest_s4.txt
python -u -W ignore::UserWarning code/models/qd-345_fourier_cnn_avg.py /home/matt/fourier cuda:0 4 | tee logs/qd-345_fourier_cnn_avg_s4.txt
python -u -W ignore::UserWarning code/models/qd-345_fourier_gnn.py /home/matt/fourier cuda:0 4 20 1.5 | tee logs/qd-345_fourier_gnn_s4.txt


python -u -W ignore::UserWarning code/models/qd-3_baseline_cnn.py /home/matt/fourier cuda:0 0 | tee logs/qd-3_baseline_cnn_s0.txt
python -u -W ignore::UserWarning code/models/qd-3_aug_cnn.py /home/matt/fourier cuda:0 0 | tee logs/qd-3_aug_cnn_s0.txt
python -u -W ignore::UserWarning code/models/qd-3_aug_mispredict_cnn.py /home/matt/fourier cuda:0 0 | tee logs/qd-3_aug_mispredict_cnn_s0.txt
python -u -W ignore::UserWarning code/models/qd-3_fourier_cnn.py /home/matt/fourier cuda:0 0 | tee logs/qd-3_fourier_cnn_s0.txt
python -u -W ignore::UserWarning code/models/qd-3_fourier_mlp.py /home/matt/fourier cuda:0 0 5 | tee logs/qd-3_fourier_mlp_s0.txt

python -u -W ignore::UserWarning code/models/qd-3_baseline_cnn.py /home/matt/fourier cuda:0 1 | tee logs/qd-3_baseline_cnn_s1.txt
python -u -W ignore::UserWarning code/models/qd-3_aug_cnn.py /home/matt/fourier cuda:0 1 | tee logs/qd-3_aug_cnn_s1.txt
python -u -W ignore::UserWarning code/models/qd-3_aug_mispredict_cnn.py /home/matt/fourier cuda:0 1 | tee logs/qd-3_aug_mispredict_cnn_s1.txt
python -u -W ignore::UserWarning code/models/qd-3_fourier_cnn.py /home/matt/fourier cuda:0 1 | tee logs/qd-3_fourier_cnn_s1.txt
python -u -W ignore::UserWarning code/models/qd-3_fourier_mlp.py /home/matt/fourier cuda:0 1 5 | tee logs/qd-3_fourier_mlp_s1.txt

python -u -W ignore::UserWarning code/models/qd-3_baseline_cnn.py /home/matt/fourier cuda:0 2 | tee logs/qd-3_baseline_cnn_s2.txt
python -u -W ignore::UserWarning code/models/qd-3_aug_cnn.py /home/matt/fourier cuda:0 2 | tee logs/qd-3_aug_cnn_s2.txt
python -u -W ignore::UserWarning code/models/qd-3_aug_mispredict_cnn.py /home/matt/fourier cuda:0 2 | tee logs/qd-3_aug_mispredict_cnn_s2.txt
python -u -W ignore::UserWarning code/models/qd-3_fourier_cnn.py /home/matt/fourier cuda:0 2 | tee logs/qd-3_fourier_cnn_s2.txt
python -u -W ignore::UserWarning code/models/qd-3_fourier_mlp.py /home/matt/fourier cuda:0 2 5 | tee logs/qd-3_fourier_mlp_s2.txt

python -u -W ignore::UserWarning code/models/qd-3_baseline_cnn.py /home/matt/fourier cuda:0 3 | tee logs/qd-3_baseline_cnn_s3.txt
python -u -W ignore::UserWarning code/models/qd-3_aug_cnn.py /home/matt/fourier cuda:0 3 | tee logs/qd-3_aug_cnn_s3.txt
python -u -W ignore::UserWarning code/models/qd-3_aug_mispredict_cnn.py /home/matt/fourier cuda:0 3 | tee logs/qd-3_aug_mispredict_cnn_s3.txt
python -u -W ignore::UserWarning code/models/qd-3_fourier_cnn.py /home/matt/fourier cuda:0 3 | tee logs/qd-3_fourier_cnn_s3.txt
python -u -W ignore::UserWarning code/models/qd-3_fourier_mlp.py /home/matt/fourier cuda:0 3 5 | tee logs/qd-3_fourier_mlp_s3.txt

python -u -W ignore::UserWarning code/models/qd-3_baseline_cnn.py /home/matt/fourier cuda:0 4 | tee logs/qd-3_baseline_cnn_s4.txt
python -u -W ignore::UserWarning code/models/qd-3_aug_cnn.py /home/matt/fourier cuda:0 4 | tee logs/qd-3_aug_cnn_s4.txt
python -u -W ignore::UserWarning code/models/qd-3_aug_mispredict_cnn.py /home/matt/fourier cuda:0 4 | tee logs/qd-3_aug_mispredict_cnn_s4.txt
python -u -W ignore::UserWarning code/models/qd-3_fourier_cnn.py /home/matt/fourier cuda:0 4 | tee logs/qd-3_fourier_cnn_s4.txt
python -u -W ignore::UserWarning code/models/qd-3_fourier_mlp.py /home/matt/fourier cuda:0 4 5 | tee logs/qd-3_fourier_mlp_s4.txt


python -u -W ignore::UserWarning code/models/mnist_baseline_cnn.py /home/matt/fourier cuda:0 0 | tee logs/mnist_baseline_cnn_s0.txt
python -u -W ignore::UserWarning code/models/mnist_aug_cnn.py /home/matt/fourier cuda:0 0 | tee logs/mnist_aug_cnn_s0.txt
python -u -W ignore::UserWarning code/models/mnist_aug_mispredict_cnn.py /home/matt/fourier cuda:0 0 | tee logs/mnist_aug_mispredict_cnn_s0.txt
python -u -W ignore::UserWarning code/models/mnist_fourier_cnn.py /home/matt/fourier cuda:0 0 | tee logs/mnist_fourier_cnn_s0.txt
python -u -W ignore::UserWarning code/models/mnist_fourier_mlp.py /home/matt/fourier cuda:0 0 5 | tee logs/mnist_fourier_mlp_s0.txt

python -u -W ignore::UserWarning code/models/mnist_baseline_cnn.py /home/matt/fourier cuda:0 1 | tee logs/mnist_baseline_cnn_s1.txt
python -u -W ignore::UserWarning code/models/mnist_aug_cnn.py /home/matt/fourier cuda:0 1 | tee logs/mnist_aug_cnn_s1.txt
python -u -W ignore::UserWarning code/models/mnist_aug_mispredict_cnn.py /home/matt/fourier cuda:0 1 | tee logs/mnist_aug_mispredict_cnn_s1.txt
python -u -W ignore::UserWarning code/models/mnist_fourier_cnn.py /home/matt/fourier cuda:0 1 | tee logs/mnist_fourier_cnn_s1.txt
python -u -W ignore::UserWarning code/models/mnist_fourier_mlp.py /home/matt/fourier cuda:0 1 5 | tee logs/mnist_fourier_mlp_s1.txt

python -u -W ignore::UserWarning code/models/qd-3_baseline_cnn.py /home/matt/fourier cuda:0 2 | tee logs/mnist_baseline_cnn_s2.txt
python -u -W ignore::UserWarning code/models/mnist_aug_cnn.py /home/matt/fourier cuda:0 2 | tee logs/mnist_aug_cnn_s2.txt
python -u -W ignore::UserWarning code/models/mnist_aug_mispredict_cnn.py /home/matt/fourier cuda:0 2 | tee logs/mnist_aug_mispredict_cnn_s2.txt
python -u -W ignore::UserWarning code/models/mnist_fourier_cnn.py /home/matt/fourier cuda:0 2 | tee logs/mnist_fourier_cnn_s2.txt
python -u -W ignore::UserWarning code/models/mnist_fourier_mlp.py /home/matt/fourier cuda:0 2 5 | tee logs/mnist_fourier_mlp_s2.txt

python -u -W ignore::UserWarning code/models/mnist_baseline_cnn.py /home/matt/fourier cuda:0 3 | tee logs/mnist_baseline_cnn_s3.txt
python -u -W ignore::UserWarning code/models/mnist_aug_cnn.py /home/matt/fourier cuda:0 3 | tee logs/mnist_aug_cnn_s3.txt
python -u -W ignore::UserWarning code/models/mnist_aug_mispredict_cnn.py /home/matt/fourier cuda:0 3 | tee logs/mnist_aug_mispredict_cnn_s3.txt
python -u -W ignore::UserWarning code/models/mnist_fourier_cnn.py /home/matt/fourier cuda:0 3 | tee logs/mnist_fourier_cnn_s3.txt
python -u -W ignore::UserWarning code/models/mnist_fourier_mlp.py /home/matt/fourier cuda:0 3 5 | tee logs/mnist_fourier_mlp_s3.txt

python -u -W ignore::UserWarning code/models/mnist_baseline_cnn.py /home/matt/fourier cuda:0 4 | tee logs/mnist_baseline_cnn_s4.txt
python -u -W ignore::UserWarning code/models/mnist_aug_cnn.py /home/matt/fourier cuda:0 4 | tee logs/mnist_aug_cnn_s4.txt
python -u -W ignore::UserWarning code/models/mnist_aug_mispredict_cnn.py /home/matt/fourier cuda:0 4 | tee logs/mnist_aug_mispredict_cnn_s4.txt
python -u -W ignore::UserWarning code/models/mnist_fourier_cnn.py /home/matt/fourier cuda:0 4 | tee logs/mnist_fourier_cnn_s4.txt
python -u -W ignore::UserWarning code/models/mnist_fourier_mlp.py /home/matt/fourier cuda:0 4 5 | tee logs/mnist_fourier_mlp_s4.txt