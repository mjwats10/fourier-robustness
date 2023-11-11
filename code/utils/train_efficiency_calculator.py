import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# extract validation accuracy from a log file
def extract_validation_accuracy(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        # Using regex to find the pattern "best val acc: X.XXXX"
        matches = re.findall(r'best val acc: (\d+\.\d+)', content)
        if matches:
            return [float(acc) for acc in matches]
    return None

# process logs
def process_directory(directory_path, model_names, time_per_run):
    model_data = {model: {'validation_accuracy': [], 'time_per_epoch': []} for model in model_names}

    for model in model_names:
        for seed in range(5):
            file_name = f"{model}_s{seed}.txt"
            file_path = os.path.join(directory_path, file_name)

            if os.path.exists(file_path):
                accuracy_values = extract_validation_accuracy(file_path)
                if accuracy_values:
                    model_data[model]['validation_accuracy'].append(accuracy_values)
                    time_per_epoch = time_per_run / len(accuracy_values)
                    model_data[model]['time_per_epoch'].append(time_per_epoch)

    return model_data

# plot the data
def plot_data(model_data, model_labels, dataset):
    plt.figure(figsize=(10, 6))

    for model, data in model_data.items():
        max_len = max(map(len, data['validation_accuracy']))
        acc_arry = np.array([acc + [None]*(max_len - len(acc)) for acc in data['validation_accuracy']], dtype=float)
        acc_arry = np.nan_to_num(acc_arry)
        num_vals = np.count_nonzero(acc_arry, axis=0)
        acc_sums = np.sum(acc_arry, axis=0, keepdims=False)
        avg_validation_accuracy = acc_sums / num_vals
        avg_time_per_epoch = sum(data['time_per_epoch']) / len(data['time_per_epoch'])
        epochs = list(range(1, len(avg_validation_accuracy) + 1))
        times = np.array([epoch * avg_time_per_epoch for epoch in epochs], dtype=float)

        plt.plot(times, avg_validation_accuracy, label=model_labels[model])

    plt.xlabel('Time (minutes)')
    plt.ylabel('Validation Accuracy')
    plt.title(f"{dataset} Validation Accuracy vs Wall-Clock Time")
    plt.legend()
    plt.show()

#-------------------------------------------------------------------------------------------

directory = "/home/matt/fourier/logs"
model_names = ['qd-345_aug_cnn', 'qd-345_aug_mispredict_cnn', 'qd-345_baseline_cnn', 'qd-345_fourier_cnn_largest', 'qd-345_fourier_gnn_w0.25_deep_skip']
model_labels = {model_names[0]: 'Augmentation-CNN', model_names[1]: 'Mispredict-CNN', model_names[2]: 'Baseline-CNN', model_names[3]: 'Fourier-CNN', model_names[4]: 'Fourier-GNN'}
dataset = "QD-345"
time_per_run = 120

model_data = process_directory(directory, model_names, time_per_run)
plot_data(model_data, model_labels, dataset)
