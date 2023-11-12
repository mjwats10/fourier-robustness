import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# extract validation accuracy from log file
def extract_validation_accuracy(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        # Using regex to find the pattern "best val acc: X.XXXX"
        matches = re.findall(r'best val acc: (\d+\.\d+)', content)
        if matches:
            return [float(acc) for acc in matches]
    return None

# extract test accuracy from log file
def extract_test_accuracy(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        # Using regex to find the pattern "Mean acc: X.XXXX"
        match = re.findall(r'Mean acc: (\d+\.\d+)', content)
        if match:
            return float(match[0])
    return None

# process logs
def process_directory(directory_path, model_names, model_labels, time_per_run):
    model_data = {model: {'validation_accuracy': [], 'test_accuracy': [], 'time_per_epoch': [], 'label': label} for model, label in zip(model_names, model_labels)}

    for model in model_names:
        for seed in range(5):
            file_name = f"{model}_s{seed}.txt"
            file_path = os.path.join(directory_path, file_name)

            if os.path.exists(file_path):
                accuracy_values = extract_validation_accuracy(file_path)
                test_acc_value = extract_test_accuracy(file_path)
                if accuracy_values and test_acc_value:
                    model_data[model]['validation_accuracy'].append(accuracy_values)
                    model_data[model]['test_accuracy'].append(test_acc_value)
                    time_per_epoch = time_per_run / len(accuracy_values)
                    model_data[model]['time_per_epoch'].append(time_per_epoch)

    return model_data

# plot the data
def plot_data(model_data, dataset):
    plt.figure(figsize=(10, 6))
    print(dataset)
    print("Mean Acc  |    Stdev")

    for model, data in model_data.items():
        max_len = max(map(len, data['validation_accuracy']))
        acc_arry = np.array([acc + [None]*(max_len - len(acc)) for acc in data['validation_accuracy']], dtype=np.float32)
        acc_arry = np.nan_to_num(acc_arry)
        num_vals = np.count_nonzero(acc_arry, axis=0)
        acc_sums = np.sum(acc_arry, axis=0, keepdims=False)
        avg_validation_accuracy = acc_sums / num_vals
        avg_test_accuracy = sum(data['test_accuracy']) / len(data['test_accuracy'])
        avg_test_accuracy = avg_validation_accuracy * avg_test_accuracy / avg_validation_accuracy[-1]
        avg_time_per_epoch = sum(data['time_per_epoch']) / len(data['time_per_epoch'])
        epochs = list(range(1, len(avg_validation_accuracy) + 1))
        times = np.array([epoch * avg_time_per_epoch for epoch in epochs], dtype=np.float32)

        plt.plot(times, avg_test_accuracy, label=data['label'])
        test_acc_arry = np.array(data['test_accuracy'])
        mean = np.mean(test_acc_arry)
        stdev = np.std(test_acc_arry)
        print(data['label'])
        print(mean, stdev)

    plt.xlabel('Time (minutes)')
    plt.ylabel('Test Accuracy')
    plt.title(f"{dataset} Test Accuracy vs Wall-Clock Time")
    plt.legend()
    fig_save = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop', f"{dataset} Efficiency")
    plt.savefig(fig_save,dpi=2000,bbox_inches='tight')
#-------------------------------------------------------------------------------------------

directory = "/home/matt/fourier/logs"
model_names = ['mnist_noise_aug_cnn', 'mnist_noise_baseline_cnn', 'mnist_noise_fourier_mlp']
model_labels = ['Augmentation-CNN', 'Baseline-CNN', 'Fourier-MLP']
dataset = "Noisy MNIST"
time_per_run = 10

model_data = process_directory(directory, model_names, model_labels, time_per_run)
plot_data(model_data, dataset)
