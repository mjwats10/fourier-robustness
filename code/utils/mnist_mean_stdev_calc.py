import torch
from torch import nn
from torchvision import datasets, models
from torchvision import transforms as T
from torch.utils.data import DataLoader
import random
import numpy as np
import pyefd
import cv2

SERVER = "matt"
if SERVER == "apg":
    MNIST_DATA = '/home/apg/mw/fourier/mnist'
else:
    MNIST_DATA = '/home/matt/fourier/mnist'
FOURIER_ORDER = 20
BATCH_SIZE = 60000
RAND_SEED = 0

# function to ensure deterministic worker re-seeding for reproduceability
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
  
# transform function - normalize img
def transform_train(img):
    raster = np.asarray(img) # convert PIL image to numpy array for openCV
    ret, raster = cv2.threshold(raster, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # since some images have artifacts disconnected from the digit, extract only
    # largest contour from the contour list (this should be the digit)
    largest_size = 0
    largest_index = 0
    for i, contour in enumerate(contours):
        if len(contour) > largest_size:
            largest_size = len(contour)
            largest_index = i

    # get translation and rotation offsets
    contour = np.squeeze(contours[largest_index])
    sketch_center = pyefd.calculate_dc_coefficients(contour)
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True)
    return torch.from_numpy(coeffs).float()


# seed RNGs
torch.manual_seed(RAND_SEED)
random.seed(RAND_SEED)

# create train, eval, and test datasets
train_data = datasets.MNIST(root=MNIST_DATA, train=True, download=True, transform=transform_train)

# load dataset
fourier_descriptors = []
for (img,label) in train_data:
  fourier_descriptors.append(img)

fourier_descriptors = np.stack(fourier_descriptors)
mean = np.mean(fourier_descriptors, axis=0)
stdev = np.std(fourier_descriptors, axis=0)
print(mean)
print('-----------------------------------')
print(stdev)
