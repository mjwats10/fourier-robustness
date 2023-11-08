import numpy as np
import pyefd
import cv2
import os
from code.modules import datasets

MNIST_DATA = os.getcwd() + '/mnist'
FOURIER_ORDER = 20
RAND_SEED = 0
  
# transform function
def transform_train(img):
    raster = np.asarray(img) # convert PIL image to numpy array for openCV
    ret, raster = cv2.threshold(raster, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # since some images have artifacts disconnected from the digit, extract only
    # largest contour from the contour list (this should be the digit)
    contour_lens = [len(contour) for contour in contours]
    largest_index = contour_lens.index(max(contour_lens))

    # get translation and rotation offsets
    contour = np.squeeze(contours[largest_index])
    sketch_center = pyefd.calculate_dc_coefficients(contour)
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True)
    return coeffs

# create train, eval, and test datasets
train_data = datasets.MNIST_VAL(root=MNIST_DATA, train=True, val=False, download=True, transform=transform_train)

# load dataset
fourier_descriptors = []
for (img,label) in train_data:
  fourier_descriptors.append(img)

fourier_descriptors = np.stack(fourier_descriptors)
mean = np.mean(fourier_descriptors, axis=0)
stdev = np.std(fourier_descriptors, axis=0)
print(np.array2string(mean, separator=', '))
print('-----------------------------------')
print(np.array2string(stdev, separator=', '))
