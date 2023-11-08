import os
import cv2
import numpy as np
import pyefd
from code.modules import misc, datasets

IMG_SIDE = 28
PADDING = 62 if IMG_SIDE == 256 else 96
NUM_CLASSES = 3
FOURIER_ORDER = 20
TRAIN_DATA = os.getcwd() + '/qd3/train/'
  
# transform functions - take sketch image, return torch tensor of descriptors
def transform_train(vector_img):
    raster_img = misc.vector_to_raster(vector_img, IMG_SIDE, PADDING)
    ret, raster = cv2.threshold(raster_img, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # since some images have artifacts disconnected from the digit, extract only
    # largest contour from the contour list (this should be the digit)
    contour_lens = [len(contour) for contour in contours]
    largest_index = contour_lens.index(max(contour_lens))

    # get translation and rotation offsets
    contour = np.squeeze(contours[largest_index])
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True)
    return coeffs

# load dataset
train_imgs = []
for class_name in os.listdir(TRAIN_DATA):
    train_folder = TRAIN_DATA + class_name
    train_drawings = datasets.unpack_drawings(train_folder)
    train_imgs += train_drawings

fourier_descriptors = []
for img in train_imgs:
  fourier_descriptors.append(transform_train(img))

fourier_descriptors = np.stack(fourier_descriptors)
mean = np.mean(fourier_descriptors, axis=0)
stdev = np.std(fourier_descriptors, axis=0)
print(np.array2string(mean, separator=', '))
print('-----------------------------------')
print(np.array2string(stdev, separator=', '))

