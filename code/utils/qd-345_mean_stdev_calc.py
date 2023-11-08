import os
import torch
import cv2
import numpy as np
import pyefd
from code.modules import misc, datasets

IMG_SIDE = 256
PADDING = 62 if IMG_SIDE == 256 else 96
NUM_CLASSES = 345
FOURIER_ORDER = 20
TRAIN_DATA = args.root_dir + '/qd-345/train/'
  
# transform functions - take sketch image, return torch tensor of descriptors
def fourier_transform(vector_img):
  stroke_rasters = misc.vector_to_raster_graph(vector_img, IMG_SIDE, PADDING)
  
  stroke_rasters_binary = []
  for raster in stroke_rasters:
    raster_binary = cv2.threshold(raster, 100, 255, cv2.THRESH_BINARY)[1]
    stroke_rasters_binary.append(raster_binary)

  stroke_fourier_descriptors = []
  for i, raster in enumerate(stroke_rasters_binary):
    contours = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    contour_lens = [len(contour) for contour in contours]
    largest_index = contour_lens.index(max(contour_lens))

    if max(contour_lens) > 1:
      contour = np.asarray(contours[largest_index]).squeeze()
      coeffs, transform = pyefd.elliptic_fourier_descriptors(contour, 
                                                          order=FOURIER_ORDER, 
                                                          normalize=True,
                                                          return_transformation=True)
      stroke_fourier_descriptors.append(coeffs)

  return stroke_fourier_descriptors


# load dataset
train_imgs = []
for class_name in os.listdir(TRAIN_DATA):
    train_folder = TRAIN_DATA + class_name
    train_drawings = datasets.unpack_drawings(train_folder)
    train_imgs += train_drawings

fourier_descriptors = []
for img in train_imgs:
  fourier_descriptors += fourier_transform(img)

fourier_descriptors = np.stack(fourier_descriptors)
mean = np.mean(fourier_descriptors, axis=0)
stdev = np.std(fourier_descriptors, axis=0)
print(np.array2string(mean, separator=', '))
print('-----------------------------------')
print(np.array2string(stdev, separator=', '))

