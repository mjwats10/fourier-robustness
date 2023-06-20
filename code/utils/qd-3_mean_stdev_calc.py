import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pyefd
import cairocffi as cairo
import struct
from struct import unpack

IMG_SIDE = 28
PADDING = 62 if IMG_SIDE == 256 else 96
NUM_CLASSES = 3
FOURIER_ORDER = 20
TRAIN_DATA = '/home/matt/fourier/qd-3/train/'

list_of_classes = ["circle", "square", "triangle"]
                   
 # define methods for unpacking Quickdraw .bin files
def unpack_drawing(file_handle):
  file_handle.read(15)
  n_strokes, = unpack('H', file_handle.read(2))
  image = []
  for i in range(n_strokes):
      n_points, = unpack('H', file_handle.read(2))
      fmt = str(n_points) + 'B'
      x = unpack(fmt, file_handle.read(n_points))
      y = unpack(fmt, file_handle.read(n_points))
      image.append((x, y))

  return image

def unpack_drawings(filename):
  imageset = []
  with open(filename, 'rb') as f:
      while True:
          try:
              imageset.append(unpack_drawing(f))
          except struct.error:
              break
  return imageset

# convert raw vector image to single raster image
def vector_to_raster(vector_image, side=IMG_SIDE, line_diameter=16, padding=PADDING, bg_color=(0,0,0), fg_color=(1,1,1)):
    """
    padding and line_diameter are relative to the original 256x256 image.
    """
    
    original_side = 256.
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)
        
    bbox = np.hstack(vector_image).max(axis=1)
    offset = ((original_side, original_side) - bbox) / 2.
    offset = offset.reshape(-1,1)
    centered = [stroke + offset for stroke in vector_image]

    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()

    # draw strokes, this is the most cpu-intensive part
    ctx.set_source_rgb(*fg_color)     
    for xv, yv in centered:   
        ctx.move_to(xv[0], yv[0])
        for x, y in zip(xv, yv):
            ctx.line_to(x, y)
        ctx.stroke()

    data = surface.get_data()
    raster = np.copy(np.asarray(data)[::4]).reshape(side, side)
    return raster
  
# transform functions - take sketch image, return torch tensor of descriptors
def transform_train(vector_img):
    raster_img = vector_to_raster(vector_img)
    ret, raster = cv2.threshold(raster_img, 100, 255, cv2.THRESH_BINARY) # binarize image
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
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True)
    return coeffs

# load dataset
train_imgs = []
for item in list_of_classes:
  train_folder = TRAIN_DATA + item + '.bin'
  train_drawings = unpack_drawings(train_folder)
  train_imgs += train_drawings

fourier_descriptors = []
for img in train_imgs:
  fourier_descriptors.append(transform_train(img))

fourier_descriptors = np.stack(fourier_descriptors)
mean = np.mean(fourier_descriptors, axis=0)
stdev = np.std(fourier_descriptors, axis=0)
print(mean)
print('-----------------------------------')
print(stdev)

