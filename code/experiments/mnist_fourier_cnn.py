import argparse
import torch
from torch import nn
from torchvision import transforms as T
from torch.utils.data import DataLoader
import random
import numpy as np
import pyefd
import cv2
from code.modules import misc, datasets, models
import os

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("device")
parser.add_argument("rand_seed", type=int)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--test_only", action="store_true")
parser.add_argument("--skip_test", action="store_true")
args = parser.parse_args()

# Const vars
EXP_NAME = f'mnist_fourier_cnn_s{args.rand_seed}'
ROOT_PATH = os.getcwd()
CHECK_PATH = ROOT_PATH + '/models/' + EXP_NAME + '_check.pt'
BEST_PATH = ROOT_PATH + '/models/' + EXP_NAME + '_best.pt'
MNIST_DATA = ROOT_PATH + '/mnist'

FOURIER_ORDER = 1
IMG_SIDE = 28
IMG_CENTER = np.asarray(((IMG_SIDE - 1) / 2, (IMG_SIDE - 1) / 2))
RAND_SEED = args.rand_seed
DEVICE = args.device
NUM_CLASSES = 10
EPOCHS = 90
LEARNING_RATE = 1e-3
BATCH_SIZE = 500
LOSS_FN = nn.CrossEntropyLoss()

#-------------------------------------------------------------------------------------------

# Define transformation(s) to be applied to dataset-
transforms_norm = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean = (0.1307,), std = (0.3081,)) # MNIST mean and stdev
    ]
)

transforms_tensor = T.ToTensor()
  
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
    coeffs, transform = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True, return_transformation=True)
    contour_angle = np.degrees(transform[1])
    img_offset = (IMG_CENTER - sketch_center).round()

    # de-translate then de-rotate
    img = transforms_norm(img)
    img = T.functional.affine(img, 0, (img_offset[0], img_offset[1]), 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    img = T.functional.affine(img, -1 * contour_angle, [0, 0], 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    return img

def transform_val(img):
    # apply random corrupting transformation to input img
    img = transforms_tensor(np.asarray(img,dtype=np.float32))
    angle = random.random()*30 - 30
    deltaX = random.randint(-3, 0)
    deltaY = random.randint(-3, 0)
    img = T.functional.affine(img, angle, [deltaX, deltaY], 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    img = np.squeeze(img.numpy()).astype(np.uint8)

    ret, raster = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # since some images have artifacts disconnected from the digit, extract only
    # largest contour from the contour list (this should be the digit)
    contour_lens = [len(contour) for contour in contours]
    largest_index = contour_lens.index(max(contour_lens))

    # get translation and rotation offsets
    contour = np.squeeze(contours[largest_index])
    sketch_center = pyefd.calculate_dc_coefficients(contour)
    coeffs, transform = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True, return_transformation=True)
    contour_angle = np.degrees(transform[1])
    img_offset = (IMG_CENTER - sketch_center).round()

    # de-translate then de-rotate
    img = transforms_norm(img)
    img = T.functional.affine(img, 0, (img_offset[0], img_offset[1]), 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    img = T.functional.affine(img, -1 * contour_angle, [0, 0], 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    return img

def transform_test(img):
    # apply random corrupting transformation to input img
    img = transforms_tensor(np.asarray(img,dtype=np.float32))
    angle = random.random()*30
    deltaX = random.randint(0, 3)
    deltaY = random.randint(0, 3)
    img = T.functional.affine(img, angle, [deltaX, deltaY], 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    img = np.squeeze(img.numpy()).astype(np.uint8)

    ret, raster = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # since some images have artifacts disconnected from the digit, extract only
    # largest contour from the contour list (this should be the digit)
    contour_lens = [len(contour) for contour in contours]
    largest_index = contour_lens.index(max(contour_lens))

    # get translation and rotation offsets
    contour = np.squeeze(contours[largest_index])
    sketch_center = pyefd.calculate_dc_coefficients(contour)
    coeffs, transform = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True, return_transformation=True)
    contour_angle = np.degrees(transform[1])
    img_offset = (IMG_CENTER - sketch_center).round()

    # de-translate then de-rotate
    img = transforms_norm(img)
    img = T.functional.affine(img, 0, (img_offset[0], img_offset[1]), 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    img = T.functional.affine(img, -1 * contour_angle, [0, 0], 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    return img

#-------------------------------------------------------------------------------------------

# seed RNGs
torch.manual_seed(RAND_SEED)
random.seed(RAND_SEED)

# create train, eval, and test datasets
train_data = datasets.MNIST_VAL(root=MNIST_DATA, train=True, val=False, download=True, transform=transform_train)
val_data = datasets.MNIST_VAL(root=MNIST_DATA, train=True, val=True, download=True, transform=transform_val)
test_data = datasets.MNIST_VAL(root=MNIST_DATA, train=False, download=True, transform=transform_test) 

# create generator for dataloaders and create dataloaders for each dataset
g = torch.Generator()
g.manual_seed(RAND_SEED)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=4, pin_memory=True, worker_init_fn=misc.seed_worker, generator=g)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=4, pin_memory=True, worker_init_fn=misc.seed_worker, generator=g)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=4, pin_memory=True, worker_init_fn=misc.seed_worker, generator=g)

# initalize model object and load model parameters into optimizer
model = models.LeNet(NUM_CLASSES)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.to(DEVICE)

current_epoch, best_acc, plateau_len = misc.get_train_state(model, optim, args.resume, CHECK_PATH)

if not args.test_only:
    misc.train(EXP_NAME, current_epoch, EPOCHS, best_acc, plateau_len, train_loader, val_loader, model, LOSS_FN, optim, CHECK_PATH, BEST_PATH, DEVICE)
 
if not args.skip_test:
    misc.test(model, BEST_PATH, RAND_SEED, test_loader, DEVICE)
