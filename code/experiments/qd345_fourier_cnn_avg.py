import argparse
import torch
from torch import nn
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision import models
import random
import cv2
import numpy as np
import pyefd
from code.modules import misc, datasets
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
EXP_NAME = f'qd345_fourier_cnn_avg_s{args.rand_seed}'
ROOT_PATH = os.getcwd()
CHECK_PATH = ROOT_PATH + '/models/' + EXP_NAME + '_check.pt'
BEST_PATH = ROOT_PATH + '/models/' + EXP_NAME + '_best.pt'
TRAIN_DATA = ROOT_PATH + '/qd345/train/'
VAL_DATA = ROOT_PATH + '/qd345/val/'
TEST_DATA = ROOT_PATH + '/qd345/test/'

FOURIER_ORDER = 1
IMG_SIDE = 256
IMG_CENTER = np.asarray(((IMG_SIDE - 1) / 2, (IMG_SIDE - 1) / 2))
PADDING = 62 if IMG_SIDE == 256 else 96
RAND_SEED = args.rand_seed
DEVICE = args.device
NUM_CLASSES = 345
EPOCHS = 90 
LEARNING_RATE = 1e-3
BATCH_SIZE = 500
LOSS_FN = nn.CrossEntropyLoss()

#-------------------------------------------------------------------------------------------

# Define transformation(s) to be applied to dataset-
transforms_norm = T.Compose(
    [
        T.ToTensor(), # scales integer inputs in the range [0, 255] into the range [0.0, 1.0]
        T.Normalize(mean=(0.138), std=(0.296)) # Quickdraw mean and stdev (35.213, 75.588), divided by 255
    ]
)

transforms_tensor = T.ToTensor()

# transform function
def transform_train(vector_img):
    raster_img = misc.vector_to_raster(vector_img, IMG_SIDE, PADDING)
    ret, raster = cv2.threshold(raster_img, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # get contour lengths
    contour_lens = [len(contour) for contour in contours]

    # get translation and rotation offsets for each contour
    img_offsets = []
    contour_angles = []
    for i, contour in enumerate(contours):
        contour = np.squeeze(contour)
        sketch_center = pyefd.calculate_dc_coefficients(contour)
        coeffs, transform = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True, return_transformation=True)
        img_offset = (IMG_CENTER - sketch_center).round()
        contour_angle = np.degrees(transform[1])
        img_offsets.append(img_offset)
        contour_angles.append(contour_angle)
        
    # average over contours
    img_offset = np.mean(np.array(img_offsets), axis=0)
    contour_lens = np.array(contour_lens)
    contour_lens_norm = contour_lens / np.sum(contour_lens)
    contour_angles = np.array(contour_angles)
    contour_angle = np.sum(contour_lens_norm * contour_angles)
    
    # de-translate then de-rotate
    img = transforms_norm(raster_img)
    img = T.functional.affine(img, 0, (img_offset[0], img_offset[1]), 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    img = T.functional.affine(img, -1 * contour_angle, [0, 0], 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    return img

def transform_val(vector_img):
    # apply random corrupting transformation to input img
    raster = misc.vector_to_raster(vector_img, IMG_SIDE, PADDING)
    img = transforms_tensor(raster.astype(np.float32))
    angle = random.random()*30 - 30
    deltaX = random.randint(-10, 0)
    deltaY = random.randint(-10, 0)
    img = T.functional.affine(img, angle, [deltaX, deltaY], 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    img = np.squeeze(img.numpy()).astype(np.uint8)

    ret, raster = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # get contour lengths
    contour_lens = [len(contour) for contour in contours]

    # get translation and rotation offsets for each contour
    img_offsets = []
    contour_angles = []
    for i, contour in enumerate(contours):
        contour = np.squeeze(contour)
        sketch_center = pyefd.calculate_dc_coefficients(contour)
        coeffs, transform = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True, return_transformation=True)
        img_offset = (IMG_CENTER - sketch_center).round()
        contour_angle = np.degrees(transform[1])
        img_offsets.append(img_offset)
        contour_angles.append(contour_angle)
        
    # average over contours
    img_offset = np.mean(np.array(img_offsets), axis=0)
    contour_lens = np.array(contour_lens)
    contour_lens_norm = contour_lens / np.sum(contour_lens)
    contour_angles = np.array(contour_angles)
    contour_angle = np.sum(contour_lens_norm * contour_angles)

    # de-translate then de-rotate
    img = transforms_norm(img)
    img = T.functional.affine(img, 0, (img_offset[0], img_offset[1]), 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    img = T.functional.affine(img, -1 * contour_angle, [0, 0], 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    return img

def transform_test(vector_img):
    # apply random corrupting transformation to input img
    raster = misc.vector_to_raster(vector_img, IMG_SIDE, PADDING)
    img = transforms_tensor(raster.astype(np.float32))
    angle = random.random()*30
    deltaX = random.randint(0, 10)
    deltaY = random.randint(0, 10)
    img = T.functional.affine(img, angle, [deltaX, deltaY], 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    img = np.squeeze(img.numpy()).astype(np.uint8)

    ret, raster = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # get contour lengths
    contour_lens = [len(contour) for contour in contours]

    # get translation and rotation offsets for each contour
    img_offsets = []
    contour_angles = []
    for i, contour in enumerate(contours):
        contour = np.squeeze(contour)
        sketch_center = pyefd.calculate_dc_coefficients(contour)
        coeffs, transform = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True, return_transformation=True)
        img_offset = (IMG_CENTER - sketch_center).round()
        contour_angle = np.degrees(transform[1])
        img_offsets.append(img_offset)
        contour_angles.append(contour_angle)
        
    # average over contours
    img_offset = np.mean(np.array(img_offsets), axis=0)
    contour_lens = np.array(contour_lens)
    contour_lens_norm = contour_lens / np.sum(contour_lens)
    contour_angles = np.array(contour_angles)
    contour_angle = np.sum(contour_lens_norm * contour_angles)

    # de-translate then de-rotate
    img = transforms_norm(img)
    img = T.functional.affine(img, 0, (img_offset[0], img_offset[1]), 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    img = T.functional.affine(img, -1 * contour_angle, [0, 0], 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    return img

#-------------------------------------------------------------------------------------------

# load dataset
train_imgs, val_imgs, test_imgs, train_counts, val_counts, test_counts = datasets.get_data(TRAIN_DATA, VAL_DATA, TEST_DATA)
  
#-------------------------------------------------------------------------------------------
  
# seed RNGs
torch.manual_seed(RAND_SEED)
random.seed(RAND_SEED)

# create train, eval, and test datasets
train_data = datasets.QuickdrawDataset(train_imgs, train_counts, transform_train)
val_data = datasets.QuickdrawDataset(val_imgs, val_counts, transform_val)
test_data = datasets.QuickdrawDataset(test_imgs, test_counts, transform_test)

# create generator for dataloaders and create dataloaders for each dataset
g = torch.Generator()
g.manual_seed(RAND_SEED)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=4, pin_memory=True, worker_init_fn=misc.seed_worker, generator=g)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=4, pin_memory=True, worker_init_fn=misc.seed_worker, generator=g)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=4, pin_memory=True, worker_init_fn=misc.seed_worker, generator=g)

# init model and optimizer
model = models.shufflenet_v2_x0_5()
model.conv1[0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model.fc = nn.Linear(in_features=1024, out_features=NUM_CLASSES, bias=True)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.to(DEVICE)

current_epoch, best_acc, plateau_len = misc.get_train_state(model, optim, args.resume, CHECK_PATH)

if not args.test_only:
    misc.train(EXP_NAME, current_epoch, EPOCHS, best_acc, plateau_len, train_loader, val_loader, model, LOSS_FN, optim, CHECK_PATH, BEST_PATH, DEVICE)
 
if not args.skip_test:
    misc.test(model, BEST_PATH, RAND_SEED, test_loader, DEVICE)
