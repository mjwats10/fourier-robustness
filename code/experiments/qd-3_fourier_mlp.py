import argparse
import ast
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
parser.add_argument("f_order", type=int)
args = parser.parse_args()

# Const vars
EXP_NAME = f'qd-3_fourier_mlp_s{args.rand_seed}'
ROOT_PATH = os.getcwd()
CHECK_PATH = ROOT_PATH + '/models/' + EXP_NAME + '_check.pt'
BEST_PATH = ROOT_PATH + '/models/' + EXP_NAME + '_best.pt'
TRAIN_DATA = ROOT_PATH + '/qd-3/train/'
VAL_DATA = ROOT_PATH + '/qd-3/val/'
TEST_DATA = ROOT_PATH + '/qd-3/test/'
LOG_PATH = ROOT_PATH + '/logs/'

FOURIER_ORDER = args.f_order
RAND_SEED = args.rand_seed
DEVICE = args.device
IMG_SIDE = 28
PADDING = 62 if IMG_SIDE == 256 else 96
NUM_CLASSES = 3
EPOCHS = 90 
LEARNING_RATE = 1e-3
BATCH_SIZE = 500
LOSS_FN = nn.CrossEntropyLoss()

#-------------------------------------------------------------------------------------------

# Define transformation(s) to be applied to dataset
transforms_tensor = T.ToTensor()

# get mean and stdevs of fourier coeffs
mean_file = open(LOG_PATH + 'qd-3_means.txt', 'r')
mean_list = ast.literal_eval(mean_file.read())
mean_file.close()
means = np.asarray(mean_list)

stdev_file = open(LOG_PATH + 'qd-3_stdevs.txt', 'r')
stdev_list = ast.literal_eval(stdev_file.read())
stdev_file.close()
stdevs = np.asarray(stdev_list)
  
# transform function - normalize img
def transform_train(vector_img):
    raster_img = misc.vector_to_raster(vector_img, IMG_SIDE, PADDING)
    ret, raster = cv2.threshold(raster_img, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # since some images have artifacts disconnected from the shape, extract only
    # largest contour from the contour list (this should be the shape)
    contour_lens = [len(contour) for contour in contours]
    largest_index = contour_lens.index(max(contour_lens))

    # get translation and rotation offsets
    contour = np.squeeze(contours[largest_index])
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True)
    coeffs = (coeffs - means[:FOURIER_ORDER]) / stdevs[:FOURIER_ORDER]
    return torch.from_numpy(coeffs.flatten()).float()

def transform_val(vector_img):
    # apply random corrupting transformation to input img
    raster = misc.vector_to_raster(vector_img, IMG_SIDE, PADDING)
    img = transforms_tensor(raster.astype(np.float32))
    angle = random.random()*30 - 30
    deltaX = random.randint(-3, 0)
    deltaY = random.randint(-3, 0)
    img = T.functional.affine(img, angle, [deltaX, deltaY], 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    img = np.squeeze(img.numpy()).astype(np.uint8)

    ret, raster = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # since some images have artifacts disconnected from the shape, extract only
    # largest contour from the contour list (this should be the shape)
    contour_lens = [len(contour) for contour in contours]
    largest_index = contour_lens.index(max(contour_lens))

    # get translation and rotation offsets
    contour = np.squeeze(contours[largest_index])
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True)
    coeffs = (coeffs - means[:FOURIER_ORDER]) / stdevs[:FOURIER_ORDER]
    return torch.from_numpy(coeffs.flatten()).float()

def transform_test(vector_img):
    # apply random corrupting transformation to input img
    raster = misc.vector_to_raster(vector_img, IMG_SIDE, PADDING)
    img = transforms_tensor(raster.astype(np.float32))
    angle = random.random()*30
    deltaX = random.randint(0, 3)
    deltaY = random.randint(0, 3)
    img = T.functional.affine(img, angle, [deltaX, deltaY], 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    img = np.squeeze(img.numpy()).astype(np.uint8)

    ret, raster = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # since some images have artifacts disconnected from the shape, extract only
    # largest contour from the contour list (this should be the shape)
    contour_lens = [len(contour) for contour in contours]
    largest_index = contour_lens.index(max(contour_lens))

    # get translation and rotation offsets
    contour = np.squeeze(contours[largest_index])
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True)
    coeffs = (coeffs - means[:FOURIER_ORDER]) / stdevs[:FOURIER_ORDER]
    return torch.from_numpy(coeffs.flatten()).float()

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

# initalize model object and load model parameters into optimizer
model = models.MLP(NUM_CLASSES, FOURIER_ORDER)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.to(DEVICE)

current_epoch, best_acc, plateau_len = misc.get_train_state(model, optim, args.resume, CHECK_PATH)

if not args.test_only:
    misc.train(EXP_NAME, current_epoch, EPOCHS, best_acc, plateau_len, train_loader, val_loader, model, LOSS_FN, optim, CHECK_PATH, BEST_PATH, DEVICE)
 
if not args.skip_test:
    misc.test(model, BEST_PATH, RAND_SEED, test_loader, DEVICE)
