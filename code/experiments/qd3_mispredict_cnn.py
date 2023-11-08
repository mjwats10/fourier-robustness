import argparse
import torch
from torch import nn
from torchvision import transforms as T
from torch.utils.data import DataLoader
import random
import numpy as np
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
EXP_NAME = f'qd3_mispredict_cnn_s{args.rand_seed}'
ROOT_PATH = os.getcwd()
CHECK_PATH = ROOT_PATH + '/models/' + EXP_NAME + '_check.pt'
BEST_PATH = ROOT_PATH + '/models/' + EXP_NAME + '_best.pt'
TRAIN_DATA = ROOT_PATH + '/qd3/train/'
VAL_DATA = ROOT_PATH + '/qd3/val/'
TEST_DATA = ROOT_PATH + '/qd3/test/'

IMG_SIDE = 28
PADDING = 62 if IMG_SIDE == 256 else 96
RAND_SEED = args.rand_seed
DEVICE = args.device
NUM_CLASSES = 3
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
  
# transform functions - take sketch image, return torch tensor of descriptors
def transform(vector_img, data_split):
    raster = misc.vector_to_raster(vector_img, IMG_SIDE, PADDING)
    raster = transforms_norm(raster)

    # add rotations and translations
    if data_split == "train" or data_split == "val":
        angle = random.random()*30 - 30
        deltaX = random.randint(3, 0)
        deltaY = random.randint(3, 0)
    else:
        angle = random.random()*30
        deltaX = random.randint(0, 3)
        deltaY = random.randint(0, 3)

    raster = T.functional.affine(raster, angle, [deltaX, deltaY], 1, 0,
                                 interpolation=T.InterpolationMode.BILINEAR)
    return raster

#-------------------------------------------------------------------------------------------

# load dataset
train_imgs, val_imgs, test_imgs, train_counts, val_counts, test_counts = datasets.get_data(TRAIN_DATA, VAL_DATA, TEST_DATA)
  
#-------------------------------------------------------------------------------------------

# seed RNGs
torch.manual_seed(RAND_SEED)
random.seed(RAND_SEED)

# create train, eval, and test datasets
train_data = datasets.QuickdrawDataset(train_imgs, train_counts, transform, data_split="train")
val_data = datasets.QuickdrawDataset(val_imgs, val_counts, transform, data_split="val")
test_data = datasets.QuickdrawDataset(test_imgs, test_counts, transform, data_split="test")

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
