import argparse
import ast
import torch
from torch import nn
from torchvision import transforms as T
import random
import cv2
import numpy as np
import pyefd
from torch_geometric.loader import DataLoader
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
parser.add_argument("width_mult", type=float)
parser.add_argument("--deep", action="store_true")
parser.add_argument("--skip_conn", action="store_true")
args = parser.parse_args()

# Const vars
EXP_NAME = f'qd345_fourier_gnn_N{args.f_order}_w{args.width_mult}_deep_s{args.rand_seed}' if args.deep else f'qd345_fourier_gnn_N{args.f_order}_w{args.width_mult}_s{args.rand_seed}'
ROOT_PATH = os.getcwd()
CHECK_PATH = ROOT_PATH + '/models/temp/' + EXP_NAME + '_check.pt'
BEST_PATH = ROOT_PATH + '/models/temp/' + EXP_NAME + '_best.pt'
TRAIN_DATA = ROOT_PATH + '/qd345/train/'
VAL_DATA = ROOT_PATH + '/qd345/val/'
TEST_DATA = ROOT_PATH + '/qd345/test/'
LOG_PATH = ROOT_PATH +'/logs/'

FOURIER_ORDER = args.f_order
RAND_SEED = args.rand_seed
DEVICE = args.device
WIDTH_MULTIPLE = args.width_mult
DEEP = args.deep
SKIP = args.skip_conn
IMG_SIDE = 256
PADDING = 62 if IMG_SIDE == 256 else 96
NUM_CLASSES = 345
EPOCHS = 90
LEARNING_RATE = 1e-3
BATCH_SIZE = 500
LOSS_FN = nn.CrossEntropyLoss()
EDGE_ATTR_DIM = 3

#-------------------------------------------------------------------------------------------

# get edge data
ELEMENT = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
ANGLE_NORM = 2 * np.pi
DIST_NORM = np.sqrt(IMG_SIDE**2 + IMG_SIDE**2)
def get_edges(stroke_angles, stroke_centers, stroke_rasters): 
  adj_1 = []
  adj_2 = []
  edge_info = []
  num_strokes = len(stroke_rasters)
  for i in range(num_strokes):
      stroke_i = cv2.dilate(stroke_rasters[i], ELEMENT).astype(np.float32)
      for j in range(i + 1, num_strokes):
          ij_info = []
          ji_info = []
          adj_1.append(i)
          adj_2.append(j)
          adj_1.append(j)
          adj_2.append(i)
          
          angle_diff = stroke_angles[i] - stroke_angles[j]
          angle_diff_norm_ij = [angle_diff / ANGLE_NORM]
          angle_diff_norm_ji = [-1 * angle_diff / ANGLE_NORM]
          ij_info.append(angle_diff_norm_ij)
          ji_info.append(angle_diff_norm_ji)

          dist = np.sqrt((stroke_centers[i][0] - stroke_centers[j][0])**2 +
                          (stroke_centers[i][1] - stroke_centers[j][1])**2)
          dist_norm = [dist / DIST_NORM]
          ij_info.append(dist_norm)
          ji_info.append(dist_norm)

          temp = stroke_i + stroke_rasters[j]
          if np.amax(temp) > 255:
            ij_info.append([1])
            ji_info.append([1])
          else:
            ij_info.append([0])
            ji_info.append([0])
          
          edge_info.append(ij_info)
          edge_info.append(ji_info)

  edge_indices = torch.LongTensor([adj_1, adj_2])
  edge_attr = torch.FloatTensor(edge_info)
  return edge_indices, edge_attr

# get mean and stdevs of fourier coeffs
mean_file = open(LOG_PATH + 'qd345_means.txt', 'r')
mean_list = ast.literal_eval(mean_file.read())
mean_file.close()
means = np.asarray(mean_list)

stdev_file = open(LOG_PATH + 'qd345_stdevs.txt', 'r')
stdev_list = ast.literal_eval(stdev_file.read())
stdev_file.close()
stdevs = np.asarray(stdev_list)

# transform functions - take sketch image, return torch tensor of descriptors
def fourier_transform(vector_img, data_split):
    stroke_rasters = misc.vector_to_raster_graph(vector_img, IMG_SIDE, PADDING)

    # add rotations and translations at test time
    if data_split == "val": 
        stroke_rasters = np.stack(stroke_rasters)
        stroke_rasters = torch.from_numpy(stroke_rasters).float()

        angle = random.random()*30 - 30
        deltaX = random.randint(-10, 0)
        deltaY = random.randint(-10, 0)

        stroke_rasters = T.functional.affine(stroke_rasters,angle,[deltaX, deltaY],1,0,
                                            interpolation=T.InterpolationMode.BILINEAR)
        stroke_rasters = stroke_rasters.numpy().astype(np.uint8)
        stroke_rasters = np.split(stroke_rasters, stroke_rasters.shape[0])
        stroke_rasters = [np.squeeze(a) for a in stroke_rasters]
    elif data_split == "test": 
        stroke_rasters = np.stack(stroke_rasters)
        stroke_rasters = torch.from_numpy(stroke_rasters).float()

        angle = random.random()*30
        deltaX = random.randint(0, 10)
        deltaY = random.randint(0, 10)

        stroke_rasters = T.functional.affine(stroke_rasters,angle,[deltaX, deltaY],1,0,
                                            interpolation=T.InterpolationMode.BILINEAR)
        stroke_rasters = stroke_rasters.numpy().astype(np.uint8)
        stroke_rasters = np.split(stroke_rasters, stroke_rasters.shape[0])
        stroke_rasters = [np.squeeze(a) for a in stroke_rasters]

    stroke_rasters_binary = []
    for raster in stroke_rasters:
        raster_binary = cv2.threshold(raster, 100, 255, cv2.THRESH_BINARY)[1]
        stroke_rasters_binary.append(raster_binary)

    stroke_fourier_descriptors = []
    strokes_to_remove = []
    stroke_angles = []
    stroke_centers = []
    for i, raster in enumerate(stroke_rasters_binary):
        contours = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        # extract only largest contour from the contour list
        contour_lens = [len(contour) for contour in contours]
        largest_size = max(contour_lens)
        largest_index = contour_lens.index(largest_size)

        if largest_size > 1:
            contour = np.asarray(contours[largest_index]).squeeze()
            coeffs, transform = pyefd.elliptic_fourier_descriptors(contour, 
                                                                order=FOURIER_ORDER, 
                                                                normalize=True,
                                                                return_transformation=True)
            stroke_angle = transform[1]
            coeffs = (coeffs - means[:FOURIER_ORDER]) / stdevs[:FOURIER_ORDER]
            stroke_fourier_descriptors.append(coeffs.flatten())
            stroke_angles.append(stroke_angle)
            stroke_center = pyefd.calculate_dc_coefficients(contour)
            stroke_centers.append(stroke_center)
        else:
            strokes_to_remove.append(i)

    for i in reversed(strokes_to_remove):
        del stroke_rasters_binary[i]

    edge_indices, edge_attr = get_edges(stroke_angles, stroke_centers,
                                        stroke_rasters_binary)
    stroke_fourier_descriptors = np.stack(stroke_fourier_descriptors)
    stroke_fourier_descriptors = torch.from_numpy(stroke_fourier_descriptors).float()
    return stroke_fourier_descriptors, edge_indices, edge_attr
  
#-------------------------------------------------------------------------------------------

# load dataset
train_imgs, val_imgs, test_imgs, train_counts, val_counts, test_counts = datasets.get_data(TRAIN_DATA, VAL_DATA, TEST_DATA)
  
#-------------------------------------------------------------------------------------------
  
# seed RNGs
torch.manual_seed(RAND_SEED)
random.seed(RAND_SEED)

# create train, eval, and test datasets
train_data = datasets.QuickdrawGraphDataset(train_imgs, train_counts, fourier_transform, data_split="train")
val_data = datasets.QuickdrawGraphDataset(val_imgs, val_counts, fourier_transform, data_split="val")
test_data = datasets.QuickdrawGraphDataset(test_imgs, test_counts, fourier_transform, data_split="test")

# create dataloaders
g = torch.Generator()
g.manual_seed(RAND_SEED)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=16, pin_memory=True, worker_init_fn=misc.seed_worker, generator=g)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=16, pin_memory=True, worker_init_fn=misc.seed_worker, generator=g)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=16, pin_memory=True, worker_init_fn=misc.seed_worker, generator=g)


# initalize model object and load model parameters into optimizer
model = models.GNN(NUM_CLASSES, FOURIER_ORDER, WIDTH_MULTIPLE, EDGE_ATTR_DIM, SKIP, DEEP)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.to(DEVICE)

current_epoch, best_acc, plateau_len = misc.get_train_state(model, optim, args.resume, CHECK_PATH)

if not args.test_only:
    misc.train_graph(EXP_NAME, current_epoch, EPOCHS, best_acc, plateau_len, train_loader, val_loader, model, LOSS_FN, optim, CHECK_PATH, BEST_PATH, DEVICE)
 
if not args.skip_test:
    misc.test_graph(model, BEST_PATH, RAND_SEED, test_loader, DEVICE)
