import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
import random
import cv2
import numpy as np
import pyefd
import cairocffi as cairo
import struct
from struct import unpack
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

# Env Vars
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True

# Const vars
EXP_NAME = 'qd-345_fourier_gnn'
CHECK_PATH = '/home/matt/fourier/models/' + EXP_NAME + '_check.pt'
BEST_PATH = '/home/matt/fourier/models/' + EXP_NAME + '_best.pt'
TRAIN_DATA = '/home/matt/fourier/qd-3/train/'
VAL_DATA = '/home/matt/fourier/qd-3/val/'
TEST_DATA = '/home/matt/fourier/qd-3/test/'
RAND_SEED = 0
DEVICE = "cuda:0"

IMG_SIDE = 256
NUM_CLASSES = 345
EPOCHS = 90
LEARNING_RATE = 1e-3
BATCH_SIZE = 500
LOSS_FN = nn.CrossEntropyLoss()
FOURIER_ORDER = 10
EDGE_ATTR_DIM = 3

#-------------------------------------------------------------------------------------------

# convert raw vector image to single raster image
def vector_to_raster(vector_image, side=IMG_SIDE, line_diameter=16, padding=80, bg_color=(0,0,0), fg_color=(1,1,1)):
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

  stroke_rasters = []
  for xv, yv in centered:
    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()

    # draw strokes, this is the most cpu-intensive part
    ctx.set_source_rgb(*fg_color)        
    ctx.move_to(xv[0], yv[0])
    for x, y in zip(xv, yv):
        ctx.line_to(x, y)
    ctx.stroke()

    data = surface.get_data()
    stroke_raster = np.copy(np.asarray(data)[::4]).reshape(side, side)
    stroke_rasters.append(stroke_raster)

  return stroke_rasters

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

means = np.asarray([[ 1.00000000e+00,  1.71533837e-19,  2.02469755e-19, -4.55006715e-01],
 [-2.44526802e-03, -4.93971948e-03, -1.98264423e-02, -2.78578175e-03],
 [ 3.60395467e-02,  2.15628834e-04,  4.09271078e-05, -6.30258948e-02],
 [-5.31979694e-05, -1.37478914e-03, -1.18044716e-03,  3.76767090e-04],
 [ 1.43644504e-02, -1.81261427e-04, -2.65732014e-04, -1.59276862e-02],
 [ 2.43380536e-04, -4.44410264e-04, -8.78416008e-04, -4.03222524e-04],
 [ 5.15026434e-03, -6.06479599e-05, -2.86628737e-05, -7.31671622e-03],
 [ 4.01537234e-05, -3.15332675e-04, -2.71719642e-04,  7.17671160e-05],
 [ 2.95224073e-03, -4.87730339e-05, -6.16868229e-05, -3.26723924e-03],
 [ 1.09736565e-04, -1.43306104e-04, -1.90696467e-04, -8.65225066e-05],
 [ 1.50643939e-03, -2.77122722e-05, -2.09268399e-05, -2.01925368e-03],
 [ 4.55949290e-05, -7.20648301e-05, -6.24892901e-05, -3.52932538e-05],
 [ 9.14688593e-04, -1.89704487e-05, -1.76250242e-05, -1.08307005e-03],
 [ 4.22255156e-05, -4.86554753e-05, -4.86603399e-05, -1.40064505e-05],
 [ 5.27235213e-04, -1.53966773e-05, -7.73577605e-06, -7.21721356e-04],
 [ 1.24775445e-05, -2.90296701e-05, -1.18877572e-05, -2.33463846e-05],
 [ 3.59514357e-04, -9.37319382e-06, -7.64706525e-06, -4.09325342e-04],
 [-7.55454498e-06, -1.74757957e-05, -1.85736557e-05, -2.36138420e-05],
 [ 2.49116020e-04, -6.19866431e-06, -2.29169767e-06, -2.85349619e-04],
 [-1.85501339e-06, -7.55794125e-06, -1.87815600e-06,  8.65967382e-06]])

stdev = np.asarray([[1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 2.77546598e-01],
 [1.38050119e-01, 1.49582106e-01, 3.61332724e-01, 1.18704408e-01],
 [1.09599799e-01, 5.09026139e-02, 7.12521591e-02, 5.82314524e-02],
 [4.35713125e-02, 4.16378369e-02, 4.93126833e-02, 3.00215217e-02],
 [3.59612024e-02, 2.54791841e-02, 2.68633370e-02, 2.67202345e-02],
 [1.90335870e-02, 1.92058465e-02, 2.65869452e-02, 2.04681377e-02],
 [1.67783362e-02, 1.28594944e-02, 1.54878290e-02, 1.64947071e-02],
 [1.07586599e-02, 1.16112085e-02, 1.29672720e-02, 1.00573940e-02],
 [1.06438523e-02, 8.90824033e-03, 9.48327812e-03, 9.88491996e-03],
 [7.22011727e-03, 7.54837138e-03, 8.65036067e-03, 7.33773138e-03],
 [6.62049264e-03, 5.79993062e-03, 6.75350800e-03, 7.14011245e-03],
 [5.12461863e-03, 5.32603919e-03, 5.70994397e-03, 5.13566599e-03],
 [4.95992671e-03, 4.46978806e-03, 4.68719281e-03, 5.02644281e-03],
 [3.78816997e-03, 3.93351360e-03, 4.20581102e-03, 3.93743953e-03],
 [3.61098197e-03, 3.27518484e-03, 3.69590012e-03, 3.94969505e-03],
 [2.86181951e-03, 3.00736143e-03, 3.23286279e-03, 3.08371526e-03],
 [2.84544097e-03, 2.64481592e-03, 2.85369602e-03, 3.01824160e-03],
 [2.32088796e-03, 2.39080780e-03, 2.56977786e-03, 2.48161846e-03],
 [2.20677405e-03, 2.09707425e-03, 2.37681990e-03, 2.48725355e-03],
 [1.85566039e-03, 1.92171891e-03, 2.14054129e-03, 2.08908828e-03]])

# transform functions - take sketch image, return torch tensor of descriptors
def fourier_transform(vector_img, is_test):
  stroke_rasters = vector_to_raster(vector_img)

  # add rotations and translations at test time
  if is_test: 
    stroke_rasters = np.stack(stroke_rasters)
    stroke_rasters = torch.from_numpy(stroke_rasters).float()

    angle = random.random()*60 - 30
    deltaX = random.randint(-10, 10)
    deltaY = random.randint(-10, 10)

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

    largest_size = 0
    largest_index = 0
    for k, contour in enumerate(contours):
        if len(contour) > largest_size:
          largest_size = len(contour)
          largest_index = k

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

# helper method to find class based on imgset index
def find_class(idx, count_list):
  class_id = 0
  sum = count_list[class_id]
  while idx >= sum:
    class_id += 1
    sum += count_list[class_id]
  return class_id

# deterministic worker re-seeding
def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)
  

# custom dataset for quickdraw
class QuickdrawDataset(Dataset):
  def __init__(self, imgs, counts, is_test):
    self.imgs = imgs
    self.counts = counts
    self.len = sum(counts)
    self.is_test = is_test

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    img = self.imgs[idx]
    x, edge_index, edge_attr = fourier_transform(img, self.is_test)
    y = find_class(idx, self.counts)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    

# pytorch-geometric GCN shallow
class GCN(nn.Module):
  def __init__(self):
    super(GCN, self).__init__()
    self.embedding = nn.Sequential(
                                    nn.Linear(FOURIER_ORDER*4, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    nn.ReLU(),
                                )
    self.edge_proj = nn.Sequential(
                                nn.Linear(EDGE_ATTR_DIM, 64),
                                nn.ReLU(),
                                nn.Linear(64, 1),
                                nn.Sigmoid()
                            )
    self.relu = nn.ReLU()
    self.conv6 = GCNConv(512, 1024)
    self.conv7 = GCNConv(1024, 1024)
    self.conv8 = GCNConv(1024, 1024)
    self.fc1 = nn.Linear(1024, 2048)
    self.head = nn.Linear(2048, NUM_CLASSES)

  def forward(self, data):
    x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

    if EDGE_ATTR_DIM > 1:
      edge_attr = edge_attr.squeeze(dim=2)
    edge_weight = self.edge_proj(edge_attr)
    x = self.embedding(x)
    x = self.conv6(x, edge_index, edge_weight)
    x = self.relu(x)
    x = self.conv7(x, edge_index, edge_weight)
    x = self.relu(x)
    x = self.conv8(x, edge_index, edge_weight)
    x = self.relu(x)
    x = global_mean_pool(x, batch)
    x = self.fc1(x)
    x = self.relu(x)
    return self.head(x)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # put the model in train mode
    total_loss = 0
    total_correct = 0
    # for each batch in the training set compute loss and update model parameters
    for batch, data in enumerate(dataloader):
      data = data.to(DEVICE)
      # Compute prediction and loss
      out = model(data)
      loss = loss_fn(out, data.y)

      # Backpropagation to update model parameters
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # print current training metrics for user
      data, out, loss = data.to("cpu"), out.to("cpu"), loss.to("cpu")
      loss_val = loss.item()
      if batch % 50 == 0:
          current = (batch + 1) * BATCH_SIZE
          print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

      pred = out.argmax(dim=1, keepdim=True)
      correct = pred.eq(data.y.view_as(pred)).sum().item()
      total_correct += correct
      total_loss += loss_val
      # print(f"train loss: {loss_val:>7f}   train accuracy: {correct / BATCH_SIZE:.7f}   [batch: {batch + 1:3d}/{(size // BATCH_SIZE) + 1:3d}]")      
    print(f"\nepoch avg train loss: {total_loss / ((size // BATCH_SIZE) + 1):.7f}   epoch avg train accuracy: {total_correct / size:.4f}")
      
def rand_test_loop(dataloader, model):
  model.eval()
  size = len(dataloader.dataset)
  with torch.no_grad():
    total_correct = 0
    for data in dataloader:
      data = data.to(DEVICE)
      out = model(data)
      data, out = data.to("cpu"), out.to("cpu")
      pred = out.argmax(dim=1, keepdim=True)
      total_correct += pred.eq(data.y.view_as(pred)).sum().item()

    accuracy = total_correct / size
    return accuracy
    
# def adv_test_loop(dataloader, model):
#   model.eval()
#   size = len(dataloader.dataset)
#   with torch.no_grad():
#     total_correct = 0
#     for x, y in dataloader:
#         passed = 1
#         for dX in range(-10, 11, 5):
#             for dY in range(-10, 11, 5):
#                 for theta in range(-30, 31, 2):
#                     x = T.functional.affine(x, theta, [dX, dY], 1, 0,
#                                  interpolation=T.InterpolationMode.BILINEAR)
#                     x = x.to(DEVICE)
#                     out = model(x).to("cpu")
#                     pred = out.argmax(dim=1, keepdim=True)
#                     passed = pred.eq(y.view_as(pred)).sum().item()
#                     if passed == 0:
#                         break
#                 if passed == 0:
#                     break
#             if passed 
# 
#     accuracy = total_correct / size
#     return accuracy
    
#-------------------------------------------------------------------------------------------

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

train_imgs = []
val_imgs = []
test_imgs = []
train_counts = []
val_counts = []
test_counts = []
list_of_classes = ["The Eiffel Tower", "The Great Wall of China", "The Mona Lisa",
                   "aircraft carrier", "airplane", "alarm clock", "ambulance", 
                   "angel", "animal migration", "ant", "anvil", "apple", "arm", "asparagus", 
                   "axe", "backpack", "banana", "bandage", "barn", "baseball bat", 
                   "baseball", "basket", "basketball", "bat", "bathtub", "beach", "bear", 
                   "beard", "bed", "bee", "belt", "bench", "bicycle", "binoculars", 
                   "bird", "birthday cake", "blackberry", "blueberry", "book", 
                   "boomerang", "bottlecap", "bowtie", "bracelet", "brain", 
                   "bread", "bridge", "broccoli", "broom", "bucket", "bulldozer", 
                   "bus", "bush", "butterfly", "cactus", "cake", "calculator", 
                   "calendar", "camel", "camera", "camouflage", "campfire", 
                   "candle", "cannon", "canoe", 'car', 'carrot', "castle", "cat",  
                   "ceiling fan", "cell phone", "cello", "chair", "chandelier", "church", 
                   "circle", "clarinet", "clock", "cloud", "coffee cup", 
                   "compass", "computer", "cookie", "cooler", "couch", "cow",
                   "crab", "crayon", "crocodile", "crown", "cruise ship", 
                   "cup", "diamond", "dishwasher", "diving board", "dog", 
                   "dolphin", "donut", "door", "dragon", "dresser", "drill", 
                   "drums", "duck", "dumbbell", "ear", "elbow", "elephant", 
                   "envelope", "eraser", "eye", "eyeglasses", "face", "fan",
                   "feather", "fence", "finger", "fire hydrant", "fireplace",
                   "firetruck", "fish", "flamingo", "flashlight", "flip flops", 
                   "floor lamp", "flower", "flying saucer", "foot", "fork", 
                   "frog", "frying pan", "garden hose", "garden", "giraffe", 
                   "goatee", "golf club", "grapes", "grass", "guitar", 
                   "hamburger", "hammer", "hand", "harp", "hat", "headphones", 
                   "hedgehog", "helicopter", "helmet", "hexagon", "hockey puck", 
                   "hockey stick", "horse", "hospital", "hot air balloon", 
                   "hot dog", "hot tub", "hourglass", "house plant", "house", 
                   "hurricane", "ice cream", "jacket", "jail", "kangaroo", 
                   "key", "keyboard", "knee", "knife", "ladder", "lantern", 
                   "laptop", "leaf", "leg", "light bulb", "lighter", "lighthouse",
                   "lightning", "line", "lion", "lipstick", "lobster", "lollipop",
                   "mailbox", "map", "marker", "matches", "megaphone", "mermaid", 
                   "microphone", "microwave", "monkey", "moon", "mosquito", 
                   "motorbike", "mountain", "mouse", "moustache", "mouth", "mug",
                   "mushroom", "nail", "necklace", "nose", "ocean", "octagon", 
                   "octopus", "onion", "oven", "owl", "paint can", "paintbrush", 
                   "palm tree", "panda", "pants", "paper clip", "parachute", 
                   "parrot", "passport", "peanut", "pear", "peas", "pencil", 
                   "penguin", "piano", "pickup truck", "picture frame", "pig", 
                   "pillow", "pineapple", "pizza", "pliers", "police car", 
                   "pond", "pool", "popsicle", "postcard", "potato", 
                   "power outlet", "purse", "rabbit", "raccoon", "radio", 
                   "rain", 'rainbow', 'rake', 'remote control', 'rhinoceros', 
                   'rifle', 'river', 'roller coaster', 'rollerskates', 
                   'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 
                   'scissors', 'scorpion', 'screwdriver', 'sea turtle', 
                   'see saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 
                   'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 
                   'smiley face', 'snail', 'snake', 'snorkel', 'snowflake', 
                   'snowman', 'soccer ball', 'sock', 'speedboat', 'spider', 
                   'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 
                   'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 
                   'stop sign', 'stove', 'strawberry', 'streetlight', 
                   'string bean', 'submarine', 'suitcase', 'sun', 'swan', 
                   'sweater', 'swing set', 'sword', 'syringe', 't-shirt', 
                   'table', 'teapot', 'teddy-bear', 'telephone', 'television', 
                   'tennis racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 
                   'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 
                   'traffic light', 'train', 'tree', 'triangle', 'trombone', 
                   'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 
                   'washing machine', 'watermelon', 'waterslide', 'whale', 
                   'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 
                   'yoga', 'zebra', 'zigzag']
                   
#-------------------------------------------------------------------------------------------

# load dataset
for item in list_of_classes:
    train_folder = TRAIN_DATA + item + '.bin'
    train_drawings = unpack_drawings(train_folder)
    train_imgs += train_drawings
    train_counts.append(len(train_drawings))
    val_folder = VAL_DATA + item + '.bin'
    val_drawings = unpack_drawings(val_folder)
    val_imgs += val_drawings
    val_counts.append(len(val_drawings))
    test_folder = TEST_DATA + item + '.bin'
    test_drawings = unpack_drawings(test_folder)
    test_imgs += test_drawings
    test_counts.append(len(test_drawings))
  
#-------------------------------------------------------------------------------------------
  
# seed RNGs
torch.manual_seed(RAND_SEED)
random.seed(RAND_SEED)

# create datasets
train_data = QuickdrawDataset(train_imgs, train_counts, is_test=False)
val_data = QuickdrawDataset(val_imgs, val_counts, is_test=False)
test_data = QuickdrawDataset(test_imgs, test_counts, is_test=True)

# create dataloaders
g = torch.Generator()
g.manual_seed(RAND_SEED)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)


# init model and optimizer
model = GCN()
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# checkpoint = torch.load(CHECK_PATH, map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
# optim.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# best_acc = checkpoint['best_acc']
epoch = 0
best_acc = 0
plateau_len = 0

model.to(DEVICE)

# train for EPOCHS number of epochs
print(EXP_NAME)
for i in range(epoch, EPOCHS):
    if plateau_len >= 10:
        break
    print("Epoch " + str(i + 1) + "\n")
    train_loop(dataloader=train_loader,model=model,loss_fn=LOSS_FN,optimizer=optim)
    torch.save({
                'epoch': i + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict()
                }, CHECK_PATH)
    acc = rand_test_loop(dataloader=val_loader,model=model)
    if acc > best_acc:
        torch.save(model.state_dict(), BEST_PATH)
        best_acc = acc
        plateau_len = 0
    else:
        plateau_len += 1
    print(f"best val acc: {best_acc:.4f}")
    print("\n-------------------------------\n")
 
# evaluate on random translations and rotations
print("Evaluating against random transformations...")
model.load_state_dict(torch.load(BEST_PATH))
random.seed(RAND_SEED)
accuracies = []
for i in range(30):
  accuracies.append(rand_test_loop(dataloader=test_loader,model=model))
accuracies = np.asarray(accuracies)
mean = np.mean(accuracies)
std = np.std(accuracies)
print(f"Mean acc: {mean:.4f}")
print(f"Acc std: {std:.7f}")
print("\n-------------------------------\n")
