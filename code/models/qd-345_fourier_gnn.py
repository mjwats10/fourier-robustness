import os
import torch
import torch.nn.functional as F
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
EXP_NAME = 'qd-345_256_fourier_gnn_wide'
LOAD_PATH = '/home/apg/Desktop/mw/fourier/models/' + EXP_NAME + '.pt'
SAVE_PATH = '/home/apg/Desktop/mw/fourier/models/' + EXP_NAME + '.pt'
TRAIN_DATA = '/home/apg/Desktop/mw/fourier/qd-345/train/'
TEST_DATA = '/home/apg/Desktop/mw/fourier/qd-345/test/'
# LOG_PATH = '/home/apg/Desktop/mw/fourier/logs/' + EXP_NAME + '.txt'
RAND_SEED = 0
DEVICE = "cuda:1"

IMG_SIDE = 256
NUM_CLASSES = 345
EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
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
  
means = np.asarray([[1.00000000e+00,  8.64704953e-20,  4.23785772e-20, -5.55188263e-01],
 [-4.21131398e-03, -8.73318590e-04, -7.46787944e-03, -1.16146910e-02],
 [ 3.45495835e-02, -2.98804448e-04,  1.64286745e-04, -5.45286175e-02],
 [-5.87583192e-06, -2.53474884e-04, -6.50432941e-05,  3.94554346e-03],
 [ 5.82578165e-03, -6.83578601e-04,  1.32397937e-04, -7.64260417e-03],
 [ 3.16973027e-05, -1.02179629e-04, -3.79339444e-04, -4.74683571e-04],
 [ 1.34949896e-03, -1.36386428e-04,  1.15424560e-04, -1.54791375e-03],
 [ 6.86240768e-05, -5.95330926e-05,  3.42253351e-05,  5.09141139e-04],
 [ 1.14870610e-03, -4.14005821e-05,  2.85670771e-05, -5.24566362e-04],
 [ 1.22019705e-04, -8.23679948e-05, -9.08114311e-05, -8.45407870e-05]])

stdevs = np.asarray([[1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 2.28322956e-01],
 [1.23763294e-01, 1.30004807e-01, 2.44674213e-01, 1.36764044e-01],
 [8.84229438e-02, 6.22169306e-02, 7.98833353e-02, 8.84160503e-02],
 [4.84725535e-02, 4.71595999e-02, 5.60523362e-02, 5.11812694e-02],
 [3.88979383e-02, 3.34160340e-02, 3.58259993e-02, 3.86328886e-02],
 [2.44812410e-02, 2.43872982e-02, 2.82021738e-02, 2.67411865e-02],
 [1.97384502e-02, 1.81130527e-02, 2.00832990e-02, 2.08832311e-02],
 [1.45924819e-02, 1.45891669e-02, 1.58764064e-02, 1.52681562e-02],
 [1.22818532e-02, 1.17304880e-02, 1.22845295e-02, 1.25736286e-02],
 [9.61639517e-03, 9.74633403e-03, 1.03337422e-02, 1.01022058e-02]])

# transform functions - take sketch image, return torch tensor of descriptors
def fourier_transform(vector_img, is_test):
  stroke_rasters = vector_to_raster(vector_img)

  # add rotations and translations at test time
  if is_test: 
    stroke_rasters = np.stack(stroke_rasters)
    stroke_rasters = torch.from_numpy(stroke_rasters).float()

    angle = random.random()*60 - 30
    deltaX = random.randint(-3, 3)
    deltaY = random.randint(-3, 3)

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
    
    
# pytorch-geometric GCN
class GCN(torch.nn.Module):
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
    print(f"\nepoch avg train loss: {total_loss / ((size // BATCH_SIZE) + 1):.7f}   epoch avg train accuracy: {total_correct / size:.7f}")
      
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
test_imgs = []
train_counts = []
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
  test_folder = TEST_DATA + item + '.bin'
  train_drawings = unpack_drawings(train_folder)
  train_imgs += train_drawings
  train_counts.append(len(train_drawings))
  test_drawings = unpack_drawings(test_folder)
  test_imgs += test_drawings
  test_counts.append(len(test_drawings))
  
#-------------------------------------------------------------------------------------------
  
# seed RNGs
torch.manual_seed(RAND_SEED)
random.seed(RAND_SEED)

# create datasets
train_data = QuickdrawDataset(train_imgs, train_counts, is_test=False)
rand_test_data = QuickdrawDataset(test_imgs, test_counts, is_test=True)
# adv_test_data = QuickdrawDataset(test_imgs, test_counts, is_test=False)

# create dataloaders
g = torch.Generator()
g.manual_seed(RAND_SEED)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
rand_test_loader = DataLoader(rand_test_data, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, generator=g)
# adv_test_loader = DataLoader(adv_test_data, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, generator=g)

# init model and optimizer
model = GCN()
# checkpoint = torch.load(LOAD_PATH, map_location=torch.device(DEVICE))
# model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optim.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
epoch = 0

# train for EPOCHS number of epochs
for i in range(epoch, EPOCHS):
    print("Epoch " + str(i + 1) + "\n")
    train_loop(dataloader=train_loader,model=model,loss_fn=LOSS_FN,optimizer=optim)
    torch.save({
                'epoch': i + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict()
                }, SAVE_PATH)
    print("\n-------------------------------\n")
    
# evaluate on random translations and rotations
print("Evaluating against random transformations...")
random.seed(RAND_SEED)
accuracies = []
for i in range(30):
  accuracies.append(rand_test_loop(dataloader=rand_test_loader,model=model))
accuracies = np.asarray(accuracies)
mean = np.mean(accuracies)
std = np.std(accuracies)
print(f"Mean acc: {mean:.4f}")
print(f"Acc std: {std:.7f}")
# print("\n-------------------------------\n")

# evaluate on adversarial translations and rotations
# print("Evaluating against adversarial transformations...")
