import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision import models
import random
import cv2
import numpy as np
import cairocffi as cairo
import struct
from struct import unpack

# Env Vars
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True

# Const vars
EXP_NAME = 'qd-345_shufflenet_256'
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

# Define transformation(s) to be applied to dataset-
transforms_norm = T.Compose(
      [
          T.ToTensor(), # scales integer inputs in the range [0, 255] into the range [0.0, 1.0]
          T.Normalize(mean=(0.138), std=(0.296)) # Quickdraw mean and stdev (35.213, 75.588), divided by 255
      ]
  )

# transform functions - take sketch image, return torch tensor of descriptors
def transform(vector_img, is_test):
  raster = vector_to_raster(vector_img)
  raster = transforms_norm(raster)

  # add rotations and translations at test time
  if is_test: 
    angle = random.random()*60 - 30
    deltaX = random.randint(-10, 10)
    deltaY = random.randint(-10, 10)

    raster = T.functional.affine(raster, angle, [deltaX, deltaY], 1, 0,
                                 interpolation=T.InterpolationMode.BILINEAR)
  return raster

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
    x = transform(img, self.is_test)
    y = find_class(idx, self.counts)
    return x, y
    
    
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
#         self.conv4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
#         self.conv5 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
#         self.conv6 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
#         self.relu = nn.ReLU()
#         self.fc1 = nn.Linear(128 * 4 * 4, 512)
#         self.head = nn.Linear(512, NUM_CLASSES)
# 
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.conv4(x)
#         x = self.relu(x)
#         x = self.conv5(x)
#         x = self.relu(x)
#         x = self.conv6(x)
#         x = self.relu(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         return self.head(x)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # put the model in train mode
    total_loss = 0
    total_correct = 0
    # for each batch in the training set compute loss and update model parameters
    for batch, (x, y) in enumerate(dataloader):
      x, y = x.to(DEVICE), y.to(DEVICE)
      # Compute prediction and loss
      out = model(x)
      loss = loss_fn(out, y)

      # Backpropagation to update model parameters
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # print current training metrics for user
      y, out, loss = y.to("cpu"), out.to("cpu"), loss.to("cpu")
      loss_val = loss.item()
      if batch % 50 == 0:
          current = (batch + 1) * BATCH_SIZE
          print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

      pred = out.argmax(dim=1, keepdim=True)
      correct = pred.eq(y.view_as(pred)).sum().item()
      total_correct += correct
      total_loss += loss_val
      # print(f"train loss: {loss_val:>7f}   train accuracy: {correct / BATCH_SIZE:.7f}   [batch: {batch + 1:3d}/{(size // BATCH_SIZE) + 1:3d}]")      
    print(f"\nepoch avg train loss: {total_loss / ((size // BATCH_SIZE) + 1):.7f}   epoch avg train accuracy: {total_correct / size:.7f}")
      
def rand_test_loop(dataloader, model):
  model.eval()
  size = len(dataloader.dataset)
  with torch.no_grad():
    total_correct = 0
    for x, y in dataloader:
      x, y = x.to(DEVICE), y.to(DEVICE)
      out = model(x)
      y, out = y.to("cpu"), out.to("cpu")
      pred = out.argmax(dim=1, keepdim=True)
      total_correct += pred.eq(y.view_as(pred)).sum().item()

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
adv_test_loader = DataLoader(adv_test_data, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, generator=g)

# init model and optimizer
model = models.shufflenet_v2_x0_5()
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
print("\n-------------------------------\n")

# evaluate on adversarial translations and rotations
# print("Evaluating against adversarial transformations...")
