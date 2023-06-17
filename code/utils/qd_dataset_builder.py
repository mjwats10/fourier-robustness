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
import subprocess
from struct import pack, unpack, error

#-----------------------------------------------------------------

# const vars
DATASET_ID = "345"
ROOT_DIR = "/home/matt/fourier/"
TRAIN_DIR = ROOT_DIR + f'qd-{DATASET_ID}/train/'
VAL_DIR = ROOT_DIR + f'qd-{DATASET_ID}/val/'
TEST_DIR = ROOT_DIR + f'qd-{DATASET_ID}/test/'
FOURIER_ORDER = 20
IMG_SIDE = 256
PADDING = 62
RAND_SEED = 0
NUM_TRAIN = 1000
NUM_VAL = 100
NUM_TEST = 100
# LIST_OF_CLASSES = ["circle", "square", "triangle"]
LIST_OF_CLASSES = ["The Eiffel Tower", "The Great Wall of China", "The Mona Lisa",
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

#-----------------------------------------------------------------

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
  drawings = []
  with open(filename, 'rb') as f:
      while True:
          try:
              drawings.append(unpack_drawing(f))
          except error:
              break
  return drawings

def sample_drawings(drawings, num_train, num_val, num_test):
  all_sampled = random.sample(drawings, k=(num_train+num_val+num_test))
  train = all_sampled[:num_train]
  val = all_sampled[num_train:(num_train+num_val)]
  test = all_sampled[(num_train+num_val):]
  return train, val, test

def pack_drawings(filename, drawings):
  with open(filename, 'wb') as f:
    for drawing in drawings:
      f.write(pack('15x'))
      num_strokes = len(drawing)
      f.write(pack('H', num_strokes))
      for stroke in drawing:
        stroke_len = len(stroke[0])
        f.write(pack('H', stroke_len))
        fmt = str(stroke_len) + 'B'
        f.write(pack(fmt, *stroke[0]))
        f.write(pack(fmt, *stroke[1]))


#-----------------------------------------------------------------

# download, sample, and create dataset
os.makedirs(TRAIN_DIR)
os.makedirs(VAL_DIR)
os.makedirs(TEST_DIR)
for item in LIST_OF_CLASSES:
  url = 'gs://quickdraw_dataset/full/binary/' + item + '.bin'
  print(url)  
  dest = ROOT_DIR + item + '.bin'
  subprocess.run(f"gsutil -m cp '{url}' '{dest}'", shell=True)
  drawings = unpack_drawings(dest)
  train, val, test = sample_drawings(drawings, NUM_TRAIN, NUM_VAL, NUM_TEST)

  train_file = TRAIN_DIR + item + '.bin'
  val_file = VAL_DIR + item + '.bin'
  test_file = TEST_DIR + item + '.bin'
  pack_drawings(train_file, train)
  pack_drawings(val_file, val)
  pack_drawings(test_file, test)
  subprocess.run(f"rm '{dest}'", shell=True)

#-----------------------------------------------------------------

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

# Define transformation(s) to be applied to dataset
transforms_tensor = T.ToTensor()

# transform functions - take sketch image, return torch tensor of descriptors
def transform(vector_img, is_test):
    raster = vector_to_raster(vector_img)

  # add rotations and translations
    raster = transforms_tensor(raster.astype(np.float32))
    if is_test:
        angle = random.random()*60 - 30
        deltaX = random.randint(-3, 3)
        deltaY = random.randint(-3, 3)
    else:
        angle = random.random()*30 - 30
        deltaX = random.randint(-3, 0)
        deltaY = random.randint(-3, 0)

    raster = T.functional.affine(raster, angle, [deltaX, deltaY], 1, 0,
                                 interpolation=T.InterpolationMode.BILINEAR)
    raster = np.squeeze(raster.numpy()).astype(np.uint8)
  
    raster_binary = cv2.threshold(raster, 100, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(raster_binary, 
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    largest_size = 0
    largest_index = 0
    for k, contour in enumerate(contours):
        if len(contour) > largest_size:
            largest_size = len(contour)
            largest_index = k

    contour = np.asarray(contours[largest_index]).squeeze()
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True)

#-----------------------------------------------------------------

# helper func for removing bad imgs
def remove_bad_imgs(imgset, is_test):
  bad_imgs = []

  for i, img in enumerate(imgset):
    try:
      transform(img, is_test)
    except Exception as e:
      print(i)
      bad_imgs.append(i)
      print(repr(e))

  for idx in reversed(bad_imgs):
    del imgset[idx]
  
  return imgset

# remove bad imgs from dataset
random.seed(RAND_SEED)
for i in range(5):
    for item in LIST_OF_CLASSES:
        print(item)
        folder = TRAIN_DIR + item + '.bin'
        drawings = unpack_drawings(folder)
        drawings = remove_bad_imgs(drawings, False)
        pack_drawings(folder, drawings)
        folder = VAL_DIR + item + '.bin'
        drawings = unpack_drawings(folder)
        drawings = remove_bad_imgs(drawings, False)
        pack_drawings(folder, drawings)
        folder = TEST_DIR + item + '.bin'
        drawings = unpack_drawings(folder)
        drawings = remove_bad_imgs(drawings, True)
        pack_drawings(folder, drawings)