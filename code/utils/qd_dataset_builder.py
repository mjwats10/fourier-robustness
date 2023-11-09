import argparse
import os
from torchvision import transforms as T
import random
import cv2
import numpy as np
import pyefd
import subprocess
from struct import pack, unpack, error
from code.modules import misc, datasets

#-----------------------------------------------------------------

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset_id")
args = parser.parse_args()

# const vars
DATASET_ID = args.dataset_id
ROOT_DIR = os.getcwd()
TRAIN_DIR = ROOT_DIR + f'/qd{DATASET_ID}/train/'
VAL_DIR = ROOT_DIR + f'/qd{DATASET_ID}/val/'
TEST_DIR = ROOT_DIR + f'/qd{DATASET_ID}/test/'
FOURIER_ORDER = 20
IMG_SIDE = 256 if DATASET_ID == "345" else 28
PADDING = 62 if IMG_SIDE == 256 else 96
TRANS_DIST = 10 if IMG_SIDE == 256 else 3
RAND_SEED = 0
NUM_TRAIN = 1000
NUM_VAL = 1000 if DATASET_ID == "345" else 100
NUM_TEST = NUM_VAL
LIST_OF_CLASSES = ["circle", "square", "triangle"]
if DATASET_ID == "345":
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

def sample_drawings(drawings, num_train, num_val, num_test):
    random.seed(RAND_SEED)
    all_sampled = random.sample(drawings, k=(num_train+num_val+num_test))
    train = all_sampled[:num_train]
    val = all_sampled[num_train:(num_train+num_val)]
    test = all_sampled[(num_train+num_val):]
    return train, val, test

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
    drawings = datasets.unpack_drawings(dest)
    train, val, test = sample_drawings(drawings, NUM_TRAIN, NUM_VAL, NUM_TEST)

    train_file = TRAIN_DIR + item + '.bin'
    val_file = VAL_DIR + item + '.bin'
    test_file = TEST_DIR + item + '.bin'
    datasets.pack_drawings(train_file, train)
    datasets.pack_drawings(val_file, val)
    datasets.pack_drawings(test_file, test)
    subprocess.run(f"rm '{dest}'", shell=True)

#-----------------------------------------------------------------

# Define transformation(s) to be applied to dataset
transforms_tensor = T.ToTensor()

# transform functions - take sketch image, return torch tensor of descriptors
def transform(vector_img, data_split):
    raster = misc.vector_to_raster(vector_img, IMG_SIDE, PADDING)

    # add rotations and translations
    if data_split == "val":
        raster = transforms_tensor(raster.astype(np.float32))
        angle = random.random()*30 - 30
        deltaX = random.randint(-TRANS_DIST, 0)
        deltaY = random.randint(-TRANS_DIST, 0)

        raster = T.functional.affine(raster, angle, [deltaX, deltaY], 1, 0,
                                    interpolation=T.InterpolationMode.BILINEAR)
        raster = np.squeeze(raster.numpy()).astype(np.uint8)
    elif data_split == "test":
        raster = transforms_tensor(raster.astype(np.float32))
        angle = random.random()*30
        deltaX = random.randint(0, TRANS_DIST)
        deltaY = random.randint(0, TRANS_DIST)

        raster = T.functional.affine(raster, angle, [deltaX, deltaY], 1, 0,
                                    interpolation=T.InterpolationMode.BILINEAR)
        raster = np.squeeze(raster.numpy()).astype(np.uint8)
  
    raster_binary = cv2.threshold(raster, 100, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(raster_binary, 
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    contour_lens = [len(contour) for contour in contours]
    largest_index = contour_lens.index(max(contour_lens))

    contour = np.asarray(contours[largest_index]).squeeze()
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True)

#-----------------------------------------------------------------

# helper func for removing bad imgs
def remove_bad_imgs(imgset, data_split, class_name):
  bad_imgs = []

  for i, img in enumerate(imgset):
    try:
      transform(img, data_split)
    except Exception as e:
      print(f"Removing image at position {i} from class {class_name} due to exception:")
      bad_imgs.append(i)
      print(repr(e))

  for idx in reversed(bad_imgs):
    del imgset[idx]
  
  return imgset

# remove bad imgs from dataset
random.seed(RAND_SEED)
print("Searching for empty images to remove...")
for i in range(5):
    for file_name in os.listdir(TRAIN_DIR):
        folder = TRAIN_DIR + file_name
        drawings = datasets.unpack_drawings(folder)
        drawings = remove_bad_imgs(drawings, "train", file_name[:-4])
        datasets.pack_drawings(folder, drawings)
        folder = VAL_DIR + file_name
        drawings = datasets.unpack_drawings(folder)
        drawings = remove_bad_imgs(drawings, "val", file_name[:-4])
        datasets.pack_drawings(folder, drawings)
        folder = TEST_DIR + file_name
        drawings = datasets.unpack_drawings(folder)
        drawings = remove_bad_imgs(drawings, "test", file_name[:-4])
        datasets.pack_drawings(folder, drawings)
print("Dataset preparation complete.")