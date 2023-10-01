import torch
from torch import nn
from torchvision import datasets, models
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import numpy as np
import pyefd
import cv2
import struct
from struct import unpack
import cairocffi as cairo

# Const vars
EXP_NAME = 'qd-3_four_mispredict_cnn5-2'
CHECK_PATH = '/home/matt/fourier/models/' + EXP_NAME + '_check.pt'
BEST_PATH = '/home/matt/fourier/models/' + EXP_NAME + '_best.pt'
TRAIN_DATA = '/home/matt/fourier/qd-3/train/'
VAL_DATA = '/home/matt/fourier/qd-3/val/'
TEST_DATA = '/home/matt/fourier/qd-3/test/'

FOURIER_ORDER = 1
IMG_SIDE = 28
IMG_CENTER = np.asarray(((IMG_SIDE - 1) / 2, (IMG_SIDE - 1) / 2))
PADDING = 62 if IMG_SIDE == 256 else 96
RAND_SEED = 0
DEVICE = "cuda:0"
NUM_CLASSES = 3
EPOCHS = 90
LEARNING_RATE = 1e-3
BATCH_SIZE = 500
LOSS_FN = nn.CrossEntropyLoss()

#-------------------------------------------------------------------------------------------

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

# function to ensure deterministic worker re-seeding for reproduceability
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
    # apply random corrupting transformation to input img
    raster = vector_to_raster(vector_img)
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

def transform_test(vector_img):
    # apply random corrupting transformation to input img
    raster = vector_to_raster(vector_img)
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

# helper method to find class based on imgset index
def find_class(idx, count_list):
    class_id = 0
    sum = count_list[class_id]
    while idx >= sum:
        class_id += 1
        sum += count_list[class_id]
    return class_id


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
        x = transform_test(img) if self.is_test else transform_train(img)
        y = find_class(idx, self.counts)
        return x, y


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.maxpool = nn.MaxPool2d(2) 
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 4 * 4, 384)
        self.head = nn.Linear(384, NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        return self.head(x)


# train_loop is called once each epoch and trains model on training set
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
      if batch % 12 == 0:
          current = (batch + 1) * BATCH_SIZE
          print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

      pred = out.argmax(dim=1, keepdim=True)
      correct = pred.eq(y.view_as(pred)).sum().item()
      total_correct += correct
      total_loss += loss_val
      # print(f"train loss: {loss_val:>7f}   train accuracy: {correct / BATCH_SIZE:.7f}   [batch: {batch + 1:3d}/{(size // BATCH_SIZE) + 1:3d}]")      
    print(f"\nepoch avg train loss: {total_loss / ((size // BATCH_SIZE) + 1):.7f}   epoch avg train accuracy: {total_correct / size:.7f}")

# rand_test_loop evaluates model performance on test set with random affine transformations
def rand_test_loop(dataloader, model):
  model.eval()
  size = len(dataloader.dataset)
  with torch.no_grad():
    total_correct = 0
    for x, y in dataloader:
      x = x.to(DEVICE)
      out = model(x)
      out = out.to("cpu")
      pred = out.argmax(dim=1, keepdim=True)
      total_correct += pred.eq(y.view_as(pred)).sum().item()

    accuracy = total_correct / size
    return accuracy

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

# init lists
train_imgs = []
val_imgs = []
test_imgs = []
train_counts = []
val_counts = []
test_counts = []
list_of_classes = ["circle", "square", "triangle"]

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

# create train, eval, and test datasets
train_data = QuickdrawDataset(train_imgs, train_counts, is_test=False)
val_data = QuickdrawDataset(val_imgs, val_counts, is_test=False)
test_data = QuickdrawDataset(test_imgs, test_counts, is_test=True)

# create generator for dataloaders and create dataloaders for each dataset
g = torch.Generator()
g.manual_seed(RAND_SEED)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)

# initalize model object and load model parameters into optimizer
model = CNN()
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# checkpoint = torch.load(CHECK_PATH, map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
# optim.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# best_acc = checkpoint['best_acc']
# plateau_len = checkpoint['plateau_len']
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
                'best_acc': best_acc,
                'plateau_len': plateau_len,
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
