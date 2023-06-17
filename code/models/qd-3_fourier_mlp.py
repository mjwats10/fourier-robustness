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

# Env Vars
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True

# Const vars
EXP_NAME = 'qd-3_fourier_mlp'
SERVER = "matt"
if SERVER == "apg":
    CHECK_PATH = '/home/apg/mw/fourier/models/' + EXP_NAME + '_check.pt'
    BEST_PATH = '/home/apg/mw/fourier/models/' + EXP_NAME + '_best.pt'
    TRAIN_DATA = '/home/apg/mw/fourier/qd-3/train/'
    VAL_DATA = '/home/apg/mw/fourier/qd-3/val/'
    TEST_DATA = '/home/apg/mw/fourier/qd-3/test/'
else:
    CHECK_PATH = '/home/matt/fourier/models/' + EXP_NAME + '_check.pt'
    BEST_PATH = '/home/matt/fourier/models/' + EXP_NAME + '_best.pt'
    TRAIN_DATA = '/home/matt/fourier/qd-3/train/'
    VAL_DATA = '/home/matt/fourier/qd-3/val/'
    TEST_DATA = '/home/matt/fourier/qd-3/test/'

FOURIER_ORDER = 10
IMG_SIDE = 28
PADDING = 80 if IMG_SIDE == 256 else 96
RAND_SEED = 0
DEVICE = "cuda:0"
NUM_CLASSES = 10
EPOCHS = 30
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

# Define transformation(s) to be applied to dataset
transforms_tensor = T.ToTensor()

means = np.asarray([[ 1.00000000e+00,  4.16406170e-19,  9.77419207e-19, -7.44817055e-01],
 [-2.00517883e-03,  1.62853060e-03, -1.95138117e-02,  3.03552236e-03],
 [-1.78710047e-02,  1.39130907e-04,  4.87185474e-04, -2.92268649e-02],
 [ 1.66247641e-03,  1.34981117e-03,  5.05098512e-04,  2.12908508e-03],
 [-7.41603839e-03,  1.42708178e-04, -3.09119432e-04, -1.77971587e-04],
 [ 4.80272874e-04,  2.83400981e-04, -4.17955330e-04, -1.41998075e-04],
 [-8.95517730e-04,  2.43358096e-04, -2.18976984e-04,  4.80068848e-04],
 [-2.10081381e-04,  8.74664807e-05,  1.81554514e-04,  6.83961503e-04],
 [-4.70502965e-04,  1.00168156e-04,  5.88821386e-05, -3.99345327e-04],
 [ 2.36248428e-04, -2.87669121e-06, -3.69962737e-04,  1.91441441e-04],
 [-6.11383641e-04,  7.92999780e-05, -3.14057546e-05, -2.10072312e-04],
 [ 1.97037704e-04, -3.76573793e-06, -1.18934120e-05,  1.97323212e-04],
 [-3.40948714e-04,  6.56489619e-05,  8.63503577e-06, -1.73714117e-04],
 [ 4.41301331e-05, -4.02759344e-05, -8.92786873e-05,  3.60538492e-06],
 [-2.77056223e-04,  4.02131253e-05, -3.99349693e-05, -6.72501753e-05],
 [ 3.82912655e-05, -4.30900384e-06, -3.74108691e-05,  1.53515331e-04],
 [-9.34478270e-05,  4.36248528e-05,  5.01808906e-06, -9.37602081e-05],
 [-8.42319966e-06,  5.01584956e-06, -3.16375931e-05,  3.30142166e-05],
 [-1.09293336e-04,  4.20281108e-05, -2.88703416e-05, -5.03026228e-05],
 [-3.32355252e-05, -1.65997936e-05, -3.16710163e-05,  8.63340775e-05]])

stdevs = np.asarray([[1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 2.42746328e-01],
 [9.87668605e-02, 7.79584479e-02, 2.93220976e-01, 1.08782793e-01],
 [1.41290863e-01, 4.76003823e-02, 5.71932165e-02, 5.34389227e-02],
 [3.95109859e-02, 3.29945398e-02, 3.86822304e-02, 2.52604451e-02],
 [3.81236049e-02, 2.19208078e-02, 2.54658253e-02, 2.23099338e-02],
 [1.71761057e-02, 1.23569567e-02, 1.97042667e-02, 1.31826816e-02],
 [1.52731902e-02, 1.22989309e-02, 1.33010352e-02, 1.26957816e-02],
 [9.64427425e-03, 8.09878894e-03, 1.24646028e-02, 9.95525158e-03],
 [1.07430361e-02, 8.69559790e-03, 8.81042075e-03, 8.23829506e-03],
 [7.50989562e-03, 6.73987266e-03, 8.91317496e-03, 7.13770893e-03],
 [7.01739340e-03, 5.95688576e-03, 6.73674680e-03, 6.52419964e-03],
 [5.28894403e-03, 5.10806731e-03, 5.94828353e-03, 5.53907242e-03],
 [5.63029758e-03, 5.03011536e-03, 5.21372670e-03, 5.14313583e-03],
 [4.28125071e-03, 4.09948451e-03, 5.01585016e-03, 4.75531622e-03],
 [4.14800232e-03, 3.94447855e-03, 4.30656279e-03, 4.24682035e-03],
 [3.58019280e-03, 3.42471444e-03, 3.95273899e-03, 3.80394144e-03],
 [3.32924231e-03, 3.24282519e-03, 3.58340164e-03, 3.52443250e-03],
 [2.90589225e-03, 2.89319573e-03, 3.34816579e-03, 3.28377283e-03],
 [2.72558678e-03, 2.72007711e-03, 2.93452316e-03, 2.92013897e-03],
 [2.41919000e-03, 2.41797797e-03, 2.68644273e-03, 2.65572993e-03]])
  
# transform function - normalize img
def transform_train(vector_img):
    raster_img = vector_to_raster(vector_img)
    ret, raster = cv2.threshold(raster_img, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # since some images have artifacts disconnected from the digit, extract only
    # largest contour from the contour list (this should be the digit)
    largest_size = 0
    largest_index = 0
    for i, contour in enumerate(contours):
        if len(contour) > largest_size:
            largest_size = len(contour)
            largest_index = i

    # get translation and rotation offsets
    contour = np.squeeze(contours[largest_index])
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True)
    coeffs = (coeffs - means[:FOURIER_ORDER]) / stdevs[:FOURIER_ORDER]
    return torch.from_numpy(coeffs.flatten()).float()

def transform_test(vector_img):
    # apply random corrupting transformation to input img
    raster = vector_to_raster(vector_img)
    img = transforms_tensor(raster.astype(np.float32))
    angle = random.random()*60 - 30
    deltaX = random.randint(-3, 3)
    deltaY = random.randint(-3, 3)
    img = T.functional.affine(img, angle, [deltaX, deltaY], 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    img = np.squeeze(img.numpy()).astype(np.uint8)

    ret, raster = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # since some images have artifacts disconnected from the digit, extract only
    # largest contour from the contour list (this should be the digit)
    largest_size = 0
    largest_index = 0
    for i, contour in enumerate(contours):
        if len(contour) > largest_size:
            largest_size = len(contour)
            largest_index = i

    # get translation and rotation offsets
    contour = np.squeeze(contours[largest_index])
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True)
    coeffs = (coeffs - means[:FOURIER_ORDER]) / stdevs[:FOURIER_ORDER]
    return torch.from_numpy(coeffs.flatten()).float()

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


# mlp taking array of normalized fourier descriptors
class FourierMLP(nn.Module):
    def __init__(self):
        super(FourierMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
        nn.Linear(FOURIER_ORDER*4, 512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,NUM_CLASSES))

    def forward(self, x):
        x = self.flatten(x)
        out = self.mlp(x)
        return out


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

train_data = QuickdrawDataset(train_imgs, train_counts, is_test=False)
val_data = QuickdrawDataset(val_imgs, val_counts, is_test=False)
test_data = QuickdrawDataset(test_imgs, test_counts, is_test=True)

# create generator for dataloaders and create dataloaders for each dataset
g = torch.Generator()
g.manual_seed(RAND_SEED)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, generator=g)

# initalize model object and load model parameters into optimizer
model = FourierMLP()
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# checkpoint = torch.load(CHECK_PATH, map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
# optim.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# best_acc = checkpoint['best_acc']
epoch = 0
best_acc = 0

model.to(DEVICE)

# train for EPOCHS number of epochs
print(EXP_NAME)
for i in range(epoch, EPOCHS):
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
    print(f"best acc: {best_acc:.4f}")
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
