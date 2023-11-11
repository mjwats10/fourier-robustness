import argparse
import torch
from torch import nn
from torchvision import datasets, models
from torchvision import transforms as T
from torch.utils.data import DataLoader
import random
import numpy as np
import pyefd
import cv2
import matplotlib.pyplot as plt
import os
from skimage.util import random_noise

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("root_path")
parser.add_argument("device")
parser.add_argument("rand_seed", type=int)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("f_order", type=int)
args = parser.parse_args()

# Const vars
EXP_NAME = 'mnist_noise_fourier_mlp'
ROOT_PATH = args.root_path
CHECK_PATH = ROOT_PATH + '/models/first_draft/' + EXP_NAME + '_check.pt'
BEST_PATH = ROOT_PATH + '/models/first_draft/' + EXP_NAME + '_best.pt'
MNIST_DATA = ROOT_PATH + '/mnist'

FOURIER_ORDER = args.f_order
RAND_SEED = args.rand_seed
DEVICE = args.device
NUM_CLASSES = 10
EPOCHS = 90
LEARNING_RATE = 1e-3
BATCH_SIZE = 500
LOSS_FN = nn.CrossEntropyLoss()
RNG = np.random.default_rng(seed=RAND_SEED)

# function to ensure deterministic worker re-seeding for reproduceability
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

means = np.asarray([[ 3.60261012e-01, -2.71080355e+00, -6.95606058e+00, -1.42660428e-01],
 [-2.45223644e-01, -5.17519985e-01, -7.47533459e-01,  2.56881480e-01],
 [ 1.97253010e-01, -5.65378079e-01, -5.33253625e-01,  2.52081157e-01],
 [-5.59832851e-02, -1.87748606e-01, -1.52407562e-01,  3.29363470e-02],
 [-2.00347684e-02, -1.16811042e-01, -1.56683080e-01,  3.91660968e-02],
 [ 2.56742871e-03, -3.05435712e-02, -8.24459107e-02,  3.38987622e-02],
 [ 1.36420429e-03, -3.40296498e-02, -6.49440445e-02,  2.13789083e-02],
 [-4.98731456e-03, -1.02267172e-02, -3.96408186e-02,  1.31229927e-02],
 [-5.34427935e-03, -1.31757914e-02, -4.06433571e-02,  1.03529157e-02],
 [-2.35582777e-03, -4.00707053e-03, -2.79016316e-02,  1.07480738e-02],
 [-2.98114287e-03, -7.04266392e-03, -2.59436232e-02,  8.48903510e-03],
 [-3.68090164e-03, -2.10842749e-03, -2.06085758e-02,  8.03354835e-03],
 [-3.53799889e-03, -3.47509141e-03, -1.89682100e-02,  6.89639067e-03],
 [-3.19977292e-03, -1.16053372e-03, -1.60456443e-02,  6.61983443e-03],
 [-2.76660305e-03, -1.80497563e-03, -1.50024954e-02,  5.44132073e-03],
 [-2.38257793e-03, -1.06859403e-04, -1.33691463e-02,  5.06156059e-03],
 [-2.61532648e-03, -5.40560203e-04, -1.26350323e-02,  4.51645103e-03],
 [-2.61135576e-03,  2.48308030e-04, -1.12789225e-02,  4.14992723e-03],
 [-2.38214517e-03,  3.97490444e-05, -1.07571088e-02,  3.68695749e-03],
 [-2.42956997e-03,  4.71281106e-04, -9.78646253e-03,  3.08791742e-03]])

stdevs = np.asarray([[2.93428344, 2.13806785, 1.81275079, 2.61003299],
 [2.11414046, 1.59152828, 1.49434875, 1.4519375 ],
 [1.27508941, 1.26768425, 0.72124805, 0.66391385],
 [0.66963166, 0.67603634, 0.48264867, 0.4752503 ],
 [0.35416702, 0.3526171 , 0.35614925, 0.34520783],
 [0.21994313, 0.2175633 , 0.19336943, 0.18925603],
 [0.16869874, 0.16769815, 0.14798171, 0.14193974],
 [0.11907087, 0.11845328, 0.11411248, 0.11027125],
 [0.09983111, 0.0999523 , 0.0960541 , 0.09159265],
 [0.0822623 , 0.08151865, 0.0778165 , 0.07411436],
 [0.07132651, 0.070599  , 0.06647898, 0.06345617],
 [0.05987423, 0.05982594, 0.055892  , 0.05396168],
 [0.05371287, 0.0531472 , 0.04910345, 0.04807303],
 [0.04695223, 0.0468781 , 0.04316355, 0.04219514],
 [0.04275125, 0.04234907, 0.03896974, 0.03842633],
 [0.03812275, 0.03808251, 0.0350922 , 0.03477163],
 [0.03529667, 0.03504137, 0.03240988, 0.03223823],
 [0.03196547, 0.03193184, 0.02952981, 0.02955216],
 [0.02987713, 0.02940787, 0.02731482, 0.02743727],
 [0.02746033, 0.0272346 , 0.02523833, 0.02517911]])
  
# transform function
def transform_train(img):
    raster = np.array(img) # convert PIL image to numpy array for openCV
    ret, raster = cv2.threshold(raster, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # since some images have artifacts disconnected from the digit, extract only
    # largest contour from the contour list (this should be the digit)
    contour_lens = [len(contour) for contour in contours]
    largest_index = contour_lens.index(max(contour_lens))

    # get translation and rotation offsets
    contour = np.squeeze(contours[largest_index])
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER)
    coeffs = (coeffs - means[:FOURIER_ORDER]) / stdevs[:FOURIER_ORDER]
    return torch.from_numpy(coeffs.flatten()).float()

def transform_test(img):
    raster = np.array(img) # convert PIL image to numpy array for openCV
    raster = random_noise(raster, mode='salt', seed=RNG, amount=0.1)
    raster = np.uint8(255 * raster)
    ret, raster = cv2.threshold(raster, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
    # since some images have artifacts disconnected from the digit, extract only
    # largest contour from the contour list (this should be the digit)
    contour_lens = [len(contour) for contour in contours]
    largest_index = contour_lens.index(max(contour_lens))

    # get translation and rotation offsets
    contour = np.squeeze(contours[largest_index])
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER)
    coeffs = (coeffs - means[:FOURIER_ORDER]) / stdevs[:FOURIER_ORDER]
    return torch.from_numpy(coeffs.flatten()).float()


class MNIST_VAL(datasets.MNIST):
    def __init__(
            self, 
            root: str, 
            train: bool = True, 
            val: bool = False,
            transform = None,
            target_transform = None,
            download: bool = False):
        super().__init__(
            root,
            train = train,
            transform = transform,
            target_transform = target_transform,
            download = download)
        if val:
            self.data = self.data[50000:]
            self.targets = self.targets[50000:]
        elif train:
            self.data = self.data[:50000]
            self.targets = self.targets[:50000]


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
        batch_num = 0
        all_incorrect = np.empty(0, dtype=np.int64)
        for x, y in dataloader:
            x = x.to(DEVICE)
            out = model(x)
            out = out.to("cpu")
            pred = out.argmax(dim=1, keepdims=True)
            correct_mask = pred.eq(y.view_as(pred))
            total_correct += correct_mask.sum().item()
            batch_incorrect = batch_num * BATCH_SIZE + np.nonzero(correct_mask.numpy() == False)[0]
            all_incorrect = np.concatenate([all_incorrect, batch_incorrect])
            batch_num += 1

        accuracy = total_correct / size
        return accuracy, all_incorrect

# seed RNGs
torch.manual_seed(RAND_SEED)
random.seed(RAND_SEED)

# create train, eval, and test datasets
train_data = MNIST_VAL(root=MNIST_DATA, train=True, val=False, download=True, transform=transform_train)
val_data = MNIST_VAL(root=MNIST_DATA, train=True, val=True, download=True, transform=transform_train)
test_data = MNIST_VAL(root=MNIST_DATA, train=False, download=True, transform=transform_test) 

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
model = FourierMLP()
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.to(DEVICE)

epoch = 0
best_acc = 0
plateau_len = 0
if args.resume:
    checkpoint = torch.load(CHECK_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    plateau_len = checkpoint['plateau_len']

if not args.skip_train:
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
        acc, __ = rand_test_loop(dataloader=val_loader,model=model)
        if acc > best_acc:
            torch.save(model.state_dict(), BEST_PATH)
            best_acc = acc
            plateau_len = 0
        else:
            plateau_len += 1
        print(f"best val acc: {best_acc:.4f}")
        print("\n-------------------------------\n")
 
if not args.skip_test:
    # evaluate on random translations and rotations
    print("Evaluating against random transformations...")
    model.load_state_dict(torch.load(BEST_PATH))
    random.seed(RAND_SEED)
    accuracies = []
    incorrect_counts = np.zeros(len(test_data), dtype=np.int64)
    for i in range(30):
        accuracy, incorrect_idx = rand_test_loop(dataloader=test_loader,model=model)
        accuracies.append(accuracy)
        incorrect_counts[incorrect_idx] += 1
    accuracies = np.asarray(accuracies)
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    print(f"Mean acc: {mean:.4f}")
    print(f"Acc std: {std:.7f}")
    print("\n-------------------------------\n")

    if RAND_SEED == 0:
        test_imgs = MNIST_VAL(root=MNIST_DATA, train=False, download=True)
        RNG = np.random.default_rng(seed=RAND_SEED)
        worst = RNG.choice(np.nonzero(incorrect_counts == 30)[0], size=9, replace=False)
        fig_save = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop', EXP_NAME)
        os.mkdir(fig_save)
        for i in range(9):
            img = random_noise(np.array(test_imgs[worst[i]][0]), mode='salt', seed=RNG, amount=0.1)
            plt.imshow(255*img,cmap='gray',vmin=0,vmax=255)
            plt.title(f"\"{test_imgs[worst[i]][1]}\"")
            plt.savefig(os.path.join(fig_save, str(worst[i])))