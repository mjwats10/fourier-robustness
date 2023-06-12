import torch
from torch import nn
from torchvision import datasets, models
from torchvision import transforms as T
from torch.utils.data import DataLoader
import random
import numpy as np
import pyefd
import cv2

# Env Vars
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True

# Const vars
EXP_NAME = 'mnist_fourier_mlp'
SERVER = "matt"
if SERVER == "apg":
    CHECK_PATH = '/home/apg/mw/fourier/models/' + EXP_NAME + '_check.pt'
    BEST_PATH = '/home/apg/mw/fourier/models/' + EXP_NAME + '_best.pt'
    MNIST_DATA = '/home/apg/mw/fourier/mnist'
else:
    CHECK_PATH = '/home/matt/fourier/models/' + EXP_NAME + '_check.pt'
    BEST_PATH = '/home/matt/fourier/models/' + EXP_NAME + '_best.pt'
    MNIST_DATA = '/home/matt/fourier/mnist'

FOURIER_ORDER = 10
RAND_SEED = 0
DEVICE = "cuda:0"
NUM_CLASSES = 10
EPOCHS = 30
LEARNING_RATE = 1e-3
BATCH_SIZE = 500
NUM_TRAIN_BATCHES = 60000 // BATCH_SIZE
NUM_VAL_BATCHES = 10000 // BATCH_SIZE
LOSS_FN = nn.CrossEntropyLoss()

# function to ensure deterministic worker re-seeding for reproduceability
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Define transformation(s) to be applied to dataset
transforms_tensor = T.ToTensor()

means = np.asarray([[ 1.0000000e+00, -5.0184957e-20, -5.5696542e-19, -3.4494427e-01],
 [-2.7267091e-02, -2.6857037e-02, -7.0028782e-02,  3.2007497e-02],
 [ 4.2432837e-02, -1.0955108e-02,  4.9947266e-02, -4.3639980e-02],
 [-1.3231365e-02, -3.1862673e-03, -3.0260244e-02,  1.0512119e-02],
 [-3.9451290e-03, -1.6365657e-03, -4.8560360e-03, -6.6966233e-03],
 [-9.9006589e-05, -3.9384090e-03, -1.6901242e-03,  1.6667262e-03],
 [ 4.1181161e-03,  3.0476106e-03,  4.4212379e-03, -3.7111118e-03],
 [-1.6818375e-04, -2.2885541e-03, -1.3834881e-03,  3.6043104e-05],
 [ 1.4858035e-03,  8.2339084e-04,  1.6646081e-03, -1.7478551e-03],
 [ 4.2826071e-04, -9.9242234e-04, -1.3389527e-03,  7.4838375e-04],
 [ 6.7811849e-04,  8.7898056e-04,  8.5074600e-04, -8.7967666e-04],
 [ 6.5689797e-05, -3.9738783e-04, -2.8131096e-04,  1.7239562e-04],
 [ 5.3711585e-04,  5.9293507e-04,  4.6619258e-04, -4.4258419e-04],
 [ 1.2194442e-04, -2.2288978e-04, -3.9203247e-04, 1.7895861e-04],
 [ 2.9932070e-04,  3.2611564e-04,  2.7608767e-04, -2.4960193e-04],
 [ 2.9204506e-05, -9.6740652e-05, -1.5082426e-04,  7.6019125e-05],
 [ 1.9806912e-04,  1.5656877e-04,  1.8582105e-04, -2.2495823e-04],
 [ 4.4551420e-05, -5.8190275e-05, -1.1468527e-04,  5.9609589e-05],
 [ 1.2926904e-04,  7.2024166e-05,  5.9584243e-05, -1.0684843e-04],
 [ 3.3954806e-05, -1.4570637e-05, -8.3436673e-05,  4.2320229e-05]])

stdevs = np.asarray([[1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 2.22759396e-01],
 [1.65906012e-01, 1.80950820e-01, 3.49689871e-01, 1.99603871e-01],
 [1.11437045e-01, 8.92437994e-02, 1.95833012e-01, 1.20404683e-01],
 [7.62690529e-02, 6.39099926e-02, 9.45072919e-02, 6.46416992e-02],
 [5.57328723e-02, 4.34280708e-02, 4.65612561e-02, 4.02896665e-02],
 [2.71149650e-02, 2.66170371e-02, 2.97385808e-02, 2.67203078e-02],
 [2.10881215e-02, 1.92879308e-02, 2.14920528e-02, 2.06694137e-02],
 [1.54367909e-02, 1.52905304e-02, 1.55936116e-02, 1.48296058e-02],
 [1.33549590e-02, 1.24201281e-02, 1.28373653e-02, 1.27569539e-02],
 [1.01144984e-02, 1.04921004e-02, 1.07883746e-02, 1.02084354e-02],
 [8.91223177e-03, 8.50528665e-03, 9.09112580e-03, 9.33418516e-03],
 [7.37450924e-03, 7.48197269e-03, 7.97509681e-03, 7.61566311e-03],
 [6.69074710e-03, 6.42373227e-03, 6.89701224e-03, 6.84088701e-03],
 [5.74082555e-03, 5.81821520e-03, 6.13429025e-03, 5.96161792e-03],
 [5.26886433e-03, 5.20167081e-03, 5.43370517e-03, 5.50555857e-03],
 [4.70780768e-03, 4.73855482e-03, 4.93523199e-03, 4.88662627e-03],
 [4.40586265e-03, 4.30541206e-03, 4.52783192e-03, 4.53947810e-03],
 [3.95527203e-03, 3.97966895e-03, 4.12503956e-03, 4.13487479e-03],
 [3.70030920e-03, 3.66408983e-03, 3.83622549e-03, 3.82866920e-03],
 [3.40025267e-03, 3.37428111e-03, 3.54929711e-03, 3.51667940e-03]])
  
# transform function - normalize img
def transform_train(img):
    raster = np.asarray(img) # convert PIL image to numpy array for openCV
    ret, raster = cv2.threshold(raster, 100, 255, cv2.THRESH_BINARY) # binarize image
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
    sketch_center = pyefd.calculate_dc_coefficients(contour)
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True)
    coeffs = (coeffs - means[:FOURIER_ORDER]) / stdevs[:FOURIER_ORDER]
    return torch.from_numpy(coeffs.flatten()).float()

def transform_test(img):
    # apply random corrupting transformation to input img
    img = transforms_tensor(img.astype(np.float32))
    angle = random.random()*60 - 30
    deltaX = random.randint(-3, 3)
    deltaY = random.randint(-3, 3)
    img = T.functional.affine(img, angle, [deltaX, deltaY], 1, 0,
                              interpolation=T.InterpolationMode.BILINEAR)
    img = np.squeeze(img.numpy()).astype(np.uint8)

    ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY) # binarize image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find outer contour of objects (digit) in image
    
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
    sketch_center = pyefd.calculate_dc_coefficients(contour)
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=FOURIER_ORDER, normalize=True)
    coeffs = (coeffs - means[:FOURIER_ORDER]) / stdevs[:FOURIER_ORDER]
    return torch.from_numpy(coeffs.flatten()).float()


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

# seed RNGs
torch.manual_seed(RAND_SEED)
random.seed(RAND_SEED)

# create train, eval, and test datasets
train_data = datasets.MNIST(root=MNIST_DATA, train=True, download=True, transform=transform_train)
test_data = datasets.MNIST(root=MNIST_DATA, train=False, download=True, transform=transform_test) 

# create generator for dataloaders and create dataloaders for each dataset
g = torch.Generator()
g.manual_seed(RAND_SEED)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
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
for i in range(epoch, EPOCHS):
    print("Epoch " + str(i + 1) + "\n")
    train_loop(dataloader=train_loader,model=model,loss_fn=LOSS_FN,optimizer=optim)
    torch.save({
                'epoch': i + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict()
                }, CHECK_PATH)
    acc = rand_test_loop(dataloader=test_loader,model=model)
    if acc > best_acc:
        torch.save(model.state_dict(), BEST_PATH)
        best_acc = acc
    print(f"best acc: {best_acc:.7f}")
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
