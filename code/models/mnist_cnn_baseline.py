import torch
from torch import nn
from torchvision import datasets, models
from torchvision import transforms as T
from torch.utils.data import DataLoader
import random
import numpy as np

# Env Vars
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True

# Const vars
EXP_NAME = 'mnist_baseline_cnn5-2'
SERVER = "matt"
if SERVER == "apg":
    CHECK_PATH = '/home/apg/mw/fourier/models/' + EXP_NAME + '_check.pt'
    BEST_PATH = '/home/apg/mw/fourier/models/' + EXP_NAME + '_best.pt'
    MNIST_DATA = '/home/apg/mw/fourier/mnist'
else:
    CHECK_PATH = '/home/matt/fourier/models/' + EXP_NAME + '_check.pt'
    BEST_PATH = '/home/matt/fourier/models/' + EXP_NAME + '_best.pt'
    MNIST_DATA = '/home/matt/fourier/mnist'

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

# Define transformation(s) to be applied to dataset-
transforms_norm = T.Compose(
      [
          T.ToTensor(),
          T.Normalize(mean = (0.1307,), std = (0.3081,)) # MNIST mean and stdev
      ]
  )

transforms_tensor = T.ToTensor()
  
# transform functions - take sketch image, return torch tensor of descriptors
def transform_train(img):
  return transforms_norm(img)

def transform_test(img):
  img = transforms_norm(img)

  # add rotations and translations at test time
  angle = random.random()*60 - 30
  deltaX = random.randint(-3, 3)
  deltaY = random.randint(-3, 3)
	
  return T.functional.affine(img, angle, [deltaX, deltaY], 1, 0,
	                             interpolation=T.InterpolationMode.BILINEAR)


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
model = CNN()
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
    print(f"best val acc: {best_acc:.7f}")
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
