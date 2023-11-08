from torchvision import datasets
from os import listdir
from struct import unpack, error
from torch_geometric.data import Data
from torch.utils.data import Dataset

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
            except error:
                break
    return imageset

def get_data(train_data, val_data, test_data):
    # init lists
    train_imgs = []
    val_imgs = []
    test_imgs = []
    train_counts = []
    val_counts = []
    test_counts = []

    # load dataset
    for class_name in listdir(train_data):
        train_folder = train_data + class_name
        train_drawings = unpack_drawings(train_folder)
        train_imgs += train_drawings
        train_counts.append(len(train_drawings))
        val_folder = val_data + class_name
        val_drawings = unpack_drawings(val_folder)
        val_imgs += val_drawings
        val_counts.append(len(val_drawings))
        test_folder = test_data + class_name
        test_drawings = unpack_drawings(test_folder)
        test_imgs += test_drawings
        test_counts.append(len(test_drawings))

    return train_imgs, val_imgs, test_imgs, train_counts, val_counts, test_counts

# helper method to find class based on imgset index
def find_class(idx, count_list):
  class_id = 0
  sum = count_list[class_id]
  while idx >= sum:
    class_id += 1
    sum += count_list[class_id]
  return class_id


# custom dataset for quickdraw raster data
class QuickdrawDataset(Dataset):
    def __init__(self, imgs, counts, transform, data_split=None):
        self.imgs = imgs
        self.counts = counts
        self.len = sum(counts)
        self.data_split = data_split

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = self.imgs[idx]
        x = transform(img) if self.data_split == None else transform(img, self.data_split)
        y = find_class(idx, self.counts)
        return x, y


# custom dataset for quickdraw graph data
class QuickdrawGraphDataset(Dataset):
  def __init__(self, imgs, counts, transform, is_test):
    self.imgs = imgs
    self.counts = counts
    self.len = sum(counts)
    self.is_test = is_test

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    img = self.imgs[idx]
    x, edge_index, edge_attr = transform(img, self.is_test)
    y = find_class(idx, self.counts)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# custom MNIST with last 10k images set aside from train set for validation
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

