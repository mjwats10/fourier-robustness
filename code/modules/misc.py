import torch
import numpy as np
import random
import cairocffi as cairo

# deterministic worker re-seeding
def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train() # put the model in train mode
    total_loss = 0
    total_correct = 0
    # for each batch in the training set compute loss and update model parameters
    for batch, (x, y) in enumerate(dataloader):
      x, y = x.to(device), y.to(device)
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
          current = (batch + 1) * dataloader.batch_size
          print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

      pred = out.argmax(dim=1, keepdim=True)
      correct = pred.eq(y.view_as(pred)).sum().item()
      total_correct += correct
      total_loss += loss_val
    print(f"\nepoch avg train loss: {total_loss / ((size // batch_size) + 1):.7f}   epoch avg train accuracy: {total_correct / size:.4f}")

def train_loop_graph(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train() # put the model in train mode
    total_loss = 0
    total_correct = 0
    # for each batch in the training set compute loss and update model parameters
    for batch, data in enumerate(dataloader):
      data = data.to(device)
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
          current = (batch + 1) * dataloader.batch_size
          print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

      pred = out.argmax(dim=1, keepdim=True)
      correct = pred.eq(data.y.view_as(pred)).sum().item()
      total_correct += correct
      total_loss += loss_val 
    print(f"\nepoch avg train loss: {total_loss / ((size // batch_size) + 1):.7f}   epoch avg train accuracy: {total_correct / size:.4f}")

def rand_test_loop(dataloader, model, device):
  model.eval()
  size = len(dataloader.dataset)
  with torch.no_grad():
    total_correct = 0
    for x, y in dataloader:
      x = x.to(device) 
      out = model(x)
      out = out.to("cpu")
      pred = out.argmax(dim=1, keepdim=True)
      total_correct += pred.eq(y.view_as(pred)).sum().item()

    accuracy = total_correct / size
    return accuracy

def rand_test_loop_graph(dataloader, model, device):
  model.eval()
  size = len(dataloader.dataset)
  with torch.no_grad():
    total_correct = 0
    for data in dataloader:
      data = data.to(device)
      out = model(data)
      data, out = data.to("cpu"), out.to("cpu")
      pred = out.argmax(dim=1, keepdim=True)
      total_correct += pred.eq(data.y.view_as(pred)).sum().item()

    accuracy = total_correct / size
    return accuracy

# convert raw vector image to a single raster image
def vector_to_raster(vector_image, side, padding, line_diameter=16, bg_color=(0,0,0), fg_color=(1,1,1)):
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

# convert raw vector image to a squence of raster images
def vector_to_raster_graph(vector_image, side, padding, line_diameter=16, bg_color=(0,0,0), fg_color=(1,1,1)):
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

def get_train_state(model, optim, resume, check_path):
    current_epoch = 0
    best_acc = 0
    plateau_len = 0
    if resume:
        checkpoint = torch.load(check_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['current_epoch']
        best_acc = checkpoint['best_acc']
        plateau_len = checkpoint['plateau_len']

    return current_epoch, best_acc, plateau_len

# train for 'num_epochs' number of epochs
def train(exp_name, current_epoch, num_epochs, best_acc, plateau_len, train_loader, val_loader, model, loss_fn, optim, check_path, best_path, device, loop=train_loop):
    print(exp_name)
    for i in range(current_epoch, num_epochs):
        if plateau_len >= 10:
            break
        print("Epoch " + str(i + 1) + "\n")
        loop(dataloader=train_loader,model=model,loss_fn=loss_fn,optimizer=optim, device=device)
        torch.save({
                    'epoch': i + 1,
                    'best_acc': best_acc,
                    'plateau_len': plateau_len,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                    }, check_path)
        acc = rand_test_loop(dataloader=val_loader,model=model)
        if acc > best_acc:
            torch.save(model.state_dict(), best_path)
            best_acc = acc
            plateau_len = 0
        else:
            plateau_len += 1
        print(f"best val acc: {best_acc:.4f}")
        print("\n-------------------------------\n")

# evaluate on random translations and rotations
def test(model, best_path, rand_seed, test_loader, device, loop=rand_test_loop):
    print("Evaluating against random transformations...")
    model.load_state_dict(torch.load(best_path))
    random.seed(rand_seed)
    accuracies = []
    for i in range(30):
        accuracies.append(loop(dataloader=test_loader,model=model,device=device))
    accuracies = np.asarray(accuracies)
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    print(f"Mean acc: {mean:.4f}")
    print(f"Acc std: {std:.7f}")
    print("\n-------------------------------\n")