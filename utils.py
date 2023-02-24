import numpy as np
from sklearn.model_selection import train_test_split
import torch
import networkx as nx
import scipy.sparse as sp
import torch_geometric
from torch_geometric.utils import coalesce, to_undirected, sort_edge_index
import copy
import random 
import os
import pickle
import torch.nn as nn
from torch_geometric.transforms import BaseTransform


def read_and_unpkl(file: str) -> object:
  with open(file, 'rb') as f:
      return pickle.load(f)

def pickle_and_write(obj: object, filename: str) -> None:
  with open(filename, 'wb') as f:
      pickle.dump(obj, f)

def seed_torch(seed=1029, enable_cudnn = True):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  # torch.use_deterministic_algorithms(True)
  torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.enabled = enable_cudnn


def train_val_test_split(train_size, val_size, test_size, X, y):
  """
      train val test split returns idx_train, idx_val and idx_test
  """
  try:
      X = X.cpu()
      y = y.cpu()
  except AttributeError:
      ## numpy object
      pass
  X = np.array(X)
  y = np.array(y)
  train_id_list = np.arange(X.shape[0])
  ## first split
  idx_train, idx_test, _, _ = train_test_split(train_id_list, y, train_size = train_size + val_size,
      test_size=test_size, shuffle=True, stratify=y)
  
  stratify_for_train_val_split = y[idx_train]
  real_train_size = train_size / (train_size + val_size)
  real_val_size = 1 - real_train_size
  idx_train, idx_val, _, _ = train_test_split(
      idx_train, stratify_for_train_val_split, train_size=real_train_size, 
      test_size = real_val_size, stratify=stratify_for_train_val_split, shuffle = True
  )
  return idx_train, idx_val, idx_test


def get_root_feature(dataset):
  dataset_list = list(dataset)
  root_node_list = [graph.x[0].numpy() for graph in dataset_list]
  label = [graph.y.numpy() for graph in dataset_list]
  features = np.vstack(root_node_list)
  label = np.concatenate(label)
  return features, label


def edge_index_to_matrix(edge_index):
  row, col = edge_index 
  length = row.max().item()
  mat = torch.zeros((length + 1, length + 1))
  mat[row, col] = 1
  return mat


def enhance_root_edge_index(x, edge_id, batch):
  row, col = edge_id
  max_node_id = edge_id.max().item()
  right = torch.arange(max_node_id + 1, dtype=torch.long).to(edge_id.device)
  root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
  root = torch.cat([root.new_zeros(1), root + 1], dim=0)
  repeat_num = root[1:]
  repeat_num_shift = torch.diff(repeat_num)
  repeat_num_shift = torch.cat([repeat_num_shift.new_zeros(1).fill_(repeat_num[0]), repeat_num_shift, repeat_num_shift.new_zeros(1).fill_((x.shape[0] - repeat_num[-1]))], dim = 0)
  left = torch.repeat_interleave(root, repeat_num_shift, dim = 0)
  row, col = torch.cat([row, left], dim = 0), torch.cat([col, right], dim = 0)
  edge_index = torch.stack([row, col], dim=0)
  edge_index = coalesce(edge_index)
  return torch_geometric.utils.remove_self_loops(torch_geometric.utils.to_undirected(edge_index))[0]


def enhance_root_edge_index_ud(edge_id):
  row, col = edge_id
  if row.min().item() < col.min().item():
    # top down
    #edges = [(row.min().item(), y) for y in col]
    left = torch.zeros(row.shape[0], dtype=torch.long).to(edge_id.device)
    right = torch.arange(row.shape[0], dtype=torch.long).to(edge_id.device)
  else:
    # bottom up
    #edges = [(y, col.min().item()) for y in row]
    right = torch.zeros(col.shape[0], dtype=torch.long).to(edge_id.device)
    left = torch.arange(row.shape[0], dtype=torch.long).to(edge_id.device)
  row, col = torch.cat([row, left], dim = 0), torch.cat([col, right], dim = 0)
  edge_index = torch.stack([row, col], dim=0)
  return coalesce(edge_index, None, row.shape[0])

def clones(module, k):
    return torch.nn.ModuleList(copy.deepcopy(module) for _ in range(k))

def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch**3 * init_lr / num_batch_warm_up**3

def flag(model_forward, perturb_shape, y, optimizer, device, criterion, step_size=8e-3, m=3) :
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()
                    
    perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
    perturb.requires_grad_()
    out = forward(perturb)
    loss = criterion(out, y)
    loss /= m
    for _ in range(m-1):
        loss.backward()
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0
        
        out = forward(perturb)
        loss = criterion(out, y)
        loss /= m                                      
    loss.backward()
    optimizer.step()
    return loss, out

def train_with_flag(model, device, loader, optimizer, multicls_criterion):
    total_loss = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            forward = lambda perturb : model(batch, perturb).to(torch.float32)
            model_forward = (model, forward)
            y = batch.y.view(-1,)
            perturb_shape = (batch.x.shape[0], model.config.hidden)
            loss, _ = flag(model_forward, perturb_shape, y, optimizer, device, multicls_criterion)
            total_loss += loss.item()
            
    #print(total_loss/len(loader))
    return total_loss/len(loader)


def dropedge(edgeindex, droprate = 0.1):
  if droprate == 0:
    return edgeindex
  device = edgeindex.device
  row = list(edgeindex[0])
  col = list(edgeindex[1])
  length = len(row)
  poslist = random.sample(range(length), int(length * (1 - droprate)))
  poslist = sorted(poslist)
  row = list(np.array(row)[poslist])
  col = list(np.array(col)[poslist])
  new_edgeindex = [row, col]
  return torch.LongTensor(new_edgeindex).to(device)

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def imbalance(dataset, ratio):
  labels = dataset.data.y
  ## 1 for fake news
  mask = torch.ones(len(dataset), dtype=torch.bool)
  fake = (labels == 1).nonzero().view(-1)
  real = (labels == 0).nonzero().view(-1)
  real_num = len(real)
  fake_num = int(real_num / ratio)
  masked_out = len(fake) - fake_num
  mask[fake[:masked_out]] = False 
  return dataset[mask]


def sub_edge_index(edge_index, node_up):
  edge_index = to_undirected(edge_index)
  row, col = edge_index
  new_row = row[row <= node_up]
  new_col = col[col <= node_up]
  edge_index = torch.stack([new_row, new_col], dim=0)
  return edge_index

class EarlyTransform(BaseTransform):
  def __init__(self, ratio):
    super().__init__()
    self.ratio = ratio

  def __call__(self, data):
    size = data.x.shape[0]
    need = int(self.ratio * size) + 1
    node_up = need - 1
    for store in data.stores:
      store['x'] =  store['x'][:need]
    for store in data.edge_stores:
       store.edge_index = coalesce(sub_edge_index(store.edge_index, node_up))
    return data