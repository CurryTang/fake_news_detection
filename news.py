import argparse
import os.path as osp

import torch
from torch.nn import Linear
import torch.nn.functional as F
from upfd import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_max_pool
from models import NewsNet, Net, MLP, UDNewsNet, BiGCN, GCNFN
from utils import get_root_feature, seed_torch, read_and_unpkl, pickle_and_write, LabelSmoothing, imbalance, EarlyTransform
from sklearn.metrics import accuracy_score
import torch_geometric.transforms as T
import optuna
import pickle as pkl
from libauc.losses import AUCMLoss 
from libauc.optimizers import PESG
import numpy as np
import random
from torchmetrics import F1


def auc_train(model, train_loader, device, optimizer, loss_fn):
  model.train()
  # print(len(train_loader.dataset))
  total_loss = 0
  for data in train_loader:
    data = data.to(device)
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.batch)
    loss = loss_fn(out, data.y)
    loss.backward()
    optimizer.step()
    total_loss += float(loss) * data.num_graphs

  return total_loss / len(train_loader.dataset)


def mixup_train(model, train_loader, device, optimizer):
  model.train()
  # print(len(train_loader.dataset))
  total_loss = 0
  for data in train_loader:
    data = data.to(device)
    optimizer.zero_grad()
    out, idx = model(data.x, data.edge_index, data.batch)
    loss = model.lam * F.nll_loss(out, data.y) + (1 - model.lam) * F.nll_loss(out, data.y[idx])
    loss.backward()
    optimizer.step()
    total_loss += float(loss) * data.num_graphs

  return total_loss / len(train_loader.dataset)

def label_smooth_train(model, train_loader, device, optimizer, loss_fn):
  model.train()
  total_loss = 0
  for data in train_loader:
    data = data.to(device)
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.batch)[0]
    loss = loss_fn(out, data.y)
    loss.backward()
    optimizer.step()
    total_loss += float(loss) * data.num_graphs

  return total_loss / len(train_loader.dataset)

def train(model, train_loader, device, optimizer):
  model.train()
  # print(len(train_loader.dataset))
  total_loss = 0
  for data in train_loader:
      data = data.to(device)
      optimizer.zero_grad()
      out = model(data.x, data.edge_index, data.batch)[0]
      loss = F.nll_loss(out, data.y)
      loss.backward()
      optimizer.step()
      total_loss += float(loss) * data.num_graphs

  return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader, model, device):
    model.eval()
    total_loss = 0
    total_correct = total_examples = 0
    total_pred = []
    total_y = []
    for data in loader:
        data = data.to(device)
        res = model(data.x, data.edge_index, data.batch)[0]
        pred = res.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
        total_examples += data.num_graphs
        epoch_loss = F.nll_loss(res, data.y)
        total_loss += epoch_loss
        total_pred.append(pred)
        total_y.append(data.y)
    total_y = torch.cat(total_y, dim = -1).to(device)
    total_pred = torch.cat(total_pred, dim = -1).to(device)
    f1 = F1(num_classes = 2, average='macro').to(device)
    f1_result = f1(total_y, total_pred)
    return total_correct / total_examples, total_loss, f1_result

def GCN(args):
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'UPFD')
  train_dataset = UPFD(path, args.dataset, args.feature, 'train', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  val_dataset = UPFD(path, args.dataset, args.feature, 'val', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  test_dataset = UPFD(path, args.dataset, args.feature, 'test', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)


  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Net(args.model, train_dataset.num_features, args.num_hidden,
              train_dataset.num_classes, concat=True).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

  best_val_acc = 0
  best_train_loss = 999
  early_stop_time = 0
  best_model = model
  for epoch in range(args.epochs):
      loss = train(model, train_loader, device, optimizer)
      if loss < best_train_loss:
          best_train_loss = loss 
          early_stop_time = 0
      else:
          if early_stop_time > args.early_stop:
              break 
          early_stop_time += 1
      
      train_acc = test(train_loader, model, device)[0]
      val_acc = test(val_loader, model, device)[0]
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          best_model = model
      print(f'Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}')
  test_res = test(test_loader, best_model, device)
  test_acc, test_f1 = test_res[0], test_res[2]
  print("Best test accuracy: {}".format(test_acc))
  print("Best test F1: {}".format(test_f1))


def LR(args):
  """
    Directly train a linear regression model with features specified in args
  """
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'UPFD')
  train_dataset = UPFD(path, args.dataset, args.feature, 'train', ToUndirected(), custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part)
  val_dataset = UPFD(path, args.dataset, args.feature, 'val', ToUndirected(), custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part)
  test_dataset = UPFD(path, args.dataset, args.feature, 'test', ToUndirected(), custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part)

  train_data, train_y = get_root_feature(train_dataset)
  test_data, test_y = get_root_feature(test_dataset)
  val_data, val_y = get_root_feature(val_dataset)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train_data = torch.Tensor(train_data).to(device)
  train_y = torch.LongTensor(train_y).to(device)
  test_data = torch.Tensor(test_data).to(device)
  test_y = torch.LongTensor(test_y).to(device)
  val_data = torch.Tensor(val_data).to(device)
  val_y = torch.LongTensor(val_y).to(device)
  
  
  model = MLP(train_data.shape[-1], args.num_hidden, 2).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
  

  best_val_acc = 0
  best_train_loss = 999
  early_stop_time = 0
  best_model = model
  for epoch in range(args.epochs):
    optimizer.zero_grad()
    output = model(train_data)
    loss = F.nll_loss(output, train_y)
    loss.backward()
    optimizer.step() 
    if loss < best_train_loss:
          best_train_loss = loss 
          early_stop_time = 0
    else:
        if early_stop_time > args.early_stop:
            break 
        early_stop_time += 1
    output = output.max(dim = -1)[1]
    train_acc = accuracy_score(output.cpu().detach().numpy(), train_y.cpu().detach().numpy())
    val_output = model(val_data)
    val_output = val_output.max(dim = -1)[1]
    val_acc = accuracy_score(val_output.cpu().detach().numpy(), val_y.cpu().detach().numpy())
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
    print(f'Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}')
  test_output = best_model(test_data)
  test_output = test_output.max(dim = -1)[1]
  test_acc = accuracy_score(test_output.cpu().detach().numpy(), test_y.cpu().detach().numpy()) 
  f1 = F1(num_classes = 2).to(device)
  f1_res = f1(test_output, test_y)
  print("Best test accuracy: {}".format(test_acc))
  print("Best test F1: {}".format(f1_res))


def NewsN(args):
  """
    NewsNet
  """
  transforms = T.Compose(
    [
      T.ToUndirected() 
    ]
  )
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'UPFD')
  train_dataset = UPFD(path, args.dataset, args.feature, 'train', transforms, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  val_dataset = UPFD(path, args.dataset, args.feature, 'val', transforms, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  test_dataset = UPFD(path, args.dataset, args.feature, 'test', transforms, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)


  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = NewsNet(train_dataset.num_features, args.num_hidden,
              train_dataset.num_classes).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

  best_val_acc = 0
  best_train_loss = 999
  early_stop_time = 0
  best_model = model
  for epoch in range(args.epochs):
      loss = train(model, train_loader, device, optimizer)
      if loss < best_train_loss:
          best_train_loss = loss 
          early_stop_time = 0
      else:
          if early_stop_time > args.early_stop:
              break 
          early_stop_time += 1
      
      train_acc = test(train_loader, model, device)
      val_acc = test(val_loader, model, device)
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          best_model = model
      print(f'Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}')
  test_acc = test(test_loader, best_model, device)
  print("Best test accuracy: {}".format(test_acc))

def UDNews(args):
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'UPFD')
  train_dataset = UPFD(path, args.dataset, args.feature, 'train', None, custom = True, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  val_dataset = UPFD(path, args.dataset, args.feature, 'val', None, custom = True, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  test_dataset = UPFD(path, args.dataset, args.feature, 'test', None, custom = True, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)


  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  num_hidden = 16
  att_hidden = 32
  need_jk = True
  pooling = 'attention'
  # pooling = 'max'
  model = UDNewsNet(train_dataset.num_features, num_hidden,
              train_dataset.num_classes, need_jk = need_jk, pooling = pooling, enhance = False, att_hidden = att_hidden).to(device)
  ## politifact: 0.008, 0.0014
  #lr = 0.003
  #weight_decay = 0.0009
  lr = 0.007
  weight_decay = 0.0033
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  best_epoch = 0
  epoch = 0
  best_val_acc = 0
  best_train_loss = 999
  early_stop_time = 0
  best_model = model
  for epoch in range(args.epochs):
      loss = train(model, train_loader, device, optimizer)
      if loss < best_train_loss:
          best_train_loss = loss 
          early_stop_time = 0
      else:
          if early_stop_time > args.early_stop:
              break 
          early_stop_time += 1
      
      train_acc = test(train_loader, model, device)[0]
      val_acc = test(val_loader, model, device)[0]
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          best_model = model
          best_epoch = epoch
      print(f'Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}')
  test_res = test(test_loader, best_model, device)
  test_acc, test_f1 = test_res[0], test_res[2]
  print("Best test accuracy: {}".format(test_acc))
  print("Best test F1 score: {}".format(test_f1))
  print(best_epoch)
  print(epoch)


def BiGCN_news(args):
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'UPFD')
  train_dataset = UPFD(path, args.dataset, args.feature, 'train', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  val_dataset = UPFD(path, args.dataset, args.feature, 'val', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  test_dataset = UPFD(path, args.dataset, args.feature, 'test', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)


  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # device = torch.device('cpu')
  model = BiGCN(train_dataset.num_features, args.num_hidden,
              train_dataset.num_classes).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

  best_val_acc = 0
  best_train_loss = 999
  early_stop_time = 0
  best_model = model
  for epoch in range(args.epochs):
      loss = train(model, train_loader, device, optimizer)
      if loss < best_train_loss:
          best_train_loss = loss 
          early_stop_time = 0
      else:
          if early_stop_time > args.early_stop:
              break 
          early_stop_time += 1
      
      train_acc = test(train_loader, model, device)[0]
      val_acc = test(val_loader, model, device)[0]
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          best_model = model
      print(f'Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}')
  test_res = test(test_loader, best_model, device)
  test_acc, test_f1 = test_res[0], test_res[2]
  print("Best test accuracy: {}".format(test_acc))
  print("Best test F1: {}".format(test_f1))
  return test_acc


def GCN_with_GDC(args):
  transform = T.Compose([
    T.ToUndirected(),
    T.GDC()
  ]
  )
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'UPFD')
  train_dataset = UPFD(path, args.dataset, args.feature, 'train', transform, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part)
  val_dataset = UPFD(path, args.dataset, args.feature, 'val', transform, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part)
  test_dataset = UPFD(path, args.dataset, args.feature, 'test', transform, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part)


  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Net(args.model, train_dataset.num_features, args.num_hidden,
              train_dataset.num_classes, concat=False).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

  best_val_acc = 0
  best_train_loss = 999
  early_stop_time = 0
  best_model = model
  for epoch in range(args.epochs):
      loss = train(model, train_loader, device, optimizer)
      if loss < best_train_loss:
          best_train_loss = loss 
          early_stop_time = 0
      else:
          if early_stop_time > args.early_stop:
              break 
          early_stop_time += 1
      
      train_acc = test(train_loader, model, device)
      val_acc = test(val_loader, model, device)
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          best_model = model
      print(f'Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}')
  test_acc = test(test_loader, best_model, device)
  print("Best test accuracy: {}".format(test_acc))


def optimize_params(trial, args):
  seed_torch()
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'UPFD')
  custom = True
  train_dataset = UPFD(path, args.dataset, args.feature, 'train', None, custom = custom, train_part = 0.05, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  val_dataset = UPFD(path, args.dataset, args.feature, 'val', None, custom = custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  test_dataset = UPFD(path, args.dataset, args.feature, 'test', None, custom = custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)


  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  num_hidden = 16
  #num_hidden = 48
  # num_hidden = trial.suggest_categorical('num_hidden', [16, 32, 48, 64])
  enhance = trial.suggest_categorical('enhance', [False])
  # enhance = False
  # dire = trial.suggest_categorical('dire', [True, False])
  dire = False
  # need_jk = trial.suggest_categorical('need_jk', [True, False])
  need_jk = True
  # pooling = trial.suggest_categorical('pooling', ['attention'])
  pooling = 'attention'
  # pooling = trial.suggest_categorical('pooling', ['max', 'mean', 'add', 'sort'])
  # if dire:
  #   model = BiGCN(train_dataset.num_features, num_hidden,
  #       train_dataset.num_classes, need_jk, pooling).to(device)
  # else:
  att_hidden = trial.suggest_categorical('att_hidden', [16, 32, 48, 64, 128])
  # mixup_alpha = trial.suggest_categorical('alpha', [0.1, 0.2, 1.0, 2.0, 4.0, 0.5])
  # att_hidden = 64
  model = UDNewsNet(train_dataset.num_features, num_hidden,
            train_dataset.num_classes, need_jk, pooling, enhance, mixup = False, att_hidden = att_hidden).to(device)
  lr = trial.suggest_float('lr', 0.001, 0.01, step = 0.001)
  # lr = 0.008
  weight_decay = trial.suggest_float('weight_decay', 0.001, 0.01, step = 1e-4)
  # weight_decay = 0.0008

  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  best_val_acc = 0
  best_train_loss = 999
  early_stop_time = 0
  best_model = model
  epoch = 0
  best_epoch = 0
  best_val_loss = 999
  for epoch in range(args.epochs):
      loss = train(model, train_loader, device, optimizer)
      if loss < best_train_loss:
          best_train_loss = loss 
          early_stop_time = 0
      else:
          if early_stop_time > args.early_stop:
              break 
          early_stop_time += 1
      
      train_acc = test(train_loader, model, device)[0]
      val_acc = test(val_loader, model, device)[0]
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          best_model = model
          best_epoch = epoch
      # if best_val_loss < val_loss:
      #   if early_stop_time > args.early_stop:
      #     break
      #   early_stop_time += 1
      # else:
      #   best_val_loss = val_loss
      #   early_stop_time = 0
      #print(f'Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
      #    f'Val: {val_acc:.4f}')
  # test_acc = test(test_loader, best_model, device)
  test_acc = test(test_loader, best_model, device)[0]
  #print("Best test accuracy: {}".format(test_acc))
  trial.set_user_attr(key = 'best', value = best_model)
  trial.set_user_attr(key = 'final_epoch', value = epoch)
  trial.set_user_attr(key = 'epoch', value = best_epoch)
  return test_acc


def test_model(model, args):
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'UPFD')
  test_dataset = UPFD(path, args.dataset, args.feature, 'test', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  test_acc = test(test_loader, model, device)
  return test_acc



def ud_news_mixup(trial, args):
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'UPFD')
  train_dataset = UPFD(path, args.dataset, args.feature, 'train', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  val_dataset = UPFD(path, args.dataset, args.feature, 'val', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  test_dataset = UPFD(path, args.dataset, args.feature, 'test', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)


  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # num_hidden = trial.suggest_int('num_hidden', 16, 128, step = 16)
  num_hidden = 16
  # enhance = trial.suggest_categorical('enhance', [True, False])
  enhance = False
  # dire = trial.suggest_categorical('dire', [True, False])
  dire = False
  # need_jk = trial.suggest_categorical('need_jk', [True, False])
  need_jk = True
  #pooling = trial.suggest_categorical('pooling', ['max', 'mean', 'add', 'sort', 'attention'])
  pooling = trial.suggest_categorical('pooling', ['attention'])
  # pooling = trial.suggest_categorical('pooling', ['max', 'mean', 'add', 'sort'])
  # if dire:
  #   model = BiGCN(train_dataset.num_features, num_hidden,
  #       train_dataset.num_classes, need_jk, pooling).to(device)
  # else:
  mix_up_alpha = trial.suggest_float('alpha', 0.1, 5.0, step = 0.1)
  mixup = True
  model = UDNewsNet(train_dataset.num_features, num_hidden,
            train_dataset.num_classes, need_jk, pooling, enhance, mixup = mixup, mixup_alpha = mix_up_alpha).to(device)
  lr = trial.suggest_float('lr', 0.001, 0.05, step = 0.001)
  weight_decay = trial.suggest_float('weight_decay', 5e-4, 0.01, step = 1e-4)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  best_val_acc = 0
  best_train_loss = 999
  early_stop_time = 0
  best_model = model
  for epoch in range(args.epochs):
      loss = mixup_train(model, train_loader, device, optimizer)
      if loss < best_train_loss:
          best_train_loss = loss 
          early_stop_time = 0
      else:
          if early_stop_time > args.early_stop:
              break 
          early_stop_time += 1
      
      train_acc, train_loss = test(train_loader, model, device)
      val_acc, val_loss = test(val_loader, model, device)
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          best_model = model
  test_acc, test_loss = test(test_loader, best_model, device)
  trial.set_user_attr(key = 'best', value = best_model)
  return test_acc

def ud_news_label_smoothing(trial, args):
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'UPFD')
  train_dataset = UPFD(path, args.dataset, args.feature, 'train', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  val_dataset = UPFD(path, args.dataset, args.feature, 'val', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  test_dataset = UPFD(path, args.dataset, args.feature, 'test', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)


  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  num_hidden = 16
  enhance = False
  dire = False
  need_jk = True
  pooling = 'attention'
  model = UDNewsNet(train_dataset.num_features, num_hidden,
            train_dataset.num_classes, need_jk, pooling, enhance, mixup = False).to(device)
  lr = trial.suggest_float('lr', 0.001, 0.01, step = 0.001)
  weight_decay = trial.suggest_float('weight_decay', 0.001, 0.01, step = 0.0001)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  smoothing_coef = trial.suggest_categorical('smooth', [0.1, 0.2])
  loss_fn = LabelSmoothing(smoothing_coef)
  best_val_acc = 0
  epoch = 0
  best_train_loss = 999
  early_stop_time = 0
  best_model = model
  for epoch in range(args.epochs):
      loss = label_smooth_train(model, train_loader, device, optimizer, loss_fn)
      if loss < best_train_loss:
          best_train_loss = loss 
          early_stop_time = 0
      else:
          if early_stop_time > args.early_stop:
              break 
          early_stop_time += 1
      
      train_acc = test(train_loader, model, device)[0]
      val_acc = test(val_loader, model, device)[0]
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          best_model = model
          best_epoch = epoch
  test_acc = test(test_loader, best_model, device)[0]
  trial.set_user_attr(key = 'best', value = best_model)
  trial.set_user_attr(key = 'final_epoch', value = epoch)
  trial.set_user_attr(key = 'epoch', value = best_epoch)
  return test_acc

def label_smooth_ud_news(args):
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'UPFD')
  train_dataset = UPFD(path, args.dataset, args.feature, 'train', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  val_dataset = UPFD(path, args.dataset, args.feature, 'val', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  test_dataset = UPFD(path, args.dataset, args.feature, 'test', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)


  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  num_hidden = 16
  enhance = False
  dire = False
  need_jk = True
  pooling = 'attention'
  model = UDNewsNet(train_dataset.num_features, num_hidden,
            train_dataset.num_classes, need_jk, pooling, enhance, mixup = False).to(device)
  lr = 0.001
  weight_decay = 0.0039

  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  smoothing_coef = 0
  loss_fn = LabelSmoothing(smoothing_coef)
  best_val_acc = 0
  epoch = 0
  best_train_loss = 999
  early_stop_time = 0
  best_model = model
  for epoch in range(args.epochs):
      loss = label_smooth_train(model, train_loader, device, optimizer, loss_fn)
      if loss < best_train_loss:
          best_train_loss = loss 
          early_stop_time = 0
      else:
          if early_stop_time > args.early_stop:
              break 
          early_stop_time += 1
      
      train_acc = test(train_loader, model, device)[0]
      val_acc = test(val_loader, model, device)[0]
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          best_model = model
          best_epoch = epoch
  test_acc = test(test_loader, best_model, device)[0]
  print(test_acc)
  return test_acc

def gcnfn(args):
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'UPFD')
  train_dataset = UPFD(path, args.dataset, args.feature, 'train', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  val_dataset = UPFD(path, args.dataset, args.feature, 'val', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  test_dataset = UPFD(path, args.dataset, args.feature, 'test', None, custom = args.custom, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)


  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # device = torch.device('cpu')
  model = GCNFN(train_dataset.num_features, args.num_hidden,
              train_dataset.num_classes).to(device)
  lr = 0.001
  weight_decay = 0.01
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  best_val_acc = 0
  best_train_loss = 999
  early_stop_time = 0
  best_model = model
  for epoch in range(args.epochs):
      loss = train(model, train_loader, device, optimizer)
      if loss < best_train_loss:
          best_train_loss = loss 
          early_stop_time = 0
      else:
          if early_stop_time > args.early_stop:
              break 
          early_stop_time += 1
      
      train_acc = test(train_loader, model, device)[0]
      val_acc = test(val_loader, model, device)[0]
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          best_model = model
      print(f'Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}')
  test_res = test(test_loader, best_model, device)
  test_acc, test_f1 = test_res[0], test_res[2]
  print("Best test accuracy: {}".format(test_acc))
  print("Best test F1: {}".format(test_f1))
  return test_acc


def special_test(args, ratio, special = 'imbalance',):
  """
    special: imbalance or early
  """
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'UPFD')
  if special == 'early':
    transform = EarlyTransform(ratio)
  else:
    transform = None
  train_dataset = UPFD(path, args.dataset, args.feature, 'train', transform, custom = False, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  val_dataset = UPFD(path, args.dataset, args.feature, 'val', None, custom = False, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)
  test_dataset = UPFD(path, args.dataset, args.feature, 'test', None, custom = False, train_part = args.train_part, val_part = args.val_part, test_part = args.test_part, feature_dim = args.feature_size)

  if special == 'imbalance':
    train_dataset = imbalance(train_dataset, ratio)

  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  num_hidden = 16
  att_hidden = 64
  need_jk = True
  pooling = 'attention'
  # model = UDNewsNet(train_dataset.num_features, num_hidden,
  #             train_dataset.num_classes, need_jk = need_jk, pooling = pooling, enhance = False, att_hidden = att_hidden).to(device)
  # model = Net(args.model, train_dataset.num_features, args.num_hidden,
  #             train_dataset.num_classes, concat=True).to(device)
  model = BiGCN(train_dataset.num_features, args.num_hidden,
    train_dataset.num_classes).to(device)
  ## politifact: 0.008, 0.0014
  ## gossip: 0.003 0.0007
  # lr = 0.001
  # weight_decay = 0.01
  print(train_dataset)
  # input()
  lr = 0.008
  weight_decay = 0.0014
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  best_epoch = 0
  epoch = 0
  best_val_acc = 0
  best_train_loss = 999
  early_stop_time = 0
  best_model = model
  for epoch in range(args.epochs):
      loss = train(model, train_loader, device, optimizer)
      if loss < best_train_loss:
          best_train_loss = loss 
          early_stop_time = 0
      else:
          if early_stop_time > args.early_stop:
              break 
          early_stop_time += 1
      
      train_acc = test(train_loader, model, device)[0]
      val_acc = test(val_loader, model, device)[0]
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          best_model = model
          best_epoch = epoch
      print(f'Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}')
  test_res = test(test_loader, best_model, device)
  test_acc, test_f1 = test_res[0], test_res[2]
  print("Best test accuracy: {}".format(test_acc))
  print("Best test F1 score: {}".format(test_f1))
  print(best_epoch)
  print(epoch)


def main(args):
  def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best", value=trial.user_attrs["best"])
        study.set_user_attr(key="epoch", value=trial.user_attrs["epoch"])
        study.set_user_attr(key="final_epoch", value=trial.user_attrs["final_epoch"])
  seed_torch()
  if args.optimize and args.model == 'UDNews':
    sampler = optuna.samplers.TPESampler(seed=1029)
    study = optuna.create_study(direction = 'maximize',sampler = sampler)
    study.optimize(lambda trial: optimize_params(trial, args), n_trials=150, callbacks=[callback], n_jobs=1)
    # study.optimize(lambda trial: ud_news_mixup(trial, args), n_trials=500, callbacks=[callback])
    best_model = study.user_attrs['best']
    pickle_and_write(best_model, 'best_model.pkl')
    test_acc = test_model(best_model, args)
    print("Reproduce acc: {}".format(test_acc))
    print(study.best_params)
    print(study.best_trial)
    print(study.user_attrs['epoch'])
    print(study.user_attrs['final_epoch'])
  elif args.optimize and args.model == 'ls':
    sampler = optuna.samplers.TPESampler(seed=1029)
    study = optuna.create_study(direction = 'maximize',sampler = sampler)
    study.optimize(lambda trial: ud_news_label_smoothing(trial, args), n_trials=200, callbacks=[callback], n_jobs=1)
    # study.optimize(lambda trial: ud_news_mixup(trial, args), n_trials=500, callbacks=[callback])
    best_model = study.user_attrs['best']
    pickle_and_write(best_model, 'best_model.pkl')
    test_acc = test_model(best_model, args)
    print("Reproduce acc: {}".format(test_acc))
    print(study.best_params)
    print(study.best_trial)
  elif args.special != 'normal':
    special_test(args, args.ratio, args.special)
  elif args.model == 'GCN' or args.model == 'SAGE' or args.model == 'GAT':
    GCN(args)
  elif args.model == 'LR':
    LR(args)
  elif args.model == 'GDC':
    GCN_with_GDC(args)
  elif args.model == 'News':
    NewsN(args)
  elif args.model == 'UDNews':
    UDNews(args)
  elif args.model == 'BiGCN':
    BiGCN_news(args)
  elif args.model == 'gcnfn':
    gcnfn(args)
  elif args.model == 'ls':
    label_smooth_ud_news(args)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='politifact',
                        choices=['politifact', 'gossipcop'])
    parser.add_argument('--feature', type=str, default='spacy',
                        choices=['profile', 'spacy', 'bert', 'content'])
    parser.add_argument('--model', type=str, default='GCN',
                        choices=['GCN', 'GAT', 'SAGE', 'LR', 'GDC', 'CL', 'News', 'UDNews', 'BiGCN', 'mixup', 'ls', 'gcnfn'])
    parser.add_argument('--pooling', type=str, default='max', choices=['max', 'mean', 'add', 'sort', 'attention'])
    parser.add_argument('--epochs', type=int, default = 300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default = 0.01)
    parser.add_argument('--num_hidden', type=int, default = 16)
    parser.add_argument('--early_stop', type=int, default=15)
    parser.add_argument('--train_part', type=float, default = 0.2)
    parser.add_argument('--val_part', type=float, default = 0.1)
    parser.add_argument('--test_part', type=float, default = 0.7)
    parser.add_argument('--feature_size', type=int, default = 1024)
    parser.add_argument('--custom', action='store_true')
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--need_jk', action='store_true')
    parser.add_argument('--att_hidden', type = int, default = 64)
    parser.add_argument('--special', type=str, default='normal', choices = ['normal', 'imbalance', 'early'])
    parser.add_argument('--ratio', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
