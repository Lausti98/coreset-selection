
import torch
# import pytorch_influence_functions as ptif
# from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import get_model, get_weight
from torchvision.transforms import transforms
import medmnist
# from medmnist import BreastMNIST
# from medmnist import Evaluator
from medmnist.evaluator import getACC, getAUC
from medmnist import INFO
import torch.utils.data as data
import torch.nn as nn
import numpy as np

import sys

from src.utils.config import Config
from src.model.evaluation import evaluate, evaluation_func
from src.model.training import train
from src.model.loader import get_my_model
from src.dataset.loader import get_my_dataloaders, load_dataset
from src.utils.helper import save_coreset



def pretrain(indices=None, fraction=None, iterations=1):
  config = Config.get_config()
  batch_size = config['batch_size']
  task = config['task']
  train_data, validation_data, test_data = load_dataset()
  trainloader, validationloader, testloader = get_my_dataloaders(train_data, validation_data, test_data)
  train_data_permutation = torch.randperm(len(train_data))
  batch_sampler = data.BatchSampler(train_data_permutation, batch_size=batch_size, drop_last=False)
  trainset_permutation_inds = list(batch_sampler)
  trainloader = data.DataLoader(train_data, batch_sampler=batch_sampler)

  model = get_my_model()
  num_epochs = config['num_epochs']
  learning_rate = config['learning_rate']

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # define loss function and optimizer¨
  if task == "binary-class":
      criterion = nn.BCEWithLogitsLoss()
  else:
      criterion = nn.CrossEntropyLoss()

  model, _, _, _, _ = train(model, trainloader, validationloader, optimizer, criterion, num_epochs)
  return model, optimizer, criterion, trainset_permutation_inds



def grand(model, optimizer, criterion, dataset, train_idx, budget: int, index=None):
  """
  Source: https://github.com/PatrickZH/DeepCore/blob/main/deepcore/methods/herding.py
  """
  config = Config.get_config()
  batch_size = config['batch_size']
  device = config['device']
  n_classes = config['n_classes']
  task = config['task']
  repeat = 10
  train_idx_flat = np.array([int(x) for sublist in train_idx for x in sublist])
  feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
  embedding_dim = model.fc.in_features
  batch_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size)
  sample_num = len(dataset)
  norm_matrix = torch.zeros([sample_num, repeat], requires_grad=False).to(device)
  num_class = n_classes if task == 'multi-class' else 1 # account for binary class

  for rep in range(repeat):
    for i, (input, targets) in enumerate(batch_loader):
      input, targets = input.to(device), targets.to(device)
      optimizer.zero_grad()
      outputs = model(input)
      if task == 'binary-class':
        targets = targets.to(torch.float32)
      else:
        targets = targets.squeeze(1).long()
      loss = criterion(outputs.requires_grad_(True), targets).sum()
      batch_num = input.shape[0]
      with torch.no_grad():
        bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]

        norm_matrix[i * batch_size:min((i + 1) * batch_size, sample_num),
        rep] = torch.norm(torch.cat([bias_parameters_grads, (
                feature_extractor(input).view(batch_num, 1, embedding_dim).repeat(1,
                                      num_class, 1) * bias_parameters_grads.view(
                                      batch_num, num_class, 1).repeat(1, 1, embedding_dim)).
                                      view(batch_num, -1)], dim=1), dim=1, p=2)


  norm_mean = torch.mean(norm_matrix, dim=1).cpu().detach().numpy()
  top_examples = train_idx_flat[np.argsort(norm_mean)][::-1][:budget]

  return top_examples

def sample(model, optimizer, criterion, dataset, train_idx, n_samples=None, fraction=None):
  print(fraction)

  if fraction is not None:
    n_samples = int(len(dataset) * fraction)

  indices = grand(model, optimizer, criterion, dataset, train_idx, n_samples)
  return [int(idx) for idx in indices]
  # indices = torch.randperm(len(dataset)).numpy()[:n_samples]
  # data.Subset(dataset, indices)

  # return [int(idx) for idx in indices]


def main():
  config = Config.get_config()
  selection_method = config['selection_method']
  train_data, val_data, test_data = load_dataset()
  assert torch.cuda.is_available() == True, "No GPUS allocated to run-time. Exiting.." ## Sometimes Slurm doesn't provide GPU
  res_indices = {}
  fracions = [0.1, 0.2, 0.4, 0.6, 0.8]
  model, optimizer, criterion, train_idx = pretrain()
  for frac in fracions:
    res_indices[frac] = sample(model, optimizer, criterion, train_data, train_idx, fraction=frac)

  save_coreset(config, res_indices, selection_method)

def run(config_name, data_flag):
  Config.set_config(config_name, data_flag, 'grand')

  main()
  
if __name__ == '__main__':
  selection_method = 'grand'
  data_flag = sys.argv[1]
  config_name = sys.argv[2]
  Config.set_config(config_name, data_flag, selection_method)
  
  main()
