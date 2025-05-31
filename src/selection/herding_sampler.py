import sys

import torch
import torch.nn as nn
import numpy as np

from src.utils.config import Config
from src.model.training import train
from src.model.loader import get_my_model
from src.dataset.loader import get_my_dataloaders, load_dataset
from src.utils.helper import save_coreset

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def pretrain(indices=None, fraction=None, iterations=1):
  config = Config.get_config()
  task = config['task']
  train_data, validation_data, test_data = load_dataset()
  trainloader, validationloader, testloader = get_my_dataloaders(train_data, validation_data, test_data)

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
  return model


def create_matrix(model):
  """
    Source: https://github.com/PatrickZH/DeepCore/blob/main/deepcore/methods/herding.py
    Find the feature embeddings of the model, and create a matrix of the embeddings.
    :return:
    feature matrix
  """
  config = Config.get_config()
  device = config['device']
  batch_size = config['batch_size']
  model.eval()

  train_data, validation_data, test_data = load_dataset()
  trainloader, _, _ = get_my_dataloaders(train_data, validation_data, test_data)
  feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
  feature_matrix = torch.zeros([len(train_data), model.fc.in_features], requires_grad=False).to(device)
  with torch.no_grad():
    for i, (inputs, labels) in enumerate(trainloader):
      inputs = inputs.to(device)
      features = feature_extractor(inputs)
      features = features.view(features.size(0), -1)
      feature_matrix[i*batch_size:(i+1)*batch_size] = features
  return feature_matrix



def herding(matrix, budget: int, index=None):
  """
  Source: https://github.com/PatrickZH/DeepCore/blob/main/deepcore/methods/herding.py
  """
  sample_num = matrix.shape[0]

  if budget < 0:
      raise ValueError("Illegal budget size.")
  elif budget > sample_num:
      budget = sample_num

  indices = np.arange(sample_num)
  with torch.no_grad():
      mu = torch.mean(matrix, dim=0)
      select_result = np.zeros(sample_num, dtype=bool)

      for i in range(budget):
          dist = euclidean_dist(((i + 1) * mu - torch.sum(matrix[select_result], dim=0)).view(1, -1),
                              matrix[~select_result])
          p = torch.argmax(dist).item()
          p = indices[~select_result][p]
          select_result[p] = True
  if index is None:
      index = indices
  return index[select_result]


def sample(dataset, model, n_samples=None, fraction=None):
  if fraction is not None:
    n_samples = int(len(dataset) * fraction)
  feature_matrix = create_matrix(model)
  indices = herding(feature_matrix, n_samples)

  return [int(idx) for idx in indices]


def main():
  config = Config.get_config()
  train_data, _, _ = load_dataset()
  assert torch.cuda.is_available() == True, "No GPUS allocated to run-time. Exiting.." ## Sometimes Slurm doesn't provide GPU
  res_indices = {}
  fracions = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8]
  if 'tissue' in config['data_flag']:
     fracions = [0.001, 0.005, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8]
  model = pretrain()
  for frac in fracions:
    res_indices[frac] = sample(train_data, model, fraction=frac)

  save_coreset(config, res_indices, config['selection_method'])

def run(config_name, data_flag):
  Config.set_config(config_name, data_flag, 'herding')

  main()
  
if __name__ == '__main__':
  selection_method = 'herding'
  data_flag = sys.argv[1]
  config_name = sys.argv[2]
  Config.set_config(config_name, data_flag, selection_method)
  
  main()