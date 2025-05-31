
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

import sys

from src.utils.config import Config
from src.model.evaluation import evaluate
from src.model.loader import get_my_model
from src.dataset.loader import get_my_dataloaders, load_dataset
from src.utils.helper import save_coreset

def sample(dataset, n_samples=None, fraction=None):
  print(fraction)
  if fraction is not None:
    n_samples = int(len(dataset) * fraction)

  indices = torch.randperm(len(dataset)).numpy()[:n_samples]
  # data.Subset(dataset, indices)

  return [int(idx) for idx in indices]

def main():
  config = Config.get_config()
  train_data, val_data, test_data = load_dataset()

  res_indices = {}
  fracions = [0.1, 0.2, 0.4, 0.6, 0.8]
  for frac in fracions:
    res_indices[frac] = sample(train_data, fraction=frac)

  save_coreset(config, res_indices, config['selection_method'])

def run(config_name, data_flag):
  Config.set_config(config_name, data_flag, 'uniform')

  main()
  
if __name__ == '__main__':
  selection_method = 'uniform'
  data_flag = sys.argv[1]
  config_name = sys.argv[2]
  Config.set_config(config_name, data_flag, selection_method)
  
  main()

