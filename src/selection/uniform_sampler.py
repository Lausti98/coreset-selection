import sys

import torch

from src.utils.config import Config
from src.dataset.loader import load_dataset
from src.utils.helper import save_coreset

def sample(dataset, n_samples=None, fraction=None):
  if fraction is not None:
    n_samples = int(len(dataset) * fraction)

  indices = torch.randperm(len(dataset)).numpy()[:n_samples]

  return [int(idx) for idx in indices]

def main():
  config = Config.get_config()
  train_data, _, _ = load_dataset()

  res_indices = {}
  fracions = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8]
  if 'tissue' in config['data_flag']:
    fracions = [0.001, 0.005, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8]
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

