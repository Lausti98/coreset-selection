import sys

import torch

import src.selection.uniform_sampler as uniform
import src.selection.herding_sampler as herding
import src.selection.forgetting_sampler as forgetting
import src.selection.grand_sampler as grand
import src.selection.moso_sampler as moso
import src.selection.clustering_sampler as clustering
from src.evaluation.thesis_trainer import execute

selection_modules = [
  uniform,
  herding,
  forgetting,
  grand,
  moso,
  clustering
]

selection_methods = [
  "uniform",
  "herding",
  "forgetting",
  "grand",
  "moso",
  "clusteringmoso"
]

data_flags = [
  "breastmnist",
  "dermamnist",
  "bloodmnist",
  "organamnist",
  "pathmnist",
  "tissuemnist"
]


if __name__ == '__main__':
  selection_index = int(sys.argv[2]) - 1 # Ensure 0-index
  data_flag = sys.argv[1]
  # config_name = sys.argv[3]
  assert torch.cuda.is_available() == True, "No GPUS allocated to run-time. Exiting.." ## Sometimes Slurm doesn't provide GPU
  execute(data_flag, selection_methods[selection_index], selection_methods[selection_index])