import sys

import torch

import src.selection.uniform_sampler as uniform
import src.selection.herding_sampler as herding
import src.selection.forgetting_sampler as forgetting
import src.selection.grand_sampler as grand
import src.selection.moso_sampler as moso
import src.selection.clustering_sampler as clustering
from src.evaluation.thesis_trainer_single_subset import execute

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
  run_idx = int(sys.argv[1]) - 1 # Ensure 0-index
  residual = run_idx % len(selection_methods)
  i = run_idx // len(selection_methods)

  data_flag = data_flags[i]
  selection_method = selection_methods[residual]

  print(data_flag)
  print(selection_method)
  subset = sys.argv[2]
  # assert torch.cuda.is_available() == True, "No GPUS allocated to run-time. Exiting.." ## Sometimes Slurm doesn't provide GPU
  execute(subset, data_flag, selection_method, selection_method)