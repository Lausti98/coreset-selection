import sys

import torch

import src.selection.uniform_sampler as uniform
import src.selection.herding_sampler as herding
import src.selection.forgetting_sampler as forgetting
import src.selection.grand_sampler as grand
import src.selection.moso_sampler as moso
import src.selection.clustering_sampler as clustering

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
  selection_module_index = int(sys.argv[1]) - 1 # Ensure 0-index
  data_flag_index = int(sys.argv[2]) - 1 
  # assert torch.cuda.is_available() == True, "No GPUS allocated to run-time. Exiting.." ## Sometimes Slurm doesn't provide GPU
  selection_modules[selection_module_index].run(selection_methods[selection_module_index], data_flags[data_flag_index])