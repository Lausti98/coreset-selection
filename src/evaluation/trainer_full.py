import sys

import torch

from src.evaluation.thesis_trainer import execute

data_flags = [
  "breastmnist",
  "dermamnist",
  "bloodmnist",
  "organamnist",
  "pathmnist",
  "tissuemnist"
]


if __name__ == '__main__':
  data_flag_idx = int(sys.argv[1]) - 1 # Ensure 0-index
  # config_name = sys.argv[3]
  assert torch.cuda.is_available() == True, "No GPUS allocated to run-time. Exiting.." ## Sometimes Slurm doesn't provide GPU
  execute(data_flags[data_flag_idx], None, None)