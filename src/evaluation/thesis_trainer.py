import sys
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

import os
import json

from src.model.evaluation import evaluate
from src.dataset.loader import get_my_dataloaders, load_dataset
from src.model.training import train
from src.model.loader import get_my_model
from src.utils.helper import get_base_path, save_result, load_checkpoint
from src.utils.config import Config

def select_subset(dataset, indices):
  if indices is None:
    return dataset
  return data.Subset(dataset, indices)


def run(indices=None, fraction=None, init_iteration=0, iterations=1):
  config = Config.get_config()
  task = config['task']
  train_data, validation_data, test_data = load_dataset()
  train_data = select_subset(train_data, indices)
  trainloader, validationloader, testloader = get_my_dataloaders(train_data, validation_data, test_data)
  # define loss function and optimizer¨
  if task == "binary-class":
      criterion = nn.BCEWithLogitsLoss()
  else:
      criterion = nn.CrossEntropyLoss()
  num_epochs = config['num_epochs']
  learning_rate = config['learning_rate']


  for iteration in range(init_iteration, iterations):
    model = get_my_model() # reinitialize model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model, training_losses, training_accuracies, validation_losses, validation_accuracies = train(model, trainloader, validationloader, optimizer, criterion, num_epochs)

    test_loss, test_auc, test_accuracy = evaluate(model, testloader, criterion, 'test')

    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
    if config['selection_method'] is not None:
      save_result(config, training_losses, validation_losses, validation_accuracies, test_loss, test_auc, test_accuracy, config['selection_method'], fraction, iteration)
    else:
      save_result(config, training_losses, validation_losses, validation_accuracies, test_loss, test_auc, test_accuracy)

def main():
  config = Config.get_config()
  assert torch.cuda.is_available() == True, "No GPUS allocated to run-time. Exiting.." ## Sometimes Slurm doesn't provide GPU
  if config['selection_method'] is not None:
    init_frac = 0.01
    init_iteration = 0
    max_iteration = 3
    
    init_frac, init_iteration, remaining = load_checkpoint(config, '/training_results', init_frac, init_iteration, max_iteration)
    if float(init_frac) > 0.4: return
    for frac, indices in remaining:
      run(indices, frac, init_iteration=init_iteration, iterations=max_iteration)
      init_iteration = 0
  else:
    run()

def execute(data_flag, selection_method, config_name):
  Config.set_config(config_name, data_flag, selection_method)
  main()

if __name__ == '__main__':
  data_flag = sys.argv[1]
  selection_method = sys.argv[2]
  config_name = sys.argv[3]

  Config.set_config(config_name, data_flag, selection_method)
  main()