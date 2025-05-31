
import torch

import torch.utils.data as data
import torch.nn as nn
import numpy as np

import sys

from src.utils.config import Config
from src.model.evaluation import evaluate, evaluation_func
from src.model.loader import get_my_model
from src.dataset.loader import get_my_dataloaders, load_dataset
from src.utils.helper import save_coreset

selection_name = 'forgetting'

def train(model, trainloader, val_loader, optimizer, criterion, num_epochs, trainset_permutation_inds):
    config = Config.get_config()
    device = config['device']
    task = config['task']
    model.to(device)
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    # Forgetting specific
    n_train = len(trainloader.dataset)
    forgetting_events = torch.zeros(n_train, requires_grad=False).to(device)
    last_acc = torch.zeros(n_train, requires_grad=False).to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            if task == 'binary-class':
                labels = labels.to(torch.float32)
            else:
              labels = labels.squeeze(1).long()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            ### Here we need to do forgetting.
            # We find the accuracy in this training step on the batch,
            # and compare to previous accuracy of this batch. If cur_acc > last_acc
            # We increment the foregetting events by one on the samples in this batch.
            # if not isinstance(outputs, torch.Tensor):
            outputs = torch.tensor([outputs]).view(-1).to(device)
            # if not isinstance(labels, torch.Tensor):
            labels = torch.tensor([labels]).view(-1).to(device)
            _, acc = evaluation_func(outputs, labels, None)
            cur_acc = torch.tensor(acc).to(device)
            batch_inds = torch.tensor(trainset_permutation_inds[i]).to(device)
            forgetting_events[batch_inds[(last_acc[batch_inds]-cur_acc)>0.01]] += 1.
            last_acc[batch_inds] = cur_acc


        training_losses.append(running_loss/len(trainloader))

        val_evals = evaluate(model, val_loader, criterion, 'val')
        val_loss = val_evals[0]
        val_acc = val_evals[2]
        validation_losses.append(val_loss)
        validation_accuracies.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(trainloader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}')

    return model, forgetting_events, last_acc

def pretrain(indices=None, fraction=None, iterations=1):
  config = Config.get_config()
  task = config['task']
  batch_size = config['batch_size']
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

  model, forgetting_events, last_acc = train(model, trainloader, validationloader, optimizer, criterion, num_epochs, trainset_permutation_inds)
  return forgetting_events, trainset_permutation_inds



def forgetting(train_idx, forgetting_events, budget: int, index=None):
  """
  Source: https://github.com/PatrickZH/DeepCore/blob/main/deepcore/methods/herding.py
  """
  train_idx_flat = np.array([int(x) for sublist in train_idx for x in sublist])
  sorted_indices = np.argsort(forgetting_events.cpu().numpy())[::-1]
  top_examples = train_idx_flat[np.argsort(forgetting_events.cpu().numpy())][::-1][:budget]
  return top_examples

def sample(dataset, train_idx, forgetting_events, n_samples=None, fraction=None):
  print(fraction)
  if fraction is not None:
    n_samples = int(len(dataset) * fraction)

  indices = forgetting(train_idx, forgetting_events,  n_samples)
  return [int(idx) for idx in indices]


def main():
  config = Config.get_config()
  train_data, val_data, test_data = load_dataset()
  assert torch.cuda.is_available() == True, "No GPUS allocated to run-time. Exiting.." ## Sometimes Slurm doesn't provide GPU
  res_indices = {}
  fracions = [0.1, 0.2, 0.4, 0.6, 0.8]
  forgetting_events, train_idx = pretrain()
  for frac in fracions:
    res_indices[frac] = sample(train_data, train_idx, forgetting_events, fraction=frac)

  save_coreset(config, res_indices, config['selection_method'])

def run(config_name, data_flag):
  Config.set_config(config_name, data_flag, 'forgetting')

  main()
  
if __name__ == '__main__':
  selection_method = 'forgetting'
  data_flag = sys.argv[1]
  config_name = sys.argv[2]
  Config.set_config(config_name, data_flag, selection_method)
  
  main()