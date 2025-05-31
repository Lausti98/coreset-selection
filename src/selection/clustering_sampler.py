import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import copy
from torch.autograd import grad
from sklearn.cluster import KMeans as KMeans_sklearn

import sys

from src.utils.config import Config
from src.model.evaluation import evaluate
from src.model.loader import get_my_model
from src.dataset.loader import get_my_dataloaders, load_dataset
from src.utils.helper import save_coreset, save_scores

def moso_scoring(net, dataloader, criterion, lr):
    config = Config.get_config()
    task = config['task']
    device = config['device']
    model = copy.deepcopy(net)
    new_dataloader = data.DataLoader( # Create a dataloader with batch_size 1 to get individual sample scores
      dataset=dataloader.dataset,
      batch_size=1,
      num_workers=dataloader.num_workers,
      pin_memory=dataloader.pin_memory,
      drop_last=False  # Ensure all samples are included
    )
    model.eval()
    overall_grad = 0

    params = [ p for p in model.parameters() if p.requires_grad ]
    for i, (inputs, labels) in enumerate(new_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        if task == 'binary-class':
          labels = labels.to(torch.float32)
        else:
          labels = labels.long().view(-1)
        logits = model(inputs)
        loss = criterion(logits, labels)
        g = list(grad(loss, params, create_graph=False))
        g = torch.nn.utils.parameters_to_vector(g)
        g = g.detach()
        overall_grad = overall_grad * i/(i+1) + g / (i+1)
        N = i+1
    overall_grad = overall_grad

    score_list = []
    for i, (inputs, labels) in enumerate(new_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        if task == 'binary-class':
          labels = labels.to(torch.float32)
        else:
          labels = labels.long().view(-1)
        logits = model(inputs)
        loss = criterion(logits, labels)
        g = list(grad(loss, params, create_graph=False))
        g = torch.nn.utils.parameters_to_vector(g)
        g = g.detach()
        score = ((overall_grad - (1/N * g)) * g).sum() * lr
        score = score.detach().cpu()
        score_list.append(score)

    score_list = torch.tensor(score_list).detach()
    return score_list

def train(model, trainloader, val_loader, optimizer, criterion, num_epochs, trainset_permutation_inds):
    config = Config.get_config()
    device = config['device']
    task = config['task']
    model.to(device)
    training_losses = []
    validation_losses = []
    validation_accuracies = []
    ### MoSo - instead of random selection of epochs, select epochs at
    ### even steps to make up 20% of total training epochs.
    moso_scores = torch.zeros(len(trainloader.dataset))
    num_scoring_epochs = int(num_epochs * 0.2)
    scoring_stride = num_epochs//num_scoring_epochs
    scoring_epochs = np.arange(scoring_stride, num_epochs, scoring_stride)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

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

        training_losses.append(running_loss/len(trainloader))


        ### MoSo scoring
        if epoch in scoring_epochs:
          v_scores = moso_scoring(model, trainloader, criterion, config['learning_rate'])
          moso_scores = moso_scores + v_scores
        val_evals = evaluate(model, val_loader, criterion, 'val')
        val_loss = val_evals[0]
        val_acc = val_evals[2]
        validation_losses.append(val_loss)
        validation_accuracies.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(trainloader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}')

    return model, moso_scores


def pretrain(indices=None, fraction=None, iterations=1):
  config = Config.get_config()
  task = config['task']
  batch_size = config['batch_size']
  train_data, validation_data, test_data = load_dataset()
  trainloader, validationloader, _ = get_my_dataloaders(train_data, validation_data, test_data)

  train_data_permutation = torch.randperm(len(train_data))

  batch_sampler = data.BatchSampler(train_data_permutation, batch_size=batch_size, drop_last=False)
  trainset_permutation_inds = list(batch_sampler)
  trainloader = data.DataLoader(train_data, batch_sampler=batch_sampler)

  model = get_my_model()
  num_epochs = config['num_pretrain_epochs']
  learning_rate = config['learning_rate']

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # define loss function and optimizer¨
  if task == "binary-class":
      criterion = nn.BCEWithLogitsLoss()
  else:
      criterion = nn.CrossEntropyLoss()

  model, moso_scores = train(model, trainloader, validationloader, optimizer, criterion, num_epochs, trainset_permutation_inds)
  return model, optimizer, criterion, trainset_permutation_inds, moso_scores

def create_matrix(model):
  """
    Find the feature embeddings of the model, and create a matrix of the embeddings.
    :return:
    feature matrix
  """
  model.eval()
  config = Config.get_config()
  device = config['device']
  batch_size = config['batch_size']

  train_data, validation_data, test_data = load_dataset()
  trainloader, _, _ = get_my_dataloaders(train_data, validation_data, test_data)
  feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])# .cpu()
  feature_matrix = torch.zeros([len(train_data), model.fc.in_features], requires_grad=False).to(device)
  with torch.no_grad():
    for i, (inputs, labels) in enumerate(trainloader):
      inputs = inputs.to(device)
      features = feature_extractor(inputs).float()
      features = features.view(features.size(0), -1)
      feature_matrix[i*batch_size:(i+1)*batch_size] = features
  return feature_matrix


def clustering_scorer(model, dataset, train_idx_flat, moso_scores, n_samples):
  ### Extract model embeddings
  feature_matrix = create_matrix(model)

  ## Convert moso scores to the original training index
  moso_scores = moso_scores[train_idx_flat]

  # Do kmeans by sklearn
  cluster_ids_x = KMeans_sklearn(n_clusters=n_samples, random_state=0).fit_predict(feature_matrix)

  res = []
  for cluster_id in range(n_samples):
    sample_ids_in_cluster = (cluster_ids_x == cluster_id).nonzero()[0]
    scores_in_cluster = moso_scores[sample_ids_in_cluster]
    res.append(sample_ids_in_cluster[torch.argmax(scores_in_cluster)])

  return res

def sample(model, dataset, train_idx, moso_scores, n_samples=None, fraction=None):
  print(fraction)
  train_idx_flat = np.array([int(x) for sublist in train_idx for x in sublist])
  if fraction is not None:
    n_samples = int(len(dataset) * fraction)

  scores = clustering_scorer(model, dataset, train_idx_flat, moso_scores, n_samples)

  
  return [int(idx) for idx in scores]


def main():
  config = Config.get_config()
  selection_name = config['selection_method']
  train_data, _, _ = load_dataset()
  assert torch.cuda.is_available() == True, "No GPUS allocated to run-time. Exiting.." ## Sometimes Slurm doesn't provide GPU
  res_indices = {}
  fracions = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8]
  if 'tissue' in config['data_flag']:
     fracions = [0.001, 0.005, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8]
  model, _, _, train_idx, moso_scores = pretrain()
  
  for frac in fracions:
    res_indices[frac] = sample(model, train_data, train_idx, moso_scores, fraction=frac)
  save_coreset(config, res_indices, selection_name)

def run(config_name, data_flag):
  Config.set_config(config_name, data_flag, 'clusteringmoso')

  main()
  
if __name__ == '__main__':
  selection_method = 'clusteringmoso'
  data_flag = sys.argv[1]
  config_name = sys.argv[2]
  Config.set_config(config_name, data_flag, selection_method)
  
  main()