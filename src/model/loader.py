from torchvision.models import get_model, get_weight
import torch.nn as nn
import torch

from src.utils.config import Config

def get_my_model(mode=None):
  config = Config.get_config()
  weights = get_weight(config['model_weights'])
  model = get_model(config['model_name'], weights=weights)

  if 'resnet' in config['model_name'].lower():
    model = define_resnet_linear_layer(model)
  elif 'efficientnet' in config['model_name'].lower():
    model = define_efficientnet_last_layer(model)
  elif 'densenet' in config['model_name'].lower():
    model = define_densenet_last_layer(model)
  if mode == 'eval':
    model.eval()

  if torch.cuda.is_available():
      model.cuda()
  return model

def define_resnet_linear_layer(model):
  config = Config.get_config()
  # Define last linear layer
  if config['task'] == 'binary-class':
    model.fc = nn.Linear(model.fc.in_features, 1)
  else:
    model.fc = nn.Linear(model.fc.in_features, config['n_classes'])
  return model

def define_efficientnet_last_layer(model):
  config = Config.get_config()
  # Define last linear layer
  num_ftrs = model.classifier[1].in_features
  if config['task'] == 'binary-class':
    model.classifier[1] = nn.Linear(num_ftrs, 1)
  else:
    model.classifier[1] = nn.Linear(num_ftrs, config['n_classes'])
  return model

def define_densenet_last_layer(model):
  config = Config.get_config()
  # Define last linear layer
  num_ftrs = model.classifier.in_features
  if config['task'] == 'binary-class':
    model.classifier = nn.Linear(num_ftrs, 1)
  else:
    model.classifier = nn.Linear(num_ftrs, config['n_classes'])
  return model