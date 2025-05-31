import medmnist

from torchvision.models import get_weight
from torchvision.transforms import transforms
import torch.utils.data as data

from src.utils.config import Config


def load_dataset(save_pth=None):
  config = Config.get_config()
  weights = get_weight(config['model_weights'])
  transform_func = transforms.Compose([
    transforms.ToTensor(),
    weights.transforms()
  ])

  train_dataset = config['DataClass'](split="train", as_rgb=True, transform=transform_func, download=True)
  test_dataset = config['DataClass'](split="test", as_rgb=True, transform=transform_func, download=True)
  validation_dataset = config['DataClass'](split="val", as_rgb=True, transform=transform_func, download=True)

  if save_pth is not None:
    train_dataset.save(save_pth + 'train')
    test_dataset.save(save_pth + 'test')
    validation_dataset.save(save_pth + 'val')

  return train_dataset, validation_dataset, test_dataset

def get_my_dataloaders(train, validation, test):
  config = Config.get_config()
  batch_size = config['batch_size']

  train_loader = data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
  validation_loader = data.DataLoader(dataset=validation, batch_size=2*batch_size, shuffle=False)
  test_loader = data.DataLoader(dataset=test, batch_size=2*batch_size, shuffle=False)

  return train_loader, validation_loader, test_loader
