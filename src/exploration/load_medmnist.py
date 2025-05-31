from medmnist import BreastMNIST
from src.utils.helper import get_base_path

def load_dataset(save_pth=None, transform_func=None):
  train_dataset = BreastMNIST(split="train", as_rgb=True, transform=transform_func, download=True)
  test_dataset = BreastMNIST(split="test", as_rgb=True, transform=transform_func, download=True)
  
  if save_pth is not None:
    train_dataset.save(save_pth + 'train')
    test_dataset.save(save_pth + 'test')

  return train_dataset, test_dataset


if __name__ == '__main__':
  dataset = BreastMNIST(split="train", download=True)
  print(dataset)
  dataset.save(f'{get_base_path()}dataset/breastmnist')

