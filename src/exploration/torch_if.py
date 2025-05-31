import pytorch_influence_functions as ptif
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import transforms
from load_medmnist import load_dataset
import torch.utils.data as data

### TODO: implement a config holder and get models from config
BATCH_SIZE = 16

def get_my_model(mode=None):
  ### TODO: implement a config holder and get models from config

  model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
  if mode == 'eval':
    model.eval()
  return model


def get_my_dataloaders():
  ### TODO: implement a config holder and get models from config
  weights = ResNet18_Weights.DEFAULT
  tranform_func = transforms.Compose([
    transforms.ToTensor(),
    weights.transforms()
  ])
  train, test = load_dataset(transform_func=tranform_func)
  print(train)
  print(test)

  train_loader = data.DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = data.DataLoader(dataset=test, batch_size=2*BATCH_SIZE, shuffle=False)
  return train_loader, test_loader


if __name__ == '__main__':
  model = get_my_model('eval')
  trainloader, testloader = get_my_dataloaders()

  ptif.init_logging()
  config = ptif.get_default_config()
  config['gpu'] = -1

  influences, harmful, helpful = ptif.calc_img_wise(config, model, trainloader, testloader)