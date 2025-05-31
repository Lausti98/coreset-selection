import json
from src.utils.helper import get_base_path
from medmnist import INFO
import medmnist
import torch

config_dir = f'{get_base_path()}/src/configs'


class Config:
  __conf = None

  @staticmethod
  def get_config():
    return Config.__conf
  
  
  def set_config(config_name, data_flag, selection_method):
    if config_name is not None:
      conf_fp = config_dir + f'/config_{config_name}.json'
    else:
      conf_fp = config_dir + f'/config.json'
    
    with open(conf_fp, 'r') as file:
        config = json.load(file)

    data_info = INFO[data_flag]
    config['data_flag'] = data_flag
    config['task'] = data_info['task']
    config['n_channels'] = data_info['n_channels']
    config['n_classes'] = len(data_info['label'])
    config['DataClass'] = getattr(medmnist, data_info['python_class'])
    config['selection_method'] = selection_method
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['outdir'] = config.get('outdir', None) # Ensure outdir key is present
    Config.__conf = config

