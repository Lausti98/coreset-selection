import os
import json

# from src.utils.config import Config

def get_base_path():
  """Get the base path of project"""

  cur_dir = os.getcwd()
  src_path = cur_dir.rfind('src')

  if src_path > 0:
    base_path = cur_dir[0:src_path]
  else:
    # Already in base directory
    base_path = f'{cur_dir}/'

  return base_path


def save_result(config, train_losses, validation_losses, validation_accuracies, test_loss, test_auc, test_accuracy, selection_method=None, fraction=None, iteration = None):
  # config = Config.get_config()
  result_dict = locals().copy()
  result_dict.pop('config')
  if config['outdir'] is None:
    result_dir = '/training_results'
  else:
    result_dir = config['outdir']
  result_path = get_base_path() + result_dir
  filepath = result_path + f'/results_{config["data_flag"]}{"_"+selection_method if selection_method is not None else ""}{"_"+fraction if fraction is not None else ""}.json'
  try:
    os.listdir(result_path)  # List files in the directory
  except FileNotFoundError:
    os.makedirs(result_path)  # Create the directory if it doesn't exist

  if iteration is not None:
    if iteration == 0:
      with open(filepath, 'w') as file:
        json.dump({iteration: result_dict}, file)
    else:
      with open(filepath, 'r+') as file:
        file_contents = json.load(file)
        file_contents[iteration] = result_dict
        file.seek(0)
        json.dump(file_contents, file)
  else:
    with open(filepath, 'w') as file:
      json.dump(result_dict, file)


def load_checkpoint(config, checkpoint_dir, init_frac = 0.1, init_iteration = 0, max_iteration = 3):
  # config = Config.get_config()
  if config['outdir'] is not None:
    checkpoint_dir = config['outdir']
  checkpoint_path = get_base_path() + checkpoint_dir
  data_flag = config['data_flag']
  # Load the selected indices
  indices_fp = get_base_path() + f'/coresets/indices_{data_flag}_{config["selection_method"]}.json'
  with open(indices_fp, 'r') as file:
    selected_indices_dict = json.load(file)

  # Load in a checkpoint if any exists
  for frac, indices in selected_indices_dict.items():
    checkpoint_fp = checkpoint_path + f'/results_{data_flag}_{config["selection_method"]}_{frac}.json'
    if os.path.exists(checkpoint_fp):
      print(f'Found checkpoint {checkpoint_fp}')
      with open(checkpoint_fp, 'r') as file:
        checkpoint = json.load(file)
        checkpoint_completed_iter = int(list(checkpoint.keys())[-1])
        init_frac = float(frac)
        if checkpoint_completed_iter < max_iteration-1:
          init_iteration = checkpoint_completed_iter + 1
        else:
          try:
            init_frac = [float(f) for f in selected_indices_dict.keys() if float(f) > float(frac)][0]
          except:
            init_frac = init_frac
    else:
      break

  remaining = [(frac, indices) for frac, indices in selected_indices_dict.items() if float(frac) >= init_frac]
  return init_frac, init_iteration, remaining

def save_coreset(config, indices : list, selection_name):
#  result_dict = {}
#  result_dict[selection_name] = indices
  result_dir = 'coresets'
  result_path = get_base_path() + f'/{result_dir}'
  data_flag = config['data_flag']
  try:
    os.listdir(result_path)  # List files in the directory
  except FileNotFoundError:
    os.makedirs(result_path)  # Create the directory if it doesn't exist

  with open(result_path + f'/indices_{data_flag}_{selection_name}.json', 'w') as file:
    json.dump(indices, file)

def save_scores(config, selection_name, scores):
  score_dir = 'saved_scores'
  result_path = get_base_path() + f'/{score_dir}'
  scores = [float(s) for s in scores]
  try:
    os.listdir(result_path)  # List files in the directory
  except FileNotFoundError:
    os.makedirs(result_path)  # Create the directory if it doesn't exist
  
  out = {'scores': scores}
  with open(result_path + f'/{config["data_flag"]}_{selection_name}.json', 'w') as file:
    json.dump(out, file)

