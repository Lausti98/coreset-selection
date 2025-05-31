import os
import json

from src.utils.helper import get_base_path

def extract_1percent_dataset(fp):
  coreset_path = f'{get_base_path()}/coresets/'
  # load in the file
  full_path = coreset_path + fp
  if not os.path.exists(full_path):
    print(f"Path {full_path} doesn't exist")
  with open(full_path, 'r') as r_file:
    sets = json.load(r_file)
  
  # locate the 10% dataset
  # select 10% top samples in 10% dataset to get 1% dataset.
  ten_p = sets["0.1"]
  one_p_size = len(ten_p) // 10
  # [0.0001, 0.005]
  sets["0.01"] = ten_p[:one_p_size]
  sets = dict(sorted(sets.items()))
  with open(full_path, 'w') as w_file:
    json.dump(sets, w_file)

if __name__ == '__main__':
  coreset_fnames = os.listdir(f'{get_base_path()}/coresets/')
  for fname in coreset_fnames:
    if ('clustering' in fname):
      continue
    elif ('tissuemnist_herding' in fname):
      extract_1percent_dataset(fname)