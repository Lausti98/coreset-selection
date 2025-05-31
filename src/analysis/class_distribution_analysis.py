import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import cv2

import medmnist
from medmnist import INFO
import torchvision.transforms as transforms
import torch.utils.data as data

from src.utils.helper import get_base_path

# Navigate to a specific folder in your Drive
drive_path = get_base_path()
try:
  os.listdir(drive_path)  # List files in the directory
except FileNotFoundError:
  os.makedirs(drive_path)  # Create the directory if it doesn't exist
finally:
  print(os.listdir(drive_path))

source_dir = "coresets"
# List all files in training_results directory
training_results_path = os.path.join(drive_path, source_dir)
res_fps = os.listdir(training_results_path)

renaming_map = {
    "test_accuracy": "Test Accuracy",
    "test_auc": "Test AUC",
    "selection_method": "Selection Method",
    "fraction": "Sampling Ratio",
    "dermamnist": "DermaMNIST",
    "breastmnist": "BreastMNIST",
    "organamnist": "OrganAMNIST",
    "bloodmnist": "BloodMNIST",
    "pathmnist": "PathMNIST",
    "tissuemnist": "TissueMNIST",
    "moso": "MoSo",
    "uniform": "Random",
    "herding": "Herding",
    "forgetting": "Forgetting",
    "grand": "GraNd",
    "clusteringmoso": "Clustering MoSo",
    "greedymoso": "Greedy MoSo",
    "full": "Full Dataset"
}

selection_color_map = {
    "moso": "#ff7f0e",          # blue
    "uniform": "#1f77b4",       # orange
    "herding": "#2ca02c",       # green
    "forgetting": "#d62728",    # red
    "grand": "#8c564b",         # purple
    "clusteringmoso": "#9467bd", # brown
    "greedymoso": "#e377c2",    # pink
    "full": "#7f7f7f",          # gray
}

# For the 1% and 10% dataset sizes and moso, clusteringmoso, uniform
# plot the histogram of classes. Possibly with the full class distribution
# normalized to the 1% and 10% respectively. Should visually show how distribution
# is kept for the different methods.

def load(data_flag):
  # preprocessing
  data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
  ])
  info = INFO[data_flag]
  DataClass = getattr(medmnist, info['python_class'])

  train_dataset = DataClass(split='train', transform=data_transform, download=True, mmap_mode='r')

  return train_dataset

def get_subset(dataset, subset_idx):
  if subset_idx is None:
    return dataset
  return data.Subset(dataset, subset_idx)

def get_label_histograms(dataset, full_set, data_flag):
  if isinstance(dataset, data.Subset):
    labels = [dataset.dataset.labels[i] for i in dataset.indices] # Get labels using indices from the Subset
  else:
    labels = dataset.labels

  full_set_labels = full_set.labels

  hist, bins = np.histogram(labels, bins=range(np.unique_counts(full_set_labels)[0].size+1), density=True)
  return hist, bins[:-1]


# For each file path in results path, load in the json
sns.set_style("whitegrid")
selection_methods = ["moso", "uniform", "clusteringmoso", "greedymoso"]
data_flags = [elm.split("_")[1] for elm in res_fps]
correlations = []
out = []
for data_flag in data_flags:
  sels = [elm for elm in res_fps if data_flag in elm]
  sels = [elm for elm in sels if elm.split("_")[2].split(".")[0] in selection_methods]
  for fp in sels:
    selection_method = fp.split("_")[2].split(".")[0]
    with open(os.path.join(training_results_path, fp), "r") as f:
      raw = json.load(f)
      # strip the fp from file ending
    data_flag = fp.split("_")[1]
    dataset = load(data_flag)
    onep_dataset = get_subset(dataset, raw["0.01"])
    tenp_dataset = get_subset(dataset, raw["0.1"])
    hist_1p = get_label_histograms(onep_dataset, dataset, data_flag)
    hist_10p = get_label_histograms(tenp_dataset, dataset, data_flag)
    hist_full = get_label_histograms(dataset, dataset, data_flag)

    hist_1p_float_type = hist_1p[0].astype(np.float32).reshape(-1, 1)
    hist_10p_float_type = hist_10p[0].astype(np.float32).reshape(-1, 1)
    hist_full_float_type = hist_full[0].astype(np.float32).reshape(-1, 1)
    print(f'{hist_1p_float_type.shape=}')
    print(f'{hist_10p_float_type.shape=}')
    print(f'{hist_full_float_type.shape=}')
    print(f'{data_flag} - {selection_method}')
    onep_correlation = cv2.compareHist(hist_full_float_type, hist_1p_float_type, cv2.HISTCMP_CORREL)
    tenp_correlation = cv2.compareHist(hist_full_float_type, hist_10p_float_type, cv2.HISTCMP_CORREL)
    print(f'1% dataset correlation: {onep_correlation}')
    print(f'10% dataset correlation: {tenp_correlation}')
    correlations.append({'data_flag': data_flag,
                         'selection_method': selection_method,
                         '1_percent_correlation': onep_correlation,
                         '10_percent_correlation': tenp_correlation})

    width = 0.25
    fig, ax = plt.subplots()
  
    print(f'Counts : {hist_1p[0]}')
    print(f'Bins : {hist_1p[1]}')
  
    ax.bar(hist_1p[1] - width, hist_1p[0], width, color=(0.2, 0.4, 0.8, 0.7), label='1% Dataset')
    ax.bar(hist_10p[1], hist_10p[0], width, color=(0.2, 0.6, 0.2, 0.7), label='10% Dataset')
    ax.bar(hist_full[1] + width, hist_full[0], width, color=(0.8, 0.2, 0.0, 0.7), label='Full Dataset')

    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Add labels and legend
    ax.set_xlabel('Labels')
    ax.set_ylabel('Pct Class Distribution')
    ax.set_xticks(hist_full[1])
    ax.legend()
    plt.savefig(f'{drive_path}/plots/{data_flag}_{selection_method}_class_dist.pgf')
    plt.show()

correlation_df = pd.DataFrame(correlations)
correlation_df.to_csv(f'{drive_path}/correlation_results.csv')