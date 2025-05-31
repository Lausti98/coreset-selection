import os
import json

import torch
from torchvision.models import get_model, get_weight
from torchvision.transforms import transforms
import medmnist

from medmnist import INFO
import torch.utils.data as data
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import colorsys
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans as KMeans_sklearn

from src.utils.helper import get_base_path

## Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

drive_path = get_base_path()
try:
  os.listdir(drive_path)  # List files in the directory
except FileNotFoundError:
  os.makedirs(drive_path)  # Create the directory if it doesn't exist
finally:
  print(os.listdir(drive_path))

"""# Common utility functions"""

config_dir = drive_path + '/configs'

# Load the config file
with open(config_dir + '/config_clusteringmoso.json', 'r') as file:
    config = json.load(file)

download = True

## Overwrite config. Leave as None-type for using config
data_flag = "dermamnist"

if data_flag is None:
  data_flag = config['data_flag']

data_info = INFO[data_flag]
task = data_info['task']
n_channels = data_info['n_channels']
n_classes = len(data_info['label'])

DataClass = getattr(medmnist, data_info['python_class'])

def load_dataset(save_pth=None, transform_func=None):

  weights = get_weight(config['model_weights'])
  transform_func = transforms.Compose([
    transforms.ToTensor(),
    weights.transforms()
  ])

  train_dataset = DataClass(split="train", as_rgb=True, transform=transform_func, download=True)
  test_dataset = DataClass(split="test", as_rgb=True, transform=transform_func, download=True)
  validation_dataset = DataClass(split="val", as_rgb=True, transform=transform_func, download=True)

  if save_pth is not None:
    train_dataset.save(save_pth + 'train')
    test_dataset.save(save_pth + 'test')
    validation_dataset.save(save_pth + 'val')

  return train_dataset, validation_dataset, test_dataset

BATCH_SIZE = config['batch_size']

def get_my_model(mode=None):
  weights = get_weight(config['model_weights'])
  model = get_model(config['model_name'], weights=weights)

  # Define last linear layer
  if task == 'binary-class':
    model.fc = nn.Linear(model.fc.in_features, 1)
  else:
    model.fc = nn.Linear(model.fc.in_features, n_classes)

  if mode == 'eval':
    model.eval()

  if torch.cuda.is_available():
      model.cuda()
  return model


def get_my_dataloaders(train, validation, test):

  train_loader = data.DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)

  validation_loader = data.DataLoader(dataset=validation, batch_size=2*BATCH_SIZE, shuffle=False)

  test_loader = data.DataLoader(dataset=test, batch_size=2*BATCH_SIZE, shuffle=False)

  return train_loader, validation_loader, test_loader

def create_matrix(model):
  """
    Source: https://github.com/PatrickZH/DeepCore/blob/main/deepcore/methods/herding.py
    Find the feature embeddings of the model, and create a matrix of the embeddings.
    :return:
    feature matrix
  """
  model.eval()

  train_data, validation_data, test_data = load_dataset()
  trainloader, _, _ = get_my_dataloaders(train_data, validation_data, test_data)
  feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])# .cpu()
  feature_matrix = torch.zeros([len(train_data), model.fc.in_features], requires_grad=False).to(device)
  with torch.no_grad():
    for i, (inputs, labels) in enumerate(trainloader):
      inputs = inputs.to(device)
      features = feature_extractor(inputs).float()
      features = features.view(features.size(0), -1)
      feature_matrix[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = features
  return feature_matrix

"""# Visualization Utilities"""

def generate_colors(n):
    return [colorsys.hsv_to_rgb(i/n, 0.55, 0.85) for i in range(n)]

def extract_labels():
  train_data, _, _ = load_dataset()
  return train_data.labels

def color_clusters(cluster_ids_x):
  colors = generate_colors(n_clusters)
  return [colors[label] for label in cluster_ids_x]


def color_labels(y):
  class_colors = generate_colors(n_classes)
  return [class_colors[label[0]] for label in y]

def get_points_in_cluster(cluster_id):
  return np.where(cluster_ids_x == cluster_id)[0]

def plot_single_cluster(cluster_id, X_embedded, class_labels, moso_scores, normalize_score=False):
  cluster_ids = get_points_in_cluster(cluster_id)
  X_cluster = X_embedded[cluster_ids]
  labels = class_labels[cluster_ids]
  scores = np.array(moso_scores['scores'])[cluster_ids]
  if normalize_score is True:
    scores = (scores - scores.min()) / (scores.max() - scores.min())

  point_colors = color_labels(labels)
  _, idx = np.unique(labels, return_index=True)

  label_map = np.array(labels).ravel()[idx]

  # Create scatter plot
  fig, ax = plt.subplots(figsize=(8, 6))
  scatter = ax.scatter(
      X_cluster[:, 0],
      X_cluster[:, 1],
      s=[max(30, score*400) for score in scores],
      c=point_colors,        # Center color: class
      alpha=0.5,
      edgecolors=point_colors,  # Edge color: class
  )

  ## Mark the max scoring sample
  max_score_idx = np.argmax(scores)
  ax.scatter(
      X_cluster[max_score_idx, 0],
      X_cluster[max_score_idx, 1],
      c='yellow',
      s=20
  )

  # Build a legend for class labels
  for label, color in zip(label_map, np.array(point_colors)[idx]):
      plt.scatter([], [], c=color, edgecolors='gray', s=100, label=f"Class {label}")
  ax.legend(title="Class Label")

  plt.savefig(f'/content/{data_flag}_single_cluster_{cluster_id}_{"normalized" if normalize_score is True else "non_normalized"}.pgf')
  plt.show()

X = create_matrix(get_my_model())
X_embedded = TSNE(n_components=2, learning_rate='auto',
                   init='random', perplexity=100).fit_transform(X.cpu())

## Scatter plot using cluster labels
fraction = 0.1
n_samples = data_info['n_samples']['train']
n_clusters = int(n_samples * fraction)
cluster_ids_x = KMeans_sklearn(n_clusters=n_clusters, random_state=0).fit_predict(X.cpu())

y = extract_labels()
class_colors = generate_colors(n_classes)
point_colors = [class_colors[label[0]] for label in y]
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=10, c = color_clusters(cluster_ids_x))
plt.savefig(f'/content/dermamnist_clusters_distribution.pgf')
plt.show()

colors = generate_colors(n_clusters)
class_colors = generate_colors(n_classes)
# Extract the class labels
class_labels = y
#point_colors = [colors[label] for label in cluster_ids_x]
point_colors = [class_colors[label[0]] for label in y]
plt.figure(2)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=10, c=point_colors)
plt.xlim(-10, 10)  # Set this based on your data range
plt.ylim(-10, 10)
plt.savefig(f'/content/dermamnist_labels_10percent_zoom.pgf')
plt.show()

## Scatter plot using class labels
def get_points_in_cluster(cluster_id):
  return np.where(cluster_ids_x == cluster_id)[0]

c0_ids = get_points_in_cluster(5)
plt.figure()
plt.scatter(X_embedded[c0_ids, 0], X_embedded[c0_ids, 1], s=50)
plt.show()

## Obtain MoSo scores for each sample
def load_moso_scores():
  with open(drive_path + f'/saved_scores/{data_flag}_clusteringmoso.json', 'r') as file:
    moso_scores = json.load(file)
  return moso_scores

moso_scores = load_moso_scores()

c0_ids = get_points_in_cluster(100)
plt.figure()
plt.scatter(X_embedded[c0_ids, 0], X_embedded[c0_ids, 1], s=50, c=np.array(moso_scores['scores'])[c0_ids])
plt.colorbar()
plt.show()

c0_ids = get_points_in_cluster(5)

colors = np.array(class_labels[c0_ids])  
alphas = np.array(moso_scores['scores'])[c0_ids]  
alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())

plt.figure()
for cls in np.unique(colors):
    idx = (colors == cls).ravel()
    print(idx.shape)
    print(X_embedded[c0_ids].shape)
    plt.scatter(
        X_embedded[c0_ids][idx, 0],
        X_embedded[c0_ids][idx, 1],
        s=50,
        c=colors[idx],
        edgecolors=alphas[idx],
        label=f'Class {cls}'
    )

plt.legend(title="Class Label")
plt.show()

c0_ids = get_points_in_cluster(5)
X_cluster = X_embedded[c0_ids]
labels = class_labels[c0_ids]
scores = np.array(moso_scores['scores'])[c0_ids]

unique_labels = np.unique(labels)
label_to_color = {label[0]: plt.cm.tab10(i % 10) for i, label in enumerate(labels)}
point_colors = [label_to_color[label[0]] for label in labels]

# Create scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    X_cluster[:, 0],
    X_cluster[:, 1],
    s=100,
    c=plt.cm.viridis(scores), # Center color: scores
    edgecolors=point_colors,  # Edge color: class
    linewidths=3.5
)

for label, color in label_to_color.items():
    plt.scatter([], [], c=color, edgecolors='gray', s=100, label=f"Class {label}")
ax.legend(title="Class Label")

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=scores.min(), vmax=scores.max()))
sm.set_array([])
fig.colorbar(sm, label='Moso Score (center color)', ax=ax)
plt.show()

c0_ids = get_points_in_cluster(5)
X_cluster = X_embedded[c0_ids]
labels = class_labels[c0_ids]
scores = np.array(moso_scores['scores'])[c0_ids]
norm_scores = (scores - scores.min()) / (scores.max() - scores.min())

unique_labels = np.unique(labels)
label_to_color = {label[0]: plt.cm.tab10(i % 10) for i, label in enumerate(labels)}
point_colors = [label_to_color[label[0]] for label in labels]

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    X_cluster[:, 0],
    X_cluster[:, 1],
    s=100,
    edgecolors=point_colors,        # Center color: class
    c=plt.cm.viridis(norm_scores),  # Edge color: score
    linewidths=3.5
)

# Build a legend for class labels
for label, color in label_to_color.items():
    plt.scatter([], [], c=color, edgecolors='gray', s=100, label=f"Class {label}")
ax.legend(title="Class Label")

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=scores.min(), vmax=scores.max()))
sm.set_array([])
fig.colorbar(sm, label='Normalized Moso Score (edge color)', ax=ax)

plt.show()

top_scores = np.argsort(moso_scores['scores'])[::-1][:n_clusters]

y = extract_labels()[top_scores]
class_colors = generate_colors(n_classes)
point_colors = [class_colors[label[0]] for label in y]

plt.figure()
plt.scatter(X_embedded[top_scores, 0], X_embedded[top_scores, 1], s=10, c = point_colors)
plt.savefig(f'/content/dermamnist_selected_samples_10pct_score_only.pgf')
plt.show()

## Plot the selection-pct using score in combination with clustering
plt.figure()
extracted_labels = extract_labels()
scores_arr = np.array(moso_scores['scores'])
class_colors = generate_colors(n_classes)
for cluster_id in range(n_clusters):

  c0_ids = get_points_in_cluster(cluster_id)
  top_score = np.argsort(scores_arr[c0_ids])[::-1][0]

  y = extracted_labels[c0_ids][top_score]
  point_colors = [class_colors[label] for label in y]

  plt.scatter(X_embedded[c0_ids][top_score, 0], X_embedded[c0_ids][top_score, 1], s=10, c = point_colors)
plt.savefig(f'/content/dermamnist_selected_samples_10pct_with_clustering.pgf')
plt.show()

print(len(np.unique(cluster_ids_x)))
print(n_clusters)

plot_single_cluster(680, X_embedded, extract_labels(), moso_scores, normalize_score=True)