import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict


data_flag = "dermamnist"
result_dir = "/Users/laust/Desktop/hendrix_data/transfer_results/"
table_dir = "/Users/laust/Desktop/coreset-selection/outdir/"
prepend_title = "ResNet18 to EfficientNet: "

renaming_map = {
    "test_accuracy": "Test Accuracy",
    "test_auc": "Test AUC",
    "selection_method": "Selection Method",
    "fraction": "Sampling Ratio",
    "dermamnist": "DermaMNIST",
    "breastmnist": "BreastMNIST",
    "organamnist": "OrganAMNIST",
    "moso": "MoSo",
    "uniform": "Random",
    "herding": "Herding",
    "forgetting": "Forgetting",
    "grand": "GraNd",
    "clusteringmoso": "Clustering MoSo",
    "greedymoso": "Greedy MoSo",
    "full": "Full Dataset"
}

def plot(df, target_key, prepend_title=""):
  # Use seaborn style
  sns.set_theme(style="whitegrid")

  # Create the plot
  plt.figure(figsize=(10, 6))

  # Use seaborn color palette
  palette = sns.color_palette("tab10", n_colors=len(df["selection_method"].unique()))


  for i, selection in enumerate(df["selection_method"].unique()):
    if selection != "full":
      subset = df[df["selection_method"] == selection]
      #plt.plot(subset["fraction"], subset["test_accuracy"], label=selection)
      sns.lineplot(x=subset["fraction"], y=subset[target_key], label=renaming_map[selection], color=palette[i], linewidth=0.8)
      plt.fill_between(subset["fraction"], subset[target_key] - subset[f'{target_key}_std'], subset[target_key] + subset[f'{target_key}_std'], color=palette[i], alpha=0.2)

  target_key_formatted = ' '.join([x.capitalize() for x in target_key.split('_')])
  full_score_df = df[df["selection_method"] == "full"]
  if not full_score_df.empty:
    full_score = df[df["selection_method"] == "full"][target_key].values[0]  # Assuming a single score for 'full'
    plt.axhline(y=full_score, color="black", linestyle="dotted", linewidth=2, label="Full Selection")
  plt.xlabel("Sampling Ratio")
  plt.ylabel(target_key_formatted)
  plt.title(prepend_title + f"{target_key_formatted} over Sampling Ratios for {renaming_map[data_flag]}")

  plt.legend(title="Selection Method", loc='lower left') # bbox_to_anchor=(1.05, 1),
  plt.tight_layout()
  plt.savefig(f'{table_dir}{df.dataset.loc[0]}_{target_key}.pgf')

  # Show the plot
  plt.show()


def save_table(df, target_key):
  df[f'{target_key}_str_format'] = df.apply(lambda r: f'{r[target_key]:.3f} $\pm$ {r[f"{target_key}_std"]:.3f}', axis=1)
  pivot = df.pivot_table(values=f'{target_key}_str_format', index='selection_method', columns='fraction', aggfunc='first')
  pivot.index.names = [renaming_map['selection_method']]
  pivot.columns.names = [renaming_map['fraction']]
  pivot = pivot.rename(renaming_map)
  print(pivot)
  pivot.to_latex(f"{table_dir}{df.dataset.loc[0]}_{target_key}.tex", float_format="%.3f")

# For each file path in results path, load in the json
plt.figure()

training_results_path = result_dir # os.path.join(drive_path, "training_results")
res_fps = os.listdir(training_results_path)
res_fps = [fp for fp in res_fps if data_flag in fp]

out = []
for res_fp in res_fps:
  with open(os.path.join(training_results_path, res_fp), "r") as f:
    raw = json.load(f)
    # strip the fp from file ending


    if "0" in raw.keys():
      res = defaultdict(float)
      stds = defaultdict(list)

      for iter_idx in raw:
        # res["test_loss"] = iteration[]

        res["test_accuracy"] += raw[iter_idx]["test_accuracy"]
        res["test_auc"] += raw[iter_idx]["test_auc"]
        stds["test_accuracy"].append(raw[iter_idx]["test_accuracy"])
        stds["test_auc"].append(raw[iter_idx]["test_auc"])
      res["test_accuracy"] /= len(raw)
      res["test_auc"] /= len(raw)
      res["test_accuracy_std"] = np.std(stds["test_accuracy"])
      res["test_auc_std"] = np.std(stds["test_auc"])
      # res["test_loss
      res = dict(res)
    else:
      res = raw

    fp_split = Path(res_fp).stem.split('_')
    if len(fp_split) == 2:
      res['dataset'] = fp_split[1]
      res['selection_method'] = 'full'
      res['fraction'] = '1'
      out.append(res)
    else:
      res['dataset'] = fp_split[1]
      res['selection_method'] = fp_split[2]
      res['fraction'] = fp_split[3]
      out.append(res)
    # Create a plot of test_accuracies for each result

df = pd.DataFrame(out)
df.head()

df = df.sort_values(by='fraction', ascending=False)
df.head()

plot(df, "test_accuracy", prepend_title)
save_table(df, "test_accuracy")

plot(df, "test_auc", prepend_title)
save_table(df, "test_auc")