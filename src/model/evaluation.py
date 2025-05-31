import torch
from medmnist.evaluator import getACC, getAUC

from src.utils.config import Config

def evaluate(model, val_loader, criterion, split):
    config = Config.get_config()
    device = config['device']
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if config['task'] == 'binary-class':
              labels = labels.to(torch.float32)
            else:
              labels = labels.squeeze(1).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            y_true = torch.cat((y_true, labels), 0)
            y_score = torch.cat((y_score, outputs), 0)

    val_acc = correct / total
    loss = val_loss/len(val_loader)
    metrics = evaluation_func(y_score, y_true, split)

    return loss, *metrics

def evaluation_func(outputs, labels, split):
  config = Config.get_config()
  y_true = labels.cpu().numpy()
  y_score = outputs.detach().cpu().numpy()

  #evaluator = Evaluator(data_flag, split)
  #metrics = evaluator.evaluate(y_score)
  ## Bypass evaluator.evaluate and get scores directly.
  ## This makes it easier to use this evaluation func
  ## on other datasets as well if necessary. It also
  ## enables evaluating the training scores for sampled
  ## training sets, though this is likely unneccessary.
  acc = getACC(y_true, y_score, config['task'])
  auc = getAUC(y_true, y_score, config['task'])

  return auc, acc
