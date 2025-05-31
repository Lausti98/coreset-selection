import torch

from src.model.evaluation import evaluate
from src.utils.config import Config


def train(model, trainloader, val_loader, optimizer, criterion, num_epochs):
    config = Config.get_config()
    device = config['device']
    task = config['task']
    model.to(device)
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    # Early stopping
    best_val_loss = float('inf')
    patience = 15
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            if task == 'binary-class':
                labels = labels.to(torch.float32)
            else:
                labels = labels.squeeze(1).long()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        training_losses.append(running_loss/len(trainloader))

        val_evals = evaluate(model, val_loader, criterion, 'val')
        val_loss = val_evals[0]
        val_acc = val_evals[2]
        validation_losses.append(val_loss)
        validation_accuracies.append(val_acc)

        # Perform early stopping with patience
        if val_loss < best_val_loss:
           best_val_loss = val_loss
           counter = 0
        else:
           counter += 1
           if counter >= patience:
            print("Early stopping triggered.")
            break

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(trainloader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}')

    return model, training_losses, training_accuracies, validation_losses, validation_accuracies
