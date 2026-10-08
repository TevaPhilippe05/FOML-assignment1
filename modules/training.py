import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from modules.architecture import Net
from modules.utils import make_optimizer, make_scheduler
from modules.plotting import plot_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, epochs=40, lr=0.001, optimizer_name="SGD", scheduler_name=None, limit_epoch=10):

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if model.regularization == "weight_decay":
        optimizer = make_optimizer(optimizer_name, model.parameters(), lr, weight_decay=model.reg_param)
    else:
        optimizer = make_optimizer(optimizer_name, model.parameters(), lr)

    scheduler = make_scheduler(scheduler_name, optimizer)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    confusion_matrices = []
    best_val_loss = float('inf')
    epoch_counter = 0

    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)

            if model.regularization == "l1":
                l1 = 0
                for p in model.parameters():
                    l1 = l1 + p.abs().sum()
                loss = loss + model.reg_param * l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_losses.append(total_loss / total)
        train_accs.append(correct / total)

        
       
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                total_loss += loss.item() * xb.size(0)

                preds = out.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        val_loss = total_loss / total
        val_losses.append(val_loss)
        val_accs.append(correct / total)
        confusion_matrices.append(confusion_matrix(all_labels, all_preds, labels=list(range(model.num_classes))))

        if scheduler is not None:
            scheduler.step()

        pbar.set_postfix({'Train Loss': f'{train_losses[-1]:.4f}',
                          'Val Loss': f'{val_loss:.4f}',
                          'Val Acc': f'{val_accs[-1]:.4f}'})

        
        if model.regularization == "early_stopping":
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epoch_counter = 0
            else:
                epoch_counter += 1
            if epoch_counter >= limit_epoch:
                tqdm.write(f"Early stopping at epoch {epoch+1}")
                break

    return {
        "loss_train": train_losses,
        "accuracy_train": train_accs,
        "loss_val": val_losses,
        "accuracy_val": val_accs,
        "confusion_matrices": confusion_matrices,
        "model": model,
    }

def run_training(train_dataset, val_dataset, num_classes, hidden_sizes=(64,), activation_function="relu", regularization=None, reg_param=0.0,
                 optimizer_name="SGD", scheduler_name=None, epochs=40, lr=0.001, batch_size=64, limit_epoch=10, seed=42, convolution=None, in_channels=None, out_channels=None, kernel_size=None, pooling=None, pr=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    first_image, first_label = train_dataset[0]
    
    if convolution is not None:
        convol = convolution(in_channels, out_channels, kernel_size, pooling)
        input_size = convol(first_image.unsqueeze(0)).numel()
    else:
        convol = None
        input_size = first_image.numel()

    net = Net(input_size, hidden_sizes=hidden_sizes, num_classes=num_classes, convolution = convol, activation_function=activation_function, regularization=regularization,
              reg_param=reg_param)

    metrics = train(net, train_loader, val_loader, epochs=epochs, lr=lr, optimizer_name=optimizer_name, scheduler_name=scheduler_name, limit_epoch=limit_epoch)

    if pr: 
        plot_metrics(metrics, [10, 20, 40], class_names=train_dataset.classes)
        
    return metrics, net