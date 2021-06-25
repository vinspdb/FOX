import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
dtype = torch.float

def make_one_hot(data, num_categories, dtype=torch.float):
    num_entries = len(data)
    # Convert data to a torch tensor of indices, with extra dimension:
    cats = torch.Tensor(data).long().unsqueeze(1)
    # Now convert this to one-hot representation:
    y = torch.zeros((num_entries, num_categories), dtype=dtype)\
        .scatter(1, cats, 1)
    y.requires_grad = True
    return y

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_anfis_cat(model, train_loader, val_loader, optimizer, EPOCHS):
    print(device)
    print("Begin training.")
    accuracy_stats = {
            'train': [],
            "val": []}
    loss_stats = {
            'train': [],
            "val": [] }
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = np.inf
    best_epoch = 0
    for e in tqdm(range(1, EPOCHS + 1)):

        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        #model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.fit_coeff(X_train_batch.float(), make_one_hot(y_train_batch.float(), num_categories=2))
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        if (val_epoch_loss / len(val_loader)) < best_loss:
            best_loss = (val_epoch_loss / len(val_loader))
            best_epoch = e
            best_model = model
        if e - best_epoch > 10:
            print(best_epoch)
            break
        print(f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.8f} | Val Loss: {val_epoch_loss / len(val_loader):.8f} | Train Acc: {train_epoch_acc / len(train_loader):.8f}| Val Acc: {val_epoch_acc / len(val_loader):.8f}')
    return best_model, loss_stats['val']

