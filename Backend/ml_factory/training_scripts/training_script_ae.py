import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Literal

def training(training_loader: DataLoader,
             val_loader: DataLoader,
             test_loader: Optional[DataLoader],
             model: nn.Module,
             optimizer: torch.optim.Optimizer,
             criterion: nn.Module, epoch: int = 100, device: Literal['cuda', 'cpu'] = 'cpu'):

    training_losses = []
    validation_losses = []
    best_model = float('inf')

    best_model_settings = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': 0}
    for _e in range(epoch):
        model.train()
        train_loss = 0
        validation_loss = 0
        for X, _ in training_loader:
            optimizer.zero_grad()
            X = X.to(device)
            X_hat, _ = model(X)
            loss = criterion(X_hat, X)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach()

        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                X_hat, _ = model(X)
                loss = criterion(X_hat, X)

                validation_loss += loss.detach()
        train_loss_n = train_loss.item() / len(training_loader)
        validation_loss_n = validation_loss.item() / len(val_loader)

        if best_model > validation_loss_n:
            best_model = validation_loss_n
            best_model_settings = {
                'model': {k: v.clone() for k, v in model.state_dict().items()},
                'optimizer': optimizer.state_dict(),
                'epoch': _e,
            }

        training_losses.append(train_loss_n)
        validation_losses.append(validation_loss_n)

        print(f"[{_e+1:<4}/{epoch}] | train loss: {train_loss_n:<10.4f} | validation loss: {validation_loss_n:<10.4f} |")

    model.load_state_dict(best_model_settings['model'])
    test_loss_n = None
    if test_loader is not None:

        test_loss = 0
        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                X_hat, _ = model(X)
                loss = criterion(X_hat, X)

                test_loss += loss.item()

        test_loss_n = test_loss / len(test_loader)

    torch.cuda.empty_cache()
    return best_model_settings, training_losses, validation_losses, test_loss_n


def compute_oe_loss(X_hat, X, y, criterion, oe_weight=1.0, oe_margin=1.0, return_components: bool = True):
    per_sample = criterion(X_hat, X).mean(dim=1)

    benign_mask = (y == 0)
    attack_mask = (y == 1)

    benign_loss = per_sample[benign_mask].mean() if benign_mask.any() else torch.tensor(0.0, device=X.device)
    if attack_mask.any():
        attack_loss = torch.clamp(oe_margin - per_sample[attack_mask], min=0).mean()

    else:
        attack_loss = torch.tensor(0.0, device=X.device)

    total = benign_loss + oe_weight * attack_loss

    if return_components:
        return total, benign_loss.detach(), attack_loss.detach()
    return total

def training_attack(training_loader: DataLoader,
             val_loader: DataLoader,
             test_loader: Optional[DataLoader],
             model: nn.Module,
             optimizer: torch.optim.Optimizer,
             epoch: int = 100, device: Literal['cuda', 'cpu'] = 'cpu'):

    training_losses, training_benign_losses, training_attack_losses = [], [], []
    validation_losses, validation_benign_losses, validation_attack_losses = [], [], []
    best_model = float('inf')
    criterion = nn.MSELoss(reduction="none")

    best_model_settings = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': 0}
    val_n = len(val_loader)
    train_n = len(training_loader)
    for _e in range(epoch):
        if hasattr(training_loader.sampler, 'set_epoch'):
            training_loader.sampler.set_epoch(_e)
        model.train()
        train_loss, train_attack_loss, train_benign_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        validation_loss, validation_attack_loss, validation_benign_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        for X, y in training_loader:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            X_hat, _ = model(X)
            loss, benign_loss, attack_loss = compute_oe_loss(X_hat, X, y, criterion)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach()
            train_attack_loss += attack_loss.detach()
            train_benign_loss += benign_loss.detach()

        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                X_hat, _ = model(X)
                loss, benign_loss, attack_loss = compute_oe_loss(X_hat, X, y, criterion)

                validation_loss += loss.detach()

                validation_attack_loss += attack_loss.detach()
                validation_benign_loss += benign_loss.detach()

        train_loss_n = train_loss.cpu().item() / train_n
        validation_loss_n = validation_loss.cpu().item() / val_n
        train_attack_loss_n = train_attack_loss.cpu().item() / train_n
        train_benign_loss_n = train_benign_loss.cpu().item() / train_n
        validation_attack_loss_n = validation_attack_loss.cpu().item() / val_n
        validation_benign_loss_n = validation_benign_loss.cpu().item() / val_n

        if best_model > validation_loss_n:
            best_model = validation_loss_n
            best_model_settings = {
                            'model': {k: v.clone() for k, v in model.state_dict().items()},
                            'optimizer': optimizer.state_dict(),
                            'epoch': _e,
                        }

        training_losses.append(train_loss_n)
        validation_losses.append(validation_loss_n)
        training_attack_losses.append(train_attack_loss_n)
        training_benign_losses.append(train_benign_loss_n)
        validation_attack_losses.append(validation_attack_loss_n)
        validation_benign_losses.append(validation_benign_loss_n)
        print(f"[{_e+1:<4}/{epoch}] | train loss: {train_loss_n:<5.5f} - a {train_attack_loss_n:<5.5f} - b {train_benign_loss_n:<5.5f} | validation loss: {validation_loss_n:<5.5f} - a {validation_attack_loss_n:<5.5f} - b {validation_benign_loss_n:<5.5f} |")

    model.load_state_dict(best_model_settings['model'])
    test_loss_n = None
    test_attack_loss_n = None
    test_benign_loss_n = None
    if test_loader is not None:

        test_loss, test_benign_loss, test_attack_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                X_hat, _ = model(X)
                loss, benign_loss, attack_loss = compute_oe_loss(X_hat, X, y, criterion)
                test_loss += loss.detach()
                test_attack_loss += attack_loss.detach()
                test_benign_loss += benign_loss.detach()

        test_loss_n = test_loss.cpu().item() / len(test_loader)
        test_attack_loss_n = test_attack_loss.cpu().item() / len(test_loader)
        test_benign_loss_n = test_benign_loss.cpu().item() / len(test_loader)

    torch.cuda.empty_cache()
    return best_model_settings, (training_losses,
                                 training_benign_losses,
                                 training_attack_losses),(validation_losses,
                                                           validation_benign_losses,
                                                          validation_attack_losses), (test_loss_n, test_benign_loss_n, test_attack_loss_n)

def contrastive_loss(z: torch.Tensor, y: torch.Tensor, margin=1.0):
    benign_mask = (y == 0)
    attack_mask = (y == 1)

    if not benign_mask.any():
        return torch.tensor(0.0, device=z.device)

    benign_centroid = z[benign_mask].mean(dim=0, keepdim=True).detach()

    benign_dist = ((z[benign_mask] - benign_centroid) ** 2).sum(dim=1)

    benign_term = benign_dist.mean()

    if attack_mask.any():
        attack_dist = ((z[attack_mask] - benign_centroid) ** 2).sum(dim=1)
        attack_term = torch.clamp(margin - attack_dist, min=0).mean()
    else:
        attack_term = torch.tensor(0.0, device=z.device)

    return benign_term + attack_term



def training_attack_contrastive(training_loader: DataLoader,
             val_loader: DataLoader,
             test_loader: Optional[DataLoader],
             model: nn.Module,
             optimizer: torch.optim.Optimizer,
             epoch: int = 100, device: Literal['cuda', 'cpu'] = 'cpu'):

    training_losses, training_benign_losses, training_attack_losses = [], [], []
    validation_losses, validation_benign_losses, validation_attack_losses = [], [], []
    training_contrastive_losses, validation_contrastive_losses = [], []
    best_model = float('inf')
    criterion = nn.MSELoss(reduction="none")

    best_model_settings = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': 0}
    val_n = len(val_loader)
    train_n = len(training_loader)
    for _e in range(epoch):
        if hasattr(training_loader.sampler, 'set_epoch'):
            training_loader.sampler.set_epoch(_e)
        model.train()
        train_loss, train_attack_loss, train_benign_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        validation_loss, validation_attack_loss, validation_benign_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        train_contrastive_loss = torch.tensor(0.0, device=device)
        validation_contrastive_loss = torch.tensor(0.0, device=device)
        for X, y in training_loader:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            X_hat, encoded = model(X)
            loss, benign_loss, attack_loss = compute_oe_loss(X_hat, X, y, criterion)
            c_loss = contrastive_loss(encoded, y)
            loss = loss + c_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.detach()
            train_contrastive_loss += c_loss.detach()
            train_attack_loss += attack_loss
            train_benign_loss += benign_loss

        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                X_hat, encoded = model(X)
                loss, benign_loss, attack_loss = compute_oe_loss(X_hat, X, y, criterion)

                c_loss = contrastive_loss(encoded, y)
                loss = loss + c_loss
                validation_loss += loss.detach()

                validation_contrastive_loss += c_loss.detach()
                validation_attack_loss += attack_loss.detach()
                validation_benign_loss += benign_loss.detach()

        train_loss_n = train_loss.cpu().item() / train_n
        validation_loss_n = validation_loss.cpu().item() / val_n
        train_attack_loss_n = train_attack_loss.cpu().item() / train_n
        train_benign_loss_n = train_benign_loss.cpu().item() / train_n
        validation_attack_loss_n = validation_attack_loss.cpu().item() / val_n
        validation_benign_loss_n = validation_benign_loss.cpu().item() / val_n
        train_contrastive_loss_n = train_contrastive_loss.cpu().item() / train_n
        validation_contrastive_loss_n = validation_contrastive_loss.cpu().item() / val_n

        if best_model > validation_loss_n:
            best_model = validation_loss_n
            best_model_settings = {
                            'model': {k: v.clone() for k, v in model.state_dict().items()},
                            'optimizer': optimizer.state_dict(),
                            'epoch': _e,
                        }

        training_losses.append(train_loss_n)
        validation_losses.append(validation_loss_n)
        training_attack_losses.append(train_attack_loss_n)
        training_benign_losses.append(train_benign_loss_n)
        validation_attack_losses.append(validation_attack_loss_n)
        validation_benign_losses.append(validation_benign_loss_n)
        training_contrastive_losses.append(train_contrastive_loss_n)
        validation_contrastive_losses.append(validation_contrastive_loss_n)
        print(f"[{_e+1:<4}/{epoch}] | train loss: {train_loss_n:<5.5f} - a {train_attack_loss_n:<5.5f} - b {train_benign_loss_n:<5.5f} - c {train_contrastive_loss_n:<5.5f} | validation loss: {validation_loss_n:<5.5f} - a {validation_attack_loss_n:<5.5f} - b {validation_benign_loss_n:<5.5f} -c {validation_contrastive_loss_n:<5.5f} |")

    model.load_state_dict(best_model_settings['model'])
    test_loss_n = None
    test_attack_loss_n = None
    test_benign_loss_n = None
    test_contrastive_loss_n = None
    if test_loader is not None:

        test_loss, test_benign_loss, test_attack_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        test_contrastive_loss = torch.tensor(0.0, device=device)
        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                X_hat, encoded = model(X)
                loss, benign_loss, attack_loss = compute_oe_loss(X_hat, X, y, criterion)
                c_loss =contrastive_loss(encoded, y)
                loss = loss + c_loss
                test_loss += loss.detach()
                test_contrastive_loss += c_loss.detach()
                test_attack_loss += attack_loss
                test_benign_loss += benign_loss

        test_loss_n = test_loss.cpu().item() / len(test_loader)
        test_attack_loss_n = test_attack_loss.cpu().item() / len(test_loader)
        test_benign_loss_n = test_benign_loss.cpu().item() / len(test_loader)
        test_contrastive_loss_n = test_contrastive_loss.cpu().item() / len(test_loader)

    torch.cuda.empty_cache()
    return best_model_settings, (training_losses,
                                 training_benign_losses,
                                 training_attack_losses, training_contrastive_losses),(validation_losses,
                                                           validation_benign_losses,
                                                          validation_attack_losses, validation_contrastive_losses),\
                                                                  (test_loss_n,
                                                                  test_benign_loss_n,
                                                                  test_attack_loss_n,
                                                                  test_contrastive_loss_n)
