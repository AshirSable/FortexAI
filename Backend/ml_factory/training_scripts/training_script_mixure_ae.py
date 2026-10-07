import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Literal
from collections import defaultdict
from sklearn.metrics import average_precision_score

from ml_factory.models.autoencoder import NormalityAE

def training(training_loader: DataLoader,
             val_loader: DataLoader,
             test_loader: Optional[DataLoader],
             model: nn.Module,
             optimizer: torch.optim.Optimizer,
             criterion: nn.Module, epoch: int = 100, device: Literal['cuda', 'cpu'] = 'cpu'):

    training_losses = []
    validation_losses = []
    best_model = float('inf')

    best_model_settings = {
        'model': {k: v.clone() for k, v in model.state_dict().items()},
        'optimizer': optimizer.state_dict(),
        'epoch': 0
    }
    for _e in range(epoch):
        model.train()
        train_loss = 0
        validation_loss = 0
        for X, _ in training_loader:
            optimizer.zero_grad()
            X = X.to(device)
            result = model(X)
            loss = criterion(result.recon, X)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach()

        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                result = model(X)
                loss = criterion(result.recon, X)

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
                result = model(X)
                loss = criterion(result.recon, X)

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

    best_model_settings = {
        'model': {k: v.clone() for k, v in model.state_dict().items()},
        'optimizer': optimizer.state_dict(),
        'epoch': 0
    }
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
            result = model(X)
            loss, benign_loss, attack_loss = compute_oe_loss(result.recon, X, y, criterion)
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
                result = model(X)
                loss, benign_loss, attack_loss = compute_oe_loss(result.recon, X, y, criterion)

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
                result = model(X)
                loss, benign_loss, attack_loss = compute_oe_loss(result.recon, X, y, criterion)
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

def diversity_loss(z: torch.Tensor):
    n_experts = z.size(1)

    normed = F.normalize(z, dim=-1)
    sim_matrix = torch.einsum('bnd,bmd->bnm', normed, normed)

    mask = ~torch.eye(n_experts, dtype=torch.bool, device=z.device)

    off_diag_sim = sim_matrix[:, mask].view(z.size(0), -1)

    return off_diag_sim.mean()


def training_attack_diversity(training_loader: DataLoader,
             val_loader: DataLoader,
             test_loader: Optional[DataLoader],
             model: NormalityAE,
             optimizer: torch.optim.Optimizer,
             epoch: int = 100, device: Literal['cuda', 'cpu'] = 'cpu'):

    training_losses = {"loss": [], "benign_loss": [], "attack_loss": [], 'diversity_loss': []}
    validation_losses = {"loss": [], "benign_loss": [], "attack_loss": [], 'diversity_loss': [], 'auprc': []}
    best_model = -float('inf')
    criterion = nn.MSELoss(reduction="none")

    best_model_settings = {
        'model': {k: v.clone() for k, v in model.state_dict().items()},
        'optimizer': optimizer.state_dict(),
        'epoch': 0
    }
    val_n = len(val_loader)
    train_n = len(training_loader)
    for _e in range(epoch):
        if hasattr(training_loader.sampler, 'set_epoch'):
            training_loader.sampler.set_epoch(_e)
        model.train()
        curr_train_loss = defaultdict(lambda: torch.tensor(0.0,device=device))
        curr_val_loss = defaultdict(lambda: torch.tensor(0.0,device=device))

        c_train_loss_n = {}
        c_val_loss_n = {}
        for X, y in training_loader:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            result = model(X)
            loss, benign_loss, attack_loss = compute_oe_loss(result.recon, X, y, criterion)
            c_loss = diversity_loss(result.experts)
            loss = loss + c_loss
            loss.backward()
            optimizer.step()
            curr_train_loss['loss'] += loss.detach()
            curr_train_loss['diversity_loss'] += c_loss.detach()
            curr_train_loss['attack_loss'] += attack_loss.detach()
            curr_train_loss['benign_loss'] += benign_loss.detach()

        model.eval()
        val_scores = []
        val_label = []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                result = model(X)
                loss, benign_loss, attack_loss = compute_oe_loss(result.recon, X, y, criterion)
                c_loss = diversity_loss(result.experts)
                loss = loss + c_loss
                scores = ((result.recon - X) ** 2).mean(dim=1)
                val_scores.append(scores.cpu())
                val_label.append(y.cpu())
                curr_val_loss['loss'] += loss.detach()
                curr_val_loss['diversity_loss'] += c_loss.detach()
                curr_val_loss['attack_loss'] += attack_loss.detach()
                curr_val_loss['benign_loss'] += benign_loss.detach()
        val_scores = torch.cat(val_scores).numpy()
        val_label = torch.cat(val_label).numpy()

        val_auprc = average_precision_score(val_label, val_scores)
        for train_key in curr_train_loss:
            c_train_loss_n[train_key] = curr_train_loss[train_key].cpu().item() / train_n

        for val_key in curr_val_loss:
            c_val_loss_n[val_key] = curr_val_loss[val_key].cpu().item() / val_n

        c_val_loss_n['auprc'] = val_auprc

        for keys in c_train_loss_n:
            training_losses[keys].append(c_train_loss_n[keys])

        for keys in c_val_loss_n:
            validation_losses[keys].append(c_val_loss_n[keys])


        if best_model < c_val_loss_n['auprc']:
            best_model = c_val_loss_n['auprc']
            best_model_settings = {
                            'model': {k: v.clone() for k, v in model.state_dict().items()},
                            'optimizer': optimizer.state_dict(),
                            'epoch': _e,
                        }

        print(f"[{_e+1:<4}/{epoch}] | train loss: {c_train_loss_n['loss']:<5.5f} \
- a {c_train_loss_n['attack_loss']:<5.5f} \
- b {c_train_loss_n['benign_loss']:<5.5f} \
- d {c_train_loss_n['diversity_loss']:<5.5f} \
validation loss: {c_val_loss_n['loss']:<5.5f} \
- a {c_val_loss_n['attack_loss']:<5.5f} \
- b {c_val_loss_n['benign_loss']:<5.5f} \
- d {c_val_loss_n['diversity_loss']:<5.5f}\
- AUPRC {c_val_loss_n['auprc']:<5.5f}|")

    model.load_state_dict(best_model_settings['model'])

    test_loss_n = None
    if test_loader is not None:

        test_loss = defaultdict(lambda: torch.tensor(0.0, device=device))
        test_loss_n = {}
        model.eval()
        val_scores = []
        val_label = []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                result = model(X)
                loss, benign_loss, attack_loss = compute_oe_loss(result.recon, X, y, criterion)
                c_loss =diversity_loss(result.experts)
                loss = loss + c_loss
                scores = ((result.recon - X) ** 2).mean(dim=1)
                val_scores.append(scores.cpu())
                val_label.append(y.cpu())
                test_loss['loss'] += loss.detach()
                test_loss['diversity_loss'] += c_loss.detach()
                test_loss['attack_loss'] += attack_loss.detach()
                test_loss['benign_loss'] += benign_loss.detach()

            val_scores = torch.cat(val_scores).numpy()
            val_label = torch.cat(val_label).numpy()

            val_auprc = average_precision_score(val_label, val_scores)

            for keys in test_loss:
                test_loss_n[keys] = test_loss[keys].cpu().item() / len(test_loader)

            test_loss_n['auprc'] = val_auprc

    torch.cuda.empty_cache()
    return best_model_settings, training_losses,validation_losses, test_loss_n


def router_load_balance_loss(router: torch.Tensor, k: int) -> torch.Tensor:
    n_experts = router.size(-1)

    router_probs = F.softmax(router, dim=-1)

    mean_prob_per_expert = router_probs.mean(dim=0)

    _, topk_idx = router.topk(k, dim=-1)

    one_hot = torch.zeros_like(router_probs)

    one_hot.scatter_(1, topk_idx, 1.0)

    fraction_selected = one_hot.mean(dim=0)

    loss = n_experts * (fraction_selected * mean_prob_per_expert).sum()

    return loss

def training_attack_diversity_router(training_loader: DataLoader,
             val_loader: DataLoader,
             test_loader: Optional[DataLoader],
             model: NormalityAE,
             optimizer: torch.optim.Optimizer,
             epoch: int = 100, device: Literal['cuda', 'cpu'] = 'cpu'):

    training_losses = {"loss": [], "benign_loss": [], "attack_loss": [], 'diversity_loss': [], 'router_loss': []}
    validation_losses = {"loss": [], "benign_loss": [], "attack_loss": [], 'diversity_loss': [], 'router_loss': [], 'auprc': []}
    best_model = -float('inf')
    criterion = nn.MSELoss(reduction="none")
    model = model.to(device)

    best_model_settings = {
        'model': {k: v.clone() for k, v in model.state_dict().items()},
        'optimizer': optimizer.state_dict(),
        'epoch': 0
    }
    val_n = len(val_loader)
    train_n = len(training_loader)
    for _e in range(epoch):
        if hasattr(training_loader.sampler, 'set_epoch'):
            training_loader.sampler.set_epoch(_e)
        model.train()

        curr_train_loss = defaultdict(lambda: torch.tensor(0.0, device=device))
        curr_val_loss = defaultdict(lambda: torch.tensor(0.0, device=device))

        c_train_loss_n = {}
        c_val_loss_n = {}
        for X, y in training_loader:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            result = model(X)
            loss, benign_loss, attack_loss = compute_oe_loss(result.recon, X, y, criterion)
            c_loss = diversity_loss(result.experts)
            r_loss = router_load_balance_loss(result.router, k= model.k_expert)
            loss = loss + c_loss + r_loss
            loss.backward()
            optimizer.step()
            curr_train_loss['loss'] += loss.detach()
            curr_train_loss['diversity_loss'] += c_loss.detach()
            curr_train_loss['attack_loss'] += attack_loss.detach()
            curr_train_loss['benign_loss'] += benign_loss.detach()
            curr_train_loss['router_loss'] += r_loss.detach()

        model.eval()
        val_scores = []
        val_label = []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                result = model(X)
                loss, benign_loss, attack_loss = compute_oe_loss(result.recon, X, y, criterion)
                c_loss = diversity_loss(result.experts)
                r_loss = router_load_balance_loss(result.router, k= model.k_expert)
                loss = loss + c_loss + r_loss

                scores = ((result.recon - X) ** 2).mean(dim=1)
                val_scores.append(scores.cpu())
                val_label.append(y.cpu())
                curr_val_loss['loss'] += loss.detach()
                curr_val_loss['diversity_loss'] += c_loss.detach()
                curr_val_loss['attack_loss'] += attack_loss.detach()
                curr_val_loss['benign_loss'] += benign_loss.detach()
                curr_val_loss['router_loss'] += r_loss.detach()

        val_scores = torch.cat(val_scores).numpy()
        val_label = torch.cat(val_label).numpy()

        val_auprc = average_precision_score(val_label, val_scores)

        for train_key in curr_train_loss:
            c_train_loss_n[train_key] = curr_train_loss[train_key].cpu().item() / train_n

        for val_key in curr_val_loss:
            c_val_loss_n[val_key] = curr_val_loss[val_key].cpu().item() / val_n

        c_val_loss_n['auprc'] = val_auprc

        if best_model < c_val_loss_n['auprc']:
            best_model = c_val_loss_n['auprc']
            best_model_settings = {
                            'model': {k: v.clone() for k, v in model.state_dict().items()},
                            'optimizer': optimizer.state_dict(),
                            'epoch': _e,
                        }


        for keys in c_train_loss_n:
            training_losses[keys].append(c_train_loss_n[keys])

        for keys in c_val_loss_n:
            validation_losses[keys].append(c_val_loss_n[keys])

        print(f"[{_e+1:<4}/{epoch}] | train loss: {c_train_loss_n['loss']:<5.5f} \
- a {c_train_loss_n['attack_loss']:<5.5f} \
- b {c_train_loss_n['benign_loss']:<5.5f} \
- d {c_train_loss_n['diversity_loss']:<5.5f} \
- r {c_train_loss_n['router_loss']:<5.5f}| \
validation loss: {c_val_loss_n['loss']:<5.5f} \
- a {c_val_loss_n['attack_loss']:<5.5f} \
- b {c_val_loss_n['benign_loss']:<5.5f} \
- d {c_val_loss_n['diversity_loss']:<5.5f} \
- r {c_val_loss_n['router_loss']:<5.5f}\
- AUPRC {c_val_loss_n['auprc']:<5.5f}|")



    model.load_state_dict(best_model_settings['model'])
    test_loss_n = None
    if test_loader is not None:

        test_loss = defaultdict(lambda: torch.tensor(0.0,device=device))
        test_loss_n = {}
        model.eval()
        val_scores = []
        val_label = []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                result = model(X)
                loss, benign_loss, attack_loss = compute_oe_loss(result.recon, X, y, criterion)
                c_loss =diversity_loss(result.experts)
                r_loss = router_load_balance_loss(result.router, model.k_expert)
                loss = loss + c_loss + r_loss

                scores = ((result.recon - X) ** 2).mean(dim=1)
                val_scores.append(scores.cpu())
                val_label.append(y.cpu())
                test_loss['loss'] += loss.detach()
                test_loss['diversity_loss'] += c_loss.detach()
                test_loss['attack_loss'] += attack_loss.detach()
                test_loss['benign_loss'] += benign_loss.detach()
                test_loss['router_loss'] += r_loss.detach()


            val_scores = torch.cat(val_scores).numpy()
            val_label = torch.cat(val_label).numpy()

            val_auprc = average_precision_score(val_label, val_scores)

            for keys in test_loss:
                test_loss_n[keys] = test_loss[keys].cpu().item() / len(test_loader)

            test_loss_n['auprc'] = val_auprc

    torch.cuda.empty_cache()

    return best_model_settings, training_losses,validation_losses, test_loss_n
