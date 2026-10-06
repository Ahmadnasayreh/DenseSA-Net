"""Two-stage training, inference and evaluation metrics."""
import copy
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast(device, enabled):
    return torch.autocast(device_type=device.type,
                          enabled=enabled and device.type == "cuda")


class EarlyStopping:
    """Keeps the weights with the lowest validation loss."""

    def __init__(self, patience=5):
        self.patience = patience
        self.best = None
        self.bad_epochs = 0
        self.best_state = None
        self.stop = False

    def step(self, val_loss, model):
        if self.best is None or val_loss < self.best:
            self.best = val_loss
            self.bad_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.bad_epochs += 1
            self.stop = self.bad_epochs >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, cfg.amp):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, cfg):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with _autocast(device, cfg.amp):
            out = model(x)
            loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def fit(model, train_loader, val_loader, cfg, device, verbose=True):
    """Stage 1: backbone frozen, new layers trained at lr_stage1.
    Stage 2: everything unfrozen and fine-tuned at lr_stage2.

    Models without a freeze_backbone method (VGG16, CustomCNN) are trained
    end to end in both stages. Within each stage the learning rate is halved
    every cfg.lr_step epochs, and the weights with the lowest validation
    loss are restored at the end of the stage.
    """
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    history = []
    t0 = time.time()
    stages = ((1, cfg.lr_stage1, True, cfg.epochs_stage1),
              (2, cfg.lr_stage2, False, cfg.epochs_stage2))
    for stage, lr, frozen, epochs in stages:
        if hasattr(model, "freeze_backbone"):
            model.freeze_backbone(frozen)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step,
                                                    gamma=cfg.lr_gamma)
        stopper = EarlyStopping(cfg.patience)
        for epoch in range(1, epochs + 1):
            tl, ta = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scaler, device, cfg)
            vl, va = evaluate_loss(model, val_loader, criterion, device, cfg)
            scheduler.step()
            history.append({"stage": stage, "epoch": epoch, "train_loss": tl,
                            "train_acc": ta, "val_loss": vl, "val_acc": va})
            if verbose:
                print(f"  stage {stage} epoch {epoch:2d}  loss {tl:.4f}/{vl:.4f}"
                      f"  acc {ta:.4f}/{va:.4f}")
            stopper.step(vl, model)
            if stopper.stop:
                break
        stopper.restore(model)
    return pd.DataFrame(history), time.time() - t0


@torch.no_grad()
def predict(model, loader, device, cfg, warmup_batches=0):
    """Returns labels, predictions, probabilities and ms per image.

    The first CUDA forward passes include cuDNN autotuning, so a few warm-up
    batches are run and discarded before timing.
    """
    model.eval()
    if warmup_batches and device.type == "cuda":
        for i, (x, _) in enumerate(loader):
            with _autocast(device, cfg.amp):
                model(x.to(device))
            if i + 1 >= warmup_batches:
                break
        torch.cuda.synchronize()
    labels, preds, probs = [], [], []
    n = 0
    t0 = time.time()
    for x, y in loader:
        with _autocast(device, cfg.amp):
            out = model(x.to(device))
        p = torch.softmax(out.float(), 1)
        labels.append(y.numpy())
        preds.append(p.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
        n += x.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / max(n, 1) * 1000
    return np.concatenate(labels), np.concatenate(preds), np.concatenate(probs), ms


def compute_metrics(labels, preds, probs, n_classes):
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    spec = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        spec.append(tn / (tn + fp) if tn + fp else 0.0)
    try:
        y = label_binarize(labels, classes=list(range(n_classes)))
        auc = roc_auc_score(y, probs, average="macro", multi_class="ovr")
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds, average="macro", zero_division=0),
        "Sensitivity": recall_score(labels, preds, average="macro", zero_division=0),
        "Specificity": float(np.mean(spec)),
        "F1-Score": f1_score(labels, preds, average="macro", zero_division=0),
        "AUC": auc,
        "cm": cm,
    }


def per_class_table(labels, preds, classes):
    """Per-class precision, sensitivity, specificity, F1 and main confusion."""
    n = len(classes)
    prec, rec, f1, sup = precision_recall_fscore_support(
        labels, preds, labels=list(range(n)), zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(n)))
    rows = []
    for i, name in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        off = cm[i].copy()
        off[i] = 0
        j = int(off.argmax())
        rows.append({"Class": name, "Support": int(sup[i]),
                     "Precision": prec[i], "Sensitivity": rec[i],
                     "Specificity": tn / (tn + fp) if tn + fp else 0.0,
                     "F1-Score": f1[i], "Misclassified": int(fn),
                     "Main_confusion": f"{classes[j]} ({off[j]})" if off.sum() else "-"})
    return pd.DataFrame(rows)
