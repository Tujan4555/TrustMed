
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        n = batch_y.size(0)
        total_loss += loss.item() * n
        total_examples += n

    return total_loss / total_examples


@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_probs = []
    all_labels = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True).float()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        probs = torch.sigmoid(logits)

        n = batch_y.size(0)
        total_loss += loss.item() * n
        total_examples += n
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch_y.cpu().numpy())

    y_true = np.concatenate(all_labels).astype(np.int8)
    y_prob = np.concatenate(all_probs).astype(np.float64)

    return {
        "val_loss": total_loss / total_examples,
        "val_auprc": float(average_precision_score(y_true, y_prob)),
        "val_auroc": float(roc_auc_score(y_true, y_prob)),
        "val_positive_count": int(y_true.sum()),
        "val_negative_count": int((y_true == 0).sum()),
        "val_examples": int(len(y_true)),
    }


def save_checkpoint(path, model, optimizer, epoch, best_auprc,
                    metrics, config):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    payload = {
        "experiment": config["experiment"],
        "epoch": int(epoch),
        "best_val_auprc": float(best_auprc),
        "metrics": metrics,
        "config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available()
            else None
        ),
    }

    torch.save(payload, temporary_path)
    temporary_path.replace(path)
