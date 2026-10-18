"""Train once per (dataset, regime, seed); cache per-epoch validation
predictions and a checkpoint. Every lambda is scored offline against this
one cached stream -- see config.yaml's `compute_plan`:

    "lambda is a MODEL SELECTION parameter, not a training parameter."

Determinism per config.yaml's `determinism` block: seeded init, seeded
batch order, torch deterministic algorithms on, cudnn benchmark off.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import build_backbone


def set_determinism(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def train_one_run(dataset_cfg, regime_cfg, backbone_cfg, model_cfg, seed,
                   train_ds, val_ds, test_ds, run_dir, device="cpu",
                   epochs_override=None, batch_size_override=None,
                   log_every=1):
    """Trains one (dataset, regime, seed) cell to the full epoch budget,
    caching per-epoch validation probabilities/labels and a checkpoint.

    Returns a dict: {epoch: {"val_probs":..., "val_labels":..., "ckpt": path}}
    plus the final test set (probs are NOT cached per epoch for test --
    only loaded lazily from a checkpoint when a selection rule asks for it;
    see score_selected_epoch below).
    """
    set_determinism(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    classes = dataset_cfg["classes"]
    in_channels = dataset_cfg["input_shape"][0]
    input_hw = dataset_cfg["input_shape"][1]
    backbone_name = regime_cfg["backbone"]

    model = build_backbone(backbone_name, in_channels, classes, input_hw,
                            backbone_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=model_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    batch_size = batch_size_override or model_cfg["batch_size"]
    epochs = epochs_override or regime_cfg["epochs"]

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               generator=g, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    epoch_cache = {}
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()

        val_probs, val_labels = predict_probs(model, val_loader, device)
        ckpt_path = run_dir / f"epoch_{epoch:03d}.pt"
        torch.save(model.state_dict(), ckpt_path)
        epoch_cache[epoch] = {
            "val_probs": val_probs, "val_labels": val_labels,
            "ckpt": str(ckpt_path),
        }
        if log_every and epoch % log_every == 0:
            acc = float((val_probs.argmax(1) == val_labels).mean())
            print(f"    epoch {epoch+1}/{epochs}  val_acc={acc:.4f}")

    return epoch_cache, model, (in_channels, classes, input_hw, backbone_name)


@torch.no_grad()
def score_selected_epoch(epoch, backbone_name, dataset_cfg, backbone_cfg,
                          test_ds, run_dir, device="cpu"):
    """Load the checkpoint for `epoch` and evaluate it on the test set --
    the one place test-set predictions are computed, only for epochs a
    selection rule actually picked.
    """
    classes = dataset_cfg["classes"]
    in_channels = dataset_cfg["input_shape"][0]
    input_hw = dataset_cfg["input_shape"][1]
    model = build_backbone(backbone_name, in_channels, classes, input_hw,
                            backbone_cfg).to(device)
    ckpt = Path(run_dir) / f"epoch_{epoch:03d}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    return predict_probs(model, test_loader, device)
