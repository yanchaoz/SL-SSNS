"""Configuration, learning-rate scheduling and checkpoint I/O for pretraining."""

from types import SimpleNamespace

import torch
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return SimpleNamespace(
        **{
            key: SimpleNamespace(**value) if isinstance(value, dict) else value
            for key, value in raw.items()
        }
    )


def calculate_lr(cfg, iteration):
    train = cfg.TRAIN
    if iteration < train.warmup_iters:
        factor = (float(iteration) / train.warmup_iters) ** train.power
    elif iteration < train.decay_iters:
        factor = (
            1 - float(iteration - train.warmup_iters) / train.decay_iters
        ) ** train.power
    else:
        return train.end_lr
    return (train.base_lr - train.end_lr) * factor + train.end_lr


def save_checkpoint(path, learner, optimizer, iteration):
    """Save CGS-compatible encoder weights plus the complete training state."""
    encoder = {
        key.removeprefix("module."): value
        for key, value in learner.net.state_dict().items()
    }
    torch.save(
        {
            "format_version": 1,
            "current_iter": iteration,
            "valid_result": None,
            "model_weights": encoder,
            "learner_weights": learner.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def restore_checkpoint(path, learner, optimizer, device):
    """Resume a full checkpoint; encoder-only releases are initialization weights."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not {"learner_weights", "optimizer", "current_iter"}.issubset(checkpoint):
        raise ValueError(
            "Encoder-only checkpoint cannot resume training; use MODEL.pre_train instead"
        )
    learner.load_state_dict(checkpoint["learner_weights"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["current_iter"])
