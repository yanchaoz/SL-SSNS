import io
from pathlib import Path
import random

import numpy as np
import pytest
import torch


def make_provider(stage_import, monkeypatch):
    provider_module = stage_import("Pretrain", "pretrain_provider")
    utility = stage_import("Pretrain", "training_utils")
    path = (
        Path(__file__).resolve().parents[1] / "Pretrain/config/pretraining_snemi3d.yaml"
    )
    cfg = utility.load_config(path)
    cfg.TRAIN.if_cuda, cfg.TRAIN.num_workers, cfg.TRAIN.batch_size = False, 0, 2
    raw = np.random.RandomState(7).randint(0, 256, (18, 160, 160), dtype=np.uint8)
    monkeypatch.setattr(provider_module.os.path, "isfile", lambda _: True)
    monkeypatch.setattr(provider_module.h5py, "is_hdf5", lambda _: False)
    monkeypatch.setattr(provider_module.imageio, "volread", lambda _: raw)
    return provider_module, cfg


def test_fixed_size_pairs_and_modern_iterator(stage_import, monkeypatch):
    module, cfg = make_provider(stage_import, monkeypatch)
    monkeypatch.setattr(module.Train, "apply_perturbations", lambda self, data: data)
    provider = module.Provider("train", cfg)
    for seed in range(10):
        random.seed(seed)
        first, second = provider.next()
        assert first.shape == second.shape == (2, 1, 18, 160, 160)
        assert first.device.type == "cpu"
        torch.testing.assert_close(first, second)
    assert provider.iteration == 10
    assert len(provider) > 0


@pytest.mark.parametrize("kind", range(7))
def test_augmentation_shapes(stage_import, monkeypatch, kind):
    module, cfg = make_provider(stage_import, monkeypatch)
    dataset = module.Train(cfg)
    # Only select the augmentation branch; retain randomness inside augmentations.
    original = module.random.randint
    monkeypatch.setattr(
        module.random,
        "randint",
        lambda a, b: kind if (a, b) == (0, 6) else original(a, b),
    )
    result = dataset.apply_perturbations(dataset.dataset[0].astype(np.float32) / 255)
    assert result.shape == (18, 160, 160)
    assert np.isfinite(result).all()


def test_full_checkpoint_and_encoder_export(stage_import):
    utils = stage_import("Pretrain", "training_utils")
    learner_module = stage_import("Pretrain", "learner")
    encoder_module = stage_import("Pretrain", "model.encoder")
    torch.manual_seed(3)
    encoder = encoder_module.UNet_PNI_encoder(filters=[2, 2, 2, 2, 2])
    learner = learner_module.BSL(
        encoder,
        hidden_layer=-1,
        projection_size=4,
        projection_hidden_size=8,
        use_momentum=False,
        sample_shape=(1, 2, 16, 16),
    )
    optimizer = torch.optim.Adam(learner.parameters(), lr=0.001)
    first, second = torch.rand(2, 1, 2, 16, 16), torch.rand(2, 1, 2, 16, 16)
    loss = learner(first, second)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    buffer = io.BytesIO()
    utils.save_checkpoint(buffer, learner, optimizer, 1)
    expected = {key: value.clone() for key, value in learner.state_dict().items()}
    with torch.no_grad():
        for parameter in learner.parameters():
            parameter.add_(10)
    optimizer.param_groups[0]["lr"] = 5
    buffer.seek(0)
    assert utils.restore_checkpoint(buffer, learner, optimizer, "cpu") == 1
    assert optimizer.param_groups[0]["lr"] == 0.001
    for key, value in learner.state_dict().items():
        torch.testing.assert_close(value, expected[key])
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    exported = encoder_module.UNet_PNI_encoder(filters=[2, 2, 2, 2, 2])
    exported.load_state_dict(checkpoint["model_weights"], strict=True)
    encoder.eval()
    exported.eval()
    with torch.no_grad():
        torch.testing.assert_close(encoder(first), exported(first))
    legacy = io.BytesIO()
    torch.save({"model_weights": checkpoint["model_weights"]}, legacy)
    legacy.seek(0)
    with pytest.raises(ValueError, match="Encoder-only"):
        utils.restore_checkpoint(legacy, learner, optimizer, "cpu")


def test_loop_stops_at_budget_and_resumes(stage_import, tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = stage_import("Pretrain", "pretrain")
    utils = stage_import("Pretrain", "training_utils")
    calls = []
    monkeypatch.setattr(
        utils,
        "save_checkpoint",
        lambda path, learner, optimizer, iteration: calls.append(iteration),
    )
    learner = torch.nn.Linear(1, 1)

    class PairLearner(torch.nn.Module):
        def forward(self, first, second):
            return (learner(first) - second).square().mean()

    pair = PairLearner()

    class Provider:
        def next(self):
            return torch.ones(2, 1), torch.zeros(2, 1)

    writer = SimpleNamespace(add_scalar=lambda *args: None)
    cfg = SimpleNamespace(
        record_path=str(tmp_path),
        save_path=str(tmp_path),
        TRAIN=SimpleNamespace(
            total_iters=3,
            display_freq=1,
            save_freq=2,
            base_lr=0.01,
            end_lr=0.01,
            decay_iters=10,
            warmup_iters=0,
            power=1.5,
            weight_decay=None,
        ),
    )
    module.loop(
        cfg, Provider(), pair, torch.optim.Adam(learner.parameters()), 1, writer
    )
    assert calls == [2, 3]
    assert len((tmp_path / "loss.txt").read_text().splitlines()) == 2


def test_learning_rate_endpoints(stage_import):
    from types import SimpleNamespace

    utils = stage_import("Pretrain", "training_utils")
    cfg = SimpleNamespace(
        TRAIN=SimpleNamespace(
            base_lr=0.01, end_lr=0.001, warmup_iters=10, decay_iters=100, power=1.5
        )
    )
    assert utils.calculate_lr(cfg, 0) == 0.001
    assert utils.calculate_lr(cfg, 10) == pytest.approx(0.01)
    assert utils.calculate_lr(cfg, 100) == 0.001
