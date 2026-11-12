import pytest
import torch


def test_weighted_mse_cpu_normalization_and_gradient(stage_import):
    module = stage_import("IIC-Net", "loss.loss")
    prediction = (
        torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 2, 1, 1, 2).requires_grad_()
    )
    loss = module.WeightedMSELoss()(
        prediction, torch.zeros_like(prediction), torch.ones_like(prediction)
    )
    assert (
        loss.item() == 15
    )  # Sum / (batch * spatial), intentionally excludes channels.
    loss.backward()
    torch.testing.assert_close(prediction.grad, prediction.detach())


@pytest.mark.parametrize("name", ["MSELoss_unlabel", "BCELoss_unlabel"])
def test_consistency_losses_and_empty_masks(stage_import, name):
    module = stage_import("IIC-Net", "loss.loss_unlabel")
    loss_fn = getattr(module, name)()
    prediction = torch.tensor([0.25, 0.75], requires_grad=True)
    target, weight = torch.tensor([0.0, 1.0]), torch.tensor([1.0, 3.0])
    expected = (
        0.0625 if name.startswith("MSE") else -torch.log(torch.tensor(0.75)).item()
    )
    assert loss_fn(prediction, target, weight).item() == pytest.approx(expected)
    empty = loss_fn(prediction, target, torch.zeros_like(weight))
    assert empty.item() == 0
    empty.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_cbst_empty_mask(stage_import):
    module = stage_import("IIC-Net", "loss.loss")
    prediction = torch.tensor([0.2, 0.8], requires_grad=True)
    loss = module.WeightedBCELoss_cbst()(prediction, torch.zeros(2), torch.zeros(2))
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros(2))
