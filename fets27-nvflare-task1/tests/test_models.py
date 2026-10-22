from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from monai.losses import DiceLoss  # noqa: E402

from fets27_challenge.models import (  # noqa: E402
    FedUNet,
    GliomaSegResNet,
    create_model_for_cohort,
)


def test_fedunet_path_e_configuration():
    model = FedUNet()

    assert model.in_channels == 4
    assert model.out_channels == 3
    assert model.norm_name == "batch"
    assert model.channels == (32, 64, 128, 256, 512)
    assert model.strides == (2, 2, 2, 2)
    assert model.dropout == 0.3
    assert model.num_res_units == 3
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_fedunet_small_batch_forward_backward():
    torch.manual_seed(17)
    model = FedUNet().train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=3e-4, weight_decay=1e-5
    )
    dice = DiceLoss(
        smooth_nr=0.0,
        smooth_dr=1e-5,
        squared_pred=True,
        to_onehot_y=False,
        sigmoid=True,
    )
    pos_weight = torch.tensor([10.0, 10.0, 10.0]).view(1, 3, 1, 1, 1)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    inputs = torch.randn(1, 4, 96, 96, 96)
    targets = (torch.rand(1, 3, 96, 96, 96) > 0.7).float()

    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    assert tuple(outputs.shape) == (1, 3, 96, 96, 96)

    loss = dice(outputs, targets) + bce(outputs, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None for parameter in model.parameters()
    )
    optimizer.step()


def test_create_model_for_cohort_selects_fedunet():
    model = create_model_for_cohort("glioma", model_name="fedunet")

    assert isinstance(model, FedUNet)
    assert isinstance(create_model_for_cohort("glioma"), GliomaSegResNet)


def test_unknown_model_name_is_rejected():
    with pytest.raises(KeyError, match="Unknown model"):
        create_model_for_cohort("glioma", model_name="unetr")
