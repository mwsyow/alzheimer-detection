import math

import pytest
import torch
from torch import nn

from conftest import FakeRun
from train import (
    build_lr_scheduler,
    restore_lr_scheduler,
    train,
)


def scheduler_config(warmup_fraction=0.2, min_lr_ratio=0.1):
    return {
        "lr_scheduler": {
            "enabled": True,
            "name": "LinearWarmupCosineAnnealingLR",
            "params": {
                "warmup_fraction": warmup_fraction,
                "min_lr_ratio": min_lr_ratio,
            },
        }
    }


def optimizer(lr=1.0):
    return torch.optim.AdamW(nn.Linear(1, 1).parameters(), lr=lr, weight_decay=0.0)


def advance(optim, scheduler, steps):
    rates = []
    for _ in range(steps):
        rates.append(optim.param_groups[0]["lr"])
        optim.step()
        scheduler.step()
    return rates


def test_scheduler_is_opt_in():
    assert build_lr_scheduler({}, optimizer(), epochs=2, steps_per_epoch=5) is None
    assert (
        build_lr_scheduler(
            {"lr_scheduler": {"enabled": False}},
            optimizer(),
            epochs=2,
            steps_per_epoch=5,
        )
        is None
    )


def test_linear_warmup_and_cosine_decay_boundaries():
    optim = optimizer()
    scheduler = build_lr_scheduler(
        scheduler_config(), optim, epochs=2, steps_per_epoch=5
    )

    rates = advance(optim, scheduler, 10)

    assert rates[:3] == pytest.approx([0.5, 1.0, 1.0])
    assert rates[-1] == pytest.approx(0.1)
    assert all(left >= right for left, right in zip(rates[2:], rates[3:]))


def test_single_update_schedule_keeps_the_base_lr():
    optim = optimizer(lr=0.25)
    scheduler = build_lr_scheduler(
        scheduler_config(warmup_fraction=0.9), optim, epochs=1, steps_per_epoch=1
    )

    assert advance(optim, scheduler, 1) == pytest.approx([0.25])


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"lr_scheduler": {"enabled": True, "name": "StepLR"}}, "Unsupported"),
        (scheduler_config(warmup_fraction=-0.1), "warmup_fraction"),
        (scheduler_config(warmup_fraction=1.0), "warmup_fraction"),
        (scheduler_config(min_lr_ratio=1.1), "min_lr_ratio"),
        (scheduler_config(min_lr_ratio=math.nan), "finite"),
    ],
)
def test_invalid_scheduler_config_is_rejected(config, match):
    with pytest.raises(ValueError, match=match):
        build_lr_scheduler(config, optimizer(), epochs=2, steps_per_epoch=5)


def test_training_steps_scheduler_per_batch_logs_lr_and_checkpoints_state(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(8, 2))
    images = torch.randn(6, 1, 2, 2, 2)
    labels = torch.arange(6) % 2
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, labels), batch_size=2
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_lr_scheduler(
        scheduler_config(warmup_fraction=0.5, min_lr_ratio=0.1),
        optim,
        epochs=2,
        steps_per_epoch=len(loader),
    )
    logger = FakeRun()

    train(
        epochs=2,
        model=model,
        optim=optim,
        lr_scheduler=scheduler,
        loss=nn.CrossEntropyLoss(),
        train_loader=loader,
        val_loader=None,
        logger=logger,
        checkpoint_dir=tmp_path,
        checkpoint_config={"save_best": False, "save_last": True},
    )

    assert scheduler.last_epoch == 6
    assert [log["Learning Rate"] for log in logger.log_calls] == pytest.approx(
        [1e-3, 1e-4]
    )
    checkpoint = torch.load(tmp_path / "last.pth", weights_only=False)
    assert checkpoint["lr_scheduler_state_dict"]["last_epoch"] == 6


def test_scheduler_state_restores_the_exact_next_lr():
    config = scheduler_config(warmup_fraction=0.3, min_lr_ratio=0.05)
    first_optim = optimizer()
    first_scheduler = build_lr_scheduler(
        config, first_optim, epochs=2, steps_per_epoch=5
    )
    advance(first_optim, first_scheduler, 4)
    checkpoint = {
        "optimizer_state_dict": first_optim.state_dict(),
        "lr_scheduler_state_dict": first_scheduler.state_dict(),
    }

    resumed_optim = optimizer()
    resumed_scheduler = build_lr_scheduler(
        config, resumed_optim, epochs=2, steps_per_epoch=5
    )
    resumed_optim.load_state_dict(checkpoint["optimizer_state_dict"])
    restore_lr_scheduler(resumed_scheduler, checkpoint)

    assert resumed_optim.param_groups[0]["lr"] == pytest.approx(
        first_optim.param_groups[0]["lr"]
    )
    assert advance(resumed_optim, resumed_scheduler, 6) == pytest.approx(
        advance(first_optim, first_scheduler, 6)
    )


def test_enabled_scheduler_requires_state_when_resuming():
    scheduler = build_lr_scheduler(
        scheduler_config(), optimizer(), epochs=2, steps_per_epoch=5
    )
    with pytest.raises(ValueError, match="lr_scheduler_state_dict"):
        restore_lr_scheduler(scheduler, {})
