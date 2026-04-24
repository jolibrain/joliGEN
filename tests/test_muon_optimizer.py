from types import SimpleNamespace

import torch

from util.optimizer_factory import build_named_parameters, build_optimizer_bundle


class ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
        self.linear_2 = torch.nn.Linear(4, 2, bias=False)
        self.embedding = torch.nn.Embedding(16, 8)
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.norm = torch.nn.LayerNorm(4)


def test_muon_optimizer_bundle_splits_parameters():
    net = ToyNet()
    opt = SimpleNamespace(
        train_optim="muon",
        train_muon_nesterov=True,
        train_muon_ns_steps=5,
        train_muon_adjust_lr_fn="original",
    )

    named_params = build_named_parameters(("toy", net))
    optimizer_bundle = build_optimizer_bundle(
        opt,
        named_params,
        lr=1e-3,
        betas=(0.95, 0.99),
        weight_decay=0.01,
        eps=1e-8,
        optimizer_name="optimizer_G",
    )

    assert [name for name, _ in optimizer_bundle] == [
        "optimizer_G",
        "optimizer_G_aux",
    ]

    muon_optimizer = optimizer_bundle[0][1]
    aux_optimizer = optimizer_bundle[1][1]

    muon_param_ids = {id(param) for param in muon_optimizer.param_groups[0]["params"]}
    aux_param_ids = {id(param) for param in aux_optimizer.param_groups[0]["params"]}

    assert id(net.linear.weight) in muon_param_ids
    assert id(net.linear_2.weight) in muon_param_ids

    assert id(net.linear.bias) in aux_param_ids
    assert id(net.embedding.weight) in aux_param_ids
    assert id(net.conv.weight) in aux_param_ids
    assert id(net.norm.weight) in aux_param_ids


def test_non_muon_optimizer_preserves_parameter_groups():
    net = ToyNet()
    opt = SimpleNamespace(train_optim="adam")

    optimizer_bundle = build_optimizer_bundle(
        opt,
        [
            {"params": [net.linear.weight]},
            {"params": [net.linear_2.weight], "lr": 2e-3},
        ],
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        eps=1e-8,
        optimizer_name="optimizer_G",
    )

    assert [name for name, _ in optimizer_bundle] == ["optimizer_G"]

    optimizer = optimizer_bundle[0][1]
    assert isinstance(optimizer, torch.optim.Adam)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[1]["lr"] == 2e-3
