import importlib
import importlib.metadata
import importlib.util
import sys
import types

import packaging
import torch
from torch import nn


def _install_pkg_resources_shim():
    if "pkg_resources" in sys.modules:
        return

    if importlib.util.find_spec("pkg_resources") is not None:
        return

    shim = types.ModuleType("pkg_resources")

    class DistributionNotFound(importlib.metadata.PackageNotFoundError):
        pass

    def get_distribution(distribution_name):
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise DistributionNotFound(str(exc)) from exc

        return types.SimpleNamespace(
            version=distribution.version,
            location=str(distribution.locate_file("")),
        )

    shim.DistributionNotFound = DistributionNotFound
    shim.get_distribution = get_distribution
    shim.packaging = packaging
    sys.modules["pkg_resources"] = shim


def _clear_failed_optional_imports():
    for module_name in list(sys.modules):
        if module_name == "gdown" or module_name.startswith("gdown."):
            sys.modules.pop(module_name, None)
        elif module_name.startswith("vision_aided_loss"):
            sys.modules.pop(module_name, None)


def _load_vision_aided_loss():
    try:
        return importlib.import_module("vision_aided_loss")
    except ModuleNotFoundError as exc:
        if exc.name != "pkg_resources":
            raise ModuleNotFoundError(
                "The `vision_aided` discriminator requires `vision_aided_loss` "
                f"and its runtime dependencies; missing module: `{exc.name}`."
            ) from exc

    # gdown 4.x still imports pkg_resources, but newer setuptools no longer
    # ships that module in some Python 3.12 environments.
    _install_pkg_resources_shim()
    _clear_failed_optional_imports()

    try:
        return importlib.import_module("vision_aided_loss")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The `vision_aided` discriminator could not import "
            "`vision_aided_loss` after installing a local `pkg_resources` "
            f"compatibility shim; missing module: `{exc.name}`."
        ) from exc


class VisionAidedDiscriminator(nn.Module):
    """Defines a vision-aided discriminator"""

    def __init__(
        self,
        cv_type="clip+dino",
    ):
        super(VisionAidedDiscriminator, self).__init__()
        vision_aided_loss = _load_vision_aided_loss()
        loss_type = ""  # loss is computed elsewhere
        self.model = vision_aided_loss.Discriminator(
            cv_type,
            loss_type,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.model.cv_ensemble.requires_grad_(False)  # freeze feature extractor

    def forward(self, input):
        return self.model(input)[0]
