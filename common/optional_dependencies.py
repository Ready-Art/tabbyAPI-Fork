"""Construct a model of all optional dependencies"""

import importlib.util
import json
from importlib.metadata import PackageNotFoundError, distribution
from importlib.metadata import version as package_version
from common.logger import xlogger
from packaging import version
from pydantic import BaseModel, computed_field


# Declare the exported parts of this module
__all__ = ["dependencies"]


class DependenciesModel(BaseModel):
    """Model of which optional dependencies are installed."""

    torch: bool
    exllamav2: bool
    exllamav3: bool
    flash_attn: bool
    infinity_emb: bool
    sentence_transformers: bool

    @computed_field
    @property
    def extras(self) -> bool:
        return self.infinity_emb and self.sentence_transformers

    @computed_field
    @property
    def inference(self) -> bool:
        return self.torch and (self.exllamav2 or self.exllamav3)


def is_installed(package_name: str) -> bool:
    """Utility function to check if a package is installed."""

    spec = importlib.util.find_spec(package_name)
    return spec is not None


def is_torch_cuda_13() -> bool:
    """Check whether the installed Torch wheel targets CUDA 13."""

    try:
        torch_version = package_version("torch")
    except PackageNotFoundError:
        return False

    _, _, local_version = torch_version.partition("+")
    return local_version.startswith("cu13")


def is_local_install(package_name: str) -> bool:
    """
    Check whether a package was installed from a local directory (editable or not).

    Source builds are compiled against the installed Torch, so they are not bound
    to the CUDA version of the prebuilt release wheels (e.g. the ReadyArt exllamav2
    fork built from source for CUDA 13 Torch).
    """

    try:
        direct_url = distribution(package_name).read_text("direct_url.json")
        if direct_url:
            return bool(json.loads(direct_url).get("dir_info"))
    except Exception:
        pass

    return False


def get_installed_deps() -> DependenciesModel:
    """Check if optional dependencies are installed by looping over the fields."""

    fields = DependenciesModel.model_fields

    installed_deps = {}

    for field_name in fields.keys():
        installed_deps[field_name] = is_installed(field_name)

    # Prebuilt exllamav2 wheels target CUDA 12 and are incompatible with CUDA 13
    # Torch. Local/source installs are built against the installed Torch instead.
    if (
        installed_deps.get("exllamav2")
        and is_torch_cuda_13()
        and not is_local_install("exllamav2")
    ):
        installed_deps["exllamav2"] = False

    return DependenciesModel(**installed_deps)


def check_package_version(package_name: str, required_version_str: str):
    """
    Fetches and verifies a given package version.

    This assumes that the required package is installed.
    """

    required_version = version.parse(required_version_str)
    current_version = version.parse(package_version(package_name).split("+")[0])

    unsupported_message = (
        f"ERROR: TabbyAPI requires {package_name} {required_version} "
        f"or greater. Your current version is {current_version}. "
        "Please update your dependencies."
    )

    if current_version < required_version:
        raise RuntimeError(unsupported_message)
    else:
        xlogger.info(f"{package_name} version: {current_version}")


dependencies = get_installed_deps()
