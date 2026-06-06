"""MotrixLab imitation learning integration package."""

from importlib.metadata import PackageNotFoundError, version


def describe() -> str:
    """Return a short description of the local MotrixLab/LeRobot integration."""
    try:
        lerobot_version = version("lerobot")
    except PackageNotFoundError:
        lerobot_version = "not installed"

    return f"motrix_il is configured with local LeRobot dependency: {lerobot_version}"
