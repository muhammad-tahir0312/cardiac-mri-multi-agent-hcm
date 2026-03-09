"""src/config — configuration helpers for the HCM MAS pipeline."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

_DEFAULT_CFG = Path(__file__).parent / "base.yaml"


def load_config(path: Path = _DEFAULT_CFG, overrides: list = None) -> DictConfig:
    """Load the OmegaConf config from a YAML file, optionally applying overrides.

    Args:
        path:      Path to the YAML file (defaults to ``src/config/base.yaml``).
        overrides: List of dot-notation override strings, e.g.
                   ``["training.batch_size=16", "seed=0"]``.

    Returns:
        Merged :class:`~omegaconf.DictConfig` object.

    Example:
        >>> from src.config import load_config
        >>> cfg = load_config(overrides=["training.batch_size=4"])
        >>> cfg.training.batch_size
        4
    """
    cfg: DictConfig = OmegaConf.load(path)
    if overrides:
        extra = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, extra)
    return cfg


__all__ = ["load_config"]
