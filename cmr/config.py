"""Configuration loading.

One YAML file is the single source of truth. Nothing in this codebase hard-codes a
path, a threshold, a model name, or a device. Change `configs/default.yaml` — or
point `--config-file` at another one (e.g. `configs/hpc.yaml`) — and the whole
pipeline follows.

Environment overrides (useful on a cluster where you cannot edit files):
    CMR_DEVICE=cuda
    CMR_DATA_ROOT=/scratch/tahir/cardio
    CMR_LLM_PROVIDER=gemini
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = _REPO / "configs" / "default.yaml"
EXPERIMENTS = _REPO / "configs" / "experiments.yaml"

log = logging.getLogger("cmr")


class Config(dict):
    """Dict with attribute access and dotted lookup. Deliberately thin."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _resolve_paths(cfg: dict) -> dict:
    """Expand `~` and `{data_root}` templates in the paths block."""
    paths = cfg["paths"]
    root = os.environ.get("CMR_DATA_ROOT", paths["data_root"])
    paths["data_root"] = str(Path(root).expanduser().resolve())
    for k, v in paths.items():
        if k == "data_root" or not isinstance(v, str):
            continue
        paths[k] = str(Path(v.format(data_root=paths["data_root"])).expanduser())
    return cfg


def _apply_env(cfg: dict) -> dict:
    """A few env overrides, for HPC job scripts that cannot edit YAML."""
    env_map = {
        "CMR_DEVICE": ("compute", "device"),
        "CMR_LLM_PROVIDER": ("llm", "provider"),
        "CMR_LLM_MODEL": ("llm", "model"),
        "CMR_LOG": ("logging", "level"),
    }
    for env, (section, key) in env_map.items():
        if env in os.environ:
            cfg[section][key] = os.environ[env]
    return cfg


def load(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = _apply_env(_resolve_paths(cfg))
    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"].upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-12s %(message)s",
        datefmt="%H:%M:%S",
    )
    return Config(cfg)


def load_experiment(name: str, base: Config | None = None) -> Config:
    """Merge one row of the ablation grid over the default config."""
    cfg = dict(base or load())
    with open(EXPERIMENTS) as f:
        grid = yaml.safe_load(f)
    if name not in grid:
        raise KeyError(f"unknown experiment '{name}'. Known: {sorted(grid)}")
    exp = grid[name]

    # The grid speaks in short keys; map them onto the real config tree.
    mapping = {
        "retriever": ("retrieval", "mode"),
        "embedder": ("retrieval", "embedder"),
        "llm": ("llm", "model"),
        "provider": ("llm", "provider"),
        "gate": ("orchestration", "gate"),
        "feedback": ("orchestration", "feedback"),
        "checkpoint": ("segmentation", "checkpoint"),
        "backend": ("segmentation", "backend"),
    }
    for k, v in exp.items():
        if k in mapping:
            section, key = mapping[k]
            cfg[section] = dict(cfg[section]) | {key: v}
        else:
            cfg[k] = v
    cfg["experiment"] = name
    cfg["config_id"] = config_id(name, exp)
    return Config(cfg)


def config_id(name: str, spec: dict) -> str:
    """Deterministic id. Same spec -> same directory; changed spec -> new directory.

    This is what stops one ablation from silently contaminating another's results.
    """
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:6]
    return f"{name}_{h}"


def device(cfg: Config) -> str:
    """auto -> cuda (HPC) | mps (this Mac) | cpu. Resolved once, here, for everyone."""
    want = cfg.compute.device
    if want != "auto":
        return want
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def artifacts(cfg: Config, *parts: str) -> Path:
    p = Path(cfg.paths.artifacts).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
