"""Load and resolve config.yaml, merging preset overrides."""

import copy
from pathlib import Path

import yaml


def load_config(path: str | Path = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["model"] = _resolve_model_preset(cfg["model"])
    return cfg


def _resolve_model_preset(model_cfg: dict) -> dict:
    cfg = copy.deepcopy(model_cfg)

    override = cfg.pop("override", False)
    presets = cfg.pop("presets", {})
    preset_name = cfg.get("preset", "small")

    if not override:
        preset_vals = presets.get(preset_name, {})
        # exclude metadata key
        for k, v in preset_vals.items():
            if k != "approx_params":
                cfg[k] = v

    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
