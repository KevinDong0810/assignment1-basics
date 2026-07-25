from dataclasses import dataclass, field, is_dataclass
from typing import Any

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    num_layers: int = 12
    num_heads: int = 12
    d_model: int = 768
    d_ff: int = 2048
    theta: int = 1000

@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8


@dataclass
class RuntimeConfig:
    max_steps: int = 5000
    batch_size: int = 32
    eval_interval: int = 500
    log_interval: int = 10
    checkpoint_interval: int = 1000
    random_seed: int = -1
    device: str = "cuda:0"
    checkpoint_path: str = "checkpoints/"

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def apply_overrides(
    config: Any,
    overrides: dict[str, Any],
) -> None:
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise KeyError(f"未知配置项: {key}")

        current_value = getattr(config, key)

        if is_dataclass(current_value) and isinstance(value, dict):
            apply_overrides(current_value, value)
        else:
            setattr(config, key, value)




