"""Training hyperparameters (justified in DECISIONS D-014) and LoRA parameter arithmetic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

STUDENT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
STUDENT_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class TrainConfig:
    name: str
    r: int
    lora_alpha: int
    lora_dropout: float = 0.0
    learning_rate: float = 2e-4
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_steps: float = 0.03  # transformers 5.17: float in [0,1) = ratio of total steps
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    optim: str = "paged_adamw_8bit"
    eval_steps: int = 25
    early_stopping_patience: int = 3
    seed: int = 3407
    fp16: bool = True
    bf16: bool = False
    load_in_4bit: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


CONFIGS: dict[str, TrainConfig] = {
    "r16": TrainConfig(name="r16", r=16, lora_alpha=16),
    "r32": TrainConfig(name="r32", r=32, lora_alpha=32),
}


def lora_trainable_params(model_config: dict[str, Any], r: int, modules: list[str]) -> int:
    """A LoRA adapter on a d_in -> d_out linear layer adds r*(d_in + d_out) parameters
    (A: r x d_in, B: d_out x r). Summed over the targeted projections of every layer."""
    h = int(model_config["hidden_size"])
    inter = int(model_config["intermediate_size"])
    n_heads = int(model_config["num_attention_heads"])
    kv = int(model_config["num_key_value_heads"])
    head_dim = int(model_config.get("head_dim") or h // n_heads)
    shapes = {
        "q_proj": (h, n_heads * head_dim),
        "k_proj": (h, kv * head_dim),
        "v_proj": (h, kv * head_dim),
        "o_proj": (n_heads * head_dim, h),
        "gate_proj": (h, inter),
        "up_proj": (h, inter),
        "down_proj": (inter, h),
    }
    per_layer = sum(r * (shapes[m][0] + shapes[m][1]) for m in modules)
    return per_layer * int(model_config["num_hidden_layers"])


def effective_batch(cfg: TrainConfig) -> int:
    return cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps


def total_steps(cfg: TrainConfig, n_train: int) -> int:
    import math

    return math.ceil(n_train / effective_batch(cfg)) * math.ceil(cfg.num_train_epochs)
