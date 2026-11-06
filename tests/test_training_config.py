import json
from pathlib import Path

from joblens.training_config import (
    CONFIGS,
    TARGET_MODULES,
    effective_batch,
    lora_trainable_params,
    total_steps,
)

CFG = json.loads(
    (Path(__file__).parent / "fixtures" / "mistral_tokenizer" / "config.json").read_text()
)


def test_lora_param_count_mistral_7b():
    # per layer r*(4096+4096 + 4096+1024 + 4096+1024 + 4096+4096 + 3*(4096+14336)) = r*81920
    assert lora_trainable_params(CFG, 16, TARGET_MODULES) == 16 * 81920 * 32 == 41_943_040
    assert lora_trainable_params(CFG, 32, TARGET_MODULES) == 83_886_080
    assert lora_trainable_params(CFG, 8, ["q_proj"]) == 8 * 8192 * 32


def test_configs():
    assert CONFIGS["r16"].lora_alpha == 16 and CONFIGS["r32"].r == 32
    assert all(c.lora_dropout == 0.0 and c.fp16 and not c.bf16 for c in CONFIGS.values())
    assert effective_batch(CONFIGS["r16"]) == 16
    assert total_steps(CONFIGS["r16"], 2700) == 169 * 2
    assert CONFIGS["r16"].as_dict()["optim"] == "paged_adamw_8bit"
