"""Stage the Hugging Face release: adapter + model card + label schema, nothing else.

Allow-list based: only known adapter/tokenizer file names are copied, so posting texts,
predictions or caches can never end up in the upload.
"""

from __future__ import annotations

import shutil
from pathlib import Path

HF_REPO = "gmohor21/joblens-mistral-7b-lora"
ADAPTER_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "chat_template.jinja",
}


def stage_release(adapter_dir: Path, card: Path, schema: Path, out: Path) -> list[str]:
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"no adapter in {adapter_dir}: run `make import-artifacts`")
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError("adapter_model.safetensors missing")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    copied = []
    for f in sorted(adapter_dir.iterdir()):
        if f.name in ADAPTER_FILES:
            shutil.copyfile(f, out / f.name)
            copied.append(f.name)
    shutil.copyfile(card, out / "README.md")
    (out / "schema").mkdir()
    shutil.copyfile(schema, out / "schema" / schema.name)
    copied += ["README.md", f"schema/{schema.name}"]
    unexpected = [
        str(p.relative_to(out))
        for p in out.rglob("*")
        if p.is_file() and p.name not in ADAPTER_FILES | {"README.md", schema.name}
    ]
    if unexpected:
        raise RuntimeError(f"unexpected files staged: {unexpected}")
    return copied
