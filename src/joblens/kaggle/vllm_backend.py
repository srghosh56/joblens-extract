"""vLLM backend for the teacher / system (d). Verified against vLLM 0.31.0 source:
`LLM(model, revision, dtype, tensor_parallel_size, max_model_len, gpu_memory_utilization, seed)`,
`SamplingParams(temperature, max_tokens, structured_outputs=StructuredOutputsParams(json=...))`,
`LLM.chat(messages, sampling_params, use_tqdm)` returning RequestOutput with
`prompt_token_ids` and `outputs[0].text / .token_ids`. T4 (sm75): fp16 only, AWQ min cap 75,
Triton attention backend supports all capabilities."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from joblens.schema import JobPosting
from joblens.teacher import Generation, Message


class VLLMBackend:
    def __init__(
        self,
        weights: str,
        revision: str,
        tensor_parallel_size: int = 2,
        max_model_len: int = 6144,
        max_tokens: int = 900,
        constrained: bool = True,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self.name = f"vllm:{weights}@{revision[:8]}"
        self.llm = LLM(
            model=weights,
            revision=revision,
            dtype="float16",  # T4 has no bf16
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=0,
        )
        so = StructuredOutputsParams(json=JobPosting.model_json_schema()) if constrained else None
        self.params = SamplingParams(temperature=0.0, max_tokens=max_tokens, structured_outputs=so)
        self.timings: list[dict[str, Any]] = []

    def generate(self, conversations: Sequence[list[Message]]) -> list[Generation]:
        t0 = time.perf_counter()
        outs = self.llm.chat(list(conversations), sampling_params=self.params, use_tqdm=False)
        dt = time.perf_counter() - t0
        gens = [
            Generation(
                text=o.outputs[0].text,
                prompt_tokens=len(o.prompt_token_ids or []),
                completion_tokens=len(o.outputs[0].token_ids),
            )
            for o in outs
        ]
        self.timings.append(
            {
                "batch": len(gens),
                "seconds": dt,
                "completion_tokens": sum(g.completion_tokens for g in gens),
            }
        )
        return gens
