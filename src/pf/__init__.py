"""persona-foqa (`pf`): preventive persona-vector steering & inoculation prompting for LoRA
fine-tuning of Qwen2.5-7B on FoQA (Faroese QA).

The pure-stdlib core (config, data formatting, scoring, persona prompts, orchestration) imports
with no heavy deps; torch/transformers/peft are imported lazily inside the GPU code paths so the
config engine, scoring and sweep orchestration stay unit-testable on a CPU/Mac box.
"""

__version__ = "0.1.0"
